//! The implementation of `UnsizedVec<T>` for `T: ?Sized + Aligned`.

use ::alloc::{alloc, collections::TryReserveErrorKind};
use core::{
    alloc::{Allocator, Layout},
    cmp,
    iter::FusedIterator,
    marker::{PhantomData, Unsize},
    mem::{self, ManuallyDrop},
    ptr::{self, addr_of, NonNull},
};

use emplacable::{Emplacable, EmplacableFn, Emplacer};

use crate::{
    helper::{decompose, valid_size::ValidSize, MetadataRemainder, SplitMetadata},
    marker::Aligned,
    unwrap_try_reserve_result,
};

use super::{AlignedVecImpl, AlignedVecProvider, TryReserveError, UnsizedVecProvider};

struct ElementInfo<T: ?Sized + Aligned> {
    /// The pointer metadata of the element.
    metadata: <T as SplitMetadata>::Remainder,
    /// The offset that the element following this one would be stored at,
    /// but disregarding padding due to over-alignment.
    /// We use this encoding to store the sizes of `Vec` elements
    /// because it allows for *O(1)* random access while only storing
    /// a single `usize`.
    ///
    /// To get the actual offset of the next element, use
    /// `unchecked_pad_to(end_offset, align)`.
    end_offset: ValidSize<T>,
}

impl<T: ?Sized + Aligned> Clone for ElementInfo<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized + Aligned> Copy for ElementInfo<T> {}

pub(in super::super) struct AlignedVecInner<T: ?Sized + Aligned> {
    ptr: NonNull<()>,
    byte_capacity: ValidSize<T>,
    elems_info: ManuallyDrop<::alloc::vec::Vec<ElementInfo<T>>>,
    _marker: PhantomData<T>,
}

impl<T: ?Sized + Aligned> AlignedVecInner<T> {
    /// The number of bytes this vec is curretly using.
    /// (sum of `size_of_val`s of all elements).
    #[inline]
    fn byte_len(&self) -> ValidSize<T> {
        self.elems_info
            .last()
            .map_or(ValidSize::ZERO, |last| last.end_offset)
    }

    /// Returns the offset of the start of this element in the vec.
    ///
    /// # Safety
    ///
    /// Does no bounds checks
    #[inline]
    unsafe fn start_offset_of_unchecked(&self, index: usize) -> ValidSize<T> {
        index.checked_sub(1).map_or(ValidSize::ZERO, |index_m_1|
                // SAFETY: precondition of function
                unsafe {
                self.elems_info
                    .get_unchecked(index_m_1)
                    .end_offset
            })
    }
}

impl<T: ?Sized + Aligned> Drop for AlignedVecInner<T> {
    fn drop(&mut self) {
        let mut start_offset: ValidSize<T> = ValidSize::ZERO;

        // SAFETY: we are in `drop`, nobody will access the `ManuallyDrop` after us
        let elems_info = unsafe { ManuallyDrop::take(&mut self.elems_info) };

        // Drop remaining elements
        for ElementInfo {
            metadata,
            end_offset,
        } in elems_info
        {
            // SAFETY: end of element can't be smaller than start
            let size_of_val = unsafe { end_offset.unchecked_sub(start_offset) };
            let metadata = metadata.as_metadata(size_of_val.as_unaligned());

            let start_of_alloc = self.ptr.as_ptr().cast::<u8>();
            // SAFETY: offset is within allocation
            let thin_ptr_to_elem = unsafe { start_of_alloc.add(start_offset.get()) };
            let wide_ptr_to_elem: *mut T =
                ptr::from_raw_parts_mut(thin_ptr_to_elem.cast(), metadata);

            start_offset = end_offset;

            // SAFETY: nobody will access this after us
            unsafe { wide_ptr_to_elem.drop_in_place() }
        }

        // Drop allocation

        let alloc_layout = self.byte_capacity.as_layout();

        // SAFETY: capacity and align come from the vec.
        unsafe {
            alloc::Global.deallocate(self.ptr.cast(), alloc_layout);
        }
    }
}

impl<T: ?Sized + Aligned> UnsizedVecProvider<T> for AlignedVecInner<T> {
    type Align = ();
    type Size = ValidSize<T>;

    type Iter<'a> = AlignedIter<'a, T> where T: 'a;
    type IterMut<'a> = AlignedIterMut<'a, T> where T: 'a;

    const NEW_ALIGN_1: Self = AlignedVecInner {
        ptr: T::DANGLING_THIN,
        byte_capacity: ValidSize::ZERO,
        elems_info: ManuallyDrop::new(::alloc::vec::Vec::new()),
        _marker: PhantomData,
    };

    const NEW_ALIGN_PTR: Self = Self::NEW_ALIGN_1;

    #[inline]
    fn capacity(&self) -> usize {
        self.elems_info.capacity()
    }

    #[inline]
    fn byte_capacity(&self) -> usize {
        self.byte_capacity.get()
    }

    #[inline]
    fn align(&self) -> usize {
        T::ALIGN.get()
    }

    #[inline]
    fn try_reserve_exact_capacity_bytes_align(
        &mut self,
        additional: usize,
        additional_bytes: usize,
        _align: (),
    ) -> Result<(), TryReserveError> {
        self.elems_info.try_reserve_exact(additional)?;

        let old_cap = self.byte_capacity;

        if additional_bytes > 0 {
            let new_cap = self
                .byte_len()
                .get()
                .checked_add(additional_bytes)
                .and_then(ValidSize::<T>::new)
                .ok_or(TryReserveError {
                    kind: TryReserveErrorKind::CapacityOverflow,
                })?;

            let new_layout = new_cap.as_layout();

            let new_ptr: NonNull<[u8]> = if old_cap == ValidSize::ZERO {
                alloc::Global.allocate(new_layout)
            } else {
                let old_layout = old_cap.as_layout();

                // SAFETY: old layout comes from vec, checked above for `old_cap < new_cap`
                unsafe { alloc::Global.grow(self.ptr.cast(), old_layout, new_layout) }
            }
            .map_err(|_| TryReserveError {
                kind: TryReserveErrorKind::AllocError {
                    layout: new_cap.as_layout(),
                    non_exhaustive: (),
                },
            })?;

            self.ptr = new_ptr.cast();

            self.byte_capacity = ValidSize::new_squished(new_ptr.len());
        }
        Ok(())
    }

    #[inline]
    fn shrink_capacity_bytes_align_to(
        &mut self,
        min_capacity: usize,
        byte_capacity: usize,
        _align: (),
    ) {
        self.elems_info.shrink_to(min_capacity);

        let old_cap = self.byte_capacity;
        let new_cap = cmp::max(self.byte_len().get(), byte_capacity);

        if new_cap < old_cap.get() {
            // SAFETY: `self.byte_capacity` is valid, so anything less than it is too
            let new_cap = unsafe { ValidSize::<T>::new_unchecked(new_cap) };

            if new_cap == ValidSize::ZERO {
                // SAFETY: layout comes from the vec
                unsafe { alloc::Global.deallocate(self.ptr.cast(), old_cap.as_layout()) }
                self.ptr = <T as Aligned>::DANGLING_THIN;
                self.byte_capacity = ValidSize::ZERO;
            } else {
                // SAFETY: `old_layout` comes from the vec, if guard ensures `new_layout` is smaller
                if let Ok(new_ptr) = unsafe {
                    alloc::Global.shrink(self.ptr.cast(), old_cap.as_layout(), new_cap.as_layout())
                } {
                    self.ptr = new_ptr.cast();
                    self.byte_capacity = ValidSize::new_squished(new_ptr.len());
                }

                // if shrink fails, we just keep old allocation
            }
        }
    }

    unsafe fn insert_unchecked(&mut self, index: usize, element: T, size_of_val: ValidSize<T>) {
        debug_assert!(self.capacity() > self.len());
        debug_assert!(self.byte_capacity() >= (self.byte_len().get() + size_of_val.get()));

        let metadata =
            <T as SplitMetadata>::Remainder::from_metadata(core::ptr::metadata(&element));

        // SAFETY: preconditions of function
        unsafe {
            let start_offset = self.start_offset_of_unchecked(index);
            let how_much_to_move = self.byte_len().unchecked_sub(start_offset);

            let start_ptr = self.ptr.cast::<u8>().as_ptr().add(index);

            ptr::copy(
                start_ptr,
                start_ptr.add(size_of_val.get()),
                how_much_to_move.get(),
            );

            ptr::copy_nonoverlapping(addr_of!(element).cast(), start_ptr, size_of_val.get());

            for ElementInfo { end_offset, .. } in self.elems_info.get_unchecked_mut(index..) {
                *end_offset = end_offset.unchecked_add(size_of_val);
            }

            self.elems_info.insert_unchecked(
                index,
                ElementInfo {
                    metadata,
                    end_offset: start_offset.unchecked_add(size_of_val),
                },
                (),
            );
        }

        mem::forget_unsized(element);
    }

    unsafe fn insert_with_unchecked(
        &mut self,
        index: usize,
        value: Emplacable<T, impl EmplacableFn<T>>,
    ) {
        /// Helper to ensure elements are moved back
        /// where they belong in case `inner_closure`
        /// panics.
        struct ElementShifterBacker<T: ?Sized + Aligned> {
            ptr_to_index: *mut u8,
            num_bytes_to_shift: ValidSize<T>,
            shift_by_bytes: ValidSize<T>,
        }

        impl<T: ?Sized + Aligned> Drop for ElementShifterBacker<T> {
            #[inline]
            fn drop(&mut self) {
                // SAFETY: shifting elements back in case of drop
                unsafe {
                    ptr::copy(
                        self.ptr_to_index.add(self.shift_by_bytes.get()),
                        self.ptr_to_index,
                        self.num_bytes_to_shift.get(),
                    );
                }
            }
        }

        debug_assert!(index <= self.len());

        let emplacable_closure = value.into_fn();

        let emplacer_closure =
            &mut |layout: Layout, metadata, inner_closure: &mut dyn FnMut(*mut PhantomData<T>)| {
                let (size_of_val, _) = decompose(layout);

                // SAFETY: by `Emplacer::new` preconditions
                let size_of_val = unsafe { ValidSize::<T>::new_unchecked(size_of_val.get()) };

                let reserve_result =
                    self.try_reserve_exact_capacity_bytes_align(1, layout.size(), ());
                unwrap_try_reserve_result(reserve_result);

                // SAFETY: precondition of function
                let start_offset = unsafe { self.start_offset_of_unchecked(index) };

                // SAFETY: getting pointer to element
                let ptr_to_elem = unsafe { self.ptr.cast::<u8>().as_ptr().add(start_offset.get()) };

                // SAFETY: by precondition of function
                let num_bytes_to_shift = unsafe { self.byte_len().unchecked_sub(start_offset) };

                let shifter_backer = ElementShifterBacker {
                    ptr_to_index: ptr_to_elem,
                    num_bytes_to_shift,
                    shift_by_bytes: size_of_val,
                };

                // SAFETY: copying elements right to make room
                unsafe {
                    ptr::copy(
                        ptr_to_elem,
                        ptr_to_elem.add(size_of_val.get()),
                        num_bytes_to_shift.get(),
                    );
                }

                // If this unwinds, `shifter_backer` will be dropped
                // and the elements will be moved back where they belong.
                inner_closure(ptr_to_elem.cast());

                // `inner_closure` succeeded, so don't want to move elements back now!
                mem::forget(shifter_backer);

                // SAFETY: by precondition of function
                let elems_to_move_back = unsafe { self.elems_info.get_unchecked_mut(index..) };

                for ElementInfo { end_offset, .. } in elems_to_move_back {
                    // SAFETY: make the offsets correct again
                    *end_offset = unsafe { end_offset.unchecked_add(size_of_val) };
                }

                // SAFETY: reserved memory earlier
                unsafe {
                    self.elems_info.insert_unchecked(
                        index,
                        ElementInfo {
                            metadata: <T as SplitMetadata>::Remainder::from_metadata(metadata),
                            end_offset: start_offset.unchecked_add(size_of_val),
                        },
                        (),
                    );
                }
            };

        // SAFETY: `emplacer_closure` runs the closure with a valid pointer to `index`
        let emplacer = unsafe { Emplacer::from_fn(emplacer_closure) };

        emplacable_closure(emplacer);
    }

    unsafe fn remove_into_unchecked(&mut self, index: usize, emplacer: &mut Emplacer<'_, T>) {
        debug_assert!(index < self.len());

        // We can't remove the metadata yet, as `emplacer_closure` might unwind,
        // so we can't leave vec metadata in an invalid state.
        // SAFETY: by precondition of function
        let removed_elem_metadata = unsafe { self.elems_info.get_unchecked(index) };

        let ElementInfo {
            metadata,
            end_offset,
        } = removed_elem_metadata;

        // SAFETY: precondition of function
        let start_offset = unsafe { self.start_offset_of_unchecked(index) };

        // SAFETY: start_offset < end_offset
        let size_of_val = unsafe { end_offset.unchecked_sub(start_offset) };

        let metadata = metadata.as_metadata(size_of_val.as_unaligned());

        // Get pointer to the element we are popping out of the vec
        // SAFETY: offset comes from vec
        let thin_ptr_to_elem = unsafe {
            self.ptr
                .as_ptr()
                .cast_const()
                .cast::<u8>()
                .add(start_offset.get())
        };

        // Copy element into the place

        // SAFETY: we call the closure right after we unwrap it
        let emplacer_closure = unsafe { emplacer.into_fn() };

        // The emplacer can choose never to run the inner closure at all! In this case, the removed value
        // is simply forgotten.
        emplacer_closure(size_of_val.as_layout(), metadata, &mut |out_ptr| {
            if !out_ptr.is_null() {
                // SAFETY: we are allowed to copy `size_of_val` bytes into `out_ptr`,
                // by the preconditions of `Emplacer::new`
                unsafe {
                    ptr::copy_nonoverlapping(
                        thin_ptr_to_elem,
                        out_ptr.cast::<u8>(),
                        size_of_val.get(),
                    );
                }
            } else {
                let wide_ptr: *mut T =
                    ptr::from_raw_parts_mut(thin_ptr_to_elem.cast_mut().cast(), metadata);

                // SAFETY: We forget the element right after by copying over it and adjusting vec metadata
                unsafe { wide_ptr.drop_in_place() }
            }
        });

        // Now that `emplacer_closure` has run successfuly, we don't need to worry
        // about exception safety anymore.
        // FIXME elide bounds check
        self.elems_info.remove(index);

        for ElementInfo { end_offset, .. } in
            // SAFETY: `index` in range by preconditions of function.
            unsafe { self.elems_info.get_unchecked_mut(index..) }
        {
            // SAFETY: `end_fooset >= size_of_val` for elements following something
            // of size `size_of_val`
            unsafe {
                *end_offset = end_offset.unchecked_sub(size_of_val);
            }
        }

        // SAFETY: new end of vec can't be to the left of old start of elem at `index`
        let how_much_to_move = unsafe { self.byte_len().unchecked_sub(start_offset) };

        // SAFETY: copying elements back where they belong
        unsafe {
            ptr::copy(
                thin_ptr_to_elem.add(size_of_val.get()),
                thin_ptr_to_elem.cast_mut(),
                how_much_to_move.get(),
            );
        }
    }

    unsafe fn push_unchecked(&mut self, value: T, size_of_val: ValidSize<T>) {
        debug_assert!(self.capacity() - self.len() > 0);
        debug_assert!(self.byte_capacity() >= (self.byte_len().get() + size_of_val.get()));

        let metadata = <T as SplitMetadata>::Remainder::from_metadata(core::ptr::metadata(&value));
        let start_offset = self.byte_len();
        // SAFETY: preconditions of function
        unsafe {
            ptr::copy_nonoverlapping(
                addr_of!(value).cast(),
                self.ptr.as_ptr().cast::<u8>().add(start_offset.get()),
                size_of_val.get(),
            );

            self.elems_info.push_unchecked(
                ElementInfo {
                    metadata,
                    end_offset: start_offset.unchecked_add(size_of_val),
                },
                (),
            );
        }

        mem::forget_unsized(value);
    }

    fn push_with(&mut self, value: Emplacable<T, impl EmplacableFn<T>>) {
        let emplacable_closure = value.into_fn();

        let emplacer_closure =
            &mut |layout: Layout, metadata, inner_closure: &mut dyn FnMut(*mut PhantomData<T>)| {
                let (size_of_val, _) = decompose(layout);

                // SAFETY: by `Emplacer::new` preconditions
                let size_of_val = unsafe { ValidSize::<T>::new_unchecked(size_of_val.get()) };

                let reserve_result =
                    self.try_reserve_exact_capacity_bytes_align(1, layout.size(), ());
                unwrap_try_reserve_result(reserve_result);

                let start_offset = self.byte_len();

                // SAFETY: getting pointer to end of allocation
                let ptr_to_elem = unsafe { self.ptr.cast::<u8>().as_ptr().add(start_offset.get()) };

                inner_closure(ptr_to_elem.cast());

                let elem_info: ElementInfo<T> = ElementInfo {
                    metadata: <T as SplitMetadata>::Remainder::from_metadata(metadata),
                    // SAFETY: neither operand can overflow `isize`, so sum
                    // can't overflow `usize`
                    end_offset: unsafe { start_offset.unchecked_add(size_of_val) },
                };

                // SAFETY: `emplacable` wrote new element at end of vec,
                // and we have reserved the needed space
                unsafe { self.elems_info.push_unchecked(elem_info, ()) };
            };

        // SAFETY: `emplacer_closure` runs the closure with a valid pointer to the end of the vec
        let emplacer = unsafe { Emplacer::from_fn(emplacer_closure) };

        emplacable_closure(emplacer);
    }

    unsafe fn pop_into_unchecked(&mut self, emplacer: &mut Emplacer<'_, T>) {
        debug_assert!(!self.elems_info.is_empty());

        // SAFETY: precondition of function
        let last_elem_metadata = unsafe { self.elems_info.pop().unwrap_unchecked() };

        let ElementInfo {
            metadata,
            end_offset,
        } = last_elem_metadata;

        let start_offset = self.byte_len();

        // SAFETY: start_offset < end_offset
        let size_of_val = unsafe { end_offset.unchecked_sub(start_offset) };

        let metadata = metadata.as_metadata(size_of_val.as_unaligned());

        // Get pointer to the element we are popping out of the vec
        // SAFETY: offset comes from vec
        let thin_ptr_to_elem = unsafe {
            self.ptr
                .as_ptr()
                .cast_const()
                .cast::<u8>()
                .add(start_offset.get())
        };

        // Copy element into the place

        // SAFETY: we call the closure right after we unwrap it
        let emplacer_closure = unsafe { emplacer.into_fn() };

        emplacer_closure(size_of_val.as_layout(), metadata, &mut |out_ptr| {
            if !out_ptr.is_null() {
                // SAFETY: we are allowed to copy `size_of_val` bytes into `out_ptr`,
                // by the preconditions of `Emplacer::new`
                unsafe {
                    ptr::copy_nonoverlapping(
                        thin_ptr_to_elem,
                        out_ptr.cast::<u8>(),
                        size_of_val.get(),
                    );
                }
            } else {
                let wide_ptr_to_elem: *mut T =
                    ptr::from_raw_parts_mut(thin_ptr_to_elem.cast_mut().cast(), metadata);

                // SAFETY: we adusted vec metadata earlier, so this won't be double-dropped
                unsafe { wide_ptr_to_elem.drop_in_place() }
            }
        });
    }

    #[inline]
    fn len(&self) -> usize {
        self.elems_info.len()
    }

    #[inline]
    unsafe fn get_unchecked_raw(&self, index: usize) -> NonNull<T> {
        debug_assert!(index < self.len());

        // SAFETY: see individual comments inside block
        unsafe {
            // SAFETY: precondition of method
            let start_offset = self.start_offset_of_unchecked(index);
            let &ElementInfo {
                end_offset,
                metadata,
            } = self.elems_info.get_unchecked(index);

            // SAFETY: end >= start
            let size_of_val = end_offset.unchecked_sub(start_offset);
            let metadata = metadata.as_metadata(size_of_val.as_unaligned());

            // SAFETY: `start_offset` in range of allocation
            NonNull::from_raw_parts(
                NonNull::new_unchecked(self.ptr.as_ptr().cast::<u8>().add(start_offset.get()))
                    .cast(),
                metadata,
            )
        }
    }

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        AlignedIter {
            elems_info: self.elems_info.iter(),
            ptr: self.ptr,
            start_offset: ValidSize::ZERO,
        }
    }

    #[inline]
    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        AlignedIterMut {
            elems_info: self.elems_info.iter(),
            ptr: self.ptr,
            start_offset: ValidSize::ZERO,
        }
    }

    #[inline]
    fn from_sized<S>(vec: ::alloc::vec::Vec<S>) -> Self
    where
        S: Unsize<T>,
    {
        let mut vec = ManuallyDrop::new(vec);
        let len_elems = vec.len();
        let cap_elems = vec.capacity();
        let heap_ptr = vec.as_mut_ptr();
        let heap_ptr_unsized: *mut T = heap_ptr;
        let metadata =
            <T as SplitMetadata>::Remainder::from_metadata(ptr::metadata(heap_ptr_unsized));
        // SAFETY: ptr comes from vec, can't be null
        let heap_ptr_thin: NonNull<()> = unsafe { NonNull::new_unchecked(heap_ptr_unsized.cast()) };

        // SAFETY: can't overflow, as otherwise allocation would be overflowing
        let byte_capacity = unsafe { cap_elems.unchecked_mul(mem::size_of::<S>()) };

        // SAFETY: same as above
        let byte_capacity = unsafe { ValidSize::new_unchecked(byte_capacity) };

        let elems_info = (0..len_elems)
            .map(|index| ElementInfo {
                metadata,
                // SAFETY: can't overflow, as otherwise allocation would be overflowing
                end_offset: unsafe {
                    ValidSize::new_unchecked(index.unchecked_mul(mem::size_of::<S>()))
                },
            })
            .collect();

        let elems_info = ManuallyDrop::new(elems_info);

        Self {
            ptr: heap_ptr_thin,
            byte_capacity,
            elems_info,
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized + Aligned> AlignedVecProvider<T> for AlignedVecInner<T> {}

impl<T: ?Sized + Aligned> AlignedVecImpl for T {
    default type Impl = AlignedVecInner<T>;
}

macro_rules! iter_ref {
    ($iter_ty:ident, $from_raw_parts:ident $($muta:ident)?) => {
        pub(in super::super) struct $iter_ty<'a, T: ?Sized + Aligned> {
            elems_info: core::slice::Iter<'a, ElementInfo<T>>,
            ptr: NonNull<()>,
            start_offset: ValidSize<T>,
        }

        impl<'a, T: ?Sized + Aligned + 'a> Iterator for $iter_ty<'a, T> {
            type Item = &'a $($muta)? T;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                let ElementInfo {
                    metadata,
                    end_offset,
                } = *self.elems_info.next()?;

                // SAFETY: end of element can't be smaller than start
                let size_of_val = unsafe { end_offset.unchecked_sub(self.start_offset) };
                let metadata = metadata.as_metadata(size_of_val.as_unaligned());

                let start_of_alloc = self.ptr.as_ptr().cast::<u8>();
                // SAFETY: offset is within allocation
                let thin_ptr_to_elem = unsafe { start_of_alloc.add(self.start_offset.get()) };
                let wide_ptr = ptr::$from_raw_parts(thin_ptr_to_elem.cast(), metadata);

                // SAFETY: pointer to element of vec
                let wide_ref = unsafe { & $($muta)? *wide_ptr };

                self.start_offset = end_offset;

                Some(wide_ref)
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.elems_info.size_hint()
            }

            #[inline]
            fn count(self) -> usize {
                self.elems_info.count()
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<Self::Item> {
                let start_offset = n
                    .checked_sub(1)
                    .and_then(|n| self.elems_info.nth(n))
                    .copied()
                    .map_or(ValidSize::ZERO, |e_i| e_i.end_offset);

                let ElementInfo {
                    metadata,
                    end_offset,
                } = *self.elems_info.next()?;

                // SAFETY: end of element can't be smaller than start`
                let size_of_val = unsafe { end_offset.unchecked_sub(start_offset) };
                let metadata = metadata.as_metadata(size_of_val.as_unaligned());

                let start_of_alloc = self.ptr.as_ptr().cast::<u8>();
                // SAFETY: offset is within allocation
                let thin_ptr_to_elem = unsafe { start_of_alloc.add(start_offset.get()) };
                let wide_ptr = ptr::$from_raw_parts(thin_ptr_to_elem.cast(), metadata);

                // SAFETY: pointer to element of vec
                let wide_ref = unsafe { & $($muta)? *wide_ptr };

                self.start_offset = end_offset;

                Some(wide_ref)
            }

            #[inline]
            fn last(mut self) -> Option<Self::Item> {
                self.nth(self.elems_info.len().checked_sub(1)?)
            }
        }

        impl<'a, T: ?Sized + Aligned + 'a> DoubleEndedIterator for $iter_ty<'a, T> {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                let ElementInfo {
                    metadata,
                    end_offset,
                } = *self.elems_info.next_back()?;

                let start_offset = self
                    .elems_info
                    .as_slice()
                    .last()
                    .map_or(ValidSize::ZERO, |e_i| e_i.end_offset);

                // SAFETY: end of element can't be smaller than start
                let size_of_val = unsafe { end_offset.unchecked_sub(start_offset) };
                let metadata = metadata.as_metadata(size_of_val.as_unaligned());

                let start_of_alloc = self.ptr.as_ptr().cast::<u8>();
                // SAFETY: offset is within allocation
                let thin_ptr_to_elem = unsafe { start_of_alloc.add(start_offset.get()) };
                let wide_ptr = ptr::$from_raw_parts(thin_ptr_to_elem.cast(), metadata);

                // SAFETY: pointer to element of vec
                let wide_ref = unsafe { & $($muta)? *wide_ptr };

                Some(wide_ref)
            }
        }

        impl<'a, T: ?Sized + Aligned + 'a> ExactSizeIterator for $iter_ty<'a, T> {
            #[inline]
            fn len(&self) -> usize {
                self.elems_info.len()
            }
        }

        impl<'a, T: ?Sized + Aligned + 'a> FusedIterator for $iter_ty<'a, T> {}
    };
}

iter_ref!(AlignedIter, from_raw_parts);
iter_ref!(AlignedIterMut, from_raw_parts_mut mut);
