//! The implementation of `UnsizedVec<T>` for `T: ?Sized + ?Aligned`.

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
    helper::{
        decompose, valid_align::ValidAlign, valid_size::ValidSizeUnaligned, MetadataRemainder,
        SplitMetadata,
    },
    marker::Aligned,
    unwrap_try_reserve_result,
};

use super::{TryReserveError, UnsizedVecImpl, UnsizedVecProvider};

struct ElementInfo<T: ?Sized> {
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
    end_offset: ValidSizeUnaligned,
}

impl<T: ?Sized> Clone for ElementInfo<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for ElementInfo<T> {}

pub(in super::super) struct UnalignedVecInner<T: ?Sized> {
    ptr: NonNull<()>,
    /// # Safety
    ///
    /// For simplicity, must be a multiple of `self.align`.
    byte_capacity: ValidSizeUnaligned,
    elems_info: ManuallyDrop<::alloc::vec::Vec<ElementInfo<T>>>,
    align: ValidAlign,
    _marker: PhantomData<T>,
}

impl<T: ?Sized> UnalignedVecInner<T> {
    /// The number of bytes this vec is curretly using,
    /// discounting padding following the last element.
    #[inline]
    fn unaligned_byte_len(&self) -> ValidSizeUnaligned {
        self.elems_info
            .last()
            .map_or(ValidSizeUnaligned::ZERO, |last| last.end_offset)
    }

    /// The number of bytes this vec is curretly using,
    /// including padding following the last element.
    #[inline]
    fn aligned_byte_len(&self) -> ValidSizeUnaligned {
        // SAFETY: it's an invariant of the capacity field that this be legal
        unsafe { self.unaligned_byte_len().unchecked_pad_to(self.align) }
    }

    /// Returns the offset of the start of this element in the vec.
    ///
    /// # Safety
    ///
    /// Does not bounds checks
    #[inline]
    unsafe fn start_offset_of_unchecked(&self, index: usize) -> ValidSizeUnaligned {
        index
            .checked_sub(1)
            .map_or(ValidSizeUnaligned::ZERO, |index_m_1|
                // SAFETY: precondition of function
                unsafe {
                self.elems_info
                    .get_unchecked(index_m_1)
                    .end_offset
                    .unchecked_pad_to(self.align)
            })
    }

    /// Returns the maximum alignment among all the elements in the vec.
    /// Used by `shrink`.
    #[inline]
    fn max_align_of_elems(&self) -> ValidAlign {
        self.iter()
            .map(ValidAlign::of_val)
            .max()
            .unwrap_or(ValidAlign::ONE)
    }

    /// Used in `try_reserve_exact_bytes_align_unchecked`.
    /// Returns what the length in bytes of the array would be
    /// after its alignment is increased from `self.align` to `new_align`
    ///
    /// # Safety
    ///
    /// `self.align <= new_align` must hold.
    unsafe fn len_after_realign_up(&mut self, new_align: ValidAlign) -> Option<ValidSizeUnaligned> {
        debug_assert!(self.align <= new_align);
        let mut new_pad_to_new: ValidSizeUnaligned = ValidSizeUnaligned::ZERO;
        self.elems_info.iter().try_fold(
            ValidSizeUnaligned::ZERO,
            |shift,
             ElementInfo {
                 end_offset: old_end_offset,
                 ..
             }| {
                let new_end_offset = old_end_offset.checked_add(shift)?;

                new_pad_to_new = new_end_offset.checked_pad_to(new_align)?;

                // SAFETY: `old_align <= new_align`, so if above call returned `Some`, this must be legal.
                let new_pad_to_old = unsafe { new_end_offset.unchecked_pad_to(self.align) };

                // SAFETY: `old_align <= new_align`, so can't underflow
                let padding_difference = unsafe { new_pad_to_new.unchecked_sub(new_pad_to_old) };

                shift.checked_add(padding_difference)
            },
        )?;

        Some(new_pad_to_new)
    }

    /// Realigns all elements in the vec to the given `new_align`,
    /// if the current align is less.
    ///
    /// # Safety
    ///
    /// `new_align > self.align` must hold.
    /// `self.len() > 1` must hold.
    ///
    /// Realigning must not lead to overflow.
    ///
    /// `new_align` must be equal to the actual alignment of the allocation,
    /// Also, this function does not allocate memory,
    /// nor does it check that enough memory has been allocated.
    ///
    /// Finally, this function doesn't update `self.align`, do that yourself.
    unsafe fn realign_up(&mut self, new_align: ValidAlign) {
        let old_align = self.align;

        debug_assert!(self.len() > 1 && new_align > old_align);

        // We compute the new offset of each element, along with the difference from the old offset.
        // Then, we copy everything over.
        // Doing this without allocating requires some complicated code.
        //
        // First we calculate how much we need to shift the very last element,
        // then we perform the copies while reversing our calculations.
        //
        // The first element is already in the right place, its offset is 0.

        // Starting here, our offsets are invalid, so unwinding is UB !!!
        // To make this explicit, we use unckecked ops for arithmetic.
        // This loop is basically `len_after_realign_up`, excpet with 0 checks and modifying metadata.
        // TODO: get from `len_after_realign_up`?
        let final_offset_shift: ValidSizeUnaligned = self.elems_info.iter_mut().fold(
            ValidSizeUnaligned::ZERO,
            |shift, ElementInfo { end_offset, .. }| {
                // SAFETY: precondition of function
                unsafe {
                    let new_end_offset = end_offset.unchecked_add(shift);
                    *end_offset = new_end_offset;

                    let new_pad_to_new = new_end_offset.unchecked_pad_to(new_align);
                    let new_pad_to_old = new_end_offset.unchecked_pad_to(old_align);
                    let padding_difference = new_pad_to_new.unchecked_sub(new_pad_to_old);

                    shift.unchecked_add(padding_difference)
                }
            },
        );

        // Now we go in reverse, and copy.
        self.elems_info.array_windows::<2>().rev().fold(
            final_offset_shift,
            |shift_end,

             &[ElementInfo {
                 end_offset: prev_end_offset,
                 ..
             }, ElementInfo {
                 end_offset: new_end_offset,
                 ..
             }]| {
                // SAFETY:: See comments inside block
                unsafe {
                    // SAFETY: Reversing computation in the last loop.
                    let new_pad_to_new = new_end_offset.unchecked_pad_to(new_align);
                    let new_pad_to_old = new_end_offset.unchecked_pad_to(old_align);
                    let padding_difference = new_pad_to_new.unchecked_sub(new_pad_to_old);
                    let shift_start = shift_end.unchecked_sub(padding_difference);

                    let new_start_offset = prev_end_offset.unchecked_pad_to(new_align);
                    let old_start_offset = new_start_offset.unchecked_sub(shift_start);

                    // SAFETY: End offset >= start offset
                    let size_of_val = new_end_offset.unchecked_sub(new_start_offset);

                    // SAFETY: moving element to new correct position, as computed above
                    ptr::copy(
                        self.ptr.as_ptr().cast::<u8>().add(old_start_offset.get()),
                        self.ptr.as_ptr().cast::<u8>().add(new_start_offset.get()),
                        size_of_val.get(),
                    );
                    shift_start
                }
            },
        );
    }

    /// Realigns all elements in the vec to the given `new_align`,
    /// if the current align is greater.
    ///
    /// Opposite of `realign_up`.
    ///
    /// # Safety
    ///
    /// `new_align < self.align` must hold.
    ///  `new_align >= self.max_align_of_elems()` must hold.
    /// `self.len() > 1` must hold.
    ///
    /// This function does not shrink the allocation,
    /// you will need to do that yourself,
    /// *even if you don't change the allocated size*,
    /// to ensure that the allocation is later deallocated
    /// with the correct alignment.
    ///
    /// Finally, this function doesn't update `self.align`, do that yourself.
    unsafe fn realign_down(&mut self, new_align: ValidAlign) {
        let old_align = self.align;

        debug_assert!(self.len() > 1 && new_align < old_align);

        // We compute the new offset of each element, along with the difference from the old offset.
        // Then, we copy the eement over.
        //
        // This is a lot simpler than `realign_up`, we can do everyting in a single pass.

        let mut shift_back = ValidSizeUnaligned::ZERO;
        for &[ElementInfo {
            end_offset: prev_new_end_offset,
            ..
        }, ElementInfo {
            end_offset: old_end_offset,
            ..
        }] in self.elems_info.array_windows::<2>()
        {
            // SAFETY: shift must be smaller than size of allocation up to this point
            let new_end_offset = unsafe { old_end_offset.unchecked_sub(shift_back) };

            // SAFETY: can't overflow, or else unshifted allocation would have overflowed
            let new_start_offset = unsafe { prev_new_end_offset.unchecked_pad_to(new_align) };

            // SAFETY: shift must be smaller than size of allocation up to this point
            let old_start_offset = unsafe { new_start_offset.unchecked_sub(shift_back) };

            // SAFETY: End offset >= start offset
            let size_of_val = unsafe { new_end_offset.unchecked_sub(new_start_offset) };

            // SAFETY: moving element to new correct position, as computed above
            unsafe {
                ptr::copy(
                    self.ptr.as_ptr().cast::<u8>().add(old_start_offset.get()),
                    self.ptr.as_ptr().cast::<u8>().add(new_start_offset.get()),
                    size_of_val.get(),
                );
            }

            // SAFETY: pads can't overfolow as otherwise old offsets would be invalid.
            // Sub can't overflow as new_align < old_align.
            // Add can't overflow as we can't shift more than the entire size of the allocation.
            shift_back = unsafe {
                shift_back.unchecked_add(
                    new_end_offset
                        .unchecked_pad_to(old_align)
                        .unchecked_sub(new_end_offset.unchecked_pad_to(new_align)),
                )
            };
        }
    }
}

impl<T: ?Sized> Drop for UnalignedVecInner<T> {
    fn drop(&mut self) {
        let mut start_offset: ValidSizeUnaligned = ValidSizeUnaligned::ZERO;

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
            let metadata = metadata.as_metadata(size_of_val);
            let start_of_alloc = self.ptr.as_ptr().cast::<u8>();
            // SAFETY: offset is within allocation
            let thin_ptr_to_elem = unsafe { start_of_alloc.add(start_offset.get()) };
            let wide_ptr_to_elem: *mut T = ptr::from_raw_parts_mut(thin_ptr_to_elem, metadata);

            // SAFETY: align comes from the vec
            start_offset = unsafe { end_offset.unchecked_pad_to(self.align) };

            // SAFETY: nobody will access this after us
            unsafe { wide_ptr_to_elem.drop_in_place() }
        }

        // Drop allocation
        //
        // SAFETY: capacity and align come from the vec.
        unsafe {
            let alloc_layout = self
                .byte_capacity
                .as_layout_with_align_unchecked(self.align);
            alloc::Global.deallocate(self.ptr.cast(), alloc_layout);
        }
    }
}

impl<T: ?Sized> UnsizedVecProvider<T> for UnalignedVecInner<T> {
    type Align = ValidAlign;
    type Size = ValidSizeUnaligned;

    type Iter<'a> = UnalignedIter<'a, T> where T: 'a;
    type IterMut<'a> = UnalignedIterMut<'a, T> where T: 'a;

    const NEW_ALIGN_1: UnalignedVecInner<T> = UnalignedVecInner {
        ptr: <() as Aligned>::DANGLING_THIN,
        byte_capacity: ValidSizeUnaligned::ZERO,
        elems_info: ManuallyDrop::new(::alloc::vec::Vec::new()),
        align: <()>::ALIGN,
        _marker: PhantomData,
    };

    const NEW_ALIGN_PTR: UnalignedVecInner<T> = UnalignedVecInner {
        ptr: <usize as Aligned>::DANGLING_THIN,
        byte_capacity: ValidSizeUnaligned::ZERO,
        elems_info: ManuallyDrop::new(::alloc::vec::Vec::new()),
        align: <usize>::ALIGN,
        _marker: PhantomData,
    };

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
        self.align.into()
    }

    #[inline]
    fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        Ok(self.elems_info.try_reserve(additional)?)
    }

    #[inline]
    fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        Ok(self.elems_info.try_reserve_exact(additional)?)
    }

    fn try_reserve_additional_bytes_align(
        &mut self,
        additional_bytes: usize,
        align: ValidAlign,
    ) -> Result<(), TryReserveError> {
        let old_align = self.align;
        let new_align = cmp::max(old_align, align);

        let old_byte_cap = self.byte_capacity;
        let old_byte_len = self.aligned_byte_len();

        let byte_len_of_existing_elems_realigned: ValidSizeUnaligned = if old_align < new_align {
            // SAFETY: just checked `old_align < new_align`
            unsafe { self.len_after_realign_up(new_align) }
                // Return early if existing elems overflow `isize`
                // when realigned.
                .ok_or(TryReserveError {
                    kind: TryReserveErrorKind::CapacityOverflow,
                })?
        } else {
            if additional_bytes == 0 {
                return Ok(());
            }

            old_byte_len
        };

        let realignment_of_existing_elems_cost =
            // SAFETY: realignment can't make byte length shorter. 
            unsafe { byte_len_of_existing_elems_realigned.unchecked_sub(old_byte_len) };

        // Now we add on the additional size requested.
        let new_byte_cap = old_byte_cap
            .checked_add_pad(realignment_of_existing_elems_cost.get(), new_align)
            .and_then(|s| s.checked_add_pad(additional_bytes, new_align))
            .ok_or(TryReserveError {
                kind: TryReserveErrorKind::CapacityOverflow,
            })?;

        if old_align < new_align || old_byte_cap < new_byte_cap {
            if new_byte_cap > ValidSizeUnaligned::ZERO {
                // SAFETY: `new_cap` checked to be legal for following call in all branches above
                let new_layout = unsafe { new_byte_cap.as_layout_with_align_unchecked(new_align) };
                let new_ptr: NonNull<[u8]> = (if old_byte_cap == ValidSizeUnaligned::ZERO {
                    alloc::Global.allocate(new_layout)
                } else {
                    //  SAFETY: `old_cap` and `old_align` come from the vec
                    unsafe {
                        let old_layout = old_byte_cap.as_layout_with_align_unchecked(old_align);
                        alloc::Global.grow(self.ptr.cast(), old_layout, new_layout)
                    }
                })
                .map_err(|_| TryReserveError {
                    kind: TryReserveErrorKind::AllocError {
                        layout: new_layout,
                        non_exhaustive: (),
                    },
                })?;

                self.byte_capacity = ValidSizeUnaligned::new_squished_to(new_ptr.len(), new_align);
                self.ptr = new_ptr.cast();

                if old_align < new_align {
                    if self.len() > 1 {
                        // SAFETY: Just performed necessary allocation, if guard.
                        // Overflow covered by earlier checks.
                        unsafe { self.realign_up(new_align) };
                    }

                    self.align = new_align;
                }
            } else {
                self.ptr = new_align.dangling_thin();
            }
        }
        Ok(())
    }

    fn shrink_capacity_bytes_align_to(
        &mut self,
        min_capacity: usize,
        min_byte_capacity: usize,
        min_align: ValidAlign,
    ) {
        self.elems_info.shrink_to(min_capacity);

        let old_align = self.align;
        let new_align = cmp::max(cmp::min(min_align, old_align), self.max_align_of_elems());

        debug_assert!(new_align <= old_align);

        let need_to_realign_elems = new_align < old_align && self.len() > 1;
        if need_to_realign_elems {
            // SAFETY: checked len, new vs old, max_align_of_elems above
            unsafe { self.realign_down(new_align) }
        }

        // SAFETY: Can't overflow, otherwise old offsets would be invalids
        let new_aligned_byte_len = unsafe { self.unaligned_byte_len().unchecked_pad_to(new_align) };

        let old_byte_capacity = self.byte_capacity;
        let new_byte_capacity = cmp::max(
            // SAFETY: `old_byte_capacity` is a valid
            // `ValidSizeUnaligned`, and result of `cmp::min`
            // can't be bigger than it
            unsafe {
                ValidSizeUnaligned::new_unchecked(cmp::min(
                    min_byte_capacity,
                    old_byte_capacity.get(),
                ))
            },
            new_aligned_byte_len,
        );

        debug_assert!(new_byte_capacity <= old_byte_capacity);

        if new_byte_capacity < old_byte_capacity || new_align < old_align {
            // SAFETY: cap and align are valid as they come from the vec
            let old_layout = unsafe { old_byte_capacity.as_layout_with_align_unchecked(old_align) };

            if new_byte_capacity > ValidSizeUnaligned::ZERO {
                let new_layout =
                    // SAFETY: cap and align are <= old (valid) cap and align
                    unsafe { new_byte_capacity.as_layout_with_align_unchecked(new_align) };

                // `shrink` can unwind, in which case we need to make sure
                // we realign everything back to how it was.

                struct Realigner<'a, T: ?Sized> {
                    vec: &'a mut UnalignedVecInner<T>,
                    new_align: ValidAlign,
                }

                impl<'a, T: ?Sized> Drop for Realigner<'a, T> {
                    #[inline]
                    fn drop(&mut self) {
                        let old_align = self.vec.align;
                        self.vec.align = self.new_align;

                        // SAFETY: old_align > new_align, checked self.len(),
                        // adjusted `self.align`
                        unsafe { self.vec.realign_up(old_align) }

                        self.vec.align = old_align;
                    }
                }

                let alloc_ptr = self.ptr.cast();

                // https://github.com/rust-lang/rust-clippy/issues/9427
                #[allow(clippy::unnecessary_lazy_evaluations)]
                let realigner = need_to_realign_elems.then(|| Realigner {
                    vec: self,
                    new_align,
                });

                let shrink_result =
                    // SAFETY: cap and align are <= old (valid) cap and align.
                    // old layout and ptr come from the vec.
                    unsafe { alloc::Global.shrink(alloc_ptr, old_layout, new_layout) };
                let Ok(new_ptr) = shrink_result else {
                    // `realigner` will be dropped, restoring offsets
                    return;
                };

                mem::forget(realigner);

                self.byte_capacity = ValidSizeUnaligned::new_squished_to(new_ptr.len(), new_align);
                self.ptr = new_ptr.cast();
            } else {
                if old_byte_capacity > ValidSizeUnaligned::ZERO {
                    // SAFETY: `old_layout` components come from the vec
                    unsafe { alloc::Global.deallocate(self.ptr.cast(), old_layout) }

                    self.byte_capacity = ValidSizeUnaligned::ZERO;
                }

                self.ptr = new_align.dangling_thin();
            }

            self.align = new_align;
        }
    }

    unsafe fn insert_unchecked(
        &mut self,
        index: usize,
        element: T,
        unaligned_size_of_val: ValidSizeUnaligned,
    ) {
        debug_assert!(index <= self.len());
        debug_assert!(self.capacity() > self.len());

        // SAFETY: preconditions of function
        let aligned_size_of_val = unsafe { unaligned_size_of_val.unchecked_pad_to(self.align) };

        debug_assert!(
            self.byte_capacity() >= (self.aligned_byte_len().get() + aligned_size_of_val.get())
        );
        debug_assert!(self.align() >= mem::align_of_val(&element));

        let metadata =
            <T as SplitMetadata>::Remainder::from_metadata(core::ptr::metadata(&element));

        // SAFETY: preconditions of function
        unsafe {
            let start_offset = self.start_offset_of_unchecked(index);
            let how_much_to_move = self.unaligned_byte_len().unchecked_sub(start_offset);

            let start_ptr = self.ptr.cast::<u8>().as_ptr().add(index);

            ptr::copy(
                start_ptr,
                start_ptr.add(aligned_size_of_val.get()),
                how_much_to_move.get(),
            );

            ptr::copy_nonoverlapping(
                addr_of!(element).cast(),
                start_ptr,
                unaligned_size_of_val.get(),
            );

            for ElementInfo { end_offset, .. } in self.elems_info.get_unchecked_mut(index..) {
                *end_offset = end_offset.unchecked_add(aligned_size_of_val);
            }

            self.elems_info.insert_unchecked(
                index,
                ElementInfo {
                    metadata,
                    end_offset: start_offset.unchecked_add(unaligned_size_of_val),
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
        struct ElementShifterBacker {
            ptr_to_index: *mut u8,
            num_bytes_to_shift: ValidSizeUnaligned,
            shift_by_bytes: ValidSizeUnaligned,
        }

        impl Drop for ElementShifterBacker {
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
            &mut |layout, metadata, inner_closure: &mut dyn FnMut(*mut PhantomData<T>)| {
                let (unaligned_size_of_val, align_of_val) = decompose(layout);

                let reserve_result = self.try_reserve(1).and_then(|()| {
                    self.try_reserve_additional_bytes_align(
                        unaligned_size_of_val.get(),
                        align_of_val,
                    )
                });
                unwrap_try_reserve_result(reserve_result);

                let aligned_size_of_val =
                    // SAFETY: `try_reserve` would have failed if this could fail
                    unsafe { unaligned_size_of_val.unchecked_pad_to(self.align) };

                // SAFETY: precondition of function
                let start_offset = unsafe { self.start_offset_of_unchecked(index) };

                // SAFETY: getting pointer to element
                let ptr_to_elem = unsafe { self.ptr.cast::<u8>().as_ptr().add(start_offset.get()) };

                let unaligned_len = self.unaligned_byte_len();

                // SAFETY: by precondition of function
                let num_bytes_to_shift = unsafe { unaligned_len.unchecked_sub(start_offset) };

                let shifter_backer = ElementShifterBacker {
                    ptr_to_index: ptr_to_elem,
                    num_bytes_to_shift,
                    shift_by_bytes: aligned_size_of_val,
                };

                // SAFETY: copying elements right to make room
                unsafe {
                    ptr::copy(
                        ptr_to_elem,
                        ptr_to_elem.add(aligned_size_of_val.get()),
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
                    *end_offset = unsafe { end_offset.unchecked_add(aligned_size_of_val) };
                }

                // SAFETY: reserved memory earlier
                unsafe {
                    self.elems_info.insert_unchecked(
                        index,
                        ElementInfo {
                            metadata: <T as SplitMetadata>::Remainder::from_metadata(metadata),
                            end_offset: start_offset.unchecked_add(unaligned_size_of_val),
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
        let unaligned_size_of_val = unsafe { end_offset.unchecked_sub(start_offset) };

        // SAFETY: `val` comes from the vec so must be paddable
        let aligned_size_of_val = unsafe { unaligned_size_of_val.unchecked_pad_to(self.align) };

        let metadata = metadata.as_metadata(unaligned_size_of_val);

        // Get pointer to the element we are popping out of the vec
        // SAFETY: offset comes from vec
        let ptr_to_elem = unsafe {
            self.ptr
                .as_ptr()
                .cast_const()
                .cast::<u8>()
                .add(start_offset.get())
        };

        let wide_ptr_to_elem: *const T = ptr::from_raw_parts(ptr_to_elem, metadata);

        // SAFETY: the element is still initialized at this point
        let align_of_val = ValidAlign::of_val(unsafe { &*wide_ptr_to_elem });

        // Copy element into the place

        // SAFETY: we call the closure right after we unwrap it
        let emplacer_closure = unsafe { emplacer.into_fn() };

        emplacer_closure(
            // SAFETY: `size_of_val` comes from the vec
            unsafe { unaligned_size_of_val.as_layout_with_align_unchecked(align_of_val) },
            metadata,
            &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: we are allowed to copy `size_of_val` bytes into `out_ptr`,
                    // by the preconditions of `Emplacer::new`
                    unsafe {
                        ptr::copy_nonoverlapping(
                            ptr_to_elem,
                            out_ptr.cast::<u8>(),
                            unaligned_size_of_val.get(),
                        );
                    }
                } else {
                    // SAFETY: we adjust vec metadata right after, so this won't be double-dropped
                    unsafe { wide_ptr_to_elem.cast_mut().drop_in_place() }
                }
            },
        );

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
                *end_offset = end_offset.unchecked_sub(aligned_size_of_val);
            }
        }

        let unaligned_len = self.unaligned_byte_len();

        // SAFETY: new end of vec can't be to the left of old start of elem at `index`
        let how_much_to_move = unsafe { unaligned_len.unchecked_sub(start_offset) };

        // SAFETY: copying elements back where they belong
        unsafe {
            ptr::copy(
                ptr_to_elem.add(aligned_size_of_val.get()),
                ptr_to_elem.cast_mut(),
                how_much_to_move.get(),
            );
        }
    }

    #[inline]
    unsafe fn push_unchecked(&mut self, value: T, size_of_val: ValidSizeUnaligned) {
        debug_assert!(self.capacity() - self.len() > 0);

        debug_assert!(self.byte_capacity() >= (self.aligned_byte_len().get() + size_of_val.get()));
        debug_assert!(self.align() >= mem::align_of_val(&value));

        let metadata = <T as SplitMetadata>::Remainder::from_metadata(core::ptr::metadata(&value));
        let start_offset = self.aligned_byte_len();

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
                let (size_of_val, align_of_val) = decompose(layout);

                let reserve_result = self.try_reserve(1).and_then(|()| {
                    self.try_reserve_additional_bytes_align(layout.size(), align_of_val)
                });
                unwrap_try_reserve_result(reserve_result);

                let start_offset = self.aligned_byte_len();

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

    #[inline]
    unsafe fn pop_into_unchecked(&mut self, emplacer: &mut Emplacer<'_, T>) {
        debug_assert!(!self.elems_info.is_empty());

        // SAFETY: precondition of function
        let last_elem_metadata = unsafe { self.elems_info.pop().unwrap_unchecked() };

        let ElementInfo {
            metadata,
            end_offset,
        } = last_elem_metadata;

        let start_offset = self.aligned_byte_len();

        // SAFETY: start_offset < end_offset
        let size_of_val = unsafe { end_offset.unchecked_sub(start_offset) };

        let metadata = metadata.as_metadata(size_of_val);

        // Get pointer to the element we are popping out of the vec
        // SAFETY: offset comes from vec
        let ptr_to_elem = unsafe {
            self.ptr
                .as_ptr()
                .cast_const()
                .cast::<u8>()
                .add(start_offset.get())
        };

        let wide_ptr_to_elem: *const T = ptr::from_raw_parts(ptr_to_elem, metadata);

        // SAFETY: the element is still initialized at this point
        let align_of_val = ValidAlign::of_val(unsafe { &*wide_ptr_to_elem });

        // Copy element into the place

        // SAFETY: we call the closure right after we unwrap it
        let emplace_closure = unsafe { emplacer.into_fn() };

        emplace_closure(
            // SAFETY: `size_of_val` comes from the vec
            unsafe { size_of_val.as_layout_with_align_unchecked(align_of_val) },
            metadata,
            &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: we are allowed to copy `size_of_val` bytes into `out_ptr`,
                    // by the preconditions of `Emplacer::new`
                    unsafe {
                        ptr::copy_nonoverlapping(
                            ptr_to_elem,
                            out_ptr.cast::<u8>(),
                            size_of_val.get(),
                        );
                    }
                } else {
                    // SAFETY: we adjusted vec metadata earlier, so this won't be double-dropped
                    unsafe { wide_ptr_to_elem.cast_mut().drop_in_place() }
                }
            },
        );
    }

    #[inline]
    fn len(&self) -> usize {
        self.elems_info.len()
    }

    #[inline]
    fn byte_len(&self) -> usize {
        self.aligned_byte_len().get()
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
            let metadata = metadata.as_metadata(size_of_val);

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
        UnalignedIter {
            elems_info: self.elems_info.iter(),
            ptr: self.ptr,
            start_offset: ValidSizeUnaligned::ZERO,
            align: self.align,
        }
    }

    #[inline]
    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        UnalignedIterMut {
            elems_info: self.elems_info.iter(),
            ptr: self.ptr,
            start_offset: ValidSizeUnaligned::ZERO,
            align: self.align,
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
        let byte_capacity = unsafe { ValidSizeUnaligned::new_unchecked(byte_capacity) };

        let elems_info = (0..len_elems)
            .map(|index| ElementInfo {
                metadata,
                // SAFETY: can't overflow, as otherwise allocation would be overflowing
                end_offset: unsafe {
                    ValidSizeUnaligned::new_unchecked(index.unchecked_mul(mem::size_of::<S>()))
                },
            })
            .collect();

        let elems_info = ManuallyDrop::new(elems_info);

        Self {
            ptr: heap_ptr_thin,
            byte_capacity,
            elems_info,
            align: S::ALIGN,
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized> UnsizedVecImpl for T {
    default type Impl = UnalignedVecInner<T>;
}

macro_rules! iter_ref {
    ($iter_ty:ident, $from_raw_parts:ident $($muta:ident)?) => {
        pub(in super::super) struct $iter_ty<'a, T: ?Sized> {
            elems_info: core::slice::Iter<'a, ElementInfo<T>>,
            ptr: NonNull<()>,
            start_offset: ValidSizeUnaligned,
            align: ValidAlign,
        }

        impl<'a, T: ?Sized + 'a> Iterator for $iter_ty<'a, T> {
            type Item = &'a $($muta)? T;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                let ElementInfo {
                    metadata,
                    end_offset,
                } = *self.elems_info.next()?;

                // SAFETY: end of element can't be smaller than start
                let size_of_val = unsafe { end_offset.unchecked_sub(self.start_offset) };
                let metadata = metadata.as_metadata(size_of_val);

                let start_of_alloc = self.ptr.as_ptr().cast::<u8>();
                // SAFETY: offset is within allocation
                let thin_ptr_to_elem = unsafe { start_of_alloc.add(self.start_offset.get()) };
                let wide_ptr = ptr::$from_raw_parts(thin_ptr_to_elem, metadata);

                // SAFETY: pointer to element of vec
                let wide_ref = unsafe { & $($muta)? *wide_ptr };

                // SAFETY: align comes from the vec
                self.start_offset = unsafe { end_offset.unchecked_pad_to(self.align) };

                Some(wide_ref)
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.elems_info.size_hint()
            }

            #[inline]
            fn count(self) -> usize
            where
                Self: Sized,
            {
                self.elems_info.count()
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<Self::Item> {
                let start_offset = n
                    .checked_sub(1)
                    .and_then(|n| self.elems_info.nth(n))
                    .copied()
                    // SAFETY: offset comes from the vec
                    .map_or(ValidSizeUnaligned::ZERO, |e_i| unsafe {
                        e_i.end_offset.unchecked_pad_to(self.align)
                    });

                let ElementInfo {
                    metadata,
                    end_offset,
                } = *self.elems_info.next()?;

                // SAFETY: end of element can't be smaller than start`
                let size_of_val = unsafe { end_offset.unchecked_sub(start_offset) };
                let metadata = metadata.as_metadata(size_of_val);

                let start_of_alloc = self.ptr.as_ptr().cast::<u8>();
                // SAFETY: offset is within allocation
                let thin_ptr_to_elem = unsafe { start_of_alloc.add(start_offset.get()) };
                let wide_ptr = ptr::$from_raw_parts(thin_ptr_to_elem, metadata);

                // SAFETY: pointer to element of vec
                let wide_ref = unsafe { & $($muta)? *wide_ptr };

                // SAFETY: offset comes from the vec
                self.start_offset = unsafe { end_offset.unchecked_pad_to(self.align) };

                Some(wide_ref)
            }

            #[inline]
            fn last(mut self) -> Option<Self::Item>
            where
                Self: Sized,
            {
                self.nth(self.elems_info.len().checked_sub(1)?)
            }
        }

        impl<'a, T: ?Sized + 'a> DoubleEndedIterator for $iter_ty<'a, T> {
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
                    // SAFETY: offset comes from the vec
                    .map_or(ValidSizeUnaligned::ZERO, |e_i| unsafe {
                        e_i.end_offset.unchecked_pad_to(self.align)
                    });

                // SAFETY: end of element can't be smaller than start
                let size_of_val = unsafe { end_offset.unchecked_sub(start_offset) };
                let metadata = metadata.as_metadata(size_of_val);

                let start_of_alloc = self.ptr.as_ptr().cast::<u8>();
                // SAFETY: offset is within allocation
                let thin_ptr_to_elem = unsafe { start_of_alloc.add(start_offset.get()) };
                let wide_ptr = ptr::$from_raw_parts(thin_ptr_to_elem, metadata);

                // SAFETY: pointer to element of vec
                let wide_ref = unsafe { & $($muta)? *wide_ptr };

                Some(wide_ref)
            }
        }

        impl<'a, T: ?Sized + 'a> ExactSizeIterator for $iter_ty<'a, T> {
            #[inline]
            fn len(&self) -> usize {
                self.elems_info.len()
            }
        }

        impl<'a, T: ?Sized + 'a> FusedIterator for $iter_ty<'a, T> {}
    };
}

iter_ref!(UnalignedIter, from_raw_parts);
iter_ref!(UnalignedIterMut, from_raw_parts_mut mut);
