//! The implementation of `UnsizedVec<T>` for `T: Sized`.

use core::{
    alloc::Layout,
    cmp,
    marker::{PhantomData, Unsize},
    mem,
    ptr::{self, NonNull},
    slice::{Iter, IterMut},
};

use emplacable::{Emplacable, EmplacableFn, Emplacer};

use crate::unwrap_try_reserve_result;

use super::{AlignedVecImpl, AlignedVecProvider, TryReserveError, UnsizedVecProvider};

impl<T> UnsizedVecProvider<T> for ::alloc::vec::Vec<T> {
    type Align = ();
    type Size = ();

    type Iter<'a>
        = Iter<'a, T>
    where
        T: 'a;
    type IterMut<'a>
        = IterMut<'a, T>
    where
        T: 'a;

    const NEW_ALIGN_1: Self = ::alloc::vec::Vec::new();
    const NEW_ALIGN_PTR: Self = Self::NEW_ALIGN_1;

    #[inline]
    fn capacity(&self) -> usize {
        self.capacity()
    }

    #[inline]
    fn byte_capacity(&self) -> usize {
        // SAFETY: capacity can't overflow `isize::MAX` bytes
        unsafe { self.capacity().unchecked_mul(mem::size_of::<T>()) }
    }

    #[inline]
    fn byte_len(&self) -> usize {
        // SAFETY: capacity can't overflow `isize::MAX` bytes
        unsafe { self.len().unchecked_mul(mem::size_of::<T>()) }
    }

    #[inline]
    fn align(&self) -> usize {
        mem::align_of::<T>()
    }

    #[inline]
    fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        Ok(self.try_reserve(additional)?)
    }

    #[inline]
    fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        Ok(self.try_reserve_exact(additional)?)
    }

    #[inline]
    fn try_reserve_additional_bytes_align(
        &mut self,
        additional_bytes: usize,
        _align: (),
    ) -> Result<(), TryReserveError> {
        // SAFETY: capacity >= len
        let free = unsafe { self.capacity().unchecked_sub(self.len()) };
        let needed = additional_bytes
            .saturating_add(free)
            .div_ceil(mem::size_of::<T>());
        Ok(self.try_reserve_exact(needed.saturating_sub(free))?)
    }

    #[inline]
    fn shrink_capacity_bytes_align_to(
        &mut self,
        min_capacity: usize,
        min_byte_capacity: usize,
        _align: (),
    ) {
        let min_capacity = cmp::max(
            min_capacity,
            min_byte_capacity.div_ceil(mem::size_of::<T>()),
        );
        self.shrink_to(min_capacity);
    }

    #[inline]
    unsafe fn insert_unchecked(&mut self, index: usize, element: T, _size: ()) {
        debug_assert!(self.capacity() > self.len());

        // SAFETY: precondition of the function
        unsafe {
            let how_much_to_move = self.len().unchecked_sub(index);
            let start_ptr = self.as_mut_ptr().add(index);
            // shift back elems to the right of pointer
            ptr::copy(start_ptr, start_ptr.add(1), how_much_to_move);
            start_ptr.write(element);
            self.set_len(self.len().unchecked_add(1));
        }
    }

    unsafe fn insert_with_unchecked(
        &mut self,
        index: usize,
        value: Emplacable<T, impl EmplacableFn<T>>,
    ) {
        /// Helper to ensure elements are moved back
        /// where they belong in case `inner_closure`
        /// panics.
        struct ElementShifterBacker<T> {
            ptr_to_index: *mut T,
            num_elems_to_shift: usize,
        }

        impl<T> Drop for ElementShifterBacker<T> {
            #[inline]
            fn drop(&mut self) {
                // SAFETY: shifting elements back in case of drops
                unsafe {
                    ptr::copy(
                        self.ptr_to_index.add(1),
                        self.ptr_to_index,
                        self.num_elems_to_shift,
                    );
                }
            }
        }

        debug_assert!(index <= self.len());

        let emplacable_closure = value.into_fn();

        let emplacer_closure =
            &mut move |_, (), inner_closure: &mut dyn FnMut(*mut PhantomData<T>)| {
                let reserve_result = <Self as UnsizedVecProvider<T>>::try_reserve_exact(self, 1);
                unwrap_try_reserve_result(reserve_result);

                // SAFETY: by precondition of function
                let ptr_to_elem = unsafe { self.as_mut_ptr().add(index) };

                // SAFETY: by precondition of function
                let num_elems_to_shift = unsafe { self.len().unchecked_sub(index) };

                let shifter_backer: ElementShifterBacker<T> = ElementShifterBacker {
                    ptr_to_index: ptr_to_elem,
                    num_elems_to_shift,
                };

                // SAFETY: copying back elements to make room
                unsafe { ptr::copy(ptr_to_elem, ptr_to_elem.add(1), num_elems_to_shift) }

                // If this unwinds, `shifter_backer` will be dropped
                // and the elements will be moved back where they belong.
                inner_closure(ptr_to_elem.cast());

                // `inner_closure` succeeded, so don't want to move elements back now!
                mem::forget(shifter_backer);

                // SAFETY: `inner_closure` wrote new element at the correct index,
                // elems to the right were shifted back
                unsafe { self.set_len(self.len().unchecked_add(1)) };
            };

        // SAFETY: `emplacer_closure` runs the closure with a valid pointer to `index`
        let emplacer = unsafe { Emplacer::from_fn(emplacer_closure) };

        emplacable_closure(emplacer);
    }

    unsafe fn remove_into_unchecked(&mut self, index: usize, emplacer: &mut Emplacer<'_, T>) {
        debug_assert!(index < self.len());

        // SAFETY: precondition of function
        let new_len = unsafe { self.len().unchecked_sub(1) };

        // Set new length of vector

        // SAFETY: new_len < old_len
        unsafe { self.set_len(new_len) };

        // Get pointer to the element we are popping out of the vec
        // SAFETY: offset comes from vec
        let thin_ptr_to_elem = unsafe { self.as_ptr().add(index) };

        // SAFETY: old_len > index, so old_len - 1 == new_len >= index
        let how_much_to_move = unsafe { new_len.unchecked_sub(index) };

        // Copy element into the place

        // SAFETY: we call the closure right after we unwrap it
        let emplacer_closure = unsafe { emplacer.into_fn() };

        emplacer_closure(Layout::new::<T>(), (), &mut |out_ptr| {
            if !out_ptr.is_null() {
                // SAFETY: we are allowed to copy `size_of::<T>()` bytes into `out_ptr`,
                // by the preconditions of `Emplacer::new`
                unsafe {
                    ptr::copy_nonoverlapping(thin_ptr_to_elem, out_ptr.cast(), 1);
                }
            } else {
                let typed_ptr_to_elem: *mut T = thin_ptr_to_elem.cast_mut().cast();

                // SAFETY: we adusted vec metadata earlier, and copy elements back right after,
                // so this won't be double-dropped
                unsafe { typed_ptr_to_elem.drop_in_place() }
            }
        });

        // SAFETY: copy elements back where they belong
        unsafe {
            ptr::copy(
                thin_ptr_to_elem.add(1),
                thin_ptr_to_elem.cast_mut(),
                how_much_to_move,
            );
        }
    }

    #[inline]
    unsafe fn push_unchecked(&mut self, value: T, _size: ()) {
        debug_assert!(self.capacity() - self.len() > 0);

        let old_len = self.len();
        // SAFETY: precondition of the function
        unsafe {
            self.as_mut_ptr().add(old_len).write(value);
            self.set_len(old_len.unchecked_add(1));
        }
    }

    fn push_with(&mut self, value: Emplacable<T, impl EmplacableFn<T>>) {
        let emplacable_closure = value.into_fn();

        let emplacer_closure =
            &mut move |_, (), inner_closure: &mut dyn FnMut(*mut PhantomData<T>)| {
                let reserve_result = <Self as UnsizedVecProvider<T>>::try_reserve(self, 1);
                unwrap_try_reserve_result(reserve_result);
                // SAFETY: getting pointer to end of allocation
                let ptr_to_elem = unsafe { self.as_mut_ptr().add(self.len()) };

                inner_closure(ptr_to_elem.cast());

                // SAFETY: `inner_closure` wrote new element at end of vec
                unsafe { self.set_len(self.len().unchecked_add(1)) };
            };

        // SAFETY: `emplacer_closure` runs the closure with a valid pointer to the end of the vec
        let emplacer = unsafe { Emplacer::from_fn(emplacer_closure) };

        emplacable_closure(emplacer);
    }

    unsafe fn pop_into_unchecked(&mut self, emplacer: &mut Emplacer<'_, T>) {
        debug_assert!(!self.is_empty());

        // Set new length of vector

        // SAFETY: precondition of function
        let new_len = unsafe { self.len().unchecked_sub(1) };

        // SAFETY: new_len < old_len
        unsafe { self.set_len(new_len) };

        // Get pointer to the element we are popping out of the vec
        // SAFETY: offset comes from vec
        let ptr_to_elem = unsafe { self.as_ptr().add(new_len) };

        // Copy element into the place

        // SAFETY: we call the closure right after we unwrap it
        let emplace_closure = unsafe { emplacer.into_fn() };

        emplace_closure(Layout::new::<T>(), (), &mut |out_ptr| {
            if !out_ptr.is_null() {
                // SAFETY: we are allowed to copy `size_of::<T>()` bytes into `out_ptr`,
                // by the preconditions of `Emplacer::new`
                unsafe {
                    ptr::copy_nonoverlapping(ptr_to_elem, out_ptr.cast(), 1);
                }
            } else {
                let typed_ptr_to_elem: *mut T = ptr_to_elem.cast_mut().cast();

                // SAFETY: we adusted vec metadata earlier, so this won't be double-dropped
                unsafe { typed_ptr_to_elem.drop_in_place() }
            }
        });
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    unsafe fn get_unchecked_raw(&self, index: usize) -> NonNull<T> {
        debug_assert!(index < self.len());

        // SAFETY: precondition of method
        unsafe { NonNull::new_unchecked(self.as_ptr().add(index).cast_mut()).cast() }
    }

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        let slice: &[T] = self;
        slice.iter()
    }

    #[inline]
    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        let slice: &mut [T] = self;
        slice.iter_mut()
    }

    fn from_sized<S>(_: ::alloc::vec::Vec<S>) -> Self
    where
        S: Unsize<T>,
    {
        unreachable!("can't unsize to a sized type, that would make 0 sense")
    }
}

impl<T> AlignedVecProvider<T> for ::alloc::vec::Vec<T> {}

impl<T> AlignedVecImpl for T {
    type Impl = ::alloc::vec::Vec<T>;
}
