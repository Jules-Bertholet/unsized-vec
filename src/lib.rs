#![deny(unsafe_op_in_unsafe_fn)]
#![allow(incomplete_features)] // For `specialization`
#![feature(
    forget_unsized,
    ptr_metadata,
    // We avoid specializing based on subtyping,
    // so should be sound.
    specialization,
    strict_provenance,
    unsized_fn_params
)]
#![no_std]

extern crate alloc as alloc_crate;

use alloc_crate::alloc::{self, Layout};
use core::{
    cmp,
    hash::Hash,
    marker::PhantomData,
    mem,
    ops::{Index, IndexMut},
    ptr::{self, NonNull, Pointee},
};

pub mod emplace;
use emplace::*;

pub mod marker;

use marker::{Aligned, MetadataRemainder, SplitMetadata};

/// Used by `UnsizedVec` for space optimization for storing sized values.
/// See module `marker` for more examples of this pattern.
trait OffsetNextRemainder<T: ?Sized>: Copy + Send + Sync + Ord + Hash + Unpin {
    #[must_use]
    fn from_offset_next(offset_next: usize) -> Self;

    #[must_use]
    fn to_offset_next(self, index: usize) -> usize;

    #[must_use]
    fn add(self, add: usize) -> Self;

    #[must_use]
    fn sub(self, sub: usize) -> Self;
}

impl<T: ?Sized> OffsetNextRemainder<T> for usize {
    #[inline]
    fn from_offset_next(offset_next: usize) -> Self {
        offset_next
    }

    #[inline]
    fn to_offset_next(self, _: usize) -> usize {
        self
    }

    #[inline]
    fn add(self, add: usize) -> Self {
        self + add
    }

    #[inline]
    fn sub(self, sub: usize) -> Self {
        self - sub
    }
}

impl<T> OffsetNextRemainder<T> for () {
    #[inline]
    fn from_offset_next(_: usize) -> Self {}

    #[inline]
    fn to_offset_next(self, index: usize) -> usize {
        mem::size_of::<T>() * index
    }

    #[inline]
    fn add(self, _: usize) {}

    #[inline]
    fn sub(self, _: usize) {}
}

trait SplitOffsetNext {
    type Remainder: OffsetNextRemainder<Self>;
}

impl<T: ?Sized> SplitOffsetNext for T {
    default type Remainder = usize;
}

impl<T> SplitOffsetNext for T {
    type Remainder = ();
}

struct MetadataStorage<T: ?Sized> {
    metadata_remainder: <T as SplitMetadata>::Remainder,
    offset_next_remainder: <T as SplitOffsetNext>::Remainder,
}

pub struct UnsizedVec<T: ?Sized + Aligned> {
    ptr: NonNull<()>,
    cap: usize,
    metadata: alloc_crate::vec::Vec<MetadataStorage<T>>,
    _marker: PhantomData<(*mut T, T)>,
}

unsafe impl<T: Send> Send for UnsizedVec<T> {}
unsafe impl<T: Sync> Sync for UnsizedVec<T> {}

impl<T: ?Sized + Aligned> UnsizedVec<T> {
    #[must_use]
    /// Make a new, empty `UnsizedVec`
    pub fn new() -> Self {
        UnsizedVec {
            ptr: <T as Aligned>::dangling_thin(),
            cap: 0,
            metadata: Default::default(),
            _marker: PhantomData,
        }
    }

    /// Grow this `UnsizedVec`'s allocation by at least `by_at_least` bytes.
    fn grow(&mut self, by_at_least: usize) {
        // since we set the capacity to usize::MAX when T has size 0,
        // getting to here necessarily means the UnsizedVec is overfull.

        let new_cap = cmp::max(2 * self.cap, by_at_least);
        let new_layout = Layout::from_size_align(new_cap, <T as Aligned>::ALIGN).unwrap();

        // Ensure that the new allocation doesn't exceed `isize::MAX` bytes.
        assert!(
            isize::try_from(new_layout.size()).is_ok(),
            "Allocation too large"
        );

        let new_ptr = if self.cap == 0 {
            unsafe { alloc::alloc(new_layout) }
        } else {
            let old_layout = Layout::from_size_align(self.cap, <T as Aligned>::ALIGN).unwrap();
            let old_ptr = self.ptr.as_ptr().cast::<u8>();
            unsafe { alloc::realloc(old_ptr, old_layout, new_layout.size()) }
        };

        // If allocation fails, `new_ptr` will be null, in which case we abort.
        self.ptr = match NonNull::new(new_ptr.cast::<()>()) {
            Some(p) => p,
            None => alloc::handle_alloc_error(new_layout),
        };
        self.cap = new_cap;
    }

    /// Get the offset of one byte past the end of the last element in th vec.
    #[inline]
    fn offset_next(&self) -> usize {
        self.metadata.last().map_or(0, |m| {
            m.offset_next_remainder.to_offset_next(self.len() - 1)
        })
    }

    /// Add element to end of the vec.
    pub fn push(&mut self, elem: T) {
        let size_of_val = mem::size_of_val(&elem);
        let offset_next = self.offset_next();
        let new_end_offset = size_of_val + self.offset_next();

        // Make sure we have enough memory
        self.metadata.reserve(1);
        if new_end_offset >= self.cap {
            self.grow(size_of_val);
        }

        let metadata = ptr::metadata(&elem);

        // Copy element to end of allocation.
        // We checked that we had enough memory earlier
        unsafe {
            ptr::copy_nonoverlapping(
                core::ptr::addr_of!(elem).cast(),
                self.ptr.as_ptr().cast::<u8>().add(offset_next),
                size_of_val,
            );
        }

        // elem is owned by the vec now, don't want to double-drop it
        mem::forget_unsized(elem);

        self.metadata.push(MetadataStorage {
            metadata_remainder: <T as SplitMetadata>::Remainder::from_metadata(metadata),
            offset_next_remainder: <T as SplitOffsetNext>::Remainder::from_offset_next(
                new_end_offset,
            ),
        });
    }

    /// Get the offset of one byte past the end of element index - 1 in the vec.
    #[inline]
    fn offset_start_idx(&self, index: usize) -> usize {
        index.checked_sub(1).map_or(0, |i| {
            self.metadata[i].offset_next_remainder.to_offset_next(i)
        })
    }

    /// Insert element into vec at given index, shifting following elements to the right.
    pub fn insert(&mut self, index: usize, elem: T) {
        let size_of_val = mem::size_of_val(&elem);
        let old_end_offset = self.offset_next();
        let new_end_offset = size_of_val + old_end_offset;

        // Make sure we have enough memory
        self.metadata.reserve(1);
        if new_end_offset >= self.cap {
            self.grow(size_of_val);
        }

        let metadata = ptr::metadata(&elem);

        // Update offsets of follwing elements
        for ms in &mut self.metadata[index..] {
            ms.offset_next_remainder = ms.offset_next_remainder.add(size_of_val);
        }

        let offset_start = self.offset_start_idx(index);

        // Shift roght and copy element to end of allocation.
        // We checked that we had enough memory earlier
        unsafe {
            ptr::copy(
                self.ptr.as_ptr().cast::<u8>().add(offset_start),
                self.ptr
                    .as_ptr()
                    .cast::<u8>()
                    .add(offset_start)
                    .add(size_of_val),
                old_end_offset - offset_start,
            );

            ptr::copy_nonoverlapping(
                core::ptr::addr_of!(elem).cast(),
                self.ptr.as_ptr().cast::<u8>().add(offset_start),
                size_of_val,
            );
        }

        // elem is owned by the vec now, don't want to double-drop it
        mem::forget_unsized(elem);

        self.metadata.insert(
            index,
            MetadataStorage {
                metadata_remainder: <T as SplitMetadata>::Remainder::from_metadata(metadata),
                offset_next_remainder: <T as SplitOffsetNext>::Remainder::from_offset_next(
                    offset_start + size_of_val,
                ),
            },
        );
    }

    // Remove an element from the end of the vec.
    // This returns the element, which is unsized, through the "emplacer" mechanism.
    // Call `box_new_with(|e| your_unsized_vec.pop_unwrap(e))` to get the element boxed.
    //
    // # Panics
    //
    // Panics if len is 0; unfortunately it's not possible to have unsized values inside enums
    // in Rust, so we can't return `Option`.
    pub fn pop_unwrap(&mut self, emplacer: &mut Emplacer<T>) {
        // Panics if len is 0.
        let MetadataStorage {
            metadata_remainder,
            offset_next_remainder: offset_end_remainder,
        } = self.metadata.pop().unwrap();
        let offset_start = self.offset_next();
        let offset_end = offset_end_remainder.to_offset_next(self.metadata.len());
        let size_of_val = offset_end - offset_start;
        let metadata = metadata_remainder.to_metadata(size_of_val);
        let emplace_fn = unsafe { emplacer.into_inner() };
        emplace_fn(
            Layout::from_size_align(size_of_val, T::ALIGN).unwrap(),
            metadata,
            &mut |dst| unsafe {
                ptr::copy_nonoverlapping(
                    self.ptr.as_ptr().cast::<u8>().add(offset_start),
                    dst.cast::<u8>(),
                    size_of_val,
                );
            },
        )
    }

    // Remove the element at the given index, shifting those after it to the left.
    // This returns the element, which is unsized, through the "emplacer" mechanism.
    // Call `box_new_with(|e| your_unsized_vec.remove(index, e))` to get the element boxed.
    //
    // # Panics
    //
    // Panics if index is out of bounds.
    pub fn remove(&mut self, index: usize, emplacer: &mut Emplacer<T>) {
        let MetadataStorage {
            metadata_remainder,
            offset_next_remainder: offset_end_remainder,
        } = self.metadata.remove(index);
        let offset_start = self.offset_start_idx(index);
        let offset_end = offset_end_remainder.to_offset_next(index);
        let size_of_val = offset_end - offset_start;
        let metadata = metadata_remainder.to_metadata(size_of_val);

        // Update offsets of follwing elements
        for ms in &mut self.metadata[index..] {
            ms.offset_next_remainder = ms.offset_next_remainder.sub(size_of_val);
        }

        let emplace_fn = unsafe { emplacer.into_inner() };
        emplace_fn(
            Layout::from_size_align(size_of_val, T::ALIGN).unwrap(),
            metadata,
            &mut |dst| unsafe {
                ptr::copy_nonoverlapping(
                    self.ptr.as_ptr().cast::<u8>().add(offset_start),
                    dst.cast::<u8>(),
                    size_of_val,
                );

                ptr::copy(
                    self.ptr
                        .as_ptr()
                        .cast::<u8>()
                        .add(offset_start)
                        .add(size_of_val),
                    dst.cast::<u8>().add(offset_start),
                    self.offset_next() - offset_start,
                );
            },
        )
    }

    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    #[must_use]
    #[inline]
    fn index_raw(&self, index: usize) -> (*mut (), <T as Pointee>::Metadata) {
        let MetadataStorage {
            metadata_remainder,
            offset_next_remainder,
        } = self.metadata[index];
        let offset = self.offset_start_idx(index);
        let offset_next = offset_next_remainder.to_offset_next(index);
        let size_of_val = offset_next - offset;
        let metadata = metadata_remainder.to_metadata(size_of_val);
        let start_ptr: *mut u8 = self.ptr.as_ptr().cast();
        let offset_ptr = unsafe { start_ptr.add(offset).cast::<()>() };
        (offset_ptr, metadata)
    }
}

impl<T: ?Sized + Aligned> Drop for UnsizedVec<T> {
    fn drop(&mut self) {
        if mem::needs_drop::<T>() {
            while let Some(MetadataStorage {
                metadata_remainder,
                offset_next_remainder,
            }) = self.metadata.pop()
            {
                let offset = self.offset_next();
                let offset_next = offset_next_remainder.to_offset_next(self.metadata.len());
                let size_of_val = offset_next - offset;
                let metadata = metadata_remainder.to_metadata(size_of_val);

                let start_ptr: *mut u8 = self.ptr.as_ptr().cast();
                let offset_ptr = unsafe { start_ptr.add(offset).cast::<()>() };

                let raw_wide_ptr: *mut T = ptr::from_raw_parts_mut(offset_ptr, metadata);
                unsafe { raw_wide_ptr.drop_in_place() };
            }
        }

        if self.cap != 0 {
            unsafe {
                alloc::dealloc(
                    self.ptr.as_ptr().cast::<u8>(),
                    Layout::from_size_align(self.cap, <T as Aligned>::ALIGN).unwrap(),
                );
            }
        }
    }
}

impl<T: ?Sized + Aligned> Index<usize> for UnsizedVec<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let (offset_ptr, metadata) = self.index_raw(index);

        let raw_wide_ptr: *const T = ptr::from_raw_parts(offset_ptr, metadata);
        unsafe { raw_wide_ptr.as_ref().unwrap_unchecked() }
    }
}

impl<T: ?Sized + Aligned> IndexMut<usize> for UnsizedVec<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let (offset_ptr, metadata) = self.index_raw(index);

        let raw_wide_ptr: *mut T = ptr::from_raw_parts_mut(offset_ptr, metadata);
        unsafe { raw_wide_ptr.as_mut().unwrap_unchecked() }
    }
}

impl<T: ?Sized + Aligned> Default for UnsizedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}
