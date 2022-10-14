//! [`UnsizedVec`] is like [`Vec`][alloc_crate::vec::Vec], but can store unsized values.
//!
//! Experimental, nightly-only.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![allow(incomplete_features)] // For `specialization`
#![feature(
    allocator_api,
    array_windows,
    const_option,
    forget_unsized,
    int_log,
    int_roundings,
    pointer_is_aligned,
    ptr_metadata,
    // We avoid specializing based on subtyping,
    // so should be sound.
    specialization,
    strict_provenance,
    unchecked_math,
    unsized_fn_params,
    vec_into_raw_parts
)]
#![no_std]

extern crate alloc as alloc_crate;

use alloc_crate::alloc::{self, Layout};
use core::{
    alloc::{AllocError, Allocator},
    cmp,
    fmt::{self, Debug, Formatter},
    hash::Hash,
    iter::FusedIterator,
    marker::PhantomData,
    mem,
    ops::{Index, IndexMut},
    ptr::{self, NonNull, Pointee},
};

pub mod emplace;
use emplace::*;

mod helper;

/// Contains the [`Aligned`] trait.
pub mod marker;

use marker::{AlignStorage, Aligned, StoreAlign};

use crate::helper::align_of_val;

/// Used by `UnsizedVec` to only store offset and pointer metadata
/// when the latter can't be derived from the former.

trait MetadataFromSize: Aligned {
    fn from_size(size: usize) -> <Self as Pointee>::Metadata;
}

impl<T> MetadataFromSize for T {
    fn from_size(_: usize) -> <Self as Pointee>::Metadata {}
}

impl<T> MetadataFromSize for [T] {
    fn from_size(size: usize) -> <Self as Pointee>::Metadata {
        size / mem::size_of::<T>()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct FullMetadataRemainder<T: Copy + Send + Sync + Ord + Hash + Unpin>(T);

/// Used by `UnsizedVec` to only store offset and pointer metadata
/// when the latter can't be derived from the former.
trait MetadataRemainder<T: ?Sized>: Copy + Send + Sync + Ord + Hash + Unpin {
    #[must_use]
    fn from_metadata(meta: <T as Pointee>::Metadata) -> Self;

    #[must_use]
    fn to_metadata(self, size: usize) -> <T as Pointee>::Metadata;
}

impl<T: ?Sized> MetadataRemainder<T> for FullMetadataRemainder<<T as Pointee>::Metadata> {
    #[inline]
    fn from_metadata(meta: <T as Pointee>::Metadata) -> Self {
        FullMetadataRemainder(meta)
    }

    #[inline]
    fn to_metadata(self, _: usize) -> <T as Pointee>::Metadata {
        self.0
    }
}

impl<T: ?Sized + MetadataFromSize> MetadataRemainder<T> for () {
    #[inline]
    fn from_metadata(_: <T as Pointee>::Metadata) -> Self {}

    #[inline]
    fn to_metadata(self, size: usize) -> <T as Pointee>::Metadata {
        <T as MetadataFromSize>::from_size(size)
    }
}
trait SplitMetadata {
    type Remainder: MetadataRemainder<Self>;
}

impl<T: ?Sized> SplitMetadata for T {
    default type Remainder = FullMetadataRemainder<<T as Pointee>::Metadata>;
}

// `MetadataFromSize` implementations are always "always applicable",
// so this specialization should be safe.
impl<T: ?Sized + MetadataFromSize> SplitMetadata for T {
    type Remainder = ();
}

/// Used by `UnsizedVec` to only have one heap allocation when storing `Sized` values.
trait OffsetNextRemainder<T: ?Sized>: Copy + Send + Sync + Ord + Hash + Unpin {
    #[must_use]
    fn from_offset_next(offset_next: usize) -> Self;

    #[must_use]
    fn to_offset_next(self, index: usize) -> usize;

    /// # Safety
    ///
    /// `self + sub` must not overflow
    #[must_use]
    unsafe fn unchecked_add(self, add: usize) -> Self;

    /// # Safety
    ///
    /// `self - sub` must not underflow
    #[must_use]
    unsafe fn unchecked_sub(self, sub: usize) -> Self;
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
    unsafe fn unchecked_add(self, add: usize) -> Self {
        // Safety: precondition of the functuon
        unsafe { self.unchecked_add(add) }
    }

    #[inline]
    unsafe fn unchecked_sub(self, sub: usize) -> Self {
        // Safety: precondition of the functuon
        unsafe { self.unchecked_sub(sub) }
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
    unsafe fn unchecked_add(self, _: usize) {}

    #[inline]
    unsafe fn unchecked_sub(self, _: usize) {}
}

/// Used by `UnsizedVec` to only have one heap allocation when storing `Sized` values.
trait SplitOffsetNext {
    type Remainder: OffsetNextRemainder<Self>;
}

impl<T: ?Sized> SplitOffsetNext for T {
    default type Remainder = usize;
}

// `Sized` implementations are always "always applicable",
// so this specialization should be safe.
impl<T> SplitOffsetNext for T {
    type Remainder = ();
}

/// This is pointer that has the highest alignment
/// that it is possible for a pointer to have.
/// It is well-aligned for any type.
const DANGLING_PERFECT_ALIGN: NonNull<()> = {
    let address: usize = 2_usize.pow(usize::MAX.ilog2());
    let ptr = ptr::invalid_mut(address);
    // Safety: 2_usize.pow(usize::MAX.ilog2()) is not 0
    unsafe { NonNull::new_unchecked(ptr) }
};

struct MetadataStorage<T: ?Sized> {
    /// The pointer metadata of the element of the `Vec`.
    metadata_remainder: <T as SplitMetadata>::Remainder,
    /// The offset that the element following this one would be stored at,
    /// but disregarding padding due to over-alignment.
    /// We use this encoding to store the sizes of `Vec` elements
    /// because it allows for *O(1)* random access while only storing
    /// a single `usize`.
    ///
    /// To get the actual offset of the next element, use
    /// `offset_next_remainder.to_offet(index).next_multiple_of(align)`.
    offset_next_remainder: <T as SplitOffsetNext>::Remainder,
}

impl<T: ?Sized> Clone for MetadataStorage<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for MetadataStorage<T> {}

/// Like [`Vec`][0], but can store unsized values.
///
/// # Memory layout
///
/// When `T` is a [`Sized`] type, the memory layout is more or less the same as [`Vec<T>`][0];
/// pointer to a heap allocation, as well as a `usize` each for length and capacity.
/// Elements are laid out in order, one after the other.
///
/// When `T` is a slice, there are two heap allocations.
/// The first is to the slices themsleves; they are laid out end-to-end, one after the other,
/// with no padding in between. The second heap allocation is to a list of offsets, to store
/// where each element begins and ends.
///
/// When `T` is neither of the above, there are still two allocations.
/// The first allocation still contains the elements of the vector laid out end-to-end,
/// but now every element is padded to at least the alignment of the most-aligned element
/// in the `UnsizedVec`. For this reason, adding a new element to the `Vec` with a larger alignment
/// than any of the elements already in it will add new padding to all the existing elements,
/// which will involve a lot of copying and probably a reallocation.
/// If you want to avoid excessive copies, use the [`UnsizedVec::with_byte_capacity_align`]
/// function to set the padding up front.
///
/// In the third case, the second allocation, in addition to storing offsets, also stores
/// the pointer metadata of each element.
///
/// # Example
///
/// ```
/// #![feature(unsized_fn_params)]
/// use core::fmt::Debug;
/// use unsized_vec::{*, emplace::box_new_with};
///
/// // `Box::new()` necessary only to coerce the values to trait objects.
/// let obj: Box<dyn Debug> = Box::new(1);
/// let obj_2: Box<dyn Debug> = Box::new((97_u128, "oh noes"));
/// let mut vec: UnsizedVec<dyn Debug> = unsized_vec![*obj, *obj_2];
/// for traitobj in &vec {
///     dbg!(traitobj);
/// };
///
/// assert_eq!(vec.len(), 2);
///
/// let popped = box_new_with(|e| vec.pop_unwrap(e));
/// dbg!(&*popped);
///
/// assert_eq!(vec.len(), 1);
/// ```
///
/// [0]: alloc_crate::vec::Vec
pub struct UnsizedVec<T: ?Sized> {
    ptr: NonNull<()>,
    cap: usize,
    metadata: alloc_crate::vec::Vec<MetadataStorage<T>>,
    align: <T as StoreAlign>::AlignStore,
    _marker: PhantomData<T>,
}

unsafe impl<T: ?Sized + Send> Send for UnsizedVec<T> {}
unsafe impl<T: ?Sized + Sync> Sync for UnsizedVec<T> {}

impl<T: ?Sized> UnsizedVec<T> {
    /// Make a new, empty `UnsizedVec`.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        UnsizedVec {
            ptr: DANGLING_PERFECT_ALIGN,
            cap: 0,
            align: <T as StoreAlign>::AlignStore::MIN_ALIGN,
            metadata: Default::default(),
            _marker: PhantomData,
        }
    }

    /// Make a new `UnsizedVec` with at least the the given capacity and alignment.
    /// (`align` is ignored for [`Aligned`] types).
    ///
    /// # Panics
    ///
    /// Panics if `align` is not a power of two (not guaranteed to panic when `T: Aligned`).
    #[must_use]
    #[inline]
    pub fn with_byte_capacity_align(byte_capacity: usize, align: usize) -> Self {
        let target_align = <T as StoreAlign>::AlignStore::from_align(align).unwrap();

        let mut ret = UnsizedVec {
            ptr: DANGLING_PERFECT_ALIGN,
            cap: 0,
            align: <T as StoreAlign>::AlignStore::MIN_ALIGN,
            metadata: Default::default(),
            _marker: PhantomData,
        };

        // Nothing in the vec, no need to adjust anything.
        // Safety: we asserted that `align` was valid earlier.
        let _ = unsafe { ret.grow_if_needed(byte_capacity, target_align) };
        ret.align = target_align;

        ret
    }

    /// Make a new `UnsizedVec` with at least the the given capacity, in bytes.
    #[must_use]
    #[inline]
    pub fn with_byte_capacity(byte_capacity: usize) -> Self
    where
        T: Aligned,
    {
        let mut ret = UnsizedVec {
            ptr: DANGLING_PERFECT_ALIGN,
            cap: 0,
            align: (),
            metadata: Default::default(),
            _marker: PhantomData,
        };

        // Nothing in the vec, no need to adjust anything.
        // Safety: `T::ALIGN` equals the type's alignment.
        let _ = unsafe { ret.grow_if_needed(byte_capacity, ()) };

        ret
    }

    /// Make a new `UnsizedVec` with at least the the given capacity, in number of elements.
    ///
    /// # Panics
    ///
    /// Panics if `capacity * mem::size_of::<T>()` overflows `isize::MAX` bytes.
    #[must_use]
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self
    where
        T: Sized,
    {
        let mut ret = UnsizedVec {
            ptr: DANGLING_PERFECT_ALIGN,
            cap: 0,
            align: (),
            metadata: Default::default(),
            _marker: PhantomData,
        };

        // Nothing in the vec, no need to adjust anything.
        // Safety: `T::ALIGN` equals the type's alignment.
        let _ =
            unsafe { ret.grow_if_needed(capacity.checked_mul(mem::size_of::<T>()).unwrap(), ()) };

        ret
    }

    /// Grow this `UnsizedVec`'s allocation by at least `by_at_least` bytes,
    /// and ensure it has at leas the alignment of `min_align`.
    /// Returns `true` iff *alignment* was increased.
    ///
    /// # Safety
    ///
    /// `min_align` must be a power of two.
    /// If `T: Aligned`, `size_delta` must be a multiple of `T::ALIGN`.
    #[must_use = "if alignment increased, you need to fix up offsets and then set the `align` field"]
    unsafe fn grow_if_needed(
        &mut self,
        size_delta: usize,
        min_align: <T as StoreAlign>::AlignStore,
    ) -> bool {
        const MAX_ALLOC_SIZE: usize = isize::MAX as usize;

        let old_align = self.align;
        let new_align = cmp::max(old_align, min_align);

        // If alignment is increasing and we need to realign existing elements,
        // we must account for extra space needed to add padding. Here,
        // We calculate how much space we need for that.
        let size_of_existing_elems_realigned = if old_align < new_align {
            let mut realigned_size: usize = 0;
            let mut last_offset: usize = 0;
            for (
                i,
                MetadataStorage {
                    offset_next_remainder,
                    ..
                },
            ) in self.metadata.iter().enumerate()
            {
                let offset_next = offset_next_remainder.to_offset_next(i);
                // Safety: `new_align` is a valid align as per function preconditions.
                // Offsets can't overflow as that would mean an allocation that's too big.
                unsafe {
                    realigned_size += new_align.unchecked_align_offset_to(
                        offset_next - old_align.unchecked_align_offset_to(last_offset),
                    );
                }

                last_offset = offset_next;
            }
            realigned_size
        } else {
            self.offset_next()
        };

        // Now we add on the size of the requested new memory.
        let new_size = size_of_existing_elems_realigned
            .checked_add(size_delta)
            .unwrap();

        // We make sure our allocation won't overflow.
        assert!(new_size <= MAX_ALLOC_SIZE);

        // `size_delta` might not have been aligned, so we need to make sure
        // that our new size is once again aligned.
        //
        // Safety: can't overflow due to check above. `isize::MAX` has the greatest
        // alignment of any value in `usize`s' range (other than 0), so anything less than
        // `isize` can't align to more than that.
        // The function's preconditions require `new_align` to be a valid align.
        let new_size = unsafe { new_align.unchecked_align_offset_to(new_size) };

        let old_cap = self.cap;

        let mut new_cap = if new_size > old_cap {
            (2 * old_cap).clamp(new_size, MAX_ALLOC_SIZE)
        } else {
            old_cap
        };

        // May be able to use `!self.ptr.as_ptr().is_aligned_to(new_align.into())`
        // instead of `new_align > old_align`?
        // https://github.com/rust-lang/rust/issues/32838 seems soundness requirement
        // not decided
        if new_cap > old_cap || new_align > old_align {
            // Safety: capacity clamped to `isize::MAX` earlier
            let mut new_layout = unsafe {
                Layout::from_size_align(new_cap, new_align.to_align().into()).unwrap_unchecked()
            };

            let new_ptr: Result<NonNull<[u8]>, AllocError> = if old_cap == 0 {
                if new_cap > 0 {
                    alloc::Global.allocate(new_layout)
                } else {
                    // Our dangling pointer is already guaranteed to be perfectly aligned,
                    // so no need to change it.

                    Ok(NonNull::from_raw_parts(self.ptr, 0))
                }
            } else {
                unsafe {
                    // Safety: old layout comes from vec, so must be valid
                    let old_layout = Layout::from_size_align(old_cap, old_align.to_align().into())
                        .unwrap_unchecked();

                    alloc::Global
                        .grow(self.ptr.cast(), old_layout, new_layout)
                        .or_else(|e| {
                            // We don't have the memory for 2 * old_cap,
                            // but maybe we can still allocate just enough for `new_size`.
                            if new_size > old_cap {
                                new_cap = new_size;
                                new_layout =
                                    Layout::from_size_align(new_cap, new_align.to_align().into())
                                        .unwrap_unchecked();
                                alloc::Global.grow(self.ptr.cast(), old_layout, new_layout)
                            } else {
                                Err(e)
                            }
                        })
                }
            };

            let Ok(new_ptr) = new_ptr else {
                alloc::handle_alloc_error(new_layout)
            };

            new_cap = new_ptr.len();
            self.ptr = new_ptr.cast();
        }

        self.cap = new_cap;

        new_align > old_align
    }

    /// Get the offset of one byte past the end of the last element in the vec.
    #[must_use]
    #[inline]
    fn offset_next(&self) -> usize {
        self.metadata.last().map_or(0, |m| {
            // Safety: if `last` returns `Some`, `len` must be greater than 0
            m.offset_next_remainder
                .to_offset_next(unsafe { self.len().unchecked_sub(1) })
        })
    }

    /// Realigns all elements in the vec to the given `new_align`.
    ///
    /// # Safety
    ///
    /// `new_align` must be greater than current alignment, less than or equal to
    /// the actual alignment of the allocation, and a valid (power of 2) alignment.
    /// `grow_if_needed` should be used to ensure these conditions are met.
    unsafe fn realign_up(&mut self, new_align: <T as StoreAlign>::AlignStore) {
        let old_align = self.align;

        // We compute the new offset of each element, along with the difference from the old offset.
        // Then, we copy everything over.
        // Doing this without allocating requires some complicated code.

        // The first element is already in the right place, its offset is 0.

        if self.len() > 1 {
            // First we calculate how much we need to shift the very last element,
            // then we go backward from there.

            // Starting here, our offsets are invalid, so unwinding is UB !!!
            // To make this explicit, we use unckecked ops for arithmetic.
            let last_offset_shift: usize = self.metadata.iter_mut().enumerate().fold(
                0,
                |cum_shift,
                 (
                    index,
                    MetadataStorage {
                        offset_next_remainder,
                        ..
                    },
                )| {
                    let old_offset_next = offset_next_remainder.to_offset_next(index);
                    // Safety: additions can't overflow because that would mean our allocation is bigger than its max value.
                    // Subtraction can't overflow because `next_aligned_to` gives greater-than-or-equal result
                    // for greater-than-or-equal align. `unchecked_next_aligned_to` safe as we are using a valid align.
                    unsafe {
                        let new_offset_next = old_offset_next.unchecked_add(cum_shift);
                        cum_shift.unchecked_add(
                            new_align
                                .unchecked_align_offset_to(new_offset_next)
                                .unchecked_sub(
                                    old_align.unchecked_align_offset_to(new_offset_next),
                                ),
                        )
                    }
                },
            );

            // Now we go in reverse.

            let _ = self.metadata.array_windows::<2>().enumerate().rev().fold(
                last_offset_shift,
                |shift_end,
                 (
                    index,
                    [MetadataStorage {
                        offset_next_remainder: offset_start_remainder,
                        ..
                    }, MetadataStorage {
                        offset_next_remainder,
                        ..
                    }],
                )| {
                    // Safety: index can't overflow, otherwise metadata would have
                    let new_offset_next =
                        offset_next_remainder.to_offset_next(unsafe { index.unchecked_add(1) });

                    // Safety: reversing computation above.
                    let shift_start = unsafe {
                        shift_end.unchecked_sub(
                            new_align
                                .unchecked_align_offset_to(new_offset_next)
                                .unchecked_sub(
                                    old_align.unchecked_align_offset_to(new_offset_next),
                                ),
                        )
                    };

                    // Safety: `new_align` is valid alignment.
                    let new_offset_start = unsafe {
                        new_align
                            .unchecked_align_offset_to(offset_start_remainder.to_offset_next(index))
                    };

                    // Safety: reversing computation above.
                    let old_offset_start = unsafe { new_offset_start.unchecked_sub(shift_start) };

                    // Safety: End offset >= start offset
                    let elem_len = unsafe { new_offset_next.unchecked_sub(new_offset_start) };

                    // Safety: moving element to new correct position, as computed above
                    unsafe {
                        ptr::copy(
                            self.ptr.as_ptr().cast::<u8>().add(old_offset_start),
                            self.ptr.as_ptr().cast::<u8>().add(new_offset_start),
                            elem_len,
                        );
                    };

                    shift_start
                },
            );
        }

        self.align = new_align;

        // Metatada is now fully correct and valid. Unwinding is no longer UB.
    }

    /// Add element to end of the vec.
    pub fn push(&mut self, elem: T) {
        let size_of_val = mem::size_of_val(&elem);
        let align_of_val = helper::align_of_val(&elem);

        // Make sure we have enough memory to store metadata
        self.metadata.reserve(1);

        // Safety: `align_of_val` is a valid alignment
        let align_changed = unsafe { self.grow_if_needed(size_of_val, align_of_val) };

        if align_changed {
            // Safety: `grow_if_needed` ensured our allocation is properly aligned
            unsafe { self.realign_up(align_of_val) };
        }

        let align = self.align.to_align();
        let new_elem_offset = self.offset_next().next_multiple_of(align.into());

        let new_offset_next = new_elem_offset + size_of_val;

        let metadata = ptr::metadata(&elem);

        // Copy element to end of allocation.
        // We checked that we had enough memory earlier
        unsafe {
            ptr::copy_nonoverlapping(
                core::ptr::addr_of!(elem).cast(),
                self.ptr.as_ptr().cast::<u8>().add(new_elem_offset),
                size_of_val,
            );
        }

        // elem is owned by the vec now, don't want to double-drop it
        mem::forget_unsized(elem);

        self.metadata.push(MetadataStorage {
            metadata_remainder: <T as SplitMetadata>::Remainder::from_metadata(metadata),
            offset_next_remainder: <T as SplitOffsetNext>::Remainder::from_offset_next(
                new_offset_next,
            ),
        });
    }

    /// Get the offset of element `index` in the vec.
    /// `None` if `index` is out of range (can be one past the end).
    #[inline]
    fn offset_start_idx(&self, index: usize) -> Option<usize> {
        let offset_next = index.checked_sub(1).map_or(Some(0), |i| {
            Some(
                self.metadata
                    .get(i)?
                    .offset_next_remainder
                    .to_offset_next(i),
            )
        })?;

        Some(unsafe { self.align.unchecked_align_offset_to(offset_next) })
    }

    /// # Safety
    ///
    /// `index` must be strictly less than `self.len()`.
    unsafe fn unchecked_insert_inner(&mut self, index: usize, elem: T) {
        let size_of_val = mem::size_of_val(&elem);
        let align_of_val = helper::align_of_val(&elem);

        // Make sure we have enough memory to store metadata
        self.metadata.reserve(1);

        // Safety: `align_of_val` is a valid alignment
        let align_changed = unsafe { self.grow_if_needed(size_of_val, align_of_val) };

        if align_changed {
            // FIXME only copy elements to the right once
            // Safety: `grow_if_needed` ensured our allocation is properly aligned
            unsafe { self.realign_up(align_of_val) };
        }

        let align = self.align.to_align();

        let aligned_size_of_val = size_of_val.next_multiple_of(align.into());

        let old_end_offset = self.offset_next();

        let metadata = ptr::metadata(&elem);

        // Update offsets of follwing elements.
        // Unwinding is UB starting here until end of function !!!;
        // Safety: safety precondition of this function
        for ms in unsafe { self.metadata.get_unchecked_mut(index..) } {
            // Safety: addition can't overflow because that would mean that the memory allocation overflowed
            ms.offset_next_remainder =
                unsafe { ms.offset_next_remainder.unchecked_add(aligned_size_of_val) };
        }

        // Safety: safety precondition of this function
        let offset_start = unsafe { self.offset_start_idx(index).unwrap_unchecked() };

        // Shift right and copy element to end of allocation.
        // We checked that we had enough memory earlier
        unsafe {
            ptr::copy(
                self.ptr.as_ptr().cast::<u8>().add(offset_start),
                self.ptr
                    .as_ptr()
                    .cast::<u8>()
                    .add(offset_start)
                    .add(aligned_size_of_val),
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

    /// Insert element into vec at given index, shifting following elements to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    #[inline]
    pub fn insert(&mut self, index: usize, elem: T) {
        // Inline the bounds check, but not the rest
        match index.cmp(&self.len()) {
            cmp::Ordering::Less => unsafe { self.unchecked_insert_inner(index, elem) },
            cmp::Ordering::Equal => self.push(elem),
            cmp::Ordering::Greater => panic!("index > self.len()"),
        };
    }

    /// Remove an element from the end of the vec.
    /// This returns the element, which is unsized, through the "emplacer" mechanism.
    /// Call `box_new_with(|e| your_unsized_vec.pop_unwrap(e))` to get the element boxed.
    ///
    /// # Panics
    ///
    /// Panics if len is 0.
    pub fn pop_unwrap(&mut self, emplacer: &mut Emplacer<T>) {
        // Panics if len is 0.
        let MetadataStorage {
            metadata_remainder,
            offset_next_remainder: offset_end_remainder,
        } = self.metadata.pop().unwrap();

        let offset_start = self
            .offset_next()
            .next_multiple_of(self.align.to_align().into());
        let offset_end = offset_end_remainder.to_offset_next(self.metadata.len());
        let size_of_val = offset_end - offset_start;
        let metadata = metadata_remainder.to_metadata(size_of_val);

        // Safety: offset computed above is within the `Vec`'s allocation
        let ptr_to_val: *const u8 = unsafe { self.ptr.as_ptr().cast::<u8>().add(offset_start) };

        // Safety: `Vec` stores a valid instance of `T` at the offset
        let align_of_val = unsafe {
            mem::align_of_val(
                ptr::from_raw_parts::<T>(ptr_to_val.cast(), metadata)
                    .as_ref()
                    .unwrap(),
            )
        };

        // Safety: we write to the pointer below
        let emplace_fn = unsafe { emplacer.into_inner() };

        emplace_fn(
            Layout::from_size_align(size_of_val, align_of_val).unwrap(),
            metadata,
            &mut |dst| unsafe {
                ptr::copy_nonoverlapping(
                    self.ptr.as_ptr().cast::<u8>().add(offset_start),
                    dst.cast::<u8>(),
                    size_of_val,
                );
            },
        );
    }

    /// Remove the element at the given index, shifting those after it to the left.
    /// This returns the element, which is unsized, through the "emplacer" mechanism.
    /// Call `box_new_with(|e| your_unsized_vec.remove(index, e))` to get the element boxed.
    ///
    /// Doesn't shink the allocation, and doesn't reduce alignment.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize, emplacer: &mut Emplacer<T>) {
        let MetadataStorage {
            metadata_remainder,
            offset_next_remainder: offset_end_remainder,
        } = self.metadata.remove(index);

        // Our metadata is incorrect, so unwind is now UB !!!

        // Safety: `remove` call above would have panicked if this indexing could fail
        let offset_start = unsafe { self.offset_start_idx(index).unwrap_unchecked() };
        let offset_end = offset_end_remainder.to_offset_next(index);
        let size_of_val = offset_end - offset_start;
        let aligned_size_of_val = size_of_val.next_multiple_of(self.align.to_align().into());
        let metadata = metadata_remainder.to_metadata(size_of_val);

        // Update offsets of follwing elements
        // Safety: `remove` call performed a bounds check already
        for ms in unsafe { self.metadata.get_unchecked_mut(index..) } {
            // Safety: subtraction can't overflow because that would mean that the offset we are adjusting
            // is for an element before the one we are removing
            ms.offset_next_remainder =
                unsafe { ms.offset_next_remainder.unchecked_sub(aligned_size_of_val) };
        }

        // Safety: offset computed above is within the `Vec`'s allocation
        let ptr_to_val: *const u8 = unsafe { self.ptr.as_ptr().cast::<u8>().add(offset_start) };

        // Safety: `Vec` stores a valid instance of `T` at the offset
        let align_of_val = unsafe {
            mem::align_of_val(
                ptr::from_raw_parts::<T>(ptr_to_val.cast(), metadata)
                    .as_ref()
                    .unwrap(),
            )
        };

        let emplace_fn = unsafe { emplacer.into_inner() };
        emplace_fn(
            Layout::from_size_align(size_of_val, align_of_val).unwrap(),
            metadata,
            &mut |dst| unsafe {
                ptr::copy_nonoverlapping(ptr_to_val, dst.cast::<u8>(), size_of_val);

                ptr::copy(
                    ptr_to_val.add(size_of_val),
                    ptr_to_val.cast_mut(),
                    self.offset_next() - offset_start,
                );
            },
        );
    }

    /// Returns the number of elements in this `UnsizedVec`.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// Returns the size of the backing allocation of this `UnsizedVec`, in bytes.
    #[must_use]
    #[inline]
    pub fn byte_capacity(&self) -> usize {
        self.cap
    }

    /// Returns the number of elements that this `UnsizedVec` can store without reallocation.
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize
    where
        T: Sized,
    {
        self.cap
            .checked_div(mem::size_of::<T>())
            .unwrap_or(usize::MAX)
    }

    /// Returns the maximum alignment of the values that this `UnsizedVec` can store without reallocating.
    /// For `T: Aligned`, this is just the type's alignent.
    #[must_use]
    #[inline]
    pub fn align(&self) -> usize {
        self.align.to_align().into()
    }

    /// Returns true iff `self.len() == 0`.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    #[must_use]
    #[inline]
    fn index_raw(&self, index: usize) -> Option<(*mut (), <T as Pointee>::Metadata)> {
        let MetadataStorage {
            metadata_remainder,
            offset_next_remainder,
        } = self.metadata.get(index)?;
        // Safety: we know we are within bounds, otherwise `get` call above would have returned `None`
        let offset = unsafe { self.offset_start_idx(index).unwrap_unchecked() };
        let offset_next = offset_next_remainder.to_offset_next(index);
        let size_of_val = offset_next - offset;
        let metadata = metadata_remainder.to_metadata(size_of_val);
        let start_ptr: *mut u8 = self.ptr.as_ptr().cast();
        let offset_ptr = unsafe { start_ptr.add(offset).cast::<()>() };
        Some((offset_ptr, metadata))
    }

    /// Returns an iterator over shared references to the elements of this `UnsizedVec`.
    #[inline]
    pub fn iter(&self) -> UnsizedVecIter<T> {
        self.into_iter()
    }

    /// Returns an iterator over mutable references to the elements of this `UnsizedVec`.
    #[inline]
    pub fn iter_mut(&mut self) -> UnsizedVecIterMut<T> {
        self.into_iter()
    }

    /// Find the maximum alignment of all the elements in the vec.
    /// This is the minimum align the vec can be shrunk to.
    #[must_use]
    #[inline]
    fn max_align_of_elements(&self) -> <T as StoreAlign>::AlignStore {
        self.iter().fold(
            <T as StoreAlign>::AlignStore::MIN_ALIGN,
            |max_align, val| cmp::max(max_align, align_of_val(val)),
        )
    }

    /// Shrinks the alignments of the elements in the vec,
    /// and returns what it was shrunk to.
    ///
    /// # Safety
    ///
    /// Doesn't adjust `self.align`; you must do that yourself,
    /// or the vec will be in an invalid state!
    ///
    /// Doesn't reallocate with lower alignment.
    #[inline]
    unsafe fn realign_down(
        &mut self,
        align: <T as StoreAlign>::AlignStore,
    ) -> <T as StoreAlign>::AlignStore {
        let new_align = cmp::max(self.max_align_of_elements(), align);
        let old_align = self.align;

        if new_align < old_align {
            // FIXME Un-nest once we have `let_chains`
            // This code would also be a lot nicer with a lending `array_windows_mut`.
            if let Some(len_m_1) = self.len().checked_sub(1) {
                let mut shift_back: usize = 0;

                // It's UB to unwind inside this loop !!!
                // (because offsets are temporarily invalid)
                for index in 0..len_m_1 {
                    // Safety: index in range as bounded by len
                    let offset_start_unaligned = unsafe {
                        self.metadata
                            .get_unchecked(index)
                            .offset_next_remainder
                            .to_offset_next(index)
                    };

                    // Safety: we are aligning to this element's old alignment, which we know is valid.
                    let offset_start_old_align =
                        unsafe { old_align.unchecked_align_offset_to(offset_start_unaligned) };

                    // Safety: shift_back can't be more than the element's offset.
                    let old_offset_start =
                        unsafe { offset_start_old_align.unchecked_sub(shift_back) };

                    // Safety: we are aligning to this element's new alignment, which we know is valid.
                    let new_offset_start =
                        unsafe { new_align.unchecked_align_offset_to(offset_start_unaligned) };

                    // Safety: index + 1 in range, as index bounded by len - 1
                    let offset_next_remainder = unsafe {
                        &mut self
                            .metadata
                            .get_unchecked_mut(index.unchecked_add(1))
                            .offset_next_remainder
                    };

                    // Safety: index can't overflow, otherwise metadata vec would be too long
                    let old_offset_next =
                        offset_next_remainder.to_offset_next(unsafe { index.unchecked_add(1) });

                    // Safety: end of element comes after its start
                    let size_of_elem = unsafe { old_offset_next.unchecked_sub(old_offset_start) };

                    // Safety: copy beck element as per calculations above
                    unsafe {
                        ptr::copy(
                            self.ptr.as_ptr().cast::<u8>().add(old_offset_start),
                            self.ptr.as_ptr().cast::<u8>().add(new_offset_start),
                            size_of_elem,
                        );
                    }

                    shift_back = unsafe { old_offset_start.unchecked_sub(new_offset_start) };

                    // Safety: shift < offset
                    *offset_next_remainder =
                        <T as SplitOffsetNext>::Remainder::from_offset_next(unsafe {
                            old_offset_next.unchecked_sub(shift_back)
                        });
                }
            }

            new_align
        } else {
            old_align
        }
    }

    #[inline]
    fn shrink_byte_capacity_align_to_inner(
        &mut self,
        byte_capacity: usize,
        align: <T as StoreAlign>::AlignStore,
    ) {
        let old_align = self.align;

        // Safety: we take care of needed adjustments right after.
        // 1 is a power of 2
        let new_align = unsafe { self.realign_down(align) };

        let old_cap = self.byte_capacity();
        // Safety: align and offset both come from the vec
        let new_cap = cmp::min(
            cmp::max(byte_capacity, unsafe {
                new_align.unchecked_align_offset_to(self.offset_next())
            }),
            old_cap,
        );

        if new_align < old_align || new_cap < old_cap {
            if old_cap > 0 {
                // Safety: aligns and capacities are valid as they come from the vec
                let old_layout = unsafe {
                    Layout::from_size_align(old_cap, old_align.to_align().into()).unwrap_unchecked()
                };

                if new_cap == 0 {
                    // Safety: New cap is 0, don't need allocation anymore!
                    // Pointer to allocation is valid, it comes from the vec.
                    // So does layout
                    unsafe { alloc::Global.deallocate(self.ptr.cast(), old_layout) };
                    self.cap = 0;
                    self.ptr = DANGLING_PERFECT_ALIGN;
                } else {
                    // Safety: aligns and capacities are valid as they come from the vec
                    let new_ptr = unsafe {
                        let new_layout =
                            Layout::from_size_align(new_cap, new_align.to_align().into())
                                .unwrap_unchecked();

                        alloc::Global.shrink(self.ptr.cast(), old_layout, new_layout)
                    };

                    let Ok(new_ptr) = new_ptr else {
                        // If `shrink fails, then we have to undo all our hard work :(
                        // Such failures should hopefully be rare.

                        self.align = new_align;
                        // Our allocation is big enough, we just failed to shrink it!
                        unsafe { self.realign_up(old_align) };
                        return;
                    };

                    self.cap = new_ptr.len();
                    self.ptr = new_ptr.cast();
                }
            }

            self.align = new_align;
        }
    }

    /// Reduce align to the smallest value that is as least as big as the supplied value,
    /// and large enough to fit all elements in the vec currently.
    ///
    /// Does nothing when `T: Aligned`, or when current align is less then or equal to supplied value.
    ///
    /// # Panics
    ///
    /// Panics if `align` is not a power of two (might not panic if `T: Aligned`).
    pub fn shrink_align_to(&mut self, align: usize) {
        self.shrink_byte_capacity_align_to_inner(
            self.byte_capacity(),
            <T as StoreAlign>::AlignStore::from_align(align).unwrap(),
        );
    }

    /// Reduce capacity, in bytes, to the smallest value that is as least as big as the supplied value,
    /// and large enough to fit all elements in the vec currently. Doesn't shrink alignment.
    ///
    /// Does nothing when current capacity is less then or equal to supplied value.
    pub fn shrink_byte_capacity_to(&mut self, byte_capacity: usize) {
        self.shrink_byte_capacity_align_to_inner(byte_capacity, self.align);
    }

    /// Reduce capacity, in bytes, to the smallest value that is as least as big as the supplied value,
    /// and large enough to fit all elements in the vec currently. Also, reduce alignment to minimum.
    ///
    /// Does nothing when current capacity and alignment are less then or equal to supplied values.
    ///
    /// # Panics
    ///
    /// Panics if `align` is not a power of two (might not panic if `T: Aligned`).
    #[inline]
    pub fn shrink_byte_capacity_align_to(&mut self, byte_capacity: usize, align: usize) {
        self.shrink_byte_capacity_align_to_inner(
            byte_capacity,
            <T as StoreAlign>::AlignStore::from_align(align).unwrap(),
        );
    }

    /// Reduce capacity and align as much as possible.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        // Safety: 1 is a power of 2
        self.shrink_byte_capacity_align_to_inner(0, unsafe {
            <T as StoreAlign>::AlignStore::from_align(1).unwrap_unchecked()
        });
    }

    /// Reduce capacity, in number of elements, to the smallest value that is as least as big as the supplied value,
    /// and large enough to fit all elements in the vec currently.
    ///
    /// Does nothing when current capacity is less then or equal to supplied value.
    pub fn shrink_to(&mut self, min_capacity: usize)
    where
        T: Sized,
    {
        self.shrink_byte_capacity_align_to_inner(
            min_capacity.saturating_mul(mem::size_of::<T>()),
            (),
        );
    }
}

impl<T: ?Sized> Drop for UnsizedVec<T> {
    fn drop(&mut self) {
        if mem::needs_drop::<T>() {
            while let Some(MetadataStorage {
                metadata_remainder,
                offset_next_remainder,
            }) = self.metadata.pop()
            {
                let offset_start = self
                    .offset_next()
                    .next_multiple_of(self.align.to_align().into());
                let offset_end = offset_next_remainder.to_offset_next(self.metadata.len());
                let size_of_val = offset_end - offset_start;
                let metadata = metadata_remainder.to_metadata(size_of_val);

                let start_ptr: *mut u8 = self.ptr.as_ptr().cast();
                let offset_ptr = unsafe { start_ptr.add(offset_start).cast::<()>() };

                let raw_wide_ptr: *mut T = ptr::from_raw_parts_mut(offset_ptr, metadata);

                // Safety: `raw_wide_ptr` is a valid pointer whatever element of type `T`
                // the user put into the vec.
                unsafe { raw_wide_ptr.drop_in_place() };
            }
        }

        if self.cap != 0 {
            // Safety: we are deallocating a previous allocation.
            unsafe {
                alloc::Global.deallocate(
                    self.ptr.cast::<u8>(),
                    Layout::from_size_align(self.cap, self.align.to_align().into()).unwrap(),
                );
            }
        }
    }
}

impl<T: ?Sized> Index<usize> for UnsizedVec<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let (offset_ptr, metadata) = self.index_raw(index).unwrap();

        let raw_wide_ptr: *const T = ptr::from_raw_parts(offset_ptr, metadata);
        unsafe { raw_wide_ptr.as_ref().unwrap_unchecked() }
    }
}

impl<T: ?Sized> IndexMut<usize> for UnsizedVec<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let (offset_ptr, metadata) = self.index_raw(index).unwrap();

        let raw_wide_ptr: *mut T = ptr::from_raw_parts_mut(offset_ptr, metadata);
        unsafe { raw_wide_ptr.as_mut().unwrap_unchecked() }
    }
}

impl<T: ?Sized> Default for UnsizedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Created by `UnsizedVec::iter()`
#[must_use]
#[derive(Debug, Clone)]
pub struct UnsizedVecIter<'a, T: ?Sized + 'a> {
    vec: &'a UnsizedVec<T>,
    index: usize,
}

impl<'a, T: ?Sized> Iterator for UnsizedVecIter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.vec.len() {
            let ret = Some(&self.vec[self.index]);
            self.index += 1;
            ret
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: ?Sized + 'a> IntoIterator for &'a UnsizedVec<T> {
    type Item = &'a T;

    type IntoIter = UnsizedVecIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        UnsizedVecIter {
            vec: self,
            index: 0,
        }
    }
}

impl<'a, T: ?Sized + 'a> ExactSizeIterator for UnsizedVecIter<'a, T> {}
impl<'a, T: ?Sized + 'a> FusedIterator for UnsizedVecIter<'a, T> {}

/// Created by `UnsizedVec::iter_mut()`

#[must_use]
#[derive(Debug)]
pub struct UnsizedVecIterMut<'a, T: ?Sized + 'a> {
    vec: &'a mut UnsizedVec<T>,
    index: usize,
}

impl<'a, T: ?Sized> Iterator for UnsizedVecIterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (offset_ptr, metadata) = self.vec.index_raw(self.index)?;
        let raw_wide_ptr: *mut T = ptr::from_raw_parts_mut(offset_ptr, metadata);
        let ret = Some(unsafe { raw_wide_ptr.as_mut().unwrap() });
        self.index += 1;
        ret
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: ?Sized + 'a> ExactSizeIterator for UnsizedVecIterMut<'a, T> {}
impl<'a, T: ?Sized + 'a> FusedIterator for UnsizedVecIterMut<'a, T> {}

impl<'a, T: ?Sized + 'a> IntoIterator for &'a mut UnsizedVec<T> {
    type Item = &'a mut T;

    type IntoIter = UnsizedVecIterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        UnsizedVecIterMut {
            vec: self,
            index: 0,
        }
    }
}

impl<T: ?Sized + Debug> Debug for UnsizedVec<T> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: ?Sized + Clone> Clone for UnsizedVec<T> {
    #[inline]
    fn clone(&self) -> Self {
        let mut ret = UnsizedVec::with_byte_capacity_align(self.byte_capacity(), self.align());
        for elem in self {
            ret.push(elem.clone());
        }
        ret
    }
}

impl<T: ?Sized + PartialEq<U>, U: ?Sized> PartialEq<UnsizedVec<U>> for UnsizedVec<T> {
    #[inline]
    fn eq(&self, other: &UnsizedVec<U>) -> bool {
        self.len() == other.len() && self.iter().zip(other).all(|(l, r)| l == r)
    }
}

impl<T: ?Sized + Eq> Eq for UnsizedVec<T> {}

impl<T: ?Sized + PartialOrd<U>, U: ?Sized> PartialOrd<UnsizedVec<U>> for UnsizedVec<T> {
    fn partial_cmp(&self, other: &UnsizedVec<U>) -> Option<cmp::Ordering> {
        for (l, r) in self.iter().zip(other) {
            match l.partial_cmp(r) {
                Some(cmp::Ordering::Equal) => (),
                res => return res,
            }
        }
        self.len().partial_cmp(&other.len())
    }
}

impl<T: ?Sized + Ord> Ord for UnsizedVec<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        for (l, r) in self.iter().zip(other) {
            match l.cmp(r) {
                cmp::Ordering::Equal => (),
                res => return res,
            }
        }
        self.len().cmp(&other.len())
    }
}

impl<T: ?Sized + Hash> Hash for UnsizedVec<T> {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        for elem in self {
            elem.hash(state);
        }
    }
}

/// Like the standard library's `vec` macro.
#[macro_export]
macro_rules! unsized_vec {
    () => (
        $crate::UnsizedVec::new()
    );
    ($($x:expr),+ $(,)?) => (
        {
            let mut ret = $crate::UnsizedVec::new();
            $(ret.push($x);)+
            ret
        }
    );
}

#[cfg(feature = "serde")]
use serde::{ser::SerializeSeq, Serialize};

#[cfg(feature = "serde")]
impl<T: ?Sized + Serialize> Serialize for UnsizedVec<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut elem_serialize = serializer.serialize_seq(Some(self.len()))?;
        for elem in self {
            elem_serialize.serialize_element(elem)?;
        }
        elem_serialize.end()
    }
}

impl<T> From<alloc_crate::vec::Vec<T>> for UnsizedVec<T> {
    fn from(value: alloc_crate::vec::Vec<T>) -> Self {
        let (ptr, len, cap) = value.into_raw_parts();

        UnsizedVec {
            // Safety: comes from `into_raw_parts`, so can't be null
            ptr: unsafe { NonNull::new_unchecked(ptr.cast()) },
            // Safety: multiplication can't overflow,
            // if it did that would mean the `Vec` had an impossibly large allocation
            cap: unsafe { cap.unchecked_mul(mem::size_of::<T>()) },
            metadata: alloc_crate::vec![
                MetadataStorage {
                    metadata_remainder: (),
                    offset_next_remainder: ()
                };
                len
            ],
            align: (),
            _marker: PhantomData,
        }
    }
}

impl<T> From<UnsizedVec<T>> for alloc_crate::vec::Vec<T> {
    fn from(value: UnsizedVec<T>) -> Self {
        // Safety: ptr, len, and cap from an allocation allocated with
        // the global allocator.
        let ret = unsafe {
            alloc_crate::vec::Vec::from_raw_parts(
                value.ptr.as_ptr().cast(),
                value.len(),
                value.cap.checked_div(mem::size_of::<T>()).unwrap_or(0),
            )
        };

        // Don't want to drop the `UnsizedVec`
        // now that we have transferred ownership of its allocation
        mem::forget(value);

        ret
    }
}
