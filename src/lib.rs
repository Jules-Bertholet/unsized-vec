//! [`UnsizedVec`] is like [`Vec`][alloc_crate::vec::Vec], but can store unsized values.
//!
//! Experimental, nightly-only.

#![deny(unsafe_op_in_unsafe_fn)]
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
    unsized_fn_params
)]
#![no_std]

extern crate alloc as alloc_crate;

use alloc_crate::{
    alloc::{self, Layout},
    vec::Vec,
};
use core::{
    alloc::{AllocError, Allocator},
    cmp,
    fmt::{self, Debug, Formatter},
    hash::Hash,
    iter::{self, FusedIterator},
    marker::PhantomData,
    mem,
    num::NonZeroUsize,
    ops::{Index, IndexMut},
    ptr::{self, NonNull, Pointee},
};

pub mod emplace;
use emplace::*;

mod helper;
use helper::next_multiple_of_unchecked;

pub mod marker;

use marker::Aligned;

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

trait AlignStorage<T: ?Sized>: Copy + Send + Sync + Ord + Hash + Unpin {
    #[must_use]
    fn from_align(align: NonZeroUsize) -> Self;

    #[must_use]
    fn to_align(self) -> NonZeroUsize;
}

impl<T: ?Sized> AlignStorage<T> for NonZeroUsize {
    #[inline]
    fn from_align(align: NonZeroUsize) -> Self {
        align
    }

    #[inline]
    fn to_align(self) -> NonZeroUsize {
        self
    }
}

impl<T: ?Sized + Aligned> AlignStorage<T> for () {
    #[inline]
    fn from_align(_: NonZeroUsize) -> Self {}

    #[inline]
    fn to_align(self) -> NonZeroUsize {
        <T as Aligned>::ALIGN
    }
}
trait StoreAlign {
    type AlignStore: AlignStorage<Self>;
}

impl<T: ?Sized> StoreAlign for T {
    default type AlignStore = NonZeroUsize;
}

impl<T: ?Sized + Aligned> StoreAlign for T {
    type AlignStore = ();
}

/// This returns a pointer that has the highest alignment
/// that it is possible for a pointer to have.
/// It is well-aligned for any type.
#[must_use]
#[inline]
const fn dangling_perfect_align() -> NonNull<()> {
    let address: usize = 2_usize.pow(usize::MAX.ilog2());
    let ptr = ptr::invalid_mut(address);
    // Safety: `ptr` id definitelty non-null, we just created it
    unsafe { NonNull::new_unchecked(ptr) }
}

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
/// [0]: alloc_crate::vec::Vec
pub struct UnsizedVec<T: ?Sized> {
    ptr: NonNull<()>,
    cap: usize,
    align: <T as StoreAlign>::AlignStore,
    metadata: alloc_crate::vec::Vec<MetadataStorage<T>>,
    _marker: PhantomData<(*mut T, T)>,
}

unsafe impl<T: ?Sized + Send> Send for UnsizedVec<T> {}
unsafe impl<T: ?Sized + Sync> Sync for UnsizedVec<T> {}

impl<T: ?Sized> UnsizedVec<T> {
    /// Make a new, empty `UnsizedVec`.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        UnsizedVec {
            ptr: dangling_perfect_align(),
            cap: 0,
            align: <T as StoreAlign>::AlignStore::from_align(NonZeroUsize::new(1).unwrap()),
            metadata: Default::default(),
            _marker: PhantomData,
        }
    }

    /// Make a new `UnsizedVec` with at least the the given capacity and alignment.
    /// (`align` is ignored for [`Aligned`] types).
    ///
    /// # Panics
    ///
    /// Panics if `align` is not a power of two.
    #[must_use]
    #[inline]
    pub fn with_byte_capacity_align(byte_capacity: usize, align: usize) -> Self {
        assert!(align.is_power_of_two());
        // Safety: above assert ensures `align` is non_zero
        let align = unsafe { NonZeroUsize::new_unchecked(align) };

        let mut ret = UnsizedVec {
            ptr: dangling_perfect_align(),
            cap: 0,
            align: <T as StoreAlign>::AlignStore::from_align(NonZeroUsize::new(1).unwrap()),
            metadata: Default::default(),
            _marker: PhantomData,
        };

        let target_align = <T as StoreAlign>::AlignStore::from_align(align);

        // Nothing in the vec, no need to adjust anything.
        // Safety: we asserted that `align` was valid earlier.
        let _ = unsafe { ret.grow_if_needed(byte_capacity, target_align.to_align()) };
        ret.align = target_align;

        ret
    }

    /// Grow this `UnsizedVec`'s allocation by at least `by_at_least` bytes,
    /// and ensure it has at leas the alignment of `min_align`.
    /// Returns `true` iff *alignment* was increased.
    ///
    /// # Safety
    ///
    /// `min_align` must be a power of two.
    #[must_use = "if alignment increased, you need to fix up offsets and then set the `align` field"]
    unsafe fn grow_if_needed(&mut self, size_delta: usize, min_align: NonZeroUsize) -> bool {
        let old_align = self.align.to_align();
        let new_align = cmp::max(old_align, min_align);
        let new_size = size_delta
            .checked_next_multiple_of(new_align.into())
            .unwrap()
            .checked_add(if old_align < new_align {
                let mut total_size: usize = 0;
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
                    total_size += (offset_next - last_offset.next_multiple_of(old_align.into()))
                        .next_multiple_of(new_align.into());
                    last_offset = offset_next;
                }
                total_size
            } else {
                self.offset_next()
            })
            .unwrap();

        let mut new_cap = if new_size > self.cap {
            cmp::max(2 * self.cap, new_size)
        } else {
            self.cap
        };

        if new_cap > self.cap || !self.ptr.as_ptr().is_aligned_to(new_align.into()) {
            let new_layout = Layout::from_size_align(new_cap, new_align.into()).unwrap();

            // Ensure that the new allocation doesn't exceed `isize::MAX` bytes.
            assert!(
                isize::try_from(new_layout.size()).is_ok(),
                "Allocation too large"
            );

            let new_ptr: Result<NonNull<[u8]>, AllocError> = if self.cap == 0 {
                if new_cap > 0 {
                    alloc::Global.allocate(new_layout)
                } else {
                    // Our dangling pointer is already guaranteed to be perfectly aligned,
                    // so no need to change it.

                    Ok(NonNull::from_raw_parts(self.ptr, 0))
                }
            } else {
                let old_layout = Layout::from_size_align(self.cap, old_align.into()).unwrap();
                unsafe { alloc::Global.grow(self.ptr.cast(), old_layout, new_layout) }
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
            m.offset_next_remainder.to_offset_next(self.len() - 1)
        })
    }

    /// Realigns all elements in the vec to the given `new_align`.
    ///
    /// # Safety
    /// - `new_align` must be greater than current alignment
    /// - `grow_if_needed` must have been called to ensure the allocation is well-aligned and large enough
    unsafe fn realign(&mut self, new_align: NonZeroUsize) {
        let old_align = self.align.to_align();

        // We compute the new offset of each element, along with the difference from the old offset.
        // Then, we copy everything over.
        // Unfortunately, doing this in O(n) requires allocating (as far as I can figure),
        // though we try our best to do so as little as possible
        // (which unfortunately requires some complicated code).

        // The first element is already in the right place, its offset is 0.

        if let Some(len_sub_2) = self.metadata.len().checked_sub(2) {
            // offset_shifts_i_plus_1[i] equals the number of bytes element `i + 1` needs to be shifted right by.
            // We allocate this up front, because we don't want to unwind while our `Vec` metadata is in an invalid state.
            let mut offset_shifts_i_plus_1: Vec<usize> = Vec::with_capacity(len_sub_2 + 1);
            let mut cumulative_shift: usize = 0;

            // Stores the new offsets into our `Vec` and populates `offset_shifts_i_plus_1`.
            // Starting from this point, it's UB if we unwind !!!
            // To make this explicit, we use only "unchecked" variants of potentially-panicking operations.
            for (
                index,
                MetadataStorage {
                    offset_next_remainder,
                    ..
                },
            ) in self.metadata.iter_mut().enumerate()
            {
                // Safety: can't overflow `usize` as allocation capped to `isize`
                let new_offset_next = unsafe {
                    offset_next_remainder
                        .to_offset_next(index)
                        .unchecked_add(cumulative_shift)
                };

                *offset_next_remainder =
                    <T as SplitOffsetNext>::Remainder::from_offset_next(new_offset_next);

                if index <= len_sub_2 {
                    // Safety: `cumulative_shift` can't overflow `usize`, as allocation capped to `isize`.
                    // Subtraction can't oveflow because shift is always positive.
                    // `next_multiple_of` is passed `new_align`, which is known to be a valid alignment,
                    // and therefore much less than `isize::MAX` (so no overflow risk).
                    unsafe {
                        cumulative_shift = cumulative_shift.unchecked_add(
                            next_multiple_of_unchecked(new_offset_next, new_align).unchecked_sub(
                                next_multiple_of_unchecked(new_offset_next, old_align),
                            ),
                        );
                    }

                    // Safety: We allocated enough memory upfront above,
                    // so `push` can't panic.
                    offset_shifts_i_plus_1.push(cumulative_shift);
                }
            }

            // Now we actually copy the elements over.
            // First, we have to set up and zip a lot of iterators;
            // we can't use `Vec`s because that would mean allocating.

            let mut offset_next_iter = self
                .metadata
                .iter()
                .enumerate()
                .map(|(index, ms)| ms.offset_next_remainder.to_offset_next(index))
                .rev();

            // We are only interested in offsets of elements already in the vec.
            offset_next_iter.next();

            // Safety: we checked the length of the `Vec` in the `if` guard above.
            let first_ms = unsafe { self.metadata.first().unwrap_unchecked() };

            let elem_sizes_iter = iter::once(first_ms.offset_next_remainder.to_offset_next(0))
                .chain(self.metadata.array_windows::<2>().enumerate().map(
                    |(
                            index,
                            [MetadataStorage {
                                offset_next_remainder: l_off_rem,
                                ..
                            }, MetadataStorage {
                                offset_next_remainder: r_off_rem,
                                ..
                            }],
                        )|
                        // Safety: Right offset can't be less than left offset,
                        // so subtraction can't overflow.
                        // `new_align` is a valid align,
                        // so `next_multiple_of` can't overflow.
                        unsafe {
                            r_off_rem.to_offset_next(index).unchecked_sub(
                                next_multiple_of_unchecked(
                                    l_off_rem.to_offset_next(index),
                                    new_align,
                                ),
                            )
                        },
                ));

            // Copy in reverse order, so as not to overwrite old stuff.
            for (new_offset, (shift_next, elem_size)) in offset_next_iter.zip(
                offset_shifts_i_plus_1
                    .into_iter()
                    .rev()
                    .zip(elem_sizes_iter.rev()),
            ) {
                // Safety:
                // next_multiple_of` can't overflow as `old_align` and `new_align` are valid alignments.
                // `ptr::copy` copies from the old offsets to the new offsets as computed above.
                unsafe {
                    let old_offset_unaligned = new_offset - shift_next;
                    let old_offset_aligned =
                        next_multiple_of_unchecked(old_offset_unaligned, old_align);
                    let new_offset_aligned = next_multiple_of_unchecked(new_offset, new_align);

                    ptr::copy(
                        self.ptr.as_ptr().cast::<u8>().add(old_offset_aligned),
                        self.ptr.as_ptr().cast::<u8>().add(new_offset_aligned),
                        elem_size,
                    );
                }
            }
        }

        self.align = <T as StoreAlign>::AlignStore::from_align(new_align);

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
            unsafe { self.realign(align_of_val) };
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
    #[inline]
    fn offset_start_idx(&self, index: usize) -> usize {
        index.checked_sub(1).map_or(0, |i| {
            self.metadata[i]
                .offset_next_remainder
                .to_offset_next(i)
                .next_multiple_of(self.align.to_align().into())
        })
    }

    /// Insert element into vec at given index, shifting following elements to the right.
    pub fn insert(&mut self, index: usize, elem: T) {
        let size_of_val = mem::size_of_val(&elem);
        let align_of_val = helper::align_of_val(&elem);

        // Make sure we have enough memory to store metadata
        self.metadata.reserve(1);

        // Safety: `align_of_val` is a valid alignment
        let align_changed = unsafe { self.grow_if_needed(size_of_val, align_of_val) };

        if align_changed {
            // FIXME only copy elements to the right once
            // Safety: `grow_if_needed` ensured our allocation is properly aligned
            unsafe { self.realign(align_of_val) };
        }

        let align = self.align.to_align();

        let aligned_size_of_val = size_of_val.next_multiple_of(align.into());

        let old_end_offset = self.offset_next();

        let metadata = ptr::metadata(&elem);

        // Update offsets of follwing elements
        for ms in &mut self.metadata[index..] {
            ms.offset_next_remainder = ms.offset_next_remainder.add(aligned_size_of_val);
        }

        let offset_start = self.offset_start_idx(index);

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
        let offset_start = self.offset_start_idx(index);
        let offset_end = offset_end_remainder.to_offset_next(index);
        let size_of_val = offset_end - offset_start;
        let aligned_size_of_val = size_of_val.next_multiple_of(self.align.to_align().into());
        let metadata = metadata_remainder.to_metadata(size_of_val);

        // Update offsets of follwing elements
        for ms in &mut self.metadata[index..] {
            ms.offset_next_remainder = ms.offset_next_remainder.sub(aligned_size_of_val);
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

    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    #[must_use]
    #[inline]
    pub fn byte_capacity(&self) -> usize {
        self.cap
    }

    #[must_use]
    #[inline]
    pub fn align(&self) -> usize {
        self.align.to_align().into()
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

    #[inline]
    pub fn iter(&self) -> UnsizedVecIter<T> {
        self.into_iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> UnsizedVecIterMut<T> {
        self.into_iter()
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
        let (offset_ptr, metadata) = self.index_raw(index);

        let raw_wide_ptr: *const T = ptr::from_raw_parts(offset_ptr, metadata);
        unsafe { raw_wide_ptr.as_ref().unwrap_unchecked() }
    }
}

impl<T: ?Sized> IndexMut<usize> for UnsizedVec<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let (offset_ptr, metadata) = self.index_raw(index);

        let raw_wide_ptr: *mut T = ptr::from_raw_parts_mut(offset_ptr, metadata);
        unsafe { raw_wide_ptr.as_mut().unwrap_unchecked() }
    }
}

impl<T: ?Sized> Default for UnsizedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[must_use]
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

#[must_use]
pub struct UnsizedVecIterMut<'a, T: ?Sized + 'a> {
    vec: &'a mut UnsizedVec<T>,
    index: usize,
}

impl<'a, T: ?Sized + 'a> Iterator for UnsizedVecIterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.vec.len() {
            let (offset_ptr, metadata) = self.vec.index_raw(self.index);
            let raw_wide_ptr: *mut T = ptr::from_raw_parts_mut(offset_ptr, metadata);
            let ret = Some(unsafe { raw_wide_ptr.as_mut().unwrap() });
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
