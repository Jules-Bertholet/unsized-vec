//! [`UnsizedVec<T>`], like [`Vec<T>`][alloc::vec::Vec], is a contiguous growable array
//! type with heap-allocated contents. Unlike [`Vec<T>`], it can store unsized values.
//!
//! Vectors have *O*(1) indexing, amortized *O*(1) push (to the end) and
//! *O*(1) pop (from the end). When `T` is [`Sized`], they use one heap
//! allocation; when it is not, they use two.
//!
//! This crate is nightly-only and experimental.
//!
//! # Examples
//!
//! You can explicitly create an [`UnsizedVec`] with [`UnsizedVec::new`]:
//!
//! ```
//! # use unsized_vec::UnsizedVec;
//! let v: UnsizedVec<i32> = UnsizedVec::new();
//! ```
//!
//! ...or by using the [`unsize_vec!`] macro:
//!
//! ```
//! # use core::fmt::Debug;
//! # use unsized_vec::{unsize_vec, UnsizedVec};
//! let v: UnsizedVec<[u32]> = unsize_vec![];
//!
//! let v: UnsizedVec<dyn Debug> = unsize_vec![1_u32, "hello!", 3.0_f64, (), -17_i32];
//! ```
//!
//! You can [`push`] or [`push_unsize`] values onto the end of a vector (which will grow the vector
//! as needed):
//!
//! ```
//! # use core::fmt::Debug;
//! # use unsized_vec::{unsize_vec, UnsizedVec};
//! let mut v: UnsizedVec<dyn Debug> = unsize_vec![1_u32, "hello!", 3.0_f64, (), -17_i32];
//!
//! v.push_unsize(3);
//! ```
//!
//! Popping values works in much the same way:
//!
//! ```
//! # use core::fmt::Debug;
//! # use emplacable::box_new_with;
//! # use unsized_vec::{unsize_vec, UnsizedVec};
//! let mut v: UnsizedVec<dyn Debug> = unsize_vec![1_u32, "hello!"];
//!
//! // "hello!" is copied directly into a new heap allocation
//! let two: Option<Box<dyn Debug>> = v.pop_into().map(box_new_with);
//! ```
//!
//! Vectors also support indexing (through the [`Index`] and [`IndexMut`] traits):
//!
//! ```
//! # use core::fmt::Debug;
//! # use unsized_vec::{unsize_vec, UnsizedVec};
//! let mut v: UnsizedVec<dyn Debug> = unsize_vec![1_u32, "hello!", [(); 0]];
//! let greeting = &v[1];
//! dbg!(greeting);
//! ```
//! [`Vec<T>`]: alloc::vec::Vec
//! [`push`]: UnsizedVec::push
//! [`push_unsize`]: UnsizedVec::push_unsize

#![allow(
    incomplete_features, // For `specialization`
    internal_features, // for `unsized_fn_params`
)]
#![feature(
    allocator_api,
    array_windows,
    forget_unsized,
    int_roundings,
    ptr_metadata,
    // We avoid specializing based on subtyping,
    // so barring compiler bugs, our usage should be sound.
    specialization,
    try_reserve_kind,
    type_alias_impl_trait,
    unsize,
    unsized_fn_params,
)]
#![no_std]

mod helper;
mod inner;
mod marker;

#[doc(hidden)]
pub extern crate alloc;
use alloc::{alloc::handle_alloc_error, collections::TryReserveErrorKind};
use core::{
    self, cmp,
    fmt::{self, Debug, Formatter},
    hash::Hash,
    iter::FusedIterator,
    marker::Unsize,
    mem,
    ops::{Index, IndexMut},
};
use emplacable::{unsize, Emplacable, EmplacableFn, Emplacer};

use inner::{Align, Size, UnsizedVecImpl, UnsizedVecProvider};

/// The error type for `try_reserve` methods.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TryReserveError {
    kind: TryReserveErrorKind,
}

#[track_caller]
#[inline]
fn to_align<T: ?Sized>(align: usize) -> AlignTypeFor<T> {
    #[cold]
    #[inline(never)]
    fn invalid_align(align: usize) -> ! {
        panic!("align {align} is not a power of 2")
    }

    let Some(ret) = AlignTypeFor::<T>::new(align) else {
        invalid_align(align)
    };

    ret
}

#[track_caller]
#[inline]
fn unwrap_try_reserve_result<T>(result: Result<T, TryReserveError>) -> T {
    #[cold]
    #[inline(never)]
    fn handle_err(e: TryReserveError) -> ! {
        match e.kind {
            TryReserveErrorKind::CapacityOverflow => panic!("Capacity overflowed `isize::MAX`"),
            TryReserveErrorKind::AllocError { layout, .. } => handle_alloc_error(layout),
        }
    }

    match result {
        Ok(val) => val,
        Err(e) => handle_err(e),
    }
}

impl From<::alloc::collections::TryReserveError> for TryReserveError {
    fn from(value: ::alloc::collections::TryReserveError) -> Self {
        TryReserveError { kind: value.kind() }
    }
}

type AlignTypeFor<T> = <<T as UnsizedVecImpl>::Impl as UnsizedVecProvider<T>>::Align;
type SizeTypeFor<T> = <<T as UnsizedVecImpl>::Impl as UnsizedVecProvider<T>>::Size;

/// Like [`Vec`][0], but can store unsized values.
///
/// # Memory layout
///
/// `UnsizedVec` is actually three different types rolled in to one;
/// specialization is used to choose the optimal implementation based on the properties
/// of `T`.
///
/// 1. When `T` is a [`Sized`] type, `UnsizedVec<T>` is a newtype around [`Vec<T>`][0],
///    with exactly the same memoy layout.
///
/// 2. When `T` is a slice, there are two heap allocations.
///    The first is to the slices themsleves; they are laid out end-to-end, one after the other,
///    with no padding in between. The second heap allocation is to a list of offsets, to store
///    where each element begins and ends.
///
/// 3. When `T` is neither of the above, there are still two allocations.
///    The first allocation still contains the elements of the vector laid out end-to-end,
///    but now every element is padded to at least the alignment of the most-aligned element
///    in the `UnsizedVec`. For this reason, adding a new element to the vec with a larger alignment
///    than any of the elements already in it will add new padding to all the existing elements,
///    which will involve a lot of copying and probably a reallocation. By default, [`UnsizedVec::new`]
///    sets the alignment to [`core::mem::align_of::<usize>()`], so as long as none of your trait objects
///    are aligned to more than that, you won't have to worry about re-padding.
///    For this last case, the second allocation, in addition to storing offsets, also stores the pointer
///    metadata of each element.
///
/// ## Managing capacity
///
/// [`Vec<T>`][0] has only one kind of capacity to worry about: elementwise capacity. And so does
/// `UnsizedVec<T>`, as long as `T: Sized`. You can use functions like [`capacity`], [`with_capacity`]
/// and [`reserve`] to manage this capacity.
///
/// When `T` is a slice, there are two kinds of capacities: element capacity and byte capacity.
/// Adding new elements to the vec is guaranteed not to reallocate as long as
/// the number of elements doesn't exceed the element capacity *and* the total size of all
/// the elements in bytes doesn't exceed the byte capacity. You can use functions like
/// [`byte_capacity`], [`with_capacity_bytes`], and [`reserve_capacity_bytes`] to manage
/// these two capacities.
///
/// When `T` is a trait object, there is a third type of capacity: alignment. To avoid
/// reallocation when adding a new element to the vec, you need to ensure that you have
/// sufficient element and byte capacity, and that the vec's align is not less than the
/// alignment of the new element. Functions like [`align`], [`with_capacity_bytes_align`], and
/// [`reserve_capacity_bytes_align`], can be used to manage all three capacities in this case.
///
/// # Limitations
///
/// - `UnsizedVec<T>` is invariant with respect to `T`; ideally, it should be covariant.
///   This is because Rust forces invariance on all structs that contain associated types
///   referencing `T`. Hopefully, future language features will allow lifting this limitation.
/// - Rust functions can't directly return unsized types. So this crate's functions return
///   them indirectly, though the "emplacer" mechanism defined in the [`emplacable`] crate.
///   See that crate's documentation for details, and the documentation of [`pop_into`] and
///   [`remove_into`] for usage examples.
///
/// # Example
///
/// ```
/// #![allow(internal_features)] // for `unsized_fn_params`
/// #![feature(unsized_fn_params)]
/// use core::fmt::Debug;
///
/// use emplacable::box_new_with;
/// use unsized_vec::{unsize_vec, UnsizedVec};
///
/// let mut vec: UnsizedVec<dyn Debug> = unsize_vec![27.53_f32, "oh the places we'll go", Some(())];
///
/// for traitobj in &vec {
///     dbg!(traitobj);
/// };
///
/// assert_eq!(vec.len(), 3);
///
/// let maybe_popped: Option<Box<dyn Debug>> = vec.pop_into().map(box_new_with);
/// let popped = maybe_popped.unwrap();
/// dbg!(&*popped);
///
/// assert_eq!(vec.len(), 2);
/// ```
///
/// [0]: alloc::vec::Vec
/// [`emplacable`]: emplacable
/// [`capacity`]: UnsizedVec::capacity
/// [`with_capacity`]: UnsizedVec::with_capacity
/// [`reserve`]: UnsizedVec::reserve
/// [`byte_capacity`]: UnsizedVec::byte_capacity
/// [`with_capacity_bytes`]: UnsizedVec::with_capacity_bytes
/// [`reserve_capacity_bytes`]: UnsizedVec::reserve_capacity_bytes
/// [`align`]: UnsizedVec::align
/// [`with_capacity_bytes_align`]: UnsizedVec::with_capacity_bytes_align
/// [`reserve_capacity_bytes_align`]: UnsizedVec::reserve_capacity_bytes_align
/// [`pop_into`]: UnsizedVec::pop_into
/// [`remove_into`]: UnsizedVec::remove_into
#[repr(transparent)]
pub struct UnsizedVec<T>
where
    T: ?Sized,
{
    inner: <T as UnsizedVecImpl>::Impl,
}

impl<T: ?Sized> UnsizedVec<T> {
    /// Create a new, empty `UnsizedVec`.
    /// Does not allocate.
    ///
    /// When `T`'s alignmnent is not known
    /// at compile-time, this uses `mem::align_of::<usize>()`
    /// as the default alignment.
    #[must_use]
    #[inline]
    pub const fn new() -> UnsizedVec<T> {
        UnsizedVec {
            inner: UnsizedVecProvider::NEW_ALIGN_PTR,
        }
    }

    /// Create a new, empty `UnsizedVec` with the given capacity.
    ///
    /// When `T`'s alignmnent is not known
    /// at compile-time, this uses `mem::align_of::<usize>()`
    /// as the default alignment.
    #[must_use]
    #[inline]
    pub fn with_capacity(capacity: usize) -> UnsizedVec<T> {
        let mut vec = UnsizedVec::new();
        vec.reserve_exact(capacity);
        vec
    }

    /// Create a new, empty `UnsizedVec` with the given capacity.
    /// (When `T: Aligned` does not hold, an alignment of 1 is used.)
    ///
    /// When `T`'s alignmnent is not known
    /// at compile-time, this uses `mem::align_of::<usize>()`
    /// as the default alignment.
    #[must_use]
    #[inline]
    pub fn with_capacity_bytes(capacity: usize, byte_capacity: usize) -> UnsizedVec<T> {
        let mut vec = UnsizedVec::new();
        vec.reserve_exact_capacity_bytes(capacity, byte_capacity);
        vec
    }

    /// Create a new, empty `UnsizedVec` with the given capacity
    /// (in bytes) and alignment.
    ///
    /// `align` is ignored when `T`'s alignment is known at compile time
    #[must_use]
    #[inline]
    pub fn with_capacity_bytes_align(
        capacity: usize,
        byte_capacity: usize,
        align: usize,
    ) -> UnsizedVec<T> {
        let mut vec = UnsizedVec {
            inner: UnsizedVecProvider::NEW_ALIGN_1,
        };
        vec.reserve_exact_capacity_bytes_align(capacity, byte_capacity, align);
        vec
    }

    /// Returns the number of elements the vector can hold without
    /// reallocating.
    ///
    /// For `T: ?Sized`, this only concers whether metadata
    /// could get reallocated, not the elements themselves.
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns the number of bytes the vector can hold without
    /// reallocating.
    #[must_use]
    #[inline]
    pub fn byte_capacity(&self) -> usize {
        self.inner.byte_capacity()
    }

    /// Returns the maximum alignment of the values this vector
    /// can hold without re-padding and reallocating.
    ///
    /// Only relevant when `T`'s alignment is not known at compile time.
    #[must_use]
    #[inline]
    pub fn align(&self) -> usize {
        self.inner.align()
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the given `UnsizedVec<T>`. The collection may reserve more space to
    /// speculatively avoid frequent reallocations.
    ///
    /// When `T` is not `Sized`, this only reseves space to store *metadata*.
    /// Consider using [`reserve_capacity_bytes`] instead in such cases.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// [`reserve_capacity_bytes`]: UnsizedVec::reserve_capacity_bytes
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        unwrap_try_reserve_result(self.try_reserve(additional));
    }

    /// Reserves capacity for at least `additional` more elements,
    /// taking up at least `additional_bytes` bytes of space, to be inserted
    /// in the given `UnsizedVec<T>`. The collection may reserve more space to
    /// speculatively avoid frequent reallocations.
    ///
    /// When `T`'s alignment is not known at compile time,
    /// the vec may still reallocate if you push a new element onto the
    /// vec with an alignment greater than `self.align()`. Consider
    /// using [`reserve_capacity_bytes_align`] instead in such cases.
    ///
    /// # Panics
    ///
    /// Panics if the either of the new capacities exceeds `isize::MAX` bytes.
    ///
    /// [`reserve_capacity_bytes_align`]: UnsizedVec::reserve_capacity_bytes_align
    #[inline]
    pub fn reserve_capacity_bytes(&mut self, additional: usize, additional_bytes: usize) {
        unwrap_try_reserve_result(self.try_reserve_capacity_bytes(additional, additional_bytes));
    }

    /// Reserves capacity for at least `additional` more elements,
    /// taking up at least `additional_bytes` bytes of space,
    /// and with alignment of at most `align`, to be inserted
    /// in the given `UnsizedVec<T>`. The collection may reserve more space to
    /// speculatively avoid frequent reallocations.
    ///
    /// When `T`'s alignment is known at compile time,
    /// `align` is ignored. Consider using [`reserve_capacity_bytes`]
    /// instead in such cases.
    ///
    /// # Panics
    ///
    /// Panics if the either of the new capacities exceeds `isize::MAX` bytes,
    /// or if `align` is not a power of two.
    ///
    /// [`reserve_capacity_bytes`]: UnsizedVec::reserve_capacity_bytes
    #[inline]
    pub fn reserve_capacity_bytes_align(
        &mut self,
        additional: usize,
        additional_bytes: usize,
        align: usize,
    ) {
        unwrap_try_reserve_result(self.try_reserve_capacity_bytes_align(
            additional,
            additional_bytes,
            align,
        ));
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the given `UnsizedVec<T>`. Unlike [`reserve`], this will not
    /// deliberately over-allocate to speculatively avoid frequent allocations.
    ///
    /// When `T` is not `Sized`, this only reseves space to store *metadata*.
    /// Consider using [`reserve_exact_capacity_bytes`] instead in such cases.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// [`reserve`]: UnsizedVec::reserve
    /// [`reserve_exact_capacity_bytes`]: UnsizedVec::reserve_exact_capacity_bytes
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        unwrap_try_reserve_result(self.try_reserve_exact_capacity_bytes(additional, 0));
    }

    /// Reserves capacity for at least `additional` more elements,
    /// taking up at least `additional_bytes` bytes of space, to be inserted
    /// in the given `UnsizedVec<T>`. Unlike [`reserve_capacity_bytes`], this will not
    /// deliberately over-allocate to speculatively avoid frequent allocations.
    ///
    /// When `T`'s alignment is not known at compile time,
    /// the vec may still reallocate if you push a new element onto the
    /// vec with an alignment greater than `self.align()`. Consider
    /// using [`reserve_exact_capacity_bytes_align`] instead in such cases.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// [`reserve_capacity_bytes`]: UnsizedVec::reserve_capacity_bytes
    /// [`reserve_exact_capacity_bytes_align`]: UnsizedVec::reserve_exact_capacity_bytes_align
    #[inline]
    pub fn reserve_exact_capacity_bytes(&mut self, additional: usize, additional_bytes: usize) {
        unwrap_try_reserve_result(
            self.try_reserve_exact_capacity_bytes(additional, additional_bytes),
        );
    }

    /// Reserves capacity for at least `additional` more elements,
    /// taking up at least `additional_bytes` bytes of space,
    /// and with alignment of at most `align`, to be inserted
    /// in the given `UnsizedVec<T>`. Unlike [`reserve_capacity_bytes_align`], this will not
    /// deliberately over-allocate to speculatively avoid frequent allocations.
    ///
    /// When `T`'s alignment is known at compile time,
    /// `align` is ignored. Consider using [`reserve_exact_capacity_bytes`]
    /// instead in such cases.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` bytes.
    ///
    /// [`reserve_capacity_bytes_align`]: UnsizedVec::reserve_capacity_bytes_align
    /// [`reserve_exact_capacity_bytes`]: UnsizedVec::reserve_exact_capacity_bytes
    #[inline]
    pub fn reserve_exact_capacity_bytes_align(
        &mut self,
        additional: usize,
        additional_bytes: usize,
        align: usize,
    ) {
        unwrap_try_reserve_result(self.try_reserve_exact_capacity_bytes_align(
            additional,
            additional_bytes,
            align,
        ));
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the given `UnsizedVec<T>`. The collection may reserve more space to
    /// speculatively avoid frequent reallocations.
    ///
    /// When `T` is not `Sized`, this only reseves space to store *metadata*.
    /// Consider using [`try_reserve_capacity_bytes`] instead in such cases.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// [`try_reserve_capacity_bytes`]: UnsizedVec::try_reserve_capacity_bytes
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve(additional)
    }

    /// Reserves capacity for at least `additional` more elements,
    /// taking up at least `additional_bytes` bytes of space, to be inserted
    /// in the given `UnsizedVec<T>`. The collection may reserve more space to
    /// speculatively avoid frequent reallocations.
    ///
    /// When `T`'s alignment is not known at compile time,
    /// the vec may still reallocate if you push a new element onto the
    /// vec with an alignment greater than `self.align()`. Consider
    /// using [`try_reserve_capacity_bytes_align`] instead in such cases.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// [`try_reserve_capacity_bytes_align`]: UnsizedVec::try_reserve_capacity_bytes_align
    #[inline]
    pub fn try_reserve_capacity_bytes(
        &mut self,
        additional: usize,
        additional_bytes: usize,
    ) -> Result<(), TryReserveError> {
        self.try_reserve_capacity_bytes_align(additional, additional_bytes, 1)
    }

    /// Reserves capacity for at least `additional` more elements,
    /// taking up at least `additional_bytes` bytes of space,
    /// and with alignment of at most `align`, to be inserted
    /// in the given `UnsizedVec<T>`. The collection may reserve more space to
    /// speculatively avoid frequent reallocations.
    ///
    /// When `T`'s alignment is known at compile time,
    /// `align` is ignored. Consider using [`try_reserve_capacity_bytes`]
    /// instead in such cases.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Panics
    ///
    /// Panics if `align` is not a power of two.
    ///
    /// [`try_reserve_capacity_bytes`]: UnsizedVec::try_reserve_capacity_bytes
    #[inline]
    pub fn try_reserve_capacity_bytes_align(
        &mut self,
        additional: usize,
        additional_bytes: usize,
        align: usize,
    ) -> Result<(), TryReserveError> {
        self.try_reserve(additional)?;

        debug_assert!(self.capacity() >= self.len() + additional);

        let align = to_align::<T>(align);

        let byte_cap = self.byte_capacity();

        let needed_bytes = additional_bytes.saturating_sub(self.unused_byte_cap());

        let optimist_bytes = if needed_bytes > 0 {
            cmp::max(needed_bytes, byte_cap)
        } else {
            0
        };

        // First we try to double capacities.
        // if that fails, we try again with only what we really need.
        if optimist_bytes > needed_bytes {
            let result = self
                .inner
                .try_reserve_additional_bytes_align(optimist_bytes, align);

            if result.is_ok() {
                return result;
            }
        }

        let result = self
            .inner
            .try_reserve_additional_bytes_align(needed_bytes, align);

        debug_assert!(self.byte_capacity() >= self.byte_len() + additional_bytes);

        result
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the given `UnsizedVec<T>`. Unlike [`try_reserve`], this will not
    /// deliberately over-allocate to speculatively avoid frequent allocations.
    ///
    /// When `T` is not `Sized`, this only reseves space to store *metadata*.
    /// Consider using [`try_reserve_exact_capacity_bytes`] instead in such cases.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// [`try_reserve`]: UnsizedVec::try_reserve
    /// [`try_reserve_exact_capacity_bytes`]: UnsizedVec::try_reserve_exact_capacity_bytes
    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve_exact(additional)
    }

    /// Reserves capacity for at least `additional` more elements,
    /// taking up at least `additional_bytes` bytes of space, to be inserted
    /// in the given `UnsizedVec<T>`. Unlike [`try_reserve_capacity_bytes`], this will not
    /// deliberately over-allocate to speculatively avoid frequent allocations.
    ///
    /// When `T`'s alignment is not known at compile time,
    /// the vec may still reallocate if you push a new element onto the
    /// vec with an alignment greater than `self.align()`. Consider
    /// using [`try_reserve_exact_capacity_bytes_align`] instead in such cases.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// [`try_reserve_capacity_bytes`]: UnsizedVec::try_reserve_capacity_bytes
    /// [`try_reserve_exact_capacity_bytes_align`]: UnsizedVec::try_reserve_exact_capacity_bytes_align
    #[inline]
    pub fn try_reserve_exact_capacity_bytes(
        &mut self,
        additional: usize,
        additional_bytes: usize,
    ) -> Result<(), TryReserveError> {
        self.try_reserve_exact_capacity_bytes_align(additional, additional_bytes, 1)
    }

    /// Reserves capacity for at least `additional` more elements,
    /// taking up at least `additional_bytes` bytes of space,
    /// and with alignment of at most `align`, to be inserted
    /// in the given `UnsizedVec<T>`. Unlike [`try_reserve_capacity_bytes_align`], this will not
    /// deliberately over-allocate to speculatively avoid frequent allocations.
    ///
    /// When `T`'s alignment is known at compile time,
    /// `align` is ignored. Consider using [`try_reserve_exact_capacity_bytes`]
    /// instead in such cases.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Panics
    ///
    /// Panics if `align` is not a power of two.
    ///
    /// [`try_reserve_capacity_bytes_align`]: UnsizedVec::try_reserve_capacity_bytes_align
    /// [`try_reserve_exact_capacity_bytes`]: UnsizedVec::try_reserve_exact_capacity_bytes
    #[inline]
    pub fn try_reserve_exact_capacity_bytes_align(
        &mut self,
        additional: usize,
        additional_bytes: usize,
        align: usize,
    ) -> Result<(), TryReserveError> {
        self.inner.try_reserve(additional)?;
        let align = to_align::<T>(align);

        self.inner
            .try_reserve_additional_bytes_align(additional_bytes, align)
    }

    /// Shrinks all the capacities of the vec as much as possible.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.inner
            .shrink_capacity_bytes_align_to(0, 0, to_align::<T>(1));
    }

    /// Shrinks the elementwise capacity of the vector with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length
    /// and the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    ///
    /// For `T: ?Sized`, this only effects elementwise capacity.
    /// Consider using [`shrink_capacity_bytes_to`] in such cases.
    ///
    /// [`shrink_capacity_bytes_to`]: UnsizedVec::shrink_capacity_bytes_to
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_capacity_bytes_align_to(
            min_capacity,
            usize::MAX,
            to_align::<T>(1 << (usize::BITS - 1)),
        );
    }

    /// Shrinks the elementwise and byte capacities of the vector with
    /// lower bounds.
    ///
    /// The capacities will remain at least as large as both the lengths
    /// and the supplied values.
    ///
    /// If the current capacities are less than the lower limits, this is a no-op.
    ///
    /// When `T`'s alignment is not known at compile-time, this only effects elementwise
    /// and bytewise capacities.
    /// Consider using [`shrink_capacity_bytes_align_to`] in such cases.
    ///
    /// [`shrink_capacity_bytes_align_to`]: UnsizedVec::shrink_capacity_bytes_align_to
    #[inline]
    pub fn shrink_capacity_bytes_to(&mut self, min_capacity: usize, min_byte_capacity: usize) {
        self.inner.shrink_capacity_bytes_align_to(
            min_capacity,
            min_byte_capacity,
            to_align::<T>(1 << (usize::BITS - 1)),
        );
    }

    /// Shrinks the elementwise, byte, and alignment capacities of the vector with
    /// lower bounds.
    ///
    /// The capacities will remain at least as large as both the lengths
    /// and the supplied values.
    ///
    /// If the current capacities are less than the lower limits, this is a no-op.
    ///
    /// # Panics
    ///
    /// Panics if `min_align` is not a power of two.
    #[inline]
    pub fn shrink_capacity_bytes_align_to(
        &mut self,
        min_capacity: usize,
        min_byte_capacity: usize,
        min_align: usize,
    ) {
        self.inner.shrink_capacity_bytes_align_to(
            min_capacity,
            min_byte_capacity,
            to_align::<T>(min_align),
        );
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// If `T` is not `Sized`, you will need
    /// `#![feature(unsized_fn_params)]` to call this.
    /// You may also need the [`unsize`] macro, which
    /// requires additional nightly features.
    ///
    /// Alternatively, you can use [`insert_unsize`][0],
    /// which takes care of unsizing for you.
    ///
    /// # Example
    ///
    /// ```
    /// #![allow(internal_features)] // for `unsized_fn_params`
    /// #![feature(allocator_api, ptr_metadata, unsized_fn_params)]
    ///
    /// use core::fmt::Debug;
    ///
    /// use emplacable::unsize;
    /// use unsized_vec::UnsizedVec;
    ///
    /// let mut vec: UnsizedVec<dyn Debug> = UnsizedVec::new();
    ///
    /// vec.push(unsize!([1, 2], ([i32; 2]) -> dyn Debug));
    /// vec.insert(0, unsize!("can you believe it", (&str) -> dyn Debug));
    /// dbg!(&vec[0]);
    /// ```
    ///
    /// [0]: UnsizedVec::insert_unsize
    #[inline]
    pub fn insert(&mut self, index: usize, value: T) {
        #[track_caller]
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("insertion index (is {index}) should be <= len (is {len})");
        }

        if index <= self.len() {
            let size_of_val = SizeTypeFor::<T>::of_val(&value);
            self.reserve_capacity_bytes_align(1, size_of_val.get(), mem::align_of_val(&value));

            // SAFETY: reserved needed capacity and performed bounds check above
            unsafe { self.inner.insert_unchecked(index, value, size_of_val) }
        } else {
            assert_failed(index, self.len())
        }
    }

    /// Appends an element to the back of a collection
    /// after unsizing it.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::fmt::Debug;
    ///
    /// use unsized_vec::UnsizedVec;
    ///
    /// let mut vec: UnsizedVec<dyn Debug> = UnsizedVec::new();
    ///
    /// vec.push_unsize([1, 2]);
    /// vec.insert_unsize(0, "can you believe it");
    /// dbg!(&vec[0]);
    /// ```
    #[inline]
    pub fn insert_unsize<S>(&mut self, index: usize, value: S)
    where
        S: Unsize<T>,
    {
        self.insert(index, unsize!(value, (S) -> T));
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// Accepts the element as an [`Emplacable<T, _>`]
    /// instead of `T` directly, analogously
    /// to [`emplacable::box_new_with`].
    ///
    /// # Example
    ///
    /// ```
    /// #![allow(internal_features)] // for `unsized_fn_params`
    /// #![feature(allocator_api, ptr_metadata, unsized_fn_params)]
    ///
    /// use core::fmt::Debug;
    ///
    /// use unsized_vec::{unsize_vec, UnsizedVec};
    ///
    /// let mut vec_1: UnsizedVec<dyn Debug> = unsize_vec![32, "hello"];
    /// let mut vec_2: UnsizedVec<dyn Debug> = unsize_vec![97];
    ///
    /// vec_2.insert_with(0, vec_1.pop_into().unwrap());
    ///
    /// assert_eq!(vec_1.len(), 1);
    /// assert_eq!(vec_2.len(), 2);
    /// dbg!(&vec_2[0]);
    /// ```
    #[inline]
    pub fn insert_with(&mut self, index: usize, value: Emplacable<T, impl EmplacableFn<T>>) {
        #[track_caller]
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("insertion index (is {index}) should be <= len (is {len})");
        }

        if index <= self.len() {
            // SAFETY: did bounds check just above
            unsafe { self.inner.insert_with_unchecked(index, value) }
        } else {
            assert_failed(index, self.len())
        }
    }

    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// Because `T` might be unsized, and functions can't return
    /// unsized values directly, this method returns the element using
    /// the "emplacer" mechanism. You can pass the returned [`Emplacable<T, _>`]
    /// to a function like [`box_new_with`] to get the contained `T`.
    ///
    /// # Example
    ///
    /// ```
    /// use core::fmt::Debug;
    ///
    /// use emplacable::box_new_with;
    /// use unsized_vec::UnsizedVec;
    ///
    /// let mut vec = UnsizedVec::<dyn Debug>::new();
    ///
    /// vec.push_unsize("A beautiful day today innit");
    /// vec.push_unsize("Quite right ol chap");
    ///
    /// let popped: Box<dyn Debug> = box_new_with(vec.remove_into(0));
    /// dbg!(&popped);
    ///
    /// ```
    ///
    /// [`box_new_with`]: emplacable::box_new_with
    #[inline]
    pub fn remove_into(&mut self, index: usize) -> Emplacable<T, impl EmplacableFn<T> + '_> {
        #[track_caller]
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("removal index (is {index}) should be < len (is {len})");
        }

        if index < self.len() {
            let closure = move |emplacer: &mut Emplacer<'_, T>| {
                // SAFETY: check `index < len` right above
                unsafe { self.inner.remove_into_unchecked(index, emplacer) };
            };
            // SAFETY: `remove_into_unchecked` upholds the requirements
            unsafe { Emplacable::from_fn(closure) }
        } else {
            assert_failed(index, self.len())
        }
    }

    /// Appends an element to the back of a collection.
    ///
    /// If `T` is not `Sized`, you will need
    /// `#![feature(unsized_fn_params)]` to call this.
    /// You may also need the [`unsize`] macro, which
    /// requires additional nightly features.
    ///
    /// Alternatively, you can use [`push_unsize`][0],
    /// which takes care of unsizing for you.
    ///
    /// # Example
    ///
    /// ```
    /// #![allow(internal_features)] // for `unsized_fn_params`
    /// #![feature(allocator_api, ptr_metadata, unsized_fn_params)]
    ///
    /// use core::fmt::Debug;
    ///
    /// use emplacable::unsize;
    /// use unsized_vec::UnsizedVec;
    ///
    /// let mut vec: UnsizedVec<dyn Debug> = UnsizedVec::new();
    ///
    /// vec.push(unsize!([1, 2], ([i32; 2]) -> dyn Debug));
    /// dbg!(&vec[0]);
    /// ```
    ///
    /// [0]: UnsizedVec::push_unsize
    #[inline]
    pub fn push(&mut self, value: T) {
        let size_of_val = SizeTypeFor::<T>::of_val(&value);

        self.reserve_capacity_bytes_align(1, size_of_val.get(), mem::align_of_val(&value));

        // SAFETY: reserved needed capacity above
        unsafe { self.inner.push_unchecked(value, size_of_val) }
    }

    /// Appends an element to the back of a collection
    /// after coercing it to an unsized type.
    ///
    /// # Example
    ///
    /// ```
    /// use core::fmt::Debug;
    ///
    /// use unsized_vec::UnsizedVec;
    ///
    /// let mut vec: UnsizedVec<dyn Debug> = UnsizedVec::new();
    ///
    /// vec.push_unsize([1, 2]);
    /// dbg!(&vec[0]);
    ///
    /// ```
    #[inline]
    pub fn push_unsize<S: Unsize<T>>(&mut self, value: S) {
        self.push(unsize!(value, (S) -> T));
    }

    /// Appends an element to the back of a collection.
    ///
    /// Accepts the element as an [`Emplacable<T, _>`]
    /// instead of `T` directly, analogously
    /// to [`emplacable::box_new_with`].
    ///
    /// ```
    /// #![allow(internal_features)] // for `unsized_fn_params`
    /// #![feature(allocator_api, ptr_metadata, unsized_fn_params)]
    ///
    /// use core::fmt::Debug;
    ///
    /// use unsized_vec::{unsize_vec, UnsizedVec};
    ///
    /// let mut vec_1: UnsizedVec<dyn Debug> = unsize_vec![32, "hello"];
    /// let mut vec_2: UnsizedVec<dyn Debug> = UnsizedVec::new();
    ///
    /// vec_2.push_with(vec_1.pop_into().unwrap());
    ///
    /// assert_eq!(vec_1.len(), 1);
    /// dbg!(&vec_2[0]);
    /// ```
    #[inline]
    pub fn push_with(&mut self, value: Emplacable<T, impl EmplacableFn<T>>) {
        self.inner.push_with(value);
    }

    /// Removes the last element from a vector and returns it, or [`None`] if it
    /// is empty.
    ///
    /// Because `T` might be unsized, and functions can't return
    /// unsized values directly, this method returns the element using
    /// the "emplacer" mechanism. You can pass the returned [`Emplacable<T, _>`]
    /// to a function like [`box_new_with`] to get the contained `T`.
    ///
    /// # Example
    ///
    /// ```
    /// use core::fmt::Debug;
    ///
    /// use emplacable::{box_new_with, Emplacable};
    /// use unsized_vec::{UnsizedVec};
    ///
    /// let mut vec = UnsizedVec::<dyn Debug>::new();
    ///
    /// dbg!(vec.is_empty());
    /// let nothing: Option<Box<dyn Debug>> = vec.pop_into().map(box_new_with);
    /// assert!(nothing.is_none());
    ///
    /// vec.push_unsize("A beautiful day today");
    /// let popped: Option<Box<dyn Debug>> = vec.pop_into().map(box_new_with);
    /// let unwrapped: Box<dyn Debug> = popped.unwrap();
    /// dbg!(&unwrapped);
    ///
    /// vec.push_unsize("innit?");
    /// dbg!(&vec);
    ///
    /// let mut popped_emplacable: Emplacable<dyn Debug, _> = vec.pop_into().unwrap();
    ///
    /// // vec.push_unsize("yea"); // error: cannot borrow `vec` as mutable more than once at a time
    /// // The `vec` will remain borrowed until you consume the `Emplacable`!
    ///
    /// // or we can just drop it...
    /// // dropping an `Emplacable` drops
    /// // the contained value.
    /// popped_emplacable;
    ///
    /// assert!(vec.is_empty());
    ///
    /// vec.push_unsize("yea"); // works now
    ///
    /// ```
    ///
    /// [`box_new_with`]: emplacable::box_new_with
    #[inline]
    pub fn pop_into(&mut self) -> Option<Emplacable<T, impl EmplacableFn<T> + '_>> {
        if !self.is_empty() {
            let closure = move |emplacer: &mut Emplacer<'_, T>| {
                // SAFETY: checked above that vec is non-empty
                unsafe { self.inner.pop_into_unchecked(emplacer) }
            };

            // SAFETY: `pop_into_unchecked` upholds the requirements of this closure
            Some(unsafe { Emplacable::from_fn(closure) })
        } else {
            None
        }
    }

    /// Returns the number of elements in the vector, also referred to
    /// as its 'length'.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of used bytes in the vector.
    #[must_use]
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.inner.byte_len()
    }

    /// Returns `true` if the vector contains no elements.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to an element,
    /// or `None` if `index` is out of range.
    #[must_use]
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        // SAFETY: Bounds check done right before
        (index < self.len()).then(|| unsafe { self.get_unchecked(index) })
    }

    /// Returns a mutable reference to an element,
    /// or `None` if `index` is out of range.
    #[must_use]
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        // SAFETY: Bounds check done right before
        (index < self.len()).then(|| unsafe { self.get_unchecked_mut(index) })
    }

    /// Returns a reference to an element, without doing bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    #[must_use]
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        // SAFETY: precondition of function
        unsafe { self.inner.get_unchecked_raw(index).as_ref() }
    }

    /// Returns a mutable reference to an element, without doing bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    #[must_use]
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        // SAFETY: precondition of function
        unsafe { self.inner.get_unchecked_raw(index).as_mut() }
    }

    /// Returns an iterator over references to the elements of this vec.
    #[must_use]
    #[inline]
    pub fn iter(&self) -> UnsizedIter<'_, T> {
        UnsizedIter {
            inner: self.inner.iter(),
        }
    }

    /// Returns an iterator over mutable references to the elements of this vec.
    #[must_use]
    #[inline]
    pub fn iter_mut(&mut self) -> UnsizedIterMut<'_, T> {
        UnsizedIterMut {
            inner: self.inner.iter_mut(),
        }
    }

    /// Coerces this Vec's elements to an unsized type.
    ///
    /// # Example
    ///
    /// ```
    /// use core::fmt::Debug;
    ///
    /// use unsized_vec::UnsizedVec;
    ///
    /// let sized: Vec<u32> = vec![3, 4, 5];
    /// let unsize: UnsizedVec<dyn Debug> = UnsizedVec::unsize(sized.into());
    /// dbg!(&unsize);
    /// ```
    #[must_use]
    #[inline]
    pub fn unsize<U>(self) -> UnsizedVec<U>
    where
        T: Sized + Unsize<U>,
        U: ?Sized,
    {
        UnsizedVec {
            inner: <U as UnsizedVecImpl>::Impl::from_sized(self.inner),
        }
    }

    #[must_use]
    #[inline]
    fn unused_byte_cap(&self) -> usize {
        // SAFETY: len <= cap
        unsafe { self.byte_capacity().unchecked_sub(self.byte_len()) }
    }
}

impl<T> Default for UnsizedVec<T>
where
    T: ?Sized,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// The iterator returned by [`UnsizedVec::iter`].
#[repr(transparent)]
pub struct UnsizedIter<'a, T>
where
    T: ?Sized + 'a,
{
    inner: <<T as UnsizedVecImpl>::Impl as UnsizedVecProvider<T>>::Iter<'a>,
}

/// The iterator returned by [`UnsizedVec::iter_mut`].
#[repr(transparent)]
pub struct UnsizedIterMut<'a, T>
where
    T: ?Sized + 'a,
{
    inner: <<T as UnsizedVecImpl>::Impl as UnsizedVecProvider<T>>::IterMut<'a>,
}

impl<T> From<::alloc::vec::Vec<T>> for UnsizedVec<T> {
    #[inline]
    fn from(value: ::alloc::vec::Vec<T>) -> Self {
        UnsizedVec { inner: value }
    }
}

impl<T> From<UnsizedVec<T>> for ::alloc::vec::Vec<T> {
    #[inline]
    fn from(value: UnsizedVec<T>) -> Self {
        value.inner
    }
}

impl<T> Index<usize> for UnsizedVec<T>
where
    T: ?Sized,
{
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of range")
    }
}

impl<T> IndexMut<usize> for UnsizedVec<T>
where
    T: ?Sized,
{
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("index out of range")
    }
}

impl<'a, T> From<core::slice::Iter<'a, T>> for UnsizedIter<'a, T>
where
    T: 'a,
{
    #[inline]
    fn from(value: core::slice::Iter<'a, T>) -> Self {
        UnsizedIter { inner: value }
    }
}

impl<'a, T> From<UnsizedIter<'a, T>> for core::slice::Iter<'a, T>
where
    T: 'a,
{
    #[inline]
    fn from(value: UnsizedIter<'a, T>) -> Self {
        value.inner
    }
}

macro_rules! iter_ref {
    ($iter_ty:ident $($muta:ident)?) => {
        impl<'a, T> Iterator for $iter_ty<'a, T>
        where
            T: ?Sized + 'a,
        {
            type Item = &'a $($muta)? T;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.inner.next()
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.inner.size_hint()
            }

            #[inline]
            fn count(self) -> usize {
                self.inner.count()
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<Self::Item> {
                self.inner.nth(n)
            }

            #[inline]
            fn last(self) -> Option<Self::Item> {
                self.inner.last()
            }

            #[inline]
            fn for_each<F>(self, f: F)
            where
                F: FnMut(Self::Item),
            {
                self.inner.for_each(f);
            }

            #[inline]
            fn all<F>(&mut self, f: F) -> bool
            where
                F: FnMut(Self::Item) -> bool,
            {
                self.inner.all(f)
            }

            #[inline]
            fn any<F>(&mut self, f: F) -> bool
            where
                F: FnMut(Self::Item) -> bool,
            {
                self.inner.any(f)
            }

            #[inline]
            fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
            where
                P: FnMut(&Self::Item) -> bool,
            {
                self.inner.find(predicate)
            }

            #[inline]
            fn find_map<B, F>(&mut self, f: F) -> Option<B>
            where
                F: FnMut(Self::Item) -> Option<B>,
            {
                self.inner.find_map(f)
            }

            #[inline]
            fn position<P>(&mut self, predicate: P) -> Option<usize>
            where
                P: FnMut(Self::Item) -> bool,
            {
                self.inner.position(predicate)
            }
        }

        impl<'a, T> DoubleEndedIterator for $iter_ty<'a, T>
        where
            T: ?Sized + 'a,
        {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                self.inner.next_back()
            }

            #[inline]
            fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
                self.inner.nth_back(n)
            }
        }

        impl<'a, T> ExactSizeIterator for $iter_ty<'a, T>
        where
            T: ?Sized + 'a,
        {}
        impl<'a, T> FusedIterator for $iter_ty<'a, T>
        where
            T: ?Sized + 'a,
        {}
    }
}

iter_ref!(UnsizedIter);
iter_ref!(UnsizedIterMut mut);

impl<'a, T> IntoIterator for &'a UnsizedVec<T>
where
    T: ?Sized + 'a,
{
    type Item = &'a T;

    type IntoIter = UnsizedIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut UnsizedVec<T>
where
    T: ?Sized + 'a,
{
    type Item = &'a mut T;

    type IntoIter = UnsizedIterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, F> FromIterator<Emplacable<T, F>> for UnsizedVec<T>
where
    T: ?Sized,
    F: EmplacableFn<T>,
{
    #[inline]
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Emplacable<T, F>>,
    {
        let mut vec = UnsizedVec::new();
        vec.extend(iter);
        vec
    }
}

impl<T, F> Extend<Emplacable<T, F>> for UnsizedVec<T>
where
    T: ?Sized,
    F: EmplacableFn<T>,
{
    #[inline]
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Emplacable<T, F>>,
    {
        fn extend_inner<T: ?Sized, F: EmplacableFn<T>, I: Iterator<Item = Emplacable<T, F>>>(
            vec: &mut UnsizedVec<T>,
            iter: I,
        ) {
            vec.reserve_exact(iter.size_hint().0);
            for emplacable in iter {
                vec.push_with(emplacable);
            }
        }

        extend_inner(self, iter.into_iter());
    }
}

impl<T> Debug for UnsizedVec<T>
where
    T: ?Sized + Debug,
{
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T> Clone for UnsizedVec<T>
where
    T: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        let mut ret = UnsizedVec::with_capacity_bytes_align(
            self.capacity(),
            self.byte_capacity(),
            self.align(),
        );
        for elem in self {
            ret.push(elem.clone());
        }
        ret
    }
}

impl<T, U> PartialEq<UnsizedVec<U>> for UnsizedVec<T>
where
    T: ?Sized + PartialEq<U>,
    U: ?Sized,
{
    #[inline]
    fn eq(&self, other: &UnsizedVec<U>) -> bool {
        self.len() == other.len() && self.iter().zip(other).all(|(l, r)| l == r)
    }
}

impl<T> Eq for UnsizedVec<T> where T: ?Sized + Eq {}

impl<T, U> PartialOrd<UnsizedVec<U>> for UnsizedVec<T>
where
    T: ?Sized + PartialOrd<U>,
    U: ?Sized,
{
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

impl<T> Ord for UnsizedVec<T>
where
    T: ?Sized + Ord,
{
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

impl<T> Hash for UnsizedVec<T>
where
    T: ?Sized + Hash,
{
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        for elem in self {
            elem.hash(state);
        }
    }
}

#[cfg(feature = "serde")]
use serde::{ser::SerializeSeq, Serialize};

#[cfg(feature = "serde")]
impl<T> Serialize for UnsizedVec<T>
where
    T: ?Sized + Serialize,
{
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

/// Impementation detail of `unsized_vec` macro.
#[doc(hidden)]
pub trait PushToUnsizedVec<U: ?Sized> {
    fn push_to_unsized_vec(self, vec: &mut UnsizedVec<U>);
}

impl<T: ?Sized> PushToUnsizedVec<T> for T {
    #[inline]
    fn push_to_unsized_vec(self, vec: &mut UnsizedVec<T>) {
        vec.push(self);
    }
}

impl<T: ?Sized, F: EmplacableFn<T>> PushToUnsizedVec<T> for Emplacable<T, F> {
    #[inline]
    fn push_to_unsized_vec(self, vec: &mut UnsizedVec<T>) {
        vec.push_with(self);
    }
}

/// Like the standard library's [`vec`] macro.
/// Accepts both raw unsized `T`s and
/// [`Emplacable<T,_>`]s.
///
/// However, this does not accept sized values implementing
/// [`Unsize<T>`]; you can use [`unsize_vec`] for that.
///
/// # Example
///
/// ```
/// #![allow(internal_features)] // for `unsized_fn_params`
/// #![feature(allocator_api, ptr_metadata, unsized_fn_params)]
///
/// use emplacable::unsize;
/// use unsized_vec::{UnsizedVec, unsized_vec};
///
/// let my_vec = unsized_vec![[23_u32, 17], [16, 34], [23, 47]];
///
/// let mut my_vec_unsized: UnsizedVec<[u32]> = my_vec.unsize();
///
/// let another_vec = unsized_vec![unsize!([42], ([u32; 1]) -> [u32]), my_vec_unsized.remove_into(2)];
/// ```
///
/// [`vec`]: macro@alloc::vec
#[macro_export]
macro_rules! unsized_vec {
    () => (
        $crate::UnsizedVec::new()
    );
    ($($x:expr),+ $(,)?) => (
        {
            let mut ret = $crate::UnsizedVec::new();
            $($crate::PushToUnsizedVec::push_to_unsized_vec($x, &mut ret);)+
            ret
        }
    );
}

/// Like [`unsized_vec`], but unsizes its arguments
/// using the [`Unsize`] trait.
///
/// Accepts sized values that can coerce to an unsized `T`.
/// If you have raw unsized `T`s or [`Emplacable<T,_>`]s,
/// use [`unsized_vec`] instead.
///
/// # Example
///
/// ```
/// use core::fmt::Debug;
///
/// use unsized_vec::{unsize_vec, UnsizedVec};
///
/// let my_vec: UnsizedVec<dyn Debug> = unsize_vec![1, "hello!", 97.5];
/// ```
#[macro_export]
macro_rules! unsize_vec {
    () => (
        $crate::UnsizedVec::new()
    );
    ($($x:expr),+ $(,)?) => (
        {
            let mut ret = $crate::UnsizedVec::new();
            $(ret.push_unsize($x);)+
            ret
        }
    );
}
