//! Defines the interface that the three implementations of
//! `UnsizedVec` (sized, unsized + aligned, unaligned)
//! implement.

use core::{
    hash::Hash,
    iter::FusedIterator,
    marker::Unsize,
    mem,
    panic::{RefUnwindSafe, UnwindSafe},
    ptr::NonNull,
};

use crate::{
    helper::{
        valid_align::ValidAlign,
        valid_size::{ValidSize, ValidSizeUnaligned},
    },
    marker::Aligned,
    TryReserveError,
};

use emplacable::{Emplacable, EmplacableFn, Emplacer};

mod aligned;
mod sized;
mod unaligned;

pub(super) trait Align<T: ?Sized>:
    Copy + Send + Sync + Ord + Hash + Unpin + UnwindSafe + RefUnwindSafe
{
    #[must_use]
    fn new(align: usize) -> Option<Self>;

    #[must_use]
    fn of_val(val: &T) -> Self;
}

impl<T: ?Sized> Align<T> for ValidAlign {
    #[inline]
    fn new(align: usize) -> Option<Self> {
        ValidAlign::new(align)
    }

    #[inline]
    fn of_val(val: &T) -> Self {
        ValidAlign::of_val(val)
    }
}

impl<T: ?Sized + Aligned> Align<T> for () {
    #[inline]
    fn new(_: usize) -> Option<Self> {
        Some(())
    }

    #[inline]
    fn of_val(_: &T) -> Self {}
}

pub(super) trait Size<T: ?Sized>:
    Copy + Send + Sync + Ord + Hash + Unpin + UnwindSafe + RefUnwindSafe
{
    #[must_use]
    fn of_val(val: &T) -> Self;

    #[must_use]
    fn get(self) -> usize;
}

impl<T: ?Sized> Size<T> for ValidSizeUnaligned {
    #[inline]
    fn of_val(val: &T) -> Self {
        Self::of_val(val)
    }

    #[inline]
    fn get(self) -> usize {
        self.get()
    }
}

impl<T: ?Sized + Aligned> Size<T> for ValidSize<T> {
    #[inline]
    fn of_val(val: &T) -> Self {
        Self::of_val(val)
    }

    #[inline]
    fn get(self) -> usize {
        self.get()
    }
}

impl<T> Size<T> for () {
    #[inline]
    fn of_val(_: &T) -> Self {}

    #[inline]
    fn get(self) -> usize {
        mem::size_of::<T>()
    }
}

/// Implementation of `UnsizedVec`.
/// Different impls for for `T: ?Aligned`, `T: Aligned`,
/// and `T: Sized`.
pub(super) trait UnsizedVecProvider<T: ?Sized> {
    type Align: Align<T>;
    type Size: Size<T>;

    type Iter<'a>: Iterator<Item = &'a T> + DoubleEndedIterator + ExactSizeIterator + FusedIterator
    where
        T: 'a,
        Self: 'a;

    type IterMut<'a>: Iterator<Item = &'a mut T>
        + DoubleEndedIterator
        + ExactSizeIterator
        + FusedIterator
    where
        T: 'a,
        Self: 'a;

    const NEW_ALIGN_1: Self;
    const NEW_ALIGN_PTR: Self;

    #[must_use]
    fn capacity(&self) -> usize;

    #[must_use]
    fn byte_len(&self) -> usize;

    #[must_use]
    fn byte_capacity(&self) -> usize;

    #[must_use]
    fn align(&self) -> usize;

    fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError>;

    fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError>;

    /// Try to reserve additional byte or align capacity,
    /// on top of existing unused capacity. Exact (doesn't overallocate).
    fn try_reserve_additional_bytes_align(
        &mut self,
        additional_bytes: usize,
        align: Self::Align,
    ) -> Result<(), TryReserveError>;

    fn shrink_capacity_bytes_align_to(
        &mut self,
        min_capacity: usize,
        min_byte_capacity: usize,
        min_align: Self::Align,
    );

    /// # Safety
    ///
    /// Does no capacity or bounds checks.
    /// Doesn't check for sufficient space for metadata, either,
    ///
    /// `size` must correpond to the actual
    /// size of `element`.
    unsafe fn insert_unchecked(&mut self, index: usize, value: T, size: Self::Size);

    /// # Safety
    ///
    /// Like `insert`, but does no bounds checks.
    unsafe fn insert_with_unchecked(
        &mut self,
        index: usize,
        value: Emplacable<T, impl EmplacableFn<T>>,
    );

    /// # Safety
    ///
    /// `index < self.len()` must hold.
    unsafe fn remove_into_unchecked(&mut self, index: usize, emplacer: &mut Emplacer<'_, T>);

    /// # Safety
    ///
    /// Like `push`, but does no capacity checks.
    /// Doesn't check for sufficient space for metadata, either.
    ///
    /// `size` must correpond to the actual
    /// size of `elem`.
    unsafe fn push_unchecked(&mut self, elem: T, size: Self::Size);

    fn push_with(&mut self, value: Emplacable<T, impl EmplacableFn<T>>);

    /// # Safety
    ///
    /// `!self.is_empty()` must hold.
    unsafe fn pop_into_unchecked(&mut self, emplacer: &mut Emplacer<'_, T>);

    #[must_use]
    fn len(&self) -> usize;

    /// # Safety
    ///
    /// `index` must be contained in `0..self.len()`
    #[must_use]
    unsafe fn get_unchecked_raw(&self, index: usize) -> NonNull<T>;

    #[must_use]
    fn iter(&self) -> Self::Iter<'_>;

    #[must_use]
    fn iter_mut(&mut self) -> Self::IterMut<'_>;

    #[must_use]
    fn from_sized<S>(vec: ::alloc::vec::Vec<S>) -> Self
    where
        S: Unsize<T>;
}

pub(super) trait UnsizedVecImpl {
    type Impl: UnsizedVecProvider<Self>;
}

pub(super) trait AlignedVecProvider<T: ?Sized + Aligned>: UnsizedVecProvider<T> {}

pub(super) trait AlignedVecImpl: Aligned {
    type Impl: AlignedVecProvider<Self>;
}

impl<T: ?Sized + Aligned> UnsizedVecImpl for T {
    type Impl = <T as AlignedVecImpl>::Impl;
}
