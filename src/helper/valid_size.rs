use core::{alloc::Layout, fmt::Debug, hash::Hash, marker::PhantomData, mem};

use crate::marker::Aligned;

use super::valid_align::ValidAlign;

/// A type storing a `usize`, that, when rounded up to the nearest multiple of `T: Align`,
/// is less than or equal to `isize::MAX`.
///
/// This mirros the requirements of [`Layout`].
///
#[repr(transparent)]
pub(crate) struct ValidSize<T: ?Sized + Aligned>(usize, PhantomData<fn() -> T>);

impl<T: ?Sized + Aligned> Clone for ValidSize<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: ?Sized + Aligned> Copy for ValidSize<T> {}
impl<T: ?Sized + Aligned> Debug for ValidSize<T> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}
impl<T: ?Sized + Aligned> PartialEq for ValidSize<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T: ?Sized + Aligned> Eq for ValidSize<T> {}
impl<T: ?Sized + Aligned> Hash for ValidSize<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}
impl<T: ?Sized + Aligned> PartialOrd for ValidSize<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: ?Sized + Aligned> Ord for ValidSize<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T: ?Sized + Aligned> ValidSize<T> {
    pub(crate) const ZERO: Self = ValidSize::new(0).unwrap();

    pub(crate) const MAX: Self = ValidSize(
        isize::MAX as usize - (<T as Aligned>::ALIGN.as_usize() - 1),
        PhantomData,
    );

    /// Creates a `ValidSize` from a `usize` that, wher rounded up
    /// to the nearest multiple of `Self::ALIGN` is less than
    /// or equal to `isize::MAX`.
    ///
    ///  # Safety
    ///
    /// `size <= isize::MAX as usize` must hold.
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn new_unchecked(size: usize) -> Self {
        debug_assert!(size <= Self::MAX.get());

        ValidSize(size, PhantomData)
    }

    /// Creates a `ValidSize` from a `usize` that is less than or equal to
    /// `isize::MAX`.
    ///
    ///  Returns `None` if `size > isize::MAX as usize`.
    #[must_use]
    #[inline]
    pub(crate) const fn new(size: usize) -> Option<Self> {
        if size <= Self::MAX.get() {
            Some(ValidSize(size, PhantomData))
        } else {
            None
        }
    }

    /// Rounds down if it would overflow.
    #[must_use]
    #[inline]
    pub(crate) const fn new_squished(size: usize) -> Self {
        if size <= Self::MAX.get() {
            // SAFETY: ensured by if guard
            unsafe { Self::new_unchecked(size) }
        } else {
            Self::MAX
        }
    }

    #[must_use]
    #[inline]
    pub(crate) const fn get(self) -> usize {
        self.0
    }

    #[must_use]
    #[inline]
    pub(crate) const fn checked_add(self, rhs: Self) -> Option<Self> {
        if let Some(sum) = self.get().checked_add(rhs.get()) {
            ValidSize::new(sum)
        } else {
            None
        }
    }

    /// # Safety
    ///
    /// Must not overflow
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn unchecked_add(self, rhs: Self) -> Self {
        // SAFETY: precondition of function
        unsafe { ValidSize::new_unchecked(self.get().unchecked_add(rhs.get())) }
    }

    /// # Safety
    ///
    /// Must not underflow
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn unchecked_sub(self, rhs: Self) -> Self {
        // SAFETY: precondition of function
        unsafe { ValidSize::new_unchecked(self.get().unchecked_sub(rhs.get())) }
    }

    #[must_use]
    #[inline]
    pub(crate) const fn as_layout(self) -> Layout {
        // SAFETY: `T::ALIGN` is a valid align, and conditions on `self` ensure it meets
        // the requirements of the call.
        unsafe { Layout::from_size_align_unchecked(self.get(), T::ALIGN.get()) }
    }

    #[must_use]
    #[inline]
    pub(crate) const fn as_unaligned(self) -> ValidSizeUnaligned {
        // SAFETY: Reducing alignment strictly increases the set of valid sizes
        unsafe { ValidSizeUnaligned::new_unchecked(self.get()) }
    }

    #[must_use]
    #[inline]
    pub(crate) const fn of_val(val: &T) -> Self {
        // SAFETY: `size_of_val` returns valid size
        unsafe { Self::new_unchecked(mem::size_of_val(val)) }
    }
}

/// For when align isn't known at compile-time
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ValidSizeUnaligned(ValidSize<()>);

impl ValidSizeUnaligned {
    pub(crate) const ZERO: Self = Self(ValidSize::ZERO);

    /// Creates a `ValidSize` from a `usize` that is less than
    /// or equal to `isize::MAX`.
    ///
    ///  # Safety
    ///
    /// `size <= isize::MAX as usize` must hold.
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn new_unchecked(size: usize) -> Self {
        // SAFETY: precondition of function
        Self(unsafe { ValidSize::new_unchecked(size) })
    }

    /// Creates a `ValidSize` from a `usize` that is less than or equal to
    /// `isize::MAX`.
    ///
    ///  Returns `None` if `size > isize::MAX as usize`.
    #[must_use]
    #[inline]
    const fn new(size: usize) -> Option<Self> {
        if let Some(size) = ValidSize::new(size) {
            Some(Self(size))
        } else {
            None
        }
    }

    #[must_use]
    #[inline]
    pub(crate) const fn get(self) -> usize {
        self.0.get()
    }

    #[must_use]
    #[inline]
    pub(crate) const fn max_for_align(align: ValidAlign) -> Self {
        // SAFETY: `1 <= align` and `align - 1 <= isize::MAX`
        // for any possible `align`.
        unsafe { Self::new_unchecked((isize::MAX as usize).unchecked_sub(align.minus_1())) }
    }

    #[must_use]
    #[inline]
    pub(crate) const fn new_padded_to(size: usize, align: ValidAlign) -> Option<Self> {
        let align_m_1 = align.minus_1();

        let Some(sum) = size.checked_add(align_m_1) else {
            return None;
        };

        let new = sum & !align_m_1;

        Self::new(new)
    }

    /// Rounds down if it would overflow.
    #[must_use]
    #[inline]
    pub(crate) const fn new_squished_to(size: usize, align: ValidAlign) -> Self {
        let max = Self::max_for_align(align);
        if size <= max.get() {
            // SAFETY: ensured by if guard
            unsafe { Self::new_unchecked(size) }
        } else {
            max
        }
    }

    #[must_use]
    #[inline]
    pub(crate) const fn checked_add(self, rhs: Self) -> Option<Self> {
        if let Some(sum) = self.0.checked_add(rhs.0) {
            Some(Self(sum))
        } else {
            None
        }
    }

    /// # Safety
    ///
    /// Must not overflow
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn unchecked_add(self, rhs: Self) -> Self {
        // SAFETY: precondition of function
        Self(unsafe { self.0.unchecked_add(rhs.0) })
    }

    /// # Safety
    ///
    /// Must not underflow
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn unchecked_sub(self, rhs: Self) -> Self {
        // SAFETY: precondition of function
        Self(unsafe { self.0.unchecked_sub(rhs.0) })
    }

    #[must_use]
    #[inline]
    pub(crate) const fn checked_pad_to(self, align: ValidAlign) -> Option<Self> {
        let align_m_1 = align.minus_1();

        // SAFETY: align - 1 <= isize::MAX, self <= isize::MAX, so no possibility of overflow
        let new = unsafe { self.get().unchecked_add(align_m_1) } & !align_m_1;

        // We are padded, so just need to check for < isize::MAX
        Self::new(new)
    }

    /// # Safety
    ///
    /// Must not overflow `isize`
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn unchecked_pad_to(self, align: ValidAlign) -> Self {
        let align_m_1 = align.minus_1();

        // SAFETY: align - 1 <= isize::MAX, self <= isize::MAX, so no possibility of overflow
        let new = unsafe { self.get().unchecked_add(align_m_1) } & !align_m_1;

        // SAFETY: preconditions of function
        unsafe { Self::new_unchecked(new) }
    }

    #[must_use]
    #[inline]
    pub(crate) const fn checked_add_pad(self, rhs: usize, align: ValidAlign) -> Option<Self> {
        if let Some(sum) = self.get().checked_add(rhs) {
            Self::new_padded_to(sum, align)
        } else {
            None
        }
    }

    /// # Safety
    ///
    /// Must meet requirements of `Layout::from_size_align_unchecked`
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn as_layout_with_align_unchecked(self, align: ValidAlign) -> Layout {
        // SAFETY: Preconditions of function
        unsafe { Layout::from_size_align_unchecked(self.get(), align.get()) }
    }

    #[must_use]
    #[inline]
    pub(crate) const fn of_val<T: ?Sized>(val: &T) -> Self {
        // SAFETY: size of values is always a multiple of alignment,
        // and less than or equal to `isize::MAX`
        unsafe { Self::new_unchecked(mem::size_of_val(val)) }
    }
}
