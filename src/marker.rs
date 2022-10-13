use core::{
    hash::Hash,
    hint::unreachable_unchecked,
    mem,
    num::NonZeroUsize,
    panic::{RefUnwindSafe, UnwindSafe},
};

/// # Safety
///
/// `ALIGN` must be equal to the alignment of all values of the type.
pub unsafe trait Aligned {
    const ALIGN: NonZeroUsize;
}

unsafe impl<T> Aligned for T {
    const ALIGN: NonZeroUsize = NonZeroUsize::new(mem::align_of::<T>()).unwrap();
}

unsafe impl<T> Aligned for [T] {
    const ALIGN: NonZeroUsize = NonZeroUsize::new(mem::align_of::<T>()).unwrap();
}

/// # Safety
///
/// If `from_align_{unchecked}` is used to create an instance of
/// an `AlignStorage` from an alignment returned by `mem::align_of_val()`,
/// `to_align` must return that same alignment.
///
/// `to_align` must always return a power of 2.
pub(super) unsafe trait AlignStorage<T: ?Sized>:
    Copy + Send + Sync + Ord + Hash + Unpin + UnwindSafe + RefUnwindSafe
{
    /// The smallest alignment that a value of type `T` can have.
    const MIN_ALIGN: Self;

    /// Turn an alignment of a value of type `T` into an `AlignStorage`.
    /// May return `None` if `align` is not a power of two, but is not guaranteed to do so
    /// if `T: Aligned`.
    #[must_use]
    fn from_align(align: usize) -> Option<Self>;

    /// Turn an alignment of a value of type `T` into an `AlignStorage`.
    /// May return `None` if `align` is not a power of two, and may also return `None`
    /// for `T: Aligned` if `align != T::ALIGN`.
    #[must_use]
    fn from_align_exact(align: usize) -> Option<Self>;

    /// Given an offset `usize`, which is assumed to be a multiple of
    /// `MIN_ALIGN`, returns the smallest number at least as big as the offset
    /// that is a multiple of the alignment that this `AlignStorage` encodes.
    ///
    /// # Safety
    ///
    /// - Must not overflow
    /// - `offset` must be a multiple of `MIN_ALIGN`
    #[must_use]
    unsafe fn unchecked_align_offset_to(self, offset: usize) -> usize;

    /// Returns the aligneent used to generate this value.
    #[must_use]
    fn to_align(self) -> NonZeroUsize;
}

unsafe impl<T: ?Sized> AlignStorage<T> for NonZeroUsize {
    const MIN_ALIGN: Self = NonZeroUsize::new(1).unwrap();

    #[inline]
    fn from_align(align: usize) -> Option<Self> {
        if align.is_power_of_two() {
            // Safety: `is_power_of_two` implies non-zero
            Some(unsafe { NonZeroUsize::new_unchecked(align) })
        } else {
            None
        }
    }

    #[inline]
    fn from_align_exact(align: usize) -> Option<Self> {
        <Self as AlignStorage<T>>::from_align(align)
    }

    #[inline]
    unsafe fn unchecked_align_offset_to(self, offset: usize) -> usize {
        // Safety: caller must assure preconditions are met
        unsafe {
            offset
                .checked_next_multiple_of(self.into())
                .unwrap_unchecked()
        }
    }

    #[inline]
    fn to_align(self) -> NonZeroUsize {
        self
    }
}

unsafe impl<T: ?Sized + Aligned> AlignStorage<T> for () {
    const MIN_ALIGN: Self = ();

    #[inline]
    fn from_align(align: usize) -> Option<Self> {
        if cfg!(debug_assertions) && !align.is_power_of_two() {
            None
        } else {
            Some(())
        }
    }

    #[inline]
    fn from_align_exact(align: usize) -> Option<Self> {
        if cfg!(debug_assertions) && align != T::ALIGN.into() {
            None
        } else {
            Some(())
        }
    }

    #[inline]
    unsafe fn unchecked_align_offset_to(self, offset: usize) -> usize {
        if offset % T::ALIGN == 0 {
            offset
        } else {
            // Safety: caller must assure preconditions are met
            unsafe { unreachable_unchecked() }
        }
    }

    #[inline]
    fn to_align(self) -> NonZeroUsize {
        <T as Aligned>::ALIGN
    }
}
pub(super) trait StoreAlign {
    type AlignStore: AlignStorage<Self>;
}

impl<T: ?Sized> StoreAlign for T {
    default type AlignStore = NonZeroUsize;
}

impl<T: ?Sized + Aligned> StoreAlign for T {
    type AlignStore = ();
}
