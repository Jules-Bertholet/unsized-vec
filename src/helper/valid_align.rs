//! Copied and pasted from the Rust standard library:
//! <https://doc.rust-lang.org/src/core/mem/valid_align.rs.html>
//!
//! All credit goes to the stdlib developers.

#![allow(clippy::enum_clike_unportable_variant)]

use core::{
    cmp, fmt, hash, mem,
    num::NonZeroUsize,
    ptr::{self, NonNull},
};

/// A type storing a `usize` which is a power of two, and thus
/// represents a possible alignment in the rust abstract machine.
///
/// Note that particularly large alignments, while representable in this type,
/// are likely not to be supported by actual allocators and linkers.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct ValidAlign(ValidAlignEnum);

// ValidAlign is `repr(usize)`, but via extra steps.
const _: () = assert!(mem::size_of::<ValidAlign>() == mem::size_of::<usize>());
const _: () = assert!(mem::align_of::<ValidAlign>() == mem::align_of::<usize>());

impl ValidAlign {
    pub(crate) const ONE: Self = Self::new(1).unwrap();

    /// Creates a `ValidAlign` from a power-of-two `usize`.
    ///
    /// Returns `None` if `align` is not a power of two.
    ///
    /// `align` must be a power of two.
    ///
    /// Equivalently, it must be `1 << exp` for some `exp` in `0..usize::BITS`.
    /// It must *not* be zero.
    #[must_use]
    #[inline]
    pub(crate) const fn new(align: usize) -> Option<Self> {
        if align.is_power_of_two() {
            Some(
                // SAFETY: By check above, this must be a power of two, and
                // our variants encompass all possible powers of two.
                unsafe { Self::new_unckecked(align) },
            )
        } else {
            None
        }
    }

    /// Creates a `ValidAlign` from a power-of-two `usize`.
    ///
    /// Returns `None` if `align` is not a power of two.
    ///
    /// # Safety
    ///
    /// `align` must be a power of two.
    ///
    /// Equivalently, it must be `1 << exp` for some `exp` in `0..usize::BITS`.
    /// It must *not* be zero.
    #[must_use]
    #[inline]
    pub(crate) const unsafe fn new_unckecked(align: usize) -> Self {
        // SAFETY: By function preconditions, this must be a power of two, and
        // our variants encompass all possible powers of two.
        unsafe { mem::transmute::<usize, ValidAlign>(align) }
    }

    #[must_use]
    #[inline]
    #[allow(clippy::cast_possible_truncation)] // Clippy is not smart enough to realize this can/t fail
    pub(crate) const fn get(self) -> usize {
        self.0 as usize
    }

    #[must_use]
    #[inline]
    pub(super) const fn minus_1(self) -> usize {
        // SAFETY: align always >= 1
        unsafe { self.get().unchecked_sub(1) }
    }

    #[must_use]
    #[inline]
    pub(crate) const fn as_nonzero(self) -> NonZeroUsize {
        // SAFETY: All the discriminants are non-zero.
        unsafe { NonZeroUsize::new_unchecked(self.0 as usize) }
    }

    #[must_use]
    #[inline]
    pub(crate) const fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Returns the base 2 logarithm of the alignment.
    ///
    /// This is always exact, as `self` represents a power of two.
    #[must_use]
    #[inline]
    pub(crate) const fn log2(self) -> u32 {
        self.as_nonzero().trailing_zeros()
    }

    /// Returns a dangling poiter with a numeric value equal to this alignment.
    #[must_use]
    #[inline]
    pub(crate) const fn dangling_thin(self) -> NonNull<()> {
        // SAFETY: self != 0
        unsafe { NonNull::new_unchecked(ptr::invalid_mut(self.get())) }
    }

    /// Returns the alignment of this value.
    #[must_use]
    #[inline]
    pub(crate) const fn of_val<T: ?Sized>(val: &T) -> Self {
        // SAFETY: `align_of_val` returns valid alignments
        unsafe { Self::new_unckecked(mem::align_of_val(val)) }
    }
}

impl fmt::Debug for ValidAlign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} (1 << {:?})", self.as_nonzero(), self.log2())
    }
}

impl From<ValidAlign> for NonZeroUsize {
    #[inline]
    fn from(value: ValidAlign) -> Self {
        value.as_nonzero()
    }
}

impl From<ValidAlign> for usize {
    #[inline]
    fn from(value: ValidAlign) -> Self {
        value.get()
    }
}

impl PartialEq for ValidAlign {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl Eq for ValidAlign {}

impl PartialOrd for ValidAlign {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ValidAlign {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.get().cmp(&other.get())
    }
}

#[allow(clippy::derive_hash_xor_eq)]
impl hash::Hash for ValidAlign {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.as_nonzero().hash(state);
    }
}

#[cfg(target_pointer_width = "16")]
#[derive(Clone, Copy)]
#[repr(usize)]
enum ValidAlignEnum {
    _Align1Shl0 = 1 << 0,
    _Align1Shl1 = 1 << 1,
    _Align1Shl2 = 1 << 2,
    _Align1Shl3 = 1 << 3,
    _Align1Shl4 = 1 << 4,
    _Align1Shl5 = 1 << 5,
    _Align1Shl6 = 1 << 6,
    _Align1Shl7 = 1 << 7,
    _Align1Shl8 = 1 << 8,
    _Align1Shl9 = 1 << 9,
    _Align1Shl10 = 1 << 10,
    _Align1Shl11 = 1 << 11,
    _Align1Shl12 = 1 << 12,
    _Align1Shl13 = 1 << 13,
    _Align1Shl14 = 1 << 14,
    _Align1Shl15 = 1 << 15,
}

#[cfg(target_pointer_width = "32")]
#[derive(Clone, Copy)]
#[repr(usize)]
enum ValidAlignEnum {
    _Align1Shl0 = 1 << 0,
    _Align1Shl1 = 1 << 1,
    _Align1Shl2 = 1 << 2,
    _Align1Shl3 = 1 << 3,
    _Align1Shl4 = 1 << 4,
    _Align1Shl5 = 1 << 5,
    _Align1Shl6 = 1 << 6,
    _Align1Shl7 = 1 << 7,
    _Align1Shl8 = 1 << 8,
    _Align1Shl9 = 1 << 9,
    _Align1Shl10 = 1 << 10,
    _Align1Shl11 = 1 << 11,
    _Align1Shl12 = 1 << 12,
    _Align1Shl13 = 1 << 13,
    _Align1Shl14 = 1 << 14,
    _Align1Shl15 = 1 << 15,
    _Align1Shl16 = 1 << 16,
    _Align1Shl17 = 1 << 17,
    _Align1Shl18 = 1 << 18,
    _Align1Shl19 = 1 << 19,
    _Align1Shl20 = 1 << 20,
    _Align1Shl21 = 1 << 21,
    _Align1Shl22 = 1 << 22,
    _Align1Shl23 = 1 << 23,
    _Align1Shl24 = 1 << 24,
    _Align1Shl25 = 1 << 25,
    _Align1Shl26 = 1 << 26,
    _Align1Shl27 = 1 << 27,
    _Align1Shl28 = 1 << 28,
    _Align1Shl29 = 1 << 29,
    _Align1Shl30 = 1 << 30,
    _Align1Shl31 = 1 << 31,
}

#[cfg(target_pointer_width = "64")]
#[derive(Clone, Copy)]
#[repr(usize)]
enum ValidAlignEnum {
    _Align1Shl0 = 1 << 0,
    _Align1Shl1 = 1 << 1,
    _Align1Shl2 = 1 << 2,
    _Align1Shl3 = 1 << 3,
    _Align1Shl4 = 1 << 4,
    _Align1Shl5 = 1 << 5,
    _Align1Shl6 = 1 << 6,
    _Align1Shl7 = 1 << 7,
    _Align1Shl8 = 1 << 8,
    _Align1Shl9 = 1 << 9,
    _Align1Shl10 = 1 << 10,
    _Align1Shl11 = 1 << 11,
    _Align1Shl12 = 1 << 12,
    _Align1Shl13 = 1 << 13,
    _Align1Shl14 = 1 << 14,
    _Align1Shl15 = 1 << 15,
    _Align1Shl16 = 1 << 16,
    _Align1Shl17 = 1 << 17,
    _Align1Shl18 = 1 << 18,
    _Align1Shl19 = 1 << 19,
    _Align1Shl20 = 1 << 20,
    _Align1Shl21 = 1 << 21,
    _Align1Shl22 = 1 << 22,
    _Align1Shl23 = 1 << 23,
    _Align1Shl24 = 1 << 24,
    _Align1Shl25 = 1 << 25,
    _Align1Shl26 = 1 << 26,
    _Align1Shl27 = 1 << 27,
    _Align1Shl28 = 1 << 28,
    _Align1Shl29 = 1 << 29,
    _Align1Shl30 = 1 << 30,
    _Align1Shl31 = 1 << 31,
    _Align1Shl32 = 1 << 32,
    _Align1Shl33 = 1 << 33,
    _Align1Shl34 = 1 << 34,
    _Align1Shl35 = 1 << 35,
    _Align1Shl36 = 1 << 36,
    _Align1Shl37 = 1 << 37,
    _Align1Shl38 = 1 << 38,
    _Align1Shl39 = 1 << 39,
    _Align1Shl40 = 1 << 40,
    _Align1Shl41 = 1 << 41,
    _Align1Shl42 = 1 << 42,
    _Align1Shl43 = 1 << 43,
    _Align1Shl44 = 1 << 44,
    _Align1Shl45 = 1 << 45,
    _Align1Shl46 = 1 << 46,
    _Align1Shl47 = 1 << 47,
    _Align1Shl48 = 1 << 48,
    _Align1Shl49 = 1 << 49,
    _Align1Shl50 = 1 << 50,
    _Align1Shl51 = 1 << 51,
    _Align1Shl52 = 1 << 52,
    _Align1Shl53 = 1 << 53,
    _Align1Shl54 = 1 << 54,
    _Align1Shl55 = 1 << 55,
    _Align1Shl56 = 1 << 56,
    _Align1Shl57 = 1 << 57,
    _Align1Shl58 = 1 << 58,
    _Align1Shl59 = 1 << 59,
    _Align1Shl60 = 1 << 60,
    _Align1Shl61 = 1 << 61,
    _Align1Shl62 = 1 << 62,
    _Align1Shl63 = 1 << 63,
}
