//! Defines the [`Aligned`] trait.

use core::{
    ffi::CStr,
    mem,
    ptr::{self, NonNull},
};

use crate::helper::valid_align::ValidAlign;

/// Implemented for types that have an alignment known at compile-time.
///
/// # Safety
///
/// `ALIGN` must be equal to the alignment of all values of the type.
pub(crate) unsafe trait Aligned {
    /// The alignment of this type.
    const ALIGN: ValidAlign;

    /// A dangling, well-aligned pointer thin pointer for the type.
    const DANGLING_THIN: NonNull<()> = NonNull::new(ptr::invalid_mut(Self::ALIGN.get())).unwrap();
}

// SAFETY: mem::align_of::<T>() is correct
unsafe impl<T> Aligned for T {
    const ALIGN: ValidAlign = ValidAlign::new(mem::align_of::<T>()).unwrap();
}

// SAFETY: alignment of `[T]` equals alignment of `T`
unsafe impl<T> Aligned for [T] {
    const ALIGN: ValidAlign = ValidAlign::new(mem::align_of::<T>()).unwrap();
}

// SAFETY: All `str`s have the same alignment
unsafe impl Aligned for str {
    const ALIGN: ValidAlign = ValidAlign::new(mem::align_of_val("")).unwrap();
}

// SAFETY: All `CStrs`s have the same alignment
unsafe impl Aligned for CStr {
    const ALIGN: ValidAlign = ValidAlign::new(mem::align_of_val(&
    // SAFETY: passed-in ends with NUL car
    unsafe {
        CStr::from_bytes_with_nul_unchecked(&[b'\0'])
    }))
    .unwrap();
}
