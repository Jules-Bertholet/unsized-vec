use core::{mem, num::NonZeroUsize};

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
