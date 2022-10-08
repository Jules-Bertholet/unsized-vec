use core::{mem, num::NonZeroUsize};

pub(super) fn align_of_val<T: ?Sized>(val: &T) -> NonZeroUsize {
    // Safety: alignment is never 0
    unsafe { NonZeroUsize::new_unchecked(mem::align_of_val(val)) }
}

// # Safety
//
// Must not overflow `usize`
pub(super) unsafe fn next_multiple_of_unchecked(num: usize, align: NonZeroUsize) -> usize {
    unsafe { num.checked_next_multiple_of(align.get()).unwrap_unchecked() }
}
