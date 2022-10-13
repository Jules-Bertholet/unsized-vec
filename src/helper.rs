use core::mem;

use crate::{AlignStorage, StoreAlign};

// Returns the alignemnt of the val pointed to by `val`.
#[must_use]
#[inline]
pub(super) fn align_of_val<T: ?Sized>(val: &T) -> <T as StoreAlign>::AlignStore {
    // Safety: Storing a valid alignment
    unsafe {
        <T as StoreAlign>::AlignStore::from_align_exact(mem::align_of_val(val)).unwrap_unchecked()
    }
}
