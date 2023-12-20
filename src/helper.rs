pub(crate) mod valid_align;
pub(crate) mod valid_size;

use core::{alloc::Layout, hash::Hash, mem, ptr::Pointee};

use crate::marker::Aligned;

use self::{valid_align::ValidAlign, valid_size::ValidSizeUnaligned};

/// Used by `UnsizedVec` to only store offset and pointer metadata
/// when the latter can't be derived from the former.

pub(crate) trait MetadataFromSize: Aligned {
    fn from_size(size: ValidSizeUnaligned) -> <Self as Pointee>::Metadata;
}

impl<T> MetadataFromSize for T {
    fn from_size(_: ValidSizeUnaligned) -> <Self as Pointee>::Metadata {}
}

impl<T> MetadataFromSize for [T] {
    fn from_size(size: ValidSizeUnaligned) -> <Self as Pointee>::Metadata {
        debug_assert!(size.get() != 0 || size == ValidSizeUnaligned::ZERO);
        size.get().checked_div(mem::size_of::<T>()).unwrap_or(0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct FullMetadataRemainder<T: Copy + Send + Sync + Ord + Hash + Unpin>(T);

/// Used by `UnsizedVec` to only store offset and pointer metadata
/// when the latter can't be derived from the former.
pub(crate) trait MetadataRemainder<T: ?Sized>:
    Copy + Send + Sync + Ord + Hash + Unpin
{
    #[must_use]
    fn from_metadata(meta: <T as Pointee>::Metadata) -> Self;

    #[must_use]
    fn as_metadata(self, size: ValidSizeUnaligned) -> <T as Pointee>::Metadata;
}

impl<T: ?Sized> MetadataRemainder<T> for FullMetadataRemainder<<T as Pointee>::Metadata> {
    #[inline]
    fn from_metadata(meta: <T as Pointee>::Metadata) -> Self {
        FullMetadataRemainder(meta)
    }

    #[inline]
    fn as_metadata(self, _: ValidSizeUnaligned) -> <T as Pointee>::Metadata {
        self.0
    }
}
impl<T: ?Sized + MetadataFromSize> MetadataRemainder<T> for () {
    #[inline]
    fn from_metadata(_: <T as Pointee>::Metadata) -> Self {}

    #[inline]
    fn as_metadata(self, size: ValidSizeUnaligned) -> <T as Pointee>::Metadata {
        <T as MetadataFromSize>::from_size(size)
    }
}
pub(crate) trait SplitMetadata {
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

pub(crate) const fn decompose(layout: Layout) -> (ValidSizeUnaligned, ValidAlign) {
    // SAFETY: `Layout` can't return invalid size/align
    unsafe {
        (
            ValidSizeUnaligned::new_unchecked(layout.size()),
            ValidAlign::new_unckecked(layout.align()),
        )
    }
}
