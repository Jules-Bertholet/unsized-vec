use core::{
    hash::Hash,
    mem,
    ptr::{NonNull, Pointee},
};

/// # Safety
///
/// `ALIGN` must be equal to the alignment of all values of the type
pub unsafe trait Aligned {
    const ALIGN: usize;

    fn dangling_thin() -> NonNull<()>;
}

unsafe impl<T> Aligned for T {
    const ALIGN: usize = mem::align_of::<Self>();

    fn dangling_thin() -> NonNull<()> {
        NonNull::<T>::dangling().cast()
    }
}

unsafe impl<T> Aligned for [T] {
    const ALIGN: usize = mem::align_of::<T>();

    fn dangling_thin() -> NonNull<()> {
        NonNull::<T>::dangling().cast()
    }
}

// The following traits are for optimizing `Vec`'s size.
// We only want to store size and metadata separately when needed,
// and only want to store size when it isn't known at compile time.

trait MetadataFromSize: Aligned {
    fn from_size(size: usize) -> <Self as Pointee>::Metadata;
}

impl<T> MetadataFromSize for T {
    fn from_size(_: usize) -> <Self as Pointee>::Metadata {}
}

impl<T> MetadataFromSize for [T] {
    fn from_size(size: usize) -> <Self as Pointee>::Metadata {
        size / mem::size_of::<T>()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct FullMetadataRemainder<T: Copy + Send + Sync + Ord + Hash + Unpin>(T);

pub(super) trait MetadataRemainder<T: ?Sized>:
    Copy + Send + Sync + Ord + Hash + Unpin
{
    #[must_use]
    fn from_metadata(meta: <T as Pointee>::Metadata) -> Self;

    #[must_use]
    fn to_metadata(self, size: usize) -> <T as Pointee>::Metadata;
}

impl<T: ?Sized> MetadataRemainder<T> for FullMetadataRemainder<<T as Pointee>::Metadata> {
    #[inline]
    fn from_metadata(meta: <T as Pointee>::Metadata) -> Self {
        FullMetadataRemainder(meta)
    }

    #[inline]
    fn to_metadata(self, _: usize) -> <T as Pointee>::Metadata {
        self.0
    }
}

impl<T: ?Sized + MetadataFromSize> MetadataRemainder<T> for () {
    #[inline]
    fn from_metadata(_: <T as Pointee>::Metadata) -> Self {}

    #[inline]
    fn to_metadata(self, size: usize) -> <T as Pointee>::Metadata {
        <T as MetadataFromSize>::from_size(size)
    }
}

pub(super) trait SplitMetadata {
    type Remainder: MetadataRemainder<Self>;
}

impl<T: ?Sized> SplitMetadata for T {
    default type Remainder = FullMetadataRemainder<<T as Pointee>::Metadata>;
}

impl<T: ?Sized + MetadataFromSize> SplitMetadata for T {
    type Remainder = ();
}
