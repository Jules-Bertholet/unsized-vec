#![no_std]
#![feature(forget_unsized, ptr_metadata, strict_provenance, unsized_fn_params)]
#![deny(unsafe_op_in_unsafe_fn)]

extern crate alloc as alloc_crate;

use alloc_crate::{
    alloc::{self, Layout},
    boxed::Box,
};
use core::{
    cmp,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ops::{Index, IndexMut},
    ptr::{self, NonNull, Pointee},
};

pub trait Aligned {
    const ALIGN: usize;

    fn dangling_thin() -> NonNull<()>;
}

impl<T> Aligned for T {
    const ALIGN: usize = mem::align_of::<Self>();

    fn dangling_thin() -> NonNull<()> {
        NonNull::<T>::dangling().cast()
    }
}

impl<T> Aligned for [T] {
    const ALIGN: usize = mem::align_of::<T>();

    fn dangling_thin() -> NonNull<()> {
        NonNull::<T>::dangling().cast()
    }
}

struct MetadataStorage<P: ?Sized> {
    metadata: <P as Pointee>::Metadata,
    offset_next: usize,
}

pub struct UnsizedVec<T: ?Sized + Aligned> {
    ptr: NonNull<()>,
    cap: usize,
    _marker: PhantomData<(*mut T, T)>,
    metadata: alloc_crate::vec::Vec<MetadataStorage<T>>,
}

unsafe impl<T: Send> Send for UnsizedVec<T> {}
unsafe impl<T: Sync> Sync for UnsizedVec<T> {}

impl<T: ?Sized + Aligned> UnsizedVec<T> {
    #[must_use]
    pub fn new() -> Self {
        UnsizedVec {
            ptr: <T as Aligned>::dangling_thin(),
            cap: 0,
            _marker: PhantomData,
            metadata: Default::default(),
        }
    }

    fn grow(&mut self, by_at_least: usize) {
        // since we set the capacity to usize::MAX when T has size 0,
        // getting to here necessarily means the UnsizedVec is overfull.

        let new_cap = cmp::max(2 * self.cap, by_at_least);
        let new_layout = Layout::from_size_align(new_cap, <T as Aligned>::ALIGN).unwrap();

        // Ensure that the new allocation doesn't exceed `isize::MAX` bytes.
        assert!(
            isize::try_from(new_layout.size()).is_ok(),
            "Allocation too large"
        );

        let new_ptr = if self.cap == 0 {
            unsafe { alloc::alloc(new_layout) }
        } else {
            let old_layout = Layout::from_size_align(self.cap, <T as Aligned>::ALIGN).unwrap();
            let old_ptr = self.ptr.as_ptr().cast::<u8>();
            unsafe { alloc::realloc(old_ptr, old_layout, new_layout.size()) }
        };

        // If allocation fails, `new_ptr` will be null, in which case we abort.
        self.ptr = match NonNull::new(new_ptr.cast::<()>()) {
            Some(p) => p,
            None => alloc::handle_alloc_error(new_layout),
        };
        self.cap = new_cap;
    }

    #[inline]
    fn offset_next(&self) -> usize {
        self.metadata.last().map_or(0, |m| m.offset_next)
    }

    pub fn push(&mut self, elem: T) {
        let size_of_val = mem::size_of_val(&elem);
        let offset_next = self.offset_next();
        let new_end_offset = size_of_val + self.offset_next();
        if new_end_offset >= self.cap {
            self.grow(size_of_val);
        }

        let metadata = ptr::metadata(&elem);

        unsafe {
            ptr::copy_nonoverlapping(
                core::ptr::addr_of!(elem).cast(),
                self.ptr.as_ptr().cast::<u8>().add(offset_next),
                size_of_val,
            );
        }

        mem::forget_unsized(elem);

        self.metadata.push(MetadataStorage {
            metadata,
            offset_next: new_end_offset,
        });
    }

    pub fn pop_unwrap(
        &mut self,
        emplacer: &mut dyn FnMut(Layout, <T as Pointee>::Metadata, &mut dyn FnMut(*mut ())),
    ) {
        let MetadataStorage {
            metadata,
            offset_next: offset_end,
        } = self.metadata.pop().unwrap();
        let offset_start = self.offset_next();
        let len = offset_end - offset_start;
        emplacer(
            Layout::from_size_align(len, T::ALIGN).unwrap(),
            metadata,
            &mut |dst| unsafe {
                ptr::copy_nonoverlapping(
                    self.ptr.as_ptr().cast::<u8>().add(offset_start),
                    dst.cast::<u8>(),
                    len,
                )
            },
        )
    }

    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    #[must_use]
    #[inline]
    fn index_raw(&self, index: usize) -> (*mut (), <T as Pointee>::Metadata) {
        let metadata = self.metadata[index].metadata;
        let offset = if index == 0 {
            0
        } else {
            self.metadata[index - 1].offset_next
        };
        let start_ptr: *mut u8 = self.ptr.as_ptr().cast();
        let offset_ptr = unsafe { start_ptr.add(offset).cast::<()>() };
        (offset_ptr, metadata)
    }
}

impl<T: ?Sized + Aligned> Drop for UnsizedVec<T> {
    fn drop(&mut self) {
        while let Some(MetadataStorage { metadata, .. }) = self.metadata.pop() {
            let offset = self.offset_next();

            let start_ptr: *mut u8 = self.ptr.as_ptr().cast();
            let offset_ptr = unsafe { start_ptr.add(offset).cast::<()>() };

            let raw_wide_ptr: *mut T = ptr::from_raw_parts_mut(offset_ptr, metadata);
            unsafe { raw_wide_ptr.drop_in_place() };
        }

        if self.cap != 0 {
            unsafe {
                alloc::dealloc(
                    self.ptr.as_ptr().cast::<u8>(),
                    Layout::from_size_align(self.cap, <T as Aligned>::ALIGN).unwrap(),
                );
            }
        }
    }
}

impl<T: ?Sized + Aligned> Index<usize> for UnsizedVec<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let (offset_ptr, metadata) = self.index_raw(index);

        let raw_wide_ptr: *const T = ptr::from_raw_parts(offset_ptr, metadata);
        unsafe { raw_wide_ptr.as_ref().unwrap_unchecked() }
    }
}

impl<T: ?Sized + Aligned> IndexMut<usize> for UnsizedVec<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let (offset_ptr, metadata) = self.index_raw(index);

        let raw_wide_ptr: *mut T = ptr::from_raw_parts_mut(offset_ptr, metadata);
        unsafe { raw_wide_ptr.as_mut().unwrap_unchecked() }
    }
}

impl<T> Default for UnsizedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub fn box_new_with<T: ?Sized>(
    unsized_ret: impl FnOnce(&mut dyn FnMut(Layout, <T as Pointee>::Metadata, &mut dyn FnMut(*mut ()))),
) -> Box<T> {
    let mut uninit_box = MaybeUninit::uninit();
    unsized_ret(&mut |layout, meta, closure| {
        let box_ptr = unsafe { alloc::alloc(layout) as *mut () };
        closure(box_ptr);
        let init_box = unsafe { Box::from_raw(ptr::from_raw_parts_mut(box_ptr, meta)) };
        uninit_box.write(init_box);
    });

    unsafe { uninit_box.assume_init() }
}
