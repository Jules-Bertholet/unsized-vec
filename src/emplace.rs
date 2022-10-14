//! Traits and functions to support functions that return unsized values.

#![allow(clippy::type_complexity)]

use alloc_crate::{
    alloc::{self, Layout},
    boxed::Box,
};
use core::{
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ptr::{self, Pointee},
};

/// Passed as the last argument to functions that return unsized values.
/// Wraps a closure that tells the function where to write its return value.
#[repr(transparent)]
pub struct Emplacer<T: ?Sized>(
    dyn FnMut(Layout, <T as Pointee>::Metadata, &mut dyn FnMut(*mut PhantomData<T>)),
);

impl<T: ?Sized> Emplacer<T> {
    /// # Safety
    ///
    /// The passed-in closure, if it runs the closure that it recieves as a third argument, must
    /// pass in a pointer that has the alignment of the `Layout` passed to it, and is valid
    /// for writes of the number of bytes corresponding to the `Layout`.
    pub unsafe fn new(
        e: &mut dyn FnMut(Layout, <T as Pointee>::Metadata, &mut dyn FnMut(*mut PhantomData<T>)),
    ) -> &mut Emplacer<T> {
        // Safety: `repr(transparent)` guarantees compatible layouts
        unsafe {
            &mut *(e as *mut dyn for<'a> FnMut(
                Layout,
                <T as Pointee>::Metadata,
                &'a mut (dyn core::ops::FnMut(*mut PhantomData<T>) + 'a),
            ) as *mut Emplacer<T>)
        }
    }

    /// # Safety
    ///
    /// If you unwrap this `Emplacer` and call the resulting closure, you must ensure
    /// that the closure you pass in writes a valid value of type `T` to the passed-in pointer
    /// (or panics, or runs forever, or exits without returning in some other otherwise-sound way).
    ///
    /// The value you write must also correspond to the layout and pointer metadata you pass in.
    pub unsafe fn into_inner(
        &mut self,
    ) -> &mut dyn FnMut(Layout, <T as Pointee>::Metadata, &mut dyn FnMut(*mut PhantomData<T>)) {
        // Safety: `repr(transparent)` guarantees compatible layouts
        unsafe { mem::transmute(self) }
    }
}

/// Run the given function, and return its result emplaced into a `Box`.
pub fn box_new_with<T: ?Sized>(unsized_ret: impl FnOnce(&mut Emplacer<T>)) -> Box<T> {
    /// Helper to ensure `box_new_with` doesn't leak memory
    /// if the unsized-value-returning function panics.
    ///
    /// Deallocates the contained pointer when dropped.
    /// Must be costructed with a pointer returned by
    /// `alloc::alloc`, and `layout` must be the same
    /// as was passed to the `alloc` call.
    /// Must `mem::forget` the struct to avoid deallocating
    /// its memory.
    struct PointerDeallocer {
        ptr: *mut u8,
        layout: Layout,
    }

    impl Drop for PointerDeallocer {
        fn drop(&mut self) {
            if self.layout.size() != 0 {
                // Safety: if layout is non-zero then this pointer
                // should have been allocated with this layout
                unsafe { alloc::dealloc(self.ptr, self.layout) }
            }
        }
    }

    let mut uninit_box = MaybeUninit::uninit();
    let mut initialized: bool = false;

    let closure = &mut |layout: Layout, meta, closure: &mut dyn FnMut(*mut PhantomData<T>)| {
        // If `closure` panics and triggers an unwind,
        // this will be dropped, which will call `dealloc` on `ptr`
        // and ensure no memory is leaked.
        let deallocer = PointerDeallocer {
            ptr: if layout.size() == 0 {
                crate::DANGLING_PERFECT_ALIGN.as_ptr().cast()
            } else {
                // Safety: just checked that layout is not zero-sized
                unsafe { alloc::alloc(layout) }
            },
            layout,
        };

        if deallocer.ptr.is_null() {
            // Don't want to deallocate a raw pointer
            mem::forget(deallocer);
            alloc::handle_alloc_error(layout);
        }

        closure(deallocer.ptr.cast());

        // Safety: Pointer either is to 0-sized allocation or comes from global allocator
        let init_box =
            unsafe { Box::from_raw(ptr::from_raw_parts_mut(deallocer.ptr.cast::<()>(), meta)) };

        // Now that we've succesfully initialized the box,
        // we don't want to deallocate its memory.
        mem::forget(deallocer);

        uninit_box.write(init_box);

        initialized = true;
    };

    unsized_ret(unsafe { Emplacer::new(closure) });

    assert!(initialized, "Emplacer wasn't run");

    unsafe { uninit_box.assume_init() }
}
