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
        unsafe { mem::transmute(e) }
    }

    /// # Safety
    ///
    /// If you unwrap this `Emplacer` and call the resulting closure, you must ensure
    /// that the closue you pass in writes a valid value of type `T` to the passed-in pointer
    /// (or panics, or runs forever, or exits without returning in some other otherwise-sound way).
    pub unsafe fn into_inner(
        &mut self,
    ) -> &mut dyn FnMut(Layout, <T as Pointee>::Metadata, &mut dyn FnMut(*mut PhantomData<T>)) {
        // Safety: `repr(transparent)` guarantees compatible layouts
        unsafe { mem::transmute(self) }
    }
}

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
        unsafe { alloc::dealloc(self.ptr, self.layout) }
    }
}

/// Run the given function, and return its result emplaced into a `Box`.
pub fn box_new_with<T: ?Sized>(unsized_ret: impl FnOnce(&mut Emplacer<T>)) -> Box<T> {
    let mut uninit_box = MaybeUninit::uninit();
    let mut initialized: bool = false;

    let closure = &mut |layout, meta, closure: &mut dyn FnMut(*mut PhantomData<T>)| {
        // If `closure` panics and triggers an unwind,
        // this will be dropped, which will call `dealloc` on `ptr`
        // and ensure no memory is leaked.
        let deallocer = PointerDeallocer {
            ptr: unsafe { alloc::alloc(layout) },
            layout,
        };

        closure(deallocer.ptr as *mut PhantomData<T>);
        let init_box =
            unsafe { Box::from_raw(ptr::from_raw_parts_mut(deallocer.ptr as *mut (), meta)) };

        // Now that we've succesfully initialized the box,
        // we don't want to deallocate its memory.
        mem::forget(deallocer);

        uninit_box.write(init_box);

        initialized = true;
    };

    unsized_ret(unsafe { Emplacer::new(closure) });

    if !initialized {
        panic!("Emplacer wasn't run")
    }

    unsafe { uninit_box.assume_init() }
}
