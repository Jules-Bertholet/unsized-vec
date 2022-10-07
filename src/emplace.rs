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
    /// that the closue you pass in writes a valid value of type `T` to the passed-in pointer.
    pub unsafe fn into_inner(
        &mut self,
    ) -> &mut dyn FnMut(Layout, <T as Pointee>::Metadata, &mut dyn FnMut(*mut PhantomData<T>)) {
        // Safety: `repr(transparent)` guarantees compatible layouts
        unsafe { mem::transmute(self) }
    }
}

/// Run the given function, and return its result emplaced into a `Box`.
pub fn box_new_with<T: ?Sized>(unsized_ret: impl FnOnce(&mut Emplacer<T>)) -> Box<T> {
    let mut uninit_box = MaybeUninit::uninit();
    let mut initialized: bool = false;

    let closure = &mut |layout, meta, closure: &mut dyn FnMut(*mut PhantomData<T>)| {
        let box_ptr = unsafe { alloc::alloc(layout) as *mut PhantomData<T> };
        closure(box_ptr);
        let init_box = unsafe { Box::from_raw(ptr::from_raw_parts_mut(box_ptr as *mut (), meta)) };
        uninit_box.write(init_box);

        initialized = true;
    };

    unsized_ret(unsafe { Emplacer::new(closure) });

    if !initialized {
        panic!("Emplacer wasn't run")
    }

    unsafe { uninit_box.assume_init() }
}
