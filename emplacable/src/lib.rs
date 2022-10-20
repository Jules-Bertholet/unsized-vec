//! Machinery to support functions that return unsized values.
//!
//! Written to support the `unsized_vec` crate, but is independent of it.
//! Requires nightly Rust.
//!
//! Unsized values can take many forms:
//!
//! - On stable Rust, values of unsized types like [`str`],
//! `[u8]`, and `dyn Any` are generally encountered behind a pointer,
//! like `&str` or `Box<dyn Any>`.
//!
//! - Nightly Rust provides limited support for passing unsized values
//! by value as arguments to functions, using the `unsized_fn_params`
//! feature. There is also `unsized_locals`, for storing these values
//! on the stack using alloca. (However, that feature is "incomplete" and
//! this crate doesn't make use of it). But even with thse two feature
//! gates enabled, functions cannot return unsized values directly.
//! Also, the only way to produce a by-value unsized value in today's Rust
//! is by dereferencing a [`Box`]; this crate provides the [`unsize`] macro
//! to work around this limitation.
//!
//! - For functions that return unsized values, this crate
//! provides the [`Emplacable`] type. Functions that want
//! to return a value of type `T`, where `T` is unsized, return an
//! `Emplacable<T, _>` instead. `Emplacable<T>` wraps a closure;
//! that closure contains instructions for writing out a `T` to
//! a caller-provided region of memory. Other functions accept the `Emplacable`
//! as an argument and call its contained closure to write out the
//! `T` to some allocation provided by them. For example, this crate
//! provides the [`box_new_with`] function, which turns an `Emplacable<T>`
//! into a [`Box<T>`].
//!
//! ## Converting between types
//!
//! | I have                    | I want                     | I can use                    |
//! |---------------------------|----------------------------|------------------------------|
//! | `[i32; 2]`                | `[i32]`                    | [`unsize`]                   |
//! | `[i32; 2]`                | `Emplacable<[i32; 2], _>`  | [`Into::into`]               |
//! | `[i32]`                   | `Box<[i32]>`               | [`box_new`]                  |
//! | `Box<[i32; 2]>`           | `Box<[i32]>`               | [`CoerceUnsized`]            |
//! | `Box<[i32]>`              | `[i32]`                    | dereference the box with `*` |
//! | `Box<[i32]>`              | `Emplacable<[i32], _>`     | [`Into::into`]               |
//! | `Vec<i32>`                | `Emplacable<[i32], _>`     | [`Into::into`]               |
//! | `Emplacable<[i32; 2], _>` | `[i32; 2]`                 | [`Emplacable::get`]          |
//! | `Emplacable<[i32; 2], _>` | `Emplacable<[i32], _>`     | [`Into::into`]               |
//! | `Emplacable<[i32; 2], _>` | `Emplacable<dyn Debug, _>` | [`Emplacable::unsize`]       |
//! | `Emplacable<[i32], _>`    | `Box<[i32]>`               | [`box_new_with`]             |
//! | `Emplacable<[i32], _>`    | `Vec<i32>`                 | [`Into::into`]               |
//! | `Emplacable<[i32], _>`    | `Rc<[i32]>`                | [`Into::into`]               |
//! | `Emplacable<[i32], _>`    | `Arc<[i32]>`               | [`Into::into`]               |
//! | `&[i32]`                  | `Box<[i32]>`               | [`Into::into`]               |
//! | `&[i32]`                  | `Emplacable<[i32], _>`     | [`Into::into`]               |
//!
//! You can replace  `[i32; 2]` and `[i32]` above by any pair of types (`T`, `U`)
//! such that [`T: Unsize<U>`][`Unsize`].
//!
//! ## A note on examples
//!
//! This crate has very few examples, as it provides tools to work with unsized types
//! but no fun things that use the tools. If you want usage sensible examples,
//! check out `unsized-vec`'s documentation and the `examples` folder on GitHub.
//!
//! [`Unsize`]: core::marker::Unsize
//! [`CoerceUnsized`]: core::ops::CoerceUnsized

#![deny(
    unsafe_op_in_unsafe_fn,
    clippy::alloc_instead_of_core,
    clippy::std_instead_of_alloc,
    clippy::std_instead_of_core
)]
#![warn(
    missing_docs,
    clippy::semicolon_if_nothing_returned,
    clippy::undocumented_unsafe_blocks
)]
#![feature(
    allocator_api,
    array_zip,
    closure_lifetime_binder,
    forget_unsized,
    min_specialization,
    ptr_metadata,
    strict_provenance,
    type_alias_impl_trait,
    unchecked_math,
    unsize,
    unsized_fn_params
)]
#![no_std]

#[doc(hidden)]
#[cfg(feature = "alloc")]
pub extern crate alloc as alloc_crate;

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "alloc")]
use alloc_crate::{
    alloc::{self, handle_alloc_error},
    boxed::Box,
    rc::Rc,
    sync::Arc,
};
use core::{
    alloc::{AllocError, Allocator, Layout},
    cell::Cell,
    ffi::CStr,
    marker::{PhantomData, Unsize},
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::FnMut,
    pin::Pin,
    ptr::{self, addr_of, addr_of_mut, NonNull, Pointee},
};
#[cfg(feature = "std")]
use std::{ffi::OsStr, path::Path};

#[doc(hidden)]
pub mod macro_exports {
    #[cfg(feature = "alloc")]
    pub use alloc_crate;
    pub use core;
    pub use u8;
}

/// Helper for coercing values to unsized types.
///
/// The `unsized_fn_params` has some rough edges when it comes to coercing
/// sized values to unsized ones by value. This macro works around that.
///
/// If you have a value `val` of type `SizedType`, and you want to coerce it
/// to `UnsizedType`, write `unsize!(val, (SizedType) -> UnsizedType))`.
///
/// Requires `#![feature(allocator_api, ptr_metadata)]`,
/// and probably useless without `unsized_fn_params` or `unsized_locals`.
///
/// Also requires the `alloc` crate feature
/// (though doesn't actually allocate on the heap).
///
/// # Example
///
/// ```
/// #![feature(allocator_api, ptr_metadata, unsized_fn_params)]
///
/// use core::fmt::Debug;
///
/// use emplacable::{box_new, unsize};
///
/// let mut my_box: Box<dyn Debug> = box_new(unsize!("hello world!", (&str) -> dyn Debug));
///
/// dbg!(&my_box);
/// ```
#[cfg(feature = "alloc")]
#[macro_export]
macro_rules! unsize {
    ($e:expr, ($src:ty) -> $dst:ty) => {{
        // To make the coercion happen, we:
        // 1. Make a fake "allocator" that just stores
        //   `mem::size_of<$src>()` bytes on the stack
        // 2. Allocate our sized value in a `Box` in our fake allocator
        // 3. Coerce the box to hold an unsized value
        // 4. Move out of the box
        use $crate::{
            ImplementationDetailDoNotUse,
            macro_exports::{
                alloc_crate::boxed::Box,
            },
        };

        let val: $src = $e;

        let new_alloc: ImplementationDetailDoNotUse<$src> = ImplementationDetailDoNotUse::NEW;
        let boxed_sized: Box<$src, &ImplementationDetailDoNotUse<$src>> = Box::new_in(val, &new_alloc);
        let boxed_unsized: Box<$dst, &ImplementationDetailDoNotUse<$src>> = boxed_sized;

        *boxed_unsized
    }};
}

// Implementation detail of `unsize` macro.
#[doc(hidden)]
#[repr(transparent)]
pub struct ImplementationDetailDoNotUse<T>(Cell<MaybeUninit<T>>);

#[doc(hidden)]
impl<T> ImplementationDetailDoNotUse<T> {
    pub const NEW: Self = Self(Cell::new(MaybeUninit::uninit()));
}

// SAFETY: this is an unsound implementation of the trait,
// you can't `allocate` more than once without UB. We are careful not
// to break this invariant inside the macro, but `ImplementationDetailDoNotUse`
// should not be leaked to arbritrary code!
unsafe impl<T> Allocator for ImplementationDetailDoNotUse<T> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert_eq!(layout, Layout::new::<T>());
        // SAFETY: Address of `self.0` can't be null
        let thin_ptr = unsafe { NonNull::new_unchecked(self.0.as_ptr()) };
        Ok(NonNull::from_raw_parts(
            thin_ptr.cast(),
            mem::size_of::<T>(),
        ))
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}

/// Construct a `str` from a string literal,
/// in its dereferenced form.
///
/// Requires `#![feature(allocator_api, ptr_metadata)]`,
/// and probably useless without `unsized_fn_params` or `unsized_locals`.
///
/// Also requires the `alloc` crate feature
/// (though doesn't actually allocate on the heap).
///
/// # Example
///
/// ```
/// #![feature(allocator_api, ptr_metadata, unsized_fn_params)]
///
/// use emplacable::{box_new, by_value_str};
///
/// let boxed_str: Box<str> = box_new(by_value_str!("why hello there"));
/// dbg!(&*boxed_str);
/// ```
#[cfg(feature = "alloc")]
#[macro_export]
macro_rules! by_value_str {
    ($s:literal) => {{
        // This implementation is similar to the one from `unsize`.
        // 1. Declare a constant `&str` for the string
        // 2. Copy the contents of the constant into a buffer
        // 5. Convert a pointer into the buffer into a `Box<str>`
        // 4. Move out of the box
        use $crate::{
            macro_exports::{
                alloc_crate::boxed::Box,
                core::{
                    alloc::{AllocError, Allocator, Layout},
                    cell::Cell,
                    mem::{self, MaybeUninit},
                    ptr::{self, addr_of_mut, NonNull},
                },
                u8,
            },
            NonAllocator,
        };

        const STRING: &str = $s;
        const LEN: usize = STRING.len();
        let mut buf: [MaybeUninit<u8>; LEN] = [MaybeUninit::uninit(); LEN];

        // SAFETY: `buf` has compatible layout
        unsafe {
            ptr::copy(STRING.as_ptr().cast::<u8>(), addr_of_mut!(buf).cast(), LEN);
        }

        let wide_ptr: *mut str = ptr::from_raw_parts_mut(addr_of_mut!(buf).cast(), LEN);

        // SAFETY: `NonAllocator::deref()`
        let boxed_str: Box<str, NonAllocator> = unsafe { Box::from_raw_in(wide_ptr, NonAllocator) };
        *boxed_str
    }};
}

// Implementation detail of `by_value_str`.
#[doc(hidden)]
pub struct NonAllocator;

// SAFETY: `allocate` is a stub that is never run and always panics
unsafe impl Allocator for NonAllocator {
    fn allocate(&self, _: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unreachable!()
    }

    #[inline]
    unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {}
}

// FIXME revisit once ICE https://github.com/rust-lang/rust/issues/103666 is fixed
/*
type WithEmplacableForFn<'a, T: ?Sized + 'a> = impl EmplacableFn<T> + 'a;

/// Accepts a possibly-unsized value as a first argument,
/// turns it into an [`Emplacable`], and passes the emplacer to
/// the given closure.
///
/// If `T` is sized, you can use [`Into::into`] instead.
///
/// # Example
///
/// ```
/// #![feature(allocator_api, ptr_metadata, unsized_fn_params)]
///
/// use emplacable::{box_new_with, with_emplacable_for, unsize};
///
/// let b: Box<[i32]> = with_emplacable_for(unsize!([23, 4 ,32], ([i32; 3]) -> [i32]), box_new_with);
/// assert_eq!(b[0], 23);
///
///
///
/// ```
#[inline]
pub fn with_emplacable_for<T, F, R>(mut val: T, mut f: F) -> R
where
    T: ?Sized,
    F: for<'a> FnMut(Emplacable<T, WithEmplacableForFn<'a, T>>) -> R,
{
    /// SAFETY: `val` must point to a valid `T`. `val` must not be dropped
    /// after this function completes.
    #[inline]
    unsafe fn with_emplacable_for_inner<'a, T: ?Sized + 'a, R>(
        val: &'a mut T,
        f: &mut dyn for<'b> FnMut(Emplacable<T, WithEmplacableForFn<'b, T>>) -> R,
    ) -> R {
        fn with_emplacable_closure<'a, T: ?Sized>(val: &'a mut T) -> WithEmplacableForFn<'a, T> {
            move |emplacer: &mut Emplacer<T>| {
                // SAFETY: We
                let layout = Layout::for_value(unsafe { &*val });
                let metadata = ptr::metadata(val);
                // Safety: we call the closure right after
                let emplacer_closure = unsafe { emplacer.into_inner() };
                emplacer_closure(layout, metadata, &mut |out_ptr| {
                    if !out_ptr.is_null() {
                        // SAFETY: copying value where it belongs.
                        // We `forget` right after to prevent double-free.
                        // `Emplacer` preconditions say this can only be run once.
                        unsafe {
                            ptr::copy_nonoverlapping(
                                addr_of_mut!(*val).cast::<u8>(),
                                out_ptr.cast(),
                                layout.size(),
                            );
                        }
                    } else {
                        // SAFETY: we `mem::forget` `val` later to avoid double-drop
                        unsafe { ptr::drop_in_place(val) }
                    }
                });
            }
        }

        // SAFETY: closure fulfills safety preconditions
        let emplacable = unsafe { Emplacable::from_closure(with_emplacable_closure(val)) };

        f(emplacable)
    }

    let ret = unsafe { with_emplacable_for_inner(&mut val, &mut f) };
    //mem::forget_unsized(val);
    ret
}

#[test]
fn testddddd() {
    let b: Box<[i32]> =
        with_emplacable_for(unsize!([23, 4 ,32], ([i32; 3]) -> [i32]), box_new_with);

    let e = with_emplacable_for(*b, |e| {
        let closure = Box::new(e.into_closure());
        let closure: Box<dyn EmplacableFn<[i32]>> = closure;
        closure
    });

    let e2 = unsafe { Emplacable::from_closure(e) };

    let b = box_new_with(e2);
}
*/

/// Alias of [`for<'a> FnOnce(&'a mut Emplacer<T>)`](Emplacer<T>).
pub trait EmplacableFn<T>: for<'a> FnOnce(&'a mut Emplacer<T>)
where
    T: ?Sized,
{
}

impl<T, F> EmplacableFn<T> for F
where
    T: ?Sized,
    F: for<'a> FnOnce(&'a mut Emplacer<T>),
{
}

/// A wrapped closure that you can pass to functions like `box_new_with`,
/// that describes how to write a value of type `T` to a caller-provided
/// allocation. You can get a `T` out of an `Emplacable` through functions like
/// [`box_new_with`]. Alternately, you can drop the value of type `T` by dropping
/// the `Emplacable`. Or you can forget the value of type `T` with [`Emplacable::forget`].
#[repr(transparent)]
pub struct Emplacable<T, C>
where
    T: ?Sized,
    C: EmplacableFn<T>,
{
    closure: ManuallyDrop<C>,
    phantom: PhantomData<fn(&mut Emplacer<T>)>,
}

impl<T, C> Emplacable<T, C>
where
    T: ?Sized,
    C: EmplacableFn<T>,
{
    /// Create a new `Emplacable` from a closure.
    /// This is only useful if you are implementing
    /// a function that returns an unsized value as
    /// an [`Emplacable`].
    ///
    /// # Safety
    ///
    /// The closure `closurse` *must*, either diverge
    /// without returning, or, if it returns, then
    /// it must have used the emplacer to fully
    /// initalize the value.
    #[must_use]
    #[inline]
    pub unsafe fn from_closure(closure: C) -> Self {
        Emplacable {
            closure: ManuallyDrop::new(closure),
            phantom: PhantomData,
        }
    }

    /// Returns the closure inside this `Emplacable`.
    ///
    /// This is only useful if you are implementing
    /// a function like [`box_new_with`].
    #[must_use]
    #[inline]
    pub fn into_closure(mut self) -> C {
        // SAFETY: we `forget` self right after, so no double drop
        let closure = unsafe { ManuallyDrop::take(&mut self.closure) };
        mem::forget(self);
        closure
    }

    /// Emplaces this sized `T` onto the stack.
    #[must_use]
    #[inline]
    pub fn get(self) -> T
    where
        T: Sized,
    {
        let mut buf: MaybeUninit<T> = MaybeUninit::uninit();

        let emplacer_closure =
            &mut |_: Layout, (), inner_closure: &mut dyn FnMut(*mut PhantomData<T>)| {
                inner_closure(buf.as_mut_ptr().cast());
            };

        // SAFETY: emplacer passes in pointer to `MaybeUninit` buffer, which is of the right size/align
        let emplacer = unsafe { Emplacer::new(emplacer_closure) };

        let closure = self.into_closure();
        closure(emplacer);

        // SAFETY: `buf` was initialized by the emplacer
        unsafe { buf.assume_init() }
    }

    /// Runs the `Emplacable` closure,
    /// but doesn't run the "inner closure",
    /// so the value of type `T` is forgotten,
    /// and its destrutor is not run.
    ///
    /// If you want to drop the `T` and run its destructor,
    /// drop the `Emplacable` instead.
    #[inline]
    pub fn forget(self) {
        #[inline]
        fn forgetting_emplacer_closure<T: ?Sized>(
            _: Layout,
            _: <T as Pointee>::Metadata,
            _: &mut dyn FnMut(*mut PhantomData<T>),
        ) {
            // Do nothing. Just forget the value ever existed.
        }

        let emplacable_closure = self.into_closure();

        let ref_to_fn = &mut forgetting_emplacer_closure::<T>;

        // SAFETY: `forgetting_emplacer` fulfills the requirements
        let forgetting_emplacer = unsafe { Emplacer::new(ref_to_fn) };

        emplacable_closure(forgetting_emplacer);
    }

    /// Turns an emplacer for a sized type into one for an unsized type
    /// via an unsizing coercion (for example, array -> slice or
    /// concrete type -> trait object).
    #[must_use]
    #[inline]
    pub fn unsize<U: ?Sized>(self) -> Emplacable<U, impl EmplacableFn<U>>
    where
        T: Sized + Unsize<U>,
    {
        const fn metadata<T: Unsize<U>, U: ?Sized>() -> <U as Pointee>::Metadata {
            // Do an unsizing coercion to get the right metadata.
            let null_ptr_to_t: *const T = ptr::null();
            let null_ptr_to_u: *const U = null_ptr_to_t;
            ptr::metadata(null_ptr_to_u)
        }

        let sized_emplacable_closure = self.into_closure();

        let unsized_emplacable_closure = move |unsized_emplacer: &mut Emplacer<U>| {
            // SAFETY: We are just wrapping this emplacer
            let unsized_emplacer_closure = unsafe { unsized_emplacer.into_inner() };

            let mut sized_emplacer_closure =
                |_: Layout, _: (), sized_inner_closure: &mut dyn FnMut(*mut PhantomData<T>)| {
                    let unsized_inner_closure: &mut dyn FnMut(*mut PhantomData<U>) =
                        &mut |unsized_ptr: *mut PhantomData<U>| {
                            sized_inner_closure(unsized_ptr.cast());
                        };

                    unsized_emplacer_closure(
                        Layout::new::<T>(),
                        metadata::<T, U>(),
                        unsized_inner_closure,
                    );
                };

            // SAFETY: just wrapping the emplacer we got, fulfills the preconditions if the inner one does
            let sized_emplacer = unsafe { Emplacer::new(&mut sized_emplacer_closure) };

            sized_emplacable_closure(sized_emplacer);
        };

        // SAFETY: Again, just wrapping our input
        unsafe { Emplacable::from_closure(unsized_emplacable_closure) }
    }

    /// Creates an `Emplacable` for a slice of values of type `T` out of an iterator
    /// of values of type `T`.
    ///
    /// This function differs from [`FromIterator::from_iter`] in that the iterator is required to
    /// be an [`ExactSizeIterator`]. If `ExactSizeIterator` is incorrectly implemented,
    /// this function may panic or otherwise misbehave (but will not trigger UB).
    #[allow(clippy::should_implement_trait)] // We only take `ExactSizeIterator`s
    pub fn from_iter<I>(iter: I) -> Emplacable<[T], impl EmplacableFn<[T]>>
    where
        T: Sized,
        I: IntoIterator<Item = Self>,
        I::IntoIter: ExactSizeIterator,
    {
        let emplacables_iter = iter.into_iter();
        let len = emplacables_iter.len();

        // Panics if size overflows `isize::MAX`.
        let layout = Layout::from_size_align(
            mem::size_of::<T>().checked_mul(len).unwrap(),
            mem::align_of::<T>(),
        )
        .unwrap();

        let slice_emplacer_closure = move |slice_emplacer: &mut Emplacer<[T]>| {
            // Move ite into closure
            let emplacables_iter = emplacables_iter;

            // SAFETY: we fulfill the preconditions
            let slice_emplacer_fn = unsafe { slice_emplacer.into_inner() };

            slice_emplacer_fn(layout, len, &mut |arr_out_ptr: *mut PhantomData<[T]>| {
                if !arr_out_ptr.is_null() {
                    let elem_emplacables: I::IntoIter =
                        // SAFETY: this "inner closure" can only be called once,
                        // per preconditions of `Emplacer::new`.
                        // we `mem::forget` `elem_emplacables` below to avoid double-drop.
                        unsafe { ptr::read(&emplacables_iter) };

                    // We can't trust `ExactSizeIterator`'s `len()`,
                    // so we keep track of how many
                    // elements were actually returned.
                    let mut num_elems_copied: usize = 0;

                    let indexed_elem_emplacables = (0..len).zip(elem_emplacables);
                    indexed_elem_emplacables.for_each(|(index, elem_emplacable)| {
                        let elem_emplacable_closure = elem_emplacable.into_closure();
                        let elem_emplacer_closure = &mut move |
                                _: Layout,
                                (),
                                inner_closure: &mut dyn FnMut(*mut PhantomData<T>),
                            | {
                                // SAFETY: by fn precondition
                                inner_closure(unsafe { arr_out_ptr.cast::<T>().add(index).cast() });
                            };

                        // SAFETY: `elem_emplacer_closure` passes a pointer with the correct offset from the
                        // start of the allocation
                        let elem_emplacer = unsafe { Emplacer::new(elem_emplacer_closure) };
                        elem_emplacable_closure(elem_emplacer);

                        num_elems_copied += 1;
                    });

                    assert_eq!(num_elems_copied, len);
                } else {
                    let emplacables_iter: I::IntoIter =
                        // SAFETY: this "inner closure" can only be called once,
                        // per preconditions of `Emplacer::new`.
                        // we `mem::forget` `elem_emplacables` below to avoid double-drop.
                        unsafe { ptr::read(&emplacables_iter) };

                    for _emplacable in emplacables_iter {
                        // drop `emplacable`
                    }
                }
            });

            // avoid double-drop
            mem::forget(emplacables_iter);
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_closure(slice_emplacer_closure) }
    }
}

impl<T, C> Drop for Emplacable<T, C>
where
    T: ?Sized,
    C: EmplacableFn<T>,
{
    /// Runs the contained closure to completion,
    /// instructing it to drop the value of type `T`.
    fn drop(&mut self) {
        #[inline]
        fn dropping_emplacer_closure<T: ?Sized>(
            _: Layout,
            _: <T as Pointee>::Metadata,
            inner_closure: &mut dyn FnMut(*mut PhantomData<T>),
        ) {
            // null ptr signals we wish to drop the value.
            inner_closure(ptr::null_mut());
        }

        let ref_to_fn = &mut dropping_emplacer_closure::<T>;

        // SAFETY: `dropping_emplacer_closure` fulfills the requirements
        let dropping_emplacer = unsafe { Emplacer::new(ref_to_fn) };

        // SAFETY: we are inside `drop`, so no one else will access this
        // `ManuallyDrop` after us
        let emplacable_closure = unsafe { ManuallyDrop::take(&mut self.closure) };

        emplacable_closure(dropping_emplacer);
    }
}

/// Implementation detail for the `From` impls
#[doc(hidden)]
pub trait IntoEmplacable<T: ?Sized> {
    type Closure: EmplacableFn<T>;

    #[must_use]
    fn into_emplacable(self) -> Emplacable<T, Self::Closure>;
}

impl<T> IntoEmplacable<T> for T {
    type Closure = impl EmplacableFn<Self>;

    #[inline]
    fn into_emplacable(mut self) -> Emplacable<T, Self::Closure> {
        let closure = move |emplacer: &mut Emplacer<T>| {
            // Safety: we call the closure right after
            let emplacer_closure = unsafe { emplacer.into_inner() };
            emplacer_closure(Layout::new::<T>(), (), &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: copying value where it belongs.
                    // We `forget` right after to prevent double-free.
                    // `Emplacer` preconditions say this can only be run once.
                    unsafe {
                        ptr::copy_nonoverlapping(addr_of!(self).cast::<T>(), out_ptr.cast(), 1);
                    }
                } else {
                    // SAFETY: we `mem::forget` later to avoid double drop
                    unsafe { ptr::drop_in_place(&mut self) }
                }
            });

            // Don't want to double-drop
            mem::forget(self);
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_closure(closure) }
    }
}

impl<T> From<T> for Emplacable<T, <T as IntoEmplacable<T>>::Closure> {
    #[inline]
    fn from(value: T) -> Self {
        value.into_emplacable()
    }
}

// Implementation detail for the `From<&[T]>` impl of `Emplacable<[T], _>`,
// allows us to specialize on `T: Copy`
trait CopyToBuf: Sized {
    /// # Safety
    ///
    /// `buf` must be valid to write `slice.len() * mem::size_of<T>()`
    /// bytes into, and bust be aligned to `mem::align_of<T>()`.
    unsafe fn copy_to_buf(slice: &[Self], buf: *mut Self);
}

impl<T: Clone> CopyToBuf for T {
    #[inline]
    default unsafe fn copy_to_buf(slice: &[Self], buf: *mut Self) {
        for (index, elem) in slice.iter().enumerate() {
            let owned = elem.clone();
            // SAFETY: copying value where it belongs. safe to write to
            // `buf` by preconditions of function
            unsafe { buf.cast::<T>().add(index).write(owned) };
        }
    }
}

impl<T: Copy> CopyToBuf for T {
    #[inline]
    unsafe fn copy_to_buf(slice: &[Self], buf: *mut Self) {
        // SAFETY: copying value where it belongs. safe to write to
        // `buf` by preconditions of function
        unsafe {
            ptr::copy_nonoverlapping(addr_of!(slice).cast(), buf, slice.len());
        }
    }
}

impl<'s, T: Clone + 's> IntoEmplacable<[T]> for &'s [T] {
    type Closure = impl for<'a> FnOnce(&'a mut Emplacer<[T]>) + 's;

    #[inline]
    fn into_emplacable(self) -> Emplacable<[T], Self::Closure> {
        let metadata = ptr::metadata(self);
        let layout = Layout::for_value(self);

        let closure = move |emplacer: &mut Emplacer<[T]>| {
            // Safety: we call the closure right after
            let emplacer_closure = unsafe { emplacer.into_inner() };
            emplacer_closure(layout, metadata, &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: by precondtion of `Emplacer::new`
                    unsafe { <T as CopyToBuf>::copy_to_buf(self, out_ptr.cast()) }
                }
            });
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_closure(closure) }
    }
}

impl<'s, T: Clone + 's> From<&'s [T]>
    for Emplacable<[T], <&'s [T] as IntoEmplacable<[T]>>::Closure>
{
    #[inline]
    fn from(value: &'s [T]) -> Self {
        <&[T] as IntoEmplacable<[T]>>::into_emplacable(value)
    }
}

impl<'s> IntoEmplacable<str> for &'s str {
    type Closure = impl for<'a> FnOnce(&'a mut Emplacer<str>) + 's;

    #[inline]
    fn into_emplacable(self) -> Emplacable<str, Self::Closure> {
        let metadata = ptr::metadata(self);
        let layout = Layout::for_value(self);

        let closure = move |emplacer: &mut Emplacer<str>| {
            // Safety: we call the closure right after
            let emplacer_closure = unsafe { emplacer.into_inner() };
            emplacer_closure(layout, metadata, &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: copying value where it belongs.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            addr_of!(self).cast::<u8>(),
                            out_ptr.cast(),
                            self.len(),
                        );
                    }
                }
            });
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_closure(closure) }
    }
}

impl<'s> From<&'s str> for Emplacable<str, <&'s str as IntoEmplacable<str>>::Closure> {
    #[inline]
    fn from(value: &'s str) -> Self {
        <&str as IntoEmplacable<str>>::into_emplacable(value)
    }
}

impl<'s> IntoEmplacable<CStr> for &'s CStr {
    type Closure = impl for<'a> FnOnce(&'a mut Emplacer<CStr>) + 's;

    #[inline]
    fn into_emplacable(self) -> Emplacable<CStr, Self::Closure> {
        let metadata = ptr::metadata(self);
        let layout = Layout::for_value(self);

        let closure = move |emplacer: &mut Emplacer<CStr>| {
            // Safety: we call the closure right after
            let emplacer_closure = unsafe { emplacer.into_inner() };
            let size_of_val = layout.size();
            emplacer_closure(layout, metadata, &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: copying value where it belongs.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            addr_of!(self).cast::<u8>(),
                            out_ptr.cast(),
                            size_of_val,
                        );
                    }
                }
            });
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_closure(closure) }
    }
}

impl<'s> From<&'s CStr> for Emplacable<CStr, <&'s CStr as IntoEmplacable<CStr>>::Closure> {
    #[inline]
    fn from(value: &'s CStr) -> Self {
        <&CStr as IntoEmplacable<CStr>>::into_emplacable(value)
    }
}

#[cfg(feature = "std")]
impl<'s> IntoEmplacable<OsStr> for &'s OsStr {
    type Closure = impl for<'a> FnOnce(&'a mut Emplacer<OsStr>) + 's;

    #[must_use]
    #[inline]
    fn into_emplacable(self) -> Emplacable<OsStr, Self::Closure> {
        let metadata = ptr::metadata(self);
        let layout = Layout::for_value(self);

        let closure = move |emplacer: &mut Emplacer<OsStr>| {
            // Safety: we call the closure right after
            let emplacer_closure = unsafe { emplacer.into_inner() };
            let size_of_val = layout.size();
            emplacer_closure(layout, metadata, &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: copying value where it belongs.
                    // We `forget` right after to prevent double-free.
                    // `Emplacer` preconditions say this can only be run once.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            addr_of!(self).cast::<u8>(),
                            out_ptr.cast(),
                            size_of_val,
                        );
                    }
                }
            });
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_closure(closure) }
    }
}

#[cfg(feature = "std")]
impl<'s> From<&'s OsStr> for Emplacable<OsStr, <&'s OsStr as IntoEmplacable<OsStr>>::Closure> {
    #[inline]
    fn from(value: &'s OsStr) -> Self {
        <&OsStr as IntoEmplacable<OsStr>>::into_emplacable(value)
    }
}

#[cfg(feature = "std")]
impl<'s> IntoEmplacable<Path> for &'s Path {
    type Closure = impl for<'a> FnOnce(&'a mut Emplacer<Path>) + 's;

    #[must_use]
    #[inline]
    fn into_emplacable(self) -> Emplacable<Path, Self::Closure> {
        let metadata = ptr::metadata(self);
        let layout = Layout::for_value(self);

        let closure = move |emplacer: &mut Emplacer<Path>| {
            // Safety: we call the closure right after
            let emplacer_closure = unsafe { emplacer.into_inner() };
            let size_of_val = layout.size();
            emplacer_closure(layout, metadata, &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: copying value where it belongs.
                    // We `forget` right after to prevent double-free.
                    // `Emplacer` preconditions say this can only be run once.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            addr_of!(self).cast::<u8>(),
                            out_ptr.cast(),
                            size_of_val,
                        );
                    }
                }
            });
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_closure(closure) }
    }
}

#[cfg(feature = "std")]
impl<'s> From<&'s Path> for Emplacable<Path, <&'s Path as IntoEmplacable<Path>>::Closure> {
    #[inline]
    fn from(value: &'s Path) -> Self {
        <&Path as IntoEmplacable<Path>>::into_emplacable(value)
    }
}

/// Implementation detail of `From<Emplacable<str, _>> for Emplacable<u8, _>`.
#[doc(hidden)]
pub trait FromEmplacable<T: ?Sized> {
    type OutputClosure<C: EmplacableFn<T>>: EmplacableFn<Self>;

    fn from_emplacable<C: EmplacableFn<T>>(
        emplacable: Emplacable<T, C>,
    ) -> Emplacable<Self, Self::OutputClosure<C>>;
}

impl<C: EmplacableFn<str>> IntoEmplacable<[u8]> for Emplacable<str, C> {
    type Closure = impl EmplacableFn<[u8]>;

    #[inline]
    fn into_emplacable(self) -> Emplacable<[u8], Self::Closure> {
        let str_closure = self.into_closure();

        #[allow(clippy::unused_unit)] // https://github.com/rust-lang/rust-clippy/issues/9748
        let u8_emplacer_closure = for<'a, 'b> move |u8_emplacer: &'a mut Emplacer<'b, [u8]>| -> () {
            let u8_emplacer_fn: &mut EmplacerFn<[u8]> =
                // SAFETY: just wrapping this in another emplacer
                unsafe { u8_emplacer.into_inner() };

            let mut str_emplacer_fn =
                |layout: Layout,
                 metadata: usize,
                 str_inner_closure: &mut dyn FnMut(*mut PhantomData<str>)| {
                    let u8_inner_closure: &mut dyn FnMut(*mut PhantomData<[u8]>) =
                        &mut |u8_ptr: *mut PhantomData<[u8]>| str_inner_closure(u8_ptr.cast());

                    u8_emplacer_fn(layout, metadata, u8_inner_closure);
                };

            // SAFETY: Emplacer just calle inner emplacer
            let str_emplacer: &mut Emplacer<str> = unsafe { Emplacer::new(&mut str_emplacer_fn) };

            str_closure(str_emplacer);
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_closure(u8_emplacer_closure) }
    }
}

impl<C> From<Emplacable<str, C>>
    for Emplacable<[u8], <Emplacable<str, C> as IntoEmplacable<[u8]>>::Closure>
where
    C: EmplacableFn<str>,
{
    #[inline]
    fn from(emplacable: Emplacable<str, C>) -> Self {
        <Emplacable<str, C> as IntoEmplacable<[u8]>>::into_emplacable(emplacable)
    }
}

#[cfg(feature = "alloc")]
impl<T: ?Sized> IntoEmplacable<T> for Box<T> {
    type Closure = impl EmplacableFn<T>;

    fn into_emplacable(self) -> Emplacable<T, Self::Closure> {
        let closure = move |emplacer: &mut Emplacer<T>| {
            let layout = Layout::for_value(&*self);
            let ptr = Box::into_raw(self);
            let metadata = ptr::metadata(ptr);

            // SAFETY: we fulfill the preconditions
            let emplacer_closure = unsafe { emplacer.into_inner() };
            emplacer_closure(layout, metadata, &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: checked for null, copying correct number of bytes.
                    unsafe {
                        ptr::copy_nonoverlapping(ptr.cast(), out_ptr.cast::<u8>(), layout.size());
                    }
                } else {
                    // SAFETY: there is no more `Box`, so no double drop
                    unsafe {
                        ptr::drop_in_place(ptr);
                    }
                }
            });

            if layout.size() > 0 {
                // SAFETY: deallocating what the `Box` allocated
                unsafe { alloc::dealloc(ptr.cast(), layout) }
            }
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_closure(closure) }
    }
}

#[cfg(feature = "alloc")]
impl<T> From<Box<T>> for Emplacable<T, <Box<T> as IntoEmplacable<T>>::Closure> {
    #[inline]
    fn from(value: Box<T>) -> Self {
        <Box<T> as IntoEmplacable<T>>::into_emplacable(value)
    }
}

#[cfg(feature = "alloc")]
impl<T> IntoEmplacable<[T]> for alloc_crate::vec::Vec<T> {
    type Closure = impl EmplacableFn<[T]>;

    fn into_emplacable(mut self) -> Emplacable<[T], Self::Closure> {
        let closure = move |emplacer: &mut Emplacer<[T]>| {
            let ptr = self.as_mut_ptr();
            let len = self.len();
            let capacity = self.capacity();
            // SAFETY: values come from the vec
            let layout = unsafe {
                Layout::from_size_align_unchecked(
                    capacity.unchecked_mul(mem::size_of::<T>()),
                    mem::align_of::<T>(),
                )
            };

            // SAFETY: we fulfill the preconditions
            let emplacer_closure = unsafe { emplacer.into_inner() };
            emplacer_closure(layout, len, &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: checked for null, copying correct number of bytes.
                    unsafe {
                        ptr::copy_nonoverlapping(ptr, out_ptr.cast::<T>(), len);
                    }
                } else {
                    for elem in &mut self {
                        // SAFETY: we later forget `Vec`, so no double drop
                        unsafe {
                            ptr::drop_in_place(elem);
                        }
                    }
                }
            });

            if layout.size() > 0 {
                // SAFETY: deallocating what the `Vec` allocated
                unsafe { alloc::dealloc(ptr.cast(), layout) }
            }
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_closure(closure) }
    }
}

#[cfg(feature = "alloc")]
impl<T> From<alloc_crate::vec::Vec<T>>
    for Emplacable<[T], <alloc_crate::vec::Vec<T> as IntoEmplacable<[T]>>::Closure>
{
    #[inline]
    fn from(value: alloc_crate::vec::Vec<T>) -> Self {
        <alloc_crate::vec::Vec<T> as IntoEmplacable<[T]>>::into_emplacable(value)
    }
}

impl<T, const N: usize, C: EmplacableFn<[T; N]>> IntoEmplacable<[T]> for Emplacable<[T; N], C> {
    type Closure = impl EmplacableFn<[T]>;

    #[inline]
    fn into_emplacable(self) -> Emplacable<[T], Self::Closure> {
        self.unsize()
    }
}

impl<T, const N: usize, C> From<Emplacable<[T; N], C>>
    for Emplacable<[T], <Emplacable<[T; N], C> as IntoEmplacable<[T]>>::Closure>
where
    C: EmplacableFn<[T; N]>,
{
    #[inline]
    fn from(value: Emplacable<[T; N], C>) -> Self {
        <Emplacable<[T; N], C> as IntoEmplacable<[T]>>::into_emplacable(value)
    }
}

impl<T, const N: usize, C: EmplacableFn<T>> IntoEmplacable<[T; N]> for [Emplacable<T, C>; N] {
    type Closure = impl EmplacableFn<[T; N]>;

    #[inline]
    fn into_emplacable(self) -> Emplacable<[T; N], Self::Closure> {
        let arr_emplacer_closure = move |arr_emplacer: &mut Emplacer<[T; N]>| {
            let mut elem_emplacables = self;
            // SAFETY: we fulfill the preconditions
            let arr_emplacer_fn = unsafe { arr_emplacer.into_inner() };

            arr_emplacer_fn(
                Layout::new::<[T; N]>(),
                (),
                &mut |arr_out_ptr: *mut PhantomData<[T; N]>| {
                    if !arr_out_ptr.is_null() {
                        let elem_emplacables: [Emplacable<T, C>; N] =
                            // SAFETY: this "inner closure" can only be called once,
                            // per preconditions of `Emplacer::new`.
                            // we `mem::forget` `elem_emplacables` below to avoid double-drop.
                            unsafe { ptr::read(&elem_emplacables) };

                        let n_zeros: [usize; N] = [0; N];
                        let mut i: usize = 0;
                        let indexes: [usize; N] = n_zeros.map(|_| {
                            i = i.wrapping_add(1);
                            i.wrapping_sub(1)
                        });

                        let indexed_elem_emplacables = indexes.zip(elem_emplacables);
                        indexed_elem_emplacables.map(|(index, elem_emplacable)| {
                            let elem_emplacable_closure = elem_emplacable.into_closure();
                            let elem_emplacer_closure = &mut move |
                                _: Layout,
                                (),
                                inner_closure: &mut dyn FnMut(*mut PhantomData<T>),
                            | {
                                // SAFETY: by fn precondition
                                inner_closure(unsafe { arr_out_ptr.cast::<T>().add(index).cast() });
                            };

                            // SAFETY: `elem_emplacer_closure` passes a pointer with the correct offset from the
                            // start of the allocation
                            let elem_emplacer = unsafe { Emplacer::new(elem_emplacer_closure) };
                            elem_emplacable_closure(elem_emplacer);
                        });
                    } else {
                        // Dropping the array of emplacers drops each emplacer,
                        // which drops the `T`s as well
                        // SAFETY: this "inner closure" can only be called once,
                        // per preconditions of `Emplacer::new`.
                        // we `mem::forget` `elem_emplacables` below to avoid double-drop.
                        unsafe {
                            ptr::drop_in_place(addr_of_mut!(elem_emplacables));
                        }
                    }
                },
            );

            // avoid double-drop
            mem::forget(elem_emplacables);
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_closure(arr_emplacer_closure) }
    }
}

impl<T, const N: usize, C> From<[Emplacable<T, C>; N]>
    for Emplacable<[T; N], <[Emplacable<T, C>; N] as IntoEmplacable<[T; N]>>::Closure>
where
    C: EmplacableFn<T>,
{
    #[inline]
    fn from(value: [Emplacable<T, C>; N]) -> Self {
        <[Emplacable<T, C>; N] as IntoEmplacable<[T; N]>>::into_emplacable(value)
    }
}

impl<T, const N: usize, C: EmplacableFn<T>> IntoEmplacable<[T]> for [Emplacable<T, C>; N] {
    type Closure = impl EmplacableFn<[T]>;

    #[inline]
    fn into_emplacable(self) -> Emplacable<[T], Self::Closure> {
        <[Emplacable<T, C>; N] as IntoEmplacable<[T; N]>>::into_emplacable(self).into()
    }
}

impl<T, const N: usize, C> From<[Emplacable<T, C>; N]>
    for Emplacable<[T], <[Emplacable<T, C>; N] as IntoEmplacable<[T]>>::Closure>
where
    C: EmplacableFn<T>,
{
    #[inline]
    fn from(value: [Emplacable<T, C>; N]) -> Self {
        <[Emplacable<T, C>; N] as IntoEmplacable<[T]>>::into_emplacable(value)
    }
}

/// The type of the closure that [`Emplacer<'a, T>`] wraps.
///
/// [`Emplacer<'a, T>`]: Emplacer
pub type EmplacerFn<'a, T> = dyn for<'b> FnMut(Layout, <T as Pointee>::Metadata, &'b mut (dyn FnMut(*mut PhantomData<T>)))
    + 'a;

/// Passed as the last argument to [`Emplacable`] closures.
/// Wraps a closure that tells the function where to write its return value.
///
/// You won't need to interact with this type directly unless you are writing a function
/// that directly produces or consumes [`Emplacable`]s.
///
/// An [`Emplacer`] closure generally does one of three things:
///
/// 1. Allocate memory with the layout of its first argument, run the closure it recieves
///    as its third argument with a pointer to the start of the allocation, then constructs a pointer
///    of type `T` to that allocation with the given metadata.
/// 2. Run the closure it recieves with a null pointer to signify that the value of type `T` should be dropped in place.
/// 3. Do nothing at all, signifying that the value of type `T` should be forgotten and its desctructor should not be run.
///
/// [`Emplacer`]s are allowed to panic, unwind, abort, etc. However, they can't unwind after
/// they have run their inner closure.
#[repr(transparent)]
pub struct Emplacer<'a, T>(EmplacerFn<'a, T>)
where
    T: ?Sized;

impl<'a, T> Emplacer<'a, T>
where
    T: ?Sized,
{
    /// Creates an `Emplacer` from a closure.
    ///
    ///  # Safety
    ///
    /// The closure `e`, if it runs the closure that it recieves as a third argument, must
    /// pass in a pointer that has the alignment of the `Layout` passed to it, and is valid
    /// for writes of the number of bytes corresponding to the `Layout`. (It can also
    /// pass in a null pointer.)
    ///
    /// Also, `e` can run that closure at most once, and must do so before returning,
    /// if it returns. It is allowed to unwind or otherwise diverge, but once it runs
    /// the closure it is no longer allowed to unwind.
    ///
    /// The `Emplacer` can't assume that is has received full ownership of
    /// the value written to the pointer of the inner closure, until the moment it returns.
    /// Specifically, it is not allowed to drop the value if it unwinds.
    #[must_use]
    #[inline]
    pub unsafe fn new<'b>(e: &'b mut EmplacerFn<'a, T>) -> &'b mut Self {
        // SAFETY: `repr(transparent)` guarantees compatible layouts
        unsafe { &mut *((e as *mut EmplacerFn<'a, T>) as *mut Self) }
    }

    /// Obtains the closure inside this `Emplacer`.
    ///
    /// # Safety
    ///
    /// If you unwrap this `Emplacer` and call the resulting closure, you must ensure
    /// that the closure you pas in writes a valid value of type `T` to the passed-in pointer
    /// (or panics, or runs forever, or exits without returning in some other otherwise-sound way).
    ///
    /// The value you write must also correspond to the layout and pointer metadata you pass in.
    ///
    /// If `T`'s size or alignment can be known at compile-time, or can be determined from the
    /// pointer metadata alone, and you pass in an oversized/overaligned `Layout`, then you are not guaranteed
    /// to get an allocation that matches the over-size/over-align guarantees.
    ///
    /// Alternatively, if the closure you pass in receives a null pointer
    /// (which you *must* check for), then you should drop the value of type
    /// `T` in place (though it's not unsound for you to simply forget it).
    ///
    ///
    /// You may not call the closure more than once.
    #[inline]
    pub unsafe fn into_inner<'b>(&'b mut self) -> &'b mut EmplacerFn<'a, T> {
        // SAFETY: `repr(transparent)` guarantees compatible layouts
        &mut self.0
    }
}

/// Like [`Box::new`], but `T` can be `?Sized`.
///
/// You will need `#![feature(unsized_fn_params)]`
/// to call this.
#[cfg(feature = "alloc")]
#[inline]
pub fn box_new<T>(x: T) -> Box<T>
where
    T: ?Sized,
{
    let layout = Layout::for_value(&x);
    let metadata = ptr::metadata(&x);
    let alloc_ptr = if layout.size() > 0 {
        // SAFETY: `layout` has non-zero size, we just checked
        let maybe_ptr = unsafe { alloc::alloc(layout) };

        if maybe_ptr.is_null() {
            handle_alloc_error(layout)
        };

        // SAFETY: copying value into allocation. We make sure to forget it after
        unsafe { ptr::copy_nonoverlapping(addr_of!(x).cast(), maybe_ptr, layout.size()) };

        maybe_ptr
    } else {
        ptr::invalid_mut(layout.align())
    };

    // Avoid double-drop
    mem::forget_unsized(x);

    let wide_ptr = ptr::from_raw_parts_mut(alloc_ptr.cast(), metadata);

    // SAFETY: `Box` allocated in global allocator, except for when it would be a zero-sized allocation
    unsafe { Box::from_raw(wide_ptr) }
}

/// Like [`Box::new`], but takes an [`Emplacer<T, _>`]
/// instead of `T` directly.
///
/// Runs the contained unsized-value-returning closure,
/// and return sits result emplaced into a `Box`.
#[cfg(feature = "alloc")]
#[inline]
pub fn box_new_with<T>(emplacable: Emplacable<T, impl EmplacableFn<T>>) -> Box<T>
where
    T: ?Sized,
{
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
        #[inline]
        fn drop(&mut self) {
            if self.layout.size() != 0 {
                // SAFETY: if layout is non-zero then this pointer
                // should have been allocated with this layout
                unsafe { alloc::dealloc(self.ptr, self.layout) }
            }
        }
    }

    let mut uninit_box = MaybeUninit::uninit();

    let emplacer_closure =
        &mut |layout: Layout, meta, closure: &mut dyn FnMut(*mut PhantomData<T>)| {
            // If `closure` panics and triggers an unwind,
            // this will be dropped, which will call `dealloc` on `ptr`
            // and ensure no memory is leaked.
            let deallocer = PointerDeallocer {
                ptr: if layout.size() == 0 {
                    ptr::invalid_mut(layout.align())
                } else {
                    // SAFETY: just checked that layout is not zero-sized
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

            let wide_ptr = ptr::from_raw_parts_mut(deallocer.ptr.cast::<()>(), meta);
            // SAFETY: Pointer either is to 0-sized allocation or comes from global allocator
            let init_box = unsafe { Box::from_raw(wide_ptr) };

            // Now that we've succesfully initialized the box,
            // we don't want to deallocate its memory.
            mem::forget(deallocer);

            uninit_box.write(init_box);
        };

    // SAFETY: `emplacer_closure` meets the preconditions
    let emplacer = unsafe { Emplacer::new(emplacer_closure) };

    let emplacing_clousre = emplacable.into_closure();
    emplacing_clousre(emplacer);

    // SAFETY: if `unsized_ret` respected `EmplacingClosure`'s safety contract,
    // then `uninit_box` should be initialized.
    unsafe { uninit_box.assume_init() }
}

impl<T, C> Emplacable<[T], C>
where
    C: EmplacableFn<[T]>,
{
    /// Turns this emplacer for a slice of `T`s into an owned [`Vec<T>`].
    ///
    /// [`Vec<T>`]: alloc_crate::vec::Vec
    #[inline]
    pub fn into_vec(self) -> alloc_crate::vec::Vec<T> {
        self.into()
    }
}

#[cfg(feature = "alloc")]
impl<T, C> From<Emplacable<[T], C>> for alloc_crate::vec::Vec<T>
where
    C: EmplacableFn<[T]>,
{
    #[inline]
    fn from(emplacable: Emplacable<[T], C>) -> Self {
        box_new_with(emplacable).into_vec()
    }
}

impl<C> Emplacable<str, C>
where
    C: EmplacableFn<str>,
{
    /// Turns this emplacer for a string slice into an owned, heap-allocated [`String`].
    ///
    /// [`String`]: alloc_crate::string::String
    #[must_use]
    #[inline]
    pub fn into_string(self) -> alloc_crate::string::String {
        self.into()
    }
}

#[cfg(feature = "alloc")]
impl<C> From<Emplacable<str, C>> for alloc_crate::string::String
where
    C: EmplacableFn<str>,
{
    #[inline]
    fn from(emplacable: Emplacable<str, C>) -> Self {
        box_new_with(emplacable).into_string()
    }
}

/// Like [`Box::pin`], but `T` can be `?Sized`.
///
/// You will need `#![feature(unsized_fn_params)]`
/// to call this.
#[cfg(feature = "alloc")]
#[inline]
pub fn box_pin<T>(x: T) -> Pin<Box<T>>
where
    T: ?Sized,
{
    Box::into_pin(box_new(x))
}

/// Like [`Box::pin`], , but takes an [`Emplacer<T, _>`]
/// instead of `T` directly.
///
/// Runs the contained unsized-value-returning closure,
/// and return sits result pinned and emplaced into a `Box`.
#[cfg(feature = "alloc")]
#[inline]
pub fn box_pin_with<T>(emplacable: Emplacable<T, impl EmplacableFn<T>>) -> Pin<Box<T>>
where
    T: ?Sized,
{
    Box::into_pin(box_new_with(emplacable))
}

/// Like [`Rc::new`], but `T` can be `?Sized`.
///
/// You will need `#![feature(unsized_fn_params)]`
/// to call this.
#[cfg(feature = "alloc")]
#[inline]
pub fn rc_new<T>(value: T) -> Rc<T>
where
    T: ?Sized,
{
    box_new(value).into()
}

/// Like [`Rc::new`], but takes an [`Emplacer<T, _>`]
/// instead of `T` directly.
///
/// Runs the contained unsized-value-returning closure,
/// and returns the result emplaced into an `Rc`.
#[cfg(feature = "alloc")]
#[inline]
pub fn rc_new_with<T>(emplacable: Emplacable<T, impl EmplacableFn<T>>) -> Rc<T>
where
    T: ?Sized,
{
    box_new_with(emplacable).into()
}

#[cfg(feature = "alloc")]
impl<T, C> From<Emplacable<T, C>> for Rc<T>
where
    T: ?Sized,
    C: EmplacableFn<T>,
{
    #[inline]
    fn from(emplacable: Emplacable<T, C>) -> Self {
        rc_new_with(emplacable)
    }
}

/// Like [`Rc::pin`], but `T` can be `?Sized`.
///
/// You will need `#![feature(unsized_fn_params)]`
/// to call this.
#[cfg(feature = "alloc")]
#[inline]
pub fn rc_pin<T>(x: T) -> Pin<Rc<T>>
where
    T: ?Sized,
{
    // SAFETY: we own `x`
    unsafe { Pin::new_unchecked(rc_new(x)) }
}

/// Like [`Rc::pin`], but takes an [`Emplacer<T, _>`]
/// instead of `T` directly.
///
/// Runs the contained unsized-value-returning closure,
/// and returns the result pinned and emplaced into an `Rc`.
#[cfg(feature = "alloc")]
#[inline]
pub fn rc_pin_with<T>(emplacable: Emplacable<T, impl EmplacableFn<T>>) -> Pin<Rc<T>>
where
    T: ?Sized,
{
    // SAFETY: we own `x`
    unsafe { Pin::new_unchecked(rc_new_with(emplacable)) }
}

/// Like [`Arc::new`], but `T` can be `?Sized`.
///
/// You will need `#![feature(unsized_fn_params)]`
/// to call this.
#[cfg(feature = "alloc")]
#[inline]
pub fn arc_new<T>(value: T) -> Arc<T>
where
    T: ?Sized,
{
    box_new(value).into()
}

/// Like [`Arc::new`], but takes an [`Emplacer<T, _>`]
/// instead of `T` directly.
///
/// Runs the contained unsized-value-returning closure,
/// and returns the result emplaced into an `Arc`.
#[cfg(feature = "alloc")]
#[inline]
pub fn arc_new_with<T>(emplacable: Emplacable<T, impl EmplacableFn<T>>) -> Arc<T>
where
    T: ?Sized,
{
    box_new_with(emplacable).into()
}

#[cfg(feature = "alloc")]
impl<T, C> From<Emplacable<T, C>> for Arc<T>
where
    T: ?Sized,
    C: EmplacableFn<T>,
{
    #[inline]
    fn from(emplacable: Emplacable<T, C>) -> Self {
        arc_new_with(emplacable)
    }
}

/// Like [`Arc::pin`], but `T` can be `?Sized`.
///
/// You will need `#![feature(unsized_fn_params)]`
/// to call this.
#[cfg(feature = "alloc")]
#[inline]
pub fn arc_pin<T>(x: T) -> Pin<Arc<T>>
where
    T: ?Sized,
{
    // SAFETY: we own `x`
    unsafe { Pin::new_unchecked(arc_new(x)) }
}

/// Like [`Arc::pin`], but takes an [`Emplacer<T, _>`]
/// instead of `T` directly.
///
/// Runs the contained unsized-value-returning closure,
/// and returns the result pinned and emplaced into an `Arc`.
#[cfg(feature = "alloc")]
#[inline]
pub fn arc_pin_with<T>(emplacable: Emplacable<T, impl EmplacableFn<T>>) -> Pin<Arc<T>>
where
    T: ?Sized,
{
    // SAFETY: we own `x`
    unsafe { Pin::new_unchecked(arc_new_with(emplacable)) }
}
