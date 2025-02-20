//! Machinery to support functions that return unsized values.
//!
//! Written to support the [`unsized-vec`] crate, but is independent of it.
//! Requires nightly Rust.
//!
//! Unsized values can take many forms:
//!
//! - On stable Rust, values of unsized types like [`str`],
//!   `[u8]`, and `dyn Any` are generally encountered behind a pointer,
//!   like `&str` or `Box<dyn Any>`.
//!
//! - Nightly Rust provides limited support for passing unsized values
//!   by value as arguments to functions, using the `unsized_fn_params`
//!   feature. There is also `unsized_locals`, for storing these values
//!   on the stack using alloca. (However, that feature is "incomplete" and
//!   this crate doesn't make use of it). But even with thse two feature
//!   gates enabled, functions cannot return unsized values directly.
//!   Also, the only way to produce a by-value unsized value in today's Rust
//!   is by dereferencing a [`Box`]; this crate provides the [`unsize`] macro
//!   to work around this limitation.
//!
//! - For functions that return unsized values, this crate
//!   provides the [`Emplacable`] type. Functions that want
//!   to return a value of type `T`, where `T` is unsized, return an
//!   `Emplacable<T, _>` instead. `Emplacable<T>` wraps a closure;
//!   that closure contains instructions for writing out a `T` to
//!   a caller-provided region of memory. Other functions accept the `Emplacable`
//!   as an argument and call its contained closure to write out the
//!   `T` to some allocation provided by them. For example, this crate
//!   provides the [`box_new_with`] function, which turns an `Emplacable<T>`
//!   into a [`Box<T>`].
//!
//! ## Converting between types
//!
//! | I have                    | I want                     | I can use                    |
//! |---------------------------|----------------------------|------------------------------|
//! | `[i32; 2]`                | `[i32]`                    | [`unsize`]                   |
//! | `[i32; 2]`                | `Emplacable<[i32; 2], _>`  | [`Into::into`]               |
//! | `[i32]`                   |  `Emplacable<[i32], _>`    | [`with_emplacable_for`]      |
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
//! but no fun things that use the tools. If you want more usage examples,
//! check out `unsized-vec`'s documentation and the `examples` folder on GitHub.
//!
//! [`unsized-vec`]: https://docs.rs/unsized-vec/
//! [`Unsize`]: core::marker::Unsize
//! [`CoerceUnsized`]: core::ops::CoerceUnsized

#![forbid(
    clippy::alloc_instead_of_core,
    clippy::std_instead_of_alloc,
    clippy::std_instead_of_core
)]
#![feature(
    allocator_api,
    closure_lifetime_binder,
    forget_unsized,
    impl_trait_in_assoc_type,
    min_specialization,
    ptr_metadata,
    type_alias_impl_trait,
    unsize,
    unsized_fn_params
)]
#![no_std]

#[cfg(feature = "alloc")]
#[doc(hidden)]
pub extern crate alloc as alloc_crate;

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "alloc")]
use alloc_crate::{alloc, boxed::Box, ffi::CString, rc::Rc, string::String, sync::Arc, vec::Vec};

use core::{
    alloc::Layout,
    ffi::CStr,
    marker::{PhantomData, Unsize},
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::FnMut,
    pin::Pin,
    ptr::{self, addr_of, Pointee},
};
#[cfg(feature = "std")]
use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
};

#[doc(hidden)]
pub mod macro_exports {
    #[cfg(feature = "alloc")]
    pub use alloc_crate;
    pub use core;
    pub use u8;

    use alloc_crate::boxed::Box;
    use core::{
        alloc::{AllocError, Allocator, Layout},
        cell::Cell,
        marker::{PhantomData, Unsize},
        mem::{self, MaybeUninit},
        ptr::{self, NonNull},
    };

    // Implementation detail of `unsize` macro.
    #[cfg_attr(not(debug_assertions), repr(transparent))]
    pub struct ImplementationDetailDoNotUse<T> {
        storage: Cell<MaybeUninit<T>>,
        #[cfg(debug_assertions)]
        allocated: bool,
    }

    pub type ImplementationDetailDoNotUseBox<'a, T, S> =
        Box<T, &'a ImplementationDetailDoNotUse<S>>;

    pub fn do_not_use_box_unsize<T, S>(
        val: S,
        a: &ImplementationDetailDoNotUse<S>,
    ) -> ImplementationDetailDoNotUseBox<'_, T, S>
    where
        T: ?Sized,
        S: Unsize<T>,
    {
        let boxed: ImplementationDetailDoNotUseBox<'_, S, S> = Box::new_in(val, a);
        boxed
    }

    impl<T> ImplementationDetailDoNotUse<T> {
        #[allow(clippy::declare_interior_mutable_const)]
        pub const NEW: Self = Self {
            storage: Cell::new(MaybeUninit::uninit()),
            #[cfg(debug_assertions)]
            allocated: false,
        };
    }

    // SAFETY: this is an unsound implementation of the trait,
    // you can't `allocate` more than once without UB. We are careful not
    // to break this invariant inside the macro, but `ImplementationDetailDoNotUse`
    // should not be leaked to arbritrary code!
    unsafe impl<T> Allocator for &ImplementationDetailDoNotUse<T> {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            debug_assert_eq!(layout, Layout::new::<T>());
            #[cfg(debug_assertions)]
            debug_assert!(!self.allocated);
            // SAFETY: Address of `self.0` can't be null
            let thin_ptr = unsafe { NonNull::new_unchecked(self.storage.as_ptr()) };
            Ok(NonNull::from_raw_parts(thin_ptr, mem::size_of::<T>()))
        }

        unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
    }

    // Implementation detail of `by_value_str`.
    pub struct NonAllocator<'a>(PhantomData<&'a mut ()>);

    // SAFETY: `allocate` is a stub that is never run and always panics
    unsafe impl Allocator for NonAllocator<'_> {
        fn allocate(&self, _: Layout) -> Result<NonNull<[u8]>, AllocError> {
            unreachable!()
        }

        #[inline]
        unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {}
    }

    pub type FakeBoxStr<'a> = Box<str, NonAllocator<'a>>;

    pub fn fake_box_str<const LEN: usize>(buf: &mut [MaybeUninit<u8>; LEN]) -> FakeBoxStr<'_> {
        let wide_ptr: *mut str = ptr::from_raw_parts_mut(buf.as_mut_ptr(), LEN);

        // SAFETY: `NonAllocator::deallocate()` is a no-op
        unsafe { Box::from_raw_in(wide_ptr, NonAllocator(PhantomData)) }
    }
}

/// Helper for coercing values to unsized types.
///
/// The `unsized_fn_params` has some rough edges when it comes to coercing
/// sized values to unsized ones by value. This macro works around that.
///
/// If you have a value `val` of type `SizedType`, and you want to coerce it
/// to `UnsizedType`, write `unsize!(val, (SizedType) -> UnsizedType))`.
///
/// Probably useless without the `unsized_fn_params` or `unsized_locals` nightly features.
///
/// Requires the `alloc` crate feature
/// (though doesn't actually allocate on the heap).
///
/// # Example
///
/// ```
/// #![feature(unsized_fn_params)]
///
/// use core::fmt::Debug;
///
/// use emplacable::{box_new, unsize};
///
/// let mut my_box: Box<dyn Debug> = box_new(unsize!("hello world!", (&str) -> dyn Debug));
///
/// dbg!(&my_box);
/// ```
#[cfg(all(feature = "alloc", not(miri)))]
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
            macro_exports::{
                do_not_use_box_unsize,
                ImplementationDetailDoNotUse,
                ImplementationDetailDoNotUseBox,
            },
        };

        let val: $src = $e;

        let new_alloc: ImplementationDetailDoNotUse<$src> = ImplementationDetailDoNotUse::NEW;
        let boxed_unsized: ImplementationDetailDoNotUseBox<$dst, $src> = do_not_use_box_unsize(val, &new_alloc);

        *boxed_unsized
    }};
}

/// Construct a `str` from a string literal,
/// in its dereferenced form.
///
/// Probably useless without the `unsized_fn_params` or `unsized_locals` nightly features.
///
/// Requires the `alloc` crate feature
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
#[cfg(all(feature = "alloc", not(miri)))]
#[macro_export]
macro_rules! by_value_str {
    ($s:literal) => {{
        // This implementation is similar to the one from `unsize`.
        // 1. Declare a constant `&str` for the string
        // 2. Copy the contents of the constant into a buffer
        // 5. Convert a pointer into the buffer into a `Box<str>`
        // 4. Move out of the box
        use $crate::macro_exports::{
            alloc_crate::boxed::Box,
            core::{
                mem::MaybeUninit,
                ptr::{self, addr_of_mut},
            },
            fake_box_str, u8,
        };

        const STRING: &str = $s;
        const LEN: usize = STRING.len();
        let mut buf: [MaybeUninit<u8>; LEN] = [MaybeUninit::uninit(); LEN];

        // SAFETY: `buf` has compatible layout
        unsafe {
            ptr::copy(STRING.as_ptr().cast::<u8>(), addr_of_mut!(buf).cast(), LEN);
        }

        let boxed = fake_box_str(&mut buf);
        *boxed
    }};
}

mod with_emplacable_for {
    use super::*;

    /// `EmplacableFn` used by [`with_emplacable_for`].
    pub type WithEmplacableForFn<'a, T: ?Sized + 'a> = impl EmplacableFn<T> + 'a;

    pub fn with_emplacable_closure<T: ?Sized>(val: &mut T) -> WithEmplacableForFn<'_, T> {
        move |emplacer: &mut Emplacer<'_, T>| {
            let layout = Layout::for_value(val);
            let metadata = ptr::metadata(val);
            // Safety: we call the closure right after
            let emplacer_closure = unsafe { emplacer.into_fn() };
            emplacer_closure(layout, metadata, &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: copying value where it belongs.
                    // We `forget` right after to prevent double-free.
                    // `Emplacer` preconditions say this can only be run once.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            ptr::addr_of_mut!(*val).cast::<u8>(),
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
}

use with_emplacable_for::with_emplacable_closure;
pub use with_emplacable_for::WithEmplacableForFn;

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
/// use emplacable::{box_new_with, unsize, with_emplacable_for};
///
/// let b = with_emplacable_for(unsize!([23_i32, 4, 32], ([i32; 3]) -> [i32]), |e| {
///   box_new_with(e)
/// });
/// assert_eq!(&*b, &[23_i32, 4, 32]);
/// ```
#[cfg(not(all(doctest, any(miri, not(feature = "alloc")))))]
#[inline]
pub fn with_emplacable_for<T, F, R>(mut val: T, mut f: F) -> R
where
    T: ?Sized + 'static,
    F: FnMut(Emplacable<T, WithEmplacableForFn<'_, T>>) -> R,
{
    /// SAFETY: `val` must not be dropped after this function completes.
    #[inline]
    unsafe fn with_emplacable_for_inner<'a, T: ?Sized + 'a, R>(
        val: &'a mut T,
        f: &mut dyn FnMut(Emplacable<T, WithEmplacableForFn<'a, T>>) -> R,
    ) -> R {
        // SAFETY: closure fulfills safety preconditions
        let emplacable = unsafe { Emplacable::from_fn(with_emplacable_closure(val)) };

        f(emplacable)
    }

    // SAFETY: we `forget_unsized` val immediately after this call
    let ret = unsafe { with_emplacable_for_inner(&mut val, &mut f) };
    mem::forget_unsized(val);
    ret
}

/// Alias of [`for<'a> FnOnce(&'a mut Emplacer<T>)`](Emplacer<T>).
pub trait EmplacableFn<T>: for<'a> FnOnce(&'a mut Emplacer<'_, T>)
where
    T: ?Sized,
{
}

impl<T, F> EmplacableFn<T> for F
where
    T: ?Sized,
    F: for<'a> FnOnce(&'a mut Emplacer<'_, T>),
{
}

/// A wrapped closure that you can pass to functions like `box_new_with`,
/// that describes how to write a value of type `T` to a caller-provided
/// allocation. You can get a `T` out of an `Emplacable` through functions like
/// [`box_new_with`]. Alternately, you can drop the value of type `T` by dropping
/// the `Emplacable`. Or you can forget the value of type `T` with [`Emplacable::forget`].
///
/// ## How it works
///
/// To make an [`Emplacable<T, _>`], you must first produce an [`EmplacableFn<T>`],
/// which is an [`FnOnce`] that accepts an [`Emplacer<T>`]. Your [`EmplacableFn<T>`] perform the follwoing steps:
///
/// 1. Call [`into_fn`][`Emplacer::into_fn`] on the [`Emplacer<T>`] to obtain a [`EmplacerFn<T>`], which is an alias for
///    `dyn FnMut(Layout, <T as Pointee>::Metadata, &mut (dyn FnMut(*mut PhantomData<T>)))`.
/// 2. Call the [`EmplacerFn<T>`] with the following arguments:
///
///     1. `Layout`: The layout of the value of type `T` you want to emplace
///     2. `<T as Pointee>::Metadata`: The pointer metadata of the value of type `T` you want to emplace
///     3. `&mut (dyn FnMut(*mut PhantomData<T>)))`: The closure you must pass for this thrid argument
///        must do one of two things, depending on the `*mut PhantomData<T>` pointer it recieves.
///         - if the pointer is null, it should drop the value of type `T`.
///         - otherwise, it should write the value of type `T` to the pointer,
///           which it can assume points to the start of an allocation with the size and alignment of
///           the `Layout` from above.
///
/// Once you have an [`EmplacableFn<T>`], use [`Emplacable::from_fn`] to turn it into an [`Emplacable<T, _>`].
///
/// There are **safety preconditions** at every step of this process that **must be respected to avoid UB.**
/// Read the documentation of all the methods involved to learn about them.
#[repr(transparent)]
pub struct Emplacable<T, F>
where
    T: ?Sized,
    F: EmplacableFn<T>,
{
    closure: ManuallyDrop<F>,
    phantom: PhantomData<fn(&mut Emplacer<'_, T>)>,
}

impl<T, F> Emplacable<T, F>
where
    T: ?Sized,
    F: EmplacableFn<T>,
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
    pub unsafe fn from_fn(closure: F) -> Self {
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
    pub fn into_fn(self) -> F {
        let mut manually_drop_self = ManuallyDrop::new(self);

        // SAFETY: `self` is in a `ManuallyDrop`, so no double drop
        unsafe { ManuallyDrop::take(&mut manually_drop_self.closure) }
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
        let emplacer = unsafe { Emplacer::from_fn(emplacer_closure) };

        let closure = self.into_fn();
        closure(emplacer);

        // SAFETY: `buf` was initialized by the emplacer
        unsafe { buf.assume_init() }
    }

    /// Runs the `Emplacable` closure,
    /// but doesn't run the "inner closure",
    /// so the value of type `T` is forgotten,
    /// and its destructor is not run.
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

        let emplacable_closure = self.into_fn();

        let ref_to_fn = &mut forgetting_emplacer_closure::<T>;

        // SAFETY: `forgetting_emplacer` fulfills the requirements
        let forgetting_emplacer = unsafe { Emplacer::from_fn(ref_to_fn) };

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

        let sized_emplacable_closure = self.into_fn();

        let unsized_emplacable_closure = move |unsized_emplacer: &mut Emplacer<'_, U>| {
            // SAFETY: We are just wrapping this emplacer
            let unsized_emplacer_closure = unsafe { unsized_emplacer.into_fn() };

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
            let sized_emplacer = unsafe { Emplacer::from_fn(&mut sized_emplacer_closure) };

            sized_emplacable_closure(sized_emplacer);
        };

        // SAFETY: Again, just wrapping our input
        unsafe { Emplacable::from_fn(unsized_emplacable_closure) }
    }

    /// Creates an `Emplacable` for a slice of values of type `T` out of an iterator
    /// of values of type `T`.
    ///
    /// This function differs from [`FromIterator::from_iter`] in that the iterator is required to
    /// be an [`ExactSizeIterator`]. If `ExactSizeIterator` is incorrectly implemented,
    /// this function may panic or otherwise misbehave (but will not trigger UB).
    #[allow(clippy::should_implement_trait)] // We only take `ExactSizeIterator`s
    #[inline]
    pub fn from_iter<I>(iter: I) -> Emplacable<[T], impl EmplacableFn<[T]>>
    where
        T: Sized,
        I: IntoIterator<Item = Self>,
        I::IntoIter: ExactSizeIterator,
    {
        fn from_iter_inner<
            T,
            F: EmplacableFn<T>,
            I: Iterator<Item = Emplacable<T, F>> + ExactSizeIterator,
        >(
            iter: I,
        ) -> Emplacable<[T], impl EmplacableFn<[T]>> {
            let len = iter.len();

            // Panics if size overflows `isize::MAX`.
            let layout = Layout::from_size_align(
                mem::size_of::<T>().checked_mul(len).unwrap(),
                mem::align_of::<T>(),
            )
            .unwrap();

            let slice_emplacer_closure = move |slice_emplacer: &mut Emplacer<'_, [T]>| {
                // Move ite into closure
                let emplacables_iter = ManuallyDrop::new(iter);

                // SAFETY: we fulfill the preconditions
                let slice_emplacer_fn = unsafe { slice_emplacer.into_fn() };

                slice_emplacer_fn(layout, len, &mut |arr_out_ptr: *mut PhantomData<[T]>| {
                    if !arr_out_ptr.is_null() {
                        let elem_emplacables: I =
                            // SAFETY: this "inner closure" can only be called once,
                            // per preconditions of `Emplacer::new`.
                            // `elem_emplacables` is inside a `ManuallyDrop`, so avoid double-drop.
                            unsafe { ptr::read(&*emplacables_iter) };

                        // We can't trust `ExactSizeIterator`'s `len()`,
                        // so we keep track of how many
                        // elements were actually returned.
                        let mut num_elems_copied: usize = 0;

                        let indexed_elem_emplacables = (0..len).zip(elem_emplacables);
                        indexed_elem_emplacables.for_each(|(index, elem_emplacable)| {
                            let elem_emplacable_closure = elem_emplacable.into_fn();
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
                            let elem_emplacer = unsafe { Emplacer::from_fn(elem_emplacer_closure) };
                            elem_emplacable_closure(elem_emplacer);

                            num_elems_copied += 1;
                        });

                        assert_eq!(num_elems_copied, len);
                    } else {
                        let emplacables_iter: I =
                            // SAFETY: this "inner closure" can only be called once,
                            // per preconditions of `Emplacer::new`.
                            // `elem_emplacables` is inside a `ManuallyDrop`, so avoid double-drop.
                            unsafe { ptr::read(&*emplacables_iter) };

                        for _emplacable in emplacables_iter {
                            // drop `emplacable`
                        }
                    }
                });
            };

            // SAFETY: `closure` properly emplaces `val`
            unsafe { Emplacable::from_fn(slice_emplacer_closure) }
        }

        let emplacables_iter = iter.into_iter();

        from_iter_inner(emplacables_iter)
    }
}

impl<T, F> Drop for Emplacable<T, F>
where
    T: ?Sized,
    F: EmplacableFn<T>,
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
        let dropping_emplacer = unsafe { Emplacer::from_fn(ref_to_fn) };

        // SAFETY: we are inside `drop`, so no one else will access this
        // `ManuallyDrop` after us
        let emplacable_closure = unsafe { ManuallyDrop::take(&mut self.closure) };

        emplacable_closure(dropping_emplacer);
    }
}

/// Implementation detail for the `From` impls.
#[doc(hidden)]
pub trait IntoEmplacable<T: ?Sized> {
    type Closure: EmplacableFn<T>;

    #[must_use]
    fn into_emplacable(self) -> Emplacable<T, Self::Closure>;
}

impl<T> IntoEmplacable<T> for T {
    type Closure = impl EmplacableFn<Self>;

    #[inline]
    fn into_emplacable(self) -> Emplacable<T, Self::Closure> {
        let closure = move |emplacer: &mut Emplacer<'_, T>| {
            let mut manually_drop_self = ManuallyDrop::new(self);
            // Safety: we call the closure right after
            let emplacer_closure = unsafe { emplacer.into_fn() };
            emplacer_closure(Layout::new::<T>(), (), &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: copying value where it belongs.
                    // We use `ManuallyDrop` prevent double-free.
                    // `Emplacer` preconditions say this can only be run once.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            addr_of!(*manually_drop_self).cast::<T>(),
                            out_ptr.cast(),
                            1,
                        );
                    }
                } else {
                    // SAFETY: we use `ManuallyDrop` to avoid double drop
                    unsafe { ManuallyDrop::drop(&mut manually_drop_self) }
                }
            });
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_fn(closure) }
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
    type Closure = impl for<'a> FnOnce(&'a mut Emplacer<'_, [T]>) + 's;

    #[inline]
    fn into_emplacable(self) -> Emplacable<[T], Self::Closure> {
        let metadata = ptr::metadata(self);
        let layout = Layout::for_value(self);

        let closure = move |emplacer: &mut Emplacer<'_, [T]>| {
            // Safety: we call the closure right after
            let emplacer_closure = unsafe { emplacer.into_fn() };
            emplacer_closure(layout, metadata, &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: by precondtion of `Emplacer::new`
                    unsafe { <T as CopyToBuf>::copy_to_buf(self, out_ptr.cast()) }
                }
            });
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_fn(closure) }
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
    type Closure = impl for<'a> FnOnce(&'a mut Emplacer<'_, str>) + 's;

    #[inline]
    fn into_emplacable(self) -> Emplacable<str, Self::Closure> {
        let metadata = ptr::metadata(self);
        let layout = Layout::for_value(self);

        let closure = move |emplacer: &mut Emplacer<'_, str>| {
            // Safety: we call the closure right after
            let emplacer_closure = unsafe { emplacer.into_fn() };
            emplacer_closure(layout, metadata, &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: copying value where it belongs.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            addr_of!(*self).cast::<u8>(),
                            out_ptr.cast(),
                            layout.size(),
                        );
                    }
                }
            });
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_fn(closure) }
    }
}

impl<'s> From<&'s str> for Emplacable<str, <&'s str as IntoEmplacable<str>>::Closure> {
    #[inline]
    fn from(value: &'s str) -> Self {
        <&str as IntoEmplacable<str>>::into_emplacable(value)
    }
}

impl<'s> IntoEmplacable<CStr> for &'s CStr {
    type Closure = impl for<'a> FnOnce(&'a mut Emplacer<'_, CStr>) + 's;

    #[inline]
    fn into_emplacable(self) -> Emplacable<CStr, Self::Closure> {
        let metadata = ptr::metadata(self);
        let layout = Layout::for_value(self);

        let closure = move |emplacer: &mut Emplacer<'_, CStr>| {
            // Safety: we call the closure right after
            let emplacer_closure = unsafe { emplacer.into_fn() };
            emplacer_closure(layout, metadata, &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: copying value where it belongs.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            addr_of!(*self).cast::<u8>(),
                            out_ptr.cast(),
                            layout.size(),
                        );
                    }
                }
            });
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_fn(closure) }
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
    type Closure = impl for<'a> FnOnce(&'a mut Emplacer<'_, OsStr>) + 's;

    #[must_use]
    #[inline]
    fn into_emplacable(self) -> Emplacable<OsStr, Self::Closure> {
        let metadata = ptr::metadata(self);
        let layout = Layout::for_value(self);

        let closure = move |emplacer: &mut Emplacer<'_, OsStr>| {
            // Safety: we call the closure right after
            let emplacer_closure = unsafe { emplacer.into_fn() };
            emplacer_closure(layout, metadata, &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: copying value where it belongs.
                    // We `forget` right after to prevent double-free.
                    // `Emplacer` preconditions say this can only be run once.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            addr_of!(*self).cast::<u8>(),
                            out_ptr.cast(),
                            layout.size(),
                        );
                    }
                }
            });
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_fn(closure) }
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
    type Closure = impl for<'a> FnOnce(&'a mut Emplacer<'_, Path>) + 's;

    #[must_use]
    #[inline]
    fn into_emplacable(self) -> Emplacable<Path, Self::Closure> {
        let metadata = ptr::metadata(self);
        let layout = Layout::for_value(self);

        let closure = move |emplacer: &mut Emplacer<'_, Path>| {
            // Safety: we call the closure right after
            let emplacer_closure = unsafe { emplacer.into_fn() };
            emplacer_closure(layout, metadata, &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: copying value where it belongs.
                    // We `forget` right after to prevent double-free.
                    // `Emplacer` preconditions say this can only be run once.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            addr_of!(*self).cast::<u8>(),
                            out_ptr.cast(),
                            layout.size(),
                        );
                    }
                }
            });
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_fn(closure) }
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
    type OutputClosure<F: EmplacableFn<T>>: EmplacableFn<Self>;

    fn from_emplacable<F: EmplacableFn<T>>(
        emplacable: Emplacable<T, F>,
    ) -> Emplacable<Self, Self::OutputClosure<F>>;
}

impl<F: EmplacableFn<str>> IntoEmplacable<[u8]> for Emplacable<str, F> {
    type Closure = impl EmplacableFn<[u8]>;

    #[inline]
    fn into_emplacable(self) -> Emplacable<[u8], Self::Closure> {
        let str_closure = self.into_fn();

        #[allow(clippy::unused_unit)] // https://github.com/rust-lang/rust-clippy/issues/9748
        let u8_emplacer_closure = for<'a, 'b> move |u8_emplacer: &'a mut Emplacer<'b, [u8]>| -> () {
            let u8_emplacer_fn: &mut EmplacerFn<'_, [u8]> =
                // SAFETY: just wrapping this in another emplacer
                unsafe { u8_emplacer.into_fn() };

            let mut str_emplacer_fn =
                |layout: Layout,
                 metadata: usize,
                 str_inner_closure: &mut dyn FnMut(*mut PhantomData<str>)| {
                    let u8_inner_closure: &mut dyn FnMut(*mut PhantomData<[u8]>) =
                        &mut |u8_ptr: *mut PhantomData<[u8]>| str_inner_closure(u8_ptr.cast());

                    u8_emplacer_fn(layout, metadata, u8_inner_closure);
                };

            let str_emplacer: &mut Emplacer<'_, str> =
                // SAFETY: Emplacer just calle inner emplacer
                unsafe { Emplacer::from_fn(&mut str_emplacer_fn) };

            str_closure(str_emplacer);
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_fn(u8_emplacer_closure) }
    }
}

impl<F> From<Emplacable<str, F>>
    for Emplacable<[u8], <Emplacable<str, F> as IntoEmplacable<[u8]>>::Closure>
where
    F: EmplacableFn<str>,
{
    #[inline]
    fn from(emplacable: Emplacable<str, F>) -> Self {
        <Emplacable<str, F> as IntoEmplacable<[u8]>>::into_emplacable(emplacable)
    }
}

#[cfg(feature = "alloc")]
impl<T: ?Sized> IntoEmplacable<T> for Box<T> {
    type Closure = impl EmplacableFn<T>;

    fn into_emplacable(self) -> Emplacable<T, Self::Closure> {
        let closure = move |emplacer: &mut Emplacer<'_, T>| {
            let layout = Layout::for_value(&*self);
            let ptr = Box::into_raw(self);
            let metadata = ptr::metadata(ptr);

            // SAFETY: we fulfill the preconditions
            let emplacer_closure = unsafe { emplacer.into_fn() };
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
        unsafe { Emplacable::from_fn(closure) }
    }
}

#[cfg(feature = "alloc")]
impl<T: ?Sized> From<Box<T>> for Emplacable<T, <Box<T> as IntoEmplacable<T>>::Closure> {
    #[inline]
    fn from(value: Box<T>) -> Self {
        <Box<T> as IntoEmplacable<T>>::into_emplacable(value)
    }
}

#[cfg(feature = "alloc")]
impl<T> IntoEmplacable<[T]> for Vec<T> {
    type Closure = impl EmplacableFn<[T]>;

    fn into_emplacable(self) -> Emplacable<[T], Self::Closure> {
        let closure = move |emplacer: &mut Emplacer<'_, [T]>| {
            let mut vec = ManuallyDrop::new(self);

            let ptr = vec.as_mut_ptr();
            let len = vec.len();
            let capacity = vec.capacity();
            // SAFETY: values come from the vec
            let layout = unsafe {
                Layout::from_size_align_unchecked(
                    capacity.unchecked_mul(mem::size_of::<T>()),
                    mem::align_of::<T>(),
                )
            };

            // SAFETY: we fulfill the preconditions
            let emplacer_closure = unsafe { emplacer.into_fn() };
            emplacer_closure(layout, len, &mut |out_ptr| {
                if !out_ptr.is_null() {
                    // SAFETY: checked for null, copying correct number of bytes.
                    // `Vec` is in a `ManuallyDrop`, so no double drop
                    unsafe {
                        ptr::copy_nonoverlapping(ptr, out_ptr.cast::<T>(), len);
                    }
                } else {
                    for elem in &mut *vec {
                        // SAFETY: `Vec` is in a `ManuallyDrop`, so no double drop
                        unsafe {
                            ptr::drop_in_place(elem);
                        }
                    }
                }
            });

            if layout.size() > 0 {
                // SAFETY: deallocating what the `Vec` allocated. The `Vec` is in a `ManuallyDrop`,
                // so no double-drop
                unsafe { alloc::dealloc(ptr.cast(), layout) }
            }
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_fn(closure) }
    }
}

#[cfg(feature = "alloc")]
impl<T> From<Vec<T>> for Emplacable<[T], <Vec<T> as IntoEmplacable<[T]>>::Closure> {
    #[inline]
    fn from(value: Vec<T>) -> Self {
        <Vec<T> as IntoEmplacable<[T]>>::into_emplacable(value)
    }
}

impl<T, const N: usize, F: EmplacableFn<[T; N]>> IntoEmplacable<[T]> for Emplacable<[T; N], F> {
    type Closure = impl EmplacableFn<[T]>;

    #[inline]
    fn into_emplacable(self) -> Emplacable<[T], Self::Closure> {
        self.unsize()
    }
}

impl<T, const N: usize, F> From<Emplacable<[T; N], F>>
    for Emplacable<[T], <Emplacable<[T; N], F> as IntoEmplacable<[T]>>::Closure>
where
    F: EmplacableFn<[T; N]>,
{
    #[inline]
    fn from(value: Emplacable<[T; N], F>) -> Self {
        <Emplacable<[T; N], F> as IntoEmplacable<[T]>>::into_emplacable(value)
    }
}

impl<T, const N: usize, F: EmplacableFn<T>> IntoEmplacable<[T; N]> for [Emplacable<T, F>; N] {
    type Closure = impl EmplacableFn<[T; N]>;

    #[inline]
    fn into_emplacable(self) -> Emplacable<[T; N], Self::Closure> {
        let arr_emplacer_closure = move |arr_emplacer: &mut Emplacer<'_, [T; N]>| {
            let mut elem_emplacables = ManuallyDrop::new(self);
            // SAFETY: we fulfill the preconditions
            let arr_emplacer_fn = unsafe { arr_emplacer.into_fn() };

            arr_emplacer_fn(
                Layout::new::<[T; N]>(),
                (),
                &mut |arr_out_ptr: *mut PhantomData<[T; N]>| {
                    if !arr_out_ptr.is_null() {
                        let elem_emplacables: [Emplacable<T, F>; N] =
                            // SAFETY: this "inner closure" can only be called once,
                            // per preconditions of `Emplacer::new`.
                            // we `mem::forget` `elem_emplacables` below to avoid double-drop.
                            unsafe { ptr::read(&*elem_emplacables) };

                        let n_zeros: [usize; N] = [0; N];
                        let mut i: usize = 0;
                        let indexes: [usize; N] = n_zeros.map(|_| {
                            i = i.wrapping_add(1);
                            i.wrapping_sub(1)
                        });

                        indexes.into_iter().zip(elem_emplacables).for_each(
                            |(index, elem_emplacable)| {
                                let elem_emplacable_closure = elem_emplacable.into_fn();
                                let elem_emplacer_closure =
                                    &mut move |_: Layout,
                                               (),
                                               inner_closure: &mut dyn FnMut(
                                        *mut PhantomData<T>,
                                    )| {
                                        // SAFETY: by fn precondition
                                        inner_closure(unsafe {
                                            arr_out_ptr.cast::<T>().add(index).cast()
                                        });
                                    };

                                let elem_emplacer =
                                    // SAFETY: `elem_emplacer_closure` passes a pointer with the correct offset from the
                                    // start of the allocation
                                    unsafe { Emplacer::from_fn(elem_emplacer_closure) };
                                elem_emplacable_closure(elem_emplacer);
                            },
                        );
                    } else {
                        // Dropping the array of emplacers drops each emplacer,
                        // which drops the `T`s as well
                        // SAFETY: this "inner closure" can only be called once,
                        // per preconditions of `Emplacer::new`.
                        // we never access `elem_emplacables` after this.
                        unsafe {
                            ManuallyDrop::drop(&mut elem_emplacables);
                        }
                    }
                },
            );
        };

        // SAFETY: `closure` properly emplaces `val`
        unsafe { Emplacable::from_fn(arr_emplacer_closure) }
    }
}

impl<T, const N: usize, F> From<[Emplacable<T, F>; N]>
    for Emplacable<[T; N], <[Emplacable<T, F>; N] as IntoEmplacable<[T; N]>>::Closure>
where
    F: EmplacableFn<T>,
{
    #[inline]
    fn from(value: [Emplacable<T, F>; N]) -> Self {
        <[Emplacable<T, F>; N] as IntoEmplacable<[T; N]>>::into_emplacable(value)
    }
}

impl<T, const N: usize, F: EmplacableFn<T>> IntoEmplacable<[T]> for [Emplacable<T, F>; N] {
    type Closure = impl EmplacableFn<[T]>;

    #[inline]
    fn into_emplacable(self) -> Emplacable<[T], Self::Closure> {
        <[Emplacable<T, F>; N] as IntoEmplacable<[T; N]>>::into_emplacable(self).into()
    }
}

impl<T, const N: usize, F> From<[Emplacable<T, F>; N]>
    for Emplacable<[T], <[Emplacable<T, F>; N] as IntoEmplacable<[T]>>::Closure>
where
    F: EmplacableFn<T>,
{
    #[inline]
    fn from(value: [Emplacable<T, F>; N]) -> Self {
        <[Emplacable<T, F>; N] as IntoEmplacable<[T]>>::into_emplacable(value)
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
/// that directly produces or consumes `Emplacable`s.
///
/// An `Emplacer` closure can do one of three things:
///
/// 1. Allocate memory with the layout of its first argument, run the closure it receives
///    as its third argument with a pointer to the start of the allocation, then construct a pointer
///    of type `T` to that allocation with the given metadata.
/// 2. Run the closure it recieves with a null pointer to signal that the value of type `T` should be dropped in place.
/// 3. Do nothing at all, signifying that the value of type `T` should be forgotten and its destructor should not be run.
///
/// `Emplacer`s are allowed to panic, unwind, abort, etc. However, they can't unwind after
/// they have run their inner closure.
#[repr(transparent)]
pub struct Emplacer<'a, T>(EmplacerFn<'a, T>)
where
    T: ?Sized;

impl<'a, T> Emplacer<'a, T>
where
    T: ?Sized,
{
    /// Creates an `Emplacer` from an [`EmplacerFn<T>`] closure
    /// (a `dyn FnMut(Layout, <T as Pointee>::Metadata, &mut dyn FnMut(*mut PhantomData<T>))`).
    ///
    /// `emplacer_fn` should do one the following things:
    ///
    /// - Allocate a chunk of memory that satisfies the requirements of the [`Layout`]
    ///   it receives as its fisrst argument, and pass a pointer to the start of that allocation
    ///   to the closure it recieves as its third argument.
    /// - Ignore its first and second arguments, and pass a null pointer to the closure
    ///   as its first argument.
    /// - Do nothing at all.
    ///
    /// # Safety
    ///
    /// The `emplacer_fn`, if it runs the closure that it recieves as a third argument, must
    /// pass in a pointer that is ether null, or has the alignment of the `Layout` passed to it,
    /// and is valid for writes of the number of bytes corresponding to the `Layout`.
    ///
    /// `emplacer_fn` is not permitted to run the closure it receives as its third argument more than once.
    ///
    /// If `emplacer_fn` runs the closure it receives as its thrid argument, it must do so before returning.
    ///
    /// `emplacer_fn` is allowed to unwind or otherwise diverge. But if it runs the closure it receives as its third argument,
    /// then once that inner closure returns, `emplacer_fn` is no longer allowed to unwind.
    ///
    /// The `emplacer_fn` can't assume that is has received full ownership of
    /// the value written to the pointer of the inner closure, until the moment `emplacer_fn`
    /// (and the [`EmplacableFn<T>`] that calls it) returns. Specifically, `emplacer_fn` is not
    /// allowed to drop the value.
    #[must_use]
    #[inline]
    pub unsafe fn from_fn<'b>(emplacer_fn: &'b mut EmplacerFn<'a, T>) -> &'b mut Self {
        // SAFETY: `repr(transparent)` guarantees compatible layouts
        unsafe { &mut *((emplacer_fn as *mut EmplacerFn<'a, T>) as *mut Self) }
    }

    /// Obtains the closure inside this `Emplacer<T>`.
    /// This should generally be called only inside [`EmplacableFn<T>`]s.
    ///
    /// # Safety
    ///
    /// If you call the resulting [`EmplacerFn`] closure, you must ensure
    /// that the closure you pass in:
    ///
    /// - If it receives a non-null pointer as an argument, it *must* write a valid value of type `T`
    ///   to the passed-in pointer. This value must correspond to the `Layout` and pointer metadata you
    ///   passed to the [`EmplacerFn<T>`].
    /// - If it recieves a null pointer as an argument, then it is recommmended that you drop the
    ///   value of type `T` that you would have written out had the pointer been non-null.
    ///   (you aren't *required* to do this.)
    /// - In either case, the closure is alterately allowed to panic, unwind, abort, or diverge
    ///   in some other way. If it does so, it is not obligated to perform the tasks listed above.
    ///
    /// If `T`'s size or alignment can be known at compile-time, or can be determined from the
    /// pointer metadata alone, and you pass in an oversized/overaligned `Layout`, then you are not guaranteed
    /// to get an allocation that matches the `Layout`'s stronger guarantees.
    ///
    /// You may not call the [`EmplacerFn`] closure more than once.
    #[inline]
    pub unsafe fn into_fn<'b>(&'b mut self) -> &'b mut EmplacerFn<'a, T> {
        // SAFETY: `repr(transparent)` guarantees compatible layouts
        &mut self.0
    }
}

/// Like [`Box::new`], but `T` can be `?Sized`.
///
/// You will need `#![feature(unsized_fn_params)]`
/// to call this.
#[cfg(all(feature = "alloc", not(miri)))]
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
            alloc::handle_alloc_error(layout)
        };

        // SAFETY: copying value into allocation. We make sure to forget it after
        unsafe { ptr::copy_nonoverlapping(addr_of!(x).cast(), maybe_ptr, layout.size()) };

        maybe_ptr
    } else {
        ptr::without_provenance_mut(layout.align())
    };

    // Avoid double-drop
    mem::forget_unsized(x);

    let wide_ptr = ptr::from_raw_parts_mut(alloc_ptr, metadata);

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
                    ptr::without_provenance_mut(layout.align())
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
    let emplacer = unsafe { Emplacer::from_fn(emplacer_closure) };

    let emplacing_clousre = emplacable.into_fn();
    emplacing_clousre(emplacer);

    // SAFETY: if `unsized_ret` respected `EmplacingClosure`'s safety contract,
    // then `uninit_box` should be initialized.
    unsafe { uninit_box.assume_init() }
}

impl<T, F> Emplacable<[T], F>
where
    F: EmplacableFn<[T]>,
{
    /// Turns this emplacer for a slice of `T`s into an owned [`Vec<T>`].
    #[cfg(feature = "alloc")]
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.into()
    }
}

#[cfg(feature = "alloc")]
impl<T, F> From<Emplacable<[T], F>> for Vec<T>
where
    F: EmplacableFn<[T]>,
{
    #[inline]
    fn from(emplacable: Emplacable<[T], F>) -> Self {
        box_new_with(emplacable).into_vec()
    }
}

#[cfg(feature = "alloc")]
impl<T, F> FromIterator<Emplacable<T, F>> for Vec<T>
where
    F: EmplacableFn<T>,
{
    #[inline]
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Emplacable<T, F>>,
    {
        let mut vec = Vec::new();
        vec.extend(iter);
        vec
    }
}

#[cfg(feature = "alloc")]
impl<T, F> Extend<Emplacable<T, F>> for Vec<T>
where
    F: EmplacableFn<T>,
{
    #[inline]
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Emplacable<T, F>>,
    {
        fn extend_inner<T, F: EmplacableFn<T>, I: Iterator<Item = Emplacable<T, F>>>(
            vec: &mut Vec<T>,
            iter: I,
        ) {
            vec.reserve_exact(iter.size_hint().0);

            for emplacable in iter {
                vec.push(emplacable.get());
            }
        }

        extend_inner(self, iter.into_iter());
    }
}

impl<F> Emplacable<str, F>
where
    F: EmplacableFn<str>,
{
    /// Turns this emplacer for a string slice into an owned, heap-allocated [`String`].
    ///
    /// [`String`]: alloc_crate::string::String
    #[cfg(feature = "alloc")]
    #[must_use]
    #[inline]
    pub fn into_string(self) -> String {
        self.into()
    }
}

#[cfg(feature = "alloc")]
impl<F> From<Emplacable<str, F>> for String
where
    F: EmplacableFn<str>,
{
    #[inline]
    fn from(emplacable: Emplacable<str, F>) -> Self {
        box_new_with(emplacable).into_string()
    }
}

#[cfg(feature = "alloc")]
impl<F> From<Emplacable<CStr, F>> for CString
where
    F: EmplacableFn<CStr>,
{
    #[inline]
    fn from(emplacable: Emplacable<CStr, F>) -> Self {
        box_new_with(emplacable).into_c_string()
    }
}

#[cfg(feature = "std")]
impl<F> From<Emplacable<OsStr, F>> for OsString
where
    F: EmplacableFn<OsStr>,
{
    #[inline]
    fn from(emplacable: Emplacable<OsStr, F>) -> Self {
        box_new_with(emplacable).into_os_string()
    }
}

#[cfg(feature = "std")]
impl<F> From<Emplacable<Path, F>> for PathBuf
where
    F: EmplacableFn<Path>,
{
    #[inline]
    fn from(emplacable: Emplacable<Path, F>) -> Self {
        box_new_with(emplacable).into_path_buf()
    }
}

/// Like [`Box::pin`], but `T` can be `?Sized`.
///
/// You will need `#![feature(unsized_fn_params)]`
/// to call this.
#[cfg(all(feature = "alloc", not(miri)))]
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
#[cfg(all(feature = "alloc", not(miri)))]
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
impl<T, F> From<Emplacable<T, F>> for Rc<T>
where
    T: ?Sized,
    F: EmplacableFn<T>,
{
    #[inline]
    fn from(emplacable: Emplacable<T, F>) -> Self {
        rc_new_with(emplacable)
    }
}

/// Like [`Rc::pin`], but `T` can be `?Sized`.
///
/// You will need `#![feature(unsized_fn_params)]`
/// to call this.
#[cfg(all(feature = "alloc", not(miri)))]
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
#[cfg(all(feature = "alloc", not(miri)))]
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
impl<T, F> From<Emplacable<T, F>> for Arc<T>
where
    T: ?Sized,
    F: EmplacableFn<T>,
{
    #[inline]
    fn from(emplacable: Emplacable<T, F>) -> Self {
        arc_new_with(emplacable)
    }
}

/// Like [`Arc::pin`], but `T` can be `?Sized`.
///
/// You will need `#![feature(unsized_fn_params)]`
/// to call this.
#[cfg(all(feature = "alloc", not(miri)))]
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
