# `unsized-vec`

[![docs.rs](https://img.shields.io/docsrs/unsized-vec)](https://docs.rs/unsized-vec/) [![Crates.io](https://img.shields.io/crates/v/unsized-vec)](https://crates.io/crates/unsized-vec)

Say goodbye to `Vec<Box<dyn Any>>`! Cut down on your heap allocations.
`UnsizedVec<T>` is like [`Vec<T>`](https://doc.rust-lang.org/alloc/vec/struct.Vec.html), but `T` can be `?Sized`.

## Features

- Familiar `Vec` API.
- Same time complexity as `alloc::vec::Vec` for major operations(indexing, push, pop, insert, remove).
  - When `T`'s alignment is not known at compile time (e.g. `T` is a trait object), this rule has one expection,
    explained in the crate docs.
- For `T: Sized`, `UnsizedVec<T>` compiles to a newtype around `alloc::vec::Vec`, and can be trivially converted to/from it.
- For unsized `T`, there are two heap allocations: one for the elements, and one for the pointer metadata.
- `#[no_std]` (but requires `alloc`).

## Drawbacks

- Invariant in `T`.
- Experimental, nightly-only.

## Example

```rust
#![allow(internal_features)] // for `unsized_fn_params`
#![feature(unsized_fn_params)]

use core::fmt::Debug;

use emplacable::box_new_with;
use unsized_vec::{unsize_vec, UnsizedVec};

fn main() {
    let mut vec: UnsizedVec<dyn Debug> = unsize_vec![27.53_f32, "oh the places we'll go", Some(())];

    for traitobj in &vec {
        dbg!(traitobj);
    };

    assert_eq!(vec.len(), 3);

    let maybe_popped: Option<Box<dyn Debug>> = vec.pop_into().map(box_new_with);
    let popped = maybe_popped.unwrap();
    dbg!(&*popped);

    assert_eq!(vec.len(), 2);
}
```

## License

`unsized-vec` is distributed under the terms of both the MIT license and the Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details.
