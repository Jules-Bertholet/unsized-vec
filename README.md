# `unsized-vec`

`UnsizedVec<T>` is like [`Vec<T>`](https://doc.rust-lang.org/alloc/vec/struct.Vec.html), but `T` can be `?Sized`.

## Features

- Similar API to `Vec`.
- Same time complexity as `Vec` for indexing, push, pop, insert, remove (more or less)
  - Exception: when `T`'s alignment isn't fixed at compile-time,
    adding a new element to the `Vec` with a greater alignment than all elements currently present
    will take $\mathcal{O}(n)$ time, and will most likely reallocate.
- For `T: Sized`, only one heap allocation, approximately same memory layout as `Vec`.
- For unsized `T`, two heap allocations (one for the elements, one for the pointer metadata).
- `#[no_std]` (but requires `alloc`).
- Experimental, nightly-only.

## Example

```rust
#![feature(unsized_fn_params)]
use core::fmt::Debug;
use unsized_vec::{*, emplace::box_new_with};

// `Box::new()` necessary only to coerce the values to trait objects.
let obj: Box<dyn Debug> = Box::new(1);
let obj_2: Box<dyn Debug> = Box::new((97_u128, "oh noes"));
let mut vec: UnsizedVec<dyn Debug> = unsized_vec![*obj, *obj_2];
for traitobj in &vec {
    dbg!(traitobj);
};

assert_eq!(vec.len(), 2);

let popped = box_new_with(|e| vec.pop_unwrap(e));
dbg!(&*popped);

assert_eq!(vec.len(), 1);
```
