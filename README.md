# `unsized-vec`

`UnsizedVec<T>` is like [`Vec<T>`](https://doc.rust-lang.org/alloc/vec/struct.Vec.html), but `T` can be `?Sized`.

Features:

- Similar API to `Vec`.
- Same time complexity as `Vec` for indexing, push, pop, insert, remove (more or less)
  - Exception: when `T`'s alignment isn't fixed at compile-time,
    adding a new element to the `Vec` with a greater alignment than all elements currently present
    will take $\mathcal{O}(n)$ time, and will most likely reallocate.
- For `T: Sized`, only one heap allocation, approximately same memory layout as `Vec`.
- For unsized `T`, two heap allocations (one for the elements, one for the pointer metadata).
- `#[no_std]` (but requires `alloc`).
- Experimental, nightly-only.
