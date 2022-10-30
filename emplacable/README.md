# `emplacable`

[![docs.rs](https://img.shields.io/docsrs/emplacable)](https://docs.rs/emplacable/) [![Crates.io](https://img.shields.io/crates/v/emplacable)](https://crates.io/crates/emplacable)

Return values of unsized types, like `[i32]` or `dyn Any`, from functions,
with a mechanism similar to placement new.

Written to support the [`unsized-vec`](https://crates.io/crates/unsized-vec) crate, but is independent of it.
Experimental, and requires nightly Rust.
