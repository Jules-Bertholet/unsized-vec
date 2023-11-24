use std::{hint::black_box, mem::size_of};

use unsized_vec::UnsizedVec;

const ARR0: [i32; 0] = [];
const ARR1: [i32; 1] = [23];
const ARR2: [i32; 2] = [-4, 27];
const ARR3: [i32; 3] = [-4, 27, 31];

const ARR4: [i32; 4] = [-4, 27, 31, 42];

const ARR13: [i32; 13] = [
    -4, 27, 31, 42, 43, 342, 2342, -324, 234, 234, 65, 123, 32465532,
];

fn main() {
    for _ in 0..100 {
        let mut v: UnsizedVec<[i32]> =
            UnsizedVec::with_capacity_bytes(256 * 6, size_of::<i32>() * 256 * (13 + 4 + 3 + 2 + 1));

        for _ in 0..255 {
            v.push_unsize(black_box(ARR0));
            v.push_unsize(black_box(ARR1));
            v.push_unsize(black_box(ARR2));
            v.push_unsize(black_box(ARR3));
            v.push_unsize(black_box(ARR4));
            v.push_unsize(black_box(ARR13));
        }
        assert_eq!(
            v.byte_capacity(),
            size_of::<i32>() * 256 * (13 + 4 + 3 + 2 + 1)
        );

        black_box(v);
    }
}
