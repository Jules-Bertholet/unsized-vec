#![feature(test)]

extern crate test;

use std::hint::black_box;

use test::bench::Bencher;
use unsized_vec::UnsizedVec;

const ARR0: [i32; 0] = [];
const ARR1: [i32; 1] = [23];
const ARR2: [i32; 2] = [-4, 27];
const ARR3: [i32; 3] = [-4, 27, 31];

const ARR4: [i32; 4] = [-4, 27, 31, 42];

const ARR13: [i32; 13] = [
    -4, 27, 31, 42, 43, 342, 2342, -324, 234, 234, 65, 123, 32465532,
];

#[bench]
fn test_push_arrays(b: &mut Bencher) {
    //let mut lock = stdout().lock();
    b.iter(|| {
        let mut v: UnsizedVec<[i32]> = UnsizedVec::new();

        for _ in 0..255 {
            // writeln!(lock, "0");
            v.push_unsize(black_box(ARR0));
            // writeln!(lock, "1");
            v.push_unsize(black_box(ARR1));
            // writeln!(lock, "2");
            v.push_unsize(black_box(ARR2));
            // writeln!(lock, "3");
            v.push_unsize(black_box(ARR3));
            // writeln!(lock, "4");
            v.push_unsize(black_box(ARR4));
            //writeln!(lock, "5");
            v.push_unsize(black_box(ARR13));
            //writeln!(lock, "6");
        }

        black_box(v);
    });
}

#[bench]
fn test_push_arrays_boxed(b: &mut Bencher) {
    b.iter(|| {
        let mut v: Vec<Box<[i32]>> = Vec::new();

        for _ in 0..255 {
            v.push(Box::new(black_box(ARR0)));
            v.push(Box::new(black_box(ARR1)));
            v.push(Box::new(black_box(ARR2)));
            v.push(Box::new(black_box(ARR3)));
            v.push(Box::new(black_box(ARR4)));
            v.push(Box::new(black_box(ARR13)));
        }

        black_box(v);
    });
}

#[bench]
fn test_push_arrays_preallocated(b: &mut Bencher) {
    b.iter(|| {
        let mut v: UnsizedVec<[i32]> =
            UnsizedVec::with_capacity_bytes(256 * 6, 256 * 4 * (13 + 4 + 3 + 2 + 1));

        for _ in 0..255 {
            v.push_unsize(black_box(ARR0));
            v.push_unsize(black_box(ARR1));
            v.push_unsize(black_box(ARR2));
            v.push_unsize(black_box(ARR3));
            v.push_unsize(black_box(ARR4));
            v.push_unsize(black_box(ARR13));
        }

        black_box(v);
    });
}

#[bench]
fn test_push_arrays_boxed_preallocated(b: &mut Bencher) {
    b.iter(|| {
        let mut v: Vec<Box<[i32]>> = Vec::with_capacity(6 * 256);

        for _ in 0..255 {
            v.push(Box::new(black_box(ARR0)));
            v.push(Box::new(black_box(ARR1)));
            v.push(Box::new(black_box(ARR2)));
            v.push(Box::new(black_box(ARR3)));
            v.push(Box::new(black_box(ARR4)));
            v.push(Box::new(black_box(ARR13)));
        }

        black_box(v);
    });
}
