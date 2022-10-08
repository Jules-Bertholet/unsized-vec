#![feature(unsized_fn_params)]

use std::fmt::Debug;

use emplace::*;
use unsized_vec::*;

#[test]
fn test_sized() {
    let mut vec = UnsizedVec::new();
    assert_eq!(vec.len(), 0);

    vec.push(32);
    assert_eq!(vec.len(), 1);
    assert_eq!(vec[0], 32);

    vec.push(34);
    assert_eq!(vec.len(), 2);
    assert_eq!(vec[1], 34);
}

#[test]
fn test_unsized() {
    let mut vec: UnsizedVec<[i32]> = UnsizedVec::new();
    assert_eq!(vec.len(), 0);

    let slice: Box<[i32]> = Box::new([1, 2]);
    vec.push(*slice);
    assert_eq!(vec.len(), 1);
    assert_eq!(&vec[0], &[1, 2]);

    let slice: Box<[i32]> = Box::new([]);
    vec.push(*slice);
    assert_eq!(&vec[1], &[]);

    let slice: Box<[i32]> = Box::new([4, 7, 3]);
    vec.push(*slice);
    vec[2][1] = 19;
    assert_eq!(&vec[2], &[4, 19, 3]);

    let popped: Box<[i32]> = box_new_with(|e| vec.pop_unwrap(e));
    assert_eq!(&*popped, &[4, 19, 3]);

    let slice: Box<[i32]> = Box::new([4, 7, 3, 4, 5, 6, 6, -1]);
    vec.insert(0, *slice);
    assert_eq!(&vec[0], &[4, 7, 3, 4, 5, 6, 6, -1]);
    assert_eq!(&vec[1], &[1, 2]);
    assert_eq!(&vec[2], &[]);

    let removed: Box<[i32]> = box_new_with(|e| vec.remove(1, e));
    assert_eq!(&*removed, &[1, 2]);
    assert_eq!(&vec[0], &[4, 7, 3, 4, 5, 6, 6, -1]);
    assert_eq!(&vec[1], &[]);
}

#[test]
fn test_unsized_drop() {
    let mut vec: UnsizedVec<[Box<i32>]> = UnsizedVec::new();
    assert_eq!(vec.len(), 0);

    let slice: Box<[Box<i32>]> = Box::new([Box::new(1), Box::new(2)]);
    vec.push(*slice);
    assert_eq!(vec.len(), 1);
    assert_eq!(&vec[0], &[Box::new(1), Box::new(2)]);

    let slice: Box<[Box<i32>]> = Box::new([]);
    vec.push(*slice);
    assert_eq!(&vec[1], &[]);

    let slice: Box<[Box<i32>]> = Box::new([Box::new(4), Box::new(7), Box::new(3)]);
    vec.push(*slice);
    vec[2][1] = Box::new(19);
    assert_eq!(&vec[2], &[Box::new(4), Box::new(19), Box::new(3)]);

    let popped: Box<[Box<i32>]> = box_new_with(|e| vec.pop_unwrap(e));
    assert_eq!(&*popped, &[Box::new(4), Box::new(19), Box::new(3)]);

    let slice: Box<[Box<i32>]> = Box::new([
        Box::new(4),
        Box::new(7),
        Box::new(3),
        Box::new(4),
        Box::new(5),
        Box::new(6),
        Box::new(6),
        Box::new(-1),
    ]);
    vec.insert(0, *slice);
    assert_eq!(
        &vec[0],
        &[
            Box::new(4),
            Box::new(7),
            Box::new(3),
            Box::new(4),
            Box::new(5),
            Box::new(6),
            Box::new(6),
            Box::new(-1)
        ]
    );
    assert_eq!(&vec[1], &[Box::new(1), Box::new(2)]);
    assert_eq!(&vec[2], &[]);

    let removed: Box<[Box<i32>]> = box_new_with(|e| vec.remove(1, e));
    assert_eq!(&*removed, &[Box::new(1), Box::new(2)]);
    assert_eq!(
        &vec[0],
        &[
            Box::new(4),
            Box::new(7),
            Box::new(3),
            Box::new(4),
            Box::new(5),
            Box::new(6),
            Box::new(6),
            Box::new(-1)
        ]
    );
    assert_eq!(&vec[1], &[]);
}

#[test]
fn test_dyn() {
    let mut vec: UnsizedVec<dyn Debug> = UnsizedVec::new();
    assert_eq!(vec.len(), 0);

    let obj: Box<dyn Debug> = Box::new(());
    vec.push(*obj);
    assert_eq!(vec.len(), 1);

    let obj: Box<dyn Debug> = Box::new(1_u16);
    vec.push(*obj);
    assert_eq!(vec.len(), 2);

    let popped = box_new_with(|e| vec.pop_unwrap(e));
    assert_eq!(vec.len(), 1);
    assert_eq!(&format!("{:?}", &*popped), "1");

    let obj: Box<dyn Debug> = Box::new("walla walla");
    vec.insert(0, *obj);
    assert_eq!(vec.len(), 2);
    assert_eq!(&format!("{:?}", &vec[0]), "\"walla walla\"");
    assert_eq!(&format!("{:?}", &vec[1]), "()");
    dbg!(&vec);

    let removed: Box<dyn Debug> = box_new_with(|e| vec.remove(0, e));
    assert_eq!(vec.len(), 1);
    assert_eq!(&format!("{:?}", &*removed), "\"walla walla\"");
    assert_eq!(&format!("{:?}", &vec[0]), "()");
}

#[test]
fn test_macro() {
    let obj: Box<dyn Debug> = Box::new(1);
    let obj_2: Box<dyn Debug> = Box::new((97_u128, "oh noes"));
    let vec: UnsizedVec<dyn Debug> = unsized_vec![*obj, *obj_2];
    assert_eq!(&format!("{:?}", &vec), "[1, (97, \"oh noes\")]");
}
