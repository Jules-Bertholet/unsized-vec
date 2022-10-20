#![feature(allocator_api, ptr_metadata, unsized_fn_params)]

use std::fmt::Debug;

use emplacable::*;
use unsized_vec::*;

#[test]
fn emplacable_from() {
    let a: Box<u32> = box_new_with(3.into());
    assert_eq!(*a, 3);
}

#[test]
fn test_sized() {
    let mut vec = UnsizedVec::new();
    assert_eq!(vec.len(), 0);

    vec.push(32);
    assert_eq!(vec.len(), 1);
    assert_eq!(vec[0], 32);

    vec.shrink_to_fit();

    vec.push(34);
    assert_eq!(vec.len(), 2);
    assert_eq!(vec[1], 34);
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

    let popped: Box<[Box<i32>]> = vec.pop_into().map(box_new_with).unwrap();
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

    vec.shrink_to_fit();

    let removed: Box<[Box<i32>]> = box_new_with(vec.remove_into(1));
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

    let popped = vec.pop_into().map(box_new_with).unwrap();
    assert_eq!(vec.len(), 1);
    assert_eq!(&format!("{:?}", &*popped), "1");

    vec.shrink_to_fit();

    let obj: Box<dyn Debug> = Box::new("walla walla");
    vec.insert(0, *obj);
    assert_eq!(vec.len(), 2);
    assert_eq!(&format!("{:?}", &vec[0]), "\"walla walla\"");
    assert_eq!(&format!("{:?}", &vec[1]), "()");
    dbg!(&vec);

    let removed: Box<dyn Debug> = box_new_with(vec.remove_into(0));
    assert_eq!(vec.len(), 1);
    assert_eq!(&format!("{:?}", &*removed), "\"walla walla\"");
    assert_eq!(&format!("{:?}", &vec[0]), "()");
}

#[test]
fn test_unsized_aligned() {
    let mut vec: UnsizedVec<[i32]> = UnsizedVec::new();
    assert_eq!(vec.len(), 0);

    let slice: Box<[i32]> = Box::new([1, 2]);
    vec.push(*slice);
    assert_eq!(vec.len(), 1);
    assert_eq!(&vec[0], &[1, 2]);

    vec.push(unsize!([], ([i32; 0]) -> [i32]));
    assert_eq!(&vec[1], &[]);

    vec.shrink_to_fit();

    vec.push_unsize([4, 7, 3]);
    vec[2][1] = 19;
    assert_eq!(&vec[2], &[4, 19, 3]);

    let popped: Box<[i32]> = vec.pop_into().map(box_new_with).unwrap();
    assert_eq!(&*popped, &[4, 19, 3]);

    vec.insert_unsize(0, [4, 7, 3, 4, 5, 6, 6, -1]);
    assert_eq!(&vec[0], &[4, 7, 3, 4, 5, 6, 6, -1]);
    assert_eq!(&vec[1], &[1, 2]);
    assert_eq!(&vec[2], &[]);

    let removed: Box<[i32]> = box_new_with(vec.remove_into(1));
    assert_eq!(&*removed, &[1, 2]);
    assert_eq!(&vec[0], &[4, 7, 3, 4, 5, 6, 6, -1]);
    assert_eq!(&vec[1], &[]);
}

#[test]
fn type_inference() {
    let mut vec: UnsizedVec<[i32; 3]> = unsized_vec![[33, 34, 35]];
    let emplacable: Emplacable<[i32; 3], _> = vec.pop_into().unwrap();
    let _: Emplacable<[i32], _> = emplacable.into();
}
