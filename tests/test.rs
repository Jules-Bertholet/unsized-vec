#![feature(unsized_fn_params)]

use unsized_vec::*;

#[test]
fn test() {
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
}
