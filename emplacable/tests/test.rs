use std::{
    ffi::{CStr, OsStr, OsString},
    path::{Path, PathBuf},
};

use emplacable::*;

#[test]
fn into_impls() {
    let a: &[Box<i32>] = &[Box::new(1), Box::new(2), Box::new(3), Box::new(4)];
    let _: Box<[Box<i32>]> = box_new_with(a.into());

    let a: &str = "iiiiii";
    let _: Box<str> = box_new_with(a.into());

    let a: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"hiii\0") };
    let _: Box<CStr> = box_new_with(a.into());

    let a: &OsStr = &OsString::from("a");
    let _: Box<OsStr> = box_new_with(a.into());

    let a: &Path = &PathBuf::from("a");
    let _: Box<Path> = box_new_with(a.into());

    let a: &str = "iiiiii";
    let e: Emplacable<str, _> = a.into();
    let b: Box<[u8]> = box_new_with(e.into());

    let _: Box<[u8]> = box_new_with(b.into());

    let v: Vec<Box<i32>> = vec![Box::new(1), Box::new(2), Box::new(3), Box::new(4)];
    let _: Box<[Box<i32>]> = box_new_with(v.into());

    let a: [Box<i32>; 4] = [Box::new(1), Box::new(2), Box::new(3), Box::new(4)];
    let e: Emplacable<[Box<i32>; 4], _> = a.into();
    let _: Box<[Box<i32>]> = box_new_with(e.into());

    let a: [Emplacable<i32, _>; 4] = [
        Box::new(1).into(),
        Box::new(2).into(),
        Box::new(3).into(),
        Box::new(4).into(),
    ];

    let _: Box<[i32; 4]> = box_new_with(a.into());
}
