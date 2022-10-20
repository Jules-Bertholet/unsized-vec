//! The example from [section 17.2 of the Rust book][0],
//! modified to use `UnsizedVec` instead of `Vec<Box<_>>>`.
//!
//! [0]: https://doc.rust-lang.org/book/ch17-02-trait-objects.html

#![allow(dead_code)]
#![feature(allocator_api, ptr_metadata, unsized_fn_params)]

use emplacable::by_value_str;
use unsized_vec::{unsize_vec, unsized_vec, UnsizedVec};

mod gui {
    //! lib.rs

    use unsized_vec::UnsizedVec;

    pub trait Draw {
        fn draw(&self);
    }
    pub struct Screen {
        pub components: UnsizedVec<dyn Draw>,
    }

    impl Screen {
        pub fn run(&self) {
            for component in self.components.iter() {
                component.draw();
            }
        }
    }
    pub struct Button {
        pub width: u32,
        pub height: u32,
        pub label: Box<str>,
    }

    impl Draw for Button {
        fn draw(&self) {
            // code to actually draw a button
        }
    }
}

// main.rs
use gui::Draw;

struct SelectBox {
    width: u32,
    height: u32,
    options: UnsizedVec<str>,
}

impl Draw for SelectBox {
    fn draw(&self) {
        // code to actually draw a select box
    }
}

use gui::{Button, Screen};

fn main() {
    let screen = Screen {
        components: unsize_vec![
            SelectBox {
                width: 75,
                height: 10,
                options: unsized_vec![
                    by_value_str!("Yes"),
                    by_value_str!("Maybe"),
                    by_value_str!("No"),
                ],
            },
            Button {
                width: 50,
                height: 10,
                label: Box::<str>::from("OK"),
            },
        ],
    };

    screen.run();
}
