// Copyright (c) 2016-2017 Guillaume Pinot <texitoi(a)texitoi.eu>
//
// This work is free. You can redistribute it and/or modify it under
// the terms of the Do What The Fuck You Want To Public License,
// Version 2, as published by Sam Hocevar. See the COPYING file for
// more details.

//! This crate provides an easy way to get parallel iteration.  The
//! contract of the added method are (almost) exacly the same as the
//! method without the `par_` prefix proposed in `std`.

#[deny(missing_docs)]

extern crate futures;
extern crate futures_cpupool;
extern crate num_cpus;

use std::collections::VecDeque;
use std::sync::Arc;
use futures::Future;
use futures_cpupool::{CpuPool, CpuFuture};

/// This trait extends `std::iter::Iterator` with parallel
/// iterator adaptors.  Just `use` it to get access to the methods:
///
/// ```
/// use par_map::ParMap;
/// ```
pub trait ParMap: Iterator + Sized {
    /// Takes a closure and creates an iterator which calls that
    /// closure on each element, exactly as
    /// `std::iter::Iterator::map`.
    ///
    /// The order of the elements are guaranted to be unchanged.  Of
    /// course, the given closures can be executed in parallel out of
    /// order.
    ///
    /// # Example
    ///
    /// ```
    /// use par_map::ParMap;
    /// let a = [1, 2, 3];
    /// let mut iter = a.iter().cloned().par_map(|x| 2 * x);
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(4));
    /// assert_eq!(iter.next(), Some(6));
    /// assert_eq!(iter.next(), None);
    /// ```
    fn par_map<B, F>(self, f: F) -> Map<Self, B, F>
        where F: Sync + Send + 'static + Fn(Self::Item) -> B,
              B: Send + 'static,
              Self::Item: Send + 'static
    {
        let num_threads = num_cpus::get();
        let mut res = Map {
            pool: CpuPool::new(num_threads),
            queue: VecDeque::new(),
            iter: self,
            f: Arc::new(f),
        };
        for _ in 0..num_threads * 2 {
            res.spawn();
        }
        res
    }

    /// Creates an iterator that works like map, but flattens nested
    /// structure, exactly as `std::iter::Iterator::flat_map`.
    ///
    /// The order of the elements are guaranted to be unchanged.  Of
    /// course, the given closures can be executed in parallel out of
    /// order.
    ///
    /// # Example
    ///
    /// ```
    /// use par_map::ParMap;
    /// let words = ["alpha", "beta", "gamma"];
    /// let merged: String = words.iter()
    ///     .cloned() // as items must be 'static
    ///     .par_flat_map(|s| s.chars()) // exactly as std::iter::Iterator::flat_map
    ///     .collect();
    /// assert_eq!(merged, "alphabetagamma");
    /// ```
    fn par_flat_map<U, F>(self, f: F) -> FlatMap<Self, U, F>
        where F: Sync + Send + 'static + Fn(Self::Item) -> U,
              U: IntoIterator,
              U::Item: Send + 'static,
              Self::Item: Send + 'static
    {
        let num_threads = num_cpus::get();
        let mut res = FlatMap {
            pool: CpuPool::new(num_threads),
            queue: VecDeque::new(),
            iter: self,
            f: Arc::new(f),
            cur_iter: vec![].into_iter(),
        };
        for _ in 0..num_threads * 2 {
            res.spawn();
        }
        res
    }
}
impl<I: Iterator> ParMap for I {}

/// An iterator that maps the values of `iter` with `f`.
///
/// This struct is created by the `flat_map()` method on
/// `ParIter`. See its documentation for more.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Map<I, B, F> {
    pool: CpuPool,
    queue: VecDeque<CpuFuture<B, ()>>,
    iter: I,
    f: Arc<F>,
}
impl<I: Iterator, B: Send + 'static, F> Map<I, B, F>
    where F: Sync + Send + 'static + Fn(I::Item) -> B,
          I::Item: Send + 'static
{
    fn spawn(&mut self) {
        let future = match self.iter.next() {
            None => return,
            Some(item) => {
                let f = self.f.clone();
                self.pool.spawn_fn(move || Ok(f(item)))
            }
        };
        self.queue.push_back(future);
    }
}
impl<I: Iterator, B: Send + 'static, F> Iterator for Map<I, B, F>
    where F: Sync + Send + 'static + Fn(I::Item) -> B,
          I::Item: Send + 'static
{
    type Item = B;
    fn next(&mut self) -> Option<Self::Item> {
        self.queue.pop_front().map(|future| {
            let i = future.wait().unwrap();
            self.spawn();
            i
        })
    }
}

/// An iterator that maps each element to an iterator, and yields the
/// elements of the produced iterators.
///
/// This struct is created by the `par_flat_map()` method on
/// `ParIter`.  See its documentation for more.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct FlatMap<I: Iterator, U: IntoIterator, F> {
    pool: CpuPool,
    queue: VecDeque<CpuFuture<Vec<U::Item>, ()>>,
    iter: I,
    f: Arc<F>,
    cur_iter: ::std::vec::IntoIter<U::Item>,
}
impl<I: Iterator, U: IntoIterator, F> FlatMap<I, U, F>
    where F: Sync + Send + 'static + Fn(I::Item) -> U,
          U::Item: Send + 'static,
          I::Item: Send + 'static
{
    fn spawn(&mut self) {
        let future = match self.iter.next() {
            None => return,
            Some(item) => {
                let f = self.f.clone();
                self.pool.spawn_fn(move || Ok(f(item).into_iter().collect()))
            }
        };
        self.queue.push_back(future);
    }
}
impl<I: Iterator, U: IntoIterator, F> Iterator for FlatMap<I, U, F>
    where F: Sync + Send + 'static + Fn(I::Item) -> U,
          U::Item: Send + 'static,
          I::Item: Send + 'static
{
    type Item = U::Item;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.cur_iter.next() {
                return Some(item);
            }
            let v = match self.queue.pop_front() {
                Some(future) => future.wait().unwrap(),
                None => return None,
            };
            self.cur_iter = v.into_iter();
            self.spawn();
        }
    }
}
