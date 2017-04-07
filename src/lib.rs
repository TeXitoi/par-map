// Copyright (c) 2016-2017 Guillaume Pinot <texitoi(a)texitoi.eu>
//
// This work is free. You can redistribute it and/or modify it under
// the terms of the Do What The Fuck You Want To Public License,
// Version 2, as published by Sam Hocevar. See the COPYING file for
// more details.

extern crate futures;
extern crate futures_cpupool;
extern crate num_cpus;

use std::collections::VecDeque;
use std::sync::Arc;
use futures::Future;
use futures_cpupool::{CpuPool, CpuFuture};

pub trait ParMap: Iterator + Sized {
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
