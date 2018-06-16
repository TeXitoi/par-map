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
#[macro_use]
extern crate pub_iterator_type;

use futures::Future;
use futures_cpupool::{CpuFuture, CpuPool};
use std::collections::VecDeque;
use std::sync::Arc;

/// This trait extends `std::iter::Iterator` with parallel
/// iterator adaptors.  Just `use` it to get access to the methods:
///
/// ```
/// use par_map::ParMap;
/// ```
///
/// Each iterator adaptor will have its own thread pool of the number
/// of CPU.  At maximum, 2 times the number of defined threads
/// (the default is the number of cpus) will be
/// launched in advance, guarantying that the memory will not be
/// exceeded if the iterator is not consumed faster that the
/// production.  To be effective, the given function should be costy
/// to compute and each call should take about the same time.  The
/// `packed` variants will do the same, processing by batch instead of
/// doing one job for each item.
///
/// The `'static` constraints are needed to have such a simple
/// interface.  These adaptors are well suited for big iterators that
/// can't be collected into a `Vec`.  Else, crates such as `rayon` are
/// more suited for this kind of task.
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
    where
        F: Sync + Send + 'static + Fn(Self::Item) -> B,
        B: Send + 'static,
        Self::Item: Send + 'static,
    {
        self.with_nb_threads(num_cpus::get()).par_map(f)
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
    where
        F: Sync + Send + 'static + Fn(Self::Item) -> U,
        U: IntoIterator,
        U::Item: Send + 'static,
        Self::Item: Send + 'static,
    {
        self.with_nb_threads(num_cpus::get()).par_flat_map(f)
    }

    /// Creates an iterator that yields `Vec<Self::Item>` of size `nb`
    /// (or less on the last element).
    ///
    /// # Example
    ///
    /// ```
    /// use par_map::ParMap;
    /// let nbs = [1, 2, 3, 4, 5, 6, 7];
    /// let mut iter = nbs.iter().cloned().pack(3);
    /// assert_eq!(Some(vec![1, 2, 3]), iter.next());
    /// assert_eq!(Some(vec![4, 5, 6]), iter.next());
    /// assert_eq!(Some(vec![7]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn pack(self, nb: usize) -> Pack<Self> {
        Pack { iter: self, nb: nb }
    }

    /// Same as `par_map`, but the parallel work is batched by `nb` items.
    ///
    /// # Example
    ///
    /// ```
    /// use par_map::ParMap;
    /// let a = [1, 2, 3];
    /// let mut iter = a.iter().cloned().par_packed_map(2, |x| 2 * x);
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(4));
    /// assert_eq!(iter.next(), Some(6));
    /// assert_eq!(iter.next(), None);
    /// ```
    fn par_packed_map<'a, B, F>(self, nb: usize, f: F) -> PackedMap<'a, B>
    where
        F: Sync + Send + 'static + Fn(Self::Item) -> B,
        B: Send + 'static,
        Self::Item: Send + 'static,
        Self: 'a,
    {
        self.with_nb_threads(num_cpus::get()).par_packed_map(nb, f)
    }

    /// Same as `par_flat_map`, but the parallel work is batched by `nb` items.
    ///
    /// # Example
    ///
    /// ```
    /// use par_map::ParMap;
    /// let words = ["alpha", "beta", "gamma"];
    /// let merged: String = words.iter()
    ///     .cloned()
    ///     .par_packed_flat_map(2, |s| s.chars())
    ///     .collect();
    /// assert_eq!(merged, "alphabetagamma");
    /// ```
    fn par_packed_flat_map<'a, U, F>(self, nb: usize, f: F) -> PackedFlatMap<'a, U::Item>
    where
        F: Sync + Send + 'static + Fn(Self::Item) -> U,
        U: IntoIterator + 'a,
        U::Item: Send + 'static,
        Self::Item: Send + 'static,
        Self: 'a,
    {
        self.with_nb_threads(num_cpus::get()).par_packed_flat_map(nb, f)
    }

    /// Configure the number of thread used.
    /// If not set, the default is the number of cpus available
    ///
    /// # Example
    ///
    /// ```
    /// use par_map::ParMap;
    /// let a = [1, 2, 3];
    /// let mut iter = a.iter().cloned().with_nb_threads(2).par_map(|x| 2 * x);
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(4));
    /// assert_eq!(iter.next(), Some(6));
    /// assert_eq!(iter.next(), None);
    /// ```
    fn with_nb_threads(self, nb: usize) -> ParMapBuilder<Self> {
        ParMapBuilder {
            iter: self,
            nb_threads: nb,
        }
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
where
    F: Sync + Send + 'static + Fn(I::Item) -> B,
    I::Item: Send + 'static,
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
    fn with_nb_threads(iter: I, f: F, nb_thread: usize) -> Self {
        let mut res = Map {
            pool: CpuPool::new(nb_thread),
            queue: VecDeque::new(),
            iter: iter,
            f: Arc::new(f),
        };
        for _ in 0..nb_thread * 2 {
            res.spawn();
        }
        res
    }
}
impl<I: Iterator, B: Send + 'static, F> Iterator for Map<I, B, F>
where
    F: Sync + Send + 'static + Fn(I::Item) -> B,
    I::Item: Send + 'static,
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
where
    F: Sync + Send + 'static + Fn(I::Item) -> U,
    U::Item: Send + 'static,
    I::Item: Send + 'static,
{
    fn spawn(&mut self) {
        let future = match self.iter.next() {
            None => return,
            Some(item) => {
                let f = self.f.clone();
                self.pool
                    .spawn_fn(move || Ok(f(item).into_iter().collect()))
            }
        };
        self.queue.push_back(future);
    }
    fn with_nb_threads(iter: I, f: F, nb_thread: usize) -> Self {
        let mut res = FlatMap {
            pool: CpuPool::new(nb_thread),
            queue: VecDeque::new(),
            iter: iter,
            f: Arc::new(f),
            cur_iter: vec![].into_iter(),
        };
        for _ in 0..nb_thread * 2 {
            res.spawn();
        }
        res
    }
}
impl<I: Iterator, U: IntoIterator, F> Iterator for FlatMap<I, U, F>
where
    F: Sync + Send + 'static + Fn(I::Item) -> U,
    U::Item: Send + 'static,
    I::Item: Send + 'static,
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

/// An iterator that yields `Vec<Self::Item>` of size `nb` (or less on
/// the last element).
///
/// This struct is created by the `pack()` method on
/// `ParIter`.  See its documentation for more.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Pack<I> {
    iter: I,
    nb: usize,
}
impl<I: Iterator> Iterator for Pack<I> {
    type Item = Vec<I::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        let item: Vec<_> = self.iter.by_ref().take(self.nb).collect();
        if item.is_empty() {
            None
        } else {
            Some(item)
        }
    }
}

pub_iterator_type! {
    #[doc="As `Map` but packed."]
    PackedMap['a, B] = Box<Iterator<Item = B> + 'a>
}
pub_iterator_type! {
    #[doc="As `FlatMap` but packed."]
    PackedFlatMap['a, T] = Box<Iterator<Item = T> + 'a>
}

/// A builder used to configure the parallele work.
///
/// This struct is created by the `with_nb_threads()` method on
/// `ParIter`.  See its documentation for more.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct ParMapBuilder<I> {
    iter: I,
    nb_threads: usize,
}

impl<I: Iterator> ParMapBuilder<I> {
    /// As ParMap::par_map, but with a custom number of threads
    /// # Example
    ///
    /// ```
    /// use par_map::ParMap;
    /// let a = [1, 2, 3];
    /// let mut iter = a.iter()
    ///     .cloned()
    ///     .with_nb_threads(2)
    ///     .par_map(|x| 2 * x);
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(4));
    /// assert_eq!(iter.next(), Some(6));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn par_map<B, F>(self, f: F) -> Map<I, B, F>
    where
        F: Sync + Send + 'static + Fn(I::Item) -> B,
        B: Send + 'static,
        I::Item: Send + 'static,
    {
        Map::with_nb_threads(self.iter, f, self.nb_threads)
    }

    /// As ParMap::par_flat_map, but with a custom number of threads
    /// # Example
    ///
    /// ```
    /// use par_map::ParMap;
    /// let words = ["alpha", "beta", "gamma"];
    /// let merged: String = words.iter()
    ///     .cloned() // as items must be 'static
    ///     .with_nb_threads(2)
    ///     .par_flat_map(|s| s.chars()) // exactly as std::iter::Iterator::flat_map
    ///     .collect();
    /// assert_eq!(merged, "alphabetagamma");
    /// ```
    pub fn par_flat_map<U, F>(self, f: F) -> FlatMap<I, U, F>
    where
        F: Sync + Send + 'static + Fn(I::Item) -> U,
        U: IntoIterator,
        U::Item: Send + 'static,
        I::Item: Send + 'static,
    {
        FlatMap::with_nb_threads(self.iter, f, self.nb_threads)
    }

    /// As ParMap::par_packed_map, but with a custom number of threads
    ///
    /// # Example
    ///
    /// ```
    /// use par_map::ParMap;
    /// let a = [1, 2, 3];
    /// let mut iter = a.iter()
    ///     .cloned()
    ///     .with_nb_threads(2)
    ///     .par_packed_map(2, |x| 2 * x);
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), Some(4));
    /// assert_eq!(iter.next(), Some(6));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn par_packed_map<'a, B, F>(self, nb: usize, f: F) -> PackedMap<'a, B>
    where
        F: Sync + Send + 'static + Fn(I::Item) -> B,
        B: Send + 'static,
        I::Item: Send + 'static,
        Self: 'a,
    {
        let f = Arc::new(f);
        let f = move |iter: Vec<I::Item>| {
            let f = f.clone();
            iter.into_iter().map(move |i| f(i))
        };
        PackedMap(Box::new(
            self.iter
                .pack(nb)
                .with_nb_threads(self.nb_threads)
                .par_flat_map(f),
        ))
    }

    /// As ParMap::par_packed_flat_map, but with a custom number of threads
    ///
    /// # Example
    ///
    /// ```
    /// use par_map::ParMap;
    /// let words = ["alpha", "beta", "gamma"];
    /// let merged: String = words.iter()
    ///     .cloned()
    ///     .with_nb_threads(2)
    ///     .par_packed_flat_map(2, |s| s.chars())
    ///     .collect();
    /// assert_eq!(merged, "alphabetagamma");
    /// ```
    pub fn par_packed_flat_map<'a, U, F>(self, nb: usize, f: F) -> PackedFlatMap<'a, U::Item>
    where
        F: Sync + Send + 'static + Fn(I::Item) -> U,
        U: IntoIterator + 'a,
        U::Item: Send + 'static,
        I::Item: Send + 'static,
        Self: 'a,
    {
        let f = Arc::new(f);
        let f = move |iter: Vec<I::Item>| {
            let f = f.clone();
            iter.into_iter().flat_map(move |i| f(i))
        };
        PackedFlatMap(Box::new(
            self.iter
                .pack(nb)
                .with_nb_threads(self.nb_threads)
                .par_flat_map(f),
        ))
    }
}
