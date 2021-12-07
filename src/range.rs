/**
 * Ranges can be used as collections of read-only indices that can be truncated from either end.
 * It can be useful to specify chunked ranges or a selection from a range.
 * Since our collections must know about the length, only finite ranges are supported.
 */
use super::*;
use std::ops::{Range, RangeInclusive, RangeTo, RangeToInclusive};

impl<T: IntBound> BoundedRange for Range<T> {
    type Index = T;
    #[inline]
    fn start(&self) -> Self::Index {
        self.start.clone()
    }
    #[inline]
    fn end(&self) -> Self::Index {
        self.end.clone()
    }
}

impl<T: IntBound> BoundedRange for RangeInclusive<T> {
    type Index = T;
    #[inline]
    fn start(&self) -> Self::Index {
        RangeInclusive::start(self).clone()
    }
    #[inline]
    fn end(&self) -> Self::Index {
        RangeInclusive::end(self).clone() + 1
    }
}

impl<T: IntBound> BoundedRange for RangeTo<T> {
    type Index = T;
    #[inline]
    fn start(&self) -> Self::Index {
        0usize.into()
    }
    #[inline]
    fn end(&self) -> Self::Index {
        self.end.clone()
    }
}

impl<T: IntBound> BoundedRange for RangeToInclusive<T> {
    type Index = T;
    #[inline]
    fn start(&self) -> Self::Index {
        0usize.into()
    }
    #[inline]
    fn end(&self) -> Self::Index {
        self.end.clone() + 1
    }
}

macro_rules! impls_for_range {
    ($range:ident) => {
        impl<T> DynamicRangeIndexType for $range<T> {}
        impl<T> ValueType for $range<T> {}
        impl<I: IntBound> Set for $range<I> {
            type Elem = <Self as BoundedRange>::Index;
            type Atom = <Self as BoundedRange>::Index;
            #[inline]
            fn len(&self) -> usize {
                (BoundedRange::end(self) - BoundedRange::start(self)).into()
            }
        }

        impl<N, I> UniChunkable<N> for $range<I> {
            type Chunk = StaticRange<N>;
        }
        impl<'a, I: IntBound> View<'a> for $range<I> {
            type Type = Self;
            #[inline]
            fn view(&'a self) -> Self::Type {
                self.clone()
            }
        }
    };
}

impls_for_range!(Range);
impls_for_range!(RangeInclusive);
impls_for_range!(RangeTo);
impls_for_range!(RangeToInclusive);

impl<'a, R> GetIndex<'a, R> for usize
where
    R: BoundedRange + Set,
{
    type Output = R::Index;
    #[inline]
    fn get(self, rng: &R) -> Option<Self::Output> {
        if self >= rng.len() {
            return None;
        }
        Some(rng.start() + self)
    }
}

impl<'a, R> GetIndex<'a, R> for Range<usize>
where
    R: BoundedRange + Set,
{
    type Output = Range<R::Index>;
    #[inline]
    fn get(self, rng: &R) -> Option<Self::Output> {
        if self.end > rng.len() {
            return None;
        }
        Some(Range {
            start: rng.start() + self.start,
            end: rng.start() + self.end,
        })
    }
}

impl<I, N> SplitPrefix<N> for Range<I>
where
    I: IntBound + Default + Copy + From<usize>,
    std::ops::RangeFrom<I>: Iterator<Item = I>,
    N: Unsigned + Array<I>,
    <N as Array<I>>::Array: Default,
{
    type Prefix = N::Array;

    #[inline]
    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::to_usize() {
            return None;
        }

        let std::ops::Range { start, end } = self;

        let mut prefix: N::Array = Default::default();
        for (i, item) in (start..).zip(N::iter_mut(&mut prefix)) {
            *item = i;
        }

        let start = start.into();

        let rest = Range {
            start: (start + N::to_usize()).into(),
            end,
        };

        Some((prefix, rest))
    }
}

impl<I, N> IntoStaticChunkIterator<N> for Range<I>
where
    Self: Set + SplitPrefix<N> + Dummy,
    N: Unsigned,
{
    type Item = <Self as SplitPrefix<N>>::Prefix;
    type IterType = UniChunkedIter<Self, N>;
    #[inline]
    fn into_static_chunk_iter(self) -> Self::IterType {
        self.into_generic_static_chunk_iter()
    }
}

impl<T> IntoStorage for Range<T> {
    type StorageType = Range<T>;
    #[inline]
    fn into_storage(self) -> Self::StorageType {
        self
    }
}

impl<T> SplitAt for Range<T>
where
    T: From<usize>,
{
    #[inline]
    fn split_at(self, mid: usize) -> (Self, Self) {
        let Range { start, end } = self;
        (
            Range {
                start,
                end: mid.into(),
            },
            Range {
                start: mid.into(),
                end,
            },
        )
    }
}

impl<T> Dummy for Range<T>
where
    T: Default,
{
    #[inline]
    unsafe fn dummy() -> Self {
        Range {
            start: T::default(),
            end: T::default(),
        }
    }
}

impl<T> Dummy for RangeTo<T>
where
    T: Default,
{
    #[inline]
    unsafe fn dummy() -> Self {
        RangeTo { end: T::default() }
    }
}

impl<T> RemovePrefix for Range<T>
where
    T: From<usize>,
{
    #[inline]
    fn remove_prefix(&mut self, n: usize) {
        self.start = n.into();
    }
}

impl<'a, T: Clone> StorageView<'a> for Range<T> {
    type StorageView = Self;
    #[inline]
    fn storage_view(&'a self) -> Self::StorageView {
        self.clone()
    }
}

impl<T> Storage for Range<T> {
    type Storage = Range<T>;
    /// A range is a type of storage, simply return an immutable reference to self.
    #[inline]
    fn storage(&self) -> &Self::Storage {
        self
    }
}

impl<T> StorageMut for Range<T> {
    /// A range is a type of storage, simply return a mutable reference to self.
    #[inline]
    fn storage_mut(&mut self) -> &mut Self::Storage {
        self
    }
}

impl<T: IntBound> Truncate for Range<T> {
    /// Truncate the range to a specified length.
    #[inline]
    fn truncate(&mut self, new_len: usize) {
        self.end = self.start.clone() + new_len;
    }
}

impl<T: IntBound> IsolateIndex<Range<T>> for usize {
    type Output = T;
    #[inline]
    unsafe fn isolate_unchecked(self, rng: Range<T>) -> Self::Output {
        rng.start + self
    }
    #[inline]
    fn try_isolate(self, rng: Range<T>) -> Option<Self::Output> {
        if self < rng.distance().into() {
            Some(rng.start + self)
        } else {
            None
        }
    }
}

impl<T: IntBound> IsolateIndex<Range<T>> for std::ops::Range<usize> {
    type Output = Range<T>;

    #[inline]
    unsafe fn isolate_unchecked(self, rng: Range<T>) -> Self::Output {
        Range {
            start: rng.start.clone() + self.start,
            end: rng.start + self.end,
        }
    }
    #[inline]
    fn try_isolate(self, rng: Range<T>) -> Option<Self::Output> {
        if self.start >= rng.distance().into() || self.end > rng.distance().into() {
            return None;
        }

        Some(Range {
            start: rng.start.clone() + self.start,
            end: rng.start + self.end,
        })
    }
}

impl<T: IntBound> IsolateIndex<RangeTo<T>> for usize {
    type Output = T;
    #[inline]
    unsafe fn isolate_unchecked(self, _: RangeTo<T>) -> Self::Output {
        self.into()
    }
    #[inline]
    fn try_isolate(self, rng: RangeTo<T>) -> Option<Self::Output> {
        if self < rng.distance().into() {
            Some(self.into())
        } else {
            None
        }
    }
}

impl<T: IntBound> IsolateIndex<RangeTo<T>> for std::ops::Range<usize> {
    type Output = Range<T>;

    #[inline]
    unsafe fn isolate_unchecked(self, _: RangeTo<T>) -> Self::Output {
        Range {
            start: self.start.into(),
            end: self.end.into(),
        }
    }
    #[inline]
    fn try_isolate(self, rng: RangeTo<T>) -> Option<Self::Output> {
        if self.start >= rng.distance().into() || self.end > rng.distance().into() {
            return None;
        }

        Some(Range {
            start: self.start.into(),
            end: self.end.into(),
        })
    }
}

impl<T> IntoOwned for Range<T> {
    type Owned = Self;
    #[inline]
    fn into_owned(self) -> Self::Owned {
        self
    }
}

impl<T> IntoOwnedData for Range<T> {
    type OwnedData = Self;
    #[inline]
    fn into_owned_data(self) -> Self::OwnedData {
        self
    }
}

impl<T> IntoOwned for RangeTo<T> {
    type Owned = Self;
    #[inline]
    fn into_owned(self) -> Self::Owned {
        self
    }
}

impl<T> IntoOwnedData for RangeTo<T> {
    type OwnedData = Self;
    #[inline]
    fn into_owned_data(self) -> Self::OwnedData {
        self
    }
}

// Ranges are lightweight and are considered to be viewed types since they are
// cheap to operate on.
impl<T> Viewed for Range<T> {}
impl<T> Viewed for RangeInclusive<T> {}
impl<T> Viewed for RangeTo<T> {}
impl<T> Viewed for RangeToInclusive<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn range_set() {
        assert_eq!(Set::len(&(0..100)), 100);
        assert_eq!(Set::len(&(1..=100)), 100);
        assert_eq!(Set::len(&(..100)), 100);
        assert_eq!(Set::len(&(..=99)), 100);
    }
}
