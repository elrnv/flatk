use super::*;
use rayon::iter::plumbing::*;
use rayon::prelude::*;

/// A parallel chunk iterator.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ChunkedParIter<I, S> {
    sizes: I,
    data: S,
}

impl<I, S> ParallelIterator for ChunkedParIter<I, S>
where
    S: Send + SplitAt + Dummy + Set,
    I: Send + GetOffset + IndexedParallelIterator + Producer<Item = usize>,
    I::IntoIter: ExactSizeIterator<Item = usize> + GetOffset,
{
    type Item = S;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<I, S> IndexedParallelIterator for ChunkedParIter<I, S>
where
    S: Send + SplitAt + Dummy + Set,
    I: Send + GetOffset + IndexedParallelIterator + Producer<Item = usize>,
    I::IntoIter: ExactSizeIterator<Item = usize> + GetOffset,
{
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.sizes.len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(ChunkedProducer {
            sizes_producer: self.sizes,
            data: self.data,
        })
    }
}

struct ChunkedProducer<I, S> {
    sizes_producer: I,
    data: S,
}

impl<I, S> Producer for ChunkedProducer<I, S>
where
    S: Send + SplitAt + Dummy + Set,
    I: Send + GetOffset + Producer<Item = usize>,
    I::IntoIter: ExactSizeIterator<Item = usize> + GetOffset,
{
    type Item = S;
    type IntoIter = ChunkedIter<I::IntoIter, S>;

    fn into_iter(self) -> Self::IntoIter {
        ChunkedIter {
            sizes: self.sizes_producer.into_iter(),
            data: self.data,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let off = self.sizes_producer.offset(index);
        let (ls, rs) = self.sizes_producer.split_at(index);
        let (l, r) = self.data.split_at(off);
        (
            ChunkedProducer {
                sizes_producer: ls,
                data: l,
            },
            ChunkedProducer {
                sizes_producer: rs,
                data: r,
            },
        )
    }
}

impl<'a, S, O> Chunked<S, O>
where
    S: View<'a>,
    O: View<'a>,
    O::Type: IntoParSizes,
{
    /// Produce a parallel iterator over elements (borrowed slices) of a `Chunked`.
    #[inline]
    pub fn par_iter(
        &'a self,
    ) -> ChunkedParIter<<<O as View<'a>>::Type as IntoParSizes>::ParIter, <S as View<'a>>::Type>
    {
        ChunkedParIter {
            sizes: self.chunks.view().into_par_sizes(),
            data: self.data.view(),
        }
    }
}

impl<S, O> IntoParallelIterator for Chunked<S, O>
where
    O: IntoParSizes,
    S: Send + SplitAt + Set + Dummy,
    O::ParIter: Producer<Item = usize> + GetOffset,
    <O::ParIter as Producer>::IntoIter: GetOffset,
{
    type Item = S;
    type Iter = ChunkedParIter<O::ParIter, S>;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        ChunkedParIter {
            sizes: self.chunks.into_par_sizes(),
            data: self.data,
        }
    }
}
