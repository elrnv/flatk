use super::*;
use rayon::iter::plumbing::*;
use rayon::prelude::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParSizes<'a> {
    offsets: Offsets<&'a [usize]>,
}

impl<'o> ParallelIterator for ParSizes<'o> {
    type Item = usize;

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

impl<'o> IndexedParallelIterator for ParSizes<'o> {
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.offsets.num_offsets() - 1
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(self)
    }
}

impl<'o> Producer for ParSizes<'o> {
    type Item = usize;
    type IntoIter = Sizes<'o>;

    fn into_iter(self) -> Self::IntoIter {
        self.offsets.into_sizes()
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.offsets.split_offsets_at(index);
        (ParSizes { offsets: left }, ParSizes { offsets: right })
    }
}

impl<'a> IntoParSizes for Offsets<&'a [usize]> {
    type ParIter = ParSizes<'a>;
    /// Returns a parallel iterator over chunk sizes represented by the stored `Offsets`.
    #[inline]
    fn into_par_sizes(self) -> Self::ParIter {
        ParSizes { offsets: self }
    }
}
