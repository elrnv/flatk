use super::*;
use rayon::iter::plumbing::*;
use rayon::prelude::*;

/// A parallel chunk iterator.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ChunkedParIter<I, S> {
    first_offset_value: usize,
    offset_values_and_sizes: I,
    data: S,
}

impl<I, S> ParallelIterator for ChunkedParIter<I, S>
where
    S: Send + SplitAt + Dummy + Set,
    I: Send + GetOffset + IndexedParallelIterator + Producer<Item = (usize, usize)>,
    I::IntoIter: ExactSizeIterator<Item = (usize, usize)> + GetOffset,
{
    type Item = S;

    #[inline]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    #[inline]
    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<I, S> IndexedParallelIterator for ChunkedParIter<I, S>
where
    S: Send + SplitAt + Dummy + Set,
    I: Send + GetOffset + IndexedParallelIterator + Producer<Item = (usize, usize)>,
    I::IntoIter: ExactSizeIterator<Item = (usize, usize)> + GetOffset,
{
    #[inline]
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    #[inline]
    fn len(&self) -> usize {
        self.offset_values_and_sizes.len()
    }

    #[inline]
    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(ChunkedProducer {
            first_offset_value: self.first_offset_value,
            offset_values_and_sizes_producer: self.offset_values_and_sizes,
            data: self.data,
        })
    }
}

struct ChunkedProducer<I, S> {
    first_offset_value: usize,
    offset_values_and_sizes_producer: I,
    data: S,
}

impl<I, S> Producer for ChunkedProducer<I, S>
where
    S: Send + SplitAt + Dummy + Set,
    I: Send + GetOffset + Producer<Item = (usize, usize)>,
    I::IntoIter: ExactSizeIterator<Item = (usize, usize)> + GetOffset,
{
    type Item = S;
    type IntoIter = ChunkedIter<I::IntoIter, S>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ChunkedIter {
            first_offset_value: self.first_offset_value,
            offset_values_and_sizes: self.offset_values_and_sizes_producer.into_iter(),
            data: self.data,
        }
    }

    #[inline]
    fn split_at(self, index: usize) -> (Self, Self) {
        let off = self.offset_values_and_sizes_producer.offset_value(index);
        let (ls, rs) = self.offset_values_and_sizes_producer.split_at(index);
        let (l, r) = self.data.split_at(off - self.first_offset_value);
        (
            ChunkedProducer {
                first_offset_value: self.first_offset_value,
                offset_values_and_sizes_producer: ls,
                data: l,
            },
            ChunkedProducer {
                first_offset_value: off,
                offset_values_and_sizes_producer: rs,
                data: r,
            },
        )
    }
}

impl<'a, S, O> Chunked<S, O>
where
    S: View<'a>,
    O: View<'a>,
    O::Type: IntoParOffsetValuesAndSizes + GetOffset,
{
    /// Produce a parallel iterator over elements (borrowed slices) of a `Chunked`.
    #[inline]
    pub fn par_iter(
        &'a self,
    ) -> ChunkedParIter<
        <<O as View<'a>>::Type as IntoParOffsetValuesAndSizes>::ParIter,
        <S as View<'a>>::Type,
    > {
        ChunkedParIter {
            first_offset_value: self.chunks.view().first_offset_value(),
            offset_values_and_sizes: self.chunks.view().into_par_offset_values_and_sizes(),
            data: self.data.view(),
        }
    }
}

impl<'a, S, O> Chunked<S, O>
where
    S: ViewMut<'a>,
    O: View<'a>,
    O::Type: IntoParOffsetValuesAndSizes + GetOffset,
{
    /// Produce a parallel iterator over elements (borrowed slices) of a `Chunked`.
    #[inline]
    pub fn par_iter_mut(
        &'a mut self,
    ) -> ChunkedParIter<
        <<O as View<'a>>::Type as IntoParOffsetValuesAndSizes>::ParIter,
        <S as ViewMut<'a>>::Type,
    > {
        ChunkedParIter {
            first_offset_value: self.chunks.view().first_offset_value(),
            offset_values_and_sizes: self.chunks.view().into_par_offset_values_and_sizes(),
            data: self.data.view_mut(),
        }
    }
}

impl<S, O> IntoParallelIterator for Chunked<S, O>
where
    O: IntoParOffsetValuesAndSizes + GetOffset,
    S: Send + SplitAt + Set + Dummy,
    O::ParIter: Producer<Item = (usize, usize)> + GetOffset,
    <O::ParIter as Producer>::IntoIter: GetOffset,
{
    type Item = S;
    type Iter = ChunkedParIter<O::ParIter, S>;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        ChunkedParIter {
            first_offset_value: self.chunks.first_offset_value(),
            offset_values_and_sizes: self.chunks.into_par_offset_values_and_sizes(),
            data: self.data,
        }
    }
}

impl<S: IntoParChunkIterator, N: Dimension> IntoParallelIterator for UniChunked<S, N> {
    type Item = <S as IntoParChunkIterator>::Item;
    type Iter = <S as IntoParChunkIterator>::IterType;

    /// Convert a `UniChunked` collection into a parallel iterator over grouped elements.
    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        self.data.into_par_chunk_iter(self.chunk_size.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_par() {
        let vecs = vec![vec![0, 1, 2, 3, 4], vec![5, 6], vec![7, 8]];
        let mut chunked = Chunked::<Vec<_>>::from_nested_vec(vecs.clone());
        let mut view_mut = chunked.view_mut();
        view_mut.par_iter_mut().for_each(|a| {
            for x in a {
                *x += 1;
            }
        });

        chunked.view().par_iter().zip(vecs.par_iter()).for_each(
            |(a, b): (&[usize], &Vec<usize>)| {
                for (&x, y) in a.iter().zip(b.iter()) {
                    assert_eq!(x, y + 1);
                }
            },
        );
    }
}
