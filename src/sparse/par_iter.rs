use super::*;
use rayon::iter::plumbing::*;
use rayon::prelude::*;

/// A parallel sparse iterator.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SparseParIter<I, S> {
    indices: I,
    source: S,
}

impl<I, S> ParallelIterator for SparseParIter<I, S>
where
    S: Send + SplitAt + SplitFirst + Dummy + Set,
    S::First: Send,
    I: Send + IndexedParallelIterator + Producer<Item = usize>,
    I::IntoIter: ExactSizeIterator<Item = usize>,
{
    type Item = (usize, S::First);

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

impl<I, S> IndexedParallelIterator for SparseParIter<I, S>
where
    S: Send + SplitAt + SplitFirst + Dummy + Set,
    S::First: Send,
    I: Send + IndexedParallelIterator + Producer<Item = usize>,
    I::IntoIter: ExactSizeIterator<Item = usize>,
{
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(SparseProducer {
            indices: self.indices,
            source: self.source,
        })
    }
}

struct SparseProducer<I, S> {
    indices: I,
    source: S,
}

impl<I, S> Producer for SparseProducer<I, S>
where
    S: Send + SplitAt + SplitFirst + Dummy + Set,
    S::First: Send,
    I: Send + IndexedParallelIterator + Producer<Item = usize>,
    I::IntoIter: ExactSizeIterator<Item = usize>,
{
    type Item = (usize, S::First);
    type IntoIter = SparseIter<I::IntoIter, S>;

    fn into_iter(self) -> Self::IntoIter {
        SparseIter {
            indices: self.indices.into_iter(),
            source: self.source,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (li, ri) = self.indices.split_at(index);
        let (ls, rs) = self.source.split_at(index);
        (
            SparseProducer {
                indices: li,
                source: ls,
            },
            SparseProducer {
                indices: ri,
                source: rs,
            },
        )
    }
}

impl<'a, S, T, I> Sparse<S, T, I>
where
    S: View<'a>,
    <S as View<'a>>::Type: Set + IntoParallelIterator,
    <<S as View<'a>>::Type as IntoParallelIterator>::Iter: IndexedParallelIterator,
    T: Set + Get<'a, usize> + View<'a> + Sync,
    T::Output: Send,
    I: AsIndexSlice + Sync,
{
    /// Produce a parallel iterator over elements (borrowed slices) of a `Sparse`.
    #[inline]
    pub fn par_iter(
        &'a self,
    ) -> impl IndexedParallelIterator<
        Item = (
            usize,
            <<S as View<'a>>::Type as IntoParallelIterator>::Item,
            <T as Get<'a, usize>>::Output,
        ),
    > {
        self.selection
            .par_iter()
            .zip(self.source.view().into_par_iter())
            .map(|((i, t), s)| (i, s, t))
    }
}

impl<'a, S, T, I> Sparse<S, T, I>
where
    S: ViewMut<'a>,
    <S as ViewMut<'a>>::Type: Set + IntoParallelIterator,
    <<S as ViewMut<'a>>::Type as IntoParallelIterator>::Iter: IndexedParallelIterator,
    I: AsMut<[usize]>,
{
    /// Produce a parallel iterator over elements (borrowed slices) of a `Sparse`.
    #[inline]
    pub fn par_iter_mut(
        &'a mut self,
    ) -> rayon::iter::Zip<
        rayon::slice::IterMut<'a, usize>,
        <<S as ViewMut<'a>>::Type as IntoParallelIterator>::Iter,
    > {
        self.selection
            .index_par_iter_mut()
            .zip(self.source.view_mut().into_par_iter())
    }
}

impl<S, T, I> IntoParallelIterator for Sparse<S, T, I>
where
    S: Send + SplitAt + SplitFirst + Set + Dummy,
    S::First: Send,
    I: Send + IndexedParallelIterator + Producer<Item = usize>,
    I::IntoIter: ExactSizeIterator<Item = usize>,
{
    type Item = (usize, S::First);
    type Iter = SparseParIter<I, S>;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        SparseParIter {
            indices: self.selection.indices,
            source: self.source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_par() {
        let values = vec![5.0, 6.0, 7.0, 8.0, 9.0];
        let mut sparse = Sparse::from_dim(vec![0, 1, 2, 0, 1], 3, values.clone());
        let mut view_mut = sparse.view_mut();
        view_mut.par_iter_mut().for_each(|(_, a)| {
            *a += 1.0;
        });

        sparse
            .view()
            .par_iter()
            .zip(values.into_par_iter())
            .for_each(|((_, &a, _), orig)| {
                assert_eq!(a, orig + 1.0);
            });
    }
}
