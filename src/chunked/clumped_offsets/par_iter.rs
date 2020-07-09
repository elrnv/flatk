use super::*;
use rayon::iter::plumbing::*;
use rayon::prelude::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParUnclumpedSizes<'a> {
    clumped_offsets: ClumpedOffsets<&'a [usize]>,
}

impl<'o> ParallelIterator for ParUnclumpedSizes<'o> {
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

impl<'o> IndexedParallelIterator for ParUnclumpedSizes<'o> {
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.clumped_offsets.num_offsets() - 1
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(self)
    }
}

impl<'o> Producer for ParUnclumpedSizes<'o> {
    type Item = usize;
    type IntoIter = DoubleEndedUnclumpedSizes<'o>;

    fn into_iter(self) -> Self::IntoIter {
        DoubleEndedUnclumpedSizes {
            stride: 0,
            cur: self.clumped_offsets.first_offset_value(),
            rev_stride: 0,
            rev_cur: self.clumped_offsets.last_offset_value(),
            clumped_offsets: self.clumped_offsets,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.clumped_offsets.split_offsets_at(index);
        (
            ParUnclumpedSizes {
                clumped_offsets: left,
            },
            ParUnclumpedSizes {
                clumped_offsets: right,
            },
        )
    }
}

/// Double ended version of the UnclumpedOffsetValuesAndSizes iterator.
#[derive(Copy, Clone)]
pub struct DEUnclumpedOffsetValuesAndSizes<'a> {
    stride: usize,
    cur: usize,
    rev_stride: usize,
    rev_cur: usize,
    clumped_offsets: ClumpedOffsets<&'a [usize]>,
}

// SAFETY: DEUnclumpedOffsetValuesAndSizes will never consume the last offset.
unsafe impl GetOffset for DEUnclumpedOffsetValuesAndSizes<'_> {
    #[inline]
    unsafe fn offset_value_unchecked(&self, i: usize) -> usize {
        self.clumped_offsets.offset_value_unchecked(i)
    }

    #[inline]
    fn num_offsets(&self) -> usize {
        self.clumped_offsets.num_offsets()
    }
}

impl<'a> Iterator for DEUnclumpedOffsetValuesAndSizes<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let DEUnclumpedOffsetValuesAndSizes {
            stride,
            cur,
            rev_stride,
            rev_cur,
            clumped_offsets: co,
        } = self;

        let n = co.offsets.num_offsets();
        // offsets should never be completely consumed.
        debug_assert!(n > 0);

        let offset = co.offsets.first_offset_value();

        if *cur == offset {
            // We have reached the end.
            if n < 2 {
                return None;
            }

            let chunk_offset = co.chunk_offsets.first_offset_value();

            // SAFETY: We know there are at least two elements in both offsets and chunk_offsets.
            unsafe {
                // Pop the last internal offset, which also changes the stride.
                co.offsets.remove_prefix_unchecked(1);
                co.chunk_offsets.remove_prefix_unchecked(1);
            }

            let next_offset = co.offsets.first_offset_value();
            let next_chunk_offset = co.chunk_offsets.first_offset_value();

            // Recompute the stride.
            let clump_dist = next_offset - offset;
            let chunk_size = next_chunk_offset - chunk_offset;
            *stride = clump_dist / chunk_size;
        }

        let off = *cur;

        // Check against the real last offset.
        if *cur == *rev_cur {}

        // Incrementing the head offset effectively pops the unclumped offset.
        *cur += *stride;

        // The current stride is equal to the next effective unclumped size
        Some((off, *stride))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.clumped_offsets.num_offsets() - 1;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.size_hint().0
    }

    // TODO: Implement this
    //#[inline]
    //fn nth(&mut self, n: usize) -> Option<Self::Item> {
    //    self.clumped_offsets.
    //}
}

impl<'a> DoubleEndedIterator for DEUnclumpedOffsetValuesAndSizes<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        let DEUnclumpedOffsetValuesAndSizes {
            stride,
            cur,
            rev_stride,
            rev_cur,
            clumped_offsets: co,
        } = self;

        let n = co.offsets.num_offsets();
        // offsets should never be completely consumed.
        debug_assert!(n > 0);

        let offset = co.offsets.first_offset_value();

        if *cur == offset {
            // We reached the end.
            if n < 2 {
                return None;
            }

            let chunk_offset = co.chunk_offsets.first_offset_value();

            // SAFETY: We know there are at least two elements in both offsets and chunk_offsets.
            unsafe {
                // Pop the last internal offset, which also changes the stride.
                co.offsets.remove_prefix_unchecked(1);
                co.chunk_offsets.remove_prefix_unchecked(1);
            }

            let next_offset = co.offsets.first_offset_value();
            let next_chunk_offset = co.chunk_offsets.first_offset_value();

            // Recompute the stride.
            let clump_dist = next_offset - offset;
            let chunk_size = next_chunk_offset - chunk_offset;
            *stride = clump_dist / chunk_size;
        }

        let off = *cur;

        // Incrementing the head offset effectively pops the unclumped offset.
        *cur += *stride;

        // The current stride is equal to the next effective unclumped size
        Some((off, *stride))
    }

    // TODO: Implement this
    //#[inline]
    //fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
    //    self.clumped_offsets.
    //}
}

impl ExactSizeIterator for DEUnclumpedOffsetValuesAndSizes<'_> {}
impl std::iter::FusedIterator for DEUnclumpedOffsetValuesAndSizes<'_> {}

/// Double ended version of the UnclumpedSizes iterator.
#[derive(Copy, Clone)]
pub struct DEUnclumpedSizes<'a> {
    iter: DEUnclumpedOffsetValuesAndSizes<'a>,
}

// SAFETY: DEUnclumpedSizes will never consume the last offset.
unsafe impl GetOffset for DEUnclumpedSizes<'_> {
    #[inline]
    unsafe fn offset_value_unchecked(&self, i: usize) -> usize {
        self.iter.offset_value_unchecked(i)
    }

    #[inline]
    fn num_offsets(&self) -> usize {
        self.iter.num_offsets()
    }
}

impl<'a> Iterator for DEUnclumpedSizes<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        self.iter.next().map(|(_, size)| size)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n).map(|(_, size)| size)
    }
}

impl<'a> DEUnclumpedSizes<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        self.iter.next_back().map(|(_, size)| size)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth_back(n).map(|(_, size)| size)
    }
}

impl ExactSizeIterator for DEUnclumpedSizes<'_> {}
impl std::iter::FusedIterator for DEUnclumpedSizes<'_> {}

impl<'a> IntoParSizes for ClumpedOffsets<&'a [usize]> {
    type ParIter = ParUnclumpedSizes<'a>;
    /// Returns a parallel iterator over chunk sizes represented by the stored `Offsets`.
    #[inline]
    fn into_par_sizes(self) -> Self::ParIter {
        ParUnclumpedSizes {
            clumped_offsets: self,
        }
    }
}

impl<O: AsRef<[usize]>> ClumpedOffsets<O> {
    /// An iterator over effective (unclumped) sizes that implements `DoubleEndedIterator`.
    ///
    /// # Notes
    ///
    /// This iterator is used to implement the parallel sizes iterator to enable parallel iteration
    /// of `Chunked` types with clumped offsets. However, this iterator can also be used as is for
    /// efficient backwards iteration.
    #[inline]
    pub fn double_ended_sizes(&self) -> par_iter::DEUnclumpedSizes {
        debug_assert!(self.chunk_offsets.num_offsets() > 0);
        DEUnclumpedSizes {
            iter: DEUnclumpedOffsetValuesAndSizes {
                stride: 0,
                cur: self.chunk_offsets.first_offset_value(),
                rev_stride: 0,
                rev_cur: self.chunk_offsets.last_offset_value(),
                clumped_offsets: self.view(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn double_ended_sizes() {
        use ExactSizeIterator;
        let offsets = Offsets::new(vec![0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39]);
        let clumped_offsets = ClumpedOffsets::from(offsets.clone());
        let mut iter = clumped_offsets.double_ended_sizes();
        let iter_len = iter.len();

        // First check that the double ended sizes works as a normal iterator
        assert_eq!(iter_len, offsets.num_offsets() - 1);
        assert_eq!(iter_len, iter.count());

        for _ in 0..4 {
            assert_eq!(iter.next().unwrap(), 3);
        }
        for _ in 0..3 {
            assert_eq!(iter.next().unwrap(), 4);
        }
        for _ in 0..5 {
            assert_eq!(iter.next().unwrap(), 3);
        }

        for i in 0..4 {
            assert_eq!(clumped_offsets.double_ended_sizes().nth(i).unwrap(), 3);
        }
        for i in 4..7 {
            assert_eq!(clumped_offsets.double_ended_sizes().nth(i).unwrap(), 4);
        }
        for i in 7..12 {
            assert_eq!(clumped_offsets.double_ended_sizes().nth(i).unwrap(), 3);
        }

        // The following checks that count is correctly implemented for the sizes iterator.
        let mut count = 0;
        for _ in clumped_offsets.double_ended_sizes() {
            count += 1;
        }
        assert_eq!(iter_len, count);

        // Check that using next and nth works together as expected

        let mut iter = clumped_offsets.sizes();
        assert_eq!(iter.next().unwrap(), 3); // Start with next
        assert_eq!(iter.nth(3).unwrap(), 4); // First in clump
        assert_eq!(iter.next().unwrap(), 4);
        assert_eq!(iter.nth(0).unwrap(), 4); // Last in clump
        assert_eq!(iter.next().unwrap(), 3);
        assert_eq!(iter.nth(3).unwrap(), 3); // Last in all
        assert_eq!(iter.next(), None);
        assert_eq!(iter.nth(0), None);

        let mut iter = clumped_offsets.sizes();
        assert_eq!(iter.nth(3).unwrap(), 3); // Start with nth last in clump
        assert_eq!(iter.next().unwrap(), 4);
        assert_eq!(iter.nth(2).unwrap(), 3); // First in clump
        assert_eq!(iter.next().unwrap(), 3);
        assert_eq!(iter.nth(1).unwrap(), 3); // Middle of clump
        assert_eq!(iter.next().unwrap(), 3);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.nth(0), None);

        // Check that double ended sizes works as a double ended iterator.

        let mut iter = clumped_offsets.sizes();
        assert_eq!(iter.next().unwrap(), 3);
        assert_eq!(iter.nth(2).unwrap(), 3); // Last in clump
        assert_eq!(iter.next().unwrap(), 4); // First in clump

        assert_eq!(iter.next_back().unwrap(), 3); // First back
        assert_eq!(iter.nth_back(2).unwrap(), 3); // Last of clump back
        assert_eq!(iter.nth_back(0).unwrap(), 4); // First of clump back
        assert_eq!(iter.nth(1).unwrap(), 4);
        assert_eq!(iter.next(), None);

        // Test that the parallel iterator returns the same offsets as the non-parallel one.
        //clumped_offsets.
    }
}
