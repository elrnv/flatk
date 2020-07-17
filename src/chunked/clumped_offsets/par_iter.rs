use super::*;
use rayon::iter::plumbing::*;
use rayon::prelude::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParUnclumpedOffsetValuesAndSizes<'a> {
    /// The offset from the first chunk offset of the first clump.
    front_clump_off: usize,
    front_stride: usize,
    /// The offset from the last chunk offset of the last clump.
    back_clump_off: usize,
    back_stride: usize,
    clumped_offsets: ClumpedOffsets<&'a [usize]>,
}

// SAFETY: ParUnclumpedOffsetValuesAndSizes will never consume the last offset.
unsafe impl GetOffset for ParUnclumpedOffsetValuesAndSizes<'_> {
    unsafe fn offset_value_unchecked(&self, index: usize) -> usize {
        self.clumped_offsets
            .offset_value_unchecked(index + self.front_clump_off)
    }

    fn num_offsets(&self) -> usize {
        self.clumped_offsets.num_offsets() - self.front_clump_off - self.back_clump_off
    }
}

impl<'o> ParallelIterator for ParUnclumpedOffsetValuesAndSizes<'o> {
    type Item = (usize, usize);

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

impl<'o> IndexedParallelIterator for ParUnclumpedOffsetValuesAndSizes<'o> {
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.num_offsets() - 1
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(self)
    }
}

impl<'o> Producer for ParUnclumpedOffsetValuesAndSizes<'o> {
    type Item = (usize, usize);
    type IntoIter = DEUnclumpedOffsetValuesAndSizes<'o>;

    fn into_iter(self) -> Self::IntoIter {
        DEUnclumpedOffsetValuesAndSizes {
            front_stride: self.front_stride,
            front: self.clumped_offsets.first_offset_value()
                + self.front_clump_off * self.front_stride,
            back_stride: self.back_stride,
            back: self.clumped_offsets.last_offset_value() - self.back_clump_off * self.back_stride,
            clumped_offsets: self.clumped_offsets,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        // Pad the index with preceeding elements before we use it to split self.clumped_offsets.
        let (left, right, chunk_off) = self
            .clumped_offsets
            .split_clumped_offsets_at(index + self.front_clump_off);
        let stride = right.clump_stride(0);
        (
            ParUnclumpedOffsetValuesAndSizes {
                back_clump_off: left.last_chunk_offset_value() - chunk_off,
                back_stride: stride,
                clumped_offsets: left,
                ..self
            },
            ParUnclumpedOffsetValuesAndSizes {
                front_clump_off: chunk_off - right.first_chunk_offset_value(),
                front_stride: stride,
                clumped_offsets: right,
                ..self
            },
        )
    }
}

/// Double ended version of the UnclumpedOffsetValuesAndSizes iterator.
#[derive(Copy, Clone)]
pub struct DEUnclumpedOffsetValuesAndSizes<'a> {
    front_stride: usize,
    front: usize,
    back_stride: usize,
    back: usize,
    clumped_offsets: ClumpedOffsets<&'a [usize]>,
}

// SAFETY: DEUnclumpedOffsetValuesAndSizes will never consume the last offset.
unsafe impl GetOffset for DEUnclumpedOffsetValuesAndSizes<'_> {
    #[inline]
    unsafe fn offset_value_unchecked(&self, mut i: usize) -> usize {
        let first_offset_value = self.clumped_offsets.first_offset_value();
        if self.front > first_offset_value {
            debug_assert!(self.front_stride > 0);
            i += (self.front - first_offset_value) / self.front_stride;
        }
        self.clumped_offsets.offset_value_unchecked(i)
    }

    #[inline]
    fn num_offsets(&self) -> usize {
        let mut n = self.clumped_offsets.num_offsets();
        let first_offset_value = self.clumped_offsets.first_offset_value();
        if self.front > first_offset_value {
            debug_assert!(self.front_stride > 0);
            n -= (self.front - first_offset_value) / self.front_stride;
        }
        let last_offset_value = self.clumped_offsets.last_offset_value();
        if self.back < last_offset_value {
            debug_assert!(self.back_stride > 0);
            n -= (last_offset_value - self.back) / self.back_stride;
        }
        n
    }
}

impl<'a> Iterator for DEUnclumpedOffsetValuesAndSizes<'a> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let DEUnclumpedOffsetValuesAndSizes {
            front_stride,
            front,
            back,
            clumped_offsets: co,
            ..
        } = self;

        // Check against the real last offset.
        if *front == *back {
            // We have reached the end.
            return None;
        }

        if co.offsets.num_offsets() < 2 {
            return None;
        }

        // SAFETY: indices at 1 are valid since there are at least two offsets (n >= 2)

        let offset = co.offsets.first_offset_value();
        let next_offset = unsafe { co.offsets.offset_value_unchecked(1) };

        if *front == offset {
            let chunk_offset = co.chunk_offsets.first_offset_value();
            let next_chunk_offset = unsafe { co.chunk_offsets.offset_value_unchecked(1) };

            // Recompute the stride.
            let clump_dist = next_offset - offset;
            let chunk_size = next_chunk_offset - chunk_offset;
            *front_stride = clump_dist / chunk_size;
        }

        let off = *front;

        // Incrementing the head offset effectively pops the unclumped offset.
        *front += *front_stride;

        if *front == next_offset {
            // SAFETY: We know there are at least two elements in both offsets and chunk_offsets.
            unsafe {
                // Pop the last internal offset, which also changes the stride.
                co.offsets.remove_prefix_unchecked(1);
                co.chunk_offsets.remove_prefix_unchecked(1);
            }
        }

        // The current stride is equal to the next effective unclumped size
        Some((off, *front_stride))
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
    fn next_back(&mut self) -> Option<Self::Item> {
        let DEUnclumpedOffsetValuesAndSizes {
            front,
            back_stride,
            back,
            clumped_offsets: co,
            ..
        } = self;

        if *back == *front {
            return None;
        }

        let n = co.offsets.num_offsets();

        if n < 2 {
            return None;
        }

        // SAFETY: indices at n-2 are valid since there are at least two offsets (n >= 2)

        let offset = co.offsets.last_offset_value();
        let next_offset = unsafe { co.offsets.offset_value_unchecked(n - 2) };

        if *back == offset {
            let chunk_offset = co.chunk_offsets.last_offset_value();
            let next_chunk_offset = unsafe { co.chunk_offsets.offset_value_unchecked(n - 2) };

            // Recompute the stride.
            let clump_dist = offset - next_offset;
            let chunk_size = chunk_offset - next_chunk_offset;
            *back_stride = clump_dist / chunk_size;
        }

        // Incrementing the head offset effectively pops the unclumped offset.
        *back -= *back_stride;

        if *back == next_offset {
            // SAFETY: We know there are at least two elements in both offsets and chunk_offsets.
            unsafe {
                // Pop the last internal offset, which also changes the stride.
                co.offsets.remove_suffix_unchecked(1);
                co.chunk_offsets.remove_suffix_unchecked(1);
            }
        }

        // The current stride is equal to the next effective unclumped size
        Some((*back, *back_stride))
    }

    // TODO: Implement this
    //#[inline]
    //fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
    //    self.clumped_offsets.
    //}
}

impl ExactSizeIterator for DEUnclumpedOffsetValuesAndSizes<'_> {}
impl std::iter::FusedIterator for DEUnclumpedOffsetValuesAndSizes<'_> {}

/// Double ended version of the UnclumpedOffsetsAndSizes iterator.
#[derive(Copy, Clone)]
pub struct DEUnclumpedOffsetsAndSizes<'a> {
    first_offset_value: usize,
    iter: DEUnclumpedOffsetValuesAndSizes<'a>,
}

impl<'a> From<DEUnclumpedOffsetValuesAndSizes<'a>> for DEUnclumpedOffsetsAndSizes<'a> {
    fn from(offset_values_and_sizes: DEUnclumpedOffsetValuesAndSizes<'a>) -> Self {
        DEUnclumpedOffsetsAndSizes {
            first_offset_value: offset_values_and_sizes.front,
            iter: offset_values_and_sizes,
        }
    }
}

impl DEUnclumpedOffsetsAndSizes<'_> {
    #[inline]
    fn mapper<'b>(&'b self) -> impl Fn((usize, usize)) -> (usize, usize) + 'b {
        move |(off, size)| (off - self.first_offset_value, size)
    }
}

// SAFETY: DEUnclumpedOffsetsAndSizes will never consume the last offset.
unsafe impl GetOffset for DEUnclumpedOffsetsAndSizes<'_> {
    #[inline]
    unsafe fn offset_value_unchecked(&self, i: usize) -> usize {
        self.iter.offset_value_unchecked(i)
    }

    #[inline]
    fn num_offsets(&self) -> usize {
        self.iter.num_offsets()
    }
}

impl<'a> Iterator for DEUnclumpedOffsetsAndSizes<'a> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(self.mapper())
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
        self.iter.nth(n).map(self.mapper())
    }
}

impl<'a> DoubleEndedIterator for DEUnclumpedOffsetsAndSizes<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(self.mapper())
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth_back(n).map(self.mapper())
    }
}

impl ExactSizeIterator for DEUnclumpedOffsetsAndSizes<'_> {}
impl std::iter::FusedIterator for DEUnclumpedOffsetsAndSizes<'_> {}

impl<'a> IntoParOffsetValuesAndSizes for ClumpedOffsets<&'a [usize]> {
    type ParIter = ParUnclumpedOffsetValuesAndSizes<'a>;
    /// Returns a parallel iterator over chunk offsets and sizes represented by the stored `Offsets`.
    #[inline]
    fn into_par_offset_values_and_sizes(self) -> Self::ParIter {
        ParUnclumpedOffsetValuesAndSizes {
            front_clump_off: 0,
            front_stride: 0,
            back_clump_off: 0,
            back_stride: 0,
            clumped_offsets: self,
        }
    }
}

/// Double ended version of the UnclumpedSizes iterator.
#[derive(Copy, Clone)]
pub struct DEUnclumpedSizes<'a> {
    iter: DEUnclumpedOffsetValuesAndSizes<'a>,
}

impl DEUnclumpedSizes<'_> {
    #[inline]
    fn mapper<'b>(&'b self) -> impl Fn((usize, usize)) -> usize + 'b {
        move |(_, size)| size
    }
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
        self.iter.next().map(self.mapper())
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
        self.iter.nth(n).map(self.mapper())
    }
}

impl<'a> DoubleEndedIterator for DEUnclumpedSizes<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(self.mapper())
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth_back(n).map(self.mapper())
    }
}

impl ExactSizeIterator for DEUnclumpedSizes<'_> {}
impl std::iter::FusedIterator for DEUnclumpedSizes<'_> {}

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
                front_stride: 0,
                front: self.offsets.first_offset_value(),
                back_stride: 0,
                back: self.offsets.last_offset_value(),
                clumped_offsets: self.view(),
            },
        }
    }

    /// Parallel version of `offsets_and_sizes`.
    #[inline]
    pub fn par_offset_values_and_sizes(&self) -> ParUnclumpedOffsetValuesAndSizes {
        debug_assert!(self.chunk_offsets.num_offsets() > 0);
        ParUnclumpedOffsetValuesAndSizes {
            front_clump_off: 0,
            front_stride: 0,
            back_clump_off: 0,
            back_stride: 0,
            clumped_offsets: self.view(),
        }
    }
}

impl<'a> ClumpedOffsets<&'a [usize]> {
    /// Same as `split_offsets_with_intersection_at`, but this function allows the output
    /// `ClumpedOffsets` to be overlapping at the clump.
    ///
    /// This allows us to split the clumped offsets in thet middle of the clump.
    #[inline]
    fn split_clumped_offsets_at(
        self,
        mid: usize,
    ) -> (
        ClumpedOffsets<&'a [usize]>,
        ClumpedOffsets<&'a [usize]>,
        usize,
    ) {
        assert!(mid < self.num_offsets());
        // SAFETY: mid is checked above.
        let ((chunk_off, _), mid_idx, on_bdry) = unsafe { self.get_chunk_at_unchecked(mid) };
        let ((los, ros), (lcos, rcos)) = if on_bdry {
            (
                self.offsets.split_offsets_at(mid_idx),
                self.chunk_offsets.split_offsets_at(mid_idx),
            )
        } else {
            (
                self.offsets.separate_offsets_with_overlap(mid_idx),
                self.chunk_offsets.separate_offsets_with_overlap(mid_idx),
            )
        };
        (
            ClumpedOffsets {
                chunk_offsets: lcos,
                offsets: los,
            },
            ClumpedOffsets {
                chunk_offsets: rcos,
                offsets: ros,
            },
            chunk_off,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_clumped_offsets() {
        let offsets = Offsets::new(vec![0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39]);
        let clumped_offsets = ClumpedOffsets::from(offsets.clone());
        // Single splits
        for i in 0..(offsets.num_offsets() - 1) {
            let (l, r) = clumped_offsets.par_offset_values_and_sizes().split_at(i);
            let (lo, ro) = offsets.view().split_offsets_at(i);
            for (l_clumped, l_o) in DEUnclumpedOffsetsAndSizes::from(l.into_iter()).zip(lo.iter()) {
                assert_eq!(l_clumped.0, l_o);
            }
            for (r_clumped, r_o) in DEUnclumpedOffsetsAndSizes::from(r.into_iter()).zip(ro.iter()) {
                assert_eq!(r_clumped.0, r_o);
            }
        }

        // Test multilevel splits
        for i in 0..(offsets.num_offsets() - 1) {
            let (l, r) = clumped_offsets.par_offset_values_and_sizes().split_at(i);
            let (lo, ro) = offsets.view().split_offsets_at(i);
            for j in 0..i {
                let (ll, lr) = l.split_at(j);
                let (llo, lro) = lo.split_offsets_at(j);
                for (l_clumped, l_o) in
                    DEUnclumpedOffsetsAndSizes::from(ll.into_iter()).zip(llo.iter())
                {
                    assert_eq!(l_clumped.0, l_o);
                }
                for (l_clumped, l_o) in
                    DEUnclumpedOffsetsAndSizes::from(lr.into_iter()).zip(lro.iter())
                {
                    assert_eq!(l_clumped.0, l_o);
                }
            }
            for j in 0..(offsets.num_offsets() - 1 - i) {
                let (rl, rr) = r.split_at(j);
                let (rlo, rro) = ro.split_offsets_at(j);
                for (r_clumped, r_o) in
                    DEUnclumpedOffsetsAndSizes::from(rl.into_iter()).zip(rlo.iter())
                {
                    assert_eq!(r_clumped.0, r_o);
                }
                for (r_clumped, r_o) in
                    DEUnclumpedOffsetsAndSizes::from(rr.into_iter()).zip(rro.iter())
                {
                    assert_eq!(r_clumped.0, r_o);
                }
            }
        }
    }

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

        let mut iter = clumped_offsets.double_ended_sizes();
        assert_eq!(iter.next().unwrap(), 3); // Start with next
        assert_eq!(iter.nth(3).unwrap(), 4); // First in clump
        assert_eq!(iter.next().unwrap(), 4);
        assert_eq!(iter.nth(0).unwrap(), 4); // Last in clump
        assert_eq!(iter.next().unwrap(), 3);
        assert_eq!(iter.nth(3).unwrap(), 3); // Last in all
        assert_eq!(iter.next(), None);
        assert_eq!(iter.nth(0), None);

        let mut iter = clumped_offsets.double_ended_sizes();
        assert_eq!(iter.nth(3).unwrap(), 3); // Start with nth last in clump
        assert_eq!(iter.next().unwrap(), 4);
        assert_eq!(iter.nth(2).unwrap(), 3); // First in clump
        assert_eq!(iter.next().unwrap(), 3);
        assert_eq!(iter.nth(1).unwrap(), 3); // Middle of clump
        assert_eq!(iter.next().unwrap(), 3);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.nth(0), None);

        // Check that double ended sizes works as a double ended iterator.

        let mut iter = clumped_offsets.double_ended_sizes();
        assert_eq!(iter.next().unwrap(), 3);
        assert_eq!(iter.nth(2).unwrap(), 3); // Last in clump
        assert_eq!(iter.next().unwrap(), 4); // First in clump

        assert_eq!(iter.next_back().unwrap(), 3); // First back
        assert_eq!(iter.nth_back(2).unwrap(), 3); // Last of clump back
        assert_eq!(iter.nth_back(1).unwrap(), 4); // First of clump back
        assert_eq!(iter.nth(0).unwrap(), 4);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn par_iter() {
        // Test that the parallel iterator returns the same offsets as the non-parallel one.
        let offsets = Offsets::new(vec![0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39]);
        let sizes: Vec<usize> = offsets.sizes().collect();
        let clumped_offsets = ClumpedOffsets::from(offsets.clone());

        assert_eq!(clumped_offsets.par_offset_values_and_sizes().len(), 12);
        assert_eq!(clumped_offsets.par_offset_values_and_sizes().count(), 12);
        clumped_offsets
            .par_offset_values_and_sizes()
            .enumerate()
            .for_each(|(i, (off, size))| {
                assert_eq!(offsets[i], off);
                assert_eq!(sizes[i], size);
            });
    }
}
