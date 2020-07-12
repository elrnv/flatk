use std::ops::Range;

use rayon::iter::plumbing::*;
use rayon::prelude::*;

use super::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParOffsetValueRanges<'a> {
    offsets: Offsets<&'a [usize]>,
}

impl<'o> ParallelIterator for ParOffsetValueRanges<'o> {
    type Item = Range<usize>;

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

impl<'o> IndexedParallelIterator for ParOffsetValueRanges<'o> {
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

impl<'o> Producer for ParOffsetValueRanges<'o> {
    type Item = Range<usize>;
    type IntoIter = OffsetValueRanges<'o>;

    fn into_iter(self) -> Self::IntoIter {
        self.offsets.into_offset_value_ranges()
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.offsets.split_offsets_at(index);
        (
            ParOffsetValueRanges { offsets: left },
            ParOffsetValueRanges { offsets: right },
        )
    }
}

impl<'a> Offsets<&'a [usize]> {
    /// Returns a parallel iterator over chunk offset value ranges represented by the stored
    /// `Offsets`.
    #[inline]
    fn into_par_offset_value_ranges(self) -> ParOffsetValueRanges<'a> {
        ParOffsetValueRanges { offsets: self }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParOffsetValuesAndSizes<'a> {
    offset_value_ranges: ParOffsetValueRanges<'a>,
}

impl<'o> ParallelIterator for ParOffsetValuesAndSizes<'o> {
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

impl<'o> IndexedParallelIterator for ParOffsetValuesAndSizes<'o> {
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.offset_value_ranges.len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(self)
    }
}

impl<'o> Producer for ParOffsetValuesAndSizes<'o> {
    type Item = (usize, usize);
    type IntoIter = OffsetValuesAndSizes<'o>;

    fn into_iter(self) -> Self::IntoIter {
        OffsetValuesAndSizes {
            offset_value_ranges: OffsetValueRanges {
                offsets: self.offset_value_ranges.offsets,
            },
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.offset_value_ranges.split_at(index);
        (
            ParOffsetValuesAndSizes {
                offset_value_ranges: left,
            },
            ParOffsetValuesAndSizes {
                offset_value_ranges: right,
            },
        )
    }
}

impl<'a> IntoParOffsetValuesAndSizes for Offsets<&'a [usize]> {
    type ParIter = ParOffsetValuesAndSizes<'a>;
    /// Returns a parallel iterator over chunk sizes represented by the stored `Offsets`.
    #[inline]
    fn into_par_offset_values_and_sizes(self) -> Self::ParIter {
        ParOffsetValuesAndSizes {
            offset_value_ranges: self.into_par_offset_value_ranges(),
        }
    }
}
