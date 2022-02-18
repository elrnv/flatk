//! This module defines the `ClumpedOffsets` type and its behaviour.

use super::*;
use std::convert::{AsMut, AsRef};
use std::ops::Range;

#[cfg(feature = "rayon")]
pub(crate) mod par_iter;

/// A collection of clumped offsets into another collection.
///
/// A version of `Offsets` that combines consecutive equidistant offsets.
///
/// # Example
///
/// To see the correspondence between `Offsets` and `ClumpedOffsets`, consider the following
/// example
///
/// ```
///     use flatk::{Offsets, ClumpedOffsets};
///
///     // with `Offsets` we might have offsets like
///     let off = Offsets::new(vec![0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39]);
///     // which can be represented in `ClumpedOffsets` as
///     let clumped_off = ClumpedOffsets::new(vec![0, 4, 7, 12], vec![0, 12, 24, 39]);
///     // Note that ClumpedOffsets would be more compact if the triplets could be combined,
///     // which would require reorganizing the data indexed by the offsets.
///
///     assert_eq!(ClumpedOffsets::from(off), clumped_off);
/// ```
///
/// `ClumpedOffsets` can be a lot more memory efficient when a chunked collection is mostly
/// uniformly chunked. However, in the worst case, when each consecutive stride is different, this
/// representation can be three times the size of standard `Offsets`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ClumpedOffsets<O = Vec<usize>> {
    pub chunk_offsets: Offsets<O>,
    pub offsets: Offsets<O>,
}

impl<O: AsRef<[usize]>> Set for ClumpedOffsets<O> {
    type Elem = usize;
    type Atom = usize;

    #[inline]
    fn len(&self) -> usize {
        self.num_offsets()
    }
}

impl<O: AsRef<[usize]>> ClumpedOffsets<O> {
    /// Get the offset value in the clumped offsets collection at the given index of the conceptual
    /// unclumped offset collection.
    ///
    /// This function returns the offset value followed by the index of the clump it belongs to in
    /// the clumped collection. The last boolean value indicates whether the offset is explicitly
    /// represented in the clumped offsets, i.e. it lies on the boundary of a clump.
    ///
    /// # Safety
    ///
    /// It is assumed that `index` is strictly less than `self.num_offsets()`.
    #[inline]
    unsafe fn offset_value_and_clump_index_unchecked(&self, index: usize) -> (usize, usize) {
        let ((_, off), clump_idx, _) = self.get_chunk_at_unchecked(index);
        (off, clump_idx)
    }

    /// Get the offset info in the clumped offsets collection located at the given
    /// index of the conceptual unclumped offset collection.
    ///
    /// This function returns the `(chunk_offset_value, offset_value)` pair followed by the index
    /// of the clump it belongs to in the clumped collection. The last boolean value indicates
    /// whether the offset is explicitly represented in the clumped offsets, i.e. it lies on the
    /// boundary of a clump.
    ///
    /// # Safety
    ///
    /// It is assumed that `index` is strictly less than `self.num_offsets()`.
    #[inline]
    unsafe fn get_chunk_at_unchecked(&self, index: usize) -> ((usize, usize), usize, bool) {
        let ClumpedOffsets {
            chunk_offsets,
            offsets,
        } = self;
        debug_assert!(chunk_offsets.num_offsets() > 0);
        debug_assert!(offsets.num_offsets() > 0);
        // The following is safe by construction of ClumpedOffsets.
        match chunk_offsets.binary_search(&index) {
            Ok(clump_idx) => (
                (
                    chunk_offsets.offset_value_unchecked(clump_idx),
                    offsets.offset_value_unchecked(clump_idx),
                ),
                clump_idx,
                true,
            ),
            Err(clump_idx) => {
                // Offset is in the middle of a clump.
                // Given that idx is not out of bounds as defined in the doc,
                // the following are safe because index >= 0.
                // so clump_idx >= 0.
                let begin_off = offsets.offset_value_unchecked(clump_idx - 1);
                let clump_dist = offsets.offset_value_unchecked(clump_idx) - begin_off;
                let begin_clump_off = chunk_offsets.offset_value_unchecked(clump_idx - 1);
                let clump_size = chunk_offsets.offset_value_unchecked(clump_idx) - begin_clump_off;
                let stride = clump_dist / clump_size;
                let chunk_offset = index + chunk_offsets.first_offset_value();
                let offset = begin_off + stride * (chunk_offset - begin_clump_off);
                ((chunk_offset, offset), clump_idx - 1, false)
            }
        }
    }

    #[inline]
    pub fn first_chunk_offset_value(&self) -> usize {
        self.chunk_offsets.first_offset_value()
    }
    #[inline]
    pub fn last_chunk_offset_value(&self) -> usize {
        self.chunk_offsets.last_offset_value()
    }

    /// Returns the number of clump offsets represented by `ClumpedOffsets`.
    ///
    /// This is typically significantly smaller than `self.num_offsets()`.
    #[inline]
    pub fn num_clump_offsets(&self) -> usize {
        self.offsets.num_offsets()
    }

    /// Returns the number of clumps represented by `ClumpedOffsets`.
    #[inline]
    pub fn num_clumps(&self) -> usize {
        self.offsets.num_offsets() - 1
    }

    /// Compute the stride of the clump at the given index.
    ///
    /// # Panics
    ///
    /// This function panics if `index` is greater than or equal to `self.num_clumps()`.
    #[inline]
    pub fn clump_stride(&self, index: usize) -> usize {
        assert!(index < self.num_clumps(), "Offset index out of bounds");

        // SAFETY: The length is checked above.
        unsafe {
            self.offsets.chunk_len_unchecked(index) / self.chunk_offsets.chunk_len_unchecked(index)
        }
    }
}

// SAFETY: offsets and chunk_offsets are guaranteed to store at least one offset at all times.
unsafe impl<O: AsRef<[usize]>> GetOffset for ClumpedOffsets<O> {
    /// A version of `offset_value` without bounds checking.
    ///
    /// # Safety
    ///
    /// It is assumed that `index` is strictly less than `self.num_offsets()`.
    #[inline]
    unsafe fn offset_value_unchecked(&self, index: usize) -> usize {
        self.offset_value_and_clump_index_unchecked(index).0
    }

    /// Get the total number of offsets.
    ///
    /// This is one more than the number of chunks represented.
    fn num_offsets(&self) -> usize {
        let offsets = self.chunk_offsets.as_ref();
        debug_assert!(!offsets.is_empty(), "Clump offsets are corrupted");
        unsafe { 1 + offsets.get_unchecked(offsets.len() - 1) - offsets.get_unchecked(0) }
    }
}

impl<O: AsRef<[usize]>> ClumpedOffsets<O> {
    /// An iterator over effective (unclumped) sizes producing an increment for advancing the
    /// data pointer.
    ///
    /// This helps for implementing iterators over `Clumped` types.
    ///
    /// # Example
    ///
    /// ```
    ///     use flatk::{Offsets, ClumpedOffsets};
    ///
    ///     let off = Offsets::new(vec![0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39]);
    ///     let clumped = ClumpedOffsets::from(off);
    ///     let mut iter = clumped.sizes();
    ///     for _ in 0..4 {
    ///         assert_eq!(iter.next(), Some(3));
    ///     }
    ///     for _ in 0..3 {
    ///         assert_eq!(iter.next(), Some(4));
    ///     }
    ///     for _ in 0..5 {
    ///         assert_eq!(iter.next(), Some(3));
    ///     }
    ///     assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    pub fn sizes(&self) -> UnclumpedSizes {
        debug_assert!(self.chunk_offsets.num_offsets() > 0);
        UnclumpedSizes {
            iter: UnclumpedOffsetValuesAndSizes {
                stride: 0,
                cur: self.offsets.first_offset_value(),
                clumped_offsets: self.view(),
            },
        }
    }

    /// An iterator over unclumped offsets.
    ///
    /// This is equivalent to iterating over `Offsets` after conversion, but it doesn't require any
    /// additional allocations.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(!self.offsets.as_ref().is_empty());
        let first = self.first_offset_value();
        UnclumpedOffsetValues {
            cur_stride: 0,
            cur_offset: self.offsets.first_offset_value(),
            chunk_offsets: self.chunk_offsets.as_ref(),
            offsets: self.offsets.as_ref(),
        }
        .map(move |x| x - first)
    }

    /// An iterator over unclumped offset values.
    ///
    /// This is equivalent to iterating over `Offsets` after conversion, but it doesn't require any
    /// additional allocations.
    #[inline]
    pub fn values(&self) -> UnclumpedOffsetValues {
        debug_assert!(!self.offsets.as_ref().is_empty());
        UnclumpedOffsetValues {
            cur_stride: 0,
            cur_offset: self.offsets.first_offset_value(),
            chunk_offsets: self.chunk_offsets.as_ref(),
            offsets: self.offsets.as_ref(),
        }
    }
}

/// Iterator over offset value and size pairs representing unclumped chunks.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct UnclumpedOffsetValuesAndSizes<'a> {
    /// Current chunk size.
    stride: usize,
    /// Current unclumped offset value.
    cur: usize,
    clumped_offsets: ClumpedOffsets<&'a [usize]>,
}

impl<'a> Iterator for UnclumpedOffsetValuesAndSizes<'a> {
    type Item = (usize, usize);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let UnclumpedOffsetValuesAndSizes {
            stride,
            cur,
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
                // Pop the last internal offset, which also potentially changes the stride.
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

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.clumped_offsets.num_offsets() - 1;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.size_hint().0
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let UnclumpedOffsetValuesAndSizes {
            stride,
            cur,
            clumped_offsets: co,
        } = self;

        let first_offset = co.offsets.first_offset_value();

        let adjusted_n = if *stride > 0 {
            let first_clump_count = (first_offset - *cur) / *stride;
            if first_clump_count > n {
                // The offset is in the first clump
                *cur += *stride * (n + 1);
                return Some((*cur, *stride));
            } else {
                n - first_clump_count
            }
        } else {
            n
        };

        if adjusted_n + 1 < co.num_offsets() {
            // Binary search for offset value.
            // SAFETY: This is safe since the bounds are checked above.
            let (cur_off, clump_idx) =
                unsafe { co.offset_value_and_clump_index_unchecked(adjusted_n) };

            // SAFETY: There are at least clump_idx+2 elements in offsets and chunk_offsets.
            let (offset, chunk_offset, next_offset, next_chunk_offset) = unsafe {
                (
                    co.offsets.offset_value_unchecked(clump_idx),
                    co.chunk_offsets.offset_value_unchecked(clump_idx),
                    co.offsets.offset_value_unchecked(clump_idx + 1),
                    co.chunk_offsets.offset_value_unchecked(clump_idx + 1),
                )
            };

            // Recompute the stride.
            let clump_dist = next_offset - offset;
            let chunk_size = next_chunk_offset - chunk_offset;
            *stride = clump_dist / chunk_size;

            // SAFETY: There are at least clump_idx+2 elements in offsets and chunk_offsets.
            // This is because offset_value_and_clump_index_unchecked will never return the last offset.
            unsafe {
                // Pop the last internal offset (if any).
                co.offsets.remove_prefix_unchecked(clump_idx + 1);
                co.chunk_offsets.remove_prefix_unchecked(clump_idx + 1);
            };

            // Incrementing the head offset effectively pops the unclumped offset.
            *cur = cur_off + *stride;

            Some((cur_off, *stride))
        } else {
            None
        }
    }
}

// TODO: Extend UnclumpedOffsetValuesAndSizes to be double ended or implement a reverse iterator.

/// Iterator over offsets and size pairs representing unclumped chunks.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct UnclumpedOffsetsAndSizes<'a> {
    first_offset_value: usize,
    iter: UnclumpedOffsetValuesAndSizes<'a>,
}

impl UnclumpedOffsetsAndSizes<'_> {
    #[inline]
    fn offset_and_size_mapper(&self) -> impl Fn((usize, usize)) -> (usize, usize) + '_ {
        move |(off, size)| (off - self.first_offset_value, size)
    }
}

impl<'a> Iterator for UnclumpedOffsetsAndSizes<'a> {
    type Item = (usize, usize);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(self.offset_and_size_mapper())
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
        self.iter.nth(n).map(self.offset_and_size_mapper())
    }
}

impl ExactSizeIterator for UnclumpedOffsetsAndSizes<'_> {}
impl std::iter::FusedIterator for UnclumpedOffsetsAndSizes<'_> {}

impl ExactSizeIterator for UnclumpedOffsetValuesAndSizes<'_> {}
impl std::iter::FusedIterator for UnclumpedOffsetValuesAndSizes<'_> {}

/// Iterator over unclumped chunk sizes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct UnclumpedSizes<'a> {
    iter: UnclumpedOffsetValuesAndSizes<'a>,
}

impl<'a> Iterator for UnclumpedSizes<'a> {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
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

impl ExactSizeIterator for UnclumpedSizes<'_> {}
impl std::iter::FusedIterator for UnclumpedSizes<'_> {}

impl<'a> IntoValues for ClumpedOffsets<&'a [usize]> {
    type Iter = UnclumpedOffsetValues<'a>;

    /// Returns an iterator over offset values represented by the stored `Offsets`.
    #[inline]
    fn into_values(self) -> UnclumpedOffsetValues<'a> {
        debug_assert!(self.chunk_offsets.num_offsets() > 0);
        UnclumpedOffsetValues {
            cur_stride: 0,
            cur_offset: self.offsets.first_offset_value(),
            chunk_offsets: self.chunk_offsets.into_inner(),
            offsets: self.offsets.into_inner(),
        }
    }
}

impl<'a> IntoSizes for ClumpedOffsets<&'a [usize]> {
    type Iter = UnclumpedSizes<'a>;

    /// Returns an iterator over chunk sizes represented by the stored `ClumpedOffsets`.
    #[inline]
    fn into_sizes(self) -> UnclumpedSizes<'a> {
        debug_assert!(self.chunk_offsets.num_offsets() > 0);
        UnclumpedSizes {
            iter: UnclumpedOffsetValuesAndSizes {
                stride: 0,
                cur: self.offsets.first_offset_value(),
                clumped_offsets: self,
            },
        }
    }
}

impl<'a> IntoOffsetValuesAndSizes for ClumpedOffsets<&'a [usize]> {
    type Iter = UnclumpedOffsetValuesAndSizes<'a>;

    #[inline]
    fn into_offset_values_and_sizes(self) -> UnclumpedOffsetValuesAndSizes<'a> {
        debug_assert!(self.chunk_offsets.num_offsets() > 0);
        UnclumpedOffsetValuesAndSizes {
            stride: 0,
            cur: self.offsets.first_offset_value(),
            clumped_offsets: self,
        }
    }
}

#[derive(Copy, Clone)]
pub struct UnclumpedOffsetValues<'a> {
    cur_stride: usize,
    cur_offset: usize,
    chunk_offsets: &'a [usize],
    offsets: &'a [usize],
}

impl<'a> Iterator for UnclumpedOffsetValues<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let UnclumpedOffsetValues {
            cur_stride,
            cur_offset,
            chunk_offsets,
            offsets,
        } = self;

        if chunk_offsets.is_empty() {
            return None;
        }

        debug_assert_eq!(chunk_offsets.len(), offsets.len());

        // SAFETY: chunk_offsets length is checked above, and offsets has the same length.
        let offset = unsafe { *offsets.get_unchecked(0) };

        if *cur_offset == offset {
            // SAFETY: chunk_offsets length is checked above, and offsets has the same length.
            let chunk_offset = unsafe { *chunk_offsets.get_unchecked(0) };

            // SAFETY: There is at least one element in both offsets and chunk_offsets.
            unsafe {
                // Pop the last internal offset.
                self.offsets = offsets.get_unchecked(1..);
                self.chunk_offsets = chunk_offsets.get_unchecked(1..);
            }

            if let Some(next_offset) = self.offsets.get(0) {
                // The following is safe since clump_offsets is expected to have the same size as
                // offsets (debug_assert above checking this invariant).
                let next_chunk_offset = unsafe { self.chunk_offsets.get_unchecked(0) };

                // Recompute the new stride.
                let clump_dist = next_offset - offset;
                let chunk_size = next_chunk_offset - chunk_offset;
                *cur_stride = clump_dist / chunk_size;
                *cur_offset += *cur_stride;
            }
            Some(offset)
        } else {
            let result = Some(*cur_offset);
            *cur_offset += *cur_stride;
            result
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self
            .chunk_offsets
            .last()
            .map(|last| {
                // SAFETY: there is a last, so there must be a first.
                last - unsafe { *self.chunk_offsets.get_unchecked(0) } + 1
            })
            .unwrap_or(0);
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.size_hint().0
    }
}

impl ExactSizeIterator for UnclumpedOffsetValues<'_> {}
impl std::iter::FusedIterator for UnclumpedOffsetValues<'_> {}

impl<'a> SplitOffsetsAt for ClumpedOffsets<&'a [usize]> {
    /// Same as `split_offsets_at`, but in addition, return the offset of the middle element
    /// (intersection): this is the value `offsets[mid] - offsets[0]`.
    ///
    /// # Panics
    ///
    /// Calling this function with an empty slice or with `mid` greater than or equal to its length
    /// will cause a panic.
    ///
    /// This function will also panic when trying to split a clump since the offsets cannot be
    /// modified or created.
    #[inline]
    fn split_offsets_with_intersection_at(
        self,
        mid_off: usize,
    ) -> (
        ClumpedOffsets<&'a [usize]>,
        ClumpedOffsets<&'a [usize]>,
        usize,
    ) {
        assert!(mid_off < self.num_offsets());
        // Try to find the mid in our chunk offsets.
        // Note that it is the responsibility of the container to ensure that the split is valid.
        let mid_idx = self
            .chunk_offsets
            .binary_search(&mid_off)
            .expect("Cannot split clumped offsets through a clump");
        let (los, ros, off) = self.offsets.split_offsets_with_intersection_at(mid_idx);
        let (lcos, rcos) = self.chunk_offsets.split_offsets_at(mid_idx);
        (
            ClumpedOffsets {
                chunk_offsets: lcos,
                offsets: los,
            },
            ClumpedOffsets {
                chunk_offsets: rcos,
                offsets: ros,
            },
            off,
        )
    }

    /// Splits clumped offsets at the given index into two such that each
    /// resulting `ClumpedOffsets` is valid. This means that the element at index
    /// `mid` is shared between the two outputs.
    ///
    /// # Panics
    ///
    /// Calling this function with an empty slice or with `mid` greater than or equal to its length
    /// will cause a panic.
    ///
    /// This function will also panic when trying to split a clump since the offsets cannot be
    /// modified or created.
    #[inline]
    fn split_offsets_at(
        self,
        mid: usize,
    ) -> (ClumpedOffsets<&'a [usize]>, ClumpedOffsets<&'a [usize]>) {
        assert!(mid < self.num_offsets());
        // Try to find the mid in our chunk offsets.
        // Note that it is the responsibility of the container to ensure that the split is valid.
        let mid_idx = self
            .chunk_offsets
            .binary_search(&mid)
            .expect("Cannot split clumped offsets through a clump");
        let (los, ros) = self.offsets.split_offsets_at(mid_idx);
        let (lcos, rcos) = self.chunk_offsets.split_offsets_at(mid_idx);
        (
            ClumpedOffsets {
                chunk_offsets: lcos,
                offsets: los,
            },
            ClumpedOffsets {
                chunk_offsets: rcos,
                offsets: ros,
            },
        )
    }
}

impl<O> IndexRange for ClumpedOffsets<O>
where
    Self: GetOffset,
{
    #[inline]
    unsafe fn index_range_unchecked(&self, range: Range<usize>) -> Range<usize> {
        let begin = self.offset_unchecked(range.start);
        let end = self.offset_unchecked(range.end);
        begin..end
    }
    /// Return the `[begin..end)` bound of the chunk at the given index.
    #[inline]
    fn index_range(&self, range: Range<usize>) -> Option<Range<usize>> {
        if range.end < self.num_offsets() {
            // SAFETY: ghecked the bound above.
            unsafe { Some(self.index_range_unchecked(range)) }
        } else {
            None
        }
    }
}

impl std::iter::FromIterator<usize> for ClumpedOffsets {
    #[inline]
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = usize>,
    {
        let mut offset_iter = iter.into_iter();
        // Start off with a more realistic capacity, this saves a few almost certain allocations.
        let mut chunk_offsets = Vec::with_capacity(8);
        let mut offsets = Vec::with_capacity(16);
        chunk_offsets.push(0);

        if let Some(first) = offset_iter.next() {
            offsets.push(first);
            if let Some(mut prev_offset) = offset_iter.next() {
                let mut prev_stride = prev_offset - first;
                let mut next_chunk_offset = 1;
                for offset in offset_iter {
                    let stride = offset - prev_offset;
                    if prev_stride != stride {
                        offsets.push(prev_offset);
                        chunk_offsets.push(next_chunk_offset);
                    }
                    next_chunk_offset += 1;
                    prev_stride = stride;
                    prev_offset = offset;
                }
                offsets.push(prev_offset);
                chunk_offsets.push(next_chunk_offset);
            }
        } else {
            offsets.push(0);
        }
        // SAFETY: chunk_offsets and offsets are guaranteed to be non-empty.
        // TODO: Abstract the incremental construction of offsets into `Offsets` to avoid having to
        // use unsafe here.
        unsafe {
            ClumpedOffsets {
                chunk_offsets: Offsets::from_raw(chunk_offsets),
                offsets: Offsets::from_raw(offsets),
            }
        }
    }
}

//impl<O: AsRef<[usize]> + Set> Set for ClumpedOffsets<O> {
//    type Elem = O::Elem;
//    type Atom = O::Atom;
//    #[inline]
//    fn len(&self) -> usize {
//        let offsets = self.chunk_offsets.as_ref();
//        assert!(!offsets.is_empty(), "Clump offsets are corrupted");
//        unsafe { 1 + offsets.get_unchecked(offsets.len() - 1) - offsets.get_unchecked(0) }
//    }
//}

impl<O: Viewed> Viewed for ClumpedOffsets<O> {}

impl<'a, O: AsRef<[usize]>> View<'a> for ClumpedOffsets<O> {
    type Type = ClumpedOffsets<&'a [usize]>;
    #[inline]
    fn view(&'a self) -> Self::Type {
        ClumpedOffsets {
            chunk_offsets: self.chunk_offsets.view(),
            offsets: self.offsets.view(),
        }
    }
}

impl<'a, O: AsMut<[usize]>> ViewMut<'a> for ClumpedOffsets<O> {
    type Type = ClumpedOffsets<&'a mut [usize]>;
    #[inline]
    fn view_mut(&'a mut self) -> Self::Type {
        ClumpedOffsets {
            chunk_offsets: self.chunk_offsets.view_mut(),
            offsets: self.offsets.view_mut(),
        }
    }
}

impl<O: AsRef<[usize]> + Set> From<Offsets<O>> for ClumpedOffsets {
    /// Convert `Offsets` to owned `ClumpedOffsets`.
    ///
    /// This function causes allocations.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::{Offsets, ClumpedOffsets};
    /// let offsets = Offsets::new(vec![0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39]);
    /// let clumped_offsets = ClumpedOffsets::new(vec![0, 4, 7, 12], vec![0, 12, 24, 39]);
    /// assert_eq!(ClumpedOffsets::from(offsets), clumped_offsets);
    /// ```
    #[inline]
    fn from(offsets: Offsets<O>) -> Self {
        debug_assert!(offsets.num_offsets() > 0, "Offsets are corrupted.");
        offsets.values().collect()
    }
}

impl<O: AsRef<[usize]> + Set> From<ClumpedOffsets<O>> for Offsets {
    /// Convert `ClumpedOffsets` into owned `Offsets`.
    ///
    /// This function causes allocations.
    #[inline]
    fn from(clumped_offsets: ClumpedOffsets<O>) -> Self {
        debug_assert!(clumped_offsets.num_offsets() > 0, "Offsets are corrupted.");
        Offsets::new(clumped_offsets.values().collect())
    }
}

impl<O: AsRef<[usize]>> ClumpedOffsets<O> {
    /// Construct new `ClumpedOffsets` from a given valid set of clumped offsets represented as an
    /// array of `usize`s.
    ///
    /// It is possible to create an invalid `ClumpedOffsets` struct using this constructor, however
    /// it is safe.
    ///
    /// # Panics
    ///
    /// This function will panic if either `chunk_offsets` or `offsets` is empty.
    #[inline]
    pub fn new(chunk_offsets: O, offsets: O) -> Self {
        ClumpedOffsets {
            chunk_offsets: Offsets::new(chunk_offsets),
            offsets: Offsets::new(offsets),
        }
    }

    /// An unchecked version of the `new` constructor.
    ///
    /// # Safety
    ///
    /// Calling this function with empty `chunk_offsets` or `offsets` may cause undefined behaviour
    /// in safe APIs.
    #[inline]
    pub unsafe fn from_raw(chunk_offsets: O, offsets: O) -> Self {
        ClumpedOffsets {
            chunk_offsets: Offsets::from_raw(chunk_offsets),
            offsets: Offsets::from_raw(offsets),
        }
    }
}

/// A default set of offsets must allocate.
impl Default for ClumpedOffsets<Vec<usize>> {
    #[inline]
    fn default() -> Self {
        ClumpedOffsets {
            chunk_offsets: Default::default(),
            offsets: Default::default(),
        }
    }
}

/// A dummy set of offsets will not allocate, but the resulting type does not correspond to valid
/// offsets.
impl<O: Dummy> Dummy for ClumpedOffsets<O> {
    /// This function creates an invalid `ClumpedOffsets`, but it does not allocate.
    #[inline]
    unsafe fn dummy() -> Self {
        ClumpedOffsets {
            chunk_offsets: Dummy::dummy(),
            offsets: Dummy::dummy(),
        }
    }
}

impl Clear for ClumpedOffsets {
    #[inline]
    fn clear(&mut self) {
        self.chunk_offsets.truncate(1);
        self.offsets.truncate(1);
    }
}

impl<'a, O> GetIndex<'a, ClumpedOffsets<O>> for usize
where
    ClumpedOffsets<O>: GetOffset,
{
    type Output = usize;
    #[inline]
    fn get(self, clumped_offsets: &ClumpedOffsets<O>) -> Option<Self::Output> {
        if self < clumped_offsets.num_offsets() {
            // The following is safe because we checked the bound above.
            unsafe { Some(clumped_offsets.offset_unchecked(self)) }
        } else {
            None
        }
    }
}

impl<'a, O> GetIndex<'a, ClumpedOffsets<O>> for &usize
where
    ClumpedOffsets<O>: GetOffset,
{
    type Output = usize;
    #[inline]
    fn get(self, clumped_offsets: &ClumpedOffsets<O>) -> Option<Self::Output> {
        GetIndex::get(*self, clumped_offsets)
    }
}

impl<O> IsolateIndex<ClumpedOffsets<O>> for usize
where
    ClumpedOffsets<O>: GetOffset,
{
    type Output = usize;
    #[inline]
    unsafe fn isolate_unchecked(self, clumped_offsets: ClumpedOffsets<O>) -> Self::Output {
        clumped_offsets.offset_unchecked(self)
    }
    #[inline]
    fn try_isolate(self, clumped_offsets: ClumpedOffsets<O>) -> Option<Self::Output> {
        if self < clumped_offsets.num_offsets() {
            // The following is safe because we checked the bound above.
            unsafe { Some(IsolateIndex::isolate_unchecked(self, clumped_offsets)) }
        } else {
            None
        }
    }
}

impl<O: Truncate> Truncate for ClumpedOffsets<O> {
    #[inline]
    fn truncate(&mut self, new_len: usize) {
        self.chunk_offsets.truncate(new_len);
        self.offsets.truncate(new_len);
    }
}

impl<O: RemovePrefix + Set> RemovePrefix for ClumpedOffsets<O> {
    #[inline]
    fn remove_prefix(&mut self, n: usize) {
        self.chunk_offsets.remove_prefix(n);
        self.offsets.remove_prefix(n);
    }
}

impl<O: IntoOwned> IntoOwned for ClumpedOffsets<O> {
    type Owned = ClumpedOffsets<O::Owned>;
    #[inline]
    fn into_owned(self) -> Self::Owned {
        ClumpedOffsets {
            chunk_offsets: self.chunk_offsets.into_owned(),
            offsets: self.offsets.into_owned(),
        }
    }
}

impl<O: Reserve> Reserve for ClumpedOffsets<O> {
    #[inline]
    fn reserve_with_storage(&mut self, n: usize, storage_n: usize) {
        self.chunk_offsets.reserve_with_storage(n, storage_n);
        self.offsets.reserve_with_storage(n, storage_n);
    }
}

impl std::iter::Extend<(usize, usize)> for ClumpedOffsets {
    /// Extend this set of clumped offsets with a given iterator over chunk-offset and offset pairs.
    ///
    /// This operation automatically shifts the merged offsets in the iterator
    /// to start from the last offset in `self`.
    ///
    /// Note that there will be 1 less offset added to `self` than produced by
    /// `iter` since the first offset is only used to determine the relative
    /// magnitude of the rest and corresponds to the last offset in `self`.
    fn extend<T: IntoIterator<Item = (usize, usize)>>(&mut self, iter: T) {
        let mut iter = iter.into_iter();
        if let Some((first_chunk_offset, first_offset)) = iter.next() {
            let last_offset = self.last_offset_value();
            let Self {
                chunk_offsets,
                offsets,
            } = self;
            chunk_offsets.extend(std::iter::once(first_chunk_offset).chain(iter.map(
                |(chunk_offset, offset)| {
                    offsets.push(offset + last_offset - first_offset);
                    chunk_offset
                },
            )));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Test for the `split_offset_at` helper function.
    #[test]
    fn split_offset_at_test() {
        let offsets = Offsets::new(vec![3, 6, 9, 12, 15, 19, 23, 27, 30, 33, 36, 39, 42]);
        let clumped_offsets = ClumpedOffsets::from(offsets);

        // Test in the middle
        let (l, r, off) = clumped_offsets.view().split_offsets_with_intersection_at(4);
        assert_eq!(Offsets::from(l).into_inner(), &[3, 6, 9, 12, 15]);
        assert_eq!(
            Offsets::from(r).into_inner(),
            &[15, 19, 23, 27, 30, 33, 36, 39, 42]
        );
        assert_eq!(off, 12);

        // Test at the beginning
        let (l, r, off) = clumped_offsets.view().split_offsets_with_intersection_at(0);
        assert_eq!(Offsets::from(l).into_inner(), &[3]);
        assert_eq!(
            Offsets::from(r).into_inner(),
            &[3, 6, 9, 12, 15, 19, 23, 27, 30, 33, 36, 39, 42]
        );
        assert_eq!(off, 0);

        // Test at the end
        let (l, r, off) = clumped_offsets
            .view()
            .split_offsets_with_intersection_at(12);
        assert_eq!(
            Offsets::from(l).into_inner(),
            &[3, 6, 9, 12, 15, 19, 23, 27, 30, 33, 36, 39, 42]
        );
        assert_eq!(Offsets::from(r).into_inner(), &[42]);
        assert_eq!(off, 39);
    }

    /// Check iterators over offsets and sizes.
    #[test]
    fn iterators() {
        let offsets = Offsets::new(vec![3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39, 42]);
        let clumped_offsets = ClumpedOffsets::from(offsets.clone());
        assert_eq!(clumped_offsets.offset(6), 21);

        for (orig, unclumped) in offsets.values().zip(clumped_offsets.values()) {
            assert_eq!(orig, unclumped);
        }
        for (orig, unclumped) in offsets.sizes().zip(clumped_offsets.sizes()) {
            assert_eq!(orig, unclumped);
        }
    }

    /// Check index_range
    #[test]
    fn index_range() {
        let offsets = Offsets::new(vec![3, 6, 9, 12, 15, 19, 23, 27, 30, 33, 36, 39, 42]);
        let clumped_offsets = ClumpedOffsets::from(offsets.clone());
        assert_eq!(clumped_offsets.index_range(2..5), Some(6..16));
        assert_eq!(clumped_offsets.index_range(0..3), Some(0..9));
        assert_eq!(clumped_offsets.index_range(6..8), Some(20..27));
        assert_eq!(clumped_offsets.index_range(6..800), None);
    }

    /// Check that clearing the clumped offsets keeps exactly one element.
    #[test]
    fn clear() {
        let mut offsets = ClumpedOffsets::from(Offsets::new(vec![0, 1, 2, 3, 4, 5]));
        offsets.clear();
        assert_eq!(offsets, ClumpedOffsets::new(vec![0], vec![0]));
    }

    /// Check indexing offsets.
    #[test]
    fn get_offset() {
        let s = ClumpedOffsets::new(vec![3, 6, 7], vec![2, 11, 15]);
        assert_eq!(0, s.offset(0));
        assert_eq!(3, s.offset(1));
        assert_eq!(6, s.offset(2));
        assert_eq!(9, s.offset(3));
        assert_eq!(13, s.offset(4));

        // get raw offset values
        assert_eq!(2, s.offset_value(0));
        assert_eq!(5, s.offset_value(1));
        assert_eq!(8, s.offset_value(2));
        assert_eq!(11, s.offset_value(3));
        assert_eq!(15, s.offset_value(4));
    }

    #[test]
    fn offset_value_iter() {
        use ExactSizeIterator;
        let offsets = Offsets::new(vec![0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39]);
        let clumped_offsets = ClumpedOffsets::from(offsets.clone());
        let iter = clumped_offsets.values();
        let iter_len = iter.len();
        assert_eq!(iter_len, offsets.num_offsets());
        assert_eq!(iter_len, iter.count());

        // The following checks that count is correctly implemented for the iterator.
        let mut count = 0;
        for _ in iter {
            count += 1;
        }
        assert_eq!(iter_len, count);
    }

    #[test]
    fn sizes_iter() {
        use ExactSizeIterator;
        let offsets = Offsets::new(vec![3, 6, 9, 12, 15, 19, 23, 27, 30, 33, 36, 39, 42]);
        let clumped_offsets = ClumpedOffsets::from(offsets.clone());
        let mut iter = clumped_offsets.sizes();
        let iter_len = iter.len();
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

        assert_eq!(clumped_offsets.sizes().nth(0).unwrap(), 3);
        assert_eq!(clumped_offsets.sizes().nth(1).unwrap(), 3);
        assert_eq!(clumped_offsets.sizes().nth(2).unwrap(), 3);
        assert_eq!(clumped_offsets.sizes().nth(3).unwrap(), 3);
        assert_eq!(clumped_offsets.sizes().nth(4).unwrap(), 4);
        assert_eq!(clumped_offsets.sizes().nth(5).unwrap(), 4);
        assert_eq!(clumped_offsets.sizes().nth(6).unwrap(), 4);
        assert_eq!(clumped_offsets.sizes().nth(7).unwrap(), 3);
        assert_eq!(clumped_offsets.sizes().nth(8).unwrap(), 3);
        assert_eq!(clumped_offsets.sizes().nth(9).unwrap(), 3);
        assert_eq!(clumped_offsets.sizes().nth(10).unwrap(), 3);
        assert_eq!(clumped_offsets.sizes().nth(11).unwrap(), 3);

        // The following checks that count is correctly implemented for the sizes iterator.
        let mut count = 0;
        for _ in clumped_offsets.sizes() {
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

        let mut iter = clumped_offsets.sizes();
        assert_eq!(iter.nth(5).unwrap(), 4); // Start with nth skipping first clump
    }
    #[test]
    fn extend_clumped_offsets() {
        let offsets = Offsets::new(vec![3, 6, 9, 10, 11]);
        let mut clumped_offsets = ClumpedOffsets::from(offsets.clone());
        let orig_clumped_offsets = clumped_offsets.clone();
        clumped_offsets.extend(vec![]); // Safe and no panics.
        assert_eq!(clumped_offsets, orig_clumped_offsets);
        clumped_offsets.extend(vec![(0, 0)]); // Nothing new added.
        assert_eq!(clumped_offsets, orig_clumped_offsets);
        clumped_offsets.extend(vec![(1, 1), (3, 5)]); // Nothing new added.
        let exp_extended_offsets = Offsets::new(vec![3, 6, 9, 10, 11, 13, 15]);
        assert_eq!(clumped_offsets, ClumpedOffsets::from(exp_extended_offsets));
    }
}
