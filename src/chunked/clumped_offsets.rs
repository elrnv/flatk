//! This module defines the `ClumpedOffsets` type and its behaviour.

use super::*;
use std::convert::{AsMut, AsRef};
use std::ops::Range;

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
///     use flatk::chunked::{Offsets, ClumpedOffsets};
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

impl<O: Set + AsRef<[usize]>> GetOffset for ClumpedOffsets<O> {
    /// A version of `offset_value` without bounds checking.
    ///
    /// # Safety
    ///
    /// It is assumed that `index` is strictly less than `self.len()`.
    #[inline]
    unsafe fn offset_value_unchecked(&self, index: usize) -> usize {
        let ClumpedOffsets {
            chunk_offsets,
            offsets,
        } = self;
        debug_assert!(!chunk_offsets.is_empty());
        debug_assert!(!offsets.is_empty());
        // The following is safe by construction of ClumpedOffsets.
        let offset = match chunk_offsets.binary_search(&index) {
            Ok(clump_idx) => offsets.offset_value_unchecked(clump_idx),
            Err(clump_idx) => {
                // Offset is in the middle of a clump.
                // Given that idx is not out of bounds as defined in the doc,
                // the following are safe because index >= 0.
                // so clump_idx > 0. The inequality is strict because binary_search failed.
                let begin_off = offsets.offset_value_unchecked(clump_idx - 1);
                let clump_dist = offsets.offset_value_unchecked(clump_idx) - begin_off;
                let begin_clump_off = chunk_offsets.offset_value_unchecked(clump_idx - 1);
                let clump_size = chunk_offsets.offset_value_unchecked(clump_idx) - begin_clump_off;
                let stride = clump_dist / clump_size;
                begin_off + stride * (index - begin_clump_off)
            }
        };
        offset
    }
}

impl<O: AsRef<[usize]> + Set> ClumpedOffsets<O> {
    /// An iterator over effective (unclumped) sizes producing an increment for advancing the
    /// data pointer.
    ///
    /// This helps for implementing iterators over `Chunked` types.
    ///
    /// # Example
    ///
    /// ```
    ///     use flatk::chunked::{Offsets, ClumpedOffsets};
    ///
    ///     let off = Offsets::new(vec![0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39]);
    ///     let clumped = ClumpedOffsets::from(off);
    ///     let mut clumped_iter = clumped.sizes();
    ///     for _ in 0..4 {
    ///         assert_eq!(clumped_iter.next(), Some(3));
    ///     }
    ///     for _ in 0..3 {
    ///         assert_eq!(clumped_iter.next(), Some(4));
    ///     }
    ///     for _ in 0..5 {
    ///         assert_eq!(clumped_iter.next(), Some(3));
    ///     }
    ///     assert_eq!(clumped_iter.next(), None);
    /// ```
    #[inline]
    pub fn sizes(&self) -> UnclumpedSizes {
        debug_assert!(!self.chunk_offsets.is_empty());
        UnclumpedSizes {
            cur: self.chunk_offsets.first_offset_value(),
            chunk_offsets: self.chunk_offsets.as_ref(),
            offsets: self.offsets.as_ref(),
        }
    }

    /// An iterator over unclumped offsets.
    ///
    /// This is equivalent to iterating over `Offsets` after conversion, but it doesn't require any
    /// additional allocations.
    #[deprecated(since = "0.2.1", note = "please use `values` instead")]
    #[inline]
    pub fn iter(&self) -> UnclumpedOffsetValues {
        debug_assert!(!self.offsets.as_ref().is_empty());
        UnclumpedOffsetValues {
            cur_stride: 0,
            cur_offset: self.offsets.first_offset_value(),
            chunk_offsets: self.chunk_offsets.as_ref(),
            offsets: self.offsets.as_ref(),
        }
    }

    /// An iterator over unclumped offsets.
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

#[derive(Copy, Clone)]
pub struct UnclumpedSizes<'a> {
    cur: usize,
    chunk_offsets: &'a [usize],
    offsets: &'a [usize],
}

impl<'a> Iterator for UnclumpedSizes<'a> {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        let UnclumpedSizes {
            cur,
            chunk_offsets,
            offsets,
        } = self;

        if offsets.len() < 2 {
            return None;
        }

        // Sanity check that chunk_offsets are in sync.
        debug_assert!(chunk_offsets.len() > 1);

        // The following is safe because we checked the length of offsets above.
        // Also the API enforces the size correspondence between
        // chunk_offsets and offsets. We also do a debug check above.
        let offset = unsafe { *offsets.get_unchecked(0) };
        let chunk_offset = unsafe { *chunk_offsets.get_unchecked(0) };
        let next_offset = unsafe { *offsets.get_unchecked(1) };
        let next_chunk_offset = unsafe { *chunk_offsets.get_unchecked(1) };

        let clump_dist = next_offset - offset;
        let clump_size = next_chunk_offset - chunk_offset;
        let stride = clump_dist / clump_size;

        // Incrementing the head offset effectively pops the unclumped offset.
        *cur += 1;
        if *cur == next_chunk_offset {
            // Pop the last internal offset, which also changes the stride.
            self.offsets = &offsets[1..];
            self.chunk_offsets = &chunk_offsets[1..];
        }

        // The current stride is equal to the next effective unclumped size
        Some(stride)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.chunk_offsets.len();
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.chunk_offsets.len()
    }
}

impl<'a> IntoValues for ClumpedOffsets<&'a [usize]> {
    type Iter = UnclumpedOffsetValues<'a>;
    /// Returns an iterator over offset values represented by the stored `Offsets`.
    #[inline]
    fn into_values(self) -> UnclumpedOffsetValues<'a> {
        debug_assert!(!self.chunk_offsets.is_empty());
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
        debug_assert!(!self.chunk_offsets.is_empty());
        UnclumpedSizes {
            cur: self.chunk_offsets.first_offset_value(),
            chunk_offsets: self.chunk_offsets.into_inner(),
            offsets: self.offsets.into_inner(),
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
        let chunk_offset = unsafe { *chunk_offsets.get_unchecked(0) };

        if *cur_offset == offset {
            // Pop the last internal offset.
            unsafe {
                self.offsets = &offsets.get_unchecked(1..);
                self.chunk_offsets = &chunk_offsets.get_unchecked(1..);
            }
            if let Some(next_offset) = self.offsets.get(0) {
                // The following is safe since clump_ofsfets is expected to have the same size as
                // offsets (debug_assert above checking this invariant).
                let next_chunk_offset = unsafe { self.chunk_offsets.get_unchecked(0) };
                // Recompute the new stride
                let clump_dist = next_offset - offset;
                let clump_size = next_chunk_offset - chunk_offset;
                *cur_stride = clump_dist / clump_size;
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
        let n = self.offsets.len();
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.offsets.len()
    }
}

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
        mid: usize,
    ) -> (
        ClumpedOffsets<&'a [usize]>,
        ClumpedOffsets<&'a [usize]>,
        usize,
    ) {
        assert!(!self.is_empty());
        assert!(mid < self.len());
        // Try to find the mid in our chunk offsets.
        // Note that it is the responsibility of the container to ensure that the split is valid.
        let mid_idx = self
            .chunk_offsets
            .binary_search(&mid)
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
        assert!(!self.is_empty());
        assert!(mid < self.len());
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
    Self: Set + GetOffset,
{
    /// Return the `[begin..end)` bound of the chunk at the given index.
    fn index_range(&self, range: Range<usize>) -> Option<Range<usize>> {
        if range.end < self.len() {
            // The following is safe because we checked the bound above.
            unsafe {
                let begin = self.offset_unchecked(range.start);
                let end = self.offset_unchecked(range.end);
                Some(begin..end)
            }
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
        ClumpedOffsets {
            chunk_offsets: Offsets(chunk_offsets),
            offsets: Offsets(offsets),
        }
    }
}

impl<O: AsRef<[usize]> + Set> Set for ClumpedOffsets<O> {
    type Elem = O::Elem;
    type Atom = O::Atom;
    #[inline]
    fn len(&self) -> usize {
        let offsets = self.chunk_offsets.as_ref();
        assert!(!offsets.is_empty(), "Clump offsets are corrupted");
        unsafe { 1 + offsets.get_unchecked(offsets.len() - 1) - offsets.get_unchecked(0) }
    }
}

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
    /// use flatk::chunked::{Offsets, ClumpedOffsets};
    /// let offsets = Offsets::new(vec![0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39]);
    /// let clumped_offsets = ClumpedOffsets::new(vec![0, 4, 7, 12], vec![0, 12, 24, 39]);
    /// assert_eq!(ClumpedOffsets::from(offsets), clumped_offsets);
    /// ```
    #[inline]
    fn from(offsets: Offsets<O>) -> Self {
        debug_assert!(!offsets.is_empty(), "Offsets are corrupted.");
        offsets.values().collect()
    }
}

impl<O: AsRef<[usize]> + Set> From<ClumpedOffsets<O>> for Offsets {
    /// Convert `ClumpedOffsets` into owned `Offsets`.
    ///
    /// This function causes allocations.
    #[inline]
    fn from(clumped_offsets: ClumpedOffsets<O>) -> Self {
        debug_assert!(!clumped_offsets.is_empty(), "Offsets are corrupted.");
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
        if self < clumped_offsets.len() {
            // The following is safe because we checked the bound above.
            unsafe { Some(clumped_offsets.offset_unchecked(self)) }
        } else {
            None
        }
    }
}

impl<O> IsolateIndex<ClumpedOffsets<O>> for usize
where
    ClumpedOffsets<O>: GetOffset,
{
    type Output = usize;
    #[inline]
    fn try_isolate(self, clumped_offsets: ClumpedOffsets<O>) -> Option<Self::Output> {
        if self < clumped_offsets.len() {
            // The following is safe because we checked the bound above.
            unsafe { Some(clumped_offsets.offset_unchecked(self)) }
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

#[cfg(test)]
mod tests {
    use super::*;
    /// Test for the `split_offset_at` helper function.
    #[test]
    fn split_offset_at_test() {
        let offsets = Offsets::new(vec![0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39]);
        let clumped_offsets = ClumpedOffsets::from(offsets);

        // Test in the middle
        let (l, r, off) = clumped_offsets.view().split_offsets_with_intersection_at(4);
        assert_eq!(Offsets::from(l).into_inner(), &[0, 3, 6, 9, 12]);
        assert_eq!(
            Offsets::from(r).into_inner(),
            &[12, 16, 20, 24, 27, 30, 33, 36, 39]
        );
        assert_eq!(off, 12);

        // Test at the beginning
        let (l, r, off) = clumped_offsets.view().split_offsets_with_intersection_at(0);
        dbg!(&l);
        assert_eq!(Offsets::from(l).into_inner(), &[0]);
        assert_eq!(
            Offsets::from(r).into_inner(),
            &[0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39]
        );
        assert_eq!(off, 0);

        // Test at the end
        let (l, r, off) = clumped_offsets
            .view()
            .split_offsets_with_intersection_at(12);
        assert_eq!(
            Offsets::from(l).into_inner(),
            &[0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39]
        );
        assert_eq!(Offsets::from(r).into_inner(), &[39]);
        assert_eq!(off, 39);
    }

    /// Check iterators over offsets and sizes.
    #[test]
    fn iterators() {
        let offsets = Offsets::new(vec![0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39]);
        let clumped_offsets = ClumpedOffsets::from(offsets.clone());
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
        let offsets = Offsets::new(vec![0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36, 39]);
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
        let s = ClumpedOffsets::from(Offsets::new(vec![2, 5, 6, 8]));
        assert_eq!(0, s.offset(0));
        assert_eq!(3, s.offset(1));
        assert_eq!(4, s.offset(2));
        assert_eq!(6, s.offset(3));

        // get raw offset values
        assert_eq!(2, s.offset_value(0));
        assert_eq!(5, s.offset_value(1));
        assert_eq!(6, s.offset_value(2));
        assert_eq!(8, s.offset_value(3));
    }
}
