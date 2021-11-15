use super::*;
use std::convert::{AsMut, AsRef};
use std::ops::Range;

#[cfg(feature = "rayon")]
pub(crate) mod par_iter;

/// A collection of offsets into another collection.
/// This newtype is intended to verify basic invariants about offsets into
/// another collection, namely that the collection is monotonically increasing
/// and non-empty.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Offsets<O = Vec<usize>>(O);

impl<O: Set<Elem = usize, Atom = usize>> Set for Offsets<O> {
    type Elem = usize;
    type Atom = usize;

    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<O: AsRef<[usize]> + Set> Offsets<O> {
    #[inline]
    fn offset_value_ranges(&self) -> OffsetValueRanges {
        self.view().into_offset_value_ranges()
    }
}

impl<'a> Offsets<&'a [usize]> {
    #[inline]
    fn into_offset_value_ranges(self) -> OffsetValueRanges<'a> {
        OffsetValueRanges { offsets: self }
    }

    /// Separate the offsets into two groups overlapping by one chunk at the given index.
    #[inline]
    pub fn separate_offsets_with_overlap(
        self,
        index: usize,
    ) -> (Offsets<&'a [usize]>, Offsets<&'a [usize]>) {
        // Check bounds, and ensure that self has at least two elements.
        assert!(index + 1 < self.0.len());
        let l = &self.0[..index + 2];
        let r = &self.0[index..];
        (Offsets(l), Offsets(r))
    }
}

impl<O: RemovePrefix + Set> Offsets<O> {
    /// Unchecked version of `RemovePrefix::remove_prefix`.
    ///
    /// # Safety
    ///
    /// This function may cause undefined behaviour in safe APIs if calling this function produces
    /// an empty `offsets` collection.
    #[inline]
    pub unsafe fn remove_prefix_unchecked(&mut self, n: usize) {
        self.0.remove_prefix(n);
    }
}

impl Offsets<&[usize]> {
    /// Remove `n` offsets from the end.
    ///
    /// # Safety
    ///
    /// This function may cause undefined behaviour in safe APIs if calling this function produces
    /// an empty `offsets` collection.
    #[inline]
    pub unsafe fn remove_suffix_unchecked(&mut self, n: usize) {
        self.0 = &self.0.get_unchecked(..self.0.len() - n);
    }
}

impl<O: Viewed> Viewed for Offsets<O> {}

// SAFETY: Offsets should always be non-empty.
unsafe impl<O: AsRef<[usize]>> GetOffset for Offsets<O> {
    /// A version of `offset_value` without bounds checking.
    ///
    /// # Safety
    ///
    /// It is assumed that `index` is strictly less than `self.num_offsets()`.
    #[inline]
    unsafe fn offset_value_unchecked(&self, index: usize) -> usize {
        *self.0.as_ref().get_unchecked(index)
    }

    /// Get the total number of offsets.
    ///
    /// This is one more than the number of chunks represented.
    fn num_offsets(&self) -> usize {
        self.0.as_ref().len()
    }
}

impl<O: AsRef<[usize]>> BinarySearch<usize> for Offsets<O> {
    /// Binary search the offsets for a given offset `off`.
    ///
    /// `off` is expected to be with respect to the beginning of the range represented by the
    /// current offsets. In other words, we are searching for offsets, not raw offset values
    /// stored in `Offsets`.
    ///
    /// The semantics of this function are identical to Rust's `std::slice::binary_search`.
    #[inline]
    fn binary_search(&self, off: &usize) -> Result<usize, usize> {
        self.as_ref()
            .binary_search(&(*off + self.first_offset_value()))
    }
}

/// An iterator over offset values.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OffsetValues<'a> {
    offset_values: &'a [usize],
}

impl<'a> Iterator for OffsetValues<'a> {
    type Item = usize;

    /// Get the next available offset.
    #[inline]
    fn next(&mut self) -> Option<usize> {
        self.offset_values.split_first().map(|(first, rest)| {
            self.offset_values = rest;
            *first
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.offset_values.len();
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        ExactSizeIterator::len(&self)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.offset_values.get(n).map(|&x| x)
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl DoubleEndedIterator for OffsetValues<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        self.offset_values.split_last().map(|(last, rest)| {
            self.offset_values = rest;
            *last
        })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.offset_values
            .get(ExactSizeIterator::len(self) - 1 - n)
            .map(|&x| x)
    }
}

impl ExactSizeIterator for OffsetValues<'_> {}
impl std::iter::FusedIterator for OffsetValues<'_> {}

unsafe impl TrustedRandomAccess for OffsetValues<'_> {
    unsafe fn get_unchecked(&mut self, i: usize) -> Self::Item {
        *self.offset_values.get_unchecked(i)
    }
}

/// Iterator over ranges of offset values.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OffsetValueRanges<'a> {
    /// Offsets used to keep track of which range we are at.
    offsets: Offsets<&'a [usize]>,
}

// SAFETY: since sizes are one less than offsets, the last offset will never be consumed.
unsafe impl GetOffset for OffsetValueRanges<'_> {
    /// Get the offset value without bounds checking.
    ///
    /// # Safety
    ///
    /// This function assumes that `i` is strictly less than `self.num_offsets()`.
    #[inline]
    unsafe fn offset_value_unchecked(&self, i: usize) -> usize {
        self.offsets.offset_value_unchecked(i)
    }
    #[inline]
    fn num_offsets(&self) -> usize {
        self.offsets.num_offsets()
    }
}

impl<'a> Iterator for OffsetValueRanges<'a> {
    type Item = std::ops::Range<usize>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.offsets.num_offsets() < 2 {
            return None;
        }
        let begin = self.offsets.first_offset_value();

        // SAFETY: there are at least two offsets as checked above.
        unsafe {
            self.offsets.remove_prefix_unchecked(1);
        }

        let end = self.offsets.first_offset_value();
        Some(begin..end)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.offsets.num_offsets() - 1;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n + 1 < self.offsets.num_offsets() {
            // SAFETY: bounds are checked above to ensure at least one more element after n
            // elements are removed.
            unsafe {
                let begin = self.offsets.offset_value_unchecked(n);
                self.offsets.remove_prefix_unchecked(n + 1);
                let end = self.offsets.first_offset_value();
                Some(begin..end)
            }
        } else {
            None
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a> DoubleEndedIterator for OffsetValueRanges<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.offsets.num_offsets() < 2 {
            return None;
        }
        let end = self.offsets.last_offset_value();
        // SAFETY: there are at least two offsets as checked above.
        unsafe {
            self.offsets.remove_suffix_unchecked(1);
        }
        let begin = self.offsets.last_offset_value();
        Some(begin..end)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let num_offsets = self.offsets.num_offsets();
        if n + 1 < num_offsets {
            // SAFETY: bounds are checked above to ensure at least one more element after n
            // elements are removed.
            unsafe {
                let end = self.offsets.offset_value_unchecked(num_offsets - 1 - n);
                self.offsets.remove_suffix_unchecked(n + 1);
                let begin = self.offsets.last_offset_value();
                Some(begin..end)
            }
        } else {
            None
        }
    }
}

impl ExactSizeIterator for OffsetValueRanges<'_> {}
impl std::iter::FusedIterator for OffsetValueRanges<'_> {}

unsafe impl TrustedRandomAccess for OffsetValueRanges<'_> {
    #[inline]
    unsafe fn get_unchecked(&mut self, i: usize) -> Self::Item {
        let begin = self.offsets.offset_value_unchecked(i - 1);
        let end = self.offsets.offset_value_unchecked(i);
        begin..end
    }
}

/// Iterator over ranges of offsets.
///
/// This is basically an adapter for `OffsetValueRanges` that subtracts the first offset value from
/// each reange to generate a useful offset range.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ranges<'a> {
    /// Offset value ranges iterator.
    offset_value_ranges: OffsetValueRanges<'a>,
    /// The very first offset value used to normalize the generated ranges.
    first_offset_value: usize,
}

impl<'a> Ranges<'a> {
    /// Produces a function that converts an offset *value* range into a bona-fide offset range by
    /// subtracting the first offset from both endpoints.
    #[inline]
    fn offset_range_converter<'b>(&'b self) -> impl Fn(Range<usize>) -> Range<usize> + 'b {
        move |Range { start, end }| start - self.first_offset_value..end - self.first_offset_value
    }
}

// SAFETY: since ranges are one less than offsets, the last offset will never be consumed.
unsafe impl GetOffset for Ranges<'_> {
    #[inline]
    unsafe fn offset_value_unchecked(&self, i: usize) -> usize {
        self.offset_value_ranges.offset_value_unchecked(i)
    }
    #[inline]
    fn num_offsets(&self) -> usize {
        self.offset_value_ranges.num_offsets()
    }
}

impl<'a> Iterator for Ranges<'a> {
    type Item = Range<usize>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.offset_value_ranges
            .next()
            .map(self.offset_range_converter())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offset_value_ranges.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.offset_value_ranges.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.offset_value_ranges
            .nth(n)
            .map(self.offset_range_converter())
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.offset_value_ranges
            .last()
            .map(self.offset_range_converter())
    }
}

impl<'a> DoubleEndedIterator for Ranges<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.offset_value_ranges
            .next_back()
            .map(self.offset_range_converter())
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.offset_value_ranges
            .nth_back(n)
            .map(self.offset_range_converter())
    }
}

impl ExactSizeIterator for Ranges<'_> {}
impl std::iter::FusedIterator for Ranges<'_> {}

unsafe impl TrustedRandomAccess for Ranges<'_> {
    #[inline]
    unsafe fn get_unchecked(&mut self, i: usize) -> Self::Item {
        let rng = self.offset_value_ranges.get_unchecked(i);
        (self.offset_range_converter())(rng)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sizes<'a> {
    offset_value_ranges: OffsetValueRanges<'a>,
}

impl<'a> Sizes<'a> {
    /// Produces a function that converts an offset value range into a range size by
    /// subtracting the last and first end points.
    #[inline]
    fn range_to_size_mapper<'b>(&'b self) -> impl Fn(Range<usize>) -> usize + 'b {
        move |Range { start, end }| end - start
    }
}

// SAFETY: since sizes are one less than offsets, the last offset will never be consumed.
unsafe impl GetOffset for Sizes<'_> {
    #[inline]
    unsafe fn offset_value_unchecked(&self, i: usize) -> usize {
        self.offset_value_ranges.offset_value_unchecked(i)
    }
    #[inline]
    fn num_offsets(&self) -> usize {
        self.offset_value_ranges.num_offsets()
    }
}

impl<'a> Iterator for Sizes<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        self.offset_value_ranges
            .next()
            .map(self.range_to_size_mapper())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offset_value_ranges.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.offset_value_ranges.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.offset_value_ranges
            .nth(n)
            .map(self.range_to_size_mapper())
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.offset_value_ranges
            .last()
            .map(self.range_to_size_mapper())
    }
}

impl<'a> DoubleEndedIterator for Sizes<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.offset_value_ranges
            .next_back()
            .map(self.range_to_size_mapper())
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.offset_value_ranges
            .nth_back(n)
            .map(self.range_to_size_mapper())
    }
}

impl ExactSizeIterator for Sizes<'_> {}
impl std::iter::FusedIterator for Sizes<'_> {}

unsafe impl TrustedRandomAccess for Sizes<'_> {
    #[inline]
    unsafe fn get_unchecked(&mut self, i: usize) -> Self::Item {
        let rng = self.offset_value_ranges.get_unchecked(i);
        (self.range_to_size_mapper())(rng)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OffsetValuesAndSizes<'a> {
    offset_value_ranges: OffsetValueRanges<'a>,
}

impl<'a> OffsetValuesAndSizes<'a> {
    /// Produces a function that converts an offset value range into a starting offset value and
    /// range size by subtracting the last and first end points.
    #[inline]
    fn range_mapper<'b>(&'b self) -> impl Fn(Range<usize>) -> (usize, usize) + 'b {
        move |Range { start, end }| (start, end - start)
    }
}

// SAFETY: since sizes are one less than offsets, the last offset will never be consumed.
unsafe impl GetOffset for OffsetValuesAndSizes<'_> {
    #[inline]
    unsafe fn offset_value_unchecked(&self, i: usize) -> usize {
        self.offset_value_ranges.offset_value_unchecked(i)
    }
    #[inline]
    fn num_offsets(&self) -> usize {
        self.offset_value_ranges.num_offsets()
    }
}

impl<'a> Iterator for OffsetValuesAndSizes<'a> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.offset_value_ranges.next().map(self.range_mapper())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offset_value_ranges.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.offset_value_ranges.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.offset_value_ranges.nth(n).map(self.range_mapper())
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.offset_value_ranges.last().map(self.range_mapper())
    }
}

impl<'a> DoubleEndedIterator for OffsetValuesAndSizes<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.offset_value_ranges
            .next_back()
            .map(self.range_mapper())
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.offset_value_ranges
            .nth_back(n)
            .map(self.range_mapper())
    }
}

impl ExactSizeIterator for OffsetValuesAndSizes<'_> {}
impl std::iter::FusedIterator for OffsetValuesAndSizes<'_> {}

unsafe impl TrustedRandomAccess for OffsetValuesAndSizes<'_> {
    #[inline]
    unsafe fn get_unchecked(&mut self, i: usize) -> Self::Item {
        let rng = self.offset_value_ranges.get_unchecked(i);
        (self.range_mapper())(rng)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OffsetsAndSizes<'a> {
    offset_value_ranges: OffsetValueRanges<'a>,
    first_offset_value: usize,
}

impl<'a> OffsetsAndSizes<'a> {
    /// Produces a function that converts an offset value range into a range size by
    /// subtracting the last and first end points.
    #[inline]
    fn range_mapper<'b>(&'b self) -> impl Fn(Range<usize>) -> (usize, usize) + 'b {
        move |Range { start, end }| (start - self.first_offset_value, end - start)
    }
}

// SAFETY: since sizes are one less than offsets, the last offset will never be consumed.
unsafe impl GetOffset for OffsetsAndSizes<'_> {
    #[inline]
    unsafe fn offset_value_unchecked(&self, i: usize) -> usize {
        self.offset_value_ranges.offset_value_unchecked(i)
    }
    #[inline]
    fn num_offsets(&self) -> usize {
        self.offset_value_ranges.num_offsets()
    }
}

impl<'a> Iterator for OffsetsAndSizes<'a> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.offset_value_ranges.next().map(self.range_mapper())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.offset_value_ranges.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.offset_value_ranges.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.offset_value_ranges.nth(n).map(self.range_mapper())
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.offset_value_ranges.last().map(self.range_mapper())
    }
}

impl<'a> DoubleEndedIterator for OffsetsAndSizes<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.offset_value_ranges
            .next_back()
            .map(self.range_mapper())
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.offset_value_ranges
            .nth_back(n)
            .map(self.range_mapper())
    }
}

impl ExactSizeIterator for OffsetsAndSizes<'_> {}
impl std::iter::FusedIterator for OffsetsAndSizes<'_> {}

unsafe impl TrustedRandomAccess for OffsetsAndSizes<'_> {
    #[inline]
    unsafe fn get_unchecked(&mut self, i: usize) -> Self::Item {
        let rng = self.offset_value_ranges.get_unchecked(i);
        (self.range_mapper())(rng)
    }
}

impl<'a> IntoValues for Offsets<&'a [usize]> {
    type Iter = OffsetValues<'a>;
    /// Returns an iterator over offset values represented by the stored `Offsets`.
    #[inline]
    fn into_values(self) -> OffsetValues<'a> {
        OffsetValues {
            offset_values: self.0,
        }
    }
}

impl<'a> IntoSizes for Offsets<&'a [usize]> {
    type Iter = Sizes<'a>;
    /// Returns an iterator over chunk sizes represented by the stored `Offsets`.
    #[inline]
    fn into_sizes(self) -> Sizes<'a> {
        Sizes {
            offset_value_ranges: self.into_offset_value_ranges(),
        }
    }
}

impl<'a> IntoOffsetValuesAndSizes for Offsets<&'a [usize]> {
    type Iter = OffsetValuesAndSizes<'a>;
    /// Returns an iterator over offset value and chunk size pairs.
    #[inline]
    fn into_offset_values_and_sizes(self) -> OffsetValuesAndSizes<'a> {
        OffsetValuesAndSizes {
            offset_value_ranges: self.into_offset_value_ranges(),
        }
    }
}

impl<'a> IntoRanges for Offsets<&'a [usize]> {
    type Iter = Ranges<'a>;
    /// Returns an iterator over offset ranges represented by the stored `Offsets`.
    #[inline]
    fn into_ranges(self) -> Ranges<'a> {
        Ranges {
            offset_value_ranges: self.into_offset_value_ranges(),
            first_offset_value: self.first_offset_value(),
        }
    }
}

impl<O: AsRef<[usize]> + Set> Offsets<O> {
    /// Returns an iterator over chunk sizes represented by the stored `Offsets`.
    #[inline]
    pub fn sizes(&self) -> Sizes {
        Sizes {
            offset_value_ranges: self.offset_value_ranges(),
        }
    }

    /// Returns an iterator over offsets.
    #[inline]
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = usize> + 'a {
        let first = self.first_offset_value();
        self.0.as_ref().iter().map(move |&x| x - first)
    }

    /// Returns an iterator over offset values.
    #[inline]
    pub fn values(&self) -> OffsetValues {
        OffsetValues {
            offset_values: self.0.as_ref(),
        }
    }

    /// Returns an iterator over offset ranges.
    #[inline]
    pub fn ranges(&self) -> Ranges {
        Ranges {
            offset_value_ranges: self.offset_value_ranges(),
            first_offset_value: self.first_offset_value(),
        }
    }
}

impl<'a, O: AsRef<[usize]>> View<'a> for Offsets<O> {
    type Type = Offsets<&'a [usize]>;
    #[inline]
    fn view(&'a self) -> Self::Type {
        Offsets(self.0.as_ref())
    }
}

impl<'a, O: AsMut<[usize]>> ViewMut<'a> for Offsets<O> {
    type Type = Offsets<&'a mut [usize]>;
    #[inline]
    fn view_mut(&'a mut self) -> Self::Type {
        Offsets(self.0.as_mut())
    }
}

//impl<'a, O: AsRef<[usize]>> Viewable for &'a Offsets<O> {
//    type View = Offsets<&'a [usize]>;
//    fn into_view(self) -> Self::View {
//        Offsets(self.0.as_ref())
//    }
//}

impl<O: AsRef<[usize]>> From<O> for Offsets<O> {
    #[inline]
    fn from(offsets: O) -> Self {
        // Note that new checks that offsets has at least one element.
        Offsets::new(offsets)
    }
}

impl<O: AsRef<[usize]>> AsRef<[usize]> for Offsets<O> {
    #[inline]
    fn as_ref(&self) -> &[usize] {
        self.0.as_ref()
    }
}

impl<O: AsMut<[usize]>> AsMut<[usize]> for Offsets<O> {
    #[inline]
    fn as_mut(&mut self) -> &mut [usize] {
        self.0.as_mut()
    }
}

/// A default set of offsets must allocate.
impl Default for Offsets<Vec<usize>> {
    #[inline]
    fn default() -> Self {
        Offsets(vec![0])
    }
}

impl<O: Dummy> Dummy for Offsets<O> {
    #[inline]
    unsafe fn dummy() -> Self {
        Offsets(Dummy::dummy())
    }
}

impl<O: AsRef<[usize]>> Offsets<O> {
    /// Construct a set `Offsets` from a given set of offsets.
    ///
    /// # Panics
    ///
    /// The given `offsets` must be a non-empty collection, otherwise this function will panic.
    #[inline]
    pub fn new(offsets: O) -> Self {
        assert!(!offsets.as_ref().is_empty());
        Offsets(offsets)
    }

    /// An unchecked version of the `new` constructor.
    ///
    /// # Safety
    ///
    /// Calling this function with empty `offsets` may cause undefined behaviour in safe APIs.
    #[inline]
    pub unsafe fn from_raw(offsets: O) -> Self {
        Offsets(offsets)
    }
}

impl<O> Offsets<O> {
    #[inline]
    pub fn into_inner(self) -> O {
        self.0
    }
}

impl<O: AsMut<[usize]>> Offsets<O> {
    /// Moves an offset back by a specified amount.
    ///
    /// This effectively transfers `by` elements to the specified `at` chunk from the preceeding chunk.
    ///
    /// # Panics
    ///
    /// This function panics if `at` is out of bounds.
    ///
    /// If `at` it zero, the beginning of the indexed range is simply extended, but an overflow
    /// panic will be caused if the first offset is moved below zero since offsets are represented
    /// by unsigned integers.
    ///
    /// It is a logic error to move an offset past its preceeding offset because this will break
    /// the monotonicity of the offset sequence, which can cause panics from other function on the
    /// Offsets.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::Offsets;
    /// let mut o = Offsets::new(vec![0, 4, 9]);
    /// o.move_back(1, 2);
    /// assert_eq!(o, vec![0, 2, 9].into());
    /// ```
    #[inline]
    pub fn move_back(&mut self, at: usize, by: usize) {
        let offsets = self.as_mut();
        offsets[at] -= by;
        debug_assert!(at == 0 || offsets[at] >= offsets[at - 1]);
    }
    /// Moves an offset forward by a specified amount.
    ///
    /// This effectively transfers `by` elements to the specified `at` chunk from the succeeding chunk.
    ///
    /// If `at` indexes the last offset, then the indexed range is simply increased.
    ///
    /// # Panics
    ///
    /// This function panics if `at` is out of bounds.
    ///
    /// It is a logic error to move an offset past its succeeding offset because this will break
    /// the monotonicity of the offset sequence, which can cause panics from other function on the
    /// Offsets.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::Offsets;
    /// let mut o = Offsets::new(vec![0, 4, 9]);
    /// o.move_forward(1, 2);
    /// assert_eq!(o, vec![0, 6, 9].into());
    /// ```
    #[inline]
    pub fn move_forward(&mut self, at: usize, by: usize) {
        let offsets = self.as_mut();
        offsets[at] += by;
        debug_assert!(at == offsets.len() - 1 || offsets[at] <= offsets[at + 1]);
    }

    /// Extend the last offset.
    ///
    /// This effectively increases the last chunk size.
    /// This function is the same as `self.move_forward(self.len() - 1, by)`.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::Offsets;
    /// let mut o = Offsets::new(vec![0, 4, 9]);
    /// o.extend_last(2);
    /// assert_eq!(o, vec![0, 4, 11].into());
    /// ```
    #[inline]
    pub fn extend_last(&mut self, by: usize) {
        let offsets = self.as_mut();
        offsets[offsets.len() - 1] += by;
    }

    /// Shrink the last offset.
    ///
    /// This effectively decreases the last chunk size.
    /// This function is the same as `self.move_back(self.len() - 1, by)`.
    ///
    /// # Panics
    ///
    /// It is a logic error to move an offset past its preceeding offset because this will break
    /// the monotonicity of the offset sequence, which can cause panics from other function on the
    /// Offsets.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::Offsets;
    /// let mut o = Offsets::new(vec![0, 4, 9]);
    /// o.shrink_last(2);
    /// assert_eq!(o, vec![0, 4, 7].into());
    /// ```
    #[inline]
    pub fn shrink_last(&mut self, by: usize) {
        let offsets = self.as_mut();
        offsets[offsets.len() - 1] -= by;
        debug_assert!(
            offsets.len() == 1 || offsets[offsets.len() - 1] >= offsets[offsets.len() - 2]
        );
    }
}

impl<O: Push<usize>> Push<usize> for Offsets<O> {
    #[inline]
    fn push(&mut self, item: usize) {
        self.0.push(item);
    }
}

impl<I: std::slice::SliceIndex<[usize]>, O: AsRef<[usize]>> std::ops::Index<I> for Offsets<O> {
    type Output = I::Output;
    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        self.0.as_ref().index(index)
    }
}

impl<I: std::slice::SliceIndex<[usize]>, O: AsRef<[usize]> + AsMut<[usize]>> std::ops::IndexMut<I>
    for Offsets<O>
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.0.as_mut().index_mut(index)
    }
}

impl Clear for Offsets {
    #[inline]
    fn clear(&mut self) {
        self.0.truncate(1);
    }
}

impl<O: std::iter::FromIterator<usize> + AsRef<[usize]>> std::iter::FromIterator<usize>
    for Offsets<O>
{
    #[inline]
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = usize>,
    {
        Offsets::new(O::from_iter(iter))
    }
}

impl<'a> SplitOffsetsAt for Offsets<&'a [usize]> {
    /// Same as `split_offsets_at`, but in addition, return the offset of the middle element
    /// (intersection): this is the value `offsets[mid] - offsets[0]`.
    ///
    /// # Panics
    ///
    /// Calling this function with an empty slice or with `mid` greater than or equal to its length
    /// will cause a panic.
    #[inline]
    fn split_offsets_with_intersection_at(
        self,
        mid: usize,
    ) -> (Offsets<&'a [usize]>, Offsets<&'a [usize]>, usize) {
        let (l, r) = self.split_offsets_at(mid);
        // This is safe since self.0 is not empty, both l and r have at least one element.
        let off = unsafe { *r.0.get_unchecked(0) - *l.0.get_unchecked(0) };
        (l, r, off)
    }

    /// Splits a slice of offsets at the given index into two slices such that each
    /// slice is a valid slice of offsets. This means that the element at index
    /// `mid` is shared between the two output slices.
    ///
    /// # Panics
    ///
    /// Calling this function with an empty slice or with `mid` greater than or equal to its length
    /// will cause a panic.
    #[inline]
    fn split_offsets_at(self, mid: usize) -> (Offsets<&'a [usize]>, Offsets<&'a [usize]>) {
        // Check bounds, and ensure that self is not empty.
        assert!(mid < self.0.len());
        let l = &self.0[..=mid];
        let r = &self.0[mid..];
        (Offsets(l), Offsets(r))
    }
}

impl<O: AsRef<[usize]> + Set> IndexRange for Offsets<O> {
    #[inline]
    unsafe fn index_range_unchecked(&self, range: Range<usize>) -> Range<usize> {
        self.offset_unchecked(range.start)..self.offset_unchecked(range.end)
    }
    /// Return the `[begin..end)` bound of the chunk at the given index.
    #[inline]
    fn index_range(&self, range: Range<usize>) -> Option<Range<usize>> {
        if range.end < self.num_offsets() {
            unsafe { Some(self.index_range_unchecked(range)) }
        } else {
            None
        }
    }
}

impl<'a, O: Get<'a, usize>> GetIndex<'a, Offsets<O>> for usize {
    type Output = O::Output;
    #[inline]
    fn get(self, offsets: &Offsets<O>) -> Option<Self::Output> {
        offsets.0.get(self)
    }
}

impl<'a, O: Get<'a, usize>> GetIndex<'a, Offsets<O>> for &usize {
    type Output = O::Output;
    #[inline]
    fn get(self, offsets: &Offsets<O>) -> Option<Self::Output> {
        offsets.0.get(*self)
    }
}

impl<'a, O: Get<'a, Range<usize>>> GetIndex<'a, Offsets<O>> for Range<usize> {
    type Output = Offsets<O::Output>;
    #[inline]
    fn get(mut self, offsets: &Offsets<O>) -> Option<Self::Output> {
        self.end += 1;
        offsets.0.get(self).map(|offsets| Offsets(offsets))
    }
}

impl<O: Isolate<usize>> IsolateIndex<Offsets<O>> for usize {
    type Output = O::Output;
    #[inline]
    unsafe fn isolate_unchecked(self, offsets: Offsets<O>) -> Self::Output {
        offsets.0.isolate_unchecked(self)
    }
    #[inline]
    fn try_isolate(self, offsets: Offsets<O>) -> Option<Self::Output> {
        offsets.0.try_isolate(self)
    }
}

impl<O: Isolate<Range<usize>>> IsolateIndex<Offsets<O>> for Range<usize> {
    type Output = Offsets<O::Output>;
    #[inline]
    unsafe fn isolate_unchecked(mut self, offsets: Offsets<O>) -> Self::Output {
        self.end += 1;
        Offsets(offsets.0.isolate_unchecked(self))
    }
    #[inline]
    fn try_isolate(mut self, offsets: Offsets<O>) -> Option<Self::Output> {
        self.end += 1;
        offsets.0.try_isolate(self).map(|offsets| Offsets(offsets))
    }
}

impl<O: Truncate> Truncate for Offsets<O> {
    #[inline]
    fn truncate(&mut self, new_len: usize) {
        self.0.truncate(new_len);
    }
}

impl<O: RemovePrefix + Set> RemovePrefix for Offsets<O> {
    /// Remove the first `n` offsets.
    ///
    /// # Panics
    ///
    /// This function will panic if all offsets are removed, which violates the `Offsets` invariant
    /// that there must always be at least one offset.
    #[inline]
    fn remove_prefix(&mut self, n: usize) {
        self.0.remove_prefix(n);
        assert!(!self.0.is_empty());
    }
}

impl<O: IntoOwned> IntoOwned for Offsets<O> {
    type Owned = Offsets<O::Owned>;
    #[inline]
    fn into_owned(self) -> Self::Owned {
        Offsets(self.0.into_owned())
    }
}

impl<O: Reserve> Reserve for Offsets<O> {
    #[inline]
    fn reserve_with_storage(&mut self, n: usize, storage_n: usize) {
        self.0.reserve_with_storage(n, storage_n);
    }
}

impl std::iter::Extend<usize> for Offsets {
    /// Extend this set of offsets with a given iterator of offsets.
    ///
    /// This operation automatically shifts the merged offsets in the iterator
    /// to start from the last offset in `self`.
    ///
    /// Note that there will be 1 less offset added to `self` than produced by
    /// `iter` since the first offset is only used to determine the relative
    /// magnitude of the rest and corresponds to the last offset in `self`.
    fn extend<T: IntoIterator<Item = usize>>(&mut self, iter: T) {
        let mut iter = iter.into_iter();
        if let Some(first) = iter.next() {
            let last = self.last_offset_value();
            self.0.extend(iter.map(|off| off + last - first));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Test for the `split_offset_at` helper function.
    #[test]
    fn split_offset_at_test() {
        // Split in the middle
        let offsets = Offsets(vec![0, 1, 2, 3, 4, 5]);
        let (l, r, off) = offsets.view().split_offsets_with_intersection_at(3);
        assert_eq!(l.0, &[0, 1, 2, 3]);
        assert_eq!(r.0, &[3, 4, 5]);
        assert_eq!(off, 3);

        // Split at the beginning
        let (l, r, off) = offsets.view().split_offsets_with_intersection_at(0);
        assert_eq!(l.0, &[0]);
        assert_eq!(r.0, &[0, 1, 2, 3, 4, 5]);
        assert_eq!(off, 0);

        // Split at the end
        let (l, r, off) = offsets.view().split_offsets_with_intersection_at(5);
        assert_eq!(l.0, &[0, 1, 2, 3, 4, 5]);
        assert_eq!(r.0, &[5]);
        assert_eq!(off, 5);
    }

    /// Check that clearing the offsets keeps exactly one element.
    #[test]
    fn clear() {
        let mut offsets = Offsets(vec![0, 1, 2, 3, 4, 5]);
        offsets.clear();
        assert_eq!(offsets.into_inner(), vec![0]);
    }

    /// Check indexing offsets.
    #[test]
    fn get_offset() {
        let s = Offsets::new(vec![2, 5, 6, 8]);
        assert_eq!(0, s.offset(0));
        assert_eq!(3, s.offset(1));
        assert_eq!(4, s.offset(2));
        assert_eq!(6, s.offset(3));
    }

    #[test]
    fn sizes_iter() {
        let offsets = Offsets(vec![0, 3, 5, 6, 10]);
        assert_eq!(3, offsets.sizes().nth(0).unwrap());
        assert_eq!(2, offsets.sizes().nth(1).unwrap());
        assert_eq!(1, offsets.sizes().nth(2).unwrap());
        assert_eq!(4, offsets.sizes().nth(3).unwrap());
        assert_eq!(None, offsets.sizes().nth(4));
    }

    #[test]
    fn double_ended_sizes_iter() {
        use ExactSizeIterator;
        let offsets = Offsets::new(vec![0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36]);
        let mut iter = offsets.sizes();
        let iter_len = iter.len();
        assert_eq!(iter_len, offsets.num_offsets() - 1);
        assert_eq!(iter_len, iter.count());

        assert_eq!(iter.next().unwrap(), 3);
        assert_eq!(iter.nth(2).unwrap(), 3);
        assert_eq!(iter.next().unwrap(), 4);

        assert_eq!(iter.next_back().unwrap(), 3);
        assert_eq!(iter.nth_back(2).unwrap(), 3);
        assert_eq!(iter.next_back().unwrap(), 4);
        assert_eq!(iter.next().unwrap(), 4);
        assert_eq!(iter.next(), None);

        // The following checks that count is correctly implemented for the sizes iterator.
        let mut count = 0;
        for _ in offsets.sizes() {
            count += 1;
        }
        assert_eq!(iter_len, count);
    }

    #[test]
    fn ranges_iter() {
        let offsets = Offsets(vec![0, 3, 5, 6, 10]);
        assert_eq!(0..3, offsets.ranges().nth(0).unwrap());
        assert_eq!(3..5, offsets.ranges().nth(1).unwrap());
        assert_eq!(5..6, offsets.ranges().nth(2).unwrap());
        assert_eq!(6..10, offsets.ranges().nth(3).unwrap());
        assert_eq!(None, offsets.ranges().nth(4));
    }

    #[test]
    fn double_ended_ranges_iter() {
        use ExactSizeIterator;
        let offsets = Offsets::new(vec![0, 3, 6, 9, 12, 16, 20, 24, 27, 30, 33, 36]);
        let mut iter = offsets.ranges();
        let iter_len = iter.len();
        assert_eq!(iter_len, offsets.num_offsets() - 1);
        assert_eq!(iter_len, iter.count());

        assert_eq!(iter.next().unwrap(), 0..3);
        assert_eq!(iter.nth(2).unwrap(), 9..12);
        assert_eq!(iter.next().unwrap(), 12..16);

        assert_eq!(iter.next_back().unwrap(), 33..36);
        assert_eq!(iter.nth_back(2).unwrap(), 24..27);
        assert_eq!(iter.next_back().unwrap(), 20..24);
        assert_eq!(iter.next().unwrap(), 16..20);
        assert_eq!(iter.next(), None);

        // The following checks that count is correctly implemented for the sizes iterator.
        let mut count = 0;
        for _ in offsets.sizes() {
            count += 1;
        }
        assert_eq!(iter_len, count);
    }

    #[test]
    fn extend_offsets() {
        let mut offsets = Offsets::new(vec![0, 3, 7]);
        let orig_offsets = offsets.clone();
        offsets.extend([]); // Safe and and no panics, nothing happens.
        assert_eq!(offsets, orig_offsets);
        offsets.extend([0]); // Nothing new added.
        assert_eq!(offsets, orig_offsets);
        offsets.extend([0, 1, 10]);
        assert_eq!(offsets, Offsets::new(vec![0, 3, 7, 8, 17]));
    }
}
