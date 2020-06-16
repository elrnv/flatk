use super::*;
use std::convert::{AsMut, AsRef};
use std::ops::Range;

/// A collection of offsets into another collection.
/// This newtype is intended to verify basic invariants about offsets into
/// another collection, namely that the collection is monotonically increasing
/// and non-empty.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Offsets<O = Vec<usize>>(O);

impl<O: Set> Set for Offsets<O> {
    type Elem = O::Elem;
    type Atom = O::Atom;
    /// Total number of offsets stored in this collection.
    ///
    /// This is one more than the number of chunks represented.
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<O: Viewed> Viewed for Offsets<O> {}

impl<O: AsRef<[usize]> + Set> GetOffset for Offsets<O> {
    /// A version of `offset_value` without bounds checking.
    ///
    /// # Safety
    ///
    /// It is assumed that `index` is strictly less than `self.len()`.
    #[inline]
    unsafe fn offset_value_unchecked(&self, index: usize) -> usize {
        *self.0.as_ref().get_unchecked(index)
    }
}

impl<'a> Iterator for Offsets<&'a [usize]> {
    type Item = usize;

    /// Get the next available offset.
    #[inline]
    fn next(&mut self) -> Option<usize> {
        self.0.split_first().map(|(first, rest)| {
            self.0 = rest;
            *first
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.len();
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        ExactSizeIterator::len(&self)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0.get(n).map(|&x| x)
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl DoubleEndedIterator for Offsets<&[usize]> {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        self.0.split_last().map(|(last, rest)| {
            self.0 = rest;
            *last
        })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.0.get(ExactSizeIterator::len(self) - 1 - n).map(|&x| x)
    }
}

impl ExactSizeIterator for Offsets<&[usize]> {}
impl std::iter::FusedIterator for Offsets<&[usize]> {}

#[derive(Copy, Clone)]
pub struct Sizes<'a> {
    offsets: &'a [usize],
}

impl<'a> Iterator for Sizes<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if let &[first, ref rest @ ..] = self.offsets {
            if let &[second, ..] = rest {
                self.offsets = rest;
                return Some(second - first);
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.offsets.len() - 1;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if let (_, &[first, second, ..]) = self.offsets.split_at(n) {
            Some(second - first)
        } else {
            None
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'a> DoubleEndedIterator for Sizes<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        if let &[ref rest @ .., last] = self.offsets {
            if let &[.., second_last] = rest {
                self.offsets = rest;
                return Some(last - second_last);
            }
        }
        None
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if let (_, &[first, second, ..]) = self.offsets.split_at(self.len() - 2 - n) {
            Some(second - first)
        } else {
            None
        }
    }
}

impl ExactSizeIterator for Sizes<'_> {}
impl std::iter::FusedIterator for Sizes<'_> {}

impl<'a> IntoSizes for Offsets<&'a [usize]> {
    type Iter = Sizes<'a>;
    /// Returns an iterator over chunk sizes represented by the stored `Offsets`.
    #[inline]
    fn into_sizes(self) -> Sizes<'a> {
        Sizes { offsets: self.0 }
    }
}

impl<O: AsRef<[usize]>> Offsets<O> {
    /// Returns an iterator over chunk sizes represented by the stored `Offsets`.
    #[inline]
    pub fn sizes(&self) -> Sizes {
        Sizes {
            offsets: self.0.as_ref(),
        }
    }
    /// Returns an iterator over offsets.
    #[inline]
    pub fn iter(&self) -> Offsets<&[usize]> {
        self.view()
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
    #[inline]
    pub fn new(offsets: O) -> Self {
        assert!(!offsets.as_ref().is_empty());
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
    /// Moves an offset back by a specified amount, effectively transferring
    /// elements from the previous chunk to the specified chunk.
    ///
    /// # Panics
    ///
    /// This function panics if `at` is out of bounds or zero.
    #[inline]
    pub(crate) fn move_back(&mut self, at: usize, by: usize) {
        let offsets = self.as_mut();
        assert!(at > 0);
        offsets[at] -= by;
    }
    /// Moves an offset forward by a specified amount, effectively transferring
    /// elements from the previous chunk to the specified chunk.
    ///
    /// # Panics
    ///
    /// This function panics if `at` is out of bounds.
    #[inline]
    pub(crate) fn move_forward(&mut self, at: usize, by: usize) {
        let offsets = self.as_mut();
        offsets[at] += by;
    }

    /// Extend the last offset, which effectively increases the last chunk size.
    /// This function is the same as `self.move_forward(self.len() - 1, by)`.
    #[inline]
    pub(crate) fn extend_last(&mut self, by: usize) {
        let offsets = self.as_mut();
        offsets[offsets.len() - 1] += by;
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
    /// Splits a slice of offsets at the given index into two slices such that each
    /// slice is a valid slice of offsets. This means that the element at index
    /// `mid` is shared between the two output slices. In addition, return the
    /// offset of the middle element: this is the value `offsets[mid] - offsets[0]`.
    ///
    /// # Panics
    ///
    /// Calling this function with an empty slice or with `mid` greater than or equal to its length
    /// will cause a panic.
    #[inline]
    fn split_offsets_at(self, mid: usize) -> (Offsets<&'a [usize]>, Offsets<&'a [usize]>, usize) {
        assert!(!Set::is_empty(&self));
        assert!(mid < self.0.len());
        let l = &self.0[..=mid];
        let r = &self.0[mid..];
        // This is safe since self.0 is not empty, both l and r have at least one element.
        let off = unsafe { *r.get_unchecked(0) - *l.get_unchecked(0) };
        (Offsets(l), Offsets(r), off)
    }
}

impl<O: AsRef<[usize]>> IndexRange for Offsets<O> {
    /// Return the `[begin..end)` bound of the chunk at the given index.
    #[inline]
    fn index_range(&self, range: Range<usize>) -> Option<Range<usize>> {
        let offsets = self.0.as_ref();
        if range.end < offsets.len() {
            let first = unsafe { offsets.get_unchecked(0) };
            let cur = unsafe { offsets.get_unchecked(range.start) };
            let next = unsafe { offsets.get_unchecked(range.end) };
            let begin = cur - first;
            let end = next - first;
            Some(begin..end)
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
    fn try_isolate(self, offsets: Offsets<O>) -> Option<Self::Output> {
        offsets.0.try_isolate(self)
    }
}

impl<O: Isolate<Range<usize>>> IsolateIndex<Offsets<O>> for Range<usize> {
    type Output = Offsets<O::Output>;
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

impl<O: RemovePrefix> RemovePrefix for Offsets<O> {
    #[inline]
    fn remove_prefix(&mut self, n: usize) {
        self.0.remove_prefix(n);
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

#[cfg(test)]
mod tests {
    use super::*;
    /// Test for the `split_offset_at` helper function.
    #[test]
    fn split_offset_at_test() {
        // Split in the middle
        let offsets = Offsets(vec![0, 1, 2, 3, 4, 5]);
        let (l, r, off) = offsets.view().split_offsets_at(3);
        assert_eq!(l.0, &[0, 1, 2, 3]);
        assert_eq!(r.0, &[3, 4, 5]);
        assert_eq!(off, 3);

        // Split at the beginning
        let (l, r, off) = offsets.view().split_offsets_at(0);
        assert_eq!(l.0, &[0]);
        assert_eq!(r.0, &[0, 1, 2, 3, 4, 5]);
        assert_eq!(off, 0);

        // Split at the end
        let (l, r, off) = offsets.view().split_offsets_at(5);
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
}
