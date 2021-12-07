#[cfg(feature = "rayon")]
mod par_iter;

use super::*;
use std::convert::{AsMut, AsRef};

/// A `Sparse` data set `S` where the sparsity pattern is given by `I` as select
/// indices into a larger range.
///
/// For example we can represent a sparse vector by assigning values to a selection of indices:
///
/// ```rust
/// use flatk::{Sparse, Get, View};
///
/// let values = vec![1.0, 2.0, 3.0, 4.0];
/// let sparse_vector = Sparse::from_dim(vec![0,5,10,100], 1000, values);
/// let sparse_vector_view = sparse_vector.view();
///
/// assert_eq!(sparse_vector_view.at(0), (0, &1.0));
/// assert_eq!(sparse_vector_view.at(1), (5, &2.0));
/// assert_eq!(sparse_vector_view.at(2), (10, &3.0));
/// assert_eq!(sparse_vector_view.at(3), (100, &4.0));
/// assert_eq!(sparse_vector_view.selection.target, ..1000);
/// ```
///
/// In this scenario, the target set is just the range `0..1000`, however in general this can be any
/// data set, which makes `Sparse` an implementation of a one-to-one mapping or a directed graph
/// with disjoint source and target node sets.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sparse<S, T = std::ops::RangeTo<usize>, I = Vec<usize>> {
    pub selection: Select<T, I>,
    pub source: S,
}

/// A borrowed view of a sparse collection.
pub type SparseView<'a, S, T> = Sparse<S, T, &'a [usize]>;

impl<S, I> Sparse<S, std::ops::RangeTo<usize>, I>
where
    S: Set,
    I: AsIndexSlice,
{
    /// Create a sparse collection from the given set of `indices`, a
    /// `dim`ension and a set of `values`.
    ///
    /// The corresponding sparse collection will represent a collection
    /// of size `dim` which stores only the given `values` at the specified
    /// `indices`. Note that `dim` may be smaller than `values.len()`, in
    /// which case a position in the sparse data structure may contain multiple
    /// values.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5,6];
    /// let sparse = Sparse::from_dim(vec![0,2,0,2,0,3], 4, v.as_slice());
    ///
    /// // The iterator traverses only non-vacant elements.
    /// let mut iter = sparse.iter(); // Returns (position, source, target) triplets
    /// assert_eq!(Some((0, &1, 0)), iter.next());
    /// assert_eq!(Some((2, &2, 2)), iter.next());
    /// assert_eq!(Some((0, &3, 0)), iter.next());
    /// assert_eq!(Some((2, &4, 2)), iter.next());
    /// assert_eq!(Some((0, &5, 0)), iter.next());
    /// assert_eq!(Some((3, &6, 3)), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    #[inline]
    pub fn from_dim(indices: I, dim: usize, values: S) -> Self {
        Sparse::new(Select::new(indices, ..dim), values)
    }
}

impl<S, T, I> Sparse<S, T, I>
where
    S: Set,
    T: Set,
    I: AsIndexSlice,
{
    /// The most general constructor for a sparse collection taking a selection
    /// of values and their corresponding data.
    ///
    /// # Panics
    /// This function will panic if `selection` and `source` have different sizes.
    #[inline]
    pub fn new(selection: Select<T, I>, source: S) -> Self {
        Self::validate(Sparse { selection, source })
    }

    /// Panics if the number of indices doesn't equal to the number of elements in the source data set.
    #[inline]
    fn validate(self) -> Self {
        assert_eq!(self.source.len(), self.selection.len());
        self
    }
}

impl<S: Set, T, I> Sparse<S, T, I> {
    /// Extend the current sparse collection with a pruned and compressed version of the given
    /// sparse collection, `other`.
    ///
    /// Each element is in the original sparse collection is guaranteed to be
    /// passed to exactly one of `keep` or `combine`.
    ///
    /// The `keep` function will get the combined value of all consecutively
    /// overlapping elements, and the source index of the first element in the
    /// consecutive group.
    ///
    /// The `map` function allows the caller to map between original `other`
    /// sparse collection and the newly populated collection `self`. The first
    /// parameter passed into `map` is the index in the output sparse array where
    /// each element is being considered for insertion. The second parameter is
    /// the position of each element in the output sparse structure. Note that `map`
    /// is not called on pruned elements.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5,6];
    ///
    /// // Create a sparse vector with overlapping elements:
    /// // [0] -> 1            [0] -> 5
    /// // [1] ->              [1] ->
    /// // [2] -> 2 + 3 + 4    [2] ->
    /// // [3] ->              [3] -> 6
    /// let sparse = Sparse::from_dim(vec![0,2,2,2,0,3], 4, v.as_slice());
    ///
    /// // Create an empty sparse vector.
    /// let mut compressed = Sparse::from_dim(Vec::new(), 4, Vec::new());
    ///
    /// // Transfer the elements from `sparse` into `compressed` while resolving the consecutive overlaps:
    /// compressed.extend_pruned(sparse, |_pos, a, b| *a += *b, |_pos, _val| true, |_src_idx, _dst_idx, | {});
    /// let mut iter = compressed.iter(); // Returns (position, source, target) triplets.
    /// assert_eq!(Some((0, &1, 0)), iter.next());
    /// assert_eq!(Some((2, &9, 2)), iter.next());
    /// assert_eq!(Some((0, &5, 0)), iter.next());
    /// assert_eq!(Some((3, &6, 3)), iter.next());
    /// assert_eq!(None, iter.next());
    /// // Note: 1 and 5 are not merged because they are not consecutive.
    /// ```
    pub fn extend_pruned<S2, T2, I2, B>(
        &mut self,
        other: Sparse<S2, T2, I2>,
        mut combine: impl FnMut(usize, &mut B::Owned, B),
        mut keep: impl FnMut(usize, &B::Owned) -> bool,
        mut map: impl FnMut(usize, usize),
    ) where
        S2: IntoIterator<Item = B>,
        I2: AsIndexSlice,
        B: IntoOwned,
        Self: Push<(usize, B::Owned)>,
    {
        let mut it = other
            .selection
            .index_iter()
            .cloned()
            .zip(other.source.into_iter())
            .enumerate();
        if let Some((mut prev_src_idx, (mut prev_idx, prev))) = it.next() {
            let mut elem = prev.into_owned();

            for (src_idx, (idx, cur)) in it {
                if prev_idx != idx {
                    if keep(prev_idx, &elem) {
                        map(prev_src_idx, self.len());
                        self.push((prev_idx, elem));
                    }
                    elem = cur.into_owned();
                    prev_idx = idx;
                    prev_src_idx = src_idx;
                } else {
                    map(src_idx, self.len());
                    combine(idx, &mut elem, cur);
                }
            }
            if keep(prev_idx, &elem) {
                map(prev_src_idx, self.len()); // Map the last element.
                self.push((prev_idx, elem)); // Push the last element.
            }
        }
    }
}

/*
impl<S, T, I> Sparse<S, T, I>
where
    S: Set + Default + Push<<S as Set>::Elem>,
    <S as Set>::Elem: Default,
{
    /// Convert this sparse collection into its dense representation.
    pub fn dense(&self) -> S {
        // TODO: This can be optimized with a trait that allows pre-allocating memory.
        let mut dense = S::default();
        for (i, v) in self.iter() {
            while dense.len() < i {
                dense.push(Default::default());
            }
            dense.push(v);
        }

        dense
    }
}
*/

impl<S, T> Extend<(usize, <S as Set>::Elem)> for Sparse<S, T>
where
    S: Set + Extend<<S as Set>::Elem>,
{
    #[inline]
    fn extend<It: IntoIterator<Item = (usize, <S as Set>::Elem)>>(&mut self, iter: It) {
        let Sparse {
            source,
            selection: Select { indices, .. },
        } = self;
        let iter = iter.into_iter();
        indices.reserve(iter.size_hint().0);
        source.extend(iter.map(|(idx, elem)| {
            indices.push(idx);
            elem
        }));
    }
}

impl<S, T, I, A> Push<(usize, A)> for Sparse<S, T, I>
where
    S: Set<Elem = A> + Push<A>,
    I: Push<usize>,
{
    #[inline]
    fn push(&mut self, (index, elem): (usize, A)) {
        self.source.push(elem);
        self.selection.indices.push(index);
    }
}

impl<'a, S, T, I> Sparse<S, T, I> {
    /// Get a reference to the underlying source data.
    #[inline]
    pub fn source(&self) -> &S {
        &self.source
    }
    /// Get a mutable reference to the underlying source data.
    #[inline]
    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }
    /// Get a reference to the underlying selection.
    #[inline]
    pub fn selection(&self) -> &Select<T, I> {
        &self.selection
    }

    #[inline]
    pub fn selection_mut(&mut self) -> &mut Select<T, I> {
        &mut self.selection
    }

    /// Get a reference to the underlying indices.
    #[inline]
    pub fn indices(&self) -> &I {
        &self.selection.indices
    }

    #[inline]
    pub fn indices_mut(&mut self) -> &mut I {
        &mut self.selection.indices
    }
}

// Note to self:
// To enable a collection to be chunked, we need to implement:
// Set, View, SplitAt
// For mutability we also need ViewMut,
// For UniChunked we need:
// Set, Vew, ReinterpretSet (this needs to be refined)

// Required for `Chunked` and `UniChunked` subsets.
impl<S: Set, T, I> Set for Sparse<S, T, I> {
    type Elem = (usize, S::Elem);
    type Atom = S::Atom;
    /// Get the length of this sparse collection.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5];
    /// let sparse = Sparse::from_dim(vec![0,2,2,1,1], 3, v.as_slice());
    /// assert_eq!(5, sparse.len());
    /// ```
    #[inline]
    fn len(&self) -> usize {
        self.source.len()
    }
}

// Required for `Chunked` and `UniChunked` subsets.
impl<'a, S, T, I> View<'a> for Sparse<S, T, I>
where
    S: View<'a>,
    T: View<'a>,
    I: AsIndexSlice,
{
    type Type = Sparse<S::Type, T::Type, &'a [usize]>;
    #[inline]
    fn view(&'a self) -> Self::Type {
        Sparse {
            selection: self.selection.view(),
            source: self.source.view(),
        }
    }
}

impl<'a, S, T, I> ViewMut<'a> for Sparse<S, T, I>
where
    S: Set + ViewMut<'a>,
    T: Set + View<'a>,
    I: AsMut<[usize]>,
{
    type Type = Sparse<S::Type, T::Type, &'a mut [usize]>;
    #[inline]
    fn view_mut(&'a mut self) -> Self::Type {
        let Sparse {
            selection: Select {
                indices,
                ref target,
            },
            source,
        } = self;
        Sparse {
            selection: Select {
                indices: indices.as_mut(),
                target: target.view(),
            },
            source: source.view_mut(),
        }
    }
}

// This impl enables `Chunked` `Subset`s
impl<S, T, I> SplitAt for Sparse<S, T, I>
where
    S: Set + SplitAt,
    T: Set + Clone,
    I: SplitAt,
{
    #[inline]
    fn split_at(self, mid: usize) -> (Self, Self) {
        let Sparse { selection, source } = self;
        let (selection_l, selection_r) = selection.split_at(mid);
        let (source_l, source_r) = source.split_at(mid);
        (
            Sparse {
                selection: selection_l,
                source: source_l,
            },
            Sparse {
                selection: selection_r,
                source: source_r,
            },
        )
    }
}

impl<S: RemovePrefix, T, I: RemovePrefix> RemovePrefix for Sparse<S, T, I> {
    #[inline]
    fn remove_prefix(&mut self, n: usize) {
        self.selection.remove_prefix(n);
        self.source.remove_prefix(n);
    }
}

/*
 * Indexing operators for convenience. Users familiar with indexing by `usize`
 * may find these implementations convenient.
 */

impl<'a, S, T, I> GetIndex<'a, Sparse<S, T, I>> for usize
where
    I: AsIndexSlice,
    S: Get<'a, usize>,
{
    type Output = (usize, <S as Get<'a, usize>>::Output);

    #[inline]
    fn get(self, sparse: &Sparse<S, T, I>) -> Option<Self::Output> {
        let Sparse { selection, source } = sparse;
        let selected = selection.indices.as_ref();
        source.get(self).map(|item| (selected[self], item))
    }
}

impl<'a, S, T, I> GetIndex<'a, Sparse<S, T, I>> for &usize
where
    I: AsIndexSlice,
    S: Get<'a, usize>,
{
    type Output = (usize, <S as Get<'a, usize>>::Output);

    #[inline]
    fn get(self, sparse: &Sparse<S, T, I>) -> Option<Self::Output> {
        GetIndex::get(*self, sparse)
    }
}

impl<S, T, I> IsolateIndex<Sparse<S, T, I>> for usize
where
    I: Isolate<usize>,
    <I as Isolate<usize>>::Output: std::borrow::Borrow<usize>,
    S: Isolate<usize>,
    T: Isolate<usize>,
{
    type Output = (
        <I as Isolate<usize>>::Output,
        <S as Isolate<usize>>::Output,
        <T as Isolate<usize>>::Output,
    );

    #[inline]
    unsafe fn isolate_unchecked(self, sparse: Sparse<S, T, I>) -> Self::Output {
        let Sparse { selection, source } = sparse;
        let item = source.isolate_unchecked(self);
        let (idx, target) = selection.isolate_unchecked(self);
        (idx, item, target)
    }

    #[inline]
    fn try_isolate(self, sparse: Sparse<S, T, I>) -> Option<Self::Output> {
        let Sparse { selection, source } = sparse;
        let item = source.try_isolate(self)?;
        // SAFETY: selection must be the same size as source.
        let (idx, target) = unsafe { selection.isolate_unchecked(self) };
        Some((idx, item, target))
    }
}

impl<S, T, I> IsolateIndex<Sparse<S, T, I>> for std::ops::Range<usize>
where
    S: Isolate<std::ops::Range<usize>>,
    I: Isolate<std::ops::Range<usize>>,
{
    type Output = Sparse<S::Output, T, I::Output>;

    #[inline]
    unsafe fn isolate_unchecked(self, sparse: Sparse<S, T, I>) -> Self::Output {
        let Sparse { selection, source } = sparse;
        let source = source.isolate_unchecked(self.clone());
        Sparse {
            selection: selection.isolate_unchecked(self),
            source,
        }
    }

    #[inline]
    fn try_isolate(self, sparse: Sparse<S, T, I>) -> Option<Self::Output> {
        let Sparse { selection, source } = sparse;
        let source = source.try_isolate(self.clone())?;
        // SAFETY: selection must be the same size as source.
        Some(Sparse {
            selection: unsafe { selection.isolate_unchecked(self) },
            source,
        })
    }
}

//impl_isolate_index_for_static_range!(impl<S, T, I> for Sparse<S, T, I>);

/*
 * Iteration
 */

impl<'a, S, T> IntoIterator for SparseView<'a, S, T>
where
    S: SplitFirst + SplitAt + Dummy + Set,
{
    type Item = (usize, S::First);
    type IntoIter = SparseIter<std::iter::Cloned<std::slice::Iter<'a, usize>>, S>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        SparseIter {
            indices: self.selection.indices.iter().cloned(),
            source: self.source,
        }
    }
}

pub struct SparseIter<I, S> {
    indices: I,
    source: S,
}

impl<I, S> Iterator for SparseIter<I, S>
where
    S: SplitFirst + SplitAt + Dummy + Set,
    I: ExactSizeIterator<Item = usize>,
{
    type Item = (usize, S::First);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: A sparse dummy is a valid sparse set.
        let source_slice = std::mem::replace(&mut self.source, unsafe { Dummy::dummy() });
        source_slice.split_first().map(|(first, rest)| {
            self.source = rest;
            // We know that sparse has at least one element, no need to check again.
            let first_idx = self.indices.next().unwrap();
            // SAFETY: We know there is at least one element in beginning.
            (first_idx, first)
        })
    }
    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        // SAFETY: A sparse dummy is a valid sparse set.
        let source_slice = std::mem::replace(&mut self.source, unsafe { Dummy::dummy() });
        self.indices.nth(n).map(|nth_idx| {
            let (_, rest) = source_slice.split_at(n);
            // SAFETY: source_slice is known to have at least n elements since indices does.
            let (nth, rest) = unsafe { rest.split_first_unchecked() };
            self.source = rest;
            (nth_idx, nth)
        })
    }
}

impl<I, S> ExactSizeIterator for SparseIter<I, S> where Self: Iterator {}

impl<I, S> DoubleEndedIterator for SparseIter<I, S>
where
    S: SplitFirst + SplitAt + Dummy + Set,
    I: ExactSizeIterator + DoubleEndedIterator<Item = usize>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let source_slice = std::mem::replace(&mut self.source, unsafe { Dummy::dummy() });
        let len = source_slice.len();
        if len < 1 {
            return None;
        }

        let (prefix, end) = source_slice.split_at(len - 1);

        self.source = prefix;
        // We know that sparse has at least one element, no need to check again.
        let last_idx = self.indices.next_back().unwrap();
        // SAFETY: We know there is at least one element in end.
        Some((last_idx, unsafe { end.split_first_unchecked().0 }))
    }
    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        // SAFETY: A sparse dummy is a valid sparse set.
        let source_slice = std::mem::replace(&mut self.source, unsafe { Dummy::dummy() });
        self.indices.nth_back(n).map(|nth_idx| {
            let (beginning, end) = source_slice.split_at(n);
            // SAFETY: source_slice is known to have at least n elements since indices does.
            let (nth, _) = unsafe { end.split_first_unchecked() };
            self.source = beginning;
            (nth_idx, nth)
        })
    }
}

impl<'a, S, T, I> Sparse<S, T, I>
where
    S: View<'a>,
    <S as View<'a>>::Type: Set + IntoIterator,
    T: Set + Get<'a, usize> + View<'a>,
    I: AsIndexSlice,
{
    #[inline]
    pub fn iter(
        &'a self,
    ) -> impl Iterator<
        Item = (
            usize,
            <<S as View<'a>>::Type as IntoIterator>::Item,
            <T as Get<'a, usize>>::Output,
        ),
    > {
        self.selection
            .iter()
            .zip(self.source.view().into_iter())
            .map(|((i, t), s)| (i, s, t))
    }
}

impl<'a, S, T, I> Sparse<S, T, I>
where
    S: View<'a>,
    <S as View<'a>>::Type: Set + IntoIterator,
{
    #[inline]
    pub fn source_iter(&'a self) -> <<S as View<'a>>::Type as IntoIterator>::IntoIter {
        self.source.view().into_iter()
    }
}

impl<'a, S, T, I> Sparse<S, T, I>
where
    I: AsIndexSlice,
{
    #[inline]
    pub fn index_iter(&'a self) -> std::iter::Cloned<std::slice::Iter<'a, usize>> {
        self.selection.index_iter().cloned()
    }
}

impl<'a, S, T, I> Sparse<S, T, I>
where
    S: View<'a>,
    <S as View<'a>>::Type: Set + IntoIterator,
    I: AsIndexSlice,
{
    #[inline]
    pub fn indexed_source_iter(
        &'a self,
    ) -> std::iter::Zip<
        std::iter::Cloned<std::slice::Iter<'a, usize>>,
        <<S as View<'a>>::Type as IntoIterator>::IntoIter,
    > {
        self.selection
            .index_iter()
            .cloned()
            .zip(self.source.view().into_iter())
    }
}

/// A mutable iterator over the source elements in `S`
impl<'a, S, T, I> Sparse<S, T, I>
where
    S: ViewMut<'a>,
    <S as ViewMut<'a>>::Type: Set + IntoIterator,
{
    #[inline]
    pub fn source_iter_mut(&'a mut self) -> <<S as ViewMut<'a>>::Type as IntoIterator>::IntoIter {
        self.source.view_mut().into_iter()
    }
}

/// A mutable iterator can only iterate over the source elements in `S` and not
/// the target elements in `T` since we would need scheduling to modify
/// potentially overlapping mutable references.
impl<'a, S, T, I> Sparse<S, T, I>
where
    S: ViewMut<'a>,
    <S as ViewMut<'a>>::Type: Set + IntoIterator,
    I: AsIndexSlice,
{
    #[inline]
    pub fn indexed_source_iter_mut(
        &'a mut self,
    ) -> impl Iterator<Item = (usize, <<S as ViewMut<'a>>::Type as IntoIterator>::Item)> {
        self.selection
            .index_iter()
            .cloned()
            .zip(self.source.view_mut().into_iter())
    }
}

/// A mutable iterator can only iterate over the source elements in `S` and not
/// the target elements in `T` since we would need scheduling to modify
/// potentially overlapping mutable references.
impl<'a, S, T, I> Sparse<S, T, I>
where
    S: ViewMut<'a>,
    <S as ViewMut<'a>>::Type: Set + IntoIterator,
    I: AsMut<[usize]>,
{
    #[inline]
    pub fn iter_mut(
        &'a mut self,
    ) -> std::iter::Zip<
        std::slice::IterMut<'a, usize>,
        <<S as ViewMut<'a>>::Type as IntoIterator>::IntoIter,
    > {
        self.selection
            .index_iter_mut()
            .zip(self.source.view_mut().into_iter())
    }
}

/// Mutably iterate over the selected indices.
impl<'a, S, T, I> Sparse<S, T, I>
where
    S: View<'a>,
    <S as View<'a>>::Type: Set + IntoIterator,
    I: AsMut<[usize]>,
{
    #[inline]
    pub fn index_iter_mut(
        &'a mut self,
    ) -> impl Iterator<Item = (&'a mut usize, <<S as View<'a>>::Type as IntoIterator>::Item)> {
        self.selection
            .index_iter_mut()
            .zip(self.source.view().into_iter())
    }
}

impl<'a, S, T, I> ViewIterator<'a> for Sparse<S, T, I>
where
    S: View<'a>,
    <S as View<'a>>::Type: Set + IntoIterator,
{
    type Item = <<S as View<'a>>::Type as IntoIterator>::Item;
    type Iter = <<S as View<'a>>::Type as IntoIterator>::IntoIter;

    #[inline]
    fn view_iter(&'a self) -> Self::Iter {
        self.source_iter()
    }
}

impl<'a, S, T, I> ViewMutIterator<'a> for Sparse<S, T, I>
where
    S: ViewMut<'a>,
    <S as ViewMut<'a>>::Type: Set + IntoIterator,
{
    type Item = <<S as ViewMut<'a>>::Type as IntoIterator>::Item;
    type Iter = <<S as ViewMut<'a>>::Type as IntoIterator>::IntoIter;

    #[inline]
    fn view_mut_iter(&'a mut self) -> Self::Iter {
        self.source_iter_mut()
    }
}

impl_atom_iterators_recursive!(impl<S, T, I> for Sparse<S, T, I> { source });

impl<S: Dummy, T: Dummy, I: Dummy> Dummy for Sparse<S, T, I> {
    #[inline]
    unsafe fn dummy() -> Self {
        Sparse {
            selection: Dummy::dummy(),
            source: Dummy::dummy(),
        }
    }
}

impl<S: Truncate, T, I: Truncate> Truncate for Sparse<S, T, I> {
    #[inline]
    fn truncate(&mut self, new_len: usize) {
        self.selection.truncate(new_len);
        self.source.truncate(new_len);
    }
}

impl<S: Clear, T, I: Clear> Clear for Sparse<S, T, I> {
    #[inline]
    fn clear(&mut self) {
        self.source.clear();
        self.selection.clear();
    }
}

impl<S, O, T, I> Sparse<Chunked<S, O>, T, I>
where
    S: Set + Truncate,
    O: AsRef<[usize]> + GetOffset + Truncate,
    I: Truncate,
{
    /// Remove any empty elements (indexed chunks) at the end of the collection and any unindexed
    /// data past the last offset.
    /// Return the number of elements removed.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let mut s = Sparse::from_dim(vec![0,1,2], 3, Chunked::from_sizes(vec![1,3,2], vec![1,2,3,4,5,6]));
    /// assert_eq!(3, s.len());
    ///
    /// // Transferring the last two elements past the indexed stack.
    /// // This creates an empty chunk at the end.
    /// s.source_mut().transfer_forward(2, 2);
    /// assert_eq!(6, s.storage().len());
    /// assert_eq!(3, s.len());
    ///
    /// s.trim(); // remove unindexed elements.
    /// assert_eq!(4, s.storage().len());
    /// ```
    #[inline]
    pub fn trim(&mut self) -> usize {
        let num_removed = self.source.trim();
        self.selection.truncate(self.source.len());
        num_removed
    }
}

/*
 * Conversions
 */

/// Pass through the conversion for structure type `Subset`.
impl<S: StorageInto<U>, T, I, U> StorageInto<U> for Sparse<S, T, I> {
    type Output = Sparse<S::Output, T, I>;
    #[inline]
    fn storage_into(self) -> Self::Output {
        Sparse {
            source: self.source.storage_into(),
            selection: self.selection,
        }
    }
}

impl<S: MapStorage<Out>, T, I, Out> MapStorage<Out> for Sparse<S, T, I> {
    type Input = S::Input;
    type Output = Sparse<S::Output, T, I>;
    #[inline]
    fn map_storage<F: FnOnce(Self::Input) -> Out>(self, f: F) -> Self::Output {
        Sparse {
            source: self.source.map_storage(f),
            selection: self.selection,
        }
    }
}

impl<S: IntoStorage, T, I> IntoStorage for Sparse<S, T, I> {
    type StorageType = S::StorageType;
    /// Convert the sparse set into its raw storage representation.
    #[inline]
    fn into_storage(self) -> Self::StorageType {
        self.source.into_storage()
    }
}

impl<T: Clone, S: CloneWithStorage<U>, I: Clone, U> CloneWithStorage<U> for Sparse<S, T, I> {
    type CloneType = Sparse<S::CloneType, T, I>;
    #[inline]
    fn clone_with_storage(&self, storage: U) -> Self::CloneType {
        Sparse {
            selection: self.selection.clone(),
            source: self.source.clone_with_storage(storage),
        }
    }
}

/*
 * Storage Access
 */

impl<'a, S: StorageView<'a>, T, I> StorageView<'a> for Sparse<S, T, I> {
    type StorageView = S::StorageView;
    /// Return a view to the underlying storage type of source data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11,12];
    /// let s0 = Chunked3::from_flat(v.clone());
    /// let s1 = Sparse::from_dim(vec![0, 2, 2, 0], 4, s0.clone());
    /// assert_eq!(s1.storage_view(), v.as_slice());
    /// ```
    #[inline]
    fn storage_view(&'a self) -> Self::StorageView {
        self.source.storage_view()
    }
}

impl<S: Storage, T, I> Storage for Sparse<S, T, I> {
    type Storage = S::Storage;
    /// Return an immutable reference to the underlying storage type of source data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11,12];
    /// let s0 = Chunked3::from_flat(v.clone());
    /// let s1 = Sparse::from_dim(vec![0, 2, 2, 0], 4, s0.clone());
    /// assert_eq!(s1.storage(), &v);
    /// ```
    #[inline]
    fn storage(&self) -> &Self::Storage {
        self.source.storage()
    }
}

impl<S: StorageMut, T, I> StorageMut for Sparse<S, T, I> {
    /// Return a mutable reference to the underlying storage type of source data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,5,6,7,8,9,10,11,12];
    /// let mut s0 = Chunked3::from_flat(v.clone());
    /// let mut s1 = Sparse::from_dim(vec![0, 2, 2, 0], 4, s0.clone());
    /// assert_eq!(s1.storage_mut(), &mut v);
    /// ```
    #[inline]
    fn storage_mut(&mut self) -> &mut Self::Storage {
        self.source.storage_mut()
    }
}

impl<S: PermuteInPlace, T, I: PermuteInPlace> PermuteInPlace for Sparse<S, T, I> {
    #[inline]
    fn permute_in_place(&mut self, permutation: &[usize], seen: &mut [bool]) {
        let Sparse {
            selection: Select { indices, .. },
            source,
        } = self;

        indices.permute_in_place(permutation, seen);
        seen.iter_mut().for_each(|x| *x = false);
        source.permute_in_place(permutation, seen);
    }
}

/*
 * Sparse uniformly chunked types
 */

impl<S: ChunkSize, T, I> ChunkSize for Sparse<S, T, I> {
    #[inline]
    fn chunk_size(&self) -> usize {
        self.source.chunk_size()
    }
}

/*
 * Convert views to owned types
 */

impl<S: IntoOwned, T: IntoOwned, I: IntoOwned> IntoOwned for Sparse<S, T, I> {
    type Owned = Sparse<S::Owned, T::Owned, I::Owned>;

    #[inline]
    fn into_owned(self) -> Self::Owned {
        Sparse {
            selection: self.selection.into_owned(),
            source: self.source.into_owned(),
        }
    }
}

impl<S, T, I> IntoOwnedData for Sparse<S, T, I>
where
    S: IntoOwnedData,
{
    type OwnedData = Sparse<S::OwnedData, T, I>;
    #[inline]
    fn into_owned_data(self) -> Self::OwnedData {
        Sparse {
            selection: self.selection,
            source: self.source.into_owned_data(),
        }
    }
}

impl<S: Reserve, T, I: Reserve> Reserve for Sparse<S, T, I> {
    #[inline]
    fn reserve_with_storage(&mut self, n: usize, storage_n: usize) {
        self.selection.reserve_with_storage(n, storage_n);
        self.source.reserve_with_storage(n, storage_n);
    }
}

/*
 * Impls for uniformly chunked sparse types
 */

impl<S, T, I, M> UniChunkable<M> for Sparse<S, T, I> {
    type Chunk = Sparse<S, T, I>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extend_pruned() {
        // Empty test
        let empty = Sparse::from_dim(Vec::new(), 4, Vec::new());
        let mut compressed = empty.clone();
        compressed.extend_pruned(empty.view(), |_, a, b| *a += *b, |_, _| true, |_, _| {});
        assert!(compressed.is_empty());

        // The basic tests from the example
        let v = vec![1, 2, 3, 4, 5, 6];
        let sparse = Sparse::from_dim(vec![0, 2, 2, 2, 0, 3], 4, v.as_slice());
        let mut compressed = empty.clone();
        compressed.extend_pruned(sparse.view(), |_, a, b| *a += *b, |__, _| true, |_, _| {});
        let mut iter = compressed.iter(); // Returns (position, source, target) pairs
        assert_eq!(Some((0, &1, 0)), iter.next());
        assert_eq!(Some((2, &9, 2)), iter.next());
        assert_eq!(Some((0, &5, 0)), iter.next());
        assert_eq!(Some((3, &6, 3)), iter.next());
        assert_eq!(None, iter.next());
    }
}
