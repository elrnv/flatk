//! This module defines functions specific to `Chunked<Sparse<_>>` types
use super::*;
use crate::select::{AsIndexSlice, AsIndexSliceMut};
use crate::Sparse;

impl<'a, S, T, I, O> Chunked<Sparse<S, T, I>, Offsets<O>>
where
    I: Dummy + AsIndexSliceMut,
    T: Dummy + Set + View<'a>,
    <T as View<'a>>::Type: Set + Clone + Dummy,
    S: Dummy + Set + ViewMut<'a>,
    <S as ViewMut<'a>>::Type: Set + SplitAt + Dummy + PermuteInPlace,
    O: Clone + AsRef<[usize]>,
{
    /// Sort each sparse chunk by the source index.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let sparse = Sparse::from_dim(vec![0,2,1,2,0], 4, vec![1,2,3,4,5]);
    /// let mut chunked = Chunked::from_sizes(vec![3,2], sparse);
    /// chunked.sort_chunks_by_index();
    /// assert_eq!(chunked.storage(), &[1,3,2,5,4]);
    /// assert_eq!(chunked.data().indices(), &[0,1,2,0,2]);
    /// ```
    pub fn sort_chunks_by_index(&'a mut self) {
        let mut indices = vec![0; self.data.selection.indices.as_mut().len()];
        let mut seen = vec![false; indices.len()];
        let mut chunked_workspace = Chunked {
            chunks: self.chunks.clone(),
            data: (indices.as_mut_slice(), seen.as_mut_slice()),
        };

        for ((permutation, seen), mut chunk) in chunked_workspace.iter_mut().zip(self.iter_mut()) {
            // Initialize permutation
            (0..permutation.len())
                .zip(permutation.iter_mut())
                .for_each(|(i, out)| *out = i);

            // Sort the permutation according to selection indices.
            chunk.selection.indices.sort_indices(permutation);

            // Apply the result of the sort (i.e. the permutation) to the whole chunk.
            chunk.permute_in_place(permutation, seen);
        }
    }
}

impl<'a, S, T, I, O> Chunked<Sparse<S, T, I>, Offsets<O>>
where
    S: Storage + IntoOwned + Set + View<'a>,
    S::Storage: Set,
    <S as View<'a>>::Type: SplitAt + Dummy + Set,
    T: View<'a> + Set + Clone,
    <T as View<'a>>::Type: Set + Dummy + Clone,
    I: AsIndexSlice,
    O: Set + AsRef<[usize]>,
{
    /// Combine elements in each sorted chunk with the same index using the given function.
    ///
    /// Assuming that the chunks are sorted by index, this function will combine adjacent
    /// elements with the same index into one element.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let sparse = Sparse::from_dim(vec![0,2,1,1,2,0,2], 4, vec![1,2,3,4,5,6,7]);
    /// let mut chunked = Chunked::from_sizes(vec![4,3], sparse);
    /// chunked.sort_chunks_by_index();
    /// let compressed = chunked.compressed(|a, b| *a += *b);
    /// assert_eq!(compressed.view().offsets().into_inner(), &[0,3,5]);
    /// assert_eq!(compressed.view().storage(), &[1,7,2,6,12]);
    /// assert_eq!(compressed.view().data().indices(), &[0,1,2,0,2]);
    /// ```
    pub fn compressed<E>(
        &'a self,
        mut combine: impl FnMut(&mut E::Owned, E),
    ) -> Chunked<Sparse<S::Owned, T>, Offsets>
    where
        <S as View<'a>>::Type: IntoIterator<Item = E>,
        E: IntoOwned,
        S::Owned: Set<Elem = E::Owned> + Default + Reserve + Push<E::Owned>,
    {
        self.pruned(&mut combine, |_, _, _| true, |_, _| {})
    }
}

impl<'a, S, T, I, O> Chunked<Sparse<S, T, I>, Offsets<O>>
where
    S: Storage + IntoOwned + Set + View<'a>,
    S::Storage: Set,
    <S as View<'a>>::Type: SplitAt + Dummy + Set,
    T: View<'a> + Set + Clone,
    <T as View<'a>>::Type: Set + Dummy + Clone,
    I: AsIndexSlice,
    O: Set + AsRef<[usize]>,
{
    /// Prune elements according to a given predicate and combine them in each
    /// sorted chunk with the same index.
    ///
    /// Assuming that the chunks are sorted by index, this function will combine adjacent
    /// elements with the same index into one element.
    ///
    /// This is a more general version of `compressed` that allows you to prune unwanted elements.
    /// In addition the `combine` and `keep` functions get an additional target
    /// index where each element will be written to.
    ///
    /// # Examples
    ///
    /// A simple example.
    ///
    /// ```
    /// use flatk::*;
    /// let sparse = Sparse::from_dim(vec![0,2,1,1,2,0,2], 4, vec![1.0, 2.0, 0.1, 0.01, 5.0, 0.001, 7.0]);
    /// let mut chunked = Chunked::from_sizes(vec![4,3], sparse);
    /// chunked.sort_chunks_by_index();
    /// let pruned = chunked.pruned(|a, b| *a += *b, |_, _, &val| val > 0.01, |_,_| {});
    /// assert_eq!(pruned.view().offsets().into_inner(), &[0,3,4]);
    /// assert_eq!(pruned.view().storage(), &[1.0, 0.11, 2.0, 12.0]); // 0.001 is pruned.
    /// assert_eq!(pruned.view().data().indices(), &[0,1,2,2]);
    /// ```
    ///
    /// The following example extends on the previous example but shows how one
    /// may construct a mapping from original elements to the pruned output.
    ///
    /// ```
    /// use flatk::*;
    /// let indices = vec![0, 2, 1, 1, 2, 0, 2];
    /// let num_indices = indices.len();
    /// let sparse = Sparse::from_dim(indices, 4, vec![1.0, 2.0, 0.1, 0.01, 5.0, 0.001, 7.0]);
    /// let mut chunked = Chunked::from_sizes(vec![4,3], sparse);
    /// chunked.sort_chunks_by_index();
    /// let mut mapping = vec![None; num_indices];
    /// let pruned = chunked.pruned(|a, b| {
    ///     *a += *b
    /// }, |_, _, &val| {
    ///     val > 0.01
    /// }, |src, dst| mapping[src] = Some(dst));
    ///
    /// // As before, the resulting structure is pruned.
    /// assert_eq!(pruned.view().offsets().into_inner(), &[0,3,4]);
    /// assert_eq!(pruned.view().storage(), &[1.0, 0.11, 2.0, 12.0]); // 0.001 is pruned.
    /// assert_eq!(pruned.view().data().indices(), &[0,1,2,2]);
    /// assert_eq!(mapping, vec![Some(0), Some(1), Some(1), Some(2), None, Some(3), Some(3)]);
    /// ```
    pub fn pruned<E>(
        &'a self,
        mut combine: impl FnMut(&mut E::Owned, E),
        mut keep: impl FnMut(usize, usize, &E::Owned) -> bool,
        mut map: impl FnMut(usize, usize),
    ) -> Chunked<Sparse<S::Owned, T>, Offsets>
    where
        <S as View<'a>>::Type: IntoIterator<Item = E>,
        E: IntoOwned,
        S::Owned: Set<Elem = E::Owned> + Default + Reserve + Push<E::Owned>,
    {
        // Initialize and allocate all output types.
        let mut data: <S as IntoOwned>::Owned = Default::default();
        data.reserve_with_storage(self.data.len(), self.storage().len());
        let mut indices = Vec::new();
        indices.reserve(self.data.selection.len());
        let mut sparse = Sparse::new(
            Select::new(indices, self.data.selection.target.clone()),
            data,
        );

        let mut offsets = Vec::with_capacity(self.chunks.num_offsets());
        offsets.push(0);

        for (i, sparse_chunk) in self.iter().enumerate() {
            sparse.extend_pruned(
                sparse_chunk,
                |_pos, a, b| combine(a, b),
                |j, e| keep(i, j, e),
                |src, dst| map(src + self.offset(i), dst),
            );
            offsets.push(sparse.len());
        }

        // Assemble the output type from constructed parts.
        Chunked::from_offsets(offsets, sparse)
    }
}
