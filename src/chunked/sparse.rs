//! This module defines functions specific to `Chunked<Sparse<_>>` types
use super::*;
use crate::Sparse;

impl<'a, S, T, I, O> Chunked<Sparse<S, T, I>, Offsets<O>>
where
    I: Dummy + AsMut<[usize]>,
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
    I: AsRef<[usize]>,
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
        combine: impl FnMut(&mut E::Owned, E),
    ) -> Chunked<Sparse<S::Owned, T>, Offsets>
    where
        <S as View<'a>>::Type: IntoIterator<Item = E>,
        E: IntoOwned,
        S::Owned: Set<Elem = E::Owned> + Default + Reserve + Push<E::Owned>,
    {
        self.pruned(combine, |_, _, _| true)
    }
}

impl<'a, S, T, I, O> Chunked<Sparse<S, T, I>, Offsets<O>>
where
    S: Storage + IntoOwned + Set + View<'a>,
    S::Storage: Set,
    <S as View<'a>>::Type: SplitAt + Dummy + Set,
    T: View<'a> + Set + Clone,
    <T as View<'a>>::Type: Set + Dummy + Clone,
    I: AsRef<[usize]>,
    O: Set + AsRef<[usize]>,
{
    pub fn pruned<E>(
        &'a self,
        mut combine: impl FnMut(&mut E::Owned, E),
        mut keep: impl FnMut(usize, usize, &E::Owned) -> bool,
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

        let mut offsets = Vec::with_capacity(self.chunks.len());
        offsets.push(0);

        for (i, sparse_chunk) in self.iter().enumerate() {
            sparse.extend_pruned(sparse_chunk, &mut combine, |j, e| keep(i, j, e));
            offsets.push(sparse.len());
        }

        // Assemble the output type from constructed parts.
        Chunked::from_offsets(offsets, sparse)
    }
}
