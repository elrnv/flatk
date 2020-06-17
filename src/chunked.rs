mod clumped_offsets;
mod offsets;
mod sorted_chunks;
mod uniform;

use super::*;
pub use clumped_offsets::*;
pub use offsets::*;
pub use sorted_chunks::*;
use std::convert::AsRef;
pub use uniform::*;

/// A partitioning of the collection `S` into distinct chunks.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Chunked<S, O = Offsets> {
    /// This can be either offsets of a uniform chunk size, if
    /// chunk size is specified at compile time.
    pub chunks: O,
    pub data: S,
}

pub type ChunkedView<'a, S> = Chunked<S, Offsets<&'a [usize]>>;

/*
 * The following traits provide abstraction over different types of offset collections.
 */

pub trait SplitOffsetsAt
where
    Self: Sized,
{
    fn split_offsets_with_intersection_at(self, mid: usize) -> (Self, Self, usize);
    fn split_offsets_at(self, mid: usize) -> (Self, Self);
}

pub trait IndexRange {
    fn index_range(&self, range: std::ops::Range<usize>) -> Option<std::ops::Range<usize>>;
}

pub trait IntoSizes {
    type Iter: Iterator<Item = usize>;
    fn into_sizes(self) -> Self::Iter;
}

pub trait IntoValues {
    type Iter: Iterator<Item = usize>;
    fn into_values(self) -> Self::Iter;
}

pub trait GetOffset: Set {
    /// A version of `offset_value` without bounds checking.
    unsafe fn offset_value_unchecked(&self, index: usize) -> usize;

    /// Get the length of the chunk at the given index.
    ///
    /// Returns the distance between offsets at `index` and `index + 1`.
    ///
    /// # Panics
    ///
    /// This funciton will panic if `chunk_index+1` is greater than or equal to `self.len()`.
    #[inline]
    fn chunk_len(&self, chunk_index: usize) -> usize {
        assert!(chunk_index + 1 < self.len(), "Offset index out of bounds");
        // SAFETY: The length is checked above.
        unsafe {
            self.offset_value_unchecked(chunk_index + 1) - self.offset_value_unchecked(chunk_index)
        }
    }

    /// Return the raw value corresponding to the offset at the given index.
    ///
    /// Using `first_*` and `last_*` variants for getting first and last offsets are preferred
    /// since they don't require bounds checking.
    ///
    /// # Panics
    ///
    /// This function panics if `index` is greater than or equal to `self.len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let s = Offsets::new(vec![2,5,6,8]);
    /// assert_eq!(2, s.offset_value(0));
    /// assert_eq!(5, s.offset_value(1));
    /// assert_eq!(6, s.offset_value(2));
    /// assert_eq!(8, s.offset_value(3));
    /// ```
    #[inline]
    fn offset_value(&self, index: usize) -> usize {
        assert!(index < self.len(), "Offset index out of bounds");
        // SAFETY: just checked the bound.
        unsafe { self.offset_value_unchecked(index) }
    }

    /// Returns the offset at the given index with respect to (minus) the first offset.
    /// This function returns the total length of `data` if `index` is equal to
    /// `self.len()`.
    ///
    /// # Panics
    ///
    /// This function panics if `index` is greater than or equal to `self.len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let s = Offsets::new(vec![2,5,6,8]);
    /// assert_eq!(0, s.offset(0));
    /// assert_eq!(3, s.offset(1));
    /// assert_eq!(4, s.offset(2));
    /// assert_eq!(6, s.offset(3));
    /// ```
    #[inline]
    fn offset(&self, index: usize) -> usize {
        self.offset_value(index) - self.first_offset_value()
    }

    /// A version of `offset` without bounds checking.
    ///
    /// # Safety
    ///
    /// It is assumed that `index` is strictly less than `self.len()`.
    #[inline]
    unsafe fn offset_unchecked(&self, index: usize) -> usize {
        self.offset_value_unchecked(index) - self.first_offset_value()
    }

    /// Get the last offset.
    ///
    /// Since offsets are never empty by construction, this will always work.
    #[inline]
    fn last_offset(&self) -> usize {
        // SAFETY: Offsets are never empty
        unsafe { self.offset_unchecked(self.len() - 1) }
    }

    /// Get the first offset.
    ///
    /// This should always return 0.
    #[inline]
    fn first_offset(&self) -> usize {
        0
    }

    /// Get the raw value corresponding to the last offset.
    #[inline]
    fn last_offset_value(&self) -> usize {
        // SAFETY: Offsets are never empty
        unsafe { self.offset_value_unchecked(self.len() - 1) }
    }

    /// Get the raw value corresponding to the first offset.
    #[inline]
    fn first_offset_value(&self) -> usize {
        // SAFETY: Offsets are never empty
        unsafe { self.offset_value_unchecked(0) }
    }
}

pub trait BinarySearch<T> {
    /// Binary search for a given element.
    ///
    /// The semantics of this function are identical to Rust's `std::slice::binary_search`.
    fn binary_search(&self, x: &T) -> Result<usize, usize>;
}

/*
 * End of offset traits
 */

/// `Clumped` is a variation of `Chunked` that compactly represents equidistant offsets as
/// "clumps", hence the name.
///
/// In order for this type to compose with other container decorators, the clumped offsets must be
/// declumped where necessary to enable efficient iteration. For this reason composition may have
/// some overhead.
pub type Clumped<S> = Chunked<S, ClumpedOffsets>;

/// A view of a `Clumped` collection.
pub type ClumpedView<'a, S> = Chunked<S, ClumpedOffsets<&'a [usize]>>;

impl<S, O> Chunked<S, O> {
    /// Get a immutable reference to the underlying data.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5,6];
    /// let s = Chunked::from_offsets(vec![0,3,4,6], v.clone());
    /// assert_eq!(&v, s.data());
    /// ```
    pub fn data(&self) -> &S {
        &self.data
    }
    /// Get a mutable reference to the underlying data.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,5,6];
    /// let mut s = Chunked::from_offsets(vec![0,3,4,6], v.clone());
    /// v[2] = 100;
    /// s.data_mut()[2] = 100;
    /// assert_eq!(&v, s.data());
    /// ```
    pub fn data_mut(&mut self) -> &mut S {
        &mut self.data
    }
}

impl<S: Set> Chunked<S> {
    /// Construct a `Chunked` collection of elements from a set of `sizes` that
    /// determine the number of elements in each chunk. The sum of the sizes
    /// must be equal to the length of the given `data`.
    ///
    /// # Panics
    ///
    /// This function will panic if the sum of all given sizes is greater than
    /// `data.len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let s = Chunked::from_sizes(vec![3,1,2], vec![1,2,3,4,5,6]);
    /// let mut iter = s.iter();
    /// assert_eq!(vec![1,2,3], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![4], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5,6], iter.next().unwrap().to_vec());
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn from_sizes(sizes: impl AsRef<[usize]>, data: S) -> Self {
        Self::from_sizes_impl(sizes.as_ref(), data)
    }

    #[inline]
    fn from_sizes_impl(sizes: &[usize], data: S) -> Self {
        assert_eq!(sizes.iter().sum::<usize>(), data.len());

        let mut offsets = Vec::with_capacity(sizes.len() + 1);
        offsets.push(0);
        offsets.extend(sizes.iter().scan(0, |prev_off, &x| {
            *prev_off += x;
            Some(*prev_off)
        }));

        Chunked {
            chunks: offsets.into(),
            data,
        }
    }
}

impl<S: Set> Clumped<S> {
    /// Construct a `Clumped` collection of elements from a set of `sizes` and `counts` that
    /// determine the number of elements in each chunk. The length of `sizes` must be equal to the
    /// the length of `counts`. Each element in `sizes` corresponds to chunk size, while the
    /// corresponding element in `counts` tells how many times this chunk size is repeated.
    ///
    /// The dot product between `sizes` and `counts` must be equal to the length of the given
    /// `data`.
    ///
    /// # Panics
    ///
    /// This function will panic if `sizes` and `counts` have different lengths or
    /// if the dot product between `sizes` and `counts` is not equal to `data.len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let s = Clumped::from_sizes_and_counts(vec![3,2], vec![2,1], vec![1,2,3,4,5,6,7,8]);
    /// let mut iter = s.iter();
    /// assert_eq!(&[1, 2, 3][..], iter.next().unwrap());
    /// assert_eq!(&[4, 5, 6][..], iter.next().unwrap());
    /// assert_eq!(&[7, 8][..], iter.next().unwrap());
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn from_sizes_and_counts(
        sizes: impl AsRef<[usize]>,
        counts: impl AsRef<[usize]>,
        data: S,
    ) -> Self {
        Self::from_sizes_and_counts_impl(sizes.as_ref(), counts.as_ref(), data)
    }

    #[inline]
    fn from_sizes_and_counts_impl(sizes: &[usize], counts: &[usize], data: S) -> Self {
        assert_eq!(sizes.len(), counts.len());
        assert_eq!(
            sizes
                .iter()
                .zip(counts.iter())
                .map(|(s, c)| s * c)
                .sum::<usize>(),
            data.len()
        );

        let mut clump_offsets = Vec::with_capacity(sizes.len() + 1);
        let mut offsets = Vec::with_capacity(sizes.len() + 1);
        clump_offsets.push(0);
        offsets.push(0);

        let mut prev_off = 0;
        let mut prev_clump_off = 0;
        for (s, c) in sizes.iter().zip(counts.iter()) {
            prev_clump_off += c;
            prev_off += s * c;
            offsets.push(prev_off);
            clump_offsets.push(prev_clump_off);
        }

        Chunked {
            chunks: ClumpedOffsets::new(clump_offsets, offsets),
            data,
        }
    }
}

impl<S: Set, O: AsRef<[usize]>> Chunked<S, Offsets<O>> {
    /// Construct a `Chunked` collection of elements given a collection of
    /// offsets into `S`. This is the most efficient constructor for creating
    /// variable sized chunks, however it is also the most error prone.
    ///
    /// # Panics
    ///
    /// The absolute value of `offsets` is not significant, however their
    /// relative quantities are. More specifically, if `x` is the first offset,
    /// then the last element of offsets must always be `data.len() + x`.
    /// This also implies that `offsets` cannot be empty. This function panics
    /// if any one of these invariants isn't satisfied.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let s = Chunked::from_offsets(vec![0,3,4,6], vec![1,2,3,4,5,6]);
    /// let mut iter = s.iter();
    /// assert_eq!(vec![1,2,3], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![4], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5,6], iter.next().unwrap().to_vec());
    /// assert_eq!(None, iter.next());
    /// ```
    #[inline]
    pub fn from_offsets(offsets: O, data: S) -> Self {
        let offsets_ref = offsets.as_ref();
        assert_eq!(
            *offsets_ref.last().unwrap(),
            data.len() + *offsets_ref.first().unwrap()
        );
        Chunked {
            chunks: Offsets::new(offsets),
            data,
        }
    }
}

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

impl<S: Set, O> Chunked<S, O> {
    /// Convert this `Chunked` into its inner representation, which consists of
    /// a collection of offsets (first output) along with the underlying data
    /// storage type (second output).
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let data = vec![1,2,3,4,5,6];
    /// let offsets = vec![0,3,4,6];
    /// let s = Chunked::from_offsets(offsets.clone(), data.clone());
    /// assert_eq!(s.into_inner(), (chunked::Offsets::new(offsets), data));
    /// ```
    #[inline]
    pub fn into_inner(self) -> (O, S) {
        let Chunked { chunks, data } = self;
        (chunks, data)
    }

    /// This function mutably borrows the inner structure of the chunked collection.
    #[inline]
    pub fn as_inner_mut(&mut self) -> (&mut O, &mut S) {
        let Chunked { chunks, data } = self;
        (chunks, data)
    }
}

impl<S, O> Chunked<S, O> {
    #[inline]
    pub fn offsets(&self) -> &O {
        &self.chunks
    }
}

impl<S, O> Chunked<S, O>
where
    O: GetOffset,
{
    /// Return the offset into `data` of the element at the given index.
    /// This function returns the total length of `data` if `index` is equal to
    /// `self.len()`.
    ///
    /// # Panics
    ///
    /// This function panics if `index` is larger than `self.len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let s = Chunked::from_offsets(vec![2,5,6,8], vec![1,2,3,4,5,6]);
    /// assert_eq!(0, s.offset(0));
    /// assert_eq!(3, s.offset(1));
    /// assert_eq!(4, s.offset(2));
    /// ```
    #[inline]
    pub fn offset(&self, index: usize) -> usize {
        self.chunks.offset(index)
    }

    /// Return the raw offset value of the element at the given index.
    ///
    /// # Panics
    ///
    /// This function panics if `index` is larger than `self.len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let s = Chunked::from_offsets(vec![2,5,6,8], vec![1,2,3,4,5,6]);
    /// assert_eq!(2, s.offset_value(0));
    /// assert_eq!(5, s.offset_value(1));
    /// assert_eq!(6, s.offset_value(2));
    /// ```
    #[inline]
    pub fn offset_value(&self, index: usize) -> usize {
        self.chunks.offset_value(index)
    }

    /// Get the length of the chunk at the given index.
    /// This is equivalent to `self.view().at(chunk_index).len()`.
    #[inline]
    pub fn chunk_len(&self, chunk_index: usize) -> usize {
        self.chunks.chunk_len(chunk_index)
    }
}

impl<S, O> Chunked<S, Offsets<O>>
where
    O: Set + AsRef<[usize]> + AsMut<[usize]>,
{
    /// Move a number of elements from a chunk at the given index to the
    /// following chunk. If the last chunk is selected, then the transferred
    /// elements are effectively removed.
    ///
    /// This operation is efficient and only performs one write on a single
    /// element in an array of `usize`.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5];
    /// let mut c = Chunked::from_sizes(vec![3,2], v);
    /// let mut c_iter = c.iter();
    /// assert_eq!(Some(&[1,2,3][..]), c_iter.next());
    /// assert_eq!(Some(&[4,5][..]), c_iter.next());
    /// assert_eq!(None, c_iter.next());
    ///
    /// // Transfer 2 elements from the first chunk to the next.
    /// c.transfer_forward(0, 2);
    /// let mut c_iter = c.iter();
    /// assert_eq!(Some(&[1][..]), c_iter.next());
    /// assert_eq!(Some(&[2,3,4,5][..]), c_iter.next());
    /// assert_eq!(None, c_iter.next());
    /// ```
    #[inline]
    pub fn transfer_forward(&mut self, chunk_index: usize, num_elements: usize) {
        self.chunks.move_back(chunk_index + 1, num_elements);
    }

    /// Like `transfer_forward` but specify the number of elements to keep
    /// instead of the number of elements to transfer in the chunk at
    /// `chunk_index`.
    #[inline]
    pub fn transfer_forward_all_but(&mut self, chunk_index: usize, num_elements_to_keep: usize) {
        let num_elements_to_transfer = self.chunk_len(chunk_index) - num_elements_to_keep;
        self.transfer_forward(chunk_index, num_elements_to_transfer);
    }
}

impl<S, O> Chunked<S, Offsets<O>>
where
    O: AsRef<[usize]> + AsMut<[usize]>,
    S: RemovePrefix,
{
    /// Move a number of elements from a chunk at the given index to the
    /// previous chunk. If the first chunk is selected, then the transferred
    /// elements are explicitly removed, which may cause reallocation if the underlying storage
    /// type manages memory.
    ///
    /// This operation is efficient and only performs one write on a single
    /// element in an array of `usize`, unless a reallocation is triggered.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5];
    /// let mut c = Chunked::from_sizes(vec![3,2], v);
    /// let mut c_iter = c.iter();
    /// assert_eq!(Some(&[1,2,3][..]), c_iter.next());
    /// assert_eq!(Some(&[4,5][..]), c_iter.next());
    /// assert_eq!(None, c_iter.next());
    ///
    /// // Transfer 2 elements from the first chunk to the next.
    /// c.transfer_backward(1, 1);
    /// let mut c_iter = c.iter();
    /// assert_eq!(Some(&[1,2,3,4][..]), c_iter.next());
    /// assert_eq!(Some(&[5][..]), c_iter.next());
    /// assert_eq!(None, c_iter.next());
    /// ```
    #[inline]
    pub fn transfer_backward(&mut self, chunk_index: usize, num_elements: usize) {
        self.chunks.move_forward(chunk_index, num_elements);
        if chunk_index == 0 {
            // Truncate data from the front to re-establish a valid chunked set.
            self.data.remove_prefix(num_elements);
        }
    }
}

impl<S: Default, O: Default> Chunked<S, O> {
    /// Construct an empty `Chunked` type.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<S> Chunked<S>
where
    S: Set + Default + ExtendFromSlice<Item = <S as Set>::Elem>, //Push<<S as Set>::Elem>,
{
    /// Construct a `Chunked` `Vec` from a nested `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let s = Chunked::<Vec<_>>::from_nested_vec(vec![vec![1,2,3],vec![4],vec![5,6]]);
    /// let mut iter = s.iter();
    /// assert_eq!(vec![1,2,3], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![4], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5,6], iter.next().unwrap().to_vec());
    /// assert_eq!(None, iter.next());
    /// ```
    #[inline]
    pub fn from_nested_vec(nested_data: Vec<Vec<<S as Set>::Elem>>) -> Self {
        nested_data.into_iter().collect()
    }

    ///// Construct a `Chunked` `Vec` of characters from a `Vec` of `String`s.
    /////
    ///// # Example
    /////
    ///// ```
    ///// use flatk::*;
    ///// let words = Chunked::<Vec<_>>::from_string_vec(vec!["Hello", "World"]);
    ///// let mut iter = s.iter();
    ///// assert_eq!("Hello", iter.next().unwrap().iter().cloned().collect::<String>());
    ///// assert_eq!("World", iter.next().unwrap().iter().cloned().collect::<String>());
    ///// assert_eq!(None, iter.next());
    ///// ```
    //pub fn from_string_vec(nested_data: Vec<Vec<<S as Set>::Elem>>) -> Self {
    //    nested_data.into_iter().collect()
    //}
}

impl<S, O> Set for Chunked<S, O>
where
    S: Set,
    O: Set,
{
    type Elem = Vec<S::Elem>;
    type Atom = S::Atom;

    /// Get the number of elements in a `Chunked`.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let s = Chunked::from_offsets(vec![0,3,4,6], vec![1,2,3,4,5,6]);
    /// assert_eq!(3, s.len());
    /// ```
    #[inline]
    fn len(&self) -> usize {
        self.chunks.len() - 1
    }
}

impl<S, O> Chunked<S, O>
where
    S: Truncate + Set,
    O: GetOffset,
{
    /// Remove any unused data past the last offset.
    /// Return the number of elements removed.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let mut s = Chunked::from_sizes(vec![1,3,2], vec![1,2,3,4,5,6]);
    /// assert_eq!(3, s.len());
    ///
    /// // Transferring the last two elements past the indexed stack.
    /// // This creates a zero sized chunk at the end.
    /// s.transfer_forward(2, 2);
    /// assert_eq!(6, s.data().len());
    /// assert_eq!(3, s.len());
    ///
    /// s.trim_data(); // Remove unindexed elements.
    /// assert_eq!(4, s.data().len());
    /// ```
    #[inline]
    pub fn trim_data(&mut self) -> usize {
        debug_assert!(self.chunks.len() > 0);
        let last_offset = self.chunks.last_offset();
        let num_removed = self.data.len() - last_offset;
        self.data.truncate(last_offset);
        debug_assert_eq!(self.data.len(), last_offset);
        num_removed
    }
}

impl<S, O> Chunked<S, O>
where
    S: Truncate + Set,
    O: AsRef<[usize]> + GetOffset + Truncate,
{
    /// Remove any empty chunks at the end of the collection and any unindexed
    /// data past the last offset.
    /// Return the number of chunks removed.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let mut s = Chunked::from_sizes(vec![1,3,2], vec![1,2,3,4,5,6]);
    /// assert_eq!(3, s.len());
    ///
    /// // Transferring the last two elements past the indexed stack.
    /// // This creates an empty chunk at the end.
    /// s.transfer_forward(2, 2);
    /// assert_eq!(6, s.data().len());
    /// assert_eq!(3, s.len());
    ///
    /// s.trim(); // Remove unindexed elements.
    /// assert_eq!(4, s.data().len());
    /// ```
    pub fn trim(&mut self) -> usize {
        let num_offsets = self.chunks.len();
        debug_assert!(num_offsets > 0);
        let last_offset = self.chunks.last_offset();
        // Count the number of identical offsets from the end.
        let num_empty = self
            .chunks
            .as_ref()
            .iter()
            .rev()
            .skip(1) // skip the actual last offset
            .take_while(|&&offset| offset == last_offset)
            .count();

        self.chunks.truncate(num_offsets - num_empty);
        self.trim_data();
        num_empty
    }
}

impl<S: Truncate, O> Truncate for Chunked<S, O>
where
    O: GetOffset,
{
    fn truncate(&mut self, new_len: usize) {
        self.data
            .truncate(self.chunks.last_offset_value() - self.chunks.offset_value(new_len));
    }
}

impl<S, O, L> Push<L> for Chunked<S, O>
where
    S: Set + ExtendFromSlice<Item = <S as Set>::Elem>,
    L: AsRef<[<S as Set>::Elem]>,
    O: Push<usize>,
{
    /// Push a slice of elements onto this `Chunked`.
    ///
    /// # Examples
    ///
    /// ```
    /// use flatk::*;
    /// let mut s = Chunked::<Vec<usize>>::from_offsets(vec![0,1,4], vec![0,1,2,3]);
    /// s.push(vec![4,5]);
    /// let v1 = s.view();
    /// let mut view1_iter = v1.into_iter();
    /// assert_eq!(Some(&[0][..]), view1_iter.next());
    /// assert_eq!(Some(&[1,2,3][..]), view1_iter.next());
    /// assert_eq!(Some(&[4,5][..]), view1_iter.next());
    /// assert_eq!(None, view1_iter.next());
    /// ```
    ///
    /// ```
    /// use flatk::*;
    /// let mut s = Chunked::from_offsets(vec![0,3,5], vec![1,2,3,4,5]);
    /// assert_eq!(2, s.len());
    /// s.push(&[1,2]);
    /// assert_eq!(3, s.len());
    /// ```
    #[inline]
    fn push(&mut self, element: L) {
        self.data.extend_from_slice(element.as_ref());
        self.chunks.push(self.data.len());
    }
}

impl<S, O> Chunked<S, O>
where
    S: Set + ExtendFromSlice<Item = <S as Set>::Elem>,
    O: Push<usize>,
{
    /// Push a slice of elements onto this `Chunked`.
    /// This can be more efficient than pushing from an iterator.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let mut s = Chunked::from_offsets(vec![0,3,5], vec![1,2,3,4,5]);
    /// assert_eq!(2, s.len());
    /// s.push_slice(&[1,2]);
    /// assert_eq!(3, s.len());
    /// ```
    #[inline]
    pub fn push_slice(&mut self, element: &[<S as Set>::Elem]) {
        self.data.extend_from_slice(element);
        self.chunks.push(self.data.len());
    }
}

impl<S, O> Chunked<S, O>
where
    S: Set,
    O: Push<usize>,
{
    /// Push a chunk using an iterator over chunk elements.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let mut s = Chunked::from_offsets(vec![0,3,5], vec![1,2,3,4,5]);
    /// assert_eq!(2, s.len());
    /// s.push_iter(std::iter::repeat(100).take(4));
    /// assert_eq!(3, s.len());
    /// assert_eq!(&[100; 4][..], s.view().at(2));
    /// ```
    #[inline]
    pub fn push_iter<I: IntoIterator>(&mut self, iter: I)
    where
        S: Extend<I::Item>,
    {
        self.data.extend(iter);
        self.chunks.push(self.data.len());
    }
}

impl<S, O> Chunked<S, Offsets<O>>
where
    S: Set + Extend<<S as Set>::Elem>,
    O: AsMut<[usize]>,
{
    /// Extend the last chunk with the given iterator.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let mut s = Chunked::from_offsets(vec![0,3,5], vec![1,2,3,4,5]);
    /// assert_eq!(2, s.len());
    /// s.extend_last(std::iter::repeat(100).take(2));
    /// assert_eq!(2, s.len());
    /// assert_eq!(&[4, 5, 100, 100][..], s.view().at(1));
    /// ```
    #[inline]
    pub fn extend_last<I: IntoIterator<Item = <S as Set>::Elem>>(&mut self, iter: I) {
        let init_len = self.data.len();
        self.data.extend(iter);
        self.chunks.extend_last(self.data.len() - init_len);
    }
}

impl<S, O> IntoOwned for Chunked<S, O>
where
    S: IntoOwned,
    O: IntoOwned,
{
    type Owned = Chunked<S::Owned, O::Owned>;

    #[inline]
    fn into_owned(self) -> Self::Owned {
        Chunked {
            chunks: self.chunks.into_owned(),
            data: self.data.into_owned(),
        }
    }
}

impl<S, O> IntoOwnedData for Chunked<S, O>
where
    S: IntoOwnedData,
{
    type OwnedData = Chunked<S::OwnedData, O>;
    #[inline]
    fn into_owned_data(self) -> Self::OwnedData {
        let Chunked { chunks, data } = self;
        Chunked {
            chunks,
            data: data.into_owned_data(),
        }
    }
}

// NOTE: There is currently no way to split ownership of a Vec without
// allocating. For this reason we opt to use a slice and defer allocation to
// a later step when the results may be collected into another Vec. This saves
// an extra allocation. We could make this more righteous with a custom
// allocator.
impl<'a, S, O> std::iter::FromIterator<&'a [<S as Set>::Elem]> for Chunked<S, O>
where
    S: Set + ExtendFromSlice<Item = <S as Set>::Elem> + Default,
    <S as Set>::Elem: 'a,
    O: Default + Push<usize>,
{
    /// Construct a `Chunked` collection from an iterator over immutable slices.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let v = [&[1,2,3][..], &[4][..], &[5,6][..]];
    /// let s: Chunked::<Vec<_>> = v.iter().cloned().collect();
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter.next());
    /// assert_eq!(Some(&[4][..]), iter.next());
    /// assert_eq!(Some(&[5,6][..]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    #[inline]
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a [<S as Set>::Elem]>,
    {
        let mut s = Chunked::default();
        for i in iter {
            s.push_slice(i);
        }
        s
    }
}

// For convenience we also implement a `FromIterator` trait for building from
// nested `Vec`s, however as mentioned in the note above, this is typically
// inefficient because it relies on intermediate allocations. This is acceptable
// during initialization, for instance.
impl<S> std::iter::FromIterator<Vec<<S as Set>::Elem>> for Chunked<S>
where
    S: Set + Default + ExtendFromSlice<Item = <S as Set>::Elem>, // + Push<<S as Set>::Elem>,
{
    /// Construct a `Chunked` from an iterator over `Vec` types.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// use std::iter::FromIterator;
    /// let s = Chunked::<Vec<_>>::from_iter(vec![vec![1,2,3],vec![4],vec![5,6]].into_iter());
    /// let mut iter = s.iter();
    /// assert_eq!(vec![1,2,3], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![4], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5,6], iter.next().unwrap().to_vec());
    /// assert_eq!(None, iter.next());
    /// ```
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Vec<<S as Set>::Elem>>,
    {
        let mut s = Chunked::default();
        for i in iter {
            s.push(i);
        }
        s
    }
}

/*
 * Indexing
 */

impl<'a, S, O> GetIndex<'a, Chunked<S, O>> for usize
where
    S: Set + View<'a> + Get<'a, std::ops::Range<usize>, Output = <S as View<'a>>::Type>,
    O: IndexRange,
{
    type Output = S::Output;

    /// Get an element of the given `Chunked` collection.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let v = vec![0, 1, 4, 6];
    /// let data = (1..=6).collect::<Vec<_>>();
    /// let s = Chunked::from_offsets(v.as_slice(), data.view());
    /// assert_eq!(Some(&[1][..]), s.get(0));
    /// assert_eq!(Some(&[2,3,4][..]), s.get(1));
    /// assert_eq!(Some(&[5,6][..]), s.get(2));
    /// ```
    #[inline]
    fn get(self, chunked: &Chunked<S, O>) -> Option<Self::Output> {
        let Chunked { ref chunks, data } = chunked;
        chunks
            .index_range(self..self + 1)
            .and_then(|index_range| data.get(index_range))
    }
}

impl<'a, S, O> GetIndex<'a, Chunked<S, O>> for std::ops::Range<usize>
where
    S: Set + View<'a> + Get<'a, std::ops::Range<usize>, Output = <S as View<'a>>::Type>,
    O: IndexRange + Get<'a, std::ops::Range<usize>>,
{
    type Output = Chunked<S::Output, O::Output>;

    /// Get a `[begin..end)` subview of the given `Chunked` collection.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let data = (1..=6).collect::<Vec<_>>();
    /// let offsets = vec![1, 2, 5, 7]; // Offsets don't have to start at 0
    /// let s = Chunked::from_offsets(offsets.as_slice(), data.view());
    /// let v = s.get(1..3).unwrap();
    /// assert_eq!(Some(&[2,3,4][..]), v.get(0));
    /// assert_eq!(Some(&[5,6][..]), v.get(1));
    /// ```
    #[inline]
    fn get(self, chunked: &Chunked<S, O>) -> Option<Self::Output> {
        assert!(self.start <= self.end);
        let Chunked { data, ref chunks } = chunked;
        chunks.index_range(self.clone()).and_then(|index_range| {
            data.get(index_range).map(|data| Chunked {
                chunks: chunks.get(self).unwrap(),
                data,
            })
        })
    }
}

impl<S, O> IsolateIndex<Chunked<S, O>> for usize
where
    S: Set + Isolate<std::ops::Range<usize>>,
    O: IndexRange,
{
    type Output = S::Output;

    /// Isolate a single chunk of the given `Chunked` collection.
    #[inline]
    fn try_isolate(self, chunked: Chunked<S, O>) -> Option<Self::Output> {
        let Chunked { chunks, data } = chunked;
        chunks
            .index_range(self..self + 1)
            .and_then(|index_range| data.try_isolate(index_range))
    }
}

impl<S, O> IsolateIndex<Chunked<S, O>> for std::ops::Range<usize>
where
    S: Set + Isolate<std::ops::Range<usize>>,
    <S as Isolate<std::ops::Range<usize>>>::Output: Set,
    O: IndexRange + Isolate<std::ops::Range<usize>>,
{
    type Output = Chunked<S::Output, O::Output>;

    /// Isolate a `[begin..end)` range of the given `Chunked` collection.
    #[inline]
    fn try_isolate(self, chunked: Chunked<S, O>) -> Option<Self::Output> {
        assert!(self.start <= self.end);
        let Chunked { data, chunks } = chunked;
        chunks
            .index_range(self.clone())
            .and_then(move |index_range| {
                data.try_isolate(index_range).map(move |data| Chunked {
                    chunks: chunks.try_isolate(self).unwrap(),
                    data,
                })
            })
    }
}

impl_isolate_index_for_static_range!(impl<S, O> for Chunked<S, O>);

//impl<S, O, I> Isolate<I> for Chunked<S, O>
//where
//    I: IsolateIndex<Self>,
//{
//    type Output = I::Output;
//    /// Isolate a sub-collection from this `Chunked` collection according to the
//    /// given range. If the range is a single index, then a single chunk
//    /// is returned instead.
//    ///
//    /// # Examples
//    ///
//    /// ```
//    /// use flatk::*;
//    /// let mut v = vec![1,2,3,4,0,0,7,8,9,10,11];
//    /// let mut s = Chunked::from_offsets(vec![0,3,4,6,9,11], v.view_mut());
//    ///
//    /// s.view_mut().try_isolate(2).unwrap().copy_from_slice(&[5,6]);  // Single index
//    /// assert_eq!(*s.data(), vec![1,2,3,4,5,6,7,8,9,10,11].as_slice());
//    /// ```
//    fn try_isolate(self, range: I) -> Option<I::Output> {
//        range.try_isolate(self)
//    }
//}

impl<T, O> std::ops::Index<usize> for Chunked<Vec<T>, O>
where
    O: std::ops::Index<usize, Output = usize>,
{
    type Output = <[T] as std::ops::Index<std::ops::Range<usize>>>::Output;

    /// Get reference to a chunk at the given index.
    ///
    /// Note that this works for `Chunked` collections that are themselves NOT `Chunked`, since a
    /// chunk of a doubly `Chunked` collection is itself `Chunked`, which cannot be represented by
    /// a single borrow. For more complex indexing use the `get` method provided by the `Get`
    /// trait.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11];
    /// let s = Chunked::from_offsets(vec![0,3,4,6,9,11], v.clone());
    /// assert_eq!(2, (&s[2]).len());
    /// assert_eq!(&[5,6], &s[2]);
    /// ```
    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[self.chunks[idx]..self.chunks[idx + 1]]
    }
}

impl<T, O> std::ops::Index<usize> for Chunked<&[T], O>
where
    O: std::ops::Index<usize, Output = usize>,
{
    type Output = <[T] as std::ops::Index<std::ops::Range<usize>>>::Output;

    /// Immutably index the `Chunked` borrowed slice by `usize`.
    ///
    /// Note that this works for chunked collections that are themselves not chunked, since the
    /// item at the index of a doubly chunked collection is itself chunked, which cannot be
    /// represented by a single borrow. For more complex indexing use the `get` method provided by
    /// the `Get` trait.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11];
    /// let s = Chunked::from_offsets(vec![0,3,4,6,9,11], v.as_slice());
    /// assert_eq!(&[5,6], &s[2]);
    /// ```
    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[self.chunks[idx]..self.chunks[idx + 1]]
    }
}

impl<T, O> std::ops::Index<usize> for Chunked<&mut [T], O>
where
    O: std::ops::Index<usize, Output = usize>,
{
    type Output = <[T] as std::ops::Index<std::ops::Range<usize>>>::Output;

    /// Immutably index the `Chunked` mutably borrowed slice by `usize`.
    ///
    /// Note that this works for chunked collections that are themselves not chunked, since the
    /// item at the index of a doubly chunked collection is itself chunked, which cannot be
    /// represented by a single borrow. For more complex indexing use the `get` method provided by
    /// the `Get` trait.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,5,6,7,8,9,10,11];
    /// let s = Chunked::from_offsets(vec![0,3,4,6,9,11], v.as_mut_slice());
    /// assert_eq!(&[5,6], &s[2]);
    /// ```
    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[self.chunks[idx]..self.chunks[idx + 1]]
    }
}

impl<T, O> std::ops::IndexMut<usize> for Chunked<Vec<T>, O>
where
    O: std::ops::Index<usize, Output = usize>,
{
    /// Mutably index the `Chunked` `Vec` by `usize`.
    ///
    /// Note that this works for chunked collections that are themselves not chunked, since the
    /// item at the index of a doubly chunked collection is itself chunked, which cannot be
    /// represented by a single borrow. For more complex indexing use the `get` method provided by
    /// the `Get` trait.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,0,0,7,8,9,10,11];
    /// let mut s = Chunked::from_offsets(vec![0,3,4,6,9,11], v.clone());
    /// s[2].copy_from_slice(&[5,6]);
    /// assert_eq!(vec![1,2,3,4,5,6,7,8,9,10,11], s.into_storage());
    /// ```
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.data[self.chunks[idx]..self.chunks[idx + 1]]
    }
}

impl<T, O> std::ops::IndexMut<usize> for Chunked<&mut [T], O>
where
    O: std::ops::Index<usize, Output = usize>,
{
    /// Mutably index the `Chunked` mutably borrowed slice by `usize`.
    ///
    /// Note that this works for chunked collections that are themselves not
    /// chunked, since the item at the index of a doubly chunked collection is
    /// itself chunked, which cannot be represented by a single borrow. For more
    /// complex indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,0,0,7,8,9,10,11];
    /// let mut s = Chunked::from_offsets(vec![0,3,4,6,9,11], v.as_mut_slice());
    /// s[2].copy_from_slice(&[5,6]);
    /// assert_eq!(vec![1,2,3,4,5,6,7,8,9,10,11], v);
    /// ```
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.data[self.chunks[idx]..self.chunks[idx + 1]]
    }
}

impl<'a, S> IntoIterator for Chunked<S, Offsets<&'a [usize]>>
where
    S: SplitAt + Set + Dummy,
{
    type Item = S;
    type IntoIter = ChunkedIter<Sizes<'a>, S>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ChunkedIter {
            sizes: self.chunks.into_sizes(),
            data: self.data,
        }
    }
}

impl<'a, S, O> ViewIterator<'a> for Chunked<S, O>
where
    S: View<'a>,
    O: View<'a, Type = Offsets<&'a [usize]>>,
    <S as View<'a>>::Type: SplitAt + Set + Dummy,
{
    type Item = <S as View<'a>>::Type;
    type Iter = ChunkedIter<Sizes<'a>, <S as View<'a>>::Type>;

    #[inline]
    fn view_iter(&'a self) -> Self::Iter {
        self.iter()
    }
}

impl<'a, S, O> ViewMutIterator<'a> for Chunked<S, O>
where
    S: ViewMut<'a>,
    O: View<'a, Type = Offsets<&'a [usize]>>,
    <S as ViewMut<'a>>::Type: SplitAt + Set + Dummy,
{
    type Item = <S as ViewMut<'a>>::Type;
    type Iter = ChunkedIter<Sizes<'a>, <S as ViewMut<'a>>::Type>;

    #[inline]
    fn view_mut_iter(&'a mut self) -> Self::Iter {
        self.iter_mut()
    }
}

impl_atom_iterators_recursive!(impl<S, O> for Chunked<S, O> { data });

impl<'a, S, O> Chunked<S, O>
where
    S: View<'a>,
    O: View<'a>,
    O::Type: IntoSizes,
{
    /// Produce an iterator over elements (borrowed slices) of a `Chunked`.
    ///
    /// # Examples
    ///
    /// The following simple example demonstrates how to iterate over a `Chunked`
    /// of integers stored in a flat `Vec`.
    ///
    /// ```
    /// use flatk::*;
    /// let s = Chunked::from_offsets(vec![0,3,4,6], vec![1,2,3,4,5,6]);
    /// let mut iter = s.iter();
    /// let mut e0_iter = iter.next().unwrap().iter();
    /// assert_eq!(Some(&1), e0_iter.next());
    /// assert_eq!(Some(&2), e0_iter.next());
    /// assert_eq!(Some(&3), e0_iter.next());
    /// assert_eq!(None, e0_iter.next());
    /// assert_eq!(Some(&[4][..]), iter.next());
    /// assert_eq!(Some(&[5,6][..]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// Nested `Chunked`s can also be used to create more complex data organization:
    ///
    /// ```
    /// use flatk::*;
    /// let s0 = Chunked::from_offsets(vec![0,3,4,6,9,11], vec![1,2,3,4,5,6,7,8,9,10,11]);
    /// let s1 = Chunked::from_offsets(vec![0,1,4,5], s0);
    /// let mut iter1 = s1.iter();
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[4][..]), iter0.next());
    /// assert_eq!(Some(&[5,6][..]), iter0.next());
    /// assert_eq!(Some(&[7,8,9][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[10,11][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// ```
    #[inline]
    pub fn iter(
        &'a self,
    ) -> ChunkedIter<<<O as View<'a>>::Type as IntoSizes>::Iter, <S as View<'a>>::Type> {
        ChunkedIter {
            sizes: self.chunks.view().into_sizes(),
            data: self.data.view(),
        }
    }
}

impl<'a, S, O> Chunked<S, O>
where
    S: ViewMut<'a>,
    O: View<'a>,
    O::Type: IntoSizes,
{
    /// Produce a mutable iterator over elements (borrowed slices) of a
    /// `Chunked`.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let mut s = Chunked::from_offsets(vec![0,3,4,6], vec![1,2,3,4,5,6]);
    /// for i in s.view_mut().iter_mut() {
    ///     for j in i.iter_mut() {
    ///         *j += 1;
    ///     }
    /// }
    /// let mut iter = s.iter();
    /// assert_eq!(vec![2,3,4], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![6,7], iter.next().unwrap().to_vec());
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// Nested `Chunked`s can also be used to create more complex data organization:
    ///
    /// ```
    /// use flatk::*;
    /// let mut s0 = Chunked::from_offsets(vec![0,3,4,6,9,11], vec![0,1,2,3,4,5,6,7,8,9,10]);
    /// let mut s1 = Chunked::from_offsets(vec![0,1,4,5], s0);
    /// for mut v0 in s1.view_mut().iter_mut() {
    ///     for i in v0.iter_mut() {
    ///         for j in i.iter_mut() {
    ///             *j += 1;
    ///         }
    ///     }
    /// }
    /// let v1 = s1.view();
    /// let mut iter1 = v1.iter();
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[4][..]), iter0.next());
    /// assert_eq!(Some(&[5,6][..]), iter0.next());
    /// assert_eq!(Some(&[7,8,9][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[10,11][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// ```
    #[inline]
    pub fn iter_mut(
        &'a mut self,
    ) -> ChunkedIter<<<O as View<'a>>::Type as IntoSizes>::Iter, <S as ViewMut<'a>>::Type> {
        ChunkedIter {
            sizes: self.chunks.view().into_sizes(),
            data: self.data.view_mut(),
        }
    }
}

impl<S, O> SplitAt for Chunked<S, O>
where
    S: SplitAt + Set,
    O: SplitOffsetsAt,
{
    #[inline]
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (offsets_l, offsets_r, off) = self.chunks.split_offsets_with_intersection_at(mid);
        let (data_l, data_r) = self.data.split_at(off);
        (
            Chunked {
                chunks: offsets_l,
                data: data_l,
            },
            Chunked {
                chunks: offsets_r,
                data: data_r,
            },
        )
    }
}

/// A special iterator capable of iterating over a `Chunked` type.
pub struct ChunkedIter<I, S> {
    sizes: I,
    data: S,
}

impl<I, V> Iterator for ChunkedIter<I, V>
where
    V: SplitAt + Set + Dummy,
    I: Iterator<Item = usize>,
{
    type Item = V;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // WARNING: After calling std::mem::replace with dummy, self.data is in a
        // temporarily invalid state.
        let data_slice = std::mem::replace(&mut self.data, unsafe { Dummy::dummy() });
        self.sizes.next().map(move |n| {
            let (l, r) = data_slice.split_at(n);
            // self.data is restored to the valid state here.
            self.data = r;
            l
        })
    }
}

/*
 * `IntoIterator` implementation for `Chunked`. Note that this type of
 * iterator allocates a new `Vec` at each iteration. This is an expensive
 * operation and is here for compatibility with the rest of Rust's ecosystem.
 * However, this iterator should be used sparingly.
 *
 * TODO: It should be possible to rewrite this implementation with unsafe to split off Box<[T]>
 * chunks, however this is not a priority at the moment since efficient iteration can always be
 * done on slices.
 */

/// IntoIter for `Chunked`.
pub struct VarIntoIter<S> {
    offsets: std::iter::Peekable<std::vec::IntoIter<usize>>,
    data: S,
}

impl<S> Iterator for VarIntoIter<S>
where
    S: SplitOff + Set,
{
    type Item = S;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let begin = self
            .offsets
            .next()
            .expect("Chunked is corrupted and cannot be iterated.");
        if self.offsets.len() <= 1 {
            return None; // Ignore the last offset
        }
        let end = *self.offsets.peek().unwrap();
        let n = end - begin;
        let mut l = self.data.split_off(n);
        std::mem::swap(&mut l, &mut self.data);
        Some(l) // These are the elements [0..n).
    }
}

impl<S: SplitOff + Set> SplitOff for Chunked<S> {
    #[inline]
    fn split_off(&mut self, mid: usize) -> Self {
        // Note: Allocations in this function heavily outweigh any cost in bounds checking.
        assert!(!self.chunks.is_empty());
        assert!(mid < self.chunks.len());
        let off = self.chunks[mid] - self.chunks[0];
        let offsets_l = self.chunks[..=mid].to_vec();
        let offsets_r = self.chunks[mid..].to_vec();
        self.chunks = offsets_l.into();
        let data_r = self.data.split_off(off);
        Chunked::from_offsets(offsets_r, data_r)
    }
}

impl<S> IntoIterator for Chunked<S>
where
    S: SplitOff + Set,
{
    type Item = S;
    type IntoIter = VarIntoIter<S>;

    fn into_iter(self) -> Self::IntoIter {
        let Chunked { chunks, data } = self;
        VarIntoIter {
            offsets: chunks.into_inner().into_iter().peekable(),
            data,
        }
    }
}

impl<'a, S, O> View<'a> for Chunked<S, O>
where
    S: View<'a>,
    O: View<'a>,
{
    type Type = Chunked<S::Type, O::Type>;

    /// Create a contiguous immutable (shareable) view into this set.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let s = Chunked::<Vec<usize>>::from_offsets(vec![0,1,4,6], vec![0,1,2,3,4,5]);
    /// let v1 = s.view();
    /// let v2 = v1.clone();
    /// let mut view1_iter = v1.clone().into_iter();
    /// assert_eq!(Some(&[0][..]), view1_iter.next());
    /// assert_eq!(Some(&[1,2,3][..]), view1_iter.next());
    /// assert_eq!(Some(&[4,5][..]), view1_iter.next());
    /// assert_eq!(None, view1_iter.next());
    /// for (a,b) in v1.into_iter().zip(v2.into_iter()) {
    ///     assert_eq!(a,b);
    /// }
    /// ```
    #[inline]
    fn view(&'a self) -> Self::Type {
        Chunked {
            chunks: self.chunks.view(),
            data: self.data.view(),
        }
    }
}

impl<'a, S, O> ViewMut<'a> for Chunked<S, O>
where
    S: ViewMut<'a>,
    O: View<'a>,
{
    type Type = Chunked<S::Type, O::Type>;

    /// Create a contiguous mutable (unique) view into this set.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let mut s = Chunked::<Vec<usize>>::from_offsets(vec![0,1,4,6], vec![0,1,2,3,4,5]);
    /// let mut v1 = s.view_mut();
    /// v1.iter_mut().next().unwrap()[0] = 100;
    /// let mut view1_iter = v1.iter();
    /// assert_eq!(Some(&[100][..]), view1_iter.next());
    /// assert_eq!(Some(&[1,2,3][..]), view1_iter.next());
    /// assert_eq!(Some(&[4,5][..]), view1_iter.next());
    /// assert_eq!(None, view1_iter.next());
    /// ```
    #[inline]
    fn view_mut(&'a mut self) -> Self::Type {
        Chunked {
            chunks: self.chunks.view(),
            data: self.data.view_mut(),
        }
    }
}

impl<S: IntoStorage, O> IntoStorage for Chunked<S, O> {
    type StorageType = S::StorageType;
    /// Strip all organizational information from this set, returning the
    /// underlying storage type.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11];
    /// let s0 = Chunked::from_offsets(vec![0,3,4,6,9,11], v.clone());
    /// let s1 = Chunked::from_offsets(vec![0,1,4,5], s0.clone());
    /// assert_eq!(s1.into_storage(), v);
    /// assert_eq!(s0.into_storage(), v);
    /// ```
    #[inline]
    fn into_storage(self) -> Self::StorageType {
        self.data.into_storage()
    }
}

impl<'a, S: StorageView<'a>, O> StorageView<'a> for Chunked<S, O> {
    type StorageView = S::StorageView;
    /// Return a view to the underlying storage type.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11];
    /// let s0 = Chunked::from_offsets(vec![0,3,4,6,9,11], v.clone());
    /// let s1 = Chunked::from_offsets(vec![0,1,4,5], s0.clone());
    /// assert_eq!(s1.storage_view(), v.as_slice());
    /// assert_eq!(s0.storage_view(), v.as_slice());
    /// ```
    #[inline]
    fn storage_view(&'a self) -> Self::StorageView {
        self.data.storage_view()
    }
}

impl<S: Storage, O> Storage for Chunked<S, O> {
    type Storage = S::Storage;
    /// Return an immutable reference to the underlying storage type.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11];
    /// let s0 = Chunked::from_offsets(vec![0,3,4,6,9,11], v.clone());
    /// let s1 = Chunked::from_offsets(vec![0,1,4,5], s0.clone());
    /// assert_eq!(s1.storage(), &v);
    /// assert_eq!(s0.storage(), &v);
    /// ```
    #[inline]
    fn storage(&self) -> &Self::Storage {
        self.data.storage()
    }
}

impl<S: StorageMut, O> StorageMut for Chunked<S, O> {
    /// Return a mutable reference to the underlying storage type.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,5,6,7,8,9,10,11];
    /// let mut s0 = Chunked::from_offsets(vec![0,3,4,6,9,11], v.clone());
    /// let mut s1 = Chunked::from_offsets(vec![0,1,4,5], s0.clone());
    /// assert_eq!(s1.storage_mut(), &mut v);
    /// assert_eq!(s0.storage_mut(), &mut v);
    /// ```
    #[inline]
    fn storage_mut(&mut self) -> &mut Self::Storage {
        self.data.storage_mut()
    }
}

impl<T, S: CloneWithStorage<T>, O: Clone> CloneWithStorage<T> for Chunked<S, O> {
    type CloneType = Chunked<S::CloneType, O>;
    #[inline]
    fn clone_with_storage(&self, storage: T) -> Self::CloneType {
        Chunked {
            chunks: self.chunks.clone(),
            data: self.data.clone_with_storage(storage),
        }
    }
}

impl<S: Default, O: Default> Default for Chunked<S, O> {
    /// Construct an empty `Chunked`.
    #[inline]
    fn default() -> Self {
        Chunked {
            data: Default::default(),
            chunks: Default::default(),
        }
    }
}

impl<S: Dummy, O: Dummy> Dummy for Chunked<S, O> {
    #[inline]
    unsafe fn dummy() -> Self {
        Chunked {
            data: Dummy::dummy(),
            chunks: Dummy::dummy(),
        }
    }
}

/// Required for subsets of chunked collections.
impl<S: RemovePrefix, O: RemovePrefix + AsRef<[usize]>> RemovePrefix for Chunked<S, O> {
    /// Remove a prefix of size `n` from a chunked collection.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let mut s = Chunked::<Vec<usize>>::from_offsets(vec![0,1,4,6], vec![0,1,2,3,4,5]);
    /// s.remove_prefix(2);
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[4,5][..]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    #[inline]
    fn remove_prefix(&mut self, n: usize) {
        let chunks = self.chunks.as_ref();
        assert!(n < chunks.len());
        let offset = *chunks.first().unwrap();

        self.chunks.remove_prefix(n);
        let data_offset = *self.chunks.as_ref().first().unwrap() - offset;
        self.data.remove_prefix(data_offset);
    }
}

impl<S: Clear> Clear for Chunked<S> {
    #[inline]
    fn clear(&mut self) {
        self.chunks.clear();
        self.chunks.push(0);
        self.data.clear();
    }
}

impl<S, O, N> SplitPrefix<N> for Chunked<S, O>
where
    S: Viewed + Set + SplitAt,
    N: Unsigned,
    O: Set + SplitOffsetsAt,
{
    type Prefix = Chunked<S, O>;
    #[inline]
    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        if N::to_usize() > self.len() {
            return None;
        }
        Some(self.split_at(N::to_usize()))
    }
}

impl<S, O> SplitFirst for Chunked<S, O>
where
    S: Viewed + Set + SplitAt,
    O: Set + SplitOffsetsAt,
{
    type First = S;
    #[inline]
    fn split_first(self) -> Option<(Self::First, Self)> {
        if self.is_empty() {
            return None;
        }
        let (_, rest_chunks, off) = self.chunks.split_offsets_with_intersection_at(1);
        let (first, rest) = self.data.split_at(off);
        Some((
            first,
            Chunked {
                data: rest,
                chunks: rest_chunks,
            },
        ))
    }
}

impl<S, I, N> UniChunkable<N> for Chunked<S, I> {
    type Chunk = Chunked<S, I>;
}

impl<S, N> IntoStaticChunkIterator<N> for ChunkedView<'_, S>
where
    Self: Set + SplitPrefix<N> + Dummy,
    N: Unsigned,
{
    type Item = <Self as SplitPrefix<N>>::Prefix;
    type IterType = UniChunkedIter<Self, N>;
    #[inline]
    fn into_static_chunk_iter(self) -> Self::IterType {
        self.into_generic_static_chunk_iter()
    }
}

/// Pass through the conversion for structure type `Chunked`.
impl<S: StorageInto<T>, O, T> StorageInto<T> for Chunked<S, O> {
    type Output = Chunked<S::Output, O>;
    #[inline]
    fn storage_into(self) -> Self::Output {
        Chunked {
            data: self.data.storage_into(),
            chunks: self.chunks,
        }
    }
}

//impl<S: PermuteInPlace + SplitAt + Swap, O: PermuteInPlace> PermuteInPlace for Chunked<S, O> {
//    fn permute_in_place(&mut self, permutation: &[usize], seen: &mut [bool]) {
//        // This algorithm involves allocating since it avoids excessive copying.
//        assert!(permutation.len(), self.len());
//        debug_assert!(permutation.all(|&i| i < self.len()));
//        self.extend_from_slice
//        permutation.iter().map(|&i| self[i]).collect()
//
//        debug_assert_eq!(permutation.len(), self.len());
//        debug_assert!(seen.len() >= self.len());
//        debug_assert!(seen.all(|&s| !s));
//
//        for unseen_i in 0..seen.len() {
//            if seen[unseen_i] {
//                continue;
//            }
//
//            let mut i = unseen_i;
//            loop {
//                let idx = permutation[i];
//                if seen[idx] {
//                    break;
//                }
//
//                // Swap elements at i and idx
//                let (l, r) = self.data.split_at(self.chunks[i], self.chunks[idx]);
//                for off in 0..self.chunks {
//                    self.data.swap(off + self.chunks * i, off + self.chunks * idx);
//                }
//
//                seen[i] = true;
//                i = idx;
//            }
//        }
//    }
//}

impl<S: Reserve, O: Reserve> Reserve for Chunked<S, O> {
    #[inline]
    fn reserve_with_storage(&mut self, n: usize, storage_n: usize) {
        self.chunks.reserve(n);
        self.data.reserve_with_storage(n, storage_n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes_constructor() {
        let empty: Vec<u32> = vec![];
        let s = Chunked::from_sizes(vec![], Vec::<u32>::new());
        assert_eq!(s.len(), 0);

        let s = Chunked::from_sizes(vec![0], Vec::<u32>::new());
        assert_eq!(s.len(), 1);
        assert_eq!(empty.as_slice(), s.view().at(0));

        let s = Chunked::from_sizes(vec![0, 0, 0], vec![]);
        assert_eq!(s.len(), 3);
        for chunk in s.iter() {
            assert_eq!(empty.as_slice(), chunk);
        }
    }

    #[test]
    fn zero_length_chunk() {
        let empty: Vec<usize> = vec![];
        // In the beginning
        let s = Chunked::from_offsets(vec![0, 0, 3, 4, 6], vec![1, 2, 3, 4, 5, 6]);
        let mut iter = s.iter();
        assert_eq!(empty.clone(), iter.next().unwrap().to_vec());
        assert_eq!(vec![1, 2, 3], iter.next().unwrap().to_vec());
        assert_eq!(vec![4], iter.next().unwrap().to_vec());
        assert_eq!(vec![5, 6], iter.next().unwrap().to_vec());
        assert_eq!(None, iter.next());

        // In the middle
        let s = Chunked::from_offsets(vec![0, 3, 3, 4, 6], vec![1, 2, 3, 4, 5, 6]);
        let mut iter = s.iter();
        assert_eq!(vec![1, 2, 3], iter.next().unwrap().to_vec());
        assert_eq!(empty.clone(), iter.next().unwrap().to_vec());
        assert_eq!(vec![4], iter.next().unwrap().to_vec());
        assert_eq!(vec![5, 6], iter.next().unwrap().to_vec());
        assert_eq!(None, iter.next());

        // At the end
        let s = Chunked::from_offsets(vec![0, 3, 4, 6, 6], vec![1, 2, 3, 4, 5, 6]);
        let mut iter = s.iter();
        assert_eq!(vec![1, 2, 3], iter.next().unwrap().to_vec());
        assert_eq!(vec![4], iter.next().unwrap().to_vec());
        assert_eq!(vec![5, 6], iter.next().unwrap().to_vec());
        assert_eq!(empty.clone(), iter.next().unwrap().to_vec());
        assert_eq!(None, iter.next());
    }

    #[test]
    fn chunked_range() {
        let c = Chunked::from_sizes(vec![0, 4, 2, 0, 1], 0..7);
        assert_eq!(c.at(0), 0..0);
        assert_eq!(c.at(1), 0..4);
        assert_eq!(c.at(2), 4..6);
        assert_eq!(c.at(3), 6..6);
        assert_eq!(c.at(4), 6..7);
        assert_eq!(c.into_storage(), 0..7);
    }

    #[test]
    fn chunked_viewable() {
        let mut s = Chunked::<Vec<usize>>::from_offsets(vec![0, 1, 4, 6], vec![0, 1, 2, 3, 4, 5]);
        let v1 = s.into_view();
        let v2 = v1.clone();
        let mut view1_iter = v1.clone().into_iter();
        assert_eq!(Some(&[0][..]), view1_iter.next());
        assert_eq!(Some(&[1, 2, 3][..]), view1_iter.next());
        assert_eq!(Some(&[4, 5][..]), view1_iter.next());
        assert_eq!(None, view1_iter.next());
        for (a, b) in v1.into_iter().zip(v2.into_iter()) {
            assert_eq!(a, b);
        }

        let v_mut = (&mut s).into_view();
        v_mut.isolate(0)[0] = 100;
        assert_eq!(&[100][..], s.into_view().at(0));
    }

    #[test]
    fn trim() {
        // This is a similar example to the one in the doc test, but is more adversarial by using
        // offsets not starting at 0.
        let mut s = Chunked::from_offsets(vec![2, 3, 6, 8], vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(3, s.len());

        // Transferring the last two elements past the indexed stack.
        // This creates a zero sized chunk at the end.
        s.transfer_forward(2, 2);
        assert_eq!(6, s.data().len());
        assert_eq!(3, s.len());

        let mut trimmed = s.clone();
        trimmed.trim_data(); // Remove unindexed elements.
        assert_eq!(4, trimmed.data().len());

        let mut trimmed = s;
        trimmed.trim(); // Remove unindexed elements.
        assert_eq!(4, trimmed.data().len());
    }
}
