use super::*;

impl<'a, T, N> GetIndex<'a, &'a [T]> for StaticRange<N>
where
    N: Unsigned + Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Output = &'a N::Array;
    #[inline]
    fn get(self, set: &&'a [T]) -> Option<Self::Output> {
        if self.end() <= set.len() {
            let slice = *set;
            Some(unsafe { &*(slice.as_ptr().add(self.start()) as *const N::Array) })
        } else {
            None
        }
    }
}

impl<'a, T, N> IsolateIndex<&'a [T]> for StaticRange<N>
where
    N: Unsigned + Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Output = &'a N::Array;
    #[inline]
    unsafe fn isolate_unchecked(self, set: &'a [T]) -> Self::Output {
        &*(set.as_ptr().add(self.start()) as *const N::Array)
    }
    #[inline]
    fn try_isolate(self, set: &'a [T]) -> Option<Self::Output> {
        if self.end() <= set.len() {
            Some(unsafe { IsolateIndex::isolate_unchecked(self, set) })
        } else {
            None
        }
    }
}

impl<'a, T, N> IsolateIndex<&'a mut [T]> for StaticRange<N>
where
    N: Unsigned + Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Output = &'a mut N::Array;
    #[inline]
    unsafe fn isolate_unchecked(self, set: &'a mut [T]) -> Self::Output {
        &mut *(set.as_mut_ptr().add(self.start()) as *mut N::Array)
    }
    #[inline]
    fn try_isolate(self, set: &'a mut [T]) -> Option<Self::Output> {
        if self.end() <= set.len() {
            Some(unsafe { IsolateIndex::isolate_unchecked(self, set) })
        } else {
            None
        }
    }
}

impl<'a, T, I> GetIndex<'a, &'a [T]> for I
where
    I: std::slice::SliceIndex<[T]>,
    <[T] as std::ops::Index<I>>::Output: 'a,
{
    type Output = &'a <[T] as std::ops::Index<I>>::Output;
    #[inline]
    fn get(self, set: &&'a [T]) -> Option<Self::Output> {
        Some(std::ops::Index::<I>::index(*set, self))
    }
}

impl<'a, T, I> IsolateIndex<&'a [T]> for I
where
    I: std::slice::SliceIndex<[T]>,
    <I as std::slice::SliceIndex<[T]>>::Output: 'a,
{
    type Output = &'a <[T] as std::ops::Index<I>>::Output;
    #[inline]
    unsafe fn isolate_unchecked(self, set: &'a [T]) -> &'a <[T] as std::ops::Index<I>>::Output {
        set.get_unchecked(self)
    }
    #[inline]
    fn try_isolate(self, set: &'a [T]) -> Option<&'a <[T] as std::ops::Index<I>>::Output> {
        Some(std::ops::Index::<I>::index(set, self))
    }
}

impl<'a, T, I> IsolateIndex<&'a mut [T]> for I
where
    I: std::slice::SliceIndex<[T]>,
    <I as std::slice::SliceIndex<[T]>>::Output: 'a,
{
    type Output = &'a mut <[T] as std::ops::Index<I>>::Output;
    #[inline]
    unsafe fn isolate_unchecked(
        self,
        set: &'a mut [T],
    ) -> &'a mut <[T] as std::ops::Index<I>>::Output {
        //let slice = std::slice::from_raw_parts_mut(set.as_mut_ptr(), set.len());
        set.get_unchecked_mut(self)
    }
    #[inline]
    fn try_isolate(self, set: &'a mut [T]) -> Option<&'a mut <[T] as std::ops::Index<I>>::Output> {
        Some(unsafe { IsolateIndex::isolate_unchecked(self, set) })
    }
}

impl<T> Set for [T] {
    type Elem = T;
    type Atom = T;
    #[inline]
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

impl<'a, T: 'a> View<'a> for [T] {
    type Type = &'a [T];

    #[inline]
    fn view(&'a self) -> Self::Type {
        self
    }
}

impl<'a, T: 'a> ViewMut<'a> for [T] {
    type Type = &'a mut [T];

    #[inline]
    fn view_mut(&'a mut self) -> Self::Type {
        self
    }
}

impl<'a, T: 'a> ViewIterator<'a> for [T] {
    type Item = &'a T;
    type Iter = std::slice::Iter<'a, T>;

    #[inline]
    fn view_iter(&'a self) -> Self::Iter {
        self.iter()
    }
}
impl<'a, T: 'a> ViewMutIterator<'a> for [T] {
    type Item = &'a mut T;
    type Iter = std::slice::IterMut<'a, T>;

    #[inline]
    fn view_mut_iter(&'a mut self) -> Self::Iter {
        self.iter_mut()
    }
}

impl<'a, T: 'a> AtomIterator<'a> for [T] {
    type Item = &'a T;
    type Iter = std::slice::Iter<'a, T>;
    #[inline]
    fn atom_iter(&'a self) -> Self::Iter {
        self.iter()
    }
}

impl<'a, T: 'a> AtomMutIterator<'a> for [T] {
    type Item = &'a mut T;
    type Iter = std::slice::IterMut<'a, T>;
    #[inline]
    fn atom_mut_iter(&'a mut self) -> Self::Iter {
        self.iter_mut()
    }
}

impl<'a, T, N> SplitPrefix<N> for &'a [T]
where
    N: Unsigned + Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Prefix = &'a N::Array;

    #[inline]
    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::to_usize() {
            None
        } else {
            Some(unsafe {
                let (l, r) = self.split_at(N::to_usize());
                (&*(l.as_ptr() as *const N::Array), r)
            })
        }
    }
}

impl<'a, T, N> SplitPrefix<N> for &'a mut [T]
where
    N: Unsigned + Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Prefix = &'a mut N::Array;

    #[inline]
    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::to_usize() {
            None
        } else {
            Some(unsafe {
                let (l, r) = self.split_at_mut(N::to_usize());
                (&mut *(l.as_mut_ptr() as *mut N::Array), r)
            })
        }
    }
}

impl<T, N: Array<T>> UniChunkable<N> for [T] {
    type Chunk = N::Array;
}

impl<'a, T, N: Array<T>> UniChunkable<N> for &'a [T]
where
    <N as Array<T>>::Array: 'a,
{
    type Chunk = &'a N::Array;
}

impl<'a, T, N: Array<T>> UniChunkable<N> for &'a mut [T]
where
    <N as Array<T>>::Array: 'a,
{
    type Chunk = &'a mut N::Array;
}

impl<'a, T, N> IntoStaticChunkIterator<N> for &'a [T]
where
    Self: SplitPrefix<N>,
    N: Unsigned,
{
    type Item = <Self as SplitPrefix<N>>::Prefix;
    type IterType = UniChunkedIter<Self, N>;
    #[inline]
    fn into_static_chunk_iter(self) -> Self::IterType {
        self.into_generic_static_chunk_iter()
    }
}

impl<'a, T, N> IntoStaticChunkIterator<N> for &'a mut [T]
where
    Self: SplitPrefix<N>,
    N: Unsigned,
{
    type Item = <Self as SplitPrefix<N>>::Prefix;
    type IterType = UniChunkedIter<Self, N>;
    #[inline]
    fn into_static_chunk_iter(self) -> Self::IterType {
        self.into_generic_static_chunk_iter()
    }
}

#[cfg(feature = "rayon")]
impl<'a, T: Send + Sync> IntoParChunkIterator for &'a [T] {
    type Item = &'a [T];
    type IterType = rayon::slice::Chunks<'a, T>;

    #[inline]
    fn into_par_chunk_iter(self, chunk_size: usize) -> Self::IterType {
        use rayon::slice::ParallelSlice;
        assert_eq!(self.len() % chunk_size, 0);
        self.par_chunks(chunk_size)
    }
}

#[cfg(feature = "rayon")]
impl<'a, T: Send + Sync> IntoParChunkIterator for &'a mut [T] {
    type Item = &'a mut [T];
    type IterType = rayon::slice::ChunksMut<'a, T>;

    #[inline]
    fn into_par_chunk_iter(self, chunk_size: usize) -> Self::IterType {
        use rayon::slice::ParallelSliceMut;
        assert_eq!(self.len() % chunk_size, 0);
        self.par_chunks_mut(chunk_size)
    }
}

impl<'a, T> SplitFirst for &'a [T] {
    type First = &'a T;

    #[inline]
    fn split_first(self) -> Option<(Self::First, Self)> {
        self.split_first()
    }
}

impl<'a, T> SplitFirst for &'a mut [T] {
    type First = &'a mut T;

    #[inline]
    fn split_first(self) -> Option<(Self::First, Self)> {
        self.split_first_mut()
    }
}

impl<'a, T> IntoStorage for &'a [T] {
    type StorageType = &'a [T];
    #[inline]
    fn into_storage(self) -> Self::StorageType {
        self
    }
}

impl<'a, T> IntoStorage for &'a mut [T] {
    type StorageType = &'a mut [T];
    #[inline]
    fn into_storage(self) -> Self::StorageType {
        self
    }
}

impl<'a, T> StorageView<'a> for &'a [T] {
    type StorageView = &'a [T];
    #[inline]
    fn storage_view(&'a self) -> Self::StorageView {
        self
    }
}

impl<'a, T> Storage for [T] {
    type Storage = [T];
    #[inline]
    fn storage(&self) -> &Self::Storage {
        self
    }
}

impl<'a, T> StorageMut for [T] {
    /// A slice is a type of storage, simply return a mutable reference to self.
    #[inline]
    fn storage_mut(&mut self) -> &mut Self::Storage {
        self
    }
}

impl<'a, T: 'a> CloneWithStorage<Vec<T>> for &'a [T] {
    type CloneType = Vec<T>;
    /// This function simply ignores self and returns storage since self is already
    /// a storage type.
    #[inline]
    fn clone_with_storage(&self, storage: Vec<T>) -> Self::CloneType {
        assert_eq!(self.len(), storage.len());
        storage
    }
}

impl<'a, T: 'a> CloneWithStorage<&'a [T]> for &'a [T] {
    type CloneType = &'a [T];
    /// This function simply ignores self and returns storage since self is already
    /// a storage type.
    #[inline]
    fn clone_with_storage(&self, storage: &'a [T]) -> Self::CloneType {
        assert_eq!(self.len(), storage.len());
        storage
    }
}

impl<'a, T: 'a> CloneWithStorage<&'a mut [T]> for &'a mut [T] {
    type CloneType = &'a mut [T];
    /// This function simply ignores self and returns storage since self is already
    /// a storage type.
    #[inline]
    fn clone_with_storage(&self, storage: &'a mut [T]) -> Self::CloneType {
        assert_eq!(self.len(), storage.len());
        storage
    }
}

impl<'a, T> SplitAt for &mut [T] {
    #[inline]
    fn split_at(self, mid: usize) -> (Self, Self) {
        self.split_at_mut(mid)
    }
}

impl<'a, T> SplitAt for &[T] {
    #[inline]
    fn split_at(self, mid: usize) -> (Self, Self) {
        self.split_at(mid)
    }
}

impl<T> Dummy for &[T] {
    #[inline]
    unsafe fn dummy() -> Self {
        &[]
    }
}

impl<T> Dummy for &mut [T] {
    #[inline]
    unsafe fn dummy() -> Self {
        &mut []
    }
}

impl<T> RemovePrefix for &[T] {
    #[inline]
    fn remove_prefix(&mut self, n: usize) {
        *self = &self[n..];
    }
}

impl<T> RemovePrefix for &mut [T] {
    /// Remove a prefix of size `n` from this mutable slice.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut s = v.as_mut_slice();
    /// s.remove_prefix(2);
    /// assert_eq!(&[3,4,5], s);
    /// ```
    #[inline]
    fn remove_prefix(&mut self, n: usize) {
        let data = std::mem::replace(self, &mut []);
        *self = &mut data[n..];
    }
}

impl<'a, T, N> ReinterpretAsGrouped<N> for &'a [T]
where
    N: Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Output = &'a [N::Array];
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        unsafe { reinterpret::reinterpret_slice(self) }
    }
}

impl<'a, T, N> ReinterpretAsGrouped<N> for &'a mut [T]
where
    N: Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Output = &'a mut [N::Array];
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        unsafe { reinterpret::reinterpret_mut_slice(self) }
    }
}

impl<T> Viewed for [T] {}

impl<T> Truncate for &[T] {
    #[inline]
    fn truncate(&mut self, new_len: usize) {
        // Simply forget about the elements past new_len.
        *self = self.split_at(new_len).0;
    }
}

impl<T> Truncate for &mut [T] {
    #[inline]
    fn truncate(&mut self, new_len: usize) {
        let data = std::mem::replace(self, &mut []);
        // Simply forget about the elements past new_len.
        *self = data.split_at_mut(new_len).0;
    }
}

/*
 * These are base cases for `ConvertStorage`. We apply the conversion at this point since slices
 * are storage types. The following are some common conversion behaviours.
 */

/// Convert a slice into an owned `Vec` type.
impl<'a, T: Clone> StorageInto<Vec<T>> for &'a [T] {
    type Output = Vec<T>;
    #[inline]
    fn storage_into(self) -> Self::Output {
        self.to_vec()
    }
}

impl<T, Out> MapStorage<Out> for &[T] {
    type Input = Self;
    type Output = Out;
    #[inline]
    fn map_storage<F: FnOnce(Self::Input) -> Out>(self, f: F) -> Self::Output {
        f(self)
    }
}

impl<T, Out> MapStorage<Out> for &mut [T] {
    type Input = Self;
    type Output = Out;
    #[inline]
    fn map_storage<F: FnOnce(Self::Input) -> Out>(self, f: F) -> Self::Output {
        f(self)
    }
}

/// Convert a mutable slice into an owned `Vec` type.
impl<'a, T: Clone> StorageInto<Vec<T>> for &'a mut [T] {
    type Output = Vec<T>;
    #[inline]
    fn storage_into(self) -> Self::Output {
        self.to_vec()
    }
}

/// Convert a mutable slice into an immutable borrow.
impl<'a, T: 'a> StorageInto<&'a [T]> for &'a mut [T] {
    type Output = &'a [T];
    #[inline]
    fn storage_into(self) -> Self::Output {
        &*self
    }
}

/*
 * End of ConvertStorage impls
 */

impl<T> SwapChunks for &mut [T] {
    /// Swap non-overlapping chunks beginning at the given indices.
    #[inline]
    fn swap_chunks(&mut self, i: usize, j: usize, chunk_size: usize) {
        assert!(i + chunk_size <= j || j + chunk_size <= i);

        let (lower, upper) = if i < j { (i, j) } else { (j, i) };
        let (l, r) = self.split_at_mut(upper);
        l[lower..lower + chunk_size].swap_with_slice(&mut r[..chunk_size]);
    }
}

impl<T: PartialOrd + Clone> Sort for [T] {
    /// Sort the given indices into this collection with respect to values provided by this collection.
    /// Invalid values like `NaN` in floats will be pushed to the end.
    #[inline]
    fn sort_indices(&self, indices: &mut [usize]) {
        indices.sort_by(|&a, &b| {
            self[a]
                .partial_cmp(&self[b])
                .unwrap_or(std::cmp::Ordering::Less)
        });
    }
}

impl<T> PermuteInPlace for &mut [T] {
    /// Permute this slice according to the given permutation.
    /// The given permutation must have length equal to this slice.
    /// The slice `seen` is provided to keep track of which elements have already been seen.
    /// `seen` is assumed to be initialized to `false` and have length equal or
    /// larger than this slice.
    #[inline]
    fn permute_in_place(&mut self, permutation: &[usize], seen: &mut [bool]) {
        let data = std::mem::replace(self, &mut []);
        UniChunked {
            chunk_size: 1,
            data,
        }
        .permute_in_place(permutation, seen);
    }
}

impl<T: Clone> CloneIntoOther<Vec<T>> for [T] {
    #[inline]
    fn clone_into_other(&self, other: &mut Vec<T>) {
        other.clear();
        other.extend_from_slice(self);
    }
}

impl<T: Clone> CloneIntoOther for [T] {
    #[inline]
    fn clone_into_other(&self, other: &mut [T]) {
        assert_eq!(self.len(), other.len());
        other.clone_from_slice(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_into_other() {
        let a = vec![1, 2, 3, 4];

        // slice -> mut slice
        let mut b = vec![5, 6, 7, 8];
        a.as_slice().clone_into_other(b.as_mut_slice());
        assert_eq!(b, a);
    }
}
