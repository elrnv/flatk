use super::*;

impl<T> ValueType for Vec<T> {}

impl<T> Clear for Vec<T> {
    #[inline]
    fn clear(&mut self) {
        Vec::<T>::clear(self);
    }
}

impl<T> Set for Vec<T> {
    type Elem = T;
    type Atom = T;
    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl<'a, T: 'a> View<'a> for Vec<T> {
    type Type = &'a [T];

    #[inline]
    fn view(&'a self) -> Self::Type {
        self.as_slice()
    }
}

impl<'a, T: 'a> ViewMut<'a> for Vec<T> {
    type Type = &'a mut [T];

    #[inline]
    fn view_mut(&'a mut self) -> Self::Type {
        self.as_mut_slice()
    }
}

impl<'a, T: 'a> ViewIterator<'a> for Vec<T> {
    type Item = &'a T;
    type Iter = std::slice::Iter<'a, T>;

    #[inline]
    fn view_iter(&'a self) -> Self::Iter {
        self.iter()
    }
}
impl<'a, T: 'a> ViewMutIterator<'a> for Vec<T> {
    type Item = &'a mut T;
    type Iter = std::slice::IterMut<'a, T>;

    #[inline]
    fn view_mut_iter(&'a mut self) -> Self::Iter {
        self.iter_mut()
    }
}

impl<'a, T: 'a> AtomIterator<'a> for Vec<T> {
    type Item = &'a T;
    type Iter = std::slice::Iter<'a, T>;
    #[inline]
    fn atom_iter(&'a self) -> Self::Iter {
        self.iter()
    }
}

impl<'a, T: 'a> AtomMutIterator<'a> for Vec<T> {
    type Item = &'a mut T;
    type Iter = std::slice::IterMut<'a, T>;
    #[inline]
    fn atom_mut_iter(&'a mut self) -> Self::Iter {
        self.iter_mut()
    }
}

impl<T> Push<T> for Vec<T> {
    #[inline]
    fn push(&mut self, element: T) {
        Vec::push(self, element);
    }
}

impl<T> SplitOff for Vec<T> {
    #[inline]
    fn split_off(&mut self, mid: usize) -> Self {
        Vec::split_off(self, mid)
    }
}

impl<T, N> SplitPrefix<N> for Vec<T>
where
    N: Unsigned + Array<T>,
    <N as Array<T>>::Array: Default,
{
    type Prefix = N::Array;

    #[inline]
    fn split_prefix(mut self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::to_usize() {
            return None;
        }
        // Note: This is inefficient ( as is the implementation for `remove_prefix` ).
        // As such it shouldn't be used when iterating over `Subset`s of
        // `Vec<T>` or `Subset`s of any other chunked collection that uses
        // `Vec<T>` for storage. We should be able to specialize the
        // implementation of subsets of `Vec<T>` types for better performance.
        self.rotate_left(N::to_usize());
        let at = self.len() - N::to_usize();
        let mut out: N::Array = Default::default();
        unsafe {
            self.set_len(at);
            std::ptr::copy_nonoverlapping(
                self.as_ptr().add(at),
                &mut out as *mut N::Array as *mut T,
                N::to_usize(),
            );
        }
        Some((out, self))
    }
}

impl<T, N: Array<T>> UniChunkable<N> for Vec<T> {
    type Chunk = N::Array;
}

impl<T: Clone, N: Array<T>> PushChunk<N> for Vec<T> {
    #[inline]
    fn push_chunk(&mut self, chunk: Self::Chunk) {
        self.extend_from_slice(N::as_slice(&chunk));
    }
}

impl<T, N> IntoStaticChunkIterator<N> for Vec<T>
where
    N: Unsigned + Array<T>,
    T: bytemuck::Pod,
{
    type Item = N::Array;
    type IterType = std::vec::IntoIter<N::Array>;

    #[inline]
    fn into_static_chunk_iter(self) -> Self::IterType {
        assert_eq!(self.len() % N::to_usize(), 0);
        ReinterpretAsGrouped::<N>::reinterpret_as_grouped(self).into_iter()
    }
}

impl<T> IntoStorage for Vec<T> {
    type StorageType = Vec<T>;
    /// Since a `Vec` has no information about the structure of its underlying
    /// data, this is effectively a no-op.
    #[inline]
    fn into_storage(self) -> Self::StorageType {
        self
    }
}

impl<'a, T: 'a> StorageView<'a> for Vec<T> {
    type StorageView = &'a [T];
    #[inline]
    fn storage_view(&'a self) -> Self::StorageView {
        self.as_slice()
    }
}

impl<T> Storage for Vec<T> {
    type Storage = Vec<T>;
    /// `Vec` is a type of storage, simply return an immutable reference to self.
    #[inline]
    fn storage(&self) -> &Self::Storage {
        self
    }
}

impl<T> StorageMut for Vec<T> {
    /// `Vec` is a type of storage, simply return a mutable reference to self.
    #[inline]
    fn storage_mut(&mut self) -> &mut Self::Storage {
        self
    }
}

impl<T, U> CloneWithStorage<Vec<U>> for Vec<T> {
    type CloneType = Vec<U>;
    /// This function simply ignores self and returns storage since self is already
    /// a storage type.
    #[inline]
    fn clone_with_storage(&self, storage: Vec<U>) -> Self::CloneType {
        assert_eq!(self.len(), storage.len());
        storage
    }
}

impl<T> SplitAt for Vec<T> {
    #[inline]
    fn split_at(mut self, mid: usize) -> (Self, Self) {
        let r = self.split_off(mid);
        (self, r)
    }
}

impl<T> RemovePrefix for Vec<T> {
    #[inline]
    fn remove_prefix(&mut self, n: usize) {
        self.rotate_left(n);
        self.truncate(self.len() - n);
    }
}

/// Since `Vec` already owns its data, this is simply a noop.
impl<T> IntoOwned for Vec<T> {
    type Owned = Self;
    #[inline]
    fn into_owned(self) -> Self::Owned {
        self
    }
}

/// Since `Vec` already owns its data, this is simply a noop.
impl<T> IntoOwnedData for Vec<T> {
    type OwnedData = Self;
    #[inline]
    fn into_owned_data(self) -> Self::OwnedData {
        self
    }
}

impl<'a, T, N> ReinterpretAsGrouped<N> for Vec<T>
where
    T: bytemuck::Pod,
    N: Array<T>,
{
    type Output = Vec<N::Array>;
    #[inline]
    fn reinterpret_as_grouped(mut self) -> Self::Output {
        // Ensure that the reserved space can be divided into chunks of size
        // `std::mem::size_of::<N::Array>()`.
        // The strategy is that it is cheaper to drop than to allocate, so we do a shrink to fit
        // here assuming that most likely the output vector wont need the additional space in most
        // cases.
        self.shrink_to_fit();
        unsafe { reinterpret::reinterpret_vec(self) }
        // TODO: switch to bytemuck when it can do cast between element types of different sizes.
        //bytemuck::cast_vec(self)
    }
}

impl<'a, T, N> ReinterpretAsGrouped<N> for &'a Vec<T>
where
    T: bytemuck::Pod,
    N: Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Output = &'a [N::Array];
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        //unsafe { reinterpret::reinterpret_slice(self.as_slice()) }
        bytemuck::cast_slice(self.as_slice())
    }
}

impl<'a, T, N> ReinterpretAsGrouped<N> for &'a mut Vec<T>
where
    T: bytemuck::Pod,
    N: Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Output = &'a mut [N::Array];
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        //unsafe { reinterpret::reinterpret_mut_slice(self.as_mut_slice()) }
        bytemuck::cast_slice_mut(self.as_mut_slice())
    }
}

impl<T> Dummy for Vec<T> {
    #[inline]
    unsafe fn dummy() -> Self {
        Vec::new()
    }
}

impl<T> Truncate for Vec<T> {
    #[inline]
    fn truncate(&mut self, new_len: usize) {
        Vec::truncate(self, new_len);
    }
}

impl<T: Clone> ExtendFromSlice for Vec<T> {
    type Item = T;
    #[inline]
    fn extend_from_slice(&mut self, other: &[Self::Item]) {
        Vec::extend_from_slice(self, other);
    }
}

/*
 * These are base cases for `ConvertStorage`. We apply the conversion at this point since `Vec` is
 * a storage type. The following are some common conversion behaviours.
 */

impl<T, S: Into<T>> StorageInto<Vec<T>> for Vec<S> {
    type Output = Vec<T>;
    /// Convert a `Vec` of one type into a `Vec` of another type given that the element types can be
    /// converted.
    ///
    /// # Example
    ///
    /// ```
    /// use flatk::*;
    /// let sentences = vec!["First", "sentence", "about", "nothing", ".", "Second", "sentence", "."];
    /// let chunked = Chunked::from_sizes(vec![5,3], sentences);
    /// let owned_sentences: Chunked<Vec<String>> = chunked.storage_into();
    /// assert_eq!(Some(&["Second".to_string(), "sentence".to_string(), ".".to_string()][..]), owned_sentences.view().get(1));
    /// ```
    #[inline]
    fn storage_into(self) -> Self::Output {
        self.into_iter().map(|x| x.into()).collect()
    }
}

impl<S, Out> MapStorage<Out> for Vec<S> {
    type Input = Self;
    type Output = Out;
    #[inline]
    fn map_storage<F: FnOnce(Self::Input) -> Out>(self, f: F) -> Self::Output {
        f(self)
    }
}

/*
 * End of ConvertStorage impls
 */

impl<T> PermuteInPlace for Vec<T> {
    /// Permute this collection according to the given permutation.
    /// The given permutation must have length equal to this collection.
    /// The slice `seen` is provided to keep track of which elements have already been seen.
    /// `seen` is assumed to be initialized to `false` and have length equal or
    /// larger than this collection.
    #[inline]
    fn permute_in_place(&mut self, permutation: &[usize], seen: &mut [bool]) {
        self.as_mut_slice().permute_in_place(permutation, seen);
    }
}

impl<T: Clone> CloneIntoOther for Vec<T> {
    #[inline]
    fn clone_into_other(&self, other: &mut Vec<T>) {
        other.clone_from(self);
    }
}

impl<T: Clone> CloneIntoOther<[T]> for Vec<T> {
    #[inline]
    fn clone_into_other(&self, other: &mut [T]) {
        other.clone_from_slice(self.as_slice());
    }
}
impl<T> Reserve for Vec<T> {
    #[inline]
    fn reserve_with_storage(&mut self, n: usize, storage_n: usize) {
        self.reserve(n.max(storage_n));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_into_other() {
        let a = vec![1, 2, 3, 4];

        // vec -> mut vec
        let mut b = vec![5, 6, 7, 8];
        a.clone_into_other(&mut b);
        assert_eq!(b, a);

        // vec -> mut slice
        let mut b = vec![5, 6, 7, 8];
        a.clone_into_other(b.as_mut_slice());
        assert_eq!(b, a);
    }
}
