use super::*;
use std::convert::AsRef;

/// A Set that is a non-contiguous subset of some larger collection.
/// `B` can be any borrowed collection type that implements the [`Set`], and [`RemovePrefix`]
/// traits.
/// For iteration of subsets, the underlying type must also implement [`SplitFirst`] and
/// [`SplitAt`] traits.
///
/// # Example
///
/// The following example shows how to create a `Subset` from a standard `Vec`.
///
/// ```rust
/// use flatk::*;
/// let v = vec![1,2,3,4,5];
/// let subset = Subset::from_indices(vec![0,2,4], v.as_slice());
/// let mut subset_iter = subset.iter();
/// assert_eq!(Some(&1), subset_iter.next());
/// assert_eq!(Some(&3), subset_iter.next());
/// assert_eq!(Some(&5), subset_iter.next());
/// assert_eq!(None, subset_iter.next());
/// ```
///
/// The next example shows how to create a `Subset` from a [`UniChunked`] collection.
///
/// ```rust
/// use flatk::*;
/// let mut v = Chunked3::from_flat(vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
/// let mut subset = Subset::from_indices(vec![0,2,4], v.view_mut());
/// {
///     let subset_view = subset.view();
///     let mut subset_iter = subset_view.iter();
///     assert_eq!(Some(&[1,2,3]), subset_iter.next());
///     assert_eq!(Some(&[7,8,9]), subset_iter.next());
///     assert_eq!(Some(&[13,14,15]), subset_iter.next());
///     assert_eq!(None, subset_iter.next());
/// }
/// *subset.view_mut().isolate(1) = [0; 3];
/// assert_eq!(&[0,0,0], subset.view().at(1));
/// ```
///
/// # Translation independence
///
/// This struct is very similar to [`Chunked`], with the main difference being that
/// each index corresponds to a single element instead of a chunk starting point.
///
/// Translation independence refers to the need to ensure that indices remain valid after
/// splitting. When the indices are owned or mutably borrowed, we could simply modify the indices
/// when we split the subset, but when the indices are a borrowed slice, this is not possible. To
/// resolve this, we chop the part of data below the first index to ensure that the first index
/// serves as an offset to the rest of the indices, making the entire index array translation
/// independent.
///
/// Because `Subset`s modify the underlying data storage, it can often be misleading when querying
/// the underlying data at any given time using one of [`Storage`], [`StorageMut`] or
/// [`StorageView`] traits.
///
/// For a more transparent data structure that preserves the original data set,
/// use [`Select`]. To expose any characteristics of the contained `data` type, use a trait.
/// See [`ChunkSize`] for an example.
///
/// [`Select`]: struct.Select.html
/// [`ChunkSize`]: trait.ChunkSize.html
/// [`Chunked`]: struct.Chunked.html
/// [`Storage`]: trait.Storage.html
/// [`StorageMut`]: trait.StorageMut.html
/// [`StorageView`]: trait.StorageView.html
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Subset<S, I = Box<[usize]>> {
    /// An optional set of indices.
    ///
    /// When this is `None`, the subset is considered to be entire.
    /// Empty subsets are represented by a zero length array of indices: either `Some(&[])` or
    /// `Some(Vec::new())`.
    pub(crate) indices: Option<I>,
    pub(crate) data: S,
}

/// A borrowed subset.
pub type SubsetView<'a, S> = Subset<S, &'a [usize]>;

impl<S, I> Subset<S, I> {
    /// Convert this subset into its internal representation.
    #[inline]
    pub fn into_raw(self) -> (Option<I>, S) {
        (self.indices, self.data)
    }

    /// Construct a subset from a set of indices and a data set.
    ///
    /// Note that independent of the value of the indices, the first element in the subset will be
    /// the first element in `data`, and all subsequent elements are taken from `data[index -
    /// first]` for each `index` in `indices` where `first` is the first index appearing in
    /// `indices`.
    ///
    /// # Safety
    ///
    /// Constructing an invalid subset using this function isn't itself unsafe, however calling
    /// various functions (except for [`Subset::validate`]) may be unsafe.
    ///
    /// The given indices must be unique and in accending sorted order.
    /// All indices (minus the first) must be strictly less than the number of elements in `data`.
    ///
    /// The `Subset` can be validated explicitly after creation using [`Subset::validate`].
    ///
    /// [`Subset::validate`]: function.Subset.validate.html
    #[inline]
    pub unsafe fn from_raw(indices: Option<I>, data: S) -> Subset<S, I> {
        Subset { indices, data }
    }
}

impl<S: Set + RemovePrefix> Subset<S, Vec<usize>> {
    /// Create a subset of elements from the original set given at the specified indices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let v = vec![1,2,3];
    /// let subset = Subset::from_indices(vec![0,2], v.as_slice());
    /// assert_eq!(1, subset[0]);
    /// assert_eq!(3, subset[1]);
    /// ```
    pub fn from_indices(mut indices: Vec<usize>, mut data: S) -> Self {
        // Ensure that indices are sorted and there are no duplicates.
        // Failure to enforce this invariant can cause race conditions.

        indices.sort_unstable();
        indices.dedup();

        if let Some(first) = indices.first() {
            data.remove_prefix(*first);
        }

        Self::validate(Subset {
            indices: Some(indices),
            data,
        })
    }
}

impl<S: Set + RemovePrefix, I: AsRef<[usize]>> Subset<S, I> {
    /// Create a subset of elements from the original collection corresponding to the given
    /// indices.
    ///
    /// In contrast to `Subset::from_indices`, this function expects the indices
    /// to be unique and in sorted order, instead of manully making it so.
    ///
    /// # Panics
    ///
    /// This function panics when given a collection of unsorted indices.
    /// It also panics when indices are repeated.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let v = vec![0,1,2,3];
    /// let indices = vec![1,3];
    ///
    /// let subset_view = Subset::from_unique_ordered_indices(indices.as_slice(), v.as_slice());
    /// assert_eq!(1, subset_view[0]);
    /// assert_eq!(3, subset_view[1]);
    ///
    /// let subset = Subset::from_unique_ordered_indices(indices, v.as_slice());
    /// assert_eq!(1, subset[0]);
    /// assert_eq!(3, subset[1]);
    /// ```
    pub fn from_unique_ordered_indices(indices: I, mut data: S) -> Self {
        // Ensure that indices are sorted and there are no duplicates.

        assert!(Self::is_sorted(&indices));
        assert!(!Self::has_duplicates(&indices));

        if let Some(first) = indices.as_ref().first() {
            data.remove_prefix(*first);
        }

        Self::validate(Subset {
            indices: Some(indices),
            data,
        })
    }
}

impl<S> Subset<S> {
    /// Create a subset with all elements from the original set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let subset = Subset::all::<Box<_>>(vec![1,2,3]);
    /// let subset_view = subset.view();
    /// let mut subset_iter = subset_view.iter();
    /// assert_eq!(Some(&1), subset_iter.next());
    /// assert_eq!(Some(&2), subset_iter.next());
    /// assert_eq!(Some(&3), subset_iter.next());
    /// assert_eq!(None, subset_iter.next());
    /// ```
    #[inline]
    pub fn all<I>(data: S) -> Subset<S, I> {
        Subset {
            indices: None,
            data,
        }
    }

    /// Create a subset with all elements from the original set.
    ///
    /// This version of `all` creates the `Subset` type with the default index type, since it
    /// cannot be determined from the function arguments. In other words this function doesn't
    /// require an additional generic parameter to be specified when the return type is ambiguous.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let subset = Subset::all_def(vec![1,2,3]);
    /// let subset_view = subset.view();
    /// let mut subset_iter = subset_view.iter();
    /// assert_eq!(Some(&1), subset_iter.next());
    /// assert_eq!(Some(&2), subset_iter.next());
    /// assert_eq!(Some(&3), subset_iter.next());
    /// assert_eq!(None, subset_iter.next());
    /// ```
    #[inline]
    pub fn all_def(data: S) -> Subset<S> {
        Self::all(data)
    }
}

impl<S: Set, I: AsRef<[usize]>> Subset<S, I> {
    /// Find an element in the subset by its index in the superset. Return the index of the element
    /// in the subset if found.
    /// Since subset indices are always in sorted order, this function performs a binary search.
    ///
    /// # Examples
    ///
    /// In the following simple example the element `3` is found at superset index `2` which is
    /// located at index `1` in the subset.
    ///
    /// ```
    /// use flatk::*;
    /// let superset =  vec![1,2,3,4,5,6];
    /// let subset = Subset::from_unique_ordered_indices(vec![1,2,5], superset);
    /// assert_eq!(Some(1), subset.find_by_index(2));
    /// assert_eq!(None, subset.find_by_index(3));
    /// ```
    ///
    /// Note that the superset index refers to the indices with which the subset was created. This
    /// means that even after we have split the subset, the input indices are expected to refer to
    /// the original subset. The following example demonstrates this by splitting the original
    /// subset in the pervious example.
    ///
    /// ```
    /// use flatk::*;
    /// let superset =  vec![1,2,3,4,5,6];
    /// let subset = Subset::from_unique_ordered_indices(vec![1,2,5], superset);
    /// let (_, r) = subset.view().split_at(1);
    /// assert_eq!(Some(0), r.find_by_index(2));
    /// assert_eq!(None, r.find_by_index(3));
    /// ```
    pub fn find_by_index(&self, index: usize) -> Option<usize> {
        match &self.indices {
            Some(indices) => indices.as_ref().binary_search(&index).ok(),
            None => {
                // If the subset is entire, then we know the element is contained.
                Some(index)
            }
        }
    }
}

impl<'a, S, I: AsRef<[usize]>> Subset<S, I> {
    /// A helper function that checks if a given collection of indices has duplicates.
    /// It is assumed that the given indices are already in sorted order.
    fn has_duplicates(indices: &I) -> bool {
        let mut index_iter = indices.as_ref().iter().cloned();
        if let Some(mut prev) = index_iter.next() {
            for cur in index_iter {
                if cur == prev {
                    return true;
                } else {
                    prev = cur;
                }
            }
        }
        false
    }

    /// Checks that the given set of indices are sorted.
    // TODO: replace this with std version when RFC 2351 lands
    // (https://github.com/rust-lang/rust/issues/53485)
    #[inline]
    fn is_sorted(indices: &I) -> bool {
        Self::is_sorted_by(indices, |a, b| a.partial_cmp(b))
    }

    /// Checks that the given set of indices are sorted by the given compare function.
    #[allow(clippy::while_let_on_iterator)]
    fn is_sorted_by<F>(indices: &I, mut compare: F) -> bool
    where
        F: FnMut(&usize, &usize) -> Option<std::cmp::Ordering>,
    {
        let mut iter = indices.as_ref().iter();
        let mut last = match iter.next() {
            Some(e) => e,
            None => return true,
        };

        while let Some(curr) = iter.next() {
            if compare(last, curr)
                .map(|o| o == std::cmp::Ordering::Greater)
                .unwrap_or(true)
            {
                return false;
            }
            last = curr;
        }

        true
    }
}

impl<'a, S: Set, I> Subset<S, I> {
    /// Get a references to the underlying indices. If `None` is returned, then
    /// this subset spans the entire domain `data`.
    #[inline]
    pub fn indices(&self) -> Option<&I> {
        self.indices.as_ref()
    }

    /// Return the superset of this `Subset`. This is just the set it was created with.
    #[inline]
    pub fn into_super(self) -> S {
        self.data
    }
}

impl<'a, S: Set, I: AsRef<[usize]>> Subset<S, I> {
    /// Panics if this subset is invald.
    #[inline]
    fn validate(self) -> Self {
        if let Some(ref indices) = self.indices {
            let indices = indices.as_ref();
            if let Some(first) = indices.first() {
                for &i in indices.iter() {
                    assert!(i - *first < self.data.len(), "Subset index out of bounds.");
                }
            }
        }
        self
    }
}

// Note to self:
// To enable a collection to be chunked, we need to implement:
// Set, View, SplitAt
// For mutability we also need ViewMut,
// For UniChunked we need:
// Set, Vew, IntoStaticChunkIterator

/// Required for `Chunked` and `UniChunked` subsets.
impl<S: Set, I: AsRef<[usize]>> Set for Subset<S, I> {
    type Elem = S::Elem;
    type Atom = S::Atom;
    /// Get the length of this subset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5];
    /// let subset = Subset::from_indices(vec![0,2,4], v.as_slice());
    /// assert_eq!(3, subset.len());
    /// ```
    #[inline]
    fn len(&self) -> usize {
        self.indices
            .as_ref()
            .map_or(self.data.len(), |indices| indices.as_ref().len())
    }
}

/// Required for `Chunked` and `UniChunked` subsets.
impl<'a, S, I> View<'a> for Subset<S, I>
where
    S: View<'a>,
    I: AsRef<[usize]>,
{
    type Type = Subset<S::Type, &'a [usize]>;
    #[inline]
    fn view(&'a self) -> Self::Type {
        // Note: it is assumed that the first index corresponds to the first
        // element in data, regardless of what the value of the index is.
        Subset {
            indices: self.indices.as_ref().map(|indices| indices.as_ref()),
            data: self.data.view(),
        }
    }
}

impl<'a, S, I> ViewMut<'a> for Subset<S, I>
where
    S: ViewMut<'a>,
    I: AsRef<[usize]>,
{
    type Type = Subset<S::Type, &'a [usize]>;
    /// Create a mutable view into this subset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// let mut view = subset.view_mut();
    /// for i in view.iter_mut() {
    ///     *i += 1;
    /// }
    /// assert_eq!(v, vec![2,2,4,4,6]);
    /// ```
    #[inline]
    fn view_mut(&'a mut self) -> Self::Type {
        // Note: it is assumed that the first index corresponds to the first
        // element in data, regardless of what the value of the index is.
        Subset {
            indices: self.indices.as_ref().map(|indices| indices.as_ref()),
            data: self.data.view_mut(),
        }
    }
}

/// This impl enables `Chunked` `Subset`s
impl<V> SplitAt for SubsetView<'_, V>
where
    V: Set + SplitAt,
{
    /// Split this subset into two at the given index `mid`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5];
    /// let indices = vec![0,2,4];
    /// let subset = Subset::from_unique_ordered_indices(indices.as_slice(), v.as_slice());
    /// let (l, r) = subset.split_at(1);
    /// let mut iter_l = l.iter();
    /// assert_eq!(Some(&1), iter_l.next());
    /// assert_eq!(None, iter_l.next());
    /// let mut iter_r = r.iter();
    /// assert_eq!(Some(&3), iter_r.next());
    /// assert_eq!(Some(&5), iter_r.next());
    /// assert_eq!(None, iter_r.next());
    /// ```
    #[inline]
    fn split_at(self, mid: usize) -> (Self, Self) {
        if let Some(indices) = self.indices {
            let (indices_l, indices_r) = indices.split_at(mid);
            let n = self.data.len();
            let offset = indices_r
                .first()
                .map(|first| *first - *indices_l.first().unwrap_or(first))
                .unwrap_or(n);
            let (data_l, data_r) = self.data.split_at(offset);
            (
                Subset {
                    indices: Some(indices_l),
                    data: data_l,
                },
                Subset {
                    indices: Some(indices_r),
                    data: data_r,
                },
            )
        } else {
            let (data_l, data_r) = self.data.split_at(mid);
            (
                Subset {
                    indices: None,
                    data: data_l,
                },
                Subset {
                    indices: None,
                    data: data_r,
                },
            )
        }
    }
}

/// This impl enables `Subset`s of `Subset`s
impl<S, I> SplitFirst for Subset<S, I>
where
    I: SplitFirst + AsRef<[usize]>,
    <I as SplitFirst>::First: std::borrow::Borrow<usize>,
    S: Set + SplitAt + SplitFirst,
{
    type First = S::First;

    /// Split the first element of this subset.
    #[inline]
    fn split_first(self) -> Option<(Self::First, Self)> {
        use std::borrow::Borrow;
        let Subset { data, indices } = self;
        if let Some(indices) = indices {
            indices.split_first().map(|(first_index, rest_indices)| {
                let n = data.len();
                let offset = rest_indices
                    .as_ref()
                    .first()
                    .map(|next| *next - *first_index.borrow())
                    .unwrap_or(n);
                let (data_l, data_r) = data.split_at(offset);
                (
                    data_l.split_first().unwrap().0,
                    Subset {
                        indices: Some(rest_indices),
                        data: data_r,
                    },
                )
            })
        } else {
            data.split_first().map(|(first, rest)| {
                (
                    first,
                    Subset {
                        indices: None,
                        data: rest,
                    },
                )
            })
        }
    }
}

impl<S: Set + RemovePrefix, I: RemovePrefix + AsRef<[usize]>> RemovePrefix for Subset<S, I> {
    /// This function will panic if `n` is larger than `self.len()`.
    #[inline]
    fn remove_prefix(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        match self.indices {
            Some(ref mut indices) => {
                let first = indices.as_ref()[0]; // Will panic if out of bounds.
                indices.remove_prefix(n);
                let data_len = self.data.len();
                let next = indices.as_ref().get(0).unwrap_or(&data_len);
                self.data.remove_prefix(*next - first);
            }
            None => {
                self.data.remove_prefix(n);
            }
        }
    }
}

impl<'a, S, I> Subset<S, I>
where
    Self: Set + ViewIterator<'a>,
{
    /// The typical way to use this function is to clone from a `SubsetView`
    /// into an owned `S` type.
    ///
    /// # Panics
    ///
    /// This function panics if `other` has a length unequal to `self.len()`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5];
    /// let indices = vec![0,2,4];
    /// let subset = Subset::from_unique_ordered_indices(indices.as_slice(), v.as_slice());
    /// let mut owned = vec![0; 4];
    /// subset.clone_into_other(&mut owned[..3]); // Need 3 elements to avoid panics.
    /// let mut iter_owned = owned.iter();
    /// assert_eq!(owned, vec![1,3,5,0]);
    /// ```
    pub fn clone_into_other<V>(&'a self, other: &'a mut V)
    where
        V: Set + ViewMutIterator<'a> + ?Sized,
        <Self as ViewIterator<'a>>::Item: CloneIntoOther<<V as ViewMutIterator<'a>>::Item>,
    {
        assert_eq!(other.len(), self.len());
        for (mut theirs, mine) in other.view_mut_iter().zip(self.view_iter()) {
            mine.clone_into_other(&mut theirs);
        }
    }
}

/*
 * Indexing operators for convenience. Users familiar with indexing by `usize`
 * may find these implementations convenient.
 */

impl<'a, S, O> GetIndex<'a, Subset<S, O>> for usize
where
    O: AsRef<[usize]>,
    S: Get<'a, usize>,
{
    type Output = <S as Get<'a, usize>>::Output;

    #[inline]
    fn get(self, subset: &Subset<S, O>) -> Option<Self::Output> {
        // TODO: too much bounds checking here, add a get_unchecked call to GetIndex.
        if let Some(ref indices) = subset.indices {
            indices.as_ref().get(0).and_then(|&first| {
                indices
                    .as_ref()
                    .get(self)
                    .and_then(|&cur| Get::get(&subset.data, cur - first))
            })
        } else {
            Get::get(&subset.data, self)
        }
    }
}

impl<'a, S, O> GetIndex<'a, Subset<S, O>> for &usize
where
    O: AsRef<[usize]>,
    S: Get<'a, usize>,
{
    type Output = <S as Get<'a, usize>>::Output;

    #[inline]
    fn get(self, subset: &Subset<S, O>) -> Option<Self::Output> {
        GetIndex::get(*self, subset)
    }
}

impl<S, O> IsolateIndex<Subset<S, O>> for usize
where
    O: AsRef<[usize]>,
    S: Isolate<usize>,
{
    type Output = <S as Isolate<usize>>::Output;

    #[inline]
    unsafe fn isolate_unchecked(self, subset: Subset<S, O>) -> Self::Output {
        let Subset { indices, data } = subset;
        Isolate::isolate_unchecked(
            data,
            if let Some(ref indices) = indices {
                let cur = indices.as_ref().get_unchecked(self);
                let first = indices.as_ref().get_unchecked(0);
                cur - first
            } else {
                self
            },
        )
    }

    #[inline]
    fn try_isolate(self, subset: Subset<S, O>) -> Option<Self::Output> {
        let Subset { indices, data } = subset;
        Isolate::try_isolate(
            data,
            if let Some(ref indices) = indices {
                let cur = indices.as_ref().get(self)?;
                // SAFETY: self must be at least zero, and we just checked it above.
                let first = unsafe { indices.as_ref().get_unchecked(0) };
                cur - first
            } else {
                self
            },
        )
    }
}

impl_isolate_index_for_static_range!(impl<S, O> for Subset<S, O>);

//impl<S, I, O> Isolate<I> for Subset<S, O>
//where
//    I: IsolateIndex<Self>,
//{
//    type Output = I::Output;
//
//    fn try_isolate(self, range: I) -> Option<Self::Output> {
//        range.try_isolate(self)
//    }
//}

macro_rules! impl_index_fn {
    ($self:ident, $idx:ident, $index_fn:ident) => {
        $self
            .data
            .$index_fn($self.indices.as_ref().map_or($idx, |indices| {
                let indices = indices.as_ref();
                indices[$idx] - *indices.first().unwrap()
            }))
    };
}

impl<'a, S, I> std::ops::Index<usize> for Subset<S, I>
where
    S: std::ops::Index<usize> + Set + ValueType,
    I: AsRef<[usize]>,
{
    type Output = S::Output;

    /// Immutably index the subset.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the subset is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let c = Chunked2::from_flat((1..=12).collect::<Vec<_>>());
    /// let subset = Subset::from_indices(vec![0,2,4], c.view());
    /// assert_eq!([1,2], subset[0]);
    /// assert_eq!([5,6], subset[1]);
    /// assert_eq!([9,10], subset[2]);
    /// ```
    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        impl_index_fn!(self, idx, index)
    }
}

impl<'a, S, I> std::ops::IndexMut<usize> for Subset<S, I>
where
    S: std::ops::IndexMut<usize> + Set + ValueType,
    I: AsRef<[usize]>,
{
    /// Mutably index the subset.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the subset is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// assert_eq!(subset[1], 3);
    /// subset[1] = 100;
    /// assert_eq!(subset[0], 1);
    /// assert_eq!(subset[1], 100);
    /// assert_eq!(subset[2], 5);
    /// ```
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        impl_index_fn!(self, idx, index_mut)
    }
}

impl<'a, T, I> std::ops::Index<usize> for Subset<&'a [T], I>
where
    I: AsRef<[usize]>,
{
    type Output = T;
    /// Immutably index the subset.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the subset is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5];
    /// let subset = Subset::from_indices(vec![0,2,4], v.as_slice());
    /// assert_eq!(3, subset[1]);
    /// ```
    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        impl_index_fn!(self, idx, index)
    }
}

impl<'a, T, I> std::ops::Index<usize> for Subset<&'a mut [T], I>
where
    I: AsRef<[usize]>,
{
    type Output = T;
    /// Immutably index the subset.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the subset is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// assert_eq!(3, subset[1]);
    /// ```
    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        impl_index_fn!(self, idx, index)
    }
}

impl<'a, T, I> std::ops::IndexMut<usize> for Subset<&'a mut [T], I>
where
    I: AsRef<[usize]>,
{
    /// Mutably index the subset.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the subset is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// assert_eq!(subset[1], 3);
    /// subset[1] = 100;
    /// assert_eq!(subset[0], 1);
    /// assert_eq!(subset[1], 100);
    /// assert_eq!(subset[2], 5);
    /// ```
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        impl_index_fn!(self, idx, index_mut)
    }
}

/*
 * Iteration
 */

impl<S, I> IntoIterator for Subset<S, I>
where
    S: SplitAt + SplitFirst + Set + Dummy,
    I: SplitFirst + Clone,
    <I as SplitFirst>::First: std::borrow::Borrow<usize>,
{
    type Item = S::First;
    type IntoIter = SubsetIter<S, I>;

    /// Convert a `Subset` into an iterator.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let mut s = Subset::from_unique_ordered_indices(vec![1,3,5], vec![1,2,3,4,5,6]);
    /// let mut iter = s.view().into_iter();
    /// assert_eq!(Some(&2), iter.next());
    /// assert_eq!(Some(&4), iter.next());
    /// assert_eq!(Some(&6), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        SubsetIter {
            indices: self.indices,
            data: self.data,
        }
    }
}

// Iterator for `Subset`s
#[derive(Copy, Clone, Debug)]
pub struct SubsetIter<S, I> {
    indices: Option<I>,
    data: S,
}

// TODO: This can be made more efficient with two distinct iterators, thus eliminating the branch
// on indices.
impl<S, I> Iterator for SubsetIter<S, I>
where
    S: SplitAt + SplitFirst + Set + Dummy,
    I: SplitFirst + Clone,
    <I as SplitFirst>::First: std::borrow::Borrow<usize>,
{
    type Item = S::First;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        use std::borrow::Borrow;
        let SubsetIter { indices, data } = self;
        let data_slice = std::mem::replace(data, unsafe { Dummy::dummy() });
        match indices {
            Some(ref mut indices) => indices.clone().split_first().map(|(first, rest)| {
                let (item, right) = data_slice.split_first().expect("Corrupt subset");
                if let Some((second, _)) = rest.clone().split_first() {
                    let (_, r) = right.split_at(*second.borrow() - *first.borrow() - 1);
                    *data = r;
                } else {
                    // No more elements, the rest is empty, just discard the rest of data.
                    // An alternative implementation simply assigns data to the empty version of S.
                    // This would require additional traits so we settle for this possibly less
                    // efficient version for now.
                    let n = right.len();
                    let (_, r) = right.split_at(n);
                    *data = r;
                }
                *indices = rest;
                item
            }),
            None => data_slice.split_first().map(|(item, rest)| {
                *data = rest;
                item
            }),
        }
    }
}

// Iterator for `Subset` indices.
pub enum SubsetIndexIter<'a> {
    All(std::ops::Range<usize>),
    Sub(&'a [usize]),
}

impl<'a> Iterator for SubsetIndexIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SubsetIndexIter::Sub(ref mut indices) => indices.split_first().map(|(first, rest)| {
                *indices = rest;
                *first
            }),
            SubsetIndexIter::All(ref mut rng) => rng.next(),
        }
    }
    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self {
            SubsetIndexIter::Sub(ref mut indices) => {
                if n >= indices.len() {
                    None
                } else {
                    // SAFETY: the bounds are checked above.
                    unsafe {
                        let item = *indices.get_unchecked(n);
                        *indices = indices.get_unchecked(n + 1..);
                        Some(item)
                    }
                }
            }
            SubsetIndexIter::All(ref mut rng) => rng.nth(n),
        }
    }
}

impl<'a, S, I> Subset<S, I>
where
    S: Set + View<'a>,
    I: AsRef<[usize]>,
{
    /// Immutably iterate over a borrowed subset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// let mut iter = subset.iter();
    /// assert_eq!(Some(&1), iter.next());
    /// assert_eq!(Some(&3), iter.next());
    /// assert_eq!(Some(&5), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    #[inline]
    pub fn iter(&'a self) -> SubsetIter<S::Type, &'a [usize]> {
        SubsetIter {
            indices: self.indices.as_ref().map(|indices| indices.as_ref()),
            data: self.data.view(),
        }
    }

    /// Immutably iterate over the indices stored by this subset.
    ///
    /// The returned indices point to the superset from which this subset was created.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// let mut iter = subset.index_iter();
    /// assert_eq!(Some(0), iter.next());
    /// assert_eq!(Some(2), iter.next());
    /// assert_eq!(Some(4), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// This also works if the subset is entire:
    ///
    /// ```
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::all_def(v.as_slice());
    /// let mut iter = subset.index_iter();
    /// assert_eq!(Some(0), iter.next());
    /// assert_eq!(Some(3), iter.nth(2));
    /// assert_eq!(Some(4), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    #[inline]
    pub fn index_iter(&'a self) -> SubsetIndexIter<'a> {
        match self.indices {
            Some(ref indices) => SubsetIndexIter::Sub(indices.as_ref()),
            None => SubsetIndexIter::All(0..self.data.len()),
        }
    }
}

impl<'a, S, I> Subset<S, I>
where
    S: Set + ViewMut<'a>,
    I: AsRef<[usize]>,
{
    /// Mutably iterate over a borrowed subset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// for i in subset.iter_mut() {
    ///     *i += 1;
    /// }
    /// assert_eq!(v, vec![2,2,4,4,6]);
    /// ```
    #[inline]
    pub fn iter_mut(&'a mut self) -> SubsetIter<<S as ViewMut<'a>>::Type, &'a [usize]> {
        SubsetIter {
            indices: self.indices.as_ref().map(|indices| indices.as_ref()),
            data: self.data.view_mut(),
        }
    }
}

impl<'a, S, I> ViewIterator<'a> for Subset<S, I>
where
    S: Set + View<'a>,
    I: AsRef<[usize]>,
    <S as View<'a>>::Type: SplitAt + SplitFirst + Set + Dummy,
{
    type Item = <<S as View<'a>>::Type as SplitFirst>::First;
    type Iter = SubsetIter<S::Type, &'a [usize]>;

    #[inline]
    fn view_iter(&'a self) -> Self::Iter {
        self.iter()
    }
}

impl<'a, S, I> ViewMutIterator<'a> for Subset<S, I>
where
    S: Set + ViewMut<'a>,
    I: AsRef<[usize]>,
    <S as ViewMut<'a>>::Type: SplitAt + SplitFirst + Set + Dummy,
{
    type Item = <<S as ViewMut<'a>>::Type as SplitFirst>::First;
    type Iter = SubsetIter<S::Type, &'a [usize]>;

    #[inline]
    fn view_mut_iter(&'a mut self) -> Self::Iter {
        self.iter_mut()
    }
}

impl<S: Dummy, I> Dummy for Subset<S, I> {
    #[inline]
    unsafe fn dummy() -> Self {
        Subset {
            data: Dummy::dummy(),
            indices: None,
        }
    }
}

impl<S: Truncate, I: Truncate> Truncate for Subset<S, I> {
    #[inline]
    fn truncate(&mut self, new_len: usize) {
        match &mut self.indices {
            // The target data remains untouched.
            Some(indices) => indices.truncate(new_len),
            // Since the subset is entire it's ok to truncate the underlying data.
            None => self.data.truncate(new_len),
        }
    }
}

/*
 * Conversions
 */

// TODO: Add conversions for other subsets.

//impl<T> From<T> for Subset<T> {
//    fn from(v: T) -> Subset<T> {
//        Subset::all(v)
//    }
//}

/// Pass through the conversion for structure type `Subset`.
impl<S: StorageInto<T>, I, T> StorageInto<T> for Subset<S, I> {
    type Output = Subset<S::Output, I>;
    #[inline]
    fn storage_into(self) -> Self::Output {
        Subset {
            data: self.data.storage_into(),
            indices: self.indices,
        }
    }
}

impl<S: MapStorage<Out>, I, Out> MapStorage<Out> for Subset<S, I> {
    type Input = S::Input;
    type Output = Subset<S::Output, I>;
    #[inline]
    fn map_storage<F: FnOnce(Self::Input) -> Out>(self, f: F) -> Self::Output {
        Subset {
            data: self.data.map_storage(f),
            indices: self.indices,
        }
    }
}

/*
 * Data access
 */

impl<'a, S: StorageView<'a>, I> StorageView<'a> for Subset<S, I> {
    type StorageView = S::StorageView;
    /// Return a view to the underlying storage type.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11,12];
    /// let s0 = Chunked3::from_flat(v.clone());
    /// let s1 = Subset::from_indices(vec![0, 2, 3], s0.clone());
    /// assert_eq!(s1.storage_view(), v.as_slice());
    /// ```
    #[inline]
    fn storage_view(&'a self) -> Self::StorageView {
        self.data.storage_view()
    }
}

impl<S: Storage, I> Storage for Subset<S, I> {
    type Storage = S::Storage;
    /// Return an immutable reference to the underlying storage type.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11,12];
    /// let s0 = Chunked3::from_flat(v.clone());
    /// let s1 = Subset::from_indices(vec![0, 2, 3], s0.clone());
    /// assert_eq!(s1.storage(), &v);
    /// ```
    #[inline]
    fn storage(&self) -> &Self::Storage {
        self.data.storage()
    }
}

impl<S: StorageMut, I> StorageMut for Subset<S, I> {
    /// Return a mutable reference to the underlying storage type.
    ///
    /// # Example
    ///
    /// ```rust
    /// use flatk::*;
    /// let mut v = vec![1,2,3,4,5,6,7,8,9,10,11,12];
    /// let mut s0 = Chunked3::from_flat(v.clone());
    /// let mut s1 = Subset::from_indices(vec![0, 2, 3], s0.clone());
    /// assert_eq!(s1.storage_mut(), &mut v);
    /// ```
    #[inline]
    fn storage_mut(&mut self) -> &mut Self::Storage {
        self.data.storage_mut()
    }
}

/*
 * Subsets of uniformly chunked collections
 */

impl<S: ChunkSize, I> ChunkSize for Subset<S, I> {
    #[inline]
    fn chunk_size(&self) -> usize {
        self.data.chunk_size()
    }
}

impl<S: ChunkSize, I, N: Dimension> Subset<UniChunked<S, N>, I> {
    #[inline]
    pub fn inner_chunk_size(&self) -> usize {
        self.data.inner_chunk_size()
    }
}

/*
 * Convert views to owned types
 */

impl<S: IntoOwned, I: IntoOwned> IntoOwned for Subset<S, I> {
    type Owned = Subset<S::Owned, I::Owned>;

    #[inline]
    fn into_owned(self) -> Self::Owned {
        Subset {
            indices: self.indices.map(|x| x.into_owned()),
            data: self.data.into_owned(),
        }
    }
}

impl<S, I> IntoOwnedData for Subset<S, I>
where
    S: IntoOwnedData,
{
    type OwnedData = Subset<S::OwnedData, I>;
    #[inline]
    fn into_owned_data(self) -> Self::OwnedData {
        Subset {
            indices: self.indices,
            data: self.data.into_owned_data(),
        }
    }
}

/*
 * Impls for uniformly chunked sparse types
 */

impl<S, I, M> UniChunkable<M> for Subset<S, I> {
    type Chunk = Subset<S, I>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subset_of_subsets_iter() {
        let set = vec![1, 2, 3, 4, 5, 6];
        let subset = Subset::from_unique_ordered_indices(vec![1, 3, 5], set);
        let subsubset = Subset::from_unique_ordered_indices(vec![0, 2], subset);
        let mut iter = subsubset.iter();
        assert_eq!(Some(&2), iter.next());
        assert_eq!(Some(&6), iter.next());
        assert_eq!(None, iter.next());
    }
}
