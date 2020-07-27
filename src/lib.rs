//! **F**lat **l**ayout **a**bstraction **t**ool**k**it.
//!
//! This library defines low level primitives for organizing flat ordered data collections (like `Vec`s
//! and `slice`s) into meaningful structures without cloning the data.
//!
//! More specifically, `flatk` provides a few core composable types intended for building more complex
//! data structures out of existing data:
//!
//! - `UniChunked`:  Subdivides a collection into a number of uniformly sized (at compile time or
//!   run-time) contiguous chunks.
//!   For example if we have a `Vec` of floats representing 3D positions, we may wish to interpret them
//!   as triplets:
//!
//!   ```rust
//!   use flatk::Chunked3;
//!
//!   let pos_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
//!
//!   let pos = Chunked3::from_flat(pos_data);
//!
//!   assert_eq!(pos[0], [0.0; 3]);
//!   assert_eq!(pos[1], [1.0; 3]);
//!   assert_eq!(pos[2], [0.0, 1.0, 0.0]);
//!   ```
//!
//!   For dynamically determined chunks sizes, the type alias `ChunkedN` can be used instead. The
//!   previous example can then be reproduced as:
//!
//!   ```rust
//!   use flatk::ChunkedN;
//!
//!   let pos_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
//!
//!   let pos = ChunkedN::from_flat_with_stride(pos_data, 3);
//!
//!   assert_eq!(pos[0], [0.0; 3]);
//!   assert_eq!(pos[1], [1.0; 3]);
//!   assert_eq!(pos[2], [0.0, 1.0, 0.0]);
//!   ```
//!
//! - `Chunked`: Subdivides a collection into a number of unstructured (non-uniformly sized) chunks.
//!   For example we may have a non-uniform grouping of nodes stored in a `Vec`, which can represent a
//!   directed graph:
//!   
//!   ```rust
//!   use flatk::Chunked;
//!   
//!   let neighbours = vec![1, 2, 0, 1, 0, 1, 2];
//!   
//!   let neigh = Chunked::from_sizes(vec![1,2,1,3], neighbours);
//!
//!   assert_eq!(&neigh[0][..], &[1][..]);
//!   assert_eq!(&neigh[1][..], &[2, 0][..]);
//!   assert_eq!(&neigh[2][..], &[1][..]);
//!   assert_eq!(&neigh[3][..], &[0, 1, 2][..]);
//!   ```
//!
//!   Here `neigh` defines the following graph:
//!
//!   ```verbatim
//!   0<--->1<--->2
//!   ^     ^     ^
//!    \    |    /
//!     \   |   /
//!      \  |  /
//!       \ | /
//!        \|/
//!         3
//!   ```
//! - `Clumped`: A hybrid between `UniChunked` and `Chunked`, this type aggregates references to
//!   uniformly spaced chunks where possible.
//!   This makes it preferable for collections with mostly uniformly spaced chunks.
//!
//!   For example, polygons can be represented as indices into some global vertex array.
//!   Polygonal meshes are often made from a combination of triangles and quadrilaterals, so we
//!   can't represent the vertex indices as a `UniChunked` vector, and it would be too wastefull to
//!   keep track of each chunk using a plain `Chunked` vector. `Clumped`, however, is perfect for
//!   this use case since it only stores an additional pair of offsets (`usize` integers) for each
//!   type of polygon. In code this may look like the following:
//!   
//!   ```rust
//!   use flatk::{Clumped, Get, View};
//!   
//!   // Indices into some vertex array (depicted below): 6 triangles followed by 2 quadrilaterals.
//!   let indices = vec![0,1,2, 2,1,3, 7,1,0, 3,5,10, 9,8,7, 4,6,5, 7,8,4,1, 1,4,5,3];
//!   
//!   let polys = Clumped::from_sizes_and_counts(vec![3,4], vec![6,2], indices);
//!   let polys_view = polys.view();
//!
//!   assert_eq!(&polys_view.at(0)[..], &[0,1,2][..]);
//!   assert_eq!(&polys_view.at(1)[..], &[2,1,3][..]);
//!   assert_eq!(&polys_view.at(2)[..], &[7,1,0][..]);
//!   assert_eq!(&polys_view.at(3)[..], &[3,5,10][..]);
//!   assert_eq!(&polys_view.at(4)[..], &[9,8,7][..]);
//!   assert_eq!(&polys_view.at(5)[..], &[4,6,5][..]);
//!   assert_eq!(&polys_view.at(6)[..], &[7,8,4,1][..]);
//!   assert_eq!(&polys_view.at(7)[..], &[1,4,5,3][..]);
//!   ```
//!
//!   These polygons could represent a mesh like below, where each number corresponds to a vertex
//!   index.
//!
//!   ```verbatim
//!   0 ---- 2 ---- 3 --10
//!   |\     |     / \  |
//!   | \    |    /   \ |
//!   |  \   |   /     \|
//!   |   \  |  /       5
//!   |    \ | /       /|
//!   |     \|/       / |
//!   7 ---- 1       /  |
//!   |\      \     /   |
//!   | \      \   /    |
//!   |  \      \ /     |
//!   9 - 8 ---- 4 ---- 6
//!   ```
//!
//! - `Select`: An ordered selection (with replacement) of elements from a
//!   given random access collection. This is usually realized with a `Vec<usize>` representing indices
//!   into the original data collection.
//!
//!   For example one may wish to select game pieces in a board game:
//!
//!   ```rust
//!   use flatk::Select;
//!   
//!   let pieces = vec!["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"];
//!   
//!   let white_pieces = Select::new(vec![3, 1, 2, 5, 4, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0], pieces.as_slice());
//!   let black_pieces = Select::new(vec![0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 2, 5, 4, 2, 1, 3], pieces.as_slice());
//!
//!   assert_eq!(white_pieces[0], "Rook");
//!   assert_eq!(white_pieces[4], "Queen");
//!   assert_eq!(black_pieces[0], "Pawn");
//!   assert_eq!(black_pieces[11], "King");
//!   ```
//!
//! - `Subset`: Similar to `Select` but `Subset` enforces an unordered selection without replacement.
//!
//!   For example we can choose a hand from a deck of cards:
//!
//!   ```rust
//!   use flatk::{Subset, Get, View};
//!
//!   let rank = vec!["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"];
//!   let suit = vec!["Clubs", "Diamonds", "Hearts", "Spades"];
//!
//!   // Natural handling of structure of arrays (SoA) style data.
//!   let deck: (Vec<_>, Vec<_>) = (
//!       rank.into_iter().cycle().take(52).collect(),
//!       suit.into_iter().cycle().take(52).collect()
//!   );
//!
//!   let hand = Subset::from_indices(vec![4, 19, 23, 1, 0, 5], deck);
//!   let hand_view = hand.view();
//!   assert_eq!(hand_view.at(0), (&"Ace", &"Clubs"));
//!   assert_eq!(hand_view.at(1), (&"2", &"Diamonds"));
//!   assert_eq!(hand_view.at(2), (&"5", &"Clubs"));
//!   assert_eq!(hand_view.at(3), (&"6", &"Diamonds"));
//!   assert_eq!(hand_view.at(4), (&"7", &"Spades"));
//!   assert_eq!(hand_view.at(5), (&"Jack", &"Spades"));
//!   ```
//!
//! - `Sparse`: A sparse data assignment to another collection. Effectively this type attaches another
//!   data set to a `Select`ion. See [`Sparse`] for examples.
//!
//!
//! # Indexing
//!
//! Due to the nature of type composition and the indexing mechanism in Rust, it is not always
//! possible to use the `Index` and `IndexMut` traits for indexing into the `flatk` collection
//! types.  To facilitate indexing, `flatk` defines two traits for indexing: [`Get`] and
//! [`Isolate`], which fill the roles of `Index` and `IndexMut` respectively.  These traits work
//! mainly on viewed collections (what is returned by calling `.view()` and `.view_mut()`).
//! `Isolate` can also work with collections that own their data, however it is not recommended
//! since methods provided by `Isolate` are destructive (they consume `self`).
//!
//! [`Get`]: trait.Get.html
//! [`Isolate`]: trait.Isolate.html
//! [`Sparse`]: struct.Sparse.html

/*
 * Define macros to be used for implementing various traits in submodules
 */

macro_rules! impl_atom_iterators_recursive {
    (impl<S, $($type_vars_decl:tt),*> for $type:ident<S, $($type_vars:tt),*> { $data_field:ident }) => {
        impl<'a, S, $($type_vars_decl,)*> AtomIterator<'a> for $type<S, $($type_vars,)*>
        where S: AtomIterator<'a>,
        {
            type Item = S::Item;
            type Iter = S::Iter;
            fn atom_iter(&'a self) -> Self::Iter {
                self.$data_field.atom_iter()
            }
        }

        impl<'a, S, $($type_vars_decl,)*> AtomMutIterator<'a> for $type<S, $($type_vars,)*>
        where S: AtomMutIterator<'a>
        {
            type Item = S::Item;
            type Iter = S::Iter;
            fn atom_mut_iter(&'a mut self) -> Self::Iter {
                self.$data_field.atom_mut_iter()
            }
        }
    }
}

macro_rules! impl_isolate_index_for_static_range {
    (impl<$($type_vars:ident),*> for $type:ty) => {
        impl_isolate_index_for_static_range! { impl<$($type_vars),*> for $type where }
    };
    (impl<$($type_vars:ident),*> for $type:ty where $($constraints:tt)*) => {
        impl<$($type_vars,)* N: Unsigned> IsolateIndex<$type> for StaticRange<N>
        where
            std::ops::Range<usize>: IsolateIndex<$type>,
            $($constraints)*
        {
            type Output = <std::ops::Range<usize> as IsolateIndex<$type>>::Output;

            #[inline]
            unsafe fn isolate_unchecked(self, set: $type) -> Self::Output {
                IsolateIndex::isolate_unchecked(self.start..self.start + N::to_usize(), set)
            }
            #[inline]
            fn try_isolate(self, set: $type) -> Option<Self::Output> {
                IsolateIndex::try_isolate(self.start..self.start + N::to_usize(), set)
            }
        }
    }
}

mod array;
mod boxed;
pub mod chunked;
#[cfg(feature = "gpu")]
pub mod gpu;
mod range;
mod select;
mod slice;
#[cfg(feature = "sparse")]
mod sparse;
mod subset;
mod tuple;
mod vec;
mod view;
mod zip;

pub use array::*;
pub use boxed::*;
pub use chunked::*;
#[cfg(feature = "gpu")]
pub use gpu::IntoGpu;
pub use range::*;
pub use select::*;
pub use slice::*;
#[cfg(feature = "sparse")]
pub use sparse::*;
pub use subset::*;
pub use tuple::*;
pub use vec::*;
pub use view::*;

#[cfg(feature = "derive")]
pub use flatk_derive::Entity;

pub use typenum::consts;
use typenum::type_operators::PartialDiv;
pub use typenum::Unsigned;

/*
 * Set is the most basic trait that annotates finite collections that contain data.
 */
/// A trait defining a raw buffer of data. This data is typed but not annotated so it can represent
/// anything. For example a buffer of floats can represent a set of vertex colours or vertex
/// positions.
pub trait Set {
    /// Owned element of the set.
    type Elem;
    /// The most basic element contained by this collection.
    /// If this collection contains other collections, this type should be
    /// different than `Elem`.
    type Atom;
    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<N: Unsigned> Set for StaticRange<N> {
    type Elem = usize;
    type Atom = usize;
    #[inline]
    fn len(&self) -> usize {
        N::to_usize()
    }
}

impl<S: Set + ?Sized> Set for &S {
    type Elem = <S as Set>::Elem;
    type Atom = <S as Set>::Elem;
    #[inline]
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}
impl<S: Set + ?Sized> Set for &mut S {
    type Elem = <S as Set>::Elem;
    type Atom = <S as Set>::Elem;
    #[inline]
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}

impl<S: Set + ?Sized> Set for std::cell::Ref<'_, S> {
    type Elem = <S as Set>::Elem;
    type Atom = <S as Set>::Elem;
    #[inline]
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}

impl<S: Set + ?Sized> Set for std::cell::RefMut<'_, S> {
    type Elem = <S as Set>::Elem;
    type Atom = <S as Set>::Elem;
    #[inline]
    fn len(&self) -> usize {
        <S as Set>::len(self)
    }
}

/*
 * Array manipulation
 */

pub trait AsSlice<T> {
    fn as_slice(&self) -> &[T];
}

impl<T> AsSlice<T> for T {
    #[inline]
    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self as *const _, 1) }
    }
}

pub trait Array<T> {
    type Array: Set<Elem = T> + bytemuck::Pod;
    fn iter_mut(array: &mut Self::Array) -> std::slice::IterMut<T>;
    fn iter(array: &Self::Array) -> std::slice::Iter<T>;
    fn as_slice(array: &Self::Array) -> &[T];
}

/*
 * Marker and utility traits to help with Coherence rules of Rust.
 */

/// A marker trait to identify types whose range indices give a dynamically sized type even if the
/// range index is given as a StaticRange.
pub trait DynamicRangeIndexType {}
#[cfg(feature = "sparse")]
impl<S, T, I> DynamicRangeIndexType for Sparse<S, T, I> {}
impl<S, I> DynamicRangeIndexType for Select<S, I> {}
impl<S, I> DynamicRangeIndexType for Subset<S, I> {}
impl<S, I> DynamicRangeIndexType for Chunked<S, I> {}
impl<S> DynamicRangeIndexType for ChunkedN<S> {}

/// A marker trait to indicate an owned collection type. This is to distinguish
/// them from borrowed types, which is essential to resolve implementation collisions.
pub trait ValueType {}
#[cfg(feature = "sparse")]
impl<S, T, I> ValueType for Sparse<S, T, I> {}
impl<S, I> ValueType for Select<S, I> {}
impl<S, I> ValueType for Subset<S, I> {}
impl<S, I> ValueType for Chunked<S, I> {}
impl<S, N> ValueType for UniChunked<S, N> {}

impl<S: Viewed + ?Sized> Viewed for &S {}
impl<S: Viewed + ?Sized> Viewed for &mut S {}
#[cfg(feature = "sparse")]
impl<S: Viewed, T: Viewed, I: Viewed> Viewed for Sparse<S, T, I> {}
impl<S: Viewed, I: Viewed> Viewed for Select<S, I> {}
impl<S: Viewed, I: Viewed> Viewed for Subset<S, I> {}
impl<S: Viewed, I: Viewed> Viewed for Chunked<S, I> {}
impl<S: Viewed, N> Viewed for UniChunked<S, N> {}

/// A marker trait to indicate a collection type that can be chunked. More precisely this is a type that can be composed with types in this crate.
//pub trait Chunkable<'a>:
//    Set + Get<'a, 'a, std::ops::Range<usize>> + RemovePrefix + View<'a> + PartialEq
//{
//}
//impl<'a, T: Clone + PartialEq> Chunkable<'a> for &'a [T] {}
//impl<'a, T: Clone + PartialEq> Chunkable<'a> for &'a mut [T] {}
//impl<'a, T: Clone + PartialEq + 'a> Chunkable<'a> for Vec<T> {}

/*
 * Aggregate traits
 */

pub trait StaticallySplittable:
    IntoStaticChunkIterator<consts::U2>
    + IntoStaticChunkIterator<consts::U3>
    + IntoStaticChunkIterator<consts::U4>
    + IntoStaticChunkIterator<consts::U5>
    + IntoStaticChunkIterator<consts::U6>
    + IntoStaticChunkIterator<consts::U7>
    + IntoStaticChunkIterator<consts::U8>
    + IntoStaticChunkIterator<consts::U9>
    + IntoStaticChunkIterator<consts::U10>
    + IntoStaticChunkIterator<consts::U11>
    + IntoStaticChunkIterator<consts::U12>
    + IntoStaticChunkIterator<consts::U13>
    + IntoStaticChunkIterator<consts::U14>
    + IntoStaticChunkIterator<consts::U15>
    + IntoStaticChunkIterator<consts::U16>
{
}

impl<T> StaticallySplittable for T where
    T: IntoStaticChunkIterator<consts::U2>
        + IntoStaticChunkIterator<consts::U3>
        + IntoStaticChunkIterator<consts::U4>
        + IntoStaticChunkIterator<consts::U5>
        + IntoStaticChunkIterator<consts::U6>
        + IntoStaticChunkIterator<consts::U7>
        + IntoStaticChunkIterator<consts::U8>
        + IntoStaticChunkIterator<consts::U9>
        + IntoStaticChunkIterator<consts::U10>
        + IntoStaticChunkIterator<consts::U11>
        + IntoStaticChunkIterator<consts::U12>
        + IntoStaticChunkIterator<consts::U13>
        + IntoStaticChunkIterator<consts::U14>
        + IntoStaticChunkIterator<consts::U15>
        + IntoStaticChunkIterator<consts::U16>
{
}

pub trait ReadSet<'a>:
    Set
    + View<'a>
    + Get<'a, usize>
    + Get<'a, std::ops::Range<usize>>
    + Isolate<usize>
    + Isolate<std::ops::Range<usize>>
    + IntoOwned
    + IntoOwnedData
    + SplitAt
    + SplitOff
    + SplitFirst
    + IntoStorage
    + Dummy
    + RemovePrefix
    + IntoChunkIterator
    + StaticallySplittable
    + Viewed
    + IntoIterator
{
}

impl<'a, T> ReadSet<'a> for T where
    T: Set
        + View<'a>
        + Get<'a, usize>
        + Get<'a, std::ops::Range<usize>>
        + Isolate<usize>
        + Isolate<std::ops::Range<usize>>
        + IntoOwned
        + IntoOwnedData
        + SplitAt
        + SplitOff
        + SplitFirst
        + IntoStorage
        + Dummy
        + RemovePrefix
        + IntoChunkIterator
        + StaticallySplittable
        + Viewed
        + IntoIterator
{
}

pub trait WriteSet<'a>: ReadSet<'a> + ViewMut<'a> {}
impl<'a, T> WriteSet<'a> for T where T: ReadSet<'a> + ViewMut<'a> {}

pub trait OwnedSet<'a>:
    Set
    + View<'a>
    + ViewMut<'a>
    + Get<'a, usize>
    + Get<'a, std::ops::Range<usize>>
    + Isolate<usize>
    + Isolate<std::ops::Range<usize>>
    + IntoOwned
    + IntoOwnedData
    + SplitOff
    + IntoStorage
    + Dummy
    + RemovePrefix
    + IntoChunkIterator
    + StaticallySplittable
    + ValueType
{
}
impl<'a, T> OwnedSet<'a> for T where
    T: Set
        + View<'a>
        + ViewMut<'a>
        + Get<'a, usize>
        + Get<'a, std::ops::Range<usize>>
        + Isolate<usize>
        + Isolate<std::ops::Range<usize>>
        + IntoOwned
        + IntoOwnedData
        + SplitOff
        + IntoStorage
        + Dummy
        + RemovePrefix
        + IntoChunkIterator
        + StaticallySplittable
        + ValueType
{
}

/*
 * Allocation
 */

/// Abstraction for pushing elements of type `T` onto a collection.
pub trait Push<T> {
    fn push(&mut self, element: T);
}

pub trait ExtendFromSlice {
    type Item;
    fn extend_from_slice(&mut self, other: &[Self::Item]);
}

/*
 * Deallocation
 */

/// Truncate the collection to be a specified length.
pub trait Truncate {
    fn truncate(&mut self, len: usize);
}

pub trait Clear {
    /// Remove all elements from the current set without necessarily
    /// deallocating the space previously used.
    fn clear(&mut self);
}

/*
 * Conversion
 */

/// Convert a collection into its underlying representation, effectively
/// stripping any organizational info.
pub trait IntoStorage {
    type StorageType;
    fn into_storage(self) -> Self::StorageType;
}

/// Convert the storage type into another using the `Into` trait.
pub trait StorageInto<Target> {
    type Output;
    fn storage_into(self) -> Self::Output;
}

/// Map the storage type into another given a conversion function.
///
/// This is useful for changing storage is not just a simple `Vec` or slice but a combination of
/// independent collections.
pub trait MapStorage<Out> {
    type Input;
    type Output;
    fn map_storage<F: FnOnce(Self::Input) -> Out>(self, f: F) -> Self::Output;
}

pub trait CloneWithStorage<S> {
    type CloneType;
    fn clone_with_storage(&self, storage: S) -> Self::CloneType;
}

/// An analog to the `ToOwned` trait from `std` that works for chunked views.
/// As the name suggests, this version of `ToOwned` takes `self` by value.
pub trait IntoOwned
where
    Self: Sized,
{
    type Owned;
    fn into_owned(self) -> Self::Owned;
    #[inline]
    fn clone_into(self, target: &mut Self::Owned) {
        *target = self.into_owned();
    }
}

/// Blanket implementation of `IntoOwned` for references of types that are already
/// `std::borrow::ToOwned`.
impl<S: std::borrow::ToOwned + ?Sized> IntoOwned for &S {
    type Owned = S::Owned;
    #[inline]
    fn into_owned(self) -> Self::Owned {
        std::borrow::ToOwned::to_owned(self)
    }
}

/// Blanket implementation of `IntoOwned` for mutable references of types that are
/// already `std::borrow::ToOwned`.
impl<S: std::borrow::ToOwned + ?Sized> IntoOwned for &mut S {
    type Owned = S::Owned;
    #[inline]
    fn into_owned(self) -> Self::Owned {
        std::borrow::ToOwned::to_owned(self)
    }
}

/// In contrast to `IntoOwned`, this trait produces a clone with owned data, but
/// potentially borrowed structure of the collection.
pub trait IntoOwnedData
where
    Self: Sized,
{
    type OwnedData;
    fn into_owned_data(self) -> Self::OwnedData;
    #[inline]
    fn clone_into(self, target: &mut Self::OwnedData) {
        *target = self.into_owned_data();
    }
}

/// Blanked implementation of `IntoOwnedData` for references of types that are
/// already `std::borrow::ToOwned`.
impl<S: std::borrow::ToOwned + ?Sized> IntoOwnedData for &S {
    type OwnedData = S::Owned;
    #[inline]
    fn into_owned_data(self) -> Self::OwnedData {
        std::borrow::ToOwned::to_owned(self)
    }
}

/// Blanked implementation of `IntoOwnedData` for mutable references of types that are
/// already `std::borrow::ToOwned`.
impl<S: std::borrow::ToOwned + ?Sized> IntoOwnedData for &mut S {
    type OwnedData = S::Owned;
    #[inline]
    fn into_owned_data(self) -> Self::OwnedData {
        std::borrow::ToOwned::to_owned(self)
    }
}

/*
 * Indexing
 */

// A Note on indexing:
// ===================
// Following the standard library we support indexing by usize only.
// However, Ranges as collections can be supported for other types as well.

/// A helper trait to identify valid types for Range bounds for use as Sets.
pub trait IntBound:
    std::ops::Sub<Self, Output = Self>
    + std::ops::Add<usize, Output = Self>
    + Into<usize>
    + From<usize>
    + Clone
{
}

impl<T> IntBound for T where
    T: std::ops::Sub<Self, Output = Self>
        + std::ops::Add<usize, Output = Self>
        + Into<usize>
        + From<usize>
        + Clone
{
}

/// A definition of a bounded range.
pub trait BoundedRange {
    type Index: IntBound;
    fn start(&self) -> Self::Index;
    fn end(&self) -> Self::Index;
    #[inline]
    fn distance(&self) -> Self::Index {
        self.end() - self.start()
    }
}

/// A type of range whose size is determined at compile time.
/// This represents a range `[start..start + N::value()]`.
/// This aids `UniChunked` types when indexing.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct StaticRange<N> {
    pub start: usize,
    pub phantom: std::marker::PhantomData<N>,
}

impl<N> StaticRange<N> {
    #[inline]
    pub fn new(start: usize) -> Self {
        StaticRange {
            start,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<N: Unsigned> BoundedRange for StaticRange<N> {
    type Index = usize;
    #[inline]
    fn start(&self) -> usize {
        self.start
    }
    #[inline]
    fn end(&self) -> usize {
        self.start + N::to_usize()
    }
}

/// A helper trait analogous to `SliceIndex` from the standard library.
pub trait GetIndex<'a, S>
where
    S: ?Sized,
{
    type Output;
    fn get(self, set: &S) -> Option<Self::Output>;
    //unsafe fn get_unchecked(self, set: &'i S) -> Self::Output;
}

/// A helper trait like `GetIndex` but for `Isolate` types.
pub trait IsolateIndex<S> {
    type Output;
    fn try_isolate(self, set: S) -> Option<Self::Output>;
    unsafe fn isolate_unchecked(self, set: S) -> Self::Output;
}

/// An index trait for collection types.
/// Here `'i` indicates the lifetime of the input while `'o` indicates that of
/// the output.
pub trait Get<'a, I> {
    type Output;
    //unsafe fn get_unchecked(&'i self, idx: I) -> Self::Output;
    fn get(&self, idx: I) -> Option<Self::Output>;
    /// Return a value at the given index. This is provided as the checked
    /// version of `get` that will panic if the equivalent `get` call is `None`,
    /// which typically means that the given index is out of bounds.
    ///
    /// # Panics
    ///
    /// This function will panic if `self.get(idx)` returns `None`.
    #[inline]
    fn at(&self, idx: I) -> Self::Output {
        self.get(idx).expect("Index out of bounds")
    }
}

/// A blanket implementation of `Get` for any collection which has an implementation for `GetIndex`.
impl<'a, S, I> Get<'a, I> for S
where
    I: GetIndex<'a, Self>,
{
    type Output = I::Output;
    #[inline]
    fn get(&self, idx: I) -> Option<I::Output> {
        idx.get(self)
    }
}

/// Since we cannot alias mutable references, in order to index a mutable view
/// of elements, we must consume the original mutable reference. Since we can't
/// use slices for general composable collections, its impossible to match
/// against a `&mut self` in the getter function to be able to use it with owned
/// collections, so we opt to have an interface that is designed specifically
/// for mutably borrowed collections. For composable collections, this is better
/// described by a subview operator, which is precisely what this trait
/// represents. Incidentally this can also work for owned collections, which is
/// why it's called `Isolate` instead of `SubView`.
pub trait Isolate<I> {
    type Output;
    unsafe fn isolate_unchecked(self, idx: I) -> Self::Output;
    fn try_isolate(self, idx: I) -> Option<Self::Output>;
    /// Return a value at the given index. This is provided as the checked
    /// version of `try_isolate` that will panic if the equivalent `try_isolate`
    /// call is `None`, which typically means that the given index is out of
    /// bounds.
    ///
    /// # Panics
    ///
    /// This function will panic if `self.get(idx)` returns `None`.
    #[inline]
    fn isolate(self, idx: I) -> Self::Output
    where
        Self: Sized,
    {
        self.try_isolate(idx).expect("Index out of bounds")
    }
}

/// A blanket implementation of `Isolate` for any collection which has an implementation for `IsolateIndex`.
impl<S, I> Isolate<I> for S
where
    I: IsolateIndex<Self>,
{
    type Output = I::Output;
    #[inline]
    unsafe fn isolate_unchecked(self, idx: I) -> Self::Output {
        idx.isolate_unchecked(self)
    }
    #[inline]
    fn try_isolate(self, idx: I) -> Option<Self::Output> {
        idx.try_isolate(self)
    }
}

impl<'a, S, N> GetIndex<'a, S> for StaticRange<N>
where
    S: Set + DynamicRangeIndexType,
    N: Unsigned,
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    #[inline]
    fn get(self, set: &S) -> Option<Self::Output> {
        (self.start..self.start + N::to_usize()).get(set)
    }
}

impl<'a, S> GetIndex<'a, S> for std::ops::RangeFrom<usize>
where
    S: Set + ValueType,
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    #[inline]
    fn get(self, set: &S) -> Option<Self::Output> {
        (self.start..set.len()).get(set)
    }
}

impl<'a, S: ValueType> GetIndex<'a, S> for std::ops::RangeTo<usize>
where
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    #[inline]
    fn get(self, set: &S) -> Option<Self::Output> {
        (0..self.end).get(set)
    }
}

impl<'a, S> GetIndex<'a, S> for std::ops::RangeFull
where
    S: Set + ValueType,
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    #[inline]
    fn get(self, set: &S) -> Option<Self::Output> {
        (0..set.len()).get(set)
    }
}

impl<'a, S: ValueType> GetIndex<'a, S> for std::ops::RangeInclusive<usize>
where
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    #[allow(clippy::range_plus_one)]
    #[inline]
    fn get(self, set: &S) -> Option<Self::Output> {
        if *self.end() == usize::max_value() {
            None
        } else {
            (*self.start()..*self.end() + 1).get(set)
        }
    }
}

impl<'a, S: ValueType> GetIndex<'a, S> for std::ops::RangeToInclusive<usize>
where
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = <std::ops::Range<usize> as GetIndex<'a, S>>::Output;

    #[inline]
    fn get(self, set: &S) -> Option<Self::Output> {
        (0..=self.end).get(set)
    }
}

impl<S, N> IsolateIndex<S> for StaticRange<N>
where
    S: Set + DynamicRangeIndexType,
    N: Unsigned,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    #[inline]
    unsafe fn isolate_unchecked(self, set: S) -> Self::Output {
        IsolateIndex::isolate_unchecked(self.start..self.start + N::to_usize(), set)
    }
    #[inline]
    fn try_isolate(self, set: S) -> Option<Self::Output> {
        IsolateIndex::try_isolate(self.start..self.start + N::to_usize(), set)
    }
}

impl<S> IsolateIndex<S> for std::ops::RangeFrom<usize>
where
    S: Set + ValueType,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    #[inline]
    unsafe fn isolate_unchecked(self, set: S) -> Self::Output {
        IsolateIndex::isolate_unchecked(self.start..set.len(), set)
    }
    #[inline]
    fn try_isolate(self, set: S) -> Option<Self::Output> {
        IsolateIndex::try_isolate(self.start..set.len(), set)
    }
}

impl<S: ValueType> IsolateIndex<S> for std::ops::RangeTo<usize>
where
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    #[inline]
    unsafe fn isolate_unchecked(self, set: S) -> Self::Output {
        IsolateIndex::isolate_unchecked(0..self.end, set)
    }
    #[inline]
    fn try_isolate(self, set: S) -> Option<Self::Output> {
        IsolateIndex::try_isolate(0..self.end, set)
    }
}

impl<S: ValueType> IsolateIndex<S> for std::ops::RangeFull
where
    S: Set,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    #[inline]
    unsafe fn isolate_unchecked(self, set: S) -> Self::Output {
        IsolateIndex::isolate_unchecked(0..set.len(), set)
    }
    #[inline]
    fn try_isolate(self, set: S) -> Option<Self::Output> {
        IsolateIndex::try_isolate(0..set.len(), set)
    }
}

impl<S: ValueType> IsolateIndex<S> for std::ops::RangeInclusive<usize>
where
    S: Set,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    #[allow(clippy::range_plus_one)]
    #[inline]
    unsafe fn isolate_unchecked(self, set: S) -> Self::Output {
        IsolateIndex::isolate_unchecked(*self.start()..*self.end() + 1, set)
    }
    #[allow(clippy::range_plus_one)]
    #[inline]
    fn try_isolate(self, set: S) -> Option<Self::Output> {
        if *self.end() == usize::max_value() {
            None
        } else {
            IsolateIndex::try_isolate(*self.start()..*self.end() + 1, set)
        }
    }
}

impl<S: ValueType> IsolateIndex<S> for std::ops::RangeToInclusive<usize>
where
    S: Set,
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = <std::ops::Range<usize> as IsolateIndex<S>>::Output;

    #[inline]
    unsafe fn isolate_unchecked(self, set: S) -> Self::Output {
        IsolateIndex::isolate_unchecked(0..=self.end, set)
    }
    #[inline]
    fn try_isolate(self, set: S) -> Option<Self::Output> {
        IsolateIndex::try_isolate(0..=self.end, set)
    }
}

/// A helper trait to split a set into two sets at a given index.
/// This trait is used to implement iteration over `ChunkedView`s.
pub trait SplitAt
where
    Self: Sized,
{
    /// Split self into two sets at the given midpoint.
    /// This function is analogous to `<[T]>::split_at`.
    fn split_at(self, mid: usize) -> (Self, Self);
}

/// A helper trait to split owned sets into two sets at a given index.
/// This trait is used to implement iteration over `Chunked`s.
pub trait SplitOff {
    /// Split self into two sets at the given midpoint.
    /// This function is analogous to `Vec::split_off`.
    /// `self` contains elements `[0, mid)`, and The returned `Self` contains
    /// elements `[mid, len)`.
    fn split_off(&mut self, mid: usize) -> Self;
}

/// Split off a number of elements from the beginning of the collection where the number is determined at compile time.
pub trait SplitPrefix<N>
where
    Self: Sized,
{
    type Prefix;

    /// Split `N` items from the beginning of the collection.
    ///
    /// Return `None` if there are not enough items.
    fn split_prefix(self) -> Option<(Self::Prefix, Self)>;
}

/// Split out the first element of a collection.
pub trait SplitFirst
where
    Self: Sized,
{
    type First;
    fn split_first(self) -> Option<(Self::First, Self)>;
}

/// Get an immutable reference to the underlying storage type.
pub trait Storage {
    type Storage: ?Sized;
    fn storage(&self) -> &Self::Storage;
}

impl<S: Storage + ?Sized> Storage for &S {
    type Storage = S::Storage;
    fn storage(&self) -> &Self::Storage {
        S::storage(*self)
    }
}

impl<S: Storage + ?Sized> Storage for &mut S {
    type Storage = S::Storage;
    fn storage(&self) -> &Self::Storage {
        S::storage(*self)
    }
}

pub trait StorageView<'a> {
    type StorageView;
    fn storage_view(&'a self) -> Self::StorageView;
}

/// Get a mutable reference to the underlying storage type.
pub trait StorageMut: Storage {
    fn storage_mut(&mut self) -> &mut Self::Storage;
}

impl<S: StorageMut + ?Sized> StorageMut for &mut S {
    fn storage_mut(&mut self) -> &mut Self::Storage {
        S::storage_mut(*self)
    }
}

/// A helper trait for constructing placeholder sets for use in `std::mem::replace`.
/// These don't necessarily have to correspond to bona-fide sets and can
/// potentially produce invalid sets. For this reason this function can be
/// unsafe since it can generate collections that don't uphold their invariants
/// for the sake of avoiding allocations.
pub trait Dummy {
    unsafe fn dummy() -> Self;
}

/// A helper trait used to help implement the Subset. This trait allows
/// abstract collections to remove a number of elements from the
/// beginning, which is what we need for subsets.
// Technically this is a deallocation trait, but it's only used to enable
// iteration on subsets so it's here.
pub trait RemovePrefix {
    /// Remove `n` elements from the beginning.
    fn remove_prefix(&mut self, n: usize);
}

/// This trait generalizes the method `chunks` available on slices in the
/// standard library. Collections that can be chunked by a runtime stride should
/// implement this behaviour such that they can be composed with `ChunkedN`
/// types.
pub trait IntoChunkIterator {
    type Item;
    type IterType: Iterator<Item = Self::Item>;

    /// Produce a chunk iterator with the given stride `chunk_size`.
    /// One notable difference between this trait and `chunks*` methods on slices is that
    /// `chunks_iter` should panic when the underlying data cannot split into `chunk_size` sized
    /// chunks exactly.
    fn into_chunk_iter(self, chunk_size: usize) -> Self::IterType;
}

// Implement IntoChunkIterator for all types that implement Set, SplitAt and Dummy.
// This should be reimplemented like IntoStaticChunkIterator to avoid expensive iteration of allocating types.
impl<S> IntoChunkIterator for S
where
    S: Set + SplitAt + Dummy,
{
    type Item = S;
    type IterType = ChunkedNIter<S>;

    #[inline]
    fn into_chunk_iter(self, chunk_size: usize) -> Self::IterType {
        assert_eq!(self.len() % chunk_size, 0);
        ChunkedNIter {
            chunk_size,
            data: self,
        }
    }
}

/// Parallel version of `IntoChunkIterator`.
#[cfg(feature = "rayon")]
pub trait IntoParChunkIterator {
    type Item: Send;
    type IterType: rayon::iter::IndexedParallelIterator<Item = Self::Item>;

    fn into_par_chunk_iter(self, chunk_size: usize) -> Self::IterType;
}

/// A trait intended to be implemented on collection types to define the type of
/// a statically sized chunk in this collection.
/// This trait is required for composing with `UniChunked`.
pub trait UniChunkable<N> {
    type Chunk;
}

/// Iterate over chunks whose size is determined at compile time.
///
/// Note that each chunk may not be a simple array, although a statically sized
/// chunk of a slice is an array.
pub trait IntoStaticChunkIterator<N>
where
    Self: Sized + Set,
    N: Unsigned,
{
    type Item;
    type IterType: Iterator<Item = Self::Item>;

    /// This function should panic if this collection length is not a multiple
    /// of `N`.
    fn into_static_chunk_iter(self) -> Self::IterType;

    /// Simply call this method for all types that implement `SplitPrefix<N>`.
    #[inline]
    fn into_generic_static_chunk_iter(self) -> UniChunkedIter<Self, N> {
        assert_eq!(self.len() % N::to_usize(), 0);
        UniChunkedIter {
            chunk_size: std::marker::PhantomData,
            data: self,
        }
    }
}

/// A trait that allows the container to allocate additional space without
/// changing any of the data. The container should allocate space for at least
/// `n` additional elements.
///
/// Composite collections such a `Chunked` or `Select` may choose to only
/// reserve primary level storage if the amount of total storage required cannot
/// be specified by a single number in `reserve`. This is the default behaviour
/// of the `reserve` function below. The `reserve_with_storage` method allows
/// the caller to also specify the amount of storage needed for the container at
/// the lowest level.
pub trait Reserve {
    #[inline]
    fn reserve(&mut self, n: usize) {
        self.reserve_with_storage(n, 0); // By default we ignore storage.
    }
    fn reserve_with_storage(&mut self, n: usize, storage_n: usize);
}

/*
 * New experimental traits below
 */

pub trait SwapChunks {
    /// Swap equal sized contiguous chunks in this collection.
    fn swap_chunks(&mut self, begin_a: usize, begin_b: usize, chunk_size: usize);
}

pub trait Sort {
    /// Sort the given indices into this collection with respect to values provided by this collection.
    fn sort_indices(&self, indices: &mut [usize]);
}

pub trait PermuteInPlace {
    fn permute_in_place(&mut self, indices: &[usize], seen: &mut [bool]);
}

/// This trait is used to produce the chunk size of a collection if it contains uniformly chunked
/// elements.
pub trait ChunkSize {
    fn chunk_size(&self) -> usize;
}

/// Clone self into a potentially different collection.
pub trait CloneIntoOther<T = Self>
where
    T: ?Sized,
{
    fn clone_into_other(&self, other: &mut T);
}

impl<T: Clone> CloneIntoOther<&mut T> for &T {
    #[inline]
    fn clone_into_other(&self, other: &mut &mut T) {
        other.clone_from(self);
    }
}

pub trait AtomIterator<'a> {
    type Item;
    type Iter: Iterator<Item = Self::Item>;
    fn atom_iter(&'a self) -> Self::Iter;
}

pub trait AtomMutIterator<'a> {
    type Item;
    type Iter: Iterator<Item = Self::Item>;
    fn atom_mut_iter(&'a mut self) -> Self::Iter;
}

// Blanket implementations of AtomIterator/AtomMutIterator for references
impl<'a, S: ?Sized> AtomIterator<'a> for &S
where
    S: AtomIterator<'a>,
{
    type Item = S::Item;
    type Iter = S::Iter;
    #[inline]
    fn atom_iter(&'a self) -> Self::Iter {
        S::atom_iter(self)
    }
}

impl<'a, S: ?Sized> AtomMutIterator<'a> for &mut S
where
    S: AtomMutIterator<'a>,
{
    type Item = S::Item;
    type Iter = S::Iter;
    #[inline]
    fn atom_mut_iter(&'a mut self) -> Self::Iter {
        S::atom_mut_iter(self)
    }
}

/// A wraper for a zip iterator that unwraps its contents into a custom struct.
pub struct StructIter<I, T> {
    iter: I,
    phantom: std::marker::PhantomData<T>,
}

impl<I, T> StructIter<I, T> {
    #[inline]
    pub fn new(iter: I) -> Self {
        StructIter {
            iter,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<I: Iterator, T: From<I::Item>> Iterator for StructIter<I, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(From::from)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n).map(From::from)
    }
}

impl<I, T> DoubleEndedIterator for StructIter<I, T>
where
    I: DoubleEndedIterator + ExactSizeIterator,
    T: From<I::Item>,
{
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back().map(From::from)
    }
}

/// An iterator whose items are random-accessible efficiently
///
/// # Safety
///
/// The iterator's .len() and size_hint() must be exact.
/// `.len()` must be cheap to call.
///
/// .get_unchecked() must return distinct mutable references for distinct
/// indices (if applicable), and must return a valid reference if index is in
/// 0..self.len().
pub unsafe trait TrustedRandomAccess: ExactSizeIterator {
    unsafe fn get_unchecked(&mut self, i: usize) -> Self::Item;
    /// Returns `true` if getting an iterator element may have
    /// side effects. Remember to take inner iterators into account.
    fn may_have_side_effect() -> bool {
        false
    }
}

/*
 * Tests
 */

///```compile_fail
/// use flatk::*;
/// // This shouldn't compile
/// let v: Vec<usize> = (1..=10).collect();
/// let chunked = Chunked::from_offsets(vec![0, 3, 5, 8, 10], v);
/// let mut chunked = Chunked::from_offsets(vec![0, 1, 4], chunked);
/// let mut mut_view = chunked.view_mut();
///
/// // The .at should not work with a mutable view.
/// let mut1 = mut_view.at(1).at(1);
/// // We should at least fail to compile when trying to get a second mut ref.
/// let mut2 = mut_view.at(1).at(1);
///```
#[doc(hidden)]
pub fn multiple_mut_refs_compile_test() {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test iteration of a `Chunked` inside a `Chunked`.
    #[test]
    fn var_of_uni_iter_test() {
        let u0 = Chunked2::from_flat((1..=12).collect::<Vec<_>>());
        let v1 = Chunked::from_offsets(vec![0, 2, 3, 6], u0);

        let mut iter1 = v1.iter();
        let v0 = iter1.next().unwrap();
        let mut iter0 = v0.iter();
        assert_eq!(Some(&[1, 2]), iter0.next());
        assert_eq!(Some(&[3, 4]), iter0.next());
        assert_eq!(None, iter0.next());
        let v0 = iter1.next().unwrap();
        let mut iter0 = v0.iter();
        assert_eq!(Some(&[5, 6]), iter0.next());
        assert_eq!(None, iter0.next());
        let v0 = iter1.next().unwrap();
        let mut iter0 = v0.iter();
        assert_eq!(Some(&[7, 8]), iter0.next());
        assert_eq!(Some(&[9, 10]), iter0.next());
        assert_eq!(Some(&[11, 12]), iter0.next());
        assert_eq!(None, iter0.next());
    }

    #[cfg(feature = "derive")]
    mod derive_tests {
        /*
         * Test the use of the `Entity` derive macro
         */

        // Needed to make the derive macro work in the test context.
        use super::*;
        use crate as flatk;
        use flatk::Entity;

        #[derive(Copy, Clone, Debug, PartialEq, Entity)]
        struct MyEntity<X, V> {
            // Unused parameter, that is simply copied through to views and items.
            id: usize,
            x: X,
            v: V,
        }

        #[test]
        fn entity_derive_test() {
            let mut e = MyEntity {
                id: 0,
                x: vec![1.0; 12],
                v: vec![7.0; 12],
            };

            // Get the size of the entity set
            assert_eq!(e.len(), 12);

            // Construct a View and Get a single element from MyEntity.
            assert_eq!(
                e.view().at(0),
                MyEntity {
                    id: 0,
                    x: &1.0,
                    v: &7.0
                }
            );

            // Construct a ViewMut and modify a single entry
            let entry_mut = e.view_mut().isolate(0);
            *entry_mut.x = 13.0;
            *entry_mut.v = 14.0;
            assert_eq!(
                e.view().at(0),
                MyEntity {
                    id: 0,
                    x: &13.0,
                    v: &14.0
                }
            );

            let chunked3 = Chunked3::from_flat(e.clone());
            assert_eq!(
                chunked3.view().at(0),
                MyEntity {
                    id: 0,
                    x: &[13.0, 1.0, 1.0],
                    v: &[14.0, 7.0, 7.0]
                }
            );

            let chunked = Chunked::from_sizes(vec![1, 3], Chunked3::from_flat(e));

            assert_eq!(
                chunked.view().at(0).at(0),
                MyEntity {
                    id: 0,
                    x: &[13.0, 1.0, 1.0],
                    v: &[14.0, 7.0, 7.0]
                }
            );
        }
    }
}
