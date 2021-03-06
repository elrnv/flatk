//!
//! Implementation of generic array manipulation.
//!

use super::*;

/// Wrapper around `typenum` types to prevent downstream trait implementations.
#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub struct U<N>(N);

impl<N: Unsigned> std::fmt::Debug for U<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&format!("U{}", N::to_usize())).finish()
    }
}

impl<N: Default> Default for U<N> {
    #[inline]
    fn default() -> Self {
        U(N::default())
    }
}

macro_rules! impl_array_for_typenum {
    ($nty:ident, $n:expr) => {
        pub type $nty = U<consts::$nty>;
        impl<T> Set for [T; $n] {
            type Elem = T;
            type Atom = T;
            #[inline]
            fn len(&self) -> usize {
                $n
            }
        }
        impl<T> AsFlatSlice<T> for [T; $n] {
            #[inline]
            fn as_flat_slice(&self) -> &[T] {
                &self[..]
            }
        }
        impl<'a, T: 'a> View<'a> for [T; $n] {
            type Type = &'a [T; $n];
            #[inline]
            fn view(&'a self) -> Self::Type {
                self
            }
        }
        impl<T> Viewed for &[T; $n] {}
        impl<T> Viewed for &mut [T; $n] {}

        impl<'a, T: 'a> ViewMut<'a> for [T; $n] {
            type Type = &'a mut [T; $n];
            #[inline]
            fn view_mut(&'a mut self) -> Self::Type {
                self
            }
        }
        impl<T: Dummy + Copy> Dummy for [T; $n] {
            #[inline]
            unsafe fn dummy() -> Self {
                [Dummy::dummy(); $n]
            }
        }

        impl<'a, T, N> GetIndex<'a, &'a [T; $n]> for StaticRange<N>
        where
            N: Unsigned + Array<T>,
            <N as Array<T>>::Array: 'a,
        {
            type Output = &'a N::Array;
            #[inline]
            fn get(self, set: &&'a [T; $n]) -> Option<Self::Output> {
                if self.end() <= set.len() {
                    let slice = *set;
                    Some(unsafe { &*(slice.as_ptr().add(self.start()) as *const N::Array) })
                } else {
                    None
                }
            }
        }

        impl<'a, T, N> IsolateIndex<&'a [T; $n]> for StaticRange<N>
        where
            N: Unsigned + Array<T>,
            <N as Array<T>>::Array: 'a,
        {
            type Output = &'a N::Array;
            #[inline]
            unsafe fn isolate_unchecked(self, set: &'a [T; $n]) -> Self::Output {
                &*(set.as_ptr().add(self.start()) as *const N::Array)
            }
            #[inline]
            fn try_isolate(self, set: &'a [T; $n]) -> Option<Self::Output> {
                if self.end() <= set.len() {
                    Some(unsafe { IsolateIndex::isolate_unchecked(self, set) })
                } else {
                    None
                }
            }
        }

        impl<T: bytemuck::Pod> Array<T> for consts::$nty {
            type Array = [T; $n];

            #[inline]
            fn iter_mut(array: &mut Self::Array) -> std::slice::IterMut<T> {
                array.iter_mut()
            }
            #[inline]
            fn iter(array: &Self::Array) -> std::slice::Iter<T> {
                array.iter()
            }
            #[inline]
            fn as_slice(array: &Self::Array) -> &[T] {
                array
            }
        }

        impl<'a, T, N> ReinterpretAsGrouped<N> for &'a [T; $n]
        where
            T: bytemuck::Pod,
            N: Unsigned + Array<T>,
            consts::$nty: PartialDiv<N>,
            <consts::$nty as PartialDiv<N>>::Output: Array<N::Array> + Unsigned,
            <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array: 'a,
        {
            type Output = &'a <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array;
            #[inline]
            fn reinterpret_as_grouped(self) -> Self::Output {
                assert_eq!(
                    $n,
                    N::to_usize()
                        * <<consts::$nty as PartialDiv<N>>::Output as Unsigned>::to_usize()
                );
                //unsafe {
                //    &*(self as *const [T; $n]
                //        as *const <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array)
                //}
                bytemuck::cast_ref(self)
            }
        }

        impl<'a, T, N> ReinterpretAsGrouped<N> for &'a mut [T; $n]
        where
            T: bytemuck::Pod,
            N: Unsigned + Array<T>,
            consts::$nty: PartialDiv<N>,
            <consts::$nty as PartialDiv<N>>::Output: Array<N::Array> + Unsigned,
            <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array: 'a,
        {
            type Output =
                &'a mut <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array;
            #[inline]
            fn reinterpret_as_grouped(self) -> Self::Output {
                assert_eq!(
                    $n,
                    N::to_usize()
                        * <<consts::$nty as PartialDiv<N>>::Output as Unsigned>::to_usize()
                );
                //unsafe {
                //    &mut *(self as *mut [T; $n]
                //        as *mut <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array)
                //}
                bytemuck::cast_mut(self)
            }
        }

        impl<T, N: Array<T>> UniChunkable<N> for [T; $n] {
            type Chunk = N::Array;
        }

        impl<'a, T, N: Array<T>> UniChunkable<N> for &'a [T; $n]
        where
            <N as Array<T>>::Array: 'a,
        {
            type Chunk = &'a N::Array;
        }

        impl<'a, T, N: Array<T>> UniChunkable<N> for &'a mut [T; $n]
        where
            <N as Array<T>>::Array: 'a,
        {
            type Chunk = &'a mut N::Array;
        }

        impl<'a, T, N> IntoStaticChunkIterator<N> for &'a [T; $n]
        where
            N: Unsigned + Array<T>,
        {
            type Item = <&'a [T] as SplitPrefix<N>>::Prefix;
            type IterType = UniChunkedIter<&'a [T], N>;
            #[inline]
            fn into_static_chunk_iter(self) -> Self::IterType {
                (&self[..]).into_generic_static_chunk_iter()
            }
        }

        impl<'a, T, N> IntoStaticChunkIterator<N> for &'a mut [T; $n]
        where
            N: Unsigned + Array<T>,
        {
            type Item = <&'a mut [T] as SplitPrefix<N>>::Prefix;
            type IterType = UniChunkedIter<&'a mut [T], N>;
            #[inline]
            fn into_static_chunk_iter(self) -> Self::IterType {
                (&mut self[..]).into_generic_static_chunk_iter()
            }
        }

        impl<T: Clone> CloneIntoOther<[T; $n]> for [T; $n] {
            #[inline]
            fn clone_into_other(&self, other: &mut [T; $n]) {
                other.clone_from(self);
            }
        }

        impl<T: Clone> CloneIntoOther<&mut [T; $n]> for [T; $n] {
            #[inline]
            fn clone_into_other(&self, other: &mut &mut [T; $n]) {
                (*other).clone_from(self);
            }
        }

        impl<'a, T: 'a> AtomIterator<'a> for [T; $n] {
            type Item = &'a T;
            type Iter = std::slice::Iter<'a, T>;
            #[inline]
            fn atom_iter(&'a self) -> Self::Iter {
                self.iter()
            }
        }

        impl<'a, T: 'a> AtomMutIterator<'a> for [T; $n] {
            type Item = &'a mut T;
            type Iter = std::slice::IterMut<'a, T>;
            #[inline]
            fn atom_mut_iter(&'a mut self) -> Self::Iter {
                self.iter_mut()
            }
        }

        // TODO: Figure out how to compile the below code.
        //        impl<T, N> ReinterpretAsGrouped<N> for [T; $n]
        //        where
        //            N: Unsigned + Array<T>,
        //            consts::$nty: PartialDiv<N>,
        //            <consts::$nty as PartialDiv<N>>::Output: Array<N::Array> + Unsigned,
        //        {
        //            type Output = <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array;
        //            #[inline]
        //            fn reinterpret_as_grouped(self) -> Self::Output {
        //                assert_eq!(
        //                    $n / N::to_usize(),
        //                    <<consts::$nty as PartialDiv<N>>::Output as Unsigned>::to_usize()
        //                );
        //                unsafe {
        //                    std::mem::transmute::<
        //                        Self,
        //                        <<consts::$nty as PartialDiv<N>>::Output as Array<N::Array>>::Array,
        //                    >(self)
        //                }
        //            }
        //        }
    };
}

impl_array_for_typenum!(U1, 1);
impl_array_for_typenum!(U2, 2);
impl_array_for_typenum!(U3, 3);
impl_array_for_typenum!(U4, 4);
impl_array_for_typenum!(U5, 5);
impl_array_for_typenum!(U6, 6);
impl_array_for_typenum!(U7, 7);
impl_array_for_typenum!(U8, 8);
impl_array_for_typenum!(U9, 9);
impl_array_for_typenum!(U10, 10);
impl_array_for_typenum!(U11, 11);
impl_array_for_typenum!(U12, 12);
impl_array_for_typenum!(U13, 13);
impl_array_for_typenum!(U14, 14);
impl_array_for_typenum!(U15, 15);
impl_array_for_typenum!(U16, 16);

macro_rules! impl_as_slice_for_2d_array {
    ($r:expr, $c:expr) => {
        impl<T: bytemuck::Pod> AsFlatSlice<T> for [[T; $c]; $r] {
            #[inline]
            fn as_flat_slice(&self) -> &[T] {
                //unsafe { reinterpret::reinterpret_slice(&self[..]) }
                bytemuck::cast_slice(self)
            }
        }
    };
}

impl_as_slice_for_2d_array!(1, 1);
impl_as_slice_for_2d_array!(1, 2);
impl_as_slice_for_2d_array!(1, 3);
impl_as_slice_for_2d_array!(1, 4);
impl_as_slice_for_2d_array!(1, 5);
impl_as_slice_for_2d_array!(1, 6);
impl_as_slice_for_2d_array!(1, 7);
impl_as_slice_for_2d_array!(1, 8);
impl_as_slice_for_2d_array!(1, 9);

impl_as_slice_for_2d_array!(2, 1);
impl_as_slice_for_2d_array!(2, 2);
impl_as_slice_for_2d_array!(2, 3);
impl_as_slice_for_2d_array!(2, 4);
impl_as_slice_for_2d_array!(2, 5);
impl_as_slice_for_2d_array!(2, 6);
impl_as_slice_for_2d_array!(2, 7);
impl_as_slice_for_2d_array!(2, 8);
impl_as_slice_for_2d_array!(2, 9);

impl_as_slice_for_2d_array!(3, 1);
impl_as_slice_for_2d_array!(3, 2);
impl_as_slice_for_2d_array!(3, 3);
impl_as_slice_for_2d_array!(3, 4);
impl_as_slice_for_2d_array!(3, 5);
impl_as_slice_for_2d_array!(3, 6);
impl_as_slice_for_2d_array!(3, 7);
impl_as_slice_for_2d_array!(3, 8);
impl_as_slice_for_2d_array!(3, 9);

impl_as_slice_for_2d_array!(4, 1);
impl_as_slice_for_2d_array!(4, 2);
impl_as_slice_for_2d_array!(4, 3);
impl_as_slice_for_2d_array!(4, 4);
impl_as_slice_for_2d_array!(4, 5);
impl_as_slice_for_2d_array!(4, 6);
impl_as_slice_for_2d_array!(4, 7);
impl_as_slice_for_2d_array!(4, 8);
impl_as_slice_for_2d_array!(4, 9);

impl_as_slice_for_2d_array!(5, 1);
impl_as_slice_for_2d_array!(5, 2);
impl_as_slice_for_2d_array!(5, 3);
impl_as_slice_for_2d_array!(5, 4);
impl_as_slice_for_2d_array!(5, 5);
impl_as_slice_for_2d_array!(5, 6);
impl_as_slice_for_2d_array!(5, 7);
impl_as_slice_for_2d_array!(5, 8);
impl_as_slice_for_2d_array!(5, 9);

impl_as_slice_for_2d_array!(6, 1);
impl_as_slice_for_2d_array!(6, 2);
impl_as_slice_for_2d_array!(6, 3);
impl_as_slice_for_2d_array!(6, 4);
impl_as_slice_for_2d_array!(6, 5);
impl_as_slice_for_2d_array!(6, 6);
impl_as_slice_for_2d_array!(6, 7);
impl_as_slice_for_2d_array!(6, 8);
impl_as_slice_for_2d_array!(6, 9);

impl_as_slice_for_2d_array!(7, 1);
impl_as_slice_for_2d_array!(7, 2);
impl_as_slice_for_2d_array!(7, 3);
impl_as_slice_for_2d_array!(7, 4);
impl_as_slice_for_2d_array!(7, 5);
impl_as_slice_for_2d_array!(7, 6);
impl_as_slice_for_2d_array!(7, 7);
impl_as_slice_for_2d_array!(7, 8);
impl_as_slice_for_2d_array!(7, 9);

impl_as_slice_for_2d_array!(8, 1);
impl_as_slice_for_2d_array!(8, 2);
impl_as_slice_for_2d_array!(8, 3);
impl_as_slice_for_2d_array!(8, 4);
impl_as_slice_for_2d_array!(8, 5);
impl_as_slice_for_2d_array!(8, 6);
impl_as_slice_for_2d_array!(8, 7);
impl_as_slice_for_2d_array!(8, 8);
impl_as_slice_for_2d_array!(8, 9);

impl_as_slice_for_2d_array!(9, 1);
impl_as_slice_for_2d_array!(9, 2);
impl_as_slice_for_2d_array!(9, 3);
impl_as_slice_for_2d_array!(9, 4);
impl_as_slice_for_2d_array!(9, 5);
impl_as_slice_for_2d_array!(9, 6);
impl_as_slice_for_2d_array!(9, 7);
impl_as_slice_for_2d_array!(9, 8);
impl_as_slice_for_2d_array!(9, 9);
