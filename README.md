# `flatk`

**F**lat **l**ayout **a**bstraction **t**ool**k**it.

[![On crates.io](https://img.shields.io/crates/v/flatk.svg)](https://crates.io/crates/flatk)
[![On docs.rs](https://docs.rs/flatk/badge.svg)](https://docs.rs/flatk/)
[![Travis CI](https://travis-ci.org/elrnv/flatk.svg?branch=master)](https://travis-ci.org/elrnv/flatk)
[![Github Actions CI](https://github.com/elrnv/flatk/workflows/CI/badge.svg)](https://github.com/elrnv/flatk/actions?query=workflow%3ACI)

This library defines low level primitives for organizing flat ordered data collections (like `Vec`s
and `slice`s) into meaningful structures without cloning the data.

More specifically, `flatk` provides a few core composable types intended for building more complex
data structures out of existing data:

- `UniChunked`:  Subdivides a collection into a number of uniformly sized (at compile time or
  run-time) contiguous groups.
  For example if we have a `Vec` of floats representing 3D positions, we may wish to interpret them
  as triplets:

  ```rust
  use flatk::Chunked3;

  let pos_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];

  let pos = Chunked3::from_flat(pos_data);

  assert_eq!(pos[0], [0.0; 3]);
  assert_eq!(pos[1], [1.0; 3]);
  assert_eq!(pos[2], [0.0, 1.0, 0.0]);
  ```

- `Chunked`: Subdivides a collection into a number of unstructured (non-uniform) groups.
  For example we may have a non-uniform grouping of nodes stored in a `Vec`, which can represent a
  directed graph:
  
  ```rust
  use flatk::Chunked;
  
  let neighbours = vec![1, 2, 0, 1, 0, 1, 2];
  
  let neigh = Chunked::from_sizes(vec![1,2,1,3], neighbours);
  
  assert_eq!(&neigh[0][..], &[1][..]);
  assert_eq!(&neigh[1][..], &[2, 0][..]);
  assert_eq!(&neigh[2][..], &[1][..]);
  assert_eq!(&neigh[3][..], &[0, 1, 2][..]);
  ```

  Here `neigh` defines the following graph:
  
  ```verbatim
  0<--->1<--->2
  ^     ^     ^
   \    |    /
    \   |   /
     \  |  /
      \ | /
       \|/
        3
  ```

- `Select`: An ordered selection (with replacement) of elements from a
  given random access collection. This is usually realized with a `Vec<usize>` representing indices
  into the original data collection.

  For example one may wish to select game pieces in a board game:

  ```rust
  use flatk::Select;
  
  let pieces = vec!["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"];
  
  let white_pieces = Select::new(vec![3, 1, 2, 5, 4, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0], pieces.as_slice());
  let black_pieces = Select::new(vec![0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 2, 5, 4, 2, 1, 3], pieces.as_slice());

  assert_eq!(white_pieces[0], "Rook");
  assert_eq!(white_pieces[4], "Queen");
  assert_eq!(black_pieces[0], "Pawn");
  assert_eq!(black_pieces[11], "King");
  ```

- `Subset`: Similar to `Select` but `Subset` enforces an unordered selection without replacement.

  For example we can choose a hand from a deck of cards:

  ```rust
  use flatk::{Subset, Get, View};

  let rank = vec!["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"];
  let suit = vec!["Clubs", "Diamonds", "Hearts", "Spades"];

  // Natural handling of structure of arrays (SoA) style data.
  let deck: (Vec<_>, Vec<_>) = (
      rank.into_iter().cycle().take(52).collect(),
      suit.into_iter().cycle().take(52).collect()
  );

  let hand = Subset::from_indices(vec![4, 19, 23, 1, 0, 5], deck);
  let hand_view = hand.view();
  assert_eq!(hand_view.at(0), (&"Ace", &"Clubs"));
  assert_eq!(hand_view.at(1), (&"2", &"Diamonds"));
  assert_eq!(hand_view.at(2), (&"5", &"Clubs"));
  assert_eq!(hand_view.at(3), (&"6", &"Diamonds"));
  assert_eq!(hand_view.at(4), (&"7", &"Spades"));
  assert_eq!(hand_view.at(5), (&"Jack", &"Spades"));
  ```

- `Sparse`: A sparse data assignment to another collection. Effectively this type attaches another
  data set to a `Select`ion.

  For example we can represent a sparse vector by assigning values to a selection of indices:
  
  ```rust
  use flatk::{Sparse, Get, View};
  
  let values = vec![1.0, 2.0, 3.0, 4.0];
  let sparse_vector = Sparse::from_dim(vec![0,5,10,100], 1000, values);
  let sparse_vector_view = sparse_vector.view();
  
  assert_eq!(sparse_vector_view.at(0), (0, &1.0));
  assert_eq!(sparse_vector_view.at(1), (5, &2.0));
  assert_eq!(sparse_vector_view.at(2), (10, &3.0));
  assert_eq!(sparse_vector_view.at(3), (100, &4.0));
  assert_eq!(sparse_vector_view.selection.target, ..1000);
  ```

  In this scenario, the target set is just the range `0..1000`, however in general this can be any
  data set, which makes `Sparse` an implementation of a one-to-one mapping or a directed graph
  with disjoint source and target node sets.


# Goals

Currently, the goal of this library is to provide useful primitives for organizing large arrays of
data with the following features:

 - composability,
 - iteration,
 - random access (indexing)

# A Motivating Example

Suppose we want to build a sparse matrix of 3x3 dense block matrices.  To motivate why we may want
to do this, we can evaluate a common alternative approach. One may choose to use an
off-the-shelf sparse matrix like a compressed sparse row (CSR) matrix, which will redistribute
the values of the 3x3 blocks into non-local positions in the value array. This can have performance
implications, for instance, when computing matrix products.  Moreover, a standard CSR matrix will
need to iterate 9x more times than the equivalent block CSR (BCSR) matrix. So let's walk through
the process of defining a BCSR matrix using traditional Rust and then using our primitive types.

The compressed sparse row structure consists of a dynamically sized array of rows, and a dynamically
sized array of elements in each row along with accompanying column indices. In traditional Rust we
may write this as `Vec<Vec<(usize, T)>>` where `usize` is the column index and `T` is a element type
like `f32`. This type loses locality between rows and incurs additional indirection for each row, as
well as other problems with Array or Structures (AoS) types (e.g. lack of automatic SIMD
optimizations).  To alleviate this, an SoA type will typically be used like 

```rust
struct CSR<T> {
    column_indices: Vec<usize>,
    offsets: Vec<usize>,
    data: Vec<T>,
}
```

where `offsets` indicates where each row begins and ends and `column_indices` and `data` are the
column indices and element values in each row stored contiguously. This also means that we must
write additional code to ensure that `offsets` is valid (bounds and monotonicity checks),
and when building algorithms, we have to conciously maintain these invariants.
To construct the equivalent BCSR matrix, one can replace `T` with a matrix type.

The pattern of maintaining offsets, for what is effectively two nested `Vec`s, is very common in
geometry processing pipelines. This mechanism is supplied by `Chunked`. Supplying a sparse set of
values to a set of indices (e.g. a range of indices from 0 to # columns) is provided by `Sparse`.
Thus a CSR matrix can be written simply as
`Chunked<Sparse<Vec<T>, Vec<usize>>, RangeTo<usize>, Vec<usize>>`.
Notice that all of the necessary `Vec`s needed to make this work are part of the type.
For convenience, the default type parameters allow one to simply write `Chunked<Sparse<Vec<T>>>`.
For a BCSR matrix, one would use `Chunked<Sparse<Chunked3<Chunked3<Vec<T>>>>>` (or alternatively
replace `T` with a matrix type as before, if the underlying data is never interpreted as a
contiguous array of numbers).


# Motivation

This library is a response to the frustration of rewriting the same bookkeeping code for managing
arrays of data in geometry processing applications.  In this setting all data is often known ahead
of time (as opposed to dealing with streams of data), and thus can be efficiently processed using
simple dynamically sized arrays (`Vec`s in Rust) of floating point or integer data (often in
structure of arrays format).  However, data often admits an intrinsically complex structure (e.g. in
an rigid or soft body animation code, a subset of 3D positions of vertices of polygonal meshes
belonging to different objects).  This induces additional cognitive load from marshalling indices
and ensuring data integrity when implementing algorithms around such data.  The goal of this library
is to reduce this cognitive load and allow users to focus on the data structures and algorithms
without any additional performance penalties.

Different applications have different performance characteristics. It is not a goal of this library
to prescribe a particular usage of the data types provided.  Instead, `flatk` aims to provide domain
agnostic abstractions for common code patterns found in data processing applications.  For instance,
processing data through a `Subset` may be faster than cloning when subsets are sufficiently large
but slower if they are small since a cloned subset can provide better data locality.

This is an exploratory project to determine if this kind of abstraction can simplify data processing
code and make it more reusable.


# Caveats and Limitations

Composability is not well supported in stable Rust (v1.41) due to some missing features like
[GAT](https://github.com/rust-lang/rust/issues/44265) (for more ergonomic indexing),
[specialization](https://github.com/rust-lang/rust/issues/31844) (for optimizations) and [const
generics](https://github.com/rust-lang/rust/issues/44580) (for better interoperability with arrays).
These limitations make the proposed abstractons difficult to implement, optimize and use in some
circumstances.  As such this library will likely remain experimental until these features are
stabilized.

To enable composability, many behaviours of contiguous collections (like `Vec` and `slice`) from the
standard library have been extracted into micro-traits (e.g. `SplitAt`), which need to be
implemented for each new composable type. This means that users who want to wrap custom `struct`s
inside types provided in this library will need to implement these traits manually.  For instance,
if a set of particles has position and velocity attributes, it may makes sense to store the
attributes in a struct as
```rust
struct Particles {
    pos: Vec<[f32; 3]>,
    vel: Vec<[f32; 3]>,
}
```
However, in order to compose `Particles` with say `Chunked`, one will need to implement `SplitAt`,
`Set` and `Dummy` traits to enable iteration.  It is possible however to automate this process
for certain structs using procedural macros.


# State

Currently this library is in the prototype stage (under heavy development) and is in no way
production ready.  It has been used to design the underlying data structures of an experimental
[tensor library](https://github.com/elrnv/tensr), which was used to develop an
[FEM](https://en.wikipedia.org/wiki/Finite_element_method) simulator for physically based animation
(TBA) capable of handling rigid bodies, cloth, soft bodies and frictional contacts between them.


# License

This repository is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.
