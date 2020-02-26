# `flatk`

**F**lat **l**ayout **a**bstraction **t**ool**k**it.

This library defines low level primitives for organizing flat ordered data collections (like `Vec`s
and `slice`s) into meaningful structures without cloning the data.

This library provides a few core composable types intended for building more complex data structures
out of existing data:

- `UniChunked`:  Subdivides a collection in to a number of uniformly sized (at compile time or
  run-time) groups.
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

- `Sparse`: TBA

- `Subset`: TBA


# License

This repository is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.
