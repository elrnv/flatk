# `flatk`

**F**lat **l**ayout **a**bstraction **t**ool**k**it.

This library defines low level primitives for organizing flat data arrays into meaningful structures
without copying the data.

For example if we have an array of floats representing 3D positions, we may wish to interpret them
as triplets:

```
use flatk::Chunked3;

let pos_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];

let pos = Chunked3::from_flat(pos_data);

assert_eq!(pos[0], [0.0; 3]);
assert_eq!(pos[1], [1.0; 3]);
assert_eq!(pos[2], [0.0, 1.0, 0.0]);
```

Similarly we may have a non-uniform grouping of array elements, which may for represent a directed
graph:

```
use flatk::Chunked;

let neighbours = vec![1, 2, 0, 1, 0, 1, 2];

let neigh = Chunked::from_sizes(vec![1,2,1,3], neighbours);

assert_eq!(&neigh[0][..], &[1][..]);
assert_eq!(&neigh[1][..], &[2, 0][..]);
assert_eq!(&neigh[2][..], &[1][..]);
assert_eq!(&neigh[3][..], &[0, 1, 2][..]);
```

Here `neigh` defines the following graph:

```
0<--->1<--->2
^     ^     ^
 \    |    /
  \   |   /
   \  |  /
    \ | /
     \|/
      3
```

A sparse array can be specified as follows:

```
use flatk::Sparse;

let 

```
