# `flat`

**F**lat **l**ayout **a**bstraction **t**ools. This library defines low level primitives for
organizing flat data arrays into meaningful structures without copying the data.

For example if we have an array of floats representing 3D positions, we may wish to interpret them
as triplets:

```
use flat::Chunked3;

let pos_data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0. 1.0, 0,0];

let pos = Chunked3::from_flat(pos_data);

for p in pos {
    println!("{}", p);
}
```


