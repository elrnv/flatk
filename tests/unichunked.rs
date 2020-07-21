mod unichunked;

use flatk::*;

#[test]
fn unichunked_vec() {
    let nested = vec![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let v = Chunked3::from_array_vec(nested.clone());

    for (a, b) in nested.into_iter().zip(v.into_iter()) {
        assert_eq!(a, b);
    }
}
