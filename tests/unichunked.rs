use flatk::*;

mod unichunked;

#[test]
fn vec_iter() {
    let nested = vec![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let v = Chunked3::from_array_vec(nested.clone());

    for (a, b) in nested.into_iter().zip(v.into_iter()) {
        assert_eq!(a, b);
    }
}

#[test]
fn array_iter() {
    let a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let c = Chunked3::from_flat(a);
    let mut iter = c.iter();
    assert_eq!(&[1, 2, 3], iter.next().unwrap());
    assert_eq!(&[4, 5, 6], iter.next().unwrap());
    assert_eq!(&[7, 8, 9], iter.next().unwrap());
    assert_eq!(None, iter.next());
}

#[test]
fn array_into_arrays() {
    let a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let c = Chunked3::from_flat(a);
    assert_eq!(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]], c.view().into_arrays());
}
