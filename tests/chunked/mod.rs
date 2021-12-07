mod chunked;
mod subset;
mod unichunked;

#[test]
fn clumped_view() {
    use flatk::{Clumped, View};
    let v = vec![1usize, 2, 3, 4, 5, 6, 7, 8, 9];

    // The following splits v into 3 pairs and a triplet.
    let s = Clumped::from_clumped_offsets(vec![0, 3, 4], vec![0, 6, 9], v);
    let view = s.view();
    let mut iter = view.iter();
    assert_eq!(&[1, 2][..], iter.next().unwrap());
    assert_eq!(&[3, 4][..], iter.next().unwrap());
    assert_eq!(&[5, 6][..], iter.next().unwrap());
    assert_eq!(&[7, 8, 9][..], iter.next().unwrap());
    assert_eq!(None, iter.next());
}
