use flatk::*;

#[test]
fn iter() {
    let v: Vec<usize> = (1..=12).collect();
    let uni0 = Chunked2::from_flat(v);
    let uni1 = Chunked3::from_flat(uni0);
    assert_eq!(&[1, 2], uni1.view().at(0).at(0));
    let mut uni1_iter = uni1.iter();
    let uni0 = uni1_iter.next().unwrap();
    let mut uni0_iter = uni0.iter();
    assert_eq!(Some(&[1, 2]), uni0_iter.next());
    assert_eq!(Some(&[3, 4]), uni0_iter.next());
    assert_eq!(Some(&[5, 6]), uni0_iter.next());
    assert_eq!(None, uni0_iter.next());
    let uni0 = uni1_iter.next().unwrap();
    let mut uni0_iter = uni0.iter();
    assert_eq!(Some(&[7, 8]), uni0_iter.next());
    assert_eq!(Some(&[9, 10]), uni0_iter.next());
    assert_eq!(Some(&[11, 12]), uni0_iter.next());
    assert_eq!(None, uni0_iter.next());
}
