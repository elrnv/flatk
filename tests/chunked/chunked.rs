use flatk::*;

mod chunked;

#[test]
fn iter_mut() {
    let v: Vec<usize> = (1..=10).collect();
    let chunked = Chunked::from_offsets(vec![0, 3, 5, 8, 10], v);
    let mut chunked = Chunked::from_offsets(vec![0, 1, 4], chunked);

    chunked.view_mut().isolate(1).isolate(1)[1] = 100;
    assert_eq!(chunked.view().at(1).at(1)[1], 100);

    // Simple iteration
    let mut iter1 = chunked.iter();
    let chunked0 = iter1.next().unwrap();
    let mut iter0 = chunked0.iter();
    assert_eq!(Some(&[1, 2, 3][..]), iter0.next());
    assert_eq!(None, iter0.next());
    let chunked0 = iter1.next().unwrap();
    let mut iter0 = chunked0.iter();
    assert_eq!(Some(&[4, 5][..]), iter0.next());
    assert_eq!(Some(&[6, 100, 8][..]), iter0.next());
    assert_eq!(Some(&[9, 10][..]), iter0.next());
    assert_eq!(None, iter0.next());

}
