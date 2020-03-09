use flatk::*;

#[test]
fn chunked_chunked_chunked_mut() {
    let v: Vec<usize> = (1..=11).collect();
    let chunked = Chunked::from_offsets(vec![1, 4, 6, 8, 10, 11, 12, 12], v);
    let chunked = Chunked::from_offsets(vec![1, 3, 5, 6, 6, 8], chunked);
    let mut chunked = Chunked::from_offsets(vec![1, 3, 6], chunked);

    chunked.view_mut().isolate(0).isolate(1).isolate(1)[1] = 100;
    assert_eq!(chunked.view().at(0).at(1).at(1)[1], 100);

    // Simple iteration
    let mut iter2 = chunked.iter();

    let chunked1 = iter2.next().unwrap();
    let mut iter1 = chunked1.iter();

    let chunked0 = iter1.next().unwrap();
    let mut iter0 = chunked0.iter();
    assert_eq!(Some(&[1, 2, 3][..]), iter0.next());
    assert_eq!(Some(&[4, 5][..]), iter0.next());
    assert_eq!(None, iter0.next());
    let chunked0 = iter1.next().unwrap();
    let mut iter0 = chunked0.iter();
    assert_eq!(Some(&[6, 7][..]), iter0.next());
    assert_eq!(Some(&[8, 100][..]), iter0.next());
    assert_eq!(None, iter0.next());
    assert_eq!(None, iter1.next());

    let chunked1 = iter2.next().unwrap();
    let mut iter1 = chunked1.iter();

    let chunked0 = iter1.next().unwrap();
    let mut iter0 = chunked0.iter();
    assert_eq!(Some(&[10][..]), iter0.next());
    assert_eq!(None, iter0.next());
    let chunked0 = iter1.next().unwrap();
    let mut iter0 = chunked0.iter();
    assert_eq!(None, iter0.next());
    let chunked0 = iter1.next().unwrap();
    let mut iter0 = chunked0.iter();
    assert_eq!(Some(&[11][..]), iter0.next());
    assert_eq!(Some(&[][..]), iter0.next());
    assert_eq!(None, iter0.next());
    assert_eq!(None, iter1.next());
    assert_eq!(None, iter2.next());
}
