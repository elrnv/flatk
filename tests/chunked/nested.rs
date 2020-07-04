/// This suite of tests checks that chunked collections compose work as expected.
use flatk::*;

#[test]
fn chunked_unichunked_iter() {
    let v: Vec<usize> = (1..13).collect();
    let uni = Chunked3::from_flat(v);
    let chunked = Chunked::from_offsets(vec![0, 1, 4], uni);

    // Simple iteration
    let mut chunked_iter = chunked.iter();
    let uni = chunked_iter.next().unwrap();
    let mut uni_iter = uni.iter();
    assert_eq!(Some(&[1, 2, 3]), uni_iter.next());
    assert_eq!(None, uni_iter.next());
    let uni = chunked_iter.next().unwrap();
    let mut uni_iter = uni.iter();
    assert_eq!(Some(&[4, 5, 6]), uni_iter.next());
    assert_eq!(Some(&[7, 8, 9]), uni_iter.next());
    assert_eq!(Some(&[10, 11, 12]), uni_iter.next());
    assert_eq!(None, uni_iter.next());
}

#[test]
fn unichunked_unichunked_as_ref() {
    let v: Vec<usize> = (1..=12).collect();
    let uni0 = Chunked2::from_flat(v);
    let uni1 = Chunked3::from_flat(uni0);
    assert_eq!(
        &[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]],
        uni1.as_ref()
    );
}

#[test]
fn unichunked_chunked() {
    let v: Vec<usize> = (1..=12).collect();
    let chunked = Chunked::from_sizes(vec![2, 1, 2, 3, 3, 1], v);
    let uni = Chunked3::from_flat(chunked);
    let mut uni_iter = uni.iter();
    let chunked = uni_iter.next().unwrap();
    let mut chunked_iter = chunked.iter();
    assert_eq!(Some(&[1, 2][..]), chunked_iter.next());
    assert_eq!(Some(&[3][..]), chunked_iter.next());
    assert_eq!(Some(&[4, 5][..]), chunked_iter.next());
    assert_eq!(None, chunked_iter.next());
    let chunked = uni_iter.next().unwrap();
    let mut chunked_iter = chunked.iter();
    assert_eq!(Some(&[6, 7, 8][..]), chunked_iter.next());
    assert_eq!(Some(&[9, 10, 11][..]), chunked_iter.next());
    assert_eq!(Some(&[12][..]), chunked_iter.next());
    assert_eq!(None, chunked_iter.next());
}

#[test]
fn chunked_chars() {
    let v = vec!["World", "Coffee", "Cat", " ", "Hello", "Refrigerator", "!"];
    let bytes: Vec<Vec<u8>> = v
        .into_iter()
        .map(|word| word.to_string().into_bytes())
        .collect();
    let words = Chunked::<Vec<u8>>::from_nested_vec(bytes);
    let selection = Select::new(vec![4, 3, 0, 6, 3, 4, 6], words);
    let collapsed = selection.view().collapse();
    assert_eq!(
        "Hello World! Hello!",
        String::from_utf8(collapsed.data().clone())
            .unwrap()
            .as_str()
    );
}
