use flatk::*;

#[test]
fn chunked_unichunked_push() {
    let mut chunked = Chunked::<Chunked2<Vec<usize>>>::new();

    for i in 0..4 {
        let data: Vec<usize> = (i..i + 4).collect();
        let uni = Chunked2::from_flat(data);
        chunked.push(uni);
    }

    assert_eq!(
        chunked.into_storage(),
        vec![0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6]
    );
}
