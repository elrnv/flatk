#[cfg(feature = "sparse")]
fn main() {
    use flatk::{Get, Sparse, View};

    let values = vec![1.0, 2.0, 3.0, 4.0];
    let sparse_vector = Sparse::from_dim(vec![0, 5, 10, 100], 1000, values);
    let sparse_vector_view = sparse_vector.view();

    assert_eq!(sparse_vector_view.at(0), (0, &1.0));
    assert_eq!(sparse_vector_view.at(1), (5, &2.0));
    assert_eq!(sparse_vector_view.at(2), (10, &3.0));
    assert_eq!(sparse_vector_view.at(3), (100, &4.0));
    assert_eq!(sparse_vector_view.selection.target, ..1000);
}

#[cfg(not(feature = "sparse"))]
fn main() {
    eprintln!("This example requires the \"sparse\" feature");
}
