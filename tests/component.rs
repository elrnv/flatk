use flatk::*;

#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
struct RigidState<T, R> {
    pub translation: T,
    pub rotation: R,
}

#[test]
fn different_topo() {
    let state = RigidState {
        translation: Chunked3::from_flat(vec![1, 2, 3, 4, 5, 6]),
        rotation: Chunked4::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8]),
    };

    let mut iter = state.into_iter();
    assert_eq!(
        iter.next(),
        Some(RigidState {
            translation: [1, 2, 3],
            rotation: [1, 2, 3, 4]
        })
    );
    assert_eq!(
        iter.next(),
        Some(RigidState {
            translation: [4, 5, 6],
            rotation: [5, 6, 7, 8]
        })
    );
    assert_eq!(iter.next(), None);
}

#[derive(Copy, Clone, Debug, PartialEq, Default, Component)]
struct Struct<A, B> {
    pub a: A,
    pub b: B,
}

#[test]
fn mixed_borrow() {
    let a = Chunked3::from_flat(vec![1, 2, 3, 4, 5, 6]);
    let mut b = Chunked3::from_flat(vec![3, 4, 5, 6, 7, 8]);

    // Prepare view to write a -> b
    let mixed_view = Struct {
        a: a.view(),
        b: b.view_mut(),
    };

    for Struct { a, b } in mixed_view.into_iter() {
        *b = *a;
    }
    assert_eq!(b, a);
}
