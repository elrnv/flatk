#[cfg(feature = "derive")]
use flatk::{Chunked, Chunked3, Component, Get, View, ViewMut};

#[cfg(feature = "derive")]
#[derive(Copy, Clone, Debug, PartialEq, Component)]
struct Object<X, V> {
    // Unused parameter, that is simply cloned through to views and items.
    // Thus it is not a good idea to put large types here.
    id: usize,
    pos: X,
    vel: V,
}

#[cfg(feature = "derive")]
type State = Object<Vec<f32>, Vec<f32>>;

#[cfg(feature = "derive")]
#[derive(Debug)]
struct Data {
    prev: Chunked<Chunked3<State>>,
    cur: Chunked<Chunked3<State>>,
}

#[cfg(feature = "derive")]
fn main() {
    let e = State {
        id: 0,
        pos: vec![1.0; 12],
        vel: vec![7.0; 12],
    };

    let chunked = Chunked::from_sizes(vec![1, 3], Chunked3::from_flat(e));
    let mut data = Data {
        prev: chunked.clone(),
        cur: chunked.clone(),
    };

    // We can access individual element in the state through indexing using the `View` and `Get`
    // traits.
    assert_eq!(
        data.prev.view().at(0).at(0),
        Object {
            id: 0,
            pos: &[1.0, 1.0, 1.0],
            vel: &[7.0, 7.0, 7.0]
        }
    );

    // A typical animation loop can be written as follows.

    // Iterate for 20 frames
    for _ in 0..20 {
        let dt = 0.1;
        for (mut prev, mut cur) in data
            .prev
            .view_mut()
            .iter_mut()
            .zip(data.cur.view_mut().iter_mut())
        {
            for (prev, cur) in prev.iter_mut().zip(cur.iter_mut()) {
                for (prev_x, cur) in prev.pos.iter_mut().zip(cur.into_iter()) {
                    *cur.pos += *cur.vel * dt;
                    *prev_x = *cur.pos;
                }
            }
        }
    }

    dbg!(data);
}

#[cfg(not(feature = "derive"))]
fn main() {
    eprintln!("The `component` example requires the \"derive\" feature flat");
}
