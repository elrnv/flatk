//! This example demonstrates how to compose two entities together to form a single entity, with a
//! unique structure shared by all the fields.
//!
//! This improves on the `entity` example by sharing the structure between `cur` and `prev` fields.

#[cfg(feature = "derive")]
use flatk::{Chunked, Chunked3, Entity, Get, MapStorage, View, ViewMut};

#[cfg(feature = "derive")]
#[derive(Copy, Clone, Debug, PartialEq, Entity)]
struct State<X, V> {
    pos: X,
    vel: V,
}

#[cfg(feature = "derive")]
#[derive(Copy, Clone, Debug, PartialEq, Entity)]
struct Object<X, V> {
    id: usize,
    #[entity]
    prev: State<X, V>,
    #[entity]
    cur: State<X, V>,
}

#[cfg(feature = "derive")]
fn main() {
    let obj = Object {
        id: 0,
        // Previous state
        prev: State {
            pos: vec![0.0; 9],
            vel: vec![1.0; 9],
        },
        // Current state
        cur: State {
            pos: vec![0.0; 9],
            vel: vec![1.0; 9],
        },
    };

    // Select a structure for the `obj` data.
    let mut state = Chunked::from_sizes(vec![1, 2], Chunked3::from_flat(obj));

    // Push another chunk onto the state.
    state.push_iter(std::iter::once(Object {
        id: 0, /* ignored */
        prev: State {
            pos: [0.0; 3],
            vel: [1.0; 3],
        },
        cur: State {
            pos: [0.0; 3],
            vel: [1.0; 3],
        },
    }));

    // We can access individual element in the state through indexing using the `View` and `Get`
    // traits.
    assert_eq!(
        state.view().at(0).at(0),
        Object {
            id: 0,
            prev: State {
                pos: &[0.0; 3],
                vel: &[1.0; 3]
            },
            cur: State {
                pos: &[0.0; 3],
                vel: &[1.0; 3]
            }
        }
    );

    // A typical animation loop can be written as follows.

    // Iterate for 20 frames
    for _ in 0..20 {
        let dt = 0.1;
        for mut element in state.view_mut().iter_mut() {
            for Object { prev, cur, .. } in element.iter_mut() {
                for (prev_x, (cur_x, cur_v)) in prev
                    .pos
                    .iter_mut()
                    .zip(cur.pos.iter_mut().zip(cur.vel.iter()))
                {
                    *cur_x += cur_v * dt;
                    *prev_x = *cur_x;
                }
            }
        }
    }

    // To isolate a particular field in the hierarchy, one can use map_storage, instead of having
    // to destructure the whole hierarchy.

    let prev_pos_only_view = state.view().map_storage(|s| s.prev.pos);

    for chunk in prev_pos_only_view.iter() {
        for pos in chunk.iter() {
            for component in pos.iter() {
                assert!((component - 2.0_f64).abs() < 1e-5);
            }
        }
    }

    dbg!(state);
}

#[cfg(not(feature = "derive"))]
fn main() {
    eprintln!("The `composite_entity` example requires the \"derive\" feature flat");
}
