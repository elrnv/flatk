use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;

// This simple benchmark checks if its faster writing to contiguous blocks than writing into
// random parts of an array.

static SEED: [u8; 32] = [3; 32];

#[inline]
fn make_random_vec(n: usize) -> Vec<f64> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    (0..n).map(|_| rng.gen()).collect()
}

#[inline]
fn compute(x: [f64; 3]) -> [f64; 3] {
    [
        (x[0] * 3.6321 + x[1] * 4.2314 - x[2]).sqrt(),
        (x[1] * 3.6321 + x[0] * 4.2314 - x[2]).sqrt(),
        (x[2] * 3.6321 + x[1] * 4.2314 - x[0]).sqrt(),
    ]
}

fn write(c: &mut Criterion) {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    let mut group = c.benchmark_group("Write");

    for &i in &[3000, 30_000, 300_000, 3_000_000] {
        group.bench_function(BenchmarkId::new("Write then distribute", i), |b| {
            let mut out = vec![0.0; i];
            let mut intermediate = make_random_vec(i);
            let input = make_random_vec(i);
            let mut indices: Vec<_> = (0..i / 3).collect();
            indices.shuffle(&mut rng);
            let mut map_back = vec![0; i / 3];
            for (i, &idx) in indices.iter().enumerate() {
                map_back[idx] = i;
            }
            b.iter(|| {
                for (&i, tmp) in indices.iter().zip(intermediate.chunks_exact_mut(3)) {
                    let input = [input[3 * i], input[3 * i + 1], input[3 * i + 2]];
                    let t = compute(input);
                    tmp[0] = t[0];
                    tmp[1] = t[1];
                    tmp[2] = t[2];
                }
                for (&i, out) in map_back.iter().zip(out.chunks_exact_mut(3)) {
                    let t = [
                        intermediate[3 * i],
                        intermediate[3 * i + 1],
                        intermediate[3 * i + 2],
                    ];
                    out[0] = t[0];
                    out[1] = t[1];
                    out[2] = t[2];
                }
            })
        });

        group.bench_function(BenchmarkId::new("Write direct", i), |b| {
            let mut out = vec![0.0; i];
            let input = make_random_vec(i);
            let mut indices: Vec<_> = (0..i / 3).collect();
            indices.shuffle(&mut rng);
            b.iter(|| {
                for &i in indices.iter() {
                    let input = [input[3 * i], input[3 * i + 1], input[3 * i + 2]];
                    let t = compute(input);
                    out[3 * i] = t[0];
                    out[3 * i + 1] = t[1];
                    out[3 * i + 2] = t[2];
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, write);
criterion_main!(benches);
