use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
/**
 * This module benchmarks the performance of zipping unused arrays together.
 */
use rayon::prelude::*;

static SEED: [u8; 32] = [3; 32];

#[inline]
fn make_random_vec(n: usize) -> Vec<f64> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    (0..n).map(|_| rng.gen()).collect()
}

#[inline]
fn compute(x: f64, y: f64, z: f64) -> f64 {
    x * 3.6321 + 4.2314 * y + z
}

#[inline]
fn process(a: &mut f64, b: &mut f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) {
    *a += compute(*a, *b, *a);
    *b += compute(*a, *b, *b);
    *a += compute(*a, *b, c);
    *b += compute(*a, *b, d);
    *a += compute(*a, *b, e);
    *b += compute(*a, *b, f);
    *a += compute(*a, *b, g);
    *b += compute(*a, *b, h);
}

// Vanilla iteration over a single vector
fn benchmark(v: &mut Vec<f64>, v1: &mut Vec<f64>) {
    for (a, b) in v.iter_mut().zip(v1.iter_mut()) {
        process(a, b, *a, *b, *a, *b, *a, *b);
    }
}

fn unnecessary_zipping(
    v: &mut Vec<f64>,
    v1: &mut Vec<f64>,
    v2: &mut Vec<f64>,
    v3: &mut Vec<f64>,
    v4: &mut Vec<f64>,
    v5: &mut Vec<f64>,
    v6: &mut Vec<f64>,
    v7: &mut Vec<f64>,
) {
    for (((((((a, b), _c), _d), _e), _f), _g), _h) in v
        .iter_mut()
        .zip(v1.iter_mut())
        .zip(v2.iter_mut())
        .zip(v3.iter_mut())
        .zip(v4.iter_mut())
        .zip(v5.iter_mut())
        .zip(v6.iter_mut())
        .zip(v7.iter_mut())
    {
        process(a, b, *a, *b, *a, *b, *a, *b);
    }
}

fn unnecessary_zipping_with_map(
    v: &mut Vec<f64>,
    v1: &mut Vec<f64>,
    v2: &mut Vec<f64>,
    v3: &mut Vec<f64>,
    v4: &mut Vec<f64>,
    v5: &mut Vec<f64>,
    v6: &mut Vec<f64>,
    v7: &mut Vec<f64>,
) {
    for (a, b, _c, _d, _e, _f, _g, _h) in v
        .iter_mut()
        .zip(v1.iter_mut())
        .zip(v2.iter_mut())
        .zip(v3.iter_mut())
        .zip(v4.iter_mut())
        .zip(v5.iter_mut())
        .zip(v6.iter_mut())
        .zip(v7.iter_mut())
        .map(|(((((((a, b), c), d), e), f), g), h)| (a, b, c, d, e, f, g, h))
    {
        process(a, b, *a, *b, *a, *b, *a, *b);
    }
}

#[allow(dead_code)]
struct Group {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
}
fn unnecessary_zipping_with_struct(
    v: &mut Vec<f64>,
    v1: &mut Vec<f64>,
    v2: &mut Vec<f64>,
    v3: &mut Vec<f64>,
    v4: &mut Vec<f64>,
    v5: &mut Vec<f64>,
    v6: &mut Vec<f64>,
    v7: &mut Vec<f64>,
) {
    for (((((((a, b), c), d), e), f), g), h) in v
        .iter_mut()
        .zip(v1.iter_mut())
        .zip(v2.iter_mut())
        .zip(v3.iter_mut())
        .zip(v4.iter_mut())
        .zip(v5.iter_mut())
        .zip(v6.iter_mut())
        .zip(v7.iter_mut())
    {
        let group = Group {
            a: *a,
            b: *b,
            c: *c,
            d: *d,
            e: *e,
            f: *f,
            g: *g,
            h: *h,
        };
        process(a, b, group.a, group.b, group.a, group.b, group.a, group.b);
    }
}

fn zipping(
    v: &mut Vec<f64>,
    v1: &mut Vec<f64>,
    v2: &mut Vec<f64>,
    v3: &mut Vec<f64>,
    v4: &mut Vec<f64>,
    v5: &mut Vec<f64>,
    v6: &mut Vec<f64>,
    v7: &mut Vec<f64>,
) {
    for (((((((a, b), c), d), e), f), g), h) in v
        .iter_mut()
        .zip(v1.iter_mut())
        .zip(v2.iter_mut())
        .zip(v3.iter_mut())
        .zip(v4.iter_mut())
        .zip(v5.iter_mut())
        .zip(v6.iter_mut())
        .zip(v7.iter_mut())
    {
        process(a, b, *c, *d, *e, *f, *g, *h);
    }
}

#[cfg(feature = "rayon")]
fn zipping_with_rayon(
    v: &mut Vec<f64>,
    v1: &mut Vec<f64>,
    v2: &mut Vec<f64>,
    v3: &mut Vec<f64>,
    v4: &mut Vec<f64>,
    v5: &mut Vec<f64>,
    v6: &mut Vec<f64>,
    v7: &mut Vec<f64>,
) {
    v.par_iter_mut()
        .zip(v1.par_iter_mut())
        .zip(v2.par_iter_mut())
        .zip(v3.par_iter_mut())
        .zip(v4.par_iter_mut())
        .zip(v5.par_iter_mut())
        .zip(v6.par_iter_mut())
        .zip(v7.par_iter_mut())
        .for_each(|(((((((a, b), c), d), e), f), g), h)| {
            process(a, b, *c, *d, *e, *f, *g, *h);
        });
}

fn zip(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zip");

    for &buf_size in &[1000, 10_000, 100_000, 1_000_000] {
        group.bench_function(BenchmarkId::new("Benchmark", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            let mut v1 = make_random_vec(buf_size);
            b.iter(|| {
                benchmark(&mut v, &mut v1);
            })
        });

        group.bench_function(BenchmarkId::new("Unnecessary Zipping", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            let mut v1 = make_random_vec(buf_size);
            let mut v2 = make_random_vec(buf_size);
            let mut v3 = make_random_vec(buf_size);
            let mut v4 = make_random_vec(buf_size);
            let mut v5 = make_random_vec(buf_size);
            let mut v6 = make_random_vec(buf_size);
            let mut v7 = make_random_vec(buf_size);
            b.iter(|| {
                unnecessary_zipping(
                    &mut v, &mut v1, &mut v2, &mut v3, &mut v4, &mut v5, &mut v6, &mut v7,
                );
            })
        });

        group.bench_function(
            BenchmarkId::new("Unnecessary Zipping With Map", buf_size),
            |b| {
                let mut v = make_random_vec(buf_size);
                let mut v1 = make_random_vec(buf_size);
                let mut v2 = make_random_vec(buf_size);
                let mut v3 = make_random_vec(buf_size);
                let mut v4 = make_random_vec(buf_size);
                let mut v5 = make_random_vec(buf_size);
                let mut v6 = make_random_vec(buf_size);
                let mut v7 = make_random_vec(buf_size);
                b.iter(|| {
                    unnecessary_zipping_with_map(
                        &mut v, &mut v1, &mut v2, &mut v3, &mut v4, &mut v5, &mut v6, &mut v7,
                    );
                })
            },
        );

        group.bench_function(
            BenchmarkId::new("Unnecessary Zipping With Struct", buf_size),
            |b| {
                let mut v = make_random_vec(buf_size);
                let mut v1 = make_random_vec(buf_size);
                let mut v2 = make_random_vec(buf_size);
                let mut v3 = make_random_vec(buf_size);
                let mut v4 = make_random_vec(buf_size);
                let mut v5 = make_random_vec(buf_size);
                let mut v6 = make_random_vec(buf_size);
                let mut v7 = make_random_vec(buf_size);
                b.iter(|| {
                    unnecessary_zipping_with_struct(
                        &mut v, &mut v1, &mut v2, &mut v3, &mut v4, &mut v5, &mut v6, &mut v7,
                    );
                })
            },
        );

        group.bench_function(BenchmarkId::new("Zipping", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            let mut v1 = make_random_vec(buf_size);
            let mut v2 = make_random_vec(buf_size);
            let mut v3 = make_random_vec(buf_size);
            let mut v4 = make_random_vec(buf_size);
            let mut v5 = make_random_vec(buf_size);
            let mut v6 = make_random_vec(buf_size);
            let mut v7 = make_random_vec(buf_size);
            b.iter(|| {
                zipping(
                    &mut v, &mut v1, &mut v2, &mut v3, &mut v4, &mut v5, &mut v6, &mut v7,
                );
            })
        });

        #[cfg(feature = "rayon")]
        group.bench_function(BenchmarkId::new("Zipping With Rayon", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            let mut v1 = make_random_vec(buf_size);
            let mut v2 = make_random_vec(buf_size);
            let mut v3 = make_random_vec(buf_size);
            let mut v4 = make_random_vec(buf_size);
            let mut v5 = make_random_vec(buf_size);
            let mut v6 = make_random_vec(buf_size);
            let mut v7 = make_random_vec(buf_size);
            b.iter(|| {
                zipping_with_rayon(
                    &mut v, &mut v1, &mut v2, &mut v3, &mut v4, &mut v5, &mut v6, &mut v7,
                );
            })
        });
    }

    group.finish();
}

criterion_group!(benches, zip);
criterion_main!(benches);
