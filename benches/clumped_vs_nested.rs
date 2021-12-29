use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use flatk::{Clumped, ClumpedView, ViewMut};
/**
 * This module benchmarks the performance of iterating over clumped data vs. a nested vec.
 */
use rand::distributions::{Distribution, Standard};
use rand::prelude::*;
use rayon::prelude::*;

static SEED: [u8; 32] = [3; 32];

#[inline]
fn make_random_vec(n: usize) -> (Vec<f64>, Vec<usize>, Vec<usize>)
where
    Standard: Distribution<f64>,
{
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    let sizes = vec![2, 3, 4, 5, 6, 10, 15, 30, 60, 300];
    let counts = sizes.iter().map(|&i| n / sizes.len() / i).collect();
    ((0..n).map(move |_| rng.gen()).collect(), sizes, counts)
}

#[inline]
fn make_random_nested_vec(n: usize) -> Vec<Vec<f64>>
where
    Standard: Distribution<f64>,
{
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    let sizes = vec![2, 3, 4, 5, 6, 10, 15, 30, 60, 300];
    sizes
        .iter()
        .map(|_| (0..n / sizes.len()).map(|_| rng.gen()).collect())
        .collect()
}

#[inline]
fn compute(a: &mut [f64]) {
    for i in a {
        *i *= 3.6321;
    }
}

#[inline]
fn clumped(v: ClumpedView<&mut [f64]>) {
    for a in v.into_iter() {
        compute(a);
    }
}

#[inline]
fn clumped_par(v: ClumpedView<&mut [f64]>) {
    v.into_par_iter().for_each(|a| {
        compute(a);
    });
}

#[inline]
fn nested(v: &mut Vec<Vec<f64>>) {
    for (v, &n) in v
        .iter_mut()
        .zip([2, 3, 4, 5, 6, 10, 15, 30, 60, 300].iter())
    {
        for a in v.chunks_exact_mut(n) {
            compute(a);
        }
    }
}

#[inline]
fn nested_par(v: &mut Vec<Vec<f64>>) {
    v.par_iter_mut()
        .zip([2, 3, 4, 5, 6, 10, 15, 30, 60, 300].par_iter())
        .for_each(|(v, &n)| {
            v.par_chunks_exact_mut(n).for_each(|a| {
                compute(a);
            });
        });
}

fn clumped_vs_nested(c: &mut Criterion) {
    let mut group = c.benchmark_group("Clumped vs. Nested");

    for &buf_size in &[3000, 30_000, 90_000, 180_000, 300_000, 600_000, 900_000] {
        group.bench_function(BenchmarkId::new("Clumped", buf_size), |b| {
            let (v, sizes, counts) = make_random_vec(buf_size);
            let mut cl = Clumped::from_sizes_and_counts(sizes, counts, v);
            b.iter(|| {
                clumped(cl.view_mut());
            })
        });
        group.bench_function(BenchmarkId::new("Clumped Par", buf_size), |b| {
            let (v, sizes, counts) = make_random_vec(buf_size);
            let mut cl = Clumped::from_sizes_and_counts(sizes, counts, v);
            b.iter(|| {
                clumped_par(cl.view_mut());
            })
        });
        group.bench_function(BenchmarkId::new("Nested", buf_size), |b| {
            let mut v = make_random_nested_vec(buf_size);
            b.iter(|| {
                nested(&mut v);
            })
        });

        group.bench_function(BenchmarkId::new("Nested Par", buf_size), |b| {
            let mut v = make_random_nested_vec(buf_size);
            b.iter(|| {
                nested_par(&mut v);
            })
        });
    }

    group.finish();
}

criterion_group!(benches, clumped_vs_nested);
criterion_main!(benches);
