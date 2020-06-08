use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
/**
 * This module benchmarks the performance of different storage schemes
 */
use rand::distributions::{Distribution, Standard};
use rand::prelude::*;

static SEED: [u8; 32] = [3; 32];

#[inline]
fn operation(a: [f64; 9], b: [f64; 9]) -> [f64; 9] {
    let mut c = [0.0; 9];
    for i in 0..9 {
        c[i] = a[i] + b[i];
    }
    c
}

#[inline]
fn make_random_vec<T: Copy>(n: usize) -> Vec<[T; 9]>
where
    Standard: Distribution<T>,
{
    let mut v = Vec::new();
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    for _ in 0..n {
        v.push([
            rng.gen(),
            rng.gen(),
            rng.gen(),
            rng.gen(),
            rng.gen(),
            rng.gen(),
            rng.gen(),
            rng.gen(),
            rng.gen(),
        ]);
    }

    v
}

fn array_of_structs(i: &Vec<([f64; 9], [f64; 9])>) -> [f64; 9] {
    let mut sum = [0.0; 9];
    for &(a, b) in i.iter() {
        let res = operation(a, b);
        for i in 0..9 {
            sum[i] += res[i];
        }
    }
    sum
}

fn struct_of_arrays(vec_a: &Vec<[f64; 9]>, vec_b: &Vec<[f64; 9]>) -> [f64; 9] {
    let mut sum = [0.0; 9];
    for (&a, &b) in vec_a.iter().zip(vec_b.iter()) {
        let res = operation(a, b);
        for i in 0..9 {
            sum[i] += res[i];
        }
    }
    sum
}

fn storage(c: &mut Criterion) {
    let mut group = c.benchmark_group("AoS vs SoA");

    for &buf_size in &[1000, 10_000, 100_000, 1_000_000] {
        group.bench_function(BenchmarkId::new("Array of Structs", buf_size), |b| {
            let v = make_random_vec(buf_size)
                .into_iter()
                .zip(make_random_vec(buf_size).into_iter())
                .collect();
            b.iter(|| array_of_structs(&v))
        });

        group.bench_function(BenchmarkId::new("Struct of Arrays", buf_size), |b| {
            let vec_a = make_random_vec(buf_size);
            let vec_b = make_random_vec(buf_size);
            b.iter(|| struct_of_arrays(&vec_a, &vec_b))
        });
    }

    group.finish();
}

criterion_group!(benches, storage);
criterion_main!(benches);
