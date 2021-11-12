use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use flatk::*;
/**
 * This module benchmarks the performance of mutable indexing via the `Isolate` trait vs. mutable
 * iteration.
 */
use rand::distributions::{uniform::SampleUniform, Distribution, Standard, Uniform};
use rand::prelude::*;

static SEED: [u8; 32] = [3; 32];

#[inline]
fn make_random_vec_standard<T>(n: usize) -> Vec<T>
where
    Standard: Distribution<T>,
{
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    (0..n).map(move |_| rng.gen()).collect()
}

#[inline]
fn make_random_vec_uniform<T>(n: usize, lo: T, hi: T) -> Vec<T>
where
    T: SampleUniform,
    Uniform<T>: Distribution<T>,
{
    let between = Uniform::new(lo, hi);
    let rng: StdRng = SeedableRng::from_seed(SEED);
    between.sample_iter(rng).take(n).collect()
}

#[inline]
fn make_random_chunked<T: bytemuck::Pod>(buf_size: usize) -> Chunked<Chunked3<Vec<T>>>
where
    Standard: Distribution<T>,
{
    let v = make_random_vec_standard::<T>(buf_size);
    let mut o = make_random_vec_uniform::<usize>(buf_size / 100, 1, buf_size / 3);
    o.push(0);
    o.push(buf_size / 3);
    o.sort();
    Chunked::from_offsets(o, Chunked3::from_flat(v))
}

#[inline]
fn compute(x: f64, y: f64, z: f64) -> [f64; 3] {
    [x * 3.6321, y * 3.6321, z * 3.6321]
}

// Vanilla mutable iteration over a single vector
#[inline]
fn iter_mut(mut v: ChunkedView<Chunked3<&mut [f64]>>) {
    for mut chunk in v.iter_mut() {
        for a in chunk.iter_mut() {
            let res = compute(a[0], a[1], a[2]);
            a[0] = res[0];
            a[1] = res[1];
            a[2] = res[2];
        }
    }
}

#[inline]
fn isolate(mut v: ChunkedView<Chunked3<&mut [f64]>>) {
    for i in 0..v.len() {
        for j in 0..v.view().at(i).len() {
            let a = v.view_mut().isolate(i).isolate(j);
            let res = compute(a[0], a[1], a[2]);
            a[0] = res[0];
            a[1] = res[1];
            a[2] = res[2];
        }
    }
}

#[inline]
fn isolate_unchecked(mut v: ChunkedView<Chunked3<&mut [f64]>>) {
    for i in 0..v.len() {
        for j in 0..unsafe { v.view().isolate_unchecked(i).len() } {
            let a = unsafe { v.view_mut().isolate_unchecked(i).isolate_unchecked(j) };
            let res = compute(a[0], a[1], a[2]);
            a[0] = res[0];
            a[1] = res[1];
            a[2] = res[2];
        }
    }
}

fn chunks_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("Isolate vs Iter Mut");

    // Make sure all of the functions being benchmarked are doing the same thing.
    let mut a = make_random_chunked::<f64>(900_000);
    let mut b = make_random_chunked::<f64>(900_000);
    let mut c = make_random_chunked::<f64>(900_000);
    iter_mut(a.view_mut());
    isolate(b.view_mut());
    isolate_unchecked(c.view_mut());
    assert_eq!(&a, &b);
    assert_eq!(&b, &c);

    for &buf_size in &[3000, 30_000, 90_000, 180_000, 300_000, 600_000, 900_000] {
        group.bench_function(BenchmarkId::new("Iter Mut", buf_size), |b| {
            let mut c = make_random_chunked::<f64>(buf_size);

            b.iter(|| iter_mut(c.view_mut()))
        });

        group.bench_function(BenchmarkId::new("Isolate", buf_size), |b| {
            let mut c = make_random_chunked::<f64>(buf_size);

            b.iter(|| isolate(c.view_mut()))
        });

        group.bench_function(BenchmarkId::new("Isolate Unchecked", buf_size), |b| {
            let mut c = make_random_chunked::<f64>(buf_size);

            b.iter(|| isolate_unchecked(c.view_mut()))
        });
    }

    group.finish();
}

criterion_group!(benches, chunks_iter);
criterion_main!(benches);
