use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use flatk::*;
/**
 * This module benchmarks the performance of gpu compute routines against their cpu counterparts.
 */
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use rayon::prelude::*;

static SEED: [u8; 32] = [3; 32];

#[inline]
fn make_random_vec(n: usize) -> Vec<u32> {
    let between = Uniform::from(1..100);
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    (0..n).map(move |_| between.sample(&mut rng)).collect()
}

#[inline]
fn collatz(mut n: u32) -> u32 {
    let mut i: u32 = 0;
    while n > 1 {
        if n % 2 == 0 {
            n = n / 2;
        } else {
            n = (3 * n) + 1;
        }
        i += 1;
    }
    i
}

#[inline]
fn seq_collatz(v: &[u32]) -> u32 {
    let mut out = v.to_vec();
    for n in out.iter_mut() {
        *n = collatz(*n);
    }
    out[0]
}

#[inline]
fn par_collatz(v: &[u32]) -> u32 {
    let mut out = v.to_vec();
    out.par_iter_mut().for_each(|n| {
        *n = collatz(*n);
    });
    out[0]
}

#[inline]
fn gpu_collatz(v: &gpu::CompiledProgram<gpu::Slice<u32>, gpu::OutputBuffer<u32>>) -> u32 {
    let out: Vec<u32> = v.run().collect();
    out[0]
}

fn gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU");

    let collatz: &str = stringify! {
        uint program(uint n) {
            uint i = 0;
            while(n > 1) {
                if (mod(n, 2) == 0) {
                    n = n / 2;
                } else {
                    n = (3 * n) + 1;
                }
                i += 1;
            }
            return i;
        }
    };

    for &buf_size in &[100_000, 500_000, 1_000_000] {
        let v = make_random_vec(buf_size);
        group.bench_with_input(
            BenchmarkId::new("Sequential Collatz", buf_size),
            v.as_slice(),
            |b, v| b.iter(|| seq_collatz(v)),
        );

        group.bench_with_input(
            BenchmarkId::new("Parallel Collatz", buf_size),
            v.as_slice(),
            |b, v| b.iter(|| par_collatz(v)),
        );

        let v_gpu = v.as_slice().into_gpu().map(collatz).compile();

        group.bench_with_input(BenchmarkId::new("GPU Collatz", buf_size), &v_gpu, |b, v| {
            b.iter(|| gpu_collatz(v))
        });

        group.bench_with_input(
            BenchmarkId::new("GPU Collatz with Compile", buf_size),
            v.as_slice(),
            |b, v| {
                b.iter(|| {
                    let v_gpu = v.into_gpu().map(collatz).compile();
                    gpu_collatz(&v_gpu)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, gpu);
criterion_main!(benches);
