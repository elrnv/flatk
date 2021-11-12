use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
/**
 * This module benchmarks the performance of iterating over chunks of data vs. reinterpreting an
 * array as chunked data.
 */
use rand::distributions::{Distribution, Standard};
use rand::prelude::*;
use rayon::prelude::*;

static SEED: [u8; 32] = [3; 32];

#[inline]
fn make_random_vec(n: usize) -> Vec<f64>
where
    Standard: Distribution<f64>,
{
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    (0..n).map(move |_| rng.gen()).collect()
}

#[inline]
fn compute(x: f64, y: f64, z: f64) -> [f64; 3] {
    [x * 3.6321, y * 3.6321, z * 3.6321]
}

// Vanilla iteration over a single vector
#[inline]
fn chunks(v: &mut Vec<f64>) {
    for a in v.chunks_mut(3) {
        let res = compute(a[0], a[1], a[2]);
        a[0] = res[0];
        a[1] = res[1];
        a[2] = res[2];
    }
}

#[inline]
fn chunks_exact(v: &mut Vec<f64>) {
    for a in v.chunks_exact_mut(3) {
        let res = compute(a[0], a[1], a[2]);
        a[0] = res[0];
        a[1] = res[1];
        a[2] = res[2];
    }
}

#[inline]
fn chunks_exact_par(v: &mut Vec<f64>) {
    v.par_chunks_exact_mut(3).for_each(|a| {
        let res = compute(a[0], a[1], a[2]);
        a[0] = res[0];
        a[1] = res[1];
        a[2] = res[2];
    });
}

#[inline]
fn chunked3(v: &mut Vec<f64>) {
    use flatk::Chunked3;
    for a in Chunked3::from_flat(v.as_mut_slice()).into_iter() {
        *a = compute(a[0], a[1], a[2]);
    }
}

#[inline]
fn chunked_n(v: &mut Vec<f64>) {
    use flatk::ChunkedN;
    for a in ChunkedN::from_flat_with_stride(3, v.as_mut_slice()).into_iter() {
        a.copy_from_slice(&compute(a[0], a[1], a[2]));
    }
}

#[inline]
fn chunked_n_par(v: &mut Vec<f64>) {
    use flatk::ChunkedN;
    ChunkedN::from_flat_with_stride(3, v.as_mut_slice())
        .into_par_iter()
        .for_each(|a| {
            a.copy_from_slice(&compute(a[0], a[1], a[2]));
        });
}

#[inline]
fn chunked(v: &mut Vec<f64>, offsets: &[usize]) {
    use flatk::Chunked;
    for a in Chunked::from_offsets(offsets, v.as_mut_slice()).into_iter() {
        a.copy_from_slice(&compute(a[0], a[1], a[2]));
    }
}

#[inline]
fn chunked_par(v: &mut Vec<f64>, offsets: &[usize]) {
    use flatk::Chunked;
    Chunked::into_par_iter(Chunked::from_offsets(offsets, v.as_mut_slice())).for_each(|a| {
        a.copy_from_slice(&compute(a[0], a[1], a[2]));
    });
}

#[inline]
fn clumped(v: &mut Vec<f64>) {
    use flatk::Clumped;
    for a in
        Clumped::from_clumped_offsets(&[0, v.len() / 3][..], &[0, v.len()][..], v.as_mut_slice())
            .into_iter()
    {
        a.copy_from_slice(&compute(a[0], a[1], a[2]));
    }
}

#[inline]
fn clumped_par(v: &mut Vec<f64>) {
    use flatk::Clumped;
    Clumped::from_clumped_offsets(&[0, v.len() / 3][..], &[0, v.len()][..], v.as_mut_slice())
        .into_par_iter()
        .for_each(|a| {
            a.copy_from_slice(&compute(a[0], a[1], a[2]));
        });
}

#[inline]
fn reinterpret(v: &mut Vec<f64>) {
    let s: &mut [[f64; 3]] = bytemuck::cast_slice_mut(v.as_mut_slice());
    for a in s.iter_mut() {
        *a = compute(a[0], a[1], a[2]);
    }
}

fn chunks_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("Chunks Iter");

    for &buf_size in &[3000, 30_000, 90_000, 180_000, 300_000, 600_000, 900_000] {
        group.bench_function(BenchmarkId::new("Chunks", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            b.iter(|| {
                chunks(&mut v);
            })
        });

        group.bench_function(BenchmarkId::new("Chunks Exact", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            b.iter(|| {
                chunks_exact(&mut v);
            })
        });

        group.bench_function(BenchmarkId::new("Chunks Exact Par", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            b.iter(|| {
                chunks_exact_par(&mut v);
            })
        });

        group.bench_function(BenchmarkId::new("Reinterpret", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            b.iter(|| {
                reinterpret(&mut v);
            })
        });

        group.bench_function(BenchmarkId::new("Chunked3", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            b.iter(|| {
                chunked3(&mut v);
            })
        });

        group.bench_function(BenchmarkId::new("ChunkedN", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            b.iter(|| {
                chunked_n(&mut v);
            })
        });
        group.bench_function(BenchmarkId::new("ChunkedN Par", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            b.iter(|| {
                chunked_n_par(&mut v);
            })
        });
        group.bench_function(BenchmarkId::new("Clumped", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            b.iter(|| {
                clumped(&mut v);
            })
        });
        group.bench_function(BenchmarkId::new("Clumped Par", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            b.iter(|| {
                clumped_par(&mut v);
            })
        });
        group.bench_function(BenchmarkId::new("Chunked", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            let offsets: Vec<_> = (0..=v.len() / 3).map(|i| 3 * i).collect();
            b.iter(|| {
                chunked(&mut v, offsets.as_slice());
            })
        });

        group.bench_function(BenchmarkId::new("Chunked Par", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            let offsets: Vec<_> = (0..=v.len() / 3).map(|i| 3 * i).collect();
            b.iter(|| {
                chunked_par(&mut v, offsets.as_slice());
            })
        });
    }

    group.finish();
}

criterion_group!(benches, chunks_iter);
criterion_main!(benches);
