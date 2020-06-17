use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use flatk::{BinarySearch, Chunked2, Chunked3, ClumpedOffsets, SplitOffsetsAt};
use rand::distributions::{Distribution, Uniform};
/**
 * This module benchmarks the performance of different storage schemes for indices.
 *
 * This is used to determine the optimal data layout for `ClumpedOffsets`.
 */

#[inline]
fn make_random_usize_vec(n: usize) -> Vec<usize> {
    let between = Uniform::from(0..100_000_000);
    let mut v = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..n {
        v.push(between.sample(&mut rng));
    }
    v
}

#[inline]
fn make_random_u8_vec(n: usize) -> Vec<u8> {
    let between = Uniform::from(0..255);
    let mut v = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..n {
        v.push(between.sample(&mut rng));
    }
    v
}

fn clumped_offsets(
    offsets: ClumpedOffsets<&[usize]>,
) -> Option<(ClumpedOffsets<&[usize]>, ClumpedOffsets<&[usize]>, usize)> {
    let mid = 10000;
    if let Ok(mid_idx) = offsets.chunk_offsets.binary_search(&mid) {
        let (los, ros, off) = offsets.offsets.split_offsets_with_intersection_at(mid_idx);
        let (lcos, rcos) = offsets.chunk_offsets.split_offsets_at(mid_idx);
        Some((
            ClumpedOffsets {
                chunk_offsets: lcos,
                offsets: los,
            },
            ClumpedOffsets {
                chunk_offsets: rcos,
                offsets: ros,
            },
            off,
        ))
    } else {
        None
    }
}

fn three_vecs<'o, 'c, 's, S>(
    offsets: &'o [usize],
    clumped_offsets: &'c [usize],
    strides: &'s [S],
) -> Option<(
    (&'o [usize], &'c [usize], &'s [S]),
    (&'o [usize], &'c [usize], &'s [S]),
    usize,
)> {
    let mid = 10000;
    if let Ok(pos) = clumped_offsets.binary_search(&mid) {
        let o = offsets[pos];
        let lo = &offsets[..=pos];
        let ro = &offsets[pos..];
        let lc = &clumped_offsets[..=pos];
        let rc = &clumped_offsets[pos..];
        let ls = &strides[..=pos];
        let rs = &strides[pos..];
        Some(((lo, lc, ls), (ro, rc, rs), o))
    } else {
        None
    }
}

fn clumped_offsets_separate<'c, 'd>(
    clumped_offsets: &'c [usize],
    data: &'d [usize],
) -> Option<(
    (&'c [usize], &'d [usize]),
    (&'c [usize], &'d [usize]),
    usize,
)> {
    let mid = 10000;
    if let Ok(pos) = clumped_offsets.binary_search(&mid) {
        let o = data[pos * 2];
        let lc = &clumped_offsets[..=pos];
        let rc = &clumped_offsets[pos..];
        let ld = &data[..=2 * pos];
        let rd = &data[2 * pos..];
        Some(((lc, ld), (rc, rd), o))
    } else {
        None
    }
}

fn stride_and_offsets<'o, 's, S>(
    offsets: &'o [usize],
    strides: &'s [S],
) -> Option<((&'o [usize], &'s [S]), (&'o [usize], &'s [S]), usize)> {
    let mid = 10000;
    let paired = Chunked2::from_flat(offsets).into_arrays();
    if let Ok(pos) = paired.binary_search_by_key(&mid, |&[offset, _]| offset) {
        let o = offsets[pos * 2];
        let lo = &offsets[..=pos * 2];
        let ro = &offsets[pos * 2..];
        let ls = &strides[..=pos];
        let rs = &strides[pos..];
        Some(((lo, ls), (ro, rs), o))
    } else {
        None
    }
}

fn all_in_one(offsets: &[usize]) -> Option<(&[usize], &[usize], usize)> {
    let mid = 10000;
    let triples = Chunked3::from_flat(offsets).into_arrays();
    if let Ok(pos) = triples.binary_search_by_key(&mid, |&[offset, _, _]| offset) {
        let o = offsets[pos * 3];
        let lo = &offsets[..=pos * 3];
        let ro = &offsets[pos * 3..];
        Some((lo, ro, o))
    } else {
        None
    }
}

fn storage_ints(c: &mut Criterion) {
    let mut group = c.benchmark_group("Clumped Offset Storage");

    for &buf_size in &[1000, 10_000, 100_000, 1_000_000] {
        group.bench_function(BenchmarkId::new("ClumpedOffsets", buf_size), |b| {
            let mut o = make_random_usize_vec(buf_size + 1);
            let mut c = make_random_usize_vec(buf_size + 1);
            o.sort();
            c.sort();
            let off = ClumpedOffsets::new(c.as_slice(), o.as_slice());
            b.iter(|| clumped_offsets(off))
        });
        group.bench_function(BenchmarkId::new("Three Vecs", buf_size), |b| {
            let o = make_random_usize_vec(buf_size + 1);
            let mut c = make_random_usize_vec(buf_size + 1);
            let s = make_random_u8_vec(buf_size);
            c.sort();
            b.iter(|| three_vecs(&o, &c, &s))
        });

        group.bench_function(BenchmarkId::new("Three Vecs Usize Stride", buf_size), |b| {
            let o = make_random_usize_vec(buf_size + 1);
            let mut c = make_random_usize_vec(buf_size + 1);
            let s = make_random_usize_vec(buf_size);
            c.sort();
            b.iter(|| three_vecs(&o, &c, &s))
        });

        group.bench_function(
            BenchmarkId::new("Clumped Offsets Separate", buf_size),
            |b| {
                let mut c = make_random_usize_vec(buf_size);
                let os = make_random_usize_vec(2 * buf_size + 1);
                c.sort();
                b.iter(|| clumped_offsets_separate(&c, &os))
            },
        );

        group.bench_function(BenchmarkId::new("Stride and Offsets", buf_size), |b| {
            let mut o = make_random_usize_vec(2 * buf_size + 2);
            let s = make_random_u8_vec(buf_size);
            o.sort();
            b.iter(|| stride_and_offsets(&o, &s))
        });

        group.bench_function(BenchmarkId::new("All in One", buf_size), |b| {
            let mut o = make_random_usize_vec(3 * buf_size + 3);
            o.sort();
            b.iter(|| all_in_one(&o))
        });
    }

    group.finish();
}

criterion_group!(benches, storage_ints);
criterion_main!(benches);
