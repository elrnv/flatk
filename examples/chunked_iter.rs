use rand::distributions::{Distribution, Standard};
use rand::prelude::*;

static SEED: [u8; 32] = [3; 32];

#[inline]
fn make_random_vec(n: usize) -> Vec<f64>
where
    Standard: Distribution<f64>,
{
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    (0..n).map(move |_| rng.gen()).collect()
}

#[derive(Debug)]
struct Error;

#[inline]
fn compute(x: f64, y: f64, z: f64) -> [f64; 3] {
    [
        x * 3.6321 + 42314.0 * y + z * 2.1,
        y * 3.6321 + 42314.0 * z + x * 2.1,
        z * 3.6321 + 42314.0 * x + y * 2.1,
    ]
}

fn main() -> Result<(), Error> {
    let mut v: Vec<f64> = make_random_vec(30_000_000);
    let s: &mut [[f64; 3]] = unsafe { reinterpret::reinterpret_mut_slice(v.as_mut_slice()) };
    for a in s.iter_mut() {
        //for (i, a) in v.chunks_exact_mut(3).enumerate() {
        //for a in flatk::Chunked3::from_flat(v.as_mut_slice()).into_iter() {
        //Chunked3::from_flat(v.as_mut_slice()).into_par_iter().enumerate().for_each(|(i, a)| {
        *a = compute(a[0], a[1], a[2]);
    }
    if *v.last().unwrap() > 100.0 {
        Err(Error)
    } else {
        Ok(())
    }
}
