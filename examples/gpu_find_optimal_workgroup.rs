//!
//! This example finds the optimal workgroup size for the collatz routine run on the GPU.
//!

#[cfg(feature = "gpu")]
fn main() {
    use flatk::*;
    use std::time::Instant;

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

    let numbers = (1..10_000_000).collect::<Vec<_>>();

    let mut prev_result: Option<Vec<u32>> = None;

    for i in 0..11 {
        let wgs = 2u32.pow(i);
        // Prepare the slice on the gpu
        let gpu_slice = numbers.as_slice().into_gpu();
        // Compile the map program to the gpu
        let map_gpu_slice = gpu_slice.map(collatz).with_workgroup_size(wgs).compile();

        let now = Instant::now();
        // Execute the gpu program and collect the results.
        let result = map_gpu_slice.run().collect();
        let elapsed = now.elapsed().as_millis();
        if let Some(prev_result) = prev_result.as_ref() {
            assert_eq!(result.as_slice(), prev_result.as_slice());
        } else {
            prev_result = Some(result);
        }
        eprintln!(
            "wgs = {}, duration = {}, result = {}",
            wgs,
            elapsed,
            prev_result.as_ref().unwrap().last().unwrap()
        );
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("This example requires the \"gpu\" feature");
}
