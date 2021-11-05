//!
//! This example finds the optimal workgroup size for the collatz routine run on the GPU.
//!

#[cfg(feature = "gpu")]
fn main() {
    use flatk::*;
    use std::time::Instant;

    let collatz: &str = stringify! {
        fn program(n_base: u32) -> u32 {
            var n: u32 = n_base;
            var i: u32 = 0u;
            loop {
                if (n <= 1u) {
                    break;
                }
                if (n % 2u == 0u) {
                    n = n / 2u;
                }
                else {
                    // Overflow? (i.e. 3*n + 1 > 0xffffffffu?)
                    if (n >= 1431655765u) {   // 0x55555555u
                        return 4294967295u;   // 0xffffffffu
                    }

                    n = 3u * n + 1u;
                }
                i = i + 1u;
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
        let map_gpu_slice = gpu_slice
            .map(collatz)
            .with_workgroup_size(wgs)
            .compile()
            .unwrap();

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
