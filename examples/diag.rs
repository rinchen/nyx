//! Diagnostic example: round-trips small/repetitive/zero/random inputs through the codec
//! and prints ratio + classification. Deliberate numeric casts (usize->f64 ratio, u32->u8
//! byte extraction) are exact here.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use nyx::classify::classify;
use nyx::codec::{compress, decompress};

fn main() {
    // Small reproducible input first
    let small = b"nyxnyxnyxnyxnyxnyxnyxnyxnyxnyx";
    println!("small input len = {}", small.len());
    println!("  classify = {:?}", classify(small));
    let c = compress(small).unwrap();
    println!("  compressed len = {}", c.len());
    let back = decompress(&c).unwrap();
    println!("  roundtrip ok = {}", back == small);

    // Larger repetitive
    let mut reps = Vec::new();
    for _ in 0..20_000 {
        reps.extend_from_slice(b"nyxnyxnyx");
    }
    println!("\nreps input len = {}", reps.len());
    println!("  classify = {:?}", classify(&reps));
    let c = compress(&reps).unwrap();
    println!("  compressed len = {}", c.len());
    println!("  ratio = {:.3}x", c.len() as f64 / reps.len() as f64);

    // Pure zero (very predictable)
    let zeros = vec![0u8; 50_000];
    println!("\nzeros input len = {}", zeros.len());
    println!("  classify = {:?}", classify(&zeros));
    let c = compress(&zeros).unwrap();
    println!("  compressed len = {}", c.len());
    println!("  ratio = {:.3}x", c.len() as f64 / zeros.len() as f64);

    // Random-ish (should expand a little, ~1x)
    let mut x = 0x1234_5678u32;
    let mut rnd = Vec::new();
    for _ in 0..50_000 {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        rnd.push(x as u8);
    }
    println!("\nrnd input len = {}", rnd.len());
    println!("  classify = {:?}", classify(&rnd));
    let c = compress(&rnd).unwrap();
    println!("  compressed len = {}", c.len());
    println!("  ratio = {:.3}x", c.len() as f64 / rnd.len() as f64);
}
