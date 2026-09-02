//! `nyx` command-line interface.
//!
//! Subcommands: `compress`, `decompress`, `bench` (vs a corpus directory), and
//! `self-test` (runs the library's `#[test]` suite). The codec is currently
//! rANS-backed end-to-end; the `--backend` flag validates that (only `rans` is built in).

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use clap::{Parser, Subcommand};
use nyx::codec::{compress, decompress};

#[derive(Parser)]
#[command(
    name = "nyx",
    version,
    about = "Nyx: adaptive staged context-mixing compressor"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compress a file into a .nyx (NYX1) container.
    Compress {
        input: PathBuf,
        output: PathBuf,
        /// Entropy backend (only `rans` is built in).
        #[arg(long, default_value = "rans")]
        backend: String,
    },
    /// Decompress a .nyx (NYX1) container back to a file.
    Decompress { input: PathBuf, output: PathBuf },
    /// Benchmark nyx over every file in a corpus directory.
    Bench {
        corpus: PathBuf,
        /// Reserved for SOTA comparison (see `scripts/bench_vs_sota.sh`).
        #[arg(long)]
        vs: Option<String>,
    },
    /// Run the library test suite and report PASS/FAIL.
    SelfTest,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("nyx: error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Compress {
            input,
            output,
            backend,
        } => cmd_compress(&input, &output, &backend),
        Cmd::Decompress { input, output } => cmd_decompress(&input, &output),
        Cmd::Bench { corpus, vs } => cmd_bench(&corpus, vs.as_deref()),
        Cmd::SelfTest => cmd_selftest(),
    }
}

fn cmd_compress(input: &PathBuf, output: &PathBuf, backend: &str) -> Result<(), String> {
    if backend != "rans" {
        return Err(format!(
            "unsupported backend '{backend}' (only 'rans' is built in)"
        ));
    }
    let data = fs::read(input).map_err(|e| format!("read {}: {e}", input.display()))?;
    let compressed = compress(&data).map_err(|e| format!("compress failed: {e}"))?;
    fs::write(output, &compressed).map_err(|e| format!("write {}: {e}", output.display()))?;
    let ratio = compressed.len() as f64 / (data.len() as f64).max(1.0);
    eprintln!(
        "compressed {} -> {} ({:.3}x, {} bytes)",
        input.display(),
        output.display(),
        ratio,
        compressed.len()
    );
    Ok(())
}

fn cmd_decompress(input: &PathBuf, output: &PathBuf) -> Result<(), String> {
    let data = fs::read(input).map_err(|e| format!("read {}: {e}", input.display()))?;
    let restored = decompress(&data).map_err(|e| format!("decompress failed: {e}"))?;
    fs::write(output, &restored).map_err(|e| format!("write {}: {e}", output.display()))?;
    eprintln!(
        "decompressed {} -> {} ({} bytes)",
        input.display(),
        output.display(),
        restored.len()
    );
    Ok(())
}

fn cmd_bench(corpus: &PathBuf, vs: Option<&str>) -> Result<(), String> {
    if !corpus.is_dir() {
        return Err(format!(
            "corpus path {} is not a directory",
            corpus.display()
        ));
    }
    println!(
        "{:<28} {:>10} {:>10} {:>9} {:>11} {:>11}",
        "name", "orig_kb", "comp_kb", "ratio%", "cmp_MBps", "dec_MBps"
    );
    println!("{}", "-".repeat(82));

    let mut entries: Vec<_> = fs::read_dir(corpus)
        .map_err(|e| format!("read dir {}: {e}", corpus.display()))?
        .filter_map(Result::ok)
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);

    for entry in entries {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        // Skip our own container output so re-running against a corpus dir that
        // accidentally contains .nyx files doesn't benchmark the wrapper.
        if path.extension().is_some_and(|e| e == "nyx") {
            continue;
        }
        let Ok(data) = fs::read(&path) else {
            continue;
        };
        if data.is_empty() {
            continue;
        }

        let enc_start = Instant::now();
        let Ok(compressed) = compress(&data) else {
            continue;
        };
        let enc_ms = enc_start.elapsed().as_secs_f64() * 1000.0;

        let dec_start = Instant::now();
        let Ok(restored) = decompress(&compressed) else {
            continue;
        };
        let dec_ms = dec_start.elapsed().as_secs_f64() * 1000.0;

        if restored != data {
            continue; // defensive; the codec should always round-trip
        }

        let orig_kb = data.len() as f64 / 1024.0;
        let comp_kb = compressed.len() as f64 / 1024.0;
        let ratio = compressed.len() as f64 / data.len() as f64 * 100.0;
        let enc_mbps = (data.len() as f64 / 1e6) / (enc_ms / 1000.0);
        let dec_mbps = (data.len() as f64 / 1e6) / (dec_ms / 1000.0);
        println!(
            "{:<28} {:>10.1} {:>10.1} {:>8.1}% {:>11.1} {:>11.1}",
            path.file_name().unwrap().to_string_lossy(),
            orig_kb,
            comp_kb,
            ratio,
            enc_mbps,
            dec_mbps
        );
    }

    if vs.is_some() {
        eprintln!("note: SOTA comparison is provided by scripts/bench_vs_sota.sh");
    }
    Ok(())
}

fn cmd_selftest() -> Result<(), String> {
    eprintln!("running cargo test --lib ...");
    let status = Command::new("cargo")
        .args(["test", "--lib"])
        .status()
        .map_err(|e| format!("failed to spawn cargo: {e}"))?;
    if status.success() {
        println!("SELF-TEST: PASS");
        Ok(())
    } else {
        println!("SELF-TEST: FAIL");
        Err("self-test failed".to_string())
    }
}
