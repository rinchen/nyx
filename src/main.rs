//! `rcn` command-line interface.
//!
//! Subcommands: `compress`, `decompress`, `bench` (vs a corpus directory), and
//! `self-test` (runs the library's `#[test]` suite). The codec is currently
//! rANS-backed end-to-end; the `--backend` flag validates that (only `rans` is built in).

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::fs;
use std::path::PathBuf;

// Use mimalloc as the global allocator for reduced BWT trial allocation overhead
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::process::Command;
use std::time::Instant;

use clap::{Parser, Subcommand};
use rcn::codec::{self, decompress, CodecMode};

#[derive(Parser)]
#[command(
    name = "rcn",
    version,
    about = "Rcn: adaptive staged context-mixing compressor",
    long_about = "Rcn is a ratio-first context-mixing compressor. It stages \
BWT, LZP, and online logistic mixing, then entropy-codes with rANS. \
Subcommands compress and decompress .rcn (RCN1) containers, bench a corpus, \
or run the library self-test.",
    after_help = "See also: man rcn (if installed), or man/rcn.1 in the source tree.",
    arg_required_else_help = true,
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum ModeArg {
    /// Bit-level CM (default). Stronger on some binary; much slower; loses text to zstd -19.
    Slow,
    /// Byte-level CM. Usually better text ratio + throughput; loses to zstd -19 on mr.
    Fast,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compress a file into a .rcn (RCN1) container.
    Compress {
        /// Path to the input file.
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        /// Path for the compressed .rcn output.
        #[arg(value_name = "OUTPUT")]
        output: PathBuf,
        /// Entropy backend (only `rans` is built in).
        #[arg(long, default_value = "rans")]
        backend: String,
        /// Entropy mode: `slow` (bit-level CM, default) or `fast` (byte-level CM).
        #[arg(long, value_enum, default_value_t = ModeArg::Slow)]
        mode: ModeArg,
        /// Print per-block kind, method, and sizes to stderr.
        #[arg(long, short = 'v')]
        verbose: bool,
    },
    /// Decompress a .rcn (RCN1) container back to a file.
    Decompress {
        /// Path to the .rcn container.
        #[arg(value_name = "INPUT")]
        input: PathBuf,
        /// Path for the restored output file.
        #[arg(value_name = "OUTPUT")]
        output: PathBuf,
    },
    /// Benchmark rcn over every file in a corpus directory.
    Bench {
        /// Directory of files to compress for timing and ratio.
        #[arg(value_name = "CORPUS")]
        corpus: PathBuf,
        /// Reserved for SOTA comparison (see `scripts/bench_vs_sota.sh`).
        #[arg(long)]
        vs: Option<String>,
        /// Use the byte-level (fast) entropy mode.
        #[arg(long)]
        fast: bool,
    },
    /// Run the library test suite and report PASS/FAIL.
    SelfTest,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("rcn: error: {e}");
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
            mode,
            verbose,
        } => cmd_compress(&input, &output, &backend, mode, verbose),
        Cmd::Decompress { input, output } => cmd_decompress(&input, &output),
        Cmd::Bench { corpus, vs, fast } => cmd_bench(&corpus, vs.as_deref(), fast),
        Cmd::SelfTest => cmd_selftest(),
    }
}

fn cmd_compress(
    input: &PathBuf,
    output: &PathBuf,
    backend: &str,
    mode: ModeArg,
    verbose: bool,
) -> Result<(), String> {
    if backend != "rans" {
        return Err(format!(
            "unsupported backend '{backend}' (only 'rans' is built in)"
        ));
    }
    let data = fs::read(input).map_err(|e| format!("read {}: {e}", input.display()))?;
    let mode = match mode {
        ModeArg::Slow => CodecMode::Slow,
        ModeArg::Fast => CodecMode::Fast,
    };
    let (compressed, diags) = codec::compress_mode_diag(&data, mode)
        .map_err(|e| format!("compress failed: {e}"))?;
    if verbose {
        for (i, d) in diags.iter().enumerate() {
            eprintln!(
                "block {i}: kind={} method={} ({}) in={} out={}",
                d.kind,
                d.method,
                codec::method_label(d.method),
                d.size_in,
                d.size_out
            );
        }
    }
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

fn cmd_bench(corpus: &PathBuf, vs: Option<&str>, fast: bool) -> Result<(), String> {
    if !corpus.is_dir() {
        return Err(format!(
            "corpus path {} is not a directory",
            corpus.display()
        ));
    }
    let mode = if fast { CodecMode::Fast } else { CodecMode::Slow };
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
        // accidentally contains .rcn files doesn't benchmark the wrapper.
        if path.extension().is_some_and(|e| e == "rcn") {
            continue;
        }
        let Ok(data) = fs::read(&path) else {
            continue;
        };
        if data.is_empty() {
            continue;
        }

        let enc_start = Instant::now();
        let Ok(compressed) = codec::compress_mode(&data, mode) else {
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
            "{:<28} {:>10.1} {:>10.1} {:>8.1}% {:>11.2} {:>11.2}",
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
