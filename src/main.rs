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
use rcn::level::Level;

#[derive(Parser)]
#[command(
    name = "rcn",
    version,
    about = "Rcn: leveled compressor (win ratio and compress speed vs each peer class)",
    long_about = "Rcn is a leveled compressor. Each level owns a peer class on \
both ratio and compress throughput: -1 vs lz4-9/zstd-1, -3 vs gzip-9, \
-9 (default, hybrid) vs zstd-19, -19 vs archival peers. It stages BWT, LZP, \
hash-chain LZ, and context mixing, then entropy-codes with rANS. \
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
    /// Level -9: Fast for Text, Slow for Binary/Exec.
    Hybrid,
    /// Level -19: bit-level CM on every block.
    Slow,
    /// Byte-level CM with BWT trials (not a numbered level).
    Fast,
    /// Level -1: hash-chain LZ + optional order-0 rANS.
    Wire,
    /// Level -3: byte CM + DP-LZP, no BWT trials.
    General,
}

impl ModeArg {
    fn to_mode(self) -> CodecMode {
        match self {
            Self::Slow => CodecMode::Slow,
            Self::Fast => CodecMode::Fast,
            Self::Hybrid => CodecMode::Hybrid,
            Self::Wire => CodecMode::Wire,
            Self::General => CodecMode::General,
        }
    }
}

fn parse_level(s: &str) -> Result<Level, String> {
    let n: i32 = s
        .parse()
        .map_err(|_| format!("invalid level '{s}' (use 1, 3, 9, 19 or -1, -3, -9, -19)"))?;
    Level::from_i32(n).ok_or_else(|| {
        format!("unsupported level {n} (use 1, 3, 9, 19 or -1, -3, -9, -19)")
    })
}

fn resolve_engine(level: Option<Level>, mode: Option<ModeArg>) -> Result<CodecMode, String> {
    match (level, mode) {
        (Some(level), Some(mode)) => {
            let from_level = level.mode();
            let from_mode = mode.to_mode();
            if from_level != from_mode {
                return Err(format!(
                    "--level {} maps to {:?}, which conflicts with --mode",
                    level.number(),
                    from_level
                ));
            }
            Ok(from_level)
        }
        (Some(level), None) => Ok(level.mode()),
        (None, Some(mode)) => Ok(mode.to_mode()),
        (None, None) => Ok(CodecMode::Hybrid),
    }
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
        /// Level: 1 (wire), 3 (general), 9 (hybrid, default), 19 (max).
        /// Also accepts -1 / -3 / -9 / -19.
        #[arg(short = 'L', long = "level", value_parser = parse_level)]
        level: Option<Level>,
        /// Engine alias: `hybrid` (-9), `slow` (-19), `fast` (BWT byte CM),
        /// `wire` (-1), or `general` (-3). Default hybrid when `--level` omitted.
        #[arg(long, value_enum)]
        mode: Option<ModeArg>,
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
        /// Level: 1 / 3 / 9 / 19 (or negative aliases). Default 9.
        #[arg(short = 'L', long = "level", value_parser = parse_level)]
        level: Option<Level>,
        /// Use the byte-level BWT (fast) entropy mode.
        #[arg(long)]
        fast: bool,
        /// Use Hybrid mode (Text→fast, Binary→slow). Default when no flags set.
        #[arg(long)]
        hybrid: bool,
        /// Use the level -1 wire engine.
        #[arg(long)]
        wire: bool,
        /// Use the level -3 general engine.
        #[arg(long)]
        general: bool,
        /// Use Slow bit CM (level -19).
        #[arg(long)]
        slow: bool,
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
            level,
            mode,
            verbose,
        } => cmd_compress(&input, &output, &backend, level, mode, verbose),
        Cmd::Decompress { input, output } => cmd_decompress(&input, &output),
        Cmd::Bench {
            corpus,
            vs,
            level,
            fast,
            hybrid,
            wire,
            general,
            slow,
        } => cmd_bench(
            &corpus,
            vs.as_deref(),
            level,
            fast,
            hybrid,
            wire,
            general,
            slow,
        ),
        Cmd::SelfTest => cmd_selftest(),
    }
}

fn cmd_compress(
    input: &PathBuf,
    output: &PathBuf,
    backend: &str,
    level: Option<Level>,
    mode: Option<ModeArg>,
    verbose: bool,
) -> Result<(), String> {
    if backend != "rans" {
        return Err(format!(
            "unsupported backend '{backend}' (only 'rans' is built in)"
        ));
    }
    let mode = resolve_engine(level, mode)?;
    let data = fs::read(input).map_err(|e| format!("read {}: {e}", input.display()))?;
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

fn cmd_bench(
    corpus: &PathBuf,
    vs: Option<&str>,
    level: Option<Level>,
    fast: bool,
    hybrid: bool,
    wire: bool,
    general: bool,
    slow: bool,
) -> Result<(), String> {
    if !corpus.is_dir() {
        return Err(format!(
            "corpus path {} is not a directory",
            corpus.display()
        ));
    }
    let flag_count = [fast, hybrid, wire, general, slow]
        .iter()
        .filter(|b| **b)
        .count();
    if flag_count > 1 {
        return Err("use only one of --fast / --hybrid / --wire / --general / --slow".into());
    }
    let mode_from_flag = if fast {
        Some(ModeArg::Fast)
    } else if hybrid {
        Some(ModeArg::Hybrid)
    } else if wire {
        Some(ModeArg::Wire)
    } else if general {
        Some(ModeArg::General)
    } else if slow {
        Some(ModeArg::Slow)
    } else {
        None
    };
    let mode = resolve_engine(level, mode_from_flag)?;
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
