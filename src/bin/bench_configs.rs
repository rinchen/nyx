//! `nyx bench:configs` — compare multiple model-stack configs on one corpus.
//!
//! Usage:
//!   cargo run --bin `bench_configs` -- <`corpus_dir`>
//!
//! Outputs one row per (file, config) pair with ratio%, cmp MB/s, dec MB/s.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::explicit_iter_loop
)]

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};

use nyx::codec;

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
    Compress {
        input: PathBuf,
        output: PathBuf,
    },
    Decompress {
        input: PathBuf,
        output: PathBuf,
    },
    Bench {
        corpus: PathBuf,
        #[arg(long)]
        vs: Option<String>,
    },
    BenchConfigs {
        corpus: PathBuf,
    },
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
        Cmd::Compress { input, output } => cmd_compress(&input, &output),
        Cmd::Decompress { input, output } => cmd_decompress(&input, &output),
        Cmd::Bench { corpus, .. } => cmd_bench(&corpus),
        Cmd::BenchConfigs { corpus } => cmd_bench_configs(&corpus),
        Cmd::SelfTest => cmd_selftest(),
    }
}

fn cmd_compress(input: &PathBuf, output: &PathBuf) -> Result<(), String> {
    let data = fs::read(input).map_err(|e| format!("read {}: {e}", input.display()))?;
    let compressed = codec::compress(&data).map_err(|e| format!("compress failed: {e}"))?;
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
    let restored = codec::decompress(&data).map_err(|e| format!("decompress failed: {e}"))?;
    fs::write(output, &restored).map_err(|e| format!("write {}: {e}", output.display()))?;
    eprintln!(
        "decompressed {} -> {} ({} bytes)",
        input.display(),
        output.display(),
        restored.len()
    );
    Ok(())
}

fn cmd_bench(corpus: &PathBuf) -> Result<(), String> {
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
        let Ok(compressed) = codec::compress(&data) else {
            continue;
        };
        let enc_ms = enc_start.elapsed().as_secs_f64() * 1000.0;

        let dec_start = Instant::now();
        let Ok(restored) = codec::decompress(&compressed) else {
            continue;
        };
        let dec_ms = dec_start.elapsed().as_secs_f64() * 1000.0;

        if restored != data {
            continue;
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
            dec_mbps,
        );
    }

    eprintln!("note: SOTA comparison is provided by scripts/bench_vs_sota.sh");
    Ok(())
}

fn measure_buf(
    label: &str,
    data: &[u8],
    build: impl Fn(
        nyx::classify::BlockKind,
    ) -> (
        Vec<Box<dyn nyx::model::BitModel>>,
        nyx::model::mixer_bank::MixerBank,
        Option<usize>,
    ),
) -> (usize, usize, f64, f64, f64) {
    let enc_start = Instant::now();
    let compress_build = &mut |kind| build(kind);
    let Ok(compressed) = codec::compress_with(data, compress_build) else {
        return (0, 0, 0.0, 0.0, 0.0);
    };
    let enc_ms = enc_start.elapsed().as_secs_f64() * 1000.0;
    let dec_start = Instant::now();
    let Ok(restored) = codec::decompress_with(&compressed, &mut |kind| build(kind)) else {
        return (compressed.len(), 0, 0.0, 0.0, 0.0);
    };
    let dec_ms = dec_start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(restored, data, "round-trip mismatch for {label}");
    let orig = data.len();
    let comp = compressed.len();
    let ratio = comp as f64 / orig as f64 * 100.0;
    let enc_mbps = (orig as f64 / 1e6) / (enc_ms / 1000.0);
    let dec_mbps = (orig as f64 / 1e6) / (dec_ms / 1000.0);
    (orig, comp, ratio, enc_mbps, dec_mbps)
}

fn cmd_bench_configs(corpus: &PathBuf) -> Result<(), String> {
    if !corpus.is_dir() {
        return Err(format!(
            "corpus path {} is not a directory",
            corpus.display()
        ));
    }

    let mut entries: Vec<_> = fs::read_dir(corpus)
        .map_err(|e| format!("read dir {}: {e}", corpus.display()))?
        .filter_map(Result::ok)
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);

    println!(
        "{:<20} {:<28} {:>10} {:>10} {:>9} {:>11} {:>11}",
        "config", "name", "orig_kb", "comp_kb", "ratio%", "cmp_MBps", "dec_MBps"
    );
    println!("{}", "-".repeat(102));

    for entry in entries {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().is_some_and(|e| e == "nyx") {
            continue;
        }
        let Ok(data) = fs::read(&path) else {
            continue;
        };
        if data.is_empty() {
            continue;
        }

        let baseline_ratio =
            measure_buf("baseline", &data, |_| nyx::stacks::BaselineBuilder::build()).2;

        let mut cases: [(&str, &mut dyn FnMut(nyx::classify::BlockKind) -> _); 5] = [
            ("baseline", &mut |_| nyx::stacks::BaselineBuilder::build()),
            ("ppm3", &mut |_| nyx::stacks::PpmBuilder::new(3).build()),
            ("ppm4", &mut |_| nyx::stacks::PpmBuilder::new(4).build()),
            ("hybrid_ppm3", &mut |_| {
                nyx::stacks::HybridPpm3Builder::build()
            }),
            ("ppmd_ssm", &mut |_| nyx::stacks::PpmdSsmBuilder::build()),
        ];

        for (label, builder) in &mut cases {
            let (orig, comp, ratio, enc_mbps, dec_mbps) = measure_buf(label, &data, &mut *builder);
            let marker = if label != &"baseline" && (ratio - baseline_ratio).abs() < 0.05 {
                "="
            } else if ratio < baseline_ratio - 0.05 {
                "↑"
            } else if ratio > baseline_ratio + 0.05 {
                "↓"
            } else {
                "~"
            };
            println!(
                "{marker:<3} {:<17} {:<28} {:>10.1} {:>10.1} {:>8.1}% {:>11.1} {:>11.1}",
                label,
                path.file_name().unwrap().to_string_lossy(),
                orig as f64 / 1024.0,
                comp as f64 / 1024.0,
                ratio,
                enc_mbps,
                dec_mbps,
            );
        }
        println!("{}", "-".repeat(102));
    }

    Ok(())
}

fn cmd_selftest() -> Result<(), String> {
    eprintln!("running cargo test --lib ...");
    let status = std::process::Command::new("cargo")
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
