#!/usr/bin/env bash
#
# bench_vs_sota.sh — compare `nyx` against reference compressors on a corpus.
#
# Builds nyx (release), then for every regular file in the corpus directory runs
# zstd -19, xz -9, brotli -11, lz4 -9 (skipping any not installed) and nyx,
# tabulating (name, orig_kb, comp_kb, ratio%, cmp_MBps, dec_MBps).
#
# Usage: scripts/bench_vs_sota.sh <corpus_dir> [nyx_bin]
#   corpus_dir  directory of files to compress (subdirs are skipped)
#   nyx_bin     optional path to a nyx binary (default: ./target/release/nyx)

set -u

CORPUS="${1:-}"
NYX="${2:-./target/release/nyx}"

if [[ -z "$CORPUS" || ! -d "$CORPUS" ]]; then
    echo "usage: $0 <corpus_dir> [nyx_bin]" >&2
    exit 1
fi

# Build nyx in release mode unless a binary was handed in.
if [[ ! -x "$NYX" ]]; then
    echo "building nyx (release)..." >&2
    cargo build --release --bin nyx || { echo "nyx build failed" >&2; exit 1; }
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Positive-float seconds via python3 (portable high-res timer on macOS).
now() { python3 -c 'import time; print("%.6f" % time.time())'; }

# run_and_time <out_file> <cmd...>  -> prints elapsed_seconds
run_and_time() {
    local out="$1"; shift
    local t0 t1
    t0="$(now)"
    "$@" >"$out" 2>/dev/null
    t1="$(now)"
    awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.6f", b - a }'
}

mbps() { # mb_per_sec(bytes, seconds)
    awk -v b="$1" -v s="$2" 'BEGIN { if (s > 0) printf "%.1f", (b/1e6)/s; else printf "0.0" }'
}

have() { command -v "$1" >/dev/null 2>&1; }

fmt() { # name orig_kb comp_kb ratio cmp dec
    printf "%-10s %10.1f %10.1f %8.1f%% %11.1f %11.1f\n" "$1" "$2" "$3" "$4" "$5" "$6"
}

echo "corpus: $CORPUS"
echo "$(printf '%-10s %10s %10s %9s %11s %11s' name orig_kb comp_kb ratio% cmp_MBps dec_MBps)"
echo "$(printf '%0.s-' $(seq 1 67))"

for f in "$CORPUS"/*; do
    [[ -f "$f" ]] || continue
    name="$(basename "$f")"
    orig="$(wc -c <"$f")"
    [[ "$orig" -eq 0 ]] && continue
    okb="$(awk -v b="$orig" 'BEGIN { printf "%.1f", b/1024 }')"

    # --- nyx ---
    if [[ -x "$NYX" ]]; then
        t="$(run_and_time "$WORK/n.nyx" "$NYX" compress "$f" "$WORK/n.nyx")"
        c="$(wc -c <"$WORK/n.nyx")"
        t2="$(run_and_time "$WORK/n.out" "$NYX" decompress "$WORK/n.nyx" "$WORK/n.out")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        fmt "nyx" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    # --- zstd -19 ---
    if have zstd; then
        t="$(run_and_time "$WORK/z.zst" zstd -19 -q -f -o "$WORK/z.zst" "$f")"
        c="$(wc -c <"$WORK/z.zst")"
        t2="$(run_and_time "$WORK/z.out" zstd -q -d -f -o "$WORK/z.out" "$WORK/z.zst")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        fmt "zstd-19" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    # --- xz -9 ---
    if have xz; then
        t="$(run_and_time "$WORK/x.xz" xz -9 -c "$f")"
        c="$(wc -c <"$WORK/x.xz")"
        t2="$(run_and_time "$WORK/x.out" xz -d -c "$WORK/x.xz")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        fmt "xz-9" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    # --- brotli -11 ---
    if have brotli; then
        t="$(run_and_time "$WORK/b.br" brotli -11 -c "$f")"
        c="$(wc -c <"$WORK/b.br")"
        t2="$(run_and_time "$WORK/b.out" brotli -d -c "$WORK/b.br")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        fmt "brotli-11" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    # --- lz4 -9 ---
    if have lz4; then
        t="$(run_and_time "$WORK/l.lz4" lz4 -9 -q -f "$f" "$WORK/l.lz4")"
        c="$(wc -c <"$WORK/l.lz4")"
        t2="$(run_and_time "$WORK/l.out" lz4 -d -q -f "$WORK/l.lz4" "$WORK/l.out")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        fmt "lz4-9" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi
done
