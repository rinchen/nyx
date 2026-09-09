#!/usr/bin/env bash
#
# bench_vs_sota.sh — compare `rcn` against reference compressors on a corpus.
#
# Builds rcn (release), then for every regular file in the corpus directory runs
# zstd -1 (fast baseline), zstd -19 (goal), xz -9, brotli -11, lz4 -9 (skipping
# any not installed) and rcn, tabulating
# (name, orig_kb, comp_kb, ratio%, cmp_MBps, dec_MBps).
#
# Usage: scripts/bench_vs_sota.sh <corpus_dir> [rcn_bin]
#   corpus_dir  directory of files to compress (subdirs are skipped)
#   rcn_bin     optional path to a rcn binary (default: ./target/release/rcn)
#
# Environment:
#   SKIP_RCN=1  skip the slow rcn pass (peer CLIs only)

set -u

CORPUS="${1:-}"
RCN="${2:-./target/release/rcn}"

if [[ -z "$CORPUS" || ! -d "$CORPUS" ]]; then
    echo "usage: $0 <corpus_dir> [rcn_bin]" >&2
    exit 1
fi

# Build rcn in release mode unless a binary was handed in or SKIP_RCN.
if [[ "${SKIP_RCN:-0}" != "1" && ! -x "$RCN" ]]; then
    echo "building rcn (release)..." >&2
    cargo build --release --bin rcn || { echo "rcn build failed" >&2; exit 1; }
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

    echo "# $name" 

    # --- rcn (default = slow) ---
    if [[ "${SKIP_RCN:-0}" != "1" && -x "$RCN" ]]; then
        t="$(run_and_time "$WORK/n.rcn" "$RCN" compress "$f" "$WORK/n.rcn")"
        c="$(wc -c <"$WORK/n.rcn")"
        t2="$(run_and_time "$WORK/n.out" "$RCN" decompress "$WORK/n.rcn" "$WORK/n.out")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        fmt "rcn" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    # --- zstd -1 (fast baseline; not the goal) ---
    if have zstd; then
        t="$(run_and_time "$WORK/z1.zst" zstd -1 -q -f -o "$WORK/z1.zst" "$f")"
        c="$(wc -c <"$WORK/z1.zst")"
        t2="$(run_and_time "$WORK/z1.out" zstd -q -d -f -o "$WORK/z1.out" "$WORK/z1.zst")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        fmt "zstd-1" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    # --- zstd -19 (primary goal) ---
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
        t="$(run_and_time "$WORK/b.br" brotli -q 11 -c "$f")"
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

    # --- gzip -9 ---
    if have gzip; then
        t="$(run_and_time "$WORK/g.gz" gzip -9 -c "$f")"
        c="$(wc -c <"$WORK/g.gz")"
        t2="$(run_and_time "$WORK/g.out" gzip -d -c "$WORK/g.gz")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        fmt "gzip-9" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi
done
