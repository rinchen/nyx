#!/usr/bin/env bash
#
# bench_vs_sota.sh — compare `rcn` against reference compressors on a corpus.
#
# Builds rcn (release), then for every regular file in the corpus directory runs
# rcn (slow + fast), zstd -1 (fast baseline), zstd -19 (goal), xz -9, brotli -11,
# lz4 -9, gzip -9 (skipping any not installed) and tabulates
# (name, orig_kb, comp_kb, ratio%, cmp_MBps, dec_MBps).
#
# When rcn is included, also prints a per-file ratio scorecard and a corpus
# W-L-T summary (lower ratio% wins; tie if equal or both < 0.5).
#
# Usage: scripts/bench_vs_sota.sh <corpus_dir> [rcn_bin]
#   corpus_dir  directory of files to compress (subdirs are skipped)
#   rcn_bin     optional path to a rcn binary (default: ./target/release/rcn)
#
# Environment:
#   SKIP_RCN=1  skip rcn (peer CLIs only; no scorecard)

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

# Ratio verdict: win / lose / tie (tie if equal or both < 0.5).
verdict() {
    awk -v a="$1" -v b="$2" 'BEGIN {
        if (a == b || (a < 0.5 && b < 0.5)) print "tie"
        else if (a < b) print "win"
        else print "lose"
    }'
}

# Per-file ratio stash: RATIO_<sanitized_label>=value
ratio_key() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9' '_'
}

set_ratio() {
    local key
    key="$(ratio_key "$1")"
    eval "RATIO_${key}=\"$2\""
}

get_ratio() {
    local key
    key="$(ratio_key "$1")"
    eval "printf '%s' \"\${RATIO_${key}:-}\""
}

clear_ratios() {
    unset RATIO_rcn RATIO_rcn_fast RATIO_zstd_1 RATIO_zstd_19 \
          RATIO_xz_9 RATIO_brotli_11 RATIO_lz4_9 RATIO_gzip_9 2>/dev/null || true
}

emit_row() { # name orig_kb comp_kb ratio cmp dec
    printf "%-10s %10.1f %10.1f %8.1f%% %11.1f %11.1f\n" "$1" "$2" "$3" "$4" "$5" "$6"
    set_ratio "$1" "$4"
}

# Corpus tallies: TALLY_<mode>_<peer>_{w,l,t}
bump_tally() {
    local mode="$1" peer="$2" v="$3"
    local base key cur
    base="$(ratio_key "${mode}_${peer}")"
    case "$v" in
        win)  key="TALLY_${base}_w" ;;
        lose) key="TALLY_${base}_l" ;;
        tie)  key="TALLY_${base}_t" ;;
        *) return ;;
    esac
    cur="$(eval "printf '%s' \"\${${key}:-0}\"")"
    eval "${key}=$((cur + 1))"
}

print_file_scorecard() {
    local mode rcn_r peer peer_r v
    local peers="zstd-19 xz-9 brotli-11 gzip-9 lz4-9 zstd-1"
    local any=0

    for mode in rcn rcn-fast; do
        rcn_r="$(get_ratio "$mode")"
        [[ -n "$rcn_r" ]] || continue
        if [[ $any -eq 0 ]]; then
            echo "# $1  (ratio scorecard; lower ratio% wins)"
            any=1
        fi
        printf "%-8s vs" "$mode"
        for peer in $peers; do
            peer_r="$(get_ratio "$peer")"
            if [[ -z "$peer_r" ]]; then
                continue
            fi
            v="$(verdict "$rcn_r" "$peer_r")"
            printf " %s=%s" "$peer" "$v"
            bump_tally "$mode" "$peer" "$v"
        done
        printf "\n"
    done
}

print_corpus_summary() {
    local mode peer base w l t
    local peers="zstd-19 xz-9 brotli-11 gzip-9 lz4-9 zstd-1"
    local any=0

    for mode in rcn rcn-fast; do
        for peer in $peers; do
            base="$(ratio_key "${mode}_${peer}")"
            eval "w=\${TALLY_${base}_w:-0}"
            eval "l=\${TALLY_${base}_l:-0}"
            eval "t=\${TALLY_${base}_t:-0}"
            if [[ $((w + l + t)) -eq 0 ]]; then
                continue
            fi
            if [[ $any -eq 0 ]]; then
                echo
                echo "corpus W-L-T (ratio only; lower ratio% wins):"
                any=1
            fi
            printf "  %-8s vs %-10s  %d-%d-%d\n" "$mode" "$peer" "$w" "$l" "$t"
        done
    done
}

echo "corpus: $CORPUS"
if [[ "${SKIP_RCN:-0}" == "1" ]]; then
    echo "note: SKIP_RCN=1 — peer numbers only; scorecard needs rcn"
fi
echo "$(printf '%-10s %10s %10s %9s %11s %11s' name orig_kb comp_kb ratio% cmp_MBps dec_MBps)"
echo "$(printf '%0.s-' $(seq 1 67))"

for f in "$CORPUS"/*; do
    [[ -f "$f" ]] || continue
    name="$(basename "$f")"
    orig="$(wc -c <"$f" | tr -d '[:space:]')"
    [[ "$orig" -eq 0 ]] && continue
    okb="$(awk -v b="$orig" 'BEGIN { printf "%.1f", b/1024 }')"
    clear_ratios

    echo "# $name"

    # --- rcn slow (default) ---
    if [[ "${SKIP_RCN:-0}" != "1" && -x "$RCN" ]]; then
        t="$(run_and_time "$WORK/n.rcn" "$RCN" compress --mode slow "$f" "$WORK/n.rcn")"
        c="$(wc -c <"$WORK/n.rcn" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/n.out" "$RCN" decompress "$WORK/n.rcn" "$WORK/n.out")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "rcn" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    # --- rcn fast ---
    if [[ "${SKIP_RCN:-0}" != "1" && -x "$RCN" ]]; then
        t="$(run_and_time "$WORK/nf.rcn" "$RCN" compress --mode fast "$f" "$WORK/nf.rcn")"
        c="$(wc -c <"$WORK/nf.rcn" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/nf.out" "$RCN" decompress "$WORK/nf.rcn" "$WORK/nf.out")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "rcn-fast" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    # --- zstd -1 (fast baseline; not the goal) ---
    if have zstd; then
        t="$(run_and_time "$WORK/z1.zst" zstd -1 -q -f -o "$WORK/z1.zst" "$f")"
        c="$(wc -c <"$WORK/z1.zst" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/z1.out" zstd -q -d -f -o "$WORK/z1.out" "$WORK/z1.zst")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "zstd-1" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    # --- zstd -19 (primary goal) ---
    if have zstd; then
        t="$(run_and_time "$WORK/z.zst" zstd -19 -q -f -o "$WORK/z.zst" "$f")"
        c="$(wc -c <"$WORK/z.zst" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/z.out" zstd -q -d -f -o "$WORK/z.out" "$WORK/z.zst")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "zstd-19" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    # --- xz -9 ---
    if have xz; then
        t="$(run_and_time "$WORK/x.xz" xz -9 -c "$f")"
        c="$(wc -c <"$WORK/x.xz" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/x.out" xz -d -c "$WORK/x.xz")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "xz-9" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    # --- brotli -11 ---
    if have brotli; then
        t="$(run_and_time "$WORK/b.br" brotli -q 11 -c "$f")"
        c="$(wc -c <"$WORK/b.br" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/b.out" brotli -d -c "$WORK/b.br")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "brotli-11" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    # --- lz4 -9 ---
    if have lz4; then
        t="$(run_and_time "$WORK/l.lz4" lz4 -9 -q -f "$f" "$WORK/l.lz4")"
        c="$(wc -c <"$WORK/l.lz4" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/l.out" lz4 -d -q -f "$WORK/l.lz4" "$WORK/l.out")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "lz4-9" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    # --- gzip -9 ---
    if have gzip; then
        t="$(run_and_time "$WORK/g.gz" gzip -9 -c "$f")"
        c="$(wc -c <"$WORK/g.gz" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/g.out" gzip -d -c "$WORK/g.gz")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "gzip-9" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    print_file_scorecard "$name"
done

print_corpus_summary
