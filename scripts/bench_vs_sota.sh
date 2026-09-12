#!/usr/bin/env bash
#
# bench_vs_sota.sh — compare `rcn` against reference compressors on a corpus.
#
# Builds rcn (release), then for every regular file in the corpus directory runs
# rcn levels -1/-3/-9/-19, rcn --mode fast (BWT byte CM), zstd -1, zstd -19,
# xz -9, brotli -11, lz4 -9, gzip -9 (skipping any not installed) and tabulates
# (name, orig_kb, comp_kb, ratio%, cmp_MBps, dec_MBps).
#
# When rcn is included, prints per-file scorecards and corpus W-L-T for
# ratio (lower wins) and compress speed (higher wins). A combined "both"
# verdict is win only if ratio is win/tie and cmp speed is win.
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

if [[ "${SKIP_RCN:-0}" != "1" && ! -x "$RCN" ]]; then
    echo "building rcn (release)..." >&2
    cargo build --release --bin rcn || { echo "rcn build failed" >&2; exit 1; }
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

now() { python3 -c 'import time; print("%.6f" % time.time())'; }

run_and_time() {
    local out="$1"; shift
    local t0 t1
    t0="$(now)"
    "$@" >"$out" 2>/dev/null
    t1="$(now)"
    awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.6f", b - a }'
}

mbps() {
    awk -v b="$1" -v s="$2" 'BEGIN { if (s > 0) printf "%.1f", (b/1e6)/s; else printf "0.0" }'
}

have() { command -v "$1" >/dev/null 2>&1; }

# Ratio: lower wins. Tie if equal or both < 0.5 (near-zero json).
verdict_ratio() {
    awk -v a="$1" -v b="$2" 'BEGIN {
        if (a == b || (a < 0.5 && b < 0.5)) print "tie"
        else if (a < b) print "win"
        else print "lose"
    }'
}

# Speed: higher wins. Tie if equal.
verdict_speed() {
    awk -v a="$1" -v b="$2" 'BEGIN {
        if (a == b) print "tie"
        else if (a > b) print "win"
        else print "lose"
    }'
}

sanitize() {
    printf '%s' "$1" | tr -c 'A-Za-z0-9' '_'
}

set_metric() {
    local kind="$1" label="$2" value="$3"
    eval "${kind}_$(sanitize "$label")=\"$value\""
}

get_metric() {
    local kind="$1" label="$2"
    eval "printf '%s' \"\${${kind}_$(sanitize "$label"):-}\""
}

clear_metrics() {
    unset RATIO_rcn_1 RATIO_rcn_3 RATIO_rcn_9 RATIO_rcn_19 RATIO_rcn_fast \
          RATIO_zstd_1 RATIO_zstd_19 RATIO_xz_9 RATIO_brotli_11 RATIO_lz4_9 RATIO_gzip_9 \
          CMP_rcn_1 CMP_rcn_3 CMP_rcn_9 CMP_rcn_19 CMP_rcn_fast \
          CMP_zstd_1 CMP_zstd_19 CMP_xz_9 CMP_brotli_11 CMP_lz4_9 CMP_gzip_9 \
          DEC_rcn_1 DEC_rcn_3 DEC_rcn_9 DEC_rcn_19 DEC_rcn_fast \
          DEC_zstd_1 DEC_zstd_19 DEC_xz_9 DEC_brotli_11 DEC_lz4_9 DEC_gzip_9 \
          2>/dev/null || true
}

emit_row() { # name orig_kb comp_kb ratio cmp dec
    printf "%-10s %10.1f %10.1f %8.1f%% %11.1f %11.1f\n" "$1" "$2" "$3" "$4" "$5" "$6"
    set_metric RATIO "$1" "$4"
    set_metric CMP "$1" "$5"
    set_metric DEC "$1" "$6"
}

bump_tally() {
    local axis="$1" mode="$2" peer="$3" v="$4"
    local base key cur
    base="$(sanitize "${axis}_${mode}_${peer}")"
    case "$v" in
        win)  key="TALLY_${base}_w" ;;
        lose) key="TALLY_${base}_l" ;;
        tie)  key="TALLY_${base}_t" ;;
        *) return ;;
    esac
    cur="$(eval "printf '%s' \"\${${key}:-0}\"")"
    eval "${key}=$((cur + 1))"
}

RCN_MODES="rcn-1 rcn-3 rcn-9 rcn-19 rcn-fast"
PEERS="zstd-19 xz-9 brotli-11 gzip-9 lz4-9 zstd-1"

print_file_scorecard() {
    local mode rcn_r rcn_c peer peer_r peer_c vr vc both any=0

    for mode in $RCN_MODES; do
        rcn_r="$(get_metric RATIO "$mode")"
        rcn_c="$(get_metric CMP "$mode")"
        [[ -n "$rcn_r" ]] || continue
        if [[ $any -eq 0 ]]; then
            echo "# $1  (ratio: lower wins; cmp MB/s: higher wins; both=win iff ratio win/tie and cmp win)"
            any=1
        fi
        printf "%-8s" "$mode"
        for peer in $PEERS; do
            peer_r="$(get_metric RATIO "$peer")"
            peer_c="$(get_metric CMP "$peer")"
            [[ -n "$peer_r" && -n "$rcn_c" && -n "$peer_c" ]] || continue
            vr="$(verdict_ratio "$rcn_r" "$peer_r")"
            vc="$(verdict_speed "$rcn_c" "$peer_c")"
            if [[ "$vr" != "lose" && "$vc" == "win" ]]; then
                both="win"
            elif [[ "$vr" == "lose" && "$vc" == "lose" ]]; then
                both="lose"
            else
                both="split"
            fi
            printf " %s=%s/%s/%s" "$peer" "$vr" "$vc" "$both"
            bump_tally ratio "$mode" "$peer" "$vr"
            bump_tally cmp "$mode" "$peer" "$vc"
            bump_tally both "$mode" "$peer" "$both"
        done
        printf "\n"
    done
}

print_axis_summary() {
    local axis="$1" label="$2"
    local mode peer base w l t any=0

    for mode in $RCN_MODES; do
        for peer in $PEERS; do
            base="$(sanitize "${axis}_${mode}_${peer}")"
            eval "w=\${TALLY_${base}_w:-0}"
            eval "l=\${TALLY_${base}_l:-0}"
            eval "t=\${TALLY_${base}_t:-0}"
            if [[ $((w + l + t)) -eq 0 ]]; then
                continue
            fi
            if [[ $any -eq 0 ]]; then
                echo
                echo "corpus W-L-T ($label):"
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
    clear_metrics

    echo "# $name"

    if [[ "${SKIP_RCN:-0}" != "1" && -x "$RCN" ]]; then
        t="$(run_and_time "$WORK/n1.rcn" "$RCN" compress --level 1 "$f" "$WORK/n1.rcn")"
        c="$(wc -c <"$WORK/n1.rcn" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/n1.out" "$RCN" decompress "$WORK/n1.rcn" "$WORK/n1.out")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "rcn-1" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"

        t="$(run_and_time "$WORK/n3.rcn" "$RCN" compress --level 3 "$f" "$WORK/n3.rcn")"
        c="$(wc -c <"$WORK/n3.rcn" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/n3.out" "$RCN" decompress "$WORK/n3.rcn" "$WORK/n3.out")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "rcn-3" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"

        t="$(run_and_time "$WORK/n9.rcn" "$RCN" compress --level 9 "$f" "$WORK/n9.rcn")"
        c="$(wc -c <"$WORK/n9.rcn" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/n9.out" "$RCN" decompress "$WORK/n9.rcn" "$WORK/n9.out")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "rcn-9" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"

        t="$(run_and_time "$WORK/n19.rcn" "$RCN" compress --level 19 "$f" "$WORK/n19.rcn")"
        c="$(wc -c <"$WORK/n19.rcn" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/n19.out" "$RCN" decompress "$WORK/n19.rcn" "$WORK/n19.out")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "rcn-19" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"

        t="$(run_and_time "$WORK/nf.rcn" "$RCN" compress --mode fast "$f" "$WORK/nf.rcn")"
        c="$(wc -c <"$WORK/nf.rcn" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/nf.out" "$RCN" decompress "$WORK/nf.rcn" "$WORK/nf.out")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "rcn-fast" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    if have zstd; then
        t="$(run_and_time "$WORK/z1.zst" zstd -1 -q -f -o "$WORK/z1.zst" "$f")"
        c="$(wc -c <"$WORK/z1.zst" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/z1.out" zstd -q -d -f -o "$WORK/z1.out" "$WORK/z1.zst")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "zstd-1" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"

        t="$(run_and_time "$WORK/z.zst" zstd -19 -q -f -o "$WORK/z.zst" "$f")"
        c="$(wc -c <"$WORK/z.zst" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/z.out" zstd -q -d -f -o "$WORK/z.out" "$WORK/z.zst")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "zstd-19" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    if have xz; then
        t="$(run_and_time "$WORK/x.xz" xz -9 -c "$f")"
        c="$(wc -c <"$WORK/x.xz" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/x.out" xz -d -c "$WORK/x.xz")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "xz-9" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    if have brotli; then
        t="$(run_and_time "$WORK/b.br" brotli -q 11 -c "$f")"
        c="$(wc -c <"$WORK/b.br" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/b.out" brotli -d -c "$WORK/b.br")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "brotli-11" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    if have lz4; then
        t="$(run_and_time "$WORK/l.lz4" lz4 -9 -q -f "$f" "$WORK/l.lz4")"
        c="$(wc -c <"$WORK/l.lz4" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/l.out" lz4 -d -q -f "$WORK/l.lz4" "$WORK/l.out")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "lz4-9" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    if have gzip; then
        t="$(run_and_time "$WORK/g.gz" gzip -9 -c "$f")"
        c="$(wc -c <"$WORK/g.gz" | tr -d '[:space:]')"
        t2="$(run_and_time "$WORK/g.out" gzip -d -c "$WORK/g.gz")"
        r="$(awk -v b="$c" -v o="$orig" 'BEGIN { printf "%.1f", b/o*100 }')"
        emit_row "gzip-9" "$okb" "$(awk -v b="$c" 'BEGIN{printf "%.1f",b/1024}')" "$r" "$(mbps "$orig" "$t")" "$(mbps "$orig" "$t2")"
    fi

    print_file_scorecard "$name"
done

print_axis_summary ratio "ratio only; lower ratio% wins"
print_axis_summary cmp "compress speed; higher cmp MB/s wins"
print_axis_summary both "both; win = ratio win/tie AND cmp win"
