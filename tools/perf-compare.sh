#!/usr/bin/env bash
# Same-machine A/B of the wall-clock benchmarks over two revisions.
#
# Usage: tools/perf-compare.sh <rev-a> <rev-b> [filter]
#
# Both revisions are built and run in one session on one machine into a
# shared target directory, then compared with critcmp. Wall-clock numbers
# taken on different machines or in different sessions are not comparable,
# so no stored history exists and no gate reads these numbers.
#
# Both revisions are built before either runs, and the runs alternate
# a, b, a, b, so neither revision is timed straight after a compilation
# and each is timed once early and once late. The two A-against-B tables
# should agree. The A-against-A and B-against-B tables are the drift
# check: a machine that is not quiet shows a difference there, and a
# comparison taken on it means nothing. Discard the run and close
# whatever was competing for the core.
#
# Requires critcmp (cargo install critcmp).

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 <rev-a> <rev-b> [filter]" >&2
    exit 2
fi

rev_a=$1
rev_b=$2
filter=${3:-}

command -v critcmp >/dev/null || {
    echo "critcmp not found: cargo install critcmp" >&2
    exit 1
}

root=$(git rev-parse --show-toplevel)
work=$root/target/perf
target=$work/target
mkdir -p "$work"
# Baselines saved by an earlier run, under another filter, would appear
# as extra rows in this run's tables.
rm -rf "$target/criterion"

sha_a=$(git rev-parse --verify "$rev_a^{commit}")
sha_b=$(git rev-parse --verify "$rev_b^{commit}")

dir_a=$work/src-a
dir_b=$work/src-b

# A checkout left by a run that could not clean up (killed, or a machine
# restart) is removed here, registration included.
remove() {
    git worktree remove --force "$1" 2>/dev/null || rm -rf "$1"
    git worktree prune
}
cleanup() {
    remove "$dir_a"
    remove "$dir_b"
}
trap cleanup EXIT

checkout() {
    local sha=$1 dir=$2
    remove "$dir"
    git worktree add --detach --quiet "$dir" "$sha"
}

checkout "$sha_a" "$dir_a"
checkout "$sha_b" "$dir_b"

bench() {
    local dir=$1
    shift
    (
        cd "$dir"
        CARGO_TARGET_DIR=$target cargo bench --locked \
            --manifest-path bench/Cargo.toml --bench wall_clock "$@"
    )
}

# One baseline name per run, so the four runs sit side by side in one
# criterion directory for critcmp to read.
run() {
    local dir=$1 baseline=$2
    echo "== $baseline ==" >&2
    bench "$dir" -- --save-baseline "$baseline" $filter
}

bench "$dir_a" --no-run
bench "$dir_b" --no-run

run "$dir_a" a
run "$dir_b" b
run "$dir_a" a-repeat
run "$dir_b" b-repeat

echo
echo "$rev_a ($sha_a) against $rev_b ($sha_b), first pair"
critcmp --target-dir "$target" a b

echo
echo "$rev_a against $rev_b, second pair"
critcmp --target-dir "$target" a-repeat b-repeat

echo
echo "drift: $rev_a against itself, first run against second"
critcmp --target-dir "$target" a a-repeat

echo
echo "drift: $rev_b against itself, first run against second"
critcmp --target-dir "$target" b b-repeat
