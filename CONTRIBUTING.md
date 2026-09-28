# Contributing

## Setup

Rust stable; the minimum supported version is 1.74. Build and test with

    cargo test --locked

## Gates

Every pull request must pass, and CI enforces:

    cargo fmt --all --check
    cargo clippy --all-targets --locked -- -D warnings
    cargo test --locked
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --locked
    cargo deny check advisories licenses bans sources

Each gate also runs with `--features experimental`. The full-size
statistical suite runs nightly; run it locally with

    cargo nextest run --locked --features experimental --run-ignored all

The full-size calibration tests write ranks and samples under
`target/calibration` (override with `CALIBRATION_DIR`);
`benchmarks/calibration/evaluate.R` turns them into rank ECDF difference
and comparison plots, as the nightly `calibration` job does.

## Performance

Benchmarks live in `bench`, off the published crate so
they carry their own toolchain floor. The registry in
`bench/src/lib.rs` holds one workload per shipped model:
a model added there is benchmarked without any other edit.

A pull request that touches a hot path reports numbers. Same-machine A/B,
old revision against new, both built and run in one session:

    tools/perf-compare.sh <rev-a> <rev-b>     # or: just perf-compare
    cargo install critcmp                     # once

The script builds both revisions first and then runs them alternately,
A, B, A, B. The two A-against-B tables should agree. The A-against-A and
B-against-B tables are the drift check: a machine that is not quiet shows
a difference there, and the comparison taken on it means nothing.

A claimed win below roughly 10% cites the instruction-count delta, not
wall-clock alone. Code layout by itself moves wall-clock by around 8%
(Mytkowicz, Diwan, Hauswirth and Sweeney 2009), which is larger than most
wins worth reporting. Instruction counts need valgrind:

    cargo install --locked --version 0.19.4 gungraun-runner
    cargo bench --manifest-path bench/Cargo.toml \
        --bench instructions

CI runs that bench on pull requests touching the core or the registry,
measuring the base revision with the pull request's benchmark code, posts
the base, pull request and change of every benchmark as a comment on the
pull request (`tools/perf-instructions-table.py` renders it), and fails
on a soft-limit breach. It is the only performance job allowed to gate,
because instruction counts are deterministic and a shared runner measures
them as well as a quiet workstation does. There is no wall-clock gate
anywhere: the runners sit at a few per cent of noise with far larger
excursions.

Binding changes report the binding's own numbers the same way, one
machine, old against new, in the pull request body. Four cases in both
languages, run against the core's time on the same designs, with the
absolute overhead beside the ratio:
[benchmarks/bindings/README.md](benchmarks/bindings/README.md).

    cargo run --release --manifest-path bench/Cargo.toml --bin overhead -- \
        designs target/bindings
    cargo run --release --manifest-path bench/Cargo.toml --bin overhead -- \
        run > target/bindings/core.json
    Rscript benchmarks/bindings/overhead.R target/bindings

Nothing there asserts and nothing is stored: the pull request is the
record and its history is the archive.

## Sampling efficiency

Speed and mixing are one measurement, not two: a sampler twice as fast per
sweep that mixes half as well has gained nothing. The suite under
`benchmarks/suite/` scores every shipped model at a fixed set of cells in
minimum effective sample size per second, bulk and tail, with R-hat as a
validity gate and ESS per sweep beside it.

    pip install -r benchmarks/suite/requirements.txt
    python benchmarks/suite/run.py --sizes small
    python benchmarks/suite/compare.py \
        benchmarks/suite/baselines/core-v0.3.0.csv target/suite/scorecard.csv

Every diagnostic comes from one pinned ArviZ; neither the crate nor the
suite estimates an effective sample size for a benchmark. Deltas are
ratios with confidence intervals, never bare point ratios.

A pull request touching the sampler kernel, an outcome model or the suite
runs it in CI against the newest committed baseline, and fails on an
unconverged cell or a separated adverse move in ESS per sweep or held-out
error. Those are the only metrics gated: they are ratios per sweep and per
row, so they do not change with the speed of the runner.

Baselines live in `benchmarks/suite/baselines/`, one per core release.
Refresh one at a release, or before a snapshot regeneration, by
dispatching the CI workflow with the sizes and repetitions wanted and
committing the scorecard artefact it uploads.

Comparisons against other implementations (upstream AddiVortes, dbarts,
BART, stochtree, an XGBoost baseline) live under
`benchmarks/comparators/` and run by hand at releases, never in CI;
[benchmarks/comparators/README.md](benchmarks/comparators/README.md) has
the commands, the pinned environments and the claim policy.

## Reproducibility and snapshots

The reproducibility contract is in the crate-root documentation
([crates/thiessen/src/lib.rs](crates/thiessen/src/lib.rs)). Two snapshot
classes exist and are regenerated by different acts, so one cannot be
refreshed by reflex for the other.

Sampled-value chains live under `crates/thiessen/tests/chains/` as
plain files: fixed-seed draws, bit-exact on `x86_64-unknown-linux-gnu`,
with other targets checking posterior summaries within Monte Carlo
error. `cargo insta` cannot touch them. Regenerate only with

    THIESSEN_UPDATE_CHAINS=1 cargo test --test snapshot

on `x86_64-unknown-linux-gnu` only. A pull request that regenerates a
chain carries a minor version bump and a changelog line "Sampled values
changed" with the reason. During a reshape of the configuration surface
a moved chain blocks the pull request: it means the reshape changed the
maths, which is a bug, not a regeneration.

Config-referencing snapshots (serialised configurations, error message
text) are managed by `insta` and refreshed through `cargo insta review`.
They may change shape when the configuration surface changes and carry
no sampled-values line.

## Stable and experimental

The published method is stable: the models and components of Stone and
Gosling (2025) and of CRAN AddiVortes. Everything else is experimental
and is compiled only with the `experimental` Cargo feature, under
`#[cfg(feature = "experimental")]`, with a row in
[docs/experimental.md](docs/experimental.md). Experimental items meet the
same acceptance criteria as the published models (known answer where one
exists, SBC and Geweke calibration at two sizes, simulation recovery,
snapshot, documentation) and are outside the semver promise.

Commits adding or changing experimental items use the scope
`feat(experimental):` or `fix(experimental):`; their changelog lines start
with "(experimental)". A sampled-value change to an experimental item
takes the "Sampled values changed" line but does not force a minor bump.
Release notes carry the sentence "Options behind the `experimental`
feature are outside the semver promise; see docs/experimental.md".

The bindings build the core without the feature, and every released
artefact ships that way; the opt-in is a build-time one,
`THIESSEN_EXPERIMENTAL=1` for the R package and maturin `build-args` for
the wheel, documented in [docs/experimental.md](docs/experimental.md).
Both binding suites run in either build, so an assertion that depends on
the setting branches or skips on it, and CI carries one opt-in leg per
binding.

An item graduates by the stabilisation rule stated once in the
crate-root documentation (`crates/thiessen/src/lib.rs`, Stability). The
stabilising pull request removes the `cfg` gate, marks the item's row in
`docs/experimental.md` stabilised with the version, and is a minor
version bump; the pull request is the public record, so no per-item
tracking issue exists.

## R package

The package under `r/` links the core as a static library through
extendr and builds offline from vendored sources, as the CRAN policy on
Rust requires. `tools/vendor.sh` packages the core as `cargo package`
would publish it (less its dev-dependencies) into `r/src/rust/core`, runs
`cargo vendor` for the third-party crates and archives those alone as
`r/src/rust/vendor.tar.xz`, and writes `r/inst/AUTHORS` and the core
version in `r/DESCRIPTION`. Run it after any change to the core or to
`r/src/rust/Cargo.toml`, commit every regenerated file including the
archive, then

    R CMD build r
    R CMD check --as-cran thiessen_*.tar.gz

CI builds the tarball once and checks it on Linux, macOS and Windows,
failing on any warning.

## Pull requests

Branch from `dev`; pull requests squash-merge into `dev` with a green
`alls-green` status, the one job of that name, reported by `ci.yml`
alone. Commit messages follow Conventional Commits. The
template's four boxes (tests, docs, changelog, breaking or
sampled-values change) are the whole checklist.

## Releases

Component tags: `core-vX.Y.Z` for the crate, `py-vX.Y.Z` for the Python
package, `r-vX.Y.Z` for the R package. Each binding versions
independently and states the core version it wraps (pyproject metadata
for Python; `Config/thiessen/core-version` in DESCRIPTION for R).

Core releases run through release-plz (`release-plz.toml`, the
release-plz workflow): a release PR carries the version bump and the
changelog section; merging it creates the tag, publishes to crates.io
and opens a GitHub release with the changelog section as notes. A
Python tag triggers the wheel matrix and trusted publishing. R releases
follow the CRAN checklist, with r-universe as the development channel.

Versions follow semver. Patch releases preserve sampled values for a
fixed seed; minor releases may change them and the changelog entry says
"Sampled values changed" with the reason, following the value-stability
policy of rand. The same rule continues past 1.0: a sampled-value
change is a minor bump with the same line, never silent.

Every GitHub release is archived on Zenodo with a DOI once the Zenodo
integration is enabled; the concept DOI badge joins the README at the
first release. `CITATION.cff` carries the software and paper references
and is validated in CI; the R package's `inst/CITATION` and the DOI in
the Python documentation join with their packages.

At each tag, `main` is fast-forwarded to the tagged commit on `dev`
with `git push origin <tag>^{commit}:main`; `main` carries releases
only. `CHANGELOG.md` is keepachangelog; the R package keeps `NEWS.md`;
Python release notes come from the changelog. The first tag of every
component is gated on the final name; the working name is never
published.

## Triage

Issues are triaged by the maintainer; bug reports need the version,
platform, seed and minimal configuration the template asks for.
Questions belong in Discussions.
