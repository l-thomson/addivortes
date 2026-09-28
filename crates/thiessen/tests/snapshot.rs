//! Fixed-seed chain against the stored chain file: bit-exact on the
//! reference target `x86_64-unknown-linux-gnu`; posterior summaries within
//! Monte Carlo error on every target. A per-draw tolerance is not used:
//! once one acceptance flips, the chains diverge entirely.
//!
//! The files under `tests/chains/` are sampled-value snapshots and sit
//! outside `insta`, where `cargo insta review` and `INSTA_UPDATE` cannot
//! touch them. Regeneration is its own act,
//! `THIESSEN_UPDATE_CHAINS=1 cargo test --test snapshot` on the reference
//! target, and carries a minor bump with the changelog line "Sampled
//! values changed". During a reshape a moved chain is a bug, not a
//! regeneration.

mod common;

use common::{
    categorical_fixture, fixture, heteroscedastic_fixture, larger_fixture, probit_fixture,
    spherical_fixture, SEED,
};
use thiessen::{Data, Fitted};

/// Rows of the fixture at which f(x) is snapshotted.
const POINTS: [usize; 3] = [0, 17, 33];

const CHAINS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/chains");

fn chain_path(name: &str) -> String {
    format!("{CHAINS}/{name}.txt")
}

/// The rendered chain against the stored file. With
/// `THIESSEN_UPDATE_CHAINS` set the file is rewritten first; run that on
/// the reference target only, and only as a sampled-values change.
fn assert_chain(name: &str, rendered: &str) {
    let path = chain_path(name);
    if std::env::var_os("THIESSEN_UPDATE_CHAINS").is_some() {
        std::fs::write(&path, rendered).expect("write the chain file");
    }
    let stored = std::fs::read_to_string(&path)
        .expect("stored chain; regenerate on the reference target first");
    assert_eq!(
        rendered, stored,
        "the {name} chain moved: a sampled-values change (minor bump, \
         changelog line), never a reflex regeneration"
    );
}

fn points(x: &Data) -> Data {
    let rows: Vec<&[f64]> = POINTS.iter().map(|&i| x.row(i)).collect();
    Data::from_rows(&rows).unwrap()
}

/// One line per draw: sigma then f(x) at each point, `{:?}` so the text
/// round-trips to the exact bits.
fn render(model: &Fitted, x: &Data) -> String {
    let sigma = model.sigma();
    let draws = model.predict_draws(&points(x)).unwrap();
    let mut out = String::from("sigma f(x0) f(x17) f(x33)\n");
    for (s, f) in sigma.iter().zip(&draws) {
        out.push_str(&format!("{:?} {:?} {:?} {:?}\n", s, f[0], f[1], f[2]));
    }
    out
}

/// One line per draw: the latent mean c + f(x) at each point.
fn render_probit(model: &Fitted, x: &Data) -> String {
    let draws = model.predict_latent(&points(x)).unwrap();
    let mut out = String::from("z(x0) z(x17) z(x33)\n");
    for f in &draws {
        out.push_str(&format!("{:?} {:?} {:?}\n", f[0], f[1], f[2]));
    }
    out
}

/// One line per draw: f(x) then s^2(x) at each point.
fn render_heteroscedastic(model: &Fitted, x: &Data) -> String {
    let draws = model.predict_draws(&points(x)).unwrap();
    let variances = model.predict_variance(&points(x)).unwrap();
    let mut out = String::from("f(x0) f(x17) f(x33) s2(x0) s2(x17) s2(x33)\n");
    for (f, v) in draws.iter().zip(&variances) {
        out.push_str(&format!(
            "{:?} {:?} {:?} {:?} {:?} {:?}\n",
            f[0], f[1], f[2], v[0], v[1], v[2]
        ));
    }
    out
}

#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
#[test]
fn reference_target_chain_is_bit_exact() {
    let (config, x, y) = fixture();
    let model = thiessen::fit(&config, &x, &y, SEED).unwrap();
    assert_chain("gaussian", &render(&model, &x));
}

#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
#[test]
fn reference_target_probit_chain_is_bit_exact() {
    let (config, x, y) = probit_fixture();
    let model = thiessen::fit(&config, &x, &y, SEED).unwrap();
    assert_chain("probit", &render_probit(&model, &x));
}

#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
#[test]
fn reference_target_heteroscedastic_chain_is_bit_exact() {
    let (config, x, y) = heteroscedastic_fixture();
    let model = thiessen::fit(&config, &x, &y, SEED).unwrap();
    assert_chain("heteroscedastic", &render_heteroscedastic(&model, &x));
}

#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
#[test]
fn reference_target_spherical_chain_is_bit_exact() {
    let (config, x, y) = spherical_fixture();
    let model = thiessen::fit(&config, &x, &y, SEED).unwrap();
    assert_chain("spherical", &render(&model, &x));
}

#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
#[test]
fn reference_target_categorical_chain_is_bit_exact() {
    let (config, x, y) = categorical_fixture();
    let model = thiessen::fit(&config, &x, &y, SEED).unwrap();
    assert_chain("categorical", &render(&model, &x));
}

#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
#[test]
fn reference_target_larger_chain_is_bit_exact() {
    let (config, x, y) = larger_fixture();
    let model = thiessen::fit(&config, &x, &y, SEED).unwrap();
    assert_chain("larger", &render(&model, &x));
}

#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
#[test]
fn reference_target_two_chain_fit_is_bit_exact() {
    let (config, x, y) = fixture();
    let (model, _) = thiessen::fit_chains_with_threads(&config, &x, &y, SEED, 2, 2).unwrap();
    assert_chain("two_chains", &render(&model, &x));
}

fn stored_columns(name: &str, n_columns: usize) -> Vec<Vec<f64>> {
    let stored = std::fs::read_to_string(chain_path(name))
        .expect("stored chain; regenerate on the reference target first");
    parse_columns(&stored, n_columns)
}

#[test]
fn posterior_summaries_match_the_stored_chain() {
    let columns = stored_columns("gaussian", 4);
    let (config, x, y) = fixture();
    let model = thiessen::fit(&config, &x, &y, SEED).unwrap();
    let rendered = render(&model, &x);
    let live = parse_columns(rendered.split_once('\n').unwrap().1, 4);

    for (a, b) in live.iter().zip(&columns) {
        assert_close_mean_and_sd(a, b);
    }
    for p in [0.05, 0.5, 0.95] {
        assert_close_quantile(&live[0], &columns[0], p);
    }
}

#[test]
fn probit_posterior_summaries_match_the_stored_chain() {
    let columns = stored_columns("probit", 3);
    let (config, x, y) = probit_fixture();
    let model = thiessen::fit(&config, &x, &y, SEED).unwrap();
    let rendered = render_probit(&model, &x);
    let live = parse_columns(rendered.split_once('\n').unwrap().1, 3);
    for (a, b) in live.iter().zip(&columns) {
        assert_close_mean_and_sd(a, b);
    }
}

#[test]
fn spherical_posterior_summaries_match_the_stored_chain() {
    let columns = stored_columns("spherical", 4);
    let (config, x, y) = spherical_fixture();
    let model = thiessen::fit(&config, &x, &y, SEED).unwrap();
    let rendered = render(&model, &x);
    let live = parse_columns(rendered.split_once('\n').unwrap().1, 4);
    for (a, b) in live.iter().zip(&columns) {
        assert_close_mean_and_sd(a, b);
    }
}

#[test]
fn categorical_posterior_summaries_match_the_stored_chain() {
    let columns = stored_columns("categorical", 4);
    let (config, x, y) = categorical_fixture();
    let model = thiessen::fit(&config, &x, &y, SEED).unwrap();
    let rendered = render(&model, &x);
    let live = parse_columns(rendered.split_once('\n').unwrap().1, 4);
    for (a, b) in live.iter().zip(&columns) {
        assert_close_mean_and_sd(a, b);
    }
}

#[test]
fn heteroscedastic_posterior_summaries_match_the_stored_chain() {
    let columns = stored_columns("heteroscedastic", 6);
    let (config, x, y) = heteroscedastic_fixture();
    let model = thiessen::fit(&config, &x, &y, SEED).unwrap();
    let rendered = render_heteroscedastic(&model, &x);
    let live = parse_columns(rendered.split_once('\n').unwrap().1, 6);
    for (a, b) in live.iter().zip(&columns) {
        assert_close_mean_and_sd(a, b);
    }
}

#[test]
fn larger_posterior_summaries_match_the_stored_chain() {
    let columns = stored_columns("larger", 4);
    let (config, x, y) = larger_fixture();
    let model = thiessen::fit(&config, &x, &y, SEED).unwrap();
    let rendered = render(&model, &x);
    let live = parse_columns(rendered.split_once('\n').unwrap().1, 4);
    for (a, b) in live.iter().zip(&columns) {
        assert_close_mean_and_sd(a, b);
    }
}

#[test]
fn two_chain_posterior_summaries_match_the_stored_chain() {
    let columns = stored_columns("two_chains", 4);
    let (config, x, y) = fixture();
    let (model, _) = thiessen::fit_chains_with_threads(&config, &x, &y, SEED, 2, 2).unwrap();
    let rendered = render(&model, &x);
    let live = parse_columns(rendered.split_once('\n').unwrap().1, 4);
    for (a, b) in live.iter().zip(&columns) {
        assert_close_mean_and_sd(a, b);
    }
}

fn parse_columns(body: &str, n_columns: usize) -> Vec<Vec<f64>> {
    let mut columns = vec![Vec::new(); n_columns];
    for line in body.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        for (c, v) in line.split_whitespace().enumerate() {
            columns[c].push(v.parse::<f64>().unwrap());
        }
    }
    columns
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn sd(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/// Type 7 quantile of a sorted sample.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let h = p * (sorted.len() - 1) as f64;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// k = 4 on the combined Monte Carlo standard errors, as in the comparison
/// fixtures.
const K: f64 = 4.0;

fn assert_close_mean_and_sd(a: &[f64], b: &[f64]) {
    let n = a.len() as f64;
    let tol_mean = K * (sd(a) + sd(b)) / n.sqrt();
    assert!(
        (mean(a) - mean(b)).abs() <= tol_mean + 1e-12,
        "means {} and {} differ beyond {tol_mean}",
        mean(a),
        mean(b)
    );
    let tol_sd = K * (sd(a) + sd(b)) / (2.0 * (n - 1.0)).sqrt();
    assert!(
        (sd(a) - sd(b)).abs() <= tol_sd + 1e-12,
        "sds {} and {} differ beyond {tol_sd}",
        sd(a),
        sd(b)
    );
}

/// SE(q_p) = sqrt(p(1-p)/n) / f(q_p), f estimated by a central difference
/// of neighbouring quantiles.
fn quantile_se(sorted: &[f64], p: f64) -> f64 {
    let lo = (p - 0.05).max(0.005);
    let hi = (p + 0.05).min(0.995);
    let spread = quantile(sorted, hi) - quantile(sorted, lo);
    let density = (hi - lo) / spread;
    (p * (1.0 - p) / sorted.len() as f64).sqrt() / density
}

fn assert_close_quantile(a: &[f64], b: &[f64], p: f64) {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let tol = K * (quantile_se(&a, p) + quantile_se(&b, p));
    let diff = (quantile(&a, p) - quantile(&b, p)).abs();
    assert!(
        diff <= tol + 1e-12,
        "quantile {p}: {} and {} differ beyond {tol}",
        quantile(&a, p),
        quantile(&b, p)
    );
}
