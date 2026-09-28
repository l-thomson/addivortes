//! Test support shared across the integration tests: the fixed-seed
//! fixture of the determinism and snapshot tests, and a self-contained
//! generator for reference simulators, which must share nothing with the
//! crate's RNG.

use thiessen::{Config, Data};

#[allow(dead_code)]
pub const SEED: u64 = 7;

/// splitmix64 (Steele, Lea and Flood 2014) with Box-Muller normals and
/// inversion sampling for the counts.
pub struct TestRng(pub u64);

#[allow(dead_code)]
impl TestRng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    pub fn normal(&mut self) -> f64 {
        let (u1, u2) = (1.0 - self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Gamma(shape, 1) for shape >= 1 by Marsaglia and Tsang (2000).
    pub fn gamma(&mut self, shape: f64) -> f64 {
        assert!(shape >= 1.0);
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let z = self.normal();
            let v = (1.0 + c * z).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.uniform();
            if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    /// Gamma(shape, 1) for any positive shape: the boost
    /// Gamma(shape + 1) U^(1 / shape) below 1.
    pub fn gamma_any(&mut self, shape: f64) -> f64 {
        assert!(shape > 0.0);
        if shape >= 1.0 {
            return self.gamma(shape);
        }
        self.gamma(shape + 1.0) * self.uniform().powf(1.0 / shape)
    }

    pub fn poisson(&mut self, lambda: f64) -> usize {
        let target = self.uniform();
        let mut pmf = (-lambda).exp();
        let mut cdf = pmf;
        let mut k = 0;
        while cdf < target && k < 200 {
            k += 1;
            pmf *= lambda / k as f64;
            cdf += pmf;
        }
        k
    }
}

/// The fixed-seed fixture of the probit model: the Gaussian fixture's
/// design with the response thresholded at its median, so both labels
/// occur.
#[allow(dead_code)]
pub fn probit_fixture() -> (Config, Data, Vec<f64>) {
    let (config, x, y) = fixture();
    let mut sorted = y.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let labels = y.iter().map(|&v| f64::from(v >= median)).collect();
    (config.with_outcome(thiessen::Outcome::probit()), x, labels)
}

/// The fixed-seed fixture of the heteroscedastic model: the Gaussian
/// fixture's design and mean with the noise scaled by 0.2 + 2 x_1, and
/// m' = 5 variance tessellations.
#[allow(dead_code)]
pub fn heteroscedastic_fixture() -> (Config, Data, Vec<f64>) {
    let (config, x, y) = fixture();
    let y = y
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let noise = 0.3 * (((i * 29) % 17) as f64 / 16.0 - 0.5);
            v - noise + noise * (0.2 + 2.0 * x.row(i)[0])
        })
        .collect();
    (config.with_m_var(5), x, y)
}

/// The fixed-seed fixture on one sphere: the Gaussian fixture's rows as
/// latitude in [-pi / 4, pi / 4] and longitude in [-pi, pi), the response
/// a smooth function of position with the fixture's noise.
#[allow(dead_code)]
pub fn spherical_fixture() -> (Config, Data, Vec<f64>) {
    use std::f64::consts::PI;
    let (config, x, _) = fixture();
    let rows: Vec<[f64; 2]> = (0..x.n_rows())
        .map(|i| {
            let r = x.row(i);
            [(r[0] - 0.5) * PI / 2.0, (r[1] - 0.5) * 2.0 * PI]
        })
        .collect();
    let y = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            r[0].cos() * r[1].cos() + 0.5 * r[0].sin() + 0.3 * (((i * 29) % 17) as f64 / 16.0 - 0.5)
        })
        .collect();
    (
        config.with_metric(vec![
            thiessen::Metric::Spherical { sphere: 0 },
            thiessen::Metric::Spherical { sphere: 0 },
        ]),
        Data::from_rows(&rows).unwrap(),
        y,
    )
}

/// The fixed-seed fixture with a categorical column: the Gaussian
/// fixture's first column and a four-level code in place of the second,
/// the response shifted by level.
#[allow(dead_code)]
pub fn categorical_fixture() -> (Config, Data, Vec<f64>) {
    let (config, x, y) = fixture();
    let rows: Vec<[f64; 2]> = (0..x.n_rows())
        .map(|i| [x.row(i)[0], ((i * 7) % 4) as f64])
        .collect();
    let y = y.iter().zip(&rows).map(|(&v, r)| v + 0.4 * r[1]).collect();
    (
        config.with_metric(vec![
            thiessen::Metric::Euclidean,
            thiessen::Metric::Categorical,
        ]),
        Data::from_rows(&rows).unwrap(),
        y,
    )
}

#[allow(dead_code)]
pub fn fixture() -> (Config, Data, Vec<f64>) {
    let n = 48;
    let rows: Vec<[f64; 2]> = (0..n)
        .map(|i| [i as f64 / (n - 1) as f64, ((i * 37) % n) as f64 / n as f64])
        .collect();
    let y: Vec<f64> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let f = 3.0 * (r[0] - 0.4) * (r[0] - 0.4) + 0.5 * r[1];
            f + 0.3 * (((i * 29) % 17) as f64 / 16.0 - 0.5)
        })
        .collect();
    let x = Data::from_rows(&rows).unwrap();
    let config = Config::new().with_m(15).with_burn_in(50).with_draws(60);
    (config, x, y)
}

/// A larger fixed-seed Gaussian fixture: 300 rows of three covariates on
/// a permuted grid and m = 5, so each tessellation holds about six cells
/// against the three of [`fixture`].
#[allow(dead_code)]
pub fn larger_fixture() -> (Config, Data, Vec<f64>) {
    let n = 300;
    let rows: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let column = |j: usize| ((i * (37 + 12 * j) + 5 * j) % n) as f64 / n as f64;
            [column(0), column(1), column(2)]
        })
        .collect();
    let y: Vec<f64> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let f = (6.0 * r[0]).sin() + 2.0 * (r[1] - 0.5) * (r[1] - 0.5) + r[0] * r[2];
            f + 0.3 * (((i * 29) % 17) as f64 / 16.0 - 0.5)
        })
        .collect();
    let x = Data::from_rows(&rows).unwrap();
    let config = Config::new().with_m(5).with_burn_in(300).with_draws(60);
    (config, x, y)
}

/// One configuration naming each gated item of `docs/experimental.md`,
/// with the name the feature error gives it. A build without the feature
/// reports each from `Config::validate` as `RequiresFeature`; a build
/// with the feature accepts each. Shared between `experimental.rs` and
/// `gated_surface.rs` so the two builds cannot disagree on the surface.
///
/// Fifteen entries, one per row of the table, plus `geometry.precision`,
/// which the Mahalanobis row requires and which has no published value.
#[allow(dead_code)]
pub const GATED_CONFIGS: &[(&str, &str)] = &[
    (r#"{"outcome": {"tobit": {"lower": 0.0}}}"#, "tobit"),
    (r#"{"outcome": {"aft": {}}}"#, "aft"),
    (
        r#"{"outcome": {"interval_censored": {}}}"#,
        "interval_censored",
    ),
    (r#"{"outcome": {"ordinal": {"categories": 3}}}"#, "ordinal"),
    (r#"{"outcome": {"student_t": {"df": 4.0}}}"#, "student_t"),
    (r#"{"outcome": {"laplace": {}}}"#, "laplace"),
    (
        r#"{"mean_params": {"geometry": {"metric": [{"minkowski": {"p": 1.5}}]}}}"#,
        "minkowski",
    ),
    (
        r#"{"mean_params": {"geometry": {"metric": [{"manhattan": {}}]}}}"#,
        "manhattan",
    ),
    (
        r#"{"mean_params": {"geometry": {"metric": [{"cosine": {}}]}}}"#,
        "cosine",
    ),
    (
        r#"{"mean_params": {"geometry": {"metric": [{"gower": {"kind": "numeric"}}]}}}"#,
        "gower",
    ),
    (
        r#"{"mean_params": {"geometry": {"metric": ["mahalanobis"],
            "precision": [1.0, 0.0, 0.0, 1.0]}}}"#,
        "mahalanobis",
    ),
    (
        r#"{"mean_params": {"geometry": {"precision": [1.0, 0.0, 0.0, 1.0]}}}"#,
        "precision",
    ),
    (
        r#"{"mean_params": {"structure": {"inclusion": {"weighted": {"weights": [1.0, 1.0]}}}}}"#,
        "weighted",
    ),
    (
        r#"{"mean_params": {"structure": {"inclusion": {"dart": {}}}}}"#,
        "dart",
    ),
    (
        r#"{"mean_params": {"geometry": {"membership": {"soft": {}}}}}"#,
        "soft",
    ),
    (
        r#"{"mean_params": {"cell": {"basis": "linear"}}}"#,
        "linear",
    ),
];

/// The published default of each gated field: what the model does when
/// the field is absent, so a build without the feature accepts it. The
/// second entry is the field name, which must stay out of the
/// serialised form.
#[allow(dead_code)]
pub const PUBLISHED_DEFAULTS: &[(&str, &str)] = &[
    (
        r#"{"mean_params": {"geometry": {"membership": "hard"}}}"#,
        "membership",
    ),
    (
        r#"{"mean_params": {"structure": {"inclusion": "uniform"}}}"#,
        "inclusion",
    ),
    (
        r#"{"mean_params": {"cell": {"basis": "constant"}}}"#,
        "basis",
    ),
];
