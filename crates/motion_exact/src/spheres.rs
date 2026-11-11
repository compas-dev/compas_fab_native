//! Sphere models of collision meshes: the spheres the core checks instead of the meshes.
//!
//! Every body (a link, a tool, a carried workpiece) gets two sets:
//!
//! * **self spheres**, inscribed in the mesh, for self-collision (covering spheres of
//!   neighbouring links overlap permanently);
//! * **obstacle spheres**, for checks against the scene: by default the inscribed spheres
//!   grown by a padding, or k-means clusters of surface samples, each enclosed by a sphere
//!   (conservative, but with many false alarms; see SPIKE.md).
//!
//! Inscribed spheres need a closed mesh. Bodies without one fall back to covering spheres
//! for both sets. `SphereSets::miss` says how far the surface reaches outside the self
//! spheres; the planner checks bodies that miss a lot on their meshes instead.
//!
//! Randomness comes from a seeded `Rng`, so a body gets the same spheres on every run and
//! platform. Distances and inside tests are exact (f64, every triangle): collision meshes
//! are small, and fits are cached on the Python side.

use std::collections::HashMap;

use motion_core::sampling::Rng;
use rayon::prelude::*;

pub type Point = [f64; 3];
pub type Triangle = [Point; 3];
/// Centre and radius.
pub type Sphere = (Point, f64);

/// What the obstacle spheres are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObstacleSpheres {
    /// The inscribed spheres grown by `FitOptions::obstacle_padding`.
    Inscribed,
    /// k-means clusters of surface samples, each enclosed by a sphere.
    Covering,
}

#[derive(Clone, Debug)]
pub struct FitOptions {
    /// Target radius of covering spheres (metres).
    pub sphere_radius: f64,
    /// At most this many covering spheres.
    pub max_spheres: usize,
    /// Base radius of inscribed sphere candidates (metres).
    pub self_sphere_radius: f64,
    /// At most this many inscribed spheres.
    pub max_self_spheres: usize,
    /// A surface sample counts as covered within this distance of an inscribed sphere.
    pub tolerance: f64,
    pub obstacle_spheres: ObstacleSpheres,
    /// Added to inscribed spheres to make obstacle spheres.
    pub obstacle_padding: f64,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            sphere_radius: 0.03,
            max_spheres: 16,
            self_sphere_radius: 0.03,
            max_self_spheres: 16,
            tolerance: 0.01,
            obstacle_spheres: ObstacleSpheres::Inscribed,
            obstacle_padding: 0.02,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SphereSets {
    pub obstacle: Vec<Sphere>,
    pub self_spheres: Vec<Sphere>,
    /// How far the surface reaches outside the self spheres: the 95th percentile, over
    /// surface samples, of the distance to the nearest sphere (0 inside one). Infinite
    /// without spheres.
    pub miss: f64,
}

/// Surface samples for covering spheres and for `miss`.
const SURFACE_SAMPLES: usize = 2000;
/// Surface samples an inscribed set must cover.
const COVER_SAMPLES: usize = 3000;
/// Surface samples pushed inward as inscribed sphere candidates (per radius).
const CANDIDATES: usize = 600;
/// Candidate radii, as multiples of `FitOptions::self_sphere_radius`.
const RADIUS_FACTORS: [f64; 6] = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0];
/// Candidates closer than this to the surface are dropped (metres).
const MIN_DEPTH: f64 = 0.002;
/// Vertices closer than this are merged (metres).
const MERGE_TOLERANCE: f64 = 1e-8;

/// Both sphere sets of one body, given as triangles.
pub fn fit_sphere_sets(triangles: &[Triangle], seed: u64, options: &FitOptions) -> SphereSets {
    let mut rng = Rng::new(seed);
    let covering = |rng: &mut Rng| {
        let points = surface_points(triangles, SURFACE_SAMPLES, rng);
        covering_spheres(&points, options.sphere_radius, options.max_spheres, rng)
    };
    let solid = Solid::from_triangles(triangles);
    let inscribed = solid
        .as_ref()
        .map(|solid| {
            let radii: Vec<f64> = RADIUS_FACTORS
                .iter()
                .map(|f| f * options.self_sphere_radius)
                .collect();
            solid.inscribed_spheres(
                options.tolerance,
                options.max_self_spheres,
                &radii,
                &mut rng,
            )
        })
        .unwrap_or_default();
    let (obstacle, self_spheres) = if inscribed.is_empty() {
        let c = covering(&mut rng);
        (c.clone(), c)
    } else if options.obstacle_spheres == ObstacleSpheres::Covering {
        (covering(&mut rng), inscribed)
    } else {
        let padded = inscribed
            .iter()
            .map(|&(c, r)| (c, r + options.obstacle_padding))
            .collect();
        (padded, inscribed)
    };
    let miss = fit_miss(triangles, &self_spheres, &mut rng);
    SphereSets {
        obstacle,
        self_spheres,
        miss,
    }
}

/// How far the surface reaches outside `spheres`, see `SphereSets::miss`.
pub fn fit_miss(triangles: &[Triangle], spheres: &[Sphere], rng: &mut Rng) -> f64 {
    if spheres.is_empty() {
        return f64::INFINITY;
    }
    let points = surface_points(triangles, SURFACE_SAMPLES, rng);
    if points.is_empty() {
        return 0.0;
    }
    let mut outside: Vec<f64> = points
        .par_iter()
        .map(|p| {
            spheres
                .iter()
                .map(|(c, r)| dist(p, c) - r)
                .fold(f64::INFINITY, f64::min)
                .max(0.0)
        })
        .collect();
    percentile(&mut outside, 95.0)
}

/// `q`-th percentile with linear interpolation between neighbours (numpy's default).
fn percentile(values: &mut [f64], q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let pos = q / 100.0 * (values.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    values[lo] + (values[hi] - values[lo]) * (pos - lo as f64)
}

/// `n` area-weighted surface samples plus every vertex, so extremes are always covered.
pub fn surface_points(triangles: &[Triangle], n: usize, rng: &mut Rng) -> Vec<Point> {
    let vertices = triangles.iter().flatten().copied();
    let Some(sampler) = AreaSampler::new(triangles.iter().map(area)) else {
        return vertices.collect();
    };
    let mut points: Vec<Point> = (0..n)
        .map(|_| point_in(&triangles[sampler.pick(rng)], rng))
        .collect();
    points.extend(vertices);
    points
}

/// k-means clusters of `points`, each enclosed by a sphere around its centroid. Enough
/// clusters for spheres of about `target_radius` across the points' extent, at most
/// `max_spheres`.
pub fn covering_spheres(
    points: &[Point],
    target_radius: f64,
    max_spheres: usize,
    rng: &mut Rng,
) -> Vec<Sphere> {
    if points.is_empty() {
        return Vec::new();
    }
    let (lo, hi) = bounds(points);
    let extent = (0..3).map(|i| hi[i] - lo[i]).fold(0.0, f64::max);
    let k = ((extent / (2.0 * target_radius)).ceil() as usize)
        .clamp(1, max_spheres.max(1))
        .min(points.len());
    let mut centers: Vec<Point> = choose(points.len(), k, rng)
        .into_iter()
        .map(|i| points[i])
        .collect();
    for _ in 0..50 {
        let labels = nearest(points, &centers);
        let mut sums = vec![([0.0; 3], 0usize); k];
        for (p, &l) in points.iter().zip(&labels) {
            sums[l].0 = add(&sums[l].0, p);
            sums[l].1 += 1;
        }
        let new: Vec<Point> = sums
            .iter()
            .zip(&centers)
            .map(|((s, n), c)| {
                if *n > 0 {
                    scale(s, 1.0 / *n as f64)
                } else {
                    *c
                }
            })
            .collect();
        let converged = new
            .iter()
            .zip(&centers)
            .all(|(a, b)| (0..3).all(|i| (a[i] - b[i]).abs() <= 1e-8 + 1e-5 * b[i].abs()));
        centers = new;
        if converged {
            break;
        }
    }
    let labels = nearest(points, &centers);
    let mut radii = vec![None::<f64>; k];
    for (p, &l) in points.iter().zip(&labels) {
        let d = dist(p, &centers[l]);
        radii[l] = Some(radii[l].map_or(d, |r: f64| r.max(d)));
    }
    centers
        .into_iter()
        .zip(radii)
        .filter_map(|(c, r)| r.map(|r| (c, r)))
        .collect()
}

/// Index of the nearest centre for each point.
fn nearest(points: &[Point], centers: &[Point]) -> Vec<usize> {
    points
        .par_iter()
        .map(|p| {
            let mut best = (0, f64::INFINITY);
            for (i, c) in centers.iter().enumerate() {
                let d = dist2(p, c);
                if d < best.1 {
                    best = (i, d);
                }
            }
            best.0
        })
        .collect()
}

/// One or more closed, consistently outward-oriented triangle meshes: the interior
/// inscribed spheres live in.
pub struct Solid {
    /// Outward triangles of every part.
    triangles: Vec<Triangle>,
}

impl Solid {
    /// The solid bounded by `triangles`, or None if they do not bound one.
    ///
    /// Vertices are merged and faces that collapse to a line or point are dropped. Faces
    /// belong to the same part when they share an edge no other face uses, so closed parts
    /// that touch along an edge (four faces on it) stay separate. Stray slivers (parts
    /// with fewer than 4 faces) are dropped. Every remaining part must be closed (each of
    /// its edges shared by exactly two of its faces) and orientable; its faces are turned
    /// to point outward. None for flat or degenerate bodies, which have no interior.
    pub fn from_triangles(triangles: &[Triangle]) -> Option<Solid> {
        let (vertices, faces) = merge_vertices(triangles);
        if faces.is_empty() {
            return None;
        }
        let (lo, hi) = bounds(&vertices);
        if (0..3).any(|i| hi[i] - lo[i] < 1e-6) {
            return None; // flat: no interior to inscribe spheres in
        }
        let mut sets = DisjointSets::new(faces.len());
        for shared in edge_faces(&faces, 0..faces.len()).values() {
            if let [a, b] = shared[..] {
                sets.union(a.0, b.0);
            }
        }
        let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
        for f in 0..faces.len() {
            components.entry(sets.find(f)).or_default().push(f);
        }
        // In order of their first face, so the result does not depend on hashing.
        let mut components: Vec<Vec<usize>> = components.into_values().collect();
        components.sort_by_key(|members| members[0]);
        let mut triangles = Vec::new();
        for members in components {
            if members.len() < 4 {
                continue; // a sliver: a couple of stray faces
            }
            let oriented = orient_outward(&faces, &vertices, &members)?;
            triangles.extend(oriented.iter().map(|f| f.map(|v| vertices[v])));
        }
        (!triangles.is_empty()).then_some(Solid { triangles })
    }

    /// Distance to the surface: positive inside, negative outside.
    ///
    /// Inside means a winding number above one half, so a point where parts overlap is
    /// inside, and the distance is to the nearest triangle of any part (where parts
    /// overlap, that can be less than the depth in the union: conservative).
    pub fn signed_distance(&self, p: &Point) -> f64 {
        let (mut nearest, mut winding) = (f64::INFINITY, 0.0);
        for t in &self.triangles {
            nearest = nearest.min(dist2(p, &closest_point(t, p)));
            winding += solid_angle(t, p);
        }
        let d = nearest.sqrt();
        if winding / (4.0 * std::f64::consts::PI) > 0.5 {
            d
        } else {
            -d
        }
    }

    /// Greedy surface cover with spheres inside the solid.
    ///
    /// Candidate centres are surface samples pushed inward by each of `radii`; each gets
    /// the largest radius that stays inside (its depth, at most the largest of `radii`).
    /// A surface sample is covered within `tolerance` of a sphere; candidates are picked
    /// greedily by how many samples they newly cover.
    pub fn inscribed_spheres(
        &self,
        tolerance: f64,
        max_spheres: usize,
        radii: &[f64],
        rng: &mut Rng,
    ) -> Vec<Sphere> {
        let Some(sampler) = AreaSampler::new(self.triangles.iter().map(area)) else {
            return Vec::new();
        };
        let samples: Vec<(Point, Point)> = (0..COVER_SAMPLES)
            .map(|_| {
                let t = &self.triangles[sampler.pick(rng)];
                (point_in(t, rng), unit_normal(t))
            })
            .collect();
        let seeds = choose(samples.len(), CANDIDATES.min(samples.len()), rng);
        let samples = &samples;
        let candidates: Vec<Point> = radii
            .iter()
            .flat_map(|r| {
                seeds
                    .iter()
                    .map(move |&i| sub(&samples[i].0, &scale(&samples[i].1, *r)))
            })
            .collect();
        let max_radius = radii.iter().copied().fold(0.0, f64::max);
        let spheres: Vec<Sphere> = candidates
            .par_iter()
            .map(|c| (*c, self.signed_distance(c)))
            .filter(|(_, depth)| *depth > MIN_DEPTH)
            .map(|(c, depth)| (c, depth.min(max_radius)))
            .collect();
        if spheres.is_empty() {
            return Vec::new();
        }
        // covers[s]: bitset of the samples sphere s covers.
        let words = samples.len().div_ceil(64);
        let covers: Vec<Vec<u64>> = spheres
            .par_iter()
            .map(|(c, r)| {
                let mut bits = vec![0u64; words];
                for (i, (p, _)) in samples.iter().enumerate() {
                    if dist(p, c) - r <= tolerance {
                        bits[i / 64] |= 1 << (i % 64);
                    }
                }
                bits
            })
            .collect();
        let mut uncovered = vec![u64::MAX; words];
        if !samples.len().is_multiple_of(64) {
            uncovered[words - 1] = (1u64 << (samples.len() % 64)) - 1;
        }
        let mut chosen = Vec::new();
        while chosen.len() < max_spheres && uncovered.iter().any(|w| *w != 0) {
            let gain = |s: &Vec<u64>| -> u32 {
                s.iter()
                    .zip(&uncovered)
                    .map(|(a, b)| (a & b).count_ones())
                    .sum()
            };
            // The first of the best, like numpy's argmax.
            let (best, most) = covers.iter().enumerate().fold((0, 0), |(bi, bg), (i, s)| {
                let g = gain(s);
                if g > bg {
                    (i, g)
                } else {
                    (bi, bg)
                }
            });
            if most == 0 {
                break;
            }
            for (u, c) in uncovered.iter_mut().zip(&covers[best]) {
                *u &= !c;
            }
            chosen.push(spheres[best]);
        }
        chosen
    }
}

/// Merged vertices and faces of distinct vertices.
fn merge_vertices(triangles: &[Triangle]) -> (Vec<Point>, Vec<[usize; 3]>) {
    let mut index: HashMap<[i64; 3], usize> = HashMap::new();
    let mut vertices = Vec::new();
    let mut faces = Vec::with_capacity(triangles.len());
    for t in triangles {
        if t.iter().flatten().any(|x| !x.is_finite()) {
            continue;
        }
        let f = t.map(|p| {
            let key = p.map(|x| (x / MERGE_TOLERANCE).round() as i64);
            *index.entry(key).or_insert_with(|| {
                vertices.push(p);
                vertices.len() - 1
            })
        });
        if f[0] != f[1] && f[1] != f[2] && f[2] != f[0] {
            faces.push(f);
        }
    }
    (vertices, faces)
}

/// For each undirected edge of the given faces, the faces that use it and whether they
/// run it low to high.
fn edge_faces(
    faces: &[[usize; 3]],
    members: impl IntoIterator<Item = usize>,
) -> HashMap<(usize, usize), Vec<(usize, bool)>> {
    let mut edges: HashMap<(usize, usize), Vec<(usize, bool)>> = HashMap::new();
    for f in members {
        let face = faces[f];
        for k in 0..3 {
            let (a, b) = (face[k], face[(k + 1) % 3]);
            edges
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push((f, a < b));
        }
    }
    edges
}

/// The faces of one connected part, turned consistently and outward; None if the part is
/// not closed, not orientable, or has no volume.
fn orient_outward(
    faces: &[[usize; 3]],
    vertices: &[Point],
    members: &[usize],
) -> Option<Vec<[usize; 3]>> {
    let edges = edge_faces(faces, members.iter().copied());
    // Closed: every edge shared by exactly two of the part's faces.
    if edges.values().any(|shared| shared.len() != 2) {
        return None;
    }
    let neighbours = |f: usize| {
        let face = faces[f];
        (0..3).map(move |k| {
            let (a, b) = (face[k], face[(k + 1) % 3]);
            (a.min(b), a.max(b))
        })
    };
    // Consistent: neighbours run their shared edge in opposite directions. Breadth-first
    // from the first face, flipping faces as needed.
    let mut flip: HashMap<usize, bool> = HashMap::new();
    flip.insert(members[0], false);
    let mut queue = std::collections::VecDeque::from([members[0]]);
    while let Some(f) = queue.pop_front() {
        for e in neighbours(f) {
            let shared = &edges[&e];
            let (mine, other) = if shared[0].0 == f {
                (shared[0], shared[1])
            } else {
                (shared[1], shared[0])
            };
            // Same direction before flipping: one of the two must be flipped.
            let want = flip[&f] ^ (mine.1 == other.1);
            match flip.get(&other.0) {
                None => {
                    flip.insert(other.0, want);
                    queue.push_back(other.0);
                }
                Some(&have) if have != want => return None, // not orientable
                Some(_) => {}
            }
        }
    }
    let mut oriented: Vec<[usize; 3]> = members
        .iter()
        .map(|&f| {
            let [a, b, c] = faces[f];
            if flip[&f] {
                [a, c, b]
            } else {
                [a, b, c]
            }
        })
        .collect();
    let volume: f64 = oriented
        .iter()
        .map(|f| {
            let [a, b, c] = f.map(|v| vertices[v]);
            dot(&a, &cross(&b, &c)) / 6.0
        })
        .sum();
    if !volume.is_finite() || volume.abs() <= 1e-12 {
        return None;
    }
    if volume < 0.0 {
        for f in &mut oriented {
            f.swap(1, 2);
        }
    }
    Some(oriented)
}

/// The point of triangle `t` closest to `p` (Ericson, Real-Time Collision Detection, 5.1.5).
fn closest_point(t: &Triangle, p: &Point) -> Point {
    let [a, b, c] = t;
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(&ab, &ap), dot(&ac, &ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return *a;
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(&ab, &bp), dot(&ac, &bp));
    if d3 >= 0.0 && d4 <= d3 {
        return *b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return add(a, &scale(&ab, d1 / (d1 - d3)));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(&ab, &cp), dot(&ac, &cp));
    if d6 >= 0.0 && d5 <= d6 {
        return *c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return add(a, &scale(&ac, d2 / (d2 - d6)));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, &scale(&sub(c, b), w));
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    add(a, &add(&scale(&ab, v), &scale(&ac, w)))
}

/// Signed solid angle of triangle `t` seen from `p` (Van Oosterom and Strackee, 1983).
/// Summed over a closed, outward mesh: 4 pi inside, 0 outside.
fn solid_angle(t: &Triangle, p: &Point) -> f64 {
    let [a, b, c] = t.map(|v| sub(&v, p));
    let (la, lb, lc) = (norm(&a), norm(&b), norm(&c));
    let numerator = dot(&a, &cross(&b, &c));
    let denominator = la * lb * lc + dot(&a, &b) * lc + dot(&b, &c) * la + dot(&c, &a) * lb;
    2.0 * numerator.atan2(denominator)
}

struct DisjointSets(Vec<usize>);

impl DisjointSets {
    fn new(n: usize) -> Self {
        Self((0..n).collect())
    }

    fn find(&mut self, mut i: usize) -> usize {
        while self.0[i] != i {
            self.0[i] = self.0[self.0[i]];
            i = self.0[i];
        }
        i
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        self.0[a] = b;
    }
}

/// Picks indices with probability proportional to their weights.
struct AreaSampler {
    cumulative: Vec<f64>,
}

impl AreaSampler {
    /// None if the weights add up to nothing.
    fn new(weights: impl Iterator<Item = f64>) -> Option<Self> {
        let mut total = 0.0;
        let cumulative: Vec<f64> = weights
            .map(|w| {
                total += w;
                total
            })
            .collect();
        (total > 0.0).then_some(Self { cumulative })
    }

    fn pick(&self, rng: &mut Rng) -> usize {
        let x = rng.next_f64() * self.cumulative[self.cumulative.len() - 1];
        self.cumulative
            .partition_point(|&c| c <= x)
            .min(self.cumulative.len() - 1)
    }
}

/// `k` distinct indices out of `n`, uniformly.
fn choose(n: usize, k: usize, rng: &mut Rng) -> Vec<usize> {
    let mut all: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (rng.next_u64() % (n - i) as u64) as usize;
        all.swap(i, j);
    }
    all.truncate(k);
    all
}

/// A uniform point on a triangle.
fn point_in(t: &Triangle, rng: &mut Rng) -> Point {
    let r1 = rng.next_f64().sqrt();
    let r2 = rng.next_f64();
    let (wa, wb, wc) = (1.0 - r1, r1 * (1.0 - r2), r1 * r2);
    std::array::from_fn(|i| wa * t[0][i] + wb * t[1][i] + wc * t[2][i])
}

fn area(t: &Triangle) -> f64 {
    0.5 * norm(&cross(&sub(&t[1], &t[0]), &sub(&t[2], &t[0])))
}

fn unit_normal(t: &Triangle) -> Point {
    let n = cross(&sub(&t[1], &t[0]), &sub(&t[2], &t[0]));
    let l = norm(&n);
    if l > 0.0 {
        scale(&n, 1.0 / l)
    } else {
        [0.0; 3]
    }
}

fn bounds(points: &[Point]) -> (Point, Point) {
    points.iter().fold(
        ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]),
        |(lo, hi), p| {
            (
                std::array::from_fn(|i| lo[i].min(p[i])),
                std::array::from_fn(|i| hi[i].max(p[i])),
            )
        },
    )
}

fn add(a: &Point, b: &Point) -> Point {
    std::array::from_fn(|i| a[i] + b[i])
}

fn sub(a: &Point, b: &Point) -> Point {
    std::array::from_fn(|i| a[i] - b[i])
}

fn scale(a: &Point, s: f64) -> Point {
    a.map(|x| x * s)
}

fn dot(a: &Point, b: &Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &Point, b: &Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: &Point) -> f64 {
    dot(a, a).sqrt()
}

fn dist2(a: &Point, b: &Point) -> f64 {
    let d = sub(a, b);
    dot(&d, &d)
}

fn dist(a: &Point, b: &Point) -> f64 {
    dist2(a, b).sqrt()
}
