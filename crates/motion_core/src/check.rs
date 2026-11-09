//! Yes/no collision checking for planners: "is this configuration free?", "is this
//! straight joint-space motion free?", as fast as possible.
//!
//! The distance kernels in `cpu.rs` compute every sphere's distance and gradient, which
//! optimization needs. Sampling planners only need a verdict, so this checker
//! (after VAMP, Thomason, Kingston and Kavraki, ICRA 2024):
//!
//! * runs `LANES` configurations at once, in explicit SIMD (`wide`: 4 lanes on ARM NEON,
//!   8 on x86);
//! * tests one bounding sphere per link before the link's own spheres;
//! * computes a link's sphere positions only when its bounding sphere hits something;
//! * stops at the first contact in any lane.
//!
//! Obstacles are spheres, capsules, oriented boxes and, optionally, an ESDF. Each obstacle
//! has a mask of links allowed to touch it.
//!
//! Attribution: the motion "rake", the sphere/cuboid test and the bounding-sphere-first,
//! early-exit structure follow VAMP (https://github.com/KavrakiLab/vamp, Apache-2.0); the
//! sine/cosine polynomials are Cephes' (Stephen L. Moshier). See `native/NOTICE`.

use std::mem::MaybeUninit;
use std::sync::Arc;

use crate::esdf::VoxelEsdf;
use crate::layout::{GpuLink, MAX_LINKS, MAX_SPHERES};
use crate::model::JointType;
pub use crate::simd::LANES;
use crate::simd::{compose, dist_sq, to_world, v, M, V};

use crate::model::{Mat34, RobotModel, Scene, Sphere};

/// One link's spheres: a contiguous range of the sphere array and their bounding sphere,
/// all in the link frame.
#[derive(Clone, Debug)]
struct Group {
    /// Robot link (for touch masks and disabled pairs).
    link: u32,
    /// Index of the FK frame the spheres are expressed in (nearest moving ancestor).
    frame: u32,
    start: u32,
    end: u32,
    /// Bounding sphere: center xyz, radius.
    bound: [f32; 4],
    /// Boxes (bit `k` = box `k`) every sphere of the group may touch.
    skip: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ObstacleSphere {
    pub center: [f32; 3],
    pub radius: f32,
    /// Bit `i` set: link `i` may touch this obstacle.
    pub touch: u32,
}

/// Capsule: segment `a`..`a + d` swept by `radius`.
#[derive(Clone, Copy, Debug)]
pub struct ObstacleCapsule {
    pub a: [f32; 3],
    pub d: [f32; 3],
    inv_dd: f32,
    pub radius: f32,
    pub touch: u32,
}

/// Oriented box: center, unit axes (rows), half extents.
#[derive(Clone, Copy, Debug)]
pub struct ObstacleBox {
    pub center: [f32; 3],
    pub axes: [[f32; 3]; 3],
    pub half_extents: [f32; 3],
    pub touch: u32,
}

#[derive(Clone, Default)]
pub struct Obstacles {
    pub spheres: Vec<ObstacleSphere>,
    pub capsules: Vec<ObstacleCapsule>,
    pub boxes: Vec<ObstacleBox>,
    pub esdf: Option<Arc<VoxelEsdf>>,
    pub esdf_margin: f32,
    /// Per robot sphere (order of `RobotModel::spheres`): bit `k` set means the sphere may
    /// touch box `k`. Empty = no exemptions.
    pub sphere_box_skip: Vec<u32>,
}

impl Obstacles {
    /// The obstacles of a native-core scene: its boxes (with their touch links), its ESDF
    /// and the per-sphere exemptions.
    pub fn from_scene(model: &RobotModel, scene: &Scene) -> Self {
        let mut obs = Obstacles::default();
        for c in &scene.cuboids {
            obs.add_box(&c.frame, c.half_extents);
            let touch = model
                .links
                .iter()
                .enumerate()
                .filter(|(_, l)| c.touch_links.contains(&l.name))
                .fold(0u32, |m, (i, _)| m | (1 << i));
            obs.boxes.last_mut().unwrap().touch = touch;
        }
        obs.esdf = scene.esdf.clone();
        obs.esdf_margin = scene.esdf_margin;
        obs.sphere_box_skip = scene.sphere_touch.clone();
        obs
    }

    pub fn add_sphere(&mut self, center: [f64; 3], radius: f64) {
        self.spheres.push(ObstacleSphere {
            center: center.map(|v| v as f32),
            radius: radius as f32,
            touch: 0,
        });
    }

    /// Capsule between `a` and `b`.
    pub fn add_capsule(&mut self, a: [f64; 3], b: [f64; 3], radius: f64) {
        let a = a.map(|v| v as f32);
        let d = [b[0] as f32 - a[0], b[1] as f32 - a[1], b[2] as f32 - a[2]];
        let dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        self.capsules.push(ObstacleCapsule {
            a,
            d,
            inv_dd: if dd > 0.0 { 1.0 / dd } else { 0.0 },
            radius: radius as f32,
            touch: 0,
        });
    }

    /// Box with world frame `frame` (3x4 row-major; columns are the box axes).
    pub fn add_box(&mut self, frame: &Mat34, half_extents: [f64; 3]) {
        let axes = [0, 1, 2].map(|c| [frame[c] as f32, frame[4 + c] as f32, frame[8 + c] as f32]);
        self.boxes.push(ObstacleBox {
            center: [frame[3] as f32, frame[7] as f32, frame[11] as f32],
            axes,
            half_extents: half_extents.map(|v| v as f32),
            touch: 0,
        });
    }

    pub fn len(&self) -> usize {
        self.spheres.len() + self.capsules.len() + self.boxes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0 && self.esdf.is_none()
    }
}

pub struct Checker {
    pub dof: usize,
    pub lower: Vec<f32>,
    pub upper: Vec<f32>,
    links: Vec<GpuLink>,
    /// Number of links FK has to compute (the last link with spheres, plus its ancestors).
    n_fk: usize,
    /// Environment spheres, grouped by link, ordered by FK frame.
    env_groups: Vec<Group>,
    /// `env_groups[env_by_frame[f]..env_by_frame[f + 1]]` are the groups on frame `f`.
    env_by_frame: Vec<u16>,
    env_spheres: Vec<[f32; 4]>,
    /// Per environment sphere: boxes it may touch (scene `sphere_touch`).
    env_skip: Vec<u32>,
    /// Self-collision spheres (may differ from the environment ones), grouped by link.
    self_groups: Vec<Group>,
    self_spheres: Vec<[f32; 4]>,
    /// Pairs of `self_groups` checked against each other.
    self_pairs: Vec<[u16; 2]>,
    /// The root link's pose: links rigidly attached to it never move.
    root: [f32; 12],
    /// World positions of self-collision spheres on the root (static) and the bounds of
    /// static groups, indexed like `self_spheres` / `self_groups`.
    static_self: Vec<[f32; 3]>,
    static_bounds: Vec<[f32; 3]>,
    /// Static links already touch an obstacle: every configuration collides.
    static_collision: bool,
    obstacles: Obstacles,
    /// A further test for configurations the spheres find free (true = collision), e.g.
    /// exact meshes for links the spheres approximate poorly. See `with_extra`.
    extra: Option<ExtraCheck>,
}

/// Collision test for one configuration (true = collision), run after the sphere checks.
pub type ExtraCheck = Arc<dyn Fn(&[f32]) -> bool + Send + Sync>;

/// Spheres grouped by link, re-expressed in each link's FK frame (`frames[link]`, with the
/// link's pose in that frame `poses[link]`).
fn groups(
    spheres: &[Sphere],
    frames: &[u32],
    poses: &[Mat34],
    box_skip: &[u32],
) -> (Vec<Group>, Vec<[f32; 4]>, Vec<u32>) {
    let mut order: Vec<usize> = (0..spheres.len()).collect();
    order.sort_by_key(|&i| spheres[i].link);
    let skips: Vec<u32> = order
        .iter()
        .map(|&i| box_skip.get(i).copied().unwrap_or(0))
        .collect();
    let sorted: Vec<&Sphere> = order.iter().map(|&i| &spheres[i]).collect();
    let flat: Vec<[f32; 4]> = sorted
        .iter()
        .map(|s| {
            let m = &poses[s.link as usize];
            let c = s.center;
            let p: [f64; 3] = std::array::from_fn(|r| {
                m[r * 4] * c[0] + m[r * 4 + 1] * c[1] + m[r * 4 + 2] * c[2] + m[r * 4 + 3]
            });
            [p[0] as f32, p[1] as f32, p[2] as f32, s.radius as f32]
        })
        .collect();
    let mut out: Vec<Group> = Vec::new();
    for (i, s) in sorted.iter().enumerate() {
        match out.last_mut() {
            Some(g) if g.link == s.link => g.end = i as u32 + 1,
            _ => out.push(Group {
                link: s.link,
                frame: frames[s.link as usize],
                start: i as u32,
                end: i as u32 + 1,
                bound: [0.0; 4],
                skip: 0,
            }),
        }
    }
    for g in &mut out {
        let members = &flat[g.start as usize..g.end as usize];
        let n = members.len() as f32;
        let mut c = [0.0f32; 3];
        for m in members {
            for k in 0..3 {
                c[k] += m[k] / n;
            }
        }
        let r = members
            .iter()
            .map(|m| {
                ((m[0] - c[0]).powi(2) + (m[1] - c[1]).powi(2) + (m[2] - c[2]).powi(2)).sqrt()
                    + m[3]
            })
            .fold(0.0f32, f32::max);
        // Rounding margin so the bound never undercuts its members in f32.
        g.bound = [c[0], c[1], c[2], r * (1.0 + 1e-6) + 1e-6];
        g.skip = skips[g.start as usize..g.end as usize]
            .iter()
            .fold(!0, |a, &b| a & b);
    }
    (out, flat, skips)
}

/// True if any lane of a sphere (world position, radius) touches an obstacle that `link`
/// may not touch. Hit masks are OR-ed over all primitives and reduced once (a reduction
/// per obstacle costs more than the extra arithmetic).
#[inline(always)]
fn hits_obstacles(obs: &Obstacles, link_bit: u32, box_skip: u32, p: &[V; 3], r: f32) -> bool {
    let mut hit = v(0.0).simd_lt(v(0.0)); // all false
    for o in &obs.spheres {
        if o.touch & link_bit == 0 {
            let rr = (r + o.radius) * (r + o.radius);
            hit |= dist_sq(p, o.center.map(v)).simd_le(v(rr));
        }
    }
    for o in &obs.capsules {
        if o.touch & link_bit == 0 {
            let rr = (r + o.radius) * (r + o.radius);
            let w = [p[0] - v(o.a[0]), p[1] - v(o.a[1]), p[2] - v(o.a[2])];
            let t = (w[0].mul_add(v(o.d[0]), w[1].mul_add(v(o.d[1]), w[2] * v(o.d[2])))
                * v(o.inv_dd))
            .max(v(0.0))
            .min(v(1.0));
            let e = [0, 1, 2].map(|k| t.mul_add(v(-o.d[k]), w[k]));
            hit |= e[0]
                .mul_add(e[0], e[1].mul_add(e[1], e[2] * e[2]))
                .simd_le(v(rr));
        }
    }
    for (k, o) in obs.boxes.iter().enumerate() {
        if o.touch & link_bit == 0 && (k >= 32 || box_skip & (1 << k) == 0) {
            let w = [
                p[0] - v(o.center[0]),
                p[1] - v(o.center[1]),
                p[2] - v(o.center[2]),
            ];
            let mut sum = v(0.0);
            for k in 0..3 {
                let a = o.axes[k];
                let proj = w[0].mul_add(v(a[0]), w[1].mul_add(v(a[1]), w[2] * v(a[2])));
                let e = (proj.abs() - v(o.half_extents[k])).max(v(0.0));
                sum = e.mul_add(e, sum);
            }
            hit |= sum.simd_le(v(r * r));
        }
    }
    if hit.any() {
        return true;
    }
    if let Some(esdf) = &obs.esdf {
        let (x, y, z) = (p[0].to_array(), p[1].to_array(), p[2].to_array());
        for l in 0..LANES {
            let (d, _) = esdf.sample([x[l], y[l], z[l]]);
            if d - obs.esdf_margin <= r {
                return true;
            }
        }
    }
    false
}

impl Checker {
    pub fn new(model: &RobotModel, obstacles: Obstacles) -> Self {
        assert!(
            model.links.len() <= MAX_LINKS,
            "more than {MAX_LINKS} links"
        );
        // Fold fixed joints into the next moving joint: FK runs over the root and the moving
        // links only; every link maps to the frame of its nearest moving ancestor.
        let mul = |a: &Mat34, b: &Mat34| -> Mat34 {
            std::array::from_fn(|k| {
                let (r, c) = (k / 4, k % 4);
                let v = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c];
                if c == 3 {
                    v + a[r * 4 + 3]
                } else {
                    v
                }
            })
        };
        const ID: Mat34 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut links: Vec<GpuLink> = Vec::new();
        let mut frames = vec![0u32; model.links.len()];
        let mut poses = vec![ID; model.links.len()];
        for (i, l) in model.links.iter().enumerate() {
            let gpu = |origin: &Mat34, parent: i32| GpuLink {
                origin: origin.map(|v| v as f32),
                axis: l.axis.map(|v| v as f32),
                parent,
                joint_type: l.joint_type as u32,
                q_index: l.q_index,
                _pad: [0; 2],
            };
            if l.parent < 0 {
                frames[i] = links.len() as u32;
                links.push(gpu(&l.origin, -1));
                continue;
            }
            let p = l.parent as usize;
            let in_parent_frame = mul(&poses[p], &l.origin);
            if l.joint_type == JointType::Fixed {
                frames[i] = frames[p];
                poses[i] = in_parent_frame;
            } else {
                frames[i] = links.len() as u32;
                links.push(gpu(&in_parent_frame, frames[p] as i32));
            }
        }
        let (mut env_groups, env_spheres, env_skip) =
            groups(&model.spheres, &frames, &poses, &obstacles.sphere_box_skip);
        env_groups.sort_by_key(|g| g.frame);
        let env_by_frame: Vec<u16> = (0..=links.len())
            .map(|f| env_groups.iter().filter(|g| (g.frame as usize) < f).count() as u16)
            .collect();
        let (self_groups, self_spheres, _) = groups(&model.self_spheres, &frames, &poses, &[]);
        assert!(env_spheres.len() <= MAX_SPHERES && self_spheres.len() <= MAX_SPHERES);
        let disabled = |a: u32, b: u32| {
            model
                .disabled_collisions
                .iter()
                .any(|&[x, y]| (x as u32 == a && y as u32 == b) || (x as u32 == b && y as u32 == a))
        };
        let mut self_pairs = Vec::new();
        for i in 0..self_groups.len() {
            for j in i + 1..self_groups.len() {
                if !disabled(self_groups[i].link, self_groups[j].link) {
                    self_pairs.push([i as u16, j as u16]);
                }
            }
        }
        let n_fk = env_groups
            .iter()
            .chain(&self_groups)
            .map(|g| g.frame as usize + 1)
            .max()
            .unwrap_or(0);
        let root = links[0].origin;
        let world = |c: &[f32; 4]| -> [f32; 3] {
            std::array::from_fn(|r| {
                root[r * 4] * c[0]
                    + root[r * 4 + 1] * c[1]
                    + root[r * 4 + 2] * c[2]
                    + root[r * 4 + 3]
            })
        };
        let static_self: Vec<[f32; 3]> = self_spheres.iter().map(world).collect();
        let static_bounds: Vec<[f32; 3]> = self_groups.iter().map(|g| world(&g.bound)).collect();
        let static_collision = env_groups.iter().filter(|g| g.frame == 0).any(|g| {
            (g.start as usize..g.end as usize).any(|i| {
                let (s, p) = (&env_spheres[i], world(&env_spheres[i]));
                let p = [v(p[0]), v(p[1]), v(p[2])];
                hits_obstacles(&obstacles, 1 << g.link, env_skip[i], &p, s[3])
            })
        });
        Self {
            dof: model.dof(),
            lower: model.joint_limits.iter().map(|l| l[0] as f32).collect(),
            upper: model.joint_limits.iter().map(|l| l[1] as f32).collect(),
            links,
            n_fk,
            env_groups,
            env_by_frame,
            env_spheres,
            env_skip,
            self_groups,
            self_spheres,
            self_pairs,
            root,
            static_self,
            static_bounds,
            static_collision,
            obstacles,
            extra: None,
        }
    }

    /// Also run `extra` on every configuration the spheres find free. It is the last word
    /// on configurations the sphere model cannot judge (links it approximates poorly), and
    /// it runs only when the spheres found no contact, so it adds nothing to the cost of
    /// configurations that collide anyway.
    pub fn with_extra(mut self, extra: ExtraCheck) -> Self {
        self.extra = Some(extra);
        self
    }

    pub fn obstacles(&self) -> &Obstacles {
        &self.obstacles
    }

    pub fn n_self_pairs(&self) -> usize {
        self.self_pairs.len()
    }

    /// Link frames for `LANES` configurations (`q[j]`: joint `j`, every lane).
    /// Initializes `frames[1..n_fk]`; frame 0 is the root, constant (`self.root`).
    #[inline(always)]
    fn fk(&self, q: &[V], frames: &mut [MaybeUninit<M>; MAX_LINKS]) {
        for i in 1..self.n_fk {
            let link = &self.links[i];
            let value = if link.q_index >= 0 {
                q[link.q_index as usize]
            } else {
                v(0.0)
            };
            let frame = if link.parent == 0 {
                compose(|k| v(self.root[k]), link, value)
            } else {
                // Parents come before children, so the parent frame is initialized.
                let p = unsafe { frames[link.parent as usize].assume_init_ref() };
                compose(|k| p[k], link, value)
            };
            frames[i].write(frame);
        }
    }

    /// True if any of the `LANES` configurations collides (environment or self).
    /// `q[j]` holds joint `j` for every lane.
    pub fn collides_lanes(&self, q: &[[f32; LANES]]) -> bool {
        debug_assert_eq!(q.len(), self.dof);
        let mut qv = [const { MaybeUninit::<V>::uninit() }; 16];
        for (j, x) in q.iter().enumerate() {
            qv[j].write(V::new(*x));
        }
        // The first `dof` entries were written just above.
        self.collides_v(unsafe { std::slice::from_raw_parts(qv.as_ptr().cast::<V>(), self.dof) })
    }

    fn collides_v(&self, q: &[V]) -> bool {
        if self.static_collision {
            return true;
        }
        // Frame by frame: compute the frame, then check its links against the environment,
        // so a configuration that collides early skips the rest of the chain (as VAMP does).
        // Frame 0 (the root) is constant: its links were checked when the checker was built.
        let mut frames = [const { MaybeUninit::<M>::uninit() }; MAX_LINKS];
        let check_env = !self.obstacles.is_empty();
        for i in 1..self.n_fk {
            let link = &self.links[i];
            let value = if link.q_index >= 0 {
                q[link.q_index as usize]
            } else {
                v(0.0)
            };
            let m = if link.parent == 0 {
                compose(|k| v(self.root[k]), link, value)
            } else {
                // Parents come before children, so the parent frame is initialized.
                let p = unsafe { frames[link.parent as usize].assume_init_ref() };
                compose(|k| p[k], link, value)
            };
            let m = frames[i].write(m);
            if !check_env {
                continue;
            }
            let groups = self.env_by_frame[i] as usize..self.env_by_frame[i + 1] as usize;
            for g in &self.env_groups[groups] {
                let bit = 1u32 << g.link;
                let b = to_world(m, [g.bound[0], g.bound[1], g.bound[2]]);
                if !hits_obstacles(&self.obstacles, bit, g.skip, &b, g.bound[3]) {
                    continue;
                }
                for i in g.start as usize..g.end as usize {
                    let s = &self.env_spheres[i];
                    let p = to_world(m, [s[0], s[1], s[2]]);
                    if hits_obstacles(&self.obstacles, bit, self.env_skip[i], &p, s[3]) {
                        return true;
                    }
                }
            }
        }
        // Every moving group's frame is in 1..n_fk, initialized above.
        let frame = |f: u32| unsafe { frames[f as usize].assume_init_ref() };
        let splat3 = |p: [f32; 3]| [v(p[0]), v(p[1]), v(p[2])];

        // Self-collision: bounding spheres of the pair, then each sphere against the other
        // link's bound, then sphere pairs.
        let bound = |i: u16| -> [V; 3] {
            let g = &self.self_groups[i as usize];
            if g.frame == 0 {
                splat3(self.static_bounds[i as usize])
            } else {
                to_world(frame(g.frame), [g.bound[0], g.bound[1], g.bound[2]])
            }
        };
        // Sphere positions of moving groups, computed per group on first use (`computed`).
        let mut world = [const { MaybeUninit::<[V; 3]>::uninit() }; MAX_SPHERES];
        let mut computed = 0u64;
        for &[i, j] in &self.self_pairs {
            let (gi, gj) = (&self.self_groups[i as usize], &self.self_groups[j as usize]);
            let (bi, bj) = (bound(i), bound(j));
            let rr = (gi.bound[3] + gj.bound[3]) * (gi.bound[3] + gj.bound[3]);
            if !dist_sq(&bi, bj).simd_le(v(rr)).any() {
                continue;
            }
            for (k, g) in [(i as usize, gi), (j as usize, gj)] {
                if g.frame != 0 && computed & (1 << k) == 0 {
                    computed |= 1 << k;
                    let m = frame(g.frame);
                    let range = g.start as usize..g.end as usize;
                    for (w, c) in world[range.clone()]
                        .iter_mut()
                        .zip(&self.self_spheres[range])
                    {
                        w.write(to_world(m, [c[0], c[1], c[2]]));
                    }
                }
            }
            // Static groups read `static_self`; moving ones were computed just above.
            let pos = |g: &Group, s: usize| -> [V; 3] {
                if g.frame == 0 {
                    splat3(self.static_self[s])
                } else {
                    unsafe { *world[s].assume_init_ref() }
                }
            };
            for a in gi.start as usize..gi.end as usize {
                let pa = pos(gi, a);
                let ra = self.self_spheres[a][3];
                if !dist_sq(&pa, bj)
                    .simd_le(v((ra + gj.bound[3]) * (ra + gj.bound[3])))
                    .any()
                {
                    continue;
                }
                for b in gj.start as usize..gj.end as usize {
                    let r = ra + self.self_spheres[b][3];
                    if dist_sq(&pa, pos(gj, b)).simd_le(v(r * r)).any() {
                        return true;
                    }
                }
            }
        }
        if let Some(extra) = &self.extra {
            let lanes: Vec<[f32; LANES]> = q.iter().map(|x| x.to_array()).collect();
            let mut config = vec![0.0f32; q.len()];
            for l in 0..LANES {
                for (c, lane) in config.iter_mut().zip(&lanes) {
                    *c = lane[l];
                }
                if extra(&config) {
                    return true;
                }
            }
        }
        false
    }

    /// Forward kinematics only, for benchmarks: returns one value so the work is not elided.
    #[doc(hidden)]
    pub fn fk_only_lanes(&self, q: &[[f32; LANES]]) -> f32 {
        let qv: Vec<V> = q.iter().map(|x| V::new(*x)).collect();
        let mut frames = [const { MaybeUninit::<M>::uninit() }; MAX_LINKS];
        self.fk(&qv, &mut frames);
        if self.n_fk < 2 {
            return 0.0;
        }
        unsafe { frames[self.n_fk - 1].assume_init_ref()[3].to_array()[0] }
    }

    /// True if the configuration collides.
    pub fn collides(&self, q: &[f32]) -> bool {
        let qv: Vec<V> = q.iter().map(|&x| v(x)).collect();
        self.collides_v(&qv)
    }

    /// True if the straight joint-space motion from `a` to `b` is free, sampled at
    /// `resolution` points per unit of joint-space (L2) distance.
    ///
    /// The `LANES` lanes start evenly spread over the motion and step back together (VAMP's
    /// "rake"), so a collision anywhere is found after few steps.
    pub fn motion_free(&self, a: &[f32], b: &[f32], resolution: f32) -> bool {
        let dof = self.dof;
        let mut dist = 0.0f32;
        for j in 0..dof {
            dist += (b[j] - a[j]) * (b[j] - a[j]);
        }
        let dist = dist.sqrt();
        let n = ((dist / LANES as f32 * resolution).ceil() as usize).max(1);
        let percents = V::new(std::array::from_fn(|l| (l as f32 + 1.0) / LANES as f32));
        let mut block = [v(0.0); 16];
        let mut back = [v(0.0); 16];
        for j in 0..dof {
            let d = b[j] - a[j];
            block[j] = v(d).mul_add(percents, v(a[j]));
            back[j] = v(d / (LANES * n) as f32);
        }
        if self.collides_v(&block[..dof]) {
            return false;
        }
        for _ in 1..n {
            for j in 0..dof {
                block[j] -= back[j];
            }
            if self.collides_v(&block[..dof]) {
                return false;
            }
        }
        true
    }

    /// World positions of the environment spheres for one configuration (for parity tests),
    /// grouped by link.
    pub fn sphere_positions(&self, q: &[f32]) -> Vec<[f32; 4]> {
        let qv: Vec<V> = q.iter().map(|&x| v(x)).collect();
        let mut frames = [const { MaybeUninit::<M>::uninit() }; MAX_LINKS];
        self.fk(&qv, &mut frames);
        let mut out = Vec::new();
        let root: M = self.root.map(v);
        for g in &self.env_groups {
            let m = if g.frame == 0 {
                &root
            } else {
                unsafe { frames[g.frame as usize].assume_init_ref() }
            };
            for s in &self.env_spheres[g.start as usize..g.end as usize] {
                let p = to_world(m, [s[0], s[1], s[2]]);
                out.push([
                    p[0].to_array()[0],
                    p[1].to_array()[0],
                    p[2].to_array()[0],
                    s[3],
                ]);
            }
        }
        out
    }
}
