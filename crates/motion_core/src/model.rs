//! Robot model and scene: loaded from the spike fixtures, flattened into the
//! layouts every compute backend shares.
//!
//! Transforms are 3x4 row-major `[r00 r01 r02 tx, r10 r11 r12 ty, r20 r21 r22 tz]`.

use serde::Deserialize;

pub type Mat34 = [f64; 12];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum JointType {
    Fixed = 0,
    Revolute = 1,
    Prismatic = 2,
}

/// One link and the joint that connects it to its parent (identity for the root).
#[derive(Clone, Debug)]
pub struct Link {
    pub name: String,
    /// Index of the parent link, always smaller than this link's index. -1 for the root.
    pub parent: i32,
    pub joint_type: JointType,
    /// Joint origin relative to the parent link.
    pub origin: Mat34,
    pub axis: [f64; 3],
    /// Index of the joint value in a configuration, -1 if not driven by one.
    pub q_index: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct Sphere {
    pub link: u32,
    pub center: [f64; 3],
    pub radius: f64,
}

#[derive(Clone, Debug)]
pub struct RobotModel {
    pub name: String,
    pub joint_names: Vec<String>,
    pub joint_limits: Vec<[f64; 2]>,
    /// Topologically ordered: every link comes after its parent.
    pub links: Vec<Link>,
    pub spheres: Vec<Sphere>,
    /// Spheres inside the link meshes, for self-collision (covering spheres of neighbouring
    /// links overlap permanently). Falls back to `spheres` when the fixture has none.
    pub self_spheres: Vec<Sphere>,
    /// Link whose frame planning targets refer to (e.g. `tool0`).
    pub end_effector_link: usize,
    /// Link pairs never checked against each other (SRDF disabled collisions).
    pub disabled_collisions: Vec<[usize; 2]>,
    pub analytic_ik: Option<crate::ik::AnalyticIk>,
}

impl RobotModel {
    pub fn dof(&self) -> usize {
        self.joint_names.len()
    }

    /// Disable self-collision checks between link pairs whose spheres overlap in at least
    /// `threshold` of `samples` random configurations (MoveIt setup assistant's "always in
    /// collision" rule, applied to the sphere model). Returns the newly disabled pairs.
    pub fn disable_always_colliding(
        &mut self,
        samples: usize,
        threshold: f64,
        seed: u64,
    ) -> Vec<[usize; 2]> {
        let n_links = self.links.len();
        let q = crate::sampling::random_configurations(self, samples, seed);
        let mut hits = vec![0usize; n_links * n_links];
        for b in 0..samples {
            let qb: Vec<f64> = (0..self.dof()).map(|j| q[j * samples + b]).collect();
            let frames = crate::reference::forward_kinematics(self, &qb);
            let world: Vec<[f64; 3]> = self
                .self_spheres
                .iter()
                .map(|s| crate::reference::transform_point(&frames[s.link as usize], s.center))
                .collect();
            let mut pair = vec![false; n_links * n_links];
            for i in 0..world.len() {
                for j in i + 1..world.len() {
                    let (a, c) = (&self.self_spheres[i], &self.self_spheres[j]);
                    if a.link == c.link {
                        continue;
                    }
                    let d2 = (0..3)
                        .map(|k| (world[i][k] - world[j][k]).powi(2))
                        .sum::<f64>();
                    if d2 < (a.radius + c.radius).powi(2) {
                        let (x, y) = (a.link.min(c.link) as usize, a.link.max(c.link) as usize);
                        pair[x * n_links + y] = true;
                    }
                }
            }
            for (h, p) in hits.iter_mut().zip(pair) {
                *h += p as usize;
            }
        }
        let mut added = Vec::new();
        for x in 0..n_links {
            for y in x + 1..n_links {
                let already = self
                    .disabled_collisions
                    .iter()
                    .any(|&[a, b]| (a == x && b == y) || (a == y && b == x));
                if !already && hits[x * n_links + y] as f64 >= threshold * samples as f64 {
                    self.disabled_collisions.push([x, y]);
                    added.push([x, y]);
                }
            }
        }
        added
    }
}

#[derive(Clone, Debug)]
pub struct Cuboid {
    pub name: String,
    /// Pose of the box in world coordinates.
    pub frame: Mat34,
    pub half_extents: [f64; 3],
}

#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub cuboids: Vec<Cuboid>,
}

/// Reference data from compas_robots: link frames for a set of configurations.
#[derive(Clone, Debug)]
pub struct ReferenceFk {
    pub configs: Vec<Vec<f64>>,
    /// `link_frames[config][link]`
    pub link_frames: Vec<Vec<Mat34>>,
}

// ---------------------------------------------------------------------------
// Fixture parsing

#[derive(Deserialize)]
struct FixtureJoint {
    #[serde(rename = "type")]
    kind: String,
    origin: Vec<f64>,
    axis: [f64; 3],
    q_index: i32,
}

#[derive(Deserialize)]
struct FixtureLink {
    name: String,
    parent: i32,
    joint: Option<FixtureJoint>,
}

#[derive(Deserialize)]
struct FixtureSphere {
    link: u32,
    center: [f64; 3],
    radius: f64,
}

#[derive(Deserialize)]
struct FixtureReference {
    configs: Vec<Vec<f64>>,
    link_frames: Vec<Vec<Vec<f64>>>,
}

#[derive(Deserialize)]
struct Fixture {
    name: String,
    joint_names: Vec<String>,
    joint_limits: Vec<[f64; 2]>,
    links: Vec<FixtureLink>,
    spheres: Vec<FixtureSphere>,
    #[serde(default)]
    self_spheres: Option<Vec<FixtureSphere>>,
    reference_fk: FixtureReference,
    #[serde(default)]
    end_effector_link: Option<usize>,
    #[serde(default)]
    disabled_collisions: Vec<[usize; 2]>,
    #[serde(default)]
    analytic_ik: Option<FixtureIk>,
}

#[derive(Deserialize)]
struct FixtureIk {
    family: String,
    params: Vec<f64>,
    flange_frame: Vec<f64>,
}

#[derive(Deserialize)]
struct FixtureCuboid {
    name: String,
    frame: Vec<f64>,
    half_extents: [f64; 3],
}

#[derive(Deserialize)]
struct FixtureScene {
    cuboids: Vec<FixtureCuboid>,
}

fn mat34(v: &[f64]) -> Result<Mat34, String> {
    v.try_into()
        .map_err(|_| format!("expected 12 values for a 3x4 transform, got {}", v.len()))
}

fn normalized(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}

pub fn load_robot(json: &str) -> Result<(RobotModel, ReferenceFk), String> {
    let f: Fixture = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let mut links = Vec::with_capacity(f.links.len());
    for (i, l) in f.links.into_iter().enumerate() {
        if l.parent >= i as i32 {
            return Err(format!("link {} is not topologically ordered", l.name));
        }
        let link = match l.joint {
            None => Link {
                name: l.name,
                parent: l.parent,
                joint_type: JointType::Fixed,
                origin: IDENTITY,
                axis: [1.0, 0.0, 0.0],
                q_index: -1,
            },
            Some(j) => {
                let joint_type = match j.kind.as_str() {
                    "fixed" => JointType::Fixed,
                    "revolute" | "continuous" => JointType::Revolute,
                    "prismatic" => JointType::Prismatic,
                    other => return Err(format!("unsupported joint type {other}")),
                };
                Link {
                    name: l.name,
                    parent: l.parent,
                    joint_type,
                    origin: mat34(&j.origin)?,
                    axis: normalized(j.axis),
                    q_index: if joint_type == JointType::Fixed {
                        -1
                    } else {
                        j.q_index
                    },
                }
            }
        };
        links.push(link);
    }
    let to_sphere = |s: FixtureSphere| Sphere {
        link: s.link,
        center: s.center,
        radius: s.radius,
    };
    let spheres: Vec<Sphere> = f.spheres.into_iter().map(to_sphere).collect();
    let self_spheres = match f.self_spheres {
        Some(v) => v.into_iter().map(to_sphere).collect(),
        None => spheres.clone(),
    };
    let reference = ReferenceFk {
        configs: f.reference_fk.configs,
        link_frames: f
            .reference_fk
            .link_frames
            .iter()
            .map(|frames| frames.iter().map(|m| mat34(m)).collect::<Result<_, _>>())
            .collect::<Result<_, _>>()?,
    };
    let analytic_ik = match f.analytic_ik {
        None => None,
        Some(ik) if ik.family == "offset_wrist" => Some(crate::ik::AnalyticIk::offset_wrist(
            ik.params
                .as_slice()
                .try_into()
                .map_err(|_| "offset_wrist needs 6 params")?,
            mat34(&ik.flange_frame)?,
        )),
        Some(ik) => return Err(format!("unsupported IK family {}", ik.family)),
    };
    let end_effector_link = f.end_effector_link.unwrap_or(links.len() - 1);
    let model = RobotModel {
        name: f.name,
        joint_names: f.joint_names,
        joint_limits: f.joint_limits,
        links,
        spheres,
        self_spheres,
        end_effector_link,
        disabled_collisions: f.disabled_collisions,
        analytic_ik,
    };
    Ok((model, reference))
}

pub fn load_scene(json: &str) -> Result<Scene, String> {
    let f: FixtureScene = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let cuboids = f
        .cuboids
        .into_iter()
        .map(|c| {
            Ok(Cuboid {
                name: c.name,
                frame: orthonormalized(mat34(&c.frame)?),
                half_extents: c.half_extents,
            })
        })
        .collect::<Result<_, String>>()?;
    Ok(Scene { cuboids })
}

pub const IDENTITY: Mat34 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

/// Re-orthonormalize the rotation part (scene files carry rounded values like 0.7071).
fn orthonormalized(m: Mat34) -> Mat34 {
    let x = normalized([m[0], m[4], m[8]]);
    let y0 = [m[1], m[5], m[9]];
    let d = x[0] * y0[0] + x[1] * y0[1] + x[2] * y0[2];
    let y = normalized([y0[0] - d * x[0], y0[1] - d * x[1], y0[2] - d * x[2]]);
    let z = [
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0],
    ];
    [
        x[0], y[0], z[0], m[3], x[1], y[1], z[1], m[7], x[2], y[2], z[2], m[11],
    ]
}
