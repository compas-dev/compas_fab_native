//! Exact collision checks on triangle meshes (parry3d): the final pass on planned paths.
//!
//! The sphere model is fast but approximate (it misses a few contacts per thousand
//! configurations, see SPIKE.md). Before a path is handed out, every configuration along
//! it is checked here against the real collision meshes: robot links pairwise (except
//! SRDF-disabled pairs) and each link against each obstacle (except its touch links).
//!
//! Surface intersection is tested with parry's BVH-accelerated mesh queries; containment
//! (one body entirely inside another) is caught by testing one vertex for inclusion.

use parry3d::math::{Mat3, Pose, Rot3, Vec3};
use parry3d::query::intersection_test;
use parry3d::shape::{Cuboid, Shape, TriMesh, TriMeshFlags};
use serde::Deserialize;

use motion_core::model::{Mat34, RobotModel, Scene};
use motion_core::reference::{forward_kinematics, mul};

#[derive(Deserialize)]
struct MeshJson {
    #[serde(default)]
    link: usize,
    #[serde(default)]
    name: String,
    #[serde(default)]
    touch_links: Vec<String>,
    vertices: Vec<f64>,
    faces: Vec<u32>,
}

#[derive(Deserialize)]
struct RobotMeshesJson {
    links: Vec<MeshJson>,
}

#[derive(Deserialize)]
struct ObstacleMeshesJson {
    obstacles: Vec<MeshJson>,
}

struct Body {
    shape: Box<dyn Shape>,
    /// A point on the body, for the containment test.
    probe: Vec3,
    /// Whether `contains_point` is meaningful (closed, consistently oriented mesh or a solid).
    solid: bool,
}

struct Obstacle {
    name: String,
    body: Body,
    pose: Pose,
    touch_links: Vec<usize>,
}

pub struct ExactChecker {
    model: RobotModel,
    links: Vec<Option<Body>>,
    obstacles: Vec<Obstacle>,
    /// Link index pairs to test against each other.
    pairs: Vec<(usize, usize)>,
}

/// What collides, for reporting.
#[derive(Clone, Debug, PartialEq)]
pub struct Contact {
    pub a: String,
    pub b: String,
}

fn pose(m: &Mat34) -> Pose {
    let rot = Mat3::from_cols(
        Vec3::new(m[0] as f32, m[4] as f32, m[8] as f32),
        Vec3::new(m[1] as f32, m[5] as f32, m[9] as f32),
        Vec3::new(m[2] as f32, m[6] as f32, m[10] as f32),
    );
    Pose::from_parts(
        Vec3::new(m[3] as f32, m[7] as f32, m[11] as f32),
        Rot3::from_mat3(&rot).normalize(),
    )
}

fn trimesh_body(mesh: &MeshJson) -> Result<Body, String> {
    let vertices: Vec<Vec3> = mesh
        .vertices
        .chunks_exact(3)
        .map(|v| Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32))
        .collect();
    let indices: Vec<[u32; 3]> = mesh
        .faces
        .chunks_exact(3)
        .map(|f| [f[0], f[1], f[2]])
        .collect();
    let probe = *vertices.first().ok_or("empty mesh")?;
    // Oriented meshes give a reliable inside test; fall back to a plain mesh otherwise.
    let flags = TriMeshFlags::ORIENTED | TriMeshFlags::MERGE_DUPLICATE_VERTICES;
    match TriMesh::with_flags(vertices.clone(), indices.clone(), flags) {
        Ok(t) => Ok(Body {
            shape: Box::new(t),
            probe,
            solid: true,
        }),
        Err(_) => {
            let t = TriMesh::new(vertices, indices).map_err(|e| format!("{e:?}"))?;
            Ok(Body {
                shape: Box::new(t),
                probe,
                solid: false,
            })
        }
    }
}

/// True if the two posed bodies overlap (surfaces intersect, or one contains the other).
fn overlap(a: &Body, pa: &Pose, b: &Body, pb: &Pose) -> bool {
    match intersection_test(pa, a.shape.as_ref(), pb, b.shape.as_ref()) {
        Ok(r) if r.intersecting => return true,
        Ok(_) => {}
        Err(_) => return false,
    }
    // No surface crossing: either disjoint or nested. Test one point of each in the other.
    (b.solid && b.shape.contains_point(pb, pa.transform_point(a.probe)))
        || (a.solid && a.shape.contains_point(pa, pb.transform_point(b.probe)))
}

impl ExactChecker {
    /// `robot_meshes`: `{"links": [{"link": i, "vertices": [...], "faces": [...]}]}` in link coordinates.
    pub fn new(model: RobotModel, robot_meshes: &str) -> Result<Self, String> {
        let parsed: RobotMeshesJson =
            serde_json::from_str(robot_meshes).map_err(|e| e.to_string())?;
        let mut links: Vec<Option<Body>> = (0..model.links.len()).map(|_| None).collect();
        for m in &parsed.links {
            if m.link >= links.len() {
                return Err(format!("mesh for unknown link {}", m.link));
            }
            links[m.link] = Some(trimesh_body(m)?);
        }
        let disabled = |a: usize, b: usize| {
            model
                .disabled_collisions
                .iter()
                .any(|&[x, y]| (x == a && y == b) || (x == b && y == a))
        };
        let mut pairs = Vec::new();
        for a in 0..links.len() {
            for b in a + 1..links.len() {
                if links[a].is_some() && links[b].is_some() && !disabled(a, b) {
                    pairs.push((a, b));
                }
            }
        }
        Ok(Self {
            model,
            links,
            obstacles: Vec::new(),
            pairs,
        })
    }

    fn link_index(&self, name: &str) -> Option<usize> {
        self.model.links.iter().position(|l| l.name == name)
    }

    /// Obstacles as meshes in robot coordinates: `{"obstacles": [{"name", "touch_links", "vertices", "faces"}]}`.
    pub fn set_obstacle_meshes(&mut self, json: &str) -> Result<(), String> {
        let parsed: ObstacleMeshesJson = serde_json::from_str(json).map_err(|e| e.to_string())?;
        self.obstacles = parsed
            .obstacles
            .iter()
            .map(|m| {
                Ok(Obstacle {
                    name: m.name.clone(),
                    body: trimesh_body(m)?,
                    pose: Pose::IDENTITY,
                    touch_links: m
                        .touch_links
                        .iter()
                        .filter_map(|n| self.link_index(n))
                        .collect(),
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(())
    }

    /// Obstacles as exact boxes (the spike's scene fixtures).
    pub fn set_obstacle_boxes(&mut self, scene: &Scene) {
        self.obstacles = scene
            .cuboids
            .iter()
            .map(|c| Obstacle {
                name: c.name.clone(),
                body: Body {
                    shape: Box::new(Cuboid::new(Vec3::new(
                        c.half_extents[0] as f32,
                        c.half_extents[1] as f32,
                        c.half_extents[2] as f32,
                    ))),
                    probe: Vec3::ZERO,
                    solid: true,
                },
                pose: pose(&c.frame),
                touch_links: c
                    .touch_links
                    .iter()
                    .filter_map(|n| self.link_index(n))
                    .collect(),
            })
            .collect();
    }

    /// First collision at configuration `q`, if any.
    pub fn check(&self, q: &[f64]) -> Option<Contact> {
        let frames = forward_kinematics(&self.model, q);
        let poses: Vec<Pose> = frames.iter().map(pose).collect();
        let name = |i: usize| self.model.links[i].name.clone();
        for &(a, b) in &self.pairs {
            if overlap(
                self.links[a].as_ref().unwrap(),
                &poses[a],
                self.links[b].as_ref().unwrap(),
                &poses[b],
            ) {
                return Some(Contact {
                    a: name(a),
                    b: name(b),
                });
            }
        }
        for (i, link) in self.links.iter().enumerate() {
            let Some(link) = link else { continue };
            for o in &self.obstacles {
                if !o.touch_links.contains(&i) && overlap(link, &poses[i], &o.body, &o.pose) {
                    return Some(Contact {
                        a: name(i),
                        b: o.name.clone(),
                    });
                }
            }
        }
        None
    }

    /// Check a joint-space path, interpolating so no joint moves more than `step` radians
    /// between checked configurations. Returns the first collision as (segment, fraction, contact).
    pub fn check_path(&self, path: &[Vec<f64>], step: f64) -> Option<(usize, f64, Contact)> {
        for (i, w) in path.windows(2).enumerate() {
            let jump = w[0]
                .iter()
                .zip(&w[1])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            let n = (jump / step).ceil().max(1.0) as usize;
            for k in 0..n {
                let f = k as f64 / n as f64;
                let q: Vec<f64> = w[0]
                    .iter()
                    .zip(&w[1])
                    .map(|(a, b)| a + f * (b - a))
                    .collect();
                if let Some(c) = self.check(&q) {
                    return Some((i, f, c));
                }
            }
        }
        let last = path.last()?;
        self.check(last).map(|c| (path.len() - 1, 0.0, c))
    }

    /// Collision flags for many configurations, checked in parallel.
    pub fn check_many(&self, configs: &[Vec<f64>]) -> Vec<Option<Contact>> {
        use rayon::prelude::*;
        configs.par_iter().map(|q| self.check(q)).collect()
    }

    pub fn model(&self) -> &RobotModel {
        &self.model
    }
}

/// Compose helper re-exported for tests.
pub fn compose(a: &Mat34, b: &Mat34) -> Mat34 {
    mul(a, b)
}
