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
use parry3d::query::{distance, intersection_test};
use parry3d::shape::{ConvexPolyhedron, Cuboid, Shape, TriMesh, TriMeshFlags};
use serde::Deserialize;

use motion_core::model::{Mat34, RobotModel, Scene};
use motion_core::reference::{forward_kinematics, mul};

pub mod spheres;

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
    /// Links `check_focus` tests (see `set_focus`), and the pairs that involve them.
    focus: Vec<usize>,
    focus_pairs: Vec<(usize, usize)>,
    /// Convex hull of every link with a mesh, for `check_focus`, and its bounding sphere
    /// (centre in link coordinates, radius) for a quick first test.
    hulls: Vec<Option<ConvexPolyhedron>>,
    hull_bounds: Vec<(Vec3, f32)>,
    /// The scene's boxes, for `check_focus` (see `set_focus_obstacles`): shape, pose,
    /// links allowed to touch it, bounding radius.
    focus_boxes: Vec<(Cuboid, Pose, Vec<usize>, f32)>,
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
        .as_chunks::<3>()
        .0
        .iter()
        .map(|v| Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32))
        .collect();
    let indices: Vec<[u32; 3]> = mesh.faces.as_chunks::<3>().0.to_vec();
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

/// Margin of the convex-hull tests of `check_focus`: parry's hulls can fall about a
/// millimetre short of their meshes (its quickhull tolerances), so hulls closer than this
/// count as touching. Keeps the test conservative.
pub const HULL_MARGIN: f32 = 0.005;

impl ExactChecker {
    /// `robot_meshes`: `{"links": [{"link": i, "vertices": [...], "faces": [...]}]}` in link coordinates.
    pub fn new(model: RobotModel, robot_meshes: &str) -> Result<Self, String> {
        let parsed: RobotMeshesJson =
            serde_json::from_str(robot_meshes).map_err(|e| e.to_string())?;
        let mut links: Vec<Option<Body>> = (0..model.links.len()).map(|_| None).collect();
        let mut hulls: Vec<Option<ConvexPolyhedron>> =
            (0..model.links.len()).map(|_| None).collect();
        let mut hull_bounds = vec![(Vec3::ZERO, 0.0f32); model.links.len()];
        for m in &parsed.links {
            if m.link >= links.len() {
                return Err(format!("mesh for unknown link {}", m.link));
            }
            links[m.link] = Some(trimesh_body(m)?);
            let points: Vec<Vec3> = m
                .vertices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|v| Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32))
                .collect();
            hulls[m.link] = ConvexPolyhedron::from_convex_hull(&points);
            let n = points.len().max(1) as f32;
            let c = points.iter().fold(Vec3::ZERO, |a, p| a + *p) / n;
            let r = points
                .iter()
                .map(|p| (*p - c).length())
                .fold(0.0f32, f32::max);
            hull_bounds[m.link] = (c, r);
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
            focus: Vec::new(),
            focus_pairs: Vec::new(),
            hulls,
            hull_bounds,
            focus_boxes: Vec::new(),
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

    /// Links for `check_focus`: those the sphere model approximates poorly. Links without a
    /// mesh (or a convex hull) are ignored.
    pub fn set_focus(&mut self, links: &[usize]) {
        self.focus = links
            .iter()
            .copied()
            .filter(|&l| self.hulls.get(l).is_some_and(|h| h.is_some()))
            .collect();
        self.focus_pairs = self
            .pairs
            .iter()
            .copied()
            .filter(|(a, b)| self.focus.contains(a) || self.focus.contains(b))
            .filter(|(a, b)| self.hulls[*a].is_some() && self.hulls[*b].is_some())
            .collect();
    }

    /// The scene's boxes, which `check_focus` tests the focus links against.
    pub fn set_focus_obstacles(&mut self, scene: &Scene) {
        self.focus_boxes = scene
            .cuboids
            .iter()
            .map(|c| {
                let half = Vec3::new(
                    c.half_extents[0] as f32,
                    c.half_extents[1] as f32,
                    c.half_extents[2] as f32,
                );
                let touch = c
                    .touch_links
                    .iter()
                    .filter_map(|n| self.link_index(n))
                    .collect();
                (Cuboid::new(half), pose(&c.frame), touch, half.length())
            })
            .collect();
    }

    pub fn has_focus(&self) -> bool {
        !self.focus.is_empty()
    }

    /// True if a focus link may collide at `q`, tested on convex hulls (conservative: a
    /// hull contains its mesh, and hulls within `HULL_MARGIN` count as touching): with
    /// another link's hull, or with a scene box it may not touch. Fast enough to run on every configuration a planner checks (convex-convex
    /// tests, unlike the mesh-mesh tests of `check`); obstacles given only as meshes (the
    /// ESDF) are left to the final `check`.
    pub fn check_focus(&self, q: &[f64]) -> bool {
        if self.focus.is_empty() {
            return false;
        }
        let frames = forward_kinematics(&self.model, q);
        let poses: Vec<Pose> = frames.iter().map(pose).collect();
        let hull = |i: usize| self.hulls[i].as_ref().unwrap();
        // Bounding sphere of link `i`'s hull, in world coordinates.
        let bound = |i: usize| {
            let (c, r) = self.hull_bounds[i];
            (poses[i].transform_point(c), r)
        };
        for &(a, b) in &self.focus_pairs {
            let ((ca, ra), (cb, rb)) = (bound(a), bound(b));
            if (ca - cb).length() > ra + rb + HULL_MARGIN {
                continue;
            }
            if distance(&poses[a], hull(a), &poses[b], hull(b))
                .is_ok_and(|d| d.distance <= HULL_MARGIN)
            {
                return true;
            }
        }
        for &i in &self.focus {
            let (ci, ri) = bound(i);
            for (cuboid, box_pose, touch, rb) in &self.focus_boxes {
                if touch.contains(&i)
                    || (ci - box_pose.translation).length() > ri + rb + HULL_MARGIN
                {
                    continue;
                }
                if distance(&poses[i], hull(i), box_pose, cuboid)
                    .is_ok_and(|d| d.distance <= HULL_MARGIN)
                {
                    return true;
                }
            }
        }
        false
    }

    /// Collision flags for many configurations, checked in parallel.
    pub fn check_many(&self, configs: &[Vec<f64>]) -> Vec<Option<Contact>> {
        use rayon::prelude::*;
        configs.par_iter().map(|q| self.check(q)).collect()
    }

    /// Replace the link transforms (same links, e.g. after `RobotModel::set_passive_joints`).
    pub fn set_links(&mut self, links: Vec<motion_core::model::Link>) {
        assert_eq!(links.len(), self.model.links.len());
        self.model.links = links;
    }

    pub fn model(&self) -> &RobotModel {
        &self.model
    }
}

/// Compose helper re-exported for tests.
pub fn compose(a: &Mat34, b: &Mat34) -> Mat34 {
    mul(a, b)
}

// ---------------------------------------------------------------------------
// Offline ESDF from meshes

/// Signed distance field of obstacle meshes (robot coordinates) in the sparse block layout.
/// Blocks whose centre is farther than `truncation` + half a block diagonal from every mesh
/// are skipped. Inside is negative (meshes must be closed for the sign to be meaningful).
pub fn build_esdf(
    json: &str,
    voxel: f32,
    truncation: f32,
) -> Result<motion_core::esdf::VoxelEsdf, String> {
    use motion_core::layout::{VOXELS_PER_BLOCK, VOXEL_BLOCK_DIM};
    use parry3d::bounding_volume::BoundingVolume;
    use rayon::prelude::*;

    let parsed: ObstacleMeshesJson = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let bodies: Vec<Body> = parsed
        .obstacles
        .iter()
        .map(trimesh_body)
        .collect::<Result<_, _>>()?;
    let mut esdf = motion_core::esdf::VoxelEsdf::new(voxel, truncation);
    if bodies.is_empty() {
        return Ok(esdf);
    }
    let mut aabb = bodies[0].shape.compute_local_aabb();
    for b in &bodies[1..] {
        aabb.merge(&b.shape.compute_local_aabb());
    }
    let dim = VOXEL_BLOCK_DIM as i32;
    let block = voxel * dim as f32;
    let lo = (aabb.mins - Vec3::splat(truncation)) / block;
    let hi = (aabb.maxs + Vec3::splat(truncation)) / block;
    let (lo, hi) = (lo.floor().as_ivec3(), hi.ceil().as_ivec3());
    let signed = |p: Vec3| -> f32 {
        bodies
            .iter()
            .map(|b| {
                let proj = b.shape.project_local_point(p, b.solid);
                let d = (proj.point - p).length();
                if proj.is_inside && b.solid {
                    -d
                } else {
                    d
                }
            })
            .fold(f32::INFINITY, f32::min)
    };
    let half_diag = 0.5 * block * 3f32.sqrt();
    let coords: Vec<[i32; 3]> = (lo.x..hi.x)
        .flat_map(|x| (lo.y..hi.y).flat_map(move |y| (lo.z..hi.z).map(move |z| [x, y, z])))
        .collect();
    let blocks: Vec<([i32; 3], Vec<f32>)> = coords
        .par_iter()
        .filter_map(|&c| {
            let centre = Vec3::new(c[0] as f32 + 0.5, c[1] as f32 + 0.5, c[2] as f32 + 0.5) * block;
            if signed(centre).abs() > truncation + half_diag {
                return None;
            }
            let mut d = vec![0.0f32; VOXELS_PER_BLOCK as usize];
            for z in 0..dim {
                for y in 0..dim {
                    for x in 0..dim {
                        let g = Vec3::new(
                            (c[0] * dim + x) as f32,
                            (c[1] * dim + y) as f32,
                            (c[2] * dim + z) as f32,
                        );
                        d[(x + dim * (y + dim * z)) as usize] = signed(g * voxel).min(truncation);
                    }
                }
            }
            Some((c, d))
        })
        .collect();
    for (c, d) in blocks {
        esdf.insert_block(c, &d);
    }
    Ok(esdf)
}
