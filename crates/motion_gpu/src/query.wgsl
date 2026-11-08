// Batched FK + link spheres + sphere/cuboid distance query.
// Byte-identical twin of motion_core::layout; arithmetic mirrors motion_core::cpu::query_scalar.
// Written against WebGPU core limits: f32 only, no atomics, no f16.

struct Params {
    batch: u32,
    dof: u32,
    n_links: u32,
    n_spheres: u32,
    n_cuboids: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

struct Link {
    origin: array<f32, 12>,
    axis: array<f32, 3>,
    parent: i32,
    joint_type: u32,
    q_index: i32,
    _p0: u32,
    _p1: u32,
}

struct Sphere {
    center: array<f32, 3>,
    radius: f32,
    link: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

struct Cuboid {
    inv: array<f32, 12>,
    half: array<f32, 3>,
    _p0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> links: array<Link>;
@group(0) @binding(2) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(3) var<storage, read> cuboids: array<Cuboid>;
@group(0) @binding(4) var<storage, read> q: array<f32>;
@group(0) @binding(5) var<storage, read_write> spheres_world: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> sphere_dist: array<f32>;
@group(0) @binding(7) var<storage, read_write> sphere_grad: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> config_min: array<f32>;

const MAX_LINKS: u32 = 32u;
const WG: u32 = 64u;
const INF: f32 = 3.0e38;

// 3x4 transform as three rows: [R | t].
struct M34 {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

fn mul34(a: M34, b: M34) -> M34 {
    let t = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return M34(
        a.r0.x * b.r0 + a.r0.y * b.r1 + a.r0.z * b.r2 + a.r0.w * t,
        a.r1.x * b.r0 + a.r1.y * b.r1 + a.r1.z * b.r2 + a.r1.w * t,
        a.r2.x * b.r0 + a.r2.y * b.r1 + a.r2.z * b.r2 + a.r2.w * t,
    );
}

fn apply(m: M34, p: vec3<f32>) -> vec3<f32> {
    let h = vec4<f32>(p, 1.0);
    return vec3<f32>(dot(m.r0, h), dot(m.r1, h), dot(m.r2, h));
}

fn link_origin(i: u32) -> M34 {
    let o = links[i].origin;
    return M34(
        vec4<f32>(o[0], o[1], o[2], o[3]),
        vec4<f32>(o[4], o[5], o[6], o[7]),
        vec4<f32>(o[8], o[9], o[10], o[11]),
    );
}

fn local_transform(i: u32, value: f32) -> M34 {
    let o = link_origin(i);
    let kind = links[i].joint_type;
    let a = vec3<f32>(links[i].axis[0], links[i].axis[1], links[i].axis[2]);
    if (kind == 1u) {
        let s = sin(value);
        let c = cos(value);
        let t = 1.0 - c;
        let r = M34(
            vec4<f32>(t * a.x * a.x + c, t * a.x * a.y - s * a.z, t * a.x * a.z + s * a.y, 0.0),
            vec4<f32>(t * a.x * a.y + s * a.z, t * a.y * a.y + c, t * a.y * a.z - s * a.x, 0.0),
            vec4<f32>(t * a.x * a.z - s * a.y, t * a.y * a.z + s * a.x, t * a.z * a.z + c, 0.0),
        );
        return mul34(o, r);
    }
    if (kind == 2u) {
        var m = o;
        m.r0.w += dot(o.r0.xyz, a) * value;
        m.r1.w += dot(o.r1.xyz, a) * value;
        m.r2.w += dot(o.r2.xyz, a) * value;
        return m;
    }
    return o;
}

// xyz: gradient of the signed distance in world coordinates, w: signed distance.
fn sphere_box(p: vec3<f32>, i: u32) -> vec4<f32> {
    let m = cuboids[i].inv;
    let r0 = vec4<f32>(m[0], m[1], m[2], m[3]);
    let r1 = vec4<f32>(m[4], m[5], m[6], m[7]);
    let r2 = vec4<f32>(m[8], m[9], m[10], m[11]);
    let h4 = vec4<f32>(p, 1.0);
    let l = vec3<f32>(dot(r0, h4), dot(r1, h4), dot(r2, h4));
    let half = vec3<f32>(cuboids[i].half[0], cuboids[i].half[1], cuboids[i].half[2]);
    let qv = abs(l) - half;
    let o = max(qv, vec3<f32>(0.0));
    let outside = length(o);
    let inside = min(max(qv.x, max(qv.y, qv.z)), 0.0);
    let s = select(vec3<f32>(1.0), vec3<f32>(-1.0), l < vec3<f32>(0.0));
    var g: vec3<f32>;
    if (outside > 0.0) {
        g = s * o * (1.0 / outside);
    } else if (qv.x >= qv.y && qv.x >= qv.z) {
        g = vec3<f32>(s.x, 0.0, 0.0);
    } else if (qv.y >= qv.z) {
        g = vec3<f32>(0.0, s.y, 0.0);
    } else {
        g = vec3<f32>(0.0, 0.0, s.z);
    }
    let gw = g.x * r0.xyz + g.y * r1.xyz + g.z * r2.xyz;
    return vec4<f32>(gw, outside + inside);
}

fn nearest(p: vec3<f32>) -> vec4<f32> {
    var best = vec4<f32>(0.0, 0.0, 0.0, INF);
    for (var k = 0u; k < params.n_cuboids; k++) {
        let d = sphere_box(p, k);
        if (d.w < best.w) {
            best = d;
        }
    }
    return best;
}

// Flattened invocation index for 2D dispatches (x is capped below the 65535 workgroup limit).
fn flat(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * WG;
}

var<private> frames: array<M34, MAX_LINKS>;

fn fk(b: u32) {
    let batch = params.batch;
    for (var i = 0u; i < params.n_links; i++) {
        let qi = links[i].q_index;
        var value = 0.0;
        if (qi >= 0) {
            value = q[u32(qi) * batch + b];
        }
        let local = local_transform(i, value);
        let parent = links[i].parent;
        if (parent < 0) {
            frames[i] = local;
        } else {
            frames[i] = mul34(frames[u32(parent)], local);
        }
    }
}

fn sphere_world(s: u32) -> vec4<f32> {
    let sp = spheres[s];
    return vec4<f32>(apply(frames[sp.link], vec3<f32>(sp.center[0], sp.center[1], sp.center[2])), sp.radius);
}

// --- Fused: one thread per configuration ------------------------------------

@compute @workgroup_size(64)
fn fused_full(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let b = flat(gid, nwg);
    let batch = params.batch;
    if (b >= batch) {
        return;
    }
    fk(b);
    var m = INF;
    for (var s = 0u; s < params.n_spheres; s++) {
        let sw = sphere_world(s);
        let best = nearest(sw.xyz);
        let d = best.w - sw.w;
        m = min(m, d);
        sphere_dist[s * batch + b] = d;
        sphere_grad[s * batch + b] = vec4<f32>(best.xyz, 0.0);
    }
    config_min[b] = m;
}

@compute @workgroup_size(64)
fn fused_min(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let b = flat(gid, nwg);
    if (b >= params.batch) {
        return;
    }
    fk(b);
    var m = INF;
    for (var s = 0u; s < params.n_spheres; s++) {
        let sw = sphere_world(s);
        m = min(m, nearest(sw.xyz).w - sw.w);
    }
    config_min[b] = m;
}

// --- Split: three dispatches, intermediate results in global memory ---------

@compute @workgroup_size(64)
fn split_fk_spheres(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let b = flat(gid, nwg);
    let batch = params.batch;
    if (b >= batch) {
        return;
    }
    fk(b);
    for (var s = 0u; s < params.n_spheres; s++) {
        spheres_world[s * batch + b] = sphere_world(s);
    }
}

// One thread per (sphere, configuration).
@compute @workgroup_size(64)
fn split_distance(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat(gid, nwg);
    if (i >= params.batch * params.n_spheres) {
        return;
    }
    let sw = spheres_world[i];
    let best = nearest(sw.xyz);
    sphere_dist[i] = best.w - sw.w;
    sphere_grad[i] = vec4<f32>(best.xyz, 0.0);
}

@compute @workgroup_size(64)
fn split_reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let b = flat(gid, nwg);
    let batch = params.batch;
    if (b >= batch) {
        return;
    }
    var m = INF;
    for (var s = 0u; s < params.n_spheres; s++) {
        m = min(m, sphere_dist[s * batch + b]);
    }
    config_min[b] = m;
}
