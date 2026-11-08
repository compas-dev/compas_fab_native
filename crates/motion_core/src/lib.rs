//! Native motion core, Phase 0 spike: batched kinematics and the sphere
//! distance query on the CPU, plus the buffer layouts shared with the GPU.

pub mod cartesian;
pub mod check;
pub mod cpu;
pub mod esdf;
pub mod freespace;
pub mod ik;
pub mod layout;
pub mod model;
pub mod reference;
pub mod sampling;
pub mod simd;
pub mod timing;

pub use layout::ModelBuffers;
pub use model::{load_robot, load_scene, RobotModel, Scene};
