//! Native motion core, Phase 0 spike: batched kinematics and the sphere
//! distance query on the CPU, plus the buffer layouts shared with the GPU.

pub mod cpu;
pub mod ik;
pub mod layout;
pub mod model;
pub mod reference;
pub mod sampling;

pub use layout::ModelBuffers;
pub use model::{load_robot, load_scene, RobotModel, Scene};
