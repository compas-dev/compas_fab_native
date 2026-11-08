"""Kinematics helpers on top of the native core for servoing: forward kinematics, a numeric
Jacobian and a damped-least-squares servo step. (IK is the backend's own:
``core.numeric_inverse_kinematics``.)

Everything works in robot coordinates (the demo robots stand at the world origin).
"""

import numpy as np
from scipy.spatial.transform import Rotation


def m34(values) -> np.ndarray:
    """4x4 matrix from the core's row-major 3x4 frame."""
    m = np.eye(4)
    m[:3, :] = np.asarray(values, dtype=float).reshape(3, 4)
    return m


def pose_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """6-vector (translation, rotation vector) taking ``current`` to ``target``, in world axes."""
    e = np.empty(6)
    e[:3] = target[:3, 3] - current[:3, 3]
    e[3:] = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
    return e


class Arm:
    """A planning group of the native core seen as a serial arm with a tool centre point.

    Parameters
    ----------
    core
        ``compas_fab_native.NativeCore``.
    lower, upper, vmax
        Joint limits and velocity limits, in the core's joint order.
    tcf
        End-effector link -> tool centre point (4x4); identity by default.
    """

    def __init__(self, core, lower, upper, vmax, tcf=None):
        self.core = core
        self.ee = core.end_effector_link
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.vmax = np.asarray(vmax, dtype=float)
        self.tcf = np.eye(4) if tcf is None else np.asarray(tcf, dtype=float)
        self.dof = len(self.lower)

    def fk(self, q) -> np.ndarray:
        return m34(self.core.link_frames([float(x) for x in q])[self.ee]) @ self.tcf

    def jacobian(self, q, current=None, eps=1e-6) -> np.ndarray:
        """Numeric geometric Jacobian of the TCP (linear rows first, world axes)."""
        q = np.asarray(q, dtype=float)
        current = self.fk(q) if current is None else current
        jac = np.zeros((6, self.dof))
        for i in range(self.dof):
            dq = q.copy()
            dq[i] += eps
            moved = self.fk(dq)
            jac[:3, i] = (moved[:3, 3] - current[:3, 3]) / eps
            dr = moved[:3, :3] @ current[:3, :3].T
            jac[3:, i] = np.array([dr[2, 1] - dr[1, 2], dr[0, 2] - dr[2, 0], dr[1, 0] - dr[0, 1]]) / (2 * eps)
        return jac

    def clip(self, q) -> np.ndarray:
        return np.clip(q, self.lower, self.upper)

    def servo_step(self, q, command, dt, home=None, damping=0.03, home_gain=0.5):
        """Joint step that moves the TCP towards ``command`` (4x4) within one period.

        Damped least squares, with the redundancy (7-axis arms) pulling towards ``home``,
        and joint speeds clamped to their limits."""
        q = np.asarray(q, dtype=float)
        current = self.fk(q)
        e = pose_error(command, current)
        jac = self.jacobian(q, current)
        jjt = jac @ jac.T + damping**2 * np.eye(6)
        dq = jac.T @ np.linalg.solve(jjt, e)
        if home is not None and self.dof > 6:
            pinv = jac.T @ np.linalg.inv(jjt)
            null = np.eye(self.dof) - pinv @ jac
            dq += null @ (home_gain * dt * (np.asarray(home) - q))
        limit = self.vmax * dt
        scale = np.max(np.abs(dq) / limit)
        if scale > 1.0:
            dq /= scale
        return self.clip(q + dq)
