"""Differentiable math helpers used by the SuperMimic learner."""

import torch


def rotvec_quat(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert rotation vectors to xyzw quaternions with a stable zero limit."""
    angle_sq = (rotvec * rotvec).sum(dim=-1, keepdim=True)
    angle = angle_sq.clamp_min(1e-12).sqrt()
    half = 0.5 * angle
    scale = torch.where(angle_sq < 1e-8, 0.5 - angle_sq / 48.0, torch.sin(half) / angle)
    return torch.cat((rotvec * scale, torch.cos(half)), dim=-1)


def quat_mul_xyzw(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Multiply xyzw quaternions."""
    x1, y1, z1, w1 = lhs.unbind(-1)
    x2, y2, z2, w2 = rhs.unbind(-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def quat_inverse(quat: torch.Tensor) -> torch.Tensor:
    """Conjugate a unit xyzw quaternion."""
    return torch.cat((-quat[..., :3], quat[..., 3:]), -1)


def quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors with broadcastable unit xyzw quaternions."""
    xyz, vector = torch.broadcast_tensors(quat[..., :3], vector)
    cross = 2 * torch.cross(xyz, vector, dim=-1)
    return vector + quat[..., 3:] * cross + torch.cross(xyz, cross, dim=-1)


def heading_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract world yaw, keeping the result in xyzw order."""
    x, y, z, w = quat.unbind(-1)
    half = 0.5 * torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    zero = torch.zeros_like(half)
    return torch.stack((zero, zero, half.sin(), half.cos()), -1)


def quat_rotvec(quat: torch.Tensor) -> torch.Tensor:
    """Shortest rotation vector with finite gradients at the identity."""
    quat = torch.where(quat[..., 3:] < 0, -quat, quat)
    xyz = quat[..., :3]
    norm = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    scale = 2 * torch.atan2(norm, quat[..., 3:]) / norm.clamp_min(1e-8)
    return xyz * torch.where(norm < 1e-6, 2 + norm.square() / 3, scale)


def state_features(state: torch.Tensor) -> torch.Tensor:
    """Remove global x/y position and heading from the physical state."""
    quat = state[..., 3:7]
    inverse = quat_inverse(quat)
    tilt = quat_mul_xyzw(quat_inverse(heading_quat(quat)), quat)
    tilt = torch.where(tilt[..., 3:] < 0, -tilt, tilt)
    return torch.cat(
        (
            state[..., 2:3],
            tilt,
            quat_apply(inverse, state[..., 7:10]),
            quat_apply(inverse, state[..., 10:13]),
            state[..., 13:],
        ),
        -1,
    )


def state_difference(state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    """Frame-local position, rotation, velocity and joint increments (70-D for G1)."""
    quat = state[..., 3:7]
    inverse = quat_inverse(quat)
    return torch.cat(
        (
            quat_apply(quat_inverse(heading_quat(quat)), next_state[..., :3] - state[..., :3]),
            quat_rotvec(quat_mul_xyzw(inverse, next_state[..., 3:7])),
            quat_apply(inverse, next_state[..., 7:10] - state[..., 7:10]),
            quat_apply(inverse, next_state[..., 10:13] - state[..., 10:13]),
            next_state[..., 13:] - state[..., 13:],
        ),
        -1,
    )


def integrate_state(state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Integrate local increments on R^3 × SO(3) and retain unit quaternions."""
    quat = state[..., 3:7]
    return torch.cat(
        (
            state[..., :3] + quat_apply(heading_quat(quat), delta[..., :3]),
            torch.nn.functional.normalize(quat_mul_xyzw(quat, rotvec_quat(delta[..., 3:6])), dim=-1),
            state[..., 7:10] + quat_apply(quat, delta[..., 6:9]),
            state[..., 10:13] + quat_apply(quat, delta[..., 9:12]),
            state[..., 13:] + delta[..., 12:],
        ),
        -1,
    )
