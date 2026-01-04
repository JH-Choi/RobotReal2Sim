import numpy as np
import math

def euler_xyz_to_matrix(rx_deg, ry_deg, rz_deg):
    """Create 3x3 rotation matrix from XYZ Euler angles in degrees."""
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx),  math.cos(rx)],
    ])

    Ry = np.array([
        [ math.cos(ry), 0, math.sin(ry)],
        [0,             1,           0],
        [-math.sin(ry), 0, math.cos(ry)],
    ])

    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz),  math.cos(rz), 0],
        [0,                        0, 1],
    ])

    # XYZ order: first Rx, then Ry, then Rz in world frame
    R = Rz @ Ry @ Rx
    return R

def build_transform(tx, ty, tz, rx, ry, rz, sx, sy, sz):
    # From your screenshot
    R = euler_xyz_to_matrix(rx, ry, rz)

    # Apply uniform or per-axis scale on the rotation part
    S = np.diag([sx, sy, sz])
    RS = R @ S

    # 4x4 homogeneous matrix
    M = np.eye(4)
    M[:3, :3] = RS
    M[:3, 3] = np.array([tx, ty, tz])
    return M

def transform_point_cloud(points, tx, ty, tz, rx, ry, rz, sx, sy, sz):
    """
    points: (N, 3) NumPy array
    returns transformed points: (N, 3)
    """
    M = build_transform(tx, ty, tz, rx, ry, rz, sx, sy, sz)
    N = points.shape[0]
    ones = np.ones((N, 1), dtype=points.dtype)
    pts_h = np.hstack([points, ones])          # (N, 4)
    pts_out_h = (M @ pts_h.T).T                # (N, 4)
    return pts_out_h[:, :3]

# Example usage:
if __name__ == "__main__":
    # Dummy point cloud
    pts = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    tx, ty, tz = 0.0, 3.23, 0.0
    rx, ry, rz = -23.31, 0.0, 180.0
    sx, sy, sz = 1.0, 1.0, 1.0

    transformed = transform_point_cloud(pts, tx, ty, tz, rx, ry, rz, sx, sy, sz)
    print("Transform matrix:\n", build_transform(tx, ty, tz, rx, ry, rz, sx, sy, sz))
    print("Transformed points:\n", transformed)
