import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Transformation matricielle Denavit-Hartenberg standard
def dh_matrix(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ])

# Calcul de la chaîne cinématique directe (Forward Kinematics)
def compute_forward_kinematics(joint_angles):
    # Paramètres DH classiques (type UR5 / manipulateur 6-DDL) : [a, alpha, d]
    dh_params = [
        (0.0,    np.pi / 2, 0.25),
        (0.35,   0.0,       0.0),
        (0.30,   0.0,       0.0),
        (0.0,    np.pi / 2, 0.10),
        (0.0,   -np.pi / 2, 0.10),
        (0.0,    0.0,       0.08)
    ]
    
    positions = [np.array([0.0, 0.0, 0.0])]
    T = np.eye(4)
    
    for i, (a, alpha, d) in enumerate(dh_params):
        T_i = dh_matrix(a, alpha, d, joint_angles[i])
        T = T @ T_i
        positions.append(T[:3, 3])
        
    return positions

def plot_capsule(ax, p0, p1, radius=0.06, resolution=10, color='royalblue', alpha=0.6):
    v = p1 - p0
    mag = np.linalg.norm(v)
    if mag < 1e-4:
        return
    v_norm = v / mag
    not_v = np.array([1, 0, 0]) if abs(v_norm[0]) < 0.9 else np.array([0, 1, 0])
    n1 = np.cross(v_norm, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v_norm, n1)

    theta = np.linspace(0, 2 * np.pi, resolution)
    s = np.linspace(0, mag, 2)
    theta_grid, s_grid = np.meshgrid(theta, s)

    x = p0[0] + v_norm[0] * s_grid + radius * (np.outer(np.ones(2), np.cos(theta)) * n1[0] + np.outer(np.ones(2), np.sin(theta)) * n2[0])
    y = p0[1] + v_norm[1] * s_grid + radius * (np.outer(np.ones(2), np.cos(theta)) * n1[1] + np.outer(np.ones(2), np.sin(theta)) * n2[1])
    z = p0[2] + v_norm[2] * s_grid + radius * (np.outer(np.ones(2), np.cos(theta)) * n1[2] + np.outer(np.ones(2), np.sin(theta)) * n2[2])

    ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=True)

# Génération de l'image de pose articulée avec collision
def render_articulated_scene():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Angles articulaires test (bras fléchi)
    thetas = [0.3, -0.6, 1.2, -0.8, 0.5, 0.0]
    joints = compute_forward_kinematics(thetas)

    capsules = []
    for i in range(len(joints) - 1):
        p0, p1 = joints[i], joints[i + 1]
        capsules.append((p0, p1, 0.06))
        plot_capsule(ax, p0, p1, radius=0.06, color='dodgerblue', alpha=0.55)

    # Nuage de voxels d'obstacles
    np.random.seed(42)
    voxels = np.random.uniform(low=[-0.6, -0.6, 0.0], high=[0.6, 0.6, 0.8], size=(1200, 3))
    colors = []
    
    for pt in voxels:
        collision = False
        for p0, p1, r in capsules:
            ab = p1 - p0
            ap = pt - p0
            t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0.0, 1.0)
            proj = p0 + t * ab
            if np.linalg.norm(pt - proj) <= r:
                collision = True
                break
        colors.append('crimson' if collision else 'mediumseagreen')

    ax.scatter(voxels[:, 0], voxels[:, 1], voxels[:, 2], c=colors, s=10, alpha=0.35)
    
    # Dessiner la structure du bras
    joint_arr = np.array(joints)
    ax.plot(joint_arr[:, 0], joint_arr[:, 1], joint_arr[:, 2], '-o', color='navy', linewidth=2, markersize=5)

    ax.set_title("CUDA Collision Engine - Articulated 6-DOF Manipulator", fontsize=13, fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([0.0, 1.0])

    plt.savefig("scripts/scene_preview.png", dpi=200, bbox_inches='tight')
    plt.show()
    print("✅ New articulated pose saved to scripts/scene_preview.png")

if __name__ == "__main__":
    render_articulated_scene()
