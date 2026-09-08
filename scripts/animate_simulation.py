import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def dh_matrix(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ])

def compute_fk(thetas):
    dh_params = [
        (0.0,   np.pi / 2, 0.25),
        (0.35,  0.0,       0.0),
        (0.30,  0.0,       0.0),
        (0.0,   np.pi / 2, 0.10),
        (0.0,  -np.pi / 2, 0.10),
        (0.0,   0.0,       0.08)
    ]
    positions = [np.array([0.0, 0.0, 0.0])]
    T = np.eye(4)
    for i, (a, alpha, d) in enumerate(dh_params):
        T = T @ dh_matrix(a, alpha, d, thetas[i])
        positions.append(T[:3, 3])
    return positions

def check_collision(pt, p0, p1, radius):
    ab = p1 - p0
    ap = pt - p0
    t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0.0, 1.0)
    proj = p0 + t * ab
    return np.linalg.norm(pt - proj) <= radius

print("🎬 Generating trajectory animation...")
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

np.random.seed(42)
voxels = np.random.uniform(low=[-0.6, -0.6, 0.0], high=[0.6, 0.6, 0.9], size=(800, 3))
num_frames = 60

def update(frame):
    ax.cla()
    t = frame / num_frames
    thetas = [
        0.5 * np.sin(2 * np.pi * t),
        -0.4 + 0.3 * np.cos(2 * np.pi * t),
        0.8 + 0.4 * np.sin(2 * np.pi * t),
        -0.5 * np.cos(2 * np.pi * t),
        0.3 * np.sin(2 * np.pi * t),
        0.0
    ]
    
    joints = compute_fk(thetas)
    capsules = [(joints[i], joints[i+1], 0.06) for i in range(len(joints)-1)]
    
    # Détection collision des voxels
    colors = []
    has_collision = False
    for pt in voxels:
        col = any(check_collision(pt, p0, p1, r) for p0, p1, r in capsules)
        if col:
            has_collision = True
            colors.append('crimson')
        else:
            colors.append('mediumseagreen')
            
    ax.scatter(voxels[:, 0], voxels[:, 1], voxels[:, 2], c=colors, s=8, alpha=0.35)
    
    # Tracer le bras
    joint_arr = np.array(joints)
    arm_color = 'red' if has_collision else 'dodgerblue'
    ax.plot(joint_arr[:, 0], joint_arr[:, 1], joint_arr[:, 2], '-o', color=arm_color, linewidth=5, markersize=7)
    
    ax.set_title(f"Real-Time Collision State: {'COLLISION' if has_collision else 'CLEAR'}", 
                 fontsize=12, fontweight='bold', color='crimson' if has_collision else 'green')
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([0.0, 1.0])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

anim = FuncAnimation(fig, update, frames=num_frames, interval=50)
anim.save("scripts/robot_simulation.gif", writer='pillow', fps=20)
print("✅ Saved animation to scripts/robot_simulation.gif")
