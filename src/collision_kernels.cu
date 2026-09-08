#pragma once
#include "kinematics.cuh"

__device__ inline float distancePointToSegmentSq(Vec3 p, Vec3 a, Vec3 b) {
    float abx = b.x - a.x, aby = b.y - a.y, abz = b.z - a.z;
    float apx = p.x - a.x, apy = p.y - a.y, apz = p.z - a.z;

    float ab_sq = abx * abx + aby * aby + abz * abz;
    if (ab_sq < 1e-6f) return apx * apx + apy * apy + apz * apz;

    float t = (apx * abx + apy * aby + apz * abz) / ab_sq;
    t = fmaxf(0.0f, fminf(1.0f, t));

    float qx = a.x + t * abx;
    float qy = a.y + t * aby;
    float qz = a.z + t * abz;

    float dx = p.x - qx, dy = p.y - qy, dz = p.z - qz;
    return dx * dx + dy * dy + dz * dz;
}

// Cinématique directe sur GPU
__device__ void computeFKDevice(const float* angles, Capsule* capsules) {
    Mat4x4 T = Mat4x4::identity();
    Vec3 prev_pos(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 6; ++i) {
        Mat4x4 T_i = Mat4x4::dh_transform(c_dh_params[i].a, c_dh_params[i].alpha, c_dh_params[i].d, angles[i]);
        T = T * T_i;
        Vec3 curr_pos(T.m[0][3], T.m[1][3], T.m[2][3]);
        
        capsules[i].p0 = prev_pos;
        capsules[i].p1 = curr_pos;
        capsules[i].radius = 0.06f;
        prev_pos = curr_pos;
    }
}

// Kernel d'évaluation de collision par lot (Batch Trajectory Checker)
__global__ void batchTrajectoryCollisionKernel(
    const float* __restrict__ trajectory_angles, // [num_poses * 6]
    int num_poses,
    const Vec3* __restrict__ voxels,
    int num_voxels,
    int* __restrict__ pose_collision_flags       // [num_poses]
) {
    int pose_idx = blockIdx.x;
    if (pose_idx >= num_poses) return;

    // Calcul des capsules pour cette pose spécifique
    __shared__ Capsule s_capsules[6];
    if (threadIdx.x == 0) {
        computeFKDevice(&trajectory_angles[pose_idx * 6], s_capsules);
        pose_collision_flags[pose_idx] = 0;
    }
    __syncthreads();

    // Vérification parallèle des voxels contre les capsules de cette pose
    for (int v_idx = threadIdx.x; v_idx < num_voxels; v_idx += blockDim.x) {
        if (pose_collision_flags[pose_idx] != 0) break;

        Vec3 voxel = voxels[v_idx];
        for (int c = 0; c < 6; ++c) {
            float r = s_capsules[c].radius;
            if (distancePointToSegmentSq(voxel, s_capsules[c].p0, s_capsules[c].p1) <= (r * r)) {
                atomicExch(&pose_collision_flags[pose_idx], 1);
                break;
            }
        }
    }
}
