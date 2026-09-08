#pragma once
#include "kinematics.cuh"

__device__ float distancePointToSegmentSq(Vec3 p, Vec3 a, Vec3 b) {
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

__global__ void checkVoxelCollisionKernel(
    const Vec3* __restrict__ voxels,
    int num_voxels,
    const Capsule* __restrict__ capsules,
    int num_capsules,
    int* __restrict__ collision_flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_voxels || *collision_flag != 0) return;

    Vec3 voxel = voxels[idx];

    for (int i = 0; i < num_capsules; ++i) {
        float r = capsules[i].radius;
        if (distancePointToSegmentSq(voxel, capsules[i].p0, capsules[i].p1) <= (r * r)) {
            atomicExch(collision_flag, 1);
            return;
        }
    }
}
