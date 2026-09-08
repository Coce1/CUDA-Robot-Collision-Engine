#include <iostream>
#include <vector>
#include <chrono>
#include "kinematics.cuh"
#include "collision_kernels.cu"

int main() {
    const int NUM_VOXELS = 1000000;
    const int NUM_CAPSULES = 6;

    std::cout << "=== CUDA Robotic Real-Time Collision Engine ===" << std::endl;
    std::cout << "Testing " << NUM_VOXELS << " obstacle voxels against a 6-DOF manipulator..." << std::endl;

    std::vector<Capsule> h_capsules(NUM_CAPSULES);
    for (int i = 0; i < NUM_CAPSULES; ++i) {
        h_capsules[i].p0 = Vec3(0.0f, 0.0f, (float)i * 0.3f);
        h_capsules[i].p1 = Vec3(0.0f, 0.0f, (float)(i + 1) * 0.3f);
        h_capsules[i].radius = 0.08f;
    }

    std::vector<Vec3> h_voxels(NUM_VOXELS);
    for (int i = 0; i < NUM_VOXELS; ++i) {
        h_voxels[i] = Vec3(
            ((float)rand() / RAND_MAX - 0.5f) * 2.0f,
            ((float)rand() / RAND_MAX - 0.5f) * 2.0f,
            ((float)rand() / RAND_MAX) * 2.0f
        );
    }

    Vec3 *d_voxels;
    Capsule *d_capsules;
    int *d_collision_flag;
    int h_collision_flag = 0;

    cudaMalloc(&d_voxels, NUM_VOXELS * sizeof(Vec3));
    cudaMalloc(&d_capsules, NUM_CAPSULES * sizeof(Capsule));
    cudaMalloc(&d_collision_flag, sizeof(int));

    cudaMemcpy(d_voxels, h_voxels.data(), NUM_VOXELS * sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_capsules, h_capsules.data(), NUM_CAPSULES * sizeof(Capsule), cudaMemcpyHostToDevice);
    cudaMemcpy(d_collision_flag, &h_collision_flag, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (NUM_VOXELS + threads - 1) / threads;

    cudaEventRecord(start);
    checkVoxelCollisionKernel<<<blocks, threads>>>(d_voxels, NUM_VOXELS, d_capsules, NUM_CAPSULES, d_collision_flag);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(&h_collision_flag, d_collision_flag, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Collision status : " << (h_collision_flag ? "COLLISION DETECTED" : "CLEAR") << std::endl;
    std::cout << "GPU Execution time: " << ms << " ms (" << (NUM_VOXELS / (ms * 1e3f)) << " Giga-tests/sec)" << std::endl;

    cudaFree(d_voxels);
    cudaFree(d_capsules);
    cudaFree(d_collision_flag);
    return 0;
}
