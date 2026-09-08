#include <iostream>
#include <vector>
#include "kinematics.cuh"
#include "collision_kernels.cu"

int main() {
    const int NUM_POSES = 2000;      // 2 000 configurations de trajectoire
    const int NUM_VOXELS = 100000;   // 100 000 voxels d'obstacles

    std::cout << "========================================================" << std::endl;
    std::cout << "  GPU-Accelerated Robot Batch Trajectory Collision Engine" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << "Trajectory Waypoints : " << NUM_POSES << " poses (6-DOF each)" << std::endl;
    std::cout << "Obstacle Cloud       : " << NUM_VOXELS << " voxels" << std::endl;
    std::cout << "Total Pairwise Checks: " << (long long)NUM_POSES * NUM_VOXELS * 6 << " capsule-voxel tests" << std::endl;

    // 1. Génération de trajectoires articulaires
    std::vector<float> h_trajectories(NUM_POSES * 6);
    for (int p = 0; p < NUM_POSES; ++p) {
        float t = (float)p / NUM_POSES;
        for (int j = 0; j < 6; ++j) {
            h_trajectories[p * 6 + j] = sinf(t * 6.28318f + j * 0.5f);
        }
    }

    // 2. Génération de la grille d'obstacles
    std::vector<Vec3> h_voxels(NUM_VOXELS);
    for (int i = 0; i < NUM_VOXELS; ++i) {
        h_voxels[i] = Vec3(
            ((float)rand() / RAND_MAX - 0.5f) * 1.6f,
            ((float)rand() / RAND_MAX - 0.5f) * 1.6f,
            ((float)rand() / RAND_MAX) * 1.2f
        );
    }

    // 3. Allocations GPU
    float* d_trajectories;
    Vec3* d_voxels;
    int* d_collision_flags;

    cudaMalloc(&d_trajectories, NUM_POSES * 6 * sizeof(float));
    cudaMalloc(&d_voxels, NUM_VOXELS * sizeof(Vec3));
    cudaMalloc(&d_collision_flags, NUM_POSES * sizeof(int));

    cudaMemcpy(d_trajectories, h_trajectories.data(), NUM_POSES * 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_voxels, h_voxels.data(), NUM_VOXELS * sizeof(Vec3), cudaMemcpyHostToDevice);

    // 4. Exécution & Profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads_per_block = 256;
    int blocks = NUM_POSES;

    cudaEventRecord(start);
    batchTrajectoryCollisionKernel<<<blocks, threads_per_block>>>(
        d_trajectories, NUM_POSES, d_voxels, NUM_VOXELS, d_collision_flags
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::vector<int> h_flags(NUM_POSES);
    cudaMemcpy(h_flags.data(), d_collision_flags, NUM_POSES * sizeof(int), cudaMemcpyDeviceToHost);

    int collision_count = 0;
    for (int f : h_flags) if (f == 1) collision_count++;

    std::cout << "\n>>> BENCHMARK RESULTS <<<" << std::endl;
    std::cout << "Execution Time       : " << ms << " ms" << std::endl;
    std::cout << "Latency per Waypoint : " << (ms / NUM_POSES) * 1000.0f << " microseconds" << std::endl;
    std::cout << "Collision Poses Found: " << collision_count << " / " << NUM_POSES << std::endl;
    std::cout << "Throughput           : " << ((double)NUM_POSES * NUM_VOXELS * 6) / (ms * 1e6) << " Billion tests/sec" << std::endl;

    cudaFree(d_trajectories);
    cudaFree(d_voxels);
    cudaFree(d_collision_flags);
    return 0;
}
