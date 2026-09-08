#pragma once
#include <cuda_runtime.h>
#include <cmath>

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3(float x_ = 0, float y_ = 0, float z_ = 0) : x(x_), y(y_), z(z_) {}
    
    __host__ __device__ Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
};

struct Mat4x4 {
    float m[4][4];

    __host__ __device__ static Mat4x4 identity() {
        Mat4x4 res;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                res.m[i][j] = (i == j) ? 1.0f : 0.0f;
        return res;
    }

    __host__ __device__ static Mat4x4 dh_transform(float a, float alpha, float d, float theta) {
        Mat4x4 T;
        float ct = cosf(theta);
        float st = sinf(theta);
        float ca = cosf(alpha);
        float sa = sinf(alpha);

        T.m[0][0] = ct;       T.m[0][1] = -st * ca; T.m[0][2] = st * sa;  T.m[0][3] = a * ct;
        T.m[1][0] = st;       T.m[1][1] = ct * ca;  T.m[1][2] = -ct * sa; T.m[1][3] = a * st;
        T.m[2][0] = 0.0f;     T.m[2][1] = sa;       T.m[2][2] = ca;       T.m[2][3] = d;
        T.m[3][0] = 0.0f;     T.m[3][1] = 0.0f;     T.m[3][2] = 0.0f;     T.m[3][3] = 1.0f;
        return T;
    }

    __host__ __device__ Mat4x4 operator*(const Mat4x4& o) const {
        Mat4x4 res;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                res.m[i][j] = 0.0f;
                for (int k = 0; k < 4; ++k)
                    res.m[i][j] += m[i][k] * o.m[k][j];
            }
        }
        return res;
    }
};

struct Capsule {
    Vec3 p0;
    Vec3 p1;
    float radius;
};

// Modèle DH d'un bras 6-DDL
struct DHParam {
    float a;
    float alpha;
    float d;
};

__constant__ DHParam c_dh_params[6] = {
    {0.0f,   1.5707963f, 0.25f},
    {0.35f,  0.0f,       0.0f},
    {0.30f,  0.0f,       0.0f},
    {0.0f,   1.5707963f, 0.10f},
    {0.0f,  -1.5707963f, 0.10f},
    {0.0f,   0.0f,       0.08f}
};
