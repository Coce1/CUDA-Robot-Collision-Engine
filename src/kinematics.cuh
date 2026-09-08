#pragma once
#include <cuda_runtime.h>
#include <cmath>

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3(float x_ = 0, float y_ = 0, float z_ = 0) : x(x_), y(y_), z(z_) {}
};

struct Mat4x4 {
    float m[4][4];

    __host__ __device__ static Mat4x4 identity() {
        Mat4x4 res = {0};
        for (int i = 0; i < 4; ++i) res.m[i][i] = 1.0f;
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
        Mat4x4 res = {0};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    res.m[i][j] += m[i][k] * o.m[k][j];
        return res;
    }
};

struct Capsule {
    Vec3 p0;
    Vec3 p1;
    float radius;
};
