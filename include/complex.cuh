#pragma once
#include <ostream>

struct complex
{
    double real;
    double img;

    __host__ __device__ complex(double _real, double _img) : real(_real), img(_img) {}
    __host__ __device__ complex(double _real) : complex(_real, 0) {}
    __host__ __device__ complex() : complex(0, 0) {}

    __host__ __device__ inline complex operator+(const complex &other) const
    {
        return complex(real + other.real, img + other.img);
    }

    __host__ __device__ inline complex operator-(const complex &other) const
    {
        return complex(real - other.real, img - other.img);
    }

    __host__ __device__ inline complex &operator/=(double scalar)
    {
        real /= scalar;
        img /= scalar;
        return *this;
    }

    __host__ __device__ inline complex &operator/=(int scalar)
    {
        real /= scalar;
        img /= scalar;
        return *this;
    }

    __host__ __device__ inline complex &operator*=(const complex &other)
    {
        double new_real = real * other.real - img * other.img;
        double new_img = real * other.img + img * other.real;
        real = new_real;
        img = new_img;
        return *this;
    }

    __host__ __device__ inline complex operator*(const complex &other) const
    {
        return complex(real * other.real - img * other.img, real * other.img + img * other.real);
    }

    __host__ __device__ bool inline operator==(const complex &other) const
    {
        return abs(real - other.real) < 1e-5 && abs(img - other.img) < 1e-5;
    }

    __host__ __device__ bool inline operator!=(const complex &other) const
    {
        return !operator==(other);
    }

    __host__ __device__ complex inline conj() const
    {
        return complex(real, -img);
    }

};

std::ostream &operator<<(std::ostream &os, complex const &m)
{
    return os << "{real: " << m.real << ", img: " << m.img << "}";
}