#pragma once

#include <Kokkos_Core.hpp>

#include <string>

void compute_density(
    Kokkos::View<double**> rho,
    Kokkos::View<double***> f,
    int Nx,
    int Ny
);

void compute_velocity(
    Kokkos::View<double**> ux,
    Kokkos::View<double**> uy,
    Kokkos::View<double**> rho,
    Kokkos::View<double***> f,
    Kokkos::View<int*> cx,
    Kokkos::View<int*> cy,
    int Nx,
    int Ny
);

void streaming(
    Kokkos::View<double***> f,
    Kokkos::View<double***> f_new,
    Kokkos::View<int*> cx,
    Kokkos::View<int*> cy,
    int Nx,
    int Ny
);

double compute_total_mass(
    Kokkos::View<double***> f,
    int Nx,
    int Ny
);

void output_fields(
    Kokkos::View<double**> rho,
    Kokkos::View<double**> ux,
    Kokkos::View<double**> uy,
    int Nx,
    int Ny,
    const std::string& filename
);

void compute_f_eq(
    Kokkos::View<double***> f_eq,
    Kokkos::View<double**> rho,
    Kokkos::View<double**> ux,
    Kokkos::View<double**> uy,
    Kokkos::View<int*> cx,
    Kokkos::View<int*> cy,
    int Nx,
    int Ny,
    Kokkos::View<double*> w
);

// Initializes f to the D2Q9 equilibrium for rho = 1,
// ux(y) = epsilon * sin(2*pi*y/Ny), and uy = 0.
void initialize_shear_wave(
    Kokkos::View<double***> f,
    Kokkos::View<int*> cx,
    Kokkos::View<int*> cy,
    Kokkos::View<double*> w,
    int Nx,
    int Ny,
    double epsilon
);

void collision(
    Kokkos::View<double***> f,
    Kokkos::View<double***> f_eq,
    double omega,
    int Nx,
    int Ny
);