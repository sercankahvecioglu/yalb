//
// Created by Sercan Kahvecioğlu on 10.07.2026.
//
#include "lbm.hpp"
#include <filesystem>
#include <fstream>

double compute_total_mass(
    Kokkos::View<double***> f,
    int Nx,
    int Ny
) {
    auto h_f = Kokkos::create_mirror_view(f);
    Kokkos::deep_copy(h_f, f);

    double mass = 0.0;

    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            for (int i = 0; i < 9; i++) {
                mass += h_f(x, y, i);
            }
        }
    }
    return mass;
}

void compute_density(
    Kokkos::View<double**> rho,
    Kokkos::View<double***> f,
    int Nx,
    int Ny
) {
    Kokkos::parallel_for("compute_density", Nx * Ny, KOKKOS_LAMBDA(const int idx) {
        int x = idx / Ny;
        int y = idx % Ny;

        double sum = 0.0;

        for (int i = 0; i < 9; i++) {
            sum += f(x, y, i);
        }

        rho(x, y) = sum;
    });
}

void compute_velocity(
    Kokkos::View<double**> ux,
    Kokkos::View<double**> uy,
    Kokkos::View<double**> rho,
    Kokkos::View<double***> f,
    Kokkos::View<int*> cx,
    Kokkos::View<int*> cy,
    int Nx,
    int Ny
) {
    Kokkos::parallel_for("compute_velocity", Nx * Ny, KOKKOS_LAMBDA(const int idx) {
        int x = idx / Ny;
        int y = idx % Ny;

        double momentum_x = 0.0;
        double momentum_y = 0.0;

        for (int i = 0; i < 9; i++) {
            momentum_x += cx(i) * f(x, y, i);
            momentum_y += cy(i) * f(x, y, i);
        }

        const double rho_xy = rho(x, y);
        if (rho_xy > 0.0) {
            ux(x, y) = momentum_x / rho_xy;
            uy(x, y) = momentum_y / rho_xy;
        } else {
            ux(x, y) = 0.0;
            uy(x, y) = 0.0;
        }
    });
}

void streaming(
    Kokkos::View<double***> f,
    Kokkos::View<double***> f_new,
    Kokkos::View<int*> cx,
    Kokkos::View<int*> cy,
    int Nx,
    int Ny
) {
    Kokkos::parallel_for("streaming", Nx * Ny, KOKKOS_LAMBDA(const int idx) {
        int x = idx / Ny;
        int y = idx % Ny;

        for (int i = 0; i < 9; i++) {
            int x_to = (x + cx(i) + Nx) % Nx;
            int y_to = (y + cy(i) + Ny) % Ny;

            f_new(x_to, y_to, i) = f(x, y, i);
        }
    });
}

void output_fields(
    Kokkos::View<double**> rho,
    Kokkos::View<double**> ux,
    Kokkos::View<double**> uy,
    int Nx,
    int Ny,
    const std::string& filename
) {
    auto h_rho = Kokkos::create_mirror_view(rho);
    auto h_ux  = Kokkos::create_mirror_view(ux);
    auto h_uy  = Kokkos::create_mirror_view(uy);

    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_ux, ux);
    Kokkos::deep_copy(h_uy, uy);

    std::ofstream file(filename);

    file << "x,y,rho,ux,uy\n";

    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            file << x << ","
                 << y << ","
                 << h_rho(x, y) << ","
                 << h_ux(x, y) << ","
                 << h_uy(x, y) << "\n";
        }
    }

    file.close();
}

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
) {
    Kokkos::parallel_for("compute_f_eq", Nx * Ny, KOKKOS_LAMBDA(const int idx) {
        int x = idx / Ny;
        int y = idx % Ny;

        double rho_xy = rho(x, y);
        double ux_xy = ux(x, y);
        double uy_xy = uy(x, y);

        double u_sq = ux_xy * ux_xy + uy_xy * uy_xy;

        for (int i = 0; i < 9; i++) {
            double c_dot_u = cx(i) * ux_xy + cy(i) * uy_xy;

            f_eq(x, y, i) =
                w(i) * rho_xy *
                (
                    1.0
                    + 3.0 * c_dot_u
                    + 4.5 * c_dot_u * c_dot_u
                    - 1.5 * u_sq
                );
        }
    });
}

void collision(
    Kokkos::View<double***> f,
    Kokkos::View<double***> f_eq,
    double omega,
    int Nx,
    int Ny
) {
    Kokkos::parallel_for("collision", Nx * Ny, KOKKOS_LAMBDA(const int idx) {
        int x = idx / Ny;
        int y = idx % Ny;
        for (int i = 0; i < 9; i++) {
            f(x, y, i) = f(x, y, i) + omega * (f_eq(x, y, i) - f(x, y, i));
        }
    });
}
