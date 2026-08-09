#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>

#include "lbm.hpp"

namespace {

constexpr int Q = 9;
constexpr std::array<int, Q> kCx = {0, 1, 0, -1, 0, 1, -1, -1, 1};
constexpr std::array<int, Q> kCy = {0, 0, 1, 0, -1, 1, 1, -1, -1};
constexpr std::array<double, Q> kWeights = {
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0
};
constexpr double kTwoPi = 6.283185307179586476925286766559;

void initialize_d2q9_constants(
    Kokkos::View<int*> cx,
    Kokkos::View<int*> cy,
    Kokkos::View<double*> w
) {
    auto h_cx = Kokkos::create_mirror_view(cx);
    auto h_cy = Kokkos::create_mirror_view(cy);
    auto h_w = Kokkos::create_mirror_view(w);

    for (int i = 0; i < Q; ++i) {
        h_cx(i) = kCx[i];
        h_cy(i) = kCy[i];
        h_w(i) = kWeights[i];
    }

    Kokkos::deep_copy(cx, h_cx);
    Kokkos::deep_copy(cy, h_cy);
    Kokkos::deep_copy(w, h_w);
}

}  // namespace

TEST(ShearWaveInitializationTest, RecoversPrescribedDensityAndVelocity) {
    constexpr int Nx = 8;
    constexpr int Ny = 16;
    constexpr double epsilon = 0.05;
    constexpr double tolerance = 1.0e-12;

    Kokkos::View<double***> f("f", Nx, Ny, Q);
    Kokkos::View<double**> rho("rho", Nx, Ny);
    Kokkos::View<double**> ux("ux", Nx, Ny);
    Kokkos::View<double**> uy("uy", Nx, Ny);
    Kokkos::View<int*> cx("cx", Q);
    Kokkos::View<int*> cy("cy", Q);
    Kokkos::View<double*> w("w", Q);

    initialize_d2q9_constants(cx, cy, w);
    initialize_shear_wave(f, cx, cy, w, Nx, Ny, epsilon);

    compute_density(rho, f, Nx, Ny);
    compute_velocity(ux, uy, rho, f, cx, cy, Nx, Ny);
    Kokkos::fence();

    auto h_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);
    auto h_ux = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ux);
    auto h_uy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), uy);

    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            const double expected_ux =
                epsilon * std::sin(kTwoPi * static_cast<double>(y) /
                                   static_cast<double>(Ny));

            EXPECT_NEAR(h_rho(x, y), 1.0, tolerance)
                << "x=" << x << ", y=" << y;
            EXPECT_NEAR(h_uy(x, y), 0.0, tolerance)
                << "x=" << x << ", y=" << y;
            EXPECT_NEAR(h_ux(x, y), expected_ux, tolerance)
                << "x=" << x << ", y=" << y;
        }
    }
}
