#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>

#include "lbm.hpp"

namespace {

constexpr int Q = 9;
constexpr std::array<int, Q> kCx = {0, 1, 0, -1, 0, 1, -1, -1, 1};
constexpr std::array<int, Q> kCy = {0, 0, 1, 0, -1, 1, 1, -1, -1};

int wrap(int value, int size) {
    return (value % size + size) % size;
}

void initialize_directions(Kokkos::View<int*> cx, Kokkos::View<int*> cy) {
    auto h_cx = Kokkos::create_mirror_view(cx);
    auto h_cy = Kokkos::create_mirror_view(cy);

    for (int i = 0; i < Q; ++i) {
        h_cx(i) = kCx[i];
        h_cy(i) = kCy[i];
    }

    Kokkos::deep_copy(cx, h_cx);
    Kokkos::deep_copy(cy, h_cy);
}

void initialize_blob(
    Kokkos::View<double***> f,
    int Nx,
    int Ny,
    int direction
) {
    auto h_f = Kokkos::create_mirror_view(f);
    Kokkos::deep_copy(h_f, 0.0);

    h_f(4, 3, direction) = 1.25;
    h_f(5, 3, direction) = 2.50;
    h_f(4, 4, direction) = 3.75;
    h_f(5, 4, direction) = 5.00;

    Kokkos::deep_copy(f, h_f);
}

void expect_shifted_blob(
    Kokkos::View<double***> initial,
    Kokkos::View<double***> current,
    int Nx,
    int Ny,
    int direction,
    int steps
) {
    auto h_initial = Kokkos::create_mirror_view(initial);
    auto h_current = Kokkos::create_mirror_view(current);

    Kokkos::deep_copy(h_initial, initial);
    Kokkos::deep_copy(h_current, current);

    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            for (int i = 0; i < Q; ++i) {
                const int source_x = wrap(x - steps * kCx[direction], Nx);
                const int source_y = wrap(y - steps * kCy[direction], Ny);
                const double expected = h_initial(source_x, source_y, i);

                EXPECT_DOUBLE_EQ(h_current(x, y, i), expected)
                    << "x=" << x << ", y=" << y << ", i=" << i;
            }
        }
    }
}

void run_blob_motion_test(int direction, int round_trip_steps) {
    constexpr int Nx = 15;
    constexpr int Ny = 10;

    Kokkos::View<double***> f("f", Nx, Ny, Q);
    Kokkos::View<double***> f_new("f_new", Nx, Ny, Q);
    Kokkos::View<double***> initial("initial", Nx, Ny, Q);
    Kokkos::View<int*> cx("cx", Q);
    Kokkos::View<int*> cy("cy", Q);

    initialize_directions(cx, cy);
    initialize_blob(f, Nx, Ny, direction);
    Kokkos::deep_copy(initial, f);

    const double initial_mass = compute_total_mass(f, Nx, Ny);
    EXPECT_DOUBLE_EQ(initial_mass, 12.5);

    for (int step = 1; step <= round_trip_steps; ++step) {
        Kokkos::deep_copy(f_new, 0.0);

        streaming(f, f_new, cx, cy, Nx, Ny);
        Kokkos::fence();
        Kokkos::deep_copy(f, f_new);

        EXPECT_DOUBLE_EQ(compute_total_mass(f, Nx, Ny), initial_mass)
            << "step=" << step;
        expect_shifted_blob(initial, f, Nx, Ny, direction, step);
    }
}

}  // namespace

TEST(StreamingTest, HorizontalBlobTranslatesAndReturnsAfterNxSteps) {
    run_blob_motion_test(1, 15);
    run_blob_motion_test(3, 15);
}

TEST(StreamingTest, VerticalBlobTranslatesAndReturnsAfterNySteps) {
    run_blob_motion_test(2, 10);
    run_blob_motion_test(4, 10);
}

TEST(MacroscopicFieldsTest, ZeroDensityProducesZeroVelocity) {
    constexpr int Nx = 2;
    constexpr int Ny = 2;

    Kokkos::View<double***> f("f", Nx, Ny, Q);
    Kokkos::View<double**> rho("rho", Nx, Ny);
    Kokkos::View<double**> ux("ux", Nx, Ny);
    Kokkos::View<double**> uy("uy", Nx, Ny);
    Kokkos::View<int*> cx("cx", Q);
    Kokkos::View<int*> cy("cy", Q);

    Kokkos::deep_copy(f, 0.0);
    initialize_directions(cx, cy);
    compute_density(rho, f, Nx, Ny);
    compute_velocity(ux, uy, rho, f, cx, cy, Nx, Ny);
    Kokkos::fence();

    auto h_ux = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ux);
    auto h_uy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), uy);

    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            EXPECT_DOUBLE_EQ(h_ux(x, y), 0.0);
            EXPECT_DOUBLE_EQ(h_uy(x, y), 0.0);
        }
    }
}
