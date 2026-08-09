#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>

#include "lbm.hpp"

namespace {

constexpr int Q = 9;
constexpr int NX = 5;
constexpr int NY = 5;
constexpr double LID_VELOCITY = 0.1;
constexpr double WALL_DENSITY = 1.0;

struct BoundaryTestData {
    Kokkos::View<double***> f{"boundary_f", NX, NY, Q};
    Kokkos::View<double***> f_new{"boundary_f_new", NX, NY, Q};
    Kokkos::View<int*> cx{"boundary_cx", Q};
    Kokkos::View<int*> cy{"boundary_cy", Q};
    Kokkos::View<int*> opposite{"boundary_opposite", Q};
    Kokkos::View<double*> weights{"boundary_weights", Q};
};

void initialize_test_data(BoundaryTestData& data) {
    constexpr std::array<int, Q> CX = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    constexpr std::array<int, Q> CY = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    constexpr std::array<int, Q> OPPOSITE = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    constexpr std::array<double, Q> WEIGHTS = {
        4.0 / 9.0,
        1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };

    auto cx = Kokkos::create_mirror_view(data.cx);
    auto cy = Kokkos::create_mirror_view(data.cy);
    auto opposite = Kokkos::create_mirror_view(data.opposite);
    auto weights = Kokkos::create_mirror_view(data.weights);

    for (int i = 0; i < Q; ++i) {
        cx(i) = CX[i];
        cy(i) = CY[i];
        opposite(i) = OPPOSITE[i];
        weights(i) = WEIGHTS[i];
    }

    Kokkos::deep_copy(data.cx, cx);
    Kokkos::deep_copy(data.cy, cy);
    Kokkos::deep_copy(data.opposite, opposite);
    Kokkos::deep_copy(data.weights, weights);
    Kokkos::deep_copy(data.f, 0.0);
    Kokkos::deep_copy(data.f_new, 0.0);
}

void stream(BoundaryTestData& data) {
    streaming_with_walls(data.f, data.f_new, data.cx, data.cy,
                         data.opposite, data.weights, NX, NY,
                         LID_VELOCITY, WALL_DENSITY);
    Kokkos::fence();
}

}  // namespace

TEST(WallStreamingTest, PopulationStreamsNormallyInsideDomain) {
    BoundaryTestData data;
    initialize_test_data(data);

    auto f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.f);
    f(2, 2, 1) = 0.25;  // f1 moves right.
    Kokkos::deep_copy(data.f, f);

    stream(data);
    auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.f_new);

    EXPECT_DOUBLE_EQ(result(3, 2, 1), 0.25);
}

TEST(WallStreamingTest, BottomWallReflectsDownIntoUp) {
    BoundaryTestData data;
    initialize_test_data(data);

    auto f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.f);
    f(2, 0, 4) = 0.25;  // f4 points down; opposite(4) is f2.
    Kokkos::deep_copy(data.f, f);

    stream(data);
    auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.f_new);

    EXPECT_DOUBLE_EQ(result(2, 0, 2), 0.25);
}

TEST(WallStreamingTest, LeftWallReflectsLeftIntoRight) {
    BoundaryTestData data;
    initialize_test_data(data);

    auto f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.f);
    f(0, 2, 3) = 0.25;  // f3 points left; opposite(3) is f1.
    Kokkos::deep_copy(data.f, f);

    stream(data);
    auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.f_new);

    EXPECT_DOUBLE_EQ(result(0, 2, 1), 0.25);
}

TEST(WallStreamingTest, MovingLidAddsRightwardMomentum) {
    BoundaryTestData data;
    initialize_test_data(data);

    auto f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.f);
    f(2, NY - 1, 5) = 0.20;  // northeast -> southwest (f7)
    f(2, NY - 1, 6) = 0.20;  // northwest -> southeast (f8)
    Kokkos::deep_copy(data.f, f);

    stream(data);
    auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.f_new);

    const double correction = LID_VELOCITY / 6.0;
    EXPECT_NEAR(result(2, NY - 1, 7), 0.20 - correction, 1.0e-14);
    EXPECT_NEAR(result(2, NY - 1, 8), 0.20 + correction, 1.0e-14);

    // f8 carries +x and f7 carries -x, so this must be positive.
    const double horizontal_momentum =
        result(2, NY - 1, 8) - result(2, NY - 1, 7);
    EXPECT_GT(horizontal_momentum, 0.0);
}

TEST(WallStreamingTest, ClosedCavityConservesTotalMass) {
    BoundaryTestData data;
    initialize_test_data(data);

    auto f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.f);
    auto weights =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.weights);

    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            for (int i = 0; i < Q; ++i) f(x, y, i) = weights(i);
        }
    }
    Kokkos::deep_copy(data.f, f);

    const double mass_before = compute_total_mass(data.f, NX, NY);
    stream(data);
    const double mass_after = compute_total_mass(data.f_new, NX, NY);

    EXPECT_NEAR(mass_after, mass_before, 1.0e-12);
}
