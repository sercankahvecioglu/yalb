#include <Kokkos_Core.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "lbm.hpp"

namespace {

constexpr int Q = 9;
constexpr int NX = 128;
constexpr int NY = 128;
constexpr int MAX_STEPS = 200000;
constexpr int CONVERGENCE_CHECK_INTERVAL = 100;
constexpr int BENCHMARK_STEPS = 2000;

constexpr double LID_VELOCITY = 0.1;
// This gives nu = 0.032 and Re = u_lid * NX / nu = 400 exactly.
constexpr double OMEGA = 1.0 / 0.596;
constexpr double CONVERGENCE_LIMIT = 1.0e-6;
constexpr double WALL_DENSITY = 1.0;

struct Lattice {
    Kokkos::View<double**> rho{"rho", NX, NY};
    Kokkos::View<double**> ux{"ux", NX, NY};
    Kokkos::View<double**> uy{"uy", NX, NY};
    Kokkos::View<double**> previous_ux{"previous_ux", NX, NY};
    Kokkos::View<double**> previous_uy{"previous_uy", NX, NY};

    Kokkos::View<double***> f{"f", NX, NY, Q};
    Kokkos::View<double***> streamed_f{"streamed_f", NX, NY, Q};
    Kokkos::View<double***> equilibrium_f{"equilibrium_f", NX, NY, Q};

    Kokkos::View<int*> cx{"cx", Q};
    Kokkos::View<int*> cy{"cy", Q};
    Kokkos::View<int*> opposite{"opposite", Q};
    Kokkos::View<double*> weights{"weights", Q};
};

void initialize_d2q9(Lattice& lattice) {
    constexpr std::array<int, Q> CX = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    constexpr std::array<int, Q> CY = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    constexpr std::array<int, Q> OPPOSITE = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    constexpr std::array<double, Q> WEIGHTS = {
        4.0 / 9.0,
        1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };

    auto host_cx = Kokkos::create_mirror_view(lattice.cx);
    auto host_cy = Kokkos::create_mirror_view(lattice.cy);
    auto host_opposite = Kokkos::create_mirror_view(lattice.opposite);
    auto host_weights = Kokkos::create_mirror_view(lattice.weights);

    for (int i = 0; i < Q; ++i) {
        host_cx(i) = CX[i];
        host_cy(i) = CY[i];
        host_opposite(i) = OPPOSITE[i];
        host_weights(i) = WEIGHTS[i];
    }

    Kokkos::deep_copy(lattice.cx, host_cx);
    Kokkos::deep_copy(lattice.cy, host_cy);
    Kokkos::deep_copy(lattice.opposite, host_opposite);
    Kokkos::deep_copy(lattice.weights, host_weights);
}

void initialize_stationary_fluid(Lattice& lattice) {
    // For rho = 1 and u = 0, the equilibrium populations are simply w_i.
    Kokkos::parallel_for("initialize_cavity", NX * NY, KOKKOS_LAMBDA(int index) {
        const int x = index / NY;
        const int y = index % NY;

        for (int i = 0; i < Q; ++i) {
            lattice.f(x, y, i) = lattice.weights(i);
        }
    });
    Kokkos::fence();
}

void update_macroscopic_fields(Lattice& lattice) {
    compute_density(lattice.rho, lattice.f, NX, NY);
    compute_velocity(lattice.ux, lattice.uy, lattice.rho, lattice.f,
                     lattice.cx, lattice.cy, NX, NY);
    Kokkos::fence();
}

void perform_timestep(Lattice& lattice) {
    compute_f_eq(lattice.equilibrium_f, lattice.rho, lattice.ux, lattice.uy,
                 lattice.cx, lattice.cy, NX, NY, lattice.weights);
    collision(lattice.f, lattice.equilibrium_f, OMEGA, NX, NY);
    streaming_with_walls(lattice.f, lattice.streamed_f,
                         lattice.cx, lattice.cy, lattice.opposite,
                         lattice.weights, NX, NY,
                         LID_VELOCITY, WALL_DENSITY);
    Kokkos::deep_copy(lattice.f, lattice.streamed_f);
    update_macroscopic_fields(lattice);
}

double maximum_velocity_change(const Lattice& lattice) {
    double maximum_change = 0.0;

    Kokkos::parallel_reduce("maximum_velocity_change", NX * NY,
        KOKKOS_LAMBDA(int index, double& local_maximum) {
            const int x = index / NY;
            const int y = index % NY;
            const double change_x =
                lattice.ux(x, y) - lattice.previous_ux(x, y);
            const double change_y =
                lattice.uy(x, y) - lattice.previous_uy(x, y);
            const double change =
                Kokkos::sqrt(change_x * change_x + change_y * change_y);

            if (change > local_maximum) local_maximum = change;
        }, Kokkos::Max<double>(maximum_change));

    return maximum_change;
}

void save_current_velocity(Lattice& lattice) {
    Kokkos::deep_copy(lattice.previous_ux, lattice.ux);
    Kokkos::deep_copy(lattice.previous_uy, lattice.uy);
}

void write_fields(const Lattice& lattice,
                  const std::filesystem::path& output_directory) {
    auto rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lattice.rho);
    auto ux = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lattice.ux);
    auto uy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lattice.uy);

    std::ofstream fields_file(output_directory / "cavity_fields.csv");
    fields_file << std::setprecision(16) << "x,y,rho,ux,uy,speed\n";

    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            const double speed = std::sqrt(ux(x, y) * ux(x, y) +
                                           uy(x, y) * uy(x, y));
            fields_file << x << ',' << y << ',' << rho(x, y) << ','
                        << ux(x, y) << ',' << uy(x, y) << ',' << speed << '\n';
        }
    }

    std::ofstream centerline_file(output_directory / "centerline_ux.csv");
    centerline_file << std::setprecision(16) << "y,y_over_L,ux,ux_over_lid_velocity\n";

    const int center_x = NX / 2;
    // The physical walls are half a lattice spacing outside the first and
    // last dry nodes. Add the exact no-slip wall values for comparison.
    centerline_file << -0.5 << ",0,0,0\n";
    for (int y = 0; y < NY; ++y) {
        const double physical_y = (static_cast<double>(y) + 0.5) / NY;
        centerline_file << y << ',' << physical_y << ','
                        << ux(center_x, y) << ','
                        << ux(center_x, y) / LID_VELOCITY << '\n';
    }
    centerline_file << NY - 0.5 << ",1," << LID_VELOCITY << ",1\n";
}

void verify_result(const Lattice& lattice) {
    auto rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lattice.rho);
    auto ux = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lattice.ux);
    auto uy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lattice.uy);

    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            if (!std::isfinite(rho(x, y)) || !std::isfinite(ux(x, y)) ||
                !std::isfinite(uy(x, y))) {
                throw std::runtime_error("simulation became unstable (NaN or infinity)");
            }
        }
    }
}

double benchmark_mlups(Lattice& lattice, double& benchmark_seconds) {
    // The flow is already steady. Time only solver timesteps: no convergence
    // reductions, file output, or terminal output is included.
    Kokkos::fence();
    const auto start = std::chrono::steady_clock::now();

    for (int step = 0; step < BENCHMARK_STEPS; ++step) {
        perform_timestep(lattice);
    }

    Kokkos::fence();
    const auto end = std::chrono::steady_clock::now();
    benchmark_seconds = std::chrono::duration<double>(end - start).count();

    const double updates = static_cast<double>(NX) * NY * BENCHMARK_STEPS;
    return updates / benchmark_seconds / 1.0e6;
}

}  // namespace

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    int return_code = 0;

    try {
        const std::filesystem::path output_directory =
            argc > 1 ? argv[1] : "milestone5_results";
        std::filesystem::create_directories(output_directory);

        Lattice lattice;
        initialize_d2q9(lattice);
        initialize_stationary_fluid(lattice);
        update_macroscopic_fields(lattice);
        save_current_velocity(lattice);

        const double viscosity = (1.0 / OMEGA - 0.5) / 3.0;
        const double reynolds_number = LID_VELOCITY * NX / viscosity;

        std::cout << "Lid-driven cavity: " << NX << " x " << NY << '\n'
                  << "omega = " << OMEGA << ", viscosity = " << viscosity << '\n'
                  << "lid velocity = " << LID_VELOCITY
                  << ", Reynolds number = " << reynolds_number << '\n';

        const auto start_time = std::chrono::steady_clock::now();
        int completed_steps = 0;
        double velocity_change = 0.0;

        for (int step = 1; step <= MAX_STEPS; ++step) {
            const bool check_convergence =
                step % CONVERGENCE_CHECK_INTERVAL == 0;

            if (check_convergence) save_current_velocity(lattice);
            perform_timestep(lattice);
            completed_steps = step;

            if (check_convergence) {
                velocity_change = maximum_velocity_change(lattice);
                if (step % 1000 == 0 || velocity_change < CONVERGENCE_LIMIT) {
                    std::cout << "step " << step
                              << ": maximum velocity change = "
                              << velocity_change << '\n';
                }

                if (velocity_change < CONVERGENCE_LIMIT) break;
            }
        }

        Kokkos::fence();
        const auto end_time = std::chrono::steady_clock::now();
        const double seconds =
            std::chrono::duration<double>(end_time - start_time).count();
        double benchmark_seconds = 0.0;
        const double mlups = benchmark_mlups(lattice, benchmark_seconds);

        verify_result(lattice);
        write_fields(lattice, output_directory);

        std::ofstream summary(output_directory / "summary.txt");
        summary << std::setprecision(16)
                << "grid=" << NX << 'x' << NY << '\n'
                << "steps=" << completed_steps << '\n'
                << "omega=" << OMEGA << '\n'
                << "viscosity=" << viscosity << '\n'
                << "lid_velocity=" << LID_VELOCITY << '\n'
                << "reynolds_number=" << reynolds_number << '\n'
                << "maximum_velocity_change=" << velocity_change << '\n'
                << "convergence_runtime_seconds=" << seconds << '\n'
                << "benchmark_steps=" << BENCHMARK_STEPS << '\n'
                << "benchmark_seconds=" << benchmark_seconds << '\n'
                << "mlups=" << mlups << '\n';

        std::cout << "Reached steady state after " << completed_steps
                  << " steps in " << seconds << " seconds\n"
                  << "Performance benchmark: " << mlups << " MLUPS ("
                  << BENCHMARK_STEPS << " solver-only steps)\n"
                  << "Results written to " << output_directory << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return_code = 1;
    }

    Kokkos::finalize();
    return return_code;
}
