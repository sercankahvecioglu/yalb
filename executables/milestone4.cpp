#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lbm.hpp"

namespace {

constexpr int Q = 9;
constexpr int NX = 32;
constexpr int NY = 64;
constexpr int NUMBER_OF_STEPS = 1000;
constexpr int FIRST_FIT_STEP = 20;
constexpr double EPSILON = 0.05;
constexpr double PI = 3.14159265358979323846;

'''To simulate different viscosities.'''
const std::array<double, 5> OMEGAS = {0.6, 0.8, 1.0, 1.2, 1.4};
const std::array<int, 5> PROFILE_TIMES = {0, 100, 200, 500, 1000};

struct Lattice {
    Kokkos::View<double**> rho{"rho", NX, NY};
    Kokkos::View<double**> ux{"ux", NX, NY};
    Kokkos::View<double**> uy{"uy", NX, NY};
    Kokkos::View<double***> f{"f", NX, NY, Q};
    Kokkos::View<double***> streamed_f{"streamed_f", NX, NY, Q};
    Kokkos::View<double***> equilibrium_f{"equilibrium_f", NX, NY, Q};
    Kokkos::View<int*> cx{"cx", Q};
    Kokkos::View<int*> cy{"cy", Q};
    Kokkos::View<double*> weights{"weights", Q};
};

struct LineFit {
    double intercept;
    double slope;
};

struct RunResult {
    double omega;
    LineFit fit;
    double measured_viscosity;
    double theoretical_viscosity;
};

void initialize_d2q9_constants(Lattice& lattice) {
    constexpr std::array<int, Q> CX = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    constexpr std::array<int, Q> CY = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    constexpr std::array<double, Q> WEIGHTS = {
        4.0 / 9.0,
        1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };

    auto host_cx = Kokkos::create_mirror_view(lattice.cx);
    auto host_cy = Kokkos::create_mirror_view(lattice.cy);
    auto host_weights = Kokkos::create_mirror_view(lattice.weights);

    for (int i = 0; i < Q; ++i) {
        host_cx(i) = CX[i];
        host_cy(i) = CY[i];
        host_weights(i) = WEIGHTS[i];
    }

    Kokkos::deep_copy(lattice.cx, host_cx);
    Kokkos::deep_copy(lattice.cy, host_cy);
    Kokkos::deep_copy(lattice.weights, host_weights);
}

void update_macroscopic_fields(Lattice& lattice) {
    compute_density(lattice.rho, lattice.f, NX, NY);
    compute_velocity(lattice.ux, lattice.uy, lattice.rho, lattice.f,
                     lattice.cx, lattice.cy, NX, NY);
    Kokkos::fence();
}

void perform_timestep(Lattice& lattice, double omega) {
    update_macroscopic_fields(lattice);
    compute_f_eq(lattice.equilibrium_f, lattice.rho, lattice.ux, lattice.uy,
                 lattice.cx, lattice.cy, NX, NY, lattice.weights);
    collision(lattice.f, lattice.equilibrium_f, omega, NX, NY);
    streaming(lattice.f, lattice.streamed_f, lattice.cx, lattice.cy, NX, NY);
    Kokkos::deep_copy(lattice.f, lattice.streamed_f);
}

void verify_initialization(const Lattice& lattice) {
    auto rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lattice.rho);
    auto ux = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lattice.ux);
    auto uy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lattice.uy);

    double largest_error = 0.0;
    for (int x = 0; x < NX; ++x) {
        for (int y = 0; y < NY; ++y) {
            const double expected_ux = EPSILON * std::sin(2.0 * PI * y / NY);
            largest_error = std::max(largest_error, std::abs(rho(x, y) - 1.0));
            largest_error = std::max(largest_error, std::abs(ux(x, y) - expected_ux));
            largest_error = std::max(largest_error, std::abs(uy(x, y)));
        }
    }

    if (largest_error > 1.0e-12) {
        throw std::runtime_error("shear-wave initialization verification failed");
    }
}

std::vector<double> calculate_velocity_profile(const Lattice& lattice) {
    auto ux = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lattice.ux);
    std::vector<double> profile(NY, 0.0);

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            profile[y] += ux(x, y);
        }
        profile[y] /= NX;
    }
    return profile;
}

/**
 * Since it makes inside sin pi/2 = 1, we can just return the value at y = NY/4 to get the amplitude.
 */
double calculate_amplitude(const std::vector<double>& profile) {
    return profile[NY / 4];
}

bool should_save_profile(int time) {
    for (int profile_time : PROFILE_TIMES) {
        if (time == profile_time) return true;
    }
    return false;
}

'''Linear Regression Line Fit to find out the'''
LineFit fit_log_amplitude(const std::vector<double>& amplitudes) {
    double sum_t = 0.0; 
    double sum_log_a = 0.0;
    double sum_t_squared = 0.0;
    double sum_t_log_a = 0.0;
    int sample_count = 0;

    for (int time = FIRST_FIT_STEP; time <= NUMBER_OF_STEPS; ++time) {
        const double log_amplitude = std::log(std::abs(amplitudes[time]));
        sum_t += time;
        sum_log_a += log_amplitude;
        sum_t_squared += time * time;
        sum_t_log_a += time * log_amplitude;
        ++sample_count;
    }

    const double denominator = sample_count * sum_t_squared - sum_t * sum_t;
    const double slope =
        (sample_count * sum_t_log_a - sum_t * sum_log_a) / denominator;
    const double intercept = (sum_log_a - slope * sum_t) / sample_count;
    return {intercept, slope};
}

RunResult run_simulation(double omega,
                         const std::filesystem::path& output_directory) {
    Lattice lattice;
    initialize_d2q9_constants(lattice);
    initialize_shear_wave(lattice.f, lattice.cx, lattice.cy, lattice.weights,
                          NX, NY, EPSILON);
    update_macroscopic_fields(lattice);
    verify_initialization(lattice);

    const std::string omega_text = std::to_string(omega);

    std::ofstream amplitude_file(
        output_directory / ("amplitude_omega_" + omega_text + ".csv"));
    amplitude_file << std::setprecision(16)
                   << "time,amplitude,log_abs_amplitude\n";

    const bool save_profiles = omega == 1.0;
    std::ofstream profile_file;
    if (save_profiles) {
        profile_file.open(output_directory / "velocity_profiles.csv");
        profile_file << std::setprecision(16) << "time,y,ux\n";
    }

    std::vector<double> amplitudes(NUMBER_OF_STEPS + 1);
    for (int time = 0; time <= NUMBER_OF_STEPS; ++time) {
        if (time > 0) {
            perform_timestep(lattice, omega);
            update_macroscopic_fields(lattice);
        }

        const std::vector<double> profile = calculate_velocity_profile(lattice);
        amplitudes[time] = calculate_amplitude(profile);
        amplitude_file << time << ',' << amplitudes[time] << ','
                       << std::log(std::abs(amplitudes[time])) << '\n';

        if (save_profiles && should_save_profile(time)) {
            for (int y = 0; y < NY; ++y) {
                profile_file << time << ',' << y << ',' << profile[y] << '\n';
            }
        }
    }

    const LineFit fit = fit_log_amplitude(amplitudes);
    const double wave_number = 2.0 * PI / NY;
    const double measured_viscosity = -fit.slope / (wave_number * wave_number);
    const double theoretical_viscosity = (1.0 / omega - 0.5) / 3.0;

    return {omega, fit, measured_viscosity, theoretical_viscosity};
}

void write_summary(const std::vector<RunResult>& results,
                   const std::filesystem::path& output_directory) {
    std::ofstream file(output_directory / "viscosity_vs_omega.csv");
    file << std::setprecision(16)
         << "omega,fit_intercept,fit_slope,nu_measured,nu_theory,relative_error\n";

    std::cout << "omega  measured_nu  theory_nu  relative_error\n";
    for (const RunResult& result : results) {
        const double relative_error =
            std::abs(result.measured_viscosity - result.theoretical_viscosity) /
            result.theoretical_viscosity;

        file << result.omega << ',' << result.fit.intercept << ','
             << result.fit.slope << ',' << result.measured_viscosity << ','
             << result.theoretical_viscosity << ',' << relative_error << '\n';

        std::cout << result.omega << "  " << result.measured_viscosity << "  "
                  << result.theoretical_viscosity << "  "
                  << 100.0 * relative_error << "%\n";
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    int return_code = 0;

    try {
        const std::filesystem::path output_directory =
            argc > 1 ? argv[1] : "milestone4_results";
        std::filesystem::create_directories(output_directory);

        std::vector<RunResult> results;
        for (double omega : OMEGAS) {
            results.push_back(run_simulation(omega, output_directory));
        }
        write_summary(results, output_directory);
        std::cout << "Results written to " << output_directory << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return_code = 1;
    }

    Kokkos::finalize();
    return return_code;
}
