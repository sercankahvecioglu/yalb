#include <Kokkos_Core.hpp>
#include "lbm.hpp"

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int Nx = 15;
        int Ny = 10;
        int Q = 9;
        int num_steps = 100;
        double omega = 1.0;

        Kokkos::View<double**> rho("rho", Nx, Ny);
        Kokkos::View<double**> ux("ux", Nx, Ny);
        Kokkos::View<double**> uy("uy", Nx, Ny);
        Kokkos::View<double***> f("f", Nx, Ny, Q);
        Kokkos::View<int*> cx("cx", Q);
        Kokkos::View<int*> cy("cy", Q);
        Kokkos::View<double***> f_new("f_new", Nx, Ny, Q);
        Kokkos::View<double***> f_eq("f_eq", Nx, Ny, Q);
        Kokkos::View<double*> w("w", Q);

        auto h_w = Kokkos::create_mirror_view(w);
        double w_values[9] = {
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

        for (int i = 0; i < Q; i++) {
            h_w(i) = w_values[i];
        }
        Kokkos::deep_copy(w, h_w);

        Kokkos::parallel_for("initialize_f", Nx * Ny, KOKKOS_LAMBDA(const int idx) {
        int x = idx / Ny;
        int y = idx % Ny;

        for (int i = 0; i < 9; i++) {
            f(x, y, i) = 1.0 / 9.0;
        }
        if (x == 7 && y == 5) {
                f(x, y, 1) += 0.1;  // sağ yönü artır
            }
    });

        auto h_cx = Kokkos::create_mirror_view(cx);
        auto h_cy = Kokkos::create_mirror_view(cy);

        int cx_values[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
        int cy_values[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

        for (int i = 0; i < 9; i++) {
            h_cx(i) = cx_values[i];
            h_cy(i) = cy_values[i];
        }

        Kokkos::deep_copy(cx, h_cx);
        Kokkos::deep_copy(cy, h_cy);

        for (int step = 0; step < num_steps; ++step) {
            compute_density(rho, f, Nx, Ny);
            Kokkos::fence();
            compute_velocity(ux, uy, rho, f, cx, cy, Nx, Ny);
            Kokkos::fence();
            compute_f_eq(f_eq, rho, ux, uy, cx, cy, Nx, Ny, w);
            Kokkos::fence();
            collision(f, f_eq, omega, Nx, Ny);
            Kokkos::fence();
            Kokkos::deep_copy(f_new, 0.0);
            streaming(f, f_new, cx, cy, Nx, Ny);
            Kokkos::fence();
            Kokkos::deep_copy(f, f_new);
        }

        compute_density(rho, f, Nx, Ny);
        Kokkos::fence();
        compute_velocity(ux, uy, rho, f, cx, cy, Nx, Ny);
        Kokkos::fence();
        output_fields(rho, ux, uy, Nx, Ny, "fields.csv");
    }

    Kokkos::finalize();
    return 0;
}
