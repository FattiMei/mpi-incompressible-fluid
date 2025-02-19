#include "ManufacturedVelocity.h"
#include "Norms.h"
#include "TimestepVelocity.h"
#include <benchmark/benchmark.h>

double Reynolds = 1e6;
constexpr Real Lx = 1.0;
constexpr Real Ly = 1.0;
constexpr Real Lz = 1.0;
constexpr Real deltat = 1e-4;

using namespace mif;

static void clear_cache_line() {
  // Determine the size of a large array (e.g., 100MB), typically larger than L3
  // cache size
  constexpr size_t cache_clear_size = 100 * 1024 * 1024; // 100 MB
  std::vector<char> cache_clear(cache_clear_size, 1);
  // Access the large array to evict cache lines
  for (std::size_t i = 0; i < cache_clear_size; ++i) {
    cache_clear[i] = i % 256;
  }
}

// Benchmarking the velocity solver.
static void timestepper(benchmark::State &state) {
  clear_cache_line();
  const size_t Nx = state.range(0);
  const size_t Ny = Nx;
  const size_t Nz = Nx;

  const Constants constants(Nx, Ny, Nz, Lx, Ly, Lz, 0.0, 0.0, 0.0, Reynolds, deltat, 1, 1, 1, 0, {false, false, false});

  VelocityTensor velocity(constants);
  VelocityTensor velocity_buffer(constants);
  VelocityTensor rhs_buffer(constants);

  TimeVectorFunction exact_velocity(u_exact_v_test, v_exact_v_test, w_exact_v_test);
  velocity.set(exact_velocity.set_time(0.0), true);

  TimeVectorFunction forcing_term(forcing_x, forcing_y, forcing_z);

  int step = 0;
  for (auto _ : state) {
    const Real t = step * constants.dt;

    timestep_velocity(velocity, velocity_buffer, rhs_buffer, exact_velocity, t);

    ++step;
  }

  state.counters["DoF"] = Nx * Ny * Nz;
  state.counters["perf"] = benchmark::Counter(
      state.counters["DoF"], benchmark::Counter::kIsIterationInvariantRate |
                                 benchmark::Counter::kInvert);
}

BENCHMARK(timestepper)
    ->RangeMultiplier(2)
    ->Range(32, 512)
    ->Unit(benchmark::kSecond);
BENCHMARK_MAIN();
