#include <benchmark/benchmark.h>
#include "Manufactured.h"
#include "Norms.h"
#include "Timestep.h"


double Reynolds = 1e6;
constexpr Real Lx = 1.0;
constexpr Real Ly = 1.0;
constexpr Real Lz = 1.0;
constexpr Real deltat = 1e-4;


using namespace mif;


static void timestepper(benchmark::State &state) {
	const size_t Nx = state.range(0);
	const size_t Ny = Nx;
	const size_t Nz = Nx;

	const Constants constants(
		Nx, Ny, Nz,
		Lx, Ly, Lz,
		Reynolds,
		deltat,
		1
	);

	VelocityTensor velocity(constants);
	VelocityTensor velocity_buffer1(constants);
	VelocityTensor velocity_buffer2(constants);

	TimeVectorFunction exact_velocity(u_exact, v_exact, w_exact);
	velocity.set(exact_velocity.set_time(0.0), true);

	TimeVectorFunction forcing_term(forcing_x, forcing_y, forcing_z);

	int step = 0;
	for (auto _ : state) {
		const Real t = step*constants.dt;

		timestep(
			velocity,
			velocity_buffer1,
			velocity_buffer2,
			t
		);

		++step;
	}

	state.counters["DoF"] = Nx*Ny*Nz;
	state.counters["perf"] = benchmark::Counter(
		state.counters["DoF"],
		benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert
	);
}


BENCHMARK(timestepper)->RangeMultiplier(2)->Range(32, 256)->Arg(300)->Arg(400)->Unit(benchmark::kSecond);
BENCHMARK_MAIN();
