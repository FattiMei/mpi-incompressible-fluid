#ifndef ENDIAN_H
#define ENDIAN_H

#include <cstdint>
#include <vector>

// In case the cluster compiler is not so modern, we provide a fallback implementation.
#ifdef MIF_LEGACY_COMPILER
#include <byteswap.h>

namespace mif {
float correct_endianness(float x) {
	uint32_t trasmute = *reinterpret_cast<uint32_t*>(&x);
	uint32_t swapped = bswap_32(trasmute);

	return *reinterpret_cast<float*>(&swapped);
}

double correct_endianness(double x) {
	uint64_t trasmute = *reinterpret_cast<uint64_t*>(&x);
	uint64_t swapped = bswap_64(trasmute);

	return *reinterpret_cast<double*>(&swapped);
}
};

#else

#include <bit>

namespace mif {
template <typename T>
constexpr T correct_endianness(const T x) noexcept {
	if constexpr (std::endian::native == std::endian::little){
		if constexpr (std::is_same_v<T, float>){
			const auto transmute = std::bit_cast<uint32_t>(x);
			auto swapped = std::byteswap(transmute);
			return std::bit_cast<T>(swapped);
		}
		else if constexpr (std::is_same_v<T, double>){
			const auto transmute = std::bit_cast<uint64_t>(x);
			auto swapped = std::byteswap(transmute);
			return std::bit_cast<T>(swapped);
		}
		else if constexpr (std::is_same_v<T, uint32_t>){
			const auto transmute = std::bit_cast<uint32_t>(x);
			auto swapped = std::byteswap(transmute);
			return std::bit_cast<T>(swapped);
		}
		else if constexpr (std::is_same_v<T, int>){
			const auto transmute = std::bit_cast<int>(x);
			auto swapped = std::byteswap(transmute);
			return std::bit_cast<T>(swapped);
		}
	}
	return x;
}

// This is not that good of a test but it is better than nothing, this would pass even if the function was x => return -x.
constexpr void test_implementations() {
	static_assert(10.0f == correct_endianness(correct_endianness(10.0f)));
	static_assert(-123.456f == correct_endianness(correct_endianness(-123.456f)));

	static_assert(10.0 == correct_endianness(correct_endianness(10.0)));
	static_assert(-123.456 == correct_endianness(correct_endianness(-123.456)));

	static_assert(123.456 != correct_endianness(123.456));
}
};

#endif

namespace mif {

template <typename T>
void vectorToBigEndian(std::vector<T> &xs) noexcept {
        for (T& x : xs) x = correct_endianness(x);
}

};


#endif // ENDIAN_H
