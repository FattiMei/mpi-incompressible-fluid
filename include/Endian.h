#ifndef ENDIAN_H
#define ENDIAN_H


namespace mif {
// in case the cluster compiler is not so modern, we provide a fallback implementation
#ifdef LEGACY_COMPILER
#include <byteswap.h>

constexpr float correct_endianness(float x) {
	uint32_t trasmute = *reinterpret_cast<uint32_t*>(&x);
	uint32_t swapped = bswap_32(x);

	return *reinterpret_cast<float*>(&swapped);
}

constexpr double correct_endianness(double x) {
	uint64_t trasmute = *reinterpret_cast<uint64_t*>(&x);
	uint64_t swapped = bswap_64(x);

	return *reinterpret_cast<float*>(&swapped);

}

#else

template <std::floating_point Type>
constexpr Type correct_endianness(const Type x) noexcept {
	if constexpr (std::endian::native == std::endian::little){
		if constexpr (std::is_same_v<Type, float>){
			const auto transmute = std::bit_cast<std::uint32_t>(x);
			auto swapped = std::byteswap(transmute);
			return std::bit_cast<Type>(swapped);
		}
		else if constexpr (std::is_same_v<Type, double>){
			const auto transmute = std::bit_cast<std::uint64_t>(x);
			auto swapped = std::byteswap(transmute);
			return std::bit_cast<Type>(swapped);
		}
	}
	return x;
}

#endif


template <typename floating>
void vectorToBigEndian(std::vector<floating> &xs) {
        for (floating& x : xs) x = correct_endianness(x);
}


constexpr void test_implementations() {
	static_assert(10.0d == correct_endianness(correct_endianness(10.0d)));
	static_assert(-1.123e7 == correct_endianness(correct_endianness(-1.123e7)));
	static_assert(123.456 != correct_endianness(123.456));
}

};


#endif
