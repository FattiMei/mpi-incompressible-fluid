#ifndef METAHELPERS_H
#define METAHELPERS_H

#include <cstdint>
#include <type_traits>

namespace mif {

/*!
 * Compute an integer sequence size
 */
template <typename T, T... Ints> struct integer_sequence_size;
/*!
 * Compute an integer sequence size with at least one element specialization
 */
template <typename T, T Head, T... Tail>
struct integer_sequence_size<T, Head, Tail...> {
  static constexpr uint8_t value = 1 + integer_sequence_size<T, Tail...>::value;
};
/*!
 * Compute an empty integer sequence specialization
 */
template <typename T> struct integer_sequence_size<T> {
  static constexpr uint8_t value = 0;
};

/*!
 * Helper struct to detect if at compile time a type is callable
 * @tparam T The type to test
 */
template <typename T> struct is_callable {
  // Test for operator ()
  template <typename U>
  static auto test(U *) -> decltype(&U::operator(), std::true_type{});

  // SFINAE Fallback
  template <typename U> static std::false_type test(...);

  // Test the type: if the type has an operator () the test will be true
  // otherwise false by default.
  static constexpr bool value = decltype(test<T>(nullptr))::value;
};

} // namespace mif

#endif // METAHELPERS_H
