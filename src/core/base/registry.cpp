#include "registry.h"
#include "object.h"

#include <string>
#include <cstdint>

namespace fem::core::base {

// Explicit template instantiation for common types
template class Registry<Object, std::string>;
template class Registry<Object, int>;
template class Registry<Object, std::uint32_t>;

} // namespace fem::core::base