#include <iostream>

#include "registry.h"

namespace fem::core::base {

// Explicit template instantiation for common types
    template class Registry<Object, std::string>;
    template class Registry<Object, int>;

} // namespace fem::core