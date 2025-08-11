#include <iostream>
#include <algorithm>

#include "factory.h"

namespace fem::core::base {

    // Explicit template instantiation for common types
    template class Factory<Object>;

} // namespace fem::core