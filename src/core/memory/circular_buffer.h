#pragma once

#ifndef CORE_MEMORY_CIRCULAR_BUFFER_H
#define CORE_MEMORY_CIRCULAR_BUFFER_H

#include "ring_buffer.h"

namespace fem::core::memory {

// Shim alias to match documentation naming: circular_buffer == ring_buffer
template<class T, class Alloc = polymorphic_allocator<T>>
using circular_buffer = ring_buffer<T, Alloc>;

} // namespace fem::core::memory

#endif // CORE_MEMORY_CIRCULAR_BUFFER_H

