#include <iostream>

#include "component.h"

namespace fem::core::base {

// === Component Implementation ===

    Component::Component(std::string_view name) : name_(name) {}

// === Entity Implementation ===

    Entity::Entity(std::string_view name) : name_(name) {}

    Entity::~Entity() {
        // Properly detach all components
        for (auto& [type, component] : components_) {
            component->on_detach();
        }
    }

} // namespace fem::core