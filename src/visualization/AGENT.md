# AGENT.md - FEM Visualization Module

## Mission
Provide high-performance, scalable visualization capabilities for FEM analysis, supporting real-time rendering, in-situ visualization, remote visualization, and publication-quality output while leveraging GPU acceleration and maintaining seamless integration with the solver pipeline.

## Architecture Philosophy
- **Performance First**: GPU-accelerated rendering with LOD support
- **Scalable Architecture**: From desktop to distributed supercomputing
- **In-Situ Capable**: Visualization during computation without I/O
- **Standards Based**: VTK, OpenGL, Vulkan, and web standards
- **Interactive & Batch**: Support both interactive exploration and batch processing
- **Memory Efficient**: Streaming and out-of-core rendering for massive datasets

## Module Structure

```
visualization/
├── core/                    # Visualization infrastructure
│   ├── scene/              # Scene graph management
│   ├── camera/             # Camera controls and navigation
│   ├── lights/             # Lighting systems
│   ├── materials/          # Material definitions
│   └── pipeline/           # Rendering pipeline abstraction
├── render/                  # Rendering engines
│   ├── opengl/             # OpenGL renderer
│   ├── vulkan/             # Vulkan renderer
│   ├── raytracing/         # Ray tracing engine
│   ├── volume/             # Volume rendering
│   └── web/                # WebGL/WebGPU renderer
├── data_mappers/           # FEM data to graphics mapping
│   ├── field_mapper/       # Field to color/glyph mapping
│   ├── mesh_mapper/        # Mesh to geometry mapping
│   ├── tensor_mapper/      # Tensor visualization
│   ├── vector_mapper/      # Vector field visualization
│   └── stream_mapper/      # Streamline/pathline generation
├── filters/                # Data filtering and processing
│   ├── clip/               # Clipping planes/volumes
│   ├── slice/              # Slicing operations
│   ├── threshold/          # Threshold filtering
│   ├── contour/            # Iso-surface/contour generation
│   ├── warp/               # Deformation visualization
│   └── decimate/           # Mesh decimation for LOD
├── insitu/                 # In-situ visualization
│   ├── catalyst/           # ParaView Catalyst integration
│   ├── ascent/             # Ascent integration
│   ├── pipeline/           # In-situ pipeline management
│   ├── triggers/           # Visualization triggers
│   └── data_reduction/     # In-situ data reduction
├── remote/                 # Remote visualization
│   ├── server/             # Rendering server
│   ├── client/             # Thin clients
│   ├── streaming/          # Video/image streaming
│   ├── compression/        # Data compression
│   └── protocols/          # Communication protocols
├── interactive/            # Interactive components
│   ├── widgets/            # 3D widgets and manipulators
│   ├── picking/            # Object selection
│   ├── probes/             # Interactive probing
│   ├── annotations/        # Text and measurement annotations
│   └── animation/          # Animation controls
├── export/                 # Export capabilities
│   ├── images/             # Image export (PNG, JPG, EXR)
│   ├── movies/             # Animation export
│   ├── geometry/           # 3D geometry export
│   ├ناز publication/       # Publication-quality output
│   └── web/                # Web-based formats
├── analytics/              # Visual analytics
│   ├── statistics/         # Statistical visualization
│   ├── comparison/         # Side-by-side comparison
│   ├── uncertainty/        # Uncertainty visualization
│   ├── correlation/        # Correlation analysis
│   └── cinema/             # Cinema database generation
├── shaders/                # GPU shaders
│   ├── vertex/             # Vertex shaders
│   ├── fragment/           # Fragment shaders
│   ├── compute/            # Compute shaders
│   ├── geometry/           # Geometry shaders
│   └── tessellation/       # Tessellation shaders
└── bindings/               # Language bindings
    ├── python/             # Python visualization API
    ├── jupyter/            # Jupyter notebook widgets
    └── web/                # JavaScript/WebAssembly
```

## Module Descriptions

### 1. **core/** - Visualization Infrastructure

#### **scene/**
```cpp
class Scene : public core::Object {
    // Scene graph with hierarchical transformations
    class SceneNode : public core::Entity {
        Transform local_transform;
        std::vector<std::unique_ptr<SceneNode>> children;
        std::vector<RenderableComponent*> renderables;
        
        // Culling support
        BoundingBox world_bounds;
        bool is_visible(const Frustum& frustum) const;
    };
    
    // Efficient scene updates
    void update_transforms();
    void update_bounds();
    
    // Level-of-detail management
    void update_lod(const Camera& camera);
    
    // GPU scene representation
    DeviceSceneData upload_to_gpu();
};

// Actor pattern for FEM entities
class FEMActor : public SceneNode {
    MeshComponent* mesh;
    FieldComponent* field;
    MaterialComponent* material;
    
    // Direct FEM data binding
    void bind_fem_data(const Element* elements,
                      const Field* field);
    
    // Incremental updates
    void update_field(const Field* new_field);
    void update_deformation(const Field* displacement);
};
```

#### **camera/**
```cpp
class Camera : public core::Component {
    // Camera modes
    enum Mode {
        TRACKBALL,
        FLY_THROUGH,
        TURNTABLE,
        ORTHOGRAPHIC,
        FEM_2D  // Specialized for 2D FEM
    };
    
    // View matrices
    Matrix4 get_view_matrix() const;
    Matrix4 get_projection_matrix() const;
    
    // Interactive controls
    void rotate(float dx, float dy);
    void pan(float dx, float dy);
    void zoom(float delta);
    
    // Automatic positioning
    void fit_to_bounds(const BoundingBox& bounds);
    void focus_on_element(const Element& elem);
    
    // Animation
    void interpolate_to(const Camera& target, float t);
    
    // Stereo/VR support
    std::pair<Matrix4, Matrix4> get_stereo_matrices() const;
};
```

#### **pipeline/**
```cpp
class RenderPipeline : public core::Object {
    // Multi-pass rendering
    std::vector<std::unique_ptr<RenderPass>> passes;
    
    // Common passes for FEM
    class GeometryPass;      // Render geometry
    class OutlinePass;       // Element outlines  
    class TransparencyPass;  // Order-independent transparency
    class PostProcessPass;   // Screen-space effects
    class SelectionPass;     // Interactive selection
    
    // Pipeline configuration
    void add_pass(std::unique_ptr<RenderPass> pass);
    void configure_for_fem();
    
    // Render targets
    std::vector<RenderTarget> targets;
    
    // GPU pipeline state
    struct PipelineState {
        VertexLayout vertex_layout;
        ShaderProgram* shader;
        RenderState render_state;
    };
    
    // Execute pipeline
    void render(const Scene& scene, const Camera& camera);
};
```

### 2. **render/** - Rendering Engines

#### **opengl/**
```cpp
class OpenGLRenderer : public IRenderer {
    // Modern OpenGL (4.5+) with DSA
    GLuint vao, vbo, ebo;
    
    // Efficient FEM mesh rendering
    void render_fem_mesh(const FEMMesh& mesh,
                        const RenderOptions& opts);
    
    // Instanced rendering for repeated elements
    void render_instanced(const InstanceData& instances);
    
    // Texture-based field visualization
    void render_field_textured(const Field& field,
                              const ColorMap& colormap);
    
    // GPU buffer management
    class BufferPool {
        std::unordered_map<size_t, GLuint> buffers;
        GLuint allocate(size_t size);
        void free(GLuint buffer);
    };
    
    // Shader hot-reload
    void reload_shaders();
};
```

#### **vulkan/**
```cpp
class VulkanRenderer : public IRenderer {
    // Vulkan for high-performance visualization
    VkDevice device;
    VkSwapchain swapchain;
    
    // Command buffer recording
    void record_commands(const Scene& scene);
    
    // Descriptor set management
    class DescriptorManager {
        void bind_fem_data(const FEMData& data);
        void update_uniforms(const UniformData& uniforms);
    };
    
    // Multi-GPU support
    std::vector<VkPhysicalDevice> physical_devices;
    void distribute_rendering(const Scene& scene);
    
    // Ray tracing acceleration structures
    VkAccelerationStructureKHR create_blas(const Mesh& mesh);
    VkAccelerationStructureKHR create_tlas(const Scene& scene);
};
```

#### **volume/**
```cpp
class VolumeRenderer : public core::Component {
    // Direct volume rendering
    void render_scalar_field(const Field3D& field,
                            const TransferFunction& tf);
    
    // GPU ray casting
    template<typename Device>
    void ray_cast_gpu(const DeviceField3D& field,
                     const Camera& camera,
                     Image& output);
    
    // Adaptive sampling
    void render_adaptive(const Field3D& field,
                        float error_threshold);
    
    // Multi-resolution volumes
    class OctreeVolume {
        void build_from_field(const Field3D& field);
        void render_lod(const Camera& camera);
    };
    
    // Pre-integration for unstructured meshes
    void preintegrate_tetrahedra(const TetraMesh& mesh,
                                const Field& field);
};
```

### 3. **data_mappers/** - FEM Data to Graphics Mapping

#### **field_mapper/**
```cpp
class FieldMapper : public core::Component {
    // Map scalar fields to colors
    void map_scalar_to_color(const Field& field,
                            const ColorMap& colormap,
                            ColorArray& colors);
    
    // Map fields to geometry attributes
    void map_to_displacement(const Field& displacement,
                            float scale,
                            VertexArray& vertices);
    
    // Complex field mapping
    void map_complex_field(const ComplexField& field,
                          MappingMode mode,  // MAGNITUDE, PHASE, REAL, IMAG
                          ColorArray& colors);
    
    // Tensor field visualization
    void map_stress_tensor(const TensorField& stress,
                          GlyphArray& glyphs);
    
    // GPU-accelerated mapping
    template<typename Device>
    void map_field_gpu(const DeviceField& field,
                      DeviceColorArray& colors);
};
```

#### **vector_mapper/**
```cpp
class VectorMapper : public core::Component {
    // Arrow glyphs
    void create_arrows(const VectorField& field,
                      const MeshPoints& points,
                      GlyphArray& arrows);
    
    // Streamlines
    void compute_streamlines(const VectorField& velocity,
                           const SeedPoints& seeds,
                           StreamlineSet& streamlines);
    
    // Line Integral Convolution (LIC)
    void compute_lic(const VectorField2D& field,
                    Image& output);
    
    // Pathlines for time-varying fields
    void compute_pathlines(const TimeVaryingField& field,
                         const SeedPoints& seeds,
                         double t_start, double t_end,
                         PathlineSet& pathlines);
    
    // GPU particle advection
    void advect_particles_gpu(const DeviceVectorField& field,
                            DeviceParticles& particles,
                            float dt);
};
```

### 4. **filters/** - Data Filtering and Processing

#### **contour/**
```cpp
class ContourFilter : public core::Component {
    // Marching cubes for iso-surfaces
    void marching_cubes(const Field3D& field,
                       double iso_value,
                       TriangleMesh& surface);
    
    // Marching tetrahedra for unstructured
    void marching_tetrahedra(const TetraMesh& mesh,
                           const Field& field,
                           double iso_value,
                           TriangleMesh& surface);
    
    // GPU-accelerated contouring
    void marching_cubes_gpu(const DeviceField3D& field,
                          double iso_value,
                          DeviceTriangleMesh& surface);
    
    // Multiple iso-values
    void multi_contour(const Field3D& field,
                      const std::vector<double>& iso_values,
                      std::vector<TriangleMesh>& surfaces);
    
    // Adaptive contouring
    void adaptive_contour(const Field3D& field,
                        double iso_value,
                        float error_tolerance);
};
```

#### **clip/**
```cpp
class ClipFilter : public core::Component {
    // Clip by plane
    void clip_by_plane(const Mesh& mesh,
                      const Plane& plane,
                      Mesh& clipped_mesh);
    
    // Clip by box
    void clip_by_box(const Mesh& mesh,
                    const Box& box,
                    Mesh& clipped_mesh);
    
    // Clip by arbitrary surface
    void clip_by_surface(const Mesh& mesh,
                        const Surface& surface,
                        Mesh& clipped_mesh);
    
    // Clip with field interpolation
    void clip_with_field(const Mesh& mesh,
                        const Field& field,
                        const ClipFunction& func,
                        Mesh& clipped_mesh,
                        Field& clipped_field);
    
    // GPU clipping
    void clip_gpu(const DeviceMesh& mesh,
                 const ClipFunction& func,
                 DeviceMesh& result);
};
```

### 5. **insitu/** - In-Situ Visualization

#### **catalyst/**
```cpp
class CatalystAdaptor : public core::Component {
    // ParaView Catalyst integration
    void initialize(const std::string& script);
    
    // Co-processing
    void coprocess(const FEMState& state,
                  double time,
                  int timestep);
    
    // Data description
    class DataDescription {
        void add_mesh(const std::string& name,
                     const Mesh& mesh);
        void add_field(const std::string& name,
                      const Field& field,
                      Association assoc);  // POINT, CELL
    };
    
    // Trigger conditions
    void set_trigger(std::function<bool(const FEMState&)> trigger);
    
    // Pipeline configuration
    void configure_pipeline(const CatalystPipeline& pipeline);
    
    // Steering support
    void enable_steering(SteeringCallback callback);
};
```

#### **data_reduction/**
```cpp
class InSituReducer : public core::Component {
    // Wavelet compression
    void compress_wavelet(const Field& field,
                         float tolerance,
                         CompressedField& compressed);
    
    // Statistical reduction
    Statistics compute_statistics(const Field& field);
    
    // Feature extraction
    void extract_features(const Field& field,
                        FeatureSet& features);
    
    // Sampling strategies
    void importance_sample(const Field& field,
                         int n_samples,
                         SampleSet& samples);
    
    // Cinema database generation
    void generate_cinema_db(const FEMState& state,
                          const CinemaSpec& spec);
};
```

### 6. **remote/** - Remote Visualization

#### **server/**
```cpp
class RenderServer : public core::Object {
    // Multi-client support
    std::vector<ClientConnection> clients;
    
    // Load balancing
    void distribute_rendering(const RenderRequest& request);
    
    // GPU rendering pool
    class GPURenderPool {
        std::vector<GPUContext> contexts;
        GPUContext* acquire();
        void release(GPUContext* ctx);
    };
    
    // Streaming encoder
    class StreamEncoder {
        void encode_frame(const Image& frame,
                         VideoStream& stream);
        void encode_h264(const Image& frame);
        void encode_vp9(const Image& frame);
    };
    
    // Adaptive quality
    void adjust_quality(const NetworkMetrics& metrics);
    
    // State synchronization
    void sync_state(const FEMState& state);
};
```

#### **client/**
```cpp
class VisualizationClient : public core::Object {
    // Thin client implementation
    void connect(const std::string& server_address);
    
    // Interaction handling
    void send_camera_update(const Camera& camera);
    void send_pick_request(int x, int y);
    
    // Progressive rendering
    void request_progressive(const RenderRequest& request);
    
    // Decoding and display
    void decode_stream(const VideoStream& stream);
    
    // Local rendering fallback
    void enable_hybrid_rendering();
    
    // WebRTC support
    void establish_webrtc_connection();
};
```

### 7. **interactive/** - Interactive Components

#### **widgets/**
```cpp
class Widget3D : public core::Component {
    // Transform widget
    class TransformWidget : public Widget3D {
        void render(const Camera& camera);
        bool handle_mouse(const MouseEvent& event);
        Signal<Transform> on_transform_changed;
    };
    
    // Plane widget for slicing
    class PlaneWidget : public Widget3D {
        Plane plane;
        void set_bounds(const BoundingBox& bounds);
        Signal<Plane> on_plane_changed;
    };
    
    // Probe widget
    class ProbeWidget : public Widget3D {
        Point3D position;
        void update_position(const Point3D& pos);
        Signal<Point3D> on_probe_moved;
    };
    
    // Box widget for ROI selection
    class BoxWidget : public Widget3D {
        Box box;
        void set_bounds(const BoundingBox& bounds);
        Signal<Box> on_box_changed;
    };
};
```

#### **picking/**
```cpp
class PickingManager : public core::Component {
    // Hardware picking with selection buffer
    PickResult pick_hardware(int x, int y);
    
    // Software ray casting
    PickResult pick_software(const Ray& ray,
                           const Scene& scene);
    
    // Pick specific entity types
    Element* pick_element(int x, int y);
    Node* pick_node(int x, int y);
    Face* pick_face(int x, int y);
    
    // Area selection
    std::vector<Entity*> pick_area(const Rectangle& area);
    
    // GPU-accelerated picking
    void pick_gpu(const DeviceScene& scene,
                 int x, int y,
                 DevicePickResult& result);
};
```

### 8. **export/** - Export Capabilities

#### **images/**
```cpp
class ImageExporter : public core::Component {
    // High-resolution rendering
    void export_high_res(const Scene& scene,
                       const Camera& camera,
                       int width, int height,
                       const std::string& filename);
    
    // Multi-sampling anti-aliasing
    void export_antialiased(const Scene& scene,
                          int samples,
                          const std::string& filename);
    
    // HDR export
    void export_hdr(const Scene& scene,
                   const std::string& filename);  // .exr
    
    // Tiled rendering for huge images
    void export_tiled(const Scene& scene,
                     int total_width,
                     int total_height,
                     int tile_size,
                     const std::string& filename);
    
    // 360° panorama
    void export_panorama(const Scene& scene,
                       const Point3D& center,
                       const std::string& filename);
};
```

#### **movies/**
```cpp
class MovieExporter : public core::Component {
    // Animation export
    void export_animation(const Animation& animation,
                        const MovieSettings& settings,
                        const std::string& filename);
    
    // Time-varying field animation
    void export_temporal(const TemporalField& field,
                       double t_start, double t_end,
                       int fps,
                       const std::string& filename);
    
    // Camera path animation
    void export_flythrough(const Scene& scene,
                         const CameraPath& path,
                         const std::string& filename);
    
    // Encoding options
    struct MovieSettings {
        Codec codec;  // H264, H265, VP9, AV1
        int bitrate;
        int fps;
        Quality quality;
    };
};
```

### 9. **analytics/** - Visual Analytics

#### **statistics/**
```cpp
class StatisticalVisualizer : public core::Component {
    // Histogram visualization
    void create_histogram(const Field& field,
                        int n_bins,
                        Chart& histogram);
    
    // Box plots
    void create_boxplot(const std::vector<Field>& fields,
                      Chart& boxplot);
    
    // Scatter plots with regression
    void create_scatter(const Field& x,
                      const Field& y,
                      Chart& scatter);
    
    // Parallel coordinates
    void create_parallel_coords(const MultiField& fields,
                              Chart& parallel);
    
    // Statistical overlays
    void overlay_statistics(const Field& field,
                          Renderable& overlay);
};
```

#### **uncertainty/**
```cpp
class UncertaintyVisualizer : public core::Component {
    // Uncertainty glyphs
    void create_uncertainty_glyphs(const Field& mean,
                                 const Field& stddev,
                                 GlyphArray& glyphs);
    
    // Confidence intervals
    void render_confidence_bands(const Field& mean,
                               const Field& confidence,
                               Mesh& bands);
    
    // Ensemble visualization
    void visualize_ensemble(const std::vector<Field>& ensemble,
                          VisualizationMode mode);
    
    // Error bars
    void add_error_bars(const Points& points,
                      const Errors& errors,
                      LineSet& bars);
};
```

### 10. **shaders/** - GPU Shaders

#### **vertex/**
```glsl
// FEM-specific vertex shader
#version 450 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in float field_value;

uniform mat4 mvp_matrix;
uniform mat4 normal_matrix;
uniform float displacement_scale;

out vec3 world_normal;
out float field_interp;

void main() {
    // Apply field-based displacement
    vec3 displaced = position + normal * field_value * displacement_scale;
    
    gl_Position = mvp_matrix * vec4(displaced, 1.0);
    world_normal = mat3(normal_matrix) * normal;
    field_interp = field_value;
}
```

#### **fragment/**
```glsl
// PBR fragment shader for FEM
#version 450 core

in vec3 world_normal;
in float field_interp;

uniform sampler1D colormap;
uniform float min_value;
uniform float max_value;

out vec4 frag_color;

vec3 apply_colormap(float value) {
    float normalized = (value - min_value) / (max_value - min_value);
    return texture(colormap, normalized).rgb;
}

void main() {
    vec3 color = apply_colormap(field_interp);
    
    // Simple shading
    vec3 light_dir = normalize(vec3(1, 1, 1));
    float ndotl = max(dot(normalize(world_normal), light_dir), 0.0);
    
    frag_color = vec4(color * (0.3 + 0.7 * ndotl), 1.0);
}
```

## Integration Examples

### Example 1: Basic FEM Visualization
```cpp
// Setup basic FEM visualization
void visualize_fem_solution(const FEMSolution& solution) {
    // Create scene
    Scene scene;
    
    // Add FEM mesh with field
    auto fem_actor = std::make_unique<FEMActor>();
    fem_actor->bind_fem_data(solution.mesh, solution.field);
    
    // Configure field mapping
    FieldMapper mapper;
    ColorMap jet_colormap = ColorMap::Jet();
    mapper.map_scalar_to_color(solution.field, jet_colormap,
                              fem_actor->colors);
    
    scene.add_actor(std::move(fem_actor));
    
    // Setup camera
    Camera camera;
    camera.fit_to_bounds(solution.mesh.bounds());
    
    // Render
    OpenGLRenderer renderer;
    renderer.render(scene, camera);
}
```

### Example 2: In-Situ Visualization
```cpp
// Configure in-situ visualization
void setup_insitu(FEMSolver& solver) {
    CatalystAdaptor catalyst;
    catalyst.initialize("pipeline.py");
    
    // Register callback
    solver.on_timestep([&](const FEMState& state, double time) {
        // Only visualize every 10 timesteps
        if (state.timestep % 10 == 0) {
            DataDescription desc;
            desc.add_mesh("mesh", state.mesh);
            desc.add_field("velocity", state.velocity, POINT);
            desc.add_field("pressure", state.pressure, CELL);
            
            catalyst.coprocess(state, time, state.timestep);
        }
    });
    
    // Data reduction for key features
    InSituReducer reducer;
    solver.on_timestep([&](const FEMState& state, double time) {
        auto features = reducer.extract_features(state.vorticity);
        features.save("features_" + std::to_string(state.timestep));
    });
}
```

### Example 3: Interactive Analysis
```cpp
// Interactive visualization with picking
class InteractiveVisualizer {
    Scene scene;
    Camera camera;
    PickingManager picker;
    ProbeWidget probe;
    
    void on_mouse_click(int x, int y) {
        // Pick element
        Element* elem = picker.pick_element(x, y);
        if (elem) {
            // Show element info
            display_element_info(elem);
            
            // Place probe at element center
            probe.set_position(elem->center());
        }
    }
    
    void on_probe_moved(const Point3D& pos) {
        // Extract field value at probe
        double value = field.interpolate_at(pos);
        display_probe_value(value);
        
        // Update streamlines from probe
        VectorMapper mapper;
        StreamlineSet streamlines;
        mapper.compute_streamlines(velocity_field, {pos},
                                 streamlines);
        update_streamlines(streamlines);
    }
};
```

### Example 4: Remote Visualization
```cpp
// Setup remote visualization server
void start_remote_server() {
    RenderServer server;
    server.listen(8080);
    
    server.on_connection([](ClientConnection& client) {
        // Send initial state
        client.send_scene(current_scene);
        
        // Handle client interactions
        client.on_camera_change([](const Camera& cam) {
            render_and_stream(current_scene, cam);
        });
        
        client.on_quality_request([](QualityLevel level) {
            adjust_rendering_quality(level);
        });
    });
    
    // Adaptive streaming
    server.enable_adaptive_quality();
    server.set_target_framerate(30);
}
```

### Example 5: Publication Quality Export
```cpp
// Generate publication figures
void export_publication_figures(const FEMSolution& solution) {
    Scene scene;
    setup_publication_scene(scene, solution);
    
    ImageExporter exporter;
    
    // High-res with anti-aliasing
    exporter.export_antialiased(scene, 8,  // 8x MSAA
                               "figure_1.png");
    
    // Multiple views
    std::vector<Camera> views = {
        Camera::front_view(),
        Camera::iso_view(),
        Camera::top_view()
    };
    
    for (size_t i = 0; i < views.size(); ++i) {
        exporter.export_high_res(scene, views[i],
                               4000, 3000,
                               "view_" + std::to_string(i) + ".png");
    }
    
    // Animation of time evolution
    MovieExporter movie_exporter;
    movie_exporter.export_temporal(solution.temporal_field,
                                  0.0, 10.0, 60,  // 60 fps
                                  "evolution.mp4");
}
```

## Python Bindings

```python
# Python visualization API
import femvis as fv

# Load solution
solution = fv.load_solution("result.fem")

# Create visualization
vis = fv.Visualizer()
vis.add_mesh(solution.mesh)
vis.add_field(solution.stress, colormap='viridis')

# Add iso-surfaces
contour = fv.ContourFilter()
iso_surface = contour.compute(solution.field, iso_value=0.5)
vis.add_surface(iso_surface)

# Interactive widgets
probe = fv.ProbeWidget()
probe.on_move = lambda pos: print(f"Value: {solution.field.at(pos)}")
vis.add_widget(probe)

# Render
vis.show()

# Export
vis.screenshot("result.png", resolution=(1920, 1080))
vis.export_animation("animation.mp4", fps=30)

# In-situ setup
catalyst = fv.CatalystAdaptor()
catalyst.add_pipeline("isosurface.py")
catalyst.add_trigger(lambda state: state.time % 0.1 < 1e-6)
```

## Performance Optimizations

### GPU Utilization
```cpp
// GPU-optimized field rendering
class GPUFieldRenderer {
    // Upload field as 3D texture
    void upload_field(const Field3D& field) {
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F,
                    field.nx, field.ny, field.nz,
                    0, GL_RED, GL_FLOAT, field.data());
    }
    
    // Hardware interpolation
    void enable_trilinear() {
        glTexParameteri(GL_TEXTURE_3D, 
                       GL_TEXTURE_MIN_FILTER,
                       GL_LINEAR);
    }
};
```

### Level-of-Detail
```cpp
// Automatic LOD generation
class LODGenerator {
    std::vector<Mesh> generate_lods(const Mesh& mesh) {
        std::vector<Mesh> lods;
        Mesh current = mesh;
        
        for (int level = 0; level < 5; ++level) {
            current = decimate(current, 0.5);  // 50% reduction
            lods.push_back(current);
        }
        
        return lods;
    }
};
```

### Memory Management
```cpp
// Out-of-core rendering for massive datasets
class OutOfCoreRenderer {
    // Octree-based data structure
    OctreeNode* root;
    
    // Stream visible nodes only
    void render(const Camera& camera) {
        auto visible_nodes = cull_nodes(root, camera.frustum());
        
        for (auto* node : visible_nodes) {
            if (!node->is_loaded()) {
                node->load_async();
            }
            render_node(node);
        }
    }
};
```

## Testing Strategy

```
visualization/tests/
├── unit/              # Component tests
├── rendering/         # Rendering correctness
├── performance/       # Frame rate benchmarks
├── integration/       # Pipeline tests
└── validation/        # Image comparison tests
```

## Build Configuration

```cmake
# Visualization module options
option(FEM_VIS_OPENGL "OpenGL renderer" ON)
option(FEM_VIS_VULKAN "Vulkan renderer" OFF)
option(FEM_VIS_INSITU "In-situ support" ON)
option(FEM_VIS_REMOTE "Remote rendering" ON)
option(FEM_VIS_VR "VR support" OFF)

# Graphics dependencies
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# Optional dependencies
find_package(VTK COMPONENTS RenderingOpenGL2)
find_package(ParaView COMPONENTS Catalyst)
find_package(Vulkan QUIET)

# Shader compilation
add_shader_directory(${CMAKE_CURRENT_SOURCE_DIR}/shaders)
```

## Success Metrics

1. **Performance**: 60+ FPS for 10M element meshes
2. **Scalability**: Linear scaling to 100+ GPUs
3. **Quality**: Publication-ready output
4. **Interactivity**: < 16ms response time
5. **Memory**: Handle 100GB+ datasets
6. **Flexibility**: Support all FEM element types
7. **Compatibility**: OpenGL 4.5+, Vulkan 1.2+

## Implementation Roadmap

### Phase 1: Core Rendering (Months 1-2)
- Basic OpenGL renderer
- Mesh and field visualization
- Camera controls
- Color mapping

### Phase 2: Advanced Features (Months 3-4)
- Filters (clip, slice, contour)
- Vector field visualization
- Volume rendering
- Interactive widgets

### Phase 3: Performance (Months 5-6)
- GPU optimizations
- Level-of-detail
- Vulkan renderer
- Out-of-core support

### Phase 4: Integration (Months 7-8)
- In-situ visualization
- Remote rendering
- Python bindings
- Export capabilities

## Notes

- All components inherit from `core::Object` for ECS integration
- GPU kernels share device abstraction with solver
- Shaders are hot-reloadable for development
- Memory pooling for dynamic scenes
- Support for both immediate and retained mode rendering