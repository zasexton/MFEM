# Core Tracing - AGENT.md

## Purpose
The `tracing/` layer provides comprehensive distributed tracing and profiling infrastructure for performance analysis, debugging, and system observability. It implements OpenTelemetry-compatible tracing with support for spans, contexts, sampling, and various export formats while maintaining minimal overhead when disabled.

## Architecture Philosophy
- **Zero-overhead when disabled**: Compile-time elimination in release builds
- **OpenTelemetry compatible**: Industry-standard trace format
- **Distributed tracing**: Support for multi-process/multi-machine traces
- **Automatic propagation**: Context flows through async operations
- **Flexible sampling**: Reduce overhead with intelligent sampling

## Files Overview

### Core Components
```cpp
tracer.hpp           // Main tracer interface and management
span.hpp             // Span creation and lifecycle
trace_context.hpp    // Trace context propagation
trace_id.hpp         // Trace and span ID generation
span_context.hpp     // Span context for propagation
```

### Span Management
```cpp
span_builder.hpp     // Fluent span construction
span_processor.hpp   // Span processing pipeline
span_attributes.hpp  // Span attribute management
span_events.hpp      // Span event recording
span_links.hpp       // Span relationship links
span_status.hpp      // Span status and errors
```

### Context Propagation
```cpp
context_propagator.hpp  // Context injection/extraction
baggage.hpp            // Cross-cutting concern data
correlation_context.hpp // Correlation ID management
thread_local_context.hpp // Thread-local storage
async_context.hpp      // Async operation context
```

### Sampling
```cpp
sampler.hpp          // Base sampler interface
always_sampler.hpp   // Always sample (100%)
never_sampler.hpp    // Never sample (0%)
probability_sampler.hpp // Probabilistic sampling
rate_limit_sampler.hpp // Rate-limited sampling
adaptive_sampler.hpp // Adaptive sampling based on load
parent_based_sampler.hpp // Respect parent sampling decision
```

### Exporters
```cpp
exporter.hpp         // Base exporter interface
console_exporter.hpp // Console output for debugging
json_exporter.hpp    // JSON format export
otlp_exporter.hpp    // OpenTelemetry Protocol
zipkin_exporter.hpp  // Zipkin compatibility
jaeger_exporter.hpp  // Jaeger compatibility
file_exporter.hpp    // File-based export
```

### Performance
```cpp
span_buffer.hpp      // Buffering for batch export
trace_metrics.hpp    // Tracing system metrics
overhead_tracker.hpp // Track tracing overhead
memory_pool.hpp      // Span memory pooling
lock_free_queue.hpp  // Lock-free span queue
```

### Utilities
```cpp
trace_macros.hpp     // Convenience macros
trace_annotations.hpp // Code annotations
flame_graph.hpp      // Flame graph generation
trace_viewer.hpp     // Built-in trace viewer
benchmark_trace.hpp  // Performance benchmarking
```

## Detailed Component Specifications

### `tracer.hpp`
```cpp
class Tracer {
public:
    struct Config {
        std::string service_name = "unknown_service";
        std::string service_version = "unknown";
        std::string deployment_environment = "unknown";
        size_t max_spans_per_trace = 1000;
        size_t max_attributes_per_span = 128;
        size_t max_events_per_span = 128;
        size_t max_links_per_span = 128;
        std::chrono::milliseconds export_timeout{30000};
        size_t export_batch_size = 512;
    };
    
private:
    Config config_;
    std::unique_ptr<Sampler> sampler_;
    std::vector<std::unique_ptr<SpanProcessor>> processors_;
    std::vector<std::unique_ptr<Exporter>> exporters_;
    std::unique_ptr<IdGenerator> id_generator_;
    std::shared_ptr<Resource> resource_;
    std::atomic<bool> enabled_{true};
    
    // Singleton
    Tracer() = default;
    static Tracer& instance() {
        static Tracer tracer;
        return tracer;
    }
    
public:
    // Configuration
    static void configure(const Config& config) {
        instance().config_ = config;
        instance().setup_resource();
    }
    
    static void set_sampler(std::unique_ptr<Sampler> sampler) {
        instance().sampler_ = std::move(sampler);
    }
    
    static void add_exporter(std::unique_ptr<Exporter> exporter) {
        instance().exporters_.push_back(std::move(exporter));
    }
    
    static void add_processor(std::unique_ptr<SpanProcessor> processor) {
        instance().processors_.push_back(std::move(processor));
    }
    
    // Span creation
    static std::unique_ptr<Span> start_span(const std::string& name,
                                            const SpanContext& parent = {}) {
        if (!instance().enabled_) {
            return std::make_unique<NoOpSpan>();
        }
        
        auto& tracer = instance();
        
        // Sampling decision
        auto sampling_result = tracer.sampler_->should_sample(
            parent, name, SpanKind::Internal, {}, {}
        );
        
        if (!sampling_result.is_sampled()) {
            return std::make_unique<NoOpSpan>();
        }
        
        auto span = std::make_unique<SpanImpl>(
            name,
            tracer.id_generator_->generate_trace_id(),
            tracer.id_generator_->generate_span_id(),
            parent,
            std::chrono::steady_clock::now()
        );
        
        span->set_attributes(sampling_result.attributes());
        
        // Notify processors
        for (auto& processor : tracer.processors_) {
            processor->on_start(*span);
        }
        
        return span;
    }
    
    // Context management
    static SpanContext current_context() {
        return ThreadLocalContext::get_current();
    }
    
    static void set_current_context(const SpanContext& ctx) {
        ThreadLocalContext::set_current(ctx);
    }
    
    // Enable/disable
    static void enable() { instance().enabled_ = true; }
    static void disable() { instance().enabled_ = false; }
    static bool is_enabled() { return instance().enabled_; }
    
    // Shutdown
    static void shutdown() {
        auto& tracer = instance();
        
        // Flush all processors
        for (auto& processor : tracer.processors_) {
            processor->shutdown();
        }
        
        // Flush all exporters
        for (auto& exporter : tracer.exporters_) {
            exporter->shutdown();
        }
    }
    
private:
    void setup_resource() {
        resource_ = std::make_shared<Resource>();
        resource_->add_attribute("service.name", config_.service_name);
        resource_->add_attribute("service.version", config_.service_version);
        resource_->add_attribute("deployment.environment", config_.deployment_environment);
        resource_->add_attribute("telemetry.sdk.name", "fem-tracer");
        resource_->add_attribute("telemetry.sdk.version", "1.0.0");
        resource_->add_attribute("telemetry.sdk.language", "cpp");
    }
};

// Global tracer access
inline Tracer& tracer() {
    return Tracer::instance();
}
```
**Why necessary**: Central management of tracing infrastructure, span creation, context propagation.
**Usage**: Application-wide tracing configuration, span lifecycle management.

### `span.hpp`
```cpp
class Span {
public:
    enum class SpanKind {
        Internal,  // Internal operation
        Server,    // Server-side request handler
        Client,    // Client-side request
        Producer,  // Message producer
        Consumer   // Message consumer
    };
    
    enum class StatusCode {
        Unset,
        Ok,
        Error
    };
    
    virtual ~Span() = default;
    
    // Core operations
    virtual void set_name(const std::string& name) = 0;
    virtual void end(std::chrono::steady_clock::time_point end_time = 
                     std::chrono::steady_clock::now()) = 0;
    
    // Context
    virtual SpanContext context() const = 0;
    virtual bool is_recording() const = 0;
    
    // Attributes
    virtual void set_attribute(const std::string& key, const AttributeValue& value) = 0;
    virtual void set_attributes(const AttributeMap& attributes) = 0;
    
    // Events
    virtual void add_event(const std::string& name,
                          const AttributeMap& attributes = {},
                          std::chrono::steady_clock::time_point timestamp = 
                          std::chrono::steady_clock::now()) = 0;
    
    // Status
    virtual void set_status(StatusCode code, const std::string& description = "") = 0;
    
    // Links
    virtual void add_link(const SpanContext& context,
                         const AttributeMap& attributes = {}) = 0;
    
    // Convenience methods
    template<typename T>
    void set_attribute(const std::string& key, T&& value) {
        set_attribute(key, AttributeValue(std::forward<T>(value)));
    }
    
    void record_exception(const std::exception& e) {
        add_event("exception", {
            {"exception.type", typeid(e).name()},
            {"exception.message", e.what()},
            {"exception.stacktrace", capture_stacktrace()}
        });
        set_status(StatusCode::Error, e.what());
    }
};

class SpanImpl : public Span {
    std::string name_;
    TraceId trace_id_;
    SpanId span_id_;
    SpanId parent_span_id_;
    SpanKind kind_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point end_time_;
    StatusCode status_{StatusCode::Unset};
    std::string status_description_;
    
    AttributeMap attributes_;
    std::vector<Event> events_;
    std::vector<Link> links_;
    
    std::atomic<bool> ended_{false};
    mutable std::mutex mutex_;
    
public:
    SpanImpl(const std::string& name,
             const TraceId& trace_id,
             const SpanId& span_id,
             const SpanContext& parent,
             std::chrono::steady_clock::time_point start)
        : name_(name)
        , trace_id_(trace_id)
        , span_id_(span_id)
        , parent_span_id_(parent.span_id())
        , start_time_(start) {
    }
    
    ~SpanImpl() {
        if (!ended_) {
            end();
        }
    }
    
    void end(std::chrono::steady_clock::time_point end_time) override {
        bool expected = false;
        if (!ended_.compare_exchange_strong(expected, true)) {
            return;  // Already ended
        }
        
        std::lock_guard lock(mutex_);
        end_time_ = end_time;
        
        // Notify processors
        for (auto& processor : Tracer::instance().processors_) {
            processor->on_end(*this);
        }
    }
    
    void set_attribute(const std::string& key, const AttributeValue& value) override {
        std::lock_guard lock(mutex_);
        attributes_[key] = value;
    }
    
    void add_event(const std::string& name,
                   const AttributeMap& attributes,
                   std::chrono::steady_clock::time_point timestamp) override {
        std::lock_guard lock(mutex_);
        events_.push_back({name, attributes, timestamp});
    }
    
    SpanContext context() const override {
        return SpanContext(trace_id_, span_id_, TraceFlags::Sampled);
    }
    
    // Export data
    SpanData to_span_data() const {
        std::lock_guard lock(mutex_);
        return SpanData{
            .name = name_,
            .trace_id = trace_id_,
            .span_id = span_id_,
            .parent_span_id = parent_span_id_,
            .kind = kind_,
            .start_time = start_time_,
            .end_time = end_time_,
            .status = status_,
            .status_description = status_description_,
            .attributes = attributes_,
            .events = events_,
            .links = links_
        };
    }
};

// RAII span guard
class ScopedSpan {
    std::unique_ptr<Span> span_;
    
public:
    explicit ScopedSpan(const std::string& name)
        : span_(Tracer::start_span(name)) {
        // Set as current context
        Tracer::set_current_context(span_->context());
    }
    
    ~ScopedSpan() {
        span_->end();
        // Restore previous context
        Tracer::set_current_context(SpanContext{});
    }
    
    Span* operator->() { return span_.get(); }
    Span& operator*() { return *span_; }
};
```
**Why necessary**: Represents a unit of work, tracks timing and metadata, hierarchical relationship.
**Usage**: Function tracing, request tracking, performance monitoring.

### `sampler.hpp`
```cpp
class Sampler {
public:
    struct SamplingResult {
        enum Decision {
            Drop,      // Don't sample
            RecordOnly, // Record but don't export
            RecordAndSample // Record and export
        };
        
        Decision decision;
        AttributeMap attributes;
        
        bool is_sampled() const {
            return decision != Drop;
        }
    };
    
    virtual ~Sampler() = default;
    
    virtual SamplingResult should_sample(
        const SpanContext& parent_context,
        const std::string& name,
        Span::SpanKind kind,
        const AttributeMap& initial_attributes,
        const std::vector<Link>& links
    ) = 0;
    
    virtual std::string description() const = 0;
};

class ProbabilitySampler : public Sampler {
    double probability_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> dist_{0.0, 1.0};
    
public:
    explicit ProbabilitySampler(double probability)
        : probability_(std::clamp(probability, 0.0, 1.0))
        , rng_(std::random_device{}()) {
    }
    
    SamplingResult should_sample(
        const SpanContext& parent_context,
        const std::string& name,
        Span::SpanKind kind,
        const AttributeMap& initial_attributes,
        const std::vector<Link>& links
    ) override {
        bool sample = dist_(rng_) < probability_;
        
        return {
            sample ? SamplingResult::RecordAndSample : SamplingResult::Drop,
            {{"sampling.probability", probability_}}
        };
    }
    
    std::string description() const override {
        return "ProbabilitySampler{" + std::to_string(probability_) + "}";
    }
};

class RateLimitSampler : public Sampler {
    size_t max_per_second_;
    std::atomic<size_t> current_second_count_{0};
    std::chrono::steady_clock::time_point current_second_start_;
    mutable std::mutex mutex_;
    
public:
    explicit RateLimitSampler(size_t max_per_second)
        : max_per_second_(max_per_second)
        , current_second_start_(std::chrono::steady_clock::now()) {
    }
    
    SamplingResult should_sample(
        const SpanContext& parent_context,
        const std::string& name,
        Span::SpanKind kind,
        const AttributeMap& initial_attributes,
        const std::vector<Link>& links
    ) override {
        auto now = std::chrono::steady_clock::now();
        
        std::lock_guard lock(mutex_);
        
        // Reset counter if new second
        if (now - current_second_start_ >= std::chrono::seconds(1)) {
            current_second_count_ = 0;
            current_second_start_ = now;
        }
        
        if (current_second_count_ >= max_per_second_) {
            return {SamplingResult::Drop, {}};
        }
        
        current_second_count_++;
        return {SamplingResult::RecordAndSample, 
                {{"sampling.rate_limit", max_per_second_}}};
    }
    
    std::string description() const override {
        return "RateLimitSampler{" + std::to_string(max_per_second_) + "/s}";
    }
};

class AdaptiveSampler : public Sampler {
    struct Statistics {
        std::atomic<size_t> total_spans{0};
        std::atomic<size_t> sampled_spans{0};
        std::atomic<double> average_latency{0};
        std::chrono::steady_clock::time_point window_start;
    };
    
    double target_rate_;
    double min_probability_;
    double max_probability_;
    std::atomic<double> current_probability_;
    Statistics stats_;
    std::chrono::seconds window_size_{60};
    
public:
    AdaptiveSampler(double target_rate, 
                   double min_prob = 0.001, 
                   double max_prob = 1.0)
        : target_rate_(target_rate)
        , min_probability_(min_prob)
        , max_probability_(max_prob)
        , current_probability_((min_prob + max_prob) / 2) {
        stats_.window_start = std::chrono::steady_clock::now();
    }
    
    SamplingResult should_sample(
        const SpanContext& parent_context,
        const std::string& name,
        Span::SpanKind kind,
        const AttributeMap& initial_attributes,
        const std::vector<Link>& links
    ) override {
        update_probability();
        
        bool sample = should_sample_probabilistic(current_probability_);
        
        stats_.total_spans++;
        if (sample) {
            stats_.sampled_spans++;
        }
        
        return {
            sample ? SamplingResult::RecordAndSample : SamplingResult::Drop,
            {{"sampling.adaptive.probability", current_probability_.load()}}
        };
    }
    
private:
    void update_probability() {
        auto now = std::chrono::steady_clock::now();
        if (now - stats_.window_start >= window_size_) {
            // Adjust probability based on actual vs target rate
            double actual_rate = static_cast<double>(stats_.sampled_spans) / 
                                window_size_.count();
            
            if (actual_rate > target_rate_ * 1.1) {
                // Decrease probability
                current_probability_ = std::max(
                    min_probability_,
                    current_probability_ * 0.9
                );
            } else if (actual_rate < target_rate_ * 0.9) {
                // Increase probability
                current_probability_ = std::min(
                    max_probability_,
                    current_probability_ * 1.1
                );
            }
            
            // Reset statistics
            stats_ = Statistics{};
            stats_.window_start = now;
        }
    }
    
    bool should_sample_probabilistic(double probability) {
        static thread_local std::mt19937_64 rng(std::random_device{}());
        static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(rng) < probability;
    }
};
```
**Why necessary**: Control overhead, reduce data volume, maintain statistical accuracy.
**Usage**: Production systems, high-throughput applications, cost management.

### `exporter.hpp`
```cpp
class Exporter {
public:
    virtual ~Exporter() = default;
    
    enum class ExportResult {
        Success,
        Failure,
        FailureRetryable
    };
    
    virtual ExportResult export_spans(const std::vector<SpanData>& spans) = 0;
    virtual void shutdown() = 0;
    virtual void flush() = 0;
};

class ConsoleExporter : public Exporter {
    std::ostream& output_;
    bool pretty_print_;
    mutable std::mutex mutex_;
    
public:
    explicit ConsoleExporter(std::ostream& out = std::cout, bool pretty = true)
        : output_(out), pretty_print_(pretty) {}
    
    ExportResult export_spans(const std::vector<SpanData>& spans) override {
        std::lock_guard lock(mutex_);
        
        for (const auto& span : spans) {
            if (pretty_print_) {
                print_pretty(span);
            } else {
                print_compact(span);
            }
        }
        
        output_.flush();
        return ExportResult::Success;
    }
    
private:
    void print_pretty(const SpanData& span) {
        auto duration = span.end_time - span.start_time;
        auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(duration);
        
        output_ << "╔══════════════════════════════════════════╗\n";
        output_ << "║ " << span.name << "\n";
        output_ << "╟──────────────────────────────────────────╢\n";
        output_ << "║ TraceID: " << span.trace_id.to_hex() << "\n";
        output_ << "║ SpanID:  " << span.span_id.to_hex() << "\n";
        output_ << "║ Parent:  " << span.parent_span_id.to_hex() << "\n";
        output_ << "║ Duration: " << duration_ms.count() << "μs\n";
        
        if (span.status != Span::StatusCode::Unset) {
            output_ << "║ Status: " << to_string(span.status);
            if (!span.status_description.empty()) {
                output_ << " (" << span.status_description << ")";
            }
            output_ << "\n";
        }
        
        if (!span.attributes.empty()) {
            output_ << "║ Attributes:\n";
            for (const auto& [key, value] : span.attributes) {
                output_ << "║   " << key << ": " << to_string(value) << "\n";
            }
        }
        
        if (!span.events.empty()) {
            output_ << "║ Events:\n";
            for (const auto& event : span.events) {
                auto event_offset = event.timestamp - span.start_time;
                auto offset_us = std::chrono::duration_cast<std::chrono::microseconds>(event_offset);
                output_ << "║   [+" << offset_us.count() << "μs] " << event.name << "\n";
            }
        }
        
        output_ << "╚══════════════════════════════════════════╝\n";
    }
    
    void print_compact(const SpanData& span) {
        auto duration = span.end_time - span.start_time;
        auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(duration);
        
        output_ << "[" << format_timestamp(span.start_time) << "] "
                << span.name << " "
                << duration_ms.count() << "μs "
                << "trace:" << span.trace_id.to_hex().substr(0, 8) << " "
                << "span:" << span.span_id.to_hex().substr(0, 8) << "\n";
    }
};

class JsonExporter : public Exporter {
    std::string endpoint_;
    std::unique_ptr<HttpClient> http_client_;
    
public:
    explicit JsonExporter(const std::string& endpoint)
        : endpoint_(endpoint)
        , http_client_(std::make_unique<HttpClient>()) {}
    
    ExportResult export_spans(const std::vector<SpanData>& spans) override {
        nlohmann::json traces_json;
        
        for (const auto& span : spans) {
            traces_json["spans"].push_back(span_to_json(span));
        }
        
        auto response = http_client_->post(endpoint_, traces_json.dump());
        
        if (response.status_code == 200) {
            return ExportResult::Success;
        } else if (response.status_code >= 500) {
            return ExportResult::FailureRetryable;
        } else {
            return ExportResult::Failure;
        }
    }
    
private:
    nlohmann::json span_to_json(const SpanData& span) {
        return {
            {"traceId", span.trace_id.to_hex()},
            {"spanId", span.span_id.to_hex()},
            {"parentSpanId", span.parent_span_id.to_hex()},
            {"name", span.name},
            {"kind", to_string(span.kind)},
            {"startTime", format_timestamp(span.start_time)},
            {"endTime", format_timestamp(span.end_time)},
            {"status", {
                {"code", to_string(span.status)},
                {"description", span.status_description}
            }},
            {"attributes", attributes_to_json(span.attributes)},
            {"events", events_to_json(span.events)},
            {"links", links_to_json(span.links)}
        };
    }
};
```
**Why necessary**: Export trace data to various backends, debugging, integration.
**Usage**: APM integration, debugging, performance analysis.

### `trace_context.hpp`
```cpp
class TraceContext {
    static thread_local std::stack<SpanContext> context_stack_;
    
public:
    // Thread-local context management
    static SpanContext current() {
        if (context_stack_.empty()) {
            return SpanContext{};
        }
        return context_stack_.top();
    }
    
    static void set_current(const SpanContext& ctx) {
        context_stack_.push(ctx);
    }
    
    static void clear_current() {
        if (!context_stack_.empty()) {
            context_stack_.pop();
        }
    }
    
    // RAII context scope
    class Scope {
        bool active_ = false;
        
    public:
        explicit Scope(const SpanContext& ctx) : active_(true) {
            TraceContext::set_current(ctx);
        }
        
        ~Scope() {
            if (active_) {
                TraceContext::clear_current();
            }
        }
        
        // Move-only
        Scope(Scope&& other) : active_(other.active_) {
            other.active_ = false;
        }
        
        Scope& operator=(Scope&& other) {
            if (this != &other) {
                if (active_) {
                    TraceContext::clear_current();
                }
                active_ = other.active_;
                other.active_ = false;
            }
            return *this;
        }
    };
    
    // Context propagation for HTTP
    static std::map<std::string, std::string> inject_http_headers(const SpanContext& ctx) {
        std::map<std::string, std::string> headers;
        
        // W3C Trace Context format
        headers["traceparent"] = format_traceparent(ctx);
        
        if (ctx.has_trace_state()) {
            headers["tracestate"] = ctx.trace_state();
        }
        
        // Baggage propagation
        if (ctx.has_baggage()) {
            headers["baggage"] = format_baggage(ctx.baggage());
        }
        
        return headers;
    }
    
    static SpanContext extract_http_headers(const std::map<std::string, std::string>& headers) {
        SpanContext ctx;
        
        // Extract W3C Trace Context
        auto traceparent_it = headers.find("traceparent");
        if (traceparent_it != headers.end()) {
            ctx = parse_traceparent(traceparent_it->second);
        }
        
        // Extract trace state
        auto tracestate_it = headers.find("tracestate");
        if (tracestate_it != headers.end()) {
            ctx.set_trace_state(tracestate_it->second);
        }
        
        // Extract baggage
        auto baggage_it = headers.find("baggage");
        if (baggage_it != headers.end()) {
            ctx.set_baggage(parse_baggage(baggage_it->second));
        }
        
        return ctx;
    }
    
private:
    static std::string format_traceparent(const SpanContext& ctx) {
        // version-trace_id-span_id-trace_flags
        return "00-" + ctx.trace_id().to_hex() + "-" + 
               ctx.span_id().to_hex() + "-" + 
               format_trace_flags(ctx.trace_flags());
    }
    
    static SpanContext parse_traceparent(const std::string& traceparent) {
        // Parse version-trace_id-span_id-trace_flags format
        // Implementation details...
    }
};

// Async context propagation
class AsyncTraceContext {
    struct AsyncContextData {
        SpanContext context;
        std::chrono::steady_clock::time_point creation_time;
    };
    
    static std::unordered_map<std::thread::id, AsyncContextData> async_contexts_;
    static std::shared_mutex mutex_;
    
public:
    // Capture current context for async operation
    static std::shared_ptr<SpanContext> capture() {
        auto current = TraceContext::current();
        return std::make_shared<SpanContext>(current);
    }
    
    // Restore context in async callback
    static TraceContext::Scope restore(std::shared_ptr<SpanContext> ctx) {
        return TraceContext::Scope(*ctx);
    }
    
    // Wrap async function with context
    template<typename F>
    static auto wrap(F&& func) {
        auto ctx = capture();
        return [ctx, func = std::forward<F>(func)](auto&&... args) {
            auto scope = restore(ctx);
            return func(std::forward<decltype(args)>(args)...);
        };
    }
};
```
**Why necessary**: Context propagation across threads, processes, and network boundaries.
**Usage**: Distributed tracing, async operations, microservices.

### `trace_macros.hpp`
```cpp
// Basic tracing macros
#ifdef ENABLE_TRACING
    #define TRACE_FUNCTION() \
        ScopedSpan _trace_span_(__FUNCTION__)
    
    #define TRACE_SCOPE(name) \
        ScopedSpan _trace_span_(name)
    
    #define TRACE_BLOCK(name) \
        if (ScopedSpan _trace_span_(name); true)
#else
    #define TRACE_FUNCTION() ((void)0)
    #define TRACE_SCOPE(name) ((void)0)
    #define TRACE_BLOCK(name) if (true)
#endif

// Attribute setting macros
#define TRACE_SET_ATTRIBUTE(key, value) \
    do { \
        if (auto* span = Tracer::current_span()) { \
            span->set_attribute(key, value); \
        } \
    } while(0)

#define TRACE_ADD_EVENT(name, ...) \
    do { \
        if (auto* span = Tracer::current_span()) { \
            span->add_event(name, {__VA_ARGS__}); \
        } \
    } while(0)

// Performance tracing
#define TRACE_TIMING(name, code) \
    do { \
        ScopedSpan _timing_span_(name); \
        auto _start_ = std::chrono::high_resolution_clock::now(); \
        code; \
        auto _end_ = std::chrono::high_resolution_clock::now(); \
        auto _duration_ = std::chrono::duration_cast<std::chrono::microseconds>(_end_ - _start_); \
        _timing_span_->set_attribute("duration_us", _duration_.count()); \
    } while(0)

// Conditional tracing
#define TRACE_IF(condition, name) \
    for (std::unique_ptr<ScopedSpan> _span_; \
         (condition) && !_span_; \
         _span_ = std::make_unique<ScopedSpan>(name))

// Async operation tracing
#define TRACE_ASYNC(future, name) \
    TracedFuture(std::move(future), name)

// Exception tracing
#define TRACE_EXCEPTION() \
    do { \
        if (auto* span = Tracer::current_span()) { \
            span->record_exception(std::current_exception()); \
        } \
    } while(0)

// Loop iteration tracing
#define TRACE_LOOP(name, iterations) \
    for (LoopTracer _loop_tracer_(name, iterations); \
         _loop_tracer_.should_continue(); \
         _loop_tracer_.next())

// Critical section tracing
#define TRACE_CRITICAL_SECTION(name, mutex) \
    ScopedSpan _cs_span_(name); \
    _cs_span_->set_attribute("mutex.wait_start", now()); \
    std::lock_guard _lock_(mutex); \
    _cs_span_->set_attribute("mutex.acquired", now())

// Database query tracing
#define TRACE_QUERY(db_type, query) \
    ScopedSpan _query_span_("db.query"); \
    _query_span_->set_attribute("db.type", db_type); \
    _query_span_->set_attribute("db.statement", query)

// HTTP request tracing
#define TRACE_HTTP_REQUEST(method, url) \
    ScopedSpan _http_span_("http.request"); \
    _http_span_->set_attribute("http.method", method); \
    _http_span_->set_attribute("http.url", url); \
    _http_span_->set_attribute("http.scheme", extract_scheme(url))

// Memory allocation tracing
#define TRACE_ALLOCATION(size) \
    do { \
        if (auto* span = Tracer::current_span()) { \
            span->add_event("memory.allocation", { \
                {"size", size}, \
                {"address", reinterpret_cast<uintptr_t>(ptr)} \
            }); \
        } \
    } while(0)
```
**Why necessary**: Convenient tracing without boilerplate, conditional compilation.
**Usage**: Throughout application code, debugging, profiling.

## Tracing Patterns

### Basic Function Tracing
```cpp
void process_data(const Data& data) {
    TRACE_FUNCTION();
    
    TRACE_SET_ATTRIBUTE("data.size", data.size());
    
    {
        TRACE_SCOPE("validation");
        validate(data);
    }
    
    {
        TRACE_SCOPE("transformation");
        auto result = transform(data);
        TRACE_SET_ATTRIBUTE("result.size", result.size());
    }
    
    TRACE_ADD_EVENT("processing_complete");
}
```

### Distributed Tracing
```cpp
// Service A - Client side
void make_request() {
    TRACE_SCOPE("outgoing_request");
    
    auto headers = TraceContext::inject_http_headers(
        Tracer::current_context()
    );
    
    http_client.post("/api/endpoint", data, headers);
}

// Service B - Server side
void handle_request(const Request& req) {
    auto parent_ctx = TraceContext::extract_http_headers(
        req.headers()
    );
    
    auto span = Tracer::start_span("incoming_request", parent_ctx);
    TraceContext::Scope scope(span->context());
    
    // Process request with tracing context
    process_request(req);
}
```

### Async Operation Tracing
```cpp
void async_workflow() {
    TRACE_SCOPE("workflow");
    
    auto ctx = AsyncTraceContext::capture();
    
    std::async(std::launch::async, [ctx]() {
        auto scope = AsyncTraceContext::restore(ctx);
        TRACE_SCOPE("async_task");
        
        // Async work with proper context
        perform_async_work();
    });
}
```

### Performance Analysis
```cpp
void algorithm() {
    TRACE_FUNCTION();
    
    TRACE_TIMING("initialization", {
        initialize();
    });
    
    TRACE_LOOP("iterations", 1000) {
        process_iteration();
    }
    
    TRACE_CRITICAL_SECTION("shared_resource", resource_mutex) {
        update_shared_resource();
    }
}
```

## Performance Considerations

- **Sampling overhead**: ~50-100ns per sampling decision
- **Span creation**: ~200-500ns for sampled spans
- **Attribute setting**: ~20-50ns per attribute
- **Context propagation**: ~100ns for thread-local access
- **Export batching**: Amortizes network overhead
- **Memory pooling**: Reduces allocation overhead

## Testing Strategy

- **Overhead measurement**: Benchmark with/without tracing
- **Sampling accuracy**: Statistical sampling validation
- **Context propagation**: Cross-thread/process testing
- **Export reliability**: Failure and retry testing
- **Memory usage**: Span buffer and pool testing

## Usage Guidelines

1. **Strategic placement**: Trace important operations, not everything
2. **Sampling strategy**: Use appropriate sampling for load
3. **Attribute selection**: Include relevant context, avoid PII
4. **Span naming**: Use consistent, descriptive names
5. **Error recording**: Always record exceptions and errors

## Anti-patterns to Avoid

- Tracing in tight loops without sampling
- Including sensitive data in attributes
- Creating spans without ending them
- Excessive span nesting (>10 levels)
- Synchronous export in critical path

## Dependencies
- `base/` - For Object patterns
- `concurrency/` - For async operations
- `logging/` - For fallback logging
- Standard library (C++20)
- Optional: OpenTelemetry SDK

## Future Enhancements
- Trace aggregation and analysis
- Automatic instrumentation
- Trace-based testing
- Performance regression detection
- Machine learning for anomaly detection