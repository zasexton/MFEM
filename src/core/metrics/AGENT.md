# Core Metrics - AGENT.md

## Purpose
The `metrics/` layer provides comprehensive application metrics collection, aggregation, and export capabilities. It implements a high-performance metrics system supporting counters, gauges, histograms, and summaries with minimal overhead, compatible with industry-standard monitoring systems like Prometheus, StatsD, and OpenTelemetry.

## Architecture Philosophy
- **Lock-free operations**: Minimize contention in hot paths
- **Zero allocation**: Pre-allocated memory pools for metrics
- **Hierarchical namespacing**: Organized metric taxonomy
- **Multiple export formats**: Prometheus, StatsD, JSON, OpenTelemetry
- **Compile-time optimization**: Template-based for zero overhead when disabled

## Files Overview

### Core Components
```cpp
metric.hpp           // Base metric interface
metric_registry.hpp  // Central metric storage and management
metric_name.hpp      // Metric naming and labels
metric_value.hpp     // Metric value types
metric_metadata.hpp  // Metric descriptions and units
```

### Metric Types
```cpp
counter.hpp          // Monotonically increasing values
gauge.hpp            // Arbitrary up/down values
histogram.hpp        // Distribution of values
summary.hpp          // Statistical summaries with quantiles
meter.hpp            // Rate measurements
timer.hpp            // Duration measurements
```

### Aggregation
```cpp
aggregator.hpp       // Metric aggregation strategies
reservoir.hpp        // Sampling reservoirs
sliding_window.hpp   // Time-based sliding windows
exponential_decay.hpp // Exponentially decaying samples
uniform_sample.hpp   // Uniform random sampling
bucket_histogram.hpp // Pre-defined bucket histograms
```

### Export & Formatting
```cpp
exporter.hpp         // Base exporter interface
prometheus_exporter.hpp // Prometheus text format
json_exporter.hpp    // JSON metrics export
statsd_exporter.hpp  // StatsD protocol
otlp_exporter.hpp    // OpenTelemetry protocol
graphite_exporter.hpp // Graphite plaintext protocol
influx_exporter.hpp  // InfluxDB line protocol
```

### Collection & Storage
```cpp
collector.hpp        // Metric collection interface
time_series.hpp      // Time series data storage
metric_snapshot.hpp  // Point-in-time metric capture
metric_buffer.hpp    // Buffering for batch export
retention_policy.hpp // Data retention management
```

### Utilities
```cpp
metric_macros.hpp    // Convenience macros
units.hpp            // Unit definitions and conversion
labels.hpp           // Label/tag management
cardinality.hpp      // Cardinality tracking and limits
metric_math.hpp      // Statistical calculations
benchmark_metrics.hpp // Performance benchmarking
```

## Detailed Component Specifications

### `metric.hpp`
```cpp
class Metric {
public:
    enum class Type {
        Counter,
        Gauge,
        Histogram,
        Summary,
        Meter,
        Timer
    };
    
    struct Metadata {
        std::string name;
        std::string description;
        std::string unit;
        Type type;
        std::map<std::string, std::string> labels;
        std::chrono::system_clock::time_point created;
    };
    
    virtual ~Metric() = default;
    
    // Core interface
    virtual Type type() const = 0;
    virtual const Metadata& metadata() const = 0;
    virtual void reset() = 0;
    
    // Serialization
    virtual void write_to(MetricWriter& writer) const = 0;
    virtual std::string to_string() const = 0;
    
    // Snapshot for export
    virtual MetricSnapshot snapshot() const = 0;
    
protected:
    Metadata metadata_;
    mutable std::atomic<uint64_t> last_updated_{0};
    
    void update_timestamp() const {
        last_updated_.store(
            std::chrono::steady_clock::now().time_since_epoch().count(),
            std::memory_order_relaxed
        );
    }
};

// Type-safe metric reference
template<typename MetricType>
class MetricRef {
    std::shared_ptr<MetricType> metric_;
    
public:
    explicit MetricRef(std::shared_ptr<MetricType> m) : metric_(m) {}
    
    MetricType* operator->() { return metric_.get(); }
    const MetricType* operator->() const { return metric_.get(); }
    MetricType& operator*() { return *metric_; }
    const MetricType& operator*() const { return *metric_; }
    
    bool valid() const { return metric_ != nullptr; }
};
```
**Why necessary**: Common interface for all metric types, metadata management, serialization support.
**Usage**: Base for all metric implementations, registry storage, export operations.

### `counter.hpp`
```cpp
class Counter : public Metric {
    std::atomic<uint64_t> value_{0};
    
public:
    Counter(const std::string& name, 
            const std::string& description = "",
            const std::string& unit = "1")
        : Metric() {
        metadata_.name = name;
        metadata_.description = description;
        metadata_.unit = unit;
        metadata_.type = Type::Counter;
        metadata_.created = std::chrono::system_clock::now();
    }
    
    // Increment operations
    void increment(uint64_t delta = 1) {
        value_.fetch_add(delta, std::memory_order_relaxed);
        update_timestamp();
    }
    
    Counter& operator++() {
        increment();
        return *this;
    }
    
    Counter& operator+=(uint64_t delta) {
        increment(delta);
        return *this;
    }
    
    // Read value
    uint64_t value() const {
        return value_.load(std::memory_order_relaxed);
    }
    
    // Rate calculation
    double rate_per_second(std::chrono::seconds window = std::chrono::seconds(60)) const {
        auto now = std::chrono::steady_clock::now();
        auto last = std::chrono::steady_clock::time_point(
            std::chrono::nanoseconds(last_updated_.load())
        );
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last);
        
        if (duration.count() == 0) return 0.0;
        return static_cast<double>(value()) / duration.count();
    }
    
    Type type() const override { return Type::Counter; }
    
    void reset() override {
        value_.store(0, std::memory_order_relaxed);
    }
    
    MetricSnapshot snapshot() const override {
        return MetricSnapshot{
            .metadata = metadata_,
            .timestamp = std::chrono::system_clock::now(),
            .value = static_cast<double>(value())
        };
    }
};

// Labeled counter for multi-dimensional metrics
template<typename... LabelTypes>
class LabeledCounter : public Metric {
    using LabelTuple = std::tuple<LabelTypes...>;
    using CounterMap = std::unordered_map<LabelTuple, std::unique_ptr<Counter>>;
    
    CounterMap counters_;
    mutable std::shared_mutex mutex_;
    std::array<std::string, sizeof...(LabelTypes)> label_names_;
    
public:
    LabeledCounter(const std::string& name,
                   const std::array<std::string, sizeof...(LabelTypes)>& labels)
        : label_names_(labels) {
        metadata_.name = name;
        metadata_.type = Type::Counter;
    }
    
    Counter& with_labels(LabelTypes... labels) {
        auto key = std::make_tuple(labels...);
        
        {
            std::shared_lock lock(mutex_);
            auto it = counters_.find(key);
            if (it != counters_.end()) {
                return *it->second;
            }
        }
        
        std::unique_lock lock(mutex_);
        auto [it, inserted] = counters_.emplace(
            key, 
            std::make_unique<Counter>(build_name(labels...))
        );
        return *it->second;
    }
    
    void increment(LabelTypes... labels, uint64_t delta = 1) {
        with_labels(labels...).increment(delta);
    }
    
    std::vector<MetricSnapshot> collect_all() const {
        std::shared_lock lock(mutex_);
        std::vector<MetricSnapshot> snapshots;
        
        for (const auto& [labels, counter] : counters_) {
            auto snapshot = counter->snapshot();
            add_labels_to_snapshot(snapshot, labels);
            snapshots.push_back(snapshot);
        }
        
        return snapshots;
    }
    
private:
    std::string build_name(LabelTypes... labels) const {
        // Build name with labels
    }
    
    void add_labels_to_snapshot(MetricSnapshot& snapshot, 
                                const LabelTuple& labels) const {
        // Add labels to snapshot
    }
};
```
**Why necessary**: Track monotonically increasing values like requests, errors, bytes processed.
**Usage**: Request counting, error tracking, throughput monitoring.

### `gauge.hpp`
```cpp
class Gauge : public Metric {
    std::atomic<double> value_{0.0};
    std::atomic<double> min_value_{std::numeric_limits<double>::max()};
    std::atomic<double> max_value_{std::numeric_limits<double>::lowest()};
    
public:
    Gauge(const std::string& name,
          const std::string& description = "",
          const std::string& unit = "1") {
        metadata_.name = name;
        metadata_.description = description;
        metadata_.unit = unit;
        metadata_.type = Type::Gauge;
    }
    
    // Set operations
    void set(double value) {
        value_.store(value, std::memory_order_relaxed);
        update_min_max(value);
        update_timestamp();
    }
    
    void increment(double delta = 1.0) {
        double old_value = value_.load(std::memory_order_relaxed);
        double new_value;
        
        do {
            new_value = old_value + delta;
        } while (!value_.compare_exchange_weak(old_value, new_value,
                                               std::memory_order_relaxed));
        
        update_min_max(new_value);
        update_timestamp();
    }
    
    void decrement(double delta = 1.0) {
        increment(-delta);
    }
    
    // Read operations
    double value() const {
        return value_.load(std::memory_order_relaxed);
    }
    
    double min() const {
        return min_value_.load(std::memory_order_relaxed);
    }
    
    double max() const {
        return max_value_.load(std::memory_order_relaxed);
    }
    
    // Operators
    Gauge& operator=(double value) {
        set(value);
        return *this;
    }
    
    Gauge& operator+=(double delta) {
        increment(delta);
        return *this;
    }
    
    Gauge& operator-=(double delta) {
        decrement(delta);
        return *this;
    }
    
    Type type() const override { return Type::Gauge; }
    
    void reset() override {
        value_.store(0.0, std::memory_order_relaxed);
        min_value_.store(std::numeric_limits<double>::max());
        max_value_.store(std::numeric_limits<double>::lowest());
    }
    
private:
    void update_min_max(double value) {
        double current_min = min_value_.load(std::memory_order_relaxed);
        while (value < current_min) {
            if (min_value_.compare_exchange_weak(current_min, value,
                                                 std::memory_order_relaxed)) {
                break;
            }
        }
        
        double current_max = max_value_.load(std::memory_order_relaxed);
        while (value > current_max) {
            if (max_value_.compare_exchange_weak(current_max, value,
                                                 std::memory_order_relaxed)) {
                break;
            }
        }
    }
};

// Functional gauge that computes value on demand
class FunctionalGauge : public Metric {
    std::function<double()> value_func_;
    
public:
    FunctionalGauge(const std::string& name,
                    std::function<double()> func)
        : value_func_(func) {
        metadata_.name = name;
        metadata_.type = Type::Gauge;
    }
    
    double value() const {
        return value_func_();
    }
    
    MetricSnapshot snapshot() const override {
        return MetricSnapshot{
            .metadata = metadata_,
            .timestamp = std::chrono::system_clock::now(),
            .value = value()
        };
    }
};
```
**Why necessary**: Track values that can go up and down like memory usage, queue size, temperature.
**Usage**: Resource monitoring, system state tracking, business metrics.

### `histogram.hpp`
```cpp
class Histogram : public Metric {
public:
    struct Buckets {
        std::vector<double> boundaries;
        std::vector<std::atomic<uint64_t>> counts;
        std::atomic<uint64_t> total_count{0};
        std::atomic<double> sum{0.0};
        std::atomic<double> sum_of_squares{0.0};
        
        Buckets(std::vector<double> bounds) 
            : boundaries(std::move(bounds))
            , counts(boundaries.size() + 1) {
            std::sort(boundaries.begin(), boundaries.end());
        }
        
        void observe(double value) {
            sum.fetch_add(value, std::memory_order_relaxed);
            sum_of_squares.fetch_add(value * value, std::memory_order_relaxed);
            total_count.fetch_add(1, std::memory_order_relaxed);
            
            // Find bucket and increment
            auto it = std::upper_bound(boundaries.begin(), boundaries.end(), value);
            size_t index = std::distance(boundaries.begin(), it);
            counts[index].fetch_add(1, std::memory_order_relaxed);
        }
    };
    
private:
    std::unique_ptr<Buckets> buckets_;
    
public:
    // Common bucket configurations
    static std::vector<double> linear_buckets(double start, double width, size_t count) {
        std::vector<double> buckets;
        buckets.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            buckets.push_back(start + i * width);
        }
        return buckets;
    }
    
    static std::vector<double> exponential_buckets(double start, double factor, size_t count) {
        std::vector<double> buckets;
        buckets.reserve(count);
        double value = start;
        for (size_t i = 0; i < count; ++i) {
            buckets.push_back(value);
            value *= factor;
        }
        return buckets;
    }
    
    // Default buckets for common use cases
    static std::vector<double> default_buckets() {
        return {0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75,
                1.0, 2.5, 5.0, 7.5, 10.0};
    }
    
    static std::vector<double> http_request_duration_buckets() {
        return {0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
                0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0,
                3.0, 4.0, 5.0, 10.0};
    }
    
    static std::vector<double> rpc_duration_buckets() {
        return {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0};
    }
    
    Histogram(const std::string& name,
             const std::vector<double>& buckets = default_buckets())
        : buckets_(std::make_unique<Buckets>(buckets)) {
        metadata_.name = name;
        metadata_.type = Type::Histogram;
    }
    
    void observe(double value) {
        buckets_->observe(value);
        update_timestamp();
    }
    
    // Statistical queries
    uint64_t count() const {
        return buckets_->total_count.load(std::memory_order_relaxed);
    }
    
    double sum() const {
        return buckets_->sum.load(std::memory_order_relaxed);
    }
    
    double mean() const {
        uint64_t n = count();
        return n > 0 ? sum() / n : 0.0;
    }
    
    double variance() const {
        uint64_t n = count();
        if (n < 2) return 0.0;
        
        double s = sum();
        double ss = buckets_->sum_of_squares.load(std::memory_order_relaxed);
        return (ss - (s * s) / n) / (n - 1);
    }
    
    double stddev() const {
        return std::sqrt(variance());
    }
    
    double quantile(double q) const {
        if (q < 0.0 || q > 1.0) {
            throw std::invalid_argument("Quantile must be in [0, 1]");
        }
        
        uint64_t total = count();
        if (total == 0) return 0.0;
        
        uint64_t target = static_cast<uint64_t>(q * total);
        uint64_t accumulated = 0;
        
        for (size_t i = 0; i < buckets_->counts.size(); ++i) {
            accumulated += buckets_->counts[i].load(std::memory_order_relaxed);
            if (accumulated >= target) {
                // Linear interpolation within bucket
                if (i == 0) {
                    return buckets_->boundaries[0];
                } else if (i == buckets_->boundaries.size()) {
                    return buckets_->boundaries.back();
                } else {
                    double lower = i > 0 ? buckets_->boundaries[i-1] : 0.0;
                    double upper = buckets_->boundaries[i];
                    return lower + (upper - lower) / 2.0;  // Midpoint
                }
            }
        }
        
        return buckets_->boundaries.back();
    }
    
    // Common quantiles
    double median() const { return quantile(0.5); }
    double p95() const { return quantile(0.95); }
    double p99() const { return quantile(0.99); }
    double p999() const { return quantile(0.999); }
    
    Type type() const override { return Type::Histogram; }
    
    MetricSnapshot snapshot() const override {
        MetricSnapshot snap;
        snap.metadata = metadata_;
        snap.timestamp = std::chrono::system_clock::now();
        
        // Add bucket data
        snap.histogram_data = std::make_unique<HistogramData>();
        for (size_t i = 0; i < buckets_->boundaries.size(); ++i) {
            snap.histogram_data->buckets.push_back({
                buckets_->boundaries[i],
                buckets_->counts[i].load(std::memory_order_relaxed)
            });
        }
        
        snap.histogram_data->inf_count = 
            buckets_->counts.back().load(std::memory_order_relaxed);
        snap.histogram_data->sum = sum();
        snap.histogram_data->count = count();
        
        return snap;
    }
};

// Time-windowed histogram
class WindowedHistogram : public Metric {
    struct Window {
        std::chrono::steady_clock::time_point start;
        std::vector<double> values;
        mutable std::mutex mutex;
    };
    
    std::vector<std::unique_ptr<Window>> windows_;
    std::chrono::seconds window_duration_;
    size_t num_windows_;
    std::atomic<size_t> current_window_{0};
    std::thread rotator_;
    std::atomic<bool> running_{true};
    
public:
    WindowedHistogram(const std::string& name,
                     std::chrono::seconds window_duration = std::chrono::seconds(60),
                     size_t num_windows = 5)
        : window_duration_(window_duration)
        , num_windows_(num_windows) {
        metadata_.name = name;
        metadata_.type = Type::Histogram;
        
        for (size_t i = 0; i < num_windows_; ++i) {
            windows_.push_back(std::make_unique<Window>());
            windows_[i]->start = std::chrono::steady_clock::now();
        }
        
        rotator_ = std::thread([this] { rotate_windows(); });
    }
    
    ~WindowedHistogram() {
        running_ = false;
        if (rotator_.joinable()) {
            rotator_.join();
        }
    }
    
    void observe(double value) {
        size_t idx = current_window_.load(std::memory_order_relaxed);
        std::lock_guard lock(windows_[idx]->mutex);
        windows_[idx]->values.push_back(value);
    }
    
    std::vector<double> get_all_values(std::chrono::seconds lookback) const {
        std::vector<double> all_values;
        auto cutoff = std::chrono::steady_clock::now() - lookback;
        
        for (const auto& window : windows_) {
            std::lock_guard lock(window->mutex);
            if (window->start >= cutoff) {
                all_values.insert(all_values.end(), 
                                 window->values.begin(), 
                                 window->values.end());
            }
        }
        
        return all_values;
    }
    
private:
    void rotate_windows() {
        while (running_) {
            std::this_thread::sleep_for(window_duration_);
            
            size_t next = (current_window_ + 1) % num_windows_;
            {
                std::lock_guard lock(windows_[next]->mutex);
                windows_[next]->values.clear();
                windows_[next]->start = std::chrono::steady_clock::now();
            }
            current_window_.store(next, std::memory_order_relaxed);
        }
    }
};
```
**Why necessary**: Track distribution of values, calculate percentiles, understand data spread.
**Usage**: Latency measurements, response size distribution, performance analysis.

### `summary.hpp`
```cpp
class Summary : public Metric {
    class QuantileEstimator {
        // T-Digest or HDRHistogram implementation for accurate quantiles
        struct Centroid {
            double mean;
            uint64_t count;
        };
        
        std::vector<Centroid> centroids_;
        mutable std::mutex mutex_;
        double compression_;
        
    public:
        explicit QuantileEstimator(double compression = 100.0)
            : compression_(compression) {}
        
        void add(double value, uint64_t weight = 1) {
            std::lock_guard lock(mutex_);
            
            // Add to centroids using t-digest algorithm
            centroids_.push_back({value, weight});
            
            // Compress if needed
            if (centroids_.size() > compression_ * 10) {
                compress();
            }
        }
        
        double quantile(double q) const {
            std::lock_guard lock(mutex_);
            
            if (centroids_.empty()) return 0.0;
            
            // Calculate quantile from centroids
            // T-Digest quantile calculation
            return calculate_quantile(q);
        }
        
    private:
        void compress() {
            // Merge adjacent centroids
        }
        
        double calculate_quantile(double q) const {
            // T-Digest quantile algorithm
            return 0.0;  // Placeholder
        }
    };
    
    struct Objectives {
        std::vector<std::pair<double, double>> quantile_objectives;  // quantile -> error
        std::chrono::seconds max_age{600};  // 10 minutes default
        uint64_t age_buckets{5};
    };
    
    QuantileEstimator estimator_;
    std::atomic<uint64_t> count_{0};
    std::atomic<double> sum_{0.0};
    Objectives objectives_;
    
public:
    Summary(const std::string& name,
           const Objectives& objectives = {{0.5, 0.05}, {0.9, 0.01}, {0.99, 0.001}})
        : objectives_(objectives) {
        metadata_.name = name;
        metadata_.type = Type::Summary;
    }
    
    void observe(double value) {
        estimator_.add(value);
        count_.fetch_add(1, std::memory_order_relaxed);
        sum_.fetch_add(value, std::memory_order_relaxed);
        update_timestamp();
    }
    
    double quantile(double q) const {
        return estimator_.quantile(q);
    }
    
    uint64_t count() const {
        return count_.load(std::memory_order_relaxed);
    }
    
    double sum() const {
        return sum_.load(std::memory_order_relaxed);
    }
    
    double mean() const {
        uint64_t n = count();
        return n > 0 ? sum() / n : 0.0;
    }
    
    Type type() const override { return Type::Summary; }
    
    MetricSnapshot snapshot() const override {
        MetricSnapshot snap;
        snap.metadata = metadata_;
        snap.timestamp = std::chrono::system_clock::now();
        
        snap.summary_data = std::make_unique<SummaryData>();
        snap.summary_data->count = count();
        snap.summary_data->sum = sum();
        
        // Add configured quantiles
        for (const auto& [q, _] : objectives_.quantile_objectives) {
            snap.summary_data->quantiles.push_back({q, quantile(q)});
        }
        
        return snap;
    }
};
```
**Why necessary**: Accurate quantile estimation over sliding time windows.
**Usage**: SLA monitoring, percentile tracking, performance objectives.

### `timer.hpp`
```cpp
class Timer : public Metric {
    Histogram histogram_;
    std::atomic<uint64_t> active_timers_{0};
    
public:
    class Scope {
        Timer* timer_;
        std::chrono::high_resolution_clock::time_point start_;
        bool completed_{false};
        
    public:
        explicit Scope(Timer* timer)
            : timer_(timer)
            , start_(std::chrono::high_resolution_clock::now()) {
            timer_->active_timers_.fetch_add(1, std::memory_order_relaxed);
        }
        
        ~Scope() {
            if (!completed_) {
                stop();
            }
        }
        
        void stop() {
            if (completed_) return;
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double>(end - start_);
            timer_->record(duration.count());
            timer_->active_timers_.fetch_sub(1, std::memory_order_relaxed);
            completed_ = true;
        }
        
        // Prevent copying
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        
        // Allow moving
        Scope(Scope&& other)
            : timer_(other.timer_)
            , start_(other.start_)
            , completed_(other.completed_) {
            other.completed_ = true;
        }
    };
    
    Timer(const std::string& name, 
          const std::vector<double>& buckets = Histogram::default_buckets())
        : histogram_(name + ".duration", buckets) {
        metadata_.name = name;
        metadata_.type = Type::Timer;
    }
    
    // Manual timing
    void record(double seconds) {
        histogram_.observe(seconds);
        update_timestamp();
    }
    
    void record(std::chrono::nanoseconds duration) {
        record(std::chrono::duration<double>(duration).count());
    }
    
    // RAII timing
    [[nodiscard]] Scope start_timer() {
        return Scope(this);
    }
    
    // Functional timing
    template<typename F>
    auto time(F&& func) -> decltype(func()) {
        auto scope = start_timer();
        return func();
    }
    
    template<typename F>
    auto time_async(F&& func) -> std::future<decltype(func())> {
        return std::async(std::launch::async, [this, func = std::forward<F>(func)]() {
            auto scope = start_timer();
            return func();
        });
    }
    
    // Statistics
    uint64_t count() const { return histogram_.count(); }
    double mean() const { return histogram_.mean(); }
    double median() const { return histogram_.median(); }
    double p95() const { return histogram_.p95(); }
    double p99() const { return histogram_.p99(); }
    uint64_t active() const { return active_timers_.load(std::memory_order_relaxed); }
    
    Type type() const override { return Type::Timer; }
    
    MetricSnapshot snapshot() const override {
        auto snap = histogram_.snapshot();
        snap.metadata = metadata_;
        return snap;
    }
};

// Rate meter for measuring events per second
class Meter : public Metric {
    struct Rate {
        std::atomic<uint64_t> count{0};
        std::chrono::steady_clock::time_point start;
    };
    
    static constexpr size_t NUM_WINDOWS = 60;  // 1-minute of per-second data
    std::array<Rate, NUM_WINDOWS> windows_;
    std::atomic<size_t> current_window_{0};
    std::thread rotator_;
    std::atomic<bool> running_{true};
    
public:
    Meter(const std::string& name) {
        metadata_.name = name;
        metadata_.type = Type::Meter;
        
        auto now = std::chrono::steady_clock::now();
        for (auto& window : windows_) {
            window.start = now;
        }
        
        rotator_ = std::thread([this] { rotate_windows(); });
    }
    
    ~Meter() {
        running_ = false;
        if (rotator_.joinable()) {
            rotator_.join();
        }
    }
    
    void mark(uint64_t n = 1) {
        size_t idx = current_window_.load(std::memory_order_relaxed);
        windows_[idx].count.fetch_add(n, std::memory_order_relaxed);
        update_timestamp();
    }
    
    double rate_per_second() const {
        return calculate_rate(std::chrono::seconds(1));
    }
    
    double rate_per_minute() const {
        return calculate_rate(std::chrono::seconds(60));
    }
    
    double one_minute_rate() const {
        return exponentially_weighted_rate(std::chrono::seconds(60));
    }
    
    double five_minute_rate() const {
        return exponentially_weighted_rate(std::chrono::seconds(300));
    }
    
    double fifteen_minute_rate() const {
        return exponentially_weighted_rate(std::chrono::seconds(900));
    }
    
    Type type() const override { return Type::Meter; }
    
private:
    void rotate_windows() {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            size_t next = (current_window_ + 1) % NUM_WINDOWS;
            windows_[next].count.store(0, std::memory_order_relaxed);
            windows_[next].start = std::chrono::steady_clock::now();
            current_window_.store(next, std::memory_order_relaxed);
        }
    }
    
    double calculate_rate(std::chrono::seconds window) const {
        auto now = std::chrono::steady_clock::now();
        auto cutoff = now - window;
        
        uint64_t total = 0;
        for (const auto& w : windows_) {
            if (w.start >= cutoff) {
                total += w.count.load(std::memory_order_relaxed);
            }
        }
        
        return static_cast<double>(total) / window.count();
    }
    
    double exponentially_weighted_rate(std::chrono::seconds window) const {
        // EWMA calculation
        const double alpha = 1.0 - std::exp(-1.0 / window.count());
        double rate = 0.0;
        
        for (const auto& w : windows_) {
            uint64_t count = w.count.load(std::memory_order_relaxed);
            rate = alpha * count + (1.0 - alpha) * rate;
        }
        
        return rate;
    }
};
```
**Why necessary**: Measure operation duration, track timing distributions, performance monitoring.
**Usage**: Function timing, request latency, operation performance.

### `metric_registry.hpp`
```cpp
class MetricRegistry : public Singleton<MetricRegistry> {
public:
    using MetricPtr = std::shared_ptr<Metric>;
    using CollectorFunc = std::function<std::vector<MetricSnapshot>()>;
    
private:
    std::unordered_map<std::string, MetricPtr> metrics_;
    std::vector<std::pair<std::string, CollectorFunc>> collectors_;
    mutable std::shared_mutex mutex_;
    
    struct Config {
        bool enable_cardinality_limits = true;
        size_t max_metrics_per_type = 10000;
        size_t max_label_cardinality = 1000;
        bool enable_metric_expiry = true;
        std::chrono::seconds metric_ttl{3600};  // 1 hour
    } config_;
    
public:
    // Metric creation and retrieval
    template<typename MetricType, typename... Args>
    std::shared_ptr<MetricType> get_or_create(const std::string& name, Args&&... args) {
        {
            std::shared_lock lock(mutex_);
            auto it = metrics_.find(name);
            if (it != metrics_.end()) {
                return std::dynamic_pointer_cast<MetricType>(it->second);
            }
        }
        
        std::unique_lock lock(mutex_);
        
        // Double-check after acquiring write lock
        auto it = metrics_.find(name);
        if (it != metrics_.end()) {
            return std::dynamic_pointer_cast<MetricType>(it->second);
        }
        
        // Check cardinality limits
        if (config_.enable_cardinality_limits) {
            size_t count = count_metrics_of_type<MetricType>();
            if (count >= config_.max_metrics_per_type) {
                throw std::runtime_error("Metric cardinality limit exceeded");
            }
        }
        
        auto metric = std::make_shared<MetricType>(name, std::forward<Args>(args)...);
        metrics_[name] = metric;
        return metric;
    }
    
    // Counter convenience
    MetricRef<Counter> counter(const std::string& name) {
        return MetricRef<Counter>(get_or_create<Counter>(name));
    }
    
    // Gauge convenience
    MetricRef<Gauge> gauge(const std::string& name) {
        return MetricRef<Gauge>(get_or_create<Gauge>(name));
    }
    
    // Histogram convenience
    MetricRef<Histogram> histogram(const std::string& name,
                                   const std::vector<double>& buckets = Histogram::default_buckets()) {
        return MetricRef<Histogram>(get_or_create<Histogram>(name, buckets));
    }
    
    // Timer convenience
    MetricRef<Timer> timer(const std::string& name) {
        return MetricRef<Timer>(get_or_create<Timer>(name));
    }
    
    // Summary convenience
    MetricRef<Summary> summary(const std::string& name) {
        return MetricRef<Summary>(get_or_create<Summary>(name));
    }
    
    // Meter convenience
    MetricRef<Meter> meter(const std::string& name) {
        return MetricRef<Meter>(get_or_create<Meter>(name));
    }
    
    // Custom collector registration
    void register_collector(const std::string& name, CollectorFunc collector) {
        std::unique_lock lock(mutex_);
        collectors_.push_back({name, collector});
    }
    
    // Metric removal
    void remove(const std::string& name) {
        std::unique_lock lock(mutex_);
        metrics_.erase(name);
    }
    
    void clear() {
        std::unique_lock lock(mutex_);
        metrics_.clear();
    }
    
    // Metric collection
    std::vector<MetricSnapshot> collect_all() const {
        std::shared_lock lock(mutex_);
        std::vector<MetricSnapshot> snapshots;
        
        // Collect from registered metrics
        for (const auto& [name, metric] : metrics_) {
            snapshots.push_back(metric->snapshot());
        }
        
        // Collect from custom collectors
        for (const auto& [name, collector] : collectors_) {
            auto custom_snapshots = collector();
            snapshots.insert(snapshots.end(), 
                           custom_snapshots.begin(), 
                           custom_snapshots.end());
        }
        
        return snapshots;
    }
    
    // Export to various formats
    std::string export_prometheus() const {
        PrometheusExporter exporter;
        return exporter.export_metrics(collect_all());
    }
    
    std::string export_json() const {
        JsonExporter exporter;
        return exporter.export_metrics(collect_all());
    }
    
    // Metric queries
    std::vector<std::string> list_metrics() const {
        std::shared_lock lock(mutex_);
        std::vector<std::string> names;
        names.reserve(metrics_.size());
        
        for (const auto& [name, _] : metrics_) {
            names.push_back(name);
        }
        
        return names;
    }
    
    size_t metric_count() const {
        std::shared_lock lock(mutex_);
        return metrics_.size();
    }
    
    // Garbage collection
    void cleanup_expired_metrics() {
        if (!config_.enable_metric_expiry) return;
        
        std::unique_lock lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        
        for (auto it = metrics_.begin(); it != metrics_.end(); ) {
            auto last_update = std::chrono::nanoseconds(
                it->second->last_updated_.load()
            );
            auto last_update_time = std::chrono::steady_clock::time_point(last_update);
            
            if (now - last_update_time > config_.metric_ttl) {
                it = metrics_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
private:
    template<typename MetricType>
    size_t count_metrics_of_type() const {
        size_t count = 0;
        for (const auto& [_, metric] : metrics_) {
            if (dynamic_cast<MetricType*>(metric.get())) {
                ++count;
            }
        }
        return count;
    }
};

// Global registry access
inline MetricRegistry& metrics() {
    return MetricRegistry::instance();
}
```
**Why necessary**: Centralized metric management, lifecycle control, export coordination.
**Usage**: Application-wide metric access, bulk export, metric discovery.

### `prometheus_exporter.hpp`
```cpp
class PrometheusExporter : public Exporter {
public:
    std::string export_metrics(const std::vector<MetricSnapshot>& snapshots) override {
        std::stringstream ss;
        
        // Group metrics by name for Prometheus family grouping
        std::map<std::string, std::vector<MetricSnapshot>> families;
        for (const auto& snapshot : snapshots) {
            std::string family_name = extract_family_name(snapshot.metadata.name);
            families[family_name].push_back(snapshot);
        }
        
        for (const auto& [family_name, family_snapshots] : families) {
            write_family(ss, family_name, family_snapshots);
        }
        
        return ss.str();
    }
    
private:
    void write_family(std::stringstream& ss,
                     const std::string& family_name,
                     const std::vector<MetricSnapshot>& snapshots) {
        if (snapshots.empty()) return;
        
        const auto& first = snapshots[0];
        
        // Write HELP line
        ss << "# HELP " << family_name << " " 
           << first.metadata.description << "\n";
        
        // Write TYPE line
        ss << "# TYPE " << family_name << " " 
           << prometheus_type(first.metadata.type) << "\n";
        
        // Write metrics
        for (const auto& snapshot : snapshots) {
            write_metric(ss, snapshot);
        }
        
        ss << "\n";
    }
    
    void write_metric(std::stringstream& ss, const MetricSnapshot& snapshot) {
        std::string name = snapshot.metadata.name;
        std::string labels = format_labels(snapshot.metadata.labels);
        
        switch (snapshot.metadata.type) {
            case Metric::Type::Counter:
            case Metric::Type::Gauge:
                ss << name << labels << " " << snapshot.value << "\n";
                break;
                
            case Metric::Type::Histogram:
                write_histogram(ss, name, labels, *snapshot.histogram_data);
                break;
                
            case Metric::Type::Summary:
                write_summary(ss, name, labels, *snapshot.summary_data);
                break;
        }
    }
    
    void write_histogram(std::stringstream& ss,
                         const std::string& name,
                         const std::string& labels,
                         const HistogramData& data) {
        // Write buckets
        uint64_t cumulative = 0;
        for (const auto& [boundary, count] : data.buckets) {
            cumulative += count;
            ss << name << "_bucket{" << trim_labels(labels) 
               << "le=\"" << boundary << "\"} " << cumulative << "\n";
        }
        
        // Write +Inf bucket
        cumulative += data.inf_count;
        ss << name << "_bucket{" << trim_labels(labels) 
           << "le=\"+Inf\"} " << cumulative << "\n";
        
        // Write sum and count
        ss << name << "_sum" << labels << " " << data.sum << "\n";
        ss << name << "_count" << labels << " " << data.count << "\n";
    }
    
    void write_summary(std::stringstream& ss,
                      const std::string& name,
                      const std::string& labels,
                      const SummaryData& data) {
        // Write quantiles
        for (const auto& [quantile, value] : data.quantiles) {
            ss << name << "{" << trim_labels(labels) 
               << "quantile=\"" << quantile << "\"} " << value << "\n";
        }
        
        // Write sum and count
        ss << name << "_sum" << labels << " " << data.sum << "\n";
        ss << name << "_count" << labels << " " << data.count << "\n";
    }
    
    std::string prometheus_type(Metric::Type type) const {
        switch (type) {
            case Metric::Type::Counter: return "counter";
            case Metric::Type::Gauge: return "gauge";
            case Metric::Type::Histogram: return "histogram";
            case Metric::Type::Summary: return "summary";
            default: return "untyped";
        }
    }
    
    std::string format_labels(const std::map<std::string, std::string>& labels) const {
        if (labels.empty()) return "";
        
        std::stringstream ss;
        ss << "{";
        bool first = true;
        
        for (const auto& [key, value] : labels) {
            if (!first) ss << ",";
            ss << key << "=\"" << escape_label_value(value) << "\"";
            first = false;
        }
        
        ss << "}";
        return ss.str();
    }
    
    std::string escape_label_value(const std::string& value) const {
        std::string escaped;
        for (char c : value) {
            if (c == '"') escaped += "\\\"";
            else if (c == '\\') escaped += "\\\\";
            else if (c == '\n') escaped += "\\n";
            else escaped += c;
        }
        return escaped;
    }
};
```
**Why necessary**: Export metrics in Prometheus format for monitoring systems.
**Usage**: Prometheus scraping endpoint, metrics exposition.

### `metric_macros.hpp`
```cpp
// Convenience macros for metric operations

// Counter macros
#define METRIC_INC(name) \
    MetricRegistry::instance().counter(name)->increment()

#define METRIC_INC_BY(name, delta) \
    MetricRegistry::instance().counter(name)->increment(delta)

#define METRIC_COUNTER(name) \
    MetricRegistry::instance().counter(name)

// Gauge macros  
#define METRIC_SET(name, value) \
    MetricRegistry::instance().gauge(name)->set(value)

#define METRIC_GAUGE_INC(name, delta) \
    MetricRegistry::instance().gauge(name)->increment(delta)

#define METRIC_GAUGE_DEC(name, delta) \
    MetricRegistry::instance().gauge(name)->decrement(delta)

#define METRIC_GAUGE(name) \
    MetricRegistry::instance().gauge(name)

// Histogram macros
#define METRIC_OBSERVE(name, value) \
    MetricRegistry::instance().histogram(name)->observe(value)

#define METRIC_HISTOGRAM(name) \
    MetricRegistry::instance().histogram(name)

// Timer macros
#define METRIC_TIME(name) \
    auto CONCAT(_timer_, __LINE__) = \
        MetricRegistry::instance().timer(name)->start_timer()

#define METRIC_TIME_BLOCK(name) \
    for (auto CONCAT(_timer_, __LINE__) = \
         MetricRegistry::instance().timer(name)->start_timer(); \
         CONCAT(_timer_, __LINE__).is_valid(); \
         CONCAT(_timer_, __LINE__).stop())

#define METRIC_TIMER(name) \
    MetricRegistry::instance().timer(name)

// Meter macros
#define METRIC_MARK(name) \
    MetricRegistry::instance().meter(name)->mark()

#define METRIC_MARK_N(name, n) \
    MetricRegistry::instance().meter(name)->mark(n)

#define METRIC_METER(name) \
    MetricRegistry::instance().meter(name)

// Labeled metrics
#define METRIC_INC_LABELED(name, ...) \
    MetricRegistry::instance().labeled_counter(name)->with_labels(__VA_ARGS__).increment()

#define METRIC_SET_LABELED(name, value, ...) \
    MetricRegistry::instance().labeled_gauge(name)->with_labels(__VA_ARGS__).set(value)

#define METRIC_OBSERVE_LABELED(name, value, ...) \
    MetricRegistry::instance().labeled_histogram(name)->with_labels(__VA_ARGS__).observe(value)

// Functional gauge registration
#define METRIC_GAUGE_FUNC(name, func) \
    MetricRegistry::instance().register_collector(name, [](){ \
        MetricSnapshot snap; \
        snap.metadata.name = name; \
        snap.metadata.type = Metric::Type::Gauge; \
        snap.value = func(); \
        return std::vector<MetricSnapshot>{snap}; \
    })

// Compile-time metric disabling
#ifdef DISABLE_METRICS
    #undef METRIC_INC
    #define METRIC_INC(name) ((void)0)
    #undef METRIC_TIME
    #define METRIC_TIME(name) ((void)0)
    // ... redefine all macros as no-ops
#endif

// Helper macro
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define CONCAT_IMPL(a, b) a##b
```
**Why necessary**: Convenient metric operations, minimal boilerplate, conditional compilation.
**Usage**: Throughout application code for metric instrumentation.

## Metrics Patterns

### Request Handling Metrics
```cpp
void handle_request(const Request& req) {
    METRIC_INC("http.requests.total");
    METRIC_GAUGE("http.requests.active").increment();
    
    METRIC_TIME("http.request.duration") {
        try {
            auto response = process_request(req);
            
            METRIC_INC_LABELED("http.responses.total",
                             req.method(), 
                             std::to_string(response.status_code()));
            
            METRIC_OBSERVE("http.response.size", response.size());
            
        } catch (const std::exception& e) {
            METRIC_INC("http.errors.total");
            throw;
        }
    }
    
    METRIC_GAUGE("http.requests.active").decrement();
}
```

### Database Metrics
```cpp
class DatabaseConnection {
    MetricRef<Counter> queries_total_;
    MetricRef<Counter> errors_total_;
    MetricRef<Histogram> query_duration_;
    MetricRef<Gauge> connections_active_;
    
public:
    DatabaseConnection() 
        : queries_total_(metrics().counter("db.queries.total"))
        , errors_total_(metrics().counter("db.errors.total"))
        , query_duration_(metrics().histogram("db.query.duration",
                                              {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0}))
        , connections_active_(metrics().gauge("db.connections.active")) {
        connections_active_->increment();
    }
    
    ~DatabaseConnection() {
        connections_active_->decrement();
    }
    
    ResultSet execute_query(const std::string& query) {
        queries_total_->increment();
        
        auto timer = query_duration_->start_timer();
        
        try {
            return database_.execute(query);
        } catch (const std::exception& e) {
            errors_total_->increment();
            throw;
        }
    }
};
```

### Business Metrics
```cpp
class OrderService {
public:
    void place_order(const Order& order) {
        // Business metrics
        METRIC_INC("orders.placed.total");
        METRIC_OBSERVE("orders.value", order.total_amount());
        METRIC_INC_LABELED("orders.by_category", order.category());
        
        // Technical metrics
        METRIC_TIME("orders.processing.duration") {
            process_order(order);
        }
        
        // Custom business logic metrics
        if (order.is_premium_customer()) {
            METRIC_INC("orders.premium.total");
        }
        
        // Revenue tracking
        METRIC_GAUGE("revenue.daily").increment(order.total_amount());
    }
};
```

### System Metrics Collection
```cpp
void collect_system_metrics() {
    // CPU usage
    METRIC_GAUGE_FUNC("system.cpu.usage", []() {
        return get_cpu_usage();
    });
    
    // Memory usage
    METRIC_GAUGE_FUNC("system.memory.used", []() {
        return get_memory_used();
    });
    
    METRIC_GAUGE_FUNC("system.memory.available", []() {
        return get_memory_available();
    });
    
    // Disk usage
    METRIC_GAUGE_FUNC("system.disk.used", []() {
        return get_disk_used();
    });
    
    // Thread count
    METRIC_GAUGE_FUNC("process.threads", []() {
        return std::thread::hardware_concurrency();
    });
    
    // File descriptors
    METRIC_GAUGE_FUNC("process.open_files", []() {
        return count_open_files();
    });
}
```

## Performance Considerations

- **Lock-free updates**: ~10-20ns for counter increment
- **Atomic operations**: Use relaxed memory ordering where safe
- **Memory pooling**: Pre-allocate histogram buckets
- **Batch exports**: Amortize serialization overhead
- **Sampling**: Use reservoir sampling for high-volume metrics
- **Cardinality limits**: Prevent metric explosion

## Testing Strategy

- **Accuracy testing**: Verify statistical calculations
- **Thread safety**: Concurrent metric updates
- **Performance benchmarks**: Overhead measurement
- **Export formats**: Validate all export formats
- **Memory usage**: Track metric memory footprint
- **Cardinality testing**: Label explosion prevention

## Usage Guidelines

1. **Naming conventions**: Use dot notation (e.g., `http.requests.total`)
2. **Label usage**: Keep cardinality low, avoid user IDs
3. **Unit specification**: Include units in metric names or metadata
4. **Histogram buckets**: Choose appropriate bucket boundaries
5. **Export frequency**: Balance freshness vs. overhead

## Anti-patterns to Avoid

- High cardinality labels (user IDs, session IDs)
- Metric name changes (breaks dashboards)
- Unbounded metrics growth
- Synchronous export in critical path
- Missing error metrics

## Dependencies
- `base/` - For Singleton pattern
- `concurrency/` - For thread safety
- Standard library (C++20)
- Optional: HTTP client for remote export

## Future Enhancements
- Metric aggregation across instances
- Anomaly detection
- Adaptive sampling
- Metric correlation analysis
- Cost-based metric retention