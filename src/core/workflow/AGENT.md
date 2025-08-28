# Core Workflow - AGENT.md

## Purpose
The `workflow/` layer provides comprehensive task orchestration including command patterns, undo/redo functionality, state machines, pipeline processing, task scheduling, and transactional operations. It enables complex multi-step operations with rollback capabilities, workflow automation, and sophisticated state management.

## Architecture Philosophy
- **Command pattern**: Encapsulate operations as objects
- **Transactional semantics**: All-or-nothing execution with rollback
- **State machine driven**: Explicit state transitions and guards
- **Pipeline composition**: Chain operations with error handling
- **Scheduling flexibility**: Time-based, dependency-based, and priority-based execution

## Files Overview

### Command Pattern
```cpp
command.hpp          // Command interface and base class
undo_stack.hpp       // Undo/redo stack management
macro_command.hpp    // Composite commands
command_processor.hpp // Command execution engine
command_history.hpp  // Command history tracking
```

### State Machines
```cpp
state_machine.hpp    // Generic state machine framework
state.hpp           // State interface and base
transition.hpp      // State transitions with guards
hsm.hpp             // Hierarchical state machines
state_chart.hpp     // UML statechart implementation
```

### Pipeline Processing
```cpp
pipeline.hpp        // Pipeline framework
stage.hpp           // Pipeline stage interface
pipeline_builder.hpp // Fluent pipeline construction
parallel_pipeline.hpp // Parallel stage execution
error_handler.hpp   // Pipeline error handling
```

### Task Scheduling
```cpp
scheduler.hpp       // Task scheduler interface
cron_scheduler.hpp  // Cron-like scheduling
priority_scheduler.hpp // Priority-based scheduling
dependency_scheduler.hpp // DAG task scheduling
task_graph.hpp      // Task dependency graph
```

### Workflow Management
```cpp
workflow.hpp        // Workflow definition
workflow_engine.hpp // Workflow execution engine
activity.hpp        // Workflow activities
decision.hpp        // Decision points
fork_join.hpp       // Parallel execution branches
```

### Transactions
```cpp
transaction.hpp     // Transaction interface
transaction_manager.hpp // Transaction coordination
rollback.hpp        // Rollback mechanisms
savepoint.hpp       // Transaction savepoints
two_phase_commit.hpp // Distributed transactions
```

### Process Control
```cpp
process.hpp         // Long-running process
process_manager.hpp // Process lifecycle management
checkpoint.hpp      // Process checkpointing
recovery.hpp        // Failure recovery
saga.hpp           // Saga pattern implementation
```

### Utilities
```cpp
context.hpp         // Execution context
workflow_macros.hpp // Convenience macros
validation.hpp      // Workflow validation
visualization.hpp   // Workflow visualization
metrics.hpp         // Workflow metrics
```

## Detailed Component Specifications

### `command.hpp`
```cpp
class Command {
public:
    using CommandId = uint64_t;
    
    enum class Status {
        Created,
        Executing,
        Completed,
        Failed,
        Undone,
        Redone
    };
    
    virtual ~Command() = default;
    
    // Core command interface
    virtual void execute() = 0;
    virtual void undo() = 0;
    virtual void redo() { execute(); }
    
    // Command properties
    virtual bool can_execute() const { return true; }
    virtual bool can_undo() const { return true; }
    virtual bool can_redo() const { return can_execute(); }
    virtual bool is_reversible() const { return can_undo(); }
    
    // Merging support for optimization
    virtual bool can_merge_with(const Command& other) const { return false; }
    virtual std::unique_ptr<Command> merge_with(const Command& other) const { return nullptr; }
    
    // Metadata
    virtual std::string name() const = 0;
    virtual std::string description() const { return name(); }
    virtual std::size_t memory_size() const { return sizeof(*this); }
    
    CommandId id() const { return id_; }
    Status status() const { return status_; }
    
    // Execution context
    void set_context(std::shared_ptr<Context> ctx) { context_ = ctx; }
    std::shared_ptr<Context> context() const { return context_; }
    
protected:
    CommandId id_ = generate_id();
    Status status_ = Status::Created;
    std::shared_ptr<Context> context_;
    
    void set_status(Status s) { status_ = s; }
    
private:
    static CommandId generate_id() {
        static std::atomic<CommandId> next_id{1};
        return next_id.fetch_add(1);
    }
};

// Template command for lambda-based commands
template<typename ExecuteFunc, typename UndoFunc>
class LambdaCommand : public Command {
    ExecuteFunc execute_func_;
    UndoFunc undo_func_;
    std::string name_;
    
public:
    LambdaCommand(ExecuteFunc exec, UndoFunc undo, const std::string& name)
        : execute_func_(std::move(exec))
        , undo_func_(std::move(undo))
        , name_(name) {}
    
    void execute() override {
        set_status(Status::Executing);
        execute_func_();
        set_status(Status::Completed);
    }
    
    void undo() override {
        undo_func_();
        set_status(Status::Undone);
    }
    
    std::string name() const override { return name_; }
};

// Factory for creating commands
template<typename ExecuteFunc, typename UndoFunc>
auto make_command(ExecuteFunc&& exec, UndoFunc&& undo, const std::string& name) {
    return std::make_unique<LambdaCommand<ExecuteFunc, UndoFunc>>(
        std::forward<ExecuteFunc>(exec),
        std::forward<UndoFunc>(undo),
        name
    );
}

// Composite command
class MacroCommand : public Command {
    std::vector<std::unique_ptr<Command>> commands_;
    std::string name_;
    
public:
    explicit MacroCommand(const std::string& name) : name_(name) {}
    
    void add_command(std::unique_ptr<Command> cmd) {
        commands_.push_back(std::move(cmd));
    }
    
    void execute() override {
        set_status(Status::Executing);
        for (auto& cmd : commands_) {
            cmd->execute();
        }
        set_status(Status::Completed);
    }
    
    void undo() override {
        // Undo in reverse order
        for (auto it = commands_.rbegin(); it != commands_.rend(); ++it) {
            (*it)->undo();
        }
        set_status(Status::Undone);
    }
    
    std::string name() const override { return name_; }
    
    std::size_t command_count() const { return commands_.size(); }
};

// Async command
class AsyncCommand : public Command {
public:
    virtual std::future<void> execute_async() = 0;
    virtual std::future<void> undo_async() = 0;
    
    void execute() override {
        execute_async().wait();
    }
    
    void undo() override {
        undo_async().wait();
    }
};
```
**Why necessary**: Encapsulate operations, enable undo/redo, command queuing, macro operations.
**Usage**: User actions, reversible operations, command history, scripting.

### `undo_stack.hpp`
```cpp
class UndoStack {
public:
    using CommandPtr = std::unique_ptr<Command>;
    
    struct Options {
        std::size_t max_size = 100;
        std::size_t max_memory = 10 * 1024 * 1024;  // 10MB
        bool merge_commands = true;
        std::chrono::milliseconds merge_interval{500};
    };
    
private:
    std::vector<CommandPtr> undo_stack_;
    std::vector<CommandPtr> redo_stack_;
    std::size_t current_index_ = 0;
    Options options_;
    
    std::size_t current_memory_usage_ = 0;
    std::chrono::steady_clock::time_point last_command_time_;
    
    // Observers
    std::vector<std::function<void()>> change_listeners_;
    
    // State flags
    std::size_t clean_index_ = 0;  // Index when set_clean() was called
    bool macro_running_ = false;
    std::unique_ptr<MacroCommand> current_macro_;
    
public:
    explicit UndoStack(const Options& opts = {}) : options_(opts) {}
    
    // Command execution
    void push(CommandPtr cmd) {
        if (!cmd || !cmd->can_execute()) {
            return;
        }
        
        // Try to merge with previous command
        if (options_.merge_commands && !undo_stack_.empty()) {
            auto now = std::chrono::steady_clock::now();
            auto time_since_last = now - last_command_time_;
            
            if (time_since_last < options_.merge_interval) {
                auto& last_cmd = undo_stack_.back();
                if (last_cmd->can_merge_with(*cmd)) {
                    auto merged = last_cmd->merge_with(*cmd);
                    if (merged) {
                        undo_stack_.back() = std::move(merged);
                        notify_change();
                        return;
                    }
                }
            }
        }
        
        // Add to macro if running
        if (macro_running_ && current_macro_) {
            current_macro_->add_command(std::move(cmd));
            return;
        }
        
        // Execute command
        cmd->execute();
        
        // Clear redo stack
        redo_stack_.clear();
        
        // Add to undo stack
        current_memory_usage_ += cmd->memory_size();
        undo_stack_.push_back(std::move(cmd));
        
        // Enforce limits
        enforce_limits();
        
        last_command_time_ = std::chrono::steady_clock::now();
        notify_change();
    }
    
    // Undo/Redo operations
    void undo() {
        if (!can_undo()) return;
        
        auto cmd = std::move(undo_stack_.back());
        undo_stack_.pop_back();
        current_memory_usage_ -= cmd->memory_size();
        
        cmd->undo();
        
        redo_stack_.push_back(std::move(cmd));
        notify_change();
    }
    
    void redo() {
        if (!can_redo()) return;
        
        auto cmd = std::move(redo_stack_.back());
        redo_stack_.pop_back();
        
        cmd->redo();
        
        current_memory_usage_ += cmd->memory_size();
        undo_stack_.push_back(std::move(cmd));
        notify_change();
    }
    
    // Macro commands
    void begin_macro(const std::string& name) {
        if (macro_running_) {
            throw std::runtime_error("Macro already running");
        }
        
        macro_running_ = true;
        current_macro_ = std::make_unique<MacroCommand>(name);
    }
    
    void end_macro() {
        if (!macro_running_) {
            throw std::runtime_error("No macro running");
        }
        
        macro_running_ = false;
        
        if (current_macro_ && current_macro_->command_count() > 0) {
            push(std::move(current_macro_));
        }
        
        current_macro_.reset();
    }
    
    void cancel_macro() {
        macro_running_ = false;
        current_macro_.reset();
    }
    
    // Stack state
    bool can_undo() const { return !undo_stack_.empty(); }
    bool can_redo() const { return !redo_stack_.empty(); }
    
    std::size_t undo_count() const { return undo_stack_.size(); }
    std::size_t redo_count() const { return redo_stack_.size(); }
    
    bool is_clean() const {
        return undo_stack_.size() == clean_index_;
    }
    
    void set_clean() {
        clean_index_ = undo_stack_.size();
    }
    
    // Stack information
    std::string undo_text() const {
        if (!can_undo()) return "";
        return "Undo " + undo_stack_.back()->name();
    }
    
    std::string redo_text() const {
        if (!can_redo()) return "";
        return "Redo " + redo_stack_.back()->name();
    }
    
    std::vector<std::string> undo_history() const {
        std::vector<std::string> history;
        for (const auto& cmd : undo_stack_) {
            history.push_back(cmd->description());
        }
        return history;
    }
    
    // Stack management
    void clear() {
        undo_stack_.clear();
        redo_stack_.clear();
        current_memory_usage_ = 0;
        clean_index_ = 0;
        notify_change();
    }
    
    void set_undo_limit(std::size_t limit) {
        options_.max_size = limit;
        enforce_limits();
    }
    
    // Observers
    void add_change_listener(std::function<void()> listener) {
        change_listeners_.push_back(std::move(listener));
    }
    
private:
    void enforce_limits() {
        // Enforce size limit
        while (undo_stack_.size() > options_.max_size && !undo_stack_.empty()) {
            current_memory_usage_ -= undo_stack_.front()->memory_size();
            undo_stack_.erase(undo_stack_.begin());
            
            if (clean_index_ > 0) {
                --clean_index_;
            }
        }
        
        // Enforce memory limit
        while (current_memory_usage_ > options_.max_memory && !undo_stack_.empty()) {
            current_memory_usage_ -= undo_stack_.front()->memory_size();
            undo_stack_.erase(undo_stack_.begin());
            
            if (clean_index_ > 0) {
                --clean_index_;
            }
        }
    }
    
    void notify_change() {
        for (const auto& listener : change_listeners_) {
            listener();
        }
    }
};

// RAII macro command
class ScopedMacro {
    UndoStack& stack_;
    bool committed_ = false;
    
public:
    ScopedMacro(UndoStack& stack, const std::string& name)
        : stack_(stack) {
        stack_.begin_macro(name);
    }
    
    ~ScopedMacro() {
        if (!committed_) {
            stack_.cancel_macro();
        }
    }
    
    void commit() {
        stack_.end_macro();
        committed_ = true;
    }
};
```
**Why necessary**: Undo/redo functionality, command history, macro recording, memory management.
**Usage**: Document editors, CAD applications, game editors, any app with reversible actions.

### `state_machine.hpp`
```cpp
template<typename StateEnum, typename EventEnum>
class StateMachine {
public:
    using State = StateEnum;
    using Event = EventEnum;
    using Guard = std::function<bool()>;
    using Action = std::function<void()>;
    using StateChangeCallback = std::function<void(State, State)>;
    
    struct Transition {
        State from;
        Event event;
        State to;
        Guard guard;
        Action action;
    };
    
private:
    State current_state_;
    State initial_state_;
    std::unordered_map<State, std::vector<Transition>> transitions_;
    std::unordered_map<State, Action> entry_actions_;
    std::unordered_map<State, Action> exit_actions_;
    std::unordered_map<State, Action> internal_actions_;
    
    std::vector<StateChangeCallback> state_change_callbacks_;
    std::queue<Event> event_queue_;
    bool processing_ = false;
    
    // State history for hierarchical machines
    std::unordered_map<State, State> history_;
    
public:
    StateMachine(State initial) 
        : current_state_(initial), initial_state_(initial) {}
    
    // State machine configuration
    StateMachine& add_transition(State from, Event event, State to,
                                 Guard guard = nullptr,
                                 Action action = nullptr) {
        transitions_[from].push_back({from, event, to, guard, action});
        return *this;
    }
    
    StateMachine& add_internal_transition(State state, Event event,
                                         Action action,
                                         Guard guard = nullptr) {
        transitions_[state].push_back({state, event, state, guard, action});
        return *this;
    }
    
    StateMachine& on_enter(State state, Action action) {
        entry_actions_[state] = action;
        return *this;
    }
    
    StateMachine& on_exit(State state, Action action) {
        exit_actions_[state] = action;
        return *this;
    }
    
    StateMachine& on_state(State state, Action action) {
        internal_actions_[state] = action;
        return *this;
    }
    
    // Event processing
    bool process_event(Event event) {
        // Queue event if already processing (avoid recursion)
        if (processing_) {
            event_queue_.push(event);
            return true;
        }
        
        processing_ = true;
        bool result = process_event_impl(event);
        
        // Process queued events
        while (!event_queue_.empty()) {
            Event queued = event_queue_.front();
            event_queue_.pop();
            process_event_impl(queued);
        }
        
        processing_ = false;
        return result;
    }
    
    // State queries
    State current_state() const { return current_state_; }
    bool is_in_state(State state) const { return current_state_ == state; }
    
    // State machine control
    void reset() {
        transition_to(initial_state_);
    }
    
    void force_state(State state) {
        current_state_ = state;
    }
    
    // History support
    void save_history(State parent_state) {
        history_[parent_state] = current_state_;
    }
    
    State get_history(State parent_state) const {
        auto it = history_.find(parent_state);
        return it != history_.end() ? it->second : parent_state;
    }
    
    // Observers
    void on_state_change(StateChangeCallback callback) {
        state_change_callbacks_.push_back(std::move(callback));
    }
    
    // Visualization
    std::string to_dot() const {
        std::stringstream ss;
        ss << "digraph StateMachine {\n";
        ss << "  rankdir=LR;\n";
        ss << "  node [shape=ellipse];\n";
        
        // Current state highlighting
        ss << "  " << static_cast<int>(current_state_) 
           << " [style=filled, fillcolor=yellow];\n";
        
        // Transitions
        for (const auto& [from, trans_list] : transitions_) {
            for (const auto& trans : trans_list) {
                ss << "  " << static_cast<int>(trans.from) 
                   << " -> " << static_cast<int>(trans.to)
                   << " [label=\"" << static_cast<int>(trans.event) << "\"];\n";
            }
        }
        
        ss << "}\n";
        return ss.str();
    }
    
private:
    bool process_event_impl(Event event) {
        auto it = transitions_.find(current_state_);
        if (it == transitions_.end()) {
            return false;
        }
        
        for (const auto& transition : it->second) {
            if (transition.event != event) {
                continue;
            }
            
            // Check guard condition
            if (transition.guard && !transition.guard()) {
                continue;
            }
            
            // Perform transition
            if (transition.from != transition.to) {
                transition_to(transition.to);
            }
            
            // Execute action
            if (transition.action) {
                transition.action();
            }
            
            return true;
        }
        
        return false;
    }
    
    void transition_to(State new_state) {
        State old_state = current_state_;
        
        // Exit current state
        if (auto it = exit_actions_.find(old_state); it != exit_actions_.end()) {
            it->second();
        }
        
        // Change state
        current_state_ = new_state;
        
        // Enter new state
        if (auto it = entry_actions_.find(new_state); it != entry_actions_.end()) {
            it->second();
        }
        
        // Notify observers
        for (const auto& callback : state_change_callbacks_) {
            callback(old_state, new_state);
        }
    }
};

// Hierarchical State Machine
template<typename StateEnum, typename EventEnum>
class HierarchicalStateMachine {
    struct StateInfo {
        std::optional<StateEnum> parent;
        std::vector<StateEnum> children;
        bool is_composite = false;
        std::optional<StateEnum> initial_child;
    };
    
    StateMachine<StateEnum, EventEnum> machine_;
    std::unordered_map<StateEnum, StateInfo> state_hierarchy_;
    
public:
    HierarchicalStateMachine(StateEnum initial) : machine_(initial) {}
    
    // Define hierarchy
    void set_composite_state(StateEnum state, StateEnum initial_child) {
        state_hierarchy_[state].is_composite = true;
        state_hierarchy_[state].initial_child = initial_child;
        state_hierarchy_[initial_child].parent = state;
    }
    
    void add_child_state(StateEnum parent, StateEnum child) {
        state_hierarchy_[parent].children.push_back(child);
        state_hierarchy_[child].parent = parent;
    }
    
    // Check if in state (including parent states)
    bool is_in_state(StateEnum state) const {
        StateEnum current = machine_.current_state();
        
        while (true) {
            if (current == state) {
                return true;
            }
            
            auto it = state_hierarchy_.find(current);
            if (it == state_hierarchy_.end() || !it->second.parent) {
                return false;
            }
            
            current = *it->second.parent;
        }
    }
    
    // Process event with bubbling
    bool process_event(EventEnum event) {
        return machine_.process_event(event);
    }
};
```
**Why necessary**: Complex state management, workflow states, protocol implementations.
**Usage**: Game AI, network protocols, UI state management, workflow engines.

### `pipeline.hpp`
```cpp
template<typename Input, typename Output>
class Pipeline {
public:
    using StageFunc = std::function<void(Input&, Output&, Context&)>;
    
    struct Stage {
        std::string name;
        StageFunc function;
        std::function<bool(const Input&)> precondition;
        std::function<void(const std::exception&, Context&)> error_handler;
        bool is_parallel = false;
        std::chrono::milliseconds timeout{0};
    };
    
    class Context {
        std::unordered_map<std::string, std::any> data_;
        std::vector<std::string> log_;
        bool cancelled_ = false;
        
    public:
        template<typename T>
        void set(const std::string& key, T value) {
            data_[key] = std::move(value);
        }
        
        template<typename T>
        T get(const std::string& key) const {
            auto it = data_.find(key);
            if (it != data_.end()) {
                return std::any_cast<T>(it->second);
            }
            throw std::runtime_error("Key not found: " + key);
        }
        
        void log(const std::string& message) {
            log_.push_back(message);
        }
        
        const std::vector<std::string>& logs() const { return log_; }
        
        void cancel() { cancelled_ = true; }
        bool is_cancelled() const { return cancelled_; }
    };
    
private:
    std::vector<Stage> stages_;
    std::function<void(const Input&, Output&, Context&)> final_handler_;
    std::function<void(const std::exception&, Context&)> global_error_handler_;
    
public:
    Pipeline() = default;
    
    // Pipeline building
    Pipeline& add_stage(const std::string& name, StageFunc func) {
        stages_.push_back({name, func});
        return *this;
    }
    
    Pipeline& add_parallel_stage(const std::string& name, StageFunc func) {
        stages_.push_back({name, func, nullptr, nullptr, true});
        return *this;
    }
    
    Pipeline& add_stage_with_precondition(const std::string& name,
                                          StageFunc func,
                                          std::function<bool(const Input&)> precond) {
        stages_.push_back({name, func, precond});
        return *this;
    }
    
    Pipeline& add_stage_with_timeout(const std::string& name,
                                     StageFunc func,
                                     std::chrono::milliseconds timeout) {
        Stage stage{name, func};
        stage.timeout = timeout;
        stages_.push_back(stage);
        return *this;
    }
    
    Pipeline& on_complete(std::function<void(const Input&, Output&, Context&)> handler) {
        final_handler_ = handler;
        return *this;
    }
    
    Pipeline& on_error(std::function<void(const std::exception&, Context&)> handler) {
        global_error_handler_ = handler;
        return *this;
    }
    
    // Pipeline execution
    Output execute(const Input& input) {
        Context context;
        Output output;
        
        execute_with_context(input, output, context);
        
        return output;
    }
    
    void execute_with_context(const Input& input, Output& output, Context& context) {
        Input current_input = input;
        
        // Group consecutive parallel stages
        std::vector<std::vector<Stage>> stage_groups = group_stages();
        
        for (const auto& group : stage_groups) {
            if (context.is_cancelled()) {
                break;
            }
            
            if (group.size() == 1 && !group[0].is_parallel) {
                // Sequential execution
                execute_stage(group[0], current_input, output, context);
            } else {
                // Parallel execution
                execute_parallel_stages(group, current_input, output, context);
            }
        }
        
        // Final handler
        if (final_handler_ && !context.is_cancelled()) {
            final_handler_(input, output, context);
        }
    }
    
    // Async execution
    std::future<Output> execute_async(const Input& input) {
        return std::async(std::launch::async, [this, input]() {
            return execute(input);
        });
    }
    
    // Pipeline composition
    template<typename IntermediateOutput>
    auto then(Pipeline<Output, IntermediateOutput>& next) {
        return ComposedPipeline<Input, Output, IntermediateOutput>(*this, next);
    }
    
    // Pipeline information
    std::size_t stage_count() const { return stages_.size(); }
    
    std::vector<std::string> stage_names() const {
        std::vector<std::string> names;
        for (const auto& stage : stages_) {
            names.push_back(stage.name);
        }
        return names;
    }
    
private:
    void execute_stage(const Stage& stage, Input& input, Output& output, Context& context) {
        // Check precondition
        if (stage.precondition && !stage.precondition(input)) {
            context.log("Stage " + stage.name + " skipped due to precondition");
            return;
        }
        
        try {
            context.log("Executing stage: " + stage.name);
            
            if (stage.timeout.count() > 0) {
                // Execute with timeout
                auto future = std::async(std::launch::async, [&]() {
                    stage.function(input, output, context);
                });
                
                if (future.wait_for(stage.timeout) == std::future_status::timeout) {
                    throw std::runtime_error("Stage " + stage.name + " timed out");
                }
            } else {
                // Execute normally
                stage.function(input, output, context);
            }
            
            context.log("Stage " + stage.name + " completed");
            
        } catch (const std::exception& e) {
            context.log("Stage " + stage.name + " failed: " + e.what());
            
            if (stage.error_handler) {
                stage.error_handler(e, context);
            } else if (global_error_handler_) {
                global_error_handler_(e, context);
            } else {
                throw;
            }
        }
    }
    
    void execute_parallel_stages(const std::vector<Stage>& stages,
                                 Input& input,
                                 Output& output,
                                 Context& context) {
        std::vector<std::future<void>> futures;
        std::mutex output_mutex;
        
        for (const auto& stage : stages) {
            futures.push_back(std::async(std::launch::async, [&]() {
                Output local_output;
                execute_stage(stage, input, local_output, context);
                
                // Merge outputs
                std::lock_guard lock(output_mutex);
                merge_output(output, local_output);
            }));
        }
        
        // Wait for all parallel stages
        for (auto& future : futures) {
            future.wait();
        }
    }
    
    std::vector<std::vector<Stage>> group_stages() const {
        std::vector<std::vector<Stage>> groups;
        std::vector<Stage> current_group;
        
        for (const auto& stage : stages_) {
            if (stage.is_parallel) {
                current_group.push_back(stage);
            } else {
                if (!current_group.empty()) {
                    groups.push_back(current_group);
                    current_group.clear();
                }
                groups.push_back({stage});
            }
        }
        
        if (!current_group.empty()) {
            groups.push_back(current_group);
        }
        
        return groups;
    }
    
    void merge_output(Output& target, const Output& source) {
        // Default implementation - should be specialized
        target = source;
    }
};

// Pipeline builder for fluent interface
template<typename Input, typename Output>
class PipelineBuilder {
    Pipeline<Input, Output> pipeline_;
    
public:
    PipelineBuilder& stage(const std::string& name,
                          typename Pipeline<Input, Output>::StageFunc func) {
        pipeline_.add_stage(name, func);
        return *this;
    }
    
    PipelineBuilder& parallel(const std::string& name,
                             typename Pipeline<Input, Output>::StageFunc func) {
        pipeline_.add_parallel_stage(name, func);
        return *this;
    }
    
    PipelineBuilder& conditional(const std::string& name,
                                typename Pipeline<Input, Output>::StageFunc func,
                                std::function<bool(const Input&)> condition) {
        pipeline_.add_stage_with_precondition(name, func, condition);
        return *this;
    }
    
    Pipeline<Input, Output> build() {
        return std::move(pipeline_);
    }
};
```
**Why necessary**: Data processing pipelines, ETL operations, multi-stage workflows.
**Usage**: Data transformation, image processing, compilation pipelines, request handling.

### `scheduler.hpp`
```cpp
class TaskScheduler {
public:
    using TaskId = uint64_t;
    using Task = std::function<void()>;
    using Duration = std::chrono::milliseconds;
    using TimePoint = std::chrono::steady_clock::time_point;
    
    enum class TaskStatus {
        Pending,
        Running,
        Completed,
        Failed,
        Cancelled
    };
    
    struct TaskInfo {
        TaskId id;
        std::string name;
        Task task;
        TimePoint scheduled_time;
        std::optional<Duration> interval;  // For recurring tasks
        std::vector<TaskId> dependencies;
        TaskStatus status = TaskStatus::Pending;
        int priority = 0;
        std::optional<std::exception_ptr> error;
        std::size_t retry_count = 0;
        std::size_t max_retries = 0;
    };
    
private:
    std::priority_queue<TaskInfo*, std::vector<TaskInfo*>, TaskComparator> task_queue_;
    std::unordered_map<TaskId, std::unique_ptr<TaskInfo>> tasks_;
    std::unordered_map<TaskId, std::set<TaskId>> dependency_graph_;
    
    std::vector<std::thread> worker_threads_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_{true};
    std::atomic<TaskId> next_task_id_{1};
    
    struct TaskComparator {
        bool operator()(const TaskInfo* a, const TaskInfo* b) const {
            if (a->priority != b->priority) {
                return a->priority < b->priority;
            }
            return a->scheduled_time > b->scheduled_time;
        }
    };
    
public:
    explicit TaskScheduler(std::size_t num_threads = std::thread::hardware_concurrency())
        : worker_threads_(num_threads) {
        
        for (auto& thread : worker_threads_) {
            thread = std::thread([this] { worker_loop(); });
        }
    }
    
    ~TaskScheduler() {
        shutdown();
    }
    
    // Schedule tasks
    TaskId schedule(Task task, const std::string& name = "") {
        return schedule_at(task, std::chrono::steady_clock::now(), name);
    }
    
    TaskId schedule_after(Task task, Duration delay, const std::string& name = "") {
        return schedule_at(task, std::chrono::steady_clock::now() + delay, name);
    }
    
    TaskId schedule_at(Task task, TimePoint when, const std::string& name = "") {
        TaskId id = next_task_id_.fetch_add(1);
        
        auto task_info = std::make_unique<TaskInfo>();
        task_info->id = id;
        task_info->name = name;
        task_info->task = std::move(task);
        task_info->scheduled_time = when;
        
        {
            std::lock_guard lock(queue_mutex_);
            task_queue_.push(task_info.get());
            tasks_[id] = std::move(task_info);
        }
        
        queue_cv_.notify_one();
        return id;
    }
    
    // Recurring tasks
    TaskId schedule_recurring(Task task, Duration interval, const std::string& name = "") {
        TaskId id = next_task_id_.fetch_add(1);
        
        auto task_info = std::make_unique<TaskInfo>();
        task_info->id = id;
        task_info->name = name;
        task_info->task = task;
        task_info->scheduled_time = std::chrono::steady_clock::now() + interval;
        task_info->interval = interval;
        
        {
            std::lock_guard lock(queue_mutex_);
            task_queue_.push(task_info.get());
            tasks_[id] = std::move(task_info);
        }
        
        queue_cv_.notify_one();
        return id;
    }
    
    // Cron-like scheduling
    TaskId schedule_cron(Task task, const std::string& cron_expr, const std::string& name = "") {
        // Parse cron expression and schedule
        auto next_time = parse_cron_next(cron_expr);
        return schedule_at(task, next_time, name);
    }
    
    // Task dependencies
    void add_dependency(TaskId task, TaskId depends_on) {
        std::lock_guard lock(queue_mutex_);
        dependency_graph_[depends_on].insert(task);
        
        if (auto it = tasks_.find(task); it != tasks_.end()) {
            it->second->dependencies.push_back(depends_on);
        }
    }
    
    // Task management
    void cancel(TaskId id) {
        std::lock_guard lock(queue_mutex_);
        
        if (auto it = tasks_.find(id); it != tasks_.end()) {
            it->second->status = TaskStatus::Cancelled;
        }
    }
    
    TaskStatus get_status(TaskId id) const {
        std::lock_guard lock(queue_mutex_);
        
        if (auto it = tasks_.find(id); it != tasks_.end()) {
            return it->second->status;
        }
        
        return TaskStatus::Cancelled;
    }
    
    void wait_for(TaskId id) {
        while (get_status(id) == TaskStatus::Pending ||
               get_status(id) == TaskStatus::Running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    // Scheduler control
    void pause() {
        running_ = false;
    }
    
    void resume() {
        running_ = true;
        queue_cv_.notify_all();
    }
    
    void shutdown() {
        running_ = false;
        queue_cv_.notify_all();
        
        for (auto& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
    
private:
    void worker_loop() {
        while (running_) {
            TaskInfo* task = nullptr;
            
            {
                std::unique_lock lock(queue_mutex_);
                
                queue_cv_.wait(lock, [this] {
                    return !running_ || !task_queue_.empty();
                });
                
                if (!running_) break;
                
                // Check if top task is ready
                if (!task_queue_.empty()) {
                    auto now = std::chrono::steady_clock::now();
                    task = const_cast<TaskInfo*>(task_queue_.top());
                    
                    if (task->scheduled_time <= now && 
                        task->status == TaskStatus::Pending &&
                        dependencies_satisfied(task)) {
                        
                        task_queue_.pop();
                        task->status = TaskStatus::Running;
                    } else {
                        task = nullptr;
                    }
                }
            }
            
            if (task) {
                execute_task(task);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }
    
    void execute_task(TaskInfo* task) {
        try {
            task->task();
            task->status = TaskStatus::Completed;
            
            // Notify dependent tasks
            notify_dependents(task->id);
            
            // Reschedule if recurring
            if (task->interval) {
                task->scheduled_time = std::chrono::steady_clock::now() + *task->interval;
                task->status = TaskStatus::Pending;
                
                std::lock_guard lock(queue_mutex_);
                task_queue_.push(task);
                queue_cv_.notify_one();
            }
            
        } catch (const std::exception& e) {
            task->error = std::current_exception();
            
            if (task->retry_count < task->max_retries) {
                // Retry with exponential backoff
                task->retry_count++;
                auto delay = std::chrono::milliseconds(100 * (1 << task->retry_count));
                task->scheduled_time = std::chrono::steady_clock::now() + delay;
                task->status = TaskStatus::Pending;
                
                std::lock_guard lock(queue_mutex_);
                task_queue_.push(task);
            } else {
                task->status = TaskStatus::Failed;
            }
        }
    }
    
    bool dependencies_satisfied(const TaskInfo* task) const {
        for (TaskId dep_id : task->dependencies) {
            if (auto it = tasks_.find(dep_id); it != tasks_.end()) {
                if (it->second->status != TaskStatus::Completed) {
                    return false;
                }
            }
        }
        return true;
    }
    
    void notify_dependents(TaskId completed_task) {
        std::lock_guard lock(queue_mutex_);
        
        if (auto it = dependency_graph_.find(completed_task); 
            it != dependency_graph_.end()) {
            
            for (TaskId dependent : it->second) {
                if (auto task_it = tasks_.find(dependent); 
                    task_it != tasks_.end()) {
                    
                    if (dependencies_satisfied(task_it->second.get())) {
                        queue_cv_.notify_one();
                    }
                }
            }
        }
    }
    
    TimePoint parse_cron_next(const std::string& cron_expr) {
        // Simplified cron parsing - would need full implementation
        return std::chrono::steady_clock::now() + std::chrono::hours(1);
    }
};
```
**Why necessary**: Task scheduling, cron jobs, dependency management, retry logic.
**Usage**: Background tasks, periodic jobs, workflow orchestration, batch processing.

### `transaction.hpp`
```cpp
class Transaction {
public:
    using TransactionId = uint64_t;
    using Savepoint = std::string;
    
    enum class Status {
        Active,
        Preparing,
        Prepared,
        Committing,
        Committed,
        Aborting,
        Aborted
    };
    
    enum class IsolationLevel {
        ReadUncommitted,
        ReadCommitted,
        RepeatableRead,
        Serializable
    };
    
    virtual ~Transaction() = default;
    
    // Transaction control
    virtual void begin() = 0;
    virtual void commit() = 0;
    virtual void rollback() = 0;
    
    // Savepoints
    virtual void save(const Savepoint& name) = 0;
    virtual void rollback_to(const Savepoint& name) = 0;
    virtual void release(const Savepoint& name) = 0;
    
    // Transaction state
    virtual TransactionId id() const = 0;
    virtual Status status() const = 0;
    virtual IsolationLevel isolation_level() const = 0;
    
    // Two-phase commit
    virtual void prepare() = 0;
    virtual bool can_commit() const = 0;
};

// Transactional command
template<typename Resource>
class TransactionalCommand : public Command {
    std::shared_ptr<Transaction> transaction_;
    std::function<void(Resource&)> execute_func_;
    std::function<void(Resource&)> compensate_func_;
    Resource& resource_;
    
public:
    TransactionalCommand(std::shared_ptr<Transaction> txn,
                        Resource& resource,
                        std::function<void(Resource&)> exec,
                        std::function<void(Resource&)> compensate)
        : transaction_(txn)
        , execute_func_(exec)
        , compensate_func_(compensate)
        , resource_(resource) {}
    
    void execute() override {
        transaction_->begin();
        try {
            execute_func_(resource_);
            transaction_->commit();
        } catch (...) {
            transaction_->rollback();
            throw;
        }
    }
    
    void undo() override {
        transaction_->begin();
        try {
            compensate_func_(resource_);
            transaction_->commit();
        } catch (...) {
            transaction_->rollback();
            throw;
        }
    }
    
    std::string name() const override {
        return "TransactionalCommand";
    }
};

// Transaction manager
class TransactionManager {
    std::unordered_map<TransactionId, std::shared_ptr<Transaction>> active_transactions_;
    std::mutex mutex_;
    std::atomic<TransactionId> next_id_{1};
    
public:
    std::shared_ptr<Transaction> begin_transaction(IsolationLevel level = 
                                                   IsolationLevel::ReadCommitted) {
        auto txn = create_transaction(next_id_.fetch_add(1), level);
        
        std::lock_guard lock(mutex_);
        active_transactions_[txn->id()] = txn;
        
        txn->begin();
        return txn;
    }
    
    void commit(TransactionId id) {
        std::lock_guard lock(mutex_);
        
        if (auto it = active_transactions_.find(id); it != active_transactions_.end()) {
            it->second->commit();
            active_transactions_.erase(it);
        }
    }
    
    void rollback(TransactionId id) {
        std::lock_guard lock(mutex_);
        
        if (auto it = active_transactions_.find(id); it != active_transactions_.end()) {
            it->second->rollback();
            active_transactions_.erase(it);
        }
    }
    
    // Two-phase commit coordinator
    bool two_phase_commit(const std::vector<std::shared_ptr<Transaction>>& participants) {
        // Phase 1: Prepare
        for (auto& txn : participants) {
            txn->prepare();
            if (!txn->can_commit()) {
                // Abort all
                for (auto& t : participants) {
                    t->rollback();
                }
                return false;
            }
        }
        
        // Phase 2: Commit
        for (auto& txn : participants) {
            txn->commit();
        }
        
        return true;
    }
    
private:
    virtual std::shared_ptr<Transaction> create_transaction(TransactionId id,
                                                           IsolationLevel level) = 0;
};
```
**Why necessary**: Transactional operations, distributed transactions, data consistency.
**Usage**: Database operations, distributed systems, saga patterns, business transactions.

## Workflow Patterns

### Command Pattern Usage
```cpp
class MoveCommand : public Command {
    Object& object_;
    Vector3 old_position_;
    Vector3 new_position_;
    
public:
    MoveCommand(Object& obj, const Vector3& new_pos)
        : object_(obj)
        , old_position_(obj.position())
        , new_position_(new_pos) {}
    
    void execute() override {
        object_.set_position(new_position_);
    }
    
    void undo() override {
        object_.set_position(old_position_);
    }
    
    std::string name() const override {
        return "Move Object";
    }
};

// Usage
UndoStack undo_stack;
undo_stack.push(std::make_unique<MoveCommand>(object, Vector3(10, 0, 0)));
```

### State Machine for Workflow
```cpp
enum class OrderState {
    Created,
    Paid,
    Processing,
    Shipped,
    Delivered,
    Cancelled
};

enum class OrderEvent {
    Pay,
    Process,
    Ship,
    Deliver,
    Cancel
};

StateMachine<OrderState, OrderEvent> order_workflow(OrderState::Created);

order_workflow
    .add_transition(OrderState::Created, OrderEvent::Pay, OrderState::Paid)
    .add_transition(OrderState::Paid, OrderEvent::Process, OrderState::Processing)
    .add_transition(OrderState::Processing, OrderEvent::Ship, OrderState::Shipped)
    .add_transition(OrderState::Shipped, OrderEvent::Deliver, OrderState::Delivered)
    .add_transition(OrderState::Created, OrderEvent::Cancel, OrderState::Cancelled)
    .on_enter(OrderState::Shipped, []() {
        send_shipping_notification();
    });

// Process order
order_workflow.process_event(OrderEvent::Pay);
```

### Pipeline Processing
```cpp
auto data_pipeline = PipelineBuilder<RawData, ProcessedData>()
    .stage("validate", [](RawData& data, ProcessedData& output, auto& ctx) {
        validate_data(data);
        ctx.log("Validation completed");
    })
    .parallel("transform", [](RawData& data, ProcessedData& output, auto& ctx) {
        output = transform_data(data);
    })
    .conditional("optimize", 
        [](RawData& data, ProcessedData& output, auto& ctx) {
            optimize_data(output);
        },
        [](const RawData& data) {
            return data.size() > 1000;
        })
    .build();

auto result = data_pipeline.execute(raw_data);
```

### Task Scheduling with Dependencies
```cpp
TaskScheduler scheduler(4);  // 4 worker threads

auto task1 = scheduler.schedule([]() {
    std::cout << "Task 1: Load data\n";
});

auto task2 = scheduler.schedule([]() {
    std::cout << "Task 2: Process data\n";
});
scheduler.add_dependency(task2, task1);

auto task3 = scheduler.schedule([]() {
    std::cout << "Task 3: Generate report\n";
});
scheduler.add_dependency(task3, task2);

// Recurring task
scheduler.schedule_recurring([]() {
    std::cout << "Health check\n";
}, std::chrono::seconds(30));
```

## Performance Considerations

- **Command memory**: Pool commands for frequent operations
- **State machine: O(1) state transitions with hash maps
- **Pipeline overhead**: ~100-200ns per stage
- **Task scheduling**: Lock-free queues for high throughput
- **Transaction isolation**: Choose appropriate level for performance

## Testing Strategy

- **Command testing**: Verify execute/undo symmetry
- **State coverage**: Test all state transitions
- **Pipeline stages**: Test individual stages and composition
- **Scheduler timing**: Verify task execution timing
- **Transaction rollback**: Test failure scenarios

## Usage Guidelines

1. **Commands**: Keep commands small and focused
2. **State machines**: Document state transitions clearly
3. **Pipelines**: Handle errors at stage level
4. **Scheduling**: Use appropriate retry strategies
5. **Transactions**: Keep transactions short

## Anti-patterns to Avoid

- Commands with side effects in constructors
- State machines with too many states
- Blocking operations in pipeline stages
- Unbounded task queues
- Long-running transactions

## Dependencies
- `base/` - For base patterns
- `events/` - For workflow events
- `concurrency/` - For async execution
- Standard library (C++20)

## Future Enhancements
- Workflow versioning
- Distributed workflows
- Visual workflow editor
- Workflow debugging tools
- Machine learning optimization