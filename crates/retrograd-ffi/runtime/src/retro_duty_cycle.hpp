#pragma once

// GPU duty-cycle limiter: the wall-clock policy that lets a training run leave
// regular compute windows to other users of the device.
//
// It has no dependency on llama.cpp, ggml or `trainer_state`, so its arithmetic
// - the debt, the coalescing floor, the sleep cap and the stale-window rule -
// is exercised without a model and without waiting, through
// `retro_probe_duty_cycle`.

#include <chrono>
#include <cstdint>

namespace retro {

using duty_cycle_clock = std::chrono::steady_clock;

// Test seam. Unit tests drive the controller on a clock they advance
// themselves, so a 25% duty cycle over ten simulated minutes is checked in
// microseconds. Production installs none and reads `steady_clock` directly; the
// branch is on the enabled path only, and the disabled path reaches neither.
struct duty_cycle_time_source {
    virtual ~duty_cycle_time_source() = default;
    virtual duty_cycle_clock::time_point now() = 0;
    // Must advance `now()` to at least `deadline`.
    virtual void sleep_until(duty_cycle_clock::time_point deadline) = 0;
};

// Idles this trainer for a share of its own wall time so another workload can
// make progress on the same device.
//
// The unit is a *duty cycle*, not GPU utilization: it bounds the fraction of
// wall time during which this trainer has work submitted and is waiting on it.
// Backend telemetry, kernel occupancy and other processes can all make a system
// monitor disagree.
//
// The accounting is one-sided on purpose. Work adds idle debt; nothing ever
// adds credit. A long CPU phase - data loading, judging, tokenization - must
// not authorize an equally long unthrottled GPU burst afterwards, so the
// unaccounted time simply passes.
//
// **Not thread-safe, and does not need to be.** One `trainer_state` is driven
// by one thread at a time (the Rust side serializes trainer calls) and every
// accounting site runs on that thread. A future concurrent generation worker
// would need a different design - one limiter arbitrating several submission
// streams - not a mutex bolted onto this one.
class duty_cycle_limiter {
public:
    using clock    = duty_cycle_clock;
    using duration = clock::duration;

    // A single uninterruptible sleep. Debt beyond it is carried to the next
    // boundary, which is what bounds the extra pause/cancel latency the policy
    // adds however large the accumulated debt gets.
    static constexpr auto max_sleep = std::chrono::milliseconds(100);
    // Debt below this is carried rather than slept: a sleep that short is host
    // scheduler noise, not released compute.
    static constexpr auto min_sleep = std::chrono::milliseconds(2);
    // Unaccounted host time - a run-control pause acknowledged inside a
    // progress callback, a reward subprocess, an ad-hoc operation between two
    // boundaries - at or above which the pending debt is discarded instead of
    // repaid. Five times `max_sleep`, so an ordinary progress row or a decode's
    // own sampling can never be mistaken for a pause, and short enough that a
    // real pause does not carry seconds of debt into the resume. Internal, not
    // configurable.
    static constexpr auto stale_window = std::chrono::milliseconds(500);
    // Ceiling on the pending debt. `fraction` is only bounded away from zero by
    // the `(0, 1]` contract, so a document naming `1e-30` would otherwise
    // overflow the nanosecond count. Saturating is the treatment a budget gets:
    // a debt this size is a stall rather than a duty cycle, and clamping keeps
    // both the arithmetic total and the symptom legible in the reported idle
    // seconds.
    static constexpr auto max_debt = std::chrono::hours(1);

    // Whether a value is a legal `max_gpu_duty_cycle`. `1.0f` is legal and
    // means "no limit"; the C ABI has no need for the TOML distinction between
    // omitted and explicit `1.0`.
    static bool is_valid_fraction(float fraction) noexcept;

    // Installs `fraction` and clears every window and counter. Precondition:
    // `is_valid_fraction(fraction)`.
    //
    // The limiter never engages on a CPU backend - `training.threads` is the
    // control there - but it keeps the requested value either way, so a report
    // can say `requested=0.5, active=false` instead of silently promising GPU
    // throttling to a run that has no GPU.
    void configure(float fraction, bool gpu_active) noexcept;

    bool  enabled()   const noexcept { return active_; }
    float requested() const noexcept { return requested_; }
    // The only gate other than the fraction itself, so a requested-but-inactive
    // limiter has exactly one possible cause.
    bool inactive_on_cpu() const noexcept { return !active_ && requested_ < 1.0f; }

    // Opens the window `account_window` closes, and in doing so decides what
    // the time since the previous boundary was: host time the limiter did not
    // choose. That time is never charged as work, and a gap of at least
    // `stale_window` also discards the pending debt.
    //
    // This is the single place the stale-window rule lives, so every kind of
    // uninvited pause goes through it - a progress callback that blocked on a
    // run-control pause, a reward subprocess, a judge, an ad-hoc operation
    // between two updates - without each call site having to recognize its own.
    // Call it immediately before submitting, and again after any host work that
    // happened between an accounting boundary and the next submission.
    void begin_window() noexcept;
    // Closes it, charging the elapsed host time as synchronized GPU work, and
    // reopens a window at the same instant.
    void account_window() noexcept;
    // Same, for a caller that timed its own window. The duration must cover
    // submission through completion: idle time only counts once the work it
    // balances is done.
    void account_synchronized_work(duration work) noexcept;
    // Repays up to `max_sleep` of the pending debt, carrying the remainder to
    // later boundaries, and reopens the window. Separate from accounting so a
    // due progress callback can deliver cancellation before deliberate idle
    // time.
    void idle_if_needed();
    // Forgets the pending debt and reopens the window. The lifetime counters
    // survive: they describe the run, not the current window.
    void reset_window() noexcept;

    // Cumulative synchronized GPU work this limiter accounted.
    double compute_seconds() const noexcept;
    // Cumulative time it actually slept.
    double idle_seconds() const noexcept;
    // Wall time since the limiter was enabled - the window over which the
    // policy has been in force, and the denominator that turns
    // `compute_seconds` into the share of the whole run rather than the share
    // of the accounted windows alone.
    double wall_seconds() const noexcept;

    // Test seam; see `duty_cycle_time_source`. Borrowed, not owned.
    void set_time_source(duty_cycle_time_source * source) noexcept { time_ = source; }

private:
    clock::time_point now() const noexcept;

    // Idle time owed per unit of work: `1 / fraction - 1`. Precomputed because
    // it is read on every accounted boundary and `fraction` only moves through
    // `configure`.
    double            idle_per_work_ = 0.0;
    float             requested_     = 1.0f;
    bool              active_        = false;
    duration          debt_ {};
    duration          compute_ {};
    duration          idle_ {};
    clock::time_point window_start_ {};
    clock::time_point activated_at_ {};

    duty_cycle_time_source * time_ = nullptr;
};

} // namespace retro
