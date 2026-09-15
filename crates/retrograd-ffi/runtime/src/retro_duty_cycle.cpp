#include "retro_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

namespace retro {

namespace {

// `work * (1 / fraction - 1)`, saturated at `duty_cycle_limiter::max_debt`.
// The multiplication is done in double because the factor is not an integer;
// the result is narrowed back to the clock's integer duration, which is the
// narrowing `max_debt` bounds.
duty_cycle_limiter::duration owed_idle(
        duty_cycle_limiter::duration work,
        double idle_per_work) {
    const double ceiling = static_cast<double>(
            std::chrono::duration_cast<duty_cycle_limiter::duration>(
                    duty_cycle_limiter::max_debt).count());
    const double owed = static_cast<double>(work.count()) * idle_per_work;
    return duty_cycle_limiter::duration(
            static_cast<duty_cycle_limiter::duration::rep>(std::min(owed, ceiling)));
}

} // namespace

bool duty_cycle_limiter::is_valid_fraction(float fraction) noexcept {
    return std::isfinite(fraction) && fraction > 0.0f && fraction <= 1.0f;
}

void duty_cycle_limiter::configure(float fraction, bool gpu_active) noexcept {
    // Every window and every counter, not just the debt: this is a mode switch,
    // and seconds accumulated under the previous fraction describe a policy that
    // is no longer in force. The installed time source survives it - it belongs
    // to the test harness, not to the setting.
    idle_per_work_ = 0.0;
    debt_ = duration::zero();
    compute_ = duration::zero();
    idle_ = duration::zero();
    window_start_ = clock::time_point {};
    activated_at_ = clock::time_point {};

    requested_ = fraction;
    active_ = gpu_active && fraction < 1.0f;
    if (!active_) {
        return;
    }
    idle_per_work_ = 1.0 / static_cast<double>(fraction) - 1.0;
    activated_at_ = now();
    window_start_ = activated_at_;
}

duty_cycle_clock::time_point duty_cycle_limiter::now() const noexcept {
    return time_ ? time_->now() : clock::now();
}

void duty_cycle_limiter::begin_window() noexcept {
    if (!active_) {
        return;
    }
    // Everything since the previous boundary was host time this limiter did not
    // choose - sampling between two decodes, a progress callback, a reward
    // subprocess, a judge, a dataset pass - and the device was free for all of
    // it. It is never charged as work; a gap long enough to be a pause rather
    // than a hand-off also discards the pending debt, because repaying it now
    // would idle for compute the neighbour has already had.
    const clock::time_point opened = now();
    if (opened - window_start_ >= stale_window) {
        debt_ = duration::zero();
    }
    window_start_ = opened;
}

void duty_cycle_limiter::account_window() noexcept {
    if (!active_) {
        return;
    }
    const clock::time_point boundary = now();
    account_synchronized_work(boundary - window_start_);
    window_start_ = boundary;
}

void duty_cycle_limiter::account_synchronized_work(duration work) noexcept {
    if (!active_ || work <= duration::zero()) {
        return;
    }
    compute_ += work;
    debt_ = std::min(debt_ + owed_idle(work, idle_per_work_),
            std::chrono::duration_cast<duration>(max_debt));
}

void duty_cycle_limiter::idle_if_needed() {
    if (!active_) {
        return;
    }
    if (debt_ < min_sleep) {
        // Carried, not dropped. The window still reopens: a caller that lets
        // host time pass after this must not have it charged as work, and
        // making that hold unconditionally is worth one clock read on a path
        // whose whole purpose is to sleep for milliseconds.
        window_start_ = now();
        return;
    }
    const clock::time_point started = now();
    const clock::time_point deadline =
            started + std::min(debt_, std::chrono::duration_cast<duration>(max_sleep));
    if (time_) {
        time_->sleep_until(deadline);
    } else {
        std::this_thread::sleep_until(deadline);
    }
    // What was actually slept, not what was asked for: an oversleeping host
    // scheduler has already released the compute, and charging the request
    // instead would repay the same debt twice.
    const duration slept = std::max(now() - started, duration::zero());
    idle_ += slept;
    debt_ = std::max(debt_ - slept, duration::zero());
    // The repaid time is not this trainer's compute, so the next window opens
    // here rather than where the previous one closed.
    window_start_ = now();
}

void duty_cycle_limiter::reset_window() noexcept {
    if (!active_) {
        return;
    }
    debt_ = duration::zero();
    window_start_ = now();
}

double duty_cycle_limiter::compute_seconds() const noexcept {
    return std::chrono::duration<double>(compute_).count();
}

double duty_cycle_limiter::idle_seconds() const noexcept {
    return std::chrono::duration<double>(idle_).count();
}

double duty_cycle_limiter::wall_seconds() const noexcept {
    if (!active_) {
        return 0.0;
    }
    return std::chrono::duration<double>(now() - activated_at_).count();
}

retro_duty_cycle_stats duty_cycle_snapshot(const duty_cycle_limiter & limiter) {
    retro_duty_cycle_stats stats {};
    stats.requested_fraction = limiter.requested();
    stats.active = limiter.enabled();
    stats.compute_seconds = limiter.compute_seconds();
    stats.idle_seconds = limiter.idle_seconds();
    stats.wall_seconds = limiter.wall_seconds();
    return stats;
}

namespace {

// Time only moves when the script says so, so a replay of ten simulated minutes
// costs microseconds. Counting reads and sleeps is not decoration: "the
// disabled path touches neither the clock nor the sleeper" is the zero-overhead
// property, and it is not observable from the accumulated seconds.
struct scripted_time_source final : duty_cycle_time_source {
    duty_cycle_clock::time_point at {};
    uint64_t reads = 0;
    uint64_t sleeps = 0;

    duty_cycle_clock::time_point now() override {
        ++reads;
        return at;
    }

    void sleep_until(duty_cycle_clock::time_point deadline) override {
        ++sleeps;
        at = std::max(at, deadline);
    }
};

} // namespace

int duty_cycle_probe_impl(
        float fraction,
        const retro_duty_cycle_event * events,
        size_t n_events,
        retro_duty_cycle_probe * out_probe) {
    if (!out_probe || (n_events > 0 && !events)
            || !duty_cycle_limiter::is_valid_fraction(fraction)) {
        set_error("retro_probe_duty_cycle requires a fraction in (0, 1] and a valid script");
        return -1;
    }
    scripted_time_source time;
    duty_cycle_limiter limiter;
    limiter.set_time_source(&time);
    // `true`: the replay answers what the controller does on a GPU. Whether a
    // CPU backend disables it is a property of `configure`, checked where the
    // trainer is.
    limiter.configure(fraction, true);
    // The reads `configure` spends opening the first window describe the
    // installation, not the replay, and would hide the disabled-path property
    // the counter exists to prove.
    time.reads = 0;

    // A narrowing with a contract rather than a proof: `micros` is a uint64
    // from the caller and the clock counts signed nanoseconds, so a script step
    // longer than a day is rejected instead of silently wrapping a time_point.
    // Nothing this probe exists to check needs one.
    constexpr uint64_t max_event_micros = 24ull * 60 * 60 * 1000 * 1000;

    for (size_t i = 0; i < n_events; ++i) {
        if (events[i].micros > max_event_micros) {
            set_error("retro_duty_cycle_event.micros must not exceed one day");
            return -1;
        }
        const auto span = std::chrono::microseconds(
                static_cast<std::chrono::microseconds::rep>(events[i].micros));
        switch (events[i].kind) {
            case RETRO_DUTY_CYCLE_EVENT_WORK:
                limiter.begin_window();
                time.at += span;
                limiter.account_window();
                break;
            case RETRO_DUTY_CYCLE_EVENT_IDLE:
                limiter.idle_if_needed();
                break;
            case RETRO_DUTY_CYCLE_EVENT_HOST:
                // Host time passes, then the next submission opens its window
                // and decides what that time was. This is the shape every real
                // call site has; the limiter measures the gap rather than being
                // told it.
                time.at += span;
                limiter.begin_window();
                break;
            case RETRO_DUTY_CYCLE_EVENT_RESET:
                limiter.reset_window();
                break;
            default:
                set_error("unknown retro_duty_cycle_event kind");
                return -1;
        }
    }

    out_probe->stats = duty_cycle_snapshot(limiter);
    out_probe->clock_reads = time.reads;
    out_probe->sleep_calls = time.sleeps;
    return 0;
}

} // namespace retro
