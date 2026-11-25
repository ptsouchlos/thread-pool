#include <doctest/doctest.h>
#include <thread_pool/work_stealing_deque.h>

#include <deque>
#include <functional>
#include <semaphore>
#include <thread>

TEST_CASE("work_stealing_deque: construct queue") {
    dp::work_stealing_deque<std::uint64_t> queue{};
}

TEST_CASE("work_stealing_deque: construct and grow queue") {
    dp::work_stealing_deque<std::uint64_t> queue{2};

    queue.push_bottom(1);
    queue.push_bottom(2);
    queue.push_bottom(3);

    REQUIRE_EQ(queue.capacity(), 4);
}

TEST_CASE("work_stealing_deque: take bottom while queue is empty") {
    dp::work_stealing_deque<std::uint64_t> queue{};

    REQUIRE_EQ(queue.take_bottom(), std::nullopt);
}

TEST_CASE("work_stealing_deque: take bottom while queue is not empty") {
    dp::work_stealing_deque<std::uint64_t> queue{};

    queue.push_bottom(1);
    queue.push_bottom(2);
    queue.push_bottom(3);

    REQUIRE_EQ(queue.take_bottom(), 3);
    REQUIRE_EQ(queue.take_bottom(), 2);
    REQUIRE_EQ(queue.take_bottom(), 1);
    REQUIRE_EQ(queue.take_bottom(), std::nullopt);
}

TEST_CASE("work_stealing_deque: multiple thread steal single item") {
    dp::work_stealing_deque<std::uint64_t> queue{};

    queue.push_bottom(23567);
    std::uint64_t value = 0;

    {
        auto thread_task = [&queue, &value]() {
            if (const auto temp = queue.pop_top()) {
                value = temp.value();
            }
        };
        std::jthread t1{thread_task};
        std::jthread t2{thread_task};
        std::jthread t3{thread_task};
        std::jthread t4{thread_task};
    }

    REQUIRE_EQ(value, 23567);
}

TEST_CASE("work_stealing_deque: steal std::function while pushing") {
    dp::work_stealing_deque<std::function<void()>> deque{};
    std::atomic_uint64_t count{0};
    constexpr auto max = 64'000;
    auto expected_sum = 0;
    std::atomic_uint64_t pending_tasks{0};
    std::atomic_bool supplier_done{false};
    std::deque<std::binary_semaphore> signals;
    signals.emplace_back(0);
    signals.emplace_back(0);
    signals.emplace_back(0);
    signals.emplace_back(0);

    auto supply_task = [&] {
        for (auto i = 0; i < max; i++) {
            deque.push_bottom([&count, i]() { count += i; });
            expected_sum += i;
            pending_tasks.fetch_add(1, std::memory_order_release);
            // wake all threads
            if ((i + 1) % 8000 == 0) {
                for (auto& signal : signals) signal.release();
            }
        }
        supplier_done.store(true, std::memory_order_release);
    };

    auto task = [&](int id) {
        signals[id].acquire();
        // exit only when the supplier has finished AND nothing is pending: a bare
        // pending_tasks == 0 can occur while the supplier is between push_bottom and
        // fetch_add, so it is not a safe exit condition on its own
        while (!(supplier_done.load(std::memory_order_acquire) &&
                 pending_tasks.load(std::memory_order_acquire) == 0)) {
            auto value = deque.pop_top();
            if (value.has_value()) {
                auto temp = std::move(value.value());
                std::invoke(temp);
                pending_tasks.fetch_sub(1, std::memory_order_release);
            }
        }
    };

    {
        std::jthread supplier(supply_task);
        std::jthread t1(task, 0);
        std::jthread t2(task, 1);
        std::jthread t3(task, 2);
        std::jthread t4(task, 3);
    }

    REQUIRE_EQ(count.load(), expected_sum);
}

class move_only {
    // a live object carries its value; an object that has been moved from carries the
    // moved-from sentinel instead, so a live value can be told apart from a moved one
    int private_value_ = 0;

    static constexpr int moved_from = 2;

  public:
    move_only() = default;
    ~move_only() = default;
    move_only(move_only&) = delete;
    move_only(move_only&& other) noexcept : private_value_(other.private_value_) {
        other.private_value_ = moved_from;
    }
    move_only& operator=(move_only&) = delete;
    move_only& operator=(move_only&& other) noexcept {
        private_value_ = other.private_value_;
        other.private_value_ = moved_from;
        return *this;
    }
    [[nodiscard]] int secret() const { return private_value_; }
};

TEST_CASE("work_stealing_deque: store move only types") {
    move_only mv_only{};
    dp::work_stealing_deque<move_only> deque{};
    deque.push_bottom(std::move(mv_only));

    const auto value = deque.take_bottom();
    REQUIRE(value.has_value());
    REQUIRE_NE(value->secret(), 2);
}

TEST_CASE("work_stealing_deque: steal move only type") {
    move_only mv_only{};
    dp::work_stealing_deque<move_only> queue{};
    queue.push_bottom(std::move(mv_only));
    std::optional<move_only> value = std::nullopt;
    {
        auto thread_task = [&queue, &value]() {
            if (auto temp = queue.pop_top()) {
                value.emplace(std::move(temp.value()));
            }
        };

        std::jthread t1{thread_task};
        std::jthread t2{thread_task};
        std::jthread t3{thread_task};
        std::jthread t4{thread_task};
    }

    REQUIRE(value.has_value());
    REQUIRE_NE(value->secret(), 2);
}

#if __cpp_lib_move_only_function

TEST_CASE("work_stealing_deque: steal std::move_only_function while pushing") {
    dp::work_stealing_deque<std::move_only_function<void()>> deque{};
    std::atomic_uint64_t count{0};
    constexpr auto max = 64'000;
    auto expected_sum = 0;
    std::atomic_uint64_t pending_tasks{0};
    std::atomic_bool supplier_done{false};
    std::deque<std::binary_semaphore> signals;
    signals.emplace_back(0);
    signals.emplace_back(0);
    signals.emplace_back(0);
    signals.emplace_back(0);

    auto supply_task = [&] {
        for (auto i = 0; i < max; i++) {
            deque.push_bottom([&count, i]() { count += i; });
            expected_sum += i;
            pending_tasks.fetch_add(1, std::memory_order_release);
            // wake all threads
            if (i % 1000 == 0) {
                for (auto& signal : signals) signal.release();
            }
        }
        supplier_done.store(true, std::memory_order_release);
    };

    auto task = [&](int id) {
        signals[id].acquire();
        // exit only when the supplier has finished AND nothing is pending: a bare
        // pending_tasks == 0 can occur while the supplier is between push_bottom and
        // fetch_add, so it is not a safe exit condition on its own
        while (!(supplier_done.load(std::memory_order_acquire) &&
                 pending_tasks.load(std::memory_order_acquire) == 0)) {
            auto value = deque.pop_top();
            if (value.has_value()) {
                auto temp = std::move(value.value());
                std::invoke(std::move(temp));
                pending_tasks.fetch_sub(1, std::memory_order_release);
            }
        }
    };

    {
        std::jthread supplier(supply_task);
        std::jthread t1(task, 0);
        std::jthread t2(task, 1);
        std::jthread t3(task, 2);
        std::jthread t4(task, 3);
    }

    REQUIRE_EQ(count.load(), expected_sum);
}
#endif
