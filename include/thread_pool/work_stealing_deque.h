#pragma once

#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace dp {

#ifdef __cpp_lib_hardware_interference_size
    using std::hardware_destructive_interference_size;
#else
    // 64 bytes on x86-64 │ L1_CACHE_BYTES │ L1_CACHE_SHIFT │ __cacheline_aligned │ ...
    inline constexpr std::size_t hardware_destructive_interference_size =
        2 * sizeof(std::max_align_t);
#endif

    /**
     * @brief Chase-Lev work stealing queue
     * @details Support single producer, multiple consumer. The producer owns the back, consumers
     * own the top. Consumers can also take from the top of the queue. The queue is "lock-free" in
     * that it does not directly use mutexes or locks.
     *
     * This is an implementation of the deque described in "Correct and Efficient Work-Stealing for
     * Weak Memory Models" and "Dynamic Circular Work-Stealing Deque" by Chase,Lev.
     *
     * The resizable SPMC construct is inspired by the following implementations:
     * - https://github.com/ConorWilliams/ConcurrentDeque/blob/main/include/riften/deque.hpp
     * - https://github.com/taskflow/work-stealing-queue/blob/master/wsq.hpp
     *
     * The slot-claim protocol and the resize index mapping in this file are original; both references
     * above are incorrect under concurrent resize + steal, so they are credited as inspiration only.
     *
     * Each queue element lives on the heap and each slot of the underlying buffer is an
     * `std::atomic<T*>`; ownership of a pointer is claimed with a single `exchange(nullptr)`, so
     * exactly one thread ever takes a given element even while the owner pushes or resizes the
     * buffer.
     *
     * I've made some minor edits and changes based on new C++ 23 features and other personal coding
     * style choices/preferences.
     */
    template <typename T>
        requires std::is_destructible_v<T>
    class work_stealing_deque final {
        /**
         * @brief Simple circular array buffer that can regrow
         */
        class circular_buffer final {
            std::size_t size_;
            std::size_t mask_;
            std::atomic<T*>* buffer_ = nullptr;

          public:
            explicit circular_buffer(const std::size_t size)
                : size_(size), mask_(size - 1), buffer_(new std::atomic<T*>[size]) {
                // size must be a power of 2
                assert(std::has_single_bit(size));
                for (std::size_t i = 0; i != size_; ++i) {
                    buffer_[i].store(nullptr, relaxed);
                }
            }

            ~circular_buffer() {
                for (std::size_t i = 0; i != size_; ++i) {
                    if (auto* item = buffer_[i].load(relaxed)) {
                        item->~T();
                        ::operator delete(static_cast<void*>(item));
                    }
                }
                delete[] buffer_;
            }

            [[nodiscard]] std::int64_t capacity() const noexcept { return size_; }

            void store(const std::int64_t index, T&& value) {
                auto* item = new T(std::move(value));
                buffer_[index & mask_].store(item, relaxed);
            }

            /**
             * @brief Claim the pointer stored in the slot at `index`, leaving the slot null.
             * @details The claim is atomic: exactly one thread ever receives a given pointer.
             * @return T* The claimed pointer, or nullptr if it was already claimed.
             */
            T* claim(const std::int64_t index) {
                return buffer_[index & mask_].exchange(nullptr, acq_rel);
            }
        };

        constexpr static std::size_t default_count = 1024;
        alignas(hardware_destructive_interference_size) std::atomic_int64_t top_;
        alignas(hardware_destructive_interference_size) std::atomic_int64_t bottom_;
        alignas(hardware_destructive_interference_size) std::atomic<circular_buffer*> buffer_;

        std::vector<std::unique_ptr<circular_buffer>> garbage_{32};

        static constexpr std::memory_order relaxed = std::memory_order_relaxed;
        static constexpr std::memory_order acquire = std::memory_order_acquire;
        static constexpr std::memory_order consume = std::memory_order_consume;
        static constexpr std::memory_order release = std::memory_order_release;
        static constexpr std::memory_order seq_cst = std::memory_order_seq_cst;
        static constexpr std::memory_order acq_rel = std::memory_order_acq_rel;

      public:
        explicit work_stealing_deque(const std::size_t& capacity = default_count)
            : top_(0), bottom_(0), buffer_(new circular_buffer(capacity)) {}

        ~work_stealing_deque() {
            delete buffer_.load(relaxed);
            // garbage_ members are destroyed by the vector's own destruction
        }

        // queue is non-copyable
        work_stealing_deque(work_stealing_deque&) = delete;
        work_stealing_deque& operator=(work_stealing_deque&) = delete;

        [[nodiscard]] std::size_t capacity() const { return buffer_.load(relaxed)->capacity(); }
        [[nodiscard]] std::size_t size() const {
            const auto bottom = bottom_.load(relaxed);
            const auto top = top_.load(relaxed);
            return static_cast<std::size_t>(bottom >= top ? bottom - top : 0);
        }

        [[nodiscard]] bool empty() const { return !size(); }

        template <typename... Args>
        void emplace(Args&&... args) {
            // construct first in case it throws
            T value(std::forward<Args>(args)...);
            push_bottom(std::move(value));
        }

        /**
         * @brief Push data to the bottom of the queue.
         * @details Only the producer thread can push data to the bottom. Consumers should take data
         * from the top. See #pop_top.
         */
        void push_bottom(T&& value) {
            auto bottom = bottom_.load(relaxed);
            auto top = top_.load(acquire);
            auto* buffer = buffer_.load(relaxed);

            // check if the buffer is full
            if (buffer->capacity() < (bottom - top) + 1) {
                garbage_.emplace_back(std::exchange(buffer, resize(buffer)));
                buffer_.store(buffer, relaxed);
            }

            buffer->store(bottom, std::forward<T>(value));

            // this synchronizes with other acquire fences
            // memory operations about this line cannot be reordered
            std::atomic_thread_fence(release);

            bottom_.store(bottom + 1, relaxed);
        }

        /**
         * @brief Steal from the top of the queue
         *
         * @return std::optional<T>
         */
        std::optional<T> pop_top() {
            auto top = top_.load(acquire);
            // this synchronizes with other release fences
            // memory ops below this line cannot be reordered with ops above this line
            std::atomic_thread_fence(seq_cst);
            auto bottom = bottom_.load(acquire);

            if (top < bottom) {
                // non-empty queue; win the top index before claiming any slot, so exactly one
                // thread drains a given index
                if (!top_.compare_exchange_strong(top, top + 1, seq_cst, relaxed)) {
                    // failed the race
                    return std::nullopt;
                }

                // claim the pointer; a null claim means the owner's resize already claimed this
                // slot and moved the pointer into the resized buffer, which is published at the
                // same absolute index — spin until the publish and claim there
                for (;;) {
                    auto* buffer = buffer_.load(acquire);
                    if (auto* claimed = buffer->claim(top)) {
                        T item = std::move(*claimed);
                        claimed->~T();
                        ::operator delete(static_cast<void*>(claimed));
                        return item;
                    }
                }
            }
            // empty queue
            return std::nullopt;
        }

        std::optional<T> take_bottom() {
            auto bottom = bottom_.load(relaxed) - 1;
            auto* buffer = buffer_.load(relaxed);

            // prevent stealing
            bottom_.store(bottom, relaxed);

            // this synchronizes with other release fences
            // memory ops below this line cannot be reordered
            std::atomic_thread_fence(seq_cst);

            auto top = top_.load(relaxed);
            if (top <= bottom) {
                // queue isn't empty
                if (top == bottom) {
                    // there is only 1 item left in the queue, we need the CAS to succeed
                    // since another thread may be trying to steal and could steal before we're able
                    // to take the bottom
                    if (!top_.compare_exchange_strong(top, top + 1, seq_cst, relaxed)) {
                        // failed race
                        bottom_.store(bottom + 1, relaxed);
                        return std::nullopt;
                    }
                    bottom_.store(bottom + 1, relaxed);
                }
                // the multi-item path intentionally leaves bottom_ at `bottom`: that decrement is
                // how the owner consumes the bottom. The owner is the only thread that touches the
                // bottom index, so the slot still holds the pointer; claim it so the slot is left
                // in a well-defined state
                auto* claimed = buffer->claim(bottom);
                if (!claimed) {
                    return std::nullopt;
                }
                T item = std::move(*claimed);
                claimed->~T();
                ::operator delete(static_cast<void*>(claimed));
                return item;
            } else {
                bottom_.store(bottom + 1, relaxed);
                return std::nullopt;
            }
        }

        /**
         * @brief Remove all items currently in the queue.
         * @details Best-effort, mirroring thread_pool::clear_tasks(): items that a consumer has
         * already popped are not recovered. Popping from the top is a legal consumer operation,
         * so this may be called from any thread. Items are destroyed as they are removed.
         * @return std::size_t The number of items removed.
         */
        std::size_t clear() {
            std::size_t removed{0};
            while (auto item = pop_top()) {
                (void)item;
                ++removed;
            }
            return removed;
        }

      private:
        /**
         * @brief Resize the internal buffer.
         * @details Legal while stealers race: every unclaimed pointer is claimed out of the old
         * buffer with an exchange and stored in the new buffer at the same absolute index, so a
         * stealer draining a given index can never double-own a pointer. A pointer that is already
         * null was claimed before the resize and is gone for good. Claiming the whole old buffer
         * (not just [top, bottom)) is what keeps the invariant that every not-yet-claimed pointer
         * lives at its absolute index in the newest published buffer — the basis a stealer
         * chasing a null claim relies on. A stealer that has won the top index but not yet claimed
         * its slot is still holding that index open, so its pointer must be relocated too.
         *
         * The payload in old slot `i` belongs to the unique absolute index in
         * [bottom - capacity, bottom) whose index & (capacity - 1) is `i`: the push that would
         * have overwritten that slot could not have happened yet, since it is exactly the push
         * this resize was triggered by. Storing at that absolute index (not at slot `i`) is what
         * lines up with the absolute-index claim a stealer performs after the swap.
         * @param old The buffer being replaced
         */
        circular_buffer* resize(circular_buffer* old) {
            auto temp = new circular_buffer(old->capacity() * 2);
            const auto capacity = old->capacity();
            const auto bottom = bottom_.load(relaxed);
            for (std::int64_t i = 0; i < capacity; ++i) {
                if (auto* claimed = old->claim(i)) {
                    const auto index = (bottom - 1) - (((bottom - 1) - i) & (capacity - 1));
                    temp->store(index, T(std::move(*claimed)));
                    claimed->~T();
                    ::operator delete(static_cast<void*>(claimed));
                }
            }
            return temp;
        }
    };
}  // namespace dp
