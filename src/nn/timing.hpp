/*
 * Copyright 2015-2016 Nicholas Andrews
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef __NN_TIMING_HPP__
#define __NN_TIMING_HPP__

#include <chrono>
#include <nn/log.hpp>

namespace nn {
    using namespace std::chrono;

    auto now() -> decltype(steady_clock::now()) {
        return steady_clock::now();
    }

    seconds::rep elapsed_seconds(decltype(now()) start_time,
                                 decltype(now()) end_time) {
        return duration_cast<seconds>(end_time - start_time).count();
    }

    milliseconds::rep elapsed_ms(decltype(now()) start_time,
                                 decltype(now()) end_time) {
        return duration_cast<milliseconds>(end_time - start_time).count();
    }

    static std::chrono::steady_clock::time_point global_start_time;

    void tic() { global_start_time = now(); }

    std::chrono::duration<float, std::chrono::steady_clock::period> toc() {
        return now() - global_start_time;
    }

    std::string prettyprint(std::chrono::duration<float, std::chrono::steady_clock::period> t) {
        minutes m = duration_cast<minutes>(t);
        seconds s = duration_cast<seconds>(t - m);
        milliseconds ms = duration_cast<milliseconds>(t - m - s);
        return std::string(std::to_string(m.count())+"m "+std::to_string(s.count())+"s "+std::to_string(ms.count())+"ms");
    }

    class progress_bar {
    public:
        progress_bar(uint64_t ticks, uint64_t refresh_rate)
            : _total_ticks(ticks),
              _ticks_occurred(0),
              _refresh_rate_sec(refresh_rate),
              _begin(std::chrono::steady_clock::now()),
              _last_update(std::chrono::steady_clock::now()),
              _ticks_at_last_update(0) {
        }

        progress_bar operator++(int) {
            using namespace std::chrono;

            ++_ticks_occurred;
            duration t = Clock::now() - _last_update;
            minutes min = duration_cast<minutes>(t);
            seconds sec = duration_cast<seconds>(t - min);

            if (static_cast<uint64_t>(sec.count()) > _refresh_rate_sec) {
                _last_update = Clock::now();
                duration time_taken = Clock::now() - _begin;
                float ticks_per_sec = (float)(_ticks_occurred - _ticks_at_last_update) / (float)duration_cast<seconds>(t).count();
                _ticks_at_last_update = _ticks_occurred;
                float percent_done = (float)_ticks_occurred/_total_ticks;
                duration time_left = time_taken * (1/percent_done - 1);
                minutes minutes_left = duration_cast<minutes>(time_left);
                seconds seconds_left = duration_cast<seconds>(time_left - minutes_left);
                LOG(INFO) << _ticks_occurred << " of " << _total_ticks
                          << " at " << ticks_per_sec
                          << " ticks/sec, time left: " << minutes_left.count()
                          << "m " << seconds_left.count() << "s";
            }

            return *this;
        }
    private:
        typedef std::chrono::steady_clock Clock;
        typedef Clock::time_point time_point;
        typedef Clock::period period;
        typedef std::chrono::duration<float, period> duration;
        std::uint64_t _total_ticks;
        std::uint64_t _ticks_occurred;
        std::uint64_t _refresh_rate_sec;
        time_point _begin;
        time_point _last_update;
        std::uint64_t _ticks_at_last_update;
    };
}

#endif
