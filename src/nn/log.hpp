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

#ifndef __NN_LOG_HPP__
#define __NN_LOG_HPP__

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

namespace logging = boost::log;

#define INFO info
#define FATAL fatal

// #define LOG(logger)  std::cout
// #define DLOG(logger) std::cout
// #define VLOG(level)  std::cout
// #define CHECK(cond)  std::cout

#define LOG(logger) \
  BOOST_LOG_TRIVIAL(logger) << "(" << __FILE__ << ", " << __LINE__ << ") "

#define DLOG(logger) \
  BOOST_LOG_TRIVIAL(logger) << "(" << __FILE__ << ", " << __LINE__ << ") "

#define VLOG(level) \
  BOOST_LOG_TRIVIAL(info) << "(" << __FILE__ << ", " << __LINE__ << ") "

#define CHECK(cond) \
  if (!(cond)) BOOST_LOG_TRIVIAL(fatal) << "(" << __FILE__ << ", " << __LINE__ << ") "

#endif
