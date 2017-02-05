cmake_minimum_required(VERSION 3.5)

# Project name
project(nicknames)

# Project version
set (nn_VERSION_MAJOR 1)
set (nn_VERSION_MINOR 0)

# CXX Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile flags
add_definitions(-Wfatal-errors)

# Build type
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
       "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
       FORCE )
endif()

# Custom modules
set (CMAKE_MODULE_PATH "${parser_SOURCE_DIR}/cmake")

# Thread support
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package( Threads REQUIRED )
find_package( OpenMP  REQUIRED )

IF(Threads_FOUND)
  INCLUDE_DIRECTORIES(${Threads_INCLUDE_DIR})
  MESSAGE("Are we using pthreads? ${CMAKE_USE_PTHREADS_INIT}")
ENDIF()

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.58.0 REQUIRED COMPONENTS
  system
  thread
  filesystem
  date_time
  iostreams
  log
  log_setup
  serialization
  timer)

# Set boost link targets
set(Boost_TARGETS
  "Boost::boost"
  "Boost::timer"
  "Boost::iostreams"
  "Boost::serialization"
  "Boost::date_time"
  "Boost::system"
  "Boost::log"
  "Boost::log_setup"
  "Boost::thread")

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_subdirectory(gflags)
add_subdirectory(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

enable_testing()

# Set include and link directories
include_directories(
  ${CMAKE_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_DIR}
  )

# Set subdirectories
include_directories(src)
add_subdirectory(src)
