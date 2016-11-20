add_executable(nname nn.cpp)
target_compile_features(nname PRIVATE cxx_decltype_auto cxx_auto_type)
target_link_libraries(nname gflags ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
