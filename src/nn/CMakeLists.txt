################################
# Unit Tests
################################

add_executable(fixed_depth_hpyp_test fixed_depth_hpyp_test.cpp)
target_link_libraries(fixed_depth_hpyp_test gtest gtest_main ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
add_test(fixed_depth_hpyp_test fixed_depth_hpyp_test)

add_executable(uniform_test uniform_test.cpp)
target_link_libraries(uniform_test gtest gtest_main ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
add_test(uniform_test uniform_test)

add_executable(node_test node_test.cpp)
target_link_libraries(node_test gtest gtest_main ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
add_test(node_test node_test)
