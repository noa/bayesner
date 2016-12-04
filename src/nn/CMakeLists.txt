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

add_executable(mutable_symbol_table_test mutable_symtab_test.cpp)
target_link_libraries(mutable_symbol_table_test gtest gtest_main ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
add_test(mutable_symbol_table_test mutable_symbol_table_test)

add_executable(reader_test reader_test.cpp)
target_link_libraries(reader_test gtest gtest_main ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
add_test(reader_test reader_test)

add_executable(simple_seq_model_test simple_seq_model_test.cpp)
target_link_libraries(simple_seq_model_test gtest gtest_main ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
add_test(simple_seq_model_test simple_seq_model_test)

add_executable(adapted_seq_model_test adapted_seq_model_test.cpp)
target_link_libraries(adapted_seq_model_test gtest gtest_main ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
add_test(adapted_seq_model_test adapted_seq_model_test)

add_executable(hidden_sequence_memoizer_test hidden_sequence_memoizer_test.cpp)
target_link_libraries(hidden_sequence_memoizer_test gtest gtest_main ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
add_test(hidden_sequence_memoizer_test hidden_sequence_memoizer_test)

add_executable(segmental_sequence_memoizer_test segmental_sequence_memoizer_test.cpp)
target_link_libraries(segmental_sequence_memoizer_test gtest gtest_main ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
add_test(segmental_sequence_memoizer_test segmental_sequence_memoizer_test)
