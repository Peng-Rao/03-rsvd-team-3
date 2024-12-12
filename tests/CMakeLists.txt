# config for unit tests
add_executable(UnitTests test_rsvd.cpp)
target_link_libraries(UnitTests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(UnitTests PRIVATE MPI::MPI_CXX)