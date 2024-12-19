# config for unit tests
add_executable(RSVDTest test_rsvd.cpp)
target_link_libraries(RSVDTest PRIVATE Catch2::Catch2WithMain)
target_link_libraries(RSVDTest PRIVATE Eigen3::Eigen)
target_link_libraries(RSVDTest PRIVATE OpenMP::OpenMP_CXX)

add_executable(GivensRotationQRTest test_givens_rotation_qr.cpp)
target_link_libraries(GivensRotationQRTest PRIVATE Catch2::Catch2WithMain)
target_link_libraries(GivensRotationQRTest PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX)


