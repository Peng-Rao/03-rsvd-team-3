# config for unit tests
add_executable(RSVDTest test_rsvd.cpp)
target_link_libraries(RSVDTest PRIVATE Catch2::Catch2WithMain)

add_executable(GivensRotationQRTest test_givens_rotation_qr.cpp)
target_link_libraries(GivensRotationQRTest PRIVATE Catch2::Catch2WithMain)

