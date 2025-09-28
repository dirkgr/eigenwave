# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Create test executables
add_executable(test_tensor test_tensor.cpp)
add_executable(test_slicing_broadcasting test_slicing_broadcasting.cpp)
add_executable(test_broadcast_view test_broadcast_view.cpp)

# Link against gtest and eigenwave
target_link_libraries(test_tensor
    PRIVATE
    eigenwave
    gtest_main
    gtest
)

target_link_libraries(test_slicing_broadcasting
    PRIVATE
    eigenwave
    gtest_main
    gtest
)

target_link_libraries(test_broadcast_view
    PRIVATE
    eigenwave
    gtest_main
    gtest
)

# Add test discovery
include(GoogleTest)
gtest_discover_tests(test_tensor)
gtest_discover_tests(test_slicing_broadcasting)
gtest_discover_tests(test_broadcast_view)