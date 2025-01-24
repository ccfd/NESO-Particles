# No need should be inherited from neso-particles set(CMAKE_CXX_STANDARD 17)

# add_definitions(-DGPU_SELECTOR=0)

set(EXECUTABLE propagateNESO)
set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
# Build the tests individually
add_executable(${EXECUTABLE} ${TEST_MAIN})

target_link_libraries(${EXECUTABLE} PRIVATE NESO-Particles)
target_compile_definitions(${EXECUTABLE} PRIVATE GPU_SELECTOR=0)
target_compile_definitions(${EXECUTABLE}
                           PRIVATE CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX})
target_compile_options(${EXECUTABLE} PRIVATE "-Wpedantic;-Wall;-Wextra")

set_target_properties(
  ${EXECUTABLE}
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF)

# define the test executable as a sycl target
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${TEST_MAIN})

# install the combined test binary
install(TARGETS ${EXECUTABLE} DESTINATION bin)
