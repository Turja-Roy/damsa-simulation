cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(Damsa LANGUAGES CXX)

# C++20 explicitly. Previously this arrived implicitly as a raw flag from
# `root-config --cflags`, which meant targets without COMPILE_FLAGS (flux_converter,
# subsample_alp_csv) silently built at the compiler default instead.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Geant4 REQUIRED ui_all vis_all multithread)
include(${Geant4_USE_FILE})

# Scoped ROOT components instead of shelling out to `root-config --libs`, which
# linked all of ROOT into every target. ROOTNTuple/ROOTDataFrame are the storage layer.
find_package(ROOT REQUIRED COMPONENTS
    Core RIO Hist Graf Graf3d Gpad Tree MathCore Physics ROOTNTuple ROOTDataFrame)

set(DAMSA_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/src/detector
    ${PROJECT_SOURCE_DIR}/src/generators
    ${PROJECT_SOURCE_DIR}/src/physics
    ${PROJECT_SOURCE_DIR}/src/actions
    ${PROJECT_SOURCE_DIR}/src/data
    ${PROJECT_SOURCE_DIR}/src/config
    ${PROJECT_SOURCE_DIR}/src/alp
)
include_directories(${DAMSA_INCLUDE_DIRS})

# NB: one-level glob — a new src/<area>/<file>.cpp needs a cmake re-run to be picked up.
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*/*.cpp)
file(GLOB headers ${PROJECT_SOURCE_DIR}/src/*/*.h)

file(GLOB MACRO_FILES "macros/*.mac")
file(COPY ${MACRO_FILES} DESTINATION ${PROJECT_BINARY_DIR})

# ── Core: built once, shared by the three Geant4 executables ─────────────────
# Previously each executable compiled the full source set independently (3x cost).
add_library(damsa_core STATIC ${sources} ${headers})
target_include_directories(damsa_core PUBLIC ${DAMSA_INCLUDE_DIRS})
target_link_libraries(damsa_core PUBLIC ${Geant4_LIBRARIES} ROOT::Core ROOT::RIO
    ROOT::Hist ROOT::Graf ROOT::Graf3d ROOT::Gpad ROOT::Tree ROOT::MathCore
    ROOT::Physics ROOT::ROOTNTuple ROOT::ROOTDataFrame)

# ── Geant4 executables (mains live at repo root, not in the src/ glob) ───────
foreach(app damsa damsa_opt damsa_alp_inject)
    add_executable(${app} ${app}.cpp)
    target_link_libraries(${app} PRIVATE damsa_core)
endforeach()

# ── Standalone tools (ROOT only, no Geant4) ─────────────────────────────────
add_executable(analyze_gap_scan analyze_gap_scan.cpp)
target_link_libraries(analyze_gap_scan PRIVATE ROOT::Core ROOT::Hist ROOT::Graf ROOT::Gpad)

add_executable(snr_separability scripts/analysis/snr_separability.cpp)
target_link_libraries(snr_separability PRIVATE ROOT::Core ROOT::Hist ROOT::Graf ROOT::Gpad)

# damsa_io.h is deliberately Geant4-free so these can read the same RNTuple files.
add_executable(flux_converter scripts/pipeline/flux_converter.cpp)
target_link_libraries(flux_converter PRIVATE ROOT::Core ROOT::RIO ROOT::ROOTNTuple ROOT::ROOTDataFrame)

add_executable(subsample_alp_csv scripts/analysis/subsample_alp_csv.cpp)
target_link_libraries(subsample_alp_csv PRIVATE ROOT::Core ROOT::RIO ROOT::ROOTNTuple ROOT::ROOTDataFrame)

add_executable(damsa_convert tools/damsa_convert.cpp)
target_link_libraries(damsa_convert PRIVATE ROOT::Core ROOT::RIO ROOT::ROOTNTuple)

add_executable(alp_xcheck tools/alp_xcheck.cpp)
target_include_directories(alp_xcheck PRIVATE ${PROJECT_SOURCE_DIR}/src/alp)

# ── Self-check (no framework; run ./build/damsa_io_test) ────────────────────
add_executable(damsa_io_test tests/test_damsa_io.cpp)
target_link_libraries(damsa_io_test PRIVATE ROOT::Core ROOT::RIO ROOT::ROOTNTuple)

add_executable(alp_test tests/test_alp.cpp)
target_include_directories(alp_test PRIVATE ${PROJECT_SOURCE_DIR}/src/alp)

add_custom_target(Damsa DEPENDS damsa damsa_opt damsa_alp_inject
                         analyze_gap_scan flux_converter subsample_alp_csv
                         snr_separability damsa_io_test damsa_convert alp_xcheck alp_test)
