cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(Damsa)

find_package(Geant4 REQUIRED ui_all vis_all multithread)

include(${Geant4_USE_FILE})

execute_process(COMMAND root-config --cflags OUTPUT_VARIABLE ROOT_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND root-config --libs OUTPUT_VARIABLE ROOT_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)

include_directories(
    ${PROJECT_SOURCE_DIR}/src/detector
    ${PROJECT_SOURCE_DIR}/src/field
    ${PROJECT_SOURCE_DIR}/src/generators
    ${PROJECT_SOURCE_DIR}/src/physics
    ${PROJECT_SOURCE_DIR}/src/actions
    ${PROJECT_SOURCE_DIR}/src/data
    ${PROJECT_SOURCE_DIR}/src/config
)

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*/*.cpp)
list(REMOVE_ITEM sources ${PROJECT_SOURCE_DIR}/damsa.cpp)
list(REMOVE_ITEM sources ${PROJECT_SOURCE_DIR}/damsa_opt.cpp)
list(REMOVE_ITEM sources ${PROJECT_SOURCE_DIR}/damsa_alp_inject.cpp)
file(GLOB headers ${PROJECT_SOURCE_DIR}/src/*/*.h)

file(GLOB MACRO_FILES
    "macros/*.mac"
)

file(COPY ${MACRO_FILES} DESTINATION ${PROJECT_BINARY_DIR})

add_executable(damsa damsa.cpp ${sources} ${headers})
set_target_properties(damsa PROPERTIES COMPILE_FLAGS "${ROOT_CXX_FLAGS}")
target_link_libraries(damsa ${Geant4_LIBRARIES} ${ROOT_LIBRARIES})

add_executable(damsa_opt damsa_opt.cpp ${sources} ${headers})
set_target_properties(damsa_opt PROPERTIES COMPILE_FLAGS "${ROOT_CXX_FLAGS}")
target_link_libraries(damsa_opt ${Geant4_LIBRARIES} ${ROOT_LIBRARIES})

add_executable(damsa_alp_inject damsa_alp_inject.cpp ${sources} ${headers})
set_target_properties(damsa_alp_inject PROPERTIES COMPILE_FLAGS "${ROOT_CXX_FLAGS}")
target_link_libraries(damsa_alp_inject ${Geant4_LIBRARIES} ${ROOT_LIBRARIES})

# ── Standalone tools (ROOT only, no Geant4) ──────────────────────
add_executable(analyze_gap_scan analyze_gap_scan.cpp)
set_target_properties(analyze_gap_scan PROPERTIES COMPILE_FLAGS "${ROOT_CXX_FLAGS}")
target_link_libraries(analyze_gap_scan ${ROOT_LIBRARIES})

add_executable(flux_converter scripts/pipeline/flux_converter.cpp)

add_executable(subsample_alp_csv scripts/analysis/subsample_alp_csv.cpp)

add_executable(snr_separability scripts/analysis/snr_separability.cpp)
set_target_properties(snr_separability PROPERTIES COMPILE_FLAGS "${ROOT_CXX_FLAGS}")
target_link_libraries(snr_separability ${ROOT_LIBRARIES})

add_custom_target(Damsa DEPENDS damsa damsa_opt damsa_alp_inject
                         analyze_gap_scan flux_converter subsample_alp_csv
                         snr_separability)
