cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(Damsa)

find_package(Geant4 REQUIRED ui_all vis_all)

include(${Geant4_USE_FILE})

execute_process(COMMAND root-config --cflags OUTPUT_VARIABLE ROOT_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND root-config --libs OUTPUT_VARIABLE ROOT_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)

file(GLOB sources ${PROJECT_SOURCE_DIR}/*.cpp)
list(REMOVE_ITEM sources ${PROJECT_SOURCE_DIR}/damsa.cpp)
list(REMOVE_ITEM sources ${PROJECT_SOURCE_DIR}/damsa_gap_scan.cpp)
list(REMOVE_ITEM sources ${PROJECT_SOURCE_DIR}/damsa_opt.cpp)
list(REMOVE_ITEM sources ${PROJECT_SOURCE_DIR}/damsa_alp_inject.cpp)
file(GLOB headers ${PROJECT_SOURCE_DIR}/*.h)

file(GLOB MACRO_FILES
    "*.mac"
)

file(COPY ${MACRO_FILES} DESTINATION ${PROJECT_BINARY_DIR})

add_executable(damsa damsa.cpp ${sources} ${headers})
set_target_properties(damsa PROPERTIES COMPILE_FLAGS "${ROOT_CXX_FLAGS}")
target_link_libraries(damsa ${Geant4_LIBRARIES} ${ROOT_LIBRARIES})

add_executable(damsa_gap_scan damsa_gap_scan.cpp ${sources} ${headers})
set_target_properties(damsa_gap_scan PROPERTIES COMPILE_FLAGS "${ROOT_CXX_FLAGS}")
target_link_libraries(damsa_gap_scan ${Geant4_LIBRARIES} ${ROOT_LIBRARIES})

add_executable(damsa_opt damsa_opt.cpp ${sources} ${headers})
set_target_properties(damsa_opt PROPERTIES COMPILE_FLAGS "${ROOT_CXX_FLAGS}")
target_link_libraries(damsa_opt ${Geant4_LIBRARIES} ${ROOT_LIBRARIES})

add_executable(damsa_alp_inject damsa_alp_inject.cpp ${sources} ${headers})
set_target_properties(damsa_alp_inject PROPERTIES COMPILE_FLAGS "${ROOT_CXX_FLAGS}")
target_link_libraries(damsa_alp_inject ${Geant4_LIBRARIES} ${ROOT_LIBRARIES})

add_custom_target(Damsa DEPENDS damsa damsa_gap_scan damsa_opt damsa_alp_inject)
