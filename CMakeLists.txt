cmake_minimum_required(VERSION 3.0)
project(network-trainer C)

include(FetchContent)

set(RAYLIB_VERSION 5.0)
set(TRAIN_MAIN src/train_main.c)
set(WAV_MAIN src/wav_main.c)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FetchContent_Declare(
    raylib
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    URL https://github.com/raysan5/raylib/archive/refs/tags/${RAYLIB_VERSION}.tar.gz
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(raylib)

FetchContent_Declare(
    portaudio
    GIT_REPOSITORY https://github.com/PortAudio/portaudio.git
    GIT_TAG        master
)
set(PA_USE_PULSEAUDIO ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(portaudio)

include_directories(include/)

file(GLOB PROJECT_HEADERS include/*.h)
file(GLOB PROJECT_COMMON_SOURCES src/*.c)
list(REMOVE_ITEM PROJECT_COMMON_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/train_main.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/wav_main.c"
)

file(GLOB PROJECT_CONFIGS CMakeLists.txt
                            Readme.md
                            .gitattributes
                            .gitignore
                            .gitmodules)

# source groups
source_group("headers" FILES ${PROJECT_HEADERS})
source_group("sources" FILES ${PROJECT_SOURCES})


# trainer
add_executable(${PROJECT_NAME} ${TRAIN_MAIN} ${PROJECT_COMMON_SOURCES}
                                             ${PROJECT_HEADERS}
                                             ${PROJECT_CONFIGS})
# utility 
add_executable(pcm_to_wav ${WAV_MAIN} ${PROJECT_COMMON_SOURCES} 
                                        ${PROJECT_HEADERS} 
                                        ${PROJECT_CONFIGS})


target_link_libraries(${PROJECT_NAME} PUBLIC "m")
target_link_libraries(pcm_to_wav PUBLIC "m")
target_link_libraries(${PROJECT_NAME} PUBLIC raylib)
target_link_libraries(${PROJECT_NAME} PRIVATE portaudio)
#target_link_libraries(${PROJECT_NAME} PUBLIC "user32.dll")
#target_link_libraries(${PROJECT_NAME} PUBLIC "imm32.dll")
#target_link_libraries(${PROJECT_NAME} PUBLIC "gdi32.dll")
#target_link_libraries(${PROJECT_NAME} PUBLIC "opengl32.dll")