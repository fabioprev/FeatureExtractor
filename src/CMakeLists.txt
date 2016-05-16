cmake_minimum_required(VERSION 2.7)

project(FeatureExtractor CXX C)

get_filename_component(PROJECT_ROOT "${PROJECT_SOURCE_DIR}" ABSOLUTE)

find_package(OpenCV REQUIRED)

if (OpenCV_FOUND)
	message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS} (found version \"${OpenCV_VERSION}\")")
endif (OpenCV_FOUND)

# Build type configuration.
if (CMAKE_BUILD_TYPE)
	string(TOUPPER ${CMAKE_BUILD_TYPE} _build_type)
	set(CMAKE_BUILD_TYPE ${_build_type} CACHE STRING "" FORCE)
endif (CMAKE_BUILD_TYPE)

# Build type must be either DEBUG or RELEASE.
if (CMAKE_BUILD_TYPE STREQUAL "")
	message("No build type specified: forcing to RELEASE.")
	set(CMAKE_BUILD_TYPE "RELEASE" CACHE STRING "" FORCE)
endif (CMAKE_BUILD_TYPE STREQUAL "")

# Linux bitness type.
if (LINUX_BITNESS)
	string(TOUPPER ${LINUX_BITNESS} _linux_bitness)
	set(LINUX_BITNESS ${_linux_bitness} CACHE STRING "" FORCE)
else (LINUX_BITNESS)
	set(LINUX_BITNESS "" CACHE STRING "" FORCE)
endif (LINUX_BITNESS)

# Linux bitness type must be either 32-BIT or 64-BIT.
if (LINUX_BITNESS STREQUAL "")
	EXECUTE_PROCESS(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
	
	if (${ARCHITECTURE} STREQUAL "x86_64")
		message("No linux bitness type specified. Detected 64-bit architecture, forcing to 64-bit bitness compilation.")
		set(LINUX_BITNESS "64-BIT" CACHE STRING "" FORCE)
	elseif (${ARCHITECTURE} STREQUAL "i686")
		message("No linux bitness type specified. Detected 32-bit architecture, forcing to 32-bit bitness compilation.")
		set(LINUX_BITNESS "32-BIT" CACHE STRING "" FORCE)
	else (${ARCHITECTURE} STREQUAL "i686")
		message(FATAL_ERROR "Unknown architecture found. The library cannot be compiled under this architecture yet.")
	endif (${ARCHITECTURE} STREQUAL "x86_64")
endif (LINUX_BITNESS STREQUAL "")

add_definitions(-Wuninitialized)
add_definitions(-Wreturn-type)
add_definitions(-Wsign-compare)
add_definitions(-Wuninitialized)
add_definitions(-Wunused-function)
add_definitions(-Wunused-label)
add_definitions(-Wunused-value)
add_definitions(-Wunused-variable)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
	add_definitions(-ggdb)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -frounding-math")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math")
elseif (CMAKE_BUILD_TYPE STREQUAL "RELEASE")
	add_definitions(-O3)
else (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
	message("Invalid build type specified: doing nothing.")
endif (CMAKE_BUILD_TYPE STREQUAL "DEBUG")

if (LINUX_BITNESS STREQUAL "32-BIT")
	add_definitions(-m32)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
elseif (LINUX_BITNESS STREQUAL "64-BIT")
	add_definitions(-m64)
	add_definitions(-fno-elide-constructors)
else (LINUX_BITNESS STREQUAL "32-BIT")
	message(FATAL_ERROR "Invalid linux bitness type specified.")
endif (LINUX_BITNESS STREQUAL "32-BIT")

set(EXECUTABLE_OUTPUT_PATH ../bin)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE FeatureExtractor_src "FeatureExtractors/*.cpp")

# FeatureExtractor executable
add_executable(FeatureExtractor ${FeatureExtractor_src})
target_link_libraries(FeatureExtractor pthread ${OpenCV_LIBS})
