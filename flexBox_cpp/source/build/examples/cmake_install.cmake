# Install script for directory: /home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/examples

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin/exampleROF" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin/exampleROF")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin/exampleROF"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin/exampleROF")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin" TYPE EXECUTABLE FILES "/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/examples/exampleROF")
  if(EXISTS "$ENV{DESTDIR}/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin/exampleROF" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin/exampleROF")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin/exampleROF")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/examples/CMakeFiles/exampleROF.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin/exampleSegmentation" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin/exampleSegmentation")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin/exampleSegmentation"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin/exampleSegmentation")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin" TYPE EXECUTABLE FILES "/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/examples/exampleSegmentation")
  if(EXISTS "$ENV{DESTDIR}/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin/exampleSegmentation" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin/exampleSegmentation")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/bin/exampleSegmentation")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/examples/CMakeFiles/exampleSegmentation.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/karpuzfa/CodeFatih/BlueMatImaging/flexBox_cpp/source/build/examples/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
