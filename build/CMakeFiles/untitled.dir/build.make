# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ahmed/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ahmed/src/build

# Include any dependencies generated for this target.
include CMakeFiles/untitled.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/untitled.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/untitled.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/untitled.dir/flags.make

CMakeFiles/untitled.dir/cmake_pch.hxx.gch: CMakeFiles/untitled.dir/flags.make
CMakeFiles/untitled.dir/cmake_pch.hxx.gch: CMakeFiles/untitled.dir/cmake_pch.hxx.cxx
CMakeFiles/untitled.dir/cmake_pch.hxx.gch: CMakeFiles/untitled.dir/cmake_pch.hxx
CMakeFiles/untitled.dir/cmake_pch.hxx.gch: CMakeFiles/untitled.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ahmed/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/untitled.dir/cmake_pch.hxx.gch"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -x c++-header -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -MD -MT CMakeFiles/untitled.dir/cmake_pch.hxx.gch -MF CMakeFiles/untitled.dir/cmake_pch.hxx.gch.d -o CMakeFiles/untitled.dir/cmake_pch.hxx.gch -c /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx.cxx

CMakeFiles/untitled.dir/cmake_pch.hxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/untitled.dir/cmake_pch.hxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -x c++-header -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -E /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx.cxx > CMakeFiles/untitled.dir/cmake_pch.hxx.i

CMakeFiles/untitled.dir/cmake_pch.hxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/untitled.dir/cmake_pch.hxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -x c++-header -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -S /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx.cxx -o CMakeFiles/untitled.dir/cmake_pch.hxx.s

CMakeFiles/untitled.dir/main.cpp.o: CMakeFiles/untitled.dir/flags.make
CMakeFiles/untitled.dir/main.cpp.o: ../main.cpp
CMakeFiles/untitled.dir/main.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx
CMakeFiles/untitled.dir/main.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx.gch
CMakeFiles/untitled.dir/main.cpp.o: CMakeFiles/untitled.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ahmed/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/untitled.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -MD -MT CMakeFiles/untitled.dir/main.cpp.o -MF CMakeFiles/untitled.dir/main.cpp.o.d -o CMakeFiles/untitled.dir/main.cpp.o -c /home/ahmed/src/main.cpp

CMakeFiles/untitled.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/untitled.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -E /home/ahmed/src/main.cpp > CMakeFiles/untitled.dir/main.cpp.i

CMakeFiles/untitled.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/untitled.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -S /home/ahmed/src/main.cpp -o CMakeFiles/untitled.dir/main.cpp.s

CMakeFiles/untitled.dir/Utility.cpp.o: CMakeFiles/untitled.dir/flags.make
CMakeFiles/untitled.dir/Utility.cpp.o: ../Utility.cpp
CMakeFiles/untitled.dir/Utility.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx
CMakeFiles/untitled.dir/Utility.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx.gch
CMakeFiles/untitled.dir/Utility.cpp.o: CMakeFiles/untitled.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ahmed/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/untitled.dir/Utility.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -MD -MT CMakeFiles/untitled.dir/Utility.cpp.o -MF CMakeFiles/untitled.dir/Utility.cpp.o.d -o CMakeFiles/untitled.dir/Utility.cpp.o -c /home/ahmed/src/Utility.cpp

CMakeFiles/untitled.dir/Utility.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/untitled.dir/Utility.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -E /home/ahmed/src/Utility.cpp > CMakeFiles/untitled.dir/Utility.cpp.i

CMakeFiles/untitled.dir/Utility.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/untitled.dir/Utility.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -S /home/ahmed/src/Utility.cpp -o CMakeFiles/untitled.dir/Utility.cpp.s

CMakeFiles/untitled.dir/stdafx.cpp.o: CMakeFiles/untitled.dir/flags.make
CMakeFiles/untitled.dir/stdafx.cpp.o: ../stdafx.cpp
CMakeFiles/untitled.dir/stdafx.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx
CMakeFiles/untitled.dir/stdafx.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx.gch
CMakeFiles/untitled.dir/stdafx.cpp.o: CMakeFiles/untitled.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ahmed/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/untitled.dir/stdafx.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -MD -MT CMakeFiles/untitled.dir/stdafx.cpp.o -MF CMakeFiles/untitled.dir/stdafx.cpp.o.d -o CMakeFiles/untitled.dir/stdafx.cpp.o -c /home/ahmed/src/stdafx.cpp

CMakeFiles/untitled.dir/stdafx.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/untitled.dir/stdafx.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -E /home/ahmed/src/stdafx.cpp > CMakeFiles/untitled.dir/stdafx.cpp.i

CMakeFiles/untitled.dir/stdafx.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/untitled.dir/stdafx.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -S /home/ahmed/src/stdafx.cpp -o CMakeFiles/untitled.dir/stdafx.cpp.s

CMakeFiles/untitled.dir/Timer.cpp.o: CMakeFiles/untitled.dir/flags.make
CMakeFiles/untitled.dir/Timer.cpp.o: ../Timer.cpp
CMakeFiles/untitled.dir/Timer.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx
CMakeFiles/untitled.dir/Timer.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx.gch
CMakeFiles/untitled.dir/Timer.cpp.o: CMakeFiles/untitled.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ahmed/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/untitled.dir/Timer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -MD -MT CMakeFiles/untitled.dir/Timer.cpp.o -MF CMakeFiles/untitled.dir/Timer.cpp.o.d -o CMakeFiles/untitled.dir/Timer.cpp.o -c /home/ahmed/src/Timer.cpp

CMakeFiles/untitled.dir/Timer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/untitled.dir/Timer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -E /home/ahmed/src/Timer.cpp > CMakeFiles/untitled.dir/Timer.cpp.i

CMakeFiles/untitled.dir/Timer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/untitled.dir/Timer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -S /home/ahmed/src/Timer.cpp -o CMakeFiles/untitled.dir/Timer.cpp.s

CMakeFiles/untitled.dir/Tensor.cpp.o: CMakeFiles/untitled.dir/flags.make
CMakeFiles/untitled.dir/Tensor.cpp.o: ../Tensor.cpp
CMakeFiles/untitled.dir/Tensor.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx
CMakeFiles/untitled.dir/Tensor.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx.gch
CMakeFiles/untitled.dir/Tensor.cpp.o: CMakeFiles/untitled.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ahmed/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/untitled.dir/Tensor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -MD -MT CMakeFiles/untitled.dir/Tensor.cpp.o -MF CMakeFiles/untitled.dir/Tensor.cpp.o.d -o CMakeFiles/untitled.dir/Tensor.cpp.o -c /home/ahmed/src/Tensor.cpp

CMakeFiles/untitled.dir/Tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/untitled.dir/Tensor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -E /home/ahmed/src/Tensor.cpp > CMakeFiles/untitled.dir/Tensor.cpp.i

CMakeFiles/untitled.dir/Tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/untitled.dir/Tensor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -S /home/ahmed/src/Tensor.cpp -o CMakeFiles/untitled.dir/Tensor.cpp.s

CMakeFiles/untitled.dir/Operators.cpp.o: CMakeFiles/untitled.dir/flags.make
CMakeFiles/untitled.dir/Operators.cpp.o: ../Operators.cpp
CMakeFiles/untitled.dir/Operators.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx
CMakeFiles/untitled.dir/Operators.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx.gch
CMakeFiles/untitled.dir/Operators.cpp.o: CMakeFiles/untitled.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ahmed/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/untitled.dir/Operators.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -MD -MT CMakeFiles/untitled.dir/Operators.cpp.o -MF CMakeFiles/untitled.dir/Operators.cpp.o.d -o CMakeFiles/untitled.dir/Operators.cpp.o -c /home/ahmed/src/Operators.cpp

CMakeFiles/untitled.dir/Operators.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/untitled.dir/Operators.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -E /home/ahmed/src/Operators.cpp > CMakeFiles/untitled.dir/Operators.cpp.i

CMakeFiles/untitled.dir/Operators.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/untitled.dir/Operators.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -S /home/ahmed/src/Operators.cpp -o CMakeFiles/untitled.dir/Operators.cpp.s

CMakeFiles/untitled.dir/Dataset.cpp.o: CMakeFiles/untitled.dir/flags.make
CMakeFiles/untitled.dir/Dataset.cpp.o: ../Dataset.cpp
CMakeFiles/untitled.dir/Dataset.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx
CMakeFiles/untitled.dir/Dataset.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx.gch
CMakeFiles/untitled.dir/Dataset.cpp.o: CMakeFiles/untitled.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ahmed/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/untitled.dir/Dataset.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -MD -MT CMakeFiles/untitled.dir/Dataset.cpp.o -MF CMakeFiles/untitled.dir/Dataset.cpp.o.d -o CMakeFiles/untitled.dir/Dataset.cpp.o -c /home/ahmed/src/Dataset.cpp

CMakeFiles/untitled.dir/Dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/untitled.dir/Dataset.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -E /home/ahmed/src/Dataset.cpp > CMakeFiles/untitled.dir/Dataset.cpp.i

CMakeFiles/untitled.dir/Dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/untitled.dir/Dataset.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -S /home/ahmed/src/Dataset.cpp -o CMakeFiles/untitled.dir/Dataset.cpp.s

CMakeFiles/untitled.dir/Ann.cpp.o: CMakeFiles/untitled.dir/flags.make
CMakeFiles/untitled.dir/Ann.cpp.o: ../Ann.cpp
CMakeFiles/untitled.dir/Ann.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx
CMakeFiles/untitled.dir/Ann.cpp.o: CMakeFiles/untitled.dir/cmake_pch.hxx.gch
CMakeFiles/untitled.dir/Ann.cpp.o: CMakeFiles/untitled.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ahmed/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/untitled.dir/Ann.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -MD -MT CMakeFiles/untitled.dir/Ann.cpp.o -MF CMakeFiles/untitled.dir/Ann.cpp.o.d -o CMakeFiles/untitled.dir/Ann.cpp.o -c /home/ahmed/src/Ann.cpp

CMakeFiles/untitled.dir/Ann.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/untitled.dir/Ann.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -E /home/ahmed/src/Ann.cpp > CMakeFiles/untitled.dir/Ann.cpp.i

CMakeFiles/untitled.dir/Ann.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/untitled.dir/Ann.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Winvalid-pch -include /home/ahmed/src/build/CMakeFiles/untitled.dir/cmake_pch.hxx -S /home/ahmed/src/Ann.cpp -o CMakeFiles/untitled.dir/Ann.cpp.s

# Object files for target untitled
untitled_OBJECTS = \
"CMakeFiles/untitled.dir/main.cpp.o" \
"CMakeFiles/untitled.dir/Utility.cpp.o" \
"CMakeFiles/untitled.dir/stdafx.cpp.o" \
"CMakeFiles/untitled.dir/Timer.cpp.o" \
"CMakeFiles/untitled.dir/Tensor.cpp.o" \
"CMakeFiles/untitled.dir/Operators.cpp.o" \
"CMakeFiles/untitled.dir/Dataset.cpp.o" \
"CMakeFiles/untitled.dir/Ann.cpp.o"

# External object files for target untitled
untitled_EXTERNAL_OBJECTS =

untitled: CMakeFiles/untitled.dir/cmake_pch.hxx.gch
untitled: CMakeFiles/untitled.dir/main.cpp.o
untitled: CMakeFiles/untitled.dir/Utility.cpp.o
untitled: CMakeFiles/untitled.dir/stdafx.cpp.o
untitled: CMakeFiles/untitled.dir/Timer.cpp.o
untitled: CMakeFiles/untitled.dir/Tensor.cpp.o
untitled: CMakeFiles/untitled.dir/Operators.cpp.o
untitled: CMakeFiles/untitled.dir/Dataset.cpp.o
untitled: CMakeFiles/untitled.dir/Ann.cpp.o
untitled: CMakeFiles/untitled.dir/build.make
untitled: CMakeFiles/untitled.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ahmed/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX executable untitled"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/untitled.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/untitled.dir/build: untitled
.PHONY : CMakeFiles/untitled.dir/build

CMakeFiles/untitled.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/untitled.dir/cmake_clean.cmake
.PHONY : CMakeFiles/untitled.dir/clean

CMakeFiles/untitled.dir/depend:
	cd /home/ahmed/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ahmed/src /home/ahmed/src /home/ahmed/src/build /home/ahmed/src/build /home/ahmed/src/build/CMakeFiles/untitled.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/untitled.dir/depend

