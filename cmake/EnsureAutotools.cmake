## EnsureAutotools.cmake
##
## Guarantees that the executables needed for ./autogen.pl exist.
## If they do not, and AUTO_INSTALL_AUTOTOOLS=ON, tries to install with
##   • apt  (Debian/Ubuntu)
##   • dnf  (Fedora/RHEL)
##   • brew (macOS)
## Otherwise it aborts with a fatal error explaining what to install.

include(CheckIncludeFile)

option(AUTO_INSTALL_AUTOTOOLS
        "Attempt to install autoconf/automake/libtool with the system package manager" OFF)

set(_req_tools autoconf automake libtoolize m4 pkg-config)

set(_missing "")
foreach(_tool IN LISTS _req_tools)
    find_program(_path ${_tool})
    if(NOT _path)
        list(APPEND _missing ${_tool})
    endif()
endforeach()

if(_missing STREQUAL "")
    message(STATUS "All required Autotools found: ${_req_tools}")
    return()
endif()

if(NOT AUTO_INSTALL_AUTOTOOLS)
    message(FATAL_ERROR
            "Building OpenMPI from Git requires Autotools, but these executables "
            "are missing: ${_missing}\n"
            "• On Ubuntu/Debian: sudo apt install autoconf automake libtool pkg-config m4\n"
            "• On Fedora/RHEL:   sudo dnf install autoconf automake libtool pkgconf-pkg-config m4\n"
            "• On macOS (brew):  brew install autoconf automake libtool pkg-config m4\n"
            "Rerun CMake afterwards or configure with -DAUTO_INSTALL_AUTOTOOLS=ON "
            "to let the script attempt installation." )
endif()

# ---------------- automatic installation branch -----------------------------
execute_process(COMMAND uname -s OUTPUT_VARIABLE _sys_kernel OUTPUT_STRIP_TRAILING_WHITESPACE)

if(_sys_kernel STREQUAL "Linux")
    # Detect package manager ----------------------------------------------------
    find_program(APT apt-get)
    find_program(DNF dnf)
    if(APT)
        set(_cmd sudo apt-get update && sudo apt-get -y install autoconf automake libtool pkg-config m4)
    elseif(DNF)
        set(_cmd sudo dnf -y install autoconf automake libtool pkgconf-pkg-config m4)
    endif()
elseif(_sys_kernel STREQUAL "Darwin")
    find_program(BREW brew)
    if(BREW)
        set(_cmd brew install autoconf automake libtool pkg-config m4)
    endif()
endif()

if(NOT _cmd)
    message(FATAL_ERROR
            "AUTO_INSTALL_AUTOTOOLS is ON but no supported package manager was found.\n"
            "Please install the tools manually: autoconf automake libtool pkg-config m4")
endif()

message(STATUS "Installing Autotools via: ${_cmd}")
execute_process(COMMAND /bin/sh -c "${_cmd}"
        RESULT_VARIABLE _res)

if(NOT _res EQUAL 0)
    message(FATAL_ERROR "Automatic Autotools installation failed (exit ${_res}).")
endif()

# Rerun detection so the caller picks up the programs from PATH
foreach(_tool IN LISTS _req_tools)
    find_program(_path ${_tool})
    if(NOT _path)
        message(FATAL_ERROR
                "Even after installation, '${_tool}' is still not on PATH.\n"
                "Check your package manager logs or install the tool manually.")
    endif()
endforeach()

message(STATUS "Autotools successfully installed.")
