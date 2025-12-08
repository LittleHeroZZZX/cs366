cmake_minimum_required(VERSION 3.10)
project(tokenizer_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

find_package(pybind11 REQUIRED)

# 寻找 Boost (Header only libraries)
# 替换原来的 find_package(Boost REQUIRED)
find_package(Boost QUIET)

if(NOT Boost_FOUND)
    message(FATAL_ERROR "\n"
        "  \n"
        "  ❌ 错误：编译中止，未在系统中找到 Boost 库！\n"
        "  -------------------------------------------------------\n"
        "  本项目依赖 Boost C++ 库，请根据您的操作系统运行以下命令进行安装：\n"
        "  \n"
        "  👉 [Ubuntu / Debian]:\n"
        "       sudo apt-get update && sudo apt-get install libboost-all-dev\n"
        "  \n"
        "  👉 [CentOS / RHEL / Fedora]:\n"
        "       sudo yum install boost-devel\n"
        "  \n"
        "  👉 [macOS (Homebrew)]:\n"
        "       brew install boost\n"
        "  -------------------------------------------------------\n"
    )
endif()

# 创建 Python 模块
# tokenizer_cpp 是模块名，必须与 PYBIND11_MODULE 中的名字一致
pybind11_add_module(tokenizer_cpp csrc/tokenizer_trainer.cc)

# 链接库
target_link_libraries(tokenizer_cpp PRIVATE Boost::boost)

# 优化选项 (Release 模式下)
if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
target_compile_options(tokenizer_cpp PRIVATE -O3)
endif()

install(TARGETS tokenizer_cpp DESTINATION .)


# copy compile_commands.json to project root
add_custom_command(TARGET tokenizer_cpp 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
)
