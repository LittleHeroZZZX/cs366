cmake_minimum_required(VERSION 3.10)
project(tokenizer_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# 寻找 Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 寻找 Pybind11
# 如果 pybind11 安装在非标准路径，可能需要设置 pybind11_DIR
find_package(pybind11 REQUIRED)

# 寻找 Boost (Header only libraries)
find_package(Boost REQUIRED)

# 创建 Python 模块
# tokenizer_cpp 是模块名，必须与 PYBIND11_MODULE 中的名字一致
pybind11_add_module(tokenizer_cpp csrc/tokenizer_trainer.cc)

# 链接库
target_link_libraries(tokenizer_cpp PRIVATE Boost::boost)

# 优化选项 (Release 模式下)
# if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
target_compile_options(tokenizer_cpp PRIVATE -O3)
# endif()


# copy compile_commands.json to project root
add_custom_command(TARGET tokenizer_cpp 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/../compile_commands.json
)

# copy so
add_custom_command(TARGET tokenizer_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:tokenizer_cpp>
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tokenizer_cpp${CMAKE_SHARED_LIBRARY_SUFFIX}
)