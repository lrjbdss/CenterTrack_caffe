cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

# set(CMAKE_PREFIX_PATH /home/lx/programs/libtorch/share/cmake/Torch)
# find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories("./include" 
"/home/lx/work/hi3519/vehicle_detect/caffe_ssd_head_tail/caffe_ssd/include"
"/usr/local/cuda-10.0/include")

aux_source_directory(./src SRC)
add_executable(example-app ${SRC})
target_link_libraries(example-app "${TORCH_LIBRARIES}" "${OpenCV_LIBS}"
"/home/lx/work/hi3519/vehicle_detect/caffe_ssd_head_tail/caffe_ssd/build/lib/libcaffe.so"
"/usr/lib/x86_64-linux-gnu/libglog.so"
"/usr/lib/x86_64-linux-gnu/libboost_system.so"
"/usr/local/cuda/lib64/libcudart.so"
"/usr/local/cuda-10.0/lib64/libcublas.so")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)