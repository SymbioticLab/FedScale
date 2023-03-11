cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 11)
include_directories(src/main/jni/include)
					    							  
add_library(MNN SHARED IMPORTED)
add_library(MNNTrain SHARED IMPORTED)
add_library(MNNExpress SHARED IMPORTED)

set_target_properties(MNN PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libMNN.so)
set_target_properties(MNNTrain PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libMNNTrain.so)
set_target_properties(MNNExpress PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libMNN_Express.so)

add_library(mnncore SHARED src/main/jni/mnntrainnative.cpp)

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

add_definitions(-DMNN_USE_LOGCAT)
target_link_libraries(mnncore MNN MNNTrain MNNExpress ${log-lib} ${jnigraphics-lib})
