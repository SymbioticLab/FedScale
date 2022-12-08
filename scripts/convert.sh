#!bin/bash
cd ../../cloud/aggregation/cache

# install and make MNN to build converter if necessary.
if [ "$1" = "--install" ] && [ ! -d "MNN" ]; then
    wget https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-Linux-x86_64.sh -O cmake.sh
    sudo sh cmake.sh --prefix=/usr/local/ --exclude-subdir
    git clone https://github.com/alibaba/MNN.git && cd MNN
    mkdir build && cd build
    cmake .. -DMNN_BUILD_CONVERTER=ON
    make -j4
    cd ../..
fi

# convert model: onnx->mnn->json.
MNN/build/MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --forTraining --bizCode fedscale
MNN/build/MNNDump2Json model.mnn model.json 
