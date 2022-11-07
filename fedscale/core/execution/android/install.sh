#!bin/bash

git clone https://github.com/alibaba/MNN.git
rm -rf MNN/resource/model/MobileNet/
rm -rf MNN/resource/model/Portrait/
rm -rf MNN/resource/model/SqueezeNet/
mv TrainTest MNN/resource/model/
rm -rf MNN/source/jni
mv jni MNN/source
mv demo/build.gradle MNN/project/android/demo
mv demo/app/build.gradle MNN/project/android/demo/app
rm -rf MNN/project/android/demo/gradle/
mv demo/gradle MNN/project/android/demo/
rm -rf MNN/project/android/demo/app/src
mv demo/app/src MNN/project/android/demo/app/
rm -rf demo