#!bin/bash

git clone https://github.com/alibaba/MNN.git
rm -rf MNN/resource/model/*
cp -r assets/* MNN/resource/model/
rm -rf MNN/source/jni
cp -r jni MNN/source
cp build.gradle MNN/project/android/demo
cp app/build.gradle MNN/project/android/demo/app
rm -rf MNN/project/android/demo/gradle/
cp -r gradle MNN/project/android/demo/
rm -rf MNN/project/android/demo/app/src
cp -r app/src MNN/project/android/demo/app/
