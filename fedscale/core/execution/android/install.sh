#!bin/bash

git clone https://github.com/alibaba/MNN.git
rm -rf MNN/resource/model/*
cp -r assets/* MNN/resource/model/
rm -rf MNN/source/jni
cp -r jni MNN/source
cp demo/build.gradle MNN/project/android/demo
cp demo/app/build.gradle MNN/project/android/demo/app
rm -rf MNN/project/android/demo/gradle/
cp -r demo/gradle MNN/project/android/demo/
rm -rf MNN/project/android/demo/app/src
cp -r demo/app/src MNN/project/android/demo/app/
