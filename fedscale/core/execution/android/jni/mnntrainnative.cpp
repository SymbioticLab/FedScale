//
//  MobilenetV2Utils.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include <android/bitmap.h>
#include <jni.h>
#include <string>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "train/source/data/DataLoader.hpp"
#include "train/source/nn/NN.hpp"
#include "train/source/optimizer/SGD.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "train/source/optimizer/ADAM.hpp"
#include "train/source/datasets/ImageDataset.hpp"
#include "train/source/optimizer/LearningRateScheduler.hpp"
#include "train/source/optimizer/Loss.hpp"
#include "train/source/transformer/Transformer.hpp"
#include "cpp/ConvertToFullQuant.hpp"
#include "module/PipelineModule.hpp"
#include "converter/include/cli.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;
using namespace rapidjson;

std::string parseString(JNIEnv* env, jstring str) {
    const char* tmpStr = env->GetStringUTFChars(str, NULL);
    std::string returnStr = tmpStr;
    env->ReleaseStringUTFChars(str, tmpStr);
    return returnStr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_taobao_android_mnn_MNNTrainNative_nativeTrain(
    JNIEnv* env,
    jclass type,
    jint numClasses_,
    jint addToLabel_,
    jstring trainModelPath_,
    jstring newModelPath_,
    jstring newJsonPath_,
    jstring trainImagesFolder_,
    jstring trainImagesTxt_,
    jstring config_
    ) {

    int numClasses = numClasses_;
    int addToLabel = addToLabel_;
    std::string trainModelPath      = parseString(env, trainModelPath_   );
    std::string newModelPath        = parseString(env, newModelPath_   );
    std::string newJsonPath         = parseString(env, newJsonPath_   );
    std::string trainImagesFolder   = parseString(env, trainImagesFolder_);
    std::string trainImagesTxt      = parseString(env, trainImagesTxt_   );
    std::string configStr           = parseString(env, config_);

    Document configJSON;
    configJSON.Parse(configStr.c_str());


    auto varMap = Variable::loadMap(trainModelPath.c_str());
    if (varMap.empty()) {
        MNN_ERROR("Can not load model %s\n", trainModelPath.c_str());
        return 0;
    }
    auto inputOutputs = Variable::getInputAndOutput(varMap);
    auto inputs = Variable::mapToSequence(inputOutputs.first);
    auto outputs = Variable::mapToSequence(inputOutputs.second);
    std::shared_ptr<Module> model(NN::extract(inputs, outputs, true));

    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_USER_1, config, 2);
    std::shared_ptr<SGD> solver(new SGD(model));
    solver->setMomentum(0.9f);
    // solver->setMomentum2(0.99f);
    solver->setWeightDecay(0.00004f);

    auto converImagesToFormat = CV::RGB;
    int resizeHeight = 28;
    int resizeWidth = 28;
    std::vector<float> means = {127.5, 127.5, 127.5};
    std::vector<float> scales = {1 / 127.5, 1 / 127.5, 1 / 127.5};
    std::vector<float> cropFraction = {
        0.875, 0.875};  // center crop fraction for height and width
    bool centerOrRandomCrop = false;  // true for random crop
    std::shared_ptr<ImageDataset::ImageConfig> datasetConfig(
        ImageDataset::ImageConfig::create(converImagesToFormat, resizeHeight,
                                          resizeWidth, scales, means,
                                          cropFraction, centerOrRandomCrop));
    bool readAllImagesToMemory = false;
    auto trainDataset =
        ImageDataset::create(trainImagesFolder, trainImagesTxt,
                             datasetConfig.get(), readAllImagesToMemory);

    const int trainBatchSize = 32;
    const int trainNumWorkers = 4;

    auto trainDataLoader =
        trainDataset.createLoader(trainBatchSize, true, true, trainNumWorkers);

    const int trainIterations = trainDataLoader->iterNumber();

    // belong to config
    float lossDecay = 0.2;
    int trainEpochs = 5;
    float epochTrainLoss = 0;
    float currentLoss = 0;

    // const int usedSize = 1000;
    // const int testIterations = usedSize / testBatchSize;
    for (int epoch = 0; epoch < trainEpochs; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            AUTOTIME;
            trainDataLoader->reset();
            model->setIsTraining(true);
            for (int i = 0; i < trainIterations; i++) {
                AUTOTIME;
                auto trainData = trainDataLoader->next();
                auto example = trainData[0];
                // Compute One-Hot
                auto newTarget = _OneHot(
                    _Cast<int32_t>(_Squeeze(
                        example.second[0] + _Scalar<int32_t>(addToLabel), {})),
                    _Scalar<int>(numClasses), _Scalar<float>(1.0f),
                    _Scalar<float>(0.0f));
                auto predict =
                    model->forward(_Convert(example.first[0], NC4HW4));
                auto loss = _CrossEntropy(predict, newTarget);
                // float rate   = LrScheduler::inv(0.0001,
                // solver->currentStep(), 0.0001, 0.75);
                float rate = 1e-2;
                solver->setLearningRate(rate);
                currentLoss = loss->readMap<float>()[0];

                if (epochTrainLoss == 0) {
                    epochTrainLoss = currentLoss;
                } else {
                    epochTrainLoss = (1 - lossDecay) * epochTrainLoss +
                            lossDecay * currentLoss;
                }

                MNN_PRINT("[train][iter][%d][loss][%f][lr][%f]",
                          solver->currentStep(), currentLoss, rate);
                solver->step(loss);
            }
        }

        {
            model->setIsTraining(false);
            auto forwardInput =
                _Input({1, 3, resizeHeight, resizeWidth}, NC4HW4);
            forwardInput->setName("data");
            auto predict = model->forward(forwardInput);
            Transformer::turnModelToInfer()->onExecute({predict});
            predict->setName("prob");
            std::string fileName = newModelPath;
            Variable::save({predict}, fileName.c_str());
            MNN_PRINT("[save][epoch][%d][path][%s]", epoch, fileName.c_str());
        }

        exe->dumpProfile();
    }

    MNN::Cli::mnn2json(newModelPath.c_str(), newJsonPath.c_str());
    MNN_PRINT("[MNN->JSON][MNN][%s][JSON][%s]", newModelPath.c_str(), newJsonPath.c_str());

    StringBuffer s;
    Writer<StringBuffer> writer(s);
    writer.StartObject();
    writer.Key("clientId");
    Value& tmp = configJSON["client_id"];
    std::string clientId = tmp.GetString();
    writer.String(clientId.c_str());
    writer.Key("moving_loss");
    writer.Double(epochTrainLoss);
    writer.Key("trained_size");
    writer.Uint(trainEpochs * trainDataLoader->size());
    writer.Key("success");
    writer.Bool(true);
    writer.Key("update_weight");
    std::ifstream t(newJsonPath);
    std::stringstream buffer;
    buffer << t.rdbuf();
    writer.String(buffer.str().c_str());
    writer.Key("utility");
    writer.Double(currentLoss * currentLoss * trainDataLoader->size());
    writer.Key("wall_duration");
    writer.Uint(0);
    writer.EndObject();
    return env->NewStringUTF(s.GetString());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_taobao_android_mnn_MNNTrainNative_nativeTest(
    JNIEnv* env,
    jclass type,
    jint numClasses_,
    jint addToLabel_,
    jstring testModelPath_,
    jstring testImagesFolder_,
    jstring testImagesTxt_,
    jstring config_) {
    int numClasses = numClasses_;
    int addToLabel = addToLabel_;
    std::string testModelPath = parseString(env, testModelPath_);
    std::string testImagesFolder = parseString(env, testImagesFolder_);
    std::string testImagesTxt = parseString(env, testImagesTxt_);
    std::string configStr = parseString(env, config_);

    Document configJSON;
    configJSON.Parse(configStr.c_str());

    auto varMap = Variable::loadMap(testModelPath.c_str());
    if (varMap.empty()) {
        MNN_ERROR("Can not load model %s\n", testModelPath.c_str());
        return 0;
    }
    auto inputOutputs = Variable::getInputAndOutput(varMap);
    auto inputs = Variable::mapToSequence(inputOutputs.first);
    auto outputs = Variable::mapToSequence(inputOutputs.second);
    std::shared_ptr<Module> model(NN::extract(inputs, outputs, true));

    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_USER_1, config, 2);

    auto converImagesToFormat = CV::RGB;
    int resizeHeight = 28;
    int resizeWidth = 28;
    std::vector<float> means = {127.5, 127.5, 127.5};
    std::vector<float> scales = {1 / 127.5, 1 / 127.5, 1 / 127.5};
    std::vector<float> cropFraction = {
        0.875, 0.875};  // center crop fraction for height and width
    bool centerOrRandomCrop = false;  // true for random crop
    std::shared_ptr<ImageDataset::ImageConfig> datasetConfig(
        ImageDataset::ImageConfig::create(converImagesToFormat, resizeHeight,
                                          resizeWidth, scales, means,
                                          cropFraction, centerOrRandomCrop));
    bool readAllImagesToMemory = false;
    auto testDataset =
        ImageDataset::create(testImagesFolder, testImagesTxt,
                             datasetConfig.get(), readAllImagesToMemory);

    const int testBatchSize = 10;
    const int testNumWorkers = 0;

    auto testDataLoader =
        testDataset.createLoader(testBatchSize, true, false, testNumWorkers);

    const int testIterations = testDataLoader->iterNumber();

    StringBuffer s;
    Writer<StringBuffer> writer(s);
    writer.StartObject();

    model->clearCache();
    exe->gc(Executor::FULL);
    exe->resetProfile();

    float testLoss = 0;
    int correct = 0;
    int sampleCount = 0;
    testDataLoader->reset();
    model->setIsTraining(false);
    exe->gc(Executor::PART);

    AUTOTIME;
    for (int i = 0; i < testIterations; i++) {
        auto data = testDataLoader->next();
        auto example = data[0];
        auto newTarget =
            _OneHot(_Cast<int32_t>(_Squeeze(
                        example.second[0] + _Scalar<int32_t>(addToLabel), {})),
                    _Scalar<int>(numClasses), _Scalar<float>(1.0f),
                    _Scalar<float>(0.0f));
        auto predict = model->forward(_Convert(example.first[0], NC4HW4));
        auto loss = _CrossEntropy(predict, newTarget);
        testLoss += loss->readMap<float>()[0];
        auto label =
            _Squeeze(example.second[0]) + _Scalar<int32_t>(addToLabel);
        sampleCount += label->getInfo()->size;
        int accu = 0;
        for (int j = 0; j < testBatchSize; ++j) {
            int _label = 0;
            int maxScore = predict->readMap<int32_t>()[j * numClasses];
            for (int k = 0; k < numClasses; ++k) {
                int currentScore =
                    predict->readMap<int32_t>()[j * numClasses + k];
                if (currentScore > maxScore) {
                    maxScore = currentScore;
                    _label = k;
                }
            }
            if (_label == label->readMap<int32_t>()[j]) accu++;
        }
        correct += accu;
        MNN_PRINT("[test][iter][%d][acc][%d/%d=%f%%]", i, correct,
                  sampleCount, float(correct) / sampleCount * 100);
    }
    auto accu = (float)correct / testDataLoader->size();
    MNN_PRINT("[test][final][acc][%f]", accu);


    exe->dumpProfile();
    writer.Key("top_1");
    writer.Double(accu);
    // TODO: add correct top_5, currently equal to top_1
    writer.Key("top_5");
    writer.Double(accu);
    writer.Key("test_loss");
    writer.Double(testLoss);
    writer.Key("test_len");
    writer.Uint(testDataLoader->size());
    writer.EndObject();
    return env->NewStringUTF(s.GetString());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_taobao_android_mnn_MNNTrainNative_nativeMNNConvert(
    JNIEnv* env,
    jclass type, 
    jstring mnnPath_,
    jstring jsonPath_,
    jboolean json2mnn_) {
    std::string mnnPath = parseString(env, mnnPath_);
    std::string jsonPath = parseString(env, jsonPath_);
    bool json2mnn = json2mnn_;
    if (json2mnn) {
        MNN::Cli::json2mnn(jsonPath.c_str(), mnnPath.c_str());
        MNN_PRINT("[JSON->MNN][JSON][%s][MNN][%s]", jsonPath.c_str(), mnnPath.c_str());
    } else {
        MNN::Cli::mnn2json(mnnPath.c_str(), jsonPath.c_str());
        MNN_PRINT("[MNN->JSON][MNN][%s][JSON][%s]", mnnPath.c_str(), jsonPath.c_str());
    }
    return 0;
}
