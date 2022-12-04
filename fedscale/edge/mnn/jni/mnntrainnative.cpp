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
Java_com_fedscale_android_mnn_MNNNative_nativeTrain(
        JNIEnv* env,
        jclass type,
        jstring directory_,
        jstring trainData_,
        jstring modelConf_,
        jstring trainingConf_
) {
    std::string directory           = parseString(env, directory_);
    std::string trainData           = parseString(env, trainData_);
    std::string modelConf           = parseString(env, modelConf_);
    std::string trainingConf        = parseString(env, trainingConf_);

    int addToLabel = 0;

    Document trainDataConfJSON;
    trainDataConfJSON.Parse(trainData.c_str());
    std::string trainImagesFolder   = directory + "/" + trainDataConfJSON["data"].GetString() + "/";
    std::string trainImagesTxt      = directory + "/" + trainDataConfJSON["label"].GetString();

    Document modelConfJSON;
    modelConfJSON.Parse(modelConf.c_str());
    std::string trainJsonPath       = directory + "/" + modelConfJSON["path"].GetString();
    std::string trainModelPath      = directory + "/model.mnn";
    std::string newModelPath        = directory + "/temp_model.mnn";
    std::string newJsonPath         = directory + "/temp_json.mnn";

    Document trainingConfJSON;
    trainingConfJSON.Parse(trainingConf.c_str());
    const std::string clientId  = trainingConfJSON["client_id"].GetString();
    const int numClasses        = trainingConfJSON["num_classes"].GetInt();
    const int trainEpochs       = trainingConfJSON["epoch"].GetInt();
    const int trainBatchSize    = trainingConfJSON["batch_size"].GetInt();
    const int trainNumWorkers   = trainingConfJSON["num_workers"].GetInt();
    const float rate            = trainingConfJSON["learning_rate"].GetFloat();
    const float lossDecay       = trainingConfJSON["loss_decay"].GetFloat();
    const int channel           = trainingConfJSON["channel"].GetInt();

    MNN_PRINT("[JSON->MNN][start][JSON][%s][MNN][%s]", trainJsonPath.c_str(), trainModelPath.c_str());
    MNN::Cli::json2mnn(trainJsonPath.c_str(), trainModelPath.c_str());
    MNN_PRINT("[JSON->MNN][end][JSON][%s][MNN][%s]", trainJsonPath.c_str(), trainModelPath.c_str());

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
    solver->setMomentum(trainingConfJSON["momentum"].GetFloat());
    // solver->setMomentum2(0.99f);
    solver->setWeightDecay(trainingConfJSON["weight_decay"].GetFloat());

    auto converImagesToFormat = CV::RGB;
    int resizeHeight = trainingConfJSON["height"].GetInt();
    int resizeWidth = trainingConfJSON["width"].GetInt();
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

    auto trainDataLoader =
            trainDataset.createLoader(trainBatchSize, true, true, trainNumWorkers);

    const int trainIterations = trainDataLoader->iterNumber();

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
                    _Input({1, channel, resizeHeight, resizeWidth}, NC4HW4);
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
Java_com_fedscale_android_mnn_MNNNative_nativeTest(
        JNIEnv* env,
        jclass type,
        jstring directory_,
        jstring testData_,
        jstring modelConf_,
        jstring testingConf_) {
    std::string directory          = parseString(env, directory_);
    std::string testData           = parseString(env, testData_);
    std::string modelConf          = parseString(env, modelConf_);
    std::string testingConf        = parseString(env, testingConf_);

    Document testDataConfJSON;
    testDataConfJSON.Parse(testData.c_str());
    std::string testImagesFolder    = directory + "/" + testDataConfJSON["data"].GetString() + "/";
    std::string testImagesTxt       = directory + "/" + testDataConfJSON["label"].GetString();

    Document modelConfJSON;
    modelConfJSON.Parse(modelConf.c_str());
    std::string testJsonPath = directory + "/" + modelConfJSON["path"].GetString();
    std::string testModelPath = directory + "/model.mnn";

    int addToLabel = 0;

    Document testingConfJSON;
    testingConfJSON.Parse(testingConf.c_str());
    int numClasses = testingConfJSON["num_classes"].GetInt();

    MNN_PRINT("[JSON->MNN][start][JSON][%s][MNN][%s]", testJsonPath.c_str(), testModelPath.c_str());
    MNN::Cli::json2mnn(testJsonPath.c_str(), testModelPath.c_str());
    MNN_PRINT("[JSON->MNN][end][JSON][%s][MNN][%s]", testJsonPath.c_str(), testModelPath.c_str());

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
    int resizeHeight = testingConfJSON["height"].GetInt();
    int resizeWidth = testingConfJSON["width"].GetInt();
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

    const int testBatchSize = testingConfJSON["test_bsz"].GetInt();
    const int testNumWorkers = testingConfJSON["num_workers"].GetInt();

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
    int correctTop1 = 0;
    int correctTop5 = 0;
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
        auto label = _Squeeze(example.second[0]) + _Scalar<int32_t>(addToLabel);
        sampleCount += label->getInfo()->size;

        auto predictTop1 = _ArgMax(predict, 1); // (N, numClasses) --> (N)
        auto accuracyTop1 = _Cast<int32_t>(_Equal(predictTop1, label).sum({}));
        correctTop1 += accuracyTop1->readMap<int32_t>()[0];

        auto predictTop5 = _TopKV2(predict, _Scalar<int>(5))[1];
        auto accuracyTop5 = _Cast<int32_t>(_Equal(predictTop5, _Unsqueeze(label, {1})).sum({}));
        correctTop5 += accuracyTop5->readMap<int32_t>()[0];

        MNN_PRINT("[test][iter][%d][accuracy][top 1][%d/%d=%f%%][top 5][%d/%d=%f%%]", i,
                  correctTop1, sampleCount, float(correctTop1) / sampleCount * 100,
                  correctTop5, sampleCount, float(correctTop5) / sampleCount * 100);
    }
    auto accuracyFinalTop1 = (float)correctTop1 / testDataLoader->size();
    auto accuracyFinalTop5 = (float)correctTop5 / testDataLoader->size();
    MNN_PRINT("[test][final][accuracy][top 1][%f][top 5][%f]", accuracyFinalTop1, accuracyFinalTop5);

    exe->dumpProfile();
    writer.Key("top_1");
    writer.Double(accuracyFinalTop1);
    writer.Key("top_5");
    writer.Double(accuracyFinalTop5);
    writer.Key("test_loss");
    writer.Double(testLoss);
    writer.Key("test_len");
    writer.Uint(testDataLoader->size());
    writer.EndObject();
    return env->NewStringUTF(s.GetString());
}
