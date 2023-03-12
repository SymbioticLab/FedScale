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

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

std::string parseString(JNIEnv* env, jstring str) {
    const char* tmpStr = env->GetStringUTFChars(str, nullptr);
    std::string returnStr = tmpStr;
    env->ReleaseStringUTFChars(str, tmpStr);
    return returnStr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_fedscale_android_mnn_MNNNative_nativeTrain(
        JNIEnv* env,
        jclass type,
        jstring directory_,
        jstring model_,
        jobject trainingDataConf_,
        jobject trainingConf_
) {
    std::string directory           = parseString(env, directory_);
    std::string modelFilename       = parseString(env, model_);

    int addToLabel = 0;

    jclass jsonObjectClass = env->GetObjectClass(trainingDataConf_);
    jmethodID getString = env->GetMethodID(jsonObjectClass, "getString","(Ljava/lang/String;)Ljava/lang/String;");
    jmethodID getInt    = env->GetMethodID(jsonObjectClass, "getInt","(Ljava/lang/String;)I");
    jmethodID getDouble  = env->GetMethodID(jsonObjectClass, "getDouble","(Ljava/lang/String;)D");
    jmethodID getBoolean  = env->GetMethodID(jsonObjectClass, "getBoolean", "(Ljava/lang/String;)Z");

    auto data = static_cast<jstring>(env->CallObjectMethod(trainingDataConf_, getString, env->NewStringUTF("data")));
    auto label = static_cast<jstring>(env->CallObjectMethod(trainingDataConf_, getString, env->NewStringUTF("label")));
    std::string trainImagesFolder   = directory + "/" + parseString(env, data) + "/";
    std::string trainImagesTxt      = directory + "/" + parseString(env, label);

    auto clientId = static_cast<jstring>(env->CallObjectMethod(trainingConf_, getString, env->NewStringUTF("client_id")));
    const int numClasses        = env->CallIntMethod(trainingConf_, getInt, env->NewStringUTF("num_classes"));
    const int trainEpochs       = env->CallIntMethod(trainingConf_, getInt, env->NewStringUTF("epoch"));
    const int trainBatchSize    = env->CallIntMethod(trainingConf_, getInt, env->NewStringUTF("batch_size"));
    const int trainNumWorkers   = env->CallIntMethod(trainingConf_, getInt, env->NewStringUTF("num_workers"));
    const int channel           = env->CallIntMethod(trainingConf_, getInt, env->NewStringUTF("channel"));
    const int height            = env->CallIntMethod(trainingConf_, getInt, env->NewStringUTF("height"));
    const int width             = env->CallIntMethod(trainingConf_, getInt, env->NewStringUTF("width"));
    const float rate            = env->CallDoubleMethod(trainingConf_, getDouble, env->NewStringUTF("learning_rate"));
    const float lossDecay       = env->CallDoubleMethod(trainingConf_, getDouble, env->NewStringUTF("loss_decay"));
    const float momentum        = env->CallDoubleMethod(trainingConf_, getDouble, env->NewStringUTF("momentum"));
    const float weight_decay    = env->CallDoubleMethod(trainingConf_, getDouble, env->NewStringUTF("weight_decay"));
    const bool fineTune         = env->CallBooleanMethod(trainingConf_, getBoolean, env->NewStringUTF("fine_tune"));

    std::string trainJsonPath       = directory + "/" + modelFilename;
    std::string trainModelPath      = directory + "/model.mnn";
    std::string newModelPath        = directory + "/temp_model.mnn";
    if (fineTune) {
        std::string newModelPath    = directory + "/model.mnn";
    }

    auto varMap = Variable::loadMap(trainModelPath.c_str());
    if (varMap.empty()) {
        MNN_ERROR("Can not load model %s\n", trainModelPath.c_str());
        return nullptr;
    }
    auto inputOutputs = Variable::getInputAndOutput(varMap);
    auto inputs = Variable::mapToSequence(inputOutputs.first);
    auto outputs = Variable::mapToSequence(inputOutputs.second);
    std::shared_ptr<Module> model(NN::extract(inputs, outputs, true));

    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_USER_1, config, 2);
    std::shared_ptr<SGD> solver(new SGD(model));
    solver->setMomentum(momentum);
    // solver->setMomentum2(0.99f);
    solver->setWeightDecay(weight_decay);

    auto convertImagesToFormat = CV::RGB;
    std::vector<float> means = {127.5, 127.5, 127.5};
    std::vector<float> scales = {1 / 127.5, 1 / 127.5, 1 / 127.5};
    std::vector<float> cropFraction = {
            0.875, 0.875};  // center crop fraction for height and width
    bool centerOrRandomCrop = false;  // true for random crop
    std::shared_ptr<ImageDataset::ImageConfig> datasetConfig(
            ImageDataset::ImageConfig::create(convertImagesToFormat, height,
                                              width, scales, means,
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
                    _Input({1, channel, height, width}, NC4HW4);
            forwardInput->setName("data");
            auto predict = model->forward(forwardInput);
            Transformer::turnModelToInfer()->onExecute({predict});
            predict->setName("prob");
            Variable::save({predict}, newModelPath.c_str());
            MNN_PRINT("[save][epoch][%d][path][%s]", epoch, newModelPath.c_str());
        }

        exe->dumpProfile();
    }

    std::ifstream t(newModelPath);
    std::stringstream buffer;
    buffer << t.rdbuf();

    jclass hashMapClass = env->FindClass("java/util/HashMap");
    jmethodID hashMapInit = env->GetMethodID(hashMapClass, "<init>", "()V");
    jobject results = env->NewObject(hashMapClass, hashMapInit);
    jmethodID put = env->GetMethodID(hashMapClass, "put","(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jclass jInteger = env->FindClass("java/lang/Integer");
    jmethodID jIntegerValueOf = env->GetStaticMethodID(jInteger, "valueOf", "(I)Ljava/lang/Integer;");
    jclass jFloat = env->FindClass("java/lang/Float");
    jmethodID jFloatValueOf = env->GetStaticMethodID(jFloat, "valueOf", "(F)Ljava/lang/Float;");
    jclass jBoolean = env->FindClass("java/lang/Boolean");
    jmethodID jBooleanValueOf = env->GetStaticMethodID(jBoolean, "valueOf", "(Z)Ljava/lang/Boolean;");
    env->CallObjectMethod(results, put, env->NewStringUTF("client_id"), clientId);
    env->CallObjectMethod(results, put, env->NewStringUTF("moving_loss"), env->CallStaticObjectMethod(jFloat, jFloatValueOf, epochTrainLoss));
    env->CallObjectMethod(results, put, env->NewStringUTF("trained_size"), env->CallStaticObjectMethod(jInteger, jIntegerValueOf, trainEpochs * (int)trainDataLoader->size()));
    env->CallObjectMethod(results, put, env->NewStringUTF("success"), env->CallStaticObjectMethod(jBoolean, jBooleanValueOf, true));
    env->CallObjectMethod(results, put, env->NewStringUTF("update_weight"), env->NewStringUTF(buffer.str().c_str()));
    env->CallObjectMethod(results, put, env->NewStringUTF("utility"), env->CallStaticObjectMethod(jFloat, jFloatValueOf, currentLoss * currentLoss * trainDataLoader->size()));
    env->CallObjectMethod(results, put, env->NewStringUTF("wall_duration"), env->CallStaticObjectMethod(jInteger, jIntegerValueOf, 0));
    return results;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_fedscale_android_mnn_MNNNative_nativeTest(
        JNIEnv* env,
        jclass type,
        jstring directory_,
        jstring model_,
        jobject testingDataConf_,
        jobject testingConf_) {
    std::string directory          = parseString(env, directory_);
    std::string modelFilename      = parseString(env, model_);

    jclass jsonObjectClass = env->GetObjectClass(testingDataConf_);
    jmethodID getString     = env->GetMethodID(jsonObjectClass, "getString","(Ljava/lang/String;)Ljava/lang/String;");
    jmethodID getInt        = env->GetMethodID(jsonObjectClass, "getInt","(Ljava/lang/String;)I");

    auto datasetFolder              = static_cast<jstring>(env->CallObjectMethod(testingDataConf_, getString, env->NewStringUTF("data")));
    auto labelFile                  = static_cast<jstring>(env->CallObjectMethod(testingDataConf_, getString, env->NewStringUTF("label")));
    std::string testImagesFolder    = directory + "/" + parseString(env, datasetFolder) + "/";
    std::string testImagesTxt       = directory + "/" + parseString(env, labelFile);

    std::string testJsonPath = directory + "/" + modelFilename;
    std::string testModelPath = directory + "/model.mnn";

    int addToLabel = 0;

    const int numClasses        = env->CallIntMethod(testingConf_, getInt, env->NewStringUTF("num_classes"));
    const int resizeHeight      = env->CallIntMethod(testingConf_, getInt, env->NewStringUTF("height"));
    const int resizeWidth       = env->CallIntMethod(testingConf_, getInt, env->NewStringUTF("width"));
    const int testBatchSize     = env->CallIntMethod(testingConf_, getInt, env->NewStringUTF("test_bsz"));
    const int testNumWorkers    = env->CallIntMethod(testingConf_, getInt, env->NewStringUTF("num_workers"));

    auto varMap = Variable::loadMap(testModelPath.c_str());
    if (varMap.empty()) {
        MNN_ERROR("Can not load model %s\n", testModelPath.c_str());
        return nullptr;
    }
    auto inputOutputs = Variable::getInputAndOutput(varMap);
    auto inputs = Variable::mapToSequence(inputOutputs.first);
    auto outputs = Variable::mapToSequence(inputOutputs.second);
    std::shared_ptr<Module> model(NN::extract(inputs, outputs, true));

    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_USER_1, config, 2);

    auto convertImagesToFormat = CV::RGB;
    std::vector<float> means = {127.5, 127.5, 127.5};
    std::vector<float> scales = {1 / 127.5, 1 / 127.5, 1 / 127.5};
    std::vector<float> cropFraction = {
            0.875, 0.875};  // center crop fraction for height and width
    bool centerOrRandomCrop = false;  // true for random crop
    std::shared_ptr<ImageDataset::ImageConfig> datasetConfig(
            ImageDataset::ImageConfig::create(convertImagesToFormat, resizeHeight,
                                              resizeWidth, scales, means,
                                              cropFraction, centerOrRandomCrop));
    bool readAllImagesToMemory = false;
    auto testDataset =
            ImageDataset::create(testImagesFolder, testImagesTxt,
                                 datasetConfig.get(), readAllImagesToMemory);

    auto testDataLoader =
            testDataset.createLoader(testBatchSize, true, false, testNumWorkers);

    const int testIterations = testDataLoader->iterNumber();

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

    jclass hashMapClass = env->FindClass("java/util/HashMap");
    jmethodID hashMapInit = env->GetMethodID(hashMapClass, "<init>", "()V");
    jobject results = env->NewObject(hashMapClass, hashMapInit);
    jmethodID put = env->GetMethodID(hashMapClass, "put","(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jclass jInteger = env->FindClass("java/lang/Integer");
    jmethodID jIntegerValueOf = env->GetStaticMethodID(jInteger, "valueOf", "(I)Ljava/lang/Integer;");
    jclass jFloat = env->FindClass("java/lang/Float");
    jmethodID jFloatValueOf = env->GetStaticMethodID(jFloat, "valueOf", "(F)Ljava/lang/Float;");
    env->CallObjectMethod(results, put, env->NewStringUTF("top_1"), env->CallStaticObjectMethod(jFloat, jFloatValueOf, accuracyFinalTop1));
    env->CallObjectMethod(results, put, env->NewStringUTF("top_5"), env->CallStaticObjectMethod(jFloat, jFloatValueOf, accuracyFinalTop5));
    env->CallObjectMethod(results, put, env->NewStringUTF("test_loss"), env->CallStaticObjectMethod(jFloat, jFloatValueOf, testLoss));
    env->CallObjectMethod(results, put, env->NewStringUTF("test_len"), env->CallStaticObjectMethod(jInteger, jIntegerValueOf, (int)testDataLoader->size()));
    return results;
}
