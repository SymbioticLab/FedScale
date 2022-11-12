package com.fedscale.android.utils;

/**
 * Training and testing interface from Java to JNI C++ backend.
 */
public interface Backend {
    /**
     * Training interface from Java to JNI C++ backend.
     *
     * @param directory Directory of model and data.
     * @param trainingData Training set and label path config.
     * @param modelConf Model path and config.
     * @param trainingConf Stringified JSON config for training.
     * @return Stringified JSON training result containing updated model in JSON.
     */
    String MLTrain(
            String directory,
            String trainingData,
            String modelConf,
            String trainingConf);

    /**
     * Testing interface from Java to JNI C++ backend.
     *
     * @param directory Directory of model and data.
     * @param testingData Testing set and label path config.
     * @param modelConf Model path and config.
     * @param testingConf Stringified JSON config for testing.
     * @return  Stringified JSON testing result.
     */
    String MLTest(
            String directory,
            String testingData,
            String modelConf,
            String testingConf);
}
