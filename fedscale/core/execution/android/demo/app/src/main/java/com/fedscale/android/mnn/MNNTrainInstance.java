package com.fedscale.android.mnn;

/**
 * Training and testing interface from Java to JNI C++ backend.
 */
public class MNNTrainInstance {
    /**
     * Training interface from Java to JNI C++ backend.
     *
     * @param directory Directory of model and data.
     * @param trainingData Training set and label path config.
     * @param modelConf Model path and config.
     * @param trainingConf Stringified JSON config for training.
     * @return Stringified JSON training result containing updated model in JSON.
     */
    public String train(
            String directory,
            String trainingData,
            String modelConf,
            String trainingConf) {
        return MNNNative.nativeTrain(
                directory,
                trainingData,
                modelConf,
                trainingConf);
    }

    /**
     * Testing interface from Java to JNI C++ backend.
     *
     * @param directory Directory of model and data.
     * @param testingData Testing set and label path config.
     * @param modelConf Model path and config.
     * @param testingConf Stringified JSON config for testing.
     * @return  Stringified JSON testing result.
     */
    public String test(
            String directory,
            String testingData,
            String modelConf,
            String testingConf) {
        return MNNNative.nativeTest(
                directory,
                testingData,
                modelConf,
                testingConf);
    }
}
