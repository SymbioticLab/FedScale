package com.taobao.android.mnn;

/**
 * Training and testing interface from Java to JNI C++ backend.
 */
public class MNNTrainInstance {
    /**
     * Training interface from Java to JNI C++ backend.
     *
     * @param numClasses Number of classes.
     * @param trainModelPath Model path before train.
     * @param newModelPath Model path after train.
     * @param newJsonPath JSON model path after train.
     * @param trainImagesFolder Training set directory path.
     * @param trainImagesTxt Training label path.
     * @param config Stringified JSON config for training.
     * @return Stringified JSON training result containing updated model in JSON.
     */
    public String train(
            int numClasses,
            String trainModelPath,
            String newModelPath,
            String newJsonPath,
            String trainImagesFolder,
            String trainImagesTxt,
            String config) {
        return MNNTrainNative.nativeTrain(
                numClasses,
                0,
                trainModelPath,
                newModelPath,
                newJsonPath,
                trainImagesFolder,
                trainImagesTxt,
                config);
    }

    /**
     * Testing interface from Java to JNI C++ backend.
     *
     * @param numClasses Number of classes.
     * @param testModelPath Model path for test.
     * @param testImagesFolder Testing set directory path.
     * @param testImagesTxt Testing label path.
     * @param config Stringified JSON config for testing.
     * @return  Stringified JSON testing result.
     */
    public String test(
            int numClasses,
            String testModelPath,
            String testImagesFolder,
            String testImagesTxt,
            String config) {
        return MNNTrainNative.nativeTest(
                numClasses,
                0,
                testModelPath,
                testImagesFolder,
                testImagesTxt,
                config);
    }

    /**
     * MNN JSON bijective conversion.
     *
     * @param mnnPath MNN Model path.
     * @param jsonPath MNN JSON path.
     * @param json2mnn True if convert JSON to MNN, false otherwise.
     */
    public void convert(String mnnPath, String jsonPath, boolean json2mnn) {
        MNNTrainNative.nativeMNNConvert(mnnPath, jsonPath, json2mnn);
    }
}
