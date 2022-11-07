package com.taobao.android.mnn;

import android.util.Log;

import com.taobao.android.utils.Common;

/**
 * Native training and testing interface, connect JNI C++ to JAVA.
 */
public class MNNTrainNative {
    /**
     * Load necessary MNN libraries.
     *
     * @param name Name of the GPU.
     */
    static void loadGpuLibrary(String name) {
        try {
            System.loadLibrary(name);
        } catch (Throwable ce) {
            Log.w(Common.TAG, "load MNN " + name + " GPU so exception=%s", ce);
        }
    }
    static {
        System.loadLibrary("MNN");
        System.loadLibrary("MNNTrain");
        System.loadLibrary("MNNConvertDeps");
        loadGpuLibrary("MNN_Vulkan");
        loadGpuLibrary("MNN_CL");
        loadGpuLibrary("MNN_GL");
        System.loadLibrary("mnncore");
    }

    /**
     * Connect JNI C++ MNN training code to JAVA.
     *
     * @param numClasses Number of classes.
     * @param addToLabel Value to be added to labels.
     * @param trainModelPath Model path before train.
     * @param newModelPath Model path after train.
     * @param newJsonPath JSON model path after train.
     * @param trainImagesFolder Training set directory path.
     * @param trainImagesTxt Training label path.
     * @param config Stringified JSON config for training.
     * @return Stringified JSON training result containing updated model in JSON.
     */
    protected static native String nativeTrain(
            int numClasses,
            int addToLabel,
            String trainModelPath,
            String newModelPath,
            String newJsonPath,
            String trainImagesFolder,
            String trainImagesTxt,
            String config);

    /**
     * Connect JNI C++ MNN testing code to JAVA.
     *
     * @param numClasses Number of classes.
     * @param addToLabel Value to be added to labels.
     * @param testModelPath Model path for test.
     * @param testImagesFolder Testing set directory path.
     * @param testImagesTxt Testing label path.
     * @param config Stringified JSON config for testing.
     * @return  Stringified JSON testing result.
     */
    protected static native String nativeTest(
            int numClasses,
            int addToLabel,
            String testModelPath,
            String testImagesFolder,
            String testImagesTxt,
            String config);

    /**
     * Connect JNI C++ MNN JSON conversion code to JAVA.
     *
     * @param mnnPath MNN Model path.
     * @param jsonPath MNN JSON path.
     * @param json2mnn True if convert JSON to MNN, false otherwise.
     * @return Always return 0.
     */
    protected static native int nativeMNNConvert(
            String mnnPath,
            String jsonPath,
            boolean json2mnn);
}
