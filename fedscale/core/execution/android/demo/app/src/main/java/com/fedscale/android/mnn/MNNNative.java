package com.fedscale.android.mnn;

import android.util.Log;

import com.fedscale.android.utils.Common;

/**
 * Native training and testing interface, connect JNI C++ to JAVA.
 */
public class MNNNative {
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
     * @param directory Directory of model and data.
     * @param trainingData Training set and label path config.
     * @param modelConf Model path and config.
     * @param trainingConf Stringified JSON config for training.
     * @return Stringified JSON training result containing updated model in JSON.
     */
    protected static native String nativeTrain(
            String directory,
            String trainingData,
            String modelConf,
            String trainingConf);

    /**
     * Connect JNI C++ MNN testing code to JAVA.
     *
     * @param directory Directory of model and data.
     * @param testingData Testing set and label path config.
     * @param modelConf Model path and config.
     * @param testingConf Stringified JSON config for testing.
     * @return  Stringified JSON testing result.
     */
    protected static native String nativeTest(
            String directory,
            String testingData,
            String modelConf,
            String testingConf);
}
