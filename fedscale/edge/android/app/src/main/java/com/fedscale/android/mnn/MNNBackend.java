package com.fedscale.android.mnn;

import com.fedscale.android.utils.Backend;

import org.json.JSONObject;

import java.util.Map;

/**
 * Implementation of MNN backend support.
 */
public class MNNBackend implements Backend{
    /**
     * Training interface from Java to JNI C++ backend.
     *
     * @param directory Directory of model and data.
     * @param model Model filename.
     * @param trainingDataConf Training set and label path config.
     * @param trainingConf JSON config for training.
     * @return Stringified JSON training result containing updated model in JSON.
     */
    public Map<String, Object> MLTrain(
            String directory,
            String model,
            JSONObject trainingDataConf,
            JSONObject trainingConf) {
        return MNNNative.nativeTrain(
                directory,
                model,
                trainingDataConf,
                trainingConf);
    }

    /**
     * Testing interface from Java to JNI C++ backend.
     *
     * @param directory Directory of model and data.
     * @param model Model filename.
     * @param testingDataConf Testing set and label path config.
     * @param testingConf JSON config for testing.
     * @return  Stringified JSON testing result.
     */
    public Map<String, Object> MLTest(
            String directory,
            String model,
            JSONObject testingDataConf,
            JSONObject testingConf) {
        return MNNNative.nativeTest(
                directory,
                model,
                testingDataConf,
                testingConf);
    }
}
