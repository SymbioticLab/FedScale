package com.fedscale.android.utils;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Map;

/**
 * Training and testing interface from Java to JNI C++ backend.
 */
public interface Backend {
    /**
     * Training interface from Java to TFLite backend.
     *
     * @param directory Directory of model and data.
     * @param model Model filename.
     * @param trainingDataConf Training set and label path config.
     * @param trainingConf JSON config for training.
     * @return Training result containing updated model in JSON.
     */
    Map<String, Object> MLTrain(
            String directory,
            String model,
            JSONObject trainingDataConf,
            JSONObject trainingConf) throws JSONException, IOException;

    /**
     * Testing interface from Java to TFLite backend.
     *
     * @param directory Directory of model and data.
     * @param model Model filename.
     * @param testingDataConf Testing set and label path config.
     * @param testingConf JSON config for testing.
     * @return Testing result.
     */
    Map<String, Object> MLTest(
            String directory,
            String model,
            JSONObject testingDataConf,
            JSONObject testingConf) throws JSONException, IOException;
}
