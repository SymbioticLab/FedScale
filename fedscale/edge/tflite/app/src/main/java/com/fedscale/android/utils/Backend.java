package com.fedscale.android.utils;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Training and testing interface from Java to JNI C++ backend.
 */
public interface Backend {
    /**
     * Training interface from Java to JNI C++ backend.
     *
     * @param directory Directory of model and data.
     * @param model Model.
     * @param trainingDataConf Training set and label path config.
     * @param trainingConf Stringified JSON config for training.
     * @return Stringified JSON training result containing updated model in JSON.
     */
    JSONObject MLTrain(
            String directory,
            ByteBuffer model,
            JSONObject trainingDataConf,
            JSONObject trainingConf) throws JSONException, IOException;

    /**
     * Testing interface from Java to JNI C++ backend.
     *
     * @param directory Directory of model and data.
     * @param model Model.
     * @param testingDataConf Testing set and label path config.
     * @param testingConf Stringified JSON config for testing.
     * @return  Stringified JSON testing result.
     */
    JSONObject MLTest(
            String directory,
            ByteBuffer model,
            JSONObject testingDataConf,
            JSONObject testingConf) throws JSONException, IOException;
}
