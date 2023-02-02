package com.fedscale.android.utils;

import android.content.Context;
import android.util.Log;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.google.common.io.Files;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * Const string representing status and tools for file operations.
 */
public class Common {
    public static String TAG = "Demo";
    public static String UPDATE_MODEL = "update_model";
    public static String MODEL_TEST = "model_test";
    public static String SHUT_DOWN = "terminate_executor";
    public static String START_ROUND = "start_round";
    public static String CLIENT_CONNECT = "client_connect";
    public static String CLIENT_TRAIN = "client_train";
    public static String DUMMY_EVENT = "dummy_event";
    public static String UPLOAD_MODEL = "upload_model";

    /**
     * Read file into string.
     *
     * @param filename The name of the file.
     * @return The string content of the file.
     * @throws IOException
     */
    public static String readStringFromFile(String filename) throws IOException {
        return Files.toString(new File(filename), StandardCharsets.UTF_8);
    }

    /**
     * Copy file from asset to com.fedscale.android.
     *
     * @param context Android context.
     * @param assetsFile Asset filename.
     * @param outFile Destination filename.
     * @throws IOException
     */
    public static void copyAssetResource2File(Context context, String assetsFile, String outFile) throws IOException {
        InputStream is = context.getAssets().open(assetsFile);
        inputStream2File(is, outFile);
    }

    /**
     * Write string content to file.
     *
     * @param str String content.
     * @param outFile Destination filename.
     * @throws IOException
     */
    public static void writeString2File(String str, String outFile) throws IOException {
        InputStream is = new ByteArrayInputStream(str.getBytes(StandardCharsets.UTF_8));
        inputStream2File(is, outFile);
    }

    /**
     * Write InputStream content into file.
     *
     * @param is InputStream content.
     * @param outFile Destination filename.
     * @throws IOException
     */
    public static void inputStream2File(InputStream is, String outFile) throws IOException {
        File outF = new File(outFile);
        FileOutputStream fos = new FileOutputStream(outF);

        int byteCount;
        byte[] buffer = new byte[1024];
        while ((byteCount = is.read(buffer)) != -1) {
            fos.write(buffer, 0, byteCount);
        }
        fos.flush();
        is.close();
        fos.close();
        outF.setReadable(true);
    }

    public static byte[] readBytes(InputStream inputStream) throws IOException {
        byte[] buffer = new byte[1024];
        ByteArrayOutputStream os = new ByteArrayOutputStream();
        int byteCount;
        while ((byteCount = inputStream.read(buffer)) != -1) {
            os.write(buffer, 0, byteCount);
        }
        inputStream.close();
        return os.toByteArray();
    }

    /**
     * Copy asset directory into com.fedscale.android.
     *
     * @param context Android context.
     * @param assetsDir Asset subdirectory.
     * @param outDir Destination directory.
     * @throws IOException
     */
    public static void copyDir(Context context, String assetsDir, File outDir) throws IOException {
        if (!outDir.exists()) {
            outDir.mkdir();
        }
        String[] assets = context.getAssets().list(assetsDir);
        for (String asset : assets) {
            String inPath = assetsDir + "/" + asset;
            if (assetsDir.equals("")) {
                inPath = asset;
            }
            String outPath = outDir + "/" + asset;
            String[] subAssets = context.getAssets().list(inPath);
            if (subAssets.length == 0) {
                // copy file
                copyAssetResource2File(context, inPath, outPath);
            } else {
                // copy directory
                copyDir(context, inPath, new File(outPath));
            }
        }
    }

    public static Map<String, Object> jsonToMap(JSONObject json) throws JSONException {
        Map<String, Object> retMap = new HashMap<>();

        if(json != JSONObject.NULL) {
            retMap = toMap(json);
        }
        return retMap;
    }

    public static Map<String, Object> toMap(JSONObject object) throws JSONException {
        Map<String, Object> map = new HashMap<>();

        Iterator<String> keysItr = object.keys();
        while(keysItr.hasNext()) {
            String key = keysItr.next();
            Object value = object.get(key);

            if(value instanceof JSONArray) {
                value = toList((JSONArray) value);
            }

            else if(value instanceof JSONObject) {
                value = toMap((JSONObject) value);
            }
            map.put(key, value);
        }
        return map;
    }

    public static List<Object> toList(JSONArray array) throws JSONException {
        List<Object> list = new ArrayList<>();
        for(int i = 0; i < array.length(); i++) {
            Object value = array.get(i);
            if(value instanceof JSONArray) {
                value = toList((JSONArray) value);
            }

            else if(value instanceof JSONObject) {
                value = toMap((JSONObject) value);
            }
            list.add(value);
        }
        return list;
    }

    /**
     * Log large content.
     *
     * @param tag Tag for this log.
     * @param content Content of the log.
     */
    public static void largeLog(String tag, String content) {
        if (content.length() > 4000) {
            Log.d(tag, content.substring(0, 4000));
            largeLog(tag, content.substring(4000));
        } else {
            Log.d(tag, content);
        }
    }
}
