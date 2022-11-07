package com.taobao.android.mnndemo;

import android.os.Handler;
import android.os.HandlerThread;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import android.widget.TextView;

import com.google.common.hash.Hashing;
import com.google.protobuf.ByteString;
import com.taobao.android.mnn.MNNTrainInstance;
import com.taobao.android.utils.ClientConnections;
import com.taobao.android.utils.Common;

import org.json.JSONException;
import org.json.JSONObject;

import io.grpc.executor.CompleteRequest;
import io.grpc.executor.PingRequest;
import io.grpc.executor.RegisterRequest;
import io.grpc.executor.ServerResponse;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Queue;

/**
 * Sample android executor with MNN backend support.
 * Training and executing will be handled inside MNN C++.
 * Server-client communication will be handled in JAVA.
 */
public class FLExecutor extends AppCompatActivity {
    private JSONObject config;

    private String mExecutorID;
    private String aggregatorIP;
    private int aggregatorPort;
    private ClientConnections communicator;

    private int round = 0;
    private boolean receivedStopRequest = false;
    private Queue<ServerResponse> eventQueue = new LinkedList<>();

    private String trainingSets;
    private String trainingLabels;
    private String testingSets;
    private String testingLabels;
    private String oldJsonPath;
    private String oldModelPath;
    private String newJsonPath;
    private String newModelPath;

    private TextView mUserId;
    private TextView mExecuteStatus;
    private TextView mExecuteResult;

    HandlerThread mThread;
    Handler mHandle;

    /**
     * Initialization of the User Interface.
     * Three lines of UI: executor id, status, result (not changed)
     */
    private void initUI() {
        runOnUiThread(()->{
            String userIdMsg = this.mExecutorID + ": Round " + this.round;
            this.mUserId.setText(userIdMsg);
            this.mExecuteStatus.setText(Common.CLIENT_CONNECT);
            this.mExecuteResult.setText("");
        });
    }

    private String initExecutorId(String username) {
        long currTime = System.currentTimeMillis();
        ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES);
        buffer.putLong(currTime);
        return Hashing
                .hmacSha256(buffer.array())
                .hashString(username, Charset.forName("UTF-8")).toString();
    }

    // No need for setupEnv

    /**
     * Set up grpc connection.
     */
    private void setupCommunication() {
        this.initControlCommunication();
    }

    // No need for setupSeed

    /**
     * Create communication channel between coordinator and executor.
     * This channel serves control messages.
     */
    private void initControlCommunication() {
        this.communicator.ConnectToServer();
    }

    // No need for initDataCommunication

    // No need for initModel

    /**
     * Move all files/directories inside assetsDir into android cache directory.
     *
     * @param assetsDir The subdirectory of assets which contains
     *                  training set, testing set, labels and config.
     */
    private void initData(String assetsDir) throws IOException {
        Log.i(Common.TAG, "Data movement starts ...");
        String mMobileTrainPath = getCacheDir() + "/" + assetsDir;
        Common.copyDir(getBaseContext(), assetsDir, mMobileTrainPath);
        Log.i(Common.TAG, "Data movement completes ...");
    }

    /**
     * Initialize variables associated to conf.json.
     *
     * @param assetsDir The subdirectory of assets which contains
     *                  training set, testing set, labels and config.
     */
    private void initAsset(String assetsDir) throws IOException, JSONException {
        String configPath = getCacheDir() + "/" + assetsDir + "/conf.json";
        String configStr = Common.readStringFromFile(configPath);
        this.config = new JSONObject(configStr);
        this.mExecutorID = this.initExecutorId(this.config.getString("username"));
        this.aggregatorIP = this.config.getJSONObject("aggregator").getString("ip");
        this.aggregatorPort = this.config.getJSONObject("aggregator").getInt("port");
        this.communicator = new ClientConnections(
                this.aggregatorIP,
                this.aggregatorPort);

        String dataDir = getCacheDir() + "/" + assetsDir;
        JSONObject assetsConfig = this.config.getJSONObject("assets");
        this.trainingSets   = dataDir + "/" + assetsConfig.getString("training_set") + "/";
        this.testingSets    = dataDir + "/" + assetsConfig.getString("testing_set") + "/";
        this.trainingLabels = dataDir + "/" + assetsConfig.getString("training_labels");
        this.testingLabels  = dataDir + "/" + assetsConfig.getString("testing_labels");
        this.oldJsonPath    = dataDir + "/" + assetsConfig.getString("old_model_json");
        this.oldModelPath   = dataDir + "/" + assetsConfig.getString("old_model_mnn");
        this.newJsonPath    = dataDir + "/" + assetsConfig.getString("new_model_json");
        this.newModelPath   = dataDir + "/" + assetsConfig.getString("new_model_mnn");
    }

    /**
     * Start running the executor by setting up execution and communication environment,
     * and monitoring the grpc message.
     *
     * @param assetsDir The subdirectory of assets which contains
     *                  training set, testing set, labels and config.
     */
    private void runExecutor(String assetsDir) throws Exception {
        this.initData(assetsDir);
        this.initAsset(assetsDir);
        this.initUI();
        this.setupCommunication();
        this.eventMonitor();
    }

    /**
     * Add new events to worker queues.
     *
     * @param request Add grpc request from server (e.g. MODEL_TEST, MODEL_TRAIN) to event_queue.
     */
    private void dispatchWorkerEvents(ServerResponse request) {
        this.eventQueue.add(request);
    }

    /**
     * Deserialize the response from server.
     *
     * @param responses Serialized response from server.
     * @return Deserialized response string from server.
     */
    private String deserializeResponse(ByteString responses) throws IOException {
        String res = responses.toString("UTF-8");
        Common.largeLog("deserialize", res);
        return res;
    }

    /**
     * Serialize the response to send to server upon assigned job completion
     *
     * @param responses Client responses after job completion.
     * @return The serialized response object to server.
     */
    private ByteString serializeResponse(String responses) throws IOException {
        Common.largeLog("serialize", responses);
        return ByteString.copyFrom(responses.getBytes("UTF-8"));
    }

    /**
     * Receive the broadcast global model for current round.
     *
     * @param config The broadcast global model config.
     */
    public void FLUpdateModel(String config) throws IOException {
        this.round++;
        this.setText(this.mExecuteStatus, Common.UPDATE_MODEL);
        this.setText(this.mUserId, this.mExecutorID + ": Round " + this.round);
        Common.writeString2File(config, this.oldJsonPath);
    }

    /**
     * Load train config and data to start training on that client.
     *
     * @param config The client training config.
     * @return The client id and train result.
     */
    public String FLTrain(String config) throws Exception {
        this.setText(this.mExecuteStatus, Common.CLIENT_TRAIN);
        MNNTrainInstance trainInstance = new MNNTrainInstance();
        trainInstance.convert(
                this.oldModelPath,
                this.oldJsonPath,
                true);
        JSONObject newConf = this.overrideConf(
                this.config.getJSONObject("training"),
                config);
        String trainResult = trainInstance.train(
                newConf.getInt("num_classes"),
                this.oldModelPath,
                this.newModelPath,
                this.newJsonPath,
                this.trainingSets,
                this.trainingLabels,
                newConf.toString());
        CompleteRequest request = CompleteRequest.newBuilder()
                .setClientId(this.mExecutorID)
                .setExecutorId(this.mExecutorID)
                .setEvent(Common.CLIENT_TRAIN)
                .setStatus(true).build();
        long startTime = System.currentTimeMillis() / 1000;
        while (System.currentTimeMillis() / 1000 - startTime < 180) {
            try {
                ServerResponse response = this.communicator.stub.cLIENTEXECUTECOMPLETION(request);
                this.dispatchWorkerEvents(response);
                break;
            } catch (Exception e) {
                Log.w(
                        Common.TAG,
                        String.format("Failed to connect to aggregator %s:%d. Will retry in 5 sec.",
                                this.aggregatorIP, this.aggregatorPort));
                Thread.sleep(5 * 1000);
            }
        }
        return trainResult;
    }

    /**
     * Model Testing. By default, we test the accuracy on all data of clients in the test group
     *
     * @param config The client testing config.
     */
    public void FLTest(String config) throws Exception {
        this.setText(this.mExecuteStatus, Common.MODEL_TEST);
        MNNTrainInstance trainInstance = new MNNTrainInstance();
        trainInstance.convert(
                this.oldModelPath,
                this.oldJsonPath,
                true);
        JSONObject newConf = this.overrideConf(
                this.config.getJSONObject("testing"),
                config);
        String testResult = trainInstance.test(
                newConf.getInt("num_classes"),
                this.oldModelPath,
                this.testingSets,
                this.testingLabels,
                newConf.toString());
        JSONObject testRes = new JSONObject();
        testRes.put("executorId", this.mExecutorID);
        testRes.put("results", new JSONObject(testResult));
        CompleteRequest request = CompleteRequest.newBuilder()
                .setClientId(this.mExecutorID)
                .setExecutorId(this.mExecutorID)
                .setEvent(Common.MODEL_TEST)
                .setStatus(true)
                .setDataResult(this.serializeResponse(testRes.toString())).
                build();
        long startTime = System.currentTimeMillis() / 1000;
        while (System.currentTimeMillis() / 1000 - startTime < 180) {
            try {
                ServerResponse response = this.communicator.stub.cLIENTEXECUTECOMPLETION(request);
                this.dispatchWorkerEvents(response);
                break;
            } catch (Exception e) {
                Log.w(
                        Common.TAG,
                        String.format("Failed to connect to aggregator %s:%d. Will retry in 5 sec.",
                                this.aggregatorIP, this.aggregatorPort));
                Thread.sleep(5 * 1000);
            }
        }
    }

    /**
     * Stop the current executor
     */
    public void FLStop() {
        this.setText(this.mExecuteStatus, Common.SHUT_DOWN);
        this.communicator.CloseServerConnection();
        this.receivedStopRequest = true;
    }

    /**
     * Return the statistics of training dataset
     *
     * @return Return the statistics of training dataset.
     */
    private String reportExecutorInfoHandler() throws JSONException {
        JSONObject executorInfo = this.config.getJSONObject("executor_info");
        return executorInfo.toString();
    }

    // No need for updateModelHandler

    // No need for loadGlobalModel

    /**
     * Override the variable arguments for different client.
     *
     * @param oldConfJSON The default client runtime config.
     * @param newConf The server specified client runtime config.
     * @return The JSONObject of the updated client runtime config.
     */
    private JSONObject overrideConf(JSONObject oldConfJSON, String newConf) throws JSONException {
        JSONObject newConfJSON = new JSONObject(newConf);
        if (newConfJSON.has("client_id")) {
            oldConfJSON.put("client_id", newConfJSON.getString("client_id"));
        }
        if (newConfJSON.has("task_config")) {
            JSONObject newTaskConfJSON = newConfJSON.getJSONObject("task_config");
            Iterator<String> keys = newTaskConfJSON.keys();
            while(keys.hasNext()) {
                String key = keys.next();
                oldConfJSON.put(key, newTaskConfJSON.get(key));
            }
        }
        return oldConfJSON;
    }

    // No need for clientTrainer

    // No need for trainingHandler

    // No need for testingHandler

    /**
     * Register the executor information to the aggregator.
     */
    private void clientRegister() throws Exception {
        this.setText(this.mExecuteStatus, "Registering");
        RegisterRequest request = RegisterRequest.newBuilder()
                .setExecutorId(this.mExecutorID)
                .setClientId(this.mExecutorID)
                .setExecutorInfo(this.serializeResponse(
                        this.reportExecutorInfoHandler()))
                .build();
        long startTime = System.currentTimeMillis() / 1000;
        while (System.currentTimeMillis() / 1000 - startTime < 180) {
            try {
                ServerResponse response = this.communicator.stub.cLIENTREGISTER(request);
                this.dispatchWorkerEvents(response);
                break;
            } catch (Exception e) {
                Log.w(
                        Common.TAG,
                        String.format("Failed to connect to aggregator %s:%d. Will retry in 5 sec.",
                                this.aggregatorIP, this.aggregatorPort));
                Thread.sleep(5 * 1000);
            }
        }
    }

    /**
     * Ping the aggregator for new task.
     */
    private void clientPing() throws InterruptedException {
        this.setText(this.mExecuteStatus, "Pinging");
        PingRequest request = PingRequest.newBuilder()
                .setClientId(this.mExecutorID)
                .setExecutorId(this.mExecutorID).build();
        long startTime = System.currentTimeMillis() / 1000;
        while (System.currentTimeMillis() / 1000 - startTime < 180) {
            try {
                ServerResponse response = this.communicator.stub.cLIENTPING(request);
                Log.i(Common.TAG, "Get EVENT " + response.getEvent());
                this.dispatchWorkerEvents(response);
                break;
            } catch (Exception e) {
                Log.w(
                        Common.TAG,
                        String.format("Failed to connect to aggregator %s:%d. Will retry in 5 sec.",
                                this.aggregatorIP, this.aggregatorPort));
                Thread.sleep(5 * 1000);
            }
        }
    }

    /**
     * Activate event handler once receiving new message
     */
    private void eventMonitor() throws Exception {
        Log.i(Common.TAG, "Start monitoring events ...");
        this.clientRegister();
        while (!this.receivedStopRequest) {
            if (this.eventQueue.size() > 0) {
                ServerResponse request = this.eventQueue.remove();
                String currentEvent = request.getEvent();
                Log.i(Common.TAG, "Handling EVENT " + currentEvent);
                if (currentEvent.equals(Common.CLIENT_TRAIN)) {
                    String trainConfigStr = this.deserializeResponse(request.getMeta());
                    String trainResult = this.FLTrain(trainConfigStr);
                    CompleteRequest cRequest = CompleteRequest.newBuilder()
                            .setClientId(this.mExecutorID)
                            .setExecutorId(this.mExecutorID)
                            .setEvent(Common.UPLOAD_MODEL)
                            .setStatus(true)
                            .setDataResult(this.serializeResponse(trainResult))
                            .build();
                    long startTime = System.currentTimeMillis() / 1000;
                    while (System.currentTimeMillis() / 1000 - startTime < 180) {
                        try {
                            ServerResponse response =
                                    this.communicator.stub.cLIENTEXECUTECOMPLETION(cRequest);
                            this.dispatchWorkerEvents(response);
                            break;
                        } catch (Exception e) {
                            Log.w(
                                    Common.TAG,
                                    String.format(
                                            "Failed to connect to aggregator %s:%d. Will retry in 5 sec.",
                                            this.aggregatorIP, this.aggregatorPort));
                            Thread.sleep(5 * 1000);
                        }
                    }
                } else if (currentEvent.equals(Common.MODEL_TEST)) {
                    this.FLTest(this.deserializeResponse(request.getMeta()));
                } else if (currentEvent.equals(Common.UPDATE_MODEL)) {
                    this.FLUpdateModel(this.deserializeResponse(request.getData()));
                } else if (currentEvent.equals(Common.SHUT_DOWN)) {
                    this.FLStop();
                }
            } else {
                Thread.sleep(1000);
                this.clientPing();
            }
        }
    }

    /**
     * Set text displayed on User Interface.
     *
     * @param text The associated TextView object.
     * @param value The value of the updated text.
     */
    private void setText(final TextView text,final String value){
        runOnUiThread(() -> text.setText(value));
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);

        this.mUserId = findViewById(R.id.userId);
        this.mExecuteStatus = findViewById(R.id.executeStatus);
        this.mExecuteResult = findViewById(R.id.executeResult);

        mThread = new HandlerThread("MNNTrain");
        mThread.start();
        mHandle = new Handler(mThread.getLooper());

        String assetsDir = "TrainTest";

        mHandle.post(() -> {
            try {
                runExecutor(assetsDir);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onDestroy() {
        mThread.interrupt();
        super.onDestroy();
    }
}
