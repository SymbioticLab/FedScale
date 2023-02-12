package com.fedscale.android.executor;

import android.os.Handler;
import android.os.HandlerThread;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import android.widget.TextView;

import com.fedscale.android.R;
import com.fedscale.android.mnn.MNNBackend;
import com.fedscale.android.utils.Backend;
import com.fedscale.android.utils.MessageProcessor;
import com.google.common.hash.Hashing;
import com.google.protobuf.ByteString;
import com.fedscale.android.tflite.TFLiteBackend;
import com.fedscale.android.utils.ClientConnections;
import com.fedscale.android.utils.Common;

import net.razorvine.pickle.Pickler;
import net.razorvine.pickle.Unpickler;

import org.json.JSONException;
import org.json.JSONObject;

import io.grpc.Server;
import io.grpc.executor.CompleteRequest;
import io.grpc.executor.PingRequest;
import io.grpc.executor.RegisterRequest;
import io.grpc.executor.ServerResponse;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

/**
 * Sample com.fedscale.android executor with MNN backend support.
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
    private Backend backend = new MNNBackend();

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

    /**
     *
     * Initialize executor ID.
     *
     * @param username Username in String.
     * @return HMAC-SHA256 hash of username.
     */
    private String initExecutorId(String username) {
        long currTime = System.currentTimeMillis();
        ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES);
        buffer.putLong(currTime);
        return Hashing
                .hmacSha256(buffer.array())
                .hashString(username, StandardCharsets.UTF_8).toString();
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
     * Move all files/directories inside assetsDir into com.fedscale.android cache directory.
     */
    private void initData() throws IOException {
        Log.i(Common.TAG, "Data movement starts ...");
        Common.copyDir(getBaseContext(), "", getCacheDir());
        Log.i(Common.TAG, "Data movement completes ...");
    }

    /**
     * Initialize variables associated to conf.json.
     */
    private void initAsset() throws IOException, JSONException {
        String configPath = getCacheDir() + "/conf.json";
        String configStr = Common.readStringFromFile(configPath);
        this.config = new JSONObject(configStr);
        this.mExecutorID = this.initExecutorId(this.config.getString("username"));
        this.aggregatorIP = this.config.getJSONObject("aggregator").getString("ip");
        this.aggregatorPort = this.config.getJSONObject("aggregator").getInt("port");
        this.communicator = new ClientConnections(
                this.aggregatorIP,
                this.aggregatorPort);
    }

    /**
     * Start running the executor by setting up execution and communication environment,
     * and monitoring the grpc message.
     */
    private void runExecutor() throws Exception {
        this.initData();
        this.initAsset();
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
    private Object deserializeResponse(ByteString responses) throws IOException {
        Unpickler unpickler = new Unpickler();
        return unpickler.loads(responses.toByteArray());
    }

    /**
     * Serialize the response to send to server upon assigned job completion
     *
     * @param responses Client responses after job completion.
     * @return The serialized response object to server.
     */
    private ByteString serializeResponse(Map<String, Object> responses) throws IOException {
        Common.largeLog("serialize", responses.toString());
        Pickler pickler = new Pickler();
        return ByteString.copyFrom(pickler.dumps(responses));
    }

    /**
     * Receive the broadcast global model for current round.
     *
     * @param model The broadcast global model config.
     */
    public void FLUpdateModel(byte[] model) throws JSONException, IOException {
        this.round++;
        this.setText(this.mExecuteStatus, Common.UPDATE_MODEL);
        this.setText(this.mUserId, this.mExecutorID + ": Round " + this.round);
        InputStream is = new ByteArrayInputStream(model);
        final String fileName = this.config.getJSONObject("model_conf").getString("path");
        final String modelPath = getCacheDir() + "/" + fileName;
        Common.inputStream2File(is, modelPath);
    }

    /**
     * Load train config and data to start training on that client.
     *
     * @param config The client training config.
     * @return The client id and train result.
     */
    public Map<String, Object> FLTrain(Map<String, Object> config) throws Exception {
        this.setText(this.mExecuteStatus, Common.CLIENT_TRAIN);
        JSONObject newTrainingConf = this.overrideConf(
                this.config.getJSONObject("training_conf"),
                config);
        Map<String, Object> trainResult = this.backend.MLTrain(
                getCacheDir().toString(),
                this.config.getJSONObject("model_conf").getString("path"),
                this.config.getJSONObject("training_data"),
                newTrainingConf);
        CompleteRequest request = CompleteRequest.newBuilder()
                .setClientId(this.mExecutorID)
                .setExecutorId(this.mExecutorID)
                .setEvent(Common.CLIENT_TRAIN)
                .setStatus(true).build();
        this.sendRequest(() -> this.communicator.stub.cLIENTEXECUTECOMPLETION(request));
        return trainResult;
    }

    /**
     * Model Testing. By default, we test the accuracy on all data of clients in the test group
     *
     * @param config The client testing config.
     */
    public void FLTest(Map<String, Object> config) throws Exception {
        this.setText(this.mExecuteStatus, Common.MODEL_TEST);
        JSONObject newTestingConf = this.overrideConf(
                this.config.getJSONObject("testing_conf"),
                config);
        Map<String, Object> testResult = this.backend.MLTest(
                getCacheDir().toString(),
                this.config.getJSONObject("model_conf").getString("path"),
                this.config.getJSONObject("testing_data"),
                newTestingConf);
        Map<String, Object> testRes = new HashMap<>();
        testRes.put("executorId", this.mExecutorID);
        testRes.put("results", testResult);
        CompleteRequest request = CompleteRequest.newBuilder()
                .setClientId(this.mExecutorID)
                .setExecutorId(this.mExecutorID)
                .setEvent(Common.MODEL_TEST)
                .setStatus(true)
                .setDataResult(this.serializeResponse(testRes)).
                build();
        this.sendRequest(() -> this.communicator.stub.cLIENTEXECUTECOMPLETION(request));
    }

    /**
     * Stop the current executor
     */
    public void FLStop() throws InterruptedException {
        this.setText(this.mExecuteStatus, Common.SHUT_DOWN);
        this.communicator.CloseServerConnection();
        this.receivedStopRequest = true;
    }

    /**
     * Return the statistics of training dataset
     *
     * @return Return the statistics of training dataset.
     */
    private Map<String, Object> reportExecutorInfoHandler() throws JSONException {
        return Common.jsonToMap(this.config.getJSONObject("executor_info"));
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
    private JSONObject overrideConf(JSONObject oldConfJSON, Map<String, Object> newConf) throws JSONException {
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
        this.sendRequest(() -> this.communicator.stub.cLIENTREGISTER(request));
    }

    /**
     * Ping the aggregator for new task.
     */
    private void clientPing() throws InterruptedException {
        this.setText(this.mExecuteStatus, "Pinging");
        PingRequest request = PingRequest.newBuilder()
                .setClientId(this.mExecutorID)
                .setExecutorId(this.mExecutorID).build();
        this.sendRequest(() -> this.communicator.stub.cLIENTPING(request));
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
                    this.FLUpdateModel((byte[])this.deserializeResponse(request.getData()));
                    Map<String, Object> trainResult = this.FLTrain(
                            (Map<String, Object>) this.deserializeResponse(request.getMeta()));
                    CompleteRequest cRequest = CompleteRequest.newBuilder()
                            .setClientId(this.mExecutorID)
                            .setExecutorId(this.mExecutorID)
                            .setEvent(Common.UPLOAD_MODEL)
                            .setStatus(true)
                            .setDataResult(this.serializeResponse(trainResult))
                            .build();
                    this.sendRequest(() -> this.communicator.stub.cLIENTEXECUTECOMPLETION(cRequest));
                } else if (currentEvent.equals(Common.MODEL_TEST)) {
                    this.FLTest((Map<String, Object>)this.deserializeResponse(request.getMeta()));
                } else if (currentEvent.equals(Common.UPDATE_MODEL)) {
                    this.FLUpdateModel((byte[])this.deserializeResponse(request.getData()));
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
     * Repeatedly send request to server until success, save response to the queue.
     *
     * @param msgProcessor lambda function of the client request
     */
    private void sendRequest(MessageProcessor msgProcessor) throws InterruptedException {
        long startTime = System.currentTimeMillis() / 1000;
        while (System.currentTimeMillis() / 1000 - startTime < 180) {
            try {
                Log.i("SendRequest", "Trying to get request");
                ServerResponse response = msgProcessor.operation();
                this.dispatchWorkerEvents(response);
                Log.i("SendRequest", "Successfully gotten request " + response.getEvent());
                break;
            } catch (Exception e) {
                Log.e(Common.TAG, "Exception", e);
                Thread.sleep(5 * 1000);
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

        mHandle.post(() -> {
            try {
                runExecutor();
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
