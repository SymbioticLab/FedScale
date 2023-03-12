package com.fedscale.android.executor;

import android.util.Log;

import com.fedscale.android.app.FLApp;
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
public class Client {
    private JSONObject config;

    private String mExecutorID;
    private ClientConnections communicator;

    private int round = 0;
    private boolean receivedStopRequest = false;
    private Queue<ServerResponse> eventQueue = new LinkedList<>();
    private Backend backend;

    private FLApp app;

    public Client(FLApp app) {
        this.app = app;
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
        ByteBuffer buffer = ByteBuffer.allocate(8);
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
        Common.copyDir(app.getBaseContext(), "dataset", app.getCacheDir());
        Log.i(Common.TAG, "Data movement completes ...");
    }

    /**
     * Initialize variables associated to conf.json.
     */
    private void initAsset() throws Exception {
        this.config = new JSONObject(Common.readFile(
                app.getBaseContext().getAssets().open("conf.json")).toString());
        this.mExecutorID = this.initExecutorId(this.config.getString("username"));
        String aggregatorIP = this.config.getJSONObject("aggregator").getString("ip");
        int aggregatorPort = this.config.getJSONObject("aggregator").getInt("port");
        this.communicator = new ClientConnections(
                aggregatorIP,
                aggregatorPort);
        final String backendName = this.config.getJSONObject("model_conf").getString("backend");
        if (backendName.equals("tflite")) this.backend = new TFLiteBackend();
        else if (backendName.equals("mnn")) this.backend = new MNNBackend();
        else throw new Exception(String.format("Unsupported backend %s", backendName));
    }

    /**
     * Start running the executor by setting up execution and communication environment,
     * and monitoring the grpc message.
     */
    public void initExecutor() throws Exception {
        this.initData();
        this.initAsset();
        this.app.initUI(this.config.getString("username"));
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
        this.app.onWriteModel();
        this.round++;
        this.app.onChangeStatus(Common.UPDATE_MODEL);
        this.app.onChangeRound(this.round);
        final String fileName = this.config.getJSONObject("model_conf").getString("path");
        final String modelPath = app.getCacheDir() + "/" + fileName;
        InputStream is = new ByteArrayInputStream(model);
        Common.inputStream2File(is, modelPath);
        this.app.onGetModel();
    }

    /**
     * Load train config and data to start training on that client.
     *
     * @param config The client training config.
     */
    public void FLTrain(Map<String, Object> config) throws Exception {
        this.app.onChangeStatus(Common.CLIENT_TRAIN);
        this.config.getJSONObject("training_conf").put("fine_tune", false);
        JSONObject newTrainingConf = this.overrideConf(
                this.config.getJSONObject("training_conf"),
                config);
        Map<String, Object> trainResult = this.backend.MLTrain(
                this.app.getCacheDir().toString(),
                this.config.getJSONObject("model_conf").getString("path"),
                this.config.getJSONObject("training_data"),
                newTrainingConf);
        // TODO: It might be better to make UPLOAD_MODEL async to utilize the resource.
        CompleteRequest request = CompleteRequest.newBuilder()
                .setClientId(this.mExecutorID)
                .setExecutorId(this.mExecutorID)
                .setEvent(Common.UPLOAD_MODEL)
                .setStatus(true)
                .setDataResult(this.serializeResponse(trainResult))
                .build();
        this.sendRequest(() -> this.communicator.stub.cLIENTEXECUTECOMPLETION(request));
    }

    /**
     * Load train config and data to start training on that client without connecting to the cloud.
     */
    public void LocalTrain() throws Exception {
        this.app.onChangeStatus(Common.CLIENT_TRAIN_LOCALLY);
        this.config.getJSONObject("training_conf").put("fine_tune", true);
        Map<String, Object> trainResult = this.backend.MLTrain(
                this.app.getCacheDir().toString(),
                this.config.getJSONObject("model_conf").getString("path"),
                this.config.getJSONObject("training_data"),
                this.config.getJSONObject("training_conf"));
        this.app.onChangeStatus(Common.CLIENT_TRAIN_LOCALLY_FIN);
    }

    /**
     * Model Testing. By default, we test the accuracy on all data of clients in the test group
     *
     * @param config The client testing config.
     */
    public void FLTest(Map<String, Object> config) throws Exception {
        this.app.onChangeStatus(Common.MODEL_TEST);
        JSONObject newTestingConf = this.overrideConf(
                this.config.getJSONObject("testing_conf"),
                config);
        Map<String, Object> testResult = this.backend.MLTest(
                app.getCacheDir().toString(),
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
     * Start the current executor
     */
    public void FLStart() throws Exception {
        this.receivedStopRequest = false;
        this.app.onChangeStatus(Common.CLIENT_CONNECT);
        this.setupCommunication();
        this.eventMonitor();
    }

    /**
     * Stop the current executor
     */
    public void FLStop() throws InterruptedException {
        this.app.onChangeStatus(Common.SHUT_DOWN);
        this.communicator.CloseServerConnection();
        this.receivedStopRequest = true;
    }

    /**
     * Return the statistics of training dataset
     *
     * @return Return the statistics of training dataset.
     */
    private Map<String, Object> reportExecutorInfoHandler() throws JSONException {
        return Common.JSON2Map(this.config.getJSONObject("executor_info"));
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
        this.app.onChangeStatus("Registering");
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
        this.app.onChangeStatus("Pinging");
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
                    this.FLUpdateModel((byte[]) this.deserializeResponse(request.getData()));
                    this.FLTrain((Map<String, Object>) this.deserializeResponse(request.getMeta()));
                } else if (currentEvent.equals(Common.MODEL_TEST)) {
                    this.FLUpdateModel((byte[]) this.deserializeResponse(request.getData()));
                    this.FLTest((Map<String, Object>)this.deserializeResponse(request.getMeta()));
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
        while (System.currentTimeMillis() / 1000 - startTime < 180 && !this.receivedStopRequest) {
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
}
