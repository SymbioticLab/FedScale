package com.fedscale.android.utils;

import android.util.Log;

import java.util.concurrent.TimeUnit;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.executor.JobServiceGrpc;

/**
 * GRPC connection from client to server.
 */
public class ClientConnections {
    public int port;
    public String address;
    public ManagedChannel channel;
    public JobServiceGrpc.JobServiceBlockingStub stub;
    public final int MAX_MESSAGE_LENGTH = 1*1024*1024*1024;

    /**
     * Initialization of IP and port.
     *
     * @param address server IP.
     * @param port server port.
     */
    public ClientConnections(String address, int port) {
        this.address = address;
        this.port = port;
    }

    /**
     * Initialize connection to server.
     */
    public void ConnectToServer() {
        Log.i("GRPC", "%%%%%%%%%% Opening grpc connection to " +
                this.address + " %%%%%%%%%%");
        this.channel = ManagedChannelBuilder
                .forAddress(this.address, this.port)
                .maxInboundMessageSize(this.MAX_MESSAGE_LENGTH)
                .usePlaintext().build();
        this.stub = JobServiceGrpc.newBlockingStub(this.channel);
    }

    /**
     * Stop connection from server.
     */
    public void CloseServerConnection() throws InterruptedException {
        Log.i("GRPC", "%%%%%%%%%% Closing grpc connection to the aggregator %%%%%%%%%%");
        this.channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }
}
