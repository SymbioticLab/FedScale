package com.fedscale.android.utils;

import io.grpc.executor.ServerResponse;

public interface MessageProcessor {
    ServerResponse operation();
}
