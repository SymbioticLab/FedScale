/*
 * Copyright (c) FedScale Project. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

package com.fedscale.android.utils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Lightweight tensor quantization utility for Android edge federated learning.
 * Provides block-wise dynamic scaling and sub-byte nibble packing for INT4 and INT8
 * parameter exchange, achieving significant bandwidth and memory reduction on mobile devices.
 */
public final class QuantizationHelper {

    private QuantizationHelper() {
        // Utility class
    }

    /**
     * Container holding quantized tensor payload and metadata.
     */
    public static class QuantizedTensor {
        public final byte[] data;
        public final float[] scales;
        public final int originalLength;
        public final int bits;

        public QuantizedTensor(byte[] data, float[] scales, int originalLength, int bits) {
            this.data = data;
            this.scales = scales;
            this.originalLength = originalLength;
            this.bits = bits;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            QuantizedTensor that = (QuantizedTensor) o;
            return originalLength == that.originalLength &&
                    bits == that.bits &&
                    Arrays.equals(data, that.data) &&
                    Arrays.equals(scales, that.scales);
        }

        @Override
        public int hashCode() {
            int result = Arrays.hashCode(data);
            result = 31 * result + Arrays.hashCode(scales);
            result = 31 * result + originalLength;
            result = 31 * result + bits;
            return result;
        }
    }

    /**
     * Quantize a float array into packed signed INT4 representation with per-block dynamic scaling.
     * Each pair of 4-bit values is packed into a single byte (~8x wire size reduction relative to float32).
     *
     * @param input     The source floating-point array.
     * @param blockSize Number of elements per block (must be a positive even integer).
     * @return A {@link QuantizedTensor} containing packed bytes and per-block scales.
     */
    public static QuantizedTensor quantizeToInt4(float[] input, int blockSize) {
        if (input == null) {
            throw new IllegalArgumentException("Input float array cannot be null");
        }
        if (blockSize <= 0 || blockSize % 2 != 0) {
            throw new IllegalArgumentException("blockSize must be a positive even integer, got " + blockSize);
        }

        int totalElements = input.length;
        if (totalElements == 0) {
            return new QuantizedTensor(new byte[0], new float[0], 0, 4);
        }

        int numBlocks = 1 + (totalElements - 1) / blockSize;
        float[] scales = new float[numBlocks];
        int packedLength = (totalElements + 1) / 2;
        byte[] packed = new byte[packedLength];

        int elemIdx = 0;
        int packedIdx = 0;

        for (int b = 0; b < numBlocks; b++) {
            int blockStart = b * blockSize;
            int blockEnd = Math.min(blockStart + blockSize, totalElements);

            float maxAbs = 0.0f;
            for (int i = blockStart; i < blockEnd; i++) {
                float v = input[i];
                if (Float.isNaN(v) || Float.isInfinite(v)) {
                    throw new IllegalArgumentException("Input contains non-finite value at index " + i + ": " + v);
                }
                float a = Math.abs(v);
                if (a > maxAbs) {
                    maxAbs = a;
                }
            }

            float scale;
            double invScale;
            if (maxAbs == 0.0f) {
                scale = 0.0f;
                invScale = 0.0;
            } else {
                scale = maxAbs / 7.0f;
                invScale = 7.0 / (double) maxAbs;
            }
            scales[b] = scale;

            int i = blockStart;
            while (i < blockEnd) {
                int q0 = quantizeNibble(input[i++], invScale, maxAbs);
                int q1 = (i < blockEnd) ? quantizeNibble(input[i++], invScale, maxAbs) : 0;
                packed[packedIdx++] = (byte) (((q0 & 0x0F) << 4) | (q1 & 0x0F));
            }
        }

        return new QuantizedTensor(packed, scales, totalElements, 4);
    }

    private static int quantizeNibble(float v, double invScale, float maxAbs) {
        if (maxAbs == 0.0f) {
            return 8; // 0 shifted by +8
        }
        int q = (int) Math.round((double) v * invScale);
        if (q > 7) q = 7;
        if (q < -7) q = -7;
        return q + 8;
    }

    /**
     * Dequantize packed INT4 payload back into a floating-point array.
     *
     * @param tensor    The quantized tensor.
     * @param blockSize Number of elements per block.
     * @return Reconstructed float array.
     */
    public static float[] dequantizeFromInt4(QuantizedTensor tensor, int blockSize) {
        if (tensor == null) {
            throw new IllegalArgumentException("QuantizedTensor cannot be null");
        }
        return dequantizeFromInt4(tensor.data, tensor.scales, tensor.originalLength, blockSize);
    }

    /**
     * Dequantize packed INT4 byte array and block scales back into a floating-point array.
     *
     * @param packed         Packed byte array.
     * @param scales         Per-block scaling factors.
     * @param originalLength Expected number of elements in the output array.
     * @param blockSize      Number of elements per block.
     * @return Reconstructed float array.
     */
    public static float[] dequantizeFromInt4(byte[] packed, float[] scales, int originalLength, int blockSize) {
        if (packed == null || scales == null) {
            throw new IllegalArgumentException("Packed buffer and scales cannot be null");
        }
        if (originalLength < 0) {
            throw new IllegalArgumentException("originalLength cannot be negative: " + originalLength);
        }
        if (blockSize <= 0 || blockSize % 2 != 0) {
            throw new IllegalArgumentException("blockSize must be a positive even integer, got " + blockSize);
        }
        if (originalLength == 0) {
            return new float[0];
        }

        int expectedBytes = (originalLength + 1) / 2;
        if (packed.length < expectedBytes) {
            throw new IllegalArgumentException("Packed buffer truncated: expected at least " + expectedBytes +
                    " bytes, got " + packed.length);
        }
        int numBlocks = 1 + (originalLength - 1) / blockSize;
        if (scales.length < numBlocks) {
            throw new IllegalArgumentException("Scales array too short: expected " + numBlocks +
                    ", got " + scales.length);
        }

        float[] restored = new float[originalLength];
        int elemIdx = 0;
        int packedIdx = 0;

        for (int b = 0; b < numBlocks && elemIdx < originalLength; b++) {
            float scale = scales[b];
            if (!Float.isFinite(scale)) {
                throw new IllegalArgumentException("Non-finite scale factor at block " + b + ": " + scale);
            }

            int inBlock = 0;
            while (inBlock < blockSize && elemIdx < originalLength) {
                int byteVal = packed[packedIdx++] & 0xFF;

                int q0 = ((byteVal >> 4) & 0x0F) - 8;
                float v0 = q0 * scale;
                if (!Float.isFinite(v0)) {
                    throw new IllegalArgumentException("Decoded value overflow at element " + elemIdx);
                }
                restored[elemIdx++] = v0;
                inBlock++;

                if (elemIdx < originalLength && inBlock < blockSize) {
                    int q1 = (byteVal & 0x0F) - 8;
                    float v1 = q1 * scale;
                    if (!Float.isFinite(v1)) {
                        throw new IllegalArgumentException("Decoded value overflow at element " + elemIdx);
                    }
                    restored[elemIdx++] = v1;
                    inBlock++;
                }
            }
        }

        return restored;
    }

    /**
     * Quantize a float array into signed INT8 representation with per-block dynamic scaling.
     *
     * @param input     The source floating-point array.
     * @param blockSize Number of elements per block (must be a positive integer).
     * @return A {@link QuantizedTensor} containing signed bytes and per-block scales.
     */
    public static QuantizedTensor quantizeToInt8(float[] input, int blockSize) {
        if (input == null) {
            throw new IllegalArgumentException("Input float array cannot be null");
        }
        if (blockSize <= 0) {
            throw new IllegalArgumentException("blockSize must be a positive integer, got " + blockSize);
        }

        int totalElements = input.length;
        if (totalElements == 0) {
            return new QuantizedTensor(new byte[0], new float[0], 0, 8);
        }

        int numBlocks = 1 + (totalElements - 1) / blockSize;
        float[] scales = new float[numBlocks];
        byte[] quantized = new byte[totalElements];

        for (int b = 0; b < numBlocks; b++) {
            int blockStart = b * blockSize;
            int blockEnd = Math.min(blockStart + blockSize, totalElements);

            float maxAbs = 0.0f;
            for (int i = blockStart; i < blockEnd; i++) {
                float v = input[i];
                if (Float.isNaN(v) || Float.isInfinite(v)) {
                    throw new IllegalArgumentException("Input contains non-finite value at index " + i + ": " + v);
                }
                float a = Math.abs(v);
                if (a > maxAbs) {
                    maxAbs = a;
                }
            }

            float scale;
            double invScale;
            if (maxAbs == 0.0f) {
                scale = 0.0f;
                invScale = 0.0;
            } else {
                scale = maxAbs / 127.0f;
                invScale = 127.0 / (double) maxAbs;
            }
            scales[b] = scale;

            for (int i = blockStart; i < blockEnd; i++) {
                if (maxAbs == 0.0f) {
                    quantized[i] = 0;
                } else {
                    int q = (int) Math.round((double) input[i] * invScale);
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    quantized[i] = (byte) q;
                }
            }
        }

        return new QuantizedTensor(quantized, scales, totalElements, 8);
    }

    /**
     * Dequantize INT8 byte array back into a floating-point array.
     *
     * @param tensor    The quantized tensor.
     * @param blockSize Number of elements per block.
     * @return Reconstructed float array.
     */
    public static float[] dequantizeFromInt8(QuantizedTensor tensor, int blockSize) {
        if (tensor == null) {
            throw new IllegalArgumentException("QuantizedTensor cannot be null");
        }
        return dequantizeFromInt8(tensor.data, tensor.scales, tensor.originalLength, blockSize);
    }

    /**
     * Dequantize INT8 byte array and block scales back into a floating-point array.
     *
     * @param quantized      Signed 8-bit byte array.
     * @param scales         Per-block scaling factors.
     * @param originalLength Expected number of elements.
     * @param blockSize      Number of elements per block.
     * @return Reconstructed float array.
     */
    public static float[] dequantizeFromInt8(byte[] quantized, float[] scales, int originalLength, int blockSize) {
        if (quantized == null || scales == null) {
            throw new IllegalArgumentException("Quantized buffer and scales cannot be null");
        }
        if (originalLength < 0) {
            throw new IllegalArgumentException("originalLength cannot be negative: " + originalLength);
        }
        if (blockSize <= 0) {
            throw new IllegalArgumentException("blockSize must be a positive integer, got " + blockSize);
        }
        if (originalLength == 0) {
            return new float[0];
        }

        if (quantized.length < originalLength) {
            throw new IllegalArgumentException("Quantized buffer truncated: expected " + originalLength +
                    " bytes, got " + quantized.length);
        }
        int numBlocks = 1 + (originalLength - 1) / blockSize;
        if (scales.length < numBlocks) {
            throw new IllegalArgumentException("Scales array too short: expected " + numBlocks +
                    ", got " + scales.length);
        }

        float[] restored = new float[originalLength];
        for (int b = 0; b < numBlocks; b++) {
            float scale = scales[b];
            if (!Float.isFinite(scale)) {
                throw new IllegalArgumentException("Non-finite scale factor at block " + b + ": " + scale);
            }
            int blockStart = b * blockSize;
            int blockEnd = Math.min(blockStart + blockSize, originalLength);
            for (int i = blockStart; i < blockEnd; i++) {
                float v = (float) quantized[i] * scale;
                if (!Float.isFinite(v)) {
                    throw new IllegalArgumentException("Decoded value overflow at element " + i);
                }
                restored[i] = v;
            }
        }

        return restored;
    }

    /**
     * Utility converting float array to native byte array in little-endian order.
     */
    public static byte[] floatsToBytes(float[] floats) {
        if (floats == null) return new byte[0];
        ByteBuffer buffer = ByteBuffer.allocate(floats.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : floats) {
            buffer.putFloat(f);
        }
        return buffer.array();
    }

    /**
     * Utility converting native byte array in little-endian order back to float array.
     */
    public static float[] bytesToFloats(byte[] bytes) {
        if (bytes == null || bytes.length == 0) return new float[0];
        if (bytes.length % 4 != 0) {
            throw new IllegalArgumentException("Byte array length must be a multiple of 4, got " + bytes.length);
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] floats = new float[bytes.length / 4];
        for (int i = 0; i < floats.length; i++) {
            floats[i] = buffer.getFloat();
        }
        return floats;
    }
}
