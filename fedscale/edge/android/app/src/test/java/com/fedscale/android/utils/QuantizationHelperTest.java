/*
 * Copyright (c) FedScale Project. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

package com.fedscale.android.utils;

import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

public class QuantizationHelperTest {

    @Test
    public void testInt4QuantizeAndDequantizeRoundtrip() {
        Random rng = new Random(42);
        int size = 256;
        float[] original = new float[size];
        for (int i = 0; i < size; i++) {
            original[i] = (float) rng.nextGaussian();
        }

        int blockSize = 64;
        QuantizationHelper.QuantizedTensor quantized = QuantizationHelper.quantizeToInt4(original, blockSize);

        Assert.assertEquals(4, quantized.bits);
        Assert.assertEquals(size, quantized.originalLength);
        Assert.assertEquals(size / 2, quantized.data.length);
        Assert.assertEquals(size / blockSize, quantized.scales.length);

        float[] restored = QuantizationHelper.dequantizeFromInt4(quantized, blockSize);
        Assert.assertEquals(original.length, restored.length);

        float totalSqError = 0.0f;
        for (int i = 0; i < size; i++) {
            float diff = original[i] - restored[i];
            totalSqError += diff * diff;
        }
        float mse = totalSqError / size;
        Assert.assertTrue("MSE should be < 0.05, got " + mse, mse < 0.05f);
    }

    @Test
    public void testInt4OddLengthArray() {
        int size = 35; // Odd length
        float[] original = new float[size];
        for (int i = 0; i < size; i++) {
            original[i] = (float) (i * 0.1);
        }

        int blockSize = 16;
        QuantizationHelper.QuantizedTensor quantized = QuantizationHelper.quantizeToInt4(original, blockSize);
        Assert.assertEquals((size + 1) / 2, quantized.data.length);

        float[] restored = QuantizationHelper.dequantizeFromInt4(quantized, blockSize);
        Assert.assertEquals(size, restored.length);

        float maxError = 0.0f;
        for (int i = 0; i < size; i++) {
            float err = Math.abs(original[i] - restored[i]);
            if (err > maxError) maxError = err;
        }
        Assert.assertTrue("Max reconstruction error should be < 0.35, got " + maxError, maxError < 0.35f);
    }

    @Test
    public void testInt4ZeroArray() {
        int size = 64;
        float[] zeros = new float[size];
        QuantizationHelper.QuantizedTensor quantized = QuantizationHelper.quantizeToInt4(zeros, 32);
        float[] restored = QuantizationHelper.dequantizeFromInt4(quantized, 32);

        for (int i = 0; i < size; i++) {
            Assert.assertEquals(0.0f, restored[i], 0.0f);
        }
    }

    @Test
    public void testInt4SmallMagnitudes() {
        int size = 64;
        float smallVal = 1e-9f;
        float[] small = new float[size];
        for (int i = 0; i < size; i++) {
            small[i] = smallVal;
        }

        QuantizationHelper.QuantizedTensor quantized = QuantizationHelper.quantizeToInt4(small, 64);
        float[] restored = QuantizationHelper.dequantizeFromInt4(quantized, 64);

        for (int i = 0; i < size; i++) {
            Assert.assertEquals(smallVal, restored[i], 1e-11f);
        }
    }

    @Test
    public void testInt8QuantizeAndDequantizeRoundtrip() {
        Random rng = new Random(123);
        int size = 128;
        float[] original = new float[size];
        for (int i = 0; i < size; i++) {
            original[i] = (float) rng.nextGaussian();
        }

        int blockSize = 32;
        QuantizationHelper.QuantizedTensor quantized = QuantizationHelper.quantizeToInt8(original, blockSize);

        Assert.assertEquals(8, quantized.bits);
        Assert.assertEquals(size, quantized.data.length);
        Assert.assertEquals(size / blockSize, quantized.scales.length);

        float[] restored = QuantizationHelper.dequantizeFromInt8(quantized, blockSize);
        float totalSqError = 0.0f;
        for (int i = 0; i < size; i++) {
            float diff = original[i] - restored[i];
            totalSqError += diff * diff;
        }
        float mse = totalSqError / size;
        Assert.assertTrue("MSE should be < 0.001, got " + mse, mse < 0.001f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNonFiniteInputRejectionNaN() {
        float[] withNaN = new float[]{1.0f, Float.NaN, 2.0f};
        QuantizationHelper.quantizeToInt4(withNaN, 2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNonFiniteInputRejectionInf() {
        float[] withInf = new float[]{1.0f, Float.POSITIVE_INFINITY, 2.0f};
        QuantizationHelper.quantizeToInt4(withInf, 2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidBlockSizeOdd() {
        float[] arr = new float[]{1.0f, 2.0f};
        QuantizationHelper.quantizeToInt4(arr, 3);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidBlockSizeZero() {
        float[] arr = new float[]{1.0f, 2.0f};
        QuantizationHelper.quantizeToInt4(arr, 0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTruncatedBufferRejection() {
        byte[] truncated = new byte[2]; // Needs 4 bytes for 8 elements
        float[] scales = new float[]{1.0f};
        QuantizationHelper.dequantizeFromInt4(truncated, scales, 8, 8);
    }

    @Test
    public void testFloatByteConversion() {
        float[] original = new float[]{1.5f, -2.25f, 0.0f, Float.MAX_VALUE};
        byte[] bytes = QuantizationHelper.floatsToBytes(original);
        Assert.assertEquals(original.length * 4, bytes.length);

        float[] restored = QuantizationHelper.bytesToFloats(bytes);
        Assert.assertArrayEquals(original, restored, 0.0f);
    }
}
