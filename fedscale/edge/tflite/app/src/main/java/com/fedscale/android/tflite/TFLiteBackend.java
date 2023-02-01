package com.fedscale.android.tflite;

import static java.lang.Math.min;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import android.util.Pair;

import com.fedscale.android.utils.Backend;
import com.fedscale.android.utils.Common;

import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TFLiteBackend implements Backend {
//    private final Handler handler = new Handler(Looper.getMainLooper());

    @Override
    public Map<String, Object> MLTrain(
            String directory,
            ByteBuffer model,
            JSONObject trainingDataConf,
            JSONObject trainingConf) throws JSONException, IOException {
        final String trainImagesFolder = directory + "/" + trainingDataConf.getString("data" ) + "/";
        final String trainImagesTxt = directory + "/" + trainingDataConf.getString("label");

        final String clientId       = trainingConf.getString("client_id");
        final int numClasses        = trainingConf.getInt("num_classes");
        final int trainEpochs       = trainingConf.getInt("epoch");
        final int trainBatchSize    = trainingConf.getInt("batch_size");
        final int trainNumWorkers   = trainingConf.getInt("num_workers");
        final float learningRate    = (float) trainingConf.getDouble("learning_rate");
        final float lossDecay       = (float) trainingConf.getDouble("loss_decay");
        final float momentum        = (float) trainingConf.getDouble("momentum");
        final float weightDecay     = (float) trainingConf.getDouble("weight_decay");
        final int channel           = trainingConf.getInt("channel");
        final int height            = trainingConf.getInt("height");
        final int width             = trainingConf.getInt("width");

        List<String> labels = FileUtil.loadLabels(new FileInputStream(trainImagesTxt));
        final int dataCount = labels.size();
        final int trainIterations = (int)Math.ceil((double)dataCount / (double)trainBatchSize);
        Collections.shuffle(labels);

        Pair<List<float[][]>, List<float[][]>> pair = dataLoader(
                labels,
                trainImagesFolder,
                trainBatchSize,
                height,
                width,
                channel,
                numClasses);
        List<float[][]> trainImageBatches = pair.first;
        List<float[][]> trainLabelBatches = pair.second;

        float epochTrainLoss = 0;
        float currentLoss = 0;

        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(trainNumWorkers);
        Interpreter interpreter = new Interpreter(model, options);

        for (int epoch = 0; epoch < trainEpochs; ++epoch) {
            for (int batchIdx = 0; batchIdx < trainIterations; ++batchIdx) {
                Map<String, Object> inputs = new HashMap<>();
                inputs.put("data", trainImageBatches.get(batchIdx));
                inputs.put("label", trainLabelBatches.get(batchIdx));

                Map<String, Object> outputs = new HashMap<>();
                FloatBuffer loss = FloatBuffer.allocate(1);
                outputs.put("loss", loss);

                interpreter.runSignature(inputs, outputs, "train");
                currentLoss = loss.get(0);

                if (epochTrainLoss == 0) {
                    epochTrainLoss = currentLoss;
                } else {
                    epochTrainLoss = (1 - lossDecay) * epochTrainLoss +
                            lossDecay * currentLoss;
                }

                Log.i(
                        "Train",
                        String.format(
                                "[train][epoch][%d][batch][%d][loss][%f]",
                                epoch, batchIdx, currentLoss
                        )
                );
            }
        }
        Map<String, Object> inputs = new HashMap<>();
        File outputFile = new File(directory + "/model.ckpt");
        inputs.put("checkpoint_path", outputFile.getAbsolutePath());
        Map<String, Object> outputs = new HashMap<>();
        interpreter.runSignature(inputs, outputs, "save");

        Map<String, Object> results = new HashMap<>();
        results.put("client_id", clientId);
        results.put("moving_loss", epochTrainLoss);
        results.put("trained_size", trainEpochs * dataCount);
        results.put("success", true);
        byte[] newModel = Common.readBytes(new FileInputStream(outputFile.getAbsolutePath()));
        results.put("update_weight", newModel);
        results.put("utility", currentLoss * currentLoss * dataCount);
        results.put("wall_duration", 0);
        return results;
    }

    @Override
    public Map<String, Object> MLTest(
            String directory,
            ByteBuffer model,
            JSONObject testingDataConf,
            JSONObject testingConf) throws JSONException, IOException {
        final String testImagesFolder = directory + "/" + testingDataConf.getString("data" ) + "/";
        final String testImagesTxt = directory + "/" + testingDataConf.getString("label");

        final int numClasses        = testingConf.getInt("num_classes");
        final int testBatchSize     = testingConf.getInt("test_bsz");
        final int testNumWorkers    = testingConf.getInt("num_workers");
        final int channel           = testingConf.getInt("channel");
        final int height            = testingConf.getInt("height");
        final int width             = testingConf.getInt("width");

        List<String> labels = FileUtil.loadLabels(new FileInputStream(testImagesTxt));
        final int dataCount = labels.size();
        final int testIterations = (int)Math.ceil((double)dataCount / (double)testBatchSize);

        Pair<List<float[][]>, List<float[][]>> pair = dataLoader(
                labels,
                testImagesFolder,
                testBatchSize,
                height,
                width,
                channel,
                numClasses);
        List<float[][]> testImageBatches = pair.first;
        List<float[][]> testLabelBatches = pair.second;

        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(testNumWorkers);
        Interpreter interpreter = new Interpreter(model, options);

        float testLoss = 0;
        int correctTop1 = 0;
        int correctTop5 = 0;
        int sampleCount = 0;

        for (int batchIdx = 0; batchIdx < testIterations; ++batchIdx) {
            Map<String, Object> inputs = new HashMap<>();
            inputs.put("data", testImageBatches.get(batchIdx));
            inputs.put("label", testLabelBatches.get(batchIdx));

            Map<String, Object> outputs = new HashMap<>();
            FloatBuffer loss = FloatBuffer.allocate(1);
            outputs.put("loss", loss);
            IntBuffer top1 = IntBuffer.allocate(1);
            outputs.put("top1", top1);
            IntBuffer top5 = IntBuffer.allocate(1);
            outputs.put("top5", top5);

            interpreter.runSignature(inputs, outputs, "test");

            testLoss += loss.get(0);
            correctTop1 += top1.get(0);
            correctTop5 += top5.get(0);
            sampleCount += testImageBatches.get(batchIdx).length;

            Log.i(
                    "Test",
                    String.format(
                            "[test][batch][%d][loss][%f][top 1][%d/%d=%f%%][top 5][%d/%d=%f%%]",
                            batchIdx, testLoss,
                            correctTop1, sampleCount, (float)correctTop1 / (float)sampleCount * 100,
                            correctTop5, sampleCount, (float)correctTop5 / (float)sampleCount * 100
                    )
            );
        }
        float accuracyFinalTop1 = (float)correctTop1 / dataCount;
        float accuracyFinalTop5 = (float)correctTop5 / dataCount;

        Map<String, Object> results = new HashMap<>();
        results.put("top_1", accuracyFinalTop1);
        results.put("top_5", accuracyFinalTop5);
        results.put("test_loss", testLoss);
        results.put("test_len", dataCount);
        return results;
    }

    private Pair<List<float[][]>, List<float[][]>> dataLoader(
            List<String> labels,
            String imagesFolder,
            int batchSize,
            int height,
            int width,
            int channel,
            int numClasses
    ) {
        List<List<float[]>> imageBatches = new ArrayList<>();
        List<List<float[]>> labelBatches = new ArrayList<>();
        for (String label: labels) {
            final String imageFileName = imagesFolder + label.split(" ")[0];
            final float[] imageBuffer = processImage(
                    BitmapFactory.decodeFile(imageFileName), height, width);
            final float[] imageLabel = encodeLabel(
                    Integer.parseInt(label.split(" ")[1]), numClasses);
            int currentLastIdx = imageBatches.size() - 1;
            if (imageBatches.size() == 0 || imageBatches.get(currentLastIdx).size() == batchSize) {
                List<float[]> newImageBatch = new ArrayList<>();
                imageBatches.add(newImageBatch);
                List<float[]> newLabelBatch = new ArrayList<>();
                labelBatches.add(newLabelBatch);
                currentLastIdx++;
            }
            imageBatches.get(currentLastIdx).add(imageBuffer);
            labelBatches.get(currentLastIdx).add(imageLabel);
        }
        return new Pair<>(
                this.generateBatchBuffers(imageBatches, channel * width * height),
                this.generateBatchBuffers(labelBatches, numClasses));
    }

    private List<float[][]> generateBatchBuffers(
            List<List<float[]>> batches,
            int allocateSizePerData) {
        List<float[][]> batchBuffers = new ArrayList<>();
        for (List<float[]> batch: batches) {
            float[][] newBatchBuffer = new float[batch.size()][allocateSizePerData];
            int iter = 0;
            for (float[] data: batch) {
                newBatchBuffer[iter++] = data;
            }
            batchBuffers.add(newBatchBuffer);
        }
        return batchBuffers;
    }

    // Preprocess the image and convert it into a TensorImage for classification.
    private float[] processImage(
            Bitmap image,
            int targetHeight,
            int targetWidth) {
        int height = image.getHeight();
        int width = image.getWidth();
        int cropSize = min(height, width);
        ImageProcessor.Builder builder = new ImageProcessor.Builder();
        ImageProcessor imageProcessor = builder
                .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                .add(new ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(0f, 255f))
                .build();
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(image);
        return imageProcessor.process(tensorImage).getTensorBuffer().getFloatArray();
    }


    // encode the classes name to float array
    private float[] encodeLabel(int id, int numClasses) {
        float[] classArray = new float[numClasses];
        Arrays.fill(classArray, 0f);
        classArray[id] = 1f;
        return classArray;
    }
}