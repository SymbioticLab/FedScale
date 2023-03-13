## FedScale Example Mobile App

We provide a sample app which you can choose to 
- Train/test models with TFLite or Alibaba MNN.
- Fine-tune models locally **after** receiving model from the cloud.

Please follow these steps to download and build the sample android app.

1. Download and unzip [sample dataset (TrainTest.zip)](https://drive.google.com/file/d/1nfi3SVzjaE0LPxwj_5DNdqi6rK7BU8kb/view?usp=sharing) to `assets/` directory. Remove `TrainTest.zip` after unzip to save space on your mobile device. After unzip, you should see 3 files and 2 directories under `assets/`:
   1. `TrainSet`: Training set directory, contains 320 images.
   2. `TestSet`: Testing set directory, contains 32 images.
   3. `conf.json`: Configuration file for mobile app.
   4. `train_labels.txt`: Training label file with format `<filename> <label>`, where `<filename>` is the path after `TrainSet/`.
   5. `test_labels.txt`: Testing label file with the same format as `train_labels.txt`.
2. Install [Android Studio](https://developer.android.com/studio) and open project `fedscale/edge/tflite`. Download necessary SDKs, NDKs and CMake when prompted. My version:
    - SDK: API 32
    - Android Gradle Plugin Version: 3.5.3
    - Gradle Version: 5.4.1
    - Source Compatibility: Java 8
    - Target Compatibility: Java 8
3. Modify `conf.json` and dataset.
   - You must modify `aggregator.ip` & `aggregator.port` to your own server.
   - You can choose your training framework by modifying `model_conf.backend` to `tflite` or `mnn`.
   - You may config your dataset information at `training_conf` and `testing_conf`.
   - You may put your own **image classification** dataset under `/TrainSet` and `/TestSet` directories and modify `train_labels.txt` and `test_labels.txt` accordingly. The format of labels must be \<filename\> \<label\>.
   - If you want to perform tasks other than image classification, you should modify framework-specific code for [MNN](https://github.com/SymbioticLab/FedScale/fedscale/edge/android/app/src/main/java/com/fedscale/android/mnn) or [TFLite](https://github.com/SymbioticLab/FedScale/fedscale/edge/android/app/src/main/java/com/fedscale/android/tflite). If you are using TFLite, you should also write your own signatures similar to [our TFLite model provider](https://github/com/SymbioticLab/FedScale/fedscale/cloud/internal/tflite_model_adapter.py)
4. Make Project. Android Studio will compile and build the app for you. Click Run if you want to run the app on a real android device.

*Note that the data paths of training data in the mobile devices and corresponding schemas are configurable when users submit jobs in the cloud to train or fine-tune a particular model (i.e., by updating ``- data_path`` in yml).* 

----
If you need any further help, feel free to contact FedScale team or the developer [website](https://continue-revolution.github.io) [email](mailto:continuerevolution@gmail.com) of this app.
