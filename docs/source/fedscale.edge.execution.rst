fedscale.edge.execution
=======================

`This directory <https://github.com/SymbioticLab/FedScale/tree/master/fedscale/edge/mnn>`__ contains minimum files modified from `MNN Android
Demo <https://github.com/alibaba/MNN/tree/master/project/android/demo>`__.
The training and testing will be conducted by MNN C++ backend, while the
task execution and communication with server will be managed by Java.
The sample has been tested upon image classification with a simple
linear model and a small subset of
`ImageNet-MINI <https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000>`__.
This documentation contains a step-by-step tutorial on how to download,
build and config this app on your own device, and modify this app for
your own implementation and deployment.

Download and build sample android app
-------------------------------------

1. Download and unzip `sample dataset
   (TrainTest.zip) <https://drive.google.com/file/d/1nfi3SVzjaE0LPxwj_5DNdqi6rK7BU8kb/view?usp=sharing>`__
   to ``assets/`` directory. Remove ``TrainTest.zip`` after unzip to
   save space on your mobile device. After unzip, you should see 3 files
   and 2 directories under ``assets/``:

   1. ``TrainSet``: Training set directory, contains 316 images.
   2. ``TestSet``: Testing set directory, contains 34 images.
   3. ``conf.json``: Configuration file for mobile app.
   4. ``train_labels.txt``: Training label file with format
      ``<filename> <label>``, where ``<filename>`` is the path after
      ``TrainSet/``.
   5. ``test_labels.txt``: Testing label file with the same format as
      ``train_labels.txt``.

2. Run ``sh install.sh``. If you are on Windows, please run this command
   in WSL. This step will clone MNN, replace `MNN Android
   Demo <https://github.com/alibaba/MNN/tree/master/project/android/demo>`__
   with FedScale Android Executor under ``MNN/project/android/demo``.
   **Note**: you may want to move MNN directory to another directory,
   otherwise the full file path might be too long for cmake to build.
3. Install `Android Studio <https://developer.android.com/studio>`__ and
   open project ``MNN/project/android/demo``. Download necessary SDKs,
   NDKs and CMake when prompted. My version:

   -  SDK: API 28
   -  NDK: 20.0.5594570
   -  Android Gradle Plugin Version: 3.5.3
   -  Gradle Version: 5.4.1
   -  Source Compatibility: Java 8
   -  Target Compatibility: Java 8

4. Make project. Android Studio will compile and build the app for you.

Test this app with default setting
----------------------------------

1. ssh to your own server and run

::

   cd fedscale/cloud/aggregation/android
   python3 android_aggregator.py --experiment_mode=mobile --num_participants=1 --model=linear

2. Change aggregator IP address inside ``assets/conf.json`` and click
   ``Run`` inside Android Studio.

Customize your own app
----------------------

1. If you want to use your own dataset, please put your data under
   ``assets/TrainSet`` and ``assets/TestSet``, make sure that your label
   has the same format as my label file.

   1. If you want to change the file/dir name under ``assets``, please
      make sure to change the corresponding config in ``assets``
      attribute inside ``assets/conf.json``.

2. If you want to use your own model for **image classification**,
   please either change ``channel``, ``width`` and ``height`` inside
   ``assets/conf.json`` to your own input and change ``num_classes`` to
   your own classes, or override these attributes when sending
   ``CLIENT_TRAIN`` request.
3. If you want to use your own model for tasks other than image
   classification, you may need to write your own JNI C++ trainer and
   tester. Please refer to `MNN <https://github.com/alibaba/MNN>`__ for
   further development guide. You may also need to change ``channel``,
   ``width`` and ``height`` inside ``assets/conf.json`` to your own
   input and change or remove ``num_classes``.
4. Please note that MNN does not support DropOut and will SegFault
   during training. MNN may have other potential bugs.