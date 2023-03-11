# FedScale Deployment

FedScale provides a cloud-based [aggregation service](https://github.com/SymbioticLab/FedScale/blob/master/fedscale/cloud/aggregation/README.md) and an [SDK](#fedscale-mobile-runtime) for smartphones on the edge that currently supports TensorflowLite and Alibaba MNN on Android (iOS support coming soon!). In this tutorial, we introduce how to:

- [Initiate FedScale Cloud Service](#fedscale-cloud-aggregation)
- [Import FedScale SDK to locally fine tune models](#fedscale-mobile-runtime)
- [Connect to FedScale cloud for federated training](#fedscale-mobile-runtime)

<p align="center">
<img src="../../../docs/fedscale-deploy.png" width="600" height="400"/>
</p>



## FedScale Cloud Aggregation
FedScale cloud aggregation orchestrates distributed mobile devices to collaboratively train ML models over the Internet. It manages client check-in, participant selection, and model aggregation for practical FL deployment. 

FedScale deployment mode follows similar setup of the [simulation mode](https://github.com/SymbioticLab/FedScale/blob/master/docs/tutorial.md) to streamline cloud-based prototyping and real-world deployment with little migration overhead. 

- **Configurate job**: Jobs are configured in the `yml` format. Here is an [example](../../../benchmark/configs/android/tflite.yml
): 

  ```
  job_conf:
      - job_name: android-tflite  
      - experiment_mode: mobile
      - log_path: $FEDSCALE_HOME/benchmark    # Path of log files
      - num_participants: 100                 # Number of participants selected in each training round
      - model: mobilenetv3                    # Model to be trained
      - data_path: assets/dataset             # Path to local database
      - input_shape: 32 32 3                  # Shape of training data stored in local database
      - num_classes: 10                       # Number of categories 
  ```

- **Submit job:** After figuring out the configuration, we can submit the FL training job in the cloud, which then will automatically coordinate edge clients. 

  ```
  cd $FEDSCALE_HOME/docker
  # If you want to run MNN backend on mobile.
  python3 driver.py submit $FEDSCALE_HOME/benchmark/configs/android/mnn.yml 
  # If you want to run TFLite backend on mobile.
  python3 driver.py submit $FEDSCALE_HOME/benchmark/configs/android/tflite.yml 
  ```

- **Check logs:** FedScale will generate logs under `data_path` you provided by default. If you use k8s deployment for cloud aggregation, keep in mind that k8s may load balancing your job to any node on the cluster, so make sure you are checking the `data_path` on the correct node.

- **Stop job:** When FL training reaches the target accuracy, we can stop FL training with the following command of line on the cloud server node.

  ```
  cd $FEDSCALE_HOME/docker
  python3 driver.py stop $YOUR_JOB
  ```

## FedScale Mobile Runtime

If you don't have an app, you may refer to [Sample App](README-App.md) to play with a sample Android app. Next, we introduce how to: 
- Train/test models with TFLite or Alibaba MNN.
- Fine-tune models locally **after** receiving model from the cloud.

To get started, you need to install the FedScale SDK and import it into your project.
Once you have installed the SDK, you can add ``fedscale_client`` to your app with the following code to fine-tune your local model: 

  ```
  import com.fedscale.android.Client;
  public class App {
      …
      private Client fedscale_client;
      protected void onCreate() {
          …
          this.fedscale_client = new Client();
          this.fedscale_client.run(); // run in background threads
      }
  }
  ```

For example, our [example app](README-App.md) uses an image classification model within the app. Our example app puts training data under ``assets/dataset``. When the user opens the app, ``fedscale_client`` carefully schedules the resource to decide whether to start fine-tuning. 

----
If you need any further help, feel free to contact FedScale team or the developer [website](https://continue-revolution.github.io) [email](mailto:continuerevolution@gmail.com) of this app.