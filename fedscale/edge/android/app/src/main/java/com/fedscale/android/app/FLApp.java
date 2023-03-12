package com.fedscale.android.app;

import android.content.Context;
import android.content.Intent;
import android.content.res.Resources;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.fedscale.android.R;
import com.fedscale.android.executor.Client;
import com.fedscale.android.utils.Common;

import java.util.HashMap;
import java.util.Map;

public class FLApp extends AppCompatActivity {

    private TextView mUserId;
    private TextView mRound;
    private TextView mExecuteStatus;
    private TextView mExecuteResult;
    private TextView mExecuteMsg;

    private Button flButton;
    private Button finetuneButton;

    private Resources res;

    HandlerThread mThread;
    Handler mHandle;

    private Client executor;

    private Boolean hasModel;
    private Boolean dataInitialized;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);

        this.mUserId = findViewById(R.id.userId);
        this.mRound = findViewById(R.id.executeRound);
        this.mExecuteStatus = findViewById(R.id.executeStatus);
        this.mExecuteResult = findViewById(R.id.executeResult);
        this.mExecuteMsg = findViewById(R.id.executeMsg);

        this.flButton = findViewById(R.id.flButton);
        this.finetuneButton = findViewById(R.id.finetuneButton);

        this.res = getResources();

        this.executor = new Client(this);

        this.hasModel = false;
        this.dataInitialized = false;

        this.flButton.setOnClickListener(v -> {
            if (this.flButton.getText().equals(this.res.getString(R.string.start_fl))) {
                if (!this.dataInitialized) {
                    try {
                        this.executor.initExecutor();
                        this.dataInitialized = true;
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                if (this.finetuneButton.isEnabled()) {
                    this.finetuneButton.setEnabled(false);
                }
                this.mThread = new HandlerThread("FL");
                this.mThread.start();
                this.mHandle = new Handler(this.mThread.getLooper());
                this.mHandle.post(() -> {
                    try {
                        this.executor.FLStart();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                });
                this.flButton.setText(this.res.getString(R.string.stop_fl));
            } else if (this.flButton.getText().equals(this.res.getString(R.string.stop_fl))) {
                try {
                    this.executor.FLStop();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                this.flButton.setText(this.res.getString(R.string.start_fl));
                if (this.hasModel) {
                    this.finetuneButton.setEnabled(true);
                }
            }
        });

        this.finetuneButton.setOnClickListener(v -> {
            this.flButton.setEnabled(false);
            this.mThread = new HandlerThread("Fine-tune");
            this.mThread.start();
            this.mHandle = new Handler(this.mThread.getLooper());
            this.mHandle.post(() -> {
                try {
                    this.executor.LocalTrain();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
            this.finetuneButton.setEnabled(false);
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
    public void onBackPressed() {
        Intent home = new Intent(Intent.ACTION_MAIN);
        home.addCategory(Intent.CATEGORY_HOME);
        startActivity(home);
    }

    @Override
    protected void onDestroy() {
        try {
            this.executor.FLStop();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        super.onDestroy();
    }

    /**
     * Change Status on User Interface.
     *
     * @param newStatus The value of the updated status.
     */
    public void onChangeStatus(final String newStatus){
        runOnUiThread(() -> {
            this.mExecuteStatus.setText(newStatus);
            if (newStatus.equals(Common.CLIENT_TRAIN_LOCALLY_FIN)) {
                this.finetuneButton.setEnabled(true);
                this.flButton.setEnabled(true);
            } else if (newStatus.equals(Common.SHUT_DOWN)) {
                this.flButton.setText(this.res.getString(R.string.start_fl));
                if (this.hasModel) {
                    this.finetuneButton.setEnabled(true);
                }
            }
        });
    }

    /**
     * Change Round on User Interface.
     *
     * @param newRound The value of the updated round.
     */
    public void onChangeRound(final int newRound){
        runOnUiThread(() -> this.mRound.setText(String.format("Round: %d", newRound)));
    }

    /**
     * When Executor get model from cloud, the local finetune button will be enabled when FL stop.
     */
    public void onGetModel(){
        runOnUiThread(() -> this.mExecuteMsg.setText(this.res.getString(R.string.has_model)));
        this.hasModel = true;
    }

    /**
     * When Executor write model to the disk, the local finetune button will be disabled when FL stop.
     */
    public void onWriteModel() {
        runOnUiThread(() -> {
            this.mExecuteMsg.setText(this.res.getString(R.string.no_model));
            this.finetuneButton.setEnabled(false);
        });
        this.hasModel = false;
    }

    /**
     * Initialization of the User Interface.
     * Three lines of UI: executor id, status, result (not changed)
     */
    public void initUI(String userIdMsg) {
        runOnUiThread(()->{
            this.mUserId.setText(userIdMsg);
            this.mExecuteStatus.setText(Common.CLIENT_CONNECT);
            this.mExecuteResult.setText("");
        });
    }
}
