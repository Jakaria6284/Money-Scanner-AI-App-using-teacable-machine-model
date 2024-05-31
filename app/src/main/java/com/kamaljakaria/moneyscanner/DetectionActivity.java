package com.kamaljakaria.moneyscanner;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.airbnb.lottie.LottieAnimationView;

import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class DetectionActivity extends AppCompatActivity {

    TextView result, confidence, Classified, confidancetxt;
    ImageView imageView;
    Button  upload,SeeDetail;
    LottieAnimationView lottieAnimationView;
    int imageSize = 224;
    byte[] byteArray;
    int[] intValues = new int[imageSize * imageSize];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detection);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        Classified = findViewById(R.id.classified);
        confidancetxt = findViewById(R.id.confidencesText);
        imageView = findViewById(R.id.imageView);

        upload = findViewById(R.id.button2);
        lottieAnimationView=findViewById(R.id.imageView2);
       // SeeDetail=findViewById(R.id.detail);








        upload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Open gallery to select an image
                lottieAnimationView.setVisibility(View.GONE);
                imageView.setVisibility(View.VISIBLE);
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, 2);


            }
        });







    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == RESULT_OK) {
            // Capture image from camera
            Bitmap image = (Bitmap) data.getExtras().get("data");
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            image.compress(Bitmap.CompressFormat.PNG, 100, stream);
            byteArray = stream.toByteArray();

            processAndClassifyImage(image);
        } else if (requestCode == 2 && resultCode == RESULT_OK && data != null) {
            // Select image from gallery
            Uri selectedImage = data.getData();
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                processAndClassifyImage(bitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void processAndClassifyImage(Bitmap image) {
        int dimension = Math.min(image.getWidth(), image.getHeight());
        image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
        imageView.setImageBitmap(image);

        image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
        classifyImage(image);
    }

    public void classifyImage(Bitmap image) {
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(1); // Use a single thread for consistency
            Interpreter model = new Interpreter(loadModelFile(), options);

            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            inputBuffer.order(ByteOrder.nativeOrder());
            preprocessImage(image, inputBuffer);

            // Log input buffer values for debugging
            inputBuffer.rewind();
            for (int i = 0; i < 10; i++) { // Log first 10 floats as example
                Log.d("InputBuffer", "Value at " + i + ": " + inputBuffer.getFloat());
            }

            float[][] output = new float[1][11]; // Adjust the size according to your model output
            model.run(inputBuffer, output);

            int maxIndex = argmax(output[0]);
            float confidenceValue = output[0][maxIndex];
            String[] classes = {"1 Taka","2 Taka","5 Taka","10 Taka","20 Taka","50 Taka"," 100 Taka","200 Taka","500 Taka","1000 Taka","Bank Coin/Something Other"};
            String predictedClass = classes[maxIndex];

            result.setText(predictedClass);
            confidence.setText(String.format("%.1f%%", confidenceValue * 100));

            model.close();
        } catch (IOException e) {
            Log.e("Model", "Error loading model: " + e.getMessage());
        }
    }

    private void preprocessImage(Bitmap bitmap, ByteBuffer buffer) {
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < imageSize; ++i) {
            for (int j = 0; j < imageSize; ++j) {
                final int val = intValues[pixel++];
                buffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                buffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                buffer.putFloat((val & 0xFF) / 255.0f);
            }
        }
    }

    private int argmax(float[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private ByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("model_unquant.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
