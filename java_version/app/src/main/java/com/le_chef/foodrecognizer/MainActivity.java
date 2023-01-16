package com.le_chef.foodrecognizer;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.service.controls.templates.ThumbnailTemplate;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;


import com.le_chef.foodrecognizer.ml.Fruits1;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    Button camera;
    ImageView imageView;
    TextView tvPlaceHolder;
    int image_size = 32; //android camera uses 32 so resize everything to 32

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.captureImageFab);
        imageView = findViewById(R.id.imageView);
        tvPlaceHolder = findViewById(R.id.tvPlaceholder);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                }
                else{
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }

            }
        });
        /**
        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, 1);
            }
        });
         */
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK) {
            if (resultCode == 3) {//camera capture access
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimensions = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimensions, dimensions);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, image_size, image_size, false);
                classifyImage(image);
            }
            else if(resultCode == 1){//gallery access
                Uri dat = data.getData();
                Bitmap image = null;
                try{
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                }
                catch(IOException e){
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, image_size, image_size, false);
                classifyImage(image);
            }
            else{//image_one, two and three is selected instead.

            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    public void classifyImage(Bitmap image)
    {
        try {
            Fruits1 model = Fruits1.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4* image_size * image_size * 3);//4 -floats * float_pixels^squared * 3 rgb on each image
            inputFeature0.loadBuffer(byteBuffer);

            int[] intValues = new int[image_size*image_size];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0 ;
            for(int i = 0; i < image_size; ++i){
                for(int j = 0; i < image_size; ++j){
                    int val = intValues[++pixel];//RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));//change to 1.f/255 if model didn't rescale in python
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));//change to 1.f/255 if model didn't rescale in python
                    byteBuffer.putFloat(((val >> 0xFF) & 0xFF) * (1.f / 1));//change to 1.f/255 if model didn't rescale in python
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Fruits1.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            //get position of highest confidence
            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0 ;
            float maxConfidence = 0;
            for(int i = 0 ; i < confidences.length; ++i){
                if(confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos=i;
                }
            }
            //classes that I used in model creation
            String[] classes = {"Apple", "Banana", "Orange"};
            //display results
            tvPlaceHolder.setText(classes[maxPos]);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}




















