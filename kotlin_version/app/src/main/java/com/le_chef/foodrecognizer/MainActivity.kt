package com.le_chef.foodrecognizer

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.graphics.*;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;
import androidx.exifinterface.media.ExifInterface;
import androidx.lifecycle.lifecycleScope;
import kotlinx.coroutines.Dispatchers;
import kotlinx.coroutines.launch;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;
import kotlin.math.max;
import kotlin.math.min;



class MainActivity : AppCompatActivity(), View.OnClickListener {

    companion object{
        const val TAG = "TFLite - ODT"
        const val REQUEST_IMAGE_CAPTURE: Int = 1
        private const val MAX_FONT_SIZE = 96F
    }

    private lateinit var captureImageFab:Button
    private lateinit var inputImageView: ImageView
    private lateinit var imgSampleOne: ImageView
    private lateinit var imgSampleTwo: ImageView
    private lateinit var imgSampleThree: ImageView
    private lateinit var tvPlaceholder: TextView
    private lateinit var currentPhotoPath: String

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputImageView = findViewById(R.id.imageView)
        captureImageFab = findViewById(R.id.captureImageFab)
        imgSampleOne = findViewById(R.id.imgSampleOne)
        imgSampleTwo = findViewById(R.id.imgSampleTwo)
        imgSampleThree = findViewById(R.id.imgSampleThree)
        tvPlaceholder  = findViewById(R.id.tvPlaceholder)

        captureImageFab.setOnClickListener(this)
        imgSampleOne.setOnClickListener(this)
        imgSampleTwo.setOnClickListener(this)
        imgSampleThree.setOnClickListener(this)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode == REQUEST_IMAGE_CAPTURE && resultCode == Activity.RESULT_OK){
            setViewAndDetect(getCapturedImage())
        }
    }
    override fun onClick(v: View?){
        when (v?.id){
            R.id.captureImageFab ->{
                try {
                    dispatchTakePictureIntent()
                }catch (e:ActivityNotFoundException){
                    Log.e(TAG, e.message.toString())
                }
            }
            R.id.imgSampleOne->{
                setViewAndDetect(getSampleImage(R.drawable.img_meal_one))
            }
            R.id.imgSampleTwo ->{
                setViewAndDetect(getSampleImage(R.drawable.img_meal_two))
            }
            R.id.imgSampleThree ->{
                setViewAndDetect(getSampleImage(R.drawable.img_meal_three))
            }
        }
    }

    private fun setViewAndDetect(bitmap: Bitmap){
        inputImageView.setImageBitmap(bitmap)
        tvPlaceholder.visibility = View.INVISIBLE
        lifecycleScope.launch(Dispatchers.Default){runObjectDetection(bitmap)}
    }

    private fun runObjectDetection(bitmap : Bitmap){
        val image = TensorImage.fromBitmap(bitmap)
        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setMaxResults(5)
            .setScoreThreshold(0.5f)
            .build()
        val detector = ObjectDetector.createFromFileAndOptions(
            this,
            "salad_colab1.tflite",
            options
        )
        val results = detector.detect(image)
        val resultToDisplay = results.map{
            val category = it.categories.first()
            val text = "${category.label}, ${category.score.times(100).toInt()}%"
            DetectionResult(it.boundingBox, text)
        }

        val imgWithResult = drawDetectionResult(bitmap, resultToDisplay)
        runOnUiThread{
            inputImageView .setImageBitmap(imgWithResult)
        }
        debugPrint(results)
    }

    private fun debugPrint(results: List<Detection>){
        for((i, obj) in results.withIndex()){
            val box = obj.boundingBox
            Log.d(TAG, "Detected object: ${i}")
            Log.d(TAG, "boundingBox: (${box.left}, ${box.top})")
            for((j, category) in obj.categories.withIndex()){
                Log.d(TAG,"     Label$j: ${category.label}")
                val confidence: Int = category.score.times(100).toInt()
                Log.d(TAG, "    confidence: ${confidence}%")
            }
        }
    }

    private fun getCapturedImage(): Bitmap{
        val targetW: Int = inputImageView.width
        val targetH: Int = inputImageView.height

        val bmOptions = BitmapFactory.Options().apply{
            inJustDecodeBounds = true
            BitmapFactory.decodeFile(currentPhotoPath, this)
            val photoW: Int = outWidth
            val photoH: Int = outHeight

            val scaleFactor: Int = max(1, min(photoW/targetW, photoH/targetH))
            inJustDecodeBounds =  false
            inSampleSize = scaleFactor
            inMutable = true
        }
        val exifInterface=ExifInterface(currentPhotoPath)
        val orientation = exifInterface.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_UNDEFINED
        )

        val bitmap = BitmapFactory.decodeFile(currentPhotoPath, bmOptions)
        return when(orientation){
            ExifInterface.ORIENTATION_ROTATE_90 ->{
                rotateImage(bitmap, 90f)
            }
            ExifInterface.ORIENTATION_ROTATE_180 ->{
                rotateImage(bitmap, 190f)
            }
            ExifInterface.ORIENTATION_ROTATE_270 ->{
                rotateImage(bitmap, 270f)
            }
            else ->{
                bitmap
            }
        }
    }


    private fun getSampleImage(drawable: Int) : Bitmap{
        return BitmapFactory.decodeResource(resources, drawable, BitmapFactory.Options().apply{
            inMutable = true
        })
    }

    private fun rotateImage(source : Bitmap, angle: Float):Bitmap{
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(
            source, 0, 0, source.width, source.height,
            matrix, true
        )
    }

    @Throws(IOException::class)
    private fun createImageFile():File{
        val timeStamp: String = SimpleDateFormat("yyyMMdd_HHmmss").format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
            "JPEG_${timeStamp}_",
            ".jpg",
            storageDir
        ).apply{
            currentPhotoPath = absolutePath
        }
    }


    private fun dispatchTakePictureIntent(){
        Intent (MediaStore.ACTION_IMAGE_CAPTURE).also{takePictureIntent ->
            takePictureIntent.resolveActivity(packageManager)?.also{
                val photoFile: File? = try{
                    createImageFile()
                }catch(e:IOException){
                    Log.e(TAG, e.message.toString())
                    null
                }
                photoFile?.also{
                    val photoURI: Uri = FileProvider.getUriForFile(
                        this,
                        "com.le_chef.foodrecognizer.fileprovider",
                        it
                    )
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                    startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
                }
            }
        }
    }

    private fun drawDetectionResult(
        bitmap: Bitmap,
        detectionResults: List<DetectionResult>
    ): Bitmap{
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)
        val pen = Paint()
        pen.textAlign = Paint.Align.LEFT
        detectionResults.forEach{
            pen.color = Color.RED
            pen.strokeWidth = 8F
            pen.style = Paint.Style.STROKE
            val box = it.boundingBox
            canvas.drawRect(box, pen)

            val tagSize = Rect(0, 0, 0, 0)
            pen.style = Paint.Style.FILL_AND_STROKE
            pen.color = Color.YELLOW
            pen.strokeWidth = 2f

            pen.textSize = MAX_FONT_SIZE
            pen.getTextBounds(it.text, 0, it.text.length, tagSize)
            val fontSize: Float = pen.textSize * box.width() / tagSize.width()
            if(fontSize < pen.textSize) pen.textSize = fontSize
            var margin = (box.width() - tagSize.width()) / 2.0F
            if(margin < 0F) margin = 0F


            canvas.drawText(
                it.text, box.left + margin,
                box.top + tagSize.height().times(1f), pen
            )
        }
        return outputBitmap
    }

    data class DetectionResult(val boundingBox: RectF, val text: String)

}