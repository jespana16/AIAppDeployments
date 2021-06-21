package com.example.imagecaptioning

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.drawable.BitmapDrawable
import android.net.ConnectivityManager
import android.net.NetworkInfo
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Base64
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.android.volley.Response
import com.android.volley.toolbox.JsonObjectRequest
import com.android.volley.toolbox.Volley
import com.loopj.android.http.AsyncHttpClient.LOG_TAG
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.lang.Exception

class MainActivity : AppCompatActivity() {
    lateinit var imageView: ImageView
    lateinit var buttonGalery: Button
    lateinit var buttonPhoto: Button
    lateinit var buttonPredict: Button
    lateinit var textViewImage: TextView
    lateinit var textViewPrediction: TextView
    private val pickImage = 100
    private var imageUri: Uri? = null
    val REQUEST_IMAGE_CAPTURE = 42
    val URL = "https://imagecaptioning-317203.rj.r.appspot.com/predict_imageCaptioning"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        imageView = findViewById(R.id.imageView)
        buttonGalery = findViewById(R.id.btnImg)
        buttonPhoto = findViewById(R.id.btnImg2)
        buttonPredict = findViewById(R.id.btnPre)
        textViewImage = findViewById(R.id.textViewImage)
        textViewPrediction = findViewById(R.id.textViewPredition)

        val connectionManager : ConnectivityManager = this.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val activeNetwork: NetworkInfo? = connectionManager.activeNetworkInfo
        val isConnected: Boolean = activeNetwork?.isConnectedOrConnecting == true

        if(ContextCompat.checkSelfPermission(applicationContext,android.Manifest.permission.CAMERA)
            ==PackageManager.PERMISSION_DENIED)
                ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.CAMERA), REQUEST_IMAGE_CAPTURE)

        buttonGalery.setOnClickListener {
            val gallery = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI)
            startActivityForResult(gallery, pickImage)
        }

        buttonPhoto.setOnClickListener {
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            if (takePictureIntent.resolveActivity(this.packageManager) != null){
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }

        buttonPredict.setOnClickListener{
            val activeNetwork: NetworkInfo? = connectionManager.activeNetworkInfo
            val isConnected: Boolean = activeNetwork?.isConnectedOrConnecting == true

            if(isConnected){
                println("imageView: " + imageView.getDrawable())
                if(imageView.getDrawable() !=null) {

                    try {
                        val queue = Volley.newRequestQueue(this)
                        val img = imageView.getDrawable()

                        val bitmap = (img as BitmapDrawable).getBitmap()
                        val stream = ByteArrayOutputStream()
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream)
                        val image: ByteArray = stream.toByteArray()

                        val jsonObject = JSONObject()
                        jsonObject.put("img", image)

                        val jsonObjectRequest = JsonObjectRequest(URL, jsonObject,
                            Response.Listener { response ->
                                Log.i(LOG_TAG, "Response is: $response")
                                textViewPrediction.setText(response.get("title").toString())
                            },
                            Response.ErrorListener { error ->
                                textViewPrediction.setText(error.message)
                            }
                        )
                        queue.add(jsonObjectRequest)
                    }catch (e: Exception){
                        Toast.makeText(applicationContext, e.message, Toast.LENGTH_LONG)
                    }
                }else{
                    textViewImage.setText("Debe cargar una imagen")
                    textViewImage.setTextColor(Color.parseColor("#FF0000"))
                }
            }else{
                textViewPrediction.setText("Debe conectarse a internet")
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK && requestCode == pickImage) {
            imageUri = data?.data
            imageView.setImageURI(imageUri)
        }else if(requestCode == REQUEST_IMAGE_CAPTURE && resultCode == Activity.RESULT_OK) {
            val takeImage:Bitmap = data?.extras?.get("data") as Bitmap
            imageView.setImageBitmap(takeImage)
        }
        textViewImage.setVisibility(View.INVISIBLE)
    }
}