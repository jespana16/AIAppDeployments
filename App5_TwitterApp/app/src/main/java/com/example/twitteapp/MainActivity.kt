package com.example.twitteapp

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.text.nlclassifier.NLClassifier
import java.util.*

class MainActivity : AppCompatActivity() {
    private lateinit var model: NLClassifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        model = NLClassifier.createFromFile(applicationContext, "model.tflite")

        val txtSentence = findViewById<TextView>(R.id.editTextTextMultiLine)
        val btnClasificar = findViewById<TextView>(R.id.button)

        btnClasificar.setOnClickListener(){
            val prediction: List<Category> = model.classify(txtSentence.text.toString())

            var mayor_pre = 0.0
            var label_pre = ""

            for (item in prediction){
                if(item.score.toDouble()>mayor_pre){
                    mayor_pre = item.score.toDouble()
                    label_pre = item.label
                }
            }
            Toast.makeText(applicationContext, "Topic: " + label_pre, Toast.LENGTH_LONG).show()
        }
    }
}