package com.example.machinelearning

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxTensor.createTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle

import java.io.FileInputStream
import java.nio.FloatBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val ortEnvironment = OrtEnvironment.getEnvironment()
        val ortSession = createORTSession( ortEnvironment )
        val inputs= floatArrayOf(3f,78f,50f,32f,88f,31f,0.248f,26f)
        val output = runPrediction( inputs , ortSession , ortEnvironment )
println(output)


    }
    private fun createORTSession( ortEnvironment: OrtEnvironment ) : OrtSession {
        val modelBytes = resources.openRawResource( R.raw.sklearn_model ).readBytes()
        return ortEnvironment.createSession( modelBytes )
    }
    private fun runPrediction( input : FloatArray , ortSession: OrtSession , ortEnvironment: OrtEnvironment ) : Long {
        // Get the name of the input node
        val inputName = ortSession.inputNames?.iterator()?.next()
        // Make a FloatBuffer of the inputs
        val floatBufferInputs = FloatBuffer.wrap( input  )
        // Create input tensor with floatBufferInputs of shape ( 1 , 1 )
        val inputTensor = OnnxTensor.createTensor( ortEnvironment , floatBufferInputs , longArrayOf( 1, 8) )
        // Run the model
        val results = ortSession.run( mapOf( inputName to inputTensor ) )
        // Fetch and return the results
        val output = results[0].value as LongArray
        return output[0]
    }
}