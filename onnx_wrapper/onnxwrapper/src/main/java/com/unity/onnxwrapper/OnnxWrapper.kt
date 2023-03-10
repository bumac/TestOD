package com.unity.onnxwrapper

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.TensorInfo
import android.os.Build
import android.os.SystemClock
import androidx.annotation.RequiresApi
import java.nio.FloatBuffer
import java.util.*



class Results() {
    var processingMs: Long = -1
    var detectedObjects: Long = -1

    var detectedBboxes: MutableList<FloatArray> = emptyList<FloatArray>().toMutableList()
    var detectedClasses: MutableList<Long> = emptyList<Long>().toMutableList()
    var detectedScores: MutableList<Float> = emptyList<Float>().toMutableList()

    constructor(processingMs: Long, detectedObjects: Long) : this() {
        this.processingMs = processingMs
        this.detectedObjects = detectedObjects
    }

    fun Add(bbox: FloatArray, class_id: Long, score: Float) {
        detectedBboxes.add(bbox)
        detectedClasses.add(class_id)
        detectedScores.add(score)
    }

    fun GetX(i: Int): Float {
        return detectedBboxes[i][0]
    }

    fun GetY(i: Int): Float {
        return detectedBboxes[i][1]
    }

    fun GetW(i: Int): Float {
        return detectedBboxes[i][2]
    }

    fun GetH(i: Int): Float {
        return detectedBboxes[i][3]
    }

    fun GetC(i: Int): Long {
        return detectedClasses[i]
    }

    fun GetS(i: Int): Float {
        return detectedScores[i]
    }

    fun GetProcessingMs(): Long {
        return processingMs
    }

    fun GetDetectedObjects(): Long {
        return detectedObjects
    }
}


class OnnxWrapper {
    var ortEnv: OrtEnvironment? = null
    var ortSession: OrtSession? = null

    fun init() {
        ortEnv = OrtEnvironment.getEnvironment()
    }

    @RequiresApi(Build.VERSION_CODES.O)
    fun createSession(model: ByteArray): Long{
        ortSession = ortEnv?.createSession(model)

        // Для отладки, возвращаем количество входов (если модель правильно загрузилась)
        return ortSession?.numInputs ?: -1
    }

    @RequiresApi(Build.VERSION_CODES.O)
    fun propagateModel(inputName: String, tensorData: FloatArray, shapeData: LongArray): Results {

        // FloatArray -> FloatBuffer
        val tensorDataBuffer = FloatBuffer.allocate(tensorData.size)
        tensorDataBuffer.put(tensorData)
        tensorDataBuffer.rewind()

        val tensor = OnnxTensor.createTensor(ortEnv, tensorDataBuffer, shapeData)

        val startTime = SystemClock.uptimeMillis()

        // Запуск модели на полученных данных:
        val output = ortSession?.run(Collections.singletonMap(inputName, tensor))

        val processTimeMs = SystemClock.uptimeMillis() - startTime

        val result = Results(processTimeMs, (output?.get(0)?.info as TensorInfo).shape[2])

        // Считывание результата:
        if(result.GetDetectedObjects() > 0){
            val raw_bboxes = (output?.get(0)?.value as Array<Array<Array<FloatArray>>>)[0][0]

            for(box: FloatArray in raw_bboxes){
                var now_class = 0
                var now_score = box[4]
                for(i in 1 .. 80 - 1){
                    if(box[4 + i] > now_score){
                        now_class = i
                        now_score = box[4 + i]
                    }
                }
                val center_x = box[0]
                val center_y = box[1]
                val width = box[2]
                val height = box[3]
                val x = center_x - width / 2f
                val y = center_y - height / 2f

                result.Add(floatArrayOf(x, y, width, height), now_class.toLong(), now_score)
            }
        }

        return result
    }

    companion object {
        val instance = OnnxWrapper()
    }
}