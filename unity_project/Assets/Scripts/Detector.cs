using System;
using UnityEngine;
using System.Linq;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Core;


public enum Library
{
    OnnxCSharp = 1,
    Onnx = 2,
}

public class Detector : MonoBehaviour
{
    public TextAsset labelsFile;
    public OnnxModelData onnxModelFile;

    public Library usedLibrary = Library.OnnxCSharp;
    
    public double lastInferenceTime = 0.0f;
    public double lastFrameProcessTime = 0.0f;
    public long lastDetectedObjects = 0;
    public double meanInferenceTime = 0.0f;
    
    public int imageSizeW = 384;
    public int imageSizeH = 480;
    
    private AndroidJavaObject _onnxModel;
    private InferenceSession _onnxCSharpModel;

    private string[] _labels;
    
    public void Start()
    {
        _labels = Regex.Split(labelsFile.text, "\n|\r|\r\n")
            .Where(s => !String.IsNullOrEmpty(s)).ToArray();

        if (usedLibrary == Library.OnnxCSharp)
        {
            
            /* XNNPACK -- Android/IOS (slow)
            var options = new SessionOptions();
            var paramsOptions = new Dictionary<string, string>()
            {
                { "intra_op_num_threads", "1" }
            };
            options.AppendExecutionProvider("XNNPACK", paramsOptions);*/
            
            
            /* SNPE -- Qualcom (Not working for me)
            var options = new SessionOptions();
            var paramsOptions = new Dictionary<string, string>()
            {
                { "runtime", "CPU_FLOAT32" },
                { "buffer_type", "FLOAT" },
            };
            options.AppendExecutionProvider("SNPE", paramsOptions);*/
            
            var options = new SessionOptions();
            // options.EnableProfiling = true;
            
            // Должно работать на всех платформах
            _onnxCSharpModel = new InferenceSession(onnxModelFile.bytes, options);
        }
        else if (usedLibrary == Library.Onnx) 
        {
            /* Только Android, модели формата .onnx */
            var pluginClass = new AndroidJavaClass("com.unity.onnxwrapper.OnnxWrapper");
            _onnxModel = pluginClass.GetStatic<AndroidJavaObject>("instance");

            _onnxModel.Call("init");
            _onnxModel.Call<long>("createSession", onnxModelFile.bytes);
        }
    }

    public IList<BoundingBox> Detect(Color32[] picture)
    {
        var startTime0 = DateTime.UtcNow;
        
        // Записываем изображение в одномерный массив:
        var tensorData = TransformInput(picture, imageSizeW, imageSizeH);

        var boxes = new List<BoundingBox>();
        if (usedLibrary == Library.OnnxCSharp)
        {
            var inputTensor = new DenseTensor<float>(tensorData, new int[] { 1, 3, imageSizeH, imageSizeW });
            var configTensor = new DenseTensor<float>(new float[] {100.0f, 0.7f, 0.2f}, new int[] { 3 });
            var input = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor<float>("images", inputTensor),
                NamedOnnxValue.CreateFromTensor<float>("config", configTensor)
            };
            
            var startTime1 = DateTime.UtcNow;
            var output = _onnxCSharpModel.Run(input).ToArray();
            lastInferenceTime = (DateTime.UtcNow - startTime1).TotalSeconds;
            
            var result = output[0].AsEnumerable<float>().ToArray();
            var objects = result.Length / 84;
            
            for (var i = 0; i < objects; i++)
            {
                var start = i * 84;
                var x = result[start];
                var y = result[start + 1];
                var w = result[start + 2];
                var h = result[start + 3];

                x -= w / 2.0f;
                y -= h / 2.0f;

                var classId = 0;
                var classScore = 0.0f;
                for (var j = 0; j < 80; j++)
                {
                    if (result[start + 4 + j] > classScore)
                    {
                        classId = j;
                        classScore = result[start + 4 + j];
                    }
                }
                
                boxes.Add(new BoundingBox
                {
                    Dimensions = new BoundingBoxDimensions
                    {
                        X = x,
                        Y = y,
                        Width = w,
                        Height = h,
                    },
                    Confidence = classScore,
                    Label = _labels[classId]
                });
            }

            meanInferenceTime = 0.95 * meanInferenceTime + 0.05 * lastInferenceTime;
            lastFrameProcessTime = (DateTime.UtcNow - startTime0).TotalSeconds;
        }
        else if (usedLibrary == Library.Onnx)
        {
            var shape = new long[] { 1, 3, imageSizeW, imageSizeH };
            AndroidJavaObject objects = _onnxModel.Call<AndroidJavaObject>(
                "propagateModel", "images", tensorData, shape
                );
            lastDetectedObjects = objects.Call<long>("GetDetectedObjects");

            for (int i = 0; i < lastDetectedObjects; i++)
            {
                var x = objects.Call<float>("GetX", i);
                var y = objects.Call<float>("GetY", i);
                var w = objects.Call<float>("GetW", i);
                var h = objects.Call<float>("GetH", i);
                var c = objects.Call<long>("GetC", i);
                var s = objects.Call<float>("GetS", i);

                boxes.Add(new BoundingBox
                {
                    Dimensions = new BoundingBoxDimensions
                    {
                        X = x,
                        Y = y,
                        Width = w,
                        Height = h,
                    },
                    Confidence = s,
                    Label = _labels[c]
                });
            }

            lastInferenceTime = objects.Call<long>("GetProcessingMs") / 1000f;
            meanInferenceTime = 0.95 * meanInferenceTime + 0.05 * lastInferenceTime;
            lastFrameProcessTime = (System.DateTime.UtcNow - startTime0).TotalSeconds;
        }
        return boxes;
    }

    private static float[] TransformInput(Color32[] pic, int width, int height)
    {
        var floatValues = new float[width * height * 3];

        var stride = width * height;
        
        for (var i = 0; i < height; ++i)
        {
            for (var j = 0; j < width; ++j)
            {
                var idx = width * i + j;
                var color = pic[idx];
                floatValues[idx] = color.r /  255.0f;
                floatValues[idx + stride] = color.g / 255.0f;
                floatValues[idx + 2 * stride] = color.b / 255.0f;
            }
        }

        return floatValues;
    }
}


public class DimensionsBase
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Height { get; set; }
    public float Width { get; set; }
}

public class BoundingBoxDimensions : DimensionsBase
{
}

class CellDimensions : DimensionsBase
{
}

public class BoundingBox
{
    public BoundingBoxDimensions Dimensions { get; set; }

    public string Label { get; set; }

    public float Confidence { get; set; }

    public Rect Rect
    {
        get { return new Rect(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height); }
    }

    public override string ToString()
    {
        return $"{Label}:{Confidence}, {Dimensions.X}:{Dimensions.Y} - {Dimensions.Width}:{Dimensions.Height}";
    }
}