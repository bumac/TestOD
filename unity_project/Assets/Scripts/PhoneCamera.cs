using System;
using System.IO;
using TFClassify;
using UnityEngine;
using System.Linq;
using UnityEngine.UI;
using System.Collections;
using System.Threading.Tasks;
using System.Collections.Generic;


public class PhoneCamera : MonoBehaviour
{
    private static Texture2D boxOutlineTexture;
    private static GUIStyle labelStyle;

    private float cameraScale = 1f;
    private float shiftX = 0f;
    private float shiftY = 0f;
    private float scaleFactor = 1;
    private bool camAvailable;

    private WebCamTexture backCamera;
    private Texture defaultBackground;

    private bool isWorking = false;
    public Detector detector;

    private IList<BoundingBox> boxOutlines;
    
    
    public RawImage background;
    public AspectRatioFitter fitter;
    public Text uiText;


    private void Start()
    {
        backCamera = new WebCamTexture();
        background.texture = backCamera;
        backCamera.Play();
        camAvailable = true;

        boxOutlineTexture = new Texture2D(1, 1);
        boxOutlineTexture.SetPixel(0, 0, Color.red);
        boxOutlineTexture.Apply();

        labelStyle = new GUIStyle
        {
            fontSize = 50,
            normal =
            {
                textColor = Color.red
            }
        };

        CalculateShift(Detector.ImageSize);

        StartCoroutine(CameraCheck());
        
        
    }

    // Clear screen:
    private void OnPreRender()
    {
        GL.Clear(true, true, Color.black);
    }
    
    private IEnumerator CameraCheck()
    {
        while (true)
        {
            if (!camAvailable)
            {
                yield break;
            }

            var ratio = (float)backCamera.width / (float)backCamera.height;
            fitter.aspectRatio = ratio;

            var scaleX = cameraScale;
            var scaleY = backCamera.videoVerticallyMirrored ? -cameraScale : cameraScale;
            background.rectTransform.localScale = new Vector3(scaleX, scaleY, 1f);

            var orient = -backCamera.videoRotationAngle;
            background.rectTransform.localEulerAngles = new Vector3(0, 0, orient);

            if (orient != 0)
            {
                cameraScale = (float)Screen.width / Screen.height;
            }

            Detect();

            yield return new WaitForSeconds(1);
        }
    }

    public void OnGUI()
    {
        if (boxOutlines == null || !boxOutlines.Any()) return;
        
        foreach (var outline in boxOutlines)
        {
            DrawBoxOutline(outline, scaleFactor, shiftX, shiftY);
        }
    }


    private void CalculateShift(int inputSize)
    {
        int smallest;

        if (Screen.width < Screen.height)
        {
            smallest = Screen.width;
            shiftY = (Screen.height - smallest) / 2f;
        }
        else
        {
            smallest = Screen.height;
            shiftX = (Screen.width - smallest) / 2f;
        }

        scaleFactor = smallest / (float)inputSize;
    }


    private void Detect()
    {
        if (isWorking)
        {
            return;
        }

        isWorking = true;
        
        StartCoroutine(ProcessImage(Detector.ImageSize, result =>
        {
            var boxes = detector.Detect(result);
            uiText.text = string.Format("{0:0.000} | {1:0.000} | {2} | {3:0.000}", 
                detector.lastInferenceTime, detector.lastFrameProcessTime, 
                detector.lastDetectedObjects, detector.meanInferenceTime
                );
            boxOutlines = boxes;
            Resources.UnloadUnusedAssets();
            isWorking = false;
        }));
    }


    private IEnumerator ProcessImage(int inputSize, Action<Color32[]> callback)
    {
        yield return StartCoroutine(TextureTools.CropSquare(backCamera,
            TextureTools.RectOptions.Center, snap =>
            {
                var scaled = Scale(snap, inputSize);
                var rotated = Rotate(scaled.GetPixels32(), scaled.width, scaled.height);
                callback(rotated);
            }));
    }


    private static void DrawBoxOutline(BoundingBox outline, float scaleFactor, float shiftX, float shiftY)
    {
        var x = outline.Dimensions.X * scaleFactor + shiftX;
        var width = outline.Dimensions.Width * scaleFactor;
        var y = outline.Dimensions.Y * scaleFactor + shiftY;
        var height = outline.Dimensions.Height * scaleFactor;

        DrawRectangle(new Rect(x, y, width, height), 4, Color.red);
        DrawLabel(new Rect(x + 10, y + 10, 200, 20), $"{outline.Label}: {(int)(outline.Confidence * 100)}%");
    }


    private static void DrawRectangle(Rect area, int frameWidth, Color color)
    {
        var lineArea = area;
        lineArea.height = frameWidth;
        GUI.DrawTexture(lineArea, boxOutlineTexture); // Top line

        lineArea.y = area.yMax - frameWidth;
        GUI.DrawTexture(lineArea, boxOutlineTexture); // Bottom line

        lineArea = area;
        lineArea.width = frameWidth;
        GUI.DrawTexture(lineArea, boxOutlineTexture); // Left line

        lineArea.x = area.xMax - frameWidth;
        GUI.DrawTexture(lineArea, boxOutlineTexture); // Right line
    }


    private static void DrawLabel(Rect position, string text)
    {
        GUI.Label(position, text, labelStyle);
    }


    private static Texture2D Scale(Texture2D texture, int imageSize)
    {
        var scaled = TextureTools.scaled(texture, imageSize, imageSize, FilterMode.Bilinear);

        return scaled;
    }


    private static Color32[] Rotate(Color32[] pixels, int width, int height)
    {
        return TextureTools.RotateImageMatrix(
                pixels, width, height, -90);
    }

    private Task<Texture2D> RotateAsync(Texture2D texture)
    {
        return Task.Run(() => TextureTools.RotateTexture(texture, -90));
    }
}
