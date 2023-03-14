using System.IO;
using UnityEngine;
using UnityEditor.AssetImporters;
using Core;

namespace Editor
{
    [ScriptedImporter(1, new[] { "onnx", "ort" })]
    public class OnnxImporter : ScriptedImporter
    {
        public override void OnImportAsset(AssetImportContext ctx)
        {
            //var mainObject = new TextAsset(File.ReadAllText(ctx.assetPath));
            var modelData = ScriptableObject.CreateInstance<OnnxModelData>();
            modelData.bytes = File.ReadAllBytes(ctx.assetPath);
            
            ctx.AddObjectToAsset("OnnxModelData", modelData);
            ctx.SetMainObject(modelData);
        }
    }
}