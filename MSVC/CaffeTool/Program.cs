using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Diagnostics;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using CmdParser;
using CaffeLibMC;
using TsvTool.Utility;
using DetectionLib;

namespace CaffeTool
{
    class Program
    {
        class ArgsExtract
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Model config file. If provided, other args (proto, model, mean, and labelmap) are optional, but can be used to overwrite the params specified in config file.")]
            public string modelcfg = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Caffe prototxt file")]
            public string proto = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Caffe model file")]
            public string model = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Image mean binary proto file or mean values in the format of b,g,r")]
            public string mean = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Caffe model label map file (for getting human-readable topk predictions)")]
            public string labelmap = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Resize target (default: 256)")]
            public int resize_target = 256;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Keep aspect ratio? (default: false)")]
            public bool keep_aspect_ratio = false;

            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
            [Argument(ArgumentType.Required, HelpText = "Column index for image stream")]
            public int imageCol = -1;
            [Argument(ArgumentType.MultipleUnique, HelpText = "Feature blob name")]
            public string[] blob = new string[]{};
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output TSV file (default: replace inTsv .ext with .blobname.tsv")]
            public string outTsv = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Top k prediction, when labelmap is provided")]
            public int topK = 5;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Confidence threshold (default: 0.001 to output all)")]
            public float conf = 0.001f;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Batch size (default: use batch size specified in prototxt)")]
            public int batch = -1;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Gpu Id (default: 0, -1 for cpu)")]
            public int gpu = 0;
        }

        static void Extract(ArgsExtract cmd)
        {
            if (cmd.outTsv == null)
                cmd.outTsv = Path.ChangeExtension(cmd.inTsv, string.Join(".", cmd.blob) + ".tsv");

            // prepare model file names
            string protoFile = null;
            string modelFile = null;
            string meanFile = null; // only one of meanFile and meanValue will be valid
            string meanValue = null;
            string labelmapFile = null;

            if (cmd.modelcfg != null)
            {
                var modelDict = File.ReadLines(cmd.modelcfg)
                    .Where(line => line.Trim().StartsWith("#") == false)
                    .Select(line => line.Split(':'))
                    .ToDictionary(cols => cols[0].Trim(), cols => cols[1].Trim(), StringComparer.OrdinalIgnoreCase);

                var modelDir = Path.GetDirectoryName(cmd.modelcfg);
                var getPath = new Func<string, string>(file => Path.Combine(modelDir, file));

                protoFile = getPath(modelDict["proto"]);
                modelFile = getPath(modelDict["model"]);
                if (File.Exists(getPath(modelDict["mean"])))
                    meanFile = getPath(modelDict["mean"]);
                else
                    meanValue = modelDict["mean"];
                labelmapFile = getPath(modelDict["labelmap"]);
                if (string.IsNullOrEmpty(modelDict["labelmap"]))
                   labelmapFile = null;
               else
                   labelmapFile = getPath(modelDict["labelmap"]);
           }

            if (cmd.proto != null)
                protoFile = cmd.proto;
            if (cmd.model != null)
                modelFile = cmd.model;
            if (cmd.mean != null)
            {
                // clear previous settings first
                meanFile = meanValue = null;
                if (File.Exists(cmd.mean))
                    meanFile = cmd.mean;
                else
                    meanValue = cmd.mean;
            }
            if (cmd.labelmap != null)
                labelmapFile = cmd.labelmap;

            if (string.IsNullOrEmpty(labelmapFile) && cmd.blob.Length > 1)
            {
                Console.WriteLine("When labelmap is provided (for getting topk prediction), only one blob (e.g. prob) can be specified.");
                return;
            }

            CaffeModel.SetDevice(cmd.gpu);
            CaffeModel predictor = new CaffeModel(protoFile, modelFile);
            if (!string.IsNullOrEmpty(meanFile))
                predictor.SetMeanFile(meanFile);
            if (!string.IsNullOrEmpty(meanValue))
            {
                var mean_values = meanValue.Split(',').Select(x => Convert.ToSingle(x)).ToArray();
                predictor.SetMeanValue(mean_values, true);
                predictor.SetResizeTarget(cmd.resize_target, cmd.keep_aspect_ratio);
            }
            if (cmd.batch > 0)
                predictor.SetInputBatchSize(cmd.batch);

            var labelmap = string.IsNullOrEmpty(labelmapFile) ? null :
                File.ReadLines(labelmapFile)
                    .Select(line => line.Split('\t')[0])
                    .ToArray();

            int batch_size = predictor.GetInputImageNum();

            Stopwatch timer = Stopwatch.StartNew();
            int count = 0;
            var lines = File.ReadLines(cmd.inTsv)
                .Select(line => line.Split('\t').ToList())
                .Batch(batch_size)
                .Select(batch =>
                {
                    // prepare batch images
                    var batch_imgs = batch.AsParallel().AsOrdered()
                    .Select(cols =>
                    {
                        using (var ms = new MemoryStream(Convert.FromBase64String(cols[cmd.imageCol])))
                        using (var img = new Bitmap(ms))
                            return predictor.ResizeImage(img);
                    }).ToArray();
                    // batch feature extraction. may extract fc6 and fc7 together
                    float[][] batch_features = predictor.ExtractOutputs(batch_imgs, cmd.blob);
                    // release images
                    foreach (var img in batch_imgs)
                        img.Dispose();

                    return new {batch = batch, batch_features = batch_features};
                })
                .SelectMany(x =>
                {
                    var feature_dim = x.batch_features.Select(f => f.Length / batch_size).ToArray();
                    // slice features for each image
                    var batch_results = x.batch
                        .Select((cols, img_idx) =>
                        {
                            // handling multiple feature (e.g. fc6, fc7)
                            var img_features = x.batch_features
                                .Select((blob_feature, blob_idx) =>
                                {
                                    var f = new float[feature_dim[blob_idx]];
                                    Buffer.BlockCopy(blob_feature, feature_dim[blob_idx] * sizeof(float) * img_idx, f, 0, f.Length * sizeof(float));
                                    return f;
                                })
                                .ToArray();
                            return new {cols = cols, img_features = img_features};
                        }).ToList();

                    return batch_results;
                })
                .Select(x =>
                {
                    count++;
                    var added_cols = x.img_features.Select(f =>
                        {
                            if (labelmap == null)
                            {
                                byte[] fea = new byte[f.Length * sizeof(float)];
                                Buffer.BlockCopy(f, 0, fea, 0, fea.Length);
                                return Convert.ToBase64String(fea);
                            }
                            else
                            {
                                var topk = f.Select((value, idx) => Tuple.Create(value, idx))
                                    .Where(tp => tp.Item1 > cmd.conf)    // small trick to speed up
                                    .OrderByDescending(tp => tp.Item1)
                                    .Take(cmd.topK)
                                    .Select(tp => labelmap[tp.Item2] + ":" + tp.Item1);
                                return string.Join(";", topk);
                            }
                        }).ToList();

                    x.cols.RemoveAt(cmd.imageCol);
                    x.cols.AddRange(added_cols);
                    return x.cols;
                })
                .ReportProgress("Images progressed")
                .Select(cols => string.Join("\t", cols));

            File.WriteAllLines(cmd.outTsv, lines);
            Console.WriteLine();

            timer.Stop();
            Console.WriteLine("Latency: {0} seconds per image", timer.Elapsed.TotalSeconds / count);

            Console.WriteLine("outTSV follows the format of inTsv, with image column removed and blob features appended");
        }

        class ArgsDetect
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Model config file. If provided, other args (proto, model, and labelmap) are optional, but can be used to overwrite the params specified in config file.")]
            public string modelcfg = null;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Caffe prototxt file")]
            public string proto = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Caffe model file")]
            public string model = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Caffe model label map file")]
            public string labelmap = null;

            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
            [Argument(ArgumentType.Required, HelpText = "Column index for image stream")]
            public int imageCol = -1;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output TSV file (default: replace inTsv .ext with .det.tsv")]
            public string outTsv = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Confidence threshold (default: 0.001 to output all)")]
            public float conf = 0.001f;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Gpu Id (default: 0, -1 for cpu)")]
            public int gpu = 0;
        }

        // for json format output
        [DataContract]
        public class JsonDetectionResult
        {
            [DataMember]
            public double conf { get; set; }
            [DataMember]
            public string @class { get; set; }
            [DataMember]
            public List<double> rect { get; set; }
        }

        static void Detect(ArgsDetect cmd)
        {
            if (cmd.outTsv == null)
                cmd.outTsv = Path.ChangeExtension(cmd.inTsv, ".det.tsv");

            ObjectDetector detector = new ObjectDetector(cmd.modelcfg, cmd.gpu);

            Stopwatch timer = Stopwatch.StartNew();
            int count = 0;
            var lines = File.ReadLines(cmd.inTsv)
                .ReportProgress("Images progressed")
                .Select(line => line.Split('\t').ToList())
                .Select(cols =>
                {
                    count++;
                    using (var ms = new MemoryStream(Convert.FromBase64String(cols[cmd.imageCol])))
                    {
                        var results = detector.Detect(new Bitmap(ms), cmd.conf);

                        // convert to json format
                        var jsonResults = results.Select(x => new JsonDetectionResult()
                            {
                                @class = x.ClassName,
                                conf = x.Confidence,
                                rect = new List<double> { x.Rect.Left, x.Rect.Top, x.Rect.Right, x.Rect.Bottom }
                            }).ToArray();
                        string jsonString;
                        using (var stream = new MemoryStream())
                        {
                            DataContractJsonSerializer ser = new DataContractJsonSerializer(typeof(JsonDetectionResult[]));
                            ser.WriteObject(stream, jsonResults);
                            stream.Position = 0;
                            using (var sr = new StreamReader(stream))
                                jsonString = sr.ReadToEnd();
                        }

                        cols.RemoveAt(cmd.imageCol);
                        cols.Add(jsonString);

                        return cols;
                    }
                })
                .Select(cols => string.Join("\t", cols));

            File.WriteAllLines(cmd.outTsv, lines);
            Console.WriteLine();

            timer.Stop();
            Console.WriteLine("Latency: {0} seconds per image", timer.Elapsed.TotalSeconds / count);

            Console.WriteLine("outTSV follows the format of inTsv, with image column removed and detection result appended");
        }

        static void Main(string[] args)
        {
            ParserX.AddTask<ArgsExtract>(Extract, "Extract caffe feature from TSV file");
            ParserX.AddTask<ArgsDetect>(Detect, "Detect objects TSV file");
            if (ParserX.ParseArgumentsWithUsage(args))
            {
                Stopwatch timer = Stopwatch.StartNew();
                ParserX.RunTask();
                timer.Stop();
                Console.WriteLine("Time used: {0}", timer.Elapsed);
            }
        }
    }
}
