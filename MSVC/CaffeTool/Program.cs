using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Diagnostics;
using System.IO;
using CmdParser;
using CaffeLibMC;
using TsvTool.Utility;

namespace CaffeExtract
{
    class Program
    {
        class ArgsExtract
        {
            [Argument(ArgumentType.Required, HelpText = "Caffe prototxt file")]
            public string proto = null;
            [Argument(ArgumentType.Required, HelpText = "Caffe model file")]
            public string model = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Image mean binary proto file")]
            public string mean = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Caffe model label map file (for getting human-readable topk predictions)")]
            public string labelmap = null;
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
            [Argument(ArgumentType.AtMostOnce, HelpText = "Gpu Id (default: 0, -1 for cpu)")]
            public int gpu = 0;
        }

        static void Extract(ArgsExtract cmd)
        {
            if (cmd.outTsv == null)
                cmd.outTsv = Path.ChangeExtension(cmd.inTsv, string.Join(".", cmd.blob) + ".tsv");

            if (cmd.labelmap != null && cmd.blob.Length > 1)
            {
                Console.WriteLine("When labelmap is provided (for getting topk prediction), only one blob (e.g. prob) can be specified.");
                return;
            }

            CaffeModel.SetDevice(cmd.gpu);
            CaffeModel predictor = new CaffeModel(cmd.proto, cmd.model);
            if (cmd.mean != null)
                predictor.SetMeanFile(cmd.mean);

            var labelmap = cmd.labelmap == null ? null :
                File.ReadLines(cmd.labelmap)
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
                    var batch_imgs = batch.Select(cols =>
                    {
                        using (var ms = new MemoryStream(Convert.FromBase64String(cols[cmd.imageCol])))
                            return new Bitmap(ms);
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
                                    .Where(tp => tp.Item1 > 0.001)    // small trick to speed up
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

        static void Main(string[] args)
        {
            ParserX.AddTask<ArgsExtract>(Extract, "Extract caffe feature from TSV file");
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
