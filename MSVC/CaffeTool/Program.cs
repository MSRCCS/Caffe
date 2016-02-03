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
            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
            [Argument(ArgumentType.Required, HelpText = "Column index for image stream")]
            public int imageCol = -1;
            [Argument(ArgumentType.MultipleUnique, HelpText = "Feature blob name")]
            public string[] blob = new string[]{};
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output TSV file (default: replace inTsv .ext with .blobname.tsv")]
            public string outTsv = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Gpu Id (default: 0, -1 for cpu)")]
            public int gpu = 0;
        }

        static void Extract(ArgsExtract cmd)
        {
            if (cmd.outTsv == null)
                cmd.outTsv = Path.ChangeExtension(cmd.inTsv, string.Join(".", cmd.blob) + ".tsv");

            CaffeModel.SetDevice(cmd.gpu);
            CaffeModel predictor = new CaffeModel(cmd.proto, cmd.model);

            Stopwatch timer = Stopwatch.StartNew();
            int count = 0;
            var lines = File.ReadLines(cmd.inTsv)
                .Select(line => line.Split('\t').ToList())
                .Select(cols =>
                {
                    using (var ms = new MemoryStream(Convert.FromBase64String(cols[cmd.imageCol])))
                    using (var img = new Bitmap(ms))
                    {
                        cols.RemoveAt(cmd.imageCol);
                        float[][] features = predictor.ExtractOutputs(img, cmd.blob);
                        foreach (var blob_feature in features)
                        {
                            byte[] fea = new byte[blob_feature.Length * sizeof(float)];
                            Buffer.BlockCopy(blob_feature, 0, fea, 0, blob_feature.Length * sizeof(float));
                            cols.Add(Convert.ToBase64String(fea));
                        }
                    }
                    Console.Write("{0}\r", ++count);
                    return cols;
                })
                .Select(cols => string.Join("\t", cols));

            File.WriteAllLines(cmd.outTsv, lines);

            timer.Stop();
            Console.WriteLine("Latency: {0} seconds per image", timer.Elapsed.TotalSeconds / count);

            Console.WriteLine("In the output TSV, image column is removed, and blob features are appended");
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
