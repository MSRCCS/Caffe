using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using CaffeLibMC;

namespace DetectionLib
{
    public static class LinqUtility
    {
        public static IEnumerable<T[]> SplitByN<T>(this IEnumerable<T> source, int N)
        {
            var result = source.Select((x, i) => Tuple.Create(i / N, x))
                .GroupBy(tp => tp.Item1)
                .Select(g => g.AsEnumerable().Select(tp => tp.Item2).ToArray());
            return result;
        }
    }

    public class DetectResult
    {
        public string ClassName;
        public int ClassId;
        public float Confidence;
        public Rectangle Rect;
    }

    public class ObjectDetector
    {
        const string ModelConfigFile = "modelcfg.txt";

        const int MaxNumOutputs = 20;

        readonly int TargetSize = 0;

        readonly CaffeModel _caffeModel;

        readonly string[] _labelMap;
        readonly ConcurrentStack<CaffeModelState> _caffeModelStates = new ConcurrentStack<CaffeModelState>();

        public ObjectDetector(string modelcfg, int deviceId = -1)
        {
            string modelDir = Path.GetDirectoryName(Path.GetFullPath(modelcfg));
            var getPath = new Func<string, string>(x => Path.Combine(modelDir, x));

            var modelDict = File.ReadLines(modelcfg)
                .Where(line => line.Trim().StartsWith("#") == false)
                .Select(line => line.Split(':'))
                .ToDictionary(cols => cols[0].Trim(), cols => cols[1].Trim(), StringComparer.OrdinalIgnoreCase);

            TargetSize = Convert.ToInt32(modelDict["target_size"]);
            string protoFile = getPath(modelDict["proto"]);
            string modelFile = getPath(modelDict["model"]);
            string labelmapFile = getPath(modelDict["labelmap"]);

            // Init image recognition
            _caffeModel = new CaffeModel(protoFile, modelFile);
            Console.WriteLine("Object detector model loaded ...");

            // Get label map
            _labelMap = File.ReadLines(labelmapFile).Select(line => line.Split('\t')[0]).ToArray();

            CaffeModel.SetDevice(deviceId);

            _caffeModelStates.Push(new CaffeModelState(_caffeModel, false));
        }

        public static Bitmap ImageResize(Bitmap img, int outputWidth, int outputHeight)
        {
            PixelFormat pf = PixelFormat.Format32bppArgb;

            Bitmap newImg = new Bitmap((int)outputWidth, (int)outputHeight, pf);
            using (Graphics g = Graphics.FromImage(newImg))
            {
                g.Clear(Color.Transparent);
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBilinear;
                g.DrawImage(img, 0, 0, outputWidth, outputHeight); // draw the image at 0, 0
            }

            return newImg;
        }

        private float[][] bbox_transform_inv(float[][] boxes, float [][]deltas)
        {
            int num_class = deltas[0].Length / 4;

            var pred_boxes = deltas.Select((delta, i) =>
            {
                var box = boxes[i];
                var w = box[3] - box[1] + 1f;
                var h = box[4] - box[2] + 1f;
                var center_x = box[1] + w * 0.5f;
                var center_y = box[2] + h * 0.5f;

                var pred = delta.SplitByN(4).Select(d =>
                    {
                        var pred_ctr_x = d[0] * w + center_x;
                        var pred_ctr_y = d[1] * h + center_y;
                        var pred_w = (float)Math.Exp(d[2]) * w;
                        var pred_h = (float)Math.Exp(d[3]) * h;

                        d[0] = pred_ctr_x - 0.5f * pred_w;
                        d[1] = pred_ctr_y - 0.5f * pred_h;
                        d[2] = pred_ctr_x + 0.5f * pred_w;
                        d[3] = pred_ctr_y + 0.5f * pred_h;

                        return d;
                    })
                    .SelectMany(d => d)
                    .ToArray();
                return pred;
            }).ToArray();

            return pred_boxes;
        }

        private float[][] clip_boxes(float[][] boxes, float width, float height)
        {
            var new_boxes = boxes.Select(box =>
            {
                return box.SplitByN(4).Select(b =>
                {
                    b[0] = Math.Max(Math.Min(b[0], width - 1f), 0f);
                    b[1] = Math.Max(Math.Min(b[1], height - 1f), 0f);
                    b[2] = Math.Max(Math.Min(b[2], width - 1f), 0f);
                    b[3] = Math.Max(Math.Min(b[3], height - 1f), 0f);
                    return b;
                })
                .SelectMany(b => b).ToArray();
            }).ToArray();

            return new_boxes;
        }

        private int[] nms(RectangleF[] boxes, float[] scores, float nms_thresh)
        {
            var order = scores.Select((x, i) => Tuple.Create(x, i))
                    .OrderByDescending(tp => tp.Item1)
                    .Select(tp => tp.Item2).ToArray();

            bool[] suppressed = new bool[boxes.Length];
            for (int i = 0; i < suppressed.Length; i++)
                suppressed[i] = false;
            List<int> keep = new List<int>();

            for (int _i = 0; _i < boxes.Length; _i++)
            {
                int i = order[_i];
                if (suppressed[i])
                    continue;
                keep.Add(i);
                var rc_i = boxes[i];
                var iarea = rc_i.Width * rc_i.Height;
                for (int _j = _i + 1; _j < boxes.Length; _j++)
                {
                    int j = order[_j];
                    if (suppressed[j])
                        continue;
                    var rc_j = boxes[j];
                    var _ij = RectangleF.Intersect(rc_i, rc_j);
                    var inter = _ij.Width * _ij.Height;
                    var ovr = inter / (iarea + rc_j.Width * rc_j.Height - inter);
                    if (ovr >= nms_thresh)
                        suppressed[j] = true;
                }
            }

            return keep.ToArray();
        }

        public DetectResult[] Detect(Bitmap image, float confidenceThreshold)
        {
            float scale = (float)TargetSize / Math.Min(image.Width, image.Height);
            image = ImageResize(image, (int)(scale * image.Width + 0.5f), (int)(scale * image.Height + 0.5f));

            CaffeModelState state;
            if (!_caffeModelStates.TryPop(out state))
                state = new CaffeModelState(_caffeModel, true);
            state.Model.SetInputs("data", new Bitmap[] { image }, false);
            state.Model.SetInputs("im_info", new float[] { image.Height, image.Width, scale });
            float[][] outputs = state.Model.Forward(new string[] { "cls_prob", "bbox_pred", "rois" });
            int[] shape = state.Model.GetBlobShape("cls_prob");
            _caffeModelStates.Push(state);

            int num_bbox = shape[0];
            int num_class = shape[1];

            var cls_prob = outputs[0].SplitByN(num_class).ToArray();
            var box_deltas = outputs[1].SplitByN(num_class * 4).ToArray();
            // # unscale back to raw image space
            var rois = outputs[2].Select(x => x / scale).SplitByN(5).ToArray();

            var pred_boxes = bbox_transform_inv(rois, box_deltas);
            pred_boxes = clip_boxes(pred_boxes, image.Width, image.Height);

            float conf_thresh = 0.05f;
            List<Tuple<RectangleF, float, int>> results = new List<Tuple<RectangleF, float, int>>();
            // # skip j = 0, because it's the background class
            for (int j = 1; j < num_class; j++)
            {
                var inds = cls_prob.Select((x, i) => Tuple.Create(x[j], i))
                    .Where(tp => tp.Item1 >= conf_thresh)
                    .Select(tp => tp.Item2).ToArray();

                var cls_scores = inds.Select(i => cls_prob[i][j]).ToArray();
                var cls_boxes = inds.Select(i => pred_boxes[i].Skip(j * 4).Take(4).ToArray())
                    .Select(b => new RectangleF(b[0], b[1], b[2] - b[0] + 1, b[3] - b[1] + 1))
                    .ToArray();

                var keep = nms(cls_boxes, cls_scores, 0.3f);
                results.AddRange(keep.Select(ind => Tuple.Create(cls_boxes[ind], cls_scores[ind], j)));
            }

            var recogResult = results
                .Where(tp => tp.Item2 > confidenceThreshold)
                .OrderByDescending(tp => tp.Item2)
                .Select(tp =>
                {
                    string name = _labelMap[tp.Item3];
                    return new DetectResult()
                    {
                        ClassName = name,
                        ClassId = tp.Item3,
                        Rect = Rectangle.Round(tp.Item1),
                        Confidence = tp.Item2
                    };
                })
                .Where(e => !string.IsNullOrEmpty(e.ClassName))
                .Take(MaxNumOutputs)
                .ToArray();

            return recogResult;
        }

        class CaffeModelState
        {
            public readonly CaffeModel Model;

            public CaffeModelState(CaffeModel caffeModel, bool cloneCaffeModel)
            {
                if (cloneCaffeModel)
                    Model = new CaffeModel(caffeModel);
                else
                    Model = caffeModel;
            }
        }
    }
}
