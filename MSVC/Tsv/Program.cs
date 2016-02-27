using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Net;
using System.IO;
using System.Drawing;
using System.Diagnostics;
using CmdParser;

namespace TsvTool
{
    class Program
    {
        class ArgsLabel
        {
            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Labelmap file (if provided, will use this to map label to class id")]
            public string labelmap = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output TSV file (default: replace inTsv file ext. with .label.tsv)")]
            public string outTsv = null;
            [Argument(ArgumentType.Required, HelpText = "Column index for label")]
            public int labelCol = -1;
        }

        class ArgsIndex
        {
            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
        }

        class ArgsShuffle
        {
            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Ignore label (default: null, include all data)")]
            public string ignoreLabel = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Label column index (valid only when ignoreLabel is not null)")]
            public int labelCol = -1;
        }

        static void Label(ArgsLabel cmd)
        {
            if (cmd.outTsv == null)
                cmd.outTsv = Path.ChangeExtension(cmd.inTsv, ".label.tsv");

            Dictionary<string, int> dict;

            if (cmd.labelmap == null)
            {
                dict = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            }
            else
            {
                dict = File.ReadLines(cmd.labelmap)
                    .Select(line => line.Split('\t'))
                    .ToDictionary(cols => cols[0], cols => Convert.ToInt32(cols[1]), StringComparer.OrdinalIgnoreCase);
                Console.WriteLine("Labelmap file loaded with # of classes: {0}", dict.Count());
            }

            int count = 0;
            var lines = File.ReadLines(cmd.inTsv)
                .Select(line => line.Split('\t')[cmd.labelCol])
                .Select(label =>
                {
                    int cls_id;
                    if (cmd.labelmap == null)
                    {
                        if (!dict.TryGetValue(label, out cls_id))
                        {
                            cls_id = dict.Count();
                            dict.Add(label, cls_id);
                        }
                    }
                    else
                    {
                        if (dict.ContainsKey(label))
                            cls_id = dict[label];
                        else
                            cls_id = -1;
                    }
                    Console.Write("Line processed: {0}\r", ++count);
                    return new List<string>() { label, cls_id.ToString() };
                })
                .Select(cols => string.Join("\t", cols));

            File.WriteAllLines(cmd.outTsv, lines);
            Console.WriteLine("\nLabel file saved.");

            if (cmd.labelmap == null)
            {
                File.WriteAllLines(Path.ChangeExtension(cmd.inTsv, ".labelmap"),
                    dict.Select(kv => kv.Key + "\t" + kv.Value.ToString()));
                Console.WriteLine("Labelmap file saved.");
            }
        }

        static void Index(ArgsIndex cmd)
        {
            FileIndex.BuildLineIndex(cmd.inTsv);
        }

        static void Shuffle(ArgsShuffle cmd)
        {
            Random rnd = new Random();
            string[] randomLineNumbers;
            if (cmd.ignoreLabel == null)
            {
                randomLineNumbers = File.ReadLines(Path.ChangeExtension(cmd.inTsv, "lineidx"))
                    .Select((line, i) => new Tuple<int, int>(i, rnd.Next()))
                    .OrderBy(tp => tp.Item2)
                    .Select(tp => tp.Item1.ToString())
                    .ToArray();
            }
            else
            {
                randomLineNumbers = File.ReadLines(cmd.inTsv)
                    .Select(line => line.Split('\t')[cmd.labelCol])
                    .Select((label, i) => Tuple.Create(label, i))
                    .Where(tp => string.Compare(tp.Item1, cmd.ignoreLabel, true) != 0)
                    .Select(tp => Tuple.Create(tp.Item2, rnd.Next()))
                    .OrderBy(tp => tp.Item2)
                    .Select(tp => tp.Item1.ToString())
                    .ToArray();
            }
            Console.WriteLine("Distinct: {0}", randomLineNumbers.Distinct().Count());
            File.WriteAllLines(Path.ChangeExtension(cmd.inTsv, "shuffle"), randomLineNumbers);
        }

        class ArgsList2Tsv
        {
            [Argument(ArgumentType.Required, HelpText = "Input list file")]
            public string inList = null;
            [Argument(ArgumentType.Required, HelpText = "Input folder that combines with filename in the list file to find images")]
            public string inFolder = null;
            [Argument(ArgumentType.Required, HelpText = "Output TSV file")]
            public string outTsv = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Column index for image")]
            public int imageCol = 0;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Column index for label")]
            public int labelCol = 1;
        }

        static void List2Tsv(ArgsList2Tsv cmd)
        {
            var list = File.ReadLines(cmd.inList).Select(line => line.Split('\t'));
            using (var sw = new StreamWriter(cmd.outTsv))
            {
                int count = 0;
                foreach (var cols in list)
                {
                    var img = File.ReadAllBytes(Path.Combine(cmd.inFolder, cols[cmd.imageCol]));
                    count++;
                    Console.Write("Read {0}\r", count);
                    sw.WriteLine("{0}\t{1}\t{2}", cols[cmd.imageCol], cols[cmd.labelCol], Convert.ToBase64String(img));
                }
                Console.WriteLine();
                Console.WriteLine("Total read: {0} images", count);
            }
        }

        class ArgsFolder2Tsv
        {
            [Argument(ArgumentType.Required, HelpText = "Input folder to find images")]
            public string inFolder = null;
            [Argument(ArgumentType.Required, HelpText = "Output TSV file")]
            public string outTsv = null;
        }

        static void Folder2Tsv(ArgsFolder2Tsv cmd)
        {
            Console.WriteLine("Finding images in {0}...", cmd.inFolder);
            var allFiles = Directory.GetFiles(cmd.inFolder, "*.*", SearchOption.AllDirectories)
                            .Where(file => file.ToLower().EndsWith("jpg")
                                            || file.ToLower().EndsWith("bmp")
                                            || file.ToLower().EndsWith("png"))
                            .OrderBy(file => file)
                            .ToArray();

            Console.WriteLine("Total images found: {0}", allFiles.Count());

            using (var sw = new StreamWriter(cmd.outTsv))
            {
                int count = 0;
                foreach (string file in allFiles)
                {
                    byte[] img = File.ReadAllBytes(file);
                    count++;
                    string imageFileName = file.Substring(cmd.inFolder.Length + 1);
                    string folderName = Path.GetDirectoryName(imageFileName);
                    Console.Write("Read {0}\r", count);
                    sw.WriteLine("{0}\t{1}\t{2}", imageFileName, folderName, Convert.ToBase64String(img));
                }
                Console.WriteLine();
                Console.WriteLine("Total {0} images written to {1}", count, cmd.outTsv);
                Console.WriteLine("Column names: Filename, Foldername, base64Image");
            }

        }

        class ArgsTsv2Folder
        {
            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output TSV file")]
            public string outFolder = null;
            [Argument(ArgumentType.Required, HelpText = "Column index for base64 encoded image")]
            public int colImage = -1;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Column index for sub folder name (default: use subfolder name in TSV)")]
            public int colSubFolder = -1;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Column index for file name (default: use GUID as filename)")]
            public int colFileName = -1;
        }

        static void Tsv2Folder(ArgsTsv2Folder cmd)
        {
            if (cmd.outFolder == null)
                cmd.outFolder = Path.GetFileNameWithoutExtension(cmd.inTsv);
            var lines = File.ReadLines(cmd.inTsv)
                .Select(line => line.Split('\t'));
            int count = 0;
            foreach (var cols in lines)
            {
                string subfolder = null, filename;
                if (cmd.colFileName >= 0)
                {
                    filename = cols[cmd.colFileName];
                    if (string.IsNullOrEmpty(Path.GetExtension(filename)))
                        filename = filename + ".jpg";
                    subfolder = Path.GetDirectoryName(filename);
                    if (string.IsNullOrEmpty(subfolder) && cmd.colSubFolder >= 0)
                        subfolder = cols[cmd.colSubFolder];
                    var invalids = System.IO.Path.GetInvalidFileNameChars();
                    subfolder = String.Join("_", subfolder.Split(invalids, StringSplitOptions.RemoveEmptyEntries)).TrimEnd('.');
                    filename = Path.GetFileName(filename);
                }
                else
                {
                    filename = Guid.NewGuid().ToString() + ".jpg";
                    if (cmd.colSubFolder >= 0)
                        subfolder = cols[cmd.colSubFolder];
                }
                string image_file_name = Path.Combine(cmd.outFolder, subfolder, filename);
                if (!Directory.Exists(Path.GetDirectoryName(image_file_name)))
                    Directory.CreateDirectory(Path.GetDirectoryName(image_file_name));
                File.WriteAllBytes(image_file_name, Convert.FromBase64String(cols[cmd.colImage]));
                Console.Write("Images saved: {0}\r", ++count);
            }
            Console.WriteLine("\nDone!");
        }

        class ArgsRemoveNonImage
        {
            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
            [Argument(ArgumentType.Required, HelpText = "Output TSV file")]
            public string outTsv = null;
            [Argument(ArgumentType.Required, HelpText = "Column index for image")]
            public int imageCol = -1;
        }

        static void RemoveNonImage(ArgsRemoveNonImage cmd)
        {
            int total_lines = 0;
            int count = 0;
            var lines = File.ReadLines(cmd.inTsv).AsParallel().AsOrdered()
            .Where(line =>
                {
                    var cols = line.Split('\t');
                    string strImage = cols[cmd.imageCol];
                    bool imageValid = true;
                    try
                    {
                        using (var ms = new MemoryStream(Convert.FromBase64String(strImage)))
                        using (var bmp = new Bitmap(ms))
                        {
                            count++;
                        }
                    }
                    catch (Exception)
                    {
                        imageValid = false;
                        Console.WriteLine();
                        Console.WriteLine("Skip: {0}", line.Substring(0, Math.Min(120, line.Length)));
                    }
                    total_lines++;
                    Console.Write("Total lines read {0}, skipped {1}\r", total_lines, total_lines - count);
                    return imageValid;
                });

            File.WriteAllLines(cmd.outTsv, lines);
            Console.WriteLine();
            Console.WriteLine("Total lines: {0}, saved {1}", total_lines, count);
        }

        class ArgsDumpB64
        {
            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
            [Argument(ArgumentType.Required, HelpText = "Col index")]
            public int col = 0;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Skip lines")]
            public int skip = 0;
            [Argument(ArgumentType.Required, HelpText = "Output file")]
            public string outFile = null;
        }

        static void DumpB64(ArgsDumpB64 cmd)
        {
            var cols = File.ReadLines(cmd.inTsv).Skip(cmd.skip).First().Split('\t');
            var data = Convert.FromBase64String(cols[cmd.col]);
            File.WriteAllBytes(cmd.outFile, data);
        }

        class ArgsSplit
        {
            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Training data ratio")]
            public double ratio = 0.8;
        }

        static void Split(ArgsSplit cmd)
        {
            string train_file = Path.GetFileNameWithoutExtension(cmd.inTsv) + ".train.tsv";
            string test_file = Path.GetFileNameWithoutExtension(cmd.inTsv) + ".test.tsv";

            Random rnd = new Random();
            int count = 0, count_train = 0, count_test = 0;
            using (var sw_train = new StreamWriter(train_file))
            using (var sw_test = new StreamWriter(test_file))
            {
                var lines = File.ReadLines(cmd.inTsv);
                foreach (var line in lines)
                {
                    if (rnd.NextDouble() < cmd.ratio)
                    {
                        sw_train.WriteLine(line);
                        count_train++;
                    }
                    else
                    {
                        sw_test.WriteLine(line);
                        count_test++;
                    }
                    count++;
                    Console.Write("Lines: {0}, train: {1}, test: {2}\r", count, count_train, count_test);
                }
            }
            Console.WriteLine("\nDone!");
        }

        class ArgsClassSplit
        {
            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Column index for label string")]
            public int labelCol = -1;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Training data ratio")]
            public double ratio = 0.8;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Minimal Training Sample Number: ignore the class with less than [min] samples")]
            public int min = 1;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximal Training Sample Number: drop the extra data after a class has [max] samples, -1 (default value) means no drop/use all samples")]
            public int max = -1;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output Train Shuffle file (default: replace inTsv file ext. with .train.shuffle")]
            public string outShuffleTrain = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output Train Shuffle file (default: replace inTsv file ext. with .test.shuffle")]
            public string outShuffleTest = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output intermediate file (default: replace inTsv file ext. with .inter.tsv")]
            public string outIntermediate = null;
        }

        static void ClassSplit(ArgsClassSplit cmd)
        {
            Random rnd = new Random();

            if (cmd.outShuffleTrain == null)
                cmd.outShuffleTrain = Path.ChangeExtension(cmd.inTsv, ".train.shuffle");
            if (cmd.outShuffleTest == null)
                cmd.outShuffleTest = Path.ChangeExtension(cmd.inTsv, ".test.shuffle");
            if (cmd.outIntermediate == null)
                cmd.outIntermediate = Path.ChangeExtension(cmd.inTsv, ".split.inter.tsv");
            var sw_intermediate = new StreamWriter(cmd.outIntermediate);

            int count = 0;
            int total_sample_selected, total_train, total_test;
            total_train = total_test = total_sample_selected = 0;
            int group = 0;
            int small_group = 0;    //counter for groups which has less than cmd.min samples, all the samples in these groups will be ignored
            int total_ignored_small = 0;
            int big_group = 0;      //counter for groups which has more than cmd.max samples, the extra samples (after cmd.max) will be ignored (randomly)
            int total_ignored_big = 0;

            var split = File.ReadLines(cmd.inTsv)
                //.AsParallel()
                .Select((line, i) =>
                {
                    Console.Write("lines processed: {0}\r", ++count);
                    var cols = line.Split('\t');
                    var label = cols[cmd.labelCol];
                    return new
                    {
                        lineNumber = i,
                        clsLabel = label
                    };
                })
                .GroupBy(tp => tp.clsLabel)
                //.AsParallel()   //will this conflict with Random shuffle below?
                .Where(g =>
                {
                    Console.Write("groups processed: {0}\r", ++group); 

                    var total = g.Count();
                    if (total < cmd.min)
                    {
                        small_group++;
                        total_ignored_small += total;
                        sw_intermediate.WriteLine(
                            "Group[{0}]\tClassLabel:{1}\tTotalSample#:{2}\t<min({3}), ignored",
                            group, g.Key, total, cmd.min);
                    }
                    return total >= cmd.min;
                })
                .Select(g =>
                {
                    int total = g.Count();
                    int sample_selected = (cmd.max >= cmd.min && cmd.max > 0) ? Math.Min(cmd.max, total) : total;
                    int train_num = (int)(Math.Ceiling(sample_selected * cmd.ratio));
                    int test_num = sample_selected - train_num;
                    int ignored_num = total - sample_selected;
                    if (total > sample_selected)
                    {
                        big_group++;
                        total_ignored_big += ignored_num;
                    }
                    total_sample_selected += sample_selected;
                    total_train += train_num;
                    total_test += test_num;

                    var shuffle = g.AsEnumerable()
                        .Select(tp => Tuple.Create(tp.lineNumber, rnd.Next()))
                        .OrderBy(tp => tp.Item2)
                        .Select(tp => tp.Item1)
                        .Take(sample_selected)
                        .ToArray();

                    var train = shuffle.Take(train_num).ToArray();
                    var test = shuffle.Skip(train_num).ToArray();

                    sw_intermediate.WriteLine(
                        "Group[{0}]\tClassLabel:{1}\tTotalSample#:{2}\tTrainSampleNum:{3}\tTestSampleNum:{4}\tIgnoredSampleNum:{5}",
                        group, g.Key, total, train_num, test_num, ignored_num);
                    return Tuple.Create(train, test, g.Key);                    
                })
                .ToList();

            var train_shuffle = split
                .SelectMany(train_test => train_test.Item1)
                .Select(num => Tuple.Create(num, rnd.Next()))
                .OrderBy(tp => tp.Item2)
                .Select(tp => tp.Item1.ToString());

            var test_shuffle = split
                .SelectMany(train_test => train_test.Item2)
                .Select(num => Tuple.Create(num, rnd.Next()))
                .OrderBy(tp => tp.Item2)
                .Select(tp => tp.Item1.ToString());

            var LabelList = split
                .Select((tp, i) => tp.Item3 + "\t" + i.ToString());

            File.WriteAllLines(cmd.outShuffleTrain, train_shuffle);
            File.WriteAllLines(cmd.outShuffleTest, test_shuffle);
            File.WriteAllLines(Path.ChangeExtension(cmd.inTsv, ".selected.labelmap"), LabelList);
            Console.WriteLine("\nDone!");

            var ClassSummary = String.Format(
                "\nTotal Classes: {0}\tsmall_class(sample#<{1}): {2}({6}%)\tbig_class(sample#>{3}({7}%)): {4}\tselected_class: {5}({8}%)\n",
                group, cmd.min, small_group, cmd.max, big_group, group - small_group,
                small_group * 100.0f / group, big_group * 100.0f / group, (group - small_group) * 100.0f / group);
            var SampleSummary = String.Format(
                "\nTotal Lines: {0}\ttrain: {1}({5}%)\ttest: {2}({6}%)\tignored_in_small_class: {3}({7}%)\tignored_in_big_class: {4}({8}%)\n",
                count, total_train, total_test, total_ignored_small, total_ignored_big,
                total_train * 100.0f / count, total_test * 100.0f / count, total_ignored_small * 100.0f / count, total_ignored_big * 100.0f / count);

            Console.WriteLine(ClassSummary);
            Console.WriteLine(SampleSummary);

            sw_intermediate.WriteLine(ClassSummary);
            sw_intermediate.WriteLine(SampleSummary);
            sw_intermediate.Close();

        }

        class ArgsTriplet
        {
            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output TSV file (default: replace inTsv file ext. with .triplet.shuffle)")]
            public string outTsv = null;
            [Argument(ArgumentType.Required, HelpText = "Column index for label")]
            public int labelCol = -1;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Batch size (default: 256)")]
            public int batchSize = 256;
            [Argument(ArgumentType.AtMostOnce, HelpText = "# of positive data in a batch (default: 10)")]
            public int posNum = 40;
        }

        static void Triplet(ArgsTriplet cmd)
        {
            if (cmd.outTsv == null)
                cmd.outTsv = Path.ChangeExtension(cmd.inTsv, ".triplet.shuffle");

            Console.WriteLine("Loading labels ...");
            var labels = File.ReadLines(cmd.inTsv)
                .Select(line => line.Split('\t'))
                .Select((cols, idx) => new { label = Convert.ToInt32(cols[cmd.labelCol]), line_num = idx })
                .ToArray();
            Console.WriteLine("Labels loaded: {0}", labels.Count());

            var groups = labels
                .GroupBy(x => x.label)
                .ToArray();
            Console.WriteLine("Distinct labels: {0}", groups.Count());

            var results = new List<int>();
            Random rnd = new Random();

            int num_batches = labels.Count() / cmd.posNum; // roughly let every data point act as anchor once
            for (int i = 0; i < num_batches; i++)
            {
                var g = groups[rnd.Next(groups.Count())];
                var pos_label = g.Key;
                var batch = g.AsEnumerable()
                    .Select(lbl => new { label = lbl, rnd = rnd.Next() })
                    .OrderBy(x => x.rnd)
                    .Take(cmd.posNum)
                    .Select(x => x.label)
                    .ToList();
                var batch_dict = batch.ToDictionary(x => x.line_num, x => 0);

                while (batch.Count() < cmd.batchSize)
                {
                    var neg = labels[rnd.Next(labels.Count())];
                    while (neg.label == pos_label || batch_dict.ContainsKey(neg.line_num))
                        neg = labels[rnd.Next(labels.Count())];

                    batch.Add(neg);
                    batch_dict.Add(neg.line_num, 0);
                }

                results.AddRange(batch.Select(x => x.line_num));

                Console.Write("Batches generated: {0} of {1}. Total samples: {2}\r", i + 1, num_batches, results.Count());
            }

            File.WriteAllLines(cmd.outTsv, results.Select(lbl => lbl.ToString()));
            Console.WriteLine("\nDone.");
        }

        class ArgsFilterLabels
        {
            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
            [Argument(ArgumentType.Required, HelpText = "Column index for label")]
            public int labelCol = -1;
            [Argument(ArgumentType.AtMostOnce, HelpText = "White list label file (the first column will be used)")]
            public string whitelist = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Black list label file (the first column will be used)")]
            public string blacklist = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output TSV file (default: replace inTsv file ext. with .selected.tsv")]
            public string outTsv = null;
        }

        static void FilterLabels(ArgsFilterLabels cmd)
        {
            if (cmd.whitelist == null && cmd.blacklist == null)
            {
                Console.WriteLine("Please provide at least a whitelist or a blacklist");
                return;
            }
            if (cmd.outTsv == null)
                cmd.outTsv = Path.ChangeExtension(cmd.inTsv, ".selected.tsv");

            List<string> whitelist;
            if (cmd.whitelist == null)
            {
                Console.WriteLine("Loading input TSV as white list...");
                whitelist = File.ReadLines(cmd.inTsv)
                    .Select(line => line.Split('\t')[cmd.labelCol])
                    .Distinct()
                    .OrderBy(label => label)
                    .ToList();
            }
            else
            {
                Console.WriteLine("Loading whilte list file...");
                whitelist = File.ReadLines(cmd.whitelist)
                    .Select(line => line.Split('\t')[0])
                    .OrderBy(label => label)
                    .ToList();
            }
            Console.WriteLine("Distinct labels in white list: {0}", whitelist.Count());
            if (cmd.blacklist != null)
            {
                Console.Write("Loading black list file...");
                var blacklist = File.ReadLines(cmd.blacklist)
                    .Select(line => line.Split('\t')[0])
                    .OrderBy(label => label)
                    .ToList();
                Console.WriteLine(" labels loaded: {0}", blacklist.Count());
                Console.WriteLine("Subtracking black list from white list...");
                whitelist = whitelist.Except(blacklist).ToList();
                Console.WriteLine("Distinct labels after black list subtraction: {0}", whitelist.Count());
            }

            var dict = whitelist.OrderBy(x => x)
                    .Select((x, i) => Tuple.Create(x, i))
                    .ToDictionary(x => x.Item1, x => x.Item2);

            string labelmap = Path.ChangeExtension(cmd.inTsv, ".labelmap");
            File.WriteAllLines(labelmap,
                    dict.ToArray().OrderBy(kv => kv.Value).Select(kv => kv.Key + "\t" + kv.Value.ToString()));
            Console.WriteLine("Labelmap file saved to: {0}", labelmap);

            int count = 0;
            var lines = File.ReadLines(cmd.inTsv)
                //.AsParallel().AsOrdered()
                .Select(line =>
                {
                    ++count;
                    if (count % 1000 == 0)
                        Console.Write("Lines processed: {0}\r", count);
                    return line.Split('\t').ToList();
                })
                .Select(cols =>
                    {
                        int cls_id;
                        if (!dict.TryGetValue(cols[cmd.labelCol], out cls_id))
                            cls_id = -1;
                        cols.Add(cls_id.ToString());
                        return cols;
                    })
                .Select(cols => string.Join("\t", cols));

            File.WriteAllLines(cmd.outTsv, lines);
            Console.Write("Lines processed: {0}\r", count);
            Console.WriteLine("\nDone.");
        }

        class ArgsCutRow
        {
            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
            [Argument(ArgumentType.Required, HelpText = "Input line number files (use column 0)")]
            public string lines = null;
            [Argument(ArgumentType.Required, HelpText = "Output TSV file")]
            public string outTsv = null;
        }

        static void CutRow(ArgsCutRow cmd)
        {
            var lines = File.ReadLines(cmd.lines)
                .Select(line => line.Split('\t')[0])
                .Select(x => Convert.ToInt32(x));
            var lineDict = new HashSet<int>(lines);
            Console.WriteLine("Line numbers loaded: {0}", lines.Count());

            int count = 0;
            var outLines = File.ReadLines(cmd.inTsv)
                .Where((tp, i) =>
                {
                    bool keep = lineDict.Contains(i);
                    if (keep)
                        ++count;
                    Console.Write("Lines processed in input TSV: {0}\r", count);
                    return keep;
                });

            File.WriteAllLines(cmd.outTsv, outLines);
            Console.WriteLine("\nDone.");
        }

        class ArgsCutCol
        {
            [Argument(ArgumentType.Required, HelpText = "Input TSV file")]
            public string inTsv = null;
            [Argument(ArgumentType.Required, HelpText = "Columns to cut, e.g. 0,2,4,7")]
            public string columns = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output TSV file (default: replace inTsv .ext with .cut.tsv)")]
            public string outTsv = null;
        }

        static void CutCol(ArgsCutCol cmd)
        {
            if (cmd.outTsv == null)
                cmd.outTsv = Path.ChangeExtension(cmd.inTsv, ".cut.tsv");

            var col_indices = cmd.columns.Split(',').Select(x => Convert.ToInt32(x)).ToArray();

            int count = 0;
            var lines = File.ReadLines(cmd.inTsv)
                .Select(line => line.Split('\t'))
                .Select(cols =>
                {
                    var new_cols = col_indices.Select(col_idx => cols[col_idx]);
                    if (++count % 100 == 0)
                        Console.Write("Lines processed: {0}\r", count);
                    return new_cols;
                })
                .Select(cols => string.Join("\t", cols));

            File.WriteAllLines(cmd.outTsv, lines);
            Console.Write("Lines processed: {0}", count);
            Console.WriteLine("\nDone.");
        }

        class ArgsFilterLines
        {
            [Argument(ArgumentType.Required, HelpText = "TSV A with optional col index (default 0) prefixed with '?'. E.g. abc.tsv?1")]
            public string a = null;
            [Argument(ArgumentType.Required, HelpText = "TSV B with optional col index (default 0) prefixed with '?'. E.g. abc.tsv?1")]
            public string b = null;
            [Argument(ArgumentType.Required, HelpText = "Output TSV file")]
            public string outTsv = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Reverse Cut (default: false). That is, keep lines in A with keys NOT in B")]
            public bool reverse = false;
        }

        static void FilterLines(ArgsFilterLines cmd)
        {
            var arg_a = cmd.a.Split('?');
            var arg_b = cmd.b.Split('?');
            cmd.a = arg_a[0];
            cmd.b = arg_b[0];
            int col_a = arg_a.Length > 1 ? Convert.ToInt32(arg_a[1]) : 0;
            int col_b = arg_b.Length > 1 ? Convert.ToInt32(arg_b[1]) : 0;

            var set_b = new HashSet<string>(File.ReadLines(cmd.b)
                                            .Select(line => line.Split('\t')[col_b])
                                            .Distinct(),
                                            StringComparer.Ordinal);
            Console.WriteLine("Distinct labels in B: {0}", set_b.Count());

            int count = 0;
            var lines = File.ReadLines(cmd.a)
                .Where(line =>
                {
                    Console.Write("Lines processed: {0}\r", ++count);
                    bool contain = set_b.Contains(line.Split('\t')[col_a]);
                    return cmd.reverse ? !contain : contain;
                });

            File.WriteAllLines(cmd.outTsv, lines);
            Console.WriteLine("\nDone.");
        }

        class ArgsPasteCol
        {
            [Argument(ArgumentType.Required, HelpText = "TSV A ")]
            public string a = null;
            [Argument(ArgumentType.Required, HelpText = "TSV B")]
            public string b = null;
            [Argument(ArgumentType.Required, HelpText = "Output TSV file")]
            public string outTsv = null;
        }

        static void PasteCol(ArgsPasteCol cmd)
        {
            int count = 0;
            var b_lines = File.ReadLines(cmd.b);
            var lines = File.ReadLines(cmd.a)
                .Zip(b_lines, (first, second) =>
                {
                    Console.Write("Lines processed: {0}\r", ++count);
                    return first + "\t" + second;
                });

            File.WriteAllLines(cmd.outTsv, lines);
            Console.WriteLine("\nDone.");
        }

        static void Main(string[] args)
        {
            ParserX.AddTask<ArgsLabel>(Label, "Generate label file with class id and generate (or use) .labelmap file");
            ParserX.AddTask<ArgsIndex>(Index, "Build line index for random access");
            ParserX.AddTask<ArgsShuffle>(Shuffle, "Shuffle data by generating shuffle line number list");
            
            ParserX.AddTask<ArgsCutCol>(CutCol, "Cut columns from TSV file");
            ParserX.AddTask<ArgsCutRow>(CutRow, "Cut rows based on line number files (normally a shuffle file)");
            ParserX.AddTask<ArgsPasteCol>(PasteCol, "Paste two TSV files by concatenating corresponding lines seperated by TAB");
            ParserX.AddTask<ArgsFilterLines>(FilterLines, "Filter lines in A with keys in B (or reversely, not in B), according to the specified column");
            ParserX.AddTask<ArgsClassSplit>(ClassSplit, "Split data into training and testing acording to class labels");

            ParserX.AddTask<ArgsList2Tsv>(List2Tsv, "Generate TSV file from list file");
            ParserX.AddTask<ArgsFolder2Tsv>(Folder2Tsv, "Generate TSV file from folder images");
            ParserX.AddTask<ArgsTsv2Folder>(Tsv2Folder, "Unpack images in TSV file to folder images");
            
            ParserX.AddTask<ArgsRemoveNonImage>(RemoveNonImage, "Remove non image lines in TSV");
            ParserX.AddTask<ArgsDumpB64>(DumpB64, "Dump and decode base64_encoded data");

            ParserX.AddTask<ArgsSplit>(Split, "Split data into training and testing");
            ParserX.AddTask<ArgsTriplet>(Triplet, "Generate triplet shuffle file");
            ParserX.AddTask<ArgsFilterLabels>(FilterLabels, "Filter data based on label dict");
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
