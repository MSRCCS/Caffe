using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace TsvTool
{
    public class FileIndex
    {
        private static long GetActualPosition(StreamReader reader)
        {
            // The current buffer of decoded characters
            char[] charBuffer = (char[])reader.GetType().InvokeMember("charBuffer"
                , System.Reflection.BindingFlags.DeclaredOnly | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.GetField
                , null, reader, null);

            // The current position in the buffer of decoded characters
            int charPos = (int)reader.GetType().InvokeMember("charPos"
                , System.Reflection.BindingFlags.DeclaredOnly | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.GetField
                , null, reader, null);

            // The number of bytes that the already-read characters need when encoded.
            int numReadBytes = reader.CurrentEncoding.GetByteCount(charBuffer, 0, charPos);

            // The number of encoded bytes that are in the current buffer
            int byteLen = (int)reader.GetType().InvokeMember("byteLen"
                , System.Reflection.BindingFlags.DeclaredOnly | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.GetField
                , null, reader, null);

            return reader.BaseStream.Position - byteLen + numReadBytes;
        }

        public static void BuildLineIndex(string inputFile)
        {
            List<long> index = new List<long>();

            int nLines = 0;
            using (StreamReader sr = new StreamReader(inputFile))
            {
                while (!sr.EndOfStream)
                {
                    long currentLinePos = GetActualPosition(sr);

                    string line = sr.ReadLine();
                    if (line.Length == 0)
                        Console.WriteLine("Warning! Empty line encountered! line#{0}", nLines + 1);

                    index.Add(currentLinePos);
                    nLines += 1;

                    if (nLines % 1000 == 0)
                        Console.Write("Scanned {0}\r", nLines);
                }
            }

            Console.WriteLine("Scanned {0}", nLines);

            File.WriteAllLines(Path.ChangeExtension(inputFile, "lineidx"),
                               index.Select(n => n.ToString()));
            Console.WriteLine("Index file saved!");
        }
    }
}