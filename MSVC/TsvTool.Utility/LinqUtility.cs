using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace TsvTool.Utility
{
    public static class LinqUtility
    {
        public static IEnumerable<T> ReportProgress<T>(this IEnumerable<T> source, string reportMsg)
        {
            int count = 0;
            foreach (var item in source)
            {
                Console.Write("{0}: {1}\r", reportMsg, ++count);
                yield return item;
            }
        }

        public static IEnumerable<IEnumerable<T>> Batch<T>(this IEnumerable<T> source, int size)
        {
            T[] bucket = null;
            var count = 0;

            foreach (var item in source)
            {
                if (bucket == null)
                    bucket = new T[size];

                bucket[count++] = item;

                if (count != size)
                    continue;

                yield return bucket.Select(x => x);

                bucket = null;
                count = 0;
            }

            // Return the last bucket with all remaining elements
            if (bucket != null && count > 0)
                yield return bucket.Take(count);
        }
    }
}
