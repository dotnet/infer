// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace CrowdsourcingWithWords
{
    using System.Collections.Generic;
    using System.IO;
    using System.IO.Compression;
    using System.Linq;

    /// <summary>
    /// This class represents a single datum, and has methods to read in data.
    /// </summary>
    public class Datum
    {
        /// <summary>
        /// The worker id.
        /// </summary>
        public string WorkerId;

        /// <summary>
        /// The task id.
        /// </summary>
        public string TaskId;

        /// <summary>
        /// The worker's label.
        /// </summary>
        public int WorkerLabel;

        /// <summary>
        /// The task's gold label (optional).
        /// </summary>
        public int? GoldLabel;

        /// <summary>
        /// The body text of the document (optional - only for text sentiment labelling tasks).
        /// </summary>
        public string BodyText;

        /// <summary>
        /// Loads the data file in the format (worker id, task id, worker label, ?gold label).
        /// </summary>
        /// <param name="filename">The data file.</param>
        /// <param name="maxLength"></param>
        /// <returns>The list of parsed data.</returns>
        public static IList<Datum> LoadData(string filename, int maxLength = short.MaxValue)
        {
            var result = new List<Datum>();
            foreach (string line in ReadLinesGzip(filename).Take(maxLength))
            {
                var strarr = line.Split('\t');
                int length = strarr.Length;

                var datum = new Datum
                {
                    WorkerId = strarr[0],
                    TaskId = strarr[1],
                    WorkerLabel = int.Parse(strarr[2]),
                    BodyText = strarr[3]
                };

                if (length >= 5)
                    datum.GoldLabel = int.Parse(strarr[4]);
                else
                    datum.GoldLabel = null;

                result.Add(datum);
            }

            return result;
        }

        public static IEnumerable<string> ReadLinesGzip(string fileName)
        {
            using (Stream stream = File.Open(fileName, FileMode.Open))
            {
                Stream gz = (Path.GetExtension(fileName) == ".gz") ? new GZipStream(stream, CompressionMode.Decompress) : stream;
                using (var streamReader = new StreamReader(gz))
                {
                    while (true)
                    {
                        string line = streamReader.ReadLine();
                        if (line == null)
                            break;
                        yield return line;
                    }
                }
            }
        }
    }
}