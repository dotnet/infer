// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System.IO;
using System.Net;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Probabilistic.Learners.Runners.DatasetGenerators
{
    /// <summary>
    /// Generator for Book Crossing dataset.
    /// </summary>
    class BookCrossingGenerator : DownloadingDatasetGenerator
    {
        private const string link = "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip";
        private const string datFileName = "BX-Book-Ratings.csv";



        protected override void DownloadArchives(string tmpDir)
        {
            string tmpZipPath = Path.Combine(tmpDir, "tmp.zip");

            new WebClient().DownloadFile(link, tmpZipPath);
            System.IO.Compression.ZipFile.ExtractToDirectory(tmpZipPath, tmpDir);
        }

        protected override void MakeDataset(string tmpDir, string outputFileName)
        {
            using (TextReader reader = new StreamReader(Path.Combine(tmpDir, datFileName)))
            {
                reader.ReadLine(); // skip first line, that holds description
                try
                {
                    using (TextWriter writer = new StreamWriter(outputFileName))
                    {
                        // Add raiting bounds
                        writer.WriteLine("R,0,10");

                        // Regular expressions for deleting unescaped characters
                        Regex comaExpr = new Regex(@"([^\\]),");
                        Regex quotesExpr = new Regex(@"([^\\])\""");
                        Regex semicolonExpr = new Regex(@"([^\\]);");

                        string line;

                        while ((line = reader.ReadLine()) != null)
                        {
                            line = comaExpr.Replace(line, "$1 ");
                            line = quotesExpr.Replace(line, "$1");
                            line = semicolonExpr.Replace(line, "$1,");

                            writer.WriteLine(line.TrimStart('"'));
                        }
                    }
                }
                catch
                {
                    if (File.Exists(outputFileName))
                    {
                        File.Delete(outputFileName);
                    }

                    throw;
                }
            }
        }
    }
}
