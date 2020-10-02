// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.IO;
using System.Net;
using ExcelDataReader;

namespace Microsoft.ML.Probabilistic.Learners.Runners.DatasetGenerators
{
    /// <summary>
    /// Generator for the Jester dataset.
    /// </summary>
    class JesterGenerator : DownloadingDatasetGenerator
    {
        private const string link1 = "http://www.ieor.berkeley.edu/~goldberg/jester-data/jester-data-1.zip";
        private const string link2 = "http://www.ieor.berkeley.edu/~goldberg/jester-data/jester-data-2.zip";
        private const string link3 = "http://www.ieor.berkeley.edu/~goldberg/jester-data/jester-data-3.zip";

        private const string datFileName1 = "jester-data-1.xls";
        private const string datFileName2 = "jester-data-2.xls";
        private const string datFileName3 = "jester-data-3.xls";

        private const int columnCount = 100;
        private const int unsetValue = 99;



        protected override void DownloadArchives(string tmpDir)
        {
            GenerateFile(link1, tmpDir);
            GenerateFile(link2, tmpDir);
            GenerateFile(link3, tmpDir);
        }

        protected override void MakeDataset(string tmpDir, string outputFileName)
        {
            string inputFile1 = Path.Combine(tmpDir, datFileName1);
            string inputFile2 = Path.Combine(tmpDir, datFileName2);
            string inputFile3 = Path.Combine(tmpDir, datFileName3);

            try
            {
                using (TextWriter writer = new StreamWriter(outputFileName))
                {
                    int user = 1;

                    writer.WriteLine("R,-10,10");

                    user = Generating(inputFile1, writer, user);
                    user = Generating(inputFile2, writer, user);
                    user = Generating(inputFile3, writer, user);
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

        /// <summary>
        /// Generate a part of the dataset.
        /// </summary>
        /// <param name="inputFile">Input file name.</param>
        /// <param name="writer">Writer for output file.</param>
        /// <param name="user">User ID from which generating begins.</param>
        /// <returns>User ID which ends generating.</returns>
        private int Generating(string inputFile, TextWriter writer, int user)
        {
#if NETCOREAPP
            System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
#endif
            using (var stream = File.Open(inputFile, FileMode.Open, FileAccess.Read))
            {
                using (var reader = ExcelReaderFactory.CreateReader(stream))
                {
                    while (reader.Read())
                    {
                        for (int column = 1; column <= columnCount; column++)
                        {
                            int val = Convert.ToInt32(System.Math.Round(reader.GetDouble(column)));

                            if (val != unsetValue)
                            {
                                writer.WriteLine($"{user},{column},{val}");
                            }
                        }

                        user++;
                    }
                }
            }

            return user;
        }

        /// <summary>
        /// Download and unpack files.
        /// </summary>
        /// <param name="link">Link for downloading.</param>
        /// <param name="downloadPath">Path to save files.</param>
        private void GenerateFile(string link, string downloadPath)
        {
            string tmpZipPath = Path.Combine(downloadPath, "tmp.zip");

            new WebClient().DownloadFile(link, tmpZipPath);
            System.IO.Compression.ZipFile.ExtractToDirectory(tmpZipPath, downloadPath);
        }
    }
}
