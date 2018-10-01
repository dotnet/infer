// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.IO;

namespace Microsoft.ML.Probabilistic.Learners.Runners.DatasetGenerators
{
    /// <summary>
    /// Generator that downloads and creates a dataset from downloaded archives.
    /// </summary>
    abstract class DownloadingDatasetGenerator : IDatasetGenerator
    {
        /// <summary>
        /// Download needed archives and unpack them.
        /// </summary>
        /// <param name="tmpDir">Path to a temporary directory.</param>
        protected abstract void DownloadArchives(string tmpDir);

        /// <summary>
        /// The main dataset creation process.
        /// </summary>
        /// <param name="tmpDir">Path to a temporary directory holds unpacked archives.</param>
        /// <param name="outputFileName">The name of the generating dataset file.</param>
        protected abstract void MakeDataset(string tmpDir, string outputFileName);

        public void Generate(string fileName)
        {
            string tmpDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
            Directory.CreateDirectory(tmpDir);

            try
            { 
                DownloadArchives(tmpDir);

                if (!Directory.Exists(Path.GetDirectoryName(fileName)))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(fileName));
                }

                MakeDataset(tmpDir, fileName);
            }
            finally
            {
                Directory.Delete(tmpDir, true);
            }
        }
    }
}
