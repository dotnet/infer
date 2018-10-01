// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.IO;
using System.Net;
using Microsoft.ML.Probabilistic.Learners.Runners.MovieLens;

namespace Microsoft.ML.Probabilistic.Learners.Runners.DatasetGenerators
{
    /// <summary>
    /// Generator for the Movie Lens dataset.
    /// </summary>
    class MovieLensGenerator : DownloadingDatasetGenerator
    {
        private const string link = "http://files.grouplens.org/datasets/movielens/ml-1m.zip";
        private const string datDirectoryName = "ml-1m";
        private const string usersDatName = "users.dat";
        private const string moviesDatName = "movies.dat";
        private const string ratingsDatName = "ratings.dat";



        protected override void DownloadArchives(string tmpDir)
        {
            string tmpZipPath = Path.Combine(tmpDir, "tmp.zip");

            new WebClient().DownloadFile(link, tmpZipPath);
            System.IO.Compression.ZipFile.ExtractToDirectory(tmpZipPath, tmpDir);
        }

        protected override void MakeDataset(string tmpDir, string outputFileName)
        {
            MovieLensConverter.Convert(
                Path.Combine(tmpDir, datDirectoryName, moviesDatName),
                Path.Combine(tmpDir, datDirectoryName, usersDatName),
                Path.Combine(tmpDir, datDirectoryName, ratingsDatName),
                outputFileName
            );
        }
    }
}
