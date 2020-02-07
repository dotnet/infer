// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners.MovieLens
{
    using System;
    using System.IO;
    using System.Text.RegularExpressions;

    public class MovieLensConverter
    {
        /// <summary>
        /// Reads the entity features from an input file,
        /// parses them and processes them (computes feature vectors),
        /// and stores the result in an output file.
        /// </summary>
        /// <param name="reader">The input file reader.</param>
        /// <param name="writer">The output file writer.</param>
        /// <param name="featureProcessor">The method which processes features.</param>
        private static void GenerateEntityFeatures(
            TextReader reader, TextWriter writer, Func<string, string> featureProcessor)
        {
            string features;
            while ((features = reader.ReadLine()) != null)
            {
                writer.WriteLine(featureProcessor(features));
            }
        }

        /// <summary>
        /// Converts raiting file.
        /// </summary>
        /// <param name="reader">The input file reader.</param>
        /// <param name="writer">The output file writer.</param>
        private static void ConvertRating(TextReader reader, TextWriter writer)
        {
            Regex expr = new Regex("::");
            string line;

            while ((line = reader.ReadLine()) != null)
            {
                int lastSemicolons = line.LastIndexOf(":") - 1;
                line = line.Remove(lastSemicolons);
                writer.WriteLine(expr.Replace(line, ","));
            }
        }


        /// <summary>
        /// Convert dataset files from the given directory to the one dataset file.
        /// </summary>
        /// <param name="moviesInfoFileName">The name of the file holds information about movies.</param>
        /// <param name="usersInfoFileName">The name of the file holds information about users.</param>
        /// <param name="ratingFileName">The name of the file holds information about rating.</param>
        /// <param name="outputFile">The output file name.</param>
        public static void Convert(string moviesInfoFileName, string usersInfoFileName, string ratingFileName, string outputFileName)
        {
            try
            {
                using (TextWriter writer = new StreamWriter(outputFileName))
                using (TextReader moviesReader = new StreamReader(moviesInfoFileName))
                using (TextReader usersReader = new StreamReader(usersInfoFileName))
                using (TextReader ratingReader = new StreamReader(ratingFileName))
                {
                    writer.WriteLine("R,1,5");
                    ConvertRating(ratingReader, writer);
                    GenerateEntityFeatures(usersReader, writer, FeatureProcessor.ProcessUserFeatures);
                    GenerateEntityFeatures(moviesReader, writer, FeatureProcessor.ProcessItemFeatures);
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