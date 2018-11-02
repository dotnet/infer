// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// Reads and writes classification data.
    /// </summary>
    public static class ClassifierPersistenceUtils
    {
        /// <summary>
        /// Reads label distributions from a file with the specified name.
        /// </summary>
        /// <param name="fileName">The file name.</param>
        /// <param name="labelSet">An optional set of labels.</param>
        /// <returns>A list of label distributions.</returns>
        public static IList<LabelDistribution> LoadLabelDistributions(string fileName, IndexedSet<string> labelSet = null)
        {
            if (string.IsNullOrWhiteSpace(fileName))
            {
                throw new ArgumentException("The name of the file must not be null or whitespace.", nameof(fileName));
            }

            var labels = new List<LabelDistribution>();
            var labelDictionary = labelSet ?? new IndexedSet<string>();
            var parsingContext = new FileParsingContext(fileName);

            using (var reader = new StreamReader(fileName))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line))
                    {
                        continue;
                    }

                    line = line.Trim();
                    if (line.StartsWith("#") || line.StartsWith("//") || line.StartsWith("%"))
                    {
                        continue;
                    }

                    try
                    {
                        labels.Add(LabelDistribution.Parse(line, labelDictionary));
                    }
                    catch (Exception e)
                    {
                        parsingContext.RaiseError("{0}", e.Message);
                    }
                }
            }

            return labels;
        }

        /// <summary>
        /// Reads labeled feature values from a file with the specified name.
        /// </summary>
        /// <param name="fileName">The file name.</param>
        /// <param name="labelSet">An optional set of labels.</param>
        /// <param name="featureSet">An optional set of features.</param>
        /// <returns>A list of labeled feature values.</returns>
        public static IList<LabeledFeatureValues> LoadLabeledFeatureValues(
            string fileName, 
            IndexedSet<string> labelSet = null,
            IndexedSet<string> featureSet = null)
        {
            if (string.IsNullOrWhiteSpace(fileName))
            {
                throw new ArgumentException("The name of the file must not be null or whitespace.", nameof(fileName));
            }

            var labeledFeatureValues = new List<LabeledFeatureValues>();
            var labelDictionary = labelSet ?? new IndexedSet<string>();
            var featureDictionary = featureSet ?? new IndexedSet<string>();
            var parsingContext = new FileParsingContext(fileName);

            using (var reader = new StreamReader(fileName))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line))
                    {
                        continue;
                    }

                    line = line.Trim();
                    if (line.StartsWith("#") || line.StartsWith("//") || line.StartsWith("%"))
                    {
                        continue;
                    }

                    try
                    {
                        labeledFeatureValues.Add(LabeledFeatureValues.Parse(line, labelDictionary, featureDictionary));
                    }
                    catch (Exception e)
                    {
                        parsingContext.RaiseError("{0}", e.Message);
                    }
                }
            }

            return labeledFeatureValues;
        }

        /// <summary>
        /// Writes a collection of label distributions to the file with the specified name.
        /// </summary>
        /// <param name="fileName">The file name.</param>
        /// <param name="labelDistributions">A collection of label distributions.</param>
        public static void SaveLabelDistributions(
            string fileName, IEnumerable<IDictionary<string, double>> labelDistributions)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            if (labelDistributions == null)
            {
                throw new ArgumentNullException(nameof(labelDistributions));
            }

            using (var writer = new StreamWriter(fileName))
            {
                foreach (var labelDistribution in labelDistributions)
                {
                    foreach (var uncertainLabel in labelDistribution)
                    {
                        writer.Write(
                            "{0}{1}{2}{3}",
                            uncertainLabel.Equals(labelDistribution.First()) ? string.Empty : string.Empty + LabelDistribution.PointSeparator,
                            uncertainLabel.Key,
                            LabelDistribution.LabelProbabilitySeparator,
                            uncertainLabel.Value);
                    }

                    writer.WriteLine();
                }
            }
        }

        /// <summary>
        /// Writes a collection of vectors to the file with the specified name.
        /// </summary>
        /// <param name="fileName">The file name.</param>
        /// <param name="vectors">A collection of vectors.</param>
        public static void SaveVectors(string fileName, IEnumerable<Vector> vectors)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            if (vectors == null)
            {
                throw new ArgumentNullException(nameof(vectors));
            }

            using (var writer = new StreamWriter(fileName))
            {
                foreach (var vector in vectors)
                {
                    writer.WriteLine(vector);
                }
            }
        }
    }
}