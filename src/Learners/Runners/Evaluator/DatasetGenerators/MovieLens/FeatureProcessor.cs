// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners.MovieLens
{
    using System;
    using System.Text;

    /// <summary>
    /// A processor of user and item features.
    /// </summary>
    public static class FeatureProcessor
    {
        /// <summary>
        /// Indicates whether to add a constant term in the beginning of each feature vector.
        /// </summary>
        public static readonly bool AddConstantTerm = true;

        /// <summary>
        /// The number of user features in the input data.
        /// </summary>
        public static readonly int InputUserFeatureCount = 5;

        /// <summary>
        /// The number of item features in the input data.
        /// </summary>
        public static readonly int InputItemFeatureCount = 3;

        /// <summary>
        /// An array of methods which process user features.
        /// </summary>
        private static readonly Func<int, string, Tuple<int, string>>[] UserFeatureProcessors =
            new Func<int, string, Tuple<int, string>>[]
                {
                    Features.ComputeUserGender, 
                    Features.ComputeUserAge, 
                    Features.ComputeUserOccupation
                };

        /// <summary>
        /// An array of methods which process item features.
        /// </summary>
        private static readonly Func<int, string, Tuple<int, string>>[] ItemFeatureProcessors =
            new Func<int, string, Tuple<int, string>>[]
                {
                    Features.ComputeMovieYear, 
                    Features.ComputeMovieGenre
                };

        /// <summary>
        /// Processes the features of a single user.
        /// </summary>
        /// <param name="inputFeatures">The input user features in a raw form.</param>
        /// <returns>The computed user feature vector.</returns>
        public static string ProcessUserFeatures(string inputFeatures)
        {
            return ProcessFeatures("U", inputFeatures, InputUserFeatureCount, UserFeatureProcessors);
        }

        /// <summary>
        /// Processes the features of a single item.
        /// </summary>
        /// <param name="inputFeatures">The input item features in a raw form.</param>
        /// <returns>The computed item feature vector.</returns>
        public static string ProcessItemFeatures(string inputFeatures)
        {
            return ProcessFeatures("I", inputFeatures, InputItemFeatureCount, ItemFeatureProcessors);
        }

        /// <summary>
        /// Processes the features of a single user or item.
        /// </summary>
        /// <param name="entityPrefix">The user or item prefix.</param>
        /// <param name="inputFeatures">The entity input features in a raw form.</param>
        /// <param name="inputFeatureCount">The expected number of input features.</param>
        /// <param name="featureProcessors">The method which computes the feature vector.</param>
        /// <returns>The internal representation of the entity features (prefix, id, feature vector).</returns>
        private static string ProcessFeatures(
            string entityPrefix, string inputFeatures, int inputFeatureCount, Func<int, string, Tuple<int, string>>[] featureProcessors)
        {
            string[] inputFeatureParts = inputFeatures.Split(new[] { "::" }, StringSplitOptions.None);

            if (inputFeatureParts.Length != inputFeatureCount)
            {
                throw new ArgumentException(
                    string.Format("There should be {0} features in {1}; given {2}.", inputFeatureCount, inputFeatures, inputFeatureParts.Length));
            }
            
            string id = inputFeatureParts[0];
            var result = new StringBuilder(string.Format("{0},{1},", entityPrefix, id));

            int offset = 0;

            if (AddConstantTerm)
            {
                result.Append("0:1|");
                offset = 1;
            }

            for (int i = 0; i < featureProcessors.Length; ++i)
            {
                var featureProcessor = featureProcessors[i];
                var inputFeature = inputFeatureParts[i + 1]; // Skip the id

                var countFeatureTuple = featureProcessor(offset, inputFeature);
                var bucketCount = countFeatureTuple.Item1;
                var feature = countFeatureTuple.Item2;

                offset += bucketCount;
                result.Append(feature);
                if (i != featureProcessors.Length - 1)
                {
                    result.Append("|");
                }
            }

            return result.ToString();
        }
    }
}
