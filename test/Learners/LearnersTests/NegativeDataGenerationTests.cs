// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Tests
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;

    using Xunit;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Tests for the negative data generator.
    /// </summary>
    public class NegativeDataGenerationTests
    {
        /// <summary>
        /// The positive only training dataset.
        /// </summary>
        private PositiveOnlyDataset positiveOnlyDataset;

        /// <summary>
        /// The feature provider for the positive only dataset.
        /// </summary>
        private FeatureProvider featureProvider;

        /// <summary>
        /// Prepares environment (datasets etc) before each test.
        /// </summary>
        public NegativeDataGenerationTests()
        {
            this.InitializePositiveOnlyDataset();
            this.InitializeFeatureProvider();
        }

        /// <summary>
        /// Correctness test for the histogram sampler. Tests both sampling and decreasing.
        /// </summary>
        [Fact]
        public void HistogramSamplerTest()
        {
            Rand.Restart(12347);
            foreach (var parameters in new[]
                                           {
                                               new[] { 2, 8, 1, 9, 4, 6, 3 },
                                               new[] { 1 },
                                               new[] { 5 },
                                               new[] { 1, 2 },
                                               new[] { 1, 2, 3 },
                                               new[] { 1, 2, 3, 4 },
                                               new[] { 1, 2, 0, 4 },
                                               new[] { 0, 0, 0, 0 }
                                           })
            {
                this.TestHistogramSampler(parameters);
            }
        }

        /// <summary>
        /// Tests if the negative data generator mapping generates the correct 
        /// number of positive and negative ratings for each user and for each item.
        /// </summary>
        [Fact]
        public void NegativeDataGeneratorMappingTest()
        {
            IDictionary<string, RatingCounts> userRatingCounts;
            IDictionary<string, RatingCounts> itemRatingCounts;

            this.ComputeRatingCounts(out userRatingCounts, out itemRatingCounts);

            this.TestRatingCounts(userRatingCounts.Values, itemRatingCounts.Count);
            this.TestRatingCounts(itemRatingCounts.Values, userRatingCounts.Count);
        }

        /// <summary>
        /// Tests if a binary recommender with a chained mapping does not throw during training 
        /// or prediction, and if it predicts positive ratings for the positive data.
        /// </summary>
        [Fact]
        public void BinaryRecommendationRegressionTest()
        {
            // TODO: We need a static class which implements mapping chainings
            Rand.Restart(12347);
            var positiveOnlyMapping = new PositiveOnlyDataMapping();
            var negativeDataGeneratorMapping = positiveOnlyMapping.WithGeneratedNegativeData(1.0);
            var recommender = MatchboxRecommender.Create(negativeDataGeneratorMapping);
            recommender.Train(this.positiveOnlyDataset, this.featureProvider); // must not throw

            foreach (var instance in positiveOnlyMapping.GetInstances(this.positiveOnlyDataset))
            {
                var user = positiveOnlyMapping.GetUser(this.positiveOnlyDataset, instance);
                var item = positiveOnlyMapping.GetItem(this.positiveOnlyDataset, instance);

                Assert.Equal(1, recommender.Predict(user, item, this.featureProvider));
            }
        }

        /// <summary>
        /// Tests custom binary serialization and deserialization of a negative data generator mapping.
        /// </summary>
        [Fact]
        public void BinaryRecommendationCustomSerializationRegressionTest()
        {
            // TODO: We need a static class which implements mapping chainings
            Rand.Restart(12347);
            var positiveOnlyMapping = new PositiveOnlyDataMapping();
            var negativeDataGeneratorMapping = positiveOnlyMapping.WithGeneratedNegativeData(1.0);
            using (Stream stream = new MemoryStream())
            {
                negativeDataGeneratorMapping.SaveForwardCompatibleAsBinary(stream);

                stream.Seek(0, SeekOrigin.Begin);

                using (var reader = new WrappedBinaryReader(new BinaryReader(stream)))
                {
                    var deserializedMapping =
                        new NegativeDataGeneratorMapping<PositiveOnlyDataset, Tuple<string, string>, string, string, FeatureProvider, Vector>(reader, positiveOnlyMapping);
                    var recommender = MatchboxRecommender.Create(deserializedMapping);
                    recommender.Train(this.positiveOnlyDataset, this.featureProvider); // must not throw

                    foreach (var instance in positiveOnlyMapping.GetInstances(this.positiveOnlyDataset))
                    {
                        var user = positiveOnlyMapping.GetUser(this.positiveOnlyDataset, instance);
                        var item = positiveOnlyMapping.GetItem(this.positiveOnlyDataset, instance);

                        Assert.Equal(1, recommender.Predict(user, item, this.featureProvider));
                    }
                }
            }
        }

        #region Helper methods

        /// <summary>
        /// Initializes the positive only dataset.
        /// </summary>
        private void InitializePositiveOnlyDataset()
        {
            this.positiveOnlyDataset = new PositiveOnlyDataset
                {
                    Observations = new List<Tuple<string, string>>
                        {
                            Tuple.Create("u0", "i0"),
                            Tuple.Create("u0", "i1"),
                            Tuple.Create("u0", "i4"),
                            Tuple.Create("u1", "i3"),
                            Tuple.Create("u1", "i4"),
                            Tuple.Create("u2", "i2"),
                        }
                };
        }

        /// <summary>
        /// Initializes the feature provider for the positive only dataset.
        /// </summary>
        private void InitializeFeatureProvider()
        {
            this.featureProvider = new FeatureProvider
            {
                UserFeatures = new Dictionary<string, Vector>
                    {
                        { "u0", Vector.FromArray(0.1) },
                        { "u1", Vector.FromArray(0.0) },
                        { "u2", Vector.FromArray(0.5) }
                    },
                ItemFeatures = new Dictionary<string, Vector>
                    {
                        { "i0", Vector.FromArray(0.2, 0.3) },
                        { "i1", Vector.FromArray(0.4, 0.5) },
                        { "i2", Vector.FromArray(0.2, 0.9) },
                        { "i3", Vector.FromArray(1.1, 1.2) }
                    }
            };
        }

        /// <summary>
        /// Computes the number of original observations, positive 
        /// ratings, and negative ratings for each user and item
        /// </summary>
        /// <param name="userRatingCounts">A map from each user to their rating counts.</param>
        /// <param name="itemRatingCounts">A map from each item to its rating counts.</param>
        private void ComputeRatingCounts(
            out IDictionary<string, RatingCounts> userRatingCounts,
            out IDictionary<string, RatingCounts> itemRatingCounts)
        {
            Rand.Restart(12347);
            var positiveOnlyMapping = new PositiveOnlyDataMapping();
            var negativeDataGeneratorMapping = positiveOnlyMapping.WithGeneratedNegativeData(1.0);

            userRatingCounts = new Dictionary<string, RatingCounts>();
            itemRatingCounts = new Dictionary<string, RatingCounts>();

            // Count the number of positive observations for each user and item.
            foreach (var instance in positiveOnlyMapping.GetInstances(this.positiveOnlyDataset))
            {
                var user = positiveOnlyMapping.GetUser(this.positiveOnlyDataset, instance);
                var item = positiveOnlyMapping.GetItem(this.positiveOnlyDataset, instance);

                RatingCounts ratingCounts;
                if (!userRatingCounts.TryGetValue(user, out ratingCounts))
                {
                    ratingCounts = new RatingCounts();
                    userRatingCounts.Add(user, ratingCounts);
                }

                ++ratingCounts.Original;

                if (!itemRatingCounts.TryGetValue(item, out ratingCounts))
                {
                    ratingCounts = new RatingCounts();
                    itemRatingCounts.Add(item, ratingCounts);
                }

                ++ratingCounts.Original;
            }

            // Compute the number of positive and negative ratings generated for each user and item.
            foreach (var instance in negativeDataGeneratorMapping.GetInstances(this.positiveOnlyDataset))
            {
                var user = negativeDataGeneratorMapping.GetUser(this.positiveOnlyDataset, instance);
                var item = negativeDataGeneratorMapping.GetItem(this.positiveOnlyDataset, instance);
                var rating = negativeDataGeneratorMapping.GetRating(this.positiveOnlyDataset, instance);

                if (rating == 1)
                {
                    ++userRatingCounts[user].Positive;
                    ++itemRatingCounts[item].Positive;
                }
                else if (rating == 0)
                {
                    ++userRatingCounts[user].Negative;
                    ++itemRatingCounts[item].Negative;
                }
            }
        }

        /// <summary>
        /// Tests if the number of generated positive and negative ratings is
        /// equal to the number of the original positive observations for each entity,
        /// except when an entity likes (or is liked) by more than half of the other entities.
        /// </summary>
        /// <param name="entityRatingCounts">The rating counts for each entity.</param>
        /// <param name="otherEntityCount">The number of other entities.</param>
        private void TestRatingCounts(IEnumerable<RatingCounts> entityRatingCounts, int otherEntityCount)
        {
            foreach (var entityRating in entityRatingCounts)
            {
                Assert.Equal(Math.Min(entityRating.Original, otherEntityCount - entityRating.Original), entityRating.Negative);
                Assert.Equal(entityRating.Original, entityRating.Positive);
            }
        }

        /// <summary>
        /// Test the histogram sampler. 
        /// Makes sure that the samples from the underlying distribution correspond to the parameters histogram.
        /// </summary>
        /// <param name="histogram">The parameters.</param>
        private void TestHistogramSampler(int[] histogram)
        {
            var histogramSampler = new HistogramSampler(histogram);
            var samplesHistogram = new int[histogram.Length];

            int sampleCount = histogram.Sum();
            for (int i = 0; i < sampleCount; ++i)
            {
                int sample = histogramSampler.Sample();
                histogramSampler.Take(sample);
                ++samplesHistogram[sample];
            }

            Assert.True(histogramSampler.IsEmpty());

            for (int i = 0; i < histogram.Length; ++i)
            {
                Assert.Equal(histogram[i], samplesHistogram[i]);
            }
        }

        #endregion

        #region Helper classes

        /// <summary>
        /// Represents a positive only dataset.
        /// </summary>
        private class PositiveOnlyDataset
        {
            /// <summary>
            /// Gets or sets the list of observations.
            /// </summary>
            public List<Tuple<string, string>> Observations { get; set; }
        }

        /// <summary>
        /// Provides features for the positive only dataset.
        /// </summary>
        private class FeatureProvider
        {
            /// <summary>
            /// Gets or sets the mapping from user to features.
            /// </summary>
            public IDictionary<string, Vector> UserFeatures { get; set; }

            /// <summary>
            /// Gets or sets the mapping from item to features.
            /// </summary>
            public IDictionary<string, Vector> ItemFeatures { get; set; }
        }

        /// <summary>
        /// An implementation of
        /// <see cref="IRecommenderMapping{TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues}"/>
        /// for the positive only test data.
        /// </summary>
        [Serializable]
        private class PositiveOnlyDataMapping
            : IRecommenderMapping<PositiveOnlyDataset, Tuple<string, string>, string, string, FeatureProvider, Vector>
        {
            /// <summary>
            /// Retrieves a list of instances from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source to retrieve instances from.</param>
            /// <returns>The list of retrieved instances.</returns>
            public IEnumerable<Tuple<string, string>> GetInstances(PositiveOnlyDataset instanceSource)
            {
                return instanceSource.Observations;
            }

            /// <summary>
            /// Extracts a user from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract the user from.</param>
            /// <returns>The extracted user.</returns>
            public string GetUser(PositiveOnlyDataset instanceSource, Tuple<string, string> instance)
            {
                return instance.Item1;
            }

            /// <summary>
            /// Extracts an item from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract the item from.</param>
            /// <returns>The extracted item.</returns>
            public string GetItem(PositiveOnlyDataset instanceSource, Tuple<string, string> instance)
            {
                return instance.Item2;
            }

            /// <summary>
            /// Provides a vector of features for a given user.
            /// </summary>
            /// <param name="featureSource">The source of the feature vector.</param>
            /// <param name="user">The user to provide the feature vector for.</param>
            /// <returns>The feature vector for <paramref name="user"/>.</returns>
            public Vector GetUserFeatures(FeatureProvider featureSource, string user)
            {
                return featureSource.UserFeatures[user];
            }

            /// <summary>
            /// Provides a vector of features for a given item.
            /// </summary>
            /// <param name="featureSource">The source of the feature vector.</param>
            /// <param name="item">The item to provide the feature vector for.</param>
            /// <returns>The feature vector for <paramref name="item"/>.</returns>
            public Vector GetItemFeatures(FeatureProvider featureSource, string item)
            {
                return featureSource.ItemFeatures[item];
            }
        }

        /// <summary>
        /// Holds the original observations count and the number of positive and negative ratings.
        /// </summary>
        private class RatingCounts
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="RatingCounts"/> class.
            /// </summary>
            public RatingCounts()
            {
                this.Original = 0;
                this.Positive = 0;
                this.Negative = 0;
            }

            /// <summary>
            /// Gets or sets the original observations count.
            /// </summary>
            public int Original { get; set; }

            /// <summary>
            /// Gets or sets the positive rating count.
            /// </summary>
            public int Positive { get; set; }

            /// <summary>
            /// Gets or sets the negative rating count.
            /// </summary>
            public int Negative { get; set; }
        }

        #endregion
    }
}
