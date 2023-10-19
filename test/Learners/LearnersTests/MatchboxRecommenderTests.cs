// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Text;

    using Xunit;
    using Assert = AssertHelper;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Serialization;

    using RatingDistribution = System.Collections.Generic.IDictionary<int, double>;

    /// <summary>
    /// Tests for Matchbox recommender.
    /// </summary>
    //[DeploymentItem(@"CustomSerializedLearners\", "CustomSerializedLearners")]
    public class MatchboxRecommenderTests
    {
        /// <summary>
        /// Tolerance for comparisons.
        /// </summary>
        private const double Tolerance = 1e-2;

        /// <summary>
        /// A low tolerance for comparisons.
        /// </summary>
        private const double LowTolerance = 1e-8;

        /// <summary>
        /// The number of inference iterations. Should be large enough for the inference to converge.
        /// </summary>
        private const int InferenceIterationCount = 1000;

        /// <summary>
        /// The minimum star rating for data in standard format.
        /// </summary>
        private const int MinStarRating = -2;

        /// <summary>
        /// The maximum star rating for data in standard format.
        /// </summary>
        private const int MaxStarRating = 3;

         /// <summary>
        /// The training set in the standard format.
        /// </summary>
        private StandardDataset standardTrainingData;

        /// <summary>
        /// The feature provider for the training set in the standard format.
        /// </summary>
        private FeatureProvider standardTrainingDataFeatures;
        
        /// <summary>
        /// The training set in the native format.
        /// </summary>
        private NativeDataset nativeTrainingData;

        /// <summary>
        /// The mapping to native Matchbox data format.
        /// </summary>
        private NativeRecommenderTestMapping nativeMapping;

        /// <summary>
        /// The mapping to standard recommender data format.
        /// </summary>
        private StandardRecommenderTestMapping standardMapping;

        /// <summary>
        /// Prepares environment (datasets etc) before each test.
        /// </summary>
        public MatchboxRecommenderTests()
        {
            this.InitializeStandardTrainingData();
            this.InitializeNativeTrainingData();
        }

        /// <summary>
        /// Regression test for recommendation with both user and item features.
        /// </summary>
        [Fact]
        public void UserItemFeaturesRegressionTest()
        {
            // Standard data format
            var standardRecommender = this.CreateStandardDataFormatMatchboxRecommender();
            standardRecommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
            RatingDistribution standardRatingDistribution = standardRecommender.PredictDistribution(User.WithId("u1"), Item.WithId("i3"), this.standardTrainingDataFeatures);
            VerifyStandardRatingDistributionOfUserOneAndItemThree(standardRatingDistribution);
            int standardRating = standardRecommender.Predict(User.WithId("u1"), Item.WithId("i3"), this.standardTrainingDataFeatures);
            Assert.Equal(MaxStarRating, standardRating);

            // Native data format
            var nativeRecommender = this.CreateNativeDataFormatMatchboxRecommender();
            nativeRecommender.Train(this.nativeTrainingData, this.nativeTrainingData);
            Discrete nativeRatingDistribution = nativeRecommender.PredictDistribution(1, 3, this.nativeTrainingData);
            this.VerifyNativeRatingDistributionOfUserOneAndItemThree(nativeRatingDistribution);
            int nativeRating = nativeRecommender.Predict(1, 3, this.nativeTrainingData);
            Assert.Equal(MaxStarRating - MinStarRating, nativeRating);
        }

        /// <summary>
        /// Regression test for recommendation only with user features.
        /// </summary>
        [Fact]
        public void UserFeaturesRegressionTest()
        {
            const double ExpectedRatingProbability = 0.313285218997476;

            // Standard data format
            var standardRecommender = this.CreateStandardDataFormatMatchboxRecommender();
            standardRecommender.Settings.Training.UseItemFeatures = false;
            standardRecommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
            double standardRating = standardRecommender.PredictDistribution(User.WithId("u1"), Item.WithId("i3"), this.standardTrainingDataFeatures)[MinStarRating];
            Assert.Equal(ExpectedRatingProbability, standardRating, Tolerance);

            // Native data format
            var nativeRecommender = this.CreateNativeDataFormatMatchboxRecommender();
            nativeRecommender.Settings.Training.UseItemFeatures = false;
            nativeRecommender.Train(this.nativeTrainingData, this.nativeTrainingData);
            double nativeRating = nativeRecommender.PredictDistribution(1, 3, this.nativeTrainingData)[0];
            Assert.Equal(ExpectedRatingProbability, nativeRating, Tolerance);
        }

        /// <summary>
        /// Regression test for recommendation only with item features.
        /// </summary>
        [Fact]
        public void ItemFeaturesRegressionTest()
        {
            const double ExpectedRatingProbability = 0.2019280142161351;

            // Standard data format
            var standardRecommender = this.CreateStandardDataFormatMatchboxRecommender();
            standardRecommender.Settings.Training.UseUserFeatures = false;
            standardRecommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
            double standardRating = standardRecommender.PredictDistribution(User.WithId("u1"), Item.WithId("i3"), this.standardTrainingDataFeatures)[MinStarRating];
            Assert.Equal(ExpectedRatingProbability, standardRating, Tolerance);

            // Native data format
            var nativeRecommender = this.CreateNativeDataFormatMatchboxRecommender();
            nativeRecommender.Settings.Training.UseUserFeatures = false;
            nativeRecommender.Train(this.nativeTrainingData, this.nativeTrainingData);
            double nativeRating = nativeRecommender.PredictDistribution(1, 3, this.nativeTrainingData)[0];
            Assert.Equal(ExpectedRatingProbability, nativeRating, Tolerance);
        }

        /// <summary>
        /// Regression test for recommendation without features.
        /// </summary>
        [Fact]
        public void NoFeaturesRegressionTest()
        {
            const double ExpectedRatingProbability = 0.1751;

            // Standard data format
            var standardRecommender = this.CreateStandardDataFormatMatchboxRecommender();
            standardRecommender.Settings.Training.UseUserFeatures = false;
            standardRecommender.Settings.Training.UseItemFeatures = false;
            standardRecommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
            double standardRating = standardRecommender.PredictDistribution(User.WithId("u1"), Item.WithId("i3"), this.standardTrainingDataFeatures)[MinStarRating];
            Assert.Equal(ExpectedRatingProbability, standardRating, Tolerance);

            // Native data format
            var nativeRecommender = this.CreateNativeDataFormatMatchboxRecommender();
            nativeRecommender.Settings.Training.UseUserFeatures = false;
            nativeRecommender.Settings.Training.UseItemFeatures = false;
            nativeRecommender.Train(this.nativeTrainingData, this.nativeTrainingData);
            double nativeRating = nativeRecommender.PredictDistribution(1, 3, this.nativeTrainingData)[0];
            Assert.Equal(ExpectedRatingProbability, nativeRating, Tolerance);
        }

        /// <summary>
        /// Regression test for the related users/items prediction using data in the standard format.
        /// </summary>
        [Fact]
        public void RelatedEntitiesStandardDataFormatNoFeaturesRegressionTest()
        {
            var recommender = this.CreateStandardDataFormatMatchboxRecommender();
            recommender.Settings.Training.UseUserFeatures = false;
            recommender.Settings.Training.UseItemFeatures = false;

            recommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);

            // Bulk related user prediction
            var relatedUsersDict = recommender.GetRelatedUsers(new[] { User.WithId("u0"), User.WithId("u1") }, 1);
            Assert.Equal(2, relatedUsersDict.Count);
            Assert.Equal(User.WithId("u1"), relatedUsersDict[User.WithId("u0")].First());
            Assert.Equal(User.WithId("u0"), relatedUsersDict[User.WithId("u1")].First());

            // Bulk related item prediction
            var relatedItemsDict = recommender.GetRelatedItems(new[] { Item.WithId("i0"), Item.WithId("i1") }, 3);
            Assert.Equal(2, relatedItemsDict.Count);
            var predictedRelatedItems = relatedItemsDict[Item.WithId("i0")].ToArray();
            Assert.Equal(Item.WithId("i2"), predictedRelatedItems[2]);

            // Per-instance related user prediction
            var relatedUsersList = recommender.GetRelatedUsers(User.WithId("u1"), 5);
            Assert.Single(relatedUsersList);
            Assert.Equal(User.WithId("u0"), relatedUsersList.First());

            // Same with user subset
            recommender.UserSubset = new[] { User.WithId("u0") };
            relatedUsersList = recommender.GetRelatedUsers(User.WithId("u1"), 5);
            Assert.Single(relatedUsersList);
            Assert.Equal(User.WithId("u0"), relatedUsersList.First());

            // Per-instance related item prediction
            var relatedItemsList = recommender.GetRelatedItems(Item.WithId("i0"), 1);
            var expected = new HashSet<Item> { Item.WithId("i1"), Item.WithId("i2"), Item.WithId("i3") };
            Assert.Single(relatedItemsList);
            Assert.Contains(relatedItemsList.First(), expected);

            // Same with item subset
            recommender.ItemSubset = new[] { Item.WithId("i1"), Item.WithId("i2"), Item.WithId("i3") };
            relatedItemsList = recommender.GetRelatedItems(Item.WithId("i0"), 1);
            Assert.Single(relatedItemsList);
            Assert.Contains(relatedItemsList.First(), expected);
        }

        /// <summary>
        /// Regression test for the related users/items prediction using data in the native Matchbox format.
        /// </summary>
        [Fact]
        public void RelatedEntitiesNativeDataFormatNoFeaturesRegressionTest()
        {
            var recommender = this.CreateNativeDataFormatMatchboxRecommender();
            recommender.Settings.Training.UseUserFeatures = false;
            recommender.Settings.Training.UseItemFeatures = false;

            recommender.Train(this.nativeTrainingData, this.nativeTrainingData);

            // Bulk related user prediction
            var relatedUsersDict = recommender.GetRelatedUsers(new[] { 0, 1 }, 1);
            Assert.Equal(2, relatedUsersDict.Count);
            Assert.Equal(1, relatedUsersDict[0].First());
            Assert.Equal(0, relatedUsersDict[1].First());

            // Bulk related item prediction
            var relatedItemsDict = recommender.GetRelatedItems(new[] { 1, 0 }, 3);
            Assert.Equal(2, relatedItemsDict.Count);
            var predictedRelatedItems = relatedItemsDict[1].ToArray();
            Assert.Equal(3, predictedRelatedItems[2]);

            // Per-instance related user prediction
            var relatedUsersList = recommender.GetRelatedUsers(0, 5);
            Assert.Single(relatedUsersList);
            Assert.Equal(1, relatedUsersList.First());

            // Same with user subset
            recommender.UserSubset = new[] { 1 };
            relatedUsersList = recommender.GetRelatedUsers(0, 5);
            Assert.Single(relatedUsersList);
            Assert.Equal(1, relatedUsersList.First());

            // Per-instance related item prediction
            var relatedItemsList = recommender.GetRelatedItems(1, 1);
            var expected = new HashSet<int> { 0, 2, 3 };
            Assert.Single(relatedItemsList);
            Assert.Contains(relatedItemsList.First(), expected);

            // Same with item subset
            recommender.ItemSubset = new[] { 0, 2, 3 };
            relatedItemsList = recommender.GetRelatedItems(1, 1);
            Assert.Single(relatedItemsList);
            Assert.Contains(relatedItemsList.First(), expected);
        }

        /// <summary>
        /// Tests if the shared user thresholds have the same values across users 
        /// and the user-specific ones have different values.
        /// </summary>
        [Fact]
        public void SharedUserThresholdsEqualityTest()
        {
            // User-specific thresholds
            var recommenderUserSpecificThresholds = this.CreateStandardDataFormatMatchboxRecommender();
            //// The UseSharedUserThresholds must be set to false by default to correspond to the model of the Matchbox paper
            recommenderUserSpecificThresholds.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
            var userSpecificThresholdsPosterior = recommenderUserSpecificThresholds.GetPosteriorDistributions().Users.Select(u => u.Value.Thresholds).ToList();
            AssertUserThresholdEquality(userSpecificThresholdsPosterior, Assert.NotEqual);

            // Shared user thresholds
            var recommenderSharedThresholds = this.CreateStandardDataFormatMatchboxRecommender();
            recommenderSharedThresholds.Settings.Training.UseSharedUserThresholds = true;
            recommenderSharedThresholds.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
            var sharedThresholdsPosterior = recommenderSharedThresholds.GetPosteriorDistributions().Users.Select(u => u.Value.Thresholds).ToList();
            AssertUserThresholdEquality(sharedThresholdsPosterior, Assert.Equal);
        }

        /// <summary>
        /// Tests if modifying one user threshold does not affect the thresholds of other users when shared thresholds are used.
        /// </summary>
        [Fact]
        public void SharedUserThresholdSeparationTest()
        {
            var nativeRecommender = this.CreateNativeDataFormatMatchboxRecommender();
            nativeRecommender.Settings.Training.UseSharedUserThresholds = true;
            nativeRecommender.Train(this.nativeTrainingData, this.nativeTrainingData);

            const int TestThreshold = 1;
            const int TestUser1 = 0;
            const int TestUser2 = 1;
            var userPosteriors = nativeRecommender.GetPosteriorDistributions().Users;
            var expectedThresholdPosterior = new Gaussian(userPosteriors[TestUser2].Thresholds[TestThreshold]);
            userPosteriors[TestUser1].Thresholds[TestThreshold] = Gaussian.PointMass(Double.NaN);
            var actualThresholdPosterior = userPosteriors[TestUser2].Thresholds[TestThreshold];
            Assert.Equal(expectedThresholdPosterior, actualThresholdPosterior);
        }

        /// <summary>
        /// Tests if it is forbidden for the same user to be presented in user subset multiple times (same for items).
        /// </summary>
        [Fact]
        public void MultipleEntriesInEntitySubsetTest()
        {
            var standardRecommender = this.CreateStandardDataFormatMatchboxRecommender();
            standardRecommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);

            Assert.Throws<ArgumentException>(() => { standardRecommender.UserSubset = new[] { User.WithId("u0"), User.WithId("u0") }; });
            Assert.Throws<ArgumentException>(() => { standardRecommender.ItemSubset = new[] { Item.WithId("i1"), Item.WithId("i0"), Item.WithId("i1") }; });

            var nativeRecommender = this.CreateNativeDataFormatMatchboxRecommender();
            nativeRecommender.Train(this.nativeTrainingData, this.nativeTrainingData);

            Assert.Throws<ArgumentException>(() => { nativeRecommender.UserSubset = new[] { 0, 1, 0 }; });
            Assert.Throws<ArgumentException>(() => { nativeRecommender.ItemSubset = new[] { 1, 1 }; });
        }

        /// <summary>
        /// Regression test for cold start rating prediction and item recommendation using the standard data format.
        /// </summary>
        [Fact]
        public void ColdStartStandardDataFormatRegressionTest()
        {
            var recommender = this.CreateStandardDataFormatMatchboxRecommender();

            recommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);

            // Test new users
            this.standardTrainingDataFeatures.UserFeatures.Add(User.WithId("u2"), Vector.FromArray(4, 2, 1.3, 1.2, 2));
            this.standardTrainingDataFeatures.UserFeatures.Add(User.WithId("u3"), Vector.FromArray(5, 1, 2.3, 3.2, 1));

            // Add a cold user to the user subset
            recommender.UserSubset = Util.ArrayInit(
                this.standardTrainingDataFeatures.UserFeatures.Count, u => User.WithId("u" + u.ToString(CultureInfo.InvariantCulture)));

            // Recommend items to a cold user
            User user = User.WithId("u2");
            int recommendationCount = 3;
            Item expectedItem = Item.WithId("i3");
            var recommendedItems = recommender.Recommend(user, recommendationCount, this.standardTrainingDataFeatures);
            Assert.Equal(expectedItem, recommendedItems.First());

            // Recommend items to a cold user, including the items' rating distributions
            var recommendation = recommender.RecommendDistribution(user, 3, this.standardTrainingDataFeatures).First();
            var expectedRatingDistribution = recommender.PredictDistribution(user, expectedItem, this.standardTrainingDataFeatures);

            Assert.Equal(expectedItem, recommendation.Item1);
            foreach (var expectedRating in expectedRatingDistribution)
            {
                Assert.Equal(expectedRating.Value, recommendation.Item2[expectedRating.Key], LowTolerance);
            }

            // Find related users to a cold user
            var relatedUsers = recommender.GetRelatedUsers(user, 3, this.standardTrainingDataFeatures);
            Assert.Equal(User.WithId("u3"), relatedUsers.First());

            // Test new items
            this.standardTrainingDataFeatures.ItemFeatures.Add(Item.WithId("i4"), Vector.FromArray(6.3, 0.5));
            this.standardTrainingDataFeatures.ItemFeatures.Add(Item.WithId("i5"), Vector.FromArray(6.1, 0.8));

            // Add a cold item to the item subset
            recommender.ItemSubset = Util.ArrayInit(
                this.standardTrainingDataFeatures.ItemFeatures.Count, i => Item.WithId("i" + i.ToString(CultureInfo.InvariantCulture)));

            // Make predictions for a cold item
            var prediction = recommender.Predict(User.WithId("u1"), Item.WithId("i4"), this.standardTrainingDataFeatures);
            Assert.Equal(MaxStarRating, prediction);

            // Find related items to a cold item
            var relatedItems = recommender.GetRelatedItems(Item.WithId("i4"), 3, this.standardTrainingDataFeatures);
            Assert.Equal(Item.WithId("i5"), relatedItems.First());
        }

        /// <summary>
        /// Regression test for cold start rating prediction and item recommendation using the native Matchbox data format.
        /// </summary>
        [Fact]
        public void ColdStartNativeDataFormatRegressionTest()
        {
            var recommender = this.CreateNativeDataFormatMatchboxRecommender();

            recommender.Train(this.nativeTrainingData, this.nativeTrainingData);

            // Test new users
            this.nativeTrainingData.NonZeroUserFeatureValues = new[]
            {
                this.nativeTrainingData.NonZeroUserFeatureValues[0],
                this.nativeTrainingData.NonZeroUserFeatureValues[1],
                new[] { 4, 2, 1.3, 1.2, 2 },
                new[] { 5, 1, 2.3, 3.2, 1 }
            };
            this.nativeTrainingData.NonZeroUserFeatureIndices = new[]
            {
                this.nativeTrainingData.NonZeroUserFeatureIndices[0],
                this.nativeTrainingData.NonZeroUserFeatureIndices[1],
                new[] { 0, 1, 2, 3, 4 },
                new[] { 0, 1, 2, 3, 4 }
            };

            // Add a cold user to the user subset
            recommender.UserSubset = Util.ArrayInit(this.nativeTrainingData.NonZeroUserFeatureValues.Length, u => u);

            // Recommend items to a cold user
            int user = 2;
            int recommendationCount = 3;
            int item = 3;
            var recommendedItems = recommender.Recommend(user, recommendationCount, this.nativeTrainingData);
            Assert.Equal(item, recommendedItems.First());

            // Recommend items to a cold user, including the items' rating distributions
            var recommendation = recommender.RecommendDistribution(user, recommendationCount, this.nativeTrainingData).First();
            var expectedRatingDistribution = recommender.PredictDistribution(user, item, this.nativeTrainingData);

            Assert.Equal(item, recommendation.Item1);
            for (int ratingIndex = 0; ratingIndex < expectedRatingDistribution.Dimension; ratingIndex++)
            {
                Assert.Equal(expectedRatingDistribution[ratingIndex], recommendation.Item2[ratingIndex], LowTolerance);
            }

            // Find related users to a cold user
            int relatedUserCount = 3;
            var relatedUsers = recommender.GetRelatedUsers(user, relatedUserCount, this.nativeTrainingData);
            Assert.Equal(3, relatedUsers.First());

            // Test new items
            this.nativeTrainingData.NonZeroItemFeatureValues = new[]
                {
                    this.nativeTrainingData.NonZeroItemFeatureValues[0],
                    this.nativeTrainingData.NonZeroItemFeatureValues[1],
                    this.nativeTrainingData.NonZeroItemFeatureValues[2],
                    this.nativeTrainingData.NonZeroItemFeatureValues[3],
                    new[] { 6.3, 0.5 },
                    new[] { 6.1, 0.8 }
                };
            this.nativeTrainingData.NonZeroItemFeatureIndices = new[]
                {
                    this.nativeTrainingData.NonZeroItemFeatureIndices[0],
                    this.nativeTrainingData.NonZeroItemFeatureIndices[1],
                    this.nativeTrainingData.NonZeroItemFeatureIndices[2],
                    this.nativeTrainingData.NonZeroItemFeatureIndices[3],
                    new[] { 0, 1 },
                    new[] { 0, 1 }
                };

            // Add a cold item to the item subset
            recommender.ItemSubset = Util.ArrayInit(this.nativeTrainingData.NonZeroItemFeatureValues.Length, i => i);

            // Make predictions for a cold item
            var prediction = recommender.Predict(0, 4, this.nativeTrainingData);
            Assert.Equal(5, prediction);

            // Find related items to a cold item
            var relatedItems = recommender.GetRelatedItems(4, 3, this.nativeTrainingData);
            Assert.Equal(5, relatedItems.First());
        }

        /// <summary>
        /// Tests if the recommender does not ignore features in training which contain only values of zero.
        /// This needs to be supported in case of a feature that is only zero in training is not zero in prediction.
        /// </summary>
        [Fact]
        public void ZeroOnlyTrainingFeatureTest()
        {
            var recommender = this.CreateStandardDataFormatMatchboxRecommender();
            this.standardTrainingDataFeatures.UserFeatures[User.WithId("u1")] = Vector.FromArray(1, 1, 1, 1, 1, 0);
            recommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
            this.standardTrainingDataFeatures.UserFeatures.Add(User.WithId("u101"), Vector.FromArray(0, 0, 0, 0, 0, 1));
            recommender.Predict(User.WithId("u101"), Item.WithId("i3"), this.standardTrainingDataFeatures); // must not throw
        }

        /// <summary>
        /// Tests if the recommender data consistency checks can correctly detect problems with the data provided by the mapping.
        /// </summary>
        [Fact]
        public void InconsistentDataCheckTest()
        {
            var data = new NativeDataset { UserIds = new[] { 0 }, ItemIds = new[] { 0 }, Ratings = new[] { 1 }, UserCount = 1, ItemCount = 1 };
            this.TestDataConsistency(data, false, false); // Must not throw

            data = new NativeDataset { UserIds = null, ItemIds = new[] { 0 }, Ratings = new[] { 1 }, UserCount = 1, ItemCount = 1 };
            this.TestDataConsistency(data, false, true); // Null user ids

            data = new NativeDataset { UserIds = new[] { 0 }, ItemIds = null, Ratings = new[] { 1 }, UserCount = 1, ItemCount = 1 };
            this.TestDataConsistency(data, false, true); // Null item ids

            data = new NativeDataset { UserIds = new[] { 0 }, ItemIds = new[] { 0 }, Ratings = null, UserCount = 1, ItemCount = 1 };
            this.TestDataConsistency(data, false, true); // Null data representations

            data = new NativeDataset { UserIds = new[] { 0, 1 }, ItemIds = new[] { 0 }, Ratings = new[] { 1 }, UserCount = 2, ItemCount = 1 };
            this.TestDataConsistency(data, false, true); // Inconsistent array length 1

            data = new NativeDataset { UserIds = new[] { 0 }, ItemIds = new[] { 0 }, Ratings = new[] { 1, 0 }, UserCount = 1, ItemCount = 1 };
            this.TestDataConsistency(data, false, true); // Inconsistent array length 2
            
            data = new NativeDataset { UserIds = new[] { 2 }, ItemIds = new[] { 0 }, Ratings = new[] { 1 }, UserCount = 1, ItemCount = 1 };
            this.TestDataConsistency(data, false, true); // Incorrect user id 1

            data = new NativeDataset { UserIds = new[] { -1 }, ItemIds = new[] { 0 }, Ratings = new[] { 1 }, UserCount = 1, ItemCount = 1 };
            this.TestDataConsistency(data, false, true); // Incorrect user id 2

            data = new NativeDataset { UserIds = new[] { 0 }, ItemIds = new[] { 2 }, Ratings = new[] { 1 }, UserCount = 1, ItemCount = 1 };
            this.TestDataConsistency(data, false, true); // Incorrect item id 1

            data = new NativeDataset { UserIds = new[] { 0 }, ItemIds = new[] { -1 }, Ratings = new[] { 1 }, UserCount = 1, ItemCount = 1 };
            this.TestDataConsistency(data, false, true); // Incorrect item id 2

            data = new NativeDataset
            {
                UserIds = new[] { 0, 1 },
                ItemIds = new[] { 1, 0 },
                Ratings = new[] { 1, 0 },
                UserCount = 2,
                ItemCount = 2,
                NonZeroUserFeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } },
                NonZeroUserFeatureIndices = new[] { new[] { 0 }, new[] { 0 } },
                NonZeroItemFeatureValues = new[] { new[] { 0.0 }, new[] { 1.0 } },
                NonZeroItemFeatureIndices = new[] { new[] { 0 }, new[] { 0 } },
            };

            this.TestDataConsistency(data, true, false); // Must not throw

            data.NonZeroUserFeatureValues = new[] { new[] { 1.0 } };
            data.NonZeroUserFeatureIndices = new[] { new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Inconsistent length of user feature arrays 1

            data.NonZeroUserFeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } };
            data.NonZeroUserFeatureIndices = new[] { new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Inconsistent length of user feature arrays 2

            data.NonZeroUserFeatureValues = new[] { new[] { 1.0 } };
            data.NonZeroUserFeatureIndices = new[] { new[] { 0 }, new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Inconsistent length of user feature arrays 3

            data.NonZeroUserFeatureValues = new[] { null, new[] { 1.0 } };
            data.NonZeroUserFeatureIndices = new[] { new[] { 0 }, new[] { 0 } };
            this.TestDataConsistency(data, true, true); // One of the user feature value arrays is null

            data.NonZeroUserFeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } };
            data.NonZeroUserFeatureIndices = new[] { null, new[] { 0 } };
            this.TestDataConsistency(data, true, true); // One of the user feature index arrays is null

            data.NonZeroUserFeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } };
            data.NonZeroUserFeatureIndices = new[] { new[] { 0, 1 }, new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Inconsistent length between user feature value and user feature index arrays

            data.NonZeroUserFeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } };
            data.NonZeroUserFeatureIndices = new[] { new[] { -1 }, new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Negative user feature indices

            data.NonZeroUserFeatureValues = new[] { new[] { 1.0, 2.0 }, new[] { 1.0 } };
            data.NonZeroUserFeatureIndices = new[] { new[] { 1, 1 }, new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Non-unique user feature indices

            data.NonZeroUserFeatureValues = new[] { new[] { 1.0 } };
            data.NonZeroUserFeatureIndices = new[] { new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Inconsistent length of user feature arrays 1

            data.NonZeroUserFeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } };
            data.NonZeroUserFeatureIndices = new[] { new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Inconsistent length of user feature arrays 2

            data.NonZeroUserFeatureValues = new[] { new[] { 1.0 } };
            data.NonZeroUserFeatureIndices = new[] { new[] { 0 }, new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Inconsistent length of user feature arrays 3

            data.NonZeroUserFeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } };
            data.NonZeroUserFeatureIndices = new[] { new[] { 0 }, new[] { 0 } };
            this.TestDataConsistency(data, true, false); // Valid user features restored, must not throw

            data.NonZeroUserFeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } };
            data.NonZeroUserFeatureIndices = new[] { new[] { 0 }, new[] { 0 } };
            this.TestDataConsistency(data, true, false); // Valid user features restored, must not throw

            data.NonZeroItemFeatureValues = new[] { new[] { 1.0 } };
            data.NonZeroItemFeatureIndices = new[] { new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Inconsistent length of item feature arrays 1

            data.NonZeroItemFeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } };
            data.NonZeroItemFeatureIndices = new[] { new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Inconsistent length of item feature arrays 2

            data.NonZeroItemFeatureValues = new[] { new[] { 1.0 } };
            data.NonZeroItemFeatureIndices = new[] { new[] { 0 }, new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Inconsistent length of item feature arrays 3

            data.NonZeroItemFeatureValues = new[] { null, new[] { 1.0 } };
            data.NonZeroItemFeatureIndices = new[] { new[] { 0 }, new[] { 0 } };
            this.TestDataConsistency(data, true, true); // One of the item feature value arrays is null

            data.NonZeroItemFeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } };
            data.NonZeroItemFeatureIndices = new[] { null, new[] { 0 } };
            this.TestDataConsistency(data, true, true); // One of the item feature index arrays is null

            data.NonZeroItemFeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } };
            data.NonZeroItemFeatureIndices = new[] { new[] { 0, 1 }, new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Inconsistent length between item feature value and user feature index arrays

            data.NonZeroItemFeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } };
            data.NonZeroItemFeatureIndices = new[] { new[] { -1 }, new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Negative item feature indices

            data.NonZeroItemFeatureValues = new[] { new[] { 1.0, 2.0 }, new[] { 1.0 } };
            data.NonZeroItemFeatureIndices = new[] { new[] { 1, 1 }, new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Non-unique item feature indices

            data.NonZeroItemFeatureValues = new[] { new[] { 1.0 } };
            data.NonZeroItemFeatureIndices = new[] { new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Inconsistent length of item feature arrays 1

            data.NonZeroItemFeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } };
            data.NonZeroItemFeatureIndices = new[] { new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Inconsistent length of item feature arrays 2

            data.NonZeroItemFeatureValues = new[] { new[] { 1.0 } };
            data.NonZeroItemFeatureIndices = new[] { new[] { 0 }, new[] { 0 } };
            this.TestDataConsistency(data, true, true); // Inconsistent length of item feature arrays 3

            data.NonZeroItemFeatureValues = new[] { new[] { 2.0 }, new[] { 1.0 } };
            data.NonZeroItemFeatureIndices = new[] { new[] { 0 }, new[] { 0 } };
            this.TestDataConsistency(data, true, false); // Valid item features restored, must not throw
        }

        /// <summary>
        /// Tests serialization/deserialization of the recommender operating on the native Matchbox data format.
        /// </summary>
        [Fact]
        public void NativeDataFormatSerializationTest()
        {
            const string TrainedFileName = "trainedNativeRecommender.bin";
            const string NotTrainedFileName = "notTrainedNativeRecommender.bin";

            // Train and serialize
            {
                var recommender = this.CreateNativeDataFormatMatchboxRecommender();
                recommender.SaveForwardCompatible(NotTrainedFileName);
                recommender.Train(this.nativeTrainingData, this.nativeTrainingData);
                recommender.SaveForwardCompatible(TrainedFileName);
            }

            // Deserialize and test
            {
                var trainedRecommender = MatchboxRecommender.LoadBackwardCompatible<NativeDataset, NativeDataset>(TrainedFileName, this.nativeMapping);
                var notTrainedRecommender = MatchboxRecommender.LoadBackwardCompatible<NativeDataset, NativeDataset>(NotTrainedFileName, this.nativeMapping);

                notTrainedRecommender.Train(this.nativeTrainingData, this.nativeTrainingData);

                this.VerifyNativeRatingDistributionOfUserOneAndItemThree(trainedRecommender.PredictDistribution(1, 3, this.nativeTrainingData));
                Assert.Equal(MaxStarRating - MinStarRating, trainedRecommender.Predict(1, 3, this.nativeTrainingData));

                this.VerifyNativeRatingDistributionOfUserOneAndItemThree(notTrainedRecommender.PredictDistribution(1, 3, this.nativeTrainingData));
                Assert.Equal(MaxStarRating - MinStarRating, notTrainedRecommender.Predict(1, 3, this.nativeTrainingData));
            }
        }

        /// <summary>
        /// Tests serialization/deserialization of the recommender operating on the standard data format.
        /// </summary>
        [Fact]
        public void StandardDataFormatSerializationTest()
        {
            const int BatchCount = 2;
            const string TrainedFileName = "trainedStandardRecommender.bin";
            const string NotTrainedFileName = "notTrainedStandardRecommender.bin";

            // Add features for cold test set user and item
            this.standardTrainingDataFeatures.UserFeatures.Add(User.WithId("u2"), Vector.FromArray(4, 2, 1.3, 1.2, 2));
            this.standardTrainingDataFeatures.ItemFeatures.Add(Item.WithId("i4"), Vector.FromArray(6.3, 0.5));

            string SerializeUserOrItem(object obj)
            {
                if (obj is User user)
                {
                    return user.Id;
                }

                if (obj is Item item)
                {
                    return item.Id;
                }

                throw new InvalidOperationException($"Cannot serialize type: {obj?.GetType()}");
            }

            object DeserializeUserOrItem(string str, Type type)
            {
                if (type == typeof(User))
                {
                    return new User { Id = str };
                }

                if (type == typeof(Item))
                {
                    return new Item { Id = str };
                }

                throw new InvalidOperationException($"Cannot deserialize type: {type}");
            }

            // Train and serialize
            {
                var recommender = this.CreateStandardDataFormatMatchboxRecommender();
                recommender.Settings.Training.BatchCount = BatchCount;

                recommender.Save<StandardDataset, User, Item, RatingDistribution, FeatureProvider>(NotTrainedFileName, SerializeUserOrItem);
                recommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
                recommender.Save<StandardDataset, User, Item, RatingDistribution, FeatureProvider>(TrainedFileName, SerializeUserOrItem);

                CheckStandardRatingPrediction(recommender, this.standardTrainingDataFeatures);
            }

            // Deserialize and test
            {
                var trainedRecommender = MatchboxRecommender.Load<StandardDataset, Tuple<User, Item, int?>, User, Item, int, FeatureProvider>(TrainedFileName, DeserializeUserOrItem, this.standardMapping);
                var notTrainedRecommender = MatchboxRecommender.Load<StandardDataset, Tuple<User, Item, int?>, User, Item, int, FeatureProvider>(NotTrainedFileName, DeserializeUserOrItem, this.standardMapping);

                notTrainedRecommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);

                CheckStandardRatingPrediction(trainedRecommender, this.standardTrainingDataFeatures);
                CheckStandardRatingPrediction(notTrainedRecommender, this.standardTrainingDataFeatures);
            }
        }

        /// <summary>
        /// Tests binary custom serialization/deserialization of the recommender operating on the native data format.
        /// </summary>
        [Fact]
        public void NativeDataFormatCustomSerializationTest()
        {
            const string TrainedFileName = "trainedCustomNativeRecommender.bin";
            const string NotTrainedFileName = "notTrainedCustomNativeRecommender.bin";

            // Train and serialize
            {
                var recommender = this.CreateNativeDataFormatMatchboxRecommender();

                this.nativeMapping.SaveForwardCompatible(NotTrainedFileName);
                recommender.SaveForwardCompatible(NotTrainedFileName, FileMode.Append);
                
                recommender.Train(this.nativeTrainingData, this.nativeTrainingData);
                recommender.SaveForwardCompatible(TrainedFileName);
            }

            // Deserialize and test
            {
                // Check wrong versions throw a serialization exception
                CheckCustomSerializationVersionException(reader => MatchboxRecommender.LoadBackwardCompatible(reader, this.nativeMapping));

                var trainedRecommender = MatchboxRecommender.LoadBackwardCompatible(TrainedFileName, this.nativeMapping);

                IMatchboxRecommender<NativeDataset, int, int, Discrete, NativeDataset> notTrainedRecommender;
                using (var stream = File.Open(NotTrainedFileName, FileMode.Open))
                {
                    using (var reader = new WrappedBinaryReader(new BinaryReader(stream)))
                    {
                        var deserializedMapping = new NativeRecommenderTestMapping(reader);
                        notTrainedRecommender = MatchboxRecommender.LoadBackwardCompatible(reader, deserializedMapping);
                    }
                }

                notTrainedRecommender.Train(this.nativeTrainingData, this.nativeTrainingData);

                this.VerifyNativeRatingDistributionOfUserOneAndItemThree(trainedRecommender.PredictDistribution(1, 3, this.nativeTrainingData));
                Assert.Equal(MaxStarRating - MinStarRating, trainedRecommender.Predict(1, 3, this.nativeTrainingData));

                this.VerifyNativeRatingDistributionOfUserOneAndItemThree(notTrainedRecommender.PredictDistribution(1, 3, this.nativeTrainingData));
                Assert.Equal(MaxStarRating - MinStarRating, notTrainedRecommender.Predict(1, 3, this.nativeTrainingData));

                // Check that trained classifier still protects settings
                Assert.Throws<InvalidOperationException>(() => { trainedRecommender.Settings.Training.IterationCount = 10; }); // Guarded: Throw
            }
        }

        /// <summary>
        /// Tests binary custom serialization/deserialization of the recommender operating on the standard data format.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")] // Fails with SerializationException : Unable to find assembly 'Infer.Learners.Tests'
        public void StandardDataFormatCustomSerializationTest()
        {
            const int BatchCount = 2;
            const string TrainedFileName = "trainedCustomStandardRecommender.bin";
            const string NotTrainedFileName = "notTrainedCustomStandardRecommender.bin";

            // Add features for cold test set user and item
            this.standardTrainingDataFeatures.UserFeatures.Add(User.WithId("u2"), Vector.FromArray(4, 2, 1.3, 1.2, 2));
            this.standardTrainingDataFeatures.ItemFeatures.Add(Item.WithId("i4"), Vector.FromArray(6.3, 0.5));

            // Train and serialize
            {
                var recommender = this.CreateStandardDataFormatMatchboxRecommender();

                recommender.Settings.Training.BatchCount = BatchCount;
                this.standardMapping.SaveForwardCompatible(NotTrainedFileName);
                recommender.SaveForwardCompatible(NotTrainedFileName, FileMode.Append);

                recommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
                recommender.SaveForwardCompatible(TrainedFileName);

                CheckStandardRatingPrediction(recommender, this.standardTrainingDataFeatures);
            }

            // Deserialize and test
            {
                // Check wrong versions throw a serialization exception
                CheckCustomSerializationVersionException(reader => MatchboxRecommender.LoadBackwardCompatible(reader, this.standardMapping));

                var trainedRecommender = MatchboxRecommender.LoadBackwardCompatible(TrainedFileName, this.standardMapping);

                IMatchboxRecommender<StandardDataset, User, Item, RatingDistribution, FeatureProvider> notTrainedRecommender;
                using (var stream = File.Open(NotTrainedFileName, FileMode.Open))
                {
                    using (var reader = new WrappedBinaryReader(new BinaryReader(stream)))
                    {
                        var deserializedMapping = new StandardRecommenderTestMapping(reader);
                        notTrainedRecommender = MatchboxRecommender.LoadBackwardCompatible(reader, deserializedMapping);
                    }
                }

                Assert.Equal(BatchCount, notTrainedRecommender.Settings.Training.BatchCount);
                notTrainedRecommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);

                CheckStandardRatingPrediction(trainedRecommender, this.standardTrainingDataFeatures);
                CheckStandardRatingPrediction(notTrainedRecommender, this.standardTrainingDataFeatures);

                // Check that trained classifier still protects settings
                Assert.Throws<InvalidOperationException>(() => { trainedRecommender.Settings.Training.IterationCount = 10; }); // Guarded: Throw
            }
        }

        /// <summary>
        /// Tests the support for negative user and item identifiers in prediction.
        /// </summary>
        [Fact]
        public void NegativeUserItemIdentifiersRegressionTest()
        {
            var recommender = this.CreateNativeDataFormatMatchboxRecommender();
            recommender.Settings.Training.UseUserFeatures = false;
            recommender.Settings.Training.UseItemFeatures = false;
            recommender.Train(this.nativeTrainingData);
            var predictedRating = recommender.Predict(-1, -5);

            Assert.Equal(2, predictedRating);
        }

        /// <summary>
        /// Tests the averaged parameters used for cold users and items in prediction.
        /// </summary>
        [Fact]
        public void ColdStartUserItemAverageParametersRegressionTest()
        {
            var recommender = this.CreateStandardDataFormatMatchboxRecommender();
            recommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);

            var featureProvider = new FeatureProvider
            {
                UserFeatures = new Dictionary<User, Vector>
                    {
                        { User.WithId("u"), SparseVector.FromArray(2.9, 3, 2.3, 3.2, 4.1) }
                    },
                ItemFeatures = new Dictionary<Item, Vector>
                    {
                        { Item.WithId("i"), SparseVector.FromArray(1.5, 1.9) }
                    }
            };

            var expectedRatingDistribution = 
                new SortedDictionary<int, double> { { -2, 0.3115 }, { -1, 0.01131 }, { 0, 0.1752 }, { 1, 0.171 }, { 2, 0.01169 }, { 3, 0.3193 } };

            var predictedRatingDistribution = recommender.PredictDistribution(User.WithId("u"), Item.WithId("i"), featureProvider);
            Assert.Equal(expectedRatingDistribution.Count, predictedRatingDistribution.Count);
            foreach (var expectedRating in expectedRatingDistribution)
            {
                Assert.Equal(expectedRating.Value, predictedRatingDistribution[expectedRating.Key], Tolerance);
            }
        }

        /// <summary>
        /// Tests the support for capabilities.
        /// </summary>
        [Fact]
        public void CapabilitiesRegressionTest()
        {
            var nativeRecommender = this.CreateNativeDataFormatMatchboxRecommender();
            Assert.True(nativeRecommender.Capabilities.SupportsColdStartUsers);
            Assert.False(nativeRecommender.Capabilities.IsResistantToShilling);

            var standardRecommender = this.CreateStandardDataFormatMatchboxRecommender();
            Assert.True(standardRecommender.Capabilities.SupportsColdStartUsers);
            Assert.False(standardRecommender.Capabilities.IsResistantToShilling);

            var randomRecommender =
                new RandomStarRatingRecommender<StandardDataset, Tuple<User, Item, int?>, User, Item, int, FeatureProvider, Vector>(
                    new StandardRecommenderTestMapping());
            Assert.True(randomRecommender.Capabilities.SupportsColdStartUsers);
            Assert.False(randomRecommender.Capabilities.IsResistantToShilling);
        }

        /// <summary>
        /// Tests the support for training and prediction settings.
        /// </summary>
        [Fact]
        public void SettingsRegressionTest()
        {
            this.TrainingPredictionSettingsTest(
                this.CreateNativeDataFormatMatchboxRecommender(), this.nativeTrainingData, this.nativeTrainingData);

            this.TrainingPredictionSettingsTest(
                this.CreateStandardDataFormatMatchboxRecommender(), this.standardTrainingData, this.standardTrainingDataFeatures);
        }

        /// <summary>
        /// Tests the case when only user features are defined.
        /// </summary>
        [Fact]
        public void UserFeaturesOnlyTest()
        {
            var recommender = MatchboxRecommender.Create(new StandardRecommenderTestMappingUserFeaturesOnly());
            recommender.Settings.Training.UseUserFeatures = true;
            recommender.Settings.Training.UseItemFeatures = false;
            recommender.Settings.Training.TraitCount = 3;
            recommender.Settings.Training.IterationCount = InferenceIterationCount;

            recommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures); // must not throw
        }

        /// <summary>
        /// Tests the case when only item features are defined.
        /// </summary>
        [Fact]
        public void ItemFeaturesOnlyTest()
        {
            var recommender = MatchboxRecommender.Create(new StandardRecommenderTestMappingItemFeaturesOnly());
            recommender.Settings.Training.UseUserFeatures = false;
            recommender.Settings.Training.UseItemFeatures = true;
            recommender.Settings.Training.TraitCount = 3;
            recommender.Settings.Training.IterationCount = InferenceIterationCount;

            recommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures); // must not throw
        }

        /// <summary>
        /// Tests whether ratings are accessed during bulk prediction.
        /// </summary>
        [Fact]
        public void RatingAccessDuringPredictionTest()
        {
            var recommender = this.CreateStandardDataFormatMatchboxRecommender();
            recommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
            var predictionPair = new List<Tuple<User, Item, int?>>
                                     {
                                         new Tuple<User, Item, int?>(User.WithId("u1"), Item.WithId("i1"), null)
                                     };

            // Must not throw
            recommender.Predict(new StandardDataset { Observations = predictionPair }, this.standardTrainingDataFeatures);
        }

        /// <summary>
        /// Tests if predictions do not change when the number of batches varies.
        /// </summary>
        [Fact]
        public void BatchingTest()
        {
            const int MaxBatchCount = 3;
            var predictions = new RatingDistribution[MaxBatchCount];

            for (int batch = 1; batch <= MaxBatchCount; ++batch)
            {
                var batchedRecommender = this.CreateStandardDataFormatMatchboxRecommender();
                batchedRecommender.Settings.Training.BatchCount = batch;
                batchedRecommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
                predictions[batch - 1] = batchedRecommender.PredictDistribution(User.WithId("u1"), Item.WithId("i3"), this.standardTrainingDataFeatures);
            }

            for (int batch = 1; batch < MaxBatchCount; ++batch)
            {
                foreach (var rating in predictions[batch])
                {
                    // The slightly larger tolerance is due to negligible mismatches 
                    // between the batched and non-batched generated code.
                    Assert.Equal(rating.Value, predictions[batch - 1][rating.Key], 0.1);
                }

                Assert.Equal(predictions[batch].GetMode(), predictions[batch - 1].GetMode());
            }
        }

        /// <summary>
        /// Tests if bulk prediction works properly when data batching is enabled.
        /// </summary>
        [Fact]
        public void BulkPredictionInBatchingTest()
        {
            var batchedRecommender = this.CreateStandardDataFormatMatchboxRecommender();
            batchedRecommender.Settings.Training.BatchCount = 2;
            batchedRecommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
            var predictions = batchedRecommender.Predict(this.standardTrainingData, this.standardTrainingDataFeatures);

            Assert.Equal(this.standardTrainingData.Observations.Count, predictions.Sum(x => x.Value.Count));
        }

        /// <summary>
        /// Regression test for recommendation on multiple threads without features.
        /// </summary>
        [Fact]
        public void MultiThreadedNoFeaturesRegressionTest()
        {
            const double ExpectedRatingProbability = 0.1751;

            for (int batchCount = 1; batchCount <= 4; ++batchCount)
            {
                var standardRecommender = this.CreateStandardDataFormatMatchboxRecommender();
                standardRecommender.Settings.Training.UseUserFeatures = false;
                standardRecommender.Settings.Training.UseItemFeatures = false;
                standardRecommender.Settings.Training.BatchCount = batchCount;
                standardRecommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
                double standardRating = standardRecommender.PredictDistribution(User.WithId("u1"), Item.WithId("i3"), this.standardTrainingDataFeatures)[MinStarRating];
                Assert.Equal(ExpectedRatingProbability, standardRating, Tolerance);
            }
        }

        /// <summary>
        /// Regression test for recommendation on multiple threads with both user and item features.
        /// </summary>
        [Fact]
        public void MultiThreadedUserItemFeaturesRegressionTest()
        {
            for (int batchCount = 1; batchCount <= 4; ++batchCount)
            {
                var standardRecommender = this.CreateStandardDataFormatMatchboxRecommender();
                standardRecommender.Settings.Training.BatchCount = batchCount;
                standardRecommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
                RatingDistribution standardRatingDistribution = standardRecommender.PredictDistribution(User.WithId("u1"), Item.WithId("i3"), this.standardTrainingDataFeatures);
                VerifyStandardRatingDistributionOfUserOneAndItemThree(standardRatingDistribution);
                int standardRating = standardRecommender.Predict(User.WithId("u1"), Item.WithId("i3"), this.standardTrainingDataFeatures);
                Assert.Equal(MaxStarRating, standardRating);
            }
        }

        [Fact]
        public void StandardDataFormatGetPosteriorsRegressionTest()
        {
            var recommender = this.CreateStandardDataFormatMatchboxRecommender();
            recommender.Settings.Training.UseItemFeatures = false;

            Assert.Throws<InvalidOperationException>(() => { recommender.GetPosteriorDistributions(); });

            recommender.Train(this.standardTrainingData, this.standardTrainingDataFeatures);
            var posteriors = recommender.GetPosteriorDistributions();

            Assert.Equal(this.standardTrainingData.Observations.Select(o => o.Item1).Distinct().Count(), posteriors.Users.Count);
            Assert.Equal(recommender.Settings.Training.TraitCount, posteriors.Users.First().Value.Traits.Count);
            Assert.Equal(this.standardTrainingData.Observations.Select(o => o.Item3).Distinct().Count() + 1, posteriors.Users.First().Value.Thresholds.Count);
            Assert.Equal(this.standardTrainingData.Observations.Select(o => o.Item2).Distinct().Count(), posteriors.Items.Count);
            Assert.Equal(this.standardTrainingDataFeatures.UserFeatures.First().Value.Count, posteriors.UserFeatures.Count);
            Assert.Equal(0, posteriors.ItemFeatures.Count);
            Assert.Equal(recommender.Settings.Training.TraitCount, posteriors.UserFeatures.First().TraitWeights.Count);
        }

        [Fact]
        public void NativeDataFormatGetPosteriorsRegressionTest()
        {
            var recommender = this.CreateNativeDataFormatMatchboxRecommender();
            recommender.Settings.Training.UseUserFeatures = false;

            Assert.Throws<InvalidOperationException>(() => { recommender.GetPosteriorDistributions(); });

            recommender.Train(this.nativeTrainingData, this.nativeTrainingData);
            var posteriors = recommender.GetPosteriorDistributions();

            Assert.Equal(this.nativeTrainingData.UserCount, posteriors.Users.Count);
            Assert.Equal(recommender.Settings.Training.TraitCount, posteriors.Users.First().Value.Traits.Count);
            Assert.Equal(this.nativeTrainingData.Ratings.Distinct().Count() + 1, posteriors.Users.First().Value.Thresholds.Count);
            Assert.Equal(this.nativeTrainingData.ItemCount, posteriors.Items.Count);
            Assert.Equal(0, posteriors.UserFeatures.Count);
            Assert.Equal(this.nativeTrainingData.NonZeroItemFeatureValues.First().Length, posteriors.ItemFeatures.Count);
            Assert.Equal(recommender.Settings.Training.TraitCount, posteriors.ItemFeatures.First().TraitWeights.Count);
        }

        #region Helpers

        /// <summary>
        /// Asserts that thresholds across users satisfy a specified equality criterion.
        /// </summary>
        /// <param name="thresholds">The user threshold posterior distribution.</param>
        /// <param name="equalityComparer">The comparer.</param>
        private static void AssertUserThresholdEquality(IList<IList<Gaussian>> thresholds, Action<object, object> equalityComparer)
        {
            for (int i = 1; i < thresholds.Count; ++i)
            {
                for (int j = 1; j < thresholds[i].Count - 1; ++j)
                {
                    if (j != thresholds[i].Count / 2)
                    {
                        equalityComparer(thresholds[0][j], thresholds[i][j]);
                    }
                    else
                    {
                        Assert.Equal(thresholds[0][j], thresholds[i][j]);
                    }
                }
            }
        }

        /// <summary>
        /// Checks that the specified deserialization action throws a serialization exception for invalid versions.
        /// </summary>
        /// <param name="deserialize">The action which deserializes from a binary reader.</param>
        private static void CheckCustomSerializationVersionException(Action<IReader> deserialize)
        {
            using (var stream = new MemoryStream())
            {
                var writer = new WrappedBinaryWriter(new BinaryWriter(stream));
                writer.Write(new Guid("2317C228-3BB2-423C-B299-33A64A1BC1F3")); // Invalid serialization version

                stream.Seek(0, SeekOrigin.Begin);

                using (var reader = new WrappedBinaryReader(new BinaryReader(stream)))
                {
                    Assert.Throws<SerializationException>(() => deserialize(reader));
                }
            }
        }

        /// <summary>
        /// Verifies the correctness of the predicted rating distribution for user 1 and item 3.
        /// </summary>
        /// <param name="actualRatingDistribution">The predicted distribution of the rating user 1 gave to the item 3.</param>
        private static void VerifyStandardRatingDistributionOfUserOneAndItemThree(RatingDistribution actualRatingDistribution)
        {
            var expectedRatingDistribution = new SortedDictionary<int, double> { { -2, 0.38 }, { -1, 0.02 }, { 0, 0.05 }, { 1, 0.035 }, { 2, 0.037 }, { 3, 0.47 } };

            Assert.Equal(expectedRatingDistribution.Count, actualRatingDistribution.Count);
            foreach (KeyValuePair<int, double> pair in expectedRatingDistribution)
            {
                Assert.Equal(pair.Value, actualRatingDistribution[pair.Key], Tolerance);
            }
        }

        /// <summary>
        /// Checks rating prediction.
        /// </summary>
        /// <param name="recommender">The recommender to test.</param>
        /// <param name="features">The features to use.</param>
        private static void CheckStandardRatingPrediction(
            IMatchboxRecommender<StandardDataset, User, Item, RatingDistribution, FeatureProvider> recommender,
            FeatureProvider features)
        {
            VerifyStandardRatingDistributionOfUserOneAndItemThree(
                recommender.PredictDistribution(User.WithId("u1"), Item.WithId("i3"), features));
            Assert.Equal(MaxStarRating, recommender.Predict(User.WithId("u1"), Item.WithId("i3"), features));

            // Test cold users and items
            Assert.Equal(MaxStarRating, recommender.Predict(User.WithId("u2"), Item.WithId("i3"), features));
            Assert.Throws<KeyNotFoundException>(() => recommender.Predict(User.WithId("u3"), Item.WithId("i3"), features));
            Assert.Equal(MaxStarRating, recommender.Predict(User.WithId("u1"), Item.WithId("i4"), features));
            Assert.Throws<KeyNotFoundException>(() => recommender.Predict(User.WithId("u1"), Item.WithId("i5"), features));
        }

        /// <summary>
        /// Creates a dataset in the native Matchbox format.
        /// </summary>
        private void InitializeNativeTrainingData()
        {
            this.nativeTrainingData = new NativeDataset
            {
                UserIds = new[] { 1, 0, 1, 0, 0, 1 },
                ItemIds = new[] { 1, 0, 2, 3, 2, 3 },
                Ratings = new[] { 0, 1, 2, 3, 4, 5 },
                UserCount = 2,
                ItemCount = 4,
                NonZeroUserFeatureValues = new[]
                {
                    new[] { 2, 3, 2.3, 3.2, 4 },
                    new[] { 6, 5.4, 4.5, 5, 4 }
                },
                NonZeroUserFeatureIndices = new[]
                {
                    new[] { 0, 1, 2, 3, 4 },
                    new[] { 4, 3, 2, 1, 0 }
                },
                NonZeroItemFeatureValues = new[]
                {
                    new[] { 0.2, 0.0 },
                    new[] { 0.0, 0.0 },
                    new[] { 0.2, 0.9 },
                    new[] { 1.2, 1.1 }
                },
                NonZeroItemFeatureIndices = new[]
                {
                    new[] { 0, 1 },
                    new[] { 1, 0 },
                    new[] { 0, 1 },
                    new[] { 1, 0 }
                }
            };

            this.nativeMapping = new NativeRecommenderTestMapping();
        }

        /// <summary>
        /// Creates a dataset in the standard recommender data format.
        /// </summary>
        private void InitializeStandardTrainingData()
        {
            this.standardTrainingData = new StandardDataset
            {
                Observations = new List<Tuple<User, Item, int?>>
                {
                    new Tuple<User, Item, int?>(User.WithId("u0"), Item.WithId("i0"), 1),
                    new Tuple<User, Item, int?>(User.WithId("u1"), Item.WithId("i1"), 0),
                    new Tuple<User, Item, int?>(User.WithId("u1"), Item.WithId("i2"), 2),
                    new Tuple<User, Item, int?>(User.WithId("u0"), Item.WithId("i3"), 3),
                    new Tuple<User, Item, int?>(User.WithId("u0"), Item.WithId("i2"), 4),
                    new Tuple<User, Item, int?>(User.WithId("u1"), Item.WithId("i3"), 5)
                }
            };

            this.standardTrainingDataFeatures = new FeatureProvider
                {
                    UserFeatures = new Dictionary<User, Vector>
                    {
                        { User.WithId("u0"), SparseVector.FromArray(2, 3, 2.3, 3.2, 4) },
                        { User.WithId("u1"), SparseVector.FromArray(4, 5, 4.5, 5.4, 6) }
                    },
                    ItemFeatures = new Dictionary<Item, Vector>
                    {
                        { Item.WithId("i0"), SparseVector.FromArray(0.2, 0.0) },
                        { Item.WithId("i1"), SparseVector.FromArray(0.0, 0.0) },
                        { Item.WithId("i2"), SparseVector.FromArray(0.2, 0.9) },
                        { Item.WithId("i3"), SparseVector.FromArray(1.1, 1.2) }
                    }
                };

            this.standardMapping = new StandardRecommenderTestMapping();
        }

        /// <summary>
        /// Verifies the correctness of the predicted rating distribution for user 1 and item 3.
        /// </summary>
        /// <param name="actualRatingDistribution">The predicted distribution of the rating user 1 gave to the item 3.</param>
        private void VerifyNativeRatingDistributionOfUserOneAndItemThree(Discrete actualRatingDistribution)
        {
            var expectedRatingDistribution = new Discrete(0.38, 0.02, 0.05, 0.035, 0.037, 0.47);

            Assert.Equal(expectedRatingDistribution.Dimension, actualRatingDistribution.Dimension);
            for (int i = 0; i < actualRatingDistribution.Dimension; ++i)
            {
                Assert.Equal(expectedRatingDistribution[i], actualRatingDistribution[i], Tolerance);
            }
        }

        /// <summary>
        /// Creates a native data format recommender with 3 traits and feature support.
        /// </summary>
        /// <returns>The created recommender.</returns>
        private IMatchboxRecommender<NativeDataset, int, int, Discrete, NativeDataset> CreateNativeDataFormatMatchboxRecommender()
        {
            var recommender = MatchboxRecommender.Create(this.nativeMapping);
            recommender.Settings.Training.UseUserFeatures = true;
            recommender.Settings.Training.UseItemFeatures = true;
            recommender.Settings.Training.TraitCount = 3;
            recommender.Settings.Training.IterationCount = InferenceIterationCount;

            return recommender;
        }

        /// <summary>
        /// Creates a standard data format recommender with 3 traits and feature support.
        /// </summary>
        /// <returns>The created recommender.</returns>
        private IMatchboxRecommender<StandardDataset, User, Item, RatingDistribution, FeatureProvider> CreateStandardDataFormatMatchboxRecommender()
        {
            var recommender = MatchboxRecommender.Create(this.standardMapping);
            recommender.Settings.Training.UseUserFeatures = true;
            recommender.Settings.Training.UseItemFeatures = true;
            recommender.Settings.Training.TraitCount = 3;
            recommender.Settings.Training.IterationCount = InferenceIterationCount;

            return recommender;
        }

        /// <summary>
        /// Tests if the training settings can only be set before training, 
        /// and if the prediction settings can be set both before and after training.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TRatingDistribution">The type of a distribution over ratings.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="recommender">The recommender to test.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="featureSource">The source of the features for the given instances.</param>
        private void TrainingPredictionSettingsTest<TInstanceSource, TUser, TItem, TRatingDistribution, TFeatureSource>(
            IMatchboxRecommender<TInstanceSource, TUser, TItem, TRatingDistribution, TFeatureSource> recommender,
            TInstanceSource instanceSource, 
            TFeatureSource featureSource)
        {
            recommender.Settings.Training.TraitCount = 10; // must not throw
            recommender.Settings.Training.Advanced.AffinityNoiseVariance = 0.01; // must not throw
            recommender.Settings.Prediction.SetPredictionLossFunction(LossFunction.Custom, Metrics.SquaredError); // must not throw

            recommender.Train(instanceSource, featureSource);

            Assert.Throws<InvalidOperationException>(
                () => { recommender.Settings.Training.TraitCount = 10; });
            Assert.Throws<InvalidOperationException>(
                () => { recommender.Settings.Training.Advanced.AffinityNoiseVariance = 0.01; });
            recommender.Settings.Prediction.SetPredictionLossFunction(LossFunction.Absolute); // must not throw
        }

        /// <summary>
        /// Tests if the recommender data consistency checks can correctly detect particular problems with data
        /// </summary>
        /// <param name="data">The data to test.</param>
        /// <param name="useFeatures">Indicates whether to use user and item features.</param>
        /// <param name="mustThrow">Indicates whether the tested code must throw a <see cref="MatchboxRecommenderException"/>.</param>
        private void TestDataConsistency(NativeDataset data, bool useFeatures, bool mustThrow)
        {
            var recommender = MatchboxRecommender.Create(this.nativeMapping);

            if (useFeatures)
            {
                recommender.Settings.Training.UseUserFeatures = true;
                recommender.Settings.Training.UseItemFeatures = true;
            }

            if (mustThrow)
            {
                Assert.Throws<MatchboxRecommenderException>(() => recommender.Train(data, data));
            }
            else
            {
                recommender.Train(data, data);
            }
        }

        #endregion

        #region Data model

        /// <summary>
        /// Base class for users and items.
        /// </summary>
        /// <typeparam name="T">The type of the derived class (<see cref="User"/> or <see cref="Item"/>).</typeparam>
        [Serializable]
        private class Entity<T> where T : Entity<T>, new()
        {
            /// <summary>
            /// Gets or sets the identifier.
            /// </summary>
            public string Id { get; set; }

            /// <summary>
            /// Creates new entity of a derived type with a given identifier.
            /// </summary>
            /// <param name="id">The identifier of a new entity.</param>
            /// <returns>The new entity.</returns>
            public static T WithId(string id)
            {
                return new T { Id = id };
            }

            /// <summary>
            /// Compares the entity with a given object. Comparison is identifier-based.
            /// </summary>
            /// <param name="obj">The object to compare with.
            /// </param>
            /// <returns>
            /// The <see cref="bool"/>.
            /// </returns>
            public override bool Equals(object obj)
            {
                if (obj == null || GetType() != obj.GetType())
                {
                    return false;
                }

                return object.Equals(((Entity<T>)obj).Id, this.Id);
            }

            /// <summary>
            /// Computes the hash code of the entity.
            /// </summary>
            /// <returns>The computed hash code.</returns>
            public override int GetHashCode()
            {
                return this.Id == null ? 0 : this.Id.GetHashCode();
            }
        }

        /// <summary>
        /// Represents users in the standard data format.
        /// </summary>
        [Serializable]
        private class User : Entity<User>
        {
            /// <summary>
            /// Returns a string that represents the current object.
            /// </summary>
            /// <returns>A string that represents the current object.</returns>
            public override string ToString()
            {
                return "User " + this.Id;
            }
        }

        /// <summary>
        /// Represents items in the standard data format.
        /// </summary>
        [Serializable]
        private class Item : Entity<Item>
        {
            /// <summary>
            /// Returns a string that represents the current object.
            /// </summary>
            /// <returns>A string that represents the current object.</returns>
            public override string ToString()
            {
                return "Item " + this.Id;
            }
        }

        /// <summary>
        /// Represents a dataset in the standard recommender format.
        /// </summary>
        private class StandardDataset
        {
            /// <summary>
            /// Gets or sets the list of observations.
            /// </summary>
            public List<Tuple<User, Item, int?>> Observations { get; set; }
        }

        /// <summary>
        /// Provides features for the data in the standard format.
        /// </summary>
        private class FeatureProvider
        {
            /// <summary>
            /// Gets or sets the mapping from user to features.
            /// </summary>
            public IDictionary<User, Vector> UserFeatures { get; set; }

            /// <summary>
            /// Gets or sets the mapping from item to features.
            /// </summary>
            public IDictionary<Item, Vector> ItemFeatures { get; set; }
        }

        /// <summary>
        /// Represents a dataset in the native Matchbox format.
        /// </summary>
        private class NativeDataset
        {
            /// <summary>
            /// Gets or sets the observed user ids.
            /// </summary>
            public int[] UserIds { get; set; }

            /// <summary>
            /// Gets or sets the observed item ids.
            /// </summary>
            public int[] ItemIds { get; set; }

            /// <summary>
            /// Gets or sets the observed ratings.
            /// </summary>
            public int[] Ratings { get; set; }

            /// <summary>
            /// Gets or sets the total number of users.
            /// </summary>
            public int UserCount { get; set; }

            /// <summary>
            /// Gets or sets the total number of items.
            /// </summary>
            public int ItemCount { get; set; }

            /// <summary>
            /// Gets or sets the array of non-zero user feature values.
            /// </summary>
            public double[][] NonZeroUserFeatureValues { get; set; }

            /// <summary>
            /// Gets or sets the array of non-zero user feature indices.
            /// </summary>
            public int[][] NonZeroUserFeatureIndices { get; set; }

            /// <summary>
            /// Gets or sets the array of non-zero item feature values.
            /// </summary>
            public double[][] NonZeroItemFeatureValues { get; set; }

            /// <summary>
            /// Gets or sets the array of non-zero item feature indices.
            /// </summary>
            public int[][] NonZeroItemFeatureIndices { get; set; }
        }

        #endregion

        #region Mapping implementations

        /// <summary>
        /// An implementation of <see cref="IMatchboxRecommenderMapping{TInstanceSource, TFeatureSource}"/> for <see cref="NativeDataset"/>.
        /// </summary>
        /// <remarks>This mapping doesn't support batching.</remarks>
        [Serializable]
        private class NativeRecommenderTestMapping : IMatchboxRecommenderMapping<NativeDataset, NativeDataset>, ICustomSerializable
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="NativeRecommenderTestMapping"/> class.
            /// </summary>
            public NativeRecommenderTestMapping()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeRecommenderTestMapping"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            public NativeRecommenderTestMapping(IReader reader)
            {
                if (reader == null)
                {
                    throw new ArgumentNullException(nameof(reader));
                }

                // Nothing to deserialize
            }

            /// <summary>
            /// Gets the list of user identifiers from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get the user identifiers from.</param>
            /// <param name="batchNumber">The number of the current batch (used only if the data is divided into batches).</param>
            /// <returns>The list of user identifiers.</returns>
            public IReadOnlyList<int> GetUserIds(NativeDataset instanceSource, int batchNumber = 0)
            {
                if (batchNumber != 0)
                {
                    throw new NotSupportedException("This mapping does not support data batching.");
                }

                return instanceSource.UserIds;
            }

            /// <summary>
            /// Gets the list of item identifiers from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get the item identifiers from.</param>
            /// <param name="batchNumber">The number of the current batch (used only if the data is divided into batches).</param>
            /// <returns>The list of item identifiers.</returns>
            public IReadOnlyList<int> GetItemIds(NativeDataset instanceSource, int batchNumber = 0)
            {
                if (batchNumber != 0)
                {
                    throw new NotSupportedException("This mapping does not support data batching.");
                }

                return instanceSource.ItemIds;
            }

            /// <summary>
            /// Gets the list of ratings from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get the ratings from.</param>
            /// <param name="batchNumber">The number of the current batch (used only if the data is divided into batches).</param>
            /// <returns>The list of ratings</returns>
            public IReadOnlyList<int> GetRatings(NativeDataset instanceSource, int batchNumber = 0)
            {
                if (batchNumber != 0)
                {
                    throw new NotSupportedException("This mapping does not support data batching.");
                }

                return instanceSource.Ratings;
            }

            /// <summary>
            /// Gets the number of users from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get number of users from.</param>
            /// <returns>The number of users.</returns>
            public int GetUserCount(NativeDataset instanceSource)
            {
                return instanceSource.UserCount;
            }

            /// <summary>
            /// Gets the number of items from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get number of items from.</param>
            /// <returns>The number of items.</returns>
            public int GetItemCount(NativeDataset instanceSource)
            {
                return instanceSource.ItemCount;
            }

            /// <summary>
            /// Gets the number of star ratings.
            /// This is equal to one plus the difference between the maximum and the minimum rating.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get number of items from.</param>
            /// <returns>The number of ratings.</returns>
            public int GetRatingCount(NativeDataset instanceSource)
            {
                return 6;
            }

            /// <summary>
            /// Gets the number of user features.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <returns>The number of user features.</returns>
            public int GetUserFeatureCount(NativeDataset featureSource)
            {
                return featureSource.NonZeroUserFeatureIndices.Max(
                    indices => indices == null || indices.Length == 0 ? 0 : indices.Max() + 1);
            }

            /// <summary>
            /// Gets the number of item features.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <returns>The number of item features.</returns>
            public int GetItemFeatureCount(NativeDataset featureSource)
            {
                return featureSource.NonZeroItemFeatureIndices.Max(
                    indices => indices == null || indices.Length == 0 ? 0 : indices.Max() + 1);
            }

            /// <summary>
            /// Gets non-zero feature values for all users present in a given feature source.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <returns>An array of non-zero user feature arrays where the outer array is indexed by user id.</returns>
            /// <remarks>This function will be called during training if the user feature support is enabled.</remarks>
            public IReadOnlyList<IReadOnlyList<double>> GetAllUserNonZeroFeatureValues(NativeDataset featureSource)
            {
                return featureSource.NonZeroUserFeatureValues;
            }

            /// <summary>
            /// Gets non-zero feature indices for all users present in a given feature source.
            /// </summary>
            /// <param name="featureSource">The source to obtain feature indices from.</param>
            /// <returns>An array of non-zero user feature index arrays where the outer array is indexed by user id.</returns>
            /// <remarks>This function will be called during training if the user feature support is enabled.</remarks>
            public IReadOnlyList<IReadOnlyList<int>> GetAllUserNonZeroFeatureIndices(NativeDataset featureSource)
            {
                return featureSource.NonZeroUserFeatureIndices;
            }

            /// <summary>
            /// Gets non-zero feature values for all items present in a given feature source.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <returns>An array of non-zero item feature arrays where the outer array is indexed by item id</returns>
            /// <remarks>This function will be called during training if the item feature support is enabled.</remarks>
            public IReadOnlyList<IReadOnlyList<double>> GetAllItemNonZeroFeatureValues(NativeDataset featureSource)
            {
                return featureSource.NonZeroItemFeatureValues;
            }

            /// <summary>
            /// Gets non-zero feature indices for all items present in a given feature source.
            /// </summary>
            /// <param name="featureSource">The source to obtain feature indices from.</param>
            /// <returns>An array of non-zero item feature index arrays where the outer array is indexed by item id</returns>
            /// <remarks>This function will be called during training if the item feature support is enabled.</remarks>
            public IReadOnlyList<IReadOnlyList<int>> GetAllItemNonZeroFeatureIndices(NativeDataset featureSource)
            {
                return featureSource.NonZeroItemFeatureIndices;
            }

            /// <summary>
            /// Gets non-zero feature values for a given user.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <param name="userId">The user identifier.</param>
            /// <returns>Non-zero feature values for the user.</returns>
            /// <remarks>This function will be called during prediction for cold users if the user feature support is enabled.</remarks>
            public IReadOnlyList<double> GetSingleUserNonZeroFeatureValues(NativeDataset featureSource, int userId)
            {
                return featureSource.NonZeroUserFeatureValues[userId];
            }

            /// <summary>
            /// Gets non-zero feature indices for a given user.
            /// </summary>
            /// <param name="featureSource">The source to obtain feature indices from.</param>
            /// <param name="userId">The user identifier.</param>
            /// <returns>Non-zero feature indices for the user.</returns>
            /// <remarks>This function will be called during prediction for cold users if the user feature support is enabled.</remarks>
            public IReadOnlyList<int> GetSingleUserNonZeroFeatureIndices(NativeDataset featureSource, int userId)
            {
                return featureSource.NonZeroUserFeatureIndices[userId];
            }

            /// <summary>
            /// Gets non-zero feature values for a given item.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <param name="itemId">The item identifier.</param>
            /// <returns>Non-zero feature values for the item.</returns>
            /// <remarks>This function will be called during prediction for cold items if the item feature support is enabled.</remarks>
            public IReadOnlyList<double> GetSingleItemNonZeroFeatureValues(NativeDataset featureSource, int itemId)
            {
                return featureSource.NonZeroItemFeatureValues[itemId];
            }

            /// <summary>
            /// Gets non-zero feature indices for a given item.
            /// </summary>
            /// <param name="featureSource">The source to obtain feature indices from.</param>
            /// <param name="itemId">The item identifier.</param>
            /// <returns>Non-zero feature values for the item.</returns>
            /// <remarks>This function will be called during prediction for cold items if the item feature support is enabled.</remarks>
            public IReadOnlyList<int> GetSingleItemNonZeroFeatureIndices(NativeDataset featureSource, int itemId)
            {
                return featureSource.NonZeroItemFeatureIndices[itemId];
            }

            /// <summary>
            /// Saves the state of the native data mapping using a writer to a binary stream.
            /// </summary>
            /// <param name="writer">The writer to save the state of the native data mapping to.</param>
            public void SaveForwardCompatible(IWriter writer)
            {
                // Nothing to serialize
            }
        }

        /// <summary>
        /// An implementation of
        /// <see cref="IStarRatingRecommenderMapping{TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues}"/>
        /// for the data format used in tests.
        /// </summary>
        [Serializable]
        private class StandardRecommenderTestMapping :
            IStarRatingRecommenderMapping<StandardDataset, Tuple<User, Item, int?>, User, Item, int, FeatureProvider, Vector>, ICustomSerializable
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="StandardRecommenderTestMapping"/> class.
            /// </summary>
            public StandardRecommenderTestMapping()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="StandardRecommenderTestMapping"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            public StandardRecommenderTestMapping(IReader reader)
            {
                reader.ReadSerializationVersion(4);

                // Nothing to deserialize
            }

            /// <summary>
            /// Retrieves a list of instances from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source to retrieve instances from.</param>
            /// <returns>The list of retrieved instances.</returns>
            public IEnumerable<Tuple<User, Item, int?>> GetInstances(StandardDataset instanceSource)
            {
                return instanceSource.Observations;
            }

            /// <summary>
            /// Extracts a user from a given instance.
            /// </summary>
            /// <param name="instanceSource">The source of instances providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract user from.</param>
            /// <returns>The extracted user.</returns>
            public User GetUser(StandardDataset instanceSource, Tuple<User, Item, int?> instance)
            {
                return instance.Item1;
            }

            /// <summary>
            /// Extracts an item from a given instance.
            /// </summary>
            /// <param name="instanceSource">The source of instances providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract item from.</param>
            /// <returns>The extracted item.</returns>
            public Item GetItem(StandardDataset instanceSource, Tuple<User, Item, int?> instance)
            {
                return instance.Item2;
            }

            /// <summary>
            /// Extracts a rating from a given instance.
            /// </summary>
            /// <param name="instanceSource">The source of instances providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract rating from.</param>
            /// <returns>The extracted rating.</returns>
            public int GetRating(StandardDataset instanceSource, Tuple<User, Item, int?> instance)
            {
                if (instance.Item3 == null)
                {
                    throw new ArgumentException("Rating is not contained in the given instance", nameof(instance));
                }

                return instance.Item3.Value + MinStarRating;
            }

            /// <summary>
            /// Provides a vector of features for a given user.
            /// </summary>
            /// <param name="featureSource">The source of features.</param>
            /// <param name="user">The user to provide features for.</param>
            /// <returns>The feature vector for <paramref name="user"/>.</returns>
            public virtual Vector GetUserFeatures(FeatureProvider featureSource, User user)
            {
                return featureSource.UserFeatures[user];
            }

            /// <summary>
            /// Provides a vector of features for a given item.
            /// </summary>
            /// <param name="featureSource">The source of features.</param>
            /// <param name="item">The item to provide features for.</param>
            /// <returns>The feature vector for <paramref name="item"/>.</returns>
            public virtual Vector GetItemFeatures(FeatureProvider featureSource, Item item)
            {
                return featureSource.ItemFeatures[item];
            }

            /// <summary>
            /// Provides the object describing how ratings provided by the instance source map to stars.
            /// </summary>
            /// <param name="instanceSource">The source of instances.</param>
            /// <returns>The object describing how ratings in the dataset map to the stars.</returns>
            public IStarRatingInfo<int> GetRatingInfo(StandardDataset instanceSource)
            {
                return new StarRatingInfo(MinStarRating, MaxStarRating);
            }

            /// <summary>
            /// Saves the state of the standard data mapping using a writer to a binary stream.
            /// </summary>
            /// <param name="writer">The writer to save the state of the standard data mapping to.</param>
            public void SaveForwardCompatible(IWriter writer)
            {
                writer.Write(3); // Fake serialization version
            }
        }

        /// <summary>
        /// An implementation of
        /// <see cref="IStarRatingRecommenderMapping{TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues}"/>
        /// for the data format used in tests. Throws when user features are accessed.
        /// </summary>
        [Serializable]
        private class StandardRecommenderTestMappingItemFeaturesOnly : StandardRecommenderTestMapping
        {
            /// <summary>
            /// Provides a vector of features for a given user.
            /// </summary>
            /// <param name="featureSource">The source of features.</param>
            /// <param name="user">The user to provide features for.</param>
            /// <returns>The feature vector for <paramref name="user"/>.</returns>
            public override Vector GetUserFeatures(FeatureProvider featureSource, User user)
            {
                throw new NotImplementedException();
            }            
        }

        /// <summary>
        /// An implementation of
        /// <see cref="IStarRatingRecommenderMapping{TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues}"/>
        /// for the data format used in tests. Throws when item features are accessed.
        /// </summary>
        [Serializable]
        private class StandardRecommenderTestMappingUserFeaturesOnly : StandardRecommenderTestMapping
        {
            /// <summary>
            /// Provides a vector of features for a given item.
            /// </summary>
            /// <param name="featureSource">The source of features.</param>
            /// <param name="item">The item to provide features for.</param>
            /// <returns>The feature vector for <paramref name="item"/>.</returns>
            public override Vector GetItemFeatures(FeatureProvider featureSource, Item item)
            {
                throw new NotImplementedException();
            }
        }

        #endregion
    }
}
