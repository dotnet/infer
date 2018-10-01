// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using Xunit;
    using Assert = AssertHelper;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    using Dataset = System.Collections.Generic.IEnumerable<System.Tuple<int, int, int>>; // User, item, rating
    

    /// <summary>
    /// Tests for the recommender data splitting routines.
    /// </summary>
    public class RecommenderDataSplittingTests
    {
        /// <summary>
        /// Tests regular splitting on datasets of different size.
        /// </summary>
        [Fact]
        public void TestSplitting()
        {
            // Empty dataset
            TestSplittingHelper(0, 0, 1, 0.5, 0.25, 0, 0, 0, 0, true);
            TestSplittingHelper(0, 0, 1, 0.5, 0.25, 0, 0, 0, 0, false);
            
            // Single instance in the dataset
            TestSplittingHelper(1, 1, 1, 0.5, 0.25, 0, 0, 0, 0, true);
            TestSplittingHelper(1, 1, 1, 0.5, 0.25, 0, 0, 0, 0, false);
            
            // Several instances in the dataset
            TestSplittingHelper(10, 10, 0.1, 0.5, 0.25, 0, 0, 0, 0, true);
            TestSplittingHelper(10, 10, 0.1, 0.5, 0.25, 0, 0, 0, 0, false);
        }

        /// <summary>
        /// Tests splitting with cold users and items on datasets of different size.
        /// </summary>
        [Fact]
        public void TestSplittingWithColdUsersAndItems()
        {
            // Empty dataset
            TestSplittingHelper(0, 0, 1, 0.5, 0.25, 0.1, 0.1, 0, 0, true);
            TestSplittingHelper(0, 0, 1, 0.5, 0.25, 0.1, 0.1, 0, 0, false);

            // Single instance in the dataset
            TestSplittingHelper(1, 1, 1, 0.5, 0.25, 0.1, 0.1, 0, 0, true);
            TestSplittingHelper(1, 1, 1, 0.5, 0.25, 0.1, 0.1, 0, 0, false);

            // Several instances in the dataset
            TestSplittingHelper(10, 10, 0.1, 0.5, 0.25, 0.1, 0.1, 0, 0, true);
            TestSplittingHelper(10, 10, 0.1, 0.5, 0.25, 0.1, 0.1, 0, 0, false);
        }

        /// <summary>
        /// Tests splitting with ignored cold users and items on datasets of different size.
        /// </summary>
        [Fact]
        public void TestSplittingWithIgnoredUsersAndItems()
        {
            // Empty dataset
            TestSplittingHelper(0, 0, 1, 0.5, 0.25, 0, 0, 0.1, 0.1, true);
            TestSplittingHelper(0, 0, 1, 0.5, 0.25, 0, 0, 0.1, 0.1, false);

            // Single instance in the dataset
            TestSplittingHelper(1, 1, 1, 0.5, 0.25, 0, 0, 0.1, 0.1, true);
            TestSplittingHelper(1, 1, 1, 0.5, 0.25, 0, 0, 0.1, 0.1, false);

            // Several instances in the dataset
            TestSplittingHelper(10, 10, 0.1, 0.5, 0.25, 0, 0, 0.1, 0.1, true);
            TestSplittingHelper(10, 10, 0.1, 0.5, 0.25, 0, 0, 0.1, 0.1, false);
        }

        /// <summary>
        /// Tests splitting with both cold and ignored users and items on datasets of different size.
        /// </summary>
        [Fact]
        public void TestSplittingWithColdAndIgnoredUsersAndItems()
        {
            // Empty dataset
            TestSplittingHelper(0, 0, 1, 0.5, 0.25, 0.1, 0.1, 0.1, 0.1, true);
            TestSplittingHelper(0, 0, 1, 0.5, 0.25, 0.1, 0.1, 0.1, 0.1, false);

            // Single instance in the dataset
            TestSplittingHelper(1, 1, 1, 0.5, 0.25, 0.1, 0.1, 0.1, 0.1, true);
            TestSplittingHelper(1, 1, 1, 0.5, 0.25, 0.1, 0.1, 0.1, 0.1, false);

            // Several instances in the dataset
            TestSplittingHelper(10, 10, 0.1, 0.5, 0.25, 0.1, 0.1, 0.1, 0.1, true);
            TestSplittingHelper(10, 10, 0.1, 0.5, 0.25, 0.1, 0.1, 0.1, 0.1, false);
        }

        /// <summary>
        /// Tests if incorrectly specified fractions of users and items are detected properly.
        /// </summary>
        [Fact]
        public void TestIncorrectFractionSums()
        {
            Assert.Throws<ArgumentException>(() => TestSplittingHelper(1, 1, 1, 0.5, 0.25, 0.5, 0, 0.1, 0, false));
            Assert.Throws<ArgumentException>(() => TestSplittingHelper(1, 1, 1, 0.5, 0.25, 0.6, 0, 0, 0, false));
            Assert.Throws<ArgumentException>(() => TestSplittingHelper(1, 1, 1, 0.5, 0.25, 0, 0.5, 0, 0.6, false));
        }

        /// <summary>
        /// Tests if the generated dataset has requested fractions of users of various kinds.
        /// </summary>
        [Fact]
        public void TestUserFractions()
        {
            const int UserCount = 1000;
            const int ItemCount = 10;
            const double TrainingOnlyUserFraction = 0.6;
            const double IgnoredUserFraction = 0.1;
            const double ColdUserFraction = 0.2;
            const double FractionTolerance = 1.0 / (UserCount * ItemCount);
            
            var datasets = TestSplittingHelper(UserCount, ItemCount, 1.0, TrainingOnlyUserFraction, 0.2, ColdUserFraction, 0.0, IgnoredUserFraction, 0.0, true);
            
            // Ignored users
            int ignoredUserCount =
                datasets.Item1.Select(t => t.Item1) // All users
                              .Except(datasets.Item2.Select(t => t.Item1)) // Minus training users
                              .Except(datasets.Item3.Select(t => t.Item1)) // Minus test users
                              .Count();
            Assert.Equal(IgnoredUserFraction, ignoredUserCount / (double)UserCount, FractionTolerance);
            
            // Cold users
            int coldUserCount =
                datasets.Item3.Select(t => t.Item1) // Test users
                              .Except(datasets.Item2.Select(t => t.Item1)) // Minus training users
                              .Count();
            Assert.Equal(ColdUserFraction, coldUserCount / (double)UserCount, FractionTolerance);
            
            // Training-only users
            int trainingOnlyUserCount =
                datasets.Item2.Select(t => t.Item1) // Training users
                              .Except(datasets.Item3.Select(t => t.Item1)) // Minus test users
                              .Count();
            Assert.Equal(TrainingOnlyUserFraction, trainingOnlyUserCount / (double)UserCount, FractionTolerance);
        }

        /// <summary>
        /// Tests if the generated dataset has requested fractions of items of various kinds.
        /// </summary>
        [Fact]
        public void TestItemFractions()
        {
            const int UserCount = 10;
            const int ItemCount = 1000;
            const double IgnoredItemFraction = 0.1;
            const double ColdItemFraction = 0.2;
            const double FractionTolerance = 1.0 / (UserCount * ItemCount);

            var datasets = TestSplittingHelper(UserCount, ItemCount, 1.0, 0.5, 0.5, 0.0, ColdItemFraction, 0.0, IgnoredItemFraction, false);

            // Ignored items
            int ignoredItemCount =
                datasets.Item1.Select(t => t.Item2) // All items
                              .Except(datasets.Item2.Select(t => t.Item2)) // Minus training items
                              .Except(datasets.Item3.Select(t => t.Item2)) // Minus test items
                              .Count();
            Assert.Equal(IgnoredItemFraction, ignoredItemCount / (double)ItemCount, FractionTolerance);

            // Cold items
            int coldItemCount =
                datasets.Item3.Select(t => t.Item2) // Test items
                              .Except(datasets.Item2.Select(t => t.Item2)) // Minus training items
                              .Count();
            Assert.Equal(ColdItemFraction, coldItemCount / (double)ItemCount, FractionTolerance);
        }

        /// <summary>
        /// Generates a random dataset of the specified size, splits it as requested and checks the correctness of the resulting split.
        /// </summary>
        /// <param name="userCount">The number of users in the dataset.</param>
        /// <param name="itemCount">The number of items in the dataset.</param>
        /// <param name="sparsity">The probability of a random item to be rated by a random user.</param>
        /// <param name="trainingOnlyUserFraction">The fraction of users presented only in the training set.</param>
        /// <param name="testUserTrainingRatingFraction">The fraction of ratings in the training set for each user who is presented in both sets.</param>
        /// <param name="coldUserFraction">The fraction of users presented only in test set.</param>
        /// <param name="coldItemFraction">The fraction of items presented only in test set.</param>
        /// <param name="ignoredUserFraction">The fraction of users not presented in any of the sets.</param>
        /// <param name="ignoredItemFraction">The fraction of items not presented in any of the sets.</param>
        /// <param name="removeOccasionalColdItems">Specifies whether the occasionally produced cold items should be removed from the test set.</param>
        /// <returns>A triple containing the generated dataset, the training subset, and the test subset.</returns>
        private static Tuple<Dataset, Dataset, Dataset> TestSplittingHelper(
            int userCount,
            int itemCount,
            double sparsity,
            double trainingOnlyUserFraction,
            double testUserTrainingRatingFraction,
            double coldUserFraction,
            double coldItemFraction,
            double ignoredUserFraction,
            double ignoredItemFraction,
            bool removeOccasionalColdItems)
        {
            Dataset dataset = GenerateDataset(userCount, itemCount, sparsity);
            
            var mapping = new Mapping();
            var splittingMapping = mapping.SplitToTrainTest(
                trainingOnlyUserFraction,
                testUserTrainingRatingFraction,
                coldUserFraction,
                coldItemFraction,
                ignoredUserFraction,
                ignoredItemFraction,
                removeOccasionalColdItems);
            
            Dataset trainingDataset = splittingMapping.GetInstances(SplitInstanceSource.Training(dataset));
            Dataset testDataset = splittingMapping.GetInstances(SplitInstanceSource.Test(dataset));

            CheckDatasetSplitCorrectness(
                dataset,
                trainingDataset,
                testDataset,
                coldUserFraction > 0,
                coldItemFraction > 0,
                ignoredUserFraction > 0,
                ignoredItemFraction > 0,
                removeOccasionalColdItems);

            return Tuple.Create(dataset, trainingDataset, testDataset);
        }

        /// <summary>
        /// Performs sanity checks on the dataset splitting results.
        /// </summary>
        /// <param name="wholeDataset">The dataset that was split.</param>
        /// <param name="trainingDataset">The training part of the <paramref name="wholeDataset"/>.</param>
        /// <param name="testDataset">The test part of the <paramref name="wholeDataset"/>.</param>
        /// <param name="hasColdUsers">Specifies whether test dataset is supposed to have cold users.</param>
        /// <param name="hasColdItems">Specifies whether test dataset is supposed to have cold items.</param>
        /// <param name="hasIgnoredUsers">Specifies whether some users were ignored when splitting data.</param>
        /// <param name="hasIgnoredItems">Specifies whether some items were ignored when splitting data.</param>
        /// <param name="areOccasionalColdItemsRemoved">Specifies whether the occasionally produced cold items were removed from the test set.</param>
        private static void CheckDatasetSplitCorrectness(
            Dataset wholeDataset,
            Dataset trainingDataset,
            Dataset testDataset,
            bool hasColdUsers,
            bool hasColdItems,
            bool hasIgnoredUsers,
            bool hasIgnoredItems,
            bool areOccasionalColdItemsRemoved)
        {
            // Training and test sets should have zero intersection
            var trainTestIntersection = trainingDataset.Intersect(testDataset);
            Assert.Empty(trainTestIntersection);

            if (!hasIgnoredUsers && !hasIgnoredItems && !areOccasionalColdItemsRemoved)
            {
                // All the observations should be either in the training set or in the test set
                var missingObservations = wholeDataset.Except(trainingDataset).Except(testDataset);
                Assert.Empty(missingObservations);
            }

            if (!hasColdItems && areOccasionalColdItemsRemoved)
            {
                // There should be no cold items
                var coldItems = testDataset.Select(obs => obs.Item2).Except(trainingDataset.Select(obs => obs.Item2));
                Assert.Empty(coldItems);
            }

            if (!hasColdUsers)
            {
                // There should be no cold users
                var coldUsers = testDataset.Select(obs => obs.Item1).Except(trainingDataset.Select(obs => obs.Item1));
                Assert.Empty(coldUsers);
            }
        }

        /// <summary>
        /// Generates a random dataset.
        /// </summary>
        /// <param name="userCount">The number of users in the dataset.</param>
        /// <param name="itemCount">The number of items in the dataset.</param>
        /// <param name="sparsity">The probability of a random item to be rated by a random user.</param>
        /// <returns>The generated dataset.</returns>
        private static Dataset GenerateDataset(int userCount, int itemCount, double sparsity)
        {
            var result = new List<Tuple<int, int, int>>();
            
            const int MinRating = 1;
            const int MaxRating = 5;

            for (int i = 0; i < userCount; i++)
            {
                for (int j = 0; j < itemCount; j++)
                {
                    if (Rand.Double() < sparsity)
                    {
                        result.Add(Tuple.Create(i, j, Rand.Int(MinRating, MaxRating + 1)));
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Mapping for the simple data model used in tests.
        /// TODO: expression-based mappings should be resurrected at least for tests
        /// </summary>
        private class Mapping : IRecommenderMapping<Dataset, Tuple<int, int, int>, int, int, NoFeatureSource, int>
        {
            /// <summary>
            /// Retrieves a list of instances from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source to retrieve instances from.</param>
            /// <returns>The list of retrieved instances.</returns>
            public Dataset GetInstances(Dataset instanceSource)
            {
                return instanceSource;
            }

            /// <summary>
            /// Extracts a user from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract user from.</param>
            /// <returns>The extracted user.</returns>
            public int GetUser(Dataset instanceSource, Tuple<int, int, int> instance)
            {
                return instance.Item1;
            }

            /// <summary>
            /// Extracts an item from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract item from.</param>
            /// <returns>The extracted item.</returns>
            public int GetItem(Dataset instanceSource, Tuple<int, int, int> instance)
            {
                return instance.Item2;
            }

            /// <summary>
            /// Extracts a rating from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract rating from.</param>
            /// <returns>The extracted rating.</returns>
            public int GetRating(Dataset instanceSource, Tuple<int, int, int> instance)
            {
                // This method should not be called by the data splitter
                throw new NotImplementedException();
            }

            /// <summary>
            /// Provides a vector of features for a given user.
            /// </summary>
            /// <param name="featureSource">The storage of features.</param>
            /// <param name="user">The user to provide features for.</param>
            /// <returns>The feature vector for <paramref name="user"/>.</returns>
            public int GetUserFeatures(NoFeatureSource featureSource, int user)
            {
                // This method should not be called by the data splitter
                throw new NotImplementedException();
            }

            /// <summary>
            /// Provides a vector of features for a given item.
            /// </summary>
            /// <param name="featureSource">The storage of features.</param>
            /// <param name="item">The item to provide features for.</param>
            /// <returns>The feature vector for <paramref name="item"/>.</returns>
            public int GetItemFeatures(NoFeatureSource featureSource, int item)
            {
                // This method should not be called by the data splitter
                throw new NotImplementedException();
            }
        }
    }
}
