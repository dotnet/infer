// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Tests
{
    using System.Collections.Generic;

    using Xunit;
    using Assert = AssertHelper;

    using Microsoft.ML.Probabilistic.Learners.Runners;

    /// <summary>
    /// Tests functions for predictions save/load.
    /// </summary>
    public class RecommenderPersistenceUtilsTests
    {
        /// <summary>
        /// Tests save/load for predicted ratings.
        /// </summary>
        [Fact]
        public void TestPredictedRatingsSaveLoad()
        {
            var predictedRatings =
                new Dictionary<User, IDictionary<Item, int>>
                {
                    { new User("1", null), new Dictionary<Item, int> { { new Item("1", null), 2 }, { new Item("aba", null), 3 } } },
                    { new User("adbda", null), new Dictionary<Item, int> { { new Item("2", null), 5 } } }
                };
            const string FileName = "predicted_ratings.txt";
            RecommenderPersistenceUtils.SavePredictedRatings(FileName, predictedRatings);
            var loadedPredictedRatings = RecommenderPersistenceUtils.LoadPredictedRatings(FileName);
            Assert.Equal(
                predictedRatings,
                loadedPredictedRatings,
                (d1, d2) => Comparers.Dictionary(d1, d2, Comparers.Generic));
        }

        /// <summary>
        /// Tests save/load for recommended items.
        /// </summary>
        [Fact]
        public void TestRecommendedItemsSaveLoad()
        {
            var recommendedItems =
                new Dictionary<User, IEnumerable<Item>>
                {
                    { new User("1", null), new[] { new Item("1", null), new Item("aba", null) } },
                    { new User("adbda", null), new[] { new Item("2", null) } }
                };
            const string FileName = "recommended_items.txt";
            RecommenderPersistenceUtils.SaveRecommendedItems(FileName, recommendedItems);
            var loadedRecommendedItems = RecommenderPersistenceUtils.LoadRecommendedItems(FileName);
            Assert.Equal(recommendedItems, loadedRecommendedItems, Comparers.Collection);
        }

        /// <summary>
        /// Tests save/load for related user predictions.
        /// </summary>
        [Fact]
        public void TestRelatedUsersSaveLoad()
        {
            var relatedUsers =
                new Dictionary<User, IEnumerable<User>>
                {
                    { new User("1", null), new[] { new User("1", null), new User("aba", null) } },
                    { new User("adbda", null), new[] { new User("2", null) } }
                };
            const string FileName = "related_users.txt";
            RecommenderPersistenceUtils.SaveRelatedUsers(FileName, relatedUsers);
            var loadedRelatedUsers = RecommenderPersistenceUtils.LoadRelatedUsers(FileName);
            Assert.Equal(relatedUsers, loadedRelatedUsers, Comparers.Collection);
        }

        /// <summary>
        /// Tests save/load for related item predictions.
        /// </summary>
        [Fact]
        public void TestRelatedItemsSaveLoad()
        {
            var relatedItems =
                new Dictionary<Item, IEnumerable<Item>>
                {
                    { new Item("1", null), new[] { new Item("1", null), new Item("aba", null) } },
                    { new Item("adbda", null), new[] { new Item("2", null) } }
                };
            const string FileName = "related_items.txt";
            RecommenderPersistenceUtils.SaveRelatedItems(FileName, relatedItems);
            var loadedRelatedItems = RecommenderPersistenceUtils.LoadRelatedItems(FileName);
            Assert.Equal(relatedItems, loadedRelatedItems, Comparers.Collection);
        }       
    }
}
