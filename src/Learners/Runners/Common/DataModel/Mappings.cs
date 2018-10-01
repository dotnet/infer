// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// Contains all the data mappings used in evaluation.
    /// </summary>
    public static class Mappings
    {
        /// <summary>
        /// Initializes static members of the <see cref="Mappings"/> class.
        /// </summary>
        static Mappings()
        {
            StarRatingRecommender = new StarRatingRecommenderMapping();
            Classifier = new ClassifierMapping();
        }

        /// <summary>
        /// Gets the mapping from dataset format to the common classifier format.
        /// </summary>
        public static IClassifierMapping<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string, Vector> Classifier
        {
            get;
            private set;
        }

        /// <summary>
        /// Gets the mapping from dataset format to the common recommender format.
        /// </summary>
        public static IStarRatingRecommenderMapping<RecommenderDataset, RatedUserItem, User, Item, int, DummyFeatureSource, Vector> StarRatingRecommender
        {
            get;
            private set;
        }

        #region StarRatingRecommenderMapping implementation

        /// <summary>
        /// Represents a star rating recommender mapping for the particular data model
        /// </summary>
        [Serializable]
        private class StarRatingRecommenderMapping : IStarRatingRecommenderMapping<RecommenderDataset, RatedUserItem, User, Item, int, DummyFeatureSource, Vector>
        {
            /// <summary>
            /// Retrieves a list of instances from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source to retrieve instances from.</param>
            /// <returns>The list of retrieved instances.</returns>
            public IEnumerable<RatedUserItem> GetInstances(RecommenderDataset instanceSource)
            {
                return instanceSource.Observations;
            }

            /// <summary>
            /// Extracts a user from a given instance.
            /// </summary>
            /// <param name="instanceSource">The parameter is not used.</param>
            /// <param name="instance">The instance to extract user from.</param>
            /// <returns>The extracted user.</returns>
            public User GetUser(RecommenderDataset instanceSource, RatedUserItem instance)
            {
                return instance.User;
            }

            /// <summary>
            /// Extracts an item from a given instance.
            /// </summary>
            /// <param name="instanceSource">The parameter is not used.</param>
            /// <param name="instance">The instance to extract item from.</param>
            /// <returns>The extracted item.</returns>
            public Item GetItem(RecommenderDataset instanceSource, RatedUserItem instance)
            {
                return instance.Item;
            }

            /// <summary>
            /// Extracts a rating from a given instance.
            /// </summary>
            /// <param name="instanceSource">The parameter is not used.</param>
            /// <param name="instance">The instance to extract rating from.</param>
            /// <returns>The extracted rating.</returns>
            public int GetRating(RecommenderDataset instanceSource, RatedUserItem instance)
            {
                return instance.Rating;
            }

            /// <summary>
            /// Provides a vector of features for a given user.
            /// </summary>
            /// <param name="featureSource">The parameter is not used.</param>
            /// <param name="user">The user to provide features for.</param>
            /// <returns>The feature vector for <paramref name="user"/>.</returns>
            public Vector GetUserFeatures(DummyFeatureSource featureSource, User user)
            {
                return user.Features;
            }

            /// <summary>
            /// Provides a vector of features for a given item.
            /// </summary>
            /// <param name="featureSource">The parameter is not used.</param>
            /// <param name="item">The item to provide features for.</param>
            /// <returns>The feature vector for <paramref name="item"/>.</returns>
            public Vector GetItemFeatures(DummyFeatureSource featureSource, Item item)
            {
                return item.Features;
            }

            /// <summary>
            /// Provides the object describing how ratings provided by the instance source map to stars.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <returns>The object describing how ratings provided by the instance source map to stars.</returns>
            public IStarRatingInfo<int> GetRatingInfo(RecommenderDataset instanceSource)
            {
                return instanceSource.StarRatingInfo;
            }
        }

        #endregion
    }
}
