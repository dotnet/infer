// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    /// <summary>
    /// Represents the metadata of the users, items and ratings - features and counts.
    /// </summary>
    internal class InstanceMetadata
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="InstanceMetadata"/> class.
        /// </summary>
        /// <param name="userCount">The number of users.</param>
        /// <param name="itemCount">The number of items.</param>
        /// <param name="ratingCount">The number of star ratings.</param>
        /// <param name="userFeatures">The user features.</param>
        /// <param name="itemFeatures">The item features.</param>
        public InstanceMetadata(
            int userCount,
            int itemCount,
            int ratingCount,
            SparseFeatureMatrix userFeatures,
            SparseFeatureMatrix itemFeatures)
        {
            this.UserCount = userCount;
            this.ItemCount = itemCount;
            this.RatingCount = ratingCount;
            this.UserFeatures = userFeatures;
            this.ItemFeatures = itemFeatures;
        }

        /// <summary>
        /// Gets the number of users.
        /// </summary>
        public int UserCount { get; private set; }

        /// <summary>
        /// Gets the number of items.
        /// </summary>
        public int ItemCount { get; private set; }

        /// <summary>
        /// Gets the number of star ratings.
        /// </summary>
        public int RatingCount { get; private set; }

        /// <summary>
        /// Gets the user features.
        /// </summary>
        public SparseFeatureMatrix UserFeatures { get; private set; }

        /// <summary>
        /// Gets the item features.
        /// </summary>
        public SparseFeatureMatrix ItemFeatures { get; private set; }
    }
}
