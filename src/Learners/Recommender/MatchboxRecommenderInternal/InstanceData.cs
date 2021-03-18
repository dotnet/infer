// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System.Collections.Generic;
    using System.Diagnostics;

    /// <summary>
    /// Represents the instance data - users, items and ratings.
    /// </summary>
    internal class InstanceData
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="InstanceData"/> class.
        /// </summary>
        /// <param name="userIds">The user identifiers.</param>
        /// <param name="itemIds">The item identifiers.</param>
        /// <param name="ratings">The ratings.</param>
        public InstanceData(IReadOnlyList<int> userIds, IReadOnlyList<int> itemIds, IReadOnlyList<int> ratings)
        {
            Debug.Assert(
                userIds != null && itemIds != null && ratings != null,
                "Valid instance arrays must be provided.");
            Debug.Assert(
                userIds.Count == itemIds.Count && userIds.Count == ratings.Count,
                "The instance arrays must be of the same length.");

            this.UserIds = userIds;
            this.ItemIds = itemIds;
            this.Ratings = ratings;
        }

        /// <summary>
        /// Gets the user identifiers.
        /// </summary>
        public IReadOnlyList<int> UserIds { get; private set; }

        /// <summary>
        /// Gets the item identifiers.
        /// </summary>
        public IReadOnlyList<int> ItemIds { get; private set; }

        /// <summary>
        /// Gets the ratings.
        /// </summary>
        public IReadOnlyList<int> Ratings { get; private set; }
    }
}
