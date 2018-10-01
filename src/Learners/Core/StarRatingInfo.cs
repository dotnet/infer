// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;

    /// <summary>
    /// Provides a mapping for the case in which ratings are already star ratings.
    /// </summary>
    [Serializable]
    public class StarRatingInfo : IStarRatingInfo<int>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="StarRatingInfo"/> class.
        /// </summary>
        /// <param name="minRating">The minimum possible rating.</param>
        /// <param name="maxRating">The maximum possible rating.</param>
        public StarRatingInfo(int minRating, int maxRating)
        {
            this.MinStarRating = minRating;
            this.MaxStarRating = maxRating;
        }

        /// <summary>
        /// Gets the minimum possible star rating.
        /// </summary>
        public int MinStarRating { get; private set; }

        /// <summary>
        /// Gets the maximum possible star rating.
        /// </summary>
        public int MaxStarRating { get; private set; }

        /// <summary>
        /// Converts a rating to the corresponding star rating.
        /// </summary>
        /// <param name="rating">The rating.</param>
        /// <returns>The corresponding star rating.</returns>
        public int ToStarRating(int rating)
        {
            return rating;
        }
    }
}