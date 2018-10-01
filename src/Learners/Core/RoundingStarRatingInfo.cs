// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;

    /// <summary>
    /// An implementation of <see cref="IStarRatingInfo{TDataRating}"/> which converts floating-point ratings to star ratings by rounding.
    /// </summary>
    public class RoundingStarRatingInfo : IStarRatingInfo<double>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RoundingStarRatingInfo"/> class.
        /// </summary>
        /// <param name="minRating">The minimum star rating.</param>
        /// <param name="maxRating">The maximum star rating.</param>
        public RoundingStarRatingInfo(int minRating, int maxRating)
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
        /// Converts a floating-point rating to a star rating by rounding to the nearest integer.
        /// </summary>
        /// <param name="rating">The floating-point rating.</param>
        /// <returns>The rounded star rating.</returns>
        public int ToStarRating(double rating)
        {
            return Convert.ToInt32(rating);
        }
    }
}
