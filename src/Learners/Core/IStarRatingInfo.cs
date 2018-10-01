// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    /// <summary>
    /// Interface to provide a mapping from ratings of arbitrary type <typeparamref name="TRating"/> to star ratings.
    /// </summary>
    /// <typeparam name="TRating">The type of the ratings.</typeparam>
    public interface IStarRatingInfo<in TRating>
    {
        /// <summary>
        /// Gets the minimum possible star rating.
        /// </summary>
        /// <remarks>
        /// This value must not be negative.
        /// </remarks>
        int MinStarRating { get; }

        /// <summary>
        /// Gets the maximum possible star rating.
        /// </summary>
        int MaxStarRating { get; }

        /// <summary>
        /// Converts a rating to a star rating.
        /// </summary>
        /// <param name="rating">The rating.</param>
        /// <returns>The corresponding star rating.</returns>
        int ToStarRating(TRating rating);
    }
}
