// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    /// <summary>
    /// A mapping used by the <see cref="StarRatingRecommenderEvaluator{TInstanceSource,TUser,TItem,TRating}"/> class.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of an instance source.</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TRating">The type of a rating.</typeparam>
    public interface IStarRatingRecommenderEvaluatorMapping<in TInstanceSource, TUser, TItem, TRating>
        : IRecommenderEvaluatorMapping<TInstanceSource, TUser, TItem, TRating>
    {
        /// <summary>
        /// Gets the object describing how ratings provided by the instance source map to stars.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <returns>The object describing how ratings provided by the instance source map to stars.</returns>
        IStarRatingInfo<TRating> GetRatingInfo(TInstanceSource instanceSource);
    }
}