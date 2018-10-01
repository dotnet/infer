// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    /// <summary>
    /// A mapping used by the implementations of the
    /// <see cref="IRecommender{TInstanceSource, TUser, TItem, TRating, TRatingDist, TFeatureSource}"/>
    /// interface which operate on star-rated data.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TRating">The type of a rating.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a source of features.</typeparam>
    /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
    public interface IStarRatingRecommenderMapping<in TInstanceSource, TInstance, TUser, TItem, TRating, in TFeatureSource, out TFeatureValues>
        : IRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>
    {
        /// <summary>
        /// Provides the object describing how ratings in the specified instance source map to stars.
        /// </summary>
        /// <param name="instanceSource">The source of the instances.</param>
        /// <returns>The object describing how ratings in the specified instance source map to stars.</returns>
        IStarRatingInfo<TRating> GetRatingInfo(TInstanceSource instanceSource);
    }
}