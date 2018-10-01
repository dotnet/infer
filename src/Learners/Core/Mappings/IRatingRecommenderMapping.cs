// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    /// <summary>
    /// A mapping used by the implementations of the
    /// <see cref="IRecommender{TInstanceSource, TUser, TItem, TRating, TRatingDist, TFeatureSource}"/>
    /// which operate on rated data.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TRating">The type of a rating.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
    /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
    public interface IRatingRecommenderMapping<in TInstanceSource, TInstance, TUser, TItem, out TRating, in TFeatureSource, out TFeatureValues>
        : IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>
    {
        /// <summary>
        /// Extracts a rating from a given instance.
        /// </summary>
        /// <param name="instanceSource">The source of instances providing the <paramref name="instance"/>.</param>
        /// <param name="instance">The instance to extract the rating from.</param>
        /// <returns>The extracted rating.</returns>
        TRating GetRating(TInstanceSource instanceSource, TInstance instance);
    }
}
