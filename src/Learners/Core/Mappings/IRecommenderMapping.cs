// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System.Collections.Generic;

    /// <summary>
    /// A mapping used by the implementations of the
    /// <see cref="IRecommender{TInstanceSource, TUser, TItem, TRating, TRatingDist, TFeatureSource}"/>
    /// interface which operate on positive-only data.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
    /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
    public interface IRecommenderMapping<in TInstanceSource, TInstance, TUser, TItem, in TFeatureSource, out TFeatureValues>
    {
        /// <summary>
        /// Retrieves a list of instances from a given instance source.
        /// </summary>
        /// <param name="instanceSource">The source to retrieve instances from.</param>
        /// <returns>The list of retrieved instances.</returns>
        IEnumerable<TInstance> GetInstances(TInstanceSource instanceSource);

        /// <summary>
        /// Extracts a user from a given instance.
        /// </summary>
        /// <param name="instanceSource">The source of instances providing the <paramref name="instance"/>.</param>
        /// <param name="instance">The instance to extract the user from.</param>
        /// <returns>The extracted user.</returns>
        TUser GetUser(TInstanceSource instanceSource, TInstance instance);

        /// <summary>
        /// Extracts an item from a given instance.
        /// </summary>
        /// <param name="instanceSource">The source of instances providing the <paramref name="instance"/>.</param>
        /// <param name="instance">The instance to extract the item from.</param>
        /// <returns>The extracted item.</returns>
        TItem GetItem(TInstanceSource instanceSource, TInstance instance);

        /// <summary>
        /// Provides the features for a given user.
        /// </summary>
        /// <param name="featureSource">The source of the features.</param>
        /// <param name="user">The user to provide the features for.</param>
        /// <returns>The features for <paramref name="user"/>.</returns>
        TFeatureValues GetUserFeatures(TFeatureSource featureSource, TUser user);

        /// <summary>
        /// Provides the features for a given item.
        /// </summary>
        /// <param name="featureSource">The source of the features.</param>
        /// <param name="item">The item to provide the features for.</param>
        /// <returns>The features for <paramref name="item"/>.</returns>
        TFeatureValues GetItemFeatures(TFeatureSource featureSource, TItem item);
    }
}
