// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Interface to a recommendation algorithm.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TRating">The type of a rating.</typeparam>
    /// <typeparam name="TRatingDistribution">The type of a distribution over ratings.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a source of features.</typeparam>
    /// <remarks>
    /// Intended usage:
    /// <para>
    /// An instance refers to a single user-item-rating triple. An instance source provides all instances of interest.
    /// A feature source provides the features for the users and the items.
    /// </para>
    /// </remarks>
    public interface IRecommender<in TInstanceSource, TUser, TItem, TRating, TRatingDistribution, in TFeatureSource> : ILearner
    {
        /// <summary>
        /// Gets the capabilities of the recommender.
        /// </summary>
        new IRecommenderCapabilities Capabilities { get; }

        /// <summary>
        /// Gets or sets the subset of the users used for related user prediction.
        /// </summary>
        IEnumerable<TUser> UserSubset { get; set; }

        /// <summary>
        /// Gets or sets the subset of the items used for related item prediction and recommendation.
        /// </summary>
        IEnumerable<TItem> ItemSubset { get; set; }

        /// <summary>
        /// Trains the recommender on the specified instances.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <param name="featureSource">The source of features for the specified instances.</param>
        void Train(TInstanceSource instanceSource, TFeatureSource featureSource = default(TFeatureSource));

        /// <summary>
        /// Predicts a rating for a specified user and item.
        /// </summary>
        /// <param name="user">The user.</param>
        /// <param name="item">The item.</param>
        /// <param name="featureSource">The source of features for the specified user and item.</param>
        /// <returns>The predicted rating.</returns>
        TRating Predict(TUser user, TItem item, TFeatureSource featureSource = default(TFeatureSource));

        /// <summary>
        /// Predicts ratings for the specified instances.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <param name="featureSource">The source of features for the specified instances.</param>
        /// <returns>The predicted ratings.</returns>
        IDictionary<TUser, IDictionary<TItem, TRating>> Predict(
            TInstanceSource instanceSource, TFeatureSource featureSource = default(TFeatureSource));

        /// <summary>
        /// Predicts the distribution of a rating for a specified user and item.
        /// </summary>
        /// <param name="user">The user.</param>
        /// <param name="item">The item.</param>
        /// <param name="featureSource">The source of features for the specified user and item.</param>
        /// <returns>The distribution of the rating.</returns>
        TRatingDistribution PredictDistribution(
            TUser user, TItem item, TFeatureSource featureSource = default(TFeatureSource));

        /// <summary>
        /// Predicts rating distributions for the specified instances.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <param name="featureSource">The source of the features for the specified instances.</param>
        /// <returns>The distributions of the predicted ratings.</returns>
        IDictionary<TUser, IDictionary<TItem, TRatingDistribution>> PredictDistribution(
            TInstanceSource instanceSource, TFeatureSource featureSource = default(TFeatureSource));

        /// <summary>
        /// Recommends items to a given user.
        /// </summary>
        /// <param name="user">The user to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend.</param>
        /// <param name="featureSource">The source of features for the specified user.</param>
        /// <returns>The list of recommended items.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        IEnumerable<TItem> Recommend(
            TUser user, int recommendationCount, TFeatureSource featureSource = default(TFeatureSource));

        /// <summary>
        /// Recommends items to a given list of users.
        /// </summary>
        /// <param name="users">The list of users to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend to a single user.</param>
        /// <param name="featureSource">The source of features for the specified users.</param>
        /// <returns>The list of recommended items for every user from <paramref name="users"/>.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        IDictionary<TUser, IEnumerable<TItem>> Recommend(
            IEnumerable<TUser> users, int recommendationCount, TFeatureSource featureSource = default(TFeatureSource));

        /// <summary>
        /// Recommend items with their rating distributions to a specified user.
        /// </summary>
        /// <param name="user">The user to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend.</param>
        /// <param name="featureSource">The source of features for the specified user.</param>
        /// <returns>The list of recommended items and their rating distributions.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        IEnumerable<Tuple<TItem, TRatingDistribution>> RecommendDistribution(
            TUser user, int recommendationCount, TFeatureSource featureSource = default(TFeatureSource));

        /// <summary>
        /// Recommends items with their rating distributions to a specified list of users.
        /// </summary>
        /// <param name="users">The list of users to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend to a single user.</param>
        /// <param name="featureSource">The source of features for the specified users.</param>
        /// <returns>The list of recommended items and their rating distributions for every user from <paramref name="users"/>.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        IDictionary<TUser, IEnumerable<Tuple<TItem, TRatingDistribution>>> RecommendDistribution(
            IEnumerable<TUser> users, int recommendationCount, TFeatureSource featureSource = default(TFeatureSource));

        /// <summary>
        /// Returns a list of users related to <paramref name="user"/>.
        /// </summary>
        /// <param name="user">The user to find related users for.</param>
        /// <param name="relatedUserCount">Maximum number of related users to return.</param>
        /// <param name="featureSource">The source of features for the specified user.</param>
        /// <returns>The list of related users.</returns>
        /// <remarks>Only users specified in <see cref="UserSubset"/> will be returned.</remarks>
        IEnumerable<TUser> GetRelatedUsers(
            TUser user, int relatedUserCount, TFeatureSource featureSource = default(TFeatureSource));

        /// <summary>
        /// Returns a list of related users to each user in <paramref name="users"/>.
        /// </summary>
        /// <param name="users">The list of users to find related users for.</param>
        /// <param name="relatedUserCount">Maximum number of related users to return for every user.</param>
        /// <param name="featureSource">The source of features for the specified users.</param>
        /// <returns>The list of related users for each user from <paramref name="users"/>.</returns>
        /// <remarks>Only users specified in <see cref="UserSubset"/> will be returned.</remarks>
        IDictionary<TUser, IEnumerable<TUser>> GetRelatedUsers(
            IEnumerable<TUser> users, int relatedUserCount, TFeatureSource featureSource = default(TFeatureSource));

        /// <summary>
        /// Returns a list of items related to <paramref name="item"/>.
        /// </summary>
        /// <param name="item">The item to find related items for.</param>
        /// <param name="relatedItemCount">Maximum number of related items to return.</param>
        /// <param name="featureSource">The source of features for the specified item.</param>
        /// <returns>The list of related items.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> will be returned.</remarks>
        IEnumerable<TItem> GetRelatedItems(
            TItem item, int relatedItemCount, TFeatureSource featureSource = default(TFeatureSource));

        /// <summary>
        /// Returns a list of related items to each item in <paramref name="items"/>.
        /// </summary>
        /// <param name="items">The list of items to find related items for.</param>
        /// <param name="relatedItemCount">Maximum number of related items to return for every item.</param>
        /// <param name="featureSource">The source of features for the specified items.</param>
        /// <returns>The list of related items for each item from <paramref name="items"/>.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> will be returned.</remarks>
        IDictionary<TItem, IEnumerable<TItem>> GetRelatedItems(
            IEnumerable<TItem> items, int relatedItemCount, TFeatureSource featureSource = default(TFeatureSource));
    }
}