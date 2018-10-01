// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System.Collections.Generic;

    /// <summary>
    /// A mapping used by the 
    /// <see cref="RecommenderEvaluator{TInstanceSource, TUser, TItem, TGroundTruthRating, TPredictedRating, TPredictedRatingDist}"/> class.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of an instance source.</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TRating">The type of a rating.</typeparam>
    public interface IRecommenderEvaluatorMapping<in TInstanceSource, TUser, TItem, out TRating>
    {
        /// <summary>
        /// Gets the list of users from a given instance source.
        /// </summary>
        /// <param name="instanceSource">The instance source to get the list of users from.</param>
        /// <returns>The list of users.</returns>
        IEnumerable<TUser> GetUsers(TInstanceSource instanceSource);
        
        /// <summary>
        /// Gets the list of items from a given instance source.
        /// </summary>
        /// <param name="instanceSource">The instance source to get the list of items from.</param>
        /// <returns>The list of items.</returns>
        IEnumerable<TItem> GetItems(TInstanceSource instanceSource);

        /// <summary>
        /// Gets the list of users who rated a particular item.
        /// </summary>
        /// <param name="instanceSource">The instance source to get the list of users from.</param>
        /// <param name="item">The item rated by the users of interest.</param>
        /// <returns>The list of users who rated the item.</returns>
        IEnumerable<TUser> GetUsersWhoRatedItem(TInstanceSource instanceSource, TItem item);

        /// <summary>
        /// Gets the list of items rated by a particular user.
        /// </summary>
        /// <param name="instanceSource">The instance source to get the list of items from.</param>
        /// <param name="user">The user who rated the items of interest.</param>
        /// <returns>The list of items rated by the user.</returns>
        IEnumerable<TItem> GetItemsRatedByUser(TInstanceSource instanceSource, TUser user);

        /// <summary>
        /// Gets the rating given by a user to an item.
        /// </summary>
        /// <param name="instanceSource">The instance source to get the rating from.</param>
        /// <param name="user">The user.</param>
        /// <param name="item">The item.</param>
        /// <returns>The rating.</returns>
        TRating GetRating(TInstanceSource instanceSource, TUser user, TItem item);
    }
}