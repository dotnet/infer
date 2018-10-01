// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// A mapping used by an evaluator of recommenders.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TRating">The type of a rating.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
    /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
    public class RecommenderEvaluatorMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues> :
        IRecommenderEvaluatorMapping<TInstanceSource, TUser, TItem, TRating>
    {
        #region Fields, constructors, properties

        /// <summary>
        /// Wrapped recommender mapping.
        /// </summary>
        private readonly IRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues> mapping;

        /// <summary>
        /// Last seen instance source.
        /// </summary>
        private TInstanceSource lastInstanceSource;

        /// <summary>
        /// Maps a user-item pair to a rating.
        /// </summary>
        private Dictionary<Tuple<TUser, TItem>, TRating> ratings;

        /// <summary>
        /// Maps a user to the list of items they have rated.
        /// </summary>
        private Dictionary<TUser, List<TItem>> itemsRatedByUser;

        /// <summary>
        /// Maps an item to the list of users it has been rated by.
        /// </summary>
        private Dictionary<TItem, List<TUser>> usersWhoRatedItem;

        /// <summary>
        /// Initializes a new instance of the 
        /// <see cref="RecommenderEvaluatorMapping{TInstanceSource,TInstance,TUser,TItem,TRating,TFeatureSource,TFeatureValues}"/>
        /// class.
        /// </summary>
        /// <param name="mapping">A recommender mapping.</param>
        public RecommenderEvaluatorMapping(
            IRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues> mapping)
        {
            this.mapping = mapping;
        }

        #endregion

        #region IRecommenderEvaluatorMapping implementation

        /// <summary>
        /// Gets the list of users from a given instance source.
        /// </summary>
        /// <param name="instanceSource">The instance source to get the list of users from.</param>
        /// <returns>The list of users.</returns>
        public IEnumerable<TUser> GetUsers(TInstanceSource instanceSource)
        {
            this.UpdateInternalRepresentation(instanceSource);
            return this.itemsRatedByUser.Keys;
        }

        /// <summary>
        /// Gets the list of items from a given instance source.
        /// </summary>
        /// <param name="instanceSource">The instance source to get the list of items from.</param>
        /// <returns>The list of items.</returns>
        public IEnumerable<TItem> GetItems(TInstanceSource instanceSource)
        {
            this.UpdateInternalRepresentation(instanceSource);
            return this.usersWhoRatedItem.Keys;
        }

        /// <summary>
        /// Gets the list of users who rated a particular item.
        /// </summary>
        /// <param name="instanceSource">The instance source to get the list of users from.</param>
        /// <param name="item">The item rated by the users of interest.</param>
        /// <returns>The list of users who rated the item.</returns>
        public IEnumerable<TUser> GetUsersWhoRatedItem(TInstanceSource instanceSource, TItem item)
        {
            this.UpdateInternalRepresentation(instanceSource);
            return this.usersWhoRatedItem[item].AsReadOnly();
        }

        /// <summary>
        /// Gets the list of items rated by a particular user.
        /// </summary>
        /// <param name="instanceSource">The instance source to get the list of items from.</param>
        /// <param name="user">The user who rated the items of interest.</param>
        /// <returns>The list of items rated by the user.</returns>
        public IEnumerable<TItem> GetItemsRatedByUser(TInstanceSource instanceSource, TUser user)
        {
            this.UpdateInternalRepresentation(instanceSource);
            return this.itemsRatedByUser[user].AsReadOnly();
        }

        /// <summary>
        /// Gets the rating given by a user to an item.
        /// </summary>
        /// <param name="instanceSource">The instance source to get the rating from.</param>
        /// <param name="user">The user.</param>
        /// <param name="item">The item.</param>
        /// <returns>The rating.</returns>
        public TRating GetRating(TInstanceSource instanceSource, TUser user, TItem item)
        {
            this.UpdateInternalRepresentation(instanceSource);
            return this.ratings[Tuple.Create(user, item)];
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Updates the state of the mapping.
        /// </summary>
        /// <param name="instanceSource">The instance source used to update the current state.</param>
        private void UpdateInternalRepresentation(TInstanceSource instanceSource)
        {
            if (ReferenceEquals(this.lastInstanceSource, instanceSource))
            {
                // The current state is up to date
                return;
            }

            this.lastInstanceSource = instanceSource;

            this.usersWhoRatedItem = new Dictionary<TItem, List<TUser>>();
            this.itemsRatedByUser = new Dictionary<TUser, List<TItem>>();
            this.ratings = new Dictionary<Tuple<TUser, TItem>, TRating>();

            var instances = this.mapping.GetInstances(instanceSource);
            foreach (var instance in instances)
            {
                var user = this.mapping.GetUser(instanceSource, instance);
                var item = this.mapping.GetItem(instanceSource, instance);
                var rating = this.mapping.GetRating(instanceSource, instance);

                // Update the mapping from a user-item pair to a rating
                var ratingsKey = Tuple.Create(user, item);
                if (this.ratings.ContainsKey(ratingsKey))
                {
                    throw new NotSupportedException("Having multiple ratings for the same user/item pair is not supported.");
                }

                this.ratings.Add(ratingsKey, rating);

                // Update the mapping from a user to the list of items they have rated
                List<TItem> items;
                if (!this.itemsRatedByUser.TryGetValue(user, out items))
                {
                    items = new List<TItem>();
                    this.itemsRatedByUser.Add(user, items);
                }

                items.Add(item);

                // Update the mapping from an item to the list of users it has been rated by
                List<TUser> users;
                if (!this.usersWhoRatedItem.TryGetValue(item, out users))
                {
                    users = new List<TUser>();
                    this.usersWhoRatedItem.Add(item, users);
                }

                users.Add(user);
            }
        }

        #endregion
    }
}