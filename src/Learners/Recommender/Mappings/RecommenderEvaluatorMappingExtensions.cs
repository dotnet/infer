// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Extensions methods for the <see cref="IRecommenderEvaluatorMapping{TInstanceSource, TUser, TItem, TRating}"/> interface.
    /// </summary>
    public static class RecommenderEvaluatorMappingExtensions
    {
        /// <summary>
        /// Finds all the users who have rated the same item as <paramref name="queryUser"/>.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TRating">The type of a rating.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instanceSource">The instance source with items.</param>
        /// <param name="queryUser">The query user.</param>
        /// <returns>
        /// A dictionary which maps users from the <paramref name="instanceSource"/> to the lists of items
        /// which have been rated by both the user and <paramref name="queryUser"/>.
        /// Only users with at least one such item are returned.
        /// </returns>
        public static Dictionary<TUser, List<TItem>> GetUsersWhoRatedSameItems<TInstanceSource, TUser, TItem, TRating>(
            this IRecommenderEvaluatorMapping<TInstanceSource, TUser, TItem, TRating> mapping,
            TInstanceSource instanceSource,
            TUser queryUser)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }
            
            return GetEntitiesWithCommonRatedEntities(
                queryUser,
                u => mapping.GetItemsRatedByUser(instanceSource, u),
                i => mapping.GetUsersWhoRatedItem(instanceSource, i));
        }

        /// <summary>
        /// Finds all the items which have been rated by the same user as <paramref name="queryItem"/>.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TRating">The type of a rating.</typeparam>
        /// <param name="mapping">The mapping.</param>
        /// <param name="instanceSource">The instance source with items.</param>
        /// <param name="queryItem">The query item.</param>
        /// <returns>
        /// A dictionary which maps items from the <paramref name="instanceSource"/> to the lists of users
        /// who has rated both the item and <paramref name="queryItem"/>. Only items with at least one such user are returned.
        /// </returns>
        public static Dictionary<TItem, List<TUser>> GetItemsRatedBySameUsers<TInstanceSource, TUser, TItem, TRating>(
            this IRecommenderEvaluatorMapping<TInstanceSource, TUser, TItem, TRating> mapping,
            TInstanceSource instanceSource,
            TItem queryItem)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return GetEntitiesWithCommonRatedEntities(
                queryItem,
                i => mapping.GetUsersWhoRatedItem(instanceSource, i),
                u => mapping.GetItemsRatedByUser(instanceSource, u));
        }

        /// <summary>
        /// Finds all the entities which have rated the entities as <paramref name="queryEntity"/>.
        /// </summary>
        /// <typeparam name="TEntity">The type of an entity.</typeparam>
        /// <typeparam name="TRatedEntity">The type of a rated entity.</typeparam>
        /// <param name="queryEntity">The query entity.</param>
        /// <param name="entityToRatedMapping">The mapping from the entity to the entities it rated.</param>
        /// <param name="ratedToEntityMapping">The mapping from the rated entity to the entities who rated it.</param>
        /// <returns>
        /// A dictionary which maps entities to the lists of rated entities
        /// which has been rated by both the entity and <paramref name="queryEntity"/>.
        /// Only entities with at least one such entity are returned.
        /// </returns>
        private static Dictionary<TEntity, List<TRatedEntity>> GetEntitiesWithCommonRatedEntities<TEntity, TRatedEntity>(
            TEntity queryEntity,
            Func<TEntity, IEnumerable<TRatedEntity>> entityToRatedMapping,
            Func<TRatedEntity, IEnumerable<TEntity>> ratedToEntityMapping)
        {
            var result = new Dictionary<TEntity, List<TRatedEntity>>();
            foreach (TRatedEntity ratedEntity in entityToRatedMapping(queryEntity))
            {
                foreach (TEntity otherEntity in ratedToEntityMapping(ratedEntity))
                {
                    if (otherEntity.Equals(queryEntity))
                    {
                        continue;
                    }

                    List<TRatedEntity> commonRatedEntities;
                    if (!result.TryGetValue(otherEntity, out commonRatedEntities))
                    {
                        commonRatedEntities = new List<TRatedEntity>();
                        result.Add(otherEntity, commonRatedEntities);
                    }

                    commonRatedEntities.Add(ratedEntity);
                }
            }

            return result;
        }
    }
}
