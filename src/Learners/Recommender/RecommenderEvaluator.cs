// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Evaluates a recommender system.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TGroundTruthRating">The type of a rating in a test dataset.</typeparam>
    /// <typeparam name="TPredictedRating">The type of a rating predicted by a recommender system.</typeparam>
    /// <typeparam name="TPredictedRatingDist">The type of a distribution over ratings predicted by a recommender system.</typeparam>
    public class RecommenderEvaluator<TInstanceSource, TUser, TItem, TGroundTruthRating, TPredictedRating, TPredictedRatingDist>
    {
        /// <summary>
        /// The mapping used for accessing data.
        /// </summary>
        private readonly IRecommenderEvaluatorMapping<TInstanceSource, TUser, TItem, TGroundTruthRating> mapping;

        /// <summary>
        /// Initializes a new instance of the
        /// <see cref="RecommenderEvaluator{TInstanceSource, TUser, TItem, TGroundTruthRating, TPredictedRating, TPredictedRatingDist}"/> class.
        /// </summary>
        /// <param name="mapping">The mapping used for accessing data.</param>
        /// <exception cref="ArgumentNullException">Thrown if the given mapping is null.</exception>
        public RecommenderEvaluator(IRecommenderEvaluatorMapping<TInstanceSource, TUser, TItem, TGroundTruthRating> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            this.mapping = mapping;
        }

        #region Predictions restricted to a particular dataset

        /// <summary>
        /// For each user in a given instance source recommends items from the set of items rated by the user.
        /// </summary>
        /// <typeparam name="TFeatureSource">The type of a feature source used by the recommendation engine.</typeparam>
        /// <param name="recommender">The recommendation engine.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="maxRecommendedItemCount">Maximum number of items to recommend to a user.</param>
        /// <param name="minRecommendationPoolSize">
        /// If a user has less than <paramref name="minRecommendationPoolSize"/> possible items to recommend,
        /// it will be skipped.
        /// </param>
        /// <param name="featureSource">The source of features.</param>
        /// <returns>The list of recommended items for every user in <paramref name="instanceSource"/>.</returns>
        public IDictionary<TUser, IEnumerable<TItem>> RecommendRatedItems<TFeatureSource>(
            IRecommender<TInstanceSource, TUser, TItem, TPredictedRating, TPredictedRatingDist, TFeatureSource> recommender,
            TInstanceSource instanceSource,
            int maxRecommendedItemCount,
            int minRecommendationPoolSize,
            TFeatureSource featureSource = default(TFeatureSource))
        {
            if (recommender == null)
            {
                throw new ArgumentNullException(nameof(recommender));
            }

            if (maxRecommendedItemCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxRecommendedItemCount), "The number of items to recommend should be positive.");
            }

            if (minRecommendationPoolSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(minRecommendationPoolSize), "The minimum size of the recommendation pool should be positive.");
            }

            IDictionary<TUser, IEnumerable<TItem>> recommendations = new Dictionary<TUser, IEnumerable<TItem>>();
            foreach (TUser user in this.mapping.GetUsers(instanceSource))
            {
                IEnumerable<TItem> itemSubset = this.mapping.GetItemsRatedByUser(instanceSource, user);
                if (itemSubset.Count() >= minRecommendationPoolSize)
                {
                    recommender.ItemSubset = itemSubset;
                    IEnumerable<TItem> recommendedItems = recommender.Recommend(user, maxRecommendedItemCount, featureSource);
                    recommendations.Add(user, recommendedItems);
                }
            }

            return recommendations;
        }

        /// <summary>
        /// Finds related users for every user in a instance source.
        /// The subset of users who will be returned as related for a particular user is restricted:
        /// it is guaranteed that all the related users have rated at least <paramref name="minCommonRatingCount"/> items in common with the query user
        /// in the dataset.
        /// </summary>
        /// <typeparam name="TFeatureSource">The type of a feature source used by the recommendation engine.</typeparam>
        /// <param name="recommender">The recommendation engine.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="maxRelatedUserCount">Maximum number of related users to return.</param>
        /// <param name="minCommonRatingCount">Minimum number of items that the query user and the related user should have rated in common.</param>
        /// <param name="minRelatedUserPoolSize">
        /// If a user has less than <paramref name="minRelatedUserPoolSize"/> possible related users,
        /// it will be skipped.
        /// </param>
        /// <param name="featureSource">The source of features.</param>
        /// <returns>The list of related users for every user in <paramref name="instanceSource"/>.</returns>
        public IDictionary<TUser, IEnumerable<TUser>> FindRelatedUsersWhoRatedSameItems<TFeatureSource>(
            IRecommender<TInstanceSource, TUser, TItem, TPredictedRating, TPredictedRatingDist, TFeatureSource> recommender,
            TInstanceSource instanceSource,
            int maxRelatedUserCount,
            int minCommonRatingCount,
            int minRelatedUserPoolSize,
            TFeatureSource featureSource = default(TFeatureSource))
        {
            if (recommender == null)
            {
                throw new ArgumentNullException(nameof(recommender));
            }

            if (maxRelatedUserCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxRelatedUserCount), "The maximum number of related users should be positive.");
            }

            if (minCommonRatingCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(minCommonRatingCount), "The minimum number of common ratings should be positive.");
            }

            if (minRelatedUserPoolSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(minRelatedUserPoolSize), "The minimum size of the related users pool should be positive.");
            }

            return FindRelatedEntitiesWhichRatedTheSame(
                this.mapping.GetUsers(instanceSource),
                user => this.mapping.GetUsersWhoRatedSameItems(instanceSource, user),
                (user, userSubset) =>
                {
                    recommender.UserSubset = userSubset;
                    return recommender.GetRelatedUsers(user, maxRelatedUserCount, featureSource);
                },
                minCommonRatingCount,
                minRelatedUserPoolSize);
        }

        /// <summary>
        /// Finds related items for every item in a given instance source.
        /// The subset of items which will be returned as related for a particular item is restricted:
        /// it is guaranteed that all the related items have been rated by at least <paramref name="minCommonRatingCount"/> users in common with the query item
        /// in the dataset.
        /// </summary>
        /// <typeparam name="TFeatureSource">The type of a feature source used by the recommendation engine.</typeparam>
        /// <param name="recommender">The recommendation engine.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="maxRelatedItemCount">Maximum number of related items to return.</param>
        /// <param name="minCommonRatingCount">Minimum number of users that the query item and the related item should have been rated by in common.</param>
        /// <param name="minRelatedItemPoolSize">
        /// If an item has less than <paramref name="minRelatedItemPoolSize"/> possible related items,
        /// it will be skipped.
        /// </param>
        /// <param name="featureSource">The source of features.</param>
        /// <returns>The list of related items for every item in <paramref name="instanceSource"/>.</returns>
        public IDictionary<TItem, IEnumerable<TItem>> FindRelatedItemsRatedBySameUsers<TFeatureSource>(
            IRecommender<TInstanceSource, TUser, TItem, TPredictedRating, TPredictedRatingDist, TFeatureSource> recommender,
            TInstanceSource instanceSource,
            int maxRelatedItemCount,
            int minCommonRatingCount,
            int minRelatedItemPoolSize,
            TFeatureSource featureSource = default(TFeatureSource))
        {
            if (recommender == null)
            {
                throw new ArgumentNullException(nameof(recommender));
            }

            if (maxRelatedItemCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxRelatedItemCount), "The maximum number of related items should be positive.");
            }

            if (minCommonRatingCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(minCommonRatingCount), "The minimum number of common ratings should be positive.");
            }

            if (minRelatedItemPoolSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(minRelatedItemPoolSize), "The minimum size of the related items pool should be positive.");
            }

            return FindRelatedEntitiesWhichRatedTheSame(
                this.mapping.GetItems(instanceSource),
                item => this.mapping.GetItemsRatedBySameUsers(instanceSource, item),
                (item, itemSubset) =>
                {
                    recommender.ItemSubset = itemSubset;
                    return recommender.GetRelatedItems(item, maxRelatedItemCount, featureSource);
                },
                minCommonRatingCount,
                minRelatedItemPoolSize);
        }

        #endregion

        #region Rating prediction metrics (data domain)

        /// <summary>
        /// Computes the average of a given data domain rating prediction metric by iterating firstly per-item and then per-user.
        /// </summary>
        /// <param name="instanceSource">The source of the instances providing the ground truth.</param>
        /// <param name="predictions">A sparse users-by-items matrix of predicted rating distributions.</param>
        /// <param name="metric">The data domain rating prediction metric to average.</param>
        /// <param name="aggregationMethod">A method specifying how metrics are aggregated over the whole dataset.</param>
        /// <returns>The computed average of the given rating prediction metric.</returns>
        public double RatingPredictionMetric(
            TInstanceSource instanceSource,
            IDictionary<TUser, IDictionary<TItem, TPredictedRating>> predictions,
            Func<TGroundTruthRating, TPredictedRating, double> metric,
            RecommenderMetricAggregationMethod aggregationMethod = RecommenderMetricAggregationMethod.Default)
        {
            return this.RatingPredictionMetric<TPredictedRating>(instanceSource, predictions, metric, aggregationMethod);
        }

        /// <summary>
        /// Computes the average of a given data domain rating prediction metric by iterating firstly per-item and then per-user.
        /// </summary>
        /// <param name="instanceSource">The source of the instances providing the ground truth.</param>
        /// <param name="predictions">A sparse users-by-items matrix of predicted rating distributions.</param>
        /// <param name="metric">The data domain rating prediction metric to average.</param>
        /// <param name="aggregationMethod">A method specifying how metrics are aggregated over the whole dataset.</param>
        /// <returns>The computed average of the given rating prediction metric.</returns>
        public double RatingPredictionMetric(
            TInstanceSource instanceSource,
            IDictionary<TUser, IDictionary<TItem, TPredictedRatingDist>> predictions,
            Func<TGroundTruthRating, TPredictedRatingDist, double> metric,
            RecommenderMetricAggregationMethod aggregationMethod = RecommenderMetricAggregationMethod.Default)
        {
            return this.RatingPredictionMetric<TPredictedRatingDist>(instanceSource, predictions, metric, aggregationMethod);
        }

        #endregion

        #region Recommendation metrics

        /// <summary>
        /// Computes the average across users of a given ranking metric for recommendation. Uses ratings to compute gains.
        /// </summary>
        /// <param name="instanceSource">The source of the instances providing the ground truth.</param>
        /// <param name="predictions">A mapping from a user to a list of recommended items.</param>
        /// <param name="metric">The ranking metric.</param>
        /// <param name="ratingConverter">The function used to convert ratings to gains. <see cref="Convert.ToDouble(object)"/> is used by default.</param>
        /// <returns>The computed average of the given ranking metric for recommendation.</returns>
        public double ItemRecommendationMetric(
            TInstanceSource instanceSource,
            IDictionary<TUser, IEnumerable<TItem>> predictions,
            Func<IEnumerable<double>, double> metric,
            Func<TGroundTruthRating, double> ratingConverter = null)
        {
            return this.ItemRecommendationMetric(instanceSource, predictions, (p, b) => metric(p), ratingConverter);
        }

        /// <summary>
        /// Computes the average across users of a given ranking metric for recommendation. Uses ratings to compute gains.
        /// </summary>
        /// <param name="instanceSource">The source of the instances providing the ground truth.</param>
        /// <param name="predictions">A mapping from a user to a list of recommended items.</param>
        /// <param name="metric">The ranking metric.</param>
        /// <param name="ratingConverter">The function used to convert ratings to gains. <see cref="Convert.ToDouble(object)"/> is used by default.</param>
        /// <returns>The computed average of the given ranking metric for recommendation.</returns>
        public double ItemRecommendationMetric(
            TInstanceSource instanceSource,
            IDictionary<TUser, IEnumerable<TItem>> predictions,
            Func<IEnumerable<double>, IEnumerable<double>, double> metric,
            Func<TGroundTruthRating, double> ratingConverter = null)
        {
            ratingConverter = ratingConverter ?? DefaultRatingConverter;

            double perUserSum = 0;
            int userCount = 0;
            foreach (var userRecommendations in predictions)
            {
                TUser user = userRecommendations.Key;
                List<TItem> recommendations = userRecommendations.Value.ToList();

                var recommendationsSet = new HashSet<TItem>(recommendations);
                if (recommendationsSet.Count < recommendations.Count)
                {
                    throw new ArgumentException("One of the recomendation lists contains the same item multiple times.");
                }

                IEnumerable<TItem> itemsRatedByUser = this.mapping.GetItemsRatedByUser(instanceSource, user);
                IEnumerable<double> allRatings = itemsRatedByUser.Select(item => ratingConverter(this.mapping.GetRating(instanceSource, user, item)));
                List<double> bestOrderedGains = Util.GetMaxKValues(allRatings, recommendations.Count).ToList();
                IEnumerable<double> orderedGains = recommendations.Select(item => ratingConverter(this.mapping.GetRating(instanceSource, user, item)));

                if (bestOrderedGains.Count < recommendations.Count)
                {
                    // Number of items we can possibly recommend is less than the number of recommended items given
                    throw new ArgumentException("The given recommendations were generated from a different dataset.");
                }

                perUserSum += metric(orderedGains, bestOrderedGains);
                ++userCount;
            }

            if (userCount == 0)
            {
                throw new ArgumentException("The given recommendations are empty.");
            }

            return perUserSum / userCount;
        }

        #endregion

        #region Related users/items metric

        /// <summary>
        /// Computes the average of a given ranking metric over a set of ordered sequences of users.
        /// Gain for ranking is defined as the rating similarity between the users in the list and the query user.
        /// </summary>
        /// <param name="instanceSource">The source of the instances providing the ground truth.</param>
        /// <param name="predictions">A mapping from a user to a list of predicted related users.</param>
        /// <param name="minCommonRatingCount">
        /// The minimum number of common ratings two users must have in order to be considered for evaluation.
        /// </param>
        /// <param name="rankingMetric">The ranking metric.</param>
        /// <param name="ratingSimilarityFunc">A method which computes the similarity between two rating vectors.</param>
        /// <param name="ratingConverter">The function applied to ratings before computing similarity. <see cref="Convert.ToDouble(object)"/> is used by default.</param>
        /// <returns>The computed average of the given ranking metric.</returns>
        public double RelatedUsersMetric(
            TInstanceSource instanceSource,
            IDictionary<TUser, IEnumerable<TUser>> predictions,
            int minCommonRatingCount,
            Func<IEnumerable<double>, IEnumerable<double>, double> rankingMetric,
            Func<Vector, Vector, double> ratingSimilarityFunc,
            Func<TGroundTruthRating, double> ratingConverter = null)
        {
            ratingConverter = ratingConverter ?? DefaultRatingConverter;

            return RelatedEntitiesMetric(
                predictions,
                minCommonRatingCount,
                (u, i) => ratingConverter(this.mapping.GetRating(instanceSource, u, i)),
                u => this.mapping.GetUsersWhoRatedSameItems(instanceSource, u),
                rankingMetric,
                ratingSimilarityFunc);
        }

        /// <summary>
        /// Computes the average of a given ranking metric over a set of ordered sequences of users.
        /// Gain for ranking is defined as the rating similarity between the users in the list and the query user.
        /// </summary>
        /// <param name="instanceSource">The source of the instances providing the ground truth.</param>
        /// <param name="predictions">A mapping from a user to a list of predicted related users.</param>
        /// <param name="minCommonRatingCount">
        /// The minimum number of common ratings two users must have in order to be considered for evaluation.
        /// </param>
        /// <param name="rankingMetric">The ranking metric.</param>
        /// <param name="ratingSimilarityFunc">A method which computes the similarity between two rating vectors.</param>
        /// <param name="ratingConverter">The function applied to ratings before computing similarity. <see cref="Convert.ToDouble(object)"/> is used by default.</param>
        /// <returns>The computed average of the given ranking metric.</returns>
        public double RelatedUsersMetric(
            TInstanceSource instanceSource,
            IDictionary<TUser, IEnumerable<TUser>> predictions,
            int minCommonRatingCount,
            Func<IEnumerable<double>, double> rankingMetric,
            Func<Vector, Vector, double> ratingSimilarityFunc,
            Func<TGroundTruthRating, double> ratingConverter = null)
        {
            return this.RelatedUsersMetric(
                instanceSource, predictions, minCommonRatingCount, (l1, l2) => rankingMetric(l1), ratingSimilarityFunc, ratingConverter);
        }

        /// <summary>
        /// Computes the average of a given ranking metric over a set of ordered sequences of items.
        /// Gain for ranking is defined as the rating similarity between the items in the list and the query item.
        /// </summary>
        /// <param name="instanceSource">The source of the instances providing the ground truth.</param>
        /// <param name="predictions">A mapping from an item to a list of predicted related items.</param>
        /// <param name="minCommonRatingCount">
        /// The minimum number of common ratings two items must have in order to be considered for evaluation.
        /// </param>
        /// <param name="rankingMetric">The ranking metric.</param>
        /// <param name="ratingSimilarityFunc">A method which computes the similarity between two rating vectors.</param>
        /// <param name="ratingConverter">The function applied to ratings before computing similarity. <see cref="Convert.ToDouble(object)"/> is used by default.</param>
        /// <returns>The computed average of the given ranking metric.</returns>
        public double RelatedItemsMetric(
            TInstanceSource instanceSource,
            IDictionary<TItem, IEnumerable<TItem>> predictions,
            int minCommonRatingCount,
            Func<IEnumerable<double>, IEnumerable<double>, double> rankingMetric,
            Func<Vector, Vector, double> ratingSimilarityFunc,
            Func<TGroundTruthRating, double> ratingConverter = null)
        {
            ratingConverter = ratingConverter ?? DefaultRatingConverter;

            return RelatedEntitiesMetric(
                predictions,
                minCommonRatingCount,
                (i, u) => ratingConverter(this.mapping.GetRating(instanceSource, u, i)),
                i => this.mapping.GetItemsRatedBySameUsers(instanceSource, i),
                rankingMetric,
                ratingSimilarityFunc);
        }

        /// <summary>
        /// Computes the average of a given ranking metric over a set of ordered sequences of items.
        /// Gain for ranking is defined as the rating similarity between the items in the list and the query item.
        /// </summary>
        /// <param name="instanceSource">The source of the instances providing the ground truth.</param>
        /// <param name="predictions">A mapping from an item to a list of predicted related items.</param>
        /// <param name="minCommonRatingCount">
        /// The minimum number of common ratings two items must have in order to be considered for evaluation.
        /// </param>
        /// <param name="rankingMetric">The ranking metric.</param>
        /// <param name="ratingSimilarityFunc">A method which computes the similarity between two rating vectors.</param>
        /// <param name="ratingConverter">The function applied to ratings before computing similarity. <see cref="Convert.ToDouble(object)"/> is used by default.</param>
        /// <returns>The computed average of the given ranking metric.</returns>
        public double RelatedItemsMetric(
            TInstanceSource instanceSource,
            IDictionary<TItem, IEnumerable<TItem>> predictions,
            int minCommonRatingCount,
            Func<IEnumerable<double>, double> rankingMetric,
            Func<Vector, Vector, double> ratingSimilarityFunc,
            Func<TGroundTruthRating, double> ratingConverter = null)
        {
            return this.RelatedItemsMetric(
                instanceSource, predictions, minCommonRatingCount, (l1, l2) => rankingMetric(l1), ratingSimilarityFunc, ratingConverter);
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Computes the average of a given ranking metric over a set of ordered sequences of entities.
        /// Gain for ranking is defined as the rating similarity between the entities in the list and the query entity.
        /// </summary>
        /// <typeparam name="TEntity">The type of an entity for which related ones were predicted.</typeparam>
        /// <typeparam name="TRatedEntity">
        /// The type of a rated entity 
        /// (the opposite of <typeparamref name="TEntity"/>).
        /// </typeparam>
        /// <param name="predictions">A mapping from an entity to a list of predicted related entities.</param>
        /// <param name="minCommonRatingCount">
        /// The minimum number of common ratings two entities must have in order to be considered for evaluation.
        /// </param>
        /// <param name="ratingSelector">A method which selects the rating of a given pair of entities.</param>
        /// <param name="entityToEntitiesWhichRatedTheSame">
        /// A mapping from an entity to a set of entities which have rated the same entities as the key one.
        /// </param>
        /// <param name="rankingMetric">The ranking metric.</param>
        /// <param name="ratingSimilarityFunc">A method which computes the similarity between two rating vectors.</param>
        /// <returns>The computed average of the given ranking metric.</returns>
        private static double RelatedEntitiesMetric<TEntity, TRatedEntity>(
            IDictionary<TEntity, IEnumerable<TEntity>> predictions,
            int minCommonRatingCount,
            Func<TEntity, TRatedEntity, double> ratingSelector,
            Func<TEntity, Dictionary<TEntity, List<TRatedEntity>>> entityToEntitiesWhichRatedTheSame,
            Func<IEnumerable<double>, IEnumerable<double>, double> rankingMetric,
            Func<Vector, Vector, double> ratingSimilarityFunc)
        {
            double sum = 0;
            int count = 0;
            foreach (var entityWithRelatedEntities in predictions)
            {
                TEntity queryEntity = entityWithRelatedEntities.Key;
                List<TEntity> relatedEntities = entityWithRelatedEntities.Value.ToList();
                var relatedEntitySet = new HashSet<TEntity>(relatedEntities);

                if (relatedEntitySet.Count < relatedEntities.Count)
                {
                    throw new ArgumentException("One of the prediction lists contains the same entity multiple times.");
                }

                if (relatedEntitySet.Contains(queryEntity))
                {
                    throw new ArgumentException("One of the prediction lists contains the query entity.");
                }

                // Find entities which have at least 1 rated entity in common
                var entityToRatedInCommon = entityToEntitiesWhichRatedTheSame(queryEntity);

                // Compute rating similarities with the given entity for every other entity with common ratings
                var similarities = new Dictionary<TEntity, double>();
                foreach (var otherEntityWithRatedInCommon in entityToRatedInCommon)
                {
                    TEntity otherEntity = otherEntityWithRatedInCommon.Key;
                    List<TRatedEntity> ratedInCommon = otherEntityWithRatedInCommon.Value;

                    if (ratedInCommon.Count < minCommonRatingCount)
                    {
                        // Too few common ratings, otherEntity similarity to entity can't be computed
                        continue;
                    }

                    Vector entityCommonRatings = Vector.FromArray(ratedInCommon.Select(rb => ratingSelector(queryEntity, rb)).ToArray());
                    Vector otherEntityCommonRatings = Vector.FromArray(ratedInCommon.Select(rb => ratingSelector(otherEntity, rb)).ToArray());
                    double similarity = ratingSimilarityFunc(entityCommonRatings, otherEntityCommonRatings);

                    similarities.Add(otherEntity, similarity);
                }

                if (relatedEntities.Any(e => !similarities.ContainsKey(e)))
                {
                    throw new ArgumentException("One or more related entities have too few ratings for the evaluation.");
                }

                IEnumerable<double> orderedGains = relatedEntities.Select(i => similarities[i]);
                List<double> bestOrderedGains = Util.GetMaxKValues(similarities.Values, relatedEntities.Count).ToList();
                if (bestOrderedGains.Count < relatedEntities.Count)
                {
                    // Number of entities we can possibly relate is less than the number of related entities given
                    throw new ArgumentException("The given predictions were generated from a different dataset.");
                }

                sum += rankingMetric(orderedGains, bestOrderedGains);
                count += 1;
            }

            if (count == 0)
            {
                throw new ArgumentException("The given predictions are empty.");
            }

            return sum / count;
        }

        /// <summary>
        /// Converts a ground truth rating to a double.
        /// </summary>
        /// <param name="rating">The ground truth rating to convert.</param>
        /// <returns>A double value representing the ground truth rating.</returns>
        private static double DefaultRatingConverter(TGroundTruthRating rating)
        {
            return Convert.ToDouble(rating);
        }

        /// <summary>
        /// Provides common implementation for
        /// <see cref="FindRelatedUsersWhoRatedSameItems{TFeatureSource}"/> and <see cref="FindRelatedItemsRatedBySameUsers{TFeatureSource}"/>.
        /// </summary>
        /// <typeparam name="TEntity">The type of an entity.</typeparam>
        /// <typeparam name="TRatedEntity">The type of a rated entity.</typeparam>
        /// <param name="entities">
        /// The entities in the dataset.
        /// </param>
        /// <param name="entitiesWhichRatedTheSameSelector">
        /// Function which returns a list of entities which have rated at least one entity in common with a given one.
        /// </param>
        /// <param name="relatedEntitySelector">
        /// Function which returns a list of related entities to a given entity by invoking recommendation engine with a
        /// specified query entity and an entity subset.
        /// </param>
        /// <param name="minCommonRatingCount">
        /// Minimum number of entities that the query entity and the related entity should have rated in common.
        /// </param>
        /// <param name="minRelatedEntityPoolSize">
        /// If an entity has less than <paramref name="minRelatedEntityPoolSize"/> possible related entities,
        /// it will be skipped.
        /// </param>
        /// <returns>
        /// The list of related entities for every entity in <paramref name="entities"/>.
        /// </returns>
        private static IDictionary<TEntity, IEnumerable<TEntity>> FindRelatedEntitiesWhichRatedTheSame<TEntity, TRatedEntity>(
            IEnumerable<TEntity> entities,
            Func<TEntity, Dictionary<TEntity, List<TRatedEntity>>> entitiesWhichRatedTheSameSelector,
            Func<TEntity, IEnumerable<TEntity>, IEnumerable<TEntity>> relatedEntitySelector,
            int minCommonRatingCount,
            int minRelatedEntityPoolSize)
        {
            var result = new Dictionary<TEntity, IEnumerable<TEntity>>();
            foreach (TEntity entity in entities)
            {
                Dictionary<TEntity, List<TRatedEntity>> entitiesWhoRatedTheSame = entitiesWhichRatedTheSameSelector(entity);

                var entitySubset = new List<TEntity>();
                foreach (var otherEntityWithCommonRatedEntities in entitiesWhoRatedTheSame)
                {
                    if (otherEntityWithCommonRatedEntities.Value.Count >= minCommonRatingCount)
                    {
                        entitySubset.Add(otherEntityWithCommonRatedEntities.Key);
                    }
                }

                if (entitySubset.Count >= minRelatedEntityPoolSize)
                {
                    IEnumerable<TEntity> relatedEntities = relatedEntitySelector(entity, entitySubset);
                    result.Add(entity, relatedEntities);
                }
            }

            return result;
        }

        /// <summary>
        /// Computes the average of a given rating prediction metric by iterating over 
        /// <paramref name="predictions"/> and using the aggregation method given in <paramref name="aggregationMethod"/>.
        /// </summary>
        /// <typeparam name="TPrediction">
        /// The type of predictions 
        /// (used to support both certain and uncertain predictions).
        /// </typeparam>
        /// <param name="instanceSource">The source of the instances providing the ground truth.</param>
        /// <param name="predictions">A sparse users-by-items matrix of predicted rating distributions.</param>
        /// <param name="metric">The rating prediction metric to average.</param>
        /// <param name="aggregationMethod">A method specifying how metrics are aggregated over the whole dataset.</param>
        /// <returns>The computed average of the given rating prediction metric.</returns>
        private double RatingPredictionMetric<TPrediction>(
            TInstanceSource instanceSource,
            IDictionary<TUser, IDictionary<TItem, TPrediction>> predictions,
            Func<TGroundTruthRating, TPrediction, double> metric,
            RecommenderMetricAggregationMethod aggregationMethod)
        {
            double totalSum = 0;
            int totalCount = 0;
            foreach (var userWithPredictionList in predictions)
            {
                double perUserSum = 0;
                int perUserCount = 0;
                foreach (var itemPrediction in userWithPredictionList.Value)
                {
                    TPrediction prediction = itemPrediction.Value;
                    TGroundTruthRating groundTruth = this.mapping.GetRating(instanceSource, userWithPredictionList.Key, itemPrediction.Key);
                    double metricValue = metric(groundTruth, prediction);
                    perUserSum += metricValue;
                    ++perUserCount;
                }

                switch (aggregationMethod)
                {
                    case RecommenderMetricAggregationMethod.PerUserFirst:
                        if (perUserCount == 0)
                        {
                            throw new ArgumentException("One of the users in the given predictions has no associated predictions.");
                        }

                        totalSum += perUserSum / perUserCount;
                        ++totalCount;
                        break;
                    case RecommenderMetricAggregationMethod.Default:
                        totalSum += perUserSum;
                        totalCount += perUserCount;
                        break;
                    default:
                        Debug.Fail("Metric aggregation method is not supported.");
                        break;
                }
            }

            if (totalCount == 0)
            {
                throw new ArgumentException("The given predictions are empty.");
            }

            return totalSum / totalCount;
        }

        #endregion
    }
}