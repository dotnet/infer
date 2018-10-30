// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    using RatingDistribution = System.Collections.Generic.IDictionary<int, double>;

    /// <summary>
    /// Represents a star rating recommender system which generates predictions purely by random guessing.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TDataRating">The type of a rating in data.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
    /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
    [SerializationVersion(1)]
    public class RandomStarRatingRecommender<TInstanceSource, TInstance, TUser, TItem, TDataRating, TFeatureSource, TFeatureValues> :
        IRecommender<TInstanceSource, TUser, TItem, int, RatingDistribution, TFeatureSource>
    {
        /// <summary>
        /// The capabilities of the recommender.
        /// </summary>
        private readonly RandomStarRatingRecommenderCapabilities capabilities;

        /// <summary>
        /// The mapping used to access the data.
        /// </summary>
        private readonly IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TDataRating, TFeatureSource, TFeatureValues> mapping;

        /// <summary>
        /// The subset of users to predict related users from.
        /// </summary>
        private TUser[] userSubset;

        /// <summary>
        /// The subset of items to predict related items and recommend items from.
        /// </summary>
        private TItem[] itemSubset;

        /// <summary>
        /// The information about ratings from the training set.
        /// </summary>
        private IStarRatingInfo<TDataRating> starRatingInfo;

        /// <summary>
        /// Initializes a new instance of the
        /// <see cref="RandomStarRatingRecommender{TInstanceSource,TInstance,TUser,TItem,TDataRating,TFeatureSource,TFeatureValues}"/> class.
        /// </summary>
        /// <param name="mapping">The mapping used to access the data.</param>
        /// <exception cref="ArgumentNullException">Thrown if one of the required parameters is null.</exception>
        public RandomStarRatingRecommender(
            IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TDataRating, TFeatureSource, TFeatureValues> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            this.capabilities = new RandomStarRatingRecommenderCapabilities();
            this.mapping = mapping;
        }

        #region Explicit ILearner implementation

        /// <summary>
        /// Gets the capabilities of the learner. These are any properties of the learner that
        /// are not captured by the type signature of the most specific learner interface below.
        /// </summary>
        ICapabilities ILearner.Capabilities
        {
            get { return this.capabilities; }
        }

        /// <summary>
        /// Gets or sets the settings (not supported).
        /// </summary>
        public ISettings Settings
        {
            get { return null; }
            set { throw new NotSupportedException("This recommender does not support settings."); }
        }

        #endregion

        #region IRecommender implementation

        /// <summary>
        /// Gets the capabilities of the recommender.
        /// </summary>
        public IRecommenderCapabilities Capabilities
        {
            get { return this.capabilities; }
        }

        /// <summary>
        /// Gets or sets the subset of the users used for related user prediction.
        /// </summary>
        public IEnumerable<TUser> UserSubset
        {
            get
            {
                return this.userSubset;
            }

            set
            {
                if (value == null)
                {
                    throw new ArgumentNullException(nameof(value));
                }

                this.userSubset = value.ToArray();
            }
        }

        /// <summary>
        /// Gets or sets the subset of the items used for related item prediction and item recommendation.
        /// </summary>
        public IEnumerable<TItem> ItemSubset
        {
            get
            {
                return this.itemSubset;
            }

            set
            {
                if (value == null)
                {
                    throw new ArgumentNullException(nameof(value));
                }

                this.itemSubset = value.ToArray();
            }
        }

        /// <summary>
        /// Trains the recommender on the specified instances. For the random recommender it results in just
        /// retrieving the rating info, as well as the list of items and users from the training set.
        /// </summary>
        /// <param name="instanceSource">The source of instances to train on.</param>
        /// <param name="featureSource">The source of features for the specified instances.</param>
        public void Train(TInstanceSource instanceSource, TFeatureSource featureSource = default(TFeatureSource))
        {
            this.starRatingInfo = this.mapping.GetRatingInfo(instanceSource);
            
            var trainingUsers = new HashSet<TUser>();
            var trainingItems = new HashSet<TItem>();
            foreach (TInstance instance in this.mapping.GetInstances(instanceSource))
            {
                trainingUsers.Add(this.mapping.GetUser(instanceSource, instance));
                trainingItems.Add(this.mapping.GetItem(instanceSource, instance));
            }

            this.userSubset = trainingUsers.ToArray();
            this.itemSubset = trainingItems.ToArray();
        }

        /// <summary>
        /// Predicts rating for a given user and item by random guessing.
        /// </summary>
        /// <param name="user">The user.</param>
        /// <param name="item">The item.</param>
        /// <param name="featureSource">The source of features for the specified user and item.</param>
        /// <returns>The predicted rating.</returns>
        public int Predict(TUser user, TItem item, TFeatureSource featureSource = default(TFeatureSource))
        {
            return Rand.Int(this.starRatingInfo.MinStarRating, this.starRatingInfo.MaxStarRating + 1);
        }

        /// <summary>
        /// Predicts ratings for the specified instances by random guessing.
        /// </summary>
        /// <param name="instanceSource">The instances to predict ratings for.</param>
        /// <param name="featureSource">The source of features for the specified instances.</param>
        /// <returns>The predicted ratings.</returns>
        public IDictionary<TUser, IDictionary<TItem, int>> Predict(
            TInstanceSource instanceSource, TFeatureSource featureSource = default(TFeatureSource))
        {
            return this.Predict(instanceSource, (u, i) => this.Predict(u, i));
        }

        /// <summary>
        /// Predicts the distribution of a rating for a given user and item by returning a random distribution over ratings.
        /// </summary>
        /// <param name="user">The user.</param>
        /// <param name="item">The item.</param>
        /// <param name="featureSource">The source of features for the given user and item.</param>
        /// <returns>The distribution of the rating.</returns>
        public RatingDistribution PredictDistribution(TUser user, TItem item, TFeatureSource featureSource = default(TFeatureSource))
        {
            Dirichlet dirichlet = Dirichlet.Uniform(this.starRatingInfo.MaxStarRating - this.starRatingInfo.MinStarRating + 1);
            Vector probabilities = dirichlet.Sample();

            var ratingDistribution = new SortedDictionary<int, double>();
            for (int i = 0; i < probabilities.Count; i++)
            {
                ratingDistribution.Add(this.starRatingInfo.MinStarRating + i, probabilities[i]);
            }

            return ratingDistribution;
        }

        /// <summary>
        /// Predicts rating distributions for the specified instances by returning the uniform 
        /// distribution over the possible rating values.
        /// </summary>
        /// <param name="instanceSource">The instances to predict ratings for.</param>
        /// <param name="featureSource">The source of features for the specified instances.</param>
        /// <returns>The distributions of the predicted ratings.</returns>
        public IDictionary<TUser, IDictionary<TItem, RatingDistribution>> PredictDistribution(
            TInstanceSource instanceSource, TFeatureSource featureSource = default(TFeatureSource))
        {
            return this.Predict(instanceSource, (u, i) => this.PredictDistribution(u, i));
        }

        /// <summary>
        /// Recommends items to a given user by randomly permuting the <see cref="ItemSubset"/>.
        /// </summary>
        /// <param name="user">The user to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend.</param>
        /// <param name="featureSource">The source of features for the specified user.</param>
        /// <returns>The list of recommended items.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        public IEnumerable<TItem> Recommend(TUser user, int recommendationCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (recommendationCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(recommendationCount), "The requested number of items to recommend should be positive.");
            }

            Rand.Shuffle(this.itemSubset);
            return this.itemSubset.Take(recommendationCount);
        }

        /// <summary>
        /// Recommends items to a given list of users by randomly permuting the <see cref="ItemSubset"/>.
        /// </summary>
        /// <param name="users">The list of users to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend to a single user.</param>
        /// <param name="featureSource">The source of features for the specified users.</param>
        /// <returns>The list of recommended items for every user from <paramref name="users"/>.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        public IDictionary<TUser, IEnumerable<TItem>> Recommend(
            IEnumerable<TUser> users, int recommendationCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (users == null)
            {
                throw new ArgumentNullException(nameof(users));
            }

            return users.ToDictionary(u => u, u => this.Recommend(u, recommendationCount));
        }

        /// <summary>
        /// Recommend items with their rating distributions to a specified user.
        /// </summary>
        /// <param name="user">The user to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend.</param>
        /// <param name="featureSource">The source of features for the specified user.</param>
        /// <returns>The list of recommended items and their rating distributions.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        public IEnumerable<Tuple<TItem, RatingDistribution>> RecommendDistribution(
            TUser user, int recommendationCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            return this.Recommend(user, recommendationCount, featureSource)
                .Select(itemId => new Tuple<TItem, RatingDistribution>(itemId, this.PredictDistribution(user, itemId, featureSource)));
        }

        /// <summary>
        /// Recommends items with their rating distributions to a specified list of users.
        /// </summary>
        /// <param name="users">The list of users to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend to a single user.</param>
        /// <param name="featureSource">The source of features for the specified users.</param>
        /// <returns>The list of recommended items and their rating distributions for every user from <paramref name="users"/>.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        public IDictionary<TUser, IEnumerable<Tuple<TItem, RatingDistribution>>> RecommendDistribution(
            IEnumerable<TUser> users, int recommendationCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (users == null)
            {
                throw new ArgumentNullException(nameof(users));
            }

            return users.ToDictionary(u => u, u => this.RecommendDistribution(u, recommendationCount));
        }

        /// <summary>
        /// Returns a list of users related to <paramref name="user"/> by randomly permuting the <see cref="UserSubset"/>.
        /// </summary>
        /// <param name="user">The user for which related users should be found.</param>
        /// <param name="relatedUserCount">Maximum number of related users to return.</param>
        /// <param name="featureSource">The source of features for the users.</param>
        /// <returns>The list of related users.</returns>
        /// <remarks>Only users specified in <see cref="UserSubset"/> will be returned.</remarks>
        public IEnumerable<TUser> GetRelatedUsers(TUser user, int relatedUserCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (relatedUserCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(relatedUserCount), "The requested number of related users should be positive.");
            }
            
            Rand.Shuffle(this.userSubset);
            return this.userSubset.Take(relatedUserCount);
        }

        /// <summary>
        /// Returns a list of related users to each user in <paramref name="users"/> by randomly permuting the <see cref="UserSubset"/>.
        /// </summary>
        /// <param name="users">The list of users for which related users should be found.</param>
        /// <param name="relatedUserCount">Maximum number of related users to return for every user.</param>
        /// <param name="featureSource">The source of features for the specified users.</param>
        /// <returns>The list of related users for each user from <paramref name="users"/>.</returns>
        /// <remarks>Only users specified in <see cref="UserSubset"/> will be returned.</remarks>
        public IDictionary<TUser, IEnumerable<TUser>> GetRelatedUsers(
            IEnumerable<TUser> users, int relatedUserCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (users == null)
            {
                throw new ArgumentNullException(nameof(users));
            }

            return users.ToDictionary(u => u, u => this.GetRelatedUsers(u, relatedUserCount));
        }

        /// <summary>
        /// Returns a list of items related to <paramref name="item"/> by randomly permuting the <see cref="ItemSubset"/>.
        /// </summary>
        /// <param name="item">The item for which related items should be found.</param>
        /// <param name="relatedItemCount">Maximum number of related items to return.</param>
        /// <param name="featureSource">The source of features for the items.</param>
        /// <returns>The list of related items.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> will be returned.</remarks>
        public IEnumerable<TItem> GetRelatedItems(TItem item, int relatedItemCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (relatedItemCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(relatedItemCount), "The requested number of related users should be positive.");
            }
            
            Rand.Shuffle(this.itemSubset);
            return this.itemSubset.Take(relatedItemCount);
        }

        /// <summary>
        /// Returns a list of related items to each item in <paramref name="items"/> by randomly permuting the <see cref="ItemSubset"/>.
        /// </summary>
        /// <param name="items">The list of items for which related items should be found.</param>
        /// <param name="relatedItemCount">Maximum number of related items to return for every item.</param>
        /// <param name="featureSource">The source of features for the specified items.</param>
        /// <returns>The list of related items for each item from <paramref name="items"/>.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> will be returned.</remarks>
        public IDictionary<TItem, IEnumerable<TItem>> GetRelatedItems(IEnumerable<TItem> items, int relatedItemCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (items == null)
            {
                throw new ArgumentNullException(nameof(items));
            }

            return items.ToDictionary(i => i, u => this.GetRelatedItems(u, relatedItemCount));
        }

        #endregion

        #region Helper functions

        /// <summary>
        /// Predicts ratings for the specified instances by using a specified prediction generator.
        /// </summary>
        /// <typeparam name="TPrediction">The type of the prediction.</typeparam>
        /// <param name="instanceSource">The source of instances to predict ratings for.</param>
        /// <param name="predictionGenerator">The prediction generator.</param>
        /// <returns>The predicted ratings.</returns>
        private IDictionary<TUser, IDictionary<TItem, TPrediction>> Predict<TPrediction>(
            TInstanceSource instanceSource, Func<TUser, TItem, TPrediction> predictionGenerator)
        {
            var result = new Dictionary<TUser, IDictionary<TItem, TPrediction>>();
            foreach (TInstance instance in this.mapping.GetInstances(instanceSource))
            {
                TUser user = this.mapping.GetUser(instanceSource, instance);
                TItem item = this.mapping.GetItem(instanceSource, instance);

                IDictionary<TItem, TPrediction> itemToRating;
                if (!result.TryGetValue(user, out itemToRating))
                {
                    itemToRating = new Dictionary<TItem, TPrediction>();
                    result.Add(user, itemToRating);
                }

                itemToRating.Add(item, predictionGenerator(user, item));
            }

            return result;
        }

        #endregion
    }
}
