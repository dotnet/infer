// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Threading.Tasks;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Represents a Matchbox recommender which operates on data in the native format.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
    [Serializable]
    [SerializationVersion(7)]
    internal class NativeDataFormatMatchboxRecommender<TInstanceSource, TFeatureSource> :
        IMatchboxRecommender<TInstanceSource, int, int, Discrete, TFeatureSource>
    {
        #region Fields, constructors, properties

        /// <summary>
        /// The current custom binary serialization version of the <see cref="NativeDataFormatMatchboxRecommender{TInstanceSource, TFeatureSource}"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// The custom binary serialization <see cref="Guid"/> of the <see cref="NativeDataFormatMatchboxRecommender{TInstanceSource, TFeatureSource}"/> class.
        /// </summary>
        private readonly Guid customSerializationGuid = new Guid("A25B3518-5977-4594-9E6A-56B4B0FAC405");

        /// <summary>
        /// The mapping used for accessing data.
        /// </summary>
        private readonly IMatchboxRecommenderMapping<TInstanceSource, TFeatureSource> mapping;

        /// <summary>
        /// The capabilities of the recommender.
        /// </summary>
        private readonly MatchboxRecommenderCapabilities capabilities;

        #region Algorithms used for making predictions

        /// <summary>
        /// The algorithm for the rating prediction.
        /// </summary>
        [NonSerialized]
        private RatingPredictionAlgorithm ratingPredictionAlgorithm;

        /// <summary>
        /// The algorithm for inferring parameters of cold users and items.
        /// </summary>
        [NonSerialized]
        private ColdUserItemParameterAlgorithm coldUserItemParameterAlgorithm;

        #endregion

        /// <summary>
        /// The posteriors over the parameters learned during training.
        /// </summary>
        private ParameterDistributions parameterPosteriorDistributions;

        /// <summary>
        /// The prior distribution of a cold user excluding the effect of features.
        /// This is computed as the average from all training users.
        /// </summary>
        private UserParameterDistribution averageUserParameterDistribution;

        /// <summary>
        /// The prior distribution of a cold item excluding the effect of features.
        /// This is computed as the average from all training items.
        /// </summary>
        private ItemParameterDistribution averageItemParameterDistribution;

        /// <summary>
        /// Subset of the user ids used for related user prediction.
        /// </summary>
        private int[] userSubset;

        /// <summary>
        /// Subset of the item ids used for related items prediction and recommendation.
        /// </summary>
        private int[] itemSubset;

        /// <summary>
        /// Initializes a new instance of the <see cref="NativeDataFormatMatchboxRecommender{TInstanceSource,TFeatureSource}"/> class. 
        /// </summary>
        /// <param name="mapping">The mapping used for accessing data.</param>
        internal NativeDataFormatMatchboxRecommender(IMatchboxRecommenderMapping<TInstanceSource, TFeatureSource> mapping)
        {
            this.capabilities = new MatchboxRecommenderCapabilities();
            this.mapping = mapping;
            this.Settings = new MatchboxRecommenderSettings(() => this.IsTrained);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NativeDataFormatMatchboxRecommender{TInstanceSource,TFeatureSource}"/> class 
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the Matchbox recommender from.</param>
        /// <param name="mapping">The mapping used for accessing data.</param>
        internal NativeDataFormatMatchboxRecommender(BinaryReader reader, IMatchboxRecommenderMapping<TInstanceSource, TFeatureSource> mapping)
        {
            Debug.Assert(reader != null, "The reader must not be null.");
            Debug.Assert(mapping != null, "The mapping must not be null.");

            this.capabilities = new MatchboxRecommenderCapabilities();
            this.mapping = mapping;

            reader.VerifySerializationGuid(this.customSerializationGuid, "The binary stream does not contain an Infer.NET Matchbox recommender.");

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);
            if (deserializedVersion == CustomSerializationVersion)
            {
                this.IsTrained = reader.ReadBoolean();
                this.Settings = new MatchboxRecommenderSettings(reader, () => this.IsTrained);

                if (this.IsTrained)
                {
                    this.parameterPosteriorDistributions = new ParameterDistributions(reader);
                    this.averageUserParameterDistribution = new UserParameterDistribution(reader);
                    this.averageItemParameterDistribution = new ItemParameterDistribution(reader);

                    this.userSubset = reader.ReadInt32Array();
                    this.itemSubset = reader.ReadInt32Array();

                    this.InitializeAlgorithmsForPrediction();
                }
            }
        }

        #endregion

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
        /// Gets the settings of the learner. These should be configured once before any 
        /// query methods are called on the learner.
        /// </summary>
        ISettings ILearner.Settings
        {
            get { return this.Settings; }
        }

        #endregion

        #region IMatchboxRecommender implementation

        /// <summary>
        /// Gets the recommender settings.
        /// </summary>
        public MatchboxRecommenderSettings Settings { get; private set; }

        /// <summary>
        /// Gets a copy of the posterior distribution over the parameters of the Matchbox model.
        /// </summary>
        public PosteriorDistributions<int, int> GetPosteriorDistributions()
        {
            if (!this.IsTrained)
            {
                throw new InvalidOperationException("The recommender is not trained. Cannot obtain posterior distributions.");
            }
            
            var userPosteriors = Enumerable.Range(0, this.parameterPosteriorDistributions.UserCount)
                .ToDictionary(
                    u => u,
                    u => new UserPosteriorDistribution(
                        this.parameterPosteriorDistributions.UserTraitDistribution[u].ToList(),
                        this.parameterPosteriorDistributions.UserBiasDistribution[u],
                        this.parameterPosteriorDistributions.UserThresholdDistribution[u].ToList()));
            var itemPosteriors = Enumerable.Range(0, this.parameterPosteriorDistributions.ItemCount)
                .ToDictionary(
                    i => i,
                    i => new ItemPosteriorDistribution(
                        this.parameterPosteriorDistributions.ItemTraitDistribution[i].ToList(),
                        this.parameterPosteriorDistributions.ItemBiasDistribution[i]));
            var userFeaturePosteriors = Enumerable.Range(0, this.parameterPosteriorDistributions.UserFeature.FeatureCount)
                .Select(f => new FeaturePosteriorDistribution(
                    this.parameterPosteriorDistributions.UserFeature.TraitWeights.Select(t => t[f]).ToList(),
                    this.parameterPosteriorDistributions.UserFeature.BiasWeights[f]))
                .ToList();
            var itemFeautrePosteriors = Enumerable.Range(0, this.parameterPosteriorDistributions.ItemFeature.FeatureCount)
                .Select(f => new FeaturePosteriorDistribution(
                    this.parameterPosteriorDistributions.ItemFeature.TraitWeights.Select(t => t[f]).ToList(),
                    this.parameterPosteriorDistributions.ItemFeature.BiasWeights[f]))
                .ToList();

            return new PosteriorDistributions<int, int>(userPosteriors, itemPosteriors, userFeaturePosteriors, itemFeautrePosteriors);
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
        /// Gets or sets the subset of the user ids used for related user prediction.
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown if the property is get or set for a recommender which has not been trained yet.</exception>
        /// <exception cref="ArgumentNullException">Thrown if the new value of the property is null.</exception>
        /// <exception cref="ArgumentException">Thrown if new value of the property has some invalid user identifiers.</exception>
        public IEnumerable<int> UserSubset
        {
            get
            {
                if (this.userSubset == null)
                {
                    // User subset is always specified for a trained model
                    throw new InvalidOperationException("The value of this property can be obtained only from a trained recommender.");
                }

                return this.userSubset;
            }

            set
            {
                if (this.userSubset == null)
                {
                    // User subset is always specified for a trained model
                    throw new InvalidOperationException("The value of this property can be set only for a trained recommender.");
                }

                if (value == null)
                {
                    throw new ArgumentNullException(nameof(value));
                }

                int[] valueArray = value.ToArray();

                if (valueArray.Distinct().Count() != valueArray.Length)
                {
                    throw new ArgumentException("The user subset contains the same element multiple times.");
                }

                this.userSubset = valueArray;
            }
        }

        /// <summary>
        /// Gets or sets the subset of the items ids used for related user prediction and item recommendation.
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown if the property is get or set for a recommender which has not been trained yet.</exception>
        /// <exception cref="ArgumentNullException">Thrown if the new value of the property is null.</exception>
        /// <exception cref="ArgumentException">Thrown if new value of the property has some invalid item identifiers.</exception>
        public IEnumerable<int> ItemSubset
        {
            get
            {
                if (this.itemSubset == null)
                {
                    // Item subset is always specified for a trained model
                    throw new InvalidOperationException("The value of this property can be obtained only from a trained recommender.");
                }

                return this.itemSubset;
            }

            set
            {
                if (this.itemSubset == null)
                {
                    // Item subset is always specified for a trained model
                    throw new InvalidOperationException("The value of this property can be set only for a trained recommender.");
                }

                if (value == null)
                {
                    throw new ArgumentNullException(nameof(value));
                }

                int[] valueArray = value.ToArray();

                if (valueArray.Distinct().Count() != valueArray.Length)
                {
                    throw new ArgumentException("The item subset contains the same element multiple times.");
                }

                this.itemSubset = value.ToArray();
            }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender is trained.
        /// </summary>
        internal bool IsTrained { get; private set; }

        /// <summary>
        /// Trains the recommender on the given dataset.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="featureSource">The source of the features for the provided instances.</param>
        public void Train(TInstanceSource instanceSource, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (this.IsTrained)
            {
                throw new InvalidOperationException("This recommender was already trained.");
            }

            if (this.mapping == null)
            {
                throw new InvalidOperationException("A mapping must be set before training.");
            }

            var metadata = this.GetInstanceMetadata(instanceSource, featureSource);
            var uniformParameters = new ParameterDistributions(metadata, this.Settings.Training.TraitCount);
            var posteriorsDividedByPriors = (ParameterDistributions)uniformParameters.Clone();
            var batchedOutputMessages = Util.ArrayInit(
                this.Settings.Training.BatchCount, m => (ParameterDistributions)uniformParameters.Clone());
            var batchedPosteriors = Util.ArrayInit(
                this.Settings.Training.BatchCount, p => (ParameterDistributions)uniformParameters.Clone());
            bool modelHasFeatureWeights = this.Settings.Training.UseUserFeatures || this.Settings.Training.UseItemFeatures;
            int iterationCountPerBatch = modelHasFeatureWeights ? 6 : 1;

            var communityTrainingAlgorithmPool = new KeyedPool<int, CommunityTrainingAlgorithm>(
                 () => new CommunityTrainingAlgorithm(
                    iterationCountPerBatch,
                    this.Settings.Training.TraitCount,
                    this.Settings.Training.UseSharedUserThresholds,
                    this.Settings.Training.Advanced.Noise,
                    this.Settings.Training.Advanced.User,
                    this.Settings.Training.Advanced.Item,
                    this.Settings.Training.Advanced.UserFeature,
                    this.Settings.Training.Advanced.ItemFeature));

            // Get the original prior by obtaining the posterior with no data
            var emptyInstanceData = new InstanceData(new int[0], new int[0], new int[0]);
            var priorObtainingAlgorithm = communityTrainingAlgorithmPool.Acquire(0);
            this.SetupCommunityTrainingAlgorithm(
                priorObtainingAlgorithm, emptyInstanceData, metadata, uniformParameters, uniformParameters);

            // The posteriors will be used for message initialization in the 
            // special first iteration, so they are initially set to the priors
            this.parameterPosteriorDistributions = priorObtainingAlgorithm.InferParameters(inferFeatures: false);

            communityTrainingAlgorithmPool.Release(priorObtainingAlgorithm, 0);

            // Train
            for (int iterationNumber = 0; iterationNumber < this.Settings.Training.IterationCount; ++iterationNumber)
            {
                // Map
                this.ProcessBatches(
                    instanceSource, metadata, uniformParameters, posteriorsDividedByPriors, batchedOutputMessages, batchedPosteriors, communityTrainingAlgorithmPool);

                // Reduce
                this.UpdatePosteriorsDividedByPriors(posteriorsDividedByPriors, batchedOutputMessages);
                this.UpdatePosteriors(batchedPosteriors);                
            }
            
            // Get the final posteriors (with the features)
            var posteriorObtainingModel = communityTrainingAlgorithmPool.Acquire(0);
            this.SetupCommunityTrainingAlgorithm(
                posteriorObtainingModel, emptyInstanceData, metadata, posteriorsDividedByPriors, this.parameterPosteriorDistributions);
            this.parameterPosteriorDistributions = posteriorObtainingModel.InferParameters(inferFeatures: true);
            communityTrainingAlgorithmPool.Release(posteriorObtainingModel, 0);

            //// Post-training

            // Duplicate the shared user thresholds across all users
            if (this.Settings.Training.UseSharedUserThresholds)
            {
                var userThresholdsPosterior = this.parameterPosteriorDistributions.UserThresholdDistribution;
                for (int u = 1; u < userThresholdsPosterior.Count; ++u)
                {
                    userThresholdsPosterior[u] = userThresholdsPosterior[0];
                }
            }

            // Initialize user and item subsets to the whole training set
            this.userSubset = Util.ArrayInit(metadata.UserCount, u => u);
            this.itemSubset = Util.ArrayInit(metadata.ItemCount, i => i);

            // Compute the average user and item posterior distributions to be used in cold predictions
            this.averageUserParameterDistribution = AlgorithmUtils.GetAverageUserParameters(this.parameterPosteriorDistributions, metadata.UserFeatures);
            this.averageItemParameterDistribution = AlgorithmUtils.GetAverageItemParameters(this.parameterPosteriorDistributions, metadata.ItemFeatures);

            // Mark the learner as trained
            this.IsTrained = true;

            // Create prediction algorithms
            this.InitializeAlgorithmsForPrediction();
        }

        /// <summary>
        /// Predicts rating for a given user and item.
        /// </summary>
        /// <param name="user">The user.</param>
        /// <param name="item">The item.</param>
        /// <param name="featureSource">The source of the features for the given user and item.</param>
        /// <returns>The predicted rating.</returns>
        public int Predict(int user, int item, TFeatureSource featureSource = default(TFeatureSource))
        {
            Func<int, int, double> customLossFunction;
            LossFunction lossFunction = this.Settings.Prediction.GetPredictionLossFunction(out customLossFunction);
            var toRating = (lossFunction == LossFunction.Custom) ? 
                PointEstimator.ForDiscrete(customLossFunction) : PointEstimator.ForDiscrete(lossFunction);

            return toRating(this.PredictDistribution(user, item, featureSource));
        }

        /// <summary>
        /// Predicts ratings for the instances provided by a given instance source.
        /// </summary>
        /// <param name="instanceSource">The source providing the instances to predict ratings for.</param>
        /// <param name="featureSource">The source of the features for the instances.</param>
        /// <returns>The predicted ratings.</returns>
        public IDictionary<int, IDictionary<int, int>> Predict(
            TInstanceSource instanceSource, TFeatureSource featureSource = default(TFeatureSource))
        {
            var uncertainPrediction = this.PredictDistribution(instanceSource, featureSource);

            Func<int, int, double> customLossFunction;
            LossFunction lossFunction = this.Settings.Prediction.GetPredictionLossFunction(out customLossFunction);
            var toRating = (lossFunction == LossFunction.Custom) ? 
                PointEstimator.ForDiscrete(customLossFunction) : PointEstimator.ForDiscrete(lossFunction);

            return uncertainPrediction.ToDictionary(
                kv => kv.Key,
                kv => (IDictionary<int, int>)kv.Value.ToDictionary(
                    kv2 => kv2.Key,
                    kv2 => toRating(kv2.Value)));
        }

        /// <summary>
        /// Predicts the distribution of a rating for a given user and item.
        /// </summary>
        /// <param name="user">The user.</param>
        /// <param name="item">The item.</param>
        /// <param name="featureSource">The source of the features for the given user and item.</param>
        /// <returns>The distribution of the rating.</returns>
        public Discrete PredictDistribution(int user, int item, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (this.mapping == null)
            {
                throw new InvalidOperationException("A mapping should be set before making predictions.");
            }

            UserParameterDistribution userPosterior = this.GetUserParameterPosterior(user, featureSource);
            ItemParameterDistribution itemPosterior = this.GetItemParameterPosterior(item, featureSource);
            return this.ratingPredictionAlgorithm.InferRatingDistribution(userPosterior, itemPosterior);
        }

        /// <summary>
        /// Predicts rating distributions for the instances provided by a given instance source.
        /// </summary>
        /// <param name="instanceSource">The source providing the instances to predict ratings for.</param>
        /// <param name="featureSource">The source of the features for the instances.</param>
        /// <returns>The distributions of the predicted ratings.</returns>
        public IDictionary<int, IDictionary<int, Discrete>> PredictDistribution(
            TInstanceSource instanceSource, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (this.mapping == null)
            {
                throw new InvalidOperationException("A mapping should be set before making predictions.");
            }

            var userIds = this.mapping.GetUserIds(instanceSource);
            var itemIds = this.mapping.GetItemIds(instanceSource);

            if (userIds == null)
            {
                throw new MatchboxRecommenderException("Invalid array of user ids was provided by the mapping.");
            }

            if (itemIds == null)
            {
                throw new MatchboxRecommenderException("Invalid array of item ids was provided by the mapping.");
            }

            if (userIds.Count != itemIds.Count)
            {
                throw new MatchboxRecommenderException("Arrays of user ids and item ids provided by the mapping should be of the same size.");
            }

            // TODO: This can be parallelized by using an algorithm pool
            IDictionary<int, IDictionary<int, Discrete>> result = new Dictionary<int, IDictionary<int, Discrete>>();
            for (int i = 0; i < userIds.Count; ++i)
            {
                int userId = userIds[i];
                int itemId = itemIds[i];

                // Add a new dictionary for each specified user if they are not already included in the result
                IDictionary<int, Discrete> itemToRating;
                if (!result.TryGetValue(userId, out itemToRating))
                {
                    itemToRating = new Dictionary<int, Discrete>();
                    result.Add(userId, itemToRating);
                }

                // Exclude duplicates
                if (!itemToRating.ContainsKey(itemId))
                {
                    itemToRating.Add(itemId, this.PredictDistribution(userId, itemId, featureSource));
                }
            }

            return result;
        }

        /// <summary>
        /// Recommends items to a given user.
        /// </summary>
        /// <param name="user">The user to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend.</param>
        /// <param name="featureSource">The source of the features for the given user and items to recommend.</param>
        /// <returns>The list of recommended items.</returns>
        /// <remarks>
        /// Only items specified in <see cref="ItemSubset"/> can be recommended.
        /// The caller of this method is expected to cache the recommendations if deemed necessary.
        /// </remarks>
        public IEnumerable<int> Recommend(
            int user, int recommendationCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (recommendationCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(recommendationCount), "The number of items to recommend must be non-negative.");
            }

            var itemMeanRatingPairs = new List<KeyValuePair<int, double>>(this.itemSubset.Length);
            foreach (int itemId in this.itemSubset)
            {
                itemMeanRatingPairs.Add(new KeyValuePair<int, double>(
                    itemId, this.PredictDistribution(user, itemId, featureSource).GetMean()));
            }

            return GetTopScoreEntities(itemMeanRatingPairs, recommendationCount);
        }

        /// <summary>
        /// Recommends items to a given list of users.
        /// </summary>
        /// <param name="users">The list of users to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend to a single user.</param>
        /// <param name="featureSource">The source of the features for the users and items.</param>
        /// <returns>The list of recommended items for every user from <paramref name="users"/>.</returns>
        /// <remarks>
        /// Only items specified in <see cref="ItemSubset"/> can be recommended.
        /// The caller of this method is expected to cache the recommendations if deemed necessary.
        /// </remarks>
        public IDictionary<int, IEnumerable<int>> Recommend(
            IEnumerable<int> users, int recommendationCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (users == null)
            {
                throw new ArgumentNullException(nameof(users));
            }

            if (recommendationCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(recommendationCount), "The number of items to recommend must be non-negative.");
            }

            IDictionary<int, IEnumerable<int>> result = new Dictionary<int, IEnumerable<int>>();
            foreach (int userId in users)
            {
                if (!result.ContainsKey(userId))
                {
                    result.Add(userId, this.Recommend(userId, recommendationCount, featureSource));
                }
            }

            return result;
        }

        /// <summary>
        /// Recommend items with their rating distributions to a specified user.
        /// </summary>
        /// <param name="user">The user to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend.</param>
        /// <param name="featureSource">The source of features for the specified user.</param>
        /// <returns>The list of recommended items and their rating distributions.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        public IEnumerable<Tuple<int, Discrete>> RecommendDistribution(
            int user, int recommendationCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (recommendationCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(recommendationCount), "The number of items to recommend must be non-negative.");
            }

            var itemMeanRatingPairs = new List<KeyValuePair<int, double>>(this.itemSubset.Length);
            var itemRatingDistributions = new Dictionary<int, Discrete>(this.itemSubset.Length);
            foreach (int itemId in this.itemSubset)
            {
                Discrete ratingDistribution = this.PredictDistribution(user, itemId, featureSource);
                itemRatingDistributions.Add(itemId, ratingDistribution);
                itemMeanRatingPairs.Add(new KeyValuePair<int, double>(itemId, ratingDistribution.GetMean()));
            }

            return GetTopScoreEntities(itemMeanRatingPairs, recommendationCount).Select(
                recommendedItem => Tuple.Create(recommendedItem, itemRatingDistributions[recommendedItem]));
        }

        /// <summary>
        /// Recommends items with their rating distributions to a specified list of users.
        /// </summary>
        /// <param name="users">The list of users to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend to a single user.</param>
        /// <param name="featureSource">The source of features for the specified users.</param>
        /// <returns>The list of recommended items and their rating distributions for every user from <paramref name="users"/>.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        public IDictionary<int, IEnumerable<Tuple<int, Discrete>>> RecommendDistribution(
            IEnumerable<int> users, int recommendationCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (users == null)
            {
                throw new ArgumentNullException(nameof(users));
            }

            if (recommendationCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(recommendationCount), "The number of items to recommend must be non-negative.");
            }

            var result = new Dictionary<int, IEnumerable<Tuple<int, Discrete>>>();
            foreach (int userId in users.Where(userId => !result.ContainsKey(userId)))
            {
                result.Add(userId, this.RecommendDistribution(userId, recommendationCount, featureSource));
            }

            return result;
        }

        /// <summary>
        /// Returns a list of users related to <paramref name="user"/>.
        /// </summary>
        /// <param name="user">The user for which related users should be found.</param>
        /// <param name="relatedUserCount">Maximum number of related users to return.</param>
        /// <param name="featureSource">The source of the features for the users.</param>
        /// <returns>The list of related users.</returns>
        /// <remarks>
        /// Only users specified in <see cref="UserSubset"/> will be returned.
        /// The caller of this method is expected to cache the result if deemed necessary.
        /// </remarks>
        public IEnumerable<int> GetRelatedUsers(
            int user, int relatedUserCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (relatedUserCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(relatedUserCount), "The number of related users to find must be non-negative.");
            }

            return this.GetRelatedEntities(
                user, relatedUserCount, this.userSubset, featureSource, this.GetUserParameterPosterior, this.Settings.Training.Advanced.ItemTraitVariance);
        }

        /// <summary>
        /// Returns a list of related users to each user in <paramref name="users"/>.
        /// </summary>
        /// <param name="users">The list of users for which related users should be found.</param>
        /// <param name="relatedUserCount">Maximum number of related users to return for every user.</param>
        /// <param name="featureSource">The source of the features for the users.</param>
        /// <returns>The list of related users for each user from <paramref name="users"/>.</returns>
        /// <remarks>
        /// Only users specified in <see cref="UserSubset"/> will be returned.
        /// The caller of this method is expected to cache the result if deemed necessary.
        /// </remarks>
        public IDictionary<int, IEnumerable<int>> GetRelatedUsers(
            IEnumerable<int> users, int relatedUserCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (users == null)
            {
                throw new ArgumentNullException(nameof(users));
            }

            if (relatedUserCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(relatedUserCount), "The number of related users to find must be non-negative.");
            }

            IDictionary<int, IEnumerable<int>> result = new Dictionary<int, IEnumerable<int>>();
            foreach (int user in users)
            {
                if (!result.ContainsKey(user))
                {
                    result.Add(user, this.GetRelatedUsers(user, relatedUserCount, featureSource));
                }
            }

            return result;
        }

        /// <summary>
        /// Returns a list of items related to <paramref name="item"/>.
        /// </summary>
        /// <param name="item">The item for which related items should be found.</param>
        /// <param name="relatedItemCount">Maximum number of related items to return.</param>
        /// <param name="featureSource">The source of the features for the items.</param>
        /// <returns>The list of related items.</returns>
        /// <remarks>
        /// Only items specified in <see cref="ItemSubset"/> will be returned.
        /// The caller of this method is expected to cache the result if deemed necessary.
        /// </remarks>
        public IEnumerable<int> GetRelatedItems(
            int item, int relatedItemCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (relatedItemCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(relatedItemCount), "The number of related items to find must be non-negative.");
            }

            return this.GetRelatedEntities(
                item, relatedItemCount, this.itemSubset, featureSource, this.GetItemParameterPosterior, this.Settings.Training.Advanced.UserTraitVariance);
        }

        /// <summary>
        /// Returns a list of related items to each item in <paramref name="items"/>.
        /// </summary>
        /// <param name="items">The list of items for which related items should be found.</param>
        /// <param name="relatedItemCount">Maximum number of related items to return for every item.</param>
        /// <param name="featureSource">The source of the features for the items.</param>
        /// <returns>The list of related items for each item from <paramref name="items"/>.</returns>
        /// <remarks>
        /// Only items specified in <see cref="ItemSubset"/> will be returned.
        /// The caller of this method is expected to cache the result if deemed necessary.
        /// </remarks>
        public IDictionary<int, IEnumerable<int>> GetRelatedItems(
            IEnumerable<int> items, int relatedItemCount, TFeatureSource featureSource = default(TFeatureSource))
        {
            if (items == null)
            {
                throw new ArgumentNullException(nameof(items));
            }

            if (relatedItemCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(relatedItemCount), "The number of related items to find must be non-negative.");
            }

            IDictionary<int, IEnumerable<int>> result = new Dictionary<int, IEnumerable<int>>();
            foreach (int item in items)
            {
                if (!result.ContainsKey(item))
                {
                    result.Add(item, this.GetRelatedItems(item, relatedItemCount, featureSource));
                }
            }

            return result;
        }

        #endregion

        #region ICustomSerializable implementation

        /// <summary>
        /// Saves the state of the Matchbox recommender using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the state of the Matchbox recommender to.</param>
        public void SaveForwardCompatible(BinaryWriter writer)
        {
            writer.Write(this.customSerializationGuid);
            writer.Write(CustomSerializationVersion);

            // No need to write capabilities as they can be created from scratch
            writer.Write(this.IsTrained);
            this.Settings.SaveForwardCompatible(writer);

            if (this.IsTrained)
            {
                this.parameterPosteriorDistributions.SaveForwardCompatible(writer);
                this.averageUserParameterDistribution.SaveForwardCompatible(writer);
                this.averageItemParameterDistribution.SaveForwardCompatible(writer);

                writer.Write(this.userSubset);
                writer.Write(this.itemSubset);
            }
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Gets the top-score entities (using a heap).
        /// </summary>
        /// <param name="entityScorePairs">An unordered enumerable of entity-to-score pairs.</param>
        /// <param name="count">The number of entities to extract.</param>
        /// <returns>Decreasingly ordered enumerable of the identifiers of the top entities.</returns>
        private static IEnumerable<int> GetTopScoreEntities(IEnumerable<KeyValuePair<int, double>> entityScorePairs, int count)
        {
            var comparer = Comparer<KeyValuePair<int, double>>.Create((l, r) => l.Value.CompareTo(r.Value));
            return Util.GetMaxKValues(entityScorePairs, count, comparer).Select(kvp => kvp.Key);
        }

        /// <summary>
        /// Performs all actions required after deserialization.
        /// </summary>
        /// <param name="context">The streaming context.</param>
        [OnDeserialized]
        private void OnDeserialized(StreamingContext context)
        {
            if (this.IsTrained)
            {
                this.InitializeAlgorithmsForPrediction();
            }
        }

        /// <summary>
        /// Gets the instance data from the mapping.
        /// Verifies that the data is consistent.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <param name="batchNumber">The number of the current batch.</param>
        /// <param name="userCount">
        /// The overall number of users. 
        /// This parameter is required for data consistency checks.
        /// </param>
        /// <param name="itemCount">
        /// The overall number of items.
        /// This parameter is required for data consistency checks.
        /// </param>
        /// <param name="ratingCount">
        /// The number of star ratings.
        /// This parameter is required for data consistency checks.
        /// </param>
        /// <returns>The instance data.</returns>
        private InstanceData GetInstanceData(
            TInstanceSource instanceSource, 
            int batchNumber, 
            int userCount, 
            int itemCount, 
            int ratingCount)
        {
            var userIds = this.mapping.GetUserIds(instanceSource, batchNumber);
            var itemIds = this.mapping.GetItemIds(instanceSource, batchNumber);
            var ratings = this.mapping.GetRatings(instanceSource, batchNumber);

            this.CheckInstanceDataConsistency(userIds, itemIds, ratings, userCount, itemCount, ratingCount);

            return new InstanceData(userIds, itemIds, ratings);
        }

        /// <summary>
        /// Gets the instance metadata from the mapping.
        /// Verifies that the data is consistent.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <param name="featureSource">The source of features.</param>
        /// <returns>The instance metadata.</returns>
        private InstanceMetadata GetInstanceMetadata(TInstanceSource instanceSource, TFeatureSource featureSource)
        {
            // Retrieve features and counts using the mapping
            int userCount = this.mapping.GetUserCount(instanceSource);
            int itemCount = this.mapping.GetItemCount(instanceSource);
            int ratingCount = this.mapping.GetRatingCount(instanceSource);

            // Retrieve the user features
            int userFeatureCount = 0;
            IList<IList<double>> nonZeroUserFeatureValues = null;
            IList<IList<int>> nonZeroUserFeatureIndices = null;
            if (this.Settings.Training.UseUserFeatures)
            {
                userFeatureCount = this.mapping.GetUserFeatureCount(featureSource);
                nonZeroUserFeatureValues = this.mapping.GetAllUserNonZeroFeatureValues(featureSource);
                nonZeroUserFeatureIndices = this.mapping.GetAllUserNonZeroFeatureIndices(featureSource);
            }

            // Retrieve the item features
            int itemFeatureCount = 0;
            IList<IList<double>> nonZeroItemFeatureValues = null;
            IList<IList<int>> nonZeroItemFeatureIndices = null;
            if (this.Settings.Training.UseItemFeatures)
            {
                itemFeatureCount = this.mapping.GetItemFeatureCount(featureSource);
                nonZeroItemFeatureValues = this.mapping.GetAllItemNonZeroFeatureValues(featureSource);
                nonZeroItemFeatureIndices = this.mapping.GetAllItemNonZeroFeatureIndices(featureSource);
            }

            // Check if the feature data is consistent
            this.CheckFeatureDataConsistency(
                nonZeroUserFeatureValues,
                nonZeroUserFeatureIndices,
                nonZeroItemFeatureValues,
                nonZeroItemFeatureIndices,
                userCount,
                itemCount);

            // Create wrappers for the features
            var userFeatures = this.Settings.Training.UseUserFeatures
                ? SparseFeatureMatrix.Create(userFeatureCount, nonZeroUserFeatureValues, nonZeroUserFeatureIndices)
                : SparseFeatureMatrix.CreateAllZero(userCount);
            var itemFeatures = this.Settings.Training.UseItemFeatures
                ? SparseFeatureMatrix.Create(itemFeatureCount, nonZeroItemFeatureValues, nonZeroItemFeatureIndices)
                : SparseFeatureMatrix.CreateAllZero(itemCount);

            return new InstanceMetadata(userCount, itemCount, ratingCount, userFeatures, itemFeatures);
        }

        /// <summary>
        /// Performs training on each data batch.
        /// </summary>
        /// <param name="instanceSource">The source of instances.</param>
        /// <param name="metadata">The instance metadata.</param>
        /// <param name="uniformParameterDistributions">A uniform parameter distribution.</param>
        /// <param name="posteriorsDividedByPriors">The current posteriors divided by priors.</param>
        /// <param name="batchedOutputMessages">The batched output messages.</param>
        /// <param name="batchedPosteriors">The batched posteriors.</param>
        /// <param name="communityTrainingAlgorithmPool">A pool of community training algorithms.</param>
        private void ProcessBatches(
            TInstanceSource instanceSource, 
            InstanceMetadata metadata,
            ParameterDistributions uniformParameterDistributions,
            ParameterDistributions posteriorsDividedByPriors,
            ParameterDistributions[] batchedOutputMessages,
            ParameterDistributions[] batchedPosteriors,
            KeyedPool<int, CommunityTrainingAlgorithm> communityTrainingAlgorithmPool)
        {
            try
            {
                Parallel.For(
                    0,
                    this.Settings.Training.BatchCount,
                    batchNumber =>
                    {
                        //// The algorithm for each batch[i] is as follows:
                        //// 1) constraints = marginalDividedByPrior / messages[i]
                        //// 2) Infer marginalDividedByPrior
                        //// 3) messages[i] = marginalDividedByPrior / constraints

                        var instanceData = GetInstanceData(
                            instanceSource, batchNumber, metadata.UserCount, metadata.ItemCount, metadata.RatingCount);

                        var constraints = (ParameterDistributions)uniformParameterDistributions.Clone();
                        constraints.SetEntityParametersToRatio(posteriorsDividedByPriors, batchedOutputMessages[batchNumber]);

                        // Try to pick up an algorithm for the same number of observations to avoid reallocations
                        var currentBatchAlgorithm = communityTrainingAlgorithmPool.Acquire(instanceData.Ratings.Count);

                        var initializers = (ParameterDistributions)uniformParameterDistributions.Clone();
                        initializers.SetEntityParametersToRatio(this.parameterPosteriorDistributions, constraints);
                        this.SetupCommunityTrainingAlgorithm(currentBatchAlgorithm, instanceData, metadata, constraints, initializers);
                        batchedPosteriors[batchNumber] = currentBatchAlgorithm.InferParameters(inferFeatures: false);
                        var currentPosteriorsDividedByPriors = currentBatchAlgorithm.GetOutputMessages();
                        batchedOutputMessages[batchNumber].SetEntityParametersToRatio(currentPosteriorsDividedByPriors, constraints);

                        communityTrainingAlgorithmPool.Release(currentBatchAlgorithm, instanceData.Ratings.Count);
                    });
            }
            catch (AggregateException ex)
            {
                // To make the Visual Studio debugger stop at the inner exception, check "Enable Just My Code" in Debug->Options.
                // throw InnerException while preserving stack trace
                // https://stackoverflow.com/questions/57383/in-c-how-can-i-rethrow-innerexception-without-losing-stack-trace
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
                throw;
            }
        }

        /// <summary>
        /// Sets up a community training algorithm for training.
        /// </summary>
        /// <param name="algorithm">The algorithm to set up.</param>
        /// <param name="instanceData">The instance data.</param>
        /// <param name="metadata">The instance metadata.</param>
        /// <param name="constraints">The constraints (required for batching).</param>
        /// <param name="initializers">The initializers (required for the special first iteration).</param>
        private void SetupCommunityTrainingAlgorithm(
            CommunityTrainingAlgorithm algorithm,
            InstanceData instanceData,
            InstanceMetadata metadata,
            ParameterDistributions constraints,
            ParameterDistributions initializers)
        {
            algorithm.SetObservedMetadata(metadata);
            algorithm.SetObservedInstanceData(instanceData);
            algorithm.ConstrainEntityParameters(constraints);
            algorithm.InitializeEntityParameters(initializers);
        }

        /// <summary>
        /// Computes the overall posterior divided by prior 
        /// from the set of output messages of each batch.
        /// </summary>
        /// <param name="posteriorsDividedByPriors">The posteriors divided by priors to update.</param>
        /// <param name="batchedOutputMessages">The set of output messages of each batch.</param>
        private void UpdatePosteriorsDividedByPriors(
            ParameterDistributions posteriorsDividedByPriors, IEnumerable<ParameterDistributions> batchedOutputMessages)
        {
            // The cummulative posteriors-divided-by-priors of
            // all batches are the product of all output messages
            posteriorsDividedByPriors.SetEntityParametersToUniform();
            foreach (var outputMessage in batchedOutputMessages)
            {
                posteriorsDividedByPriors.SetEntityParametersToProduct(posteriorsDividedByPriors, outputMessage);
            }        
        }

        /// <summary>
        /// Computes the overall posteriors from the set of posteriors of each batch.
        /// </summary>
        /// <param name="batchedPosteriors">The set of posteriors of each batch.</param>
        private void UpdatePosteriors(IEnumerable<ParameterDistributions> batchedPosteriors)
        {
            // The cummulative posteriors of all batches
            // are the geometric mean of all posteriors
            this.parameterPosteriorDistributions.SetEntityParametersToUniform();
            foreach (var posterior in batchedPosteriors)
            {
                this.parameterPosteriorDistributions.SetEntityParametersToProduct(this.parameterPosteriorDistributions, posterior);
            }

            this.parameterPosteriorDistributions.SetEntityParametersToPower(this.parameterPosteriorDistributions, 1.0 / this.Settings.Training.BatchCount);        
        }

        /// <summary>
        /// Initializes the algorithms needed for making predictions.
        /// </summary>
        private void InitializeAlgorithmsForPrediction()
        {
            Debug.Assert(this.IsTrained, "The recommender must be trained before creating prediction algorithms.");

            this.ratingPredictionAlgorithm = new RatingPredictionAlgorithm(this.Settings.Training.Advanced.Noise);
            this.coldUserItemParameterAlgorithm =
                new ColdUserItemParameterAlgorithm(
                    this.parameterPosteriorDistributions.UserFeature,
                    this.parameterPosteriorDistributions.ItemFeature,
                    this.averageUserParameterDistribution,
                    this.averageItemParameterDistribution);
        }

        /// <summary>
        /// Gets the parameter posteriors for the user with a given identifier.
        /// </summary>
        /// <param name="userId">The user identifier.</param>
        /// <param name="featureSource">The feature source.</param>
        /// <returns>The parameter posteriors for the user.</returns>
        private UserParameterDistribution GetUserParameterPosterior(int userId, TFeatureSource featureSource)
        {
            bool isColdUser = userId < 0 || userId >= this.parameterPosteriorDistributions.UserCount;
            if (!isColdUser)
            {
                return this.parameterPosteriorDistributions.ForUser(userId);
            }

            SparseFeatureVector features = this.GetEntityFeatures(
                this.Settings.Training.UseUserFeatures,
                this.parameterPosteriorDistributions.UserFeature,
                () => this.mapping.GetSingleUserNonZeroFeatureValues(featureSource, userId),
                () => this.mapping.GetSingleUserNonZeroFeatureIndices(featureSource, userId));
            return this.coldUserItemParameterAlgorithm.InferUserParameters(features);
        }

        /// <summary>
        /// Gets the parameter posteriors for the item with a given identifier.
        /// </summary>
        /// <param name="itemId">The item identifier.</param>
        /// <param name="featureSource">The feature source.</param>
        /// <returns>The parameter posteriors for the item.</returns>
        private ItemParameterDistribution GetItemParameterPosterior(int itemId, TFeatureSource featureSource)
        {
            bool isColdItem = itemId < 0 || itemId >= this.parameterPosteriorDistributions.ItemCount;
            if (!isColdItem)
            {
                return this.parameterPosteriorDistributions.ForItem(itemId);
            }

            SparseFeatureVector features = this.GetEntityFeatures(
                this.Settings.Training.UseItemFeatures,
                this.parameterPosteriorDistributions.ItemFeature,
                () => this.mapping.GetSingleItemNonZeroFeatureValues(featureSource, itemId),
                () => this.mapping.GetSingleItemNonZeroFeatureIndices(featureSource, itemId));
            return this.coldUserItemParameterAlgorithm.InferItemParameters(features);
        }

        /// <summary>
        /// Retrieves a feature vector for a given entity (user or item). If the model doesn't use features for this kind of entity,
        /// zero feature vector is returned.
        /// </summary>
        /// <param name="useFeatures">Specifies whether the model uses features for this kind of entity.</param>
        /// <param name="featureParameterPosterior">The posterior over feature weights learned during training.</param>
        /// <param name="nonZeroFeatureValuesRetriever">A function to retrieve non-zero feature values for the entity.</param>
        /// <param name="nonZeroFeatureIndicesRetriever">A function to retrieve non-zero feature indices for the entity.</param>
        /// <returns>An instance of <see cref="SparseFeatureVector"/> representing entity features.</returns>
        private SparseFeatureVector GetEntityFeatures(
            bool useFeatures,
            FeatureParameterDistribution featureParameterPosterior,
            Func<IList<double>> nonZeroFeatureValuesRetriever,
            Func<IList<int>> nonZeroFeatureIndicesRetriever)
        {
            if (!useFeatures)
            {
                return SparseFeatureVector.CreateAllZero(featureParameterPosterior.FeatureCount);
            }

            var nonZeroFeatureValues = nonZeroFeatureValuesRetriever();
            var nonZeroFeatureIndices = nonZeroFeatureIndicesRetriever();

            if (!SparseFeatureVector.CanBeCreatedFrom(
                nonZeroFeatureValues, nonZeroFeatureIndices, featureParameterPosterior.FeatureCount))
            {
                throw new MatchboxRecommenderException("Provided features are either not valid.");
            }

            return SparseFeatureVector.Create(
                nonZeroFeatureValues, nonZeroFeatureIndices, featureParameterPosterior.FeatureCount);
        }

        /// <summary>
        /// Returns a list of entities related to <paramref name="entity"/>.
        /// </summary>
        /// <typeparam name="TEntityParameterDistribution">The type of an entity parameter distribution.</typeparam>
        /// <param name="entity">The entity to search related to.</param>
        /// <param name="relatedEntityCount">The maximum number of related entities to return.</param>
        /// <param name="entitySubset">Specifies the list of entities which can be returned.</param>
        /// <param name="featureSource">The feature source.</param>
        /// <param name="getEntityParameterDistribution">A method to obtain the entity parameter distribution.</param>
        /// <param name="otherEntityTraitVariance">The variance of the traits of the other entity.</param>
        /// <returns>The list of related entities.</returns>
        /// <remarks>'Other entity' refers to a user when 'entity' is an item and vice versa.</remarks>
        private IEnumerable<int> GetRelatedEntities<TEntityParameterDistribution>(
            int entity,
            int relatedEntityCount,
            int[] entitySubset,
            TFeatureSource featureSource,
            Func<int, TFeatureSource, TEntityParameterDistribution> getEntityParameterDistribution,
            double otherEntityTraitVariance)
            where TEntityParameterDistribution : EntityParameterDistribution
        {
            var queryEntityParameterDistribution = getEntityParameterDistribution(entity, featureSource);
            var queryEntityBias = queryEntityParameterDistribution.Bias.GetMean();
            var queryEntityTraits = Vector.FromArray(Util.ArrayInit(
                this.Settings.Training.TraitCount, t => queryEntityParameterDistribution.Traits[t].GetMean()));

            // Exclude the current entity from the result
            var entitiesToScore = entitySubset.Where(e => e != entity);
            var entityScorePairs = new List<KeyValuePair<int, double>>(entitySubset.Length);

            foreach (var entityId in entitiesToScore)
            {
                var relatedEntityParameterDistribution = getEntityParameterDistribution(entityId, featureSource);
                var relatedEntityBias = relatedEntityParameterDistribution.Bias.GetMean();
                var relatedEntityTraits = Vector.FromArray(Util.ArrayInit(
                    this.Settings.Training.TraitCount, t => relatedEntityParameterDistribution.Traits[t].GetMean()));

                var mean = queryEntityBias - relatedEntityBias;
                var traitDelta = queryEntityTraits - relatedEntityTraits;
                var variance = (2 * this.Settings.Training.Advanced.AffinityNoiseVariance) + (traitDelta.Inner(traitDelta) * otherEntityTraitVariance);
                var score = Gaussian.GetLogProb(0, mean, variance);

                entityScorePairs.Add(new KeyValuePair<int, double>(entityId, score));
            }

            return GetTopScoreEntities(entityScorePairs, relatedEntityCount);
        }

        /// <summary>
        /// Checks if data returned by the native mapping is consistent.
        /// </summary>
        /// <param name="userIds">The array of user ids returned by the mapping.</param>
        /// <param name="itemIds">The array of item ids returned by the mapping.</param>
        /// <param name="ratings">The array of ratings returned by the mapping.</param>
        /// <param name="userCount">The number of users returned by the mapping.</param>
        /// <param name="itemCount">The number of items returned by the mapping.</param>
        /// <param name="ratingCount">The upper bound on the rating values.</param>
        private void CheckInstanceDataConsistency(
            IList<int> userIds,
            IList<int> itemIds,
            IList<int> ratings,
            int userCount,
            int itemCount,
            int ratingCount)
        {
            if (userIds == null)
            {
                throw new MatchboxRecommenderException("Invalid array of user ids was provided by the mapping.");
            }

            if (itemIds == null)
            {
                throw new MatchboxRecommenderException("Invalid array of item ids was provided by the mapping.");
            }

            if (ratings == null)
            {
                throw new MatchboxRecommenderException("Invalid array of ratings was provided by the mapping.");
            }

            if (userIds.Count != itemIds.Count || userIds.Count != ratings.Count)
            {
                throw new MatchboxRecommenderException(
                    "The arrays of user ids, item ids and ratings provided by the mapping should be of the same size.");
            }

            if (userIds.Any(u => u < 0 || u >= userCount) || itemIds.Any(i => i < 0 || i >= itemCount))
            {
                throw new MatchboxRecommenderException(
                    "The user and item identifiers provided by the mapping should be in the range defined by it.");
            }

            if (ratings.Any(r => r < 0 || r >= ratingCount))
            {
                throw new MatchboxRecommenderException("All ratings provided by the mapping should be in the range defined by it.");
            }
        }

        /// <summary>
        /// Checks if the feature data returned by the native mapping is consistent.
        /// </summary>
        /// <param name="nonZeroUserFeatureValues">The array of arrays of non-zero user feature values.</param>
        /// <param name="nonZeroUserFeatureIndices">The array of arrays of non-zero user feature indices.</param>
        /// <param name="nonZeroItemFeatureValues">The array of arrays of non-zero item feature values.</param>
        /// <param name="nonZeroItemFeatureIndices">The array of arrays of non-zero item feature indices.</param>
        /// <param name="userCount">The number of users returned by the mapping.</param>
        /// <param name="itemCount">The number of items returned by the mapping.</param>
        private void CheckFeatureDataConsistency(
            IList<IList<double>> nonZeroUserFeatureValues,
            IList<IList<int>> nonZeroUserFeatureIndices,
            IList<IList<double>> nonZeroItemFeatureValues,
            IList<IList<int>> nonZeroItemFeatureIndices,
            int userCount,
            int itemCount)
        {
            if (this.Settings.Training.UseUserFeatures)
            {
                if (!SparseFeatureMatrix.CanBeCreatedFrom(nonZeroUserFeatureValues, nonZeroUserFeatureIndices))
                {
                    throw new MatchboxRecommenderException("Provided user features are not valid.");
                }

                if (nonZeroUserFeatureValues.Count != userCount)
                {
                    throw new MatchboxRecommenderException("Features should be provided for every user in the training set, and only for them.");
                }
            }

            if (this.Settings.Training.UseItemFeatures)
            {
                if (!SparseFeatureMatrix.CanBeCreatedFrom(nonZeroItemFeatureValues, nonZeroItemFeatureIndices))
                {
                    throw new MatchboxRecommenderException("Provided item features are not valid.");
                }

                if (nonZeroItemFeatureValues.Count != itemCount)
                {
                    throw new MatchboxRecommenderException("Features should be provided for every item in the training set, and only for them.");
                }
            }
        }

        #endregion
    }
}
