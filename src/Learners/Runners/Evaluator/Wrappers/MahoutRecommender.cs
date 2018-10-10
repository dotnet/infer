// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Net;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    using RatingDistribution = System.Collections.Generic.IDictionary<int, double>;

    /// <summary>
    /// The wrapper for the recommendation algorithms from Mahout.
    /// </summary>
    /// <remarks>
    /// This wrapper operates on the data model represented by <see cref="RecommenderDataset"/> and uses mapping
    /// only to retrieve list of instances and rating info.
    /// </remarks>
    /// <typeparam name="TInstanceSource">The type of an instance source.</typeparam>
    internal class MahoutRecommender<TInstanceSource> :
        IRecommender<TInstanceSource, User, Item, int, RatingDistribution, DummyFeatureSource>
    {
        /// <summary>
        /// The relative path to the java classes needed to invoke the Mahout.
        /// </summary>
        private static readonly string PathToClasses = Path.Combine("Data", "Bin");

        /// <summary>
        /// The relative path to the mahout-core-x-job.jar file.
        /// </summary>
        private static readonly string PathToJar = Path.Combine("Data", "Bin", "mahout-core-0.8-job.jar");

        /// <summary>
        /// Url of released zip with Mahout
        /// </summary>
        private static readonly string ReleaseUrl = @"http://archive.apache.org/dist/mahout/0.8/mahout-distribution-0.8.zip";

        /// <summary>
        /// Path to mahout-core-x-job.jar within zip
        /// </summary>
        private static readonly string CoreJobJarPath = @"mahout-distribution-0.8/mahout-core-0.8-job.jar";

        /// <summary>
        /// Directory to store temp files during zip extraction
        /// </summary>
        private static readonly string InstallationWorkDir = Path.Combine("Data", "Bin");

        private static readonly string JarSeparator = WrapperUtils.DetectedOS == WrapperUtils.OS.Windows ? ";" : ":";

        /// <summary>
        /// The mapping used to access the data.
        /// </summary>
        private readonly IStarRatingRecommenderMapping<TInstanceSource, RatedUserItem, User, Item, int, DummyFeatureSource, Vector> mapping;

        /// <summary>
        /// The mapping from users to dense integer identifiers as required by Mahout.
        /// </summary>
        private readonly Dictionary<User, int> userToId = new Dictionary<User, int>();

        /// <summary>
        /// The mapping from dense integer identifiers to users as required by Mahout.
        /// </summary>
        private readonly List<User> idToUser = new List<User>();

        /// <summary>
        /// The mapping from items to dense integer identifiers as required by Mahout.
        /// </summary>
        private readonly Dictionary<Item, int> itemToId = new Dictionary<Item, int>();

        /// <summary>
        /// The mapping from dense integer identifiers to items as required by Mahout.
        /// </summary>
        private readonly List<Item> idToItem = new List<Item>();

        /// <summary>
        /// The name of the temporary file with the training data in Mahout format.
        /// </summary>
        private string trainingDatasetFile;

        /// <summary>
        /// The subset of items to find related items from.
        /// </summary>
        private IEnumerable<Item> itemSubset;

        /// <summary>
        /// The subset of users to find related users from.
        /// </summary>
        private IEnumerable<User> userSubset;

        /// <summary>
        /// The last context for lazy related user search.
        /// </summary>
        private LazyFindRelatedUsersContext lastFindRelatedUsersContext;

        /// <summary>
        /// The last context for lazy related item search.
        /// </summary>
        private LazyFindRelatedItemsContext lastFindRelatedItemsContext;

        /// <summary>
        /// The rating info.
        /// </summary>
        private IStarRatingInfo<int> starRatingInfo;
        
        /// <summary>
        /// The rating distribution in the training set (with the support shifted by -MinRating to handle negative ratings).
        /// </summary>
        private Discrete trainingSetRatingDistribution;

        /// <summary>
        /// Initializes a new instance of the <see cref="MahoutRecommender{TInstanceSource}"/> class.
        /// </summary>
        /// <param name="mapping">The mapping used to access the data.</param>
        private MahoutRecommender(
            IStarRatingRecommenderMapping<TInstanceSource, RatedUserItem, User, Item, int, DummyFeatureSource, Vector> mapping)
        {
            Debug.Assert(mapping != null, "A valid mapping should be provided.");

            this.mapping = mapping;
            this.Settings = new MahoutRecommenderSettings();
        }

        /// <summary>
        /// Finalizes an instance of the <see cref="MahoutRecommender{TInstanceSource}"/> class. 
        /// </summary>
        ~MahoutRecommender()
        {
            // TODO: implement IDisposable pattern?
            if (this.trainingDatasetFile != null)
            {
                File.Delete(this.trainingDatasetFile);
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MahoutRecommender{TInstanceSource}"/> class. Downloads recommender if necessary.
        /// </summary>
        /// <param name="mapping">The mapping used to access the data.</param>
        public static MahoutRecommender<TInstanceSource> Create(
            IStarRatingRecommenderMapping<TInstanceSource, RatedUserItem, User, Item, int, DummyFeatureSource, Vector> mapping)
        {
            if (!File.Exists(PathToJar))
            {
                // getting zip
                var zipName = "mahout.zip";
                var tmpFile = Path.Combine(InstallationWorkDir, zipName);
                ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
                new WebClient().DownloadFile(ReleaseUrl, tmpFile);
                // extracting zip
                using (var file = File.OpenRead(tmpFile))
                {
                    var zip = new System.IO.Compression.ZipArchive(file);
                    var entry = zip.GetEntry(CoreJobJarPath);
                    using (Stream source = entry.Open(), dest = File.OpenWrite(PathToJar))
                    {
                        source.CopyTo(dest);
                    }
                }
                // deleting temp files
                File.Delete(tmpFile);
            }
            return new MahoutRecommender<TInstanceSource>(mapping);
        }

        /// <summary>
        /// Gets the settings.
        /// </summary>
        public MahoutRecommenderSettings Settings { get; private set; }

        #region Explicit ILearner implementation

        /// <summary>
        /// Gets the capabilities.
        /// </summary>
        ICapabilities ILearner.Capabilities
        {
            get { throw new NotImplementedException(); }
        }

        /// <summary>
        /// Gets the settings.
        /// </summary>
        ISettings ILearner.Settings
        {
            get { return this.Settings; }
        }

        #endregion

        #region IRecommender implementation

        /// <summary>
        /// Gets the capabilities of the recommender.
        /// </summary>
        public IRecommenderCapabilities Capabilities
        {
            get { throw new NotImplementedException(); }
        }

        /// <summary>
        /// Gets or sets the subset of the items used for related user prediction.
        /// </summary>
        public IEnumerable<User> UserSubset
        {
            get
            {
                Debug.Assert(this.userSubset != null, "The user subset can not be requested before training.");
                return this.userSubset;
            }

            set
            {
                Debug.Assert(this.userSubset != null, "The user subset can not be set up before training.");
                Debug.Assert(value != null, "The user subset can not be null.");
                
                if (value.Any(u => !this.userToId.ContainsKey(u)))
                {
                    throw new NotSupportedException("Cold users are not supported by this recommender.");    
                }

                this.userSubset = value;
            }
        }

        /// <summary>
        /// Gets or sets the subset of the items used for related item prediction.
        /// </summary>
        public IEnumerable<Item> ItemSubset
        {
            get
            {
                Debug.Assert(this.itemSubset != null, "The item subset can not be requested before training.");
                return this.itemSubset;
            }

            set
            {
                Debug.Assert(this.itemSubset != null, "The item subset can not be set up before training.");
                Debug.Assert(value != null, "The item subset can not be null.");

                if (value.Any(i => !this.itemToId.ContainsKey(i)))
                {
                    throw new NotSupportedException("Cold items are not supported by this recommender.");
                }
                
                this.itemSubset = value;
            }
        }

        /// <summary>
        /// Trains the recommender on a given dataset. Since Mahout doesn't support model persistence,
        /// the training procedure just saves the training data to a temporary file to use it later during prediction.
        /// </summary>
        /// <param name="instanceSource">The instances of the dataset.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        public void Train(TInstanceSource instanceSource, DummyFeatureSource featureSource = null)
        {
            IEnumerable<RatedUserItem> trainingInstances = this.mapping.GetInstances(instanceSource).ToList();
            this.starRatingInfo = this.mapping.GetRatingInfo(instanceSource);
            this.trainingSetRatingDistribution = this.EstimateRatingDistribution(trainingInstances);
            
            // Map users and items to dense integers as required by Mahout
            foreach (RatedUserItem trainingInstance in trainingInstances)
            {
                if (!this.userToId.ContainsKey(trainingInstance.User))
                {
                    this.idToUser.Add(trainingInstance.User);
                    this.userToId.Add(trainingInstance.User, this.userToId.Count);
                }

                if (!this.itemToId.ContainsKey(trainingInstance.Item))
                {
                    this.idToItem.Add(trainingInstance.Item);
                    this.itemToId.Add(trainingInstance.Item, this.itemToId.Count);
                }
            }

            // Setup user and item subsets
            this.userSubset = this.userToId.Keys;
            this.itemSubset = this.itemToId.Keys;
            
            // Save the training dataset in Mahout format
            this.trainingDatasetFile = this.CreateDatasetFile(trainingInstances);
        }

        /// <summary>
        /// This query is not supported.
        /// </summary>
        /// <param name="user">The parameter is not used.</param>
        /// <param name="item">The parameter is not used.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>Nothing, since the method always throws.</returns>
        public int Predict(User user, Item item, DummyFeatureSource featureSource = null)
        {
            throw new NotSupportedException("Only bulk rating prediction is supported by this recommender.");
        }

        /// <summary>
        /// This query is not supported.
        /// </summary>
        /// <param name="instanceSource">The instances to predict ratings for.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>The predicted ratings.</returns>
        public IDictionary<User, IDictionary<Item, int>> Predict(
            TInstanceSource instanceSource, DummyFeatureSource featureSource = null)
        {
            string queryFileName = null;
            string predictionsFileName = null;

            try
            {
                IEnumerable<RatedUserItem> observations = this.mapping.GetInstances(instanceSource).ToList();
                if (observations.Any(o => !this.userToId.ContainsKey(o.User) || !this.itemToId.ContainsKey(o.Item)))
                {
                    throw new NotSupportedException("Cold users and items are not supported by this recommender.");    
                }
                
                predictionsFileName = Path.GetTempFileName();
                queryFileName = this.CreateRatingPredictionQueryFile(observations);
                string command = null;
                switch (this.Settings.RatingPredictionAlgorithm)
                {
                    case MahoutRatingPredictionAlgorithm.UserBased:
                        command = string.Format(
                            "PredictRatings_UserBased \"{0}\" \"{1}\" \"{2}\" {3} {4}",
                            this.trainingDatasetFile,
                            queryFileName,
                            predictionsFileName,
                            this.Settings.RatingSimilarity,
                            this.Settings.UserNeighborhoodSize);
                        break;
                    case MahoutRatingPredictionAlgorithm.ItemBased:
                        command = string.Format(
                            "PredictRatings_ItemBased \"{0}\" \"{1}\" \"{2}\" {3}",
                            this.trainingDatasetFile,
                            queryFileName,
                            predictionsFileName,
                            this.Settings.RatingSimilarity);
                        break;
                    case MahoutRatingPredictionAlgorithm.SlopeOne:
                        command = string.Format(
                            "PredictRatings_SlopeOne \"{0}\" \"{1}\" \"{2}\"",
                            this.trainingDatasetFile,
                            queryFileName,
                            predictionsFileName);
                        break;
                    case MahoutRatingPredictionAlgorithm.Svd:
                        command = string.Format(
                            "PredictRatings_Svd \"{0}\" \"{1}\" \"{2}\" {3} {4}",
                            this.trainingDatasetFile,
                            queryFileName,
                            predictionsFileName,
                            this.Settings.TraitCount,
                            this.Settings.IterationCount);
                        break;
                    default:
                        Debug.Fail("Unsupported rating prediction algorithm.");
                        break;
                }

                this.RunMahoutWrapper(command);
                return this.ReadRatingPredictions(predictionsFileName);
            }
            finally
            {
                if (queryFileName != null)
                {
                    File.Delete(queryFileName);
                }

                if (predictionsFileName != null)
                {
                    File.Delete(predictionsFileName);
                }
            }
        }

        /// <summary>
        /// This query is not supported.
        /// </summary>
        /// <param name="user">The parameter is not used.</param>
        /// <param name="item">The parameter is not used.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>Nothing, since the method always throws.</returns>
        public RatingDistribution PredictDistribution(User user, Item item, DummyFeatureSource featureSource = null)
        {
            throw new NotSupportedException("Uncertain rating prediction is not supported by this recommender.");
        }

        /// <summary>
        /// This query is not supported.
        /// </summary>
        /// <param name="instanceSource">The parameter is not used.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>Nothing, since the method always throws.</returns>
        public IDictionary<User, IDictionary<Item, RatingDistribution>> PredictDistribution(
            TInstanceSource instanceSource, DummyFeatureSource featureSource = null)
        {
            throw new NotSupportedException("Uncertain rating prediction is not supported by this recommender.");
        }

        /// <summary>
        /// This query is not supported.
        /// </summary>
        /// <param name="user">The parameter is not used.</param>
        /// <param name="recommendationCount">The parameter is not used.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>Nothing, since the method always throws.</returns>
        public IEnumerable<Item> Recommend(User user, int recommendationCount, DummyFeatureSource featureSource = null)
        {
            throw new NotSupportedException("Item recommendation is not supported by this recommender.");
        }

        /// <summary>
        /// This query is not supported.
        /// </summary>
        /// <param name="users">The parameter is not used.</param>
        /// <param name="recommendationCount">The parameter is not used.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>Nothing, since the method always throws.</returns>
        public IDictionary<User, IEnumerable<Item>> Recommend(
            IEnumerable<User> users, int recommendationCount, DummyFeatureSource featureSource = null)
        {
            throw new NotSupportedException("Item recommendation is not supported by this recommender.");
        }

        /// <summary>
        /// Recommend items with their rating distributions to a specified user.
        /// </summary>
        /// <param name="user">The user to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend.</param>
        /// <param name="featureSource">The source of features for the specified user.</param>
        /// <returns>The list of recommended items and their rating distributions.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        public IEnumerable<Tuple<Item, RatingDistribution>> RecommendDistribution(
            User user, int recommendationCount, DummyFeatureSource featureSource = null)
        {
            throw new NotSupportedException("Item recommendation with rating distributions is not supported by this recommender.");
        }

        /// <summary>
        /// Recommends items with their rating distributions to a specified list of users.
        /// </summary>
        /// <param name="users">The list of users to recommend items to.</param>
        /// <param name="recommendationCount">Maximum number of items to recommend to a single user.</param>
        /// <param name="featureSource">The source of features for the specified users.</param>
        /// <returns>The list of recommended items and their rating distributions for every user from <paramref name="users"/>.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        public IDictionary<User, IEnumerable<Tuple<Item, RatingDistribution>>> RecommendDistribution(
            IEnumerable<User> users, int recommendationCount, DummyFeatureSource featureSource = null)
        {
            throw new NotSupportedException("Item recommendation with rating distributions is not supported by this recommender.");
        }

        /// <summary>
        /// Returns a list of users related to <paramref name="user"/>.
        /// </summary>
        /// <param name="user">The user for which related users should be found.</param>
        /// <param name="relatedUserCount">The maximum number of related users to return.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>The list of related users.</returns>
        /// <remarks>Only users specified in <see cref="UserSubset"/> will be returned.</remarks>
        public IEnumerable<User> GetRelatedUsers(User user, int relatedUserCount, DummyFeatureSource featureSource = null)
        {
            Debug.Assert(user != null, "A valid user should be provided.");

            if (!this.userToId.ContainsKey(user))
            {
                throw new NotSupportedException("Cold users are not supported by this recommender.");
            }
            
            if (this.lastFindRelatedUsersContext == null || this.lastFindRelatedUsersContext.IsNewContextNeeded(user, relatedUserCount))
            {
                this.lastFindRelatedUsersContext = new LazyFindRelatedUsersContext(this, relatedUserCount);
            }

            return this.lastFindRelatedUsersContext.CreateLazyPredictionResults(user, this.userSubset);
        }

        /// <summary>
        /// This query is not supported.
        /// </summary>
        /// <param name="users">The parameter is not used.</param>
        /// <param name="relatedUserCount">The parameter is not used.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>Nothing, since the method always throws.</returns>
        public IDictionary<User, IEnumerable<User>> GetRelatedUsers(
            IEnumerable<User> users, int relatedUserCount, DummyFeatureSource featureSource = null)
        {
            throw new NotSupportedException("Bulk related user prediction is not supported by this recommender.");
        }

        /// <summary>
        /// Returns a list of items related to <paramref name="item"/>.
        /// </summary>
        /// <param name="item">The item for which related items should be found.</param>
        /// <param name="relatedItemCount">The maximum number of related items to return.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>The list of related items.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> will be returned.</remarks>
        public IEnumerable<Item> GetRelatedItems(Item item, int relatedItemCount, DummyFeatureSource featureSource = null)
        {
            Debug.Assert(item != null, "A valid item should be provided.");

            if (!this.itemToId.ContainsKey(item))
            {
                throw new NotSupportedException("Cold items are not supported by this recommender.");
            }
            
            if (this.lastFindRelatedItemsContext == null || this.lastFindRelatedItemsContext.IsNewContextNeeded(item, relatedItemCount))
            {
                this.lastFindRelatedItemsContext = new LazyFindRelatedItemsContext(this, relatedItemCount);
            }

            return this.lastFindRelatedItemsContext.CreateLazyPredictionResults(item, this.itemSubset);
        }

        /// <summary>
        /// Returns a list of related items to each item in <paramref name="items"/>.
        /// </summary>
        /// <param name="items">The parameter is not used.</param>
        /// <param name="relatedItemCount">The parameter is not used.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>Nothing, since the method always throws.</returns>
        public IDictionary<Item, IEnumerable<Item>> GetRelatedItems(
            IEnumerable<Item> items, int relatedItemCount, DummyFeatureSource featureSource = null)
        {
            throw new NotSupportedException("Bulk related item prediction is not supported by this recommender.");
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Creates a temporary file with a given related entity search queries in the format expected by the Mahout wrapper.
        /// </summary>
        /// <typeparam name="TEntity">The type of an entity (user or item).</typeparam>
        /// <param name="queries">A dictionary mapping a query entity to a list of entities that can possibly
        /// be returned in the result list for the corresponding query.</param>
        /// <param name="entityIdSelector">The mapping from an entity to its identifier.</param>
        /// <returns>The name of the created temporary file with the queries.</returns>
        private string CreateRelatedEntitiesQueryFile<TEntity>(
            IDictionary<TEntity, IEnumerable<TEntity>> queries,
            Func<TEntity, int> entityIdSelector)
        {
            string queryFileName = Path.GetTempFileName();
            using (var writer = new StreamWriter(queryFileName))
            {
                foreach (KeyValuePair<TEntity, IEnumerable<TEntity>> query in queries)
                {
                    writer.Write("{0}", entityIdSelector(query.Key));
                    foreach (TEntity entity in query.Value)
                    {
                        writer.Write(",{0}", entityIdSelector(entity));
                    }

                    writer.WriteLine();
                }    
            }

            return queryFileName;
        }

        /// <summary>
        /// Creates a temporary file with a given rating prediction queries in the format expected by the Mahout wrapper.
        /// </summary>
        /// <param name="observations">A set of observations ratings for which should be predicted.</param>
        /// <returns>The name of the created temporary file with the queries.</returns>
        private string CreateRatingPredictionQueryFile(IEnumerable<RatedUserItem> observations)
        {
            string queryFileName = Path.GetTempFileName();
            using (var writer = new StreamWriter(queryFileName))
            {
                foreach (RatedUserItem observation in observations)
                {
                    writer.WriteLine("{0},{1}", this.userToId[observation.User], this.itemToId[observation.Item]);
                }
            }

            return queryFileName;
        }

        /// <summary>
        /// Saves a given collection of observations to a temporary file in Mahout format.
        /// </summary>
        /// <param name="observations">The collection of observations to save.</param>
        /// <returns>The name of the created temporary file.</returns>
        private string CreateDatasetFile(IEnumerable<RatedUserItem> observations)
        {
            string tempFile = Path.GetTempFileName();
            using (var writer = new StreamWriter(tempFile))
            {
                foreach (RatedUserItem observation in observations)
                {
                    writer.WriteLine("{0},{1},{2}", this.userToId[observation.User], this.itemToId[observation.Item], observation.Rating);
                }
            }

            return tempFile;
        }

        /// <summary>
        /// Reads a related entity prediction file produced by the Mahout wrapper.
        /// </summary>
        /// <typeparam name="TEntity">The type of an entity (user or item).</typeparam>
        /// <param name="predictionsFileName">The name of the file with predictions.</param>
        /// <param name="entityFromIdSelector">The mapping from an identifier to the corresponding entity.</param>
        /// <returns>The read predictions.</returns>
        private Dictionary<TEntity, IEnumerable<TEntity>> ReadRelatedEntityPredictions<TEntity>(
            string predictionsFileName,
            Func<int, TEntity> entityFromIdSelector)
        {
            var result = new Dictionary<TEntity, IEnumerable<TEntity>>();
            using (var reader = new StreamReader(predictionsFileName))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    // Mahout wrapper writes results for each query on a separate line.
                    // Lines have the following format: queryEntityId,relatedEntityId1,...,relatedEntityIdN
                    string[] parts = line.Split(',');
                    TEntity queryEntity = entityFromIdSelector(int.Parse(parts[0]));
                    var queryResults = new List<TEntity>();
                    for (int i = 1; i < parts.Length; ++i)
                    {
                        queryResults.Add(entityFromIdSelector(int.Parse(parts[i])));
                    }

                    result.Add(queryEntity, queryResults);
                }
            }

            return result;
        }

        /// <summary>
        /// Reads a rating prediction file produced by the Mahout wrapper.
        /// </summary>
        /// <param name="predictionsFileName">The name of the file with predictions.</param>
        /// <returns>The read predictions.</returns>
        private Dictionary<User, IDictionary<Item, int>> ReadRatingPredictions(string predictionsFileName)
        {
            var result = new Dictionary<User, IDictionary<Item, int>>();
            using (var reader = new StreamReader(predictionsFileName))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    // Lines have the following format: userId,itemId,rating
                    // Ratings are fractional
                    string[] parts = line.Split(',');
                    User user = this.idToUser[int.Parse(parts[0])];
                    Item item = this.idToItem[int.Parse(parts[1])];
                    double fractionalPredictedRating = double.Parse(parts[2]);

                    // NaN indicates missng prediction which we should fill anyway
                    if (double.IsNaN(fractionalPredictedRating))
                    {
                        switch (this.Settings.MissingRatingPredictionAlgorithm)
                        {
                            // TODO: mean/mode/median shift should be removed as soon as Discrete supports negative values
                            case MahoutMissingRatingPredictionAlgorithm.TakeMean:
                                fractionalPredictedRating = this.trainingSetRatingDistribution.GetMean() + this.starRatingInfo.MinStarRating;
                                break;
                            case MahoutMissingRatingPredictionAlgorithm.TakeMedian:
                                fractionalPredictedRating = this.trainingSetRatingDistribution.GetMedian() + this.starRatingInfo.MinStarRating;
                                break;
                            case MahoutMissingRatingPredictionAlgorithm.TakeMode:
                                fractionalPredictedRating = this.trainingSetRatingDistribution.GetMode() + this.starRatingInfo.MinStarRating;
                                break;
                        }
                    }

                    int rating = Convert.ToInt32(fractionalPredictedRating);
                    IDictionary<Item, int> itemToRating;
                    if (!result.TryGetValue(user, out itemToRating))
                    {
                        itemToRating = new Dictionary<Item, int>();
                        result.Add(user, itemToRating);
                    }

                    itemToRating.Add(item, rating);
                }
            }

            return result;
        }

        /// <summary>
        /// Performs a given set of related user queries using the Mahout wrapper.
        /// </summary>
        /// <param name="queries">The mapping from a query user to a subset of users that can possibly be included in the query results for that user.</param>
        /// <param name="relatedUserCount">The maximum number of related users to return for every user.</param>
        /// <returns>The list of related users for every query user.</returns>
        private IDictionary<User, IEnumerable<User>> GetRelatedUsersImpl(IDictionary<User, IEnumerable<User>> queries, int relatedUserCount)
        {
            string queryFileName = null;
            string predictionsFileName = null;

            try
            {
                predictionsFileName = Path.GetTempFileName();
                queryFileName = CreateRelatedEntitiesQueryFile(queries, u => this.userToId[u]);
                string command = string.Format(
                    "FindRelatedUsers \"{0}\" \"{1}\" \"{2}\" {3} {4}",
                    this.trainingDatasetFile,
                    queryFileName,
                    predictionsFileName,
                    relatedUserCount,
                    this.Settings.RatingSimilarity);
                this.RunMahoutWrapper(command);
                return ReadRelatedEntityPredictions(predictionsFileName, uId => this.idToUser[uId]);
            }
            finally
            {
                if (queryFileName != null)
                {
                    File.Delete(queryFileName);
                }

                if (predictionsFileName != null)
                {
                    File.Delete(predictionsFileName);
                }
            }
        }

        /// <summary>
        /// Performs a given set of related item queries using the Mahout wrapper.
        /// </summary>
        /// <param name="queries">The mapping from a query item to a subset of items that can possibly be included in the query results for that item.</param>
        /// <param name="relatedItemCount">The maximum number of related items to return for every item.</param>
        /// <returns>The list of related items for every query item.</returns>
        private IDictionary<Item, IEnumerable<Item>> GetRelatedItemsImpl(IDictionary<Item, IEnumerable<Item>> queries, int relatedItemCount)
        {
            string queryFileName = null;
            string predictionsFileName = null;

            try
            {
                predictionsFileName = Path.GetTempFileName();
                queryFileName = CreateRelatedEntitiesQueryFile(queries, i => this.itemToId[i]);
                string command = string.Format(
                    "FindRelatedItems \"{0}\" \"{1}\" \"{2}\" {3} {4}",
                    this.trainingDatasetFile,
                    queryFileName,
                    predictionsFileName,
                    relatedItemCount,
                    this.Settings.RatingSimilarity);
                this.RunMahoutWrapper(command);
                return ReadRelatedEntityPredictions(predictionsFileName, iId => this.idToItem[iId]);
            }
            finally
            {
                if (queryFileName != null)
                {
                    File.Delete(queryFileName);
                }

                if (predictionsFileName != null)
                {
                    File.Delete(predictionsFileName);
                }
            }
        }

        /// <summary>
        /// Runs the Mahout wrapper with the specified command line.
        /// </summary>
        /// <param name="mahoutRunnerCommandLine">The command line for the Mahout wrapper.</param>
        private void RunMahoutWrapper(string mahoutRunnerCommandLine)
        {
            string command =
                $"java {(this.Settings.UseX64JVM ? "-d64" : "")} -Xmx{this.Settings.JavaMaxHeapSizeInMb}m -classpath \"{Path.Combine(PathToClasses, "MahoutRunner.jar")}{JarSeparator}{Path.Combine(PathToClasses, "mahout-core-0.8-job.jar")}\" MahoutRunner {mahoutRunnerCommandLine}";
            WrapperUtils.ExecuteExternalCommand(command);
        }

        /// <summary>
        /// Estimates rating distribution in the given set of observations.
        /// </summary>
        /// <param name="observations">The set of observations for which the rating distribution should be estimated.</param>
        /// <returns>The estimated rating distribution.</returns>
        /// <remarks>The support of the resulting distribution is shifted by -MinRating to handle negative rating values.</remarks>
        private Discrete EstimateRatingDistribution(IEnumerable<RatedUserItem> observations)
        {
            // TODO: this code should be rewritten properly (without offseting ratings) as soon as Discrete supports negative values
            var estimator = new DiscreteEstimator(this.starRatingInfo.MaxStarRating - this.starRatingInfo.MinStarRating + 1);
            foreach (RatedUserItem observation in observations)
            {
                estimator.Add(observation.Rating - this.starRatingInfo.MinStarRating);
            }

            var result = Discrete.Uniform(this.starRatingInfo.MaxStarRating - this.starRatingInfo.MinStarRating + 1);
            return estimator.GetDistribution(result);
        }

        #endregion

        #region Lazy list prediction context implementations

        /// <summary>
        /// Base class for aggregating multiple list prediction queries for Mahout.
        /// </summary>
        /// <typeparam name="TQueryEntity">The type of an entity acting as a query.</typeparam>
        /// <typeparam name="TResultEntity">The type of a query result entity.</typeparam>
        private abstract class MahoutLazyListPredictionContextBase<TQueryEntity, TResultEntity>
             : LazyListPredictionContext<TQueryEntity, TResultEntity>
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="MahoutLazyListPredictionContextBase{TQueryEntity,TResultEntity}"/> class.
            /// </summary>
            /// <param name="recommender">The recommender used to actually perform the queries.</param>
            /// <param name="maxPredictionListSize">The maximum size of the prediction list.</param>
            protected MahoutLazyListPredictionContextBase(
                MahoutRecommender<TInstanceSource> recommender, int maxPredictionListSize)
                : base(maxPredictionListSize)
            {
                this.Recommender = recommender;
                this.Queries = new Dictionary<TQueryEntity, IEnumerable<TResultEntity>>();
            }

            /// <summary>
            /// Gets the queries aggregated so far.
            /// </summary>
            protected Dictionary<TQueryEntity, IEnumerable<TResultEntity>> Queries { get; private set; }

            /// <summary>
            /// Gets the recommender used to actually perform the queries.
            /// </summary>
            protected MahoutRecommender<TInstanceSource> Recommender { get; private set; }

            /// <summary>
            /// Appends a given query to the context.
            /// </summary>
            /// <param name="queryEntity">The query to make predictions for.</param>
            /// <param name="possibleResultEntities">The set of entities that can possibly be presented in the prediction.</param>
            protected override void AppendQuery(TQueryEntity queryEntity, IEnumerable<TResultEntity> possibleResultEntities)
            {
                this.Queries.Add(queryEntity, possibleResultEntities.ToList());
            }
        }
        
        /// <summary>
        /// Aggregates multiple related user search queries to execute them in one Mahout call (helps performance a lot).
        /// </summary>
        private class LazyFindRelatedUsersContext : MahoutLazyListPredictionContextBase<User, User>
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="LazyFindRelatedUsersContext"/> class.
            /// </summary>
            /// <param name="recommender">The recommender used to actually perform the queries.</param>
            /// <param name="maxRelatedUserCount">The maximum number of related users to return for every user.</param>
            public LazyFindRelatedUsersContext(
                MahoutRecommender<TInstanceSource> recommender, int maxRelatedUserCount)
                : base(recommender, maxRelatedUserCount)
            {
            }

            /// <summary>
            /// Computes the predictions for all the queries stored in the context.
            /// </summary>
            /// <returns>The list of predictions for every query entity stored in the context.</returns>
            protected override IDictionary<User, IEnumerable<User>> EvaluateContext()
            {
                return this.Recommender.GetRelatedUsersImpl(this.Queries, this.MaxPredictionListSize);
            }
        }

        /// <summary>
        /// Aggregates multiple related item search queries to execute them in one Mahout call (helps performance a lot).
        /// </summary>
        private class LazyFindRelatedItemsContext : MahoutLazyListPredictionContextBase<Item, Item>
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="LazyFindRelatedItemsContext"/> class.
            /// </summary>
            /// <param name="recommender">The recommender used to actually perform the queries.</param>
            /// <param name="maxRelatedItemCount">The maximum number of related items to return for every item.</param>
            public LazyFindRelatedItemsContext(
                MahoutRecommender<TInstanceSource> recommender, int maxRelatedItemCount)
                : base(recommender, maxRelatedItemCount)
            {
            }

            /// <summary>
            /// Computes the predictions for all the queries stored in the context.
            /// </summary>
            /// <returns>The list of predictions for every query entity stored in the context.</returns>
            protected override IDictionary<Item, IEnumerable<Item>> EvaluateContext()
            {
                return this.Recommender.GetRelatedItemsImpl(this.Queries, this.MaxPredictionListSize);
            }
        }

        #endregion
    }
}
