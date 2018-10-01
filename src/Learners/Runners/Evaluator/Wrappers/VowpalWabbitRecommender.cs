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
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    using RatingDistribution = System.Collections.Generic.IDictionary<int, double>;

    /// <summary>
    /// The wrapper for the VW recommender.
    /// </summary>
    /// <remarks>
    /// This wrapper operates on the data model represented by <see cref="RecommenderDataset"/> and uses mapping
    /// only to retrieve list of instances and rating info.
    /// </remarks>
    /// <typeparam name="TInstanceSource">The type of an instance source.</typeparam>
    internal class VowpalWabbitRecommender<TInstanceSource> :
        IRecommender<TInstanceSource, User, Item, int,RatingDistribution, DummyFeatureSource>
    {
        /// <summary>
        /// The relative path to the VW executable.
        /// </summary>
        private static string PathToExe = null;

        /// <summary>
        /// Url of released msi with VW
        /// </summary>
        private static readonly string ReleaseUrl = @"https://github.com/eisber/vowpal_wabbit/releases/download/v8.4.0.3/VowpalWabbit-8.4.0.3.msi";

        /// <summary>
        /// Url of released VW Linux binaries
        /// </summary>
        private static readonly string ReleaseUrlLinux = @"http://finance.yendor.com/ML/VW/Binaries/vw-8.20170920";

        /// <summary>
        /// Url of released VW Linux binaries
        /// </summary>
        private static readonly string PathToInstalledBinMacOS = @"vw";

        /// <summary>
        /// Directory to store temp files during msi extraction
        /// </summary>
        private static readonly string InstallationWorkDir = Path.Combine("Data", "Bin");

        /// <summary>
        /// The mapping used to access the data.
        /// </summary>
        private readonly IRatingRecommenderMapping<TInstanceSource, RatedUserItem, User, Item, int, DummyFeatureSource, Vector> mapping;

        /// <summary>
        /// The items from the training set.
        /// </summary>
        private HashSet<Item> trainingItems;

        /// <summary>
        /// The users from the training set.
        /// </summary>
        private HashSet<User> trainingUsers;

        /// <summary>
        /// The subset of items to recommend items from.
        /// </summary>
        private IEnumerable<Item> itemSubset;

        /// <summary>
        /// The name of the VW model file.
        /// </summary>
        private string modelFileName;

        /// <summary>
        /// The last lazy recommendation context.
        /// </summary>
        private LazyRecommendationContext lastRecommendationContext;

        /// <summary>
        /// Initializes a new instance of the <see cref="VowpalWabbitRecommender{TInstanceSource}"/> class.
        /// </summary>
        /// <param name="mapping">The mapping used to access the data.</param>
        private VowpalWabbitRecommender(
            IRatingRecommenderMapping<TInstanceSource, RatedUserItem, User, Item, int, DummyFeatureSource, Vector> mapping)
        {
            Debug.Assert(mapping != null, "A valid mapping should be provided.");

            this.mapping = mapping;
            this.Settings = new VowpalWabbitRecommenderSettings();
        }

        /// <summary>
        /// Finalizes an instance of the <see cref="VowpalWabbitRecommender{TInstanceSource}"/> class. 
        /// </summary>
        ~VowpalWabbitRecommender()
        {
            // TODO: implement IDisposable pattern?
            if (this.modelFileName != null)
            {
                File.Delete(this.modelFileName);
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="VowpalWabbitRecommender{TInstanceSource}"/> class. Downloads recommender executable if necessary.
        /// </summary>
        /// <param name="mapping">The mapping used to access the data.</param>
        public static VowpalWabbitRecommender<TInstanceSource> Create(
            IRatingRecommenderMapping<TInstanceSource, RatedUserItem, User, Item, int, DummyFeatureSource, Vector> mapping)
        {
            if (string.IsNullOrEmpty(PathToExe))
            {
                switch (WrapperUtils.DetectedOS)
                {
                    case WrapperUtils.OS.Windows:
                        PathToExe = Path.Combine("Data", "Bin", "vw.exe");
                        if (!File.Exists(PathToExe))
                        {
                            // getting msi
                            var msiName = "vwsetup.msi";
                            var tmpDir = Path.Combine(InstallationWorkDir, "tmp");
                            var tmpFile = Path.Combine(tmpDir, msiName);
                            Directory.CreateDirectory(tmpDir);
                            // Github does not support ssl, which is the default in .NET 4.5, anymore
                            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
                            new WebClient().DownloadFile(ReleaseUrl, tmpFile);
                            // extracting msi contents
                            // "/a" means "administrative installation" https://docs.microsoft.com/en-us/windows/desktop/Msi/administrative-installation
                            // it extracts files from msi to TARGETDIR\VowpalWabbit
                            // and creates a smaller msi with the same name in the TARGETDIR
                            WrapperUtils.ExecuteExternalCommand($"msiexec /a \"{Path.GetFullPath(tmpFile)}\" TARGETDIR=\"{Path.GetFullPath(InstallationWorkDir)}\" /q");
                            // copying executable to the correct place
                            File.Copy(Path.Combine(InstallationWorkDir, "VowpalWabbit", "vw.exe"), PathToExe);
                            // deleting temp files
                            Directory.Delete(tmpDir, true);
                            Directory.Delete(Path.Combine(InstallationWorkDir, "VowpalWabbit"), true);
                            File.Delete(Path.Combine(InstallationWorkDir, msiName));
                        }
                        break;
                    case WrapperUtils.OS.Linux:
                        PathToExe = Path.Combine(".", "Data", "Bin", "vw");
                        if (!File.Exists(PathToExe))
                        {
                            new WebClient().DownloadFile(ReleaseUrlLinux, PathToExe);
                        }
                        break;
                    case WrapperUtils.OS.OSX:
                        // Expecting that vw is installed via brew install vowpal-wabbit
                        PathToExe = PathToInstalledBinMacOS;
                        break;
                    default:
                        throw new PlatformNotSupportedException($"Running Vowpal Wabbit recommender is not supported on OS {WrapperUtils.DetectedOSDescription}.");
                }
            }
            return new VowpalWabbitRecommender<TInstanceSource>(mapping);
        }

        /// <summary>
        /// Gets the settings.
        /// </summary>
        public VowpalWabbitRecommenderSettings Settings { get; private set; }

#region ILearner implementation

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
        /// Gets or sets the subset of the users used for related user prediction. It is not supported for this recommender.
        /// </summary>
        public IEnumerable<User> UserSubset
        {
            get { throw new NotSupportedException("User subset specification is not supported by this recommender."); }
            set { throw new NotSupportedException("User subset specification is not supported by this recommender."); }
        }

        /// <summary>
        /// Gets or sets the subset of the items used for related item prediction and recommendation.
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
                
                this.itemSubset = value;
            }
        }

        /// <summary>
        /// Trains the recommender on a given dataset.
        /// </summary>
        /// <param name="instanceSource">The instances of the dataset.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        public void Train(TInstanceSource instanceSource, DummyFeatureSource featureSource = null)
        {
            string trainingDatasetFile = null;
            string cacheFileName = null;

            try
            {
                List<RatedUserItem> trainingInstanceList = this.mapping.GetInstances(instanceSource).ToList();

                // Create temporary files
                trainingDatasetFile = this.CreateDatasetFile(trainingInstanceList);
                cacheFileName = Path.GetTempFileName();
                this.modelFileName = Path.GetTempFileName();

                // Invoke VW to train the model
                string trainingCommand =
                string.Format(
                    "{0} -d \"{1}\" -b {2} -q ui --rank {3} --l1 {4} --l2 {5} --learning_rate {6} --decay_learning_rate {7} --passes {8} --power_t 0 -f \"{9}\" --cache_file \"{10}\" -k",
                    PathToExe,
                    trainingDatasetFile,
                    this.Settings.BitPrecision,
                    this.Settings.TraitCount,
                    this.Settings.L1Regularization,
                    this.Settings.L2Regularization,
                    this.Settings.LearningRate,
                    this.Settings.LearningRateDecay,
                    this.Settings.PassCount,
                    this.modelFileName,
                    cacheFileName);
                WrapperUtils.ExecuteExternalCommand(trainingCommand);

                // Remember users and items used for training
                this.trainingItems = new HashSet<Item>();
                this.trainingUsers = new HashSet<User>();
                foreach (RatedUserItem observation in trainingInstanceList)
                {
                    this.trainingItems.Add(observation.Item);
                    this.trainingUsers.Add(observation.User);
                }

                // Setup item subset
                this.itemSubset = this.trainingItems;
            }
            finally
            {
                if (trainingDatasetFile != null)
                {
                    File.Delete(trainingDatasetFile);
                }

                if (cacheFileName != null)
                {
                    File.Delete(cacheFileName);
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
        public int Predict(User user, Item item, DummyFeatureSource featureSource = null)
        {
            throw new NotSupportedException("Only bulk predictions are supported by this recommender.");
        }

        /// <summary>
        /// Predicts ratings for the instances provided by a given instance source.
        /// </summary>
        /// <param name="instanceSource">The source providing the instances to predict ratings for.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>The predicted ratings.</returns>
        public IDictionary<User, IDictionary<Item, int>> Predict(
            TInstanceSource instanceSource, DummyFeatureSource featureSource = null)
        {
            IDictionary<User, IDictionary<Item, double>> fractionalPredictions = this.PredictFractionalRatings(this.mapping.GetInstances(instanceSource));
            return fractionalPredictions.ToDictionary(
                kv => kv.Key,
                kv => (IDictionary<Item, int>)kv.Value.ToDictionary(kv2 => kv2.Key, kv2 => Convert.ToInt32(kv2.Value)));
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
            throw new NotSupportedException("Uncertain predictions are not supported by this recommender.");
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
            throw new NotSupportedException("Uncertain predictions are not supported by this recommender.");
        }

        /// <summary>
        /// Recommends items to a given user.
        /// </summary>
        /// <param name="user">The user to recommend items to.</param>
        /// <param name="recommendationCount">The maximum number of items to recommend.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>The list of recommended items.</returns>
        /// <remarks>Only items specified in <see cref="ItemSubset"/> can be recommended.</remarks>
        public IEnumerable<Item> Recommend(User user, int recommendationCount, DummyFeatureSource featureSource = null)
        {
            Debug.Assert(user != null, "A valid user should be provided.");

            if (!this.trainingUsers.Contains(user))
            {
                throw new NotSupportedException("Cold users are not supported by this recommender.");
            }
            
            if (this.lastRecommendationContext == null || this.lastRecommendationContext.IsNewContextNeeded(user, recommendationCount))                               
            {
                this.lastRecommendationContext = new LazyRecommendationContext(this, recommendationCount);
            }

            return this.lastRecommendationContext.CreateLazyPredictionResults(user, this.itemSubset);
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
            throw new NotSupportedException("Bulk item recommendation is not supported by this recommender.");
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
        /// This query is not supported.
        /// </summary>
        /// <param name="user">The parameter is not used.</param>
        /// <param name="relatedUserCount">The parameter is not used.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>Nothing, since the method always throws.</returns>
        public IEnumerable<User> GetRelatedUsers(User user, int relatedUserCount, DummyFeatureSource featureSource = null)
        {
            throw new NotSupportedException("Related user prediction is not supported by this recommender");
        }

        /// <summary>
        /// This query is not supported.
        /// </summary>
        /// <param name="users">The parameter is not used.</param>
        /// <param name="relatedUserCount">The parameter is not used.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>Nothing, since the method always throws.</returns>
        public IDictionary<User, IEnumerable<User>> GetRelatedUsers(IEnumerable<User> users, int relatedUserCount, DummyFeatureSource featureSource = null)
        {
            throw new NotSupportedException("Related user prediction is not supported by this recommender");
        }

        /// <summary>
        /// This query is not supported.
        /// </summary>
        /// <param name="item">The item for which related items should be found.</param>
        /// <param name="relatedItemCount">The maximum number of related items to return.</param>
        /// <param name="featureSource">The source of the features for the items. Unused.</param>
        /// <returns>Nothing, since the method always throws.</returns>
        public IEnumerable<Item> GetRelatedItems(Item item, int relatedItemCount, DummyFeatureSource featureSource = null)
        {
            throw new NotSupportedException("Related item prediction is not supported by this recommender");
        }

        /// <summary>
        /// This query is not supported.
        /// </summary>
        /// <param name="items">The parameter is not used.</param>
        /// <param name="relatedItemCount">The parameter is not used.</param>
        /// <param name="featureSource">The parameter is not used.</param>
        /// <returns>Nothing, since the method always throws.</returns>
        public IDictionary<Item, IEnumerable<Item>> GetRelatedItems(
            IEnumerable<Item> items, int relatedItemCount, DummyFeatureSource featureSource = null)
        {
            throw new NotSupportedException("Related item prediction is not supported by this recommender");
        }

#endregion

#region Helpers

        /// <summary>
        /// Reads predictions from a VW prediction file.
        /// </summary>
        /// <param name="fileName">The name of the file to read.</param>
        /// <returns>The read prediction list.</returns>
        private static List<double> ReadPredictions(string fileName)
        {
            var result = new List<double>();
            using (var reader = new StreamReader(fileName))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    result.Add(double.Parse(line));
                }
            }

            return result;
        }

        /// <summary>
        /// Computes fractional rating predictions for a given list of pairs of users and items.
        /// </summary>
        /// <param name="observations">The list of observations containing pairs of users and items to make predictions for.</param>
        /// <returns>The predicted fractional ratings.</returns>
        private Dictionary<User, IDictionary<Item, double>> PredictFractionalRatings(IEnumerable<RatedUserItem> observations)
        {
            string predictionDatasetFile = null;
            string predictionsFile = null;

            try
            {
                List<RatedUserItem> observationList = observations.ToList();

                // Create temporary files
                predictionDatasetFile = this.CreateDatasetFile(observationList);
                predictionsFile = Path.GetTempFileName();

                // Invoke VW to compute the fractional rating predictions
                string predictionCommand = string.Format(
                   "{0} -d \"{1}\" -i {2} -t -p {3}",
                   PathToExe,
                   predictionDatasetFile,
                   this.modelFileName,
                   predictionsFile);
                WrapperUtils.ExecuteExternalCommand(predictionCommand);
                
                // Read the predictions from file
                List<double> predictions = ReadPredictions(predictionsFile);

                // Pair predicted ratings with users and items
                var result = new Dictionary<User, IDictionary<Item, double>>();
                int observationIndex = 0;
                foreach (RatedUserItem observation in observationList)
                {
                    double predictedRating = predictions[observationIndex++];
                    IDictionary<Item, double> itemToRating;
                    if (!result.TryGetValue(observation.User, out itemToRating))
                    {
                        itemToRating = new Dictionary<Item, double>();
                        result.Add(observation.User, itemToRating);
                    }

                    itemToRating.Add(observation.Item, predictedRating);
                }

                return result;
            }
            finally
            {
                if (predictionDatasetFile != null)
                {
                    File.Delete(predictionDatasetFile);
                }

                if (predictionsFile != null)
                {
                    File.Delete(predictionsFile);
                }
            }
        }

        /// <summary>
        /// Saves a given collection of observation to a temporary file in VW format.
        /// </summary>
        /// <param name="observations">The collection of observations to save.</param>
        /// <returns>The name of the created temporary file containing the observations.</returns>
        private string CreateDatasetFile(IEnumerable<RatedUserItem> observations)
        {
            string tempFile = Path.GetTempFileName();
            using (var writer = new StreamWriter(tempFile))
            {
                foreach (RatedUserItem observation in observations)
                {
                    writer.Write("{0} ", observation.Rating);
                    
                    writer.Write("|u id_{0} ", observation.User.Id);
                    if (this.Settings.UseUserFeatures)
                    {
                        this.WriteFeatures(writer, observation.User.Features);
                    }

                    writer.Write("|i id_{0} ", observation.Item.Id);
                    if (this.Settings.UseItemFeatures)
                    {
                        this.WriteFeatures(writer, observation.Item.Features);
                    }

                    writer.WriteLine();
                }
            }

            return tempFile;
        }

        /// <summary>
        /// Writes a feature vector to a stream in VW format.
        /// </summary>
        /// <param name="writer">The stream to write features to.</param>
        /// <param name="features">The feature vector.</param>
        /// <remarks>Only non-zero elements of a feature vector are written.</remarks>
        private void WriteFeatures(StreamWriter writer, Vector features)
        {
            if (features == null)
            {
                return;
            }

            const double Tolerance = 1e-6;
            IEnumerable<ValueAtIndex<double>> nonZeroFeatures = features.FindAll(x => Math.Abs(x) >= Tolerance).ToArray();
            foreach (var valueIndex in nonZeroFeatures)
            {
                writer.Write("f{0}:{1} ", valueIndex.Index, valueIndex.Value);
            }
        }

#endregion

#region Lazy list prediction context implementations

        /// <summary>
        /// Aggregates multiple item recommendation queries to execute them in one VW call (helps performance a lot).
        /// </summary>
        private class LazyRecommendationContext : LazyListPredictionContext<User, Item>
        {
            /// <summary>
            /// The list of pairs of users and items which need ratings in order to perform recommendation.
            /// </summary>
            private readonly List<RatedUserItem> observationList = new List<RatedUserItem>();

            /// <summary>
            /// The recommender used to actually execute the queries.
            /// </summary>
            private readonly VowpalWabbitRecommender<TInstanceSource> recommender;

            /// <summary>
            /// Initializes a new instance of the <see cref="LazyRecommendationContext"/> class.
            /// </summary>
            /// <param name="recommender">The recommender.</param>
            /// <param name="recommendationCount">The maximum number of items to recommend.</param>
            public LazyRecommendationContext(
                VowpalWabbitRecommender<TInstanceSource> recommender, int recommendationCount)
                : base(recommendationCount)
            {
                this.recommender = recommender;
            }

            /// <summary>
            /// Appends a given query to the context.
            /// </summary>
            /// <param name="queryEntity">The query to make predictions for.</param>
            /// <param name="possibleResultEntities">The set of entities that can possibly be presented in the prediction.</param>
            protected override void AppendQuery(User queryEntity, IEnumerable<Item> possibleResultEntities)
            {
                this.observationList.AddRange(possibleResultEntities.Select(i => new RatedUserItem(queryEntity, i, 0)));
            }

            /// <summary>
            /// Computes the predictions for all the queries stored in the context.
            /// </summary>
            /// <returns>The list of predictions for every query entity stored in the context.</returns>
            protected override IDictionary<User, IEnumerable<Item>> EvaluateContext()
            {
                IDictionary<User, IDictionary<Item, double>> fractionalPredictions = this.recommender.PredictFractionalRatings(this.observationList);
                return fractionalPredictions.ToDictionary(
                    kv => kv.Key,
                    kv => kv.Value.OrderByDescending(kv2 => kv2.Value).Take(this.MaxPredictionListSize).Select(kv2 => kv2.Key));
            }
        }

#endregion
    }
}
