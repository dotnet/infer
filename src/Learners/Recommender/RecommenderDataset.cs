// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// The dataset for the recommendation.
    /// </summary>
    public class RecommenderDataset
    {
        /// <summary>
        /// The list of (user, item, rating) triples in the dataset.
        /// </summary>
        private readonly List<RatedUserItem> observations = new List<RatedUserItem>();

        /// <summary>
        /// The mapping from string user ids to user object instances.
        /// </summary>
        private readonly Dictionary<string, User> idToUser = new Dictionary<string, User>();

        /// <summary>
        /// The mapping from string item ids to item object instances.
        /// </summary>
        private readonly Dictionary<string, Item> idToItem = new Dictionary<string, Item>();

        /// <summary>
        /// Initializes a new instance of the <see cref="RecommenderDataset"/> class.
        /// </summary>
        /// <param name="observations">The list of observations to create the dataset from.</param>
        /// <param name="starRatingInfo">The information about ratings in the dataset.</param>
        public RecommenderDataset(IEnumerable<RatedUserItem> observations, IStarRatingInfo<int> starRatingInfo)
        {
            if (starRatingInfo == null)
            {
                throw new ArgumentNullException(nameof(starRatingInfo));
            }

            this.StarRatingInfo = starRatingInfo;
            foreach (RatedUserItem observation in observations)
            {
                this.AddObservation(observation);
            }
        }

        /// <summary>
        /// Prevents a default instance of the <see cref="RecommenderDataset"/> class from being created without factory.
        /// </summary>
        private RecommenderDataset()
        {
        }

        /// <summary>
        /// Gets the information about ratings in the dataset.
        /// </summary>
        public IStarRatingInfo<int> StarRatingInfo { get; private set; }

        /// <summary>
        /// Gets the observations.
        /// </summary>
        public ReadOnlyCollection<RatedUserItem> Observations
        {
            get { return this.observations.AsReadOnly(); }
        }

        /// <summary>
        /// Gets the list of all users in the dataset.
        /// </summary>
        public IEnumerable<User> Users
        {
            get { return this.idToUser.Values; }
        }

        /// <summary>
        /// Gets the list of all items in the dataset.
        /// </summary>
        public IEnumerable<Item> Items
        {
            get { return this.idToItem.Values; }
        }

        /// <summary>
        /// Loads dataset from a given file.
        /// <para>
        /// Data file format:
        /// Row starting with 'R' describes min and max ratings and has form 'R,Min,Max'.
        /// Rows starting with 'U' describe a single user and have form 'U,UserId,UserFeatures'.
        /// Rows starting with 'I' describe a single item and have form 'I,ItemId,ItemFeatures'.
        /// Rows other than that describe instances and should have form 'UserID,ItemID,Rating'.
        /// Feature description has form 'FeatureIndex1:Value1|FeatureIndex2:Value2|...'
        /// If all the user features are zero or there are no user features in the dataset at all, the user description can be omitted. Same is true for items.
        /// </para>
        /// </summary>
        /// <param name="fileName">File to load data from.</param>
        /// <returns>The loaded dataset.</returns>
        public static RecommenderDataset Load(string fileName)
        {
            var rawObservations = new List<Tuple<string, string, int>>();
            var userIdToFeatures = new Dictionary<string, Vector>();
            var itemIdToFeatures = new Dictionary<string, Vector>();
            int? minRating = null, maxRating = null;
            int userFeatureCount = 0, itemFeatureCount = 0;

            var parsingContext = new FileParsingContext(fileName);
            using (var reader = new StreamReader(fileName))
            {
                string line;
                bool isFirstRecord = true;
                while ((line = reader.ReadLine()) != null)
                {
                    parsingContext.NextLine(line);
                    if (line.Length == 0 || line.StartsWith("#"))
                    {
                        continue; // Skip comments and empty lines
                    }

                    string[] splits = line.Split(',');

                    if (isFirstRecord)
                    {
                        //// Parse rating record

                        int minRatingValue = 0, maxRatingValue = 0;
                        if (splits.Length != 3 || 
                            splits[0].Trim() != "R" || 
                            !int.TryParse(splits[1], out minRatingValue) || 
                            !int.TryParse(splits[2], out maxRatingValue))
                        {
                            parsingContext.RaiseError("Invalid rating info record.");
                        }

                        minRating = minRatingValue;
                        maxRating = maxRatingValue;
                        isFirstRecord = false;
                    }
                    else if (splits[0].Trim() == "U")
                    {
                        //// Parse user record
                        
                        if (splits.Length != 3)
                        {
                            parsingContext.RaiseError("Invalid user record.");
                        }

                        string userId = splits[1].Trim();
                        if (userIdToFeatures.ContainsKey(userId))
                        {
                            parsingContext.RaiseError("Record describing user '{0}' is presented more than once.", userId);
                        }

                        Vector features = ParseFeatures(splits[2], parsingContext, ref userFeatureCount);
                        userIdToFeatures.Add(userId, features);
                    }
                    else if (splits[0].Trim() == "I")
                    {
                        //// Parse item record
                        
                        if (splits.Length != 3)
                        {
                            parsingContext.RaiseError("Invalid item record.");
                        }

                        string itemId = splits[1].Trim();
                        if (itemIdToFeatures.ContainsKey(itemId))
                        {
                            parsingContext.RaiseError("Record describing item '{0}' is presented more than once.", itemId);
                        }

                        Vector features = ParseFeatures(splits[2], parsingContext, ref itemFeatureCount);
                        itemIdToFeatures.Add(itemId, features);
                    }
                    else
                    {
                        //// Parse instance record
                        
                        string userId = splits[0].Trim();
                        string itemId = splits[1].Trim();
                        int rating = 0;
                        if (splits.Length != 3 || !int.TryParse(splits[2], out rating))
                        {
                            parsingContext.RaiseError("Invalid instance record.", line);
                        }

                        rawObservations.Add(Tuple.Create(userId, itemId, rating));
                    }
                }
            }

            if (!minRating.HasValue)
            {
                parsingContext.RaiseGlobalError("Rating info is missing.");
            }

            var result = new RecommenderDataset { StarRatingInfo = new StarRatingInfo(minRating.Value, maxRating.Value) };
            foreach (var observation in rawObservations)
            {
                string userId = observation.Item1;
                string itemId = observation.Item2;
                int rating = observation.Item3;

                if (rating < minRating.Value || rating > maxRating.Value)
                {
                    parsingContext.RaiseGlobalError("One of the ratings is inconsistent with the specified rating info.");
                }

                User user = RetrieveEntity(userId, result.idToUser, userIdToFeatures, userFeatureCount, (id, features) => new User(id, features));
                Item item = RetrieveEntity(itemId, result.idToItem, itemIdToFeatures, itemFeatureCount, (id, features) => new Item(id, features));
                result.observations.Add(new RatedUserItem(user, item, rating));
            }

            return result;
        }

        /// <summary>
        /// Saves the dataset to a specified file.
        /// </summary>
        /// <param name="fileName">The name of the dataset.</param>
        public void Save(string fileName)
        {
            using (var writer = new StreamWriter(fileName))
            {
                writer.WriteLine("R,{0},{1}", this.StarRatingInfo.MinStarRating, this.StarRatingInfo.MaxStarRating);
                
                // Save users
                foreach (var user in this.idToUser.Values)
                {
                    WriteFeatures(writer, "U", user.Id, user.Features);
                }

                // Save items
                foreach (var item in this.idToItem.Values)
                {
                    WriteFeatures(writer, "I", item.Id, item.Features);
                }
                
                // Save observations
                foreach (RatedUserItem observation in this.observations)
                {
                    writer.WriteLine("{0},{1},{2}", observation.User.Id, observation.Item.Id, observation.Rating);
                }
            }
        }

        /// <summary>
        /// Writes a given entity feature vector to a stream.
        /// </summary>
        /// <param name="writer">The writer of the stream.</param>
        /// <param name="typeString">A string identifying type of the entity that owns features.</param>
        /// <param name="id">The identifier of an entity that owns features.</param>
        /// <param name="features">The feature vector. If null, nothing will be written.</param>
        private static void WriteFeatures(StreamWriter writer, string typeString, string id, Vector features)
        {
            Debug.Assert(typeString == "U" || typeString == "I", "A valid type string should be provided.");
            Debug.Assert(writer != null, "A valid writer should be provided.");
            Debug.Assert(!string.IsNullOrEmpty(id), "A valid id should be provided.");
            
            if (features == null)
            {
                return;
            }

            const double Tolerance = 1e-6;
            ValueAtIndex<double>[] nonZeroFeatures = features.FindAll(x => Math.Abs(x) >= Tolerance).ToArray();
            if (nonZeroFeatures.Length == 0)
            {
                return;
            }

            writer.Write("{0},{1},", typeString, id);
            for (int i = 0; i < nonZeroFeatures.Length; ++i)
            {
                writer.Write("{0}:{1}", nonZeroFeatures[i].Index, nonZeroFeatures[i].Value);
                if (i != nonZeroFeatures.Length - 1)
                {
                    writer.Write("|");
                }
            }

            writer.WriteLine();
        }

        /// <summary>
        /// Retrieves a new user or item by its id from the pool. If it was not in the pool, it will be created.
        /// </summary>
        /// <typeparam name="TEntity">Type of entity (<see cref="User"/> or <see cref="Item"/>).</typeparam>
        /// <param name="entityId">The id of the entity.</param>
        /// <param name="entityPool">The pool of already created entities.</param>
        /// <param name="entityIdToFeatures">The dictionary mapping entity identifiers to feature vectors.</param>
        /// <param name="featureCount">The count of the features for this type of entity in the dataset.</param>
        /// <param name="entityFactory">The factory to create new entities.</param>
        /// <returns>The retrieved entity.</returns>
        private static TEntity RetrieveEntity<TEntity>(
            string entityId,
            Dictionary<string, TEntity> entityPool,
            Dictionary<string, Vector> entityIdToFeatures,
            int featureCount,
            Func<string, Vector, TEntity> entityFactory)
        {
            TEntity entity;
            if (entityPool.TryGetValue(entityId, out entity))
            {
                return entity;
            }

            Vector features;
            if (!entityIdToFeatures.TryGetValue(entityId, out features))
            {
                features = SparseVector.Zero(featureCount);
            }
            else
            {
                // Make all the feature vectors have the same length
                features = features.Append(SparseVector.Zero(featureCount - features.Count));
            }

            entity = entityFactory(entityId, features);
            entityPool.Add(entityId, entity);

            return entity;
        }

        /// <summary>
        /// Parses a string describing a list of features.
        /// </summary>
        /// <param name="featureString">The string containing the list of features.</param>
        /// <param name="parsingContext">The file parsing context.</param>
        /// <param name="featureCount">The number of features in the dataset, which would be updated from the parsed feature indices.</param>
        /// <returns>A sparse array of features extracted from <paramref name="featureString"/>.</returns>
        private static Vector ParseFeatures(string featureString, FileParsingContext parsingContext, ref int featureCount)
        {
            Debug.Assert(featureString != null, "A valid feature string should be specified.");

            var featureIndexToValue = new SortedDictionary<int, double>();
            string[] featureDescriptions = featureString.Split('|');
            foreach (string featureDescription in featureDescriptions)
            {
                if (featureDescription.Trim().Length == 0)
                {
                    continue;
                }

                string[] featureDescriptionParts = featureDescription.Split(':');
                int featureIndex = 0;
                double featureValue = 0;
                if (featureDescriptionParts.Length != 2 ||
                    !int.TryParse(featureDescriptionParts[0], out featureIndex) ||
                    !double.TryParse(featureDescriptionParts[1], out featureValue))
                {
                    parsingContext.RaiseError("Invalid feature description string.");
                }

                if (featureIndexToValue.ContainsKey(featureIndex))
                {
                    parsingContext.RaiseError("Feature {0} is referenced several times.", featureIndex);
                }

                featureIndexToValue.Add(featureIndex, featureValue);
                featureCount = Math.Max(featureCount, featureIndex + 1);
            }

            return SparseVector.FromSparseValues(featureCount, 0, featureIndexToValue.Select(kv => new ValueAtIndex<double>(kv.Key, kv.Value)).ToList());
        }

        /// <summary>
        /// Adds an observation to the dataset. 
        /// </summary>
        /// <param name="ratedUserItem">An observation to be added</param> 
        private void AddObservation(RatedUserItem ratedUserItem)
        {
            Debug.Assert(ratedUserItem.Rating >= this.StarRatingInfo.MinStarRating && ratedUserItem.Rating <= this.StarRatingInfo.MaxStarRating, "The rating value is not valid.");
            
            if (!this.idToItem.ContainsKey(ratedUserItem.Item.Id))
            {
                this.idToItem.Add(ratedUserItem.Item.Id, ratedUserItem.Item);
            }

            if (!this.idToUser.ContainsKey(ratedUserItem.User.Id))
            {
                this.idToUser.Add(ratedUserItem.User.Id, ratedUserItem.User);
            }

            this.observations.Add(ratedUserItem);
        }
    }
}
