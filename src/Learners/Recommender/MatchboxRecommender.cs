// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Runtime.Serialization.Json;
    using System.Text;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal;
    using Microsoft.ML.Probabilistic.Learners.Runners;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Serialization;

    using RatingDistribution = System.Collections.Generic.IDictionary<int, double>;
    
    /// <summary>
    /// Matchbox recommender factory.
    /// </summary>
    public static class MatchboxRecommender
    {
        #region Creation
        
        /// <summary>
        /// Creates a Matchbox recommender from a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The recommender instance.</returns>
        public static IMatchboxRecommender<TInstanceSource, int, int, Discrete, TFeatureSource>
            Create<TInstanceSource, TFeatureSource>(
                IMatchboxRecommenderMapping<TInstanceSource, TFeatureSource> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new NativeDataFormatMatchboxRecommender<TInstanceSource, TFeatureSource>(mapping);
        }

        /// <summary>
        /// Creates a recommender from a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TRating">The type of a rating.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <param name="writeUser">Write a user.</param>
        /// <param name="writeItem">Write an item.</param>
        /// <returns>The recommender instance.</returns>
        public static IMatchboxRecommender<TInstanceSource, TUser, TItem, RatingDistribution, TFeatureSource>
            Create<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(
                IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, Vector> mapping, Action<IWriter, TUser> writeUser, Action<IWriter, TItem> writeItem)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new StandardDataFormatMatchboxRecommender<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(mapping, writeUser, writeItem);
        }

        #endregion

        #region .NET binary deserialization

        /// <summary>
        /// Deserializes a Matchbox recommender from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of instances.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TDataRating">The type of data rating.</typeparam>
        /// <typeparam name="TRatingDistribution">The type of a distribution over ratings.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <param name="readUser">Read a user.</param>
        /// <param name="writeUser">Write a user.</param>
        /// <param name="readItem">Read an item.</param>
        /// <param name="writeItem">Write an item.</param>
        /// <returns>The deserialized recommender object.</returns>
        public static IMatchboxRecommender<TInstanceSource, TUser, TItem, TRatingDistribution, TFeatureSource>
            Load<TInstanceSource, TInstance, TUser, TItem, TDataRating, TRatingDistribution, TFeatureSource>(string fileName, Func<IReader, TUser> readUser, Action<IWriter, TUser> writeUser, Func<IReader, TItem> readItem, Action<IWriter, TItem> writeItem) =>
            ReadFile(
                fileName,
                reader => LoadMatchboxRecommender<TInstanceSource, TInstance, TUser, TItem, TDataRating, TRatingDistribution, TFeatureSource>(reader, readUser, writeUser, readItem, writeItem));

        private static T ReadFile<T>(string fileName, Func<IReader, T> read)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            using (Stream stream = File.Open(fileName, FileMode.Open))
            using (var reader = new StreamReader(stream))
            {
                var textReader = new WrappedTextReader(reader);
                return read(textReader);
            }
        }

        private static IMatchboxRecommender<TInstanceSource, TUser, TItem, TRatingDistribution, TFeatureSource>
            LoadMatchboxRecommender<TInstanceSource, TInstance, TUser, TItem, TDataRating, TRatingDistribution, TFeatureSource>(IReader reader, Func<IReader, TUser> readUser, Action<IWriter, TUser> writeUser, Func<IReader, TItem> readItem, Action<IWriter, TItem> writeItem)
        {
            var type = reader.ReadString();
            switch (type)
            {
                case nameof(NativeDataFormatMatchboxRecommender<TInstanceSource, TFeatureSource>):
                    {
                        var mapping = LoadMatchboxRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TDataRating, TFeatureSource>(reader);
                        return (IMatchboxRecommender<TInstanceSource, TUser, TItem, TRatingDistribution, TFeatureSource>)new NativeDataFormatMatchboxRecommender<TInstanceSource, TFeatureSource>(reader, mapping);
                    }
                case nameof(StandardDataFormatMatchboxRecommender<TInstanceSource, TInstance, TUser, TItem, TDataRating, TFeatureSource>):
                    {
                        var topLevelMapping = LoadStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TDataRating, TFeatureSource, Vector>(reader);
                        return (IMatchboxRecommender<TInstanceSource, TUser, TItem, TRatingDistribution, TFeatureSource>)new StandardDataFormatMatchboxRecommender<TInstanceSource, TInstance, TUser, TItem, TDataRating, TFeatureSource>(reader, topLevelMapping, readUser, writeUser, readItem, writeItem);
                    }
                default:
                    throw new InvalidOperationException($"Unrecognised type: {type}.");
            }
        }

        private static IMatchboxRecommenderMapping<TInstanceSource, TFeatureSource> LoadMatchboxRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(IReader reader)
        {
            var type = reader.ReadString();
            switch (type)
            {
                case nameof(NativeRecommenderTestMapping):
                    return (IMatchboxRecommenderMapping<TInstanceSource, TFeatureSource>)new NativeRecommenderTestMapping(reader);
                case nameof(StandardDataFormatMatchboxRecommender<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>.NativeRecommenderMapping):
                    {
                        var topLevelMapping = LoadStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, Vector>(reader);
                        return (IMatchboxRecommenderMapping<TInstanceSource, TFeatureSource>)new StandardDataFormatMatchboxRecommender<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>.NativeRecommenderMapping(reader, topLevelMapping);
                    }
                default:
                    throw new InvalidOperationException($"Unrecognised type: {type}.");
            }
        }

        private static IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues> LoadStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>(IReader reader)
        {
            var type = reader.ReadString();
            switch (type)
            {
                case nameof(CsvMapping):
                    return (IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>)new CsvMapping();
                case nameof(StandardRecommenderTestMapping):
                    return (IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>)new StandardRecommenderTestMapping(reader);
                case nameof(StandardRecommenderTestMappingItemFeaturesOnly):
                    return (IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>)new StandardRecommenderTestMappingItemFeaturesOnly();
                case nameof(StandardRecommenderTestMappingUserFeaturesOnly):
                    return (IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>)new StandardRecommenderTestMappingUserFeaturesOnly();
                case nameof(TestStarRatingRecommenderMapping):
                    return (IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>)new TestStarRatingRecommenderMapping();
                case nameof(RunnerStarRatingRecommenderMapping):
                    return (IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>)new RunnerStarRatingRecommenderMapping();
                case nameof(NegativeDataGeneratorMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>):
                    {
                        var mapping = LoadRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>(reader);
                        return (IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>)new NegativeDataGeneratorMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>(reader, mapping);
                    }
                default:
                    throw new InvalidOperationException($"Unrecognised type: {type}.");
            }
        }

        private static IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues> LoadRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>(IReader reader)
        {
            var type = reader.ReadString();
            switch (type)
            {
                case nameof(PositiveOnlyDataMapping):
                    return (IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>)new PositiveOnlyDataMapping();
                case nameof(TestMapping):
                    return (IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>)new TestMapping();
                case nameof(NegativeDataGeneratorMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>):
                    {
                        var mapping = LoadRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>(reader);
                        return (IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>)new NegativeDataGeneratorMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>(reader, mapping);
                    }
                case nameof(TestStarRatingRecommenderMapping):
                    return (IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>)new TestStarRatingRecommenderMapping();
                case nameof(StandardRecommenderTestMappingItemFeaturesOnly):
                    return (IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>)new StandardRecommenderTestMappingItemFeaturesOnly();
                case nameof(StandardRecommenderTestMappingUserFeaturesOnly):
                    return (IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>)new StandardRecommenderTestMappingUserFeaturesOnly();
                case nameof(StandardRecommenderTestMapping):
                    return (IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>)new StandardRecommenderTestMapping(reader);
                case nameof(CsvMapping):
                    return (IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>)new CsvMapping();
                case nameof(RunnerStarRatingRecommenderMapping):
                    return (IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>)new RunnerStarRatingRecommenderMapping();
                default:
                    throw new InvalidOperationException($"Unrecognised type: {type}.");
            }
        }

        /// <summary>
        /// Represents a dataset in the native Matchbox format.
        /// </summary>
        internal class NativeDataset
        {
            /// <summary>
            /// Gets or sets the observed user ids.
            /// </summary>
            public int[] UserIds { get; set; }

            /// <summary>
            /// Gets or sets the observed item ids.
            /// </summary>
            public int[] ItemIds { get; set; }

            /// <summary>
            /// Gets or sets the observed ratings.
            /// </summary>
            public int[] Ratings { get; set; }

            /// <summary>
            /// Gets or sets the total number of users.
            /// </summary>
            public int UserCount { get; set; }

            /// <summary>
            /// Gets or sets the total number of items.
            /// </summary>
            public int ItemCount { get; set; }

            /// <summary>
            /// Gets or sets the array of non-zero user feature values.
            /// </summary>
            public double[][] NonZeroUserFeatureValues { get; set; }

            /// <summary>
            /// Gets or sets the array of non-zero user feature indices.
            /// </summary>
            public int[][] NonZeroUserFeatureIndices { get; set; }

            /// <summary>
            /// Gets or sets the array of non-zero item feature values.
            /// </summary>
            public double[][] NonZeroItemFeatureValues { get; set; }

            /// <summary>
            /// Gets or sets the array of non-zero item feature indices.
            /// </summary>
            public int[][] NonZeroItemFeatureIndices { get; set; }
        }

        /// <summary>
        /// An implementation of <see cref="IMatchboxRecommenderMapping{TInstanceSource, TFeatureSource}"/> for <see cref="NativeDataset"/>.
        /// </summary>
        /// <remarks>This mapping doesn't support batching.</remarks>
        [Serializable]
        internal class NativeRecommenderTestMapping : IMatchboxRecommenderMapping<NativeDataset, NativeDataset>, ICustomSerializable
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="NativeRecommenderTestMapping"/> class.
            /// </summary>
            public NativeRecommenderTestMapping()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="NativeRecommenderTestMapping"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            public NativeRecommenderTestMapping(IReader reader)
            {
                if (reader == null)
                {
                    throw new ArgumentNullException(nameof(reader));
                }

                // Nothing to deserialize
            }

            /// <summary>
            /// Gets the list of user identifiers from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get the user identifiers from.</param>
            /// <param name="batchNumber">The number of the current batch (used only if the data is divided into batches).</param>
            /// <returns>The list of user identifiers.</returns>
            public IReadOnlyList<int> GetUserIds(NativeDataset instanceSource, int batchNumber = 0)
            {
                if (batchNumber != 0)
                {
                    throw new NotSupportedException("This mapping does not support data batching.");
                }

                return instanceSource.UserIds;
            }

            /// <summary>
            /// Gets the list of item identifiers from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get the item identifiers from.</param>
            /// <param name="batchNumber">The number of the current batch (used only if the data is divided into batches).</param>
            /// <returns>The list of item identifiers.</returns>
            public IReadOnlyList<int> GetItemIds(NativeDataset instanceSource, int batchNumber = 0)
            {
                if (batchNumber != 0)
                {
                    throw new NotSupportedException("This mapping does not support data batching.");
                }

                return instanceSource.ItemIds;
            }

            /// <summary>
            /// Gets the list of ratings from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get the ratings from.</param>
            /// <param name="batchNumber">The number of the current batch (used only if the data is divided into batches).</param>
            /// <returns>The list of ratings</returns>
            public IReadOnlyList<int> GetRatings(NativeDataset instanceSource, int batchNumber = 0)
            {
                if (batchNumber != 0)
                {
                    throw new NotSupportedException("This mapping does not support data batching.");
                }

                return instanceSource.Ratings;
            }

            /// <summary>
            /// Gets the number of users from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get number of users from.</param>
            /// <returns>The number of users.</returns>
            public int GetUserCount(NativeDataset instanceSource)
            {
                return instanceSource.UserCount;
            }

            /// <summary>
            /// Gets the number of items from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get number of items from.</param>
            /// <returns>The number of items.</returns>
            public int GetItemCount(NativeDataset instanceSource)
            {
                return instanceSource.ItemCount;
            }

            /// <summary>
            /// Gets the number of star ratings.
            /// This is equal to one plus the difference between the maximum and the minimum rating.
            /// </summary>
            /// <param name="instanceSource">The source of instances to get number of items from.</param>
            /// <returns>The number of ratings.</returns>
            public int GetRatingCount(NativeDataset instanceSource)
            {
                return 6;
            }

            /// <summary>
            /// Gets the number of user features.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <returns>The number of user features.</returns>
            public int GetUserFeatureCount(NativeDataset featureSource)
            {
                return featureSource.NonZeroUserFeatureIndices.Max(
                    indices => indices == null || indices.Length == 0 ? 0 : indices.Max() + 1);
            }

            /// <summary>
            /// Gets the number of item features.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <returns>The number of item features.</returns>
            public int GetItemFeatureCount(NativeDataset featureSource)
            {
                return featureSource.NonZeroItemFeatureIndices.Max(
                    indices => indices == null || indices.Length == 0 ? 0 : indices.Max() + 1);
            }

            /// <summary>
            /// Gets non-zero feature values for all users present in a given feature source.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <returns>An array of non-zero user feature arrays where the outer array is indexed by user id.</returns>
            /// <remarks>This function will be called during training if the user feature support is enabled.</remarks>
            public IReadOnlyList<IReadOnlyList<double>> GetAllUserNonZeroFeatureValues(NativeDataset featureSource)
            {
                return featureSource.NonZeroUserFeatureValues;
            }

            /// <summary>
            /// Gets non-zero feature indices for all users present in a given feature source.
            /// </summary>
            /// <param name="featureSource">The source to obtain feature indices from.</param>
            /// <returns>An array of non-zero user feature index arrays where the outer array is indexed by user id.</returns>
            /// <remarks>This function will be called during training if the user feature support is enabled.</remarks>
            public IReadOnlyList<IReadOnlyList<int>> GetAllUserNonZeroFeatureIndices(NativeDataset featureSource)
            {
                return featureSource.NonZeroUserFeatureIndices;
            }

            /// <summary>
            /// Gets non-zero feature values for all items present in a given feature source.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <returns>An array of non-zero item feature arrays where the outer array is indexed by item id</returns>
            /// <remarks>This function will be called during training if the item feature support is enabled.</remarks>
            public IReadOnlyList<IReadOnlyList<double>> GetAllItemNonZeroFeatureValues(NativeDataset featureSource)
            {
                return featureSource.NonZeroItemFeatureValues;
            }

            /// <summary>
            /// Gets non-zero feature indices for all items present in a given feature source.
            /// </summary>
            /// <param name="featureSource">The source to obtain feature indices from.</param>
            /// <returns>An array of non-zero item feature index arrays where the outer array is indexed by item id</returns>
            /// <remarks>This function will be called during training if the item feature support is enabled.</remarks>
            public IReadOnlyList<IReadOnlyList<int>> GetAllItemNonZeroFeatureIndices(NativeDataset featureSource)
            {
                return featureSource.NonZeroItemFeatureIndices;
            }

            /// <summary>
            /// Gets non-zero feature values for a given user.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <param name="userId">The user identifier.</param>
            /// <returns>Non-zero feature values for the user.</returns>
            /// <remarks>This function will be called during prediction for cold users if the user feature support is enabled.</remarks>
            public IReadOnlyList<double> GetSingleUserNonZeroFeatureValues(NativeDataset featureSource, int userId)
            {
                return featureSource.NonZeroUserFeatureValues[userId];
            }

            /// <summary>
            /// Gets non-zero feature indices for a given user.
            /// </summary>
            /// <param name="featureSource">The source to obtain feature indices from.</param>
            /// <param name="userId">The user identifier.</param>
            /// <returns>Non-zero feature indices for the user.</returns>
            /// <remarks>This function will be called during prediction for cold users if the user feature support is enabled.</remarks>
            public IReadOnlyList<int> GetSingleUserNonZeroFeatureIndices(NativeDataset featureSource, int userId)
            {
                return featureSource.NonZeroUserFeatureIndices[userId];
            }

            /// <summary>
            /// Gets non-zero feature values for a given item.
            /// </summary>
            /// <param name="featureSource">The source to obtain features from.</param>
            /// <param name="itemId">The item identifier.</param>
            /// <returns>Non-zero feature values for the item.</returns>
            /// <remarks>This function will be called during prediction for cold items if the item feature support is enabled.</remarks>
            public IReadOnlyList<double> GetSingleItemNonZeroFeatureValues(NativeDataset featureSource, int itemId)
            {
                return featureSource.NonZeroItemFeatureValues[itemId];
            }

            /// <summary>
            /// Gets non-zero feature indices for a given item.
            /// </summary>
            /// <param name="featureSource">The source to obtain feature indices from.</param>
            /// <param name="itemId">The item identifier.</param>
            /// <returns>Non-zero feature values for the item.</returns>
            /// <remarks>This function will be called during prediction for cold items if the item feature support is enabled.</remarks>
            public IReadOnlyList<int> GetSingleItemNonZeroFeatureIndices(NativeDataset featureSource, int itemId)
            {
                return featureSource.NonZeroItemFeatureIndices[itemId];
            }

            /// <summary>
            /// Saves the state of the native data mapping using a writer to a binary stream.
            /// </summary>
            /// <param name="writer">The writer to save the state of the native data mapping to.</param>
            public void SaveForwardCompatible(IWriter writer)
            {
                // Nothing to serialize
            }
        }

        /// <summary>
        /// Provides features for the positive only dataset.
        /// </summary>
        internal class FeatureProvider
        {
            /// <summary>
            /// Gets or sets the mapping from user to features.
            /// </summary>
            public IDictionary<string, Vector> UserFeatures { get; set; }

            /// <summary>
            /// Gets or sets the mapping from item to features.
            /// </summary>
            public IDictionary<string, Vector> ItemFeatures { get; set; }
        }

        /// <summary>
        /// Represents a positive only dataset.
        /// </summary>
        internal class PositiveOnlyDataset
        {
            /// <summary>
            /// Gets or sets the list of observations.
            /// </summary>
            public List<Tuple<string, string>> Observations { get; set; }
        }

        /// <summary>
        /// An implementation of
        /// <see cref="IRecommenderMapping{TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues}"/>
        /// for the positive only test data.
        /// </summary>
        [Serializable]
        internal class PositiveOnlyDataMapping
            : IRecommenderMapping<PositiveOnlyDataset, Tuple<string, string>, string, string, FeatureProvider, Vector>
        {
            /// <summary>
            /// Retrieves a list of instances from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source to retrieve instances from.</param>
            /// <returns>The list of retrieved instances.</returns>
            public IEnumerable<Tuple<string, string>> GetInstances(PositiveOnlyDataset instanceSource)
            {
                return instanceSource.Observations;
            }

            /// <summary>
            /// Extracts a user from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract the user from.</param>
            /// <returns>The extracted user.</returns>
            public string GetUser(PositiveOnlyDataset instanceSource, Tuple<string, string> instance)
            {
                return instance.Item1;
            }

            /// <summary>
            /// Extracts an item from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract the item from.</param>
            /// <returns>The extracted item.</returns>
            public string GetItem(PositiveOnlyDataset instanceSource, Tuple<string, string> instance)
            {
                return instance.Item2;
            }

            /// <summary>
            /// Provides a vector of features for a given user.
            /// </summary>
            /// <param name="featureSource">The source of the feature vector.</param>
            /// <param name="user">The user to provide the feature vector for.</param>
            /// <returns>The feature vector for <paramref name="user"/>.</returns>
            public Vector GetUserFeatures(FeatureProvider featureSource, string user)
            {
                return featureSource.UserFeatures[user];
            }

            /// <summary>
            /// Provides a vector of features for a given item.
            /// </summary>
            /// <param name="featureSource">The source of the feature vector.</param>
            /// <param name="item">The item to provide the feature vector for.</param>
            /// <returns>The feature vector for <paramref name="item"/>.</returns>
            public Vector GetItemFeatures(FeatureProvider featureSource, string item)
            {
                return featureSource.ItemFeatures[item];
            }
        }

        /// <summary>
        /// Mapping for the simple data model used in tests.
        /// TODO: expression-based mappings should be resurrected at least for tests
        /// </summary>
        internal class TestMapping : IRecommenderMapping<IEnumerable<Tuple<int, int, int>>, Tuple<int, int, int>, int, int, NoFeatureSource, int>
        {
            /// <summary>
            /// Retrieves a list of instances from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source to retrieve instances from.</param>
            /// <returns>The list of retrieved instances.</returns>
            public IEnumerable<Tuple<int, int, int>> GetInstances(IEnumerable<Tuple<int, int, int>> instanceSource)
            {
                return instanceSource;
            }

            /// <summary>
            /// Extracts a user from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract user from.</param>
            /// <returns>The extracted user.</returns>
            public int GetUser(IEnumerable<Tuple<int, int, int>> instanceSource, Tuple<int, int, int> instance)
            {
                return instance.Item1;
            }

            /// <summary>
            /// Extracts an item from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract item from.</param>
            /// <returns>The extracted item.</returns>
            public int GetItem(IEnumerable<Tuple<int, int, int>> instanceSource, Tuple<int, int, int> instance)
            {
                return instance.Item2;
            }

            /// <summary>
            /// Extracts a rating from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract rating from.</param>
            /// <returns>The extracted rating.</returns>
            public int GetRating(IEnumerable<Tuple<int, int, int>> instanceSource, Tuple<int, int, int> instance)
            {
                // This method should not be called by the data splitter
                throw new NotImplementedException();
            }

            /// <summary>
            /// Provides a vector of features for a given user.
            /// </summary>
            /// <param name="featureSource">The storage of features.</param>
            /// <param name="user">The user to provide features for.</param>
            /// <returns>The feature vector for <paramref name="user"/>.</returns>
            public int GetUserFeatures(NoFeatureSource featureSource, int user)
            {
                // This method should not be called by the data splitter
                throw new NotImplementedException();
            }

            /// <summary>
            /// Provides a vector of features for a given item.
            /// </summary>
            /// <param name="featureSource">The storage of features.</param>
            /// <param name="item">The item to provide features for.</param>
            /// <returns>The feature vector for <paramref name="item"/>.</returns>
            public int GetItemFeatures(NoFeatureSource featureSource, int item)
            {
                // This method should not be called by the data splitter
                throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Represents a testing star rating recommender mapping
        /// </summary>
        internal class TestStarRatingRecommenderMapping : IStarRatingRecommenderMapping<IEnumerable<Tuple<string, string, int, IDictionary<int, double>, double>>, Tuple<string, string, int, IDictionary<int, double>, double>, string, string, double, Int16, double[]>
        {
            /// <summary>
            /// Retrieves a list of instances from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source to retrieve instances from.</param>
            /// <returns>The list of retrieved instances.</returns>
            public IEnumerable<Tuple<string, string, int, IDictionary<int, double>, double>> GetInstances(IEnumerable<Tuple<string, string, int, IDictionary<int, double>, double>> instanceSource)
            {
                return instanceSource;
            }

            /// <summary>
            /// Extracts a user from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract user from.</param>
            /// <returns>The extracted user.</returns>
            public string GetUser(IEnumerable<Tuple<string, string, int, IDictionary<int, double>, double>> instanceSource, Tuple<string, string, int, IDictionary<int, double>, double> instance)
            {
                return instance.Item1;
            }

            /// <summary>
            /// Extracts an item from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract item from.</param>
            /// <returns>The extracted item.</returns>
            public string GetItem(IEnumerable<Tuple<string, string, int, IDictionary<int, double>, double>> instanceSource, Tuple<string, string, int, IDictionary<int, double>, double> instance)
            {
                return instance.Item2;
            }

            /// <summary>
            /// Extracts a rating from a given instance.
            /// </summary>
            /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract rating from.</param>
            /// <returns>The extracted rating.</returns>
            public double GetRating(IEnumerable<Tuple<string, string, int, IDictionary<int, double>, double>> instanceSource, Tuple<string, string, int, IDictionary<int, double>, double> instance)
            {
                return instance.Item5;
            }

            /// <summary>
            /// Provides an array of feature values for a given user.
            /// </summary>
            /// <param name="featureSource">The source of features.</param>
            /// <param name="user">The user to provide features for.</param>
            /// <returns>The array of features for <paramref name="user"/>.</returns>
            public double[] GetUserFeatures(short featureSource, string user)
            {
                return null;
            }

            /// <summary>
            /// Provides an array of feature values for a given item.
            /// </summary>
            /// <param name="featureSource">The source of features.</param>
            /// <param name="item">The item to provide features for.</param>
            /// <returns>The array of features for <paramref name="item"/>.</returns>
            public double[] GetItemFeatures(short featureSource, string item)
            {
                return null;
            }

            /// <summary>
            /// Provides the object describing how ratings provided by the instance source map to stars.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <returns>The object describing how ratings provided by the instance source map to stars.</returns>
            public IStarRatingInfo<double> GetRatingInfo(IEnumerable<Tuple<string, string, int, IDictionary<int, double>, double>> instanceSource)
            {
                return new RoundingStarRatingInfo(1, 5);
            }
        }

        /// <summary>
        /// An implementation of
        /// <see cref="IStarRatingRecommenderMapping{TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues}"/>
        /// for the data format used in tests. Throws when user features are accessed.
        /// </summary>
        [Serializable]
        internal class StandardRecommenderTestMappingItemFeaturesOnly : StandardRecommenderTestMapping
        {
            /// <summary>
            /// Provides a vector of features for a given user.
            /// </summary>
            /// <param name="featureSource">The source of features.</param>
            /// <param name="user">The user to provide features for.</param>
            /// <returns>The feature vector for <paramref name="user"/>.</returns>
            public override Vector GetUserFeatures(StandardFeatureProvider featureSource, User user)
            {
                throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Represents users in the standard data format.
        /// </summary>
        [Serializable]
        internal class User : Entity<User>
        {
            /// <summary>
            /// Returns a string that represents the current object.
            /// </summary>
            /// <returns>A string that represents the current object.</returns>
            public override string ToString()
            {
                return "User " + this.Id;
            }
        }

        /// <summary>
        /// Base class for users and items.
        /// </summary>
        /// <typeparam name="T">The type of the derived class (<see cref="User"/> or <see cref="Item"/>).</typeparam>
        [Serializable]
        internal class Entity<T> where T : Entity<T>, new()
        {
            /// <summary>
            /// Gets or sets the identifier.
            /// </summary>
            public string Id { get; set; }

            /// <summary>
            /// Creates new entity of a derived type with a given identifier.
            /// </summary>
            /// <param name="id">The identifier of a new entity.</param>
            /// <returns>The new entity.</returns>
            public static T WithId(string id)
            {
                return new T { Id = id };
            }

            /// <summary>
            /// Compares the entity with a given object. Comparison is identifier-based.
            /// </summary>
            /// <param name="obj">The object to compare with.
            /// </param>
            /// <returns>
            /// The <see cref="bool"/>.
            /// </returns>
            public override bool Equals(object obj)
            {
                if (obj == null || GetType() != obj.GetType())
                {
                    return false;
                }

                return object.Equals(((Entity<T>)obj).Id, this.Id);
            }

            /// <summary>
            /// Computes the hash code of the entity.
            /// </summary>
            /// <returns>The computed hash code.</returns>
            public override int GetHashCode()
            {
                return this.Id == null ? 0 : this.Id.GetHashCode();
            }
        }

        /// <summary>
        /// Represents items in the standard data format.
        /// </summary>
        [Serializable]
        internal class Item : Entity<Item>
        {
            /// <summary>
            /// Returns a string that represents the current object.
            /// </summary>
            /// <returns>A string that represents the current object.</returns>
            public override string ToString()
            {
                return "Item " + this.Id;
            }
        }

        /// <summary>
        /// An implementation of
        /// <see cref="IStarRatingRecommenderMapping{TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues}"/>
        /// for the data format used in tests. Throws when item features are accessed.
        /// </summary>
        [Serializable]
        internal class StandardRecommenderTestMappingUserFeaturesOnly : StandardRecommenderTestMapping
        {
            /// <summary>
            /// Provides a vector of features for a given item.
            /// </summary>
            /// <param name="featureSource">The source of features.</param>
            /// <param name="item">The item to provide features for.</param>
            /// <returns>The feature vector for <paramref name="item"/>.</returns>
            public override Vector GetItemFeatures(StandardFeatureProvider featureSource, Item item)
            {
                throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Represents a dataset in the standard recommender format.
        /// </summary>
        internal class StandardDataset
        {
            /// <summary>
            /// Gets or sets the list of observations.
            /// </summary>
            public List<Tuple<User, Item, int?>> Observations { get; set; }
        }

        /// <summary>
        /// Provides features for the data in the standard format.
        /// </summary>
        internal class StandardFeatureProvider
        {
            /// <summary>
            /// Gets or sets the mapping from user to features.
            /// </summary>
            public IDictionary<User, Vector> UserFeatures { get; set; }

            /// <summary>
            /// Gets or sets the mapping from item to features.
            /// </summary>
            public IDictionary<Item, Vector> ItemFeatures { get; set; }
        }

        /// <summary>
        /// An implementation of
        /// <see cref="IStarRatingRecommenderMapping{TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues}"/>
        /// for the data format used in tests.
        /// </summary>
        [Serializable]
        internal class StandardRecommenderTestMapping :
            IStarRatingRecommenderMapping<StandardDataset, Tuple<User, Item, int?>, User, Item, int, StandardFeatureProvider, Vector>, ICustomSerializable
        {
            /// <summary>
            /// The minimum star rating for data in standard format.
            /// </summary>
            internal const int MinStarRating = -2;

            /// <summary>
            /// The maximum star rating for data in standard format.
            /// </summary>
            internal const int MaxStarRating = 3;

            /// <summary>
            /// Initializes a new instance of the <see cref="StandardRecommenderTestMapping"/> class.
            /// </summary>
            public StandardRecommenderTestMapping()
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="StandardRecommenderTestMapping"/> class.
            /// </summary>
            /// <param name="reader">The reader to load the mapping from.</param>
            public StandardRecommenderTestMapping(IReader reader)
            {
                reader.ReadSerializationVersion(4);

                // Nothing to deserialize
            }

            /// <summary>
            /// Retrieves a list of instances from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source to retrieve instances from.</param>
            /// <returns>The list of retrieved instances.</returns>
            public IEnumerable<Tuple<User, Item, int?>> GetInstances(StandardDataset instanceSource)
            {
                return instanceSource.Observations;
            }

            /// <summary>
            /// Extracts a user from a given instance.
            /// </summary>
            /// <param name="instanceSource">The source of instances providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract user from.</param>
            /// <returns>The extracted user.</returns>
            public User GetUser(StandardDataset instanceSource, Tuple<User, Item, int?> instance)
            {
                return instance.Item1;
            }

            /// <summary>
            /// Extracts an item from a given instance.
            /// </summary>
            /// <param name="instanceSource">The source of instances providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract item from.</param>
            /// <returns>The extracted item.</returns>
            public Item GetItem(StandardDataset instanceSource, Tuple<User, Item, int?> instance)
            {
                return instance.Item2;
            }

            /// <summary>
            /// Extracts a rating from a given instance.
            /// </summary>
            /// <param name="instanceSource">The source of instances providing the <paramref name="instance"/>.</param>
            /// <param name="instance">The instance to extract rating from.</param>
            /// <returns>The extracted rating.</returns>
            public int GetRating(StandardDataset instanceSource, Tuple<User, Item, int?> instance)
            {
                if (instance.Item3 == null)
                {
                    throw new ArgumentException("Rating is not contained in the given instance", nameof(instance));
                }

                return instance.Item3.Value + MinStarRating;
            }

            /// <summary>
            /// Provides a vector of features for a given user.
            /// </summary>
            /// <param name="featureSource">The source of features.</param>
            /// <param name="user">The user to provide features for.</param>
            /// <returns>The feature vector for <paramref name="user"/>.</returns>
            public virtual Vector GetUserFeatures(StandardFeatureProvider featureSource, User user)
            {
                return featureSource.UserFeatures[user];
            }

            /// <summary>
            /// Provides a vector of features for a given item.
            /// </summary>
            /// <param name="featureSource">The source of features.</param>
            /// <param name="item">The item to provide features for.</param>
            /// <returns>The feature vector for <paramref name="item"/>.</returns>
            public virtual Vector GetItemFeatures(StandardFeatureProvider featureSource, Item item)
            {
                return featureSource.ItemFeatures[item];
            }

            /// <summary>
            /// Provides the object describing how ratings provided by the instance source map to stars.
            /// </summary>
            /// <param name="instanceSource">The source of instances.</param>
            /// <returns>The object describing how ratings in the dataset map to the stars.</returns>
            public IStarRatingInfo<int> GetRatingInfo(StandardDataset instanceSource)
            {
                return new StarRatingInfo(MinStarRating, MaxStarRating);
            }

            /// <summary>
            /// Saves the state of the standard data mapping using a writer to a binary stream.
            /// </summary>
            /// <param name="writer">The writer to save the state of the standard data mapping to.</param>
            public void SaveForwardCompatible(IWriter writer)
            {
                writer.Write(3); // Fake serialization version
            }
        }

        [Serializable]
        internal class CsvMapping : IStarRatingRecommenderMapping<string, Tuple<string, string, int>, string, string, int, NoFeatureSource, Vector>
        {
            public IEnumerable<Tuple<string, string, int>> GetInstances(string instanceSource)
            {
                foreach (string line in File.ReadLines(instanceSource))
                {
                    string[] split = line.Split(new[] { ',' });
                    yield return Tuple.Create(split[0], split[1], Convert.ToInt32(split[2]));
                }
            }

            public string GetUser(string instanceSource, Tuple<string, string, int> instance)
            { return instance.Item1; }

            public string GetItem(string instanceSource, Tuple<string, string, int> instance)
            { return instance.Item2; }

            public int GetRating(string instanceSource, Tuple<string, string, int> instance)
            { return instance.Item3; }

            public IStarRatingInfo<int> GetRatingInfo(string instanceSource)
            { return new StarRatingInfo(0, 5); }

            public Vector GetUserFeatures(NoFeatureSource featureSource, string user)
            { throw new NotImplementedException(); }

            public Vector GetItemFeatures(NoFeatureSource featureSource, string item)
            { throw new NotImplementedException(); }
        }

        /// <summary>
        /// Represents a star rating recommender mapping for the particular data model
        /// </summary>
        [Serializable]
        internal class RunnerStarRatingRecommenderMapping : IStarRatingRecommenderMapping<RecommenderDataset, RatedUserItem, Runners.User, Runners.Item, int, DummyFeatureSource, Vector>
        {
            /// <summary>
            /// Retrieves a list of instances from a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source to retrieve instances from.</param>
            /// <returns>The list of retrieved instances.</returns>
            public IEnumerable<RatedUserItem> GetInstances(RecommenderDataset instanceSource)
            {
                return instanceSource.Observations;
            }

            /// <summary>
            /// Extracts a user from a given instance.
            /// </summary>
            /// <param name="instanceSource">The parameter is not used.</param>
            /// <param name="instance">The instance to extract user from.</param>
            /// <returns>The extracted user.</returns>
            public Runners.User GetUser(RecommenderDataset instanceSource, RatedUserItem instance)
            {
                return instance.User;
            }

            /// <summary>
            /// Extracts an item from a given instance.
            /// </summary>
            /// <param name="instanceSource">The parameter is not used.</param>
            /// <param name="instance">The instance to extract item from.</param>
            /// <returns>The extracted item.</returns>
            public Runners.Item GetItem(RecommenderDataset instanceSource, RatedUserItem instance)
            {
                return instance.Item;
            }

            /// <summary>
            /// Extracts a rating from a given instance.
            /// </summary>
            /// <param name="instanceSource">The parameter is not used.</param>
            /// <param name="instance">The instance to extract rating from.</param>
            /// <returns>The extracted rating.</returns>
            public int GetRating(RecommenderDataset instanceSource, RatedUserItem instance)
            {
                return instance.Rating;
            }

            /// <summary>
            /// Provides a vector of features for a given user.
            /// </summary>
            /// <param name="featureSource">The parameter is not used.</param>
            /// <param name="user">The user to provide features for.</param>
            /// <returns>The feature vector for <paramref name="user"/>.</returns>
            public Vector GetUserFeatures(DummyFeatureSource featureSource, Runners.User user)
            {
                return user.Features;
            }

            /// <summary>
            /// Provides a vector of features for a given item.
            /// </summary>
            /// <param name="featureSource">The parameter is not used.</param>
            /// <param name="item">The item to provide features for.</param>
            /// <returns>The feature vector for <paramref name="item"/>.</returns>
            public Vector GetItemFeatures(DummyFeatureSource featureSource, Runners.Item item)
            {
                return item.Features;
            }

            /// <summary>
            /// Provides the object describing how ratings provided by the instance source map to stars.
            /// </summary>
            /// <param name="instanceSource">The instance source.</param>
            /// <returns>The object describing how ratings provided by the instance source map to stars.</returns>
            public IStarRatingInfo<int> GetRatingInfo(RecommenderDataset instanceSource)
            {
                return instanceSource.StarRatingInfo;
            }
        }

        /// <summary>
        /// Deserializes a recommender from a given stream and formatter.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TRatingDistribution">The type of a distribution over ratings.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="stream">The stream.</param>
        /// <param name="formatter">The formatter.</param>
        /// <returns>The deserialized recommender object.</returns>
        public static IMatchboxRecommender<TInstanceSource, TUser, TItem, TRatingDistribution, TFeatureSource>
            Load<TInstanceSource, TUser, TItem, TRatingDistribution, TFeatureSource>(Stream stream, IFormatter formatter)
        {
            return Utilities.Load<IMatchboxRecommender<TInstanceSource, TUser, TItem, TRatingDistribution, TFeatureSource>>(stream, formatter);
        }

        #endregion

        #region Custom binary deserialization

        /// <summary>
        /// Deserializes a Matchbox recommender from a reader to a binary stream and a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="reader">The reader of a binary stream to a serialized Matchbox recommender.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The deserialized recommender object.</returns>
        public static IMatchboxRecommender<TInstanceSource, int, int, Discrete, TFeatureSource>
            LoadBackwardCompatible<TInstanceSource, TFeatureSource>(
                IReader reader, IMatchboxRecommenderMapping<TInstanceSource, TFeatureSource> mapping)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new NativeDataFormatMatchboxRecommender<TInstanceSource, TFeatureSource>(reader, mapping);
        }

        /// <summary>
        /// Deserializes a Matchbox recommender from a binary stream and a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="stream">The binary stream to a serialized Matchbox recommender.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The deserialized recommender object.</returns>
        public static IMatchboxRecommender<TInstanceSource, int, int, Discrete, TFeatureSource>
            LoadBackwardCompatible<TInstanceSource, TFeatureSource>(
                Stream stream, IMatchboxRecommenderMapping<TInstanceSource, TFeatureSource> mapping)
        {
            if (stream == null)
            {
                throw new ArgumentNullException(nameof(stream));
            }

            using (var reader = new WrappedBinaryReader(new BinaryReader(stream, Encoding.UTF8, true)))
            {
                return LoadBackwardCompatible(reader, mapping);
            }
        }

        /// <summary>
        /// Deserializes a Matchbox recommender from a file with the specified name and a native data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="fileName">The name of the file of a serialized Matchbox recommender.</param>
        /// <param name="mapping">The mapping used for accessing data in the native format.</param>
        /// <returns>The deserialized recommender object.</returns>
        public static IMatchboxRecommender<TInstanceSource, int, int, Discrete, TFeatureSource>
            LoadBackwardCompatible<TInstanceSource, TFeatureSource>(
                string fileName, IMatchboxRecommenderMapping<TInstanceSource, TFeatureSource> mapping)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            using (Stream stream = File.Open(fileName, FileMode.Open))
            {
                return LoadBackwardCompatible(stream, mapping);
            }
        }

        /// <summary>
        /// Deserializes a Matchbox recommender from a reader to a binary stream and a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TRating">The type of a rating.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="reader">The reader of a binary stream to a serialized Matchbox recommender.</param>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <param name="readUser">Read a user.</param>
        /// <param name="writeUser">Write a user.</param>
        /// <param name="readItem">Read an item.</param>
        /// <param name="writeItem">Write an item.</param>
        /// <returns>The deserialized recommender object.</returns>
        public static IMatchboxRecommender<TInstanceSource, TUser, TItem, RatingDistribution, TFeatureSource>
            LoadBackwardCompatible<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(
                IReader reader, IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, Vector> mapping, Func<IReader, TUser> readUser, Action<IWriter, TUser> writeUser, Func<IReader, TItem> readItem, Action<IWriter, TItem> writeItem)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new StandardDataFormatMatchboxRecommender<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(reader, mapping, readUser, writeUser, readItem, writeItem);
        }

        /// <summary>
        /// Deserializes a Matchbox recommender from a binary stream and a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TRating">The type of a rating.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="stream">The binary stream to a serialized Matchbox recommender.</param>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <param name="readUser">Read a user.</param>
        /// <param name="writeUser">Write a user.</param>
        /// <param name="readItem">Read an item.</param>
        /// <param name="writeItem">Write an item.</param>
        /// <returns>The deserialized recommender object.</returns>
        public static IMatchboxRecommender<TInstanceSource, TUser, TItem, RatingDistribution, TFeatureSource>
            LoadBackwardCompatible<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(
                Stream stream, IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, Vector> mapping, Func<IReader, TUser> readUser, Action<IWriter, TUser> writeUser, Func<IReader, TItem> readItem, Action<IWriter, TItem> writeItem)
        {
            if (stream == null)
            {
                throw new ArgumentNullException(nameof(stream));
            }

            using (var reader = new WrappedBinaryReader(new BinaryReader(stream, Encoding.UTF8, true)))
            {
                return LoadBackwardCompatible(reader, mapping, readUser, writeUser, readItem, writeItem);
            }
        }

        /// <summary>
        /// Deserializes a Matchbox recommender from a file with the specified name and a standard data format mapping.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TRating">The type of a rating.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="fileName">The name of the file of a serialized Matchbox recommender.</param>
        /// <param name="mapping">The mapping used for accessing data in the standard format.</param>
        /// <param name="readUser">Read a user.</param>
        /// <param name="writeUser">Write a user.</param>
        /// <param name="readItem">Read an item.</param>
        /// <param name="writeItem">Write an item.</param>
        /// <returns>The deserialized recommender object.</returns>
        public static IMatchboxRecommender<TInstanceSource, TUser, TItem, RatingDistribution, TFeatureSource>
            LoadBackwardCompatible<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(
                string fileName, IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, Vector> mapping, Func<IReader, TUser> readUser, Action<IWriter, TUser> writeUser, Func<IReader, TItem> readItem, Action<IWriter, TItem> writeItem)
        {
            if (fileName == null)
            {
                throw new ArgumentNullException(nameof(fileName));
            }

            using (Stream stream = File.Open(fileName, FileMode.Open))
            {
                return LoadBackwardCompatible(stream, mapping, readUser, writeUser, readItem, writeItem);
            }
        }

        #endregion
    }
}
