// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Represents a generator of negative ratings.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of an instance source.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
    /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
    /// <remarks>
    /// A description of the algorithm for negative data generation used here can be found in Section 4.1
    /// of <see href="http://www.ulrichpaquet.com/Papers/PaquetKoenigsteinWWW.pdf">this</see> paper.
    /// </remarks>
    [Serializable]
    public class NegativeDataGeneratorMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>
        : IStarRatingRecommenderMapping<TInstanceSource, RatingInstance<TUser, TItem, byte>, TUser, TItem, int, TFeatureSource, TFeatureValues>, ICustomSerializable
    {
        #region Fields, constructors, properties

        /// <summary>
        /// The current custom binary serialization version of the 
        /// <see cref="NegativeDataGeneratorMapping{TInstanceSource,TInstance,TUser,TItem,TFeatureSource,TFeatureValues}"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// The negative rating value
        /// </summary>
        private const byte NegativeRatingValue = 0;

        /// <summary>
        /// The positive rating value
        /// </summary>
        private const byte PositiveRatingValue = 1;

        /// <summary>
        /// The number of attempts to sample an item for a user.
        /// </summary>
        private const int SampleAttemptsCount = 10;

        /// <summary>
        /// The default item histogram adjustment.
        /// </summary>
        private const double DefaultItemHistogramAdjustment = 1.2;

        /// <summary>
        /// The custom binary serialization <see cref="Guid"/> of the
        /// <see cref="NegativeDataGeneratorMapping{TInstanceSource,TInstance,TUser,TItem,TFeatureSource,TFeatureValues}"/> class.
        /// </summary>
        private readonly Guid customSerializationGuid = new Guid("09AA0E98-FD9A-4F2F-8791-6322DE0B0265");

        /// <summary>
        /// The wrapped recommender mapping.
        /// </summary>
        private readonly IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues> mapping;

        /// <summary>
        /// The degree to which each element of the item histogram is raised before sampling.
        /// Should be different from 1.0 if it is believed that there is some quality
        /// bar which drives popular items to be more generally liked.
        /// </summary>
        private readonly double itemHistogramAdjustment;

        /// <summary>
        /// The last used instance source.
        /// </summary>
        /// <remarks>Used to avoid redundant conversions.</remarks>
        [NonSerialized]
        private TInstanceSource lastInstanceSource;

        /// <summary>
        /// The cached collection of rated instances.
        /// </summary>
        [NonSerialized]
        private IDictionary<TUser, HashSet<RatedItem>> ratedInstances;

        /// <summary>
        /// Initializes a new instance of the <see cref="NegativeDataGeneratorMapping{TInstanceSource,TInstance,TUser,TItem,TFeatureSource,TFeatureValues}"/> class.
        /// </summary>
        /// <param name="mapping">The top level mapping.</param>
        /// <param name="itemHistogramAdjustment">
        /// The degree to which each element of the item histogram is raised before sampling.
        /// Should be different from 1.0 if it is believed that there is some quality
        /// bar which drives popular items to be more generally liked.
        /// </param>
        public NegativeDataGeneratorMapping(
            IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues> mapping,
            double itemHistogramAdjustment = DefaultItemHistogramAdjustment)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            this.mapping = mapping;
            this.itemHistogramAdjustment = itemHistogramAdjustment;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NegativeDataGeneratorMapping{TInstanceSource,TInstance,TUser,TItem,TFeatureSource,TFeatureValues}"/> 
        /// class from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the state of the negative data generation mapping from.</param>
        /// <param name="mapping">The top level mapping.</param>
        public NegativeDataGeneratorMapping(
            IReader reader,
            IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues> mapping)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            this.mapping = mapping;

            reader.VerifySerializationGuid(
                this.customSerializationGuid, "The binary stream does not contain the negative data generation mapping of an Infer.NET Matchbox recommender.");

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);
            if (deserializedVersion == CustomSerializationVersion)
            {
                this.itemHistogramAdjustment = reader.ReadDouble();
            }
        }

        #endregion

        #region IStarRatingRecommenderMapping implementation

        /// <summary>
        /// Retrieves a list of instances from a given instance source.
        /// </summary>
        /// <param name="instanceSource">The instance source to retrieve instances from.</param>
        /// <returns>The list of retrieved instances.</returns>
        public IEnumerable<RatingInstance<TUser, TItem, byte>> GetInstances(TInstanceSource instanceSource)
        {
            this.UpdateInstanceRepresentation(instanceSource);

            return this.ratedInstances.SelectMany(
                userToRatedItems =>
                userToRatedItems.Value.Select(
                    ratedItem =>
                    new RatingInstance<TUser, TItem, byte>(userToRatedItems.Key, ratedItem.Item, ratedItem.Rating)));
        }

        /// <summary>
        /// Extracts a user from a given instance.
        /// </summary>
        /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
        /// <param name="instance">The instance to extract the user from.</param>
        /// <returns>The extracted user.</returns>
        public TUser GetUser(TInstanceSource instanceSource, RatingInstance<TUser, TItem, byte> instance)
        {
            this.UpdateInstanceRepresentation(instanceSource);
            return instance.User;
        }

        /// <summary>
        /// Extracts an item from a given instance.
        /// </summary>
        /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
        /// <param name="instance">The instance to extract the item from.</param>
        /// <returns>The extracted item.</returns>
        public TItem GetItem(TInstanceSource instanceSource, RatingInstance<TUser, TItem, byte> instance)
        {
            this.UpdateInstanceRepresentation(instanceSource);
            return instance.Item;
        }

        /// <summary>
        /// Extracts a rating from a given instance.
        /// </summary>
        /// <param name="instanceSource">The instance source providing the <paramref name="instance"/>.</param>
        /// <param name="instance">The instance to extract the rating from.</param>
        /// <returns>The extracted rating.</returns>
        public int GetRating(TInstanceSource instanceSource, RatingInstance<TUser, TItem, byte> instance)
        {
            this.UpdateInstanceRepresentation(instanceSource);
            return instance.Rating;
        }

        /// <summary>
        /// Provides the features for a given user.
        /// </summary>
        /// <param name="featureSource">The source of the features.</param>
        /// <param name="user">The user to provide the features for.</param>
        /// <returns>The features for <paramref name="user"/>.</returns>
        public TFeatureValues GetUserFeatures(TFeatureSource featureSource, TUser user)
        {
            return this.mapping.GetUserFeatures(featureSource, user);
        }

        /// <summary>
        /// Provides the features for a given item.
        /// </summary>
        /// <param name="featureSource">The source of the features.</param>
        /// <param name="item">The item to provide the features for.</param>
        /// <returns>The features for <paramref name="item"/>.</returns>
        public TFeatureValues GetItemFeatures(TFeatureSource featureSource, TItem item)
        {
            return this.mapping.GetItemFeatures(featureSource, item);
        }

        /// <summary>
        /// Provides the object describing how ratings in the dataset map to the stars.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <returns>The object describing how ratings provided by the instance source map to stars.</returns>
        public IStarRatingInfo<int> GetRatingInfo(TInstanceSource instanceSource)
        {
            return new StarRatingInfo(0, 1);
        }

        #endregion

        #region ICustomSerializable implementation

        /// <summary>
        /// Saves the negative data generation mapping using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the negative data generation mapping to.</param>
        public void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(this.customSerializationGuid);
            writer.Write(CustomSerializationVersion);
            writer.Write(this.itemHistogramAdjustment);
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Updates the representation of the rated instances.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <remarks>Call this method every time before accessing instance data.</remarks>
        private void UpdateInstanceRepresentation(TInstanceSource instanceSource)
        {
            if (object.Equals(this.lastInstanceSource, instanceSource))
            {
                return;
            }

            this.ratedInstances = new Dictionary<TUser, HashSet<RatedItem>>();
            this.lastInstanceSource = instanceSource;

            IDictionary<TItem, int> itemHistogram;
            this.GeneratePositiveRatingsAndItemHistogram(instanceSource, out itemHistogram);
            this.GenerateNegativeRatings(itemHistogram);
        }

        /// <summary>
        /// Generates the positive ratings and the item histogram.
        /// </summary>
        /// <param name="instanceSource">The source of positive instances.</param>
        /// <param name="itemHistogram">The item histogram.</param>
        /// <remarks>
        /// The generation of these two entities is implemented in the same method for performance reasons.
        /// By doing this, iteration over the positive observations is performed only once.
        /// </remarks>
        private void GeneratePositiveRatingsAndItemHistogram(
            TInstanceSource instanceSource, out IDictionary<TItem, int> itemHistogram)
        {
            itemHistogram = new Dictionary<TItem, int>();
            var instances = this.mapping.GetInstances(instanceSource);

            foreach (TInstance instance in instances)
            {
                TUser user = this.mapping.GetUser(instanceSource, instance);
                TItem item = this.mapping.GetItem(instanceSource, instance);

                // Update the item histogram
                if (!itemHistogram.ContainsKey(item))
                {
                    itemHistogram.Add(item, 0);
                }

                ++itemHistogram[item];

                // Update the rated instances
                HashSet<RatedItem> ratedItems;
                if (!this.ratedInstances.TryGetValue(user, out ratedItems))
                {
                    ratedItems = new HashSet<RatedItem>();
                    this.ratedInstances.Add(user, ratedItems);
                }

                var ratedItem = new RatedItem(item, PositiveRatingValue);
                if (ratedItems.Contains(ratedItem))
                {
                    throw new NotSupportedException(
                        string.Format(
                            "Multiple occurances of the same user-item pair are not supported. User: {0}, Item: {1}.",
                            user,
                            item));
                }

                ratedItems.Add(ratedItem);
            }
        }

        /// <summary>
        /// Generates the negative ratings.
        /// </summary>
        /// <param name="itemHistogram">The item histogram.</param>
        private void GenerateNegativeRatings(IDictionary<TItem, int> itemHistogram)
        {
            // Create the binary sum tree.
            var items = itemHistogram.Keys.Select(key => key).ToList();
            var histogramSampler = new HistogramSampler(
                itemHistogram.Values.Select(value => Convert.ToInt32(Math.Pow(value, this.itemHistogramAdjustment))));

            foreach (var userToRatedItems in this.ratedInstances)
            {
                var ratedItems = userToRatedItems.Value;

                // In case the user has liked more than half of the items
                int samplesCount = Math.Min(ratedItems.Count, items.Count - ratedItems.Count);

                for (int i = 0; i < samplesCount; ++i)
                {
                    // This is required not to run into an infinite loop.
                    for (int sampleAttempts = 0; sampleAttempts < SampleAttemptsCount; ++sampleAttempts)
                    {
                        var itemIndex = histogramSampler.Sample();
                        var ratedItem = new RatedItem(items[itemIndex], NegativeRatingValue);

                        if (!ratedItems.Contains(ratedItem))
                        {
                            histogramSampler.Take(itemIndex);
                            ratedItems.Add(ratedItem);
                            break;
                        }
                    }
                }
            }
        }

        #endregion

        #region RatedItem nested class

        /// <summary>
        /// Represents a rated item.
        /// </summary>
        private class RatedItem
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="RatedItem"/> class.
            /// </summary>
            /// <param name="item">The item.</param>
            /// <param name="rating">The rating.</param>
            public RatedItem(TItem item, byte rating)
            {
                this.Item = item;
                this.Rating = rating;
            }

            /// <summary>
            /// Gets the item.
            /// </summary>
            public TItem Item { get; private set; }

            /// <summary>
            /// Gets the rating of the item.
            /// </summary>
            public byte Rating { get; private set; }

            /// <summary>
            /// Determines whether the specified object is equal to the current object.
            /// </summary>
            /// <param name="obj">The object to compare with the current object.</param>
            /// <returns><b>true</b> if the specified object is equal to the current object; otherwise, <b>false</b>.</returns>
            public override bool Equals(object obj)
            {
                if (obj == null || GetType() != obj.GetType())
                {
                    return false;
                }

                var ratedItem = (RatedItem)obj;
                return this.Item.Equals(ratedItem.Item);
            }

            /// <summary>
            /// Serves as a hash function for the <see cref="RatedItem"/> class.
            /// </summary>
            /// <returns>A hash code for the current object.</returns>
            /// <remarks>Two rated items are considered the same if the items they contain are the same.</remarks>
            public override int GetHashCode()
            {
                return this.Item.GetHashCode();
            }
        }

        #endregion
    }
}
