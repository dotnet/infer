// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.IO;
    using System.Runtime.Serialization;
    using System.Text;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal;
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
        /// <returns>The recommender instance.</returns>
        public static IMatchboxRecommender<TInstanceSource, TUser, TItem, RatingDistribution, TFeatureSource>
            Create<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(
                IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, Vector> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new StandardDataFormatMatchboxRecommender<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(mapping);
        }

        #endregion

        #region .NET binary deserialization
        
        /// <summary>
        /// Deserializes a Matchbox recommender from a file.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TRatingDistribution">The type of a distribution over ratings.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <param name="fileName">The file name.</param>
        /// <returns>The deserialized recommender object.</returns>
        public static IMatchboxRecommender<TInstanceSource, TUser, TItem, TRatingDistribution, TFeatureSource>
            Load<TInstanceSource, TUser, TItem, TRatingDistribution, TFeatureSource>(string fileName)
        {
            return Utilities.Load<IMatchboxRecommender<TInstanceSource, TUser, TItem, TRatingDistribution, TFeatureSource>>(fileName);
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
        /// <returns>The deserialized recommender object.</returns>
        public static IMatchboxRecommender<TInstanceSource, TUser, TItem, RatingDistribution, TFeatureSource>
            LoadBackwardCompatible<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(
                IReader reader, IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, Vector> mapping)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            return new StandardDataFormatMatchboxRecommender<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(reader, mapping);
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
        /// <returns>The deserialized recommender object.</returns>
        public static IMatchboxRecommender<TInstanceSource, TUser, TItem, RatingDistribution, TFeatureSource>
            LoadBackwardCompatible<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(
                Stream stream, IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, Vector> mapping)
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
        /// <returns>The deserialized recommender object.</returns>
        public static IMatchboxRecommender<TInstanceSource, TUser, TItem, RatingDistribution, TFeatureSource>
            LoadBackwardCompatible<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource>(
                string fileName, IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, Vector> mapping)
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

        #endregion
    }
}
