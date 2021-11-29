// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Represents the model noise variance.
    /// </summary>
    [Serializable]
    internal class NoiseHyperparameters : ICustomSerializable
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="NoiseHyperparameters"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the <see cref="NoiseHyperparameters"/> class.
        /// </summary>
        public NoiseHyperparameters()
        {
            this.UserThresholdVariance = 0.25;
            this.AffinityVariance = 1.0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NoiseHyperparameters"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the model noise variance from.</param>
        public NoiseHyperparameters(IReader reader)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);
            
            if (deserializedVersion == CustomSerializationVersion)
            {
                this.UserThresholdVariance = reader.ReadDouble();
                this.AffinityVariance = reader.ReadDouble();
            }
        }

        /// <summary>
        /// Gets or sets the variance of the noise of the user thresholds.
        /// </summary>
        public double UserThresholdVariance { get; set; }

        /// <summary>
        /// Gets or sets the variance of the affinity noise.
        /// </summary>
        public double AffinityVariance { get; set; }

        /// <summary>
        /// Saves the noise variance of the Matchbox recommender using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the model noise variance to.</param>
        public void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(CustomSerializationVersion);

            writer.Write(this.UserThresholdVariance);
            writer.Write(this.AffinityVariance);
        }
    }
}
