// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Represents the feature-related hyper-parameters of the Matchbox recommender.
    /// </summary>
    [Serializable]
    internal class FeatureHyperparameters : ICustomSerializable
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="FeatureHyperparameters"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureHyperparameters"/> class.
        /// </summary>
        public FeatureHyperparameters()
        {
            this.TraitWeightPriorVariance = 1.0;
            this.BiasWeightPriorVariance = 1.0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureHyperparameters"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the feature-related hyper-parameters from.</param>
        public FeatureHyperparameters(IReader reader)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.TraitWeightPriorVariance = reader.ReadDouble();
                this.BiasWeightPriorVariance = reader.ReadDouble();
            }
        }

        /// <summary>
        /// Gets or sets the variance of the weights of the feature contribution to the traits.
        /// </summary>
        public double TraitWeightPriorVariance { get; set; }

        /// <summary>
        /// Gets or sets the variance of the weights of the feature contribution to the biases.
        /// </summary>
        public double BiasWeightPriorVariance { get; set; }

        /// <summary>
        /// Saves the feature-related hyper-parameters of the Matchbox recommender using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the feature-related hyper-parameters to.</param>
        public void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(CustomSerializationVersion);

            writer.Write(this.TraitWeightPriorVariance);
            writer.Write(this.BiasWeightPriorVariance);
        }
    }
}
