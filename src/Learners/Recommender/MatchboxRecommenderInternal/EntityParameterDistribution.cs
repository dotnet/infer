// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System;
    using System.Diagnostics;
    using System.IO;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Serialization;

    using GaussianArray = Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>;

    /// <summary>
    /// Represents distribution over traits and bias of a user or an item.
    /// </summary>
    [Serializable]
    internal abstract class EntityParameterDistribution : ICustomSerializable
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="EntityParameterDistribution"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the <see cref="EntityParameterDistribution"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the distribution over traits and bias of a user or an item from.</param>
        protected EntityParameterDistribution(IReader reader)
        {
            Debug.Assert(reader != null, "The reader must not be null.");

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.Traits = reader.ReadGaussianArray();
                this.Bias = reader.ReadGaussian();
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="EntityParameterDistribution"/> class.
        /// </summary>
        /// <param name="traitDistribution">The distribution over the entity traits.</param>
        /// <param name="biasDistribution">The distribution over the entity bias.</param>
        protected EntityParameterDistribution(GaussianArray traitDistribution, Gaussian biasDistribution)
        {
            Debug.Assert(traitDistribution != null, "A valid distribution over traits must be provided.");
            
            this.Traits = traitDistribution;
            this.Bias = biasDistribution;
        }

        /// <summary>
        /// Gets the distribution over the traits.
        /// </summary>
        public GaussianArray Traits { get; private set; }

        /// <summary>
        /// Gets the distribution over the bias.
        /// </summary>
        public Gaussian Bias { get; private set; }

        /// <summary>
        /// Saves the distribution over traits and bias of a user or an item using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the distribution over traits and bias of a user or an item to.</param>
        public virtual void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(CustomSerializationVersion);

            writer.Write(this.Traits);
            writer.Write(this.Bias);
        }
    }
}
