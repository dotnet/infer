// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Serialization;

    using GaussianArray = Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>;
    using GaussianMatrix = Distributions.DistributionRefArray<Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>, double[]>;

    /// <summary>
    /// Represents the distribution over feature weights.
    /// </summary>
    [Serializable]
    internal class FeatureParameterDistribution : ICloneable, ICustomSerializable
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="FeatureParameterDistribution"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureParameterDistribution"/> class.
        /// </summary>
        public FeatureParameterDistribution()
        {
            this.TraitWeights = new GaussianMatrix(0);
            this.BiasWeights = new GaussianArray(0);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureParameterDistribution"/> class
        /// from a reader of a stream.
        /// </summary>
        /// <param name="reader">The reader to read the distribution over feature weights from.</param>
        public FeatureParameterDistribution(IReader reader)
        {
            if (reader == null) throw new ArgumentNullException(nameof(reader));

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.TraitWeights = reader.ReadGaussianMatrix();
                this.BiasWeights = reader.ReadGaussianArray();
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureParameterDistribution"/> class.
        /// </summary>
        /// <param name="traitFeatureWeightDistribution">The distribution over weights of the feature contribution to traits.</param>
        /// <param name="biasFeatureWeightDistribution">The distribution over weights of the feature contribution to biases.</param>
        public FeatureParameterDistribution(GaussianMatrix traitFeatureWeightDistribution, GaussianArray biasFeatureWeightDistribution)
        {
            if (traitFeatureWeightDistribution == null)
                traitFeatureWeightDistribution = new GaussianMatrix(0);
            if (biasFeatureWeightDistribution == null)
                biasFeatureWeightDistribution = new GaussianArray(0);
            foreach (var w in traitFeatureWeightDistribution)
            {
                if (w == null)
                    throw new ArgumentException(nameof(traitFeatureWeightDistribution), "Element is null");
                if (w.Count != biasFeatureWeightDistribution.Count)
                    throw new ArgumentException(nameof(traitFeatureWeightDistribution), "Feature count does not match biasFeatureWeightDistribution.Count");
            }

            this.TraitWeights = traitFeatureWeightDistribution;
            this.BiasWeights = biasFeatureWeightDistribution;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="FeatureParameterDistribution"/> class.
        /// </summary>
        /// <param name="traitCount">The number of traits.</param>
        /// <param name="featureCount">The number of features.</param>
        /// <param name="value">
        /// The value to which each element of the 
        /// contained distributions will be initialized.
        /// </param>
        public FeatureParameterDistribution(int traitCount, int featureCount, Gaussian value)
        {
            this.TraitWeights = new GaussianMatrix(new GaussianArray(value, featureCount), traitCount);
            this.BiasWeights = new GaussianArray(value, featureCount);
        }

        /// <summary>
        /// Gets the number of features.
        /// </summary>
        public int FeatureCount
        {
            get { return this.BiasWeights.Count; }
        }

        /// <summary>
        /// Gets the distribution over weights of the feature contribution to traits.
        /// </summary>
        public GaussianMatrix TraitWeights { get; private set; }

        /// <summary>
        /// Gets the distribution over weights of the feature contribution to biases.
        /// </summary>
        public GaussianArray BiasWeights { get; private set; }

        /// <summary>
        /// Creates a new object that is a copy of the current instance.
        /// </summary>
        /// <returns>A new object that is a copy of this instance.</returns>
        public object Clone()
        {
            return new FeatureParameterDistribution
            {
                TraitWeights = (GaussianMatrix)this.TraitWeights.Clone(),
                BiasWeights = (GaussianArray)this.BiasWeights.Clone(),
            };
        }

        /// <summary>
        /// Saves the distributions over feature weights using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the distributions over feature weights to.</param>
        public void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(CustomSerializationVersion);
            writer.Write(this.TraitWeights);
            writer.Write(this.BiasWeights);
        }
    }
}