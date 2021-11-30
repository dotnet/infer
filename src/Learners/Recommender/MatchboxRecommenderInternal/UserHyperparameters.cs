// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System;
    using System.IO;

    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Represents the user-related hyper-parameters of the Matchbox recommender.
    /// </summary>
    [Serializable]
    internal class UserHyperparameters : ICustomSerializable
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="UserHyperparameters"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the <see cref="UserHyperparameters"/> class.
        /// </summary>
        public UserHyperparameters()
        {
            this.TraitVariance = 1.0;
            this.BiasVariance = 1.0;
            this.ThresholdPriorVariance = 1.0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="UserHyperparameters"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the user-related hyper-parameters from.</param>
        public UserHyperparameters(IReader reader)
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.TraitVariance = reader.ReadDouble();
                this.BiasVariance = reader.ReadDouble();
                this.ThresholdPriorVariance = reader.ReadDouble();
            }
        }

        /// <summary>
        /// Gets or sets the variance of the user traits.
        /// </summary>
        public double TraitVariance { get; set; }

        /// <summary>
        /// Gets or sets the variance of the user bias.
        /// </summary>
        public double BiasVariance { get; set; }

        /// <summary>
        /// Gets or sets the variance of the prior distribution of the user threshold.
        /// </summary>
        public double ThresholdPriorVariance { get; set; }

        /// <summary>
        /// Saves the user-related hyper-parameters of the Matchbox recommender using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the user-related hyper-parameters to.</param>
        public void SaveForwardCompatible(IWriter writer)
        {
            writer.Write(CustomSerializationVersion);

            writer.Write(this.TraitVariance);
            writer.Write(this.BiasVariance);
            writer.Write(this.ThresholdPriorVariance);
        }
    }
}
