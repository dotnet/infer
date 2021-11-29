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
    /// Represents the distribution over user parameters.
    /// </summary>
    [Serializable]
    internal class UserParameterDistribution : EntityParameterDistribution
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="UserParameterDistribution"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the <see cref="UserParameterDistribution"/> class.
        /// </summary>
        /// <param name="traitDistribution">The distribution over the user traits.</param>
        /// <param name="biasDistribution">The distribution over the user bias.</param>
        /// <param name="thresholdDistribution">The distribution over the user thresholds.</param>
        public UserParameterDistribution(GaussianArray traitDistribution, Gaussian biasDistribution, GaussianArray thresholdDistribution)
            : base(traitDistribution, biasDistribution)
        {
            Debug.Assert(thresholdDistribution != null, "A valid distribution over thresholds must be provided.");
            
            this.Thresholds = thresholdDistribution;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="UserParameterDistribution"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the distribution over user parameters from.</param>
        public UserParameterDistribution(IReader reader) : base(reader)
        {
            Debug.Assert(reader != null, "The reader must not be null.");

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                this.Thresholds = reader.ReadGaussianArray();
            }
        }

        /// <summary>
        /// Gets the distribution over thresholds.
        /// </summary>
        public GaussianArray Thresholds { get; private set; }

        /// <summary>
        /// Saves the distribution over user parameters using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the distribution over user parameters to.</param>
        public override void SaveForwardCompatible(IWriter writer)
        {
            base.SaveForwardCompatible(writer);

            writer.Write(CustomSerializationVersion);
            writer.Write(this.Thresholds);
        }
    }
}