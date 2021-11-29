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
    /// Represents the distribution over item parameters.
    /// </summary>
    [Serializable]
    internal class ItemParameterDistribution : EntityParameterDistribution
    {
        /// <summary>
        /// The current custom binary serialization version of the <see cref="ItemParameterDistribution"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// Initializes a new instance of the <see cref="ItemParameterDistribution"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The binary reader to read the distribution over item parameters from.</param>
        public ItemParameterDistribution(IReader reader) : base(reader)
        {
            Debug.Assert(reader != null, "The reader must not be null.");

            reader.ReadSerializationVersion(CustomSerializationVersion);

            // Nothing to deserialize
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ItemParameterDistribution"/> class.
        /// </summary>
        /// <param name="traitDistribution">The distribution over the user traits.</param>
        /// <param name="biasDistribution">The distribution over the user bias.</param>
        public ItemParameterDistribution(GaussianArray traitDistribution, Gaussian biasDistribution)
            : base(traitDistribution, biasDistribution)
        {
        }

        /// <summary>
        /// Saves the distribution over item parameters using the specified writer to a binary stream.
        /// </summary>
        /// <param name="writer">The writer to save the distribution over item parameters to.</param>
        public override void SaveForwardCompatible(IWriter writer)
        {
            base.SaveForwardCompatible(writer);

            writer.Write(CustomSerializationVersion);
        }
    }
}