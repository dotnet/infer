// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    /// <summary>
    /// Represents the settings of the <see cref="VowpalWabbitRecommender{TInstanceSource}"/> class.
    /// </summary>
    internal class VowpalWabbitRecommenderSettings : ISettings
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="VowpalWabbitRecommenderSettings"/> class.
        /// </summary>
        public VowpalWabbitRecommenderSettings()
        {
            this.TraitCount = 10;
            this.BitPrecision = 18;
            this.LearningRate = 0.021;
            this.LearningRateDecay = 0.97;
            this.PassCount = 10;
            this.L1Regularization = 1.0;
            this.L2Regularization = 0.001;
        }

        /// <summary>
        /// Gets or sets the number of traits.
        /// </summary>
        public int TraitCount { get; set; }

        /// <summary>
        /// Gets or sets the number of bits in the feature table.
        /// </summary>
        public int BitPrecision { get; set; }

        /// <summary>
        /// Gets or sets the learning rate.
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// Gets or sets the learning rate decay.
        /// </summary>
        public double LearningRateDecay { get; set; }

        /// <summary>
        /// Gets or sets the number of passes.
        /// </summary>
        public int PassCount { get; set; }

        /// <summary>
        /// Gets or sets the weight of the L1 regularization term.
        /// </summary>
        public double L1Regularization { get; set; }

        /// <summary>
        /// Gets or sets the weight of the L2 regularization term.
        /// </summary>
        public double L2Regularization { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether user features should be used.
        /// </summary>
        public bool UseUserFeatures { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether item features should be used.
        /// </summary>
        public bool UseItemFeatures { get; set; }
    }
}