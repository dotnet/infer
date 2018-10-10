// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    /// <summary>
    /// Represents a rating similarity function used by the Mahout recommender.
    /// </summary>
    public enum MahoutRatingSimilarity
    {
        /// <summary>
        /// The Euclidian distance based similarity function.
        /// </summary>
        Euclidean,

        /// <summary>
        /// The Manhattan distance based similarity function.
        /// </summary>
        Manhattan,

        /// <summary>
        /// The similarity function based on Pearson's correlation.
        /// </summary>
        PearsonCorrelation
    }

    /// <summary>
    /// Represents an algorithm used by the Mahout recommender to predict ratings.
    /// </summary>
    public enum MahoutRatingPredictionAlgorithm
    {
        /// <summary>
        /// User-based rating prediction.
        /// </summary>
        UserBased,

        /// <summary>
        /// Item-based rating prediction.
        /// </summary>
        ItemBased,

        /// <summary>
        /// Slope One rating prediction.
        /// </summary>
        SlopeOne,

        /// <summary>
        /// SVD rating prediction.
        /// </summary>
        Svd
    }

    /// <summary>
    /// Represents an algorithm used by the Mahout recommender to fill missing ratings.
    /// </summary>
    public enum MahoutMissingRatingPredictionAlgorithm
    {
        /// <summary>
        /// Use the mean of the training set rating distribution.
        /// </summary>
        TakeMean,

        /// <summary>
        /// Use the mode of the training set rating distribution.
        /// </summary>
        TakeMode,

        /// <summary>
        /// Use the median of the training set rating distribution.
        /// </summary>
        TakeMedian,
    }
    
    /// <summary>
    /// Represents the settings of the <see cref="MahoutRecommender{TInstanceSource}"/>.
    /// </summary>
    internal class MahoutRecommenderSettings : ISettings
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="MahoutRecommenderSettings"/> class.
        /// </summary>
        public MahoutRecommenderSettings()
        {
            this.RatingSimilarity = MahoutRatingSimilarity.Euclidean;
            this.RatingPredictionAlgorithm = MahoutRatingPredictionAlgorithm.Svd;
            this.MissingRatingPredictionAlgorithm = MahoutMissingRatingPredictionAlgorithm.TakeMean;
            this.UserNeighborhoodSize = 500;
            this.TraitCount = 5;
            this.IterationCount = 10;
        }

        /// <summary>
        /// Gets or sets the rating similarity function used by the Mahout recommender.
        /// </summary>
        public MahoutRatingSimilarity RatingSimilarity { get; set; }

        /// <summary>
        /// Gets or sets the algorithm used for rating prediction by the Mahout recommender.
        /// </summary>
        public MahoutRatingPredictionAlgorithm RatingPredictionAlgorithm { get; set; }

        /// <summary>
        /// Gets or sets the algorithm used to fill the ratings that the Mahout recommender failed to estimate.
        /// </summary>
        public MahoutMissingRatingPredictionAlgorithm MissingRatingPredictionAlgorithm { get; set; }

        /// <summary>
        /// Gets or sets the size of the user neighborhood used by the user-based rating prediction algorithm.
        /// </summary>
        public int UserNeighborhoodSize { get; set; }

        /// <summary>
        /// Gets or sets the number of traits used by the SVD recommender.
        /// </summary>
        public int TraitCount { get; set; }

        /// <summary>
        /// Gets or sets the number of iterations used by the SVD recommender.
        /// </summary>
        public int IterationCount { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to use 64-bit JVM to run Mahout
        /// </summary>
        public bool UseX64JVM { get; set; }

        /// <summary>
        /// Gets or sets maximum heap size in MB for JVM running Mahout
        /// </summary>
        public int JavaMaxHeapSizeInMb { get; set; }
    }
}
