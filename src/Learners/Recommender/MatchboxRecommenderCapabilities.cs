// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;

    /// <summary>
    /// Defines the capabilities of the Matchbox recommender.
    /// </summary>
    [Serializable]
    public class MatchboxRecommenderCapabilities : IRecommenderCapabilities
    {
        #region ICapabilities implementation

        /// <summary>
        /// Gets a value indicating whether the learner is precompiled.
        /// </summary>
        /// <remarks>This is currently only relevant for Infer.NET.</remarks>
        public bool IsPrecompiled
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the learner supports missing data.
        /// </summary>
        public bool SupportsMissingData
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the learner supports sparse data.
        /// </summary>
        public bool SupportsSparseData
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the learner supports streamed data.
        /// </summary>
        public bool SupportsStreamedData
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the learner supports training on batched data.
        /// </summary>
        public bool SupportsBatchedTraining
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the learner supports distributed training.
        /// </summary>
        public bool SupportsDistributedTraining
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the learner supports incremental training.
        /// </summary>
        public bool SupportsIncrementalTraining
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the learner can compute how well it matches the training data 
        /// (usually for a given set of hyper-parameters).
        /// </summary>
        public bool SupportsModelEvidenceComputation
        {
            get { return false; }
        }

        #endregion

        #region IRecommenderCapabilities implementation

        /// <summary>
        /// Gets a value indicating whether the recommender supports positive-only data.
        /// </summary>
        public bool SupportsPositiveOnlyData
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender supports implicit feedback
        /// (e.g. counts of interactions, duration of interactions).
        /// </summary>
        public bool SupportsImplicitFeedback
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender supports non-binary ratings.
        /// </summary>
        public bool SupportsStarRatings
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender supports real-valued ratings.
        /// </summary>  
        public bool SupportsRealValuedRatings
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender supports multiple forms of feedback.
        /// </summary>
        public bool SupportsMultipleFormsOfFeedback
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender supports user-specific metadata.
        /// </summary>
        public bool SupportsUserFeatures
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender supports item-specific metadata.
        /// </summary>
        public bool SupportsItemFeatures
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender supports rating-specific metadata.
        /// </summary>
        public bool SupportsRatingFeatures
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender supports preference ratings
        /// (e.g. item A is better than item B).
        /// </summary>
        public bool SupportsPreferenceRatings
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender supports ratings of friends.
        /// </summary>
        public bool SupportsRatingsOfFriends
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender can predict 
        /// the rating distribution of a given user-item pair.
        /// </summary>
        /// <remakrks>This is only relevant for star-rating recommenders.</remakrks>
        public bool SupportsPredictionDistribution
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender can find related users.
        /// </summary>
        public bool SupportsRelatedUsers
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender can find related items.
        /// </summary>
        public bool SupportsRelatedItems
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender supports item diversity in predictions.
        /// </summary>
        public bool SupportsRecommendationDiversity
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender can compute predictive point estimates from a user-defined loss function.
        /// </summary>
        public bool SupportsCustomPredictionLossFunction 
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender can make 
        /// predictions for users who were not included in training.
        /// </summary>
        public bool SupportsColdStartUsers
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender can make 
        /// predictions for items which were not included in training.
        /// </summary>
        public bool SupportsColdStartItems
        {
            get { return true; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender can model not missing-at-random data
        /// (e.g. content restrictions due to provider).
        /// </summary>
        public bool SupportsNotMissingAtRandomData
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender is resistant to shilling.
        /// </summary>
        public bool IsResistantToShilling
        {
            get { return false; }
        }

        /// <summary>
        /// Gets a value indicating whether the recommender supports changes in state over time.
        /// </summary>
        public bool SupportsNonStationarity
        {
            get { return false; }
        }

        #endregion
    }
}
