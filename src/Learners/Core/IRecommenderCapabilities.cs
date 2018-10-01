// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    /// <summary>
    /// Interface to a recommender capabilities.
    /// </summary>
    public interface IRecommenderCapabilities : ICapabilities
    {
        #region Type of data

        /// <summary>
        /// Gets a value indicating whether the recommender supports positive-only data.
        /// </summary>
        bool SupportsPositiveOnlyData { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender supports implicit feedback
        /// (e.g. counts of interactions, duration of interactions).
        /// </summary>        
        bool SupportsImplicitFeedback { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender supports non-binary ratings.
        /// </summary>
        bool SupportsStarRatings { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender supports real-valued ratings.
        /// </summary>        
        bool SupportsRealValuedRatings { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender supports multiple forms of feedback.
        /// </summary>
        bool SupportsMultipleFormsOfFeedback { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender supports user-specific metadata.
        /// </summary>
        bool SupportsUserFeatures { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender supports item-specific metadata.
        /// </summary>
        bool SupportsItemFeatures { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender supports rating-specific metadata.
        /// </summary>
        bool SupportsRatingFeatures { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender supports preference ratings
        /// (e.g. item A is better than item B).
        /// </summary>
        bool SupportsPreferenceRatings { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender supports ratings of friends.
        /// </summary>
        bool SupportsRatingsOfFriends { get; }

        #endregion

        #region Prediction

        /// <summary>
        /// Gets a value indicating whether the recommender can predict 
        /// the rating distribution of a given user-item pair.
        /// </summary>
        /// <remakrks>This is only relevant for star-rating recommenders.</remakrks>
        bool SupportsPredictionDistribution { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender can find related users.
        /// </summary>
        bool SupportsRelatedUsers { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender can find related items.
        /// </summary>
        bool SupportsRelatedItems { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender supports item diversity in predictions.
        /// </summary>
        bool SupportsRecommendationDiversity { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender can compute predictive point estimates from a user-defined loss function.
        /// </summary>
        bool SupportsCustomPredictionLossFunction { get; }

        #endregion

        #region Model

        /// <summary>
        /// Gets a value indicating whether the recommender can make 
        /// predictions for users who were not included in training.
        /// </summary>
        bool SupportsColdStartUsers { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender can make 
        /// predictions for items which were not included in training.
        /// </summary>
        bool SupportsColdStartItems { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender can model not missing-at-random data
        /// (e.g. content restrictions due to provider).
        /// </summary>
        bool SupportsNotMissingAtRandomData { get; }

        /// <summary>
        /// Gets a value indicating whether the recommender is resistant to shilling.
        /// </summary>
        bool IsResistantToShilling { get; }
        
        /// <summary>
        /// Gets a value indicating whether the recommender supports changes in state over time.
        /// </summary>
        bool SupportsNonStationarity { get; }

        #endregion
    }
}
