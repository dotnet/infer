// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    /// <summary>
    /// Represents 
    /// <see cref="TrainTestSplittingRatingRecommenderMapping{TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues}"/>
    /// for star-rating based recommenders.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TRating">The type of a rating.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
    /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
    public class TrainTestSplittingStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues> :
        TrainTestSplittingRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>,
        IStarRatingRecommenderMapping<SplitInstanceSource<TInstanceSource>, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>
    {
        /// <summary>
        /// The wrapped recommender mapping.
        /// </summary>
        private readonly IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues> mapping;

        /// <summary>
        /// Initializes a new instance of the
        /// <see cref="TrainTestSplittingStarRatingRecommenderMapping{TInstanceSource,TInstance,TUser,TItem,TRating,TFeatureSource,TFeatureValues}"/> class.
        /// </summary>
        /// <param name="mapping">The wrapped recommender mapping.</param>
        /// <param name="trainingOnlyUserFraction">The fraction of users included in the training set only.</param>
        /// <param name="testUserRatingTrainingFraction">The fraction of ratings in the training set for each user who is presented in both sets.</param>
        /// <param name="coldUserFraction">The fraction of users included in the test set only.</param>
        /// <param name="coldItemFraction">The fraction of items included in the test set only.</param>
        /// <param name="ignoredUserFraction">The fraction of users not included in either the training or the test set.</param>
        /// <param name="ignoredItemFraction">The fraction of items not included in either the training or the test set.</param>
        /// <param name="removeOccasionalColdItems">Whether the occasionally produced cold items should be removed from the test set.</param>
        public TrainTestSplittingStarRatingRecommenderMapping(
            IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues> mapping,
            double trainingOnlyUserFraction,
            double testUserRatingTrainingFraction,
            double coldUserFraction = 0,
            double coldItemFraction = 0,
            double ignoredUserFraction = 0,
            double ignoredItemFraction = 0,
            bool removeOccasionalColdItems = false)
            : base(mapping, trainingOnlyUserFraction, testUserRatingTrainingFraction, coldUserFraction, coldItemFraction, ignoredUserFraction, ignoredItemFraction, removeOccasionalColdItems)
        {
            this.mapping = mapping;
        }

        /// <summary>
        /// Provides the object describing how ratings provided by the instance source map to stars by delegating the call to the wrapped mapping.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <returns>The object describing how ratings provided by the instance source map to stars.</returns>
        public IStarRatingInfo<TRating> GetRatingInfo(SplitInstanceSource<TInstanceSource> instanceSource)
        {
            return this.mapping.GetRatingInfo(instanceSource.InstanceSource);
        }
    }
}
