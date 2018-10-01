// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    /// <summary>
    /// Extension methods to simplify mapping chaining syntax.
    /// </summary>
    public static class RecommenderMappingExtensions
    {
        /// <summary>
        /// Chains a given recommender mapping with a splitting mapping, which splits the mapped data into training and test parts.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
        /// <param name="mapping">The recommender mapping to chain with the splitting mapping.</param>
        /// <param name="trainingOnlyUserFraction">The fraction of users included in the training set only.</param>
        /// <param name="testUserRatingTrainingFraction">The fraction of ratings in the training set for each user who is presented in both sets.</param>
        /// <param name="coldUserFraction">The fraction of users included in the test set only.</param>
        /// <param name="coldItemFraction">The fraction of items included in the test set only.</param>
        /// <param name="ignoredUserFraction">The fraction of users not included in either the training or the test set.</param>
        /// <param name="ignoredItemFraction">The fraction of items not included in either the training or the test set.</param>
        /// <param name="removeOccasionalColdItems">Whether the occasionally produced cold items should be removed from the test set.</param>
        /// <returns>The train/test splitting mapping.</returns>
        public static TrainTestSplittingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>
            SplitToTrainTest<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>(
            this IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues> mapping,
            double trainingOnlyUserFraction,
            double testUserRatingTrainingFraction,
            double coldUserFraction = 0,
            double coldItemFraction = 0,
            double ignoredUserFraction = 0,
            double ignoredItemFraction = 0,
            bool removeOccasionalColdItems = false)
        {
            return new TrainTestSplittingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>(
                mapping, trainingOnlyUserFraction, testUserRatingTrainingFraction, coldUserFraction, coldItemFraction, ignoredUserFraction, ignoredItemFraction, removeOccasionalColdItems);
        }

        /// <summary>
        /// Chains a given recommender mapping with a splitting mapping, which splits the mapped data into training and test parts.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TRating">The type of a rating.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
        /// <param name="mapping">The recommender mapping to chain with the splitting mapping.</param>
        /// <param name="trainingOnlyUserFraction">The fraction of users included in the training set only.</param>
        /// <param name="testUserRatingTrainingFraction">The fraction of ratings in the training set for each user who is presented in both sets.</param>
        /// <param name="coldUserFraction">The fraction of users included in the test set only.</param>
        /// <param name="coldItemFraction">The fraction of items included in the test set only.</param>
        /// <param name="ignoredUserFraction">The fraction of users not included in either the training or the test set.</param>
        /// <param name="ignoredItemFraction">The fraction of items not included in either the training or the test set.</param>
        /// <param name="removeOccasionalColdItems">Whether the occasionally produced cold items should be removed from the test set.</param>
        /// <returns>The train/test splitting mapping.</returns>
        public static TrainTestSplittingRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>
            SplitToTrainTest<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>(
            this IRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues> mapping,
            double trainingOnlyUserFraction,
            double testUserRatingTrainingFraction,
            double coldUserFraction = 0,
            double coldItemFraction = 0,
            double ignoredUserFraction = 0,
            double ignoredItemFraction = 0,
            bool removeOccasionalColdItems = false)
        {
            return new TrainTestSplittingRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>(
                mapping, trainingOnlyUserFraction, testUserRatingTrainingFraction, coldUserFraction, coldItemFraction, ignoredUserFraction, ignoredItemFraction, removeOccasionalColdItems);
        }

        /// <summary>
        /// Chains a given recommender mapping with a splitting mapping, which splits the mapped data into training and test parts.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TRating">The type of a rating.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
        /// <param name="mapping">The recommender mapping to chain with the splitting mapping.</param>
        /// <param name="trainingOnlyUserFraction">The fraction of users included in the training set only.</param>
        /// <param name="testUserRatingTrainingFraction">The fraction of ratings in the training set for each user who is presented in both sets.</param>
        /// <param name="coldUserFraction">The fraction of users included in the test set only.</param>
        /// <param name="coldItemFraction">The fraction of items included in the test set only.</param>
        /// <param name="ignoredUserFraction">The fraction of users not included in either the training or the test set.</param>
        /// <param name="ignoredItemFraction">The fraction of items not included in either the training or the test set.</param>
        /// <param name="removeOccasionalColdItems">Whether the occasionally produced cold items should be removed from the test set.</param>
        /// <returns>The train/test splitting mapping.</returns>
        public static TrainTestSplittingStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>
            SplitToTrainTest<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>(
            this IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues> mapping,
            double trainingOnlyUserFraction,
            double testUserRatingTrainingFraction,
            double coldUserFraction = 0,
            double coldItemFraction = 0,
            double ignoredUserFraction = 0,
            double ignoredItemFraction = 0,
            bool removeOccasionalColdItems = false)
        {
            return new TrainTestSplittingStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>(
                mapping, trainingOnlyUserFraction, testUserRatingTrainingFraction, coldUserFraction, coldItemFraction, ignoredUserFraction, ignoredItemFraction, removeOccasionalColdItems);
        }

        /// <summary>
        /// Chains a given recommender mapping with a mapping that augments the mapped data with negative examples,
        /// while treating all instances provided by the original mapping as positive.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
        /// <param name="mapping">The mapping to chain with the negative data generator mapping.</param>
        /// <param name="itemHistogramAdjustment">
        /// The degree to which each element of the item histogram is raised before sampling.
        /// Should be different from 1.0 if it is believed that there is some quality
        /// bar which drives popular items to be more generally liked.
        /// </param>
        /// <returns>The negative data generator mapping.</returns>
        public static NegativeDataGeneratorMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>
            WithGeneratedNegativeData<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>(
            this IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues> mapping,
            double itemHistogramAdjustment = 1.2)
        {
            return new NegativeDataGeneratorMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>(
                mapping, itemHistogramAdjustment);
        }

        /// <summary>
        /// Chains a given recommender mapping with an evaluator mapping,
        /// making it possible to evaluate a recommender on the mapped data.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TRating">The type of a rating.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
        /// <param name="mapping">The recommender mapping to chain with the evaluator mapping.</param>
        /// <returns>The evaluator mapping.</returns>
        public static RecommenderEvaluatorMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>
            ForEvaluation<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>(
            this IRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues> mapping)
        {
            return new RecommenderEvaluatorMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>(mapping);
        }

        /// <summary>
        /// Chains a given recommender mapping with an evaluator mapping,
        /// making it possible to evaluate a recommender on the mapped data.
        /// </summary>
        /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <typeparam name="TUser">The type of a user.</typeparam>
        /// <typeparam name="TItem">The type of an item.</typeparam>
        /// <typeparam name="TRating">The type of a rating.</typeparam>
        /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
        /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
        /// <param name="mapping">The recommender mapping to chain with the evaluator mapping.</param>
        /// <returns>The evaluator mapping.</returns>
        public static StarRatingRecommenderEvaluatorMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>
            ForEvaluation<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>(
            this IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues> mapping)
        {
            return new StarRatingRecommenderEvaluatorMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>(mapping);
        }
    }
}