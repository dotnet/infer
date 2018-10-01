// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    /// <summary>
    /// A mapping used by an evaluator of recommenders.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TRating">The type of a rating.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
    /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
    public class StarRatingRecommenderEvaluatorMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues> :
        RecommenderEvaluatorMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues>,
        IStarRatingRecommenderEvaluatorMapping<TInstanceSource, TUser, TItem, TRating>
    {
        /// <summary>
        /// Wrapped recommender mapping.
        /// </summary>
        private readonly IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues> mapping;

        /// <summary>
        /// Initializes a new instance of the
        /// <see cref="StarRatingRecommenderEvaluatorMapping{TInstanceSource,TInstance,TUser,TItem,TRating,TFeatureSource,TFeatureValues}"/> class.
        /// </summary>
        /// <param name="mapping">The recommender mapping.</param>
        public StarRatingRecommenderEvaluatorMapping(
            IStarRatingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues> mapping)
            : base(mapping)
        {
            this.mapping = mapping;
        }
        
        /// <summary>
        /// Gets the object describing how ratings provided by the instance source map to stars.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <returns>The object describing how ratings provided by the instance source map to stars.</returns>
        public IStarRatingInfo<TRating> GetRatingInfo(TInstanceSource instanceSource)
        {
            return this.mapping.GetRatingInfo(instanceSource);
        }
    }
}