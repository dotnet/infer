// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    /// <summary>
    /// Interface to a Matchbox recommender system.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TRatingDistribution">The type of a distribution over ratings.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a source of features.</typeparam>
    public interface IMatchboxRecommender<in TInstanceSource, TUser, TItem, TRatingDistribution, in TFeatureSource> :
        IRecommender<TInstanceSource, TUser, TItem, int, TRatingDistribution, TFeatureSource>, ICustomSerializable
    {
        /// <summary>
        /// Gets the recommender settings.
        /// </summary>
        new MatchboxRecommenderSettings Settings { get; }

        /// <summary>
        /// Gets a copy of the posterior distribution over the parameters of the Matchbox model.
        /// </summary>
        /// <returns>The posterior distributions.</returns>
        PosteriorDistributions<TUser, TItem> GetPosteriorDistributions();
    }
}
