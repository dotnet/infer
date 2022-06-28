// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;
    using static Microsoft.ML.Probabilistic.Learners.MatchboxRecommender;

    /// <summary>
    /// Contains all the data mappings used in evaluation.
    /// </summary>
    public static class RecommenderMappings
    {
        /// <summary>
        /// Initializes static members of the <see cref="Mappings"/> class.
        /// </summary>
        static RecommenderMappings()
        {
            StarRatingRecommender = new RunnerStarRatingRecommenderMapping();
        }

        /// <summary>
        /// Gets the mapping from dataset format to the common recommender format.
        /// </summary>
        public static IStarRatingRecommenderMapping<RecommenderDataset, RatedUserItem, User, Item, int, DummyFeatureSource, Vector> StarRatingRecommender
        {
            get;
            private set;
        }
    }
}
