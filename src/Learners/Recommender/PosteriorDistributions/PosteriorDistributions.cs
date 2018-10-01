// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System.Collections.Generic;

    /// <summary>
    /// The posterior distribution over the parameters of the Matchbox model.
    /// </summary>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    public class PosteriorDistributions<TUser, TItem>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PosteriorDistributions{TUser,TItem}"/> class.
        /// </summary>
        /// <param name="users">The user posterior distributions.</param>
        /// <param name="items">The item posterior distributions.</param>
        /// <param name="userFeatures">The user feature posterior distributions.</param>
        /// <param name="itemFeatures">The item feature posterior distributions.</param>
        public PosteriorDistributions(
            IDictionary<TUser, UserPosteriorDistribution> users,
            IDictionary<TItem, ItemPosteriorDistribution> items,
            IList<FeaturePosteriorDistribution> userFeatures,
            IList<FeaturePosteriorDistribution> itemFeatures)
        {
            this.Users = users;
            this.Items = items;
            this.UserFeatures = userFeatures;
            this.ItemFeatures = itemFeatures;
        }

        /// <summary>
        /// Gets the user posterior distributions.
        /// </summary>
        public IDictionary<TUser, UserPosteriorDistribution> Users { get; }

        /// <summary>
        /// Gets the item posterior distributions.
        /// </summary>
        public IDictionary<TItem, ItemPosteriorDistribution> Items { get; }

        /// <summary>
        /// Gets the user feature posterior distributions.
        /// </summary>
        public IList<FeaturePosteriorDistribution> UserFeatures { get; }

        /// <summary>
        /// Gets the item feature posterior distributions.
        /// </summary>
        public IList<FeaturePosteriorDistribution> ItemFeatures { get; }
    }
}
