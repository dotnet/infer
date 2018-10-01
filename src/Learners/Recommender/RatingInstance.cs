// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    /// <summary>
    /// Represents a user-item-rating triple.
    /// </summary>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TRating">The type of a rating.</typeparam>
    public class RatingInstance<TUser, TItem, TRating>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RatingInstance{TUser,TItem,TRating}"/> class.
        /// </summary>
        /// <param name="user">The user.</param>
        /// <param name="item">The item.</param>
        /// <param name="rating">The rating.</param>
        public RatingInstance(TUser user, TItem item, TRating rating)
        {
            this.User = user;
            this.Item = item;
            this.Rating = rating;
        }

        /// <summary>
        /// Gets the user.
        /// </summary>
        public TUser User { get; private set; }

        /// <summary>
        /// Gets the item.
        /// </summary>
        public TItem Item { get; private set; }

        /// <summary>
        /// Gets the rating.
        /// </summary>
        public TRating Rating { get; private set; }
    }
}
