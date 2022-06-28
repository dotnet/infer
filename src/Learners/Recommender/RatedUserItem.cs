// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System.Diagnostics;

    /// <summary>
    /// A triple of user, item, and rating.
    /// </summary>
    public class RatedUserItem
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RatedUserItem"/> class with a given user, item, and rating.
        /// </summary>
        /// <param name="user">The user.</param>
        /// <param name="item">The item.</param>
        /// <param name="rating">The rating.</param>
        public RatedUserItem(User user, Item item, int rating)
        {
            Debug.Assert(user != null, "Observation user can not be null.");
            Debug.Assert(item != null, "Observation item can not be null.");

            this.User = user;
            this.Item = item;
            this.Rating = rating;
        }

        /// <summary>
        /// Gets the user.
        /// </summary>
        public User User { get; private set; }

        /// <summary>
        /// Gets the item.
        /// </summary>
        public Item Item { get; private set; }

        /// <summary>
        /// Gets the rating.
        /// </summary>
        public int Rating { get; private set; }

        /// <summary>
        /// Returns a string that represents the current object.
        /// </summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString()
        {
            return string.Format("({0}, {1}) -> {2}", this.User, this.Item, this.Rating);
        }
    }
}