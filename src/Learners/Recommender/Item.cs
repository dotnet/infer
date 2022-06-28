// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;

    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// Represents an item in a recommendation system.
    /// </summary>
    [Serializable]
    public class Item
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Item"/> class.
        /// </summary>
        /// <param name="id">The identifier of the item.</param>
        /// <param name="features">The features of the item or null if item has no features.</param>
        public Item(string id, Vector features)
        {
            if (string.IsNullOrWhiteSpace(id))
            {
                throw new ArgumentException("The identifier of the item can not be null or whitespace.", nameof(id));
            }
            
            this.Id = id;
            this.Features = features;
        }

        /// <summary>
        /// Gets the identifier.
        /// </summary>
        public string Id { get; private set; }

        /// <summary>
        /// Gets the feature vector.
        /// </summary>
        public Vector Features { get; private set; }

        /// <summary>
        /// Determines whether the specified <see cref="Object"/> is equal to this item.
        /// </summary>
        /// <param name="obj">The object to compare with this item.</param>
        /// <returns>True if <paramref name="obj"/> is <see cref="Item"/> and has the same <see cref="Id"/>, false otherwise.</returns>
        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
            {
                return false;
            }

            return this.Id.Equals(((Item)obj).Id);
        }

        /// <summary>
        /// Gets the hash code of this item, which is based entirely on its <see cref="Id"/>.
        /// </summary>
        /// <returns>A value of the hash code.</returns>
        public override int GetHashCode()
        {
            return this.Id.GetHashCode();
        }

        /// <summary>
        /// Returns a string that represents the current object.
        /// </summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString()
        {
            return this.Id;
        }
    }
}