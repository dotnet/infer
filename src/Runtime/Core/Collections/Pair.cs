// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Collections
{
    using System;

    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represent a pair of elements.
    /// Should be used instead of <see cref="Tuple{T1,T2}"/> when a value type is preferred, e.g. for performance reasons.
    /// </summary>
    /// <typeparam name="T1">The type of a first element.</typeparam>
    /// <typeparam name="T2">The type of a second element.</typeparam>
    public struct Pair<T1, T2>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Pair{T1,T2}"/> struct.
        /// </summary>
        /// <param name="first">The first element of the pair.</param>
        /// <param name="second">The second element of the pair.</param>
        public Pair(T1 first, T2 second)
            : this()
        {
            this.First = first;
            this.Second = second;
        }

        /// <summary>
        /// Gets or sets the first element of the pair.
        /// </summary>
        public T1 First { get; set; }

        /// <summary>
        /// Gets or sets the second element of the pair.
        /// </summary>
        public T2 Second { get; set; }

        /// <summary>
        /// Gets the string representation of this pair.
        /// </summary>
        /// <returns>The string representation of the pair.</returns>
        public override string ToString()
        {
            return "[" + this.First + "," + this.Second + "]";
        }

        /// <summary>
        /// Checks if this object is equal to <paramref name="obj"/>.
        /// </summary>
        /// <param name="obj">The object to compare this object with.</param>
        /// <returns>
        /// <see langword="true"/> if this object is equal to <paramref name="obj"/>,
        /// <see langword="false"/> otherwise.
        /// </returns>
        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
            {
                return false;
            }

            var objCasted = (Pair<T1, T2>)obj;
            return object.Equals(this.First, objCasted.First) && object.Equals(this.Second, objCasted.Second);
        }

        /// <summary>
        /// Computes the hash code of this object.
        /// </summary>
        /// <returns>The computed hash code.</returns>
        public override int GetHashCode()
        {
            int result = Hash.Start;
            result = Hash.Combine(result, this.First != null ? this.First.GetHashCode() : 0);
            result = Hash.Combine(result, this.Second != null ? this.Second.GetHashCode() : 0);
            return result;
        }
    }

    /// <summary>
    /// Represents a pair of elements.
    /// </summary>
    public static class Pair
    {
        /// <summary>
        /// Creates a pair of elements.
        /// </summary>
        /// <param name="first">The first element of the pair.</param>
        /// <param name="second">The second element of the pair.</param>
        /// <typeparam name="T1">The type of the first element of the pair.</typeparam>
        /// <typeparam name="T2">The type of the second element of the pair.</typeparam>
        /// <returns>The created pair.</returns>
        public static Pair<T1, T2> Create<T1, T2>(T1 first, T2 second)
        {
            return new Pair<T1, T2>(first, second);
        }
    }
}