// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// Hand-rolled implementation of <see cref="ValueTuple{T1,T2}"/> for integers.
    /// </summary>
    /// <remarks>
    /// <see cref="ValueTuple{T1,T2}.Equals(ValueTuple{T1,T2})"/> uses
    /// <see cref="EqualityComparer{T}.Default"/> to implement he comparison. Accessing this
    /// static property involves a lookup in a hash-table each time it is used due to the way
    /// static fields in generic classes are implemented in CLR. It appears to be measurably
    /// slower when `(int, int)` is used as a key in a dictionary. To avoid this, this class
    /// specializes `Equals()` to integers and thus avoids lookup for comparer.
    /// </remarks>
    public readonly struct IntPair : IEquatable<IntPair>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="IntPair"/> struct.
        /// </summary>
        /// <param name="first">The first element of the pair.</param>
        /// <param name="second">The second element of the pair.</param>
        public IntPair(int first, int second)
        {
            this.First = first;
            this.Second = second;
            
        }

        /// <summary>
        /// Gets or sets the first element of the pair.
        /// </summary>
        public int First { get; }

        /// <summary>
        /// Gets or sets the second element of the pair.
        /// </summary>
        public int Second { get; }

        /// <summary>
        /// Gets the string representation of this pair.
        /// </summary>
        /// <returns>The string representation of the pair.</returns>
        public override string ToString()
        {
            return "[" + this.First + "," + this.Second + "]";
        }

        /// <summary>
        /// Checks if this object is equal to <paramref name="that"/>.
        /// </summary>
        /// <param name="that">The object to compare this object with.</param>
        /// <returns>
        /// <see langword="true"/> if this object is equal to <paramref name="that"/>,
        /// <see langword="false"/> otherwise.
        /// </returns>
        public bool Equals(IntPair that)
        {
            return this.First == that.First && this.Second == that.Second;

        }

        /// <summary>
        /// Checks if this object is equal to <paramref name="that"/>.
        /// </summary>
        /// <param name="that">The object to compare this object with.</param>
        /// <returns>
        /// <see langword="true"/> if this object is equal to <paramref name="that"/>,
        /// <see langword="false"/> otherwise.
        /// </returns>
        public override bool Equals(object that)
        {
            return that is IntPair thatPair && this.Equals(thatPair);
        }

        /// <summary>
        /// Computes the hash code of this object.
        /// </summary>
        /// <returns>The computed hash code.</returns>
        public override int GetHashCode()
        {
            int h1 = this.First;
            int h2 = this.Second;
            uint rol5 = ((uint)h1 << 5) | ((uint)h1 >> 27);
            return ((int)rol5 + h1) ^ h2;
        }

        public void Deconstruct(out int first, out int second)
        {
            first = this.First;
            second = this.Second;
        }
    }
}