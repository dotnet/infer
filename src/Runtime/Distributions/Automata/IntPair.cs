// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represents a pair of integers.
    /// </summary>
    /// <remarks>
    /// This structure equipped with the provided equality comparer can be used as an efficient dictionary key.
    /// </remarks>
    internal struct IntPair
    {
        /// <summary>
        /// Initializes static members of the <see cref="IntPair"/> struct.
        /// </summary>
        static IntPair()
        {
            DefaultEqualityComparer = new EqualityComparer();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="IntPair"/> struct.
        /// </summary>
        /// <param name="first">The first integer.</param>
        /// <param name="second">The second integer.</param>
        public IntPair(int first, int second)
            : this()
        {
            this.First = first;
            this.Second = second;
        }

        /// <summary>
        /// Gets the default equality comparer for this class.
        /// </summary>
        public static EqualityComparer DefaultEqualityComparer { get; private set; }
        
        /// <summary>
        /// Gets the first integer.
        /// </summary>
        public int First { get; private set; }

        /// <summary>
        /// Gets the second integer.
        /// </summary>
        public int Second { get; private set; }

        /// <summary>
        /// Gets a string representation of this object.
        /// </summary>
        /// <returns>A string representation of this object.</returns>
        public override string ToString()
        {
            return "[" + this.First + "," + this.Second + "]";
        }

        /// <summary>
        /// Checks whether this pair is equal to a given object.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns><see langword="true"/> if this pair is equal to the given one, <see langword="false"/> otherwise.</returns>
        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
            {
                return false;
            }

            return this.Equals((IntPair)obj);
        }

        /// <summary>
        /// Checks whether this object is equal to a given pair.
        /// </summary>
        /// <param name="pair">The pair to compare with.</param>
        /// <returns><see langword="true"/> if this pair is equal to the given one, <see langword="false"/> otherwise.</returns>
        public bool Equals(IntPair pair)
        {
            return this.First == pair.First && this.Second == pair.Second;
        }

        /// <summary>
        /// Gets the hash code of this pair.
        /// </summary>
        /// <returns>The has code of this pair.</returns>
        public override int GetHashCode()
        {
            return Hash.Combine(this.First.GetHashCode(), this.Second.GetHashCode());
        }

        /// <summary>
        /// The default equality comparer for the <see cref="IntPair"/> class.
        /// </summary>
        public class EqualityComparer : IEqualityComparer<IntPair>
        {
            /// <summary>
            /// Checks whether two given pairs are equal.
            /// </summary>
            /// <param name="pair1">The first pair.</param>
            /// <param name="pair2">The second pair.</param>
            /// <returns><see langword="true"/> if the given pairs are equal, <see langword="false"/> otherwise.</returns>
            public bool Equals(IntPair pair1, IntPair pair2)
            {
                return pair1.Equals(pair2);
            }

            /// <summary>
            /// Computes the hash code of a given pair.
            /// </summary>
            /// <param name="pair">The pair to compute the hash code of.</param>
            /// <returns>The computed hash code.</returns>
            public int GetHashCode(IntPair pair)
            {
                return pair.GetHashCode();
            }
        }
    }
}