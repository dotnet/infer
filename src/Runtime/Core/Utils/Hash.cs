// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Reference: "Fowler/Noll/Vo" or FNV hash (1991)
// http://www.isthe.com/chongo/tech/comp/fnv/

using System;
using System.Collections;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Utilities
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Utilities for implementing GetHashCode().
    /// </summary>
    /// <remarks><para>
    /// To hash an object with two fields x and y:
    /// <code>return Hash.Combine(x.GetHashCode(), y.GetHashCode());</code>
    /// </para><para>
    /// To hash an array:
    /// <example><code>
    /// int hash = Hash.Start;
    /// for(int i = 0; i &lt; Count; i++)
    ///     hash = Hash.Combine(hash, this[i].GetHashCode());
    /// return hash;
    /// </code></example>
    /// </para><para>
    /// Algorithm: FNV hash from http://www.isthe.com/chongo/tech/comp/fnv/
    /// </para><para>
    /// Note: You should not use xor to combine hash codes, even though it is
    /// recommended by MSDN.  xor is invariant to permutation, which means
    /// "abc" and "bac" and "cba" will hash to the same value (bad).
    /// Also xoring a hash value with itself produces 0, so "aab" and "b"
    /// will hash to the same value (bad).
    /// </para></remarks>
    public static class Hash
    {
        /// <summary>
        /// The recommended start value for a combined hash value
        /// </summary>
        public const Int32 Start = unchecked((Int32) 2166136261);

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1707:IdentifiersShouldNotContainUnderscores", MessageId = "Member")] private const Int32
            FNV_32_prime = 16777619;

        /// <summary>
        /// Combines an existing hash key with a new byte value
        /// </summary>
        /// <param name="hash">Current hash code</param>
        /// <param name="key">New byte of the new hash code to be integrated</param>
        /// <returns>A new hash-count that is neither order invariant nor "idempotent"</returns>
        private static Int32 Combine(Int32 hash, byte key)
        {
            return unchecked((hash ^ key)*FNV_32_prime);
        }

        /// <summary>
        /// Combines two int32 hash keys
        /// </summary>
        /// <param name="hash">First hash key</param>
        /// <param name="key">Second hash key</param>
        /// <returns>Incorporates the second hash key into the first hash key and returns the new, combined hash key</returns>
        public static Int32 Combine(Int32 hash, Int32 key)
        {
            unchecked
            {
                hash = Combine(hash, (byte)key);
                hash = Combine(hash, (byte)(key >> 8));
                hash = Combine(hash, (byte)(key >> 16));
                hash = Combine(hash, (byte)(key >> 24));
            }
            return hash;
        }

        /// <summary>
        /// Incorporates the hash key of a double into an existing hash key
        /// </summary>
        /// <param name="hash">Exisiting hash key</param>
        /// <param name="number">Floating point number to incorporate</param>
        /// <returns>The new, combined hash key</returns>
        public static int Combine(int hash, double number)
        {
            return Combine(hash, number.GetHashCode());
        }

        public static int GetHashCodeAsSequence(IEnumerable seq)
        {
            int hash = Hash.Start;
            foreach (object item in seq)
            {
                hash = Hash.Combine(hash, item == null ? 0 : item.GetHashCode());
            }
            return hash;
        }

        public static int GetHashCodeAsSequence<T>(IEnumerable<T> seq)
        {
            int hash = Hash.Start;
            foreach (T item in seq)
            {
                hash = Hash.Combine(hash, item == null ? 0 : item.GetHashCode());
            }
            return hash;
        }

        public static int GetHashCodeAsSet<T>(IEnumerable<T> set)
        {
            int count = 0;
            int hash = 0;
            // use a symmetric combining function for the elements.
            foreach (T item in set)
            {
                hash ^= item.GetHashCode();
                count++;
            }
            return Hash.Combine(Hash.Combine(Hash.Start, count), hash);
        }

        public static int GetHashCodeAsSet<T>(IEnumerable<T> set, IEqualityComparer<T> comparer)
        {
            int count = 0;
            int hash = 0;
            // use a symmetric combining function for the elements.
            foreach (T item in set)
            {
                hash ^= comparer.GetHashCode(item);
                count++;
            }
            return Hash.Combine(Hash.Combine(Hash.Start, count), hash);
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}