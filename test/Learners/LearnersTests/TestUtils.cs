// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Tests
{
    using System;
    using System.Collections.Generic;
    using Xunit;

    /// <summary>
    /// Help class for assert options that are not implemented in xUnit.
    /// </summary>
    internal class AssertHelper : Xunit.Assert
    {
        /// <summary>
        /// Compare two doubles with given precision.
        /// </summary>
        /// <param name="expected">Expected value.</param>
        /// <param name="observed">Actual value.</param>
        /// <param name="eps">Precision.</param>
        public static void Equal(double expected, double observed, double eps)
        {
            // Infinty check
            if (expected == observed)
            {
                return;
            }

            Assert.True(Math.Abs(expected - observed) < eps, $"Equality failure\n. Expected: {expected}\nActual:   {observed}");
        }

        /// <summary>
        /// Asserts the equality of two given dictionaries using a given value comparer.
        /// </summary>
        /// <typeparam name="TKey">The type of the key in the dictionary.</typeparam>
        /// <typeparam name="TValue">The type of the value in the dictionary.</typeparam>
        /// <param name="expectedDictionary">The expected dictionary.</param>
        /// <param name="actualDictionary">The actual dictionary.</param>
        /// <param name="valueComparer">The comparer for dictionary values.</param>
        public static void Equal<TKey, TValue>(
            IDictionary<TKey, TValue> expectedDictionary,
            IDictionary<TKey, TValue> actualDictionary,
            Func<TValue, TValue, bool> valueComparer)
        {
            Assert.True(Comparers.Dictionary(expectedDictionary, actualDictionary, valueComparer), "Dictionaries are not equal.");
        }
    }
}
