// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Xunit;
    using Microsoft.ML.Probabilistic.Collections;

    /// <summary>
    /// Tests for the <see cref="ArrayDictionary{T}"/> class.
    /// </summary>
    public class ArrayDictionaryTests
    {
        /// <summary>
        /// A simple dictionary test.
        /// </summary>
        [Fact]
        public void SimpleTest()
        {
            var dict = new ArrayDictionary<double>();

            double valueFor1;
            Assert.False(dict.ContainsKey(1));
            Assert.False(dict.TryGetValue(1, out valueFor1));
            Assert.Throws<KeyNotFoundException>(() => valueFor1 = dict[1]);

            dict.Add(1, 3.0);
            Assert.True(dict.ContainsKey(1));
            Assert.True(dict.TryGetValue(1, out valueFor1));
            Assert.Equal(3.0, valueFor1);
            Assert.Equal(3.0, dict[1]);
        }

        /// <summary>
        /// Tests adding multiple items into the dictionary.
        /// </summary>
        [Fact]
        public void ManyItemsTest()
        {
            const int LowerBoundInclusive = 5;
            const int UpperBoundExclusive = 15;
            const int InitialCapacity = 10;

            var dict = new ArrayDictionary<int>(InitialCapacity);

            for (int i = LowerBoundInclusive; i < UpperBoundExclusive; ++i)
            {
                if (i%2 == 0)
                {
                    dict.Add(i, i + 1);
                }
            }

            KeyValuePair<int, int>[] keyValuePairs = dict.ToArray();
            int keyValueIndex = 0;
            for (int key = 0; key < UpperBoundExclusive + 10; ++key)
            {
                int value;

                if (key < LowerBoundInclusive || key >= UpperBoundExclusive || key%2 != 0)
                {
                    Assert.True(!dict.ContainsKey(key));
                    Assert.False(dict.TryGetValue(key, out value));
                }
                else
                {
                    int expectedValue = key + 1;

                    Assert.True(dict.ContainsKey(key));
                    Assert.Equal(expectedValue, dict[key]);
                    Assert.True(dict.TryGetValue(key, out value));
                    Assert.Equal(expectedValue, value);

                    Assert.True(keyValueIndex < keyValuePairs.Length);
                    Assert.Equal(key, keyValuePairs[keyValueIndex].Key);
                    Assert.Equal(expectedValue, keyValuePairs[keyValueIndex].Value);
                    ++keyValueIndex;
                }
            }
        }

        /// <summary>
        /// Tests adding the same key with different values to the dictionary in various ways.
        /// </summary>
        [Fact]
        public void SameKeyDifferentValuesTest()
        {
            var dict = new ArrayDictionary<int>();
            dict.Add(1, 2);
            Assert.Equal(2, dict[1]);
            Assert.Throws<ArgumentException>(() => dict.Add(1, 3));
            Assert.Equal(2, dict[1]);
            dict[1] = 3;
            Assert.Equal(3, dict[1]);
        }
    }
}