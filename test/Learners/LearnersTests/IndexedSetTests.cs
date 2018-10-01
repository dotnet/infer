// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System;
    using System.Linq;
    using Xunit;
    using Microsoft.ML.Probabilistic.Collections;

    /// <summary>
    /// Tests for the <see cref="IndexedSet{T}"/> class.
    /// </summary>
    public class IndexedSetTests
    {
        /// <summary>
        /// A simple test for <see cref="IndexedSet{T}"/>.
        /// </summary>
        [Fact]
        public void SimpleIndexedSetTest()
        {
            const double FirstElement = 17;
            const double SecondElement = Math.PI;
            const double ThirdElement = 6.23123438874;
            const double FourthElement = 43;

            var indexedSet = new IndexedSet<double>(new[] { FirstElement, SecondElement });
            Assert.Equal(2, indexedSet.Count);
            Assert.True(indexedSet.Contains(FirstElement));
            var elements = indexedSet.Elements.ToArray();
            Assert.Equal(SecondElement, elements[1]);
            Assert.Equal(SecondElement, indexedSet.GetElementByIndex(1));

            int secondElementIndex;
            Assert.True(indexedSet.TryGetIndex(Math.PI, out secondElementIndex));
            Assert.Equal(1, secondElementIndex);

            indexedSet.Add(ThirdElement);
            Assert.Equal(3, indexedSet.Count);
            Assert.True(indexedSet.Contains(ThirdElement));

            int fourthElementIndex;
            Assert.Throws<ArgumentException>(() => indexedSet.Add(ThirdElement));
            Assert.Throws<ArgumentOutOfRangeException>(() => indexedSet.GetElementByIndex(3));
            Assert.False(indexedSet.TryGetIndex(FourthElement, out fourthElementIndex));
            
            indexedSet.Add(ThirdElement, false); 
            indexedSet.Add(FourthElement);
            Assert.Equal(FourthElement, indexedSet.GetElementByIndex(3));

            indexedSet.Clear();
            Assert.Equal(0, indexedSet.Count);

            Assert.Throws<ArgumentException>(() => new IndexedSet<double>(new[] { FirstElement, SecondElement, SecondElement }));
        }
    }
}