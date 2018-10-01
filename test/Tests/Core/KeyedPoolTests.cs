// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests
{
    using System.Threading;
    using System.Threading.Tasks;
    using Xunit;
    using Microsoft.ML.Probabilistic.Collections;
    using Assert = Xunit.Assert;

    /// <summary>
    /// Tests for <see cref="KeyedPool{TKey, TItem}"/>.
    /// </summary>
    public class KeyedPoolTests
    {
        /// <summary>
        /// Tests the correctness of the pool logic.
        /// </summary>
        [Fact]
        public void LogicTest()
        {
            int itemId = 0;
            var pool = new KeyedPool<string, TestItem>(() => new TestItem { Data = itemId++ });
            
            // Acquire an item from an empty pool (will create a new item)
            var item = pool.Acquire("1");
            Assert.Equal(0, item.Data);

            // Return the item back under a different key
            pool.Release(item, "2");
            
            // Get an item for that key, should get the existing item back
            item = pool.Acquire("2");
            Assert.Equal(0, item.Data);

            // Create another new item
            var item2 = pool.Acquire("3");
            Assert.Equal(1, item2.Data);

            // Return both to the pool
            pool.Release(item, "1");
            pool.Release(item2, "2");

            // Get an item for "1", should return the one registered against it
            Assert.Equal(0, pool.Acquire("1").Data);

            // Get another item for "1", should return the one registered against "2"
            Assert.Equal(1, pool.Acquire("1").Data);
        }

        /// <summary>
        /// Make sure we don't create unnecessary items
        /// and don't return the same item to two different callers at the same time.
        /// </summary>
        [Fact]
        public void StressTest()
        {
            int itemId = 0;
            var pool = new KeyedPool<int, TestItem>(() => new TestItem { Data = itemId++ });

            const int MaxThreads = 4;
            bool[] usedItems = new bool[MaxThreads];

            Parallel.For(
                0,
                MaxThreads * 3,
                new ParallelOptions { MaxDegreeOfParallelism = MaxThreads },
                i =>
                    {
                        for (int j = 0; j < 100; ++j)
                        {
                            TestItem item;
                            lock (usedItems)
                            {
                                item = pool.Acquire(i);
                                Assert.True(item.Data < MaxThreads);
                                Assert.False(usedItems[item.Data]);
                                usedItems[item.Data] = true;
                            }

                            Thread.Sleep(1);

                            lock (usedItems)
                            {
                                Assert.True(usedItems[item.Data]);
                                usedItems[item.Data] = false;
                                pool.Release(item, i);
                            }
                        }
                    });
        }

        /// <summary>
        /// A simple item class for testing the pool.
        /// </summary>
        private class TestItem
        {
            /// <summary>
            /// Gets or sets item data.
            /// </summary>
            public int Data { get; set; }
        }
    }
}
