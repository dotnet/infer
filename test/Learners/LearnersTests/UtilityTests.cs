// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Tests
{
    using System;
    using Xunit;
    using Microsoft.ML.Probabilistic.Utilities;

    

    /// <summary>
    /// Tests for learner utilities.
    /// </summary>
    public class UtilityTests
    {
        /// <summary>
        /// Tests batch selection.
        /// </summary>
        [Fact]
        public void TestBatchSelection()
        {
            var array = Util.ArrayInit(736, i => i);

            // Test invalid arguments
            Assert.Throws<ArgumentNullException>(() => Utilities.GetBatch<int>(0, null, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => Utilities.GetBatch(0, array, 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => Utilities.GetBatch(1, array, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => Utilities.GetBatch(-1, array, 1));
            Assert.Throws<ArgumentException>(() => Utilities.GetBatch(0, array, 737));

            // A single batch
            var segment = Utilities.GetBatch(0, array, 1);
            Assert.NotNull(segment);
            Assert.Equal(736, segment.Count);
            Assert.Equal(0, segment[0]);
            Assert.Equal(43, segment[43]);

            // 3 batches
            segment = Utilities.GetBatch(0, array, 3);
            Assert.NotNull(segment);
            Assert.Equal(246, segment.Count);
            Assert.Equal(0, segment[0]);

            segment = Utilities.GetBatch(1, array, 3);
            Assert.NotNull(segment);
            Assert.Equal(245, segment.Count);
            Assert.Equal(246, segment[0]);

            segment = Utilities.GetBatch(2, array, 3);
            Assert.NotNull(segment);
            Assert.Equal(245, segment.Count);
            Assert.Equal(491, segment[0]);

            Assert.Throws<ArgumentOutOfRangeException>(() => Utilities.GetBatch(3, array, 3));

            // 48 batches
            segment = Utilities.GetBatch(0, array, 48);
            Assert.NotNull(segment);
            Assert.Equal(16, segment.Count);
            Assert.Equal(0, segment[0]);

            segment = Utilities.GetBatch(16, array, 48);
            Assert.NotNull(segment);
            Assert.Equal(15, segment.Count);
            Assert.Equal(256, segment[0]);

            segment = Utilities.GetBatch(47, array, 48);
            Assert.NotNull(segment);
            Assert.Equal(15, segment.Count);
            Assert.Equal(735, segment[14]);

            // 735 batches
            segment = Utilities.GetBatch(0, array, 735);
            Assert.NotNull(segment);
            Assert.Equal(2, segment.Count);
            Assert.Equal(0, segment[0]);

            segment = Utilities.GetBatch(433, array, 735);
            Assert.NotNull(segment);
            Assert.Equal(1, segment.Count);
            Assert.Equal(434, segment[0]);

            segment = Utilities.GetBatch(734, array, 735);
            Assert.NotNull(segment);
            Assert.Equal(1, segment.Count);
            Assert.Equal(735, segment[0]);

            Assert.Throws<ArgumentOutOfRangeException>(() => Utilities.GetBatch(735, array, 735));
            Assert.Throws<ArgumentException>(() => Utilities.GetBatch(734, array, 737));
        }
    }
}
