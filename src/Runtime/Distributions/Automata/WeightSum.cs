// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    /// <summary>
    /// Helper class for calculating sum of multiple <see cref="Weight"/>. It supports adding
    /// and then substracting infinite values. Only values which were added into sum can be substracted from it.
    /// </summary>
    public struct WeightSum
    {
        /// <summary>
        /// Number of non-infinite weights participating in the sum
        /// </summary>
        private readonly int count;

        /// <summary>
        /// Number of infinite weights participating in the sum
        /// </summary>
        private readonly int infCount;

        /// <summary>
        /// Sum of all non-infinite weights
        /// </summary>
        private readonly Weight sum;

        public WeightSum(int count, int infCount, Weight sum)
        {
            this.count = count;
            this.infCount = infCount;
            this.sum = sum;
        }

        public WeightSum(Weight init)
        {
            this.count = 1;
            if (init.IsInfinity)
            {
                this.infCount = 1;
                this.sum = Weight.Zero;
            }
            else
            {
                this.infCount = 0;
                this.sum = init;
            }
        }

        /// <summary>
        /// Constructs new empty accumulator instance
        /// </summary>
        public static WeightSum Zero() => new WeightSum(0, 0, Weight.Zero);

        public static WeightSum operator +(WeightSum a, Weight b) =>
            b.IsInfinity
                ? new WeightSum(a.count + 1, a.infCount + 1, a.sum)
                : new WeightSum(a.count + 1, a.infCount, a.sum + b);

        public static WeightSum operator -(WeightSum a, Weight b) =>
            a.count == 1
                ? WeightSum.Zero()
                : (b.IsInfinity
                    ? new WeightSum(a.count - 1, a.infCount - 1, a.sum)
                    : new WeightSum(a.count - 1, a.infCount, Weight.AbsoluteDifference(a.sum, b)));

        public int Count => this.count;

        public Weight Sum =>
            this.infCount != 0
                ? Weight.Infinity
                : this.sum;
    }
}
