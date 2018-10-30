// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// Efficiently implements histogram sampling without replacement.
    /// </summary>
    /// <remarks>
    /// Represents a binary sum tree. The leaves hold user-specified values,
    /// and the parent nodes contain the sum of their children.
    /// </remarks>
    internal class HistogramSampler
    {
        /// <summary>
        /// The nodes of the tree.
        /// </summary>
        private readonly List<int> nodes;

        /// <summary>
        /// The index of the first leaf in the tree.
        /// </summary>
        private readonly int firstLeafIndex;

        /// <summary>
        /// Initializes a new instance of the <see cref="HistogramSampler"/> class.
        /// </summary>
        /// <param name="histogram">The histogram to sample from.</param>
        public HistogramSampler(IEnumerable<int> histogram)
        {
            int leavesCount = histogram.Count();

            if (leavesCount < 1)
            {
                throw new ArgumentException("There given histogram must not be empty.", nameof(histogram));
            }

            this.firstLeafIndex = ComputeNextPowerOf2(leavesCount) - 1;
            this.nodes = new List<int>(this.firstLeafIndex + leavesCount);
            this.nodes.AddRange(Enumerable.Repeat(0, this.firstLeafIndex).ToList());

            foreach (var leaf in histogram)
            {
                if (leaf < 0)
                {
                    throw new ArgumentException(
                        string.Format(
                            "All elements of the histogram must have non-negative values. Histogram element: {0}.", leaf),
                        nameof(histogram));
                }

                this.nodes.Add(leaf);
            }

            this.ComputeInnerNodes();
        }

        /// <summary>
        /// Samples from the histogram.
        /// </summary>
        /// <returns>A sample from the histogram.</returns>
        public int Sample()
        {
            int currentIndex = this.GetRootIndex();

            if (this.GetValueAtIndex(currentIndex) == 0)
            {
                throw new InvalidOperationException(
                    "Cannot sample from the underlying distribution, because the tree that represents it is empty.");
            }

            while (currentIndex < this.firstLeafIndex)
            {
                int currentValue = this.GetValueAtIndex(currentIndex);
                Debug.Assert(
                    currentValue > 0, "All inner nodes which participate in sampling must have positive values.");

                int leftChildIndex = this.GetLeftChildIndex(currentIndex);

                if (Rand.Int(currentValue) < this.GetValueAtIndex(leftChildIndex))
                {
                    // Go left
                    currentIndex = leftChildIndex;
                }
                else
                {
                    // Go right
                    currentIndex = this.GetRightChildIndex(currentIndex);
                }
            }

            return currentIndex - this.firstLeafIndex;
        }

        /// <summary>
        /// Decreases a histogram element by one.
        /// </summary>
        /// <param name="leaf">The element to update.</param>
        public void Take(int leaf)
        {
            if (leaf < 0 || leaf > this.nodes.Count - this.firstLeafIndex)
            {
                var outOfRangeExceptionMessage = string.Format(
                    "The specified leaf must be non-negative and less than or equal to {0}. Given: {1}.",
                    this.nodes.Count - this.firstLeafIndex,
                    leaf);

                throw new ArgumentOutOfRangeException(nameof(leaf), outOfRangeExceptionMessage);
            }

            int currentIndex = this.firstLeafIndex + leaf;

            if (this.GetValueAtIndex(currentIndex) == 0)
            {
                throw new InvalidOperationException(string.Format("The value at leaf {0} is 0 and cannot be decreased.", leaf));
            }

            // Update all parents to the root
            while (currentIndex != this.GetRootIndex())
            {
                --this.nodes[currentIndex];
                currentIndex = this.GetParentIndex(currentIndex);
            }

            // Update the root
            --this.nodes[currentIndex];
        }

        /// <summary>
        /// Indicates whether the histogram is empty or not.
        /// </summary>
        /// <returns>True if the histogram is empty; otherwise, false.</returns>
        public bool IsEmpty()
        {
            return this.GetValueAtIndex(this.GetRootIndex()) == 0;
        }

        #region Helper methods

        /// <summary>
        /// Computes the next power of 2 for a given integer.
        /// For example, the next power of 2 for the number 6 is 8.
        /// </summary>
        /// <param name="n">The number to compute the next power of 2 for.</param>
        /// <returns>The next power of 2.</returns>
        private static int ComputeNextPowerOf2(int n)
        {
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;

            return n + 1;
        }

        /// <summary>
        /// Computes the inner nodes of the tree.
        /// Each inner node holds the sum of the values of its left and right child.
        /// </summary>
        private void ComputeInnerNodes()
        {
            for (int i = this.firstLeafIndex - 1; i >= 0; --i)
            {
                this.nodes[i] += this.GetValueAtIndex(this.GetLeftChildIndex(i))
                                 + this.GetValueAtIndex(this.GetRightChildIndex(i));
            }
        }

        /// <summary>
        /// Gets the index of the root of the tree.
        /// </summary>
        /// <returns>The index of the root of the tree</returns>
        private int GetRootIndex()
        {
            return 0;
        }

        /// <summary>
        /// Gets the node value at a given index.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns>The node value.</returns>
        private int GetValueAtIndex(int index)
        {
            return index >= this.nodes.Count ? 0 : this.nodes[index];
        }

        /// <summary>
        /// Gets the index of the left child.
        /// </summary>
        /// <param name="parentIndex">The index of the parent.</param>
        /// <returns>The index of the left child.</returns>
        private int GetLeftChildIndex(int parentIndex)
        {
            return (2 * parentIndex) + 1;
        }

        /// <summary>
        /// Gets the index of the right child.
        /// </summary>
        /// <param name="parentIndex">The index of the parent.</param>
        /// <returns>The index of the right child.</returns>
        private int GetRightChildIndex(int parentIndex)
        {
            return (2 * parentIndex) + 2;
        }

        /// <summary>
        /// Gets the index of the parent.
        /// </summary>
        /// <param name="childIndex">The index of the child.</param>
        /// <returns>The index of the parent</returns>
        private int GetParentIndex(int childIndex)
        {
            return (childIndex - 1) / 2;
        }

        #endregion
    }
}
