// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Diagnostics;

    /// <summary>
    /// Represents the distribution of a metric value.
    /// </summary>
    public class MetricValueDistribution
    {
        /// <summary>
        /// The sum of the metric values.
        /// </summary>
        private double sum;

        /// <summary>
        /// The sum of squares of the metric values.
        /// </summary>
        private double sumSqr;

        /// <summary>
        /// The number of the metric values.
        /// </summary>
        private int count;

        /// <summary>
        /// Initializes a new instance of the <see cref="MetricValueDistribution"/> class with a specified metric value.
        /// </summary>
        /// <param name="value">The metric value.</param>
        public MetricValueDistribution(double value)
        {
            this.sum = value;
            this.sumSqr = value * value;
            this.count = 1;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MetricValueDistribution"/> class.
        /// </summary>
        /// <param name="sum">The sum of metric values.</param>
        /// <param name="sumSqr">The sum of squares of the metric values.</param>
        /// <param name="count">The number of the metric values.</param>
        private MetricValueDistribution(double sum, double sumSqr, int count)
        {
            this.sum = sum;
            this.sumSqr = sumSqr;
            this.count = count;
        }

        /// <summary>
        /// Gets the mean value of the metric.
        /// </summary>
        public double Mean
        {
            get
            {
                Debug.Assert(this.count > 0, "At least one metric value should be provided before computing mean.");
                return this.sum / this.count;
            }
        }

        /// <summary>
        /// Gets the standard deviation of the metric.
        /// </summary>
        public double StdDev
        {
            get
            {
                double mean = this.Mean;
                double variance = (this.sumSqr / this.count) - (mean * mean);
                return Math.Sqrt(variance);
            }
        }

        /// <summary>
        /// Merges this instance with another metric distribution.
        /// </summary>
        /// <param name="other">The other metric distribution.</param>
        public void MergeWith(MetricValueDistribution other)
        {
            this.sum += other.sum;
            this.sumSqr += other.sumSqr;
            this.count += other.count;
        }

        /// <summary>
        /// Creates a copy of this object.
        /// </summary>
        /// <returns>The created copy.</returns>
        public MetricValueDistribution Clone()
        {
            return new MetricValueDistribution(this.sum, this.sumSqr, this.count);
        }

        /// <summary>
        /// Returns a string that represents the current object.
        /// </summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString()
        {
            return string.Format("{0:0.000} ± {1:0.000}", this.Mean, this.StdDev);
        }
    }
}
