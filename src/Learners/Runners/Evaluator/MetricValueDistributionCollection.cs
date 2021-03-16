// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System.Collections;
    using System.Collections.Generic;
    using System.Diagnostics;

    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represents a collection of named metric value distributions.
    /// </summary>
    public class MetricValueDistributionCollection : IEnumerable<KeyValuePair<string, MetricValueDistribution>>
    {
        /// <summary>
        /// The mapping from metric names to metric value distributions.
        /// </summary>
        private readonly Dictionary<string, MetricValueDistribution> metricNameToValueDistribution =
            new Dictionary<string, MetricValueDistribution>();

        /// <summary>
        /// Adds a new metric to the collection.
        /// </summary>
        /// <param name="metricName">The name of the metric.</param>
        /// <param name="metricValueDistribution">The distribution of metric values.</param>
        public void Add(string metricName, MetricValueDistribution metricValueDistribution)
        {
            Debug.Assert(!string.IsNullOrEmpty(metricName), "Metric name can not be null or empty.");
            Debug.Assert(metricValueDistribution != null, "A valid metric value distribution should be provided.");
            Debug.Assert(!this.metricNameToValueDistribution.ContainsKey(metricName), "Value distribution for this metric has been already added.");
            
            this.metricNameToValueDistribution.Add(metricName, metricValueDistribution.Clone());
        }

        /// <summary>
        /// Merges metric value distributions in this collection with the distributions in a given collection.
        /// Both collections must have the same set of metrics.
        /// </summary>
        /// <param name="other">The collection to merge with.</param>
        public void MergeWith(MetricValueDistributionCollection other)
        {
            Debug.Assert(other != null, "A valid collection should be provided.");
            Debug.Assert(
                this.metricNameToValueDistribution.Count == other.metricNameToValueDistribution.Count,
                "Both collections should have the same set of metrics.");
            
            foreach (KeyValuePair<string, MetricValueDistribution> metricValueDistribution in other.metricNameToValueDistribution)
            {
                Debug.Assert(
                    this.metricNameToValueDistribution.ContainsKey(metricValueDistribution.Key),
                    "Both collections should have the same set of metrics.");
                this.metricNameToValueDistribution[metricValueDistribution.Key].MergeWith(metricValueDistribution.Value);
            }
        }

        /// <summary>
        /// Adds metric value distributions from the given collection to this collection.
        /// Collections must have disjoint sets of metrics.
        /// </summary>
        /// <param name="other">The collection to compute union with.</param>
        public void SetToUnionWith(MetricValueDistributionCollection other)
        {
            Debug.Assert(other != null, "A valid collection should be provided.");

            foreach (KeyValuePair<string, MetricValueDistribution> metricValueDistribution in other.metricNameToValueDistribution)
            {
                this.Add(metricValueDistribution.Key, metricValueDistribution.Value);
            }
        }

        /// <summary>
        /// Returns an enumerator that iterates through the collection.
        /// </summary>
        /// <returns>An enumerator.</returns>
        public IEnumerator<KeyValuePair<string, MetricValueDistribution>> GetEnumerator()
        {
            return this.metricNameToValueDistribution.GetEnumerator();
        }

        /// <summary>
        /// Returns an enumerator that iterates through the collection.
        /// </summary>
        /// <returns>An enumerator.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        /// <summary>
        /// Returns a string that represents the current object.
        /// </summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString()
        {
            return StringUtil.DictionaryToString<string,MetricValueDistribution>(this.metricNameToValueDistribution, " | ");
        }
    }
}
