// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    /// <summary>
    /// Specifies how metrics are aggregated over the whole dataset.
    /// </summary>
    public enum RecommenderMetricAggregationMethod
    {
        /// <summary>
        /// Metric values for every user-item pair are summed up and then divided by the number of pairs.
        /// </summary>
        Default,

        /// <summary>
        /// Metric values for each user are averaged separately first, then summed up and divided by the number of users.
        /// </summary>
        PerUserFirst
    }
}