// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Diagnostics;

    /// <summary>
    /// The arguments of the <see cref="RecommenderRun.Completed"/> event.
    /// </summary>
    public class RecommenderRunCompletedEventArgs : EventArgs
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RecommenderRunCompletedEventArgs"/> class.
        /// </summary>
        /// <param name="totalTime">The total time of the run.</param>
        /// <param name="trainingTime">The total training time.</param>
        /// <param name="predictionTime">The total prediction time.</param>
        /// <param name="evaluationTime">The total evaluation time.</param>
        /// <param name="metrics">The collection of metric names and value distributions.</param>
        public RecommenderRunCompletedEventArgs(
            TimeSpan totalTime,
            TimeSpan trainingTime,
            TimeSpan predictionTime,
            TimeSpan evaluationTime,
            MetricValueDistributionCollection metrics)
        {
            Debug.Assert(totalTime >= trainingTime + predictionTime + evaluationTime, "Given time spans are inconsistent.");
            Debug.Assert(metrics != null, "Valid metrics should be provided.");

            this.TotalTime = totalTime;
            this.TrainingTime = trainingTime;
            this.PredictionTime = predictionTime;
            this.EvaluationTime = evaluationTime;
            this.Metrics = metrics;
        }

        /// <summary>
        /// Gets the total time of the run.
        /// </summary>
        public TimeSpan TotalTime { get; private set; }

        /// <summary>
        /// Gets the total training time.
        /// </summary>
        public TimeSpan TrainingTime { get; private set; }

        /// <summary>
        /// Gets the total prediction time.
        /// </summary>
        public TimeSpan PredictionTime { get; private set; }

        /// <summary>
        /// Gets the total evaluation time.
        /// </summary>
        public TimeSpan EvaluationTime { get; private set; }

        /// <summary>
        /// Gets the dictionary which maps metric names to values.
        /// </summary>
        public MetricValueDistributionCollection Metrics { get; private set; }
    }
}