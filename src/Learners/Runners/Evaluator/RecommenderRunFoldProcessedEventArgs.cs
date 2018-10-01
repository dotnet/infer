// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Diagnostics;

    /// <summary>
    /// The arguments of the <see cref="RecommenderRun.FoldProcessed"/> event.
    /// </summary>
    public class RecommenderRunFoldProcessedEventArgs : EventArgs
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RecommenderRunFoldProcessedEventArgs"/> class.
        /// </summary>
        /// <param name="foldNumber">The number of the fold that has been processed.</param>
        /// <param name="totalTime">The total time spent while processing the fold.</param>
        /// <param name="trainingTime">The total training time for the fold.</param>
        /// <param name="predictionTime">The total prediction time for the fold.</param>
        /// <param name="evaluationTime">The total evaluation time for the fold.</param>
        /// <param name="metrics">The collection of metric names and value distributions.</param>
        public RecommenderRunFoldProcessedEventArgs(
            int foldNumber,
            TimeSpan totalTime,
            TimeSpan trainingTime,
            TimeSpan predictionTime,
            TimeSpan evaluationTime,
            MetricValueDistributionCollection metrics)
        {
            Debug.Assert(foldNumber >= 0, "A valid fold number should be provided.");
            Debug.Assert(totalTime >= trainingTime + predictionTime + evaluationTime, "Given time spans are inconsistent.");
            Debug.Assert(metrics != null, "Valid metrics should be provided.");

            this.FoldNumber = foldNumber;
            this.TotalTime = totalTime;
            this.TrainingTime = trainingTime;
            this.PredictionTime = predictionTime;
            this.EvaluationTime = evaluationTime;
            this.Metrics = metrics;
        }

        /// <summary>
        /// Gets the number of the fold that has been processed.
        /// </summary>
        public int FoldNumber { get; private set; }

        /// <summary>
        /// Gets the total time spent while processing the fold.
        /// </summary>
        public TimeSpan TotalTime { get; private set; }

        /// <summary>
        /// Gets the total training time for the fold.
        /// </summary>
        public TimeSpan TrainingTime { get; private set; }

        /// <summary>
        /// Gets the total prediction time for the fold.
        /// </summary>
        public TimeSpan PredictionTime { get; private set; }

        /// <summary>
        /// Gets the total evaluation time for the fold.
        /// </summary>
        public TimeSpan EvaluationTime { get; private set; }

        /// <summary>
        /// Gets the dictionary which maps metric names to values of the metrics computed on the fold.
        /// </summary>
        public MetricValueDistributionCollection Metrics { get; private set; }
    }
}