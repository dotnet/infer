// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    using Evaluator = StarRatingRecommenderEvaluator<Learners.Mappings.SplitInstanceSource<RecommenderDataset>, User, Item, int>;
    using EvaluatorMapping = Microsoft.ML.Probabilistic.Learners.Mappings.StarRatingRecommenderEvaluatorMapping<Learners.Mappings.SplitInstanceSource<RecommenderDataset>, RatedUserItem, User, Item, int, DummyFeatureSource, Math.Vector>;
    using Recommender = IRecommender<Learners.Mappings.SplitInstanceSource<RecommenderDataset>, User, Item, int, System.Collections.Generic.IDictionary<int, double>, DummyFeatureSource>;
    using SplittingMapping = Microsoft.ML.Probabilistic.Learners.Mappings.TrainTestSplittingStarRatingRecommenderMapping<RecommenderDataset, RatedUserItem, User, Item, int, DummyFeatureSource, Math.Vector>;

    /// <summary>
    /// Represents a run of a particular test on a given data using a specified recommendation engine.
    /// </summary>
    public class RecommenderRun
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RecommenderRun"/> class.
        /// </summary>
        /// <param name="name">The name of the run.</param>
        /// <param name="dataset">The dataset to run the test on.</param>
        /// <param name="foldCount">The number of folds to split the dataset into.</param>
        /// <param name="splittingMappingFactory">The factory to create instances of the splitting mapping.</param>
        /// <param name="recommenderFactory">The factory to create instances of the recommender with the specified mapping.</param>
        /// <param name="tests">The test for the recommender.</param>
        public RecommenderRun(
            string name,
            RecommenderDataset dataset,
            int foldCount,
            Func<SplittingMapping> splittingMappingFactory,
            Func<SplittingMapping, Recommender> recommenderFactory,
            IEnumerable<RecommenderTest> tests)
        {
            Debug.Assert(!string.IsNullOrEmpty(name), "Test run name can not be null or empty.");
            Debug.Assert(dataset != null, "A valid dataset should be provided.");
            Debug.Assert(foldCount > 0, "A valid number of folds should be provided.");
            Debug.Assert(splittingMappingFactory != null, "A valid splitting mapping factory should be provided.");
            Debug.Assert(recommenderFactory != null, "A valid recommender factory should be provided.");
            Debug.Assert(tests != null, "A valid collection of recommender tests should be provided.");

            this.Name = name;
            this.RecommenderDataset = dataset;
            this.FoldCount = foldCount;
            this.SplittingMappingFactory = splittingMappingFactory;
            this.RecommenderFactory = recommenderFactory;
            this.Tests = tests.ToList();
        }
        
        /// <summary>
        /// Occurs when the run is started.
        /// </summary>
        public event EventHandler Started;

        /// <summary>
        /// Occurs when the run is complete.
        /// </summary>
        public event EventHandler<RecommenderRunCompletedEventArgs> Completed;

        /// <summary>
        /// Occurs when a fold is processed.
        /// </summary>
        public event EventHandler<RecommenderRunFoldProcessedEventArgs> FoldProcessed;

        /// <summary>
        /// Occurs when the run is interrupted.
        /// </summary>
        public event EventHandler<RecommenderRunInterruptedEventArgs> Interrupted;

        /// <summary>
        /// Gets the name of the run.
        /// </summary>
        public string Name { get; private set; }

        /// <summary>
        /// Gets the dataset to run the test on.
        /// </summary>
        public RecommenderDataset RecommenderDataset { get; private set; }

        /// <summary>
        /// Gets the number of folds to split the dataset into.
        /// </summary>
        public int FoldCount { get; private set; }

        /// <summary>
        /// Gets the factory to create instances of the splitting mapping.
        /// </summary>
        public Func<SplittingMapping> SplittingMappingFactory { get; private set; }

        /// <summary>
        /// Gets the factory to create instances of the recommender with the specified mapping.
        /// </summary>
        public Func<SplittingMapping, Recommender> RecommenderFactory { get; private set; }

        /// <summary>
        /// Gets the test for the recommender.
        /// </summary>
        public IEnumerable<RecommenderTest> Tests { get; private set; }

        /// <summary>
        /// Executes the test for a given recommender under a specified name.
        /// </summary>
        public void Execute()
        {
            // Report that the run has been started
            this.Started?.Invoke(this, EventArgs.Empty);

            try
            {
                Rand.Restart(1984); // Run should produce the same results every time

                TimeSpan totalTrainingTime = TimeSpan.Zero;
                TimeSpan totalPredictionTime = TimeSpan.Zero;
                TimeSpan totalEvaluationTime = TimeSpan.Zero;
                Stopwatch totalTimer = Stopwatch.StartNew();
                MetricValueDistributionCollection metrics = null;

                for (int i = 0; i < this.FoldCount; ++i)
                {
                    // Start timer measuring total time spent on this fold
                    Stopwatch totalFoldTimer = Stopwatch.StartNew();

                    SplittingMapping splittingMapping = this.SplittingMappingFactory();
                    Recommender recommender = this.RecommenderFactory(splittingMapping);
                    Evaluator evaluator = new Evaluator(new EvaluatorMapping(splittingMapping));

                    // Train the recommender
                    Stopwatch foldTrainingTimer = Stopwatch.StartNew();
                    recommender.Train(SplitInstanceSource.Training(this.RecommenderDataset));
                    TimeSpan foldTrainingTime = foldTrainingTimer.Elapsed;
                    
                    // Run each test on the trained recommender
                    var foldMetrics = new MetricValueDistributionCollection();
                    TimeSpan foldPredictionTime = TimeSpan.Zero;
                    TimeSpan foldEvaluationTime = TimeSpan.Zero;
                    foreach (RecommenderTest test in this.Tests)
                    {
                        // Perform the test
                        TimeSpan testPredictionTime, testEvaluationTime;
                        MetricValueDistributionCollection testMetrics;
                        test.Execute(
                            recommender,
                            evaluator,
                            SplitInstanceSource.Test(this.RecommenderDataset),
                            out testPredictionTime,
                            out testEvaluationTime,
                            out testMetrics);

                        // Merge the timings and the metrics
                        foldPredictionTime += testPredictionTime;
                        foldEvaluationTime += testEvaluationTime;
                        foldMetrics.SetToUnionWith(testMetrics);
                    }

                    // Stop timer measuring total time spent on this fold
                    TimeSpan totalFoldTime = totalFoldTimer.Elapsed;

                    // Report that the fold has been processed
                    this.FoldProcessed?.Invoke(this,
                        new RecommenderRunFoldProcessedEventArgs(i, totalFoldTime, foldTrainingTime, foldPredictionTime, foldEvaluationTime, foldMetrics));

                    // Merge the timings
                    totalTrainingTime += foldTrainingTime;
                    totalPredictionTime += foldPredictionTime;
                    totalEvaluationTime += foldEvaluationTime;

                    // Merge the metrics
                    if (metrics == null)
                    {
                        metrics = foldMetrics;
                    }
                    else
                    {
                        metrics.MergeWith(foldMetrics);
                    }
                }

                // Report that the run has been completed
                TimeSpan totalTime = totalTimer.Elapsed;
                this.Completed?.Invoke(this,
                    new RecommenderRunCompletedEventArgs(totalTime, totalTrainingTime, totalPredictionTime, totalEvaluationTime, metrics));
            }
            catch (Exception e)
            {
                this.Interrupted?.Invoke(this, new RecommenderRunInterruptedEventArgs(e));
            }
        }
    }
}
