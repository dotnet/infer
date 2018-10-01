// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Utilities;

    using RatingDistribution = System.Collections.Generic.IDictionary<int, double>;

    /// <summary>
    /// Evaluates a recommender system which predicts star ratings.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TGroundTruthRating">The type of a rating in a test dataset.</typeparam>
    public class StarRatingRecommenderEvaluator<TInstanceSource, TUser, TItem, TGroundTruthRating>
        : RecommenderEvaluator<TInstanceSource, TUser, TItem, TGroundTruthRating, int, RatingDistribution>
    {
        #region Fields, properties, constructor

        /// <summary>
        /// The mapping used for accessing data.
        /// </summary>
        private readonly IStarRatingRecommenderEvaluatorMapping<TInstanceSource, TUser, TItem, TGroundTruthRating> mapping;

        /// <summary>
        /// Initializes a new instance of the <see cref="StarRatingRecommenderEvaluator{TInstanceSource,TUser,TItem,TGroundTruthRating}"/> class.
        /// </summary>
        /// <param name="mapping">The mapping used for accessing data.</param>
        public StarRatingRecommenderEvaluator(
            IStarRatingRecommenderEvaluatorMapping<TInstanceSource, TUser, TItem, TGroundTruthRating> mapping)
            : base(mapping)
        {
            this.mapping = mapping;
        }

        #endregion

        #region Rating probability calibration

        /// <summary>
        /// Computes the probability calibration plot for a particular rating value.
        /// </summary>
        /// <param name="instanceSource">The instance source providing the ground truth.</param>
        /// <param name="predictions">A sparse users-by-items matrix of predicted rating distributions.</param>
        /// <param name="rating">The rating value to generate the calibration plot for.</param>
        /// <param name="bins">The number of bins to use.</param>
        /// <returns>The computed probability calibration plot.</returns>
        public double[] ProbabilityCalibrationPlot(
            TInstanceSource instanceSource,
            IDictionary<TUser, IDictionary<TItem, RatingDistribution>> predictions,
            int rating,
            int bins)
        {
            IStarRatingInfo<TGroundTruthRating> starRatingInfo = this.mapping.GetRatingInfo(instanceSource);

            var countTotal = new int[bins];
            var countGuessed = new int[bins];

            foreach (var userWithPredictionList in predictions)
            {
                foreach (var itemPrediction in userWithPredictionList.Value)
                {
                    TUser user = userWithPredictionList.Key;
                    TItem item = itemPrediction.Key;
                    double prob = itemPrediction.Value[rating];
                    int groundTruth = starRatingInfo.ToStarRating(this.mapping.GetRating(instanceSource, user, item));
                    int probBin = Math.Min((int)(prob * bins), bins - 1);

                    countTotal[probBin] += 1;
                    countGuessed[probBin] += (groundTruth == rating) ? 1 : 0;
                }
            }

            return Util.ArrayInit(bins, i => (double)countGuessed[i] / countTotal[i]);
        }

        /// <summary>
        /// Computes the probability calibration error for a given rating.
        /// </summary>
        /// <param name="instanceSource">The instance source providing the ground truth.</param>
        /// <param name="predictions">A sparse users-by-items matrix of predicted rating distributions.</param>
        /// <param name="rating">The rating value to generate the calibration plot for.</param>
        /// <param name="bins">The number of bins to use.</param>
        /// <returns>The computed probability calibration error.</returns>
        public double ProbabilityCalibrationError(
            TInstanceSource instanceSource,
            IDictionary<TUser, IDictionary<TItem, RatingDistribution>> predictions,
            int rating,
            int bins)
        {
            double[] plot = this.ProbabilityCalibrationPlot(instanceSource, predictions, rating, bins);
            return plot.Select((v, i) => Math.Abs(v - ((i + 0.5) / bins))).Average(); // Mean absolute difference from bin center
        }

        #endregion

        #region Confusion matrices

        /// <summary>
        /// Computes a confusion matrix.
        /// </summary>
        /// <param name="instanceSource">The instance source providing the ground truth.</param>
        /// <param name="predictions">A sparse users-by-items matrix of predicted rating distributions.</param>
        /// <param name="aggregationMethod">A method specifying how metrics are aggregated over the whole dataset.</param>
        /// <returns>The computed confusion matrix.</returns>
        public RatingMatrix ConfusionMatrix(
            TInstanceSource instanceSource,
            IDictionary<TUser, IDictionary<TItem, int>> predictions,
            RecommenderMetricAggregationMethod aggregationMethod = RecommenderMetricAggregationMethod.Default)
        {
            IStarRatingInfo<TGroundTruthRating> starRatingInfo = this.mapping.GetRatingInfo(instanceSource);

            var result = new RatingMatrix(starRatingInfo.MinStarRating, starRatingInfo.MaxStarRating);
            for (int predictedRating = starRatingInfo.MinStarRating; predictedRating <= starRatingInfo.MaxStarRating; ++predictedRating)
            {
                for (int trueRating = starRatingInfo.MinStarRating; trueRating <= starRatingInfo.MaxStarRating; ++trueRating)
                {
                    result[predictedRating, trueRating] = this.ModelDomainRatingPredictionMetric(
                        instanceSource,
                        predictions,
                        (t, p) => p == predictedRating && t == trueRating ? 1.0 : 0.0,
                        aggregationMethod);
                }
            }

            return result;
        }

        /// <summary>
        /// Computes the component-wise product of a confusion matrix and a loss matrix.
        /// </summary>
        /// <param name="instanceSource">The instance source providing the ground truth (used to compute the confusion matrix).</param>
        /// <param name="predictions">
        /// A sparse users-by-items matrix of predicted rating distributions (used to compute the confusion matrix).
        /// </param>
        /// <param name="lossMatrix">The loss matrix.</param>
        /// <param name="aggregationMethod">
        /// A method specifying how metrics are aggregated over all instances (used to compute the confusion matrix).
        /// </param>
        /// <returns>The computed weighted confusion.</returns>
        public double WeightedConfusion(
            TInstanceSource instanceSource,
            IDictionary<TUser, IDictionary<TItem, int>> predictions,
            RatingMatrix lossMatrix,
            RecommenderMetricAggregationMethod aggregationMethod = RecommenderMetricAggregationMethod.Default)
        {
            RatingMatrix confusionMatrix = this.ConfusionMatrix(instanceSource, predictions, aggregationMethod);
            return RatingMatrix.ComponentwiseProduct(confusionMatrix, lossMatrix);
        }

        /// <summary>
        /// Computes the expected confusion matrix.
        /// </summary>
        /// <param name="instanceSource">The instance source providing the ground truth.</param>
        /// <param name="predictions">A sparse users-by-items matrix of predicted rating distributions.</param>
        /// <param name="aggregationMethod">A method specifying how metrics are aggregated over all instances.</param>
        /// <returns>The computed expected confusion matrix.</returns>
        public RatingMatrix ExpectedConfusionMatrix(
            TInstanceSource instanceSource,
            IDictionary<TUser, IDictionary<TItem, RatingDistribution>> predictions,
            RecommenderMetricAggregationMethod aggregationMethod = RecommenderMetricAggregationMethod.Default)
        {
            IStarRatingInfo<TGroundTruthRating> starRatingInfo = this.mapping.GetRatingInfo(instanceSource);

            var result = new RatingMatrix(starRatingInfo.MinStarRating, starRatingInfo.MaxStarRating);
            for (int predictedRating = starRatingInfo.MinStarRating; predictedRating <= starRatingInfo.MaxStarRating; ++predictedRating)
            {
                for (int trueRating = starRatingInfo.MinStarRating; trueRating <= starRatingInfo.MaxStarRating; ++trueRating)
                {
                    result[predictedRating, trueRating] = this.ModelDomainRatingPredictionMetricExpectation(
                        instanceSource,
                        predictions,
                        (p, t) => p == predictedRating && t == trueRating ? 1.0 : 0.0,
                        aggregationMethod);
                }
            }

            return result;
        }

        /// <summary>
        /// Computes the expected component-wise product of a confusion matrix and a loss matrix.
        /// </summary>
        /// <param name="instanceSource">The instance source providing the ground truth (used to compute the expected confusion matrix).</param>
        /// <param name="predictions">
        /// A sparse users-by-items matrix of predicted rating distributions (used to compute the expected confusion matrix).
        /// </param>
        /// <param name="lossMatrix">The loss matrix.</param>
        /// <param name="aggregationMethod">
        /// A method specifying how metrics are aggregated over all instances (used to compute the expected confusion matrix).</param>
        /// <returns>The computed expected weighted confusion.</returns>
        public double ExpectedWeightedConfusion(
            TInstanceSource instanceSource,
            IDictionary<TUser, IDictionary<TItem, RatingDistribution>> predictions,
            RatingMatrix lossMatrix,
            RecommenderMetricAggregationMethod aggregationMethod = RecommenderMetricAggregationMethod.Default)
        {
            RatingMatrix confusionMatrix = this.ExpectedConfusionMatrix(instanceSource, predictions, aggregationMethod);
            return RatingMatrix.ComponentwiseProduct(confusionMatrix, lossMatrix);
        }

        #endregion

        #region Rating prediction metrics (model domain)

        /// <summary>
        /// Computes the average of a given rating prediction metric using ground truth in model domain by iterating over 
        /// <paramref name="predictions"/> and using the aggregation method given in <paramref name="aggregationMethod"/>.
        /// </summary>
        /// <param name="instanceSource">The instance source providing the ground truth.</param>
        /// <param name="predictions">A sparse users-by-items matrix of predicted rating distributions.</param>
        /// <param name="metric">The rating prediction metric using ground truth in model domain.</param>
        /// <param name="aggregationMethod">A method specifying how metrics are aggregated over all instances.</param>
        /// <returns>The computed average of the given rating prediction metric.</returns>
        public double ModelDomainRatingPredictionMetric(
            TInstanceSource instanceSource,
            IDictionary<TUser, IDictionary<TItem, int>> predictions,
            Func<int, int, double> metric,
            RecommenderMetricAggregationMethod aggregationMethod = RecommenderMetricAggregationMethod.Default)
        {
            IStarRatingInfo<TGroundTruthRating> starRatingInfo = this.mapping.GetRatingInfo(instanceSource);
            Func<TGroundTruthRating, int, double> metricWrapper = (g, p) => metric(starRatingInfo.ToStarRating(g), p);
            return this.RatingPredictionMetric(instanceSource, predictions, metricWrapper, aggregationMethod);
        }

        /// <summary>
        /// Computes the average of a given rating prediction metric using ground truth in model domain by iterating over 
        /// <paramref name="predictions"/> and using the aggregation method given in <paramref name="aggregationMethod"/>.
        /// </summary>
        /// <param name="instanceSource">The instance source providing the ground truth.</param>
        /// <param name="predictions">A sparse users-by-items matrix of predicted rating distributions.</param>
        /// <param name="metric">The rating prediction metric using ground truth in model domain.</param>
        /// <param name="aggregationMethod">A method specifying how metrics are aggregated over all instances.</param>
        /// <returns>The computed average of the given rating prediction metric.</returns>
        public double ModelDomainRatingPredictionMetric(
            TInstanceSource instanceSource,
            IDictionary<TUser, IDictionary<TItem, RatingDistribution>> predictions,
            Func<int, RatingDistribution, double> metric,
            RecommenderMetricAggregationMethod aggregationMethod = RecommenderMetricAggregationMethod.Default)
        {
            IStarRatingInfo<TGroundTruthRating> starRatingInfo = this.mapping.GetRatingInfo(instanceSource);
            Func<TGroundTruthRating, RatingDistribution, double> metricWrapper = (g, up) => metric(starRatingInfo.ToStarRating(g), up);
            return this.RatingPredictionMetric(instanceSource, predictions, metricWrapper, aggregationMethod);
        }

        /// <summary>
        /// Computes the average of a given rating prediction metric using ground truth in model domain by iterating over 
        /// <paramref name="predictions"/> and using the aggregation method given in <paramref name="aggregationMethod"/>.
        /// </summary>
        /// <param name="instanceSource">The instance source providing the ground truth.</param>
        /// <param name="predictions">A sparse users-by-items matrix of predicted rating distributions.</param>
        /// <param name="metric">The rating prediction metric using ground truth in model domain.</param>
        /// <param name="aggregationMethod">A method specifying how metrics are aggregated over all instances.</param>
        /// <returns>The computed average of the given rating prediction metric.</returns>
        public double ModelDomainRatingPredictionMetricExpectation(
            TInstanceSource instanceSource,
            IDictionary<TUser, IDictionary<TItem, RatingDistribution>> predictions,
            Func<int, int, double> metric,
            RecommenderMetricAggregationMethod aggregationMethod = RecommenderMetricAggregationMethod.Default)
        {
            return this.ModelDomainRatingPredictionMetric(
                instanceSource,
                predictions,
                (r, ur) => RatingMetricExpectation(r, ur, metric),
                aggregationMethod);
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Computes the expected rating metric.
        /// </summary>
        /// <param name="trueRating">The true rating.</param>
        /// <param name="predictedRatingDistribution">The predicted rating distribution.</param>
        /// <param name="metric">The method used to compute the metric.</param>
        /// <returns>The rating metric expectation.</returns>
        private static double RatingMetricExpectation(
             int trueRating, RatingDistribution predictedRatingDistribution, Func<int, int, double> metric)
        {
            return predictedRatingDistribution.Sum(prediction => metric(trueRating, prediction.Key) * prediction.Value);
        }

        #endregion
    }
}
