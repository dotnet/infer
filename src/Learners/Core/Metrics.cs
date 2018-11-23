// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// A diverse set of metrics to evaluate various kinds of predictors.
    /// </summary>
    public static class Metrics
    {
        /// <summary>
        /// Linear position discount function.
        /// </summary>
        public static readonly Func<int, double> LinearDiscountFunc = i => 1.0 / (i + 1);

        /// <summary>
        /// Logarithmic position discount function.
        /// </summary>
        public static readonly Func<int, double> LogarithmicDiscountFunc = i => 1.0 / Math.Log(i + 2, 2);

        /// <summary>
        /// The tolerance for comparisons of real numbers.
        /// </summary>
        private const double Tolerance = 1e-9;

        #region Pointwise metrics

        /// <summary>
        /// Returns 0 if a prediction and the ground truth are the same and 1 otherwise.
        /// </summary>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="prediction">The prediction.</param>
        /// <returns>The computed metric value.</returns>
        public static double ZeroOneError<TLabel>(TLabel groundTruth, TLabel prediction)
        {
            if (groundTruth == null)
            {
                throw new ArgumentNullException(nameof(groundTruth));
            }

            if (prediction == null)
            {
                throw new ArgumentNullException(nameof(prediction));
            }

            return prediction.Equals(groundTruth) ? 0.0 : 1.0;
        }

        /// <summary>
        /// Returns 0 if a prediction and the ground truth are the same and 1 otherwise.
        /// </summary>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="prediction">The prediction.</param>
        /// <returns>The computed metric value.</returns>
        public static double ZeroOneError(int groundTruth, int prediction)
        {
            return prediction == groundTruth ? 0.0 : 1.0; 
        }

        /// <summary>
        /// Computes squared difference between a prediction and the ground truth.
        /// </summary>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="prediction">The prediction.</param>
        /// <returns>The computed metric value.</returns>
        public static double SquaredError(bool groundTruth, bool prediction)
        {
            return ZeroOneError(groundTruth, prediction);
        }

        /// <summary>
        /// Computes squared difference between a prediction and the ground truth.
        /// </summary>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="prediction">The prediction.</param>
        /// <returns>The computed metric value.</returns>
        public static double SquaredError(int groundTruth, int prediction)
        {
            return SquaredError((double)groundTruth, (double)prediction);
        }

        /// <summary>
        /// Computes squared difference between a prediction and the ground truth.
        /// </summary>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="prediction">The prediction.</param>
        /// <returns>The computed metric value.</returns>
        public static double SquaredError(double groundTruth, int prediction)
        {
            return SquaredError(groundTruth, (double)prediction);
        }

        /// <summary>
        /// Computes the squared difference between a prediction and the ground truth.
        /// </summary>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="prediction">The prediction.</param>
        /// <returns>The computed metric value.</returns>
        public static double SquaredError(double groundTruth, double prediction)
        {
            double difference = groundTruth - prediction;
            return difference * difference;
        }

        /// <summary>
        /// Computes the absolute difference between a prediction and the ground truth.
        /// </summary>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="prediction">The prediction.</param>
        /// <returns>The computed metric value.</returns>
        public static double AbsoluteError(bool groundTruth, bool prediction)
        {
            return ZeroOneError(groundTruth, prediction);
        }

        /// <summary>
        /// Computes the absolute difference between a prediction and the ground truth.
        /// </summary>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="prediction">The prediction.</param>
        /// <returns>The computed metric value.</returns>
        public static double AbsoluteError(int groundTruth, int prediction)
        {
            return AbsoluteError((double)groundTruth, (double)prediction);
        }

        /// <summary>
        /// Computes the absolute difference between a prediction and the ground truth.
        /// </summary>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="prediction">The prediction.</param>
        /// <returns>The computed metric value.</returns>
        public static double AbsoluteError(double groundTruth, int prediction)
        {
            return AbsoluteError(groundTruth, (double)prediction);
        }

        /// <summary>
        /// Computes the absolute difference between a prediction and the ground truth.
        /// </summary>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="prediction">The prediction.</param>
        /// <returns>The computed metric value.</returns>
        public static double AbsoluteError(double groundTruth, double prediction)
        {
            return Math.Abs(groundTruth - prediction); 
        }

        /// <summary>
        /// Returns the negative natural logarithm of the probability of ground truth for a given predictive distribution.
        /// </summary>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="prediction">The prediction as a <see cref="Discrete"/> distribution.</param>
        /// <returns>The negative natural logarithm of the probability of <paramref name="groundTruth"/>.</returns>
        public static double NegativeLogProbability(bool groundTruth, Bernoulli prediction)
        {
            return -prediction.GetLogProb(groundTruth);
        }

        /// <summary>
        /// Returns the negative natural logarithm of the probability of ground truth for a given predictive distribution.
        /// </summary>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="prediction">The prediction as a <see cref="Discrete"/> distribution.</param>
        /// <returns>The negative natural logarithm of the probability of <paramref name="groundTruth"/>.</returns>
        public static double NegativeLogProbability(int groundTruth, Discrete prediction)
        {
            if (prediction == null)
            {
                throw new ArgumentNullException(nameof(prediction));
            }
            
            return -prediction.GetLogProb(groundTruth);
        }

        /// <summary>
        /// Returns the negative natural logarithm of the probability of ground truth for a given predictive distribution.
        /// </summary>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="prediction">The prediction as a discrete distribution over labels.</param>
        /// <returns>The negative natural logarithm of the probability of <paramref name="groundTruth"/>.</returns>
        public static double NegativeLogProbability<TLabel>(TLabel groundTruth, IDictionary<TLabel, double> prediction)
        {
            if (groundTruth == null)
            {
                throw new ArgumentNullException(nameof(groundTruth));
            }

            if (prediction == null)
            {
                throw new ArgumentNullException(nameof(prediction));
            }

            if (prediction.Count < 2)
            {
                throw new ArgumentException("The predicted distribution over labels must contain at least two entries.", nameof(prediction));
            }

            if (Math.Abs(prediction.Values.Sum() - 1) > Tolerance)
            {
                throw new ArgumentException("The predicted distribution over labels must sum to 1.", nameof(prediction));
            }

            if (prediction.Values.Any(p => p < 0.0 || p > 1.0))
            {
                throw new ArgumentException("The label probability must be between 0 and 1.", nameof(prediction));
            }

            double probability;
            if (!prediction.TryGetValue(groundTruth, out probability))
            {
                throw new ArgumentException("The predicted distribution over labels does not contain a probability for the ground truth label '" + groundTruth + "'.");
            }

            return -Math.Log(probability);
        }

        #endregion

        #region Listwise metrics

        /// <summary>
        /// Computes discounted cumulative gain for the given list of gains.
        /// </summary>
        /// <param name="orderedGains">List of gains ordered according to some external criteria.</param>
        /// <param name="discountFunc">Position discount function.</param>
        /// <returns>The computed metric value.</returns>
        /// <exception cref="ArgumentNullException">Thrown if one of the arguments is null.</exception>
        public static double Dcg(IEnumerable<double> orderedGains, Func<int, double> discountFunc)
        {
            if (orderedGains == null)
            {
                throw new ArgumentNullException(nameof(orderedGains));
            }

            if (discountFunc == null)
            {
                throw new ArgumentNullException(nameof(discountFunc));
            }

            int index = 0;
            return orderedGains.Sum(gain => gain * discountFunc(index++));
        }

        /// <summary>
        /// Computes discounted cumulative gain for the given list of gains
        /// using <see cref="LogarithmicDiscountFunc"/> discount function.
        /// </summary>
        /// <param name="orderedGains">List of gains ordered according to some external criteria.</param>
        /// <returns>The computed metric value.</returns>
        /// <exception cref="ArgumentNullException">Thrown if one of the arguments is null.</exception>
        public static double Dcg(IEnumerable<double> orderedGains)
        {
            return Dcg(orderedGains, LogarithmicDiscountFunc);
        }

        /// <summary>
        /// Computes discounted cumulative gain for the given list of gains
        /// using <see cref="LinearDiscountFunc"/> discount function.
        /// </summary>
        /// <param name="orderedGains">List of gains ordered according to some external criteria.</param>
        /// <returns>The computed metric value.</returns>
        /// <exception cref="ArgumentNullException">Thrown if one of the arguments is null.</exception>
        public static double LinearDcg(IEnumerable<double> orderedGains)
        {
            return Dcg(orderedGains, LinearDiscountFunc);
        }

        /// <summary>
        /// Computes discounted cumulative gain for the given list of gains normalized given another list of gains.
        /// </summary>
        /// <param name="orderedGains">List of gains ordered according to some external criteria.</param>
        /// <param name="bestOrderedGains">List of gains used to compute a normalizer for discounted cumulative gain.</param>
        /// <param name="discountFunc">Position discount function.</param>
        /// <returns>The computed metric value.</returns>
        /// <exception cref="ArgumentNullException">Thrown if one of the arguments is null.</exception>
        /// <exception cref="ArgumentException">
        /// Thrown if <paramref name="orderedGains"/> and <paramref name="bestOrderedGains"/> are of different size.
        /// </exception>
        public static double Ndcg(IEnumerable<double> orderedGains, IEnumerable<double> bestOrderedGains, Func<int, double> discountFunc)
        {
            if (orderedGains == null)
            {
                throw new ArgumentNullException(nameof(orderedGains));
            }

            if (bestOrderedGains == null)
            {
                throw new ArgumentNullException(nameof(bestOrderedGains));
            }

            if (discountFunc == null)
            {
                throw new ArgumentNullException(nameof(discountFunc));
            }
            
            List<double> orderedGainList = orderedGains.ToList();
            List<double> bestOrderedGainList = bestOrderedGains.ToList();

            if (bestOrderedGainList.Count == 0)
            {
                throw new ArgumentException("NDCG is not defined for empty gain lists.");
            }

            if (orderedGainList.Count != bestOrderedGainList.Count)
            {
                throw new ArgumentException("The gain lists must be of the same size in order to compute NDCG.");
            }

            double result = Dcg(orderedGainList, discountFunc) / Dcg(bestOrderedGainList, discountFunc);
            if (double.IsNaN(result) || double.IsInfinity(result))
            {
                throw new ArgumentException("NDCG is not defined for the given pair of gain lists.");
            }

            if (result < 0 || result > 1)
            {
                throw new ArgumentException("NDCG is out of the [0, 1] range for the given pair of gain lists.");
            }
            
            return result;
        }

        /// <summary>
        /// Computes discounted cumulative gain for the given list of gains normalized given another list of gains using <see cref="LogarithmicDiscountFunc"/>.
        /// </summary>
        /// <param name="orderedGains">List of gains ordered according to some external criteria.</param>
        /// <param name="bestOrderedGains">List of gains used to compute a normalizer for discounted cumulative gain.</param>
        /// <returns>The computed metric value.</returns>
        /// <exception cref="ArgumentNullException">Thrown if one of the arguments is null.</exception>
        /// <exception cref="ArgumentException">
        /// Thrown if <paramref name="orderedGains"/> and <paramref name="bestOrderedGains"/> are of different size.
        /// </exception>
        public static double Ndcg(IEnumerable<double> orderedGains, IEnumerable<double> bestOrderedGains)
        {
            return Ndcg(orderedGains, bestOrderedGains, LogarithmicDiscountFunc);
        }

        /// <summary>
        /// Computes discounted cumulative gain for the given list of gains normalized given another list of gains using <see cref="LinearDiscountFunc"/>.
        /// </summary>
        /// <param name="orderedGains">List of gains ordered according to some external criteria.</param>
        /// <param name="bestOrderedGains">List of gains used to compute a normalizer for discounted cumulative gain.</param>
        /// <returns>The computed metric value.</returns>
        /// <exception cref="ArgumentNullException">Thrown if one of the arguments is null.</exception>
        /// <exception cref="ArgumentException">
        /// Thrown if <paramref name="orderedGains"/> and <paramref name="bestOrderedGains"/> are of different size.
        /// </exception>
        public static double LinearNdcg(IEnumerable<double> orderedGains, IEnumerable<double> bestOrderedGains)
        {
            return Ndcg(orderedGains, bestOrderedGains, LinearDiscountFunc);
        }

        /// <summary>
        /// Computes graded average precision for the given list of relevance values.
        /// </summary>
        /// <param name="orderedRelevances">List of relevance values ordered according to some external criteria.</param>
        /// <returns>The computed metric value.</returns>
        /// <exception cref="ArgumentNullException">Thrown if one of the arguments is null.</exception>
        public static double GradedAveragePrecision(IEnumerable<double> orderedRelevances)
        {
            if (orderedRelevances == null)
            {
                throw new ArgumentNullException(nameof(orderedRelevances));
            }

            int outerIndex = 0;
            double outerSum = 0;
            double relevanceSum = 0;
            List<double> orderedRelevanceList = orderedRelevances.ToList();
            foreach (double relevanceOuter in orderedRelevanceList)
            {
                relevanceSum += relevanceOuter;

                int innerIndex = 0;
                double innerSum = 0;
                foreach (double relevanceInner in orderedRelevanceList)
                {
                    if (innerIndex > outerIndex)
                    {
                        break;
                    }

                    innerSum += Math.Min(relevanceOuter, relevanceInner);
                    ++innerIndex;
                }

                outerSum += innerSum / (outerIndex + 1);
                ++outerIndex;
            }

            return outerSum / relevanceSum;
        }

        /// <summary>
        /// Computes the precision-recall curve.
        /// </summary>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <param name="positiveInstances">The instances with 'positive' ground truth labels.</param>
        /// <param name="instanceScores">
        /// The predicted instance scores. The larger a predicted score, the more likely the instance is 
        /// to belong to the 'positive' class.
        /// </param>
        /// <returns>The points on the precision-recall curve, increasing by recall.</returns>
        /// <remarks>
        /// All instances not contained in <paramref name="positiveInstances"/> are assumed to belong to the 'negative' class.
        /// </remarks>
        public static IEnumerable<PrecisionRecall> PrecisionRecallCurve<TInstance>(
            IEnumerable<TInstance> positiveInstances, IEnumerable<KeyValuePair<TInstance, double>> instanceScores)
        {
            if (positiveInstances == null)
            {
                throw new ArgumentNullException(nameof(positiveInstances));
            }

            if (instanceScores == null)
            {
                throw new ArgumentNullException(nameof(instanceScores));
            }

            // Compute the number of instances with positive ground truth labels
            var positiveInstanceSet = new HashSet<TInstance>(positiveInstances);
            long positivesCount = positiveInstanceSet.Count();

            if (positivesCount == 0)
            {
                throw new ArgumentException("There must be at least one instance with a 'positive' ground truth label.");
            }

            // Sort instances by their scores
            var sortedInstanceScores = from pair in instanceScores orderby pair.Value descending select pair;

            // Add (0,1) to PR curve
            long falsePositivesCount = 0;
            long truePositivesCount = 0;

            double recall = 0.0;
            double precision = 1.0;

            var precisionRecallCurve = new List<PrecisionRecall> { new PrecisionRecall(precision, recall) };

            // Add further points to PR curve
            foreach (var instance in sortedInstanceScores)
            {
                if (positiveInstanceSet.Contains(instance.Key))
                {
                    truePositivesCount++;
                }
                else
                {
                    falsePositivesCount++;
                }

                recall = truePositivesCount / (double)positivesCount;
                precision = truePositivesCount / ((double)truePositivesCount + falsePositivesCount);
                precisionRecallCurve.Add(new PrecisionRecall(precision, recall));
            }

            return precisionRecallCurve;
        }

        /// <summary>
        /// Computes the receiver operating characteristic curve.
        /// <para>
        /// The implementation follows Algorithm 2 as described in Fawcett, T. (2004): ROC Graphs: Notes and Practical 
        /// Considerations for Researchers.
        /// </para>
        /// </summary>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <param name="positiveInstances">The instances with 'positive' ground truth labels.</param>
        /// <param name="instanceScores">
        /// The predicted instance scores. The larger a predicted score, the more likely the instance is 
        /// to belong to the 'positive' class.
        /// </param>
        /// <returns>The points on the receiver operating characteristic curve, increasing by false positive rate.</returns>
        /// <remarks>
        /// All instances not contained in <paramref name="positiveInstances"/> are assumed to belong to the 'negative' class.
        /// </remarks>
        public static IEnumerable<FalseAndTruePositiveRate> ReceiverOperatingCharacteristicCurve<TInstance>(
            IEnumerable<TInstance> positiveInstances, IEnumerable<KeyValuePair<TInstance, double>> instanceScores)
        {
            if (positiveInstances == null)
            {
                throw new ArgumentNullException(nameof(positiveInstances));
            }

            if (instanceScores == null)
            {
                throw new ArgumentNullException(nameof(instanceScores));
            }

            // Compute the number of instances with positive and negative ground truth labels
            var positiveInstanceSet = new HashSet<TInstance>(positiveInstances);
            long positivesCount = positiveInstanceSet.Count();
            long negativesCount = instanceScores.Count() - positivesCount;

            if (positivesCount == 0)
            {
                throw new ArgumentException("There must be at least one instance with a 'positive' ground truth label.");
            }

            if (negativesCount <= 0)
            {
                throw new ArgumentException("There must be at least one instance with a 'negative' ground truth label.");
            }

            // Sort instances by their scores
            var sortedInstanceScores = from pair in instanceScores orderby pair.Value descending select pair;

            long falsePositivesCount = 0;
            long truePositivesCount = 0;
            double falsePositiveRate;
            double truePositiveRate;
            double previousScore = double.NaN;
            var rocCurve = new List<FalseAndTruePositiveRate>();

            foreach (var instance in sortedInstanceScores)
            {
                double score = instance.Value;
                if (score != previousScore)
                {
                    falsePositiveRate = falsePositivesCount / (double)negativesCount;
                    truePositiveRate = truePositivesCount / (double)positivesCount;
                    rocCurve.Add(new FalseAndTruePositiveRate(falsePositiveRate, truePositiveRate));
                    previousScore = score;
                }

                if (positiveInstanceSet.Contains(instance.Key))
                {
                    truePositivesCount++;
                }
                else
                {
                    falsePositivesCount++;
                }
            }

            // Add point for (1,1)
            falsePositiveRate = falsePositivesCount / (double)negativesCount;
            truePositiveRate = truePositivesCount / (double)positivesCount;
            rocCurve.Add(new FalseAndTruePositiveRate(falsePositiveRate, truePositiveRate));

            return rocCurve;
        }

        /// <summary>
        /// Computes the area under the receiver operating characteristic curve.
        /// <para>
        /// The implementation follows Algorithm 3 as described in Fawcett, T. (2004): ROC Graphs: Notes and Practical 
        /// Considerations for Researchers.
        /// </para>
        /// </summary>
        /// <typeparam name="TInstance">The type of an instance.</typeparam>
        /// <param name="positiveInstances">The instances with 'positive' ground truth labels.</param>
        /// <param name="instanceScores">
        /// The predicted instance scores. The larger a predicted score, the more likely the instance is 
        /// to belong to the 'positive' class.
        /// </param>
        /// <returns>AUC - the area under the receiver operating characteristic curve.</returns>
        /// <remarks>
        /// All instances not contained in <paramref name="positiveInstances"/> are assumed to belong to the 'negative' class.
        /// </remarks>
        public static double AreaUnderRocCurve<TInstance>(
            IEnumerable<TInstance> positiveInstances, IEnumerable<KeyValuePair<TInstance, double>> instanceScores)
        {
            if (positiveInstances == null)
            {
                throw new ArgumentNullException(nameof(positiveInstances));
            }

            if (instanceScores == null)
            {
                throw new ArgumentNullException(nameof(instanceScores));
            }

            // Compute the number of instances with positive and negative ground truth labels
            var positiveInstanceSet = new HashSet<TInstance>(positiveInstances);
            long positivesCount = positiveInstanceSet.Count();
            long negativesCount = instanceScores.Count() - positivesCount;

            if (positivesCount == 0)
            {
                throw new ArgumentException("There must be at least one instance with a 'positive' ground truth label.");
            }

            if (negativesCount <= 0)
            {
                throw new ArgumentException("There must be at least one instance with a 'negative' ground truth label.");
            }

            // Sort instances by their scores
            var sortedInstanceScores = from pair in instanceScores orderby pair.Value descending select pair;

            long falsePositivesCount = 0;
            long truePositivesCount = 0;
            long previousFalsePositivesCount = 0;
            long previousTruePositivesCount = 0;
            double area = 0.0;
            double previousScore = double.NaN;

            foreach (var instance in sortedInstanceScores)
            {
                double score = instance.Value;
                if (score != previousScore)
                {
                    // Add trapezoid area between current instance and previous instance
                    area += 0.5 * Math.Abs(falsePositivesCount - previousFalsePositivesCount) * (truePositivesCount + previousTruePositivesCount);
                    previousScore = score;
                    previousFalsePositivesCount = falsePositivesCount;
                    previousTruePositivesCount = truePositivesCount;
                }

                if (positiveInstanceSet.Contains(instance.Key))
                {
                    truePositivesCount++;
                }
                else
                {
                    falsePositivesCount++;
                }
            }

            // Add trapezoid area between last instance and (1, 1)
            area += 0.5 * Math.Abs(falsePositivesCount - previousFalsePositivesCount) * (truePositivesCount + previousTruePositivesCount);

            // Scale to unit square
            return area / (positivesCount * negativesCount);
        }

        #endregion

        #region Correlation measures

        /// <summary>
        /// Computes cosine similarity between two given vectors.
        /// </summary>
        /// <param name="vector1">The first vector.</param>
        /// <param name="vector2">The second vector.</param>
        /// <returns>The computed similarity value.</returns>
        /// <exception cref="ArgumentNullException">Thrown if one of the arguments is null.</exception>
        /// <exception cref="ArgumentException">
        /// Thrown if the similarity is not defined for the given pair of vectors.
        /// It can happen if the vectors are of different length or one of them has zero magnitude.
        /// </exception>
        public static double CosineSimilarity(Vector vector1, Vector vector2)
        {
            CheckVectorPair(vector1, vector2);

            double result = vector1.Inner(vector2) / Math.Sqrt(vector1.Inner(vector1) * vector2.Inner(vector2));
            if (double.IsNaN(result))
            {
                throw new ArgumentException("Similarity is not defined for the given pair of sequences.");
            }

            Debug.Assert(result >= -1 && result <= 1, "Similarity should always be in the [-1, 1] range.");
            return result;
        }

        /// <summary>
        /// Computes Pearson's correlation coefficient for a given pair of vectors.
        /// </summary>
        /// <param name="vector1">The first vector.</param>
        /// <param name="vector2">The second vector.</param>
        /// <returns>The computed correlation value.</returns>
        /// <exception cref="ArgumentNullException">Thrown if one of the arguments is null.</exception>
        /// <exception cref="ArgumentException">
        /// Thrown if the correlation is not defined for the given pair of vectors.
        /// It can happen if the vectors are of different length or one of them has zero variance.
        /// </exception>
        public static double PearsonCorrelation(Vector vector1, Vector vector2)
        {
            CheckVectorPair(vector1, vector2);

            double mean1 = vector1.Sum() / vector1.Count;
            double mean2 = vector2.Sum() / vector2.Count;
            return CosineSimilarity(
                vector1 - Vector.Constant(vector1.Count, mean1),
                vector2 - Vector.Constant(vector2.Count, mean2));
        }

        #endregion

        #region Distance measures

        /// <summary>
        /// Computes the similarity measure for a given pair of vectors based on the Euclidean distance between them.
        /// The similarity is computed as 1.0 / (1.0 + D(<paramref name="vector1"/>, <paramref name="vector2"/>),
        /// where D stands for the Euclidean distance divided by the square root of the dimensionality.
        /// </summary>
        /// <param name="vector1">The first vector.</param>
        /// <param name="vector2">The second vector.</param>
        /// <returns>The computed similarity value.</returns>
        /// <exception cref="ArgumentNullException">Thrown if one of the arguments is null.</exception>
        /// <exception cref="ArgumentException">
        /// Thrown if the distance is not defined for the given pair of vectors.
        /// It can happen if the vectors are of different or zero length.
        /// </exception>
        public static double NormalizedEuclideanSimilarity(Vector vector1, Vector vector2)
        {
            CheckVectorPair(vector1, vector2);

            Vector diff = vector1 - vector2;
            double distance = Math.Sqrt(diff.Inner(diff) / diff.Count);
            return 1.0 / (1.0 + distance);
        }

        /// <summary>
        /// Computes the similarity measure for a given pair of vectors based on Manhattan distance between them.
        /// The similarity is computed as 1.0 / (1.0 + D(<paramref name="vector1"/>, <paramref name="vector2"/>),
        /// where D stands for the Manhattan distance divided by the dimensionality.
        /// </summary>
        /// <param name="vector1">The first vector.</param>
        /// <param name="vector2">The second vector.</param>
        /// <returns>The computed similarity value.</returns>
        /// <exception cref="ArgumentNullException">Thrown if one of the arguments is null.</exception>
        /// <exception cref="ArgumentException">
        /// Thrown if the distance is not defined for the given pair of vectors.
        /// It can happen if the vectors are of different or zero length.
        /// </exception>
        public static double NormalizedManhattanSimilarity(Vector vector1, Vector vector2)
        {
            CheckVectorPair(vector1, vector2);

            double absoluteDifferenceSum = (vector1 - vector2).Sum(Math.Abs);
            double distance = absoluteDifferenceSum / vector1.Count;
            return 1.0 / (1.0 + distance);
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Checks if the given pair of vectors is of the same non-zero dimensionality.
        /// </summary>
        /// <param name="vector1">The first vector.</param>
        /// <param name="vector2">The second vector.</param>
        /// <exception cref="ArgumentNullException">Thrown if one of the arguments is null.</exception>
        /// <exception cref="ArgumentException">Thrown if arguments are of different or zero length.</exception>
        private static void CheckVectorPair(Vector vector1, Vector vector2)
        {
            if (vector1 == null)
            {
                throw new ArgumentNullException(nameof(vector1));
            }

            if (vector2 == null)
            {
                throw new ArgumentNullException(nameof(vector2));
            }

            int count1 = vector1.Count;
            int count2 = vector2.Count;

            if (count1 == 0)
            {
                throw new ArgumentException("The given vectors should have non-zero size.");
            }
            
            if (count1 != count2)
            {
                throw new ArgumentException("The given vectors should be of the same size.");
            }
        }

        #endregion
    }
}
