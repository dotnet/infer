// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Tests
{
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Xunit;
    using Assert = AssertHelper;
    using LabelDistribution = System.Collections.Generic.IDictionary<string, double>;


    /// <summary>
    /// Tests classifier evaluation methods.
    /// </summary>
    public class ClassifierEvaluatorTests
    {
        #region Test initialization

        /// <summary>
        /// Tolerance for comparisons.
        /// </summary>
        private const double Tolerance = 1e-15;

        /// <summary>
        /// The set of labels.
        /// </summary>
        private static readonly string[] LabelSet = { "A", "B", "C" };

        /// <summary>
        /// The ground truth labels.
        /// </summary>
        private LabelDistribution[] groundTruth;

        /// <summary>
        /// The predictive distributions.
        /// </summary>
        private LabelDistribution[] predictions;

        /// <summary>
        /// The classifier evaluator.
        /// </summary>
        private ClassifierEvaluator<IEnumerable<LabelDistribution>, LabelDistribution, IEnumerable<LabelDistribution>, string> evaluator;

        /// <summary>
        /// Prepares the environment before each test.
        /// </summary>
        public ClassifierEvaluatorTests()
        {
            // Ground truth labels (no uncertainty)
            this.groundTruth = new LabelDistribution[5];
            this.groundTruth[0] = new Dictionary<string, double> { { LabelSet[0], 1 }, { LabelSet[1], 0 }, { LabelSet[2], 0 } };
            this.groundTruth[1] = new Dictionary<string, double> { { LabelSet[0], 0 }, { LabelSet[1], 1 }, { LabelSet[2], 0 } };
            this.groundTruth[2] = new Dictionary<string, double> { { LabelSet[0], 0 }, { LabelSet[1], 0 }, { LabelSet[2], 1 } };
            this.groundTruth[3] = new Dictionary<string, double> { { LabelSet[0], 1 }, { LabelSet[1], 0 }, { LabelSet[2], 0 } };
            this.groundTruth[4] = new Dictionary<string, double> { { LabelSet[0], 1 }, { LabelSet[1], 0 }, { LabelSet[2], 0 } };

            // Predictions
            this.predictions = new LabelDistribution[5];
            this.predictions[0] = new Dictionary<string, double> { { LabelSet[0], 0 }, { LabelSet[1], 0 }, { LabelSet[2], 1 } };
            this.predictions[1] = new Dictionary<string, double> { { LabelSet[0], 0 }, { LabelSet[1], 1 }, { LabelSet[2], 0 } };
            this.predictions[2] = new Dictionary<string, double> { { LabelSet[0], 1 }, { LabelSet[1], 0 }, { LabelSet[2], 0 } };
            this.predictions[3] = new Dictionary<string, double> { { LabelSet[0], 1 / 6.0 }, { LabelSet[1], 2 / 3.0 }, { LabelSet[2], 1 / 6.0 } };
            this.predictions[4] = new Dictionary<string, double> { { LabelSet[0], 1 / 8.0 }, { LabelSet[1], 1 / 8.0 }, { LabelSet[2], 3 / 4.0 } };

            // Classifier evaluator
            var classifierMapping = new ClassifierMapping();
            var evaluatorMapping = classifierMapping.ForEvaluation();

            this.evaluator = new ClassifierEvaluator<IEnumerable<LabelDistribution>, LabelDistribution, IEnumerable<LabelDistribution>, string>(evaluatorMapping);
        }

        #endregion

        #region Test methods

        /// <summary>
        /// Tests correctness of the performance metric evaluation.
        /// </summary>
        [Fact]
        public void PerformanceMetricEvaluationTest()
        {
            // Perfect predictions
            double negativeLogProbability = this.evaluator.Evaluate(this.groundTruth, this.groundTruth, Metrics.NegativeLogProbability);
            Assert.Equal(0.0, negativeLogProbability);

            // Imperfect predictions
            negativeLogProbability = this.evaluator.Evaluate(this.groundTruth, this.predictions, Metrics.NegativeLogProbability);
            Assert.Equal(double.PositiveInfinity, negativeLogProbability);

            var uncertainPredictions = new LabelDistribution[5];
            uncertainPredictions[0] = new Dictionary<string, double> { { LabelSet[0], 0.5 }, { LabelSet[1], 0.25 }, { LabelSet[2], 0.25 } };
            uncertainPredictions[1] = new Dictionary<string, double> { { LabelSet[0], 0 }, { LabelSet[1], 1 }, { LabelSet[2], 0 } };
            uncertainPredictions[2] = new Dictionary<string, double> { { LabelSet[0], 0.25 }, { LabelSet[1], 0.25 }, { LabelSet[2], 0.5 } };
            uncertainPredictions[3] = new Dictionary<string, double> { { LabelSet[0], 1 / 6.0 }, { LabelSet[1], 2 / 3.0 }, { LabelSet[2], 1 / 6.0 } };
            uncertainPredictions[4] = new Dictionary<string, double> { { LabelSet[0], 1 / 8.0 }, { LabelSet[1], 1 / 8.0 }, { LabelSet[2], 3 / 4.0 } };

            negativeLogProbability = this.evaluator.Evaluate(this.groundTruth, uncertainPredictions, Metrics.NegativeLogProbability);
            Assert.Equal(5.2574953720277815, negativeLogProbability, Tolerance);

            // Insufficient number of predictions
            var insufficientPredictions = new LabelDistribution[1];
            insufficientPredictions[0] = new Dictionary<string, double> { { LabelSet[0], 0.5 }, { LabelSet[1], 0.25 }, { LabelSet[2], 0.25 } };
            Assert.Throws<ArgumentException>(() => this.evaluator.Evaluate(this.groundTruth, insufficientPredictions, Metrics.NegativeLogProbability));
        }

        /// <summary>
        /// Tests correctness of the confusion matrix.
        /// </summary>
        [Fact]
        public void ConfusionMatrixTest()
        {
            string expectedConfusionMatrixString = "Truth \\ Prediction ->" + Environment.NewLine + 
                                                         "     A  B  C" + Environment.NewLine + 
                                                         "  A  3  0  0" + Environment.NewLine + 
                                                         "  B  1  0  0" + Environment.NewLine +
                                                         "  C  1  0  0" + Environment.NewLine;

            var predictedLabels = new[] { LabelSet[0], LabelSet[0], LabelSet[0], LabelSet[0], LabelSet[0] };
            var confusionMatrix = this.evaluator.ConfusionMatrix(this.groundTruth, predictedLabels);

            // Verify ToString method
            Assert.Equal(expectedConfusionMatrixString, confusionMatrix.ToString());

            // Counts
            Assert.Equal(3, confusionMatrix[LabelSet[0], LabelSet[0]]);
            Assert.Equal(0, confusionMatrix[LabelSet[0], LabelSet[1]]);
            Assert.Equal(0, confusionMatrix[LabelSet[0], LabelSet[2]]);

            Assert.Equal(1, confusionMatrix[LabelSet[1], LabelSet[0]]);
            Assert.Equal(0, confusionMatrix[LabelSet[1], LabelSet[1]]);
            Assert.Equal(0, confusionMatrix[LabelSet[1], LabelSet[2]]);

            Assert.Equal(1, confusionMatrix[LabelSet[2], LabelSet[0]]);
            Assert.Equal(0, confusionMatrix[LabelSet[2], LabelSet[1]]);
            Assert.Equal(0, confusionMatrix[LabelSet[2], LabelSet[2]]);

            // Precision
            Assert.Equal(0.6, confusionMatrix.Precision(LabelSet[0]), Tolerance);
            Assert.Equal(double.NaN, confusionMatrix.Precision(LabelSet[1])); // undefined result
            Assert.Equal(double.NaN, confusionMatrix.Precision(LabelSet[2])); // undefined result
            Assert.Equal(0.6, confusionMatrix.MacroPrecision, Tolerance);
            Assert.Equal(0.36, confusionMatrix.MicroPrecision, Tolerance);

            // Recall
            Assert.Equal(1, confusionMatrix.Recall(LabelSet[0]));
            Assert.Equal(0, confusionMatrix.Recall(LabelSet[1]));
            Assert.Equal(0, confusionMatrix.Recall(LabelSet[2]));
            Assert.Equal(1 / 3.0, confusionMatrix.MacroRecall, Tolerance);
            Assert.Equal(0.6, confusionMatrix.MicroRecall, Tolerance);

            // Accuracy
            Assert.Equal(1, confusionMatrix.Accuracy(LabelSet[0]));
            Assert.Equal(0, confusionMatrix.Accuracy(LabelSet[1]));
            Assert.Equal(0, confusionMatrix.Accuracy(LabelSet[2]));
            Assert.Equal(1 / 3.0, confusionMatrix.MacroAccuracy, Tolerance);
            Assert.Equal(0.6, confusionMatrix.MicroAccuracy, Tolerance);

            // F1-measure
            Assert.Equal(0.75, confusionMatrix.F1(LabelSet[0]), Tolerance);
            Assert.Equal(double.NaN, confusionMatrix.F1(LabelSet[1])); // undefined result
            Assert.Equal(double.NaN, confusionMatrix.F1(LabelSet[2])); // undefined result
            Assert.Equal(0.75, confusionMatrix.MacroF1, Tolerance);
            Assert.Equal(0.45, confusionMatrix.MicroF1, Tolerance);
        }

        /// <summary>
        /// Tests correctness of the receiver operating characteristic curve (ROC).
        /// </summary>
        [Fact]
        public void RocCurveTest()
        {
            // Curve for perfect predictions
            var expected = new[] { new FalseAndTruePositiveRate(0.0, 0.0), new FalseAndTruePositiveRate(0.0, 1.0), new FalseAndTruePositiveRate(1.0, 1.0) };
            var actual = this.evaluator.ReceiverOperatingCharacteristicCurve(LabelSet[0], this.groundTruth, this.groundTruth).ToArray();
            Xunit.Assert.Equal(expected, actual);

            // Curve for imperfect predictions (one-versus-rest)
            expected = new[] { new FalseAndTruePositiveRate(0.0, 0.0), new FalseAndTruePositiveRate(0.5, 0.0), new FalseAndTruePositiveRate(0.5, 1 / 3.0), new FalseAndTruePositiveRate(0.5, 2 / 3.0), new FalseAndTruePositiveRate(1.0, 1.0) };
            actual = this.evaluator.ReceiverOperatingCharacteristicCurve(LabelSet[0], this.groundTruth, this.predictions).ToArray();
            Xunit.Assert.Equal(expected, actual); // matches below AUC = 5/12

            // Curve for imperfect predictions (one-versus-another)
            expected = new[] { new FalseAndTruePositiveRate(0.0, 0.0), new FalseAndTruePositiveRate(0.0, 1 / 3.0), new FalseAndTruePositiveRate(0.0, 2 / 3.0), new FalseAndTruePositiveRate(1.0, 1.0) };
            actual = this.evaluator.ReceiverOperatingCharacteristicCurve(LabelSet[0], LabelSet[1], this.groundTruth, this.predictions).ToArray();
            Xunit.Assert.Equal(expected, actual); // matches below AUC = 5/6

            // No positive or negative class labels
            var actualLabelDistribution = new LabelDistribution[1];
            actualLabelDistribution[0] = new Dictionary<string, double> { { LabelSet[0], 1 }, { LabelSet[1], 0 }, { LabelSet[2], 0 } };
            
            // One-versus-rest
            Assert.Throws<ArgumentException>(() => this.evaluator.ReceiverOperatingCharacteristicCurve(LabelSet[0], actualLabelDistribution, actualLabelDistribution));
            Assert.Throws<ArgumentException>(() => this.evaluator.ReceiverOperatingCharacteristicCurve(LabelSet[1], actualLabelDistribution, actualLabelDistribution));
            Assert.Throws<ArgumentException>(() => this.evaluator.ReceiverOperatingCharacteristicCurve(LabelSet[2], actualLabelDistribution, actualLabelDistribution));
            
            // One-versus-another
            Assert.Throws<ArgumentException>(() => this.evaluator.ReceiverOperatingCharacteristicCurve(LabelSet[0], LabelSet[2], actualLabelDistribution, actualLabelDistribution));
            Assert.Throws<ArgumentException>(() => this.evaluator.ReceiverOperatingCharacteristicCurve(LabelSet[1], LabelSet[0], actualLabelDistribution, actualLabelDistribution));
            Assert.Throws<ArgumentException>(() => this.evaluator.ReceiverOperatingCharacteristicCurve(LabelSet[2], LabelSet[1], actualLabelDistribution, actualLabelDistribution));

            // Positive and negative class labels are identical
            Assert.Throws<ArgumentException>(() => this.evaluator.ReceiverOperatingCharacteristicCurve(LabelSet[0], LabelSet[0], actualLabelDistribution, actualLabelDistribution));
        }

        /// <summary>
        /// Tests correctness of the precision-recall curve.
        /// </summary>
        [Fact]
        public void PrecisionRecallCurveTest()
        {
            // Curve for perfect predictions
            var expected = new[] { new PrecisionRecall(1.0, 0.0), new PrecisionRecall(1.0, 1 / 3.0), new PrecisionRecall(1.0, 2 / 3.0), new PrecisionRecall(1.0, 1.0), new PrecisionRecall(0.75, 1.0), new PrecisionRecall(0.6, 1.0) };
            var actual = this.evaluator.PrecisionRecallCurve(LabelSet[0], this.groundTruth, this.groundTruth).ToArray();
            Xunit.Assert.Equal(expected, actual);

            // Curve for imperfect predictions (one-versus-rest)
            expected = new[] { new PrecisionRecall(1.0, 0.0), new PrecisionRecall(0.0, 0.0), new PrecisionRecall(0.5, 1 / 3.0), new PrecisionRecall(2 / 3.0, 2 / 3.0), new PrecisionRecall(0.75, 1.0), new PrecisionRecall(0.6, 1.0) };
            actual = this.evaluator.PrecisionRecallCurve(LabelSet[0], this.groundTruth, this.predictions).ToArray();
            Xunit.Assert.Equal(expected, actual);

            // Curve for imperfect predictions (one-versus-another)
            expected = new[] { new PrecisionRecall(1.0, 0.0), new PrecisionRecall(1.0, 1.0), new PrecisionRecall(0.5, 1.0) };
            actual = this.evaluator.PrecisionRecallCurve(LabelSet[1], LabelSet[2], this.groundTruth, this.predictions).ToArray();
            Xunit.Assert.Equal(expected, actual);

            // No positive class labels
            var actualLabelDistribution = new LabelDistribution[1];
            actualLabelDistribution[0] = new Dictionary<string, double> { { LabelSet[0], 1 }, { LabelSet[1], 0 }, { LabelSet[2], 0 } };

            // One-versus-rest
            Assert.Throws<ArgumentException>(() => this.evaluator.PrecisionRecallCurve(LabelSet[1], actualLabelDistribution, actualLabelDistribution));
            Assert.Throws<ArgumentException>(() => this.evaluator.PrecisionRecallCurve(LabelSet[2], actualLabelDistribution, actualLabelDistribution));

            // One-versus-another
            Assert.Throws<ArgumentException>(() => this.evaluator.PrecisionRecallCurve(LabelSet[1], LabelSet[0], actualLabelDistribution, actualLabelDistribution));
            Assert.Throws<ArgumentException>(() => this.evaluator.PrecisionRecallCurve(LabelSet[2], LabelSet[0], actualLabelDistribution, actualLabelDistribution));

            // Positive and negative class labels are identical
            Assert.Throws<ArgumentException>(() => this.evaluator.PrecisionRecallCurve(LabelSet[0], LabelSet[0], actualLabelDistribution, actualLabelDistribution));
        }

        /// <summary>
        /// Tests correctness of the calibration curve.
        /// </summary>
        [Fact]
        public void CalibrationCurveTest()
        {
            // Curve for perfect predictions
            var expected = new[] { new CalibrationPair(0.25, 0.0), new CalibrationPair(0.75, 1.0) };
            var actual = this.evaluator.CalibrationCurve(LabelSet[0], this.groundTruth, this.groundTruth).ToArray();
            Xunit.Assert.Equal(expected, actual);

            // Curve for imperfect predictions (one-versus-rest)
            expected = new[] { new CalibrationPair(0.25, 0.75), new CalibrationPair(0.75, 0.0) };
            actual = this.evaluator.CalibrationCurve(LabelSet[0], this.groundTruth, this.predictions).ToArray();
            Xunit.Assert.Equal(expected, actual);

            // Curve for imperfect predictions (3 bins)
            const int BinCount = 4;
            expected = new[] { new CalibrationPair(1 / 8.0, 0.75), new CalibrationPair(7 / 8.0, 0.0) };
            actual = this.evaluator.CalibrationCurve(LabelSet[0], this.groundTruth, this.predictions, BinCount).ToArray();
            Xunit.Assert.Equal(expected, actual);

            // Ground truth instances without corresponding predictions
            var insufficientPredictions = new LabelDistribution[1];
            insufficientPredictions[0] = this.predictions[0];
            Assert.Throws<ArgumentException>(() => this.evaluator.CalibrationCurve(LabelSet[0], this.groundTruth, insufficientPredictions));
        }

        /// <summary>
        /// Tests correctness of the area under the receiver operating characteristic curve (AUC).
        /// </summary>
        [Fact]
        public void AreaUnderRocCurveTest()
        {
            // AUC for perfect predictions

            // Per-label AUC
            Assert.Equal(1.0, this.evaluator.AreaUnderRocCurve(LabelSet[0], this.groundTruth, this.groundTruth));
            Assert.Equal(1.0, this.evaluator.AreaUnderRocCurve(LabelSet[1], this.groundTruth, this.groundTruth));
            Assert.Equal(1.0, this.evaluator.AreaUnderRocCurve(LabelSet[2], this.groundTruth, this.groundTruth));

            // M-measure
            IDictionary<string, IDictionary<string, double>> computedAucMatrix;
            Assert.Equal(1.0, this.evaluator.AreaUnderRocCurve(this.groundTruth, this.groundTruth, out computedAucMatrix));
            Assert.Equal(1.0, this.evaluator.AreaUnderRocCurve(this.groundTruth, this.groundTruth));

            // Pairwise AUC (upper triangle)
            Assert.Equal(1.0, computedAucMatrix[LabelSet[0]][LabelSet[1]]);
            Assert.Equal(1.0, computedAucMatrix[LabelSet[0]][LabelSet[2]]);
            Assert.Equal(1.0, computedAucMatrix[LabelSet[1]][LabelSet[2]]);

            // Pairwise AUC (diagnonal)
            foreach (string label in LabelSet)
            {
                Assert.Equal(double.NaN, computedAucMatrix[label][label]); // undefined result
            }

            // Pairwise AUC (lower triangle)
            Assert.Equal(1.0, computedAucMatrix[LabelSet[1]][LabelSet[0]]);
            Assert.Equal(1.0, computedAucMatrix[LabelSet[2]][LabelSet[0]]);
            Assert.Equal(1.0, computedAucMatrix[LabelSet[2]][LabelSet[1]]);

            // AUC for imperfect predictions

            // Per-label AUC
            Assert.Equal(5 / 12.0, this.evaluator.AreaUnderRocCurve(LabelSet[0], this.groundTruth, this.predictions)); // matches ROC curve
            Assert.Equal(1.0, this.evaluator.AreaUnderRocCurve(LabelSet[1], this.groundTruth, this.predictions));
            Assert.Equal(1 / 8.0, this.evaluator.AreaUnderRocCurve(LabelSet[2], this.groundTruth, this.predictions));

            // M-measure
            Assert.Equal(5 / 9.0, this.evaluator.AreaUnderRocCurve(this.groundTruth, this.predictions, out computedAucMatrix));
            Assert.Equal(5 / 9.0, this.evaluator.AreaUnderRocCurve(this.groundTruth, this.predictions));

            // Pairwise AUC (upper triangle)
            Assert.Equal(5 / 6.0, computedAucMatrix[LabelSet[0]][LabelSet[1]]); // matches ROC curve
            Assert.Equal(0.0, computedAucMatrix[LabelSet[0]][LabelSet[2]]);
            Assert.Equal(1.0, computedAucMatrix[LabelSet[1]][LabelSet[2]]);

            // Pairwise AUC (diagnonal)
            foreach (string label in LabelSet)
            {
                Assert.Equal(double.NaN, computedAucMatrix[label][label]); // undefined result
            }

            // Pairwise AUC (lower triangle)
            Assert.Equal(1.0, computedAucMatrix[LabelSet[1]][LabelSet[0]]);
            Assert.Equal(0.0, computedAucMatrix[LabelSet[2]][LabelSet[0]]);
            Assert.Equal(0.5, computedAucMatrix[LabelSet[2]][LabelSet[1]]);

            // Test code path for symmetric two-class case
            var binaryGroundTruth = new LabelDistribution[2];
            binaryGroundTruth[0] = new Dictionary<string, double> { { LabelSet[0], 1 }, { LabelSet[1], 0 } };
            binaryGroundTruth[1] = new Dictionary<string, double> { { LabelSet[0], 0 }, { LabelSet[1], 1 } };

            var binaryPredictions = new LabelDistribution[2];
            binaryPredictions[0] = new Dictionary<string, double> { { LabelSet[0], 0.9 }, { LabelSet[1], 0.1 } };
            binaryPredictions[1] = new Dictionary<string, double> { { LabelSet[0], 0.8 }, { LabelSet[1], 0.2 } };

            Assert.Equal(1.0, this.evaluator.AreaUnderRocCurve(binaryGroundTruth, binaryPredictions, out computedAucMatrix));
            Assert.Equal(1.0, this.evaluator.AreaUnderRocCurve(binaryGroundTruth, binaryPredictions));
            Assert.Equal(double.NaN, computedAucMatrix[LabelSet[0]][LabelSet[0]]); // undefined result
            Assert.Equal(1.0, computedAucMatrix[LabelSet[0]][LabelSet[1]]);
            Assert.Equal(1.0, computedAucMatrix[LabelSet[1]][LabelSet[0]]);
            Assert.Equal(double.NaN, computedAucMatrix[LabelSet[1]][LabelSet[1]]); // undefined result

            // No positive or negative class labels
            var actualLabelDistribution = new LabelDistribution[1];
            actualLabelDistribution[0] = new Dictionary<string, double> { { LabelSet[0], 1 }, { LabelSet[1], 0 }, { LabelSet[2], 0 } };

            // One-versus-rest
            Assert.Throws<ArgumentException>(() => this.evaluator.AreaUnderRocCurve(LabelSet[0], actualLabelDistribution, actualLabelDistribution));
            Assert.Throws<ArgumentException>(() => this.evaluator.AreaUnderRocCurve(LabelSet[1], actualLabelDistribution, actualLabelDistribution));
            Assert.Throws<ArgumentException>(() => this.evaluator.AreaUnderRocCurve(LabelSet[2], actualLabelDistribution, actualLabelDistribution));

            // One-versus-another
            Assert.Throws<ArgumentException>(() => this.evaluator.AreaUnderRocCurve(LabelSet[0], LabelSet[2], actualLabelDistribution, actualLabelDistribution));
            Assert.Throws<ArgumentException>(() => this.evaluator.AreaUnderRocCurve(LabelSet[1], LabelSet[0], actualLabelDistribution, actualLabelDistribution));
            Assert.Throws<ArgumentException>(() => this.evaluator.AreaUnderRocCurve(LabelSet[2], LabelSet[1], actualLabelDistribution, actualLabelDistribution));

            // Positive and negative class labels are identical
            Assert.Throws<ArgumentException>(() => this.evaluator.ReceiverOperatingCharacteristicCurve(LabelSet[0], LabelSet[0], actualLabelDistribution, actualLabelDistribution));
        }

        #endregion

        #region IClassifierMapping implementation

        /// <summary>
        /// The classifier mapping.
        /// </summary>
        private class ClassifierMapping : ClassifierMapping<IEnumerable<LabelDistribution>, LabelDistribution, IEnumerable<LabelDistribution>, string, Vector>
        {
            /// <summary>
            /// Provides the instances for a given instance source.
            /// </summary>
            /// <param name="instanceSource">The source of instances.</param>
            /// <returns>The instances provided by the instance source.</returns>
            /// <remarks>Assumes that the same instance source always provides the same instances.</remarks>
            public override IEnumerable<LabelDistribution> GetInstances(IEnumerable<LabelDistribution> instanceSource)
            {
                if (instanceSource == null)
                {
                    throw new ArgumentNullException(nameof(instanceSource));
                }

                return instanceSource;
            }

            /// <summary>
            /// Provides the features for a given instance.
            /// </summary>
            /// <param name="instance">The instance to provide features for.</param>
            /// <param name="instanceSource">An optional source of instances.</param>
            /// <returns>The features for the given instance.</returns>
            /// <remarks>Assumes that the same instance source always provides the same features for a given instance.</remarks>
            public override Vector GetFeatures(LabelDistribution instance, IEnumerable<LabelDistribution> instanceSource = null)
            {
                throw new NotImplementedException("Features are not required in evaluation.");
            }

            /// <summary>
            /// Provides the label for a given instance.
            /// </summary>
            /// <param name="instance">The instance to provide the label for.</param>
            /// <param name="instanceSource">An optional source of instances.</param>
            /// <param name="labelSource">An optional source of labels.</param>
            /// <returns>The label of the given instance.</returns>
            /// <remarks>Assumes that the same sources always provide the same label for a given instance.</remarks>
            public override string GetLabel(LabelDistribution instance, IEnumerable<LabelDistribution> instanceSource = null, IEnumerable<LabelDistribution> labelSource = null)
            {
                if (instance == null)
                {
                    throw new ArgumentNullException(nameof(instance));
                }

                // Use zero-one loss function to determine point estimate (mode of distribution)
                string mode = string.Empty;
                double maximum = double.NegativeInfinity;
                foreach (var element in instance)
                {
                    if (element.Value > maximum)
                    {
                        maximum = element.Value;
                        mode = element.Key;
                    }
                }

                return mode;
            }

            /// <summary>
            /// Gets all class labels.
            /// </summary>
            /// <param name="instanceSource">An optional instance source.</param>
            /// <param name="labelSource">An optional label source.</param>
            /// <returns>All possible values of a label.</returns>
            public override IEnumerable<string> GetClassLabels(IEnumerable<LabelDistribution> instanceSource = null, IEnumerable<LabelDistribution> labelSource = null)
            {
                if (instanceSource == null)
                {
                    throw new ArgumentNullException(nameof(instanceSource));
                }

                return new HashSet<string>(instanceSource.SelectMany(instance => instance.Keys));
            }
        }

        #endregion
    }
}
