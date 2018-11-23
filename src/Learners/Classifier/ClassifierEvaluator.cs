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
    using Microsoft.ML.Probabilistic.Learners.Mappings;

    /// <summary>
    /// Evaluates the predictions of a classifier.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <remarks>
    /// Assumes that there are as many predictions as ground truth instances and that the order of 
    /// the predictions matches the order of the ground truth instances.
    /// </remarks>
    public class ClassifierEvaluator<TInstanceSource, TInstance, TLabelSource, TLabel>
    {
        #region Fields and constructors

        /// <summary>
        /// The mapping to access the ground truth labels.
        /// </summary>
        private readonly IClassifierEvaluatorMapping<TInstanceSource, TInstance, TLabelSource, TLabel> mapping;

        /// <summary>
        /// Initializes a new instance of the <see cref="ClassifierEvaluator{TInstanceSource, TInstance, TLabelSource, TLabel}"/> class.
        /// </summary>
        /// <param name="mapping">The mapping to access the ground truth labels.</param>
        public ClassifierEvaluator(
            IClassifierEvaluatorMapping<TInstanceSource, TInstance, TLabelSource, TLabel> mapping)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            this.mapping = mapping;
        }

        #endregion

        #region Generic evaluation methods

        /// <summary>
        /// Computes a performance metric for specified predictions and ground truth as provided by a given instance source.
        /// </summary>
        /// <typeparam name="TPrediction">The type of the prediction.</typeparam>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <param name="performanceMetric">The performance metric.</param>
        /// <returns>The computed performance metric.</returns>
        public double Evaluate<TPrediction>(
            TInstanceSource instanceSource,
            IEnumerable<TPrediction> predictions,
            Func<TLabel, TPrediction, double> performanceMetric)
        {
            return this.Evaluate(instanceSource, default(TLabelSource), predictions, performanceMetric);
        }

        /// <summary>
        /// Computes a performance metric for specified predictions and ground truth as provided by a given instance or label source.
        /// </summary>
        /// <typeparam name="TPrediction">The type of the prediction.</typeparam>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <param name="performanceMetric">The performance metric.</param>
        /// <returns>The computed performance metric.</returns>
        public double Evaluate<TPrediction>(
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<TPrediction> predictions,
            Func<TLabel, TPrediction, double> performanceMetric)
        {
            if (predictions == null)
            {
                throw new ArgumentNullException(nameof(predictions));
            }

            if (performanceMetric == null)
            {
                throw new ArgumentNullException(nameof(performanceMetric));
            }

            double sum = 0;
            var groundTruthInstances = this.mapping.GetInstancesSafe(instanceSource);
            using (var predictionIterator = predictions.GetEnumerator())
            {
                foreach (var instance in groundTruthInstances)
                {
                    if (!predictionIterator.MoveNext())
                    {
                        throw new ArgumentException(
                            "There must be a prediction for each ground truth instance.", nameof(predictions));
                    }

                    TPrediction prediction = predictionIterator.Current;
                    TLabel groundTruthLabel = this.mapping.GetLabelSafe(instance, instanceSource, labelSource);

                    sum += performanceMetric(groundTruthLabel, prediction);
                }
            }

            return sum;
        }

        /// <summary>
        /// Computes a performance metric for a specified positive class, assuming all other classes are negative.
        /// </summary>
        /// <typeparam name="TResult">The result of the performance metric.</typeparam>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <param name="performanceMetric">The performance metric.</param>
        /// <returns>The computed performance metric.</returns>
        public TResult EvaluateOneVersusRest<TResult>(
            TLabel positiveClassLabel,
            TInstanceSource instanceSource,
            IEnumerable<IDictionary<TLabel, double>> predictions,
            Func<IEnumerable<int>, IDictionary<int, double>, TResult> performanceMetric)
        {
            return this.EvaluateOneVersusRest(positiveClassLabel, instanceSource, default(TLabelSource), predictions, performanceMetric);
        }

        /// <summary>
        /// Computes a performance metric for a specified positive class, assuming all other classes are negative.
        /// </summary>
        /// <typeparam name="TResult">The result of the performance metric.</typeparam>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <param name="performanceMetric">The performance metric.</param>
        /// <returns>The computed performance metric.</returns>
        public TResult EvaluateOneVersusRest<TResult>(
            TLabel positiveClassLabel,
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<IDictionary<TLabel, double>> predictions,
            Func<IEnumerable<int>, IDictionary<int, double>, TResult> performanceMetric)
        {
            if (positiveClassLabel == null)
            {
                throw new ArgumentNullException(nameof(positiveClassLabel));
            }

            if (predictions == null)
            {
                throw new ArgumentNullException(nameof(predictions));
            }

            if (performanceMetric == null)
            {
                throw new ArgumentNullException(nameof(performanceMetric));
            }

            // Get positive instances
            var positiveInstances = this.GetPositiveInstances(positiveClassLabel, instanceSource, labelSource);

            // Get instance scores
            var instanceScores = GetScoresOneVersusRest(positiveClassLabel, predictions);

            // Compute performance metric
            return performanceMetric(positiveInstances, instanceScores);
        }

        /// <summary>
        /// Computes a performance metric for specified positive and negative classes.
        /// </summary>
        /// <typeparam name="TResult">The result of the performance metric.</typeparam>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="negativeClassLabel">The label of the negative class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <param name="performanceMetric">The performance metric.</param>
        /// <returns>The computed performance metric.</returns>
        public TResult EvaluateOneVersusAnother<TResult>(
            TLabel positiveClassLabel,
            TLabel negativeClassLabel,
            TInstanceSource instanceSource,
            IEnumerable<IDictionary<TLabel, double>> predictions,
            Func<IEnumerable<int>, IDictionary<int, double>, TResult> performanceMetric)
        {
            return this.EvaluateOneVersusAnother(
                positiveClassLabel, negativeClassLabel, instanceSource, default(TLabelSource), predictions, performanceMetric);
        }

        /// <summary>
        /// Computes a performance metric for specified positive and negative classes.
        /// </summary>
        /// <typeparam name="TResult">The result of the performance metric.</typeparam>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="negativeClassLabel">The label of the negative class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <param name="performanceMetric">The performance metric.</param>
        /// <returns>The computed performance metric.</returns>
        public TResult EvaluateOneVersusAnother<TResult>(
            TLabel positiveClassLabel,
            TLabel negativeClassLabel,
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<IDictionary<TLabel, double>> predictions,
            Func<IEnumerable<int>, IDictionary<int, double>, TResult> performanceMetric)
        {
            if (positiveClassLabel == null)
            {
                throw new ArgumentNullException(nameof(positiveClassLabel));
            }

            if (negativeClassLabel == null)
            {
                throw new ArgumentNullException(nameof(negativeClassLabel));
            }

            if (predictions == null)
            {
                throw new ArgumentNullException(nameof(predictions));
            }

            if (performanceMetric == null)
            {
                throw new ArgumentNullException(nameof(performanceMetric));
            }

            if (positiveClassLabel.Equals(negativeClassLabel))
            {
                throw new ArgumentException("The positive and negative class labels must be distinct, but are both '" + positiveClassLabel + "'.");
            }

            // Get positive instances
            var positiveInstances = this.GetPositiveInstances(positiveClassLabel, instanceSource, labelSource);

            // Get instance scores
            var instanceScores = this.GetScoresOneVersusAnother(positiveClassLabel, negativeClassLabel, instanceSource, labelSource, predictions);

            // Compute performance metric
            return performanceMetric(positiveInstances, instanceScores);
        }

        #endregion

        #region Confusion matrix

        /// <summary>
        /// Computes a confusion matrix for specified predictions given the ground truth provided by an instance source.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The computed confusion matrix.</returns>
        public ConfusionMatrix<TLabel> ConfusionMatrix(TInstanceSource instanceSource, IEnumerable<TLabel> predictions)
        {
            return this.ConfusionMatrix(instanceSource, default(TLabelSource), predictions);
        }

        /// <summary>
        /// Computes a confusion matrix for specified predictions given the ground truth provided by an instance or label source.
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The computed confusion matrix.</returns>
        public ConfusionMatrix<TLabel> ConfusionMatrix(
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<TLabel> predictions)
        {
            if (predictions == null)
            {
                throw new ArgumentNullException(nameof(predictions));
            }
            
            var confusionMatrix = new ConfusionMatrix<TLabel>(this.mapping.GetClassLabelsSafe(instanceSource, labelSource));

            var groundTruthInstances = this.mapping.GetInstancesSafe(instanceSource);
            using (var predictionIterator = predictions.GetEnumerator())
            {
                foreach (var instance in groundTruthInstances)
                {
                    if (!predictionIterator.MoveNext())
                    {
                        throw new ArgumentException(
                            "There must be a prediction for each ground truth instance.", nameof(predictions));
                    }

                    TLabel groundTruthLabel = this.mapping.GetLabelSafe(instance, instanceSource, labelSource);
                    TLabel predictedLabel = predictionIterator.Current;

                    confusionMatrix[groundTruthLabel, predictedLabel]++;
                }
            }

            return confusionMatrix;
        }

        #endregion

        #region Precision-recall curve

        /// <summary>
        /// Computes the precision-recall curve for specified predictions 
        /// given the ground truth provided by an instance source.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The computed precision-recall curve.</returns>
        public IEnumerable<PrecisionRecall> PrecisionRecallCurve(
            TLabel positiveClassLabel,
            TInstanceSource instanceSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            return this.PrecisionRecallCurve(positiveClassLabel, instanceSource, default(TLabelSource), predictions);
        }

        /// <summary>
        /// Computes the precision-recall curve for specified predictions 
        /// given the ground truth provided by an instance or label source.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The computed precision-recall curve.</returns>
        public IEnumerable<PrecisionRecall> PrecisionRecallCurve(
            TLabel positiveClassLabel,
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            return this.EvaluateOneVersusRest(positiveClassLabel, instanceSource, labelSource, predictions, Metrics.PrecisionRecallCurve);
        }

        /// <summary>
        /// Computes the precision-recall curve for specified predictions 
        /// given the ground truth provided by an instance source.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="negativeClassLabel">The label of the negative class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The computed precision-recall curve.</returns>
        public IEnumerable<PrecisionRecall> PrecisionRecallCurve(
            TLabel positiveClassLabel,
            TLabel negativeClassLabel,
            TInstanceSource instanceSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            return this.PrecisionRecallCurve(positiveClassLabel, negativeClassLabel, instanceSource, default(TLabelSource), predictions);
        }

        /// <summary>
        /// Computes the precision-recall curve for specified predictions 
        /// given the ground truth provided by an instance or label source.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="negativeClassLabel">The label of the negative class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The computed precision-recall curve.</returns>
        public IEnumerable<PrecisionRecall> PrecisionRecallCurve(
            TLabel positiveClassLabel,
            TLabel negativeClassLabel,
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            return this.EvaluateOneVersusAnother(
                positiveClassLabel, negativeClassLabel, instanceSource, labelSource, predictions, Metrics.PrecisionRecallCurve);
        }

        #endregion

        #region Receiver operating characteristic (ROC) curve

        /// <summary>
        /// Computes the receiver operating characteristic curve for specified predictions 
        /// given the ground truth provided by an instance source.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The computed receiver operating characteristic curve.</returns>
        public IEnumerable<FalseAndTruePositiveRate> ReceiverOperatingCharacteristicCurve(
            TLabel positiveClassLabel,
            TInstanceSource instanceSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            return this.ReceiverOperatingCharacteristicCurve(positiveClassLabel, instanceSource, default(TLabelSource), predictions);
        }

        /// <summary>
        /// Computes the receiver operating characteristic curve for specified predictions 
        /// given the ground truth provided by an instance or label source.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The computed receiver operating characteristic curve.</returns>
        public IEnumerable<FalseAndTruePositiveRate> ReceiverOperatingCharacteristicCurve(
            TLabel positiveClassLabel,
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            return this.EvaluateOneVersusRest(
                positiveClassLabel, instanceSource, labelSource, predictions, Metrics.ReceiverOperatingCharacteristicCurve);
        }

        /// <summary>
        /// Computes the receiver operating characteristic curve for specified predictions 
        /// given the ground truth provided by an instance source.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="negativeClassLabel">The label of the negative class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The computed receiver operating characteristic curve.</returns>
        public IEnumerable<FalseAndTruePositiveRate> ReceiverOperatingCharacteristicCurve(
            TLabel positiveClassLabel,
            TLabel negativeClassLabel,
            TInstanceSource instanceSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            return this.ReceiverOperatingCharacteristicCurve(positiveClassLabel, negativeClassLabel, instanceSource, default(TLabelSource), predictions);
        }

        /// <summary>
        /// Computes the receiver operating characteristic curve for specified predictions 
        /// given the ground truth provided by an instance or label source.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="negativeClassLabel">The label of the negative class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The computed receiver operating characteristic curve.</returns>
        public IEnumerable<FalseAndTruePositiveRate> ReceiverOperatingCharacteristicCurve(
            TLabel positiveClassLabel,
            TLabel negativeClassLabel,
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            return this.EvaluateOneVersusAnother(
                positiveClassLabel, negativeClassLabel, instanceSource, labelSource, predictions, Metrics.ReceiverOperatingCharacteristicCurve);
        }

        #endregion

        #region Area under the ROC curve

        /// <summary>
        /// Computes the area under the receiver operating characteristic curve for specified predictions 
        /// given the ground truth provided by an instance source.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The area under the receiver operating characteristic curve.</returns>
        public double AreaUnderRocCurve(
            TLabel positiveClassLabel,
            TInstanceSource instanceSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            return this.AreaUnderRocCurve(positiveClassLabel, instanceSource, default(TLabelSource), predictions);
        }

        /// <summary>
        /// Computes the area under the receiver operating characteristic curve for specified predictions 
        /// given the ground truth provided by an instance or label source.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The area under the receiver operating characteristic curve.</returns>
        public double AreaUnderRocCurve(
            TLabel positiveClassLabel,
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            return this.EvaluateOneVersusRest(positiveClassLabel, instanceSource, labelSource, predictions, Metrics.AreaUnderRocCurve);
        }

        /// <summary>
        /// Computes the area under the receiver operating characteristic curve for specified predictions 
        /// given the ground truth provided by an instance source.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="negativeClassLabel">The label of the negative class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The area under the receiver operating characteristic curve.</returns>
        public double AreaUnderRocCurve(
            TLabel positiveClassLabel,
            TLabel negativeClassLabel,
            TInstanceSource instanceSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            return this.AreaUnderRocCurve(positiveClassLabel, negativeClassLabel, instanceSource, default(TLabelSource), predictions);
        }

        /// <summary>
        /// Computes the area under the receiver operating characteristic curve for specified predictions 
        /// given the ground truth provided by an instance or label source.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        /// <param name="negativeClassLabel">The label of the negative class.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The area under the receiver operating characteristic curve.</returns>
        public double AreaUnderRocCurve(
            TLabel positiveClassLabel,
            TLabel negativeClassLabel,
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            return this.EvaluateOneVersusAnother(
                positiveClassLabel, negativeClassLabel, instanceSource, labelSource, predictions, Metrics.AreaUnderRocCurve);
        }

        /// <summary>
        /// Computes the area under the receiver operating characteristic curve for specified predictions 
        /// given the ground truth provided by an instance source.
        /// <para>
        /// See Hand, D.J. and Till, R.J. (2001): "A simple generalization of the Area Under the ROC 
        /// Curve for Multiple Class Classification Problems". Machine Learning, 45, pages 171-186.
        /// </para>
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>
        /// The area under the receiver operating characteristic curve. Returns <see cref="double.NaN"/> 
        /// if there are no positive or no negative instances.
        /// </returns>
        public double AreaUnderRocCurve(
            TInstanceSource instanceSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            return this.AreaUnderRocCurve(instanceSource, default(TLabelSource), predictions);
        }

        /// <summary>
        /// Computes the area under the receiver operating characteristic curve for specified predictions 
        /// given the ground truth provided by an instance source.
        /// <para>
        /// See Hand, D.J. and Till, R.J. (2001): "A simple generalization of the Area Under the ROC 
        /// Curve for Multiple Class Classification Problems". Machine Learning, 45, pages 171-186.
        /// </para>
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <param name="aucMatrix">
        /// A matrix of pairwise AUC metrics. If the area under the curve is not defined for a given pair
        /// of classes, the corresponding element in the matrix is <see cref="double.NaN"/>.
        /// </param>
        /// <returns>
        /// The area under the receiver operating characteristic curve. Returns <see cref="double.NaN"/> 
        /// if there are no positive or no negative instances.
        /// </returns>
        public double AreaUnderRocCurve(
            TInstanceSource instanceSource,
            IEnumerable<IDictionary<TLabel, double>> predictions,
            out IDictionary<TLabel, IDictionary<TLabel, double>> aucMatrix)
        {
            return this.AreaUnderRocCurve(instanceSource, default(TLabelSource), predictions, out aucMatrix);
        }

        /// <summary>
        /// Computes the area under the receiver operating characteristic curve for specified predictions 
        /// given the ground truth provided by an instance or label source.
        /// <para>
        /// See Hand, D.J. and Till, R.J. (2001): "A simple generalization of the Area Under the ROC 
        /// Curve for Multiple Class Classification Problems". Machine Learning, 45, pages 171-186.
        /// </para>
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>
        /// The area under the receiver operating characteristic curve. Returns <see cref="double.NaN"/> 
        /// if there are no positive or no negative instances.
        /// </returns>
        public double AreaUnderRocCurve(
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            if (predictions == null)
            {
                throw new ArgumentNullException(nameof(predictions));
            }

            // Get class labels
            var classLabelSet = new IndexedSet<TLabel>(this.mapping.GetClassLabelsSafe(instanceSource, labelSource));
            var classLabels = classLabelSet.Elements.ToList();

            if (classLabels.Count == 2)
            {
                // For two classes, AUC(i,j) = AUC(j,i)
                try
                {
                    return this.AreaUnderRocCurve(classLabels[0], classLabels[1], instanceSource, labelSource, predictions);
            }
                catch (ArgumentException)
                {
                    // No positive or no negative instances
                    return double.NaN;
                }
            }

            // For all distinct pairs of label classes...
            double averagedAuc = 0.0;
            foreach (var groundTruthLabel in classLabels)
            {
                foreach (var predictedLabel in classLabelSet.Elements)
                {
                    if (groundTruthLabel.Equals(predictedLabel))
                    {
                        continue;
                    }

                    // Get positive instances
                    var positiveInstances = this.GetPositiveInstances(groundTruthLabel, instanceSource, labelSource, classLabelSet);

                    // Get instance scores
                    var instanceScores = this.GetScoresOneVersusAnother(groundTruthLabel, predictedLabel, instanceSource, labelSource, predictions);

                    // Compute AUC
                    double auc;
                    try
                    {
                        auc = Metrics.AreaUnderRocCurve(positiveInstances, instanceScores);
                    }
                    catch (ArgumentException)
                    {
                        // No positive or no negative instances
                        return double.NaN;
                    }

                    averagedAuc += auc;
                }
            }

            // Average AUC over all pairs
            averagedAuc = averagedAuc / ((double)classLabels.Count * (classLabels.Count - 1));

            return averagedAuc;
        }

        /// <summary>
        /// Computes the area under the receiver operating characteristic curve for specified predictions 
        /// given the ground truth provided by an instance or label source.
        /// <para>
        /// See Hand, D.J. and Till, R.J. (2001): "A simple generalization of the Area Under the ROC 
        /// Curve for Multiple Class Classification Problems". Machine Learning, 45, pages 171-186.
        /// </para>
        /// </summary>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <param name="aucMatrix">
        /// A matrix of pairwise AUC metrics. If the area under the curve is not defined for a given pair
        /// of classes, the corresponding element in the matrix is <see cref="double.NaN"/>.
        /// </param>
        /// <returns>
        /// The area under the receiver operating characteristic curve. Returns <see cref="double.NaN"/> 
        /// if there are no positive or no negative instances.
        /// </returns>
        public double AreaUnderRocCurve(
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<IDictionary<TLabel, double>> predictions, 
            out IDictionary<TLabel, IDictionary<TLabel, double>> aucMatrix)
        {
            if (predictions == null)
            {
                throw new ArgumentNullException(nameof(predictions));
            }

            // Get class labels
            var classLabelSet = new IndexedSet<TLabel>(this.mapping.GetClassLabelsSafe(instanceSource, labelSource));
            var classLabels = classLabelSet.Elements.ToList();

            if (classLabels.Count == 2)
            {
                // For two classes, AUC(i,j) = AUC(j,i)
                double auc;
                try
                {
                    auc = this.AreaUnderRocCurve(classLabels[0], classLabels[1], instanceSource, labelSource, predictions);
                }
                catch (ArgumentException)
                {
                    // No positive or no negative instances
                    auc = double.NaN;
                }

                aucMatrix = new Dictionary<TLabel, IDictionary<TLabel, double>>();
                aucMatrix.Add(classLabels[0], new Dictionary<TLabel, double> { { classLabels[0], double.NaN }, { classLabels[1], auc } });
                aucMatrix.Add(classLabels[1], new Dictionary<TLabel, double> { { classLabels[0], auc }, { classLabels[1], double.NaN } });
                return auc;
            }

            // Setup matrix of pairwise AUC metrics
            aucMatrix = new Dictionary<TLabel, IDictionary<TLabel, double>>();

            // For all distinct pairs of label classes...
            double averagedAuc = 0.0;
            foreach (var groundTruthLabel in classLabels)
            {
                aucMatrix.Add(groundTruthLabel, new Dictionary<TLabel, double>());
                foreach (var predictedLabel in classLabelSet.Elements)
                {
                    if (groundTruthLabel.Equals(predictedLabel))
                    {
                        aucMatrix[groundTruthLabel].Add(predictedLabel, double.NaN);
                        continue;
                    }

                    // Get positive instances
                    var positiveInstances = this.GetPositiveInstances(groundTruthLabel, instanceSource, labelSource, classLabelSet);

                    // Get instance scores
                    var instanceScores = this.GetScoresOneVersusAnother(groundTruthLabel, predictedLabel, instanceSource, labelSource, predictions);

                    // Compute AUC
                    double auc;
                    try
                    {
                        auc = Metrics.AreaUnderRocCurve(positiveInstances, instanceScores);
                    }
                    catch (ArgumentException)
                    {
                        // No positive or no negative instances
                        auc = double.NaN;
                    }

                    aucMatrix[groundTruthLabel].Add(predictedLabel, auc);
                    averagedAuc += auc;
                }
            }

            // Average AUC over all pairs
            averagedAuc = averagedAuc / ((double)classLabels.Count * (classLabels.Count - 1));

            return averagedAuc;
        }

        #endregion

        #region Empirical probability calibration curve

        /// <summary>
        /// Computes the empirical probability calibration curve for the class of the specified label.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the class to compute the curve for.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The computed empirical probability calibration curve.</returns>
        public IEnumerable<CalibrationPair> CalibrationCurve(
            TLabel positiveClassLabel,
            TInstanceSource instanceSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            return this.CalibrationCurve(positiveClassLabel, instanceSource, default(TLabelSource), predictions);
        }

        /// <summary>
        /// Computes the empirical probability calibration curve for the class of the specified label.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the class to compute the curve for.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <param name="binCount">The number of bins to use.</param>
        /// <param name="minBinInstanceCount">The minimal number of instances per bin. Defaults to 1.</param>
        /// <returns>The computed empirical probability calibration curve.</returns>
        public IEnumerable<CalibrationPair> CalibrationCurve(
            TLabel positiveClassLabel,
            TInstanceSource instanceSource,
            IEnumerable<IDictionary<TLabel, double>> predictions,
            int binCount,
            int minBinInstanceCount = 1)
        {
            return this.CalibrationCurve(positiveClassLabel, instanceSource, default(TLabelSource), predictions, binCount, minBinInstanceCount);
        }

        /// <summary>
        /// Computes the empirical probability calibration curve for the class of the specified label.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the class to compute the curve for.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The computed empirical probability calibration curve.</returns>
        public IEnumerable<CalibrationPair> CalibrationCurve(
            TLabel positiveClassLabel,
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            if (positiveClassLabel == null)
            {
                throw new ArgumentNullException(nameof(positiveClassLabel));
            }

            if (predictions == null)
            {
                throw new ArgumentNullException(nameof(predictions));
            }

            var groundTruthInstances = this.mapping.GetInstancesSafe(instanceSource).ToArray();
            var binCount = (int)Math.Sqrt(groundTruthInstances.Length);

            return this.CalibrationCurve(
                positiveClassLabel, instanceSource, labelSource, predictions, binCount, groundTruthInstances);
        }

        /// <summary>
        /// Computes the empirical probability calibration curve for the class of the specified label.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the class to compute the curve for.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <param name="binCount">The number of bins to use.</param>
        /// <param name="minBinInstanceCount">The minimal number of instances per bin. Defaults to 1.</param>
        /// <returns>The computed empirical probability calibration curve.</returns>
        public IEnumerable<CalibrationPair> CalibrationCurve(
            TLabel positiveClassLabel, 
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<IDictionary<TLabel, double>> predictions,
            int binCount,
            int minBinInstanceCount = 1)
        {
            if (positiveClassLabel == null)
            {
                throw new ArgumentNullException(nameof(positiveClassLabel));
            }

            if (predictions == null)
            {
                throw new ArgumentNullException(nameof(predictions));
            }

            if (binCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(binCount), "The number of bins must be greater than 0.");
            }

            if (minBinInstanceCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(minBinInstanceCount), "The minimal number of instance per bin must be greater than 0.");
            }

            var groundTruthInstances = this.mapping.GetInstancesSafe(instanceSource);

            return this.CalibrationCurve(
                positiveClassLabel, instanceSource, labelSource, predictions, binCount, groundTruthInstances, minBinInstanceCount);
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Gets the instance scores of one class versus all other classes.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the class to get the instance scores of.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The computed instance scores.</returns>
        private static IDictionary<int, double> GetScoresOneVersusRest(TLabel positiveClassLabel, IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            Debug.Assert(positiveClassLabel != null, "The label of the positive class must not be null.");
            Debug.Assert(predictions != null, "The predictions must not be null.");

            int instanceId = 0;
            var instanceScores = new Dictionary<int, double>();
            foreach (var prediction in predictions)
            {
                double score = prediction[positiveClassLabel];
                instanceScores.Add(instanceId, score);

                instanceId++;
            }

            return instanceScores;
        }

        /// <summary>
        /// Computes the empirical probability calibration curve for the class of the specified label.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the class to generate the calibration curve for.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <param name="binCount">The number of bins to use.</param>
        /// <param name="groundTruthInstances">The ground truth instances.</param>
        /// <param name="minBinInstanceCount">The minimal number of instances per bin. Defaults to 1.</param>
        /// <returns>The computed empirical probability calibration curve.</returns>
        private IEnumerable<CalibrationPair> CalibrationCurve(
            TLabel positiveClassLabel,
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<IDictionary<TLabel, double>> predictions,
            int binCount,
            IEnumerable<TInstance> groundTruthInstances,
            int minBinInstanceCount = 1)
        {
            Debug.Assert(positiveClassLabel != null, "The label of the positive class must not be null.");
            Debug.Assert(instanceSource != null, "The instance source must not be null.");
            Debug.Assert(predictions != null, "The predictions must not be null.");
            Debug.Assert(groundTruthInstances != null, "The ground truth instances must not be null.");
            Debug.Assert(binCount > 0, "The number of bins must be greater than 0.");
            Debug.Assert(minBinInstanceCount > 0, "The minimal number of instances per bin must be greater than 0.");

            var positiveClassCount = new int[binCount];
            var predictedCount = new int[binCount];

            using (var predictionIterator = predictions.GetEnumerator())
            {
                foreach (var instance in groundTruthInstances)
                {
                    if (!predictionIterator.MoveNext())
                    {
                        throw new ArgumentException(
                            "There must be a prediction for each ground truth instance.", nameof(predictions));
                    }

                    var groundTruthLabel = this.mapping.GetLabelSafe(instance, instanceSource, labelSource);
                    var prediction = predictionIterator.Current;

                    double prob = prediction[positiveClassLabel];

                    int probBin = Math.Min((int)(prob * binCount), binCount - 1);

                    predictedCount[probBin] += 1;
                    positiveClassCount[probBin] += groundTruthLabel.Equals(positiveClassLabel) ? 1 : 0;
                }
            }

            var calibrationCurve = new List<CalibrationPair>();
            for (int bin = 0; bin < binCount; bin++)
            {
                if (predictedCount[bin] >= minBinInstanceCount)
                {
                    calibrationCurve.Add(new CalibrationPair(
                        (bin + 0.5) / binCount,
                        (double)positiveClassCount[bin] / predictedCount[bin]));
                }
            }

            return calibrationCurve;
        }

        /// <summary>
        /// Gets a collection of instances identifiers for the class with the specified label.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the class to get the instances of.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">An optional label source.</param>
        /// <param name="labelSet">An optional set of labels.</param>
        /// <returns> A collection of instance identifiers for the class with the specified label.</returns>
        private IEnumerable<int> GetPositiveInstances(
            TLabel positiveClassLabel, 
            TInstanceSource instanceSource, 
            TLabelSource labelSource = default(TLabelSource), 
            IndexedSet<TLabel> labelSet = null)
        {
            Debug.Assert(positiveClassLabel != null, "The label of the positive class must not be null.");
            Debug.Assert(instanceSource != null, "The instance source must not be null.");

            if (labelSet == null)
            {
                labelSet = new IndexedSet<TLabel>(this.mapping.GetClassLabelsSafe(instanceSource, labelSource));
            }

            int positiveClassLabelId;
            if (!labelSet.TryGetIndex(positiveClassLabel, out positiveClassLabelId))
            {
                throw new ArgumentException("The positive class label '" + positiveClassLabel + "' must be present in the ground truth.");
            }

            var groundTruthInstances = this.mapping.GetInstancesSafe(instanceSource);

            int instanceId = 0;
            var positiveInstances = new List<int>();
            foreach (var instance in groundTruthInstances)
            {
                TLabel label = this.mapping.GetLabelSafe(instance, instanceSource, labelSource);
                int labelIndex;
                if (!labelSet.TryGetIndex(label, out labelIndex))
                {
                    // Should not occur!
                    throw new ArgumentException("Unknown label '" + label + "'.");
                }

                if (labelIndex == positiveClassLabelId)
                {
                    positiveInstances.Add(instanceId);
                }

                instanceId++;
            }

            return positiveInstances;
        }

        /// <summary>
        /// Gets the instance scores of one class versus another class.
        /// </summary>
        /// <param name="positiveClassLabel">The label of the class to get the instance scores of.</param>
        /// <param name="negativeClassLabel">The label of the class to compare to.</param>
        /// <param name="instanceSource">The instance source.</param>
        /// <param name="labelSource">The label source.</param>
        /// <param name="predictions">The predictions.</param>
        /// <returns>The computed instance scores.</returns>
        private IDictionary<int, double> GetScoresOneVersusAnother(
            TLabel positiveClassLabel, 
            TLabel negativeClassLabel, 
            TInstanceSource instanceSource,
            TLabelSource labelSource,
            IEnumerable<IDictionary<TLabel, double>> predictions)
        {
            Debug.Assert(positiveClassLabel != null, "The label of the positive class must not be null.");
            Debug.Assert(negativeClassLabel != null, "The label of the negative class must not be null.");
            Debug.Assert(predictions != null, "The predictions must not be null.");

            var groundTruthInstances = this.mapping.GetInstancesSafe(instanceSource);

            int instanceId = 0;
            var instanceScores = new Dictionary<int, double>();
            using (var predictionIterator = predictions.GetEnumerator())
            {
                foreach (var instance in groundTruthInstances)
                {
                    if (!predictionIterator.MoveNext())
                    {
                        throw new ArgumentException(
                            "There must be a prediction for each ground truth instance.", nameof(predictions));
                    }

                    TLabel groundTruthLabel = this.mapping.GetLabelSafe(instance, instanceSource, labelSource);

                    if (groundTruthLabel.Equals(positiveClassLabel) || groundTruthLabel.Equals(negativeClassLabel))
                    {
                        double positiveScore = predictionIterator.Current[positiveClassLabel];
                        double negativeScore = predictionIterator.Current[negativeClassLabel];
                        double score = positiveScore == 0 ? 0 : positiveScore / (positiveScore + negativeScore);

                        instanceScores.Add(instanceId, score);
                    }

                    instanceId++;
                }
            }

            return instanceScores;
        }

        #endregion
    }
}