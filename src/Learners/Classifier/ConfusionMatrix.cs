// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.Text;

    using Microsoft.ML.Probabilistic.Collections;

    /// <summary>
    /// Implements a confusion matrix.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    [Serializable]
    public class ConfusionMatrix<TLabel>
    {
        #region Fields and constructors

        /// <summary>
        /// The confusion matrix over class label identifiers.
        /// </summary>
        private readonly long[,] confusionMatrix;

        #region Evaluation metrics

        /// <summary>
        /// True if the evaluation metrics are up-to-date and false otherwise
        /// </summary>
        private bool isMetricsUpdated;

        /// <summary>
        /// The true label counts.
        /// </summary>
        private long[] trueLabelCounts;

        /// <summary>
        /// The predicted label counts.
        /// </summary>
        private long[] predictedLabelCounts;

        /// <summary>
        /// The precision for all labels.
        /// </summary>
        private double[] labelPrecision;

        /// <summary>
        /// The micro-averaged precision.
        /// </summary>
        private double microPrecision;

        /// <summary>
        /// The macro-averaged precision.
        /// </summary>
        private double macroPrecision;

        /// <summary>
        /// The number of distinct class labels used to compute the macro-averaged precision.
        /// </summary>
        private int macroPrecisionClassLabelCount;

        /// <summary>
        /// The recall for all labels.
        /// </summary>
        private double[] labelRecall;

        /// <summary>
        /// The micro-averaged recall.
        /// </summary>
        private double microRecall;

        /// <summary>
        /// The macro-averaged recall.
        /// </summary>
        private double macroRecall;

        /// <summary>
        /// The number of distinct class labels used to compute the macro-averaged recall.
        /// </summary>
        private int macroRecallClassLabelCount;

        /// <summary>
        /// The F1-measure for all labels.
        /// </summary>
        private double[] labelF1;

        /// <summary>
        /// The micro-averaged F1-measure.
        /// </summary>
        private double microF1;

        /// <summary>
        /// The macro-averaged F1-measure.
        /// </summary>
        private double macroF1;

        /// <summary>
        /// The number of class labels available to compute the macro-averaged F1-measure.
        /// </summary>
        private int macroF1ClassLabelCount;

        #endregion

        /// <summary>
        /// Initializes a new instance of the <see cref="ConfusionMatrix{TLabel}"/> class.
        /// </summary>
        /// <param name="classLabels">The class labels.</param>
        public ConfusionMatrix(IEnumerable<TLabel> classLabels)
        {
            this.ClassLabelSet = new IndexedSet<TLabel>(classLabels);

            int classLabelCount = this.ClassLabelSet.Count;
            this.confusionMatrix = new long[classLabelCount, classLabelCount];

            this.isMetricsUpdated = false;
        }

        #endregion

        #region Properties for averaged evaluation metrics

        /// <summary>
        /// Gets the bidirectional dictionary mapping class labels to class label indexes.
        /// </summary>
        public IndexedSet<TLabel> ClassLabelSet { get; private set; }

        /// <summary>
        /// Gets the micro-averaged precision.
        /// </summary>
        public double MicroPrecision
        {
            get
            {
                this.UpdateMetrics();
                return this.microPrecision;
            }
        }

        /// <summary>
        /// Gets the macro-averaged precision.
        /// </summary>
        public double MacroPrecision
        {
            get
            {
                this.UpdateMetrics();
                return this.macroPrecision;
            }
        }

        /// <summary>
        /// Gets the number of distinct class labels used to compute the macro-averaged precision.
        /// </summary>
        public int MacroPrecisionClassLabelCount
        {
            get
            {
                this.UpdateMetrics();
                return this.macroPrecisionClassLabelCount;
            }
        }

        /// <summary>
        /// Gets the micro-averaged recall.
        /// </summary>
        public double MicroRecall
        {
            get
            {
                this.UpdateMetrics();
                return this.microRecall;
            }
        }

        /// <summary>
        /// Gets the macro-averaged recall.
        /// </summary>
        public double MacroRecall
        {
            get
            {
                this.UpdateMetrics();
                return this.macroRecall;
            }
        }

        /// <summary>
        /// Gets the number of distinct class labels used to compute the macro-averaged recall.
        /// </summary>
        public int MacroRecallClassLabelCount
        {
            get
            {
                this.UpdateMetrics();
                return this.macroRecallClassLabelCount;
            }
        }

        /// <summary>
        /// Gets the micro-averaged F1-measure.
        /// </summary>
        public double MicroF1
        {
            get
            {
                this.UpdateMetrics();
                return this.microF1;
            }
        }

        /// <summary>
        /// Gets the macro-averaged F1-measure.
        /// </summary>
        public double MacroF1
        {
            get
            {
                this.UpdateMetrics();
                return this.macroF1;
            }
        }

        /// <summary>
        /// Gets the number of class labels available to compute the macro-averaged F1-measure.
        /// </summary>
        public int MacroF1ClassLabelCount
        {
            get
            {
                this.UpdateMetrics();
                return this.macroF1ClassLabelCount;
            }
        }

        /// <summary>
        /// Gets the micro-averaged accuracy.
        /// </summary>
        public double MicroAccuracy
        {
            get
            {
                return this.MicroRecall;
            }
        }

        /// <summary>
        /// Gets the macro-averaged accuracy.
        /// </summary>
        public double MacroAccuracy
        {
            get
            {
                return this.MacroRecall;
            }
        }

        /// <summary>
        /// Gets the number of class labels available to compute the macro-averaged accuracy.
        /// </summary>
        public int MacroAccuracyClassLabelCount
        {
            get
            {
                return this.MacroRecallClassLabelCount;
            }
        }

        #endregion

        #region Indexer

        /// <summary>
        /// Gets or sets the number of instances for given true and predicted labels.
        /// </summary>
        /// <param name="trueLabel">The true label.</param>
        /// <param name="predictedLabel">The predicted label.</param>
        /// <exception cref="ArgumentOutOfRangeException">If a given label is unknown.</exception>
        /// <returns>The number of instances for the specified true and predicted labels.</returns>
        public long this[TLabel trueLabel, TLabel predictedLabel]
        {
            get
            {
                int trueLabelIdentifier, predictedLabelIdentifier;
                this.GetIndexesByLabels(trueLabel, predictedLabel, out trueLabelIdentifier, out predictedLabelIdentifier);
                return this.confusionMatrix[trueLabelIdentifier, predictedLabelIdentifier];
            }

            set
            {
                int trueLabelIdentifier, predictedLabelIdentifier;
                this.GetIndexesByLabels(trueLabel, predictedLabel, out trueLabelIdentifier, out predictedLabelIdentifier);
                this.confusionMatrix[trueLabelIdentifier, predictedLabelIdentifier] = value;
                this.isMetricsUpdated = false;
            }
        }

        #endregion

        #region Methods for label-specific evaluation metrics

        /// <summary>
        /// Gets the number of instances whose ground truth label equals the specified label.
        /// </summary>
        /// <param name="label">The label to get the number of instances for.</param>
        /// <returns>The number of instances with the specified ground truth label.</returns>
        public long TrueLabelCount(TLabel label)
        {
            this.UpdateMetrics();
            return this.trueLabelCounts[this.GetIndexByLabel(label)];
        }

        /// <summary>
        /// Gets the number of instances whose predicted label equals the specified label.
        /// </summary>
        /// <param name="label">The label to get the number of instances for.</param>
        /// <returns>The number of instances with the specified predicted label.</returns>
        public long PredictedLabelCount(TLabel label)
        {
            this.UpdateMetrics();
            return this.predictedLabelCounts[this.GetIndexByLabel(label)];
        }

        /// <summary>
        /// Gets the precision for a specified label.
        /// </summary>
        /// <param name="label">The label to get the precision for.</param>
        /// <returns>The precision of the specified label.</returns>
        public double Precision(TLabel label)
        {
            this.UpdateMetrics();
            return this.labelPrecision[this.GetIndexByLabel(label)];
        }

        /// <summary>
        /// Gets the recall for a specified label.
        /// </summary>
        /// <param name="label">The label to get the recall for.</param>
        /// <returns>The recall of the specified label.</returns>
        public double Recall(TLabel label)
        {
            this.UpdateMetrics();
            return this.labelRecall[this.GetIndexByLabel(label)];
        }

        /// <summary>
        /// Gets the F1-measure for a specified label.
        /// </summary>
        /// <param name="label">The label to get the F1-measure for.</param>
        /// <returns>The F1-measure of the specified label.</returns>
        public double F1(TLabel label)
        {
            this.UpdateMetrics();
            return this.labelF1[this.GetIndexByLabel(label)];
        }

        /// <summary>
        /// Gets the accuracy for a specified label.
        /// </summary>
        /// <param name="label">The label to get the accuracy for.</param>
        /// <returns>The accuracy of the specified label.</returns>
        public double Accuracy(TLabel label)
        {
            return this.Recall(label);
        }

        #endregion

        #region General purpose methods

        /// <summary>
        /// Returns the confusion matrix as a string.
        /// </summary>
        /// <returns>The confusion matrix as a <see cref="string"/>.</returns>
        public override string ToString()
        {
            this.UpdateMetrics();
            return this.ConfusionMatrixToString(string.Empty);
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Appends a count to a specified string builder.
        /// </summary>
        /// <param name="builder">The <see cref="StringBuilder"/>.</param>
        /// <param name="count">The count.</param>
        /// <param name="width">The width in characters used to print the count.</param>
        private static void AppendCount(StringBuilder builder, long count, int width)
        {
            string paddedCount = (count > 0) ? count.ToString(CultureInfo.InvariantCulture) : "0";
            paddedCount = paddedCount.Length > width ? paddedCount.Substring(0, width) : paddedCount;
            paddedCount = paddedCount.PadLeft(width + 2);
            builder.Append(paddedCount);
        }

        /// <summary>
        /// Appends a label to a specified string builder.
        /// </summary>
        /// <param name="builder">The <see cref="StringBuilder"/>.</param>
        /// <param name="label">The label.</param>
        /// <param name="width">The width in characters used to print the label.</param>
        private static void AppendLabel(StringBuilder builder, string label, int width)
        {
            string paddedLabel = label.Length > width ? label.Substring(0, width) : label;
            paddedLabel = paddedLabel.PadLeft(width + 2);
            builder.Append(paddedLabel);
        }

        /// <summary>
        /// Computes a set of performance metrics from the confusion matrix.
        /// </summary>
        private void UpdateMetrics()
        {
            if (this.isMetricsUpdated)
            {
                return;
            }

            long instanceCount = 0;
            int classLabelCount = this.ClassLabelSet.Count;

            this.trueLabelCounts = new long[classLabelCount];
            this.predictedLabelCounts = new long[classLabelCount];

            for (int trueLabel = 0; trueLabel < classLabelCount; trueLabel++)
            {
                for (int predictedLabel = 0; predictedLabel < classLabelCount; predictedLabel++)
                {
                    long count = this.confusionMatrix[trueLabel, predictedLabel];
                    this.trueLabelCounts[trueLabel] += count;
                    this.predictedLabelCounts[predictedLabel] += count;
                    instanceCount += count;
                }
            }

            // Compute per-label precision, recall, and F1.
            var divisionByZeroPrecision = new bool[classLabelCount];
            var divisionByZeroRecall = new bool[classLabelCount];
            var divisionByZeroF1 = new bool[classLabelCount];

            this.macroPrecisionClassLabelCount = classLabelCount;
            this.macroRecallClassLabelCount = classLabelCount;
            this.macroF1ClassLabelCount = classLabelCount;

            this.labelPrecision = new double[classLabelCount];
            this.labelRecall = new double[classLabelCount];
            this.labelF1 = new double[classLabelCount];

            for (int label = 0; label < classLabelCount; label++)
            {
                double correct = this.confusionMatrix[label, label];

                divisionByZeroPrecision[label] = false;
                divisionByZeroRecall[label] = false;
                divisionByZeroF1[label] = false;

                // Precision
                if (this.predictedLabelCounts[label] == 0)
                {
                    divisionByZeroPrecision[label] = true;
                    this.macroPrecisionClassLabelCount--;
                    this.labelPrecision[label] = double.NaN;
                }
                else
                {
                    this.labelPrecision[label] = correct / this.predictedLabelCounts[label];
                }

                // Recall
                if (this.trueLabelCounts[label] == 0)
                {
                    this.labelRecall[label] = double.NaN;
                    divisionByZeroRecall[label] = true;
                    this.macroRecallClassLabelCount--;
                }
                else
                {
                    this.labelRecall[label] = correct / this.trueLabelCounts[label];
                }

                // F1-measure
                if (divisionByZeroPrecision[label] || divisionByZeroRecall[label])
                {
                    this.labelF1[label] = double.NaN;
                    divisionByZeroF1[label] = true;
                    this.macroF1ClassLabelCount--;
                }
                else if (this.labelPrecision[label] <= 0 && this.labelRecall[label] <= 0)
                {
                    // It makes sense to define F1 as 0 when both precision and recall are 0 (but defined!)
                    this.labelF1[label] = 0.0;
                }
                else
                {
                    this.labelF1[label] = (2 * this.labelPrecision[label] * this.labelRecall[label])
                            / (this.labelPrecision[label] + this.labelRecall[label]);
                }
            }

            // Compute micro-averages (instances have equal weight) and macro-averages (label classes have equal weight)
            this.microPrecision = 0;
            this.macroPrecision = 0;
            this.microRecall = 0;
            this.macroRecall = 0;
            this.microF1 = 0;
            this.macroF1 = 0;

            for (int label = 0; label < classLabelCount; label++)
            {
                if (!divisionByZeroPrecision[label])
                {
                    this.microPrecision += this.trueLabelCounts[label] * this.labelPrecision[label] / instanceCount;
                    this.macroPrecision += this.labelPrecision[label] / this.macroPrecisionClassLabelCount;
                }

                if (!divisionByZeroRecall[label])
                {
                    this.microRecall += this.trueLabelCounts[label] * this.labelRecall[label] / instanceCount;
                    this.macroRecall += this.labelRecall[label] / this.macroRecallClassLabelCount;
                }

                if (!divisionByZeroF1[label])
                {
                    this.microF1 += this.trueLabelCounts[label] * this.labelF1[label] / instanceCount;
                    this.macroF1 += this.labelF1[label] / this.macroF1ClassLabelCount;
                }
            }

            // All evaluation metrics now up-to-date
            this.isMetricsUpdated = true;
        }

        /// <summary>
        /// Gets the index for the given label.
        /// </summary>
        /// <param name="label">The label to get the index for.</param>
        /// <returns>The index for the given label.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if label is unknown.</exception>
        private int GetIndexByLabel(TLabel label)
        {
            if (label == null)
            {
                throw new ArgumentNullException(nameof(label));
            }

            int labelIndex;
            if (!this.ClassLabelSet.TryGetIndex(label, out labelIndex))
            {
                throw new ArgumentOutOfRangeException(nameof(label), "The class label '" + label + "' is unknown.");
            }

            return labelIndex;
        }

        /// <summary>
        /// Gets the label indexes for the given labels.
        /// </summary>
        /// <param name="trueLabel">The true label to get the index for.</param>
        /// <param name="predictedLabel">The predicted label to get the index for.</param>
        /// <param name="trueLabelIndex">The index of the true label.</param>
        /// <param name="predictedLabelIndex">The index of the predicted label.</param>
        private void GetIndexesByLabels(TLabel trueLabel, TLabel predictedLabel, out int trueLabelIndex, out int predictedLabelIndex)
        {
            if (trueLabel == null)
            {
                throw new ArgumentNullException(nameof(trueLabel));
            }

            if (predictedLabel == null)
            {
                throw new ArgumentNullException(nameof(predictedLabel));
            }

            if (!this.ClassLabelSet.TryGetIndex(trueLabel, out trueLabelIndex))
            {
                throw new ArgumentOutOfRangeException(nameof(trueLabel), "The class label '" + trueLabel + "' is unknown.");
            }

            if (!this.ClassLabelSet.TryGetIndex(predictedLabel, out predictedLabelIndex))
            {
                throw new ArgumentOutOfRangeException(nameof(predictedLabel), "The class label '" + predictedLabel + "' is unknown.");
            }
        }

        /// <summary>
        /// Returns the confusion matrix as a string.
        /// </summary>
        /// <param name="linePrefix">The prefix used on each line.</param>
        /// <returns>The confusion matrix as a <see cref="string"/>.</returns>
        private string ConfusionMatrixToString(string linePrefix)
        {
            const int MaxLabelWidth = 15;

            // Widths of the columns
            int classLabelCount = this.ClassLabelSet.Count;
            var columnWidths = new int[classLabelCount + 1];

            // For each column of the confusion matrix...
            for (int c = 0; c < classLabelCount; c++)
            {
                // ...find the longest string among counts and label
                int labelWidth = this.ClassLabelSet.GetElementByIndex(c).ToString().Length;

                columnWidths[c + 1] = labelWidth > MaxLabelWidth ? MaxLabelWidth : labelWidth;
                for (int r = 0; r < classLabelCount; r++)
                {
                    int countWidth = this.confusionMatrix[r, c].ToString(CultureInfo.InvariantCulture).Length;
                    if (countWidth > columnWidths[c + 1])
                    {
                        columnWidths[c + 1] = countWidth;
                    }
                }

                if (labelWidth > columnWidths[0])
                {
                    columnWidths[0] = labelWidth > MaxLabelWidth ? MaxLabelWidth : labelWidth;
                }
            }

            // Print title row
            var builder = new StringBuilder();
            builder.Append(linePrefix);
            string format = string.Format("{{0,{0}}} \\ Prediction ->", columnWidths[0]);
            builder.AppendLine(string.Format(format, "Truth"));

            // Print column labels
            builder.Append(linePrefix);
            AppendLabel(builder, string.Empty, columnWidths[0]);
            for (int c = 0; c < classLabelCount; c++)
            {
                AppendLabel(builder, this.ClassLabelSet.GetElementByIndex(c).ToString(), columnWidths[c + 1]);
            }

            builder.AppendLine();

            // For each row (true labels) in confusion matrix...
            for (int r = 0; r < classLabelCount; r++)
            {
                builder.Append(linePrefix);

                // Print row label
                AppendLabel(builder, this.ClassLabelSet.GetElementByIndex(r).ToString(), columnWidths[0]);

                // For each column (predicted labels) in the confusion matrix...
                for (int c = 0; c < classLabelCount; c++)
                {
                    // Print count
                    AppendCount(builder, this.confusionMatrix[r, c], columnWidths[c + 1]);
                }

                builder.AppendLine();
            }

            return builder.ToString();
        }

        #endregion
    }
}
