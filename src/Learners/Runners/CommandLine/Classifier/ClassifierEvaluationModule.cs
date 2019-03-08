// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Globalization;
    using System.IO;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Learners.Mappings;

    /// <summary>
    /// A command-line module to evaluate the label predictions of classifiers.
    /// </summary>
    internal class ClassifierEvaluationModule : CommandLineModule
    {
        /// <summary>
        /// Runs the module.
        /// </summary>
        /// <param name="args">The command line arguments for the module.</param>
        /// <param name="usagePrefix">The prefix to print before the usage string.</param>
        /// <returns>True if the run was successful, false otherwise.</returns>
        public override bool Run(string[] args, string usagePrefix)
        {
            string groundTruthFileName = string.Empty;
            string predictionsFileName = string.Empty;
            string reportFileName = string.Empty;
            string calibrationCurveFileName = string.Empty;
            string rocCurveFileName = string.Empty;
            string precisionRecallCurveFileName = string.Empty;
            string positiveClassLabel = string.Empty;

            var parser = new CommandLineParser();
            parser.RegisterParameterHandler("--ground-truth", "FILE", "File with ground truth labels", v => groundTruthFileName = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--predictions", "FILE", "File with label predictions", v => predictionsFileName = v, CommandLineParameterType.Required);
            parser.RegisterParameterHandler("--report", "FILE", "File to store the evaluation report", v => reportFileName = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--calibration-curve", "FILE", "File to store the empirical calibration curve", v => calibrationCurveFileName = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--roc-curve", "FILE", "File to store the receiver operating characteristic curve", v => rocCurveFileName = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--precision-recall-curve", "FILE", "File to store the precision-recall curve", v => precisionRecallCurveFileName = v, CommandLineParameterType.Optional);
            parser.RegisterParameterHandler("--positive-class", "STRING", "Label of the positive class to use in curves", v => positiveClassLabel = v, CommandLineParameterType.Optional); 
            if (!parser.TryParse(args, usagePrefix))
            {
                return false;
            }

            // Read ground truth
            var groundTruth = ClassifierPersistenceUtils.LoadLabeledFeatureValues(groundTruthFileName);

            // Read predictions using ground truth label dictionary
            var predictions = ClassifierPersistenceUtils.LoadLabelDistributions(predictionsFileName, groundTruth.First().LabelDistribution.LabelSet);

            // Check that there are at least two distinct class labels
            if (predictions.First().LabelSet.Count < 2)
            {
                throw new InvalidFileFormatException("Ground truth and predictions must contain at least two distinct class labels.");
            }

            // Distill distributions and point estimates
            var predictiveDistributions = predictions.Select(i => i.ToDictionary()).ToList();
            var predictivePointEstimates = predictions.Select(i => i.GetMode()).ToList();

            // Create evaluator 
            var evaluatorMapping = Mappings.Classifier.ForEvaluation();
            var evaluator = new ClassifierEvaluator<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string>(evaluatorMapping);

            // Write evaluation report
            if (!string.IsNullOrEmpty(reportFileName))
            {
                using (var writer = new StreamWriter(reportFileName))
                {
                    this.WriteReportHeader(writer, groundTruthFileName, predictionsFileName);
                    this.WriteReport(writer, evaluator, groundTruth, predictiveDistributions, predictivePointEstimates);
                }
            }

            // Compute and write the empirical probability calibration curve
            positiveClassLabel = this.CheckPositiveClassLabel(groundTruth, positiveClassLabel);
            if (!string.IsNullOrEmpty(calibrationCurveFileName))
            {
                this.WriteCalibrationCurve(calibrationCurveFileName, evaluator, groundTruth, predictiveDistributions, positiveClassLabel);
            }

            // Compute and write the precision-recall curve
            if (!string.IsNullOrEmpty(precisionRecallCurveFileName))
            {
                this.WritePrecisionRecallCurve(precisionRecallCurveFileName, evaluator, groundTruth, predictiveDistributions, positiveClassLabel);
            }

            // Compute and write the receiver operating characteristic curve
            if (!string.IsNullOrEmpty(rocCurveFileName))
            {
                this.WriteRocCurve(rocCurveFileName, evaluator, groundTruth, predictiveDistributions, positiveClassLabel);
            }

            return true;
        }

        #region Helper methods

        /// <summary>
        /// Writes the evaluation results to a file with the specified name.
        /// </summary>
        /// <param name="writer">The name of the file to write the report to.</param>
        /// <param name="evaluator">The classifier evaluator.</param>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="predictiveDistributions">The predictive distributions.</param>
        /// <param name="predictedLabels">The predicted labels.</param>
        private void WriteReport(
            StreamWriter writer,
            ClassifierEvaluator<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string> evaluator,
            IList<LabeledFeatureValues> groundTruth,
            ICollection<IDictionary<string, double>> predictiveDistributions,
            IEnumerable<string> predictedLabels)
        {
            // Compute confusion matrix
            var confusionMatrix = evaluator.ConfusionMatrix(groundTruth, predictedLabels);
            
            // Compute mean negative log probability
            double meanNegativeLogProbability = 
                evaluator.Evaluate(groundTruth, predictiveDistributions, Metrics.NegativeLogProbability) / predictiveDistributions.Count;

            // Compute M-measure (averaged pairwise AUC)
            IDictionary<string, IDictionary<string, double>> aucMatrix;
            double auc = evaluator.AreaUnderRocCurve(groundTruth, predictiveDistributions, out aucMatrix);

            // Compute per-label AUC as well as micro- and macro-averaged AUC
            double microAuc;
            double macroAuc;
            int macroAucClassLabelCount;
            var labelAuc = this.ComputeLabelAuc(
                confusionMatrix,
                evaluator,
                groundTruth,
                predictiveDistributions,
                out microAuc,
                out macroAuc,
                out macroAucClassLabelCount);

            // Instance-averaged performance
            this.WriteInstanceAveragedPerformance(writer, confusionMatrix, meanNegativeLogProbability, microAuc);

            // Class-averaged performance
            this.WriteClassAveragedPerformance(writer, confusionMatrix, auc, macroAuc, macroAucClassLabelCount);

            // Performance on individual classes
            this.WriteIndividualClassPerformance(writer, confusionMatrix, labelAuc);

            // Confusion matrix
            this.WriteConfusionMatrix(writer, confusionMatrix);

            // Pairwise AUC
            this.WriteAucMatrix(writer, aucMatrix);
        }

        /// <summary>
        /// Computes all per-label AUCs as well as the micro- and macro-averaged AUCs.
        /// </summary>
        /// <param name="confusionMatrix">The confusion matrix.</param>
        /// <param name="evaluator">The classifier evaluator.</param>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="predictiveDistributions">The predictive distributions.</param>
        /// <param name="microAuc">The micro-averaged area under the receiver operating characteristic curve.</param>
        /// <param name="macroAuc">The macro-averaged area under the receiver operating characteristic curve.</param>
        /// <param name="macroAucClassLabelCount">The number of class labels for which the AUC if defined.</param>
        /// <returns>The area under the receiver operating characteristic curve for each class label.</returns>
        private IDictionary<string, double> ComputeLabelAuc(
            ConfusionMatrix<string> confusionMatrix,
            ClassifierEvaluator<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string> evaluator,
            IList<LabeledFeatureValues> groundTruth,
            ICollection<IDictionary<string, double>> predictiveDistributions,
            out double microAuc,
            out double macroAuc,
            out int macroAucClassLabelCount)
        {
            int instanceCount = predictiveDistributions.Count;
            var classLabels = confusionMatrix.ClassLabelSet.Elements.ToArray();
            int classLabelCount = classLabels.Length;
            var labelAuc = new Dictionary<string, double>();

            // Compute per-label AUC
            macroAucClassLabelCount = classLabelCount;
            foreach (var classLabel in classLabels)
            {
                // One versus rest
                double auc;
                try
                {
                    auc = evaluator.AreaUnderRocCurve(classLabel, groundTruth, predictiveDistributions);
                }
                catch (ArgumentException)
                {
                    auc = double.NaN;
                    macroAucClassLabelCount--;
                }

                labelAuc.Add(classLabel, auc);
            }

            // Compute micro- and macro-averaged AUC
            microAuc = 0;
            macroAuc = 0;
            foreach (var label in classLabels)
            {
                if (double.IsNaN(labelAuc[label]))
                {
                    continue;
                }

                microAuc += confusionMatrix.TrueLabelCount(label) * labelAuc[label] / instanceCount;
                macroAuc += labelAuc[label] / macroAucClassLabelCount;
            }

            return labelAuc;
        }

        /// <summary>
        /// Writes the header of the evaluation report to a specified stream writer.
        /// </summary>
        /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
        /// <param name="groundTruthFileName">The name of the file containing the ground truth.</param>
        /// <param name="predictionsFileName">The name of the file containing the predictions.</param>
        private void WriteReportHeader(StreamWriter writer, string groundTruthFileName, string predictionsFileName)
        {
            writer.WriteLine();
            writer.WriteLine(" Classifier evaluation report ");
            writer.WriteLine("******************************");
            writer.WriteLine();
            writer.WriteLine("           Date:      {0}", DateTime.Now);
            writer.WriteLine("   Ground truth:      {0}", groundTruthFileName);
            writer.WriteLine("    Predictions:      {0}", predictionsFileName);
        }

        /// <summary>
        /// Writes instance-averaged performance results to a specified stream writer.
        /// </summary>
        /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
        /// <param name="confusionMatrix">The confusion matrix.</param>
        /// <param name="negativeLogProbability">The negative log-probability.</param>
        /// <param name="microAuc">The micro-averaged AUC.</param>
        private void WriteInstanceAveragedPerformance(
            StreamWriter writer,
            ConfusionMatrix<string> confusionMatrix, 
            double negativeLogProbability,
            double microAuc)
        {
            long instanceCount = 0;
            long correctInstanceCount = 0;
            foreach (var classLabelIndex in confusionMatrix.ClassLabelSet.Indexes)
            {
                string classLabel = confusionMatrix.ClassLabelSet.GetElementByIndex(classLabelIndex);
                instanceCount += confusionMatrix.TrueLabelCount(classLabel);
                correctInstanceCount += confusionMatrix[classLabel, classLabel];
            }

            writer.WriteLine();
            writer.WriteLine(" Instance-averaged performance (micro-averages)");
            writer.WriteLine("================================================");
            writer.WriteLine();
            writer.WriteLine("                Precision = {0,10:0.0000}", confusionMatrix.MicroPrecision);
            writer.WriteLine("                   Recall = {0,10:0.0000}", confusionMatrix.MicroRecall);
            writer.WriteLine("                       F1 = {0,10:0.0000}", confusionMatrix.MicroF1);
            writer.WriteLine();
            writer.WriteLine("                 #Correct = {0,10}", correctInstanceCount);
            writer.WriteLine("                   #Total = {0,10}", instanceCount);
            writer.WriteLine("                 Accuracy = {0,10:0.0000}", confusionMatrix.MicroAccuracy);
            writer.WriteLine("                    Error = {0,10:0.0000}", 1 - confusionMatrix.MicroAccuracy);

            writer.WriteLine();
            writer.WriteLine("                      AUC = {0,10:0.0000}", microAuc);

            writer.WriteLine();
            writer.WriteLine("                 Log-loss = {0,10:0.0000}", negativeLogProbability);
        }

        /// <summary>
        /// Writes class-averaged performance results to a specified stream writer.
        /// </summary>
        /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
        /// <param name="confusionMatrix">The confusion matrix.</param>
        /// <param name="auc">The AUC.</param>
        /// <param name="macroAuc">The macro-averaged AUC.</param>
        /// <param name="macroAucClassLabelCount">The number of distinct class labels used to compute macro-averaged AUC.</param>
        private void WriteClassAveragedPerformance(
            StreamWriter writer,
            ConfusionMatrix<string> confusionMatrix,
            double auc,
            double macroAuc,
            int macroAucClassLabelCount)
        {
            int classLabelCount = confusionMatrix.ClassLabelSet.Count;

            writer.WriteLine();
            writer.WriteLine(" Class-averaged performance (macro-averages)");
            writer.WriteLine("=============================================");
            writer.WriteLine();
            if (confusionMatrix.MacroPrecisionClassLabelCount < classLabelCount)
            {
                writer.WriteLine(
                    "                Precision = {0,10:0.0000}     {1,10}",
                    confusionMatrix.MacroPrecision,
                    "[only " + confusionMatrix.MacroPrecisionClassLabelCount + "/" + classLabelCount + " classes defined]");
            }
            else
            {
                writer.WriteLine("                Precision = {0,10:0.0000}", confusionMatrix.MacroPrecision);
            }

            if (confusionMatrix.MacroRecallClassLabelCount < classLabelCount)
            {
                writer.WriteLine(
                    "                   Recall = {0,10:0.0000}     {1,10}",
                    confusionMatrix.MacroRecall,
                    "[only " + confusionMatrix.MacroRecallClassLabelCount + "/" + classLabelCount + " classes defined]");
            }
            else
            {
                writer.WriteLine("                   Recall = {0,10:0.0000}", confusionMatrix.MacroRecall);
            }

            if (confusionMatrix.MacroF1ClassLabelCount < classLabelCount)
            {
                writer.WriteLine(
                    "                       F1 = {0,10:0.0000}     {1,10}",
                    confusionMatrix.MacroF1,
                    "[only " + confusionMatrix.MacroF1ClassLabelCount + "/" + classLabelCount + " classes defined]");
            }
            else
            {
                writer.WriteLine("                       F1 = {0,10:0.0000}", confusionMatrix.MacroF1);
            }

            writer.WriteLine();
            if (confusionMatrix.MacroF1ClassLabelCount < classLabelCount)
            {
                writer.WriteLine(
                    "                 Accuracy = {0,10:0.0000}     {1,10}",
                    confusionMatrix.MacroAccuracy,
                    "[only " + confusionMatrix.MacroAccuracyClassLabelCount + "/" + classLabelCount + " classes defined]");
                writer.WriteLine(
                    "                    Error = {0,10:0.0000}     {1,10}",
                    1 - confusionMatrix.MacroAccuracy,
                    "[only " + confusionMatrix.MacroAccuracyClassLabelCount + "/" + classLabelCount + " classes defined]");
            }
            else
            {
                writer.WriteLine("                 Accuracy = {0,10:0.0000}", confusionMatrix.MacroAccuracy);
                writer.WriteLine("                    Error = {0,10:0.0000}", 1 - confusionMatrix.MacroAccuracy);
            }

            writer.WriteLine();
            if (macroAucClassLabelCount < classLabelCount)
            {
                writer.WriteLine(
                    "                      AUC = {0,10:0.0000}     {1,10}",
                    macroAuc,
                    "[only " + macroAucClassLabelCount + "/" + classLabelCount + " classes defined]");
            }
            else
            {
                writer.WriteLine("                      AUC = {0,10:0.0000}", macroAuc);
            }

            writer.WriteLine();
            writer.WriteLine("         M (pairwise AUC) = {0,10:0.0000}", auc);
        }

        /// <summary>
        /// Writes performance results for individual classes to a specified stream writer.
        /// </summary>
        /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
        /// <param name="confusionMatrix">The confusion matrix.</param>
        /// <param name="auc">The per-class AUC.</param>
        private void WriteIndividualClassPerformance(
            StreamWriter writer,
            ConfusionMatrix<string> confusionMatrix,
            IDictionary<string, double> auc)
        {
            writer.WriteLine();
            writer.WriteLine(" Performance on individual classes");
            writer.WriteLine("===================================");
            writer.WriteLine();
            writer.WriteLine(
                " {0,5} {1,15} {2,10} {3,11} {4,9} {5,10} {6,10} {7,10} {8,10}",
                "Index",
                "Label",
                "#Truth",
                "#Predicted",
                "#Correct",
                "Precision",
                "Recall",
                "F1",
                "AUC");

            writer.WriteLine("----------------------------------------------------------------------------------------------------");

            foreach (var classLabelIndex in confusionMatrix.ClassLabelSet.Indexes)
            {
                string classLabel = confusionMatrix.ClassLabelSet.GetElementByIndex(classLabelIndex);

                writer.WriteLine(
                    " {0,5} {1,15} {2,10} {3,11} {4,9} {5,10:0.0000} {6,10:0.0000} {7,10:0.0000} {8,10:0.0000}",
                    classLabelIndex + 1,
                    classLabel,
                    confusionMatrix.TrueLabelCount(classLabel),
                    confusionMatrix.PredictedLabelCount(classLabel),
                    confusionMatrix[classLabel, classLabel],
                    confusionMatrix.Precision(classLabel),
                    confusionMatrix.Recall(classLabel),
                    confusionMatrix.F1(classLabel),
                    auc[classLabel]);
            }
        }

        /// <summary>
        /// Writes the confusion matrix to a specified stream writer.
        /// </summary>
        /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
        /// <param name="confusionMatrix">The confusion matrix.</param>
        private void WriteConfusionMatrix(StreamWriter writer, ConfusionMatrix<string> confusionMatrix)
        {
            writer.WriteLine();
            writer.WriteLine(" Confusion matrix");
            writer.WriteLine("==================");
            writer.WriteLine();
            writer.WriteLine(confusionMatrix);
        }

        /// <summary>
        /// Writes the matrix of pairwise AUC metrics to a specified stream writer.
        /// </summary>
        /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
        /// <param name="aucMatrix">The matrix containing the pairwise AUC metrics.</param>
        private void WriteAucMatrix(StreamWriter writer, IDictionary<string, IDictionary<string, double>> aucMatrix)
        {
            writer.WriteLine();
            writer.WriteLine(" Pairwise AUC matrix");
            writer.WriteLine("=====================");
            writer.WriteLine();

            const int MaxLabelWidth = 20;
            const int MaxValueWidth = 6;

            // Widths of the columns
            string[] labels = aucMatrix.Keys.ToArray();
            int classLabelCount = aucMatrix.Count;
            var columnWidths = new int[classLabelCount + 1];

            // For each column of the confusion matrix...
            for (int c = 0; c < classLabelCount; c++)
            {
                // ...find the longest string among counts and label
                int labelWidth = Math.Min(labels[c].Length, MaxLabelWidth);

                columnWidths[c + 1] = Math.Max(MaxValueWidth, labelWidth);

                if (labelWidth > columnWidths[0])
                {
                    columnWidths[0] = labelWidth;
                }
            }

            // Print title row
            string format = string.Format("{{0,{0}}} \\ Prediction ->", columnWidths[0]);
            writer.WriteLine(format, "Truth");

            // Print column labels
            this.WriteLabel(writer, string.Empty, columnWidths[0]);
            for (int c = 0; c < classLabelCount; c++)
            {
                this.WriteLabel(writer, labels[c], columnWidths[c + 1]);
            }

            writer.WriteLine();

            // For each row (true labels) in confusion matrix...
            for (int r = 0; r < classLabelCount; r++)
            {
                // Print row label
                this.WriteLabel(writer, labels[r], columnWidths[0]);

                // For each column (predicted labels) in the confusion matrix...
                for (int c = 0; c < classLabelCount; c++)
                {
                    // Print count
                    this.WriteAucValue(writer, labels[r].Equals(labels[c]) ? -1 : aucMatrix[labels[r]][labels[c]], columnWidths[c + 1]);
                }

                writer.WriteLine();
            }
        }

        /// <summary>
        /// Writes the probability calibration plot to the file with the specified name.
        /// </summary>
        /// <param name="fileName">The name of the file to write the calibration plot to.</param>
        /// <param name="evaluator">The classifier evaluator.</param>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="predictiveDistributions">The predictive distributions.</param>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        private void WriteCalibrationCurve(
            string fileName,
            ClassifierEvaluator<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string> evaluator,
            IList<LabeledFeatureValues> groundTruth,
            IList<IDictionary<string, double>> predictiveDistributions,
            string positiveClassLabel)
        {
            Debug.Assert(predictiveDistributions != null, "The predictive distributions must not be null.");
            Debug.Assert(predictiveDistributions.Count > 0, "The predictive distributions must not be empty.");
            Debug.Assert(positiveClassLabel != null, "The label of the positive class must not be null.");

            var calibrationCurve = evaluator.CalibrationCurve(positiveClassLabel, groundTruth, predictiveDistributions);
            double calibrationError = calibrationCurve.Select(i => Metrics.AbsoluteError(i.EmpiricalProbability, i.PredictedProbability)).Average();

            using (var writer = new StreamWriter(fileName))
            {
                writer.WriteLine("# Empirical probability calibration plot");
                writer.WriteLine("#");
                writer.WriteLine("# Class '" + positiveClassLabel + "'     (versus the rest)");
                writer.WriteLine("# Calibration error = {0}     (mean absolute error)", calibrationError);
                writer.WriteLine("#");
                writer.WriteLine("# Predicted probability, empirical probability");
                foreach (var point in calibrationCurve)
                {
                    writer.WriteLine(point);
                }
            }
        }

        /// <summary>
        /// Writes the precision-recall curve to the file with the specified name.
        /// </summary>
        /// <param name="fileName">The name of the file to write the precision-recall curve to.</param>
        /// <param name="evaluator">The classifier evaluator.</param>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="predictiveDistributions">The predictive distributions.</param>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        private void WritePrecisionRecallCurve(
            string fileName,
            ClassifierEvaluator<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string> evaluator,
            IList<LabeledFeatureValues> groundTruth,
            IList<IDictionary<string, double>> predictiveDistributions,
            string positiveClassLabel)
        {
            Debug.Assert(predictiveDistributions != null, "The predictive distributions must not be null.");
            Debug.Assert(predictiveDistributions.Count > 0, "The predictive distributions must not be empty.");
            Debug.Assert(positiveClassLabel != null, "The label of the positive class must not be null.");

            var precisionRecallCurve = evaluator.PrecisionRecallCurve(positiveClassLabel, groundTruth, predictiveDistributions);
            using (var writer = new StreamWriter(fileName))
            {
                writer.WriteLine("# Precision-recall curve");
                writer.WriteLine("#");
                writer.WriteLine("# Class '" + positiveClassLabel + "'     (versus the rest)");
                writer.WriteLine("#");
                writer.WriteLine("# precision (P), Recall (R)");
                foreach (var point in precisionRecallCurve)
                {
                    writer.WriteLine(point);
                }
            }
        }

        /// <summary>
        /// Writes the receiver operating characteristic curve to the file with the specified name.
        /// </summary>
        /// <param name="fileName">The name of the file to write the receiver operating characteristic curve to.</param>
        /// <param name="evaluator">The classifier evaluator.</param>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="predictiveDistributions">The predictive distributions.</param>
        /// <param name="positiveClassLabel">The label of the positive class.</param>
        private void WriteRocCurve(
            string fileName,
            ClassifierEvaluator<IList<LabeledFeatureValues>, LabeledFeatureValues, IList<LabelDistribution>, string> evaluator,
            IList<LabeledFeatureValues> groundTruth,
            IList<IDictionary<string, double>> predictiveDistributions,
            string positiveClassLabel)
        {
            Debug.Assert(predictiveDistributions != null, "The predictive distributions must not be null.");
            Debug.Assert(predictiveDistributions.Count > 0, "The predictive distributions must not be empty.");
            Debug.Assert(positiveClassLabel != null, "The label of the positive class must not be null.");

            var rocCurve = evaluator.ReceiverOperatingCharacteristicCurve(positiveClassLabel, groundTruth, predictiveDistributions);

            using (var writer = new StreamWriter(fileName))
            {
                writer.WriteLine("# Receiver operating characteristic (ROC) curve");
                writer.WriteLine("#");
                writer.WriteLine("# Class '" + positiveClassLabel + "'     (versus the rest)");
                writer.WriteLine("#");
                writer.WriteLine("# False positive rate (FPR), True positive rate (TPR)");
                foreach (var point in rocCurve)
                {
                    writer.WriteLine(point);
                }
            }
        }

        /// <summary>
        /// Writes a count to a specified stream writer.
        /// </summary>
        /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
        /// <param name="auc">The count.</param>
        /// <param name="width">The width in characters used to print the count.</param>
        private void WriteAucValue(StreamWriter writer, double auc, int width)
        {
            string paddedCount;
            if (auc > 0)
            {
                paddedCount = auc.ToString(CultureInfo.InvariantCulture);
            }
            else
            {
                if (auc < 0)
                {
                    paddedCount = ".";
                }
                else
                {
                    paddedCount = double.IsNaN(auc) ? "NaN" : "0";
                }                
            }

            paddedCount = paddedCount.Length > width ? paddedCount.Substring(0, width) : paddedCount;
            paddedCount = paddedCount.PadLeft(width + 2);
            writer.Write(paddedCount);
        }

        /// <summary>
        /// Writes a label to a specified stream writer.
        /// </summary>
        /// <param name="writer">The <see cref="StreamWriter"/> to write to.</param>
        /// <param name="label">The label.</param>
        /// <param name="width">The width in characters used to print the label.</param>
        private void WriteLabel(StreamWriter writer, string label, int width)
        {
            string paddedLabel = label.Length > width ? label.Substring(0, width) : label;
            paddedLabel = paddedLabel.PadLeft(width + 2);
            writer.Write(paddedLabel);
        }

        /// <summary>
        /// Checks the positive class label.
        /// </summary>
        /// <param name="groundTruth">The ground truth.</param>
        /// <param name="positiveClassLabel">An optional positive class label provided by the user. Defaults to the first class label.</param>
        /// <returns>The actually used positive class label.</returns>
        private string CheckPositiveClassLabel(IList<LabeledFeatureValues> groundTruth, string positiveClassLabel = null)
        {
            Debug.Assert(groundTruth != null, "The ground truth labels must not be null.");
            Debug.Assert(groundTruth.Count > 0, "There must be at least one ground truth label.");

            if (string.IsNullOrEmpty(positiveClassLabel))
            {
                positiveClassLabel = groundTruth[0].LabelDistribution.LabelSet.Elements.First();
            }
            else
            {
                if (!groundTruth.First().LabelDistribution.LabelSet.Contains(positiveClassLabel))
                {
                    throw new ArgumentException(
                        "The label '" + positiveClassLabel + "' of the positive class is not a valid class label.");
                }
            }

            return positiveClassLabel;
        }

        #endregion
    }
}
