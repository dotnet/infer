// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CrowdsourcingWithWords
{
    /// <summary>
    /// Results class containing posteriors and predictions.
    /// </summary>
    public class Results
    {
        /// <summary>
        /// The posterior of the true label for each task.
        /// </summary>
        public Dictionary<string, Discrete> TrueLabel
        {
            get;
            protected set;
        }

        /// <summary>
        /// The predicted label for each task when doing simulations from the current
        /// model state. It avoids overwriting the true label posterior.
        /// </summary>
        public Dictionary<string, Discrete> LookAheadTrueLabel
        {
            get;
            protected set;
        }

        /// <summary>
        /// The posterior for the constraint that allows online learning for the true label variable.
        /// </summary>
        public Dictionary<string, Discrete> TrueLabelConstraint
        {
            get;
            protected set;
        }

        /// <summary>
        /// The predicted label for each task
        /// </summary>
        public Dictionary<string, int?> PredictedLabel
        {
            get;
            protected set;
        }

        /// <summary>
        /// The probabilities that generate the true label of all the tasks.
        /// </summary>
        public Dirichlet BackgroundLabelProb
        {
            get;
            protected set;
        }

        /// <summary>
        /// The posterior of the confusion matrix of each worker.
        /// </summary>
        public Dictionary<string, Dirichlet[]> WorkerConfusionMatrix
        {
            get;
            protected set;
        }

        /// <summary>
        /// The look-ahead posterior of the confusion matrix of each worker obtained after simulating
        /// a new label in look-ahead run mode.
        /// </summary>
        public Dictionary<string, Dirichlet[]> LookAheadWorkerConfusionMatrix
        {
            get;
            protected set;
        }

        /// <summary>
        /// The predictive probabilities of the labels produced by each worker.
        /// </summary>
        public Dictionary<string, Dictionary<string, Discrete>> WorkerPrediction
        {
            get;
            protected set;
        }

        /// <summary>
        /// The community membership probabilities of each worker.
        /// </summary>
        public Dictionary<string, Discrete> WorkerCommunity
        {
            get;
            protected set;
        }

        /// <summary>
        /// The confusion matrix of each community.
        /// </summary>
        public Dirichlet[][] CommunityConfusionMatrix
        {
            get;
            protected set;
        }

        /// <summary>
        /// The score matrix of each community.
        /// </summary>
        public VectorGaussian[][] CommunityScoreMatrix
        {
            get;
            protected set;
        }

        /// <summary>
        /// The posterior for the constraint that allows online learning for worker confusion matrices
        /// int the community model.
        /// </summary>
        public Dictionary<string, VectorGaussian[]> WorkerScoreMatrixConstraint
        {
            get;
            protected set;
        }

        /// <summary>
        /// The probabilities that generate the community memberships of all the workers.
        /// </summary>
        public Dirichlet CommunityProb
        {
            get;
            protected set;
        }

        /// <summary>
        /// The posterior for the constraint that allows online learning for community membership.
        /// int the community model.
        /// </summary>
        public Dictionary<string, Discrete> CommunityConstraint
        {
            get;
            protected set;
        }

        /// <summary>
        /// Model evidence.
        /// </summary>
        public Bernoulli ModelEvidence
        {
            get;
            protected set;
        }

        /// <summary>
        /// The data mapping.
        /// </summary>
        public DataMapping Mapping
        {
            get;
            set;
        }

        /// <summary>
        /// The full data mapping.
        /// </summary>
        public DataMapping FullMapping
        {
            get;
            set;
        }

        /// <summary>
        /// The gold labels of each task. The gold label type is nullable to
        /// support the (usual) situation without labels.
        /// </summary>
        public Dictionary<string, int?> GoldLabels
        {
            get;
            protected set;
        }

        /// <summary>
        /// The accuracy of the current true label predictions.
        /// </summary>
        public double Accuracy
        {
            get;
            private set;
        }

        /// <summary>
        /// The accuracy of the worker labels.
        /// </summary>
        public double WorkerLabelAccuracy
        {
            get;
            protected set;
        }

        /// <summary>
        /// The negative log probability density (NLPD) scores of the current true label predictions.
        /// </summary>
        public double NegativeLogProb
        {
            get;
            private set;
        }

        /// <summary>
        /// The average recall of the current true label predictions.
        /// </summary>
        public double AvgRecall
        {
            get;
            private set;
        }

        /// <summary>
        /// The confusion matrix of the predicted true labels against the gold labels
        /// The rows are the gold labels and the columns are the predicted labels.
        /// </summary>
        public double[,] ModelConfusionMatrix
        {
            get;
            private set;
        }

        /// <summary>
        /// The number of communities.
        /// </summary>
        public int CommunityCount
        {
            get;
            private set;
        }

        public ReceiverOperatingCharacteristic.ConfusionMatrix BinaryConfusionMatrix
        {
            get;
            private set;
        }

        public ReceiverOperatingCharacteristic RocCurve
        {
            get;
            private set;
        }

        public List<double> trueBinaryLabel;
        public List<double> probTrueBinaryLabel;

        public enum RunMode
        {
            ClearResults,
            BatchTraining,
            IncrementalExperiment,
            OnlineExperiment,
            LookAheadExperiment,
            LoadAndUseCommunityPriors,
            Prediction,
        };

        protected virtual void ClearResults()
        {
            BackgroundLabelProb = Dirichlet.Uniform(Mapping.LabelCount);
            WorkerConfusionMatrix = new Dictionary<string, Dirichlet[]>();
            WorkerPrediction = new Dictionary<string, Dictionary<String, Discrete>>();
            WorkerCommunity = new Dictionary<string, Discrete>();
            TrueLabel = new Dictionary<string, Discrete>();
            TrueLabelConstraint = new Dictionary<string, Discrete>();
            CommunityConfusionMatrix = null;
            WorkerScoreMatrixConstraint = new Dictionary<string, VectorGaussian[]>();
            CommunityProb = null;
            CommunityScoreMatrix = null;
            CommunityConstraint = new Dictionary<string, Discrete>();
            LookAheadTrueLabel = new Dictionary<string, Discrete>();
            LookAheadWorkerConfusionMatrix = new Dictionary<string, Dirichlet[]>();
            ModelEvidence = new Bernoulli(0.5);
            PredictedLabel = new Dictionary<string, int?>();
        }

        protected virtual void UpdateResults(BCCPosteriors posteriors, RunMode mode)
        {
            if (mode == RunMode.LookAheadExperiment)
            {
                if (posteriors.TrueLabel != null)
                {
                    for (int t = 0; t < posteriors.TrueLabel.Length; t++)
                    {
                        LookAheadTrueLabel[Mapping.TaskIndexToId[t]] = posteriors.TrueLabel[t];
                    }
                }

                if (posteriors.WorkerConfusionMatrix != null)
                {
                    for (int w = 0; w < posteriors.WorkerConfusionMatrix.Length; w++)
                    {
                        LookAheadWorkerConfusionMatrix[Mapping.WorkerIndexToId[w]] = posteriors.WorkerConfusionMatrix[w];
                    }
                }

            }
            else if (mode == RunMode.Prediction)
            {
                if (posteriors.WorkerConfusionMatrix != null)
                {
                    for (int w = 0; w < posteriors.WorkerConfusionMatrix.Length; w++)
                    {
                        WorkerPrediction[Mapping.WorkerIndexToId[w]] = new Dictionary<string, Discrete>();
                        for (int tw = 0; tw < posteriors.WorkerPrediction[w].Length; tw++)
                        {
                            WorkerPrediction[Mapping.WorkerIndexToId[w]][Mapping.TaskIndexToId[tw]] = posteriors.WorkerPrediction[w][tw];
                        }
                    }
                }
            }
            else
            {
                // Update results for BCC
                BackgroundLabelProb = posteriors.BackgroundLabelProb;
                if (posteriors.WorkerConfusionMatrix != null)
                {
                    for (int w = 0; w < posteriors.WorkerConfusionMatrix.Length; w++)
                    {
                        WorkerConfusionMatrix[Mapping.WorkerIndexToId[w]] = posteriors.WorkerConfusionMatrix[w];
                    }
                }
                if (posteriors.TrueLabel != null)
                {
                    for (int t = 0; t < posteriors.TrueLabel.Length; t++)
                    {
                        TrueLabel[Mapping.TaskIndexToId[t]] = posteriors.TrueLabel[t];
                    }
                }
                if (posteriors.TrueLabelConstraint != null)
                {
                    for (int t = 0; t < posteriors.TrueLabelConstraint.Length; t++)
                    {
                        TrueLabelConstraint[Mapping.TaskIndexToId[t]] = posteriors.TrueLabelConstraint[t];
                    }
                }

                this.ModelEvidence = posteriors.Evidence;
            }
        }

        /// <summary>
        /// Updates the accuracy using the current results.
        /// </summary>
        protected virtual void UpdateAccuracy()
        {
            double nlpdThreshold = -Math.Log(0.001);
            int labelCount = TrueLabel.First(kvp => kvp.Value != null).Value.Dimension;
            var confusionMatrix = Util.ArrayInit(labelCount, labelCount, (i, j) => 0.0);
            int correct = 0;
            double logProb = 0.0;

            int goldX = 0;

            // Only for binary labels
            if (Mapping.LabelCount == 2)
            {
                trueBinaryLabel = new List<double>();
                probTrueBinaryLabel = new List<double>();
            }

            foreach (var kvp in GoldLabels)
            {
                if (kvp.Value == null)
                    continue;

                // We have a gold label
                goldX++;

                Discrete trueLabel = null;
                if (TrueLabel.ContainsKey(kvp.Key))
                    trueLabel = TrueLabel[kvp.Key];

                if (trueLabel == null)
                {
                    trueLabel = Discrete.Uniform(Mapping.LabelCount);
                    //continue;  // No inferred label
                }

                var probs = trueLabel.GetProbs();
                double max = probs.Max();
                var predictedLabels = probs.Select((p, i) => new
                {
                    prob = p,
                    idx = i
                }).Where(a => a.prob == max).Select(a => a.idx).ToArray();

                int predictedLabel = predictedLabels.Length == 1 ? predictedLabels[0] : predictedLabels[Rand.Int(predictedLabels.Length)];

                this.PredictedLabel[kvp.Key] = predictedLabel;

                int goldLabel = kvp.Value.Value;

                confusionMatrix[goldLabel, predictedLabel] = confusionMatrix[goldLabel, predictedLabel] + 1.0;

                var nlp = -trueLabel.GetLogProb(goldLabel);
                if (nlp > nlpdThreshold)
                    nlp = nlpdThreshold;
                logProb += nlp;

                if (trueBinaryLabel != null)
                {
                    trueBinaryLabel.Add(goldLabel);
                    probTrueBinaryLabel.Add(probs[goldLabel]);
                }
            }

            Accuracy = correct / (double)goldX;
            NegativeLogProb = logProb / goldX;
            ModelConfusionMatrix = confusionMatrix;

            // Average recall
            double sumRec = 0;
            for (int i = 0; i < labelCount; i++)
            {
                double classSum = 0;
                for (int j = 0; j < labelCount; j++)
                {
                    classSum += confusionMatrix[i, j];
                }

                sumRec += confusionMatrix[i, i] / classSum;
            }
            AvgRecall = sumRec / labelCount;

            // WorkerLabelAccuracy: Perc. agreement between worker label and gold label
            int sumAcc = 0;
            var LabelSet = Mapping.DataWithGold;
            int numLabels = LabelSet.Count();
            foreach (var datum in LabelSet)
            {
                sumAcc += datum.WorkerLabel == datum.GoldLabel ? 1 : 0;
            }
            WorkerLabelAccuracy = sumAcc / (double)numLabels;

            if (trueBinaryLabel != null && trueBinaryLabel.Count > 0)
            {
                RocCurve = new ReceiverOperatingCharacteristic(trueBinaryLabel.ToArray(), probTrueBinaryLabel.ToArray());
                RocCurve.Compute(0.001);
                BinaryConfusionMatrix = new ReceiverOperatingCharacteristic.ConfusionMatrix((int)confusionMatrix[1, 1], (int)confusionMatrix[0, 0], (int)confusionMatrix[0, 1], (int)confusionMatrix[1, 0]);
            }
        }

        public static void WriteConfusionMatrix(StreamWriter writer, string worker, Dirichlet[] confusionMatrix)
        {
            int labelCount = confusionMatrix.Length;
            var meanConfusionMatrix = confusionMatrix.Select(cm => cm.GetMean()).ToArray();
            var printableConfusionMatrix = Util.ArrayInit(labelCount, labelCount, (i, j) => meanConfusionMatrix[i][j]);
            WriteWorkerConfusionMatrix(writer, worker, printableConfusionMatrix);
        }

        public static void WriteWorkerConfusionMatrix(StreamWriter writer, string worker, double[,] confusionMatrix)
        {
            int labelCount = confusionMatrix.GetLength(0);
            writer.WriteLine(worker);
            for (int j = 0; j < labelCount; j++)
                writer.Write(",{0}", j);
            writer.WriteLine();

            for (int i = 0; i < labelCount; i++)
            {
                writer.Write(i);
                for (int j = 0; j < labelCount; j++)
                    writer.Write(",{0:0.0000}", confusionMatrix[i, j]);

                writer.WriteLine();
            }
        }

        public virtual void WriteResults(StreamWriter writer, bool writeCommunityParameters, bool writeWorkerParameters, bool writeWorkerCommunities, IList<Datum> data = null)
        {
            if (writeCommunityParameters && this.CommunityConfusionMatrix != null)
            {
                for (int communityIndex = 0; communityIndex < this.CommunityConfusionMatrix.Length; communityIndex++)
                {
                    WriteConfusionMatrix(writer, "Community" + communityIndex, this.CommunityConfusionMatrix[communityIndex]);
                }
            }

            if (writeWorkerParameters && this.WorkerConfusionMatrix != null)
            {
                foreach (var kvp in this.WorkerConfusionMatrix.Distinct().Take(5))
                {
                    WriteConfusionMatrix(writer, kvp.Key, kvp.Value);
                }
            }

            if (this.TrueLabel != null)
            {
                foreach (var kvp in this.TrueLabel.OrderBy(kvp => kvp.Value.GetProbs()[0]))
                {
                    if (data != null)
                    {
                        var taskLabels = data.Where(d => d.TaskId == kvp.Key).Select(l => l.WorkerLabel);
                        var pos = taskLabels.Where(l => l == 1);
                        var neg = taskLabels.Where(l => l == 0);
                        int numPos = pos.Count();
                        int numNeg = neg.Count();
                        writer.WriteLine($"{kvp.Key}:\t{kvp.Value}\tnum_neg: {numNeg}\t num_pos: {numPos}");
                    }
                    else
                    {
                        writer.WriteLine($"{kvp.Key}:\t{kvp.Value}");
                    }
                }
            }
        }
    }
}
