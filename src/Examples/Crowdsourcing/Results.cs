// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Xml;

using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Serialization;
using Microsoft.ML.Probabilistic.Utilities;

namespace Crowdsourcing
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
            private set;
        }

        /// <summary>
        /// The predicted label for each task when doing simulations from the current
        /// model state. It avoids overwriting the true label posterior.
        /// </summary>
        public Dictionary<string, Discrete> LookAheadTrueLabel
        {
            get;
            private set;
        }

        /// <summary>
        /// The posterior for the constraint that allows online learning for the true label variable.
        /// </summary>
        public Dictionary<string, Discrete> TrueLabelConstraint
        {
            get;
            private set;
        }

        /// <summary>
        /// The probabilities that generate the true label of all the tasks.
        /// </summary>
        public Dirichlet BackgroundLabelProb
        {
            get;
            private set;
        }

        /// <summary>
        /// The posterior of the confusion matrix of each worker.
        /// </summary>
        public Dictionary<string, Dirichlet[]> WorkerConfusionMatrix
        {
            get;
            private set;
        }

        /// <summary>
        /// The look-ahead posterior of the confusion matrix of each worker obtained after simulating
        /// a new label in look-ahead run mode.
        /// </summary>
        public Dictionary<string, Dirichlet[]> LookAheadWorkerConfusionMatrix
        {
            get;
            private set;
        }


        /// <summary>
        /// The predictive probabilities of the labels produced by each worker.
        /// </summary>
        public Dictionary<string, Dictionary<string, Discrete>> WorkerPrediction
        {
            get;
            private set;
        }

        /// <summary>
        /// The community membership probabilities of each worker.
        /// </summary>
        public Dictionary<string, Discrete> WorkerCommunity
        {
            get;
            private set;
        }

        /// <summary>
        /// The confusion matrix of each community.
        /// </summary>
        public Dirichlet[][] CommunityConfusionMatrix
        {
            get;
            private set;
        }

        /// <summary>
        /// The score matrix of each community.
        /// </summary>
        public VectorGaussian[][] CommunityScoreMatrix
        {
            get;
            private set;
        }

        /// <summary>
        /// The posterior for the constraint that allows online learning for worker confusion matrices
        /// int the community model.
        /// </summary>
        public Dictionary<string, VectorGaussian[]> WorkerScoreMatrixConstraint
        {
            get;
            private set;
        }

        /// <summary>
        /// The probabilities that generate the community memberships of all the workers.
        /// </summary>
        public Dirichlet CommunityProb
        {
            get;
            private set;
        }

        /// <summary>
        /// The posterior for the constraint that allows online learning for community membership.
        /// int the community model.
        /// </summary>
        public Dictionary<string, Discrete> CommunityConstraint
        {
            get;
            private set;
        }

        /// <summary>
        /// Model evidence.
        /// </summary>
        public Bernoulli ModelEvidence
        {
            get;
            private set;
        }

        /// <summary>
        /// The data mapping.
        /// </summary>
        public DataMapping Mapping
        {
            get;
            private set;
        }

        /// <summary>
        /// The gold labels of each task. The gold label type is nullable to
        /// support the (usual) situation where the is no labels.
        /// </summary>
        public Dictionary<string, int?> GoldLabels
        {
            get;
            private set;
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
        /// Flags whether the model instance is CBCC (true) or BCC (false).
        /// </summary>
        public bool IsCommunityModel
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

        /// <summary>
        /// Runs the majority vote method on the data.
        /// </summary>
        /// <param name="data">The data</param>
        /// <param name="calculateAccuracy">Compute the accuracy (true).</param>
        /// <param name="useVoteDistribution">The true label is sampled from the vote distribution (true) or it is
        /// taken as the mode of the vote counts (false).
        /// In the latter case, ties are broken by sampling from the most voted classes.</param>
        /// <returns>The updated results</returns>
        public Results RunMajorityVote(IList<Datum> data, bool calculateAccuracy, bool useVoteDistribution)
        {
            var dataMapping = new DataMapping(data);
            Mapping = dataMapping;
            GoldLabels = Mapping.GetGoldLabelsPerTaskId();
            var inferredLabels = useVoteDistribution ? dataMapping.GetVoteDistribPerTaskIndex() : dataMapping.GetMajorityVotesPerTaskIndex().Select(mv => mv == null ? (Discrete)null : Discrete.PointMass(mv.Value, dataMapping.LabelCount)).ToArray();
            TrueLabel = inferredLabels.Select((lab, i) => new
            {
                key = dataMapping.TaskIndexToId[i],
                val = lab
            }).ToDictionary(a => a.key, a => a.val);

            if (calculateAccuracy)
            {
                UpdateAccuracy();
            }
            return this;
        }

        /// <summary>
        /// Run Dawid-Skene on the data.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="calculateAccuracy">Whether to calculate accuracy</param>
        /// <returns>A results instance</returns>
        public Results RunDawidSkene(IList<Datum> data, bool calculateAccuracy)
        {
            // If you want to run Dawid-Skene code, download his code, integrate it into
            // the project, and change false to true below.
#if false
            var dataMapping = new DataMapping(data);
            Mapping = dataMapping;
            var labelings = data.Select(d => new Labeling(d.WorkerId, d.TaskId, d.WorkerLabel.ToString(), d.GoldLabel.ToString())).ToList();
            DawidSkene ds = new DawidSkene(labelings, null, null);
            // The labels may be in a different order from our data labeling - we need to create a map.
            int[] labelIndexMap = new int[dataMapping.LabelCount];
            var dwLabels = ds.classes.Keys.ToArray();
            for (int i = 0; i < dataMapping.LabelCount; i++)
            {
                labelIndexMap[i] = Array.IndexOf(dwLabels, (i + dataMapping.LabelMin).ToString());
            }

            GoldLabels = Mapping.GetGoldLabelsPerTaskId().
                ToDictionary(kvp => kvp.Key, kvp => kvp.Value == null ? (int?)null : (int?)labelIndexMap[kvp.Value.Value]);

            ds.Estimate(10);

            var inferredLabels = ds.GetObjectClassProbabilities().Select(r => new Discrete(r)).ToArray();
            TrueLabel = inferredLabels.Select((lab, i) => new
            {
                key = dataMapping.TaskIndexToId[i],
                val = lab
            }).ToDictionary(a => a.key, a => a.val);

            if (calculateAccuracy)
            {
                UpdateAccuracy();
            }

            return this;
#else
            throw new ApplicationException("To support Dawid-Skene, you must link to the C# version of their code");
#endif
        }

        /// <summary>
        /// The different modes in which the model can be run.
        /// </summary>
        public enum RunMode
        {
            /// <summary>
            /// Clears all posteriors
            /// </summary>
            ClearResults,
            /// <summary>
            /// Training from a batch of data - uses initial priors.
            /// </summary>
            BatchTraining,
            /// <summary>
            /// Online training from a batch of data - uses previous posteriors as priors.
            /// </summary>
            OnlineTraining,
            /// <summary>
            /// Online training where we don't update the posteriors
            /// </summary>
            LookAheadExperiment,
            /// <summary>
            /// Use communities as workers in a BCC
            /// </summary>
            LoadAndUseCommunityPriors,
            /// <summary>
            /// Prediction of worker labels
            /// </summary>
            Prediction,
        };

        /// <summary>
        /// The structure with the model parameters.
        /// </summary>
        [Serializable]
        public struct NonTaskWorkerParameters
        {
            public Dirichlet BackgroundLabelProb;
            public Dirichlet CommunityProb;
            public VectorGaussian[][] CommunityScoreMatrix;
        }

        /// <summary>
        /// Runs the BCC or CBCC model.
        /// </summary>
        /// <param name="modelName">The model name.</param>
        /// <param name="data">The data that will be used for this run.</param>
        /// <param name="fullData">The full data set of data.</param>
        /// <param name="model">The model instance (BCC or CBCC).</param>
        /// <param name="mode">The mode (for example training, prediction, etc.).</param>
        /// <param name="calculateAccuracy">Whether to calculate accuracy.</param>
        /// <param name="numCommunities">The number of communities (community model only).</param>
        /// <param name="serialize">Whether to serialize all posteriors.</param>
        /// <param name="serializeCommunityPosteriors">Whether to serialize community posteriors.</param>
        public void RunBCC(string modelName, IList<Datum> data, IList<Datum> fullData, BCC model, RunMode mode, bool calculateAccuracy, int numCommunities = -1, bool serialize = false, bool serializeCommunityPosteriors = false)
        {
            CommunityModel communityModel = model as CommunityModel;
            IsCommunityModel = communityModel != null;
            string communityPriorsFileName = modelName + "CommunityPriors.xml";

            if (this.Mapping == null)
            {
                this.Mapping = new DataMapping(fullData, numCommunities);
                this.GoldLabels = this.Mapping.GetGoldLabelsPerTaskId();
            }

            /// A new model is created if the label count or the task count has changed
            bool createModel = (Mapping.LabelCount != model.LabelCount) || (Mapping.TaskCount != model.TaskCount);

            if (IsCommunityModel)
            {

                /// Creates a new CBCC model instance
                CommunityCount = numCommunities;
                createModel = createModel || (numCommunities != communityModel.CommunityCount);
                if (createModel)
                {
                    communityModel.CreateModel(Mapping.TaskCount, Mapping.LabelCount, numCommunities);
                }

            }
            else if (createModel)
            {
                /// Creates a new BCC model instance
                model.CreateModel(Mapping.TaskCount, Mapping.LabelCount);
            }

            /// Selects the prior according to the run mode
            BCC.Posteriors priors = null;
            switch (mode)
            {
                /// Use existing priors
                case RunMode.OnlineTraining:
                case RunMode.LookAheadExperiment:
                case RunMode.Prediction:
                    priors = ToPriors();
                    break;
                default:

                    /// Use default priors
                    ClearResults();
                    if (mode == RunMode.LoadAndUseCommunityPriors && IsCommunityModel)
                    {
                        priors = DeserializeCommunityPosteriors(communityPriorsFileName, numCommunities);
                    }
                    break;
            }

            /// Get data to observe
            var labelsPerWorkerIndex = Mapping.GetLabelsPerWorkerIndex(data);
            if (mode == RunMode.Prediction)
            {
                /// Signal prediction mode by setting all labels to null
                labelsPerWorkerIndex = labelsPerWorkerIndex.Select(arr => (int[])null).ToArray();
            }

            /// Run model inference
            BCC.Posteriors posteriors = model.Infer(
                Mapping.GetTaskIndicesPerWorkerIndex(data),
                labelsPerWorkerIndex, priors);
            UpdateResults(posteriors, mode);

            /// Compute accuracy
            if (calculateAccuracy)
            {
                UpdateAccuracy();
            }

            /// Serialize parameters
            if (serialize)
            {
                var type = IsCommunityModel ? typeof(CommunityModel.Posteriors) : typeof(BCC.Posteriors);
                DataContractSerializer serializer = new DataContractSerializer(type, new DataContractSerializerSettings { DataContractResolver = new InferDataContractResolver() });
                string posteriorsFileName = modelName + ".xml";
                using (XmlDictionaryWriter writer = XmlDictionaryWriter.CreateTextWriter(new FileStream(posteriorsFileName, FileMode.Create)))
                {
                    serializer.WriteObject(writer, posteriors);
                }
            }

            if (serializeCommunityPosteriors && IsCommunityModel)
            {
                SerializeCommunityPosteriors(communityPriorsFileName);
            }
        }

        /// <summary>
        /// Serializes the posteriors on an xml file.
        /// </summary>
        /// <param name="fileName">The file name.</param>
        void SerializeCommunityPosteriors(string fileName)
        {
            NonTaskWorkerParameters ntwp = new NonTaskWorkerParameters();
            ntwp.BackgroundLabelProb = BackgroundLabelProb;
            ntwp.CommunityProb = CommunityProb;
            ntwp.CommunityScoreMatrix = CommunityScoreMatrix;
            DataContractSerializer serializer = new DataContractSerializer(typeof(NonTaskWorkerParameters), new DataContractSerializerSettings { DataContractResolver = new InferDataContractResolver() });
            using (XmlDictionaryWriter writer = XmlDictionaryWriter.CreateTextWriter(new FileStream(fileName, FileMode.Create)))
            {
                serializer.WriteObject(writer, ntwp);
            }
        }

        /// <summary>
        /// Deserializes the parameters of CBCC from an xml file (used in the LoadAndUseCommunityPriors mode).
        /// </summary>
        /// <param name="fileName">The file name.</param>
        /// <param name="numCommunities">The number of communities.</param>
        /// <returns></returns>
        CommunityModel.Posteriors DeserializeCommunityPosteriors(string fileName, int numCommunities)
        {
            CommunityModel.Posteriors cbccPriors = new CommunityModel.Posteriors();
            DataContractSerializer serializer = new DataContractSerializer(typeof(NonTaskWorkerParameters), new DataContractSerializerSettings { DataContractResolver = new InferDataContractResolver() });
            using (XmlDictionaryReader reader = XmlDictionaryReader.CreateTextReader(new FileStream(fileName, FileMode.Open), new XmlDictionaryReaderQuotas()))
            {
                var ntwp = (NonTaskWorkerParameters)serializer.ReadObject(reader);

                if (ntwp.BackgroundLabelProb.Dimension != Mapping.LabelCount)
                {
                    throw new ApplicationException("Unexpected number of labels");
                }

                BackgroundLabelProb = ntwp.BackgroundLabelProb;
                cbccPriors.BackgroundLabelProb = ntwp.BackgroundLabelProb;
                if (ntwp.CommunityScoreMatrix.Length != numCommunities)
                {
                    throw new ApplicationException("Unexpected number of communities");
                }

                if (ntwp.CommunityScoreMatrix[0][0].Dimension != Mapping.LabelCount)
                {
                    throw new ApplicationException("Unexpected number of labels");
                }

                CommunityScoreMatrix = ntwp.CommunityScoreMatrix;
                cbccPriors.CommunityScoreMatrix = ntwp.CommunityScoreMatrix;

                if (ntwp.CommunityProb.Dimension != numCommunities)
                {
                    throw new ApplicationException("Unexpected number of communities");
                }

                CommunityProb = ntwp.CommunityProb;
                cbccPriors.CommunityProb = ntwp.CommunityProb;
            }

            return cbccPriors;
        }

        /// <summary>
        /// Resets all the parameters to the default values.
        /// </summary>
        void ClearResults()
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
        }

        /// <summary>
        /// Updates the results with the new posteriors.
        /// </summary>
        /// <param name="posteriors">The posteriors.</param>
        /// <param name="mode">The mode (for example training, prediction, etc.).</param>
        void UpdateResults(BCC.Posteriors posteriors, RunMode mode)
        {

            /// In the lookAheadExperiment mode, update only the LookAhead results
            if (mode == RunMode.LookAheadExperiment)
            {
                for (int t = 0; t < posteriors.TrueLabel.Length; t++)
                {
                    LookAheadTrueLabel[Mapping.TaskIndexToId[t]] = posteriors.TrueLabel[t];
                }
                for (int w = 0; w < posteriors.WorkerConfusionMatrix.Length; w++)
                {
                    LookAheadWorkerConfusionMatrix[Mapping.WorkerIndexToId[w]] = posteriors.WorkerConfusionMatrix[w];
                }
            }

            /// In the prediction mode, update only the worker prediction results
            else if (mode == RunMode.Prediction)
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
            else
            {
                /// In the all other modes, update all the results
                CommunityModel.Posteriors communityPosteriors = posteriors as CommunityModel.Posteriors;
                bool isCommunityModel = communityPosteriors != null;
                BackgroundLabelProb = posteriors.BackgroundLabelProb;
                for (int w = 0; w < posteriors.WorkerConfusionMatrix.Length; w++)
                {
                    WorkerConfusionMatrix[Mapping.WorkerIndexToId[w]] = posteriors.WorkerConfusionMatrix[w];
                }
                for (int t = 0; t < posteriors.TrueLabel.Length; t++)
                {
                    TrueLabel[Mapping.TaskIndexToId[t]] = posteriors.TrueLabel[t];
                }
                for (int t = 0; t < posteriors.TrueLabelConstraint.Length; t++)
                {
                    TrueLabelConstraint[Mapping.TaskIndexToId[t]] = posteriors.TrueLabelConstraint[t];
                }

                if (isCommunityModel)
                {
                    CommunityConfusionMatrix = communityPosteriors.CommunityConfusionMatrix;
                    for (int w = 0; w < communityPosteriors.WorkerScoreMatrixConstraint.Length; w++)
                    {
                        WorkerScoreMatrixConstraint[Mapping.WorkerIndexToId[w]] = communityPosteriors.WorkerScoreMatrixConstraint[w];
                        CommunityConstraint[Mapping.WorkerIndexToId[w]] = communityPosteriors.WorkerCommunityConstraint[w];
                        WorkerCommunity[Mapping.WorkerIndexToId[w]] = communityPosteriors.Community[w];
                    }

                    CommunityProb = communityPosteriors.CommunityProb;
                    CommunityScoreMatrix = communityPosteriors.CommunityScoreMatrix;
                }

                this.ModelEvidence = posteriors.Evidence;
            }
        }

        /// <summary>
        /// Loads the priors of BCC and CBCC.
        /// </summary>
        /// <returns>A BCC posterior instance with the loaded priors.</returns>
        BCC.Posteriors ToPriors()
        {
            int numClasses = Mapping.LabelCount;
            int numTasks = Mapping.TaskCount;
            int numWorkers = Mapping.WorkerCount;
            CommunityModel.Posteriors cbccPriors = new CommunityModel.Posteriors();
            BCC.Posteriors priors = IsCommunityModel ? cbccPriors : new BCC.Posteriors();

            /// Loads the prior of the background probabilities of the tasks
            priors.BackgroundLabelProb = BackgroundLabelProb;

            /// Loads the prior of the confusion matrix of each worker
            priors.WorkerConfusionMatrix = Util.ArrayInit(numWorkers,
                w =>
                {
                    string wid = Mapping.WorkerIndexToId[w];
                    if (WorkerConfusionMatrix.ContainsKey(wid))
                        return Util.ArrayInit(numClasses, lab => WorkerConfusionMatrix[wid][lab]);
                    else
                        return Util.ArrayInit(numClasses, lab => Dirichlet.Uniform(numClasses));
                });

            /// Loads the true label constraint of each task
            priors.TrueLabelConstraint = Util.ArrayInit(numTasks,
                t =>
                {
                    string tid = Mapping.TaskIndexToId[t];
                    if (TrueLabelConstraint.ContainsKey(tid))
                        return TrueLabelConstraint[Mapping.TaskIndexToId[t]];
                    else
                        return Discrete.Uniform(numClasses);
                });

            /// Loads the priors of the parameters of CBCC
            if (IsCommunityModel)
            {
                cbccPriors.CommunityConfusionMatrix = CommunityConfusionMatrix;
                cbccPriors.WorkerScoreMatrixConstraint = Util.ArrayInit(numWorkers,
                w =>
                {
                    string wid = Mapping.WorkerIndexToId[w];
                    if (WorkerScoreMatrixConstraint.ContainsKey(wid))
                        return Util.ArrayInit(numClasses, lab => WorkerScoreMatrixConstraint[wid][lab]);
                    else
                        return Util.ArrayInit(numClasses, lab => VectorGaussian.Uniform(numClasses));
                });
                cbccPriors.CommunityProb = CommunityProb;
                cbccPriors.CommunityScoreMatrix = CommunityScoreMatrix;
                cbccPriors.WorkerCommunityConstraint = Util.ArrayInit(numWorkers,
                w =>
                {
                    string wid = Mapping.WorkerIndexToId[w];
                    if (CommunityConstraint.ContainsKey(wid))
                        return CommunityConstraint[wid];
                    else
                        return Discrete.Uniform(CommunityCount);
                });
            }

            priors.Evidence = ModelEvidence;

            return priors;
        }

        /// <summary>
        /// Updates the accuracy using the current results.
        /// </summary>
        private void UpdateAccuracy()
        {
            double nlpdThreshold = -Math.Log(0.001);
            int labelCount = TrueLabel.Where(kvp => kvp.Value != null).First().Value.Dimension;
            var confusionMatrix = Util.ArrayInit(labelCount, labelCount, (i, j) => 0.0);
            int correct = 0;
            double logProb = 0.0;

            int goldX = 0;
            foreach (var kvp in GoldLabels)
            {
                if (kvp.Value == null)
                    continue;

                // We have a gold label
                goldX++;

                var trueLabel = TrueLabel[kvp.Key];
                if (trueLabel == null)
                    continue;  // No inferred label

                var probs = trueLabel.GetProbs();
                var max = probs.Max();
                var predictedLabels = probs.Select((p, i) => new
                {
                    prob = p,
                    idx = i
                }).Where(a => a.prob == max).Select(a => a.idx).ToArray();
                int predictedLabel = predictedLabels.Length == 1 ? predictedLabels[0] : predictedLabels[Rand.Int(predictedLabels.Length)];
                int goldLabel = kvp.Value.Value;

                confusionMatrix[goldLabel, predictedLabel] = confusionMatrix[goldLabel, predictedLabel] + 1.0;

                if (goldLabel == predictedLabel)
                    correct++;

                var nlp = -trueLabel.GetLogProb(goldLabel);
                if (nlp > nlpdThreshold)
                    nlp = nlpdThreshold;
                logProb += nlp;
            }

            if (goldX == 0) Console.WriteLine($"Accuracy and recall are NaN because no gold labels were provided.");
            Accuracy = correct / (double)goldX;
            NegativeLogProb = logProb / goldX;
            ModelConfusionMatrix = confusionMatrix;

            // Compute average recall
            double sumRec = 0;
            int actualLabelCount = 0;
            for (int goldLabel = 0; goldLabel < labelCount; goldLabel++)
            {
                double goldLabelCount = 0;
                for (int predictedLabel = 0; predictedLabel < labelCount; predictedLabel++)
                {
                    goldLabelCount += confusionMatrix[goldLabel, predictedLabel];
                }

                if (goldLabelCount > 0)
                {
                    actualLabelCount++;
                    sumRec += confusionMatrix[goldLabel, goldLabel] / goldLabelCount;
                }
            }
            AvgRecall = sumRec / actualLabelCount;
        }

        /// <summary>
        /// Writes out the mean of an uncertain confusion matrix to a StreamWriter.
        /// The confusion matrix is passed as an of Dirichlets, one for each row
        /// of the confusion matrix (as given by the posteriors from the model).
        /// </summary>
        /// <param name="writer">A| StreamWriter instance.</param>
        /// <param name="worker">The worker id.</param>
        /// <param name="confusionMatrix">The confusion matrix</param>
        private static void WriteConfusionMatrix(StreamWriter writer, string worker, Dirichlet[] confusionMatrix)
        {
            int labelCount = confusionMatrix.Length;
            var meanConfusionMatrix = confusionMatrix.Select(cm => cm.GetMean()).ToArray();
            var printableConfusionMatrix = Util.ArrayInit(labelCount, labelCount, (i, j) => meanConfusionMatrix[i][j]);
            WriteWorkerConfusionMatrix(writer, worker, printableConfusionMatrix);
        }

        /// <summary>
        /// Writes the a confusion matrix to a stream writer.
        /// </summary>
        /// <param name="writer">A StreamWriter instance.</param>
        /// <param name="worker">The worker id.</param>
        /// <param name="confusionMatrix">The confusion matrix.</param>
        private static void WriteWorkerConfusionMatrix(StreamWriter writer, string worker, double[,] confusionMatrix)
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

        /// <summary>
        /// Writes various results to a StreamWriter.
        /// </summary>
        /// <param name="writer">A StreamWriter instance.</param>
        /// <param name="writeCommunityParameters">Set true to write community parameters.</param>
        /// <param name="writeWorkerParameters">Set true to write worker parameters.</param>
        /// <param name="writeWorkerCommunities">Set true to write worker communities.</param>
        public void WriteResults(StreamWriter writer, bool writeCommunityParameters, bool writeWorkerParameters, bool writeWorkerCommunities)
        {
            this.WriteAccuracy(writer);

            if (writeCommunityParameters && this.CommunityConfusionMatrix != null)
            {
                for (int communityIndex = 0; communityIndex < this.CommunityConfusionMatrix.Length; communityIndex++)
                {
                    WriteConfusionMatrix(writer, "Community" + communityIndex, this.CommunityConfusionMatrix[communityIndex]);
                }
            }

            if (writeWorkerParameters && this.WorkerConfusionMatrix != null)
            {
                foreach (var kvp in this.WorkerConfusionMatrix)
                {
                    WriteConfusionMatrix(writer, kvp.Key, kvp.Value);
                }
            }

            if (writeWorkerCommunities && this.WorkerCommunity != null)
            {
                foreach (var kvp in this.WorkerCommunity)
                {
                    writer.WriteLine(string.Format("{0}:\t{1}", kvp.Key, kvp.Value));
                }
            }

            writer.WriteLine("Log Evidence = {0:0.0000}", ModelEvidence.LogOdds);
        }

        /// <summary>
        /// Writes the accuracy results on the StreamWriter.
        /// </summary>
        /// <param name="writer">The StreamWriter.</param>
        public void WriteAccuracy(StreamWriter writer)
        {
            writer.WriteLine("Accuracy = {0:0.000}", this.Accuracy);
            writer.WriteLine("Mean negative log prob density = {0:0.000}", this.NegativeLogProb);
            WriteWorkerConfusionMatrix(writer, "Model confusion matrix", this.ModelConfusionMatrix);
        }
    }
}
