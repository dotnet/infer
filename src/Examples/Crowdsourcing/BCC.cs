// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Crowdsourcing
{
    /// <summary>
    /// Implements statistical inference for (community-based and non-community-based) Bayesian classifier combination.
    /// </summary>
    /// <remarks>
    /// References:
    /// Matteo Venanzi, John Guiver, Gabriella Kazai, Pushmeet Kohli, and Milad Shokouhi. Community-Based Bayesian Aggregation Models for Crowdsourcing. In Proceedings of the 23rd International World Wide Web Conference, WWW2014, ACM, April 2014.
    /// H.C. Kim and Z. Ghahramani. Bayesian classifier combination. International Conference on Articial Intelligence and Statistics, pages 619-627, 2012.
    /// </remarks>
    public class BCC
    {
        /// <summary>
        /// The number of label values.
        /// </summary>
        public int LabelCount
        {
            get
            {
                return c == null ? 0 : c.SizeAsInt;
            }
        }

        /// <summary>
        /// The number of tasks.
        /// </summary>
        public int TaskCount
        {
            get
            {
                return n == null ? 0 : n.SizeAsInt;
            }
        }

        // Ranges
        protected Range n;
        protected Range k;
        protected Range c;
        protected Range kn;

        // Variables in the model
        protected Variable<int> WorkerCount;
        protected VariableArray<int> TrueLabel;
        protected VariableArray<int> WorkerTaskCount;
        protected VariableArray<VariableArray<int>, int[][]> WorkerTaskIndex;
        protected VariableArray<VariableArray<int>, int[][]> WorkerLabel;
        protected Variable<Vector> BackgroundLabelProb;
        protected VariableArray<VariableArray<Vector>, Vector[][]> WorkerConfusionMatrix;
        protected Variable<bool> Evidence;

        // Prior distributions
        protected Variable<Dirichlet> BackgroundLabelProbPrior;
        protected VariableArray<VariableArray<Dirichlet>, Dirichlet[][]> ConfusionMatrixPrior;
        protected VariableArray<Discrete> TrueLabelConstraint;
        protected Variable<Bernoulli> EvidencePrior;

        // Inference engine
        protected InferenceEngine Engine;

        // Hyperparameters and inference settings
        public double InitialWorkerBelief
        {
            get;
            set;
        }

        /// <summary>
        /// The number of inference iterations.
        /// </summary>
        public int NumberOfIterations
        {
            get;
            set;
        }

        /// <summary>
        /// Creates a BCC model instance.
        /// </summary>
        public BCC()
        {
            InitialWorkerBelief = 0.5;
            NumberOfIterations = 35;
            EvidencePrior = new Bernoulli(0.5);
        }

        /// <summary>
        /// Initializes the ranges, the generative process and the inference engine of the BCC model.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="labelCount">The number of labels.</param>
        public virtual void CreateModel(int taskCount, int labelCount)
        {
            Evidence = Variable<bool>.Random(this.EvidencePrior);
            var evidenceBlock = Variable.If(Evidence);
            DefineVariablesAndRanges(taskCount, labelCount);
            DefineGenerativeProcess();
            DefineInferenceEngine();
            evidenceBlock.CloseBlock();
        }

        /// <summary>
        /// Initializes the ranges of the variables.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="labelCount">The number of labels.</param>
        protected virtual void DefineVariablesAndRanges(int taskCount, int labelCount)
        {
            WorkerCount = Variable.New<int>().Named("WorkerCount");
            n = new Range(taskCount).Named("n");
            c = new Range(labelCount).Named("c");
            k = new Range(WorkerCount).Named("k");

            // The tasks for each worker
            WorkerTaskCount = Variable.Array<int>(k).Named("WorkerTaskCount");
            kn = new Range(WorkerTaskCount[k]).Named("kn");
            WorkerTaskIndex = Variable.Array(Variable.Array<int>(kn), k).Named("WorkerTaskIndex");
            WorkerTaskIndex.SetValueRange(n);
            WorkerLabel = Variable.Array(Variable.Array<int>(kn), k).Named("WorkerLabel");

            // The background probability vector
            BackgroundLabelProbPrior = Variable.New<Dirichlet>().Named("BackgroundLabelProbPrior");
            BackgroundLabelProb = Variable<Vector>.Random(BackgroundLabelProbPrior).Named("BackgroundLabelProb");
            BackgroundLabelProb.SetValueRange(c);

            // The confusion matrices for each worker
            ConfusionMatrixPrior = Variable.Array(Variable.Array<Dirichlet>(c), k).Named("ConfusionMatrixPrior");
            WorkerConfusionMatrix = Variable.Array(Variable.Array<Vector>(c), k).Named("ConfusionMatrix");
            WorkerConfusionMatrix[k][c] = Variable<Vector>.Random(ConfusionMatrixPrior[k][c]);
            WorkerConfusionMatrix.SetValueRange(c);

            // The unobserved 'true' label for each task
            TrueLabel = Variable.Array<int>(n).Attrib(QueryTypes.Marginal).Attrib(QueryTypes.MarginalDividedByPrior).Named("Truth");
            TrueLabelConstraint = Variable.Array<Discrete>(n).Named("TruthConstraint");
            // Constraint for online learning
            TrueLabel[n] = Variable.Discrete(BackgroundLabelProb).ForEach(n);
            Variable.ConstrainEqualRandom(TrueLabel[n], TrueLabelConstraint[n]);
            // The worker labels
            WorkerLabel = Variable.Array(Variable.Array<int>(kn), k).Named("WorkerLabel");
        }

        /// <summary>
        /// Defines the BCC generative process.
        /// </summary>
        protected virtual void DefineGenerativeProcess()
        {
            // The process that generates the worker's label
            using (Variable.ForEach(k))
            {
                var trueLabel = Variable.Subarray(TrueLabel, WorkerTaskIndex[k]);
                trueLabel.SetValueRange(c);
                using (Variable.ForEach(kn))
                {
                    using (Variable.Switch(trueLabel[kn]))
                    {
                        WorkerLabel[k][kn] = Variable.Discrete(WorkerConfusionMatrix[k][trueLabel[kn]]);
                    }
                }
            }
        }

        /// <summary>
        /// Initializes the BCC inference engine.
        /// </summary>
        protected virtual void DefineInferenceEngine()
        {
            Engine = new InferenceEngine(new ExpectationPropagation());
            Engine.Compiler.UseParallelForLoops = true;
            Engine.ShowProgress = false;
            Engine.Compiler.WriteSourceFiles = false;
        }

        /// <summary>
        /// Sets the priors of BCC.
        /// </summary>
        /// <param name="workerCount">The number of workers.</param>
        /// <param name="priors">The priors.</param>
        protected virtual void SetPriors(int workerCount, Posteriors priors)
        {
            WorkerCount.ObservedValue = workerCount;
            if (priors == null)
            {
                BackgroundLabelProbPrior.ObservedValue = Dirichlet.Uniform(LabelCount);
                var confusionMatrixPrior = GetConfusionMatrixPrior();
                ConfusionMatrixPrior.ObservedValue = Util.ArrayInit(workerCount, worker => Util.ArrayInit(LabelCount, lab => confusionMatrixPrior[lab]));
                TrueLabelConstraint.ObservedValue = Util.ArrayInit(TaskCount, t => Discrete.Uniform(LabelCount));
            }
            else
            {
                BackgroundLabelProbPrior.ObservedValue = priors.BackgroundLabelProb;
                ConfusionMatrixPrior.ObservedValue = priors.WorkerConfusionMatrix;
                TrueLabelConstraint.ObservedValue = priors.TrueLabelConstraint;
            }
        }

        /// <summary>
        /// Attachs the data to the workers labels.
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        protected virtual void AttachData(int[][] taskIndices, int[][] workerLabels)
        {
            AttachData(taskIndices, workerLabels, null);
        }

        /// <summary>
        /// Attachs the data to the workers labels with and sets the workers' confusion matrix priors.
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        /// <param name="confusionMatrixPrior">The workers' confusion matrix priors.</param>
        protected virtual void AttachData(int[][] taskIndices, int[][] workerLabels, Dirichlet[][] confusionMatrixPrior)
        {
            int numClasses = c.SizeAsInt;
            WorkerCount.ObservedValue = taskIndices.Length;
            WorkerTaskCount.ObservedValue = taskIndices.Select(tasks => tasks.Length).ToArray();
            WorkerTaskIndex.ObservedValue = taskIndices;
            // Prediction mode is indicated by none of the workers having a label.
            // We can just look at the first one
            if (workerLabels[0] != null)
            {
                WorkerLabel.ObservedValue = workerLabels;
            }
            else
            {
                WorkerLabel.ClearObservedValue();
            }

            if (confusionMatrixPrior != null)
            {
                ConfusionMatrixPrior.ObservedValue = Util.ArrayInit(confusionMatrixPrior.Length, worker => Util.ArrayInit(numClasses, lab => confusionMatrixPrior[worker][lab]));
            }
        }

        /// <summary>
        /// Infers the posteriors of BCC using the attached data and priors.
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        /// <param name="priors">The priors of the BCC parameters.</param>
        /// <returns></returns>
        public virtual Posteriors Infer(int[][] taskIndices, int[][] workerLabels, Posteriors priors)
        {
            int workerCount = workerLabels.Length;
            SetPriors(workerCount, priors);
            AttachData(taskIndices, workerLabels, null);
            var result = new Posteriors();
            Engine.NumberOfIterations = NumberOfIterations;
            result.Evidence = Engine.Infer<Bernoulli>(Evidence);
            result.BackgroundLabelProb = Engine.Infer<Dirichlet>(BackgroundLabelProb);
            result.WorkerConfusionMatrix = Engine.Infer<Dirichlet[][]>(WorkerConfusionMatrix);
            result.TrueLabel = Engine.Infer<Discrete[]>(TrueLabel);
            result.TrueLabelConstraint = Engine.Infer<Discrete[]>(TrueLabel, QueryTypes.MarginalDividedByPrior);

            // Prediction mode is indicated by none of the workers having a label.
            // We can just look at the first one
            if (workerLabels[0] == null)
            {
                result.WorkerPrediction = Engine.Infer<Discrete[][]>(WorkerLabel);
            }

            return result;
        }

        /// <summary>
        /// Returns the confusion matrix prior of each worker.
        /// </summary>
        /// <returns>The confusion matrix prior of each worker.</returns>
        public Dirichlet[] GetConfusionMatrixPrior()
        {
            var confusionMatrixPrior = new Dirichlet[LabelCount];
            for (int d = 0; d < LabelCount; d++)
            {
                // The prior prefers diagonal confusion matrices.
                // The paper says "Each row of π(k)_c has a Dirichlet prior with pseudo counts 1 expect for the diagonal count set to C − 1." but that does not work for C=2.
                // Instead of following the paper, this code sets the diagonal to C.
                confusionMatrixPrior[d] = new Dirichlet(Util.ArrayInit(LabelCount, i => (i == d) ? (InitialWorkerBelief / (1 - InitialWorkerBelief)) * LabelCount : 1.0));
            }

            return confusionMatrixPrior;
        }

        /// <summary>
        /// The BCC posteriors class.
        /// </summary>
        [Serializable]
        public class Posteriors
        {
            /// <summary>
            /// The probabilities that generate the true labels of all the tasks.
            /// </summary>
            public Dirichlet BackgroundLabelProb;

            /// <summary>
            /// The probabilities of the true label of each task.
            /// </summary>
            public Discrete[] TrueLabel;

            /// <summary>
            /// The Dirichlet parameters of the confusion matrix of each worker.
            /// </summary>
            public Dirichlet[][] WorkerConfusionMatrix;

            /// <summary>
            /// The predictive probabilities of the worker's labels.
            /// </summary>
            public Discrete[][] WorkerPrediction;

            /// <summary>
            /// The true label constraint used in online training.
            /// </summary>
            public Discrete[] TrueLabelConstraint;

            /// <summary>
            /// The model evidence.
            /// </summary>
            public Bernoulli Evidence;
        }
    }
}
