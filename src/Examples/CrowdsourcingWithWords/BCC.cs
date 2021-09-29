// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;

namespace CrowdsourcingWithWords
{
    /// <summary>
    /// The BCC model class.
    /// </summary>
    public class BCC
    {
        /// <summary>
        /// The number of label values.
        /// </summary>
        public int LabelCount => c?.SizeAsInt ?? 0;

        /// <summary>
        /// The number of tasks.
        /// </summary>
        public int TaskCount => n?.SizeAsInt ?? 0;

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
        /// Returns the confusion matrix prior of each worker.
        /// </summary>
        /// <returns>The confusion matrix prior of each worker.</returns>
        public Dirichlet[] GetConfusionMatrixPrior()
        {
            var confusionMatrixPrior = new Dirichlet[LabelCount];
            for (int d = 0; d < LabelCount; d++)
            {
                confusionMatrixPrior[d] = new Dirichlet(Util.ArrayInit(LabelCount, i => i == d ? (InitialWorkerBelief / (1 - InitialWorkerBelief)) * LabelCount : 1.0));
            }

            return confusionMatrixPrior;
        }
    }
}
