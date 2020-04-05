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
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Crowdsourcing
{
    /// <summary>
    /// The CommunityBCC model class.
    /// </summary>
    public class CommunityModel : BCC
    {
        // Additional ranges
        protected Range m;

        // Additional variables
        protected VariableArray<int> Community;
        protected VariableArray<Discrete> CommunityInit;
        protected Variable<Vector> CommunityProb;
        protected VariableArray<VariableArray<Vector>, Vector[][]> ScoreMatrix;
        protected VariableArray<VariableArray<Vector>, Vector[][]> CommunityScoreMatrix;
        protected VariableArray<VariableArray<Vector>, Vector[][]> CommunityConfusionMatrix;
        protected Variable<PositiveDefiniteMatrix> NoiseMatrix = Variable.New<PositiveDefiniteMatrix>().Named("NoiseMatrix");

        // Additional priors
        protected VariableArray<Discrete> CommunityConstraint;
        protected VariableArray<VariableArray<VectorGaussian>, VectorGaussian[][]> ScoreMatrixConstraint;
        protected VariableArray<VariableArray<VectorGaussian>, VectorGaussian[][]> CommunityScoreMatrixPrior;
        protected Variable<Dirichlet> CommunityProbPrior;

        /// <summary>
        /// The noise precision that generates the workers score matrix from the communities score matrix.
        /// </summary>
        public double NoisePrecision
        {
            get;
            set;
        }

        /// <summary>
        /// The number of communities.
        /// </summary>
        public int CommunityCount
        {
            get;
            protected set;
        }

        /// <summary>
        /// The mean vector of the Gaussian distribution generating the community score matrices.
        /// </summary>
        public Tuple<double, double>[] ScoreMeanParameters
        {
            get;
            set;
        }

        /// <summary>
        /// The precision matrix of the Gaussian distribution generating the community score matrices.
        /// </summary>
        public double[] ScorePrecisionParameters
        {
            get;
            set;
        }

        /// <summary>
        /// The hyperparameter governing community membership.
        /// </summary>
        public double CommunityPseudoCount
        {
            get;
            set;
        }

        /// <summary>
        /// The prior for the score matrices.
        /// </summary>
        public VectorGaussian[][] CommunityScoreMatrixPriorObserved
        {
            get;
            protected set;
        }

        /// <summary>
        /// The prior for community membership.
        /// </summary>
        public Dirichlet CommunityProbPriorObserved
        {
            get;
            protected set;
        }

        /// <summary>
        /// Creates a CBCC model instance.
        /// </summary>
        public CommunityModel()
            : base()
        {
            NoisePrecision = 5;
            CommunityCount = 3;
            CommunityPseudoCount = 10.0;
            ScoreMeanParameters = null;
            ScorePrecisionParameters = null;
        }

        /// <summary>
        /// Initializes the CBCC model.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="labelCount">The number of labels.</param>
        public override void CreateModel(int taskCount, int labelCount)
        {
            CreateModel(taskCount, labelCount, CommunityCount);
        }

        /// <summary>
        /// Initializes the CBCC model with a number of communities.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="labelCount">The number of labels.</param>
        /// <param name="communityCount">The number of communities.</param>
        public virtual void CreateModel(int taskCount, int labelCount, int communityCount)
        {
            Evidence = Variable<bool>.Random(this.EvidencePrior);
            var evidenceBlock = Variable.If(Evidence);
            CommunityCount = communityCount;
            CommunityProbPriorObserved = Dirichlet.Symmetric(communityCount, CommunityPseudoCount);
            DefineVariablesAndRanges(taskCount, labelCount);
            DefineGenerativeProcess();
            DefineInferenceEngine();
            evidenceBlock.CloseBlock();

            if (ScoreMeanParameters == null)
            {
                var scoreMatrixPrior = GetScoreMatrixPrior();
                CommunityScoreMatrixPriorObserved = Util.ArrayInit(CommunityCount, comm => Util.ArrayInit(labelCount, lab => new VectorGaussian(scoreMatrixPrior[lab])));
            }
            else
            {
                CommunityScoreMatrixPriorObserved = Util.ArrayInit(
                    CommunityCount,
                    comm => Util.ArrayInit(
                        labelCount, lab => VectorGaussian.FromMeanAndPrecision(
                            Vector.FromArray(
                            Util.ArrayInit(labelCount, lab1 => lab == lab1 ? ScoreMeanParameters[comm].Item1 : ScoreMeanParameters[comm].Item2)),
                            PositiveDefiniteMatrix.IdentityScaledBy(LabelCount, ScorePrecisionParameters[comm]))));
            }
        }

        /// <summary>
        /// Defines the variables and the ranges of CBCC.
        /// </summary>
        /// <param name="taskCount">The number of tasks.</param>
        /// <param name="labelCount">The number of labels.</param>
        protected override void DefineVariablesAndRanges(int taskCount, int labelCount)
        {
            WorkerCount = Variable.New<int>().Named("WorkerCount");
            m = new Range(CommunityCount).Named("m");
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

            // Community membership
            CommunityProbPrior = Variable.New<Dirichlet>().Named("CommunityProbPrior");
            CommunityProb = Variable<Vector>.Random(CommunityProbPrior).Named("CommunityProb");
            CommunityProb.SetValueRange(m);
            Community = Variable.Array<int>(k).Attrib(QueryTypes.Marginal).Attrib(QueryTypes.MarginalDividedByPrior).Named("Community");
            CommunityConstraint = Variable.Array<Discrete>(k).Named("CommunityConstraint");
            Community[k] = Variable.Discrete(CommunityProb).ForEach(k);
            Variable.ConstrainEqualRandom(Community[k], CommunityConstraint[k]);
            // Initialiser to break symmetry for community membership
            CommunityInit = Variable.Array<Discrete>(k).Named("CommunityInit");
            Community[k].InitialiseTo(CommunityInit[k]);

            // Community parameters
            CommunityScoreMatrixPrior = Variable.Array(Variable.Array<VectorGaussian>(c), m).Named("CommunityScoreMatrixPrior");
            CommunityScoreMatrix = Variable.Array(Variable.Array<Vector>(c), m).Named("CommunityScoreMatrix");
            CommunityScoreMatrix[m][c] = Variable<Vector>.Random(CommunityScoreMatrixPrior[m][c]);
            CommunityConfusionMatrix = Variable.Array(Variable.Array<Vector>(c), m).Named("CommunityConfusionMatrix");
            CommunityConfusionMatrix[m][c] = Variable.Softmax(CommunityScoreMatrix[m][c]);
            CommunityScoreMatrix.SetValueRange(c);

            // Parameters for each worker
            ScoreMatrix = Variable.Array(Variable.Array<Vector>(c), k).Attrib(QueryTypes.Marginal).Attrib(QueryTypes.MarginalDividedByPrior).Named("ScoreMatrix");
            ScoreMatrixConstraint = Variable.Array(Variable.Array<VectorGaussian>(c), k).Named("ScoreMatrixConstraint");
            WorkerConfusionMatrix = Variable.Array(Variable.Array<Vector>(c), k).Named("ConfusionMatrix");

            // The unobserved 'true' label for each task
            TrueLabel = Variable.Array<int>(n).Attrib(QueryTypes.Marginal).Attrib(QueryTypes.MarginalDividedByPrior).Named("Truth");
            TrueLabelConstraint = Variable.Array<Discrete>(n).Named("TruthConstraint");
            TrueLabel[n] = Variable.Discrete(BackgroundLabelProb).ForEach(n);
            Variable.ConstrainEqualRandom(TrueLabel[n], TrueLabelConstraint[n]);

            // The labels given by the workers
            WorkerLabel = Variable.Array(Variable.Array<int>(kn), k).Named("WorkerLabel");
        }

        /// <summary>
        /// Defines the generative process of CBCC.
        /// </summary>
        protected override void DefineGenerativeProcess()
        {
            // The process that generates the worker's label
            using (Variable.ForEach(k))
            {
                using (Variable.Switch(Community[k]))
                {
                    ScoreMatrix[k][c] = Variable.VectorGaussianFromMeanAndPrecision(CommunityScoreMatrix[Community[k]][c], NoiseMatrix);
                }

                Variable.ConstrainEqualRandom(ScoreMatrix[k][c], ScoreMatrixConstraint[k][c]);
                WorkerConfusionMatrix[k][c] = Variable.Softmax(ScoreMatrix[k][c]);
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
        /// Initializes the CBCC inference engine.
        /// </summary>
        protected override void DefineInferenceEngine()
        {
            Engine = new InferenceEngine(new VariationalMessagePassing());
            Engine.ShowProgress = false;
            Engine.Compiler.UseParallelForLoops = true;
            Engine.Compiler.GivePriorityTo(typeof(SoftmaxOp_BL06));
            Engine.Compiler.WriteSourceFiles = false;
        }

        /// <summary>
        /// Attachs the data to the workers labels.
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        protected override void AttachData(int[][] taskIndices, int[][] workerLabels)
        {
            AttachData(taskIndices, workerLabels, null, null);
        }

        /// <summary>
        /// Attachs the data to the workers labels and sets the constraints on the community score matrices and
        /// the community memberships (used for online training).
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        /// <param name="scoreConstraint">The constraint of the community score matrices.</param>
        /// <param name="communityConstraint">The constraint of the workers community membership.</param>
        protected void AttachData(int[][] taskIndices, int[][] workerLabels, VectorGaussian[][] scoreConstraint, Discrete[] communityConstraint)
        {
            int communityCount = m.SizeAsInt;
            int workerCount = workerLabels.Length;
            int labelCount = c.SizeAsInt;
            base.AttachData(taskIndices, workerLabels);
            CommunityInit.ObservedValue = Util.ArrayInit(workerCount, worker => Discrete.PointMass(Rand.Int(communityCount), communityCount));

            if (scoreConstraint != null)
            {
                ScoreMatrixConstraint.ObservedValue = scoreConstraint;
            }
            else
            {
                ScoreMatrixConstraint.ObservedValue = Util.ArrayInit(workerCount, w => Util.ArrayInit(labelCount, lab => VectorGaussian.Uniform(labelCount)));
            }

            if (communityConstraint != null)
            {
                CommunityConstraint.ObservedValue = communityConstraint;
            }
            else
            {
                CommunityConstraint.ObservedValue = Util.ArrayInit(workerCount, w => Discrete.Uniform(communityCount));
            }
        }

        /// <summary>
        /// Sets the priors of CBCC.
        /// </summary>
        /// <param name="workerCount">The number of workers.</param>
        /// <param name="priors">The priors.</param>
        protected override void SetPriors(int workerCount, BCC.Posteriors priors)
        {
            int communityCount = m.SizeAsInt;
            int labelCount = c.SizeAsInt;
            WorkerCount.ObservedValue = workerCount;
            NoiseMatrix.ObservedValue = PositiveDefiniteMatrix.IdentityScaledBy(labelCount, NoisePrecision);
            CommunityModel.Posteriors cbccPriors = (CommunityModel.Posteriors)priors;

            if (cbccPriors == null || cbccPriors.BackgroundLabelProb == null)
                BackgroundLabelProbPrior.ObservedValue = Dirichlet.Uniform(labelCount);
            else
                BackgroundLabelProbPrior.ObservedValue = cbccPriors.BackgroundLabelProb;

            if (cbccPriors == null || cbccPriors.CommunityProb == null)
                CommunityProbPrior.ObservedValue = CommunityProbPriorObserved;
            else
                CommunityProbPrior.ObservedValue = cbccPriors.CommunityProb;

            if (cbccPriors == null || cbccPriors.CommunityScoreMatrix == null)
                CommunityScoreMatrixPrior.ObservedValue = CommunityScoreMatrixPriorObserved;
            else
                CommunityScoreMatrixPrior.ObservedValue = cbccPriors.CommunityScoreMatrix;

            if (cbccPriors == null || cbccPriors.TrueLabelConstraint == null)
                TrueLabelConstraint.ObservedValue = Util.ArrayInit(TaskCount, t => Discrete.Uniform(labelCount));
            else
                TrueLabelConstraint.ObservedValue = cbccPriors.TrueLabelConstraint;
        }

        /// <summary>
        /// Infers the posteriors of CBCC using the attached data.
        /// </summary>
        /// <param name="taskIndices">The matrix of the task indices (columns) of each worker (rows).</param>
        /// <param name="workerLabels">The matrix of the labels (columns) of each worker (rows).</param>
        /// <param name="priors">The priors.</param>
        /// <returns></returns>
        public override BCC.Posteriors Infer(int[][] taskIndices, int[][] workerLabels, BCC.Posteriors priors)
        {
            var cbccPriors = (CommunityModel.Posteriors)priors;
            VectorGaussian[][] scoreConstraint = (cbccPriors == null ? null : cbccPriors.WorkerScoreMatrixConstraint);
            Discrete[] communityConstraint = (cbccPriors == null ? null : cbccPriors.WorkerCommunityConstraint);
            SetPriors(workerLabels.Length, priors);
            AttachData(taskIndices, workerLabels, scoreConstraint, communityConstraint);
            var result = new CommunityModel.Posteriors();
            Engine.NumberOfIterations = NumberOfIterations;
            result.Evidence = Engine.Infer<Bernoulli>(Evidence);
            result.BackgroundLabelProb = Engine.Infer<Dirichlet>(BackgroundLabelProb);
            result.WorkerConfusionMatrix = Engine.Infer<Dirichlet[][]>(WorkerConfusionMatrix);
            result.TrueLabel = Engine.Infer<Discrete[]>(TrueLabel);
            result.TrueLabelConstraint = Engine.Infer<Discrete[]>(TrueLabel, QueryTypes.MarginalDividedByPrior);
            result.CommunityScoreMatrix = Engine.Infer<VectorGaussian[][]>(CommunityScoreMatrix);
            result.CommunityConfusionMatrix = Engine.Infer<Dirichlet[][]>(CommunityConfusionMatrix);
            result.WorkerScoreMatrixConstraint = Engine.Infer<VectorGaussian[][]>(ScoreMatrix, QueryTypes.MarginalDividedByPrior);
            result.CommunityProb = Engine.Infer<Dirichlet>(CommunityProb);
            result.Community = Engine.Infer<Discrete[]>(Community);
            result.WorkerCommunityConstraint = Engine.Infer<Discrete[]>(Community, QueryTypes.MarginalDividedByPrior);
            return result;
        }

        /// <summary>
        /// Returns the community score matrix prior.
        /// </summary>
        /// <returns>The community score matrix prior.</returns>
        private VectorGaussian[] GetScoreMatrixPrior()
        {
            var dim = new Range(LabelCount);
            var mean = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(LabelCount), PositiveDefiniteMatrix.IdentityScaledBy(LabelCount, 1));
            var prec = Variable.WishartFromShapeAndRate(1.0, PositiveDefiniteMatrix.IdentityScaledBy(LabelCount, 1));
            var score = Variable.VectorGaussianFromMeanAndPrecision(mean, prec);
            var confusion = Variable.Softmax(score);
            confusion.SetValueRange(dim);
            var confusionConstraint = Variable.New<Dirichlet>();
            Variable.ConstrainEqualRandom(confusion, confusionConstraint);
            var engine = new InferenceEngine(new VariationalMessagePassing())
            {
                ShowProgress = false
            };

            engine.Compiler.WriteSourceFiles = false;
            var scorePrior = new VectorGaussian[LabelCount];
            for (int d = 0; d < LabelCount; d++)
            {
                confusionConstraint.ObservedValue = new Dirichlet(Util.ArrayInit(LabelCount, i => i == d ? (InitialWorkerBelief / (1 - InitialWorkerBelief)) * (LabelCount - 1) : 1.0));
                scorePrior[d] = engine.Infer<VectorGaussian>(score);
            }

            return scorePrior;
        }

        /// <summary>
        /// CBCC posterior object.
        /// </summary>
        [Serializable]
        public new class Posteriors : BCC.Posteriors
        {
            /// <summary>
            /// The Dirichlet posteriors of the workers community membership.
            /// </summary>
            public Dirichlet CommunityProb;

            /// <summary>
            /// The posterior probabilities of the workers community membnerships.
            /// </summary>
            public Discrete[] Community;

            /// <summary>
            /// The Dirichlet posteriors of the community confusion matrix.
            /// </summary>
            public Dirichlet[][] CommunityConfusionMatrix;

            /// <summary>
            /// The Gaussian posteriors of the community score matrix.
            /// </summary>
            public VectorGaussian[][] CommunityScoreMatrix;

            /// <summary>
            /// The Gaussian constraint of the community score matrix (used for online training).
            /// </summary>
            public VectorGaussian[][] WorkerScoreMatrixConstraint;

            /// <summary>
            /// Theconstraint of the workers community membership (used for online training).
            /// </summary>
            public Discrete[] WorkerCommunityConstraint;
        }
    }
}
