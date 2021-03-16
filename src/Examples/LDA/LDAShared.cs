// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using Microsoft.ML.Probabilistic.Utilities;

namespace LDAExample
{
    using DirichletArray = DistributionRefArray<Dirichlet, Vector>;

    /// <summary>
    /// Latent Dirichlet Allocation (LDA) model implemented in Infer.NET.
    /// This version scales with number of documents.
    /// An optional parameter to the constructor specifies whether to use the
    /// fast version of the model (which uses power plates to deal efficiently
    /// with repeated words in the document) or the slower version where
    /// each word is considered separately. The only advantage of the latter
    /// is that it supports an evidence calculation.
    /// </summary>
    public class LDAShared : ILDA
    {
        /// <summary>
        /// Number of batches
        /// </summary>
        public int NumBatches { get; protected set; }

        /// <summary>
        /// Number of passes over the data
        /// </summary>
        public int NumPasses
        {
            get
            {
                return (IterationsPerPass == null) ? 0 : IterationsPerPass.Length;
            }
        }

        /// <summary>
        /// Number of iterations for each pass of the data
        /// </summary>
        public int[] IterationsPerPass { get; set; }

        /// <summary>
        /// Size of vocabulary
        /// </summary>
        public int SizeVocab { get; protected set; }

        /// <summary>
        /// Number of Topics
        /// </summary>
        public int NumTopics { get; protected set; }

        /// <summary>
        /// Sparsity specification for per-document distributions over topics
        /// </summary>
        public Sparsity ThetaSparsity { get; protected set; }

        /// <summary>
        /// Sparsity specification for per-topic distributions over words
        /// </summary>
        public Sparsity PhiSparsity { get; protected set; }

        /// <summary>
        /// Inference Engine for Phi definition model
        /// </summary>
        public InferenceEngine EnginePhiDef { get; protected set; }

        /// <summary>
        /// Main inference Engine
        /// </summary>
        public InferenceEngine Engine { get; protected set; }

        /// <summary>
        /// Shared variable array for per-topic word mixture variables - to be inferred
        /// </summary>
        protected SharedVariableArray<Vector> Phi;

        /// <summary>
        /// Shared variable for evidence
        /// </summary>
        protected SharedVariable<bool> Evidence;

        /// <summary>
        /// Model for documents (many copies)
        /// </summary>
        protected Model DocModel;

        /// <summary>
        /// Model for Phi definition (only one copy)
        /// </summary>
        protected Model PhiDefModel;

        /// <summary>
        /// Total number of documents (observed)
        /// </summary>
        protected Variable<int> NumDocuments;

        /// <summary>
        /// Number of words in each document (observed).
        /// For the fast version of the model, this is the number of unique words.
        /// </summary>
        protected VariableArray<int> NumWordsInDoc;

        /// <summary>
        /// Word indices in each document (observed)
        /// For the fast version of the model, these are the unique word indices.
        /// </summary>
        protected VariableArray<VariableArray<int>, int[][]> Words;

        /// <summary>
        /// Counts of unique words in each document (observed).
        /// This is used for the fast version only
        /// </summary>
        protected VariableArray<VariableArray<double>, double[][]> WordCounts;

        /// <summary>
        /// Per document distribution over topics (to be inferred)
        /// </summary>
        protected VariableArray<Vector> Theta;

        /// <summary>
        /// Copy of Phi for document model
        /// </summary>
        protected VariableArray<Vector> PhiDoc;

        /// <summary>
        /// Copy of Phi for definition model
        /// </summary>
        protected VariableArray<Vector> PhiDef;

        /// <summary>
        /// Prior for <see cref="Theta"/>
        /// </summary>
        protected VariableArray<Dirichlet> ThetaPrior;

        /// <summary>
        /// Prior for <see cref="Phi"/>
        /// </summary>
        protected VariableArray<Dirichlet> PhiPrior;

        /// <summary>
        /// Copy of model evidence variable for document model
        /// </summary>
        protected Variable<bool> EvidenceDoc;

        /// <summary>
        /// Copy of model evidence variable for phi definition model
        /// </summary>
        protected Variable<bool> EvidencePhiDef;

        /// <summary>
        /// Initialisation for breaking symmetry with respect to <see cref="Theta"/> (observed)
        /// </summary>
        protected VariableArray<Dirichlet> ThetaInit;

        /// <summary>
        /// Constructs an LDA model
        /// </summary>
        /// <param name="sizeVocab">Size of vocabulary</param>
        /// <param name="numTopics">Number of topics</param>
        public LDAShared(int numBatches, int sizeVocab, int numTopics)
        {
            SizeVocab = sizeVocab;
            NumTopics = numTopics;
            ThetaSparsity = Sparsity.Dense;
            PhiSparsity = Sparsity.ApproximateWithTolerance(0.00000000001); // Allow for round-off error
            NumDocuments = Variable.New<int>().Named("NumDocuments");
            NumBatches = numBatches;
            IterationsPerPass = new int[] { 1, 3, 5, 7, 9 };

            //---------------------------------------------
            // The model
            //---------------------------------------------
            Range D = new Range(NumDocuments).Named("D");
            Range W = new Range(SizeVocab).Named("W");
            Range T = new Range(NumTopics).Named("T");
            NumWordsInDoc = Variable.Array<int>(D).Named("NumWordsInDoc");
            Range WInD = new Range(NumWordsInDoc[D]).Named("WInD");

            Evidence = SharedVariable<bool>.Random(new Bernoulli(0.5)).Named("Evidence");
            Evidence.IsEvidenceVariable = true;

            Phi = SharedVariable<Vector>.Random(T, CreateUniformDirichletArray(numTopics, sizeVocab, PhiSparsity)).Named("Phi");

            // Phi definition sub-model - just one copy
            PhiDefModel = new Model(1).Named("PhiDefModel");

            IfBlock evidencePhiDefBlock = null;
            EvidencePhiDef = Evidence.GetCopyFor(PhiDefModel).Named("EvidencePhiDef");
            evidencePhiDefBlock = Variable.If(EvidencePhiDef);
            PhiDef = Variable.Array<Vector>(T).Named("PhiDef");
            PhiDef.SetSparsity(PhiSparsity);
            PhiDef.SetValueRange(W);
            PhiPrior = Variable.Array<Dirichlet>(T).Named("PhiPrior");
            PhiDef[T] = Variable<Vector>.Random(PhiPrior[T]);
            Phi.SetDefinitionTo(PhiDefModel, PhiDef);
            evidencePhiDefBlock.CloseBlock();

            // Document sub-model - many copies
            DocModel = new Model(numBatches).Named("DocModel");

            IfBlock evidenceDocBlock = null;
            EvidenceDoc = Evidence.GetCopyFor(DocModel).Named("EvidenceDoc");
            evidenceDocBlock = Variable.If(EvidenceDoc);
            Theta = Variable.Array<Vector>(D).Named("Theta");
            Theta.SetSparsity(ThetaSparsity);
            Theta.SetValueRange(T);
            ThetaPrior = Variable.Array<Dirichlet>(D).Named("ThetaPrior");
            Theta[D] = Variable<Vector>.Random(ThetaPrior[D]);
            PhiDoc = Phi.GetCopyFor(DocModel);
            PhiDoc.AddAttribute(new MarginalPrototype(Dirichlet.Uniform(sizeVocab, PhiSparsity)));
            Words = Variable.Array(Variable.Array<int>(WInD), D).Named("Words");
            WordCounts = Variable.Array(Variable.Array<double>(WInD), D).Named("WordCounts");
            using (Variable.ForEach(D))
            {
                using (Variable.ForEach(WInD))
                {
                    using (Variable.Repeat(WordCounts[D][WInD]))
                    {
                        Variable<int> topic = Variable.Discrete(Theta[D]).Named("topic");
                        using (Variable.Switch(topic))
                        {
                            Words[D][WInD] = Variable.Discrete(PhiDoc[topic]);
                        }
                    } 
                }
            }

            evidenceDocBlock.CloseBlock();

            // Initialization to break symmetry
            ThetaInit = Variable.Array<Dirichlet>(D).Named("ThetaInit");
            Theta[D].InitialiseTo(ThetaInit[D]);
            EnginePhiDef = new InferenceEngine(new VariationalMessagePassing());
            EnginePhiDef.Compiler.ShowWarnings = false;
            EnginePhiDef.ModelName = "LDASharedPhiDef";

            Engine = new InferenceEngine(new VariationalMessagePassing());
            Engine.OptimiseForVariables = new IVariable[] { Theta, PhiDoc, EvidenceDoc };

            Engine.Compiler.ShowWarnings = false;
            Engine.ModelName = "LDAShared";
            Engine.Compiler.ReturnCopies = false;
            Engine.Compiler.FreeMemory = true;
        }

        /// <summary>
        /// Runs inference on the LDA model. 
        /// <para>
        /// Words in documents are observed, topic distributions per document (<see cref="Theta"/>)
        /// and word distributions per topic (<see cref="Phi"/>) are inferred.
        /// </para>
        /// </summary>
        /// <param name="wordsInDoc">For each document, the unique word counts in the document</param>
        /// <param name="alpha">Hyper-parameter for <see cref="Theta"/></param>
        /// <param name="beta">Hyper-parameter for <see cref="Phi"/></param>
        /// <param name="postTheta">Posterior marginals for <see cref="Theta"/></param>
        /// <param name="postPhi">Posterior marginals for <see cref="Phi"/></param>
        /// <returns>Log evidence - can be used for model selection.</returns>
        public virtual double Infer(Dictionary<int, int>[] wordsInDoc, double alpha, double beta, out Dirichlet[] postTheta, out Dirichlet[] postPhi)
        {
            int numDocs = wordsInDoc.Length;
            var thetaPosterior = new Dirichlet[numDocs];
            int numIters = Engine.NumberOfIterations;
            bool showProgress = Engine.ShowProgress;
            Engine.ShowProgress = false; // temporarily disable Infer.NET progress

            // Set up document index boundaries for each batch
            double numDocsPerBatch = ((double)numDocs) / NumBatches;
            if (numDocsPerBatch == 0)
            {
                numDocsPerBatch = 1;
            }

            int[] boundary = new int[NumBatches + 1];
            boundary[0] = 0;
            double currBoundary = 0.0;
            for (int batch = 1; batch <= NumBatches; batch++)
            {
                currBoundary += numDocsPerBatch;
                int bnd = (int)currBoundary;
                if (bnd > numDocs)
                {
                    bnd = numDocs;
                }

                boundary[batch] = bnd;
            }

            boundary[NumBatches] = numDocs;

            PhiPrior.ObservedValue = new Dirichlet[NumTopics];
            for (int i = 0; i < NumTopics; i++)
            {
                PhiPrior.ObservedValue[i] = Dirichlet.Symmetric(SizeVocab, beta);
            }

            NumDocuments.ObservedValue = -1;
            try
            {
                for (int pass = 0; pass < NumPasses; pass++)
                {
                    Engine.NumberOfIterations = IterationsPerPass[pass];
                    if (showProgress)
                    {
                        Console.Write(String.Format(
                        "\nPass {0} ({1} iteration{2} per batch)",
                        pass, IterationsPerPass[pass], IterationsPerPass[pass] == 1 ? "" : "s"));
                    }

                    PhiDefModel.InferShared(EnginePhiDef, 0);
                    for (int batch = 0; batch < NumBatches; batch++)
                    {
                        int startDoc = boundary[batch];
                        int endDoc = boundary[batch + 1];
                        if (startDoc >= numDocs)
                        {
                            break;
                        }

                        int numDocsInThisBatch = endDoc - startDoc;

                        // Set up the observed values
                        if (NumDocuments.ObservedValue != numDocsInThisBatch)
                        {
                            NumDocuments.ObservedValue = numDocsInThisBatch;

                            ThetaPrior.ObservedValue = new Dirichlet[numDocsInThisBatch];
                            for (int i = 0; i < numDocsInThisBatch; i++)
                            {
                                ThetaPrior.ObservedValue[i] = Dirichlet.Symmetric(NumTopics, alpha);
                            }
                        }
                        if (pass == 0)
                        {
                            ThetaInit.ObservedValue = LDAModel.GetInitialisation(numDocsInThisBatch, NumTopics, ThetaSparsity);
                        }
                        else
                        {
                            ThetaInit.ObservedValue = Util.ArrayInit(numDocsInThisBatch, d => new Dirichlet(thetaPosterior[d + startDoc]));
                        }


                        int[] numWordsInDocBatch = new int[numDocsInThisBatch];
                        int[][] wordsInDocBatch = new int[numDocsInThisBatch][];
                        double[][] wordCountsInDocBatch = new double[numDocsInThisBatch][];
                        for (int i = 0, j = startDoc; j < endDoc; i++, j++)
                        {
                            numWordsInDocBatch[i] = wordsInDoc[j].Count;
                            wordsInDocBatch[i] = wordsInDoc[j].Keys.ToArray();
                            ICollection<int> cnts = wordsInDoc[j].Values;
                            wordCountsInDocBatch[i] = new double[cnts.Count];
                            int k = 0;
                            foreach (int val in cnts)
                            {
                                wordCountsInDocBatch[i][k++] = (double)val;
                            }
                        }

                        NumWordsInDoc.ObservedValue = numWordsInDocBatch;
                        Words.ObservedValue = wordsInDocBatch;
                        WordCounts.ObservedValue = wordCountsInDocBatch;

                        DocModel.InferShared(Engine, batch);
                        var postThetaBatch = Engine.Infer<Dirichlet[]>(Theta);
                        for (int i = 0, j = startDoc; j < endDoc; i++, j++)
                        {
                            thetaPosterior[j] = postThetaBatch[i];
                        }

                        if (showProgress)
                        {
                            if ((batch % 80) == 0)
                            {
                                Console.WriteLine("");
                            }

                            Console.Write(".");
                        }
                    }
                }
            }
            finally
            {
                Engine.NumberOfIterations = numIters;
                Engine.ShowProgress = showProgress;
            }

            if (showProgress)
            {
                Console.WriteLine();
            }

            postTheta = thetaPosterior;
            postPhi = Phi.Marginal<Dirichlet[]>();

            return Model.GetEvidenceForAll(PhiDefModel, DocModel);
        }

        /// <summary>
        /// Creates a uniform distribution array over Dirichlets
        /// </summary>
        /// <param name="length">Length of array</param>
        /// <param name="valueLength">Dimension of each Dirichlet</param>
        /// <returns></returns>
        private static DirichletArray CreateUniformDirichletArray(
            int length, int valueLength, Sparsity sparsity)
        {
            return new DirichletArray(length, i => Dirichlet.Uniform(valueLength, sparsity));
        }
    }
}
