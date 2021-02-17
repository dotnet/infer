// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
#define newversion
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Utilities;

namespace LDAExample
{
    public interface ILDA
    {
        double Infer(Dictionary<int, int>[] wordsInDoc, double alpha, double beta, out Dirichlet[] postTheta, out Dirichlet[] postPhi);

        InferenceEngine Engine { get; }
    }

    /// <summary>
    /// Latent Dirichlet Allocation (LDA) model implemented in Infer.NET.
    /// It keeps all messages in memory, and so scales poorly with respect to
    /// number of documents.
    /// An optional parameter to the constructor specifies whether to use the
    /// fast version of the model (which uses power plates to deal efficiently
    /// with repeated words in the document) or the slower version where
    /// each word is considered separately. The only advantage of the latter
    /// is that it supports an evidence calculation.
    /// See <see cref="LDAShared"/> for a version which scales better with number
    /// of documents.
    /// </summary>
    public class LDAModel : ILDA
    {
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
        /// Inference engine
        /// </summary>
        public InferenceEngine Engine { get; protected set; }

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
        /// Per topic distribution over words (to be inferred)
        /// </summary>
        protected VariableArray<Vector> Phi;

        /// <summary>
        /// Prior for <see cref="Theta"/>
        /// </summary>
        protected VariableArray<Dirichlet> ThetaPrior;

        /// <summary>
        /// Prior for <see cref="Phi"/>
        /// </summary>
        protected VariableArray<Dirichlet> PhiPrior;

        /// <summary>
        /// Model evidence
        /// </summary>
        protected Variable<bool> Evidence;

        /// <summary>
        /// Initialisation for breaking symmetry with respect to <see cref="Theta"/> (observed)
        /// </summary>
        protected VariableArray<Dirichlet> ThetaInit;

        /// <summary>
        /// Constructs an LDA model
        /// </summary>
        /// <param name="sizeVocab">Size of vocabulary</param>
        /// <param name="numTopics">Number of topics</param>
        public LDAModel(int sizeVocab, int numTopics)
        {
            SizeVocab = sizeVocab;
            NumTopics = numTopics;
            ThetaSparsity = Sparsity.Dense;
            PhiSparsity = Sparsity.ApproximateWithTolerance(0.00000000001); // Allow for round-off error
            NumDocuments = Variable.New<int>().Named("NumDocuments");

            //---------------------------------------------
            // The model
            //---------------------------------------------
            Range D = new Range(NumDocuments).Named("D");
            Range W = new Range(SizeVocab).Named("W");
            Range T = new Range(NumTopics).Named("T");
            NumWordsInDoc = Variable.Array<int>(D).Named("NumWordsInDoc");
            Range WInD = new Range(NumWordsInDoc[D]).Named("WInD");

            // Surround model by a stochastic If block so that we can compute model evidence
            Evidence = Variable.Bernoulli(0.5).Named("Evidence");
            IfBlock evidenceBlock = Variable.If(Evidence);

            Theta = Variable.Array<Vector>(D);
            Theta.SetSparsity(ThetaSparsity);
            Theta.SetValueRange(T);
            ThetaPrior = Variable.Array<Dirichlet>(D).Named("ThetaPrior");
            Theta[D] = Variable<Vector>.Random(ThetaPrior[D]);
            Phi = Variable.Array<Vector>(T); 
            Phi.SetSparsity(PhiSparsity);
            Phi.SetValueRange(W);
            PhiPrior = Variable.Array<Dirichlet>(T).Named("PhiPrior");
            Phi[T] = Variable<Vector>.Random(PhiPrior[T]);
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
                            Words[D][WInD] = Variable.Discrete(Phi[topic]);
                        }
                    }
                }
            }

            evidenceBlock.CloseBlock();

            ThetaInit = Variable.Array<Dirichlet>(D).Named("ThetaInit");
            Theta[D].InitialiseTo(ThetaInit[D]);
            Engine = new InferenceEngine(new VariationalMessagePassing());
            Engine.Compiler.ShowWarnings = false;
            Engine.ModelName = "LDAModel";
        }

        /// <summary>
        /// Gets random initialisation for <see cref="Theta"/>. This initialises downward messages from <see cref="Theta"/>.
        /// The sole purpose is to break symmetry in the inference - it does not change the model.
        /// </summary>
        /// <param name="sparsity">The sparsity settings</param>
        /// <returns></returns>
        /// <remarks>This is implemented so as to support sparse initialisations</remarks>
        public static Dirichlet[] GetInitialisation(
            int numDocs, int numTopics, Sparsity sparsity)
        {
            return Util.ArrayInit(numDocs, i =>
            {
                // Choose a random topic
                Vector v = Vector.Zero(numTopics, sparsity);
                int topic = Rand.Int(numTopics);
                v[topic] = 1.0;
                return Dirichlet.PointMass(v);
            });
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
            // Set up the observed values
            int numDocs = wordsInDoc.Length;
            NumDocuments.ObservedValue = numDocs;

            int[] numWordsInDoc = new int[numDocs];
            int[][] wordIndices = new int[numDocs][];
            double[][] wordCounts = new double[numDocs][];
            for (int i = 0; i < numDocs; i++)
            {
                numWordsInDoc[i] = wordsInDoc[i].Count;
                wordIndices[i] = wordsInDoc[i].Keys.ToArray();
                ICollection<int> cnts = wordsInDoc[i].Values;
                wordCounts[i] = new double[cnts.Count];
                int k = 0;
                foreach (int val in cnts)
                {
                    wordCounts[i][k++] = (double)val;
                }
            }

            NumWordsInDoc.ObservedValue = numWordsInDoc;
            Words.ObservedValue = wordIndices;
            WordCounts.ObservedValue = wordCounts;
            ThetaInit.ObservedValue = GetInitialisation(numDocs, NumTopics, ThetaSparsity);
            ThetaPrior.ObservedValue = new Dirichlet[numDocs];
            for (int i = 0; i < numDocs; i++)
            {
                ThetaPrior.ObservedValue[i] = Dirichlet.Symmetric(NumTopics, alpha);
            }

            PhiPrior.ObservedValue = new Dirichlet[NumTopics];
            for (int i = 0; i < NumTopics; i++)
            {
                PhiPrior.ObservedValue[i] = Dirichlet.Symmetric(SizeVocab, beta);
            }

            Engine.OptimiseForVariables = new IVariable[] { Theta, Phi, Evidence };
            postTheta = Engine.Infer<Dirichlet[]>(Theta);
            postPhi = Engine.Infer<Dirichlet[]>(Phi);
            return Engine.Infer<Bernoulli>(Evidence).LogOdds;
        }
    }
}
