// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace LDAExample
{
    /// <summary>
    /// Latent Dirichlet Allocation (LDA) prediction model implemented in Infer.NET.
    /// Use this class for obtaining predictive distributions over words for
    /// documents with known topic distributions
    /// </summary>
    public class LDAPredictionModel
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
        /// Sparsity specification for predictive distributions over words
        /// </summary>
        public Sparsity PredictionSparsity { get; protected set; }

        /// <summary>
        /// Inference engine
        /// </summary>
        public InferenceEngine Engine { get; protected set; }

        /// <summary>
        /// A predicted word
        /// </summary>
        protected Variable<int> Word;

        /// <summary>
        /// Per document distribution over topics (to be inferred)
        /// </summary>
        protected Variable<Vector> Theta;

        /// <summary>
        /// Per topic distribution over words (to be inferred)
        /// </summary>
        protected VariableArray<Vector> Phi;

        /// <summary>
        /// Prior for <see cref="Theta"/>
        /// </summary>
        protected Variable<Dirichlet> ThetaPrior;

        /// <summary>
        /// Prior for <see cref="Phi"/>
        /// </summary>
        protected VariableArray<Dirichlet> PhiPrior;

        /// <summary>
        /// Constructs an LDA model
        /// </summary>
        /// <param name="sizeVocab">Size of vocabulary</param>
        /// <param name="numTopics">Number of topics</param>
        public LDAPredictionModel(int sizeVocab, int numTopics)
        {
            SizeVocab = sizeVocab;
            NumTopics = numTopics;
            PredictionSparsity = Sparsity.Sparse;

            //---------------------------------------------
            // The model
            //---------------------------------------------
            Range W = new Range(SizeVocab).Named("W");
            Range T = new Range(NumTopics).Named("T");

            ThetaPrior = Variable.New<Dirichlet>().Named("ThetaPrior");
            PhiPrior = Variable.Array<Dirichlet>(T).Named("PhiPrior").Attrib(new ValueRange(W));
            Theta = Variable.New<Vector>().Named("Theta");
            Phi = Variable.Array<Vector>(T).Named("Phi");
            Theta = Variable.Random<Vector, Dirichlet>(ThetaPrior);
            Phi[T] = Variable.Random<Vector, Dirichlet>(PhiPrior[T]);

            Word = Variable.New<int>().Named("Word");
            Word.SetSparsity(PredictionSparsity);
            var topic = Variable.Discrete(Theta).Attrib(new ValueRange(T)).Named("topic");
            using (Variable.Switch(topic))
            {
                Word.SetTo(Variable.Discrete(Phi[topic]));
            }

            Engine = new InferenceEngine(new VariationalMessagePassing());
            Engine.Compiler.ShowWarnings = false;
        }

        /// <summary>
        /// Gets the predictive distributions for a set of documents
        /// <para>
        /// Topic distributions per document (<see cref="Theta"/>) and word distributions
        /// per topic (<see cref="Phi"/>) are observed, document distributions over words
        /// are inferred.
        /// </para>
        /// </summary>
        /// <param name="postTheta">The posterior document topic distributions</param>
        /// <param name="postPhi">The posterior topic word distributions</param>
        /// <returns>The predictive distribution over words for each document</returns>
        public virtual Discrete[] Predict(Dirichlet[] postTheta, Dirichlet[] postPhi)
        {
            int numVocab = postPhi[0].Dimension;
            int numTopics = postPhi.Length;
            int numDocs = postTheta.Length;
            Discrete[] result = new Discrete[numDocs];
            bool showProgress = Engine.ShowProgress;
            Engine.ShowProgress = false;
            try
            {
                for (int i = 0; i < numDocs; i++)
                {
                    ThetaPrior.ObservedValue = postTheta[i];
                    PhiPrior.ObservedValue = postPhi;
                    result[i] = Engine.Infer<Discrete>(Word);
                    if (showProgress)
                    {
                        if ((i % 80) == 0)
                        {
                            Console.WriteLine("");
                        }

                        Console.Write(".");
                    }
                }
            }
            finally
            {
                Engine.ShowProgress = showProgress;
            }

            if (showProgress)
            {
                Console.WriteLine();
            }

            return result;
        }
    }
}
