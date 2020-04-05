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

namespace LDAExample
{
    /// <summary>
    /// Latent Dirichlet Allocation (LDA) prediction model implemented in Infer.NET.
    /// Use this class for obtaining predictive distributions over words for
    /// documents with known topic distributions
    /// </summary>
    public class LDATopicInferenceModel
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
        /// Inference engine
        /// </summary>
        public InferenceEngine Engine { get; protected set; }

        protected Variable<int> NumDocuments;

        /// <summary>
        /// Number of words in the document (observed).
        /// For the fast version of the model, this is the number of unique words.
        /// </summary>
        protected Variable<int> NumWordsInDoc;

        /// <summary>
        /// Word indices in the document (observed)
        /// For the fast version of the model, these are the unique word indices.
        /// </summary>
        protected VariableArray<int> Words;

        /// <summary>
        /// Counts of unique words in the document (observed).
        /// This is used for the fast version only
        /// </summary>
        protected VariableArray<double> WordCounts;

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
        public LDATopicInferenceModel(
            int sizeVocab, 
            int numTopics)
        {
            SizeVocab = sizeVocab;
            NumTopics = numTopics;

            //---------------------------------------------
            // The model
            //---------------------------------------------
            NumWordsInDoc = Variable.New<int>().Named("NumWordsInDoc");
            Range W = new Range(SizeVocab).Named("W");
            Range T = new Range(NumTopics).Named("T");
            Range WInD = new Range(NumWordsInDoc).Named("WInD");

            Theta = Variable.New<Vector>().Named("Theta");
            ThetaPrior = Variable.New<Dirichlet>().Named("ThetaPrior");
            ThetaPrior.SetValueRange(T);
            Theta = Variable<Vector>.Random(ThetaPrior);
            PhiPrior = Variable.Array<Dirichlet>(T).Named("PhiPrior");
            PhiPrior.SetValueRange(W);
            Phi = Variable.Array<Vector>(T).Named("Phi");
            Phi[T] = Variable.Random<Vector, Dirichlet>(PhiPrior[T]);

            Words = Variable.Array<int>(WInD).Named("Words");
            WordCounts = Variable.Array<double>(WInD).Named("WordCounts");
            using (Variable.ForEach(WInD))
            {
                using (Variable.Repeat(WordCounts[WInD]))
                {
                    var topic = Variable.Discrete(Theta).Attrib(new ValueRange(T)).Named("topic");
                    topic.SetValueRange(T);
                    using (Variable.Switch(topic))
                    {
                        Words[WInD] = Variable.Discrete(Phi[topic]);
                    }
                }
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
        /// <param name="alpha">Hyper-parameter for <see cref="Theta"/></param>
        /// <param name="postPhi">The posterior topic word distributions</param>
        /// <param name="wordsInDoc">The unique word counts in the documents</param>
        /// <returns>The predictive distribution over words for each document</returns>
        public virtual Dirichlet[] InferTopic(
            double alpha, 
            Dirichlet[] postPhi, 
            Dictionary<int, int>[] wordsInDoc)
        {
            int numVocab = postPhi[0].Dimension;
            int numTopics = postPhi.Length;
            int numDocs = wordsInDoc.Length;
            Dirichlet[] result = new Dirichlet[numDocs];
            bool showProgress = Engine.ShowProgress;
            Engine.ShowProgress = false;
            PhiPrior.ObservedValue = postPhi;
            ThetaPrior.ObservedValue = Dirichlet.Symmetric(numTopics, alpha);

            try
            {
                for (int i = 0; i < numDocs; i++)
                {
                    NumWordsInDoc.ObservedValue = wordsInDoc[i].Count;
                    Words.ObservedValue = wordsInDoc[i].Keys.ToArray();
                    ICollection<int> cnts = wordsInDoc[i].Values;
                    var wordCounts = new double[cnts.Count];
                    int k = 0;
                    foreach (int val in cnts)
                    {
                        wordCounts[k++] = (double)val;
                    }

                    WordCounts.ObservedValue = wordCounts;

                    result[i] = Engine.Infer<Dirichlet>(Theta);
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
