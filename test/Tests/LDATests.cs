// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Xunit;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using Range = Microsoft.ML.Probabilistic.Models.Range;

    internal class LDAModel
    {
        // Following are all observed
        public Variable<int> TotalTopics = Variable.New<int>().Named("NumTopics");
        public Variable<int> TotalDocuments = Variable.New<int>().Named("NumDocuments");
        public Variable<int> TotalWords = Variable.New<int>().Named("NumWords");
        public VariableArray<int> NumWordsInDoc;
        public VariableArray<VariableArray<int>, int[][]> Words;
        public Variable<Dirichlet> AlphaPrior = Variable.New<Dirichlet>().Named("AlphaPrior");
        public Variable<Dirichlet> BetaPrior = Variable.New<Dirichlet>().Named("BetaPrior");
        public Variable<IDistribution<Vector[]>> PhiInit = Variable.New<IDistribution<Vector[]>>().Named("PhiInit");
        // These are latent
        public VariableArray<Vector> Phi;
        public VariableArray<Vector> Theta;
        public InferenceEngine Engine;
        // The posteriors
        public Dirichlet[] PostPhi;
        public Dirichlet[] PostTheta;

        public LDAModel()
        {
            //----------------------------------------------
            // Ranges
            //----------------------------------------------
            // Constant ranges
            Range D = new Range(TotalDocuments).Named("D");
            Range W = new Range(TotalWords).Named("W");
            Range T = new Range(TotalTopics).Named("T");
            NumWordsInDoc = Variable.Array<int>(D).Named("NumWordsInDoc");
            Range WInD = new Range(NumWordsInDoc[D]).Named("WInD");

            //----------------------------------------------
            // Data
            //----------------------------------------------
            Words = Variable.Array(Variable.Array<int>(WInD), D).Named("Words");

            //---------------------------------------------
            // The model
            //---------------------------------------------
            Theta = Variable.Array<Vector>(D).Named("Theta").Attrib(new ValueRange(T));
            Theta[D] = Variable.Random<Vector, Dirichlet>(AlphaPrior).ForEach(D);
            Phi = Variable.Array<Vector>(T).Named("Phi").Attrib(new ValueRange(W));
            Phi[T] = Variable.Random<Vector, Dirichlet>(BetaPrior).ForEach(T);
            Phi.InitialiseTo(PhiInit);

            using (Variable.ForEach(D))
            {
                using (Variable.ForEach(WInD))
                {
                    var topic = Variable.Discrete(Theta[D]).Attrib(new ValueRange(T)).Named("topic");
                    using (Variable.Switch(topic))
                        Words[D][WInD] = Variable.Discrete(Phi[topic]);
                }
            }
            // Inference engine
            Engine = new InferenceEngine(new VariationalMessagePassing());
        }

        private IDistribution<Vector[]> GetInitialisation(double initMaxPseudoCount, double initWordsPerTopic, Sparsity sparsity, double beta)
        {
            Dirichlet[] initPhi = new Dirichlet[TotalTopics.ObservedValue];
            Random r = new Random(12347);
            for (int i = 0; i < TotalTopics.ObservedValue; i++)
            {
                Vector v = Vector.Constant(TotalWords.ObservedValue, beta, sparsity);
                int[] perm = Rand.Perm(TotalWords.ObservedValue);
                int numWords = Poisson.Sample(initWordsPerTopic);
                for (int j = 0; j < numWords; j++)
                    v[perm[j]] += initMaxPseudoCount*r.NextDouble();
                initPhi[i] = new Dirichlet(v);
            }
            return Distribution<Vector>.Array(initPhi);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>
        /// Run inference on the LDA model
        /// </summary>
        /// <param name="initMaxPseudoCount">Max psudo-count for initialisation</param>
        /// <param name="alpha">Pseudo-counts for theta</param>
        /// <param name="beta">Pseudo-counts for phi</param>
        public void Infer(double initMaxPseudoCount, int numVocab, int numTopics, int[][] wordsInDoc, double alpha, double beta, Sparsity alphaSparsity, Sparsity betaSparsity,
                          Sparsity phiSparsity)
        {
            // Set up the observed values
            TotalWords.ObservedValue = numVocab;
            TotalDocuments.ObservedValue = wordsInDoc.Length;
            TotalTopics.ObservedValue = numTopics;
            int[] numWordsInDoc = new int[wordsInDoc.Length];
            for (int i = 0; i < wordsInDoc.Length; i++)
                numWordsInDoc[i] = wordsInDoc[i].Length;
            NumWordsInDoc.ObservedValue = numWordsInDoc;
            Words.ObservedValue = wordsInDoc;
            Vector alphaVector = Vector.Constant(numTopics, alpha, alphaSparsity);
            Vector betaVector = Vector.Constant(numVocab, beta, betaSparsity);

            AlphaPrior.ObservedValue = new Dirichlet(alphaVector);
            BetaPrior.ObservedValue = new Dirichlet(betaVector);
            // This has to occur after the other observed values are set:
            PhiInit.ObservedValue = GetInitialisation(initMaxPseudoCount, numVocab/numTopics, phiSparsity, beta);

            if (false)
            {
                // for debugging, put the observed values in the code.
                TotalWords.IsReadOnly = true;
                TotalDocuments.IsReadOnly = true;
                TotalTopics.IsReadOnly = true;
                NumWordsInDoc.IsReadOnly = true;
                Words.IsReadOnly = true;
                AlphaPrior.IsReadOnly = true;
                BetaPrior.IsReadOnly = true;
                PhiInit.IsReadOnly = true;
            }

            Engine.OptimiseForVariables = new List<IVariable>() {Theta, Phi};
            PostTheta = Engine.Infer<Dirichlet[]>(Theta);
            PostPhi = Engine.Infer<Dirichlet[]>(Phi);
        }


#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <summary>
        /// For each topic, count how many documents have that topic as their
        /// primary topic
        /// </summary>
        /// <returns></returns>
        public Vector CountTopicSize()
        {
            Vector counts = Vector.Zero(TotalTopics.ObservedValue);
            for (int i = 0; i < TotalTopics.ObservedValue; i++)
                counts[i] = 0;
            foreach (Dirichlet d in PostTheta)
                counts[d.GetMean().IndexOfMaximum()]++;
            return counts;
        }

        /// <summary>
        /// For each word, find the most likely topic
        /// </summary>
        /// <returns></returns>
        public Vector ComputeArgmaxTopicOfWord()
        {
            Vector argmaxTOfW = Vector.Zero(TotalWords.ObservedValue);
            // probability of a topic P(t)=sum_t P(t|d)*P(d) but disregarding P(d)=1/|D|
            Vector probT = Vector.Zero(TotalTopics.ObservedValue);
            foreach (Dirichlet d in PostTheta)
                probT += d.GetMean();

            for (int i = 0; i < TotalWords.ObservedValue; i++)
            {
                // Probability of this word given a topic
                Vector TOfW = Vector.Zero(TotalTopics.ObservedValue);
                int t = 0;
                foreach (Dirichlet d in PostPhi)
                {
                    TOfW[t] = d.GetMean()[i];
                    t++;
                }

                TOfW *= probT;
                argmaxTOfW[i] = 0;
                double maxCOfM = 0;
                for (int j = 0; j < TotalTopics.ObservedValue; j++)

                    if (maxCOfM < TOfW[j])
                    {
                        maxCOfM = TOfW[j];
                        argmaxTOfW[i] = j;
                    }
            }
            return argmaxTOfW;
        }
    }

    /// <summary>
    /// Summary description for LDATests
    /// </summary>
    public class LDATests
    {
        // Generate toy data - returns indices into vocab for each doc
        private int[][] GenerateToyLDAData(
            int numTopics, int numVocab, int numDocs, int expectedDocLength,
            out Dirichlet[] trueTheta, out Dirichlet[] truePhi)
        {
            truePhi = new Dirichlet[numTopics];
            for (int i = 0; i < numTopics; i++)
            {
                truePhi[i] = Dirichlet.Uniform(numVocab);
                truePhi[i].PseudoCount.SetAllElementsTo(0.0);
                // Draw the number of unique words in the topic
                int numUniqueWordsPerTopic = Poisson.Sample((double) numVocab/numTopics);
                if (numUniqueWordsPerTopic >= numVocab) numUniqueWordsPerTopic = numVocab;
                double expectedRepeatOfWordInTopic =
                    ((double) numDocs)*expectedDocLength/numUniqueWordsPerTopic;
                int[] shuffledWordIndices = Rand.Perm(numVocab);
                for (int j = 0; j < numUniqueWordsPerTopic; j++)
                {
                    int wordIndex = shuffledWordIndices[j];
                    // Draw the count for that word
                    int cnt = Poisson.Sample(expectedRepeatOfWordInTopic);
                    truePhi[i].PseudoCount[wordIndex] = cnt + 1.0;
                }
            }

            trueTheta = new Dirichlet[numDocs];
            for (int i = 0; i < numDocs; i++)
            {
                trueTheta[i] = Dirichlet.Uniform(numTopics);
                trueTheta[i].PseudoCount.SetAllElementsTo(0.0);
                // Draw the number of unique topics in the doc. We expect this to be
                // very sparse
                int numUniqueTopicsPerDoc = System.Math.Min(1 + Poisson.Sample(1.0), numTopics);
                double expectedRepeatOfTopicInDoc =
                    expectedDocLength/numUniqueTopicsPerDoc;
                int[] shuffledTopicIndices = Rand.Perm(numTopics);
                for (int j = 0; j < numUniqueTopicsPerDoc; j++)
                {
                    int topicIndex = shuffledTopicIndices[j];
                    // Draw the count for that topic
                    int cnt = Poisson.Sample(expectedRepeatOfTopicInDoc);
                    trueTheta[i].PseudoCount[topicIndex] = cnt + 1.0;
                }
            }

            // Sample from the model
            Vector[] topicDist = new Vector[numDocs];
            Vector[] wordDist = new Vector[numTopics];
            for (int i = 0; i < numDocs; i++)
                topicDist[i] = trueTheta[i].Sample();
            for (int i = 0; i < numTopics; i++)
                wordDist[i] = truePhi[i].Sample();


            int[][] wordsInDoc = new int[numDocs][];
            for (int i = 0; i < numDocs; i++)
            {
                int LengthOfDoc = Poisson.Sample((double) expectedDocLength);
                wordsInDoc[i] = new int[LengthOfDoc];
                for (int j = 0; j < LengthOfDoc; j++)
                {
                    int topic = Discrete.Sample(topicDist[i]);
                    wordsInDoc[i][j] = Discrete.Sample(wordDist[topic]);
                }
            }
            return wordsInDoc;
        }

        /// <summary>
        /// Compare sparse (exact) and dense LDA models - these should give identical results
        /// </summary>
        [Fact]
        public void LDATest()
        {
            int numTopics = 2;
            int numVocab = 10;
            int numDocs = 5;
            double alpha = 1.0/numTopics;
            double beta = 1.0/numVocab;
            Dirichlet[] truePhi, trueTheta;
            Rand.Restart(12347);
            int[][] wordsInDoc = GenerateToyLDAData(numTopics, numVocab, numDocs, 100,
                                                    out trueTheta, out truePhi);

            Rand.Restart(12347);
            var modelDense = new LDAModel();
            modelDense.Engine.NumberOfIterations = 15;
            modelDense.Engine.ModelName = "LdaDense";
            modelDense.Infer(10.0, numVocab, numTopics, wordsInDoc, alpha, beta, Sparsity.Dense, Sparsity.Dense, Sparsity.Dense);

            // Same model, but run as sparse
            Rand.Restart(12347);
            var modelSparse = new LDAModel();
            modelSparse.Engine.NumberOfIterations = 15;
            modelSparse.Engine.ModelName = "LdaSparse";
            modelSparse.Engine.Compiler.ReturnCopies = false;
            // previous tolerances appeared to be set to alpha and beta but were in fact set to dense due to a downstream bug.
            modelSparse.Infer(10.0, numVocab, numTopics, wordsInDoc, alpha, beta, Sparsity.ApproximateWithTolerance(1e-7), Sparsity.ApproximateWithTolerance(1e-7),
                              Sparsity.ApproximateWithTolerance(1e-6));

            Console.WriteLine(StringUtil.JoinColumns("Phi sparsity = ", SparsityFraction(modelSparse.PostPhi)));
            Console.WriteLine(StringUtil.JoinColumns("Theta sparsity = ", SparsityFraction(modelSparse.PostTheta)));
            for (int i = 0; i < numDocs; i++)
            {
                Assert.Equal(Sparsity.Dense, modelDense.PostTheta[i].Sparsity);
                //Assert.True(modelSparse.PostTheta[i].Sparsity.IsApproximate);
                Assert.Equal(0.0, modelDense.PostTheta[i].MaxDiff(modelSparse.PostTheta[i]), 1e-3);
            }
            for (int i = 0; i < numTopics; i++)
            {
                Assert.Equal(Sparsity.Dense, modelDense.PostPhi[i].Sparsity);
                // Assert.True(modelSparse.PostPhi[i].Sparsity.IsApproximate);
                Assert.Equal(0.0, modelDense.PostPhi[i].MaxDiff(modelSparse.PostPhi[i]), 1e-3);
            }
        }

        public static double SparsityFraction(object obj)
        {
            if (obj is Vector) return SparsityFraction((Vector) obj);
            else if (obj is Dirichlet) return SparsityFraction(((Dirichlet) obj).PseudoCount);
            else if (obj is Discrete) return SparsityFraction(((Discrete) obj).GetProbs());
            else
            {
                Type type = obj.GetType();
                if (Util.IsIList(type))
                {
                    return (double)Compiler.Reflection.Invoker.InvokeStatic(typeof (LDATests), "SparsityFraction", obj);
                }
                else throw new NotImplementedException();
            }
        }

        public static double SparsityFraction<T>(IEnumerable<T> array)
        {
            double sum = 0.0;
            int count = 0;
            foreach (T item in array)
            {
                sum += SparsityFraction(item);
                count++;
            }
            return sum/count;
        }

        public static double SparsityFraction(Vector vector)
        {
            if (vector is SparseVector)
            {
                SparseVector sv = (SparseVector) vector;
                return (double) sv.SparseValues.Count/sv.Count;
            }
            else return 1.0;
        }

        public static double MaxDiff(object a, object b)
        {
            if ((a is Diffable) || Util.IsIList(a.GetType()))
            {
                return (double)Compiler.Reflection.Invoker.InvokeStatic(typeof (LDATests), "MaxDiff", a, b);
            }
            else throw new NotImplementedException();
        }

        public static double MaxDiff(Diffable a, object b)
        {
            return a.MaxDiff(b);
        }

        public static double MaxDiff<T, U>(IList<T> a, IList<U> b)
        {
            if (a.Count != b.Count) return Double.PositiveInfinity;
            double diff = 0.0;
            for (int i = 0; i < a.Count; i++)
            {
                diff = System.Math.Max(diff, MaxDiff(a[i], b[i]));
            }
            return diff;
        }
    }
}