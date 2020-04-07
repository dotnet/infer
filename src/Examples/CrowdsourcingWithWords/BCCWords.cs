// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using System;
using System.Linq;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace CrowdsourcingWithWords
{
    /// <summary>
    /// The BCCWords model
    /// </summary>
    public class BCCWords : BCC
    {
        // Add extra ranges
        private Range w;
        private Range nw;

        // Model evidence
        private Variable<bool> evidence;

        // Additional variables for BCCWords
        private VariableArray<Vector> ProbWord;
        private VariableArray<int> WordCount;
        private VariableArray<VariableArray<int>, int[][]> Words;
        private Variable<Dirichlet> ProbWordPrior;

        public void CreateModel(int NumTasks, int NumClasses, int VocabSize, int numBatches = 3)
        {
            WorkerCount = Variable.New<int>().Named("WorkerCount");

            // Set up inference engine
            Engine = new InferenceEngine(new VariationalMessagePassing());

            // Set engine flags
            Engine.Compiler.WriteSourceFiles = true;
            Engine.Compiler.UseParallelForLoops = true;

            evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);

            // Set up ranges
            n = new Range(NumTasks).Named("N");
            c = new Range(NumClasses).Named("C");
            k = new Range(WorkerCount).Named("K");
            WorkerTaskCount = Variable.Array<int>(k).Named("WorkerTaskCount");
            kn = new Range(WorkerTaskCount[k]).Named("KN");
            WorkerTaskIndex = Variable.Array(Variable.Array<int>(kn), k).Named("Task");
            WorkerTaskIndex.SetValueRange(n);

            // Initialise truth
            BackgroundLabelProbPrior = Variable.New<Dirichlet>().Named("TruthProbPrior");
            BackgroundLabelProb = Variable<Vector>.Random(BackgroundLabelProbPrior).Named("TruthProb");
            BackgroundLabelProb.SetValueRange(c);

            // Truth distributions
            TrueLabel = Variable.Array<int>(n).Named("Truth");
            TrueLabel[n] = Variable.Discrete(BackgroundLabelProb).ForEach(n);

            //VocabSize = Variable.New<int>();
            w = new Range(VocabSize).Named("W");
            ProbWord = Variable.Array<Vector>(c).Named("ProbWord");
            ProbWord.SetValueRange(w);
            WordCount = Variable.Array<int>(n).Named("WordCount");
            nw = new Range(WordCount[n]).Named("WN");
            Words = Variable.Array(Variable.Array<int>(nw), n).Named("Word");
            ProbWordPrior = Variable.New<Dirichlet>().Named("ProbWordPrior");
            ProbWord[c] = Variable<Vector>.Random(ProbWordPrior).ForEach(c);

            // Initialise user profiles
            ConfusionMatrixPrior = Variable.Array(Variable.Array<Dirichlet>(c), k).Named("WorkerConfusionMatrixPrior");
            WorkerConfusionMatrix = Variable.Array(Variable.Array<Vector>(c), k).Named("WorkerConfusionMatrix");
            WorkerConfusionMatrix[k][c] = Variable<Vector>.Random(ConfusionMatrixPrior[k][c]);
            WorkerConfusionMatrix.SetValueRange(c);

            // Vote distributions
            WorkerLabel = Variable.Array(Variable.Array<int>(kn), k).Named("WorkerLabel");

            using (Variable.ForEach(k))
            {
                var trueLabel = Variable.Subarray(TrueLabel, WorkerTaskIndex[k]).Named("TrueLabelSubarray");
                trueLabel.SetValueRange(c);
                using (Variable.ForEach(kn))
                {
                    using (Variable.Switch(trueLabel[kn]))
                    {
                        WorkerLabel[k][kn] = Variable.Discrete(WorkerConfusionMatrix[k][trueLabel[kn]]);
                    }
                }
            }

            // Words inference
            using (Variable.ForEach(n))
            {
                using (Variable.Switch(TrueLabel[n]))
                {
                    Words[n][nw] = Variable.Discrete(ProbWord[TrueLabel[n]]).ForEach(nw);
                }
            }
            block.CloseBlock();
        }

        private void ObserveCrowdLabels(int[][] workerLabel, int[][] workerTaskIndex)
        {
            BackgroundLabelProbPrior.ObservedValue = Dirichlet.Uniform(c.SizeAsInt);
            WorkerCount.ObservedValue = workerLabel.Length;
            WorkerLabel.ObservedValue = workerLabel;
            WorkerTaskCount.ObservedValue = workerTaskIndex.Select(tasks => tasks.Length).ToArray();
            WorkerTaskIndex.ObservedValue = workerTaskIndex;
            SetBiasedPriors(WorkerCount.ObservedValue);
        }

        private void ObserveWords(int[][] words, int[] wordCounts)
        {
            Words.ObservedValue = words;
            WordCount.ObservedValue = wordCounts;
        }

        private void ObserveTrueLabels(int[] trueLabels)
        {
            TrueLabel.ObservedValue = trueLabels;
        }

        public void SetBiasedPriors(int workerCount)
        {
            // uniform over true values
            BackgroundLabelProbPrior.ObservedValue = Dirichlet.Uniform(c.SizeAsInt);
            ConfusionMatrixPrior.ObservedValue = Util.ArrayInit(workerCount, input => Util.ArrayInit(c.SizeAsInt, l => new Dirichlet(Util.ArrayInit(c.SizeAsInt, l1 => l1 == l ? 5.5 : 1))));
            ProbWordPrior.ObservedValue = Dirichlet.Symmetric(w.SizeAsInt, 1);
        }

        /* Inference */
        public BCCWordsPosteriors InferPosteriors(
                    int[][] workerLabel, int[][] workerTaskIndex, int[][] words, int[] wordCounts, int[] trueLabels = null,
                    int numIterations = 35)
        {
            ObserveCrowdLabels(workerLabel, workerTaskIndex);

            ObserveWords(words, wordCounts);

            if (trueLabels != null)
            {
                ObserveTrueLabels(trueLabels);
            }

            BCCWordsPosteriors posteriors = new BCCWordsPosteriors();

            Console.WriteLine("\n***** BCC Words *****\n");
            for (int it = 1; it <= numIterations; it++)
            {
                Engine.NumberOfIterations = it;
                posteriors.TrueLabel = Engine.Infer<Discrete[]>(TrueLabel);
                posteriors.WorkerConfusionMatrix = Engine.Infer<Dirichlet[][]>(WorkerConfusionMatrix);
                posteriors.BackgroundLabelProb = Engine.Infer<Dirichlet>(BackgroundLabelProb);
                posteriors.ProbWordPosterior = Engine.Infer<Dirichlet[]>(ProbWord);
                Console.WriteLine("Iteration {0}:\t{1:0.0000}", it, posteriors.TrueLabel[0]);
            }

            posteriors.Evidence = Engine.Infer<Bernoulli>(evidence);
            return posteriors;
        }
    }
}
