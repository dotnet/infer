// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    [Example("Applications", "A model which reconciles document click counts and human relevance judgements")]
    public class ClickModel
    {
        public void Run()
        {
            // Number of label classes for this example
            int numLabels = 3;

            // Train the model
            ClickModelMarginals marginals = Model1(numLabels, false);
            if (marginals == null)
            {
                return;
            }

            //-----------------------------------------------------------------------------
            // The prediction model
            //-----------------------------------------------------------------------------

            // The observations will be in the form of an array of distributions
            Variable<int> numberOfObservations = Variable.New<int>().Named("NumObs");
            Range r = new Range(numberOfObservations).Named("N");
            VariableArray<Gaussian> observationDistribs = Variable.Array<Gaussian>(r).Named("Obs");

            // Use the marginals from the trained model
            Variable<double> scoreMean = Variable.Random(marginals.marginalScoreMean).Named("scoreMean");
            Variable<double> scorePrec = Variable.Random(marginals.marginalScorePrec).Named("scorePrec");
            Variable<double> judgePrec = Variable.Random(marginals.marginalJudgePrec).Named("judgePrec");
            Variable<double> clickPrec = Variable.Random(marginals.marginalClickPrec).Named("clickPrec");
            Variable<double>[] thresholds = new Variable<double>[numLabels + 1];

            // Variables for each observation
            VariableArray<double> scores = Variable.Array<double>(r).Named("Scores");
            VariableArray<double> scoresJ = Variable.Array<double>(r).Named("ScoresJ");
            VariableArray<double> scoresC = Variable.Array<double>(r).Named("ScoresC");
            scores[r] = Variable.GaussianFromMeanAndPrecision(scoreMean, scorePrec).ForEach(r);
            scoresJ[r] = Variable.GaussianFromMeanAndPrecision(scores[r], judgePrec);
            scoresC[r] = Variable.GaussianFromMeanAndPrecision(scores[r], clickPrec);

            // Constrain to the click observation
            Variable.ConstrainEqualRandom(scoresC[r], observationDistribs[r]);

            // The threshold variables
            thresholds[0] = Variable.GaussianFromMeanAndVariance(Double.NegativeInfinity, 0.0).Named("thresholds0");
            for (int i = 1; i < thresholds.Length - 1; i++)
            {
                thresholds[i] = Variable.Random(marginals.marginalThresh[i]).Named("thresholds" + i);
            }

            thresholds[thresholds.Length - 1] = Variable.GaussianFromMeanAndVariance(Double.PositiveInfinity, 0.0).Named("thresholds" + (thresholds.Length - 1));

            // Boolean label variables
            VariableArray<bool>[] testLabels = new VariableArray<bool>[numLabels];
            for (int j = 0; j < numLabels; j++)
            {
                testLabels[j] = Variable.Array<bool>(r).Named("TestLabels" + j);
                testLabels[j][r] = Variable.IsBetween(scoresJ[r], thresholds[j], thresholds[j + 1]);
            }

            //--------------------------------------------------------------------
            // Running the prediction model
            //--------------------------------------------------------------------
            int[] clicks = { 10, 100, 1000, 9, 99, 999, 10, 10, 10 };
            int[] exams = { 20, 200, 2000, 10, 100, 1000, 100, 1000, 10000 };
            Gaussian[] obs = new Gaussian[clicks.Length];
            for (int i = 0; i < clicks.Length; i++)
            {
                int nC = clicks[i];    // Number of clicks 
                int nE = exams[i];     // Number of examinations
                int nNC = nE - nC;     // Number of non-clicks
                Beta b = new Beta(1.0 + nC, 1.0 + nNC);
                double m, v;
                b.GetMeanAndVariance(out m, out v);
                obs[i] = Gaussian.FromMeanAndVariance(m, v);
            }

            numberOfObservations.ObservedValue = obs.Length;
            observationDistribs.ObservedValue = obs;
            InferenceEngine engine = new InferenceEngine();
            Gaussian[] latentScore = engine.Infer<Gaussian[]>(scores);
            Bernoulli[][] predictedLabels = new Bernoulli[numLabels][];
            for (int j = 0; j < numLabels; j++)
            {
                predictedLabels[j] = engine.Infer<Bernoulli[]>(testLabels[j]);
            }

            Console.WriteLine("\n******   Some Predictions  ******\n");
            Console.WriteLine("Clicks\tExams\t\tScore\t\tLabel0\t\tLabel1\t\tLabel2");
            for (int i = 0; i < clicks.Length; i++)
            {
                Console.WriteLine(
                    "{0}\t{1}\t\t{2}\t\t{3}\t\t{4}\t\t{5}",
                    clicks[i],
                    exams[i],
                    latentScore[i].GetMean().ToString("F4"),
                    predictedLabels[0][i].GetProbTrue().ToString("F4"),
                    predictedLabels[1][i].GetProbTrue().ToString("F4"),
                    predictedLabels[2][i].GetProbTrue().ToString("F4"));
            }
        }

        static private ClickModelMarginals Model1(int numLabels, bool allowNoExams)
        {
            // Inference engine must be EP because of the ConstrainBetween constraint
            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is Algorithms.ExpectationPropagation))
            {
                Console.WriteLine("This example only runs with Expectation Propagation");
                return null;
            }

            engine.NumberOfIterations = 10;  // Restrict the number of iterations

            // Includes lower and upper bounds
            int numThresholds = numLabels + 1;

            //-------------------------------------------------------------
            // Specify prior distributions
            //-------------------------------------------------------------
            Gaussian priorScoreMean = Gaussian.FromMeanAndVariance(0.5, 1.0);
            Gamma priorScorePrec = Gamma.FromMeanAndVariance(2.0, 0.0);
            Gamma priorJudgePrec = Gamma.FromMeanAndVariance(2.0, 1.0);
            Gamma priorClickPrec = Gamma.FromMeanAndVariance(2.0, 1.0);
            Gaussian[] priorThresholds;
            CreateThresholdPriors(numLabels, out priorThresholds);

            //-------------------------------------------------------------
            // Variables to infer
            //-------------------------------------------------------------
            Variable<double> scoreMean = Variable.Random(priorScoreMean).Named("scoreMean");
            Variable<double> scorePrec = Variable.Random(priorScorePrec).Named("scorePrec");
            Variable<double> judgePrec = Variable.Random(priorJudgePrec).Named("judgePrec");
            Variable<double> clickPrec = Variable.Random(priorClickPrec).Named("clickPrec");
            Variable<double>[] thresholds = new Variable<double>[numLabels + 1];
            for (int i = 0; i < thresholds.Length; i++)
            {
                thresholds[i] = Variable.Random(priorThresholds[i]).Named("thresholds" + i);
            }

            //----------------------------------------------------------------------------------
            // The model
            //----------------------------------------------------------------------------------
            VariableArray<Gaussian>[] observationDistribs = new VariableArray<Gaussian>[numLabels];
            Variable<int>[] numberOfObservations = new Variable<int>[numLabels];
            for (int i = 0; i < numLabels; i++)
            {
                numberOfObservations[i] = Variable.New<int>().Named("NumObs" + i);
                Range r = new Range(numberOfObservations[i]).Named("N" + i);
                observationDistribs[i] = Variable.Array<Gaussian>(r).Named("Obs" + i);
                VariableArray<double> scores = Variable.Array<double>(r).Named("Scores" + i);
                VariableArray<double> scoresJ = Variable.Array<double>(r).Named("ScoresJ" + i);
                VariableArray<double> scoresC = Variable.Array<double>(r).Named("ScoresC" + i);
                scores[r] = Variable.GaussianFromMeanAndPrecision(scoreMean, scorePrec).ForEach(r);
                scoresJ[r] = Variable.GaussianFromMeanAndPrecision(scores[r], judgePrec);
                scoresC[r] = Variable.GaussianFromMeanAndPrecision(scores[r], clickPrec);
                Variable.ConstrainBetween(scoresJ[r], thresholds[i], thresholds[i + 1]);
                Variable.ConstrainEqualRandom(scoresC[r], observationDistribs[i][r]);
            }

            // Get the arrays of human judgement labels, clicks, and examinations
            int[] labels;
            int[] clicks;
            int[] exams;
            string fileName = Path.Combine(
#if NETCOREAPP
                    Path.GetDirectoryName(typeof(ClickModel).Assembly.Location), // work dir is not the one with Microsoft.ML.Probabilistic.Tests.dll on netcore and neither is .Location on netfull
#endif
                "TutorialData", "ClickModel.txt");
            if (!File.Exists(fileName))
            {
                fileName = Path.Combine(
#if NETCOREAPP
                    Path.GetDirectoryName(typeof(ClickModel).Assembly.Location), // work dir is not the one with Microsoft.ML.Probabilistic.Tests.dll on netcore and neither is .Location on netfull
#endif
                    "..", "Samples", "C#", "ExamplesBrowser", "TutorialData", "ClickModel.txt");
            }

            LoadData(fileName, allowNoExams, out labels, out clicks, out exams);

            // Convert the raw click data into uncertain Gaussian observations chunk-by-chunk
            Gaussian[][] allObs = getClickObservations(numLabels, labels, clicks, exams);

            // (a) Set the observation and observation count parameters in the model
            for (int i = 0; i < numLabels; i++)
            {
                numberOfObservations[i].ObservedValue = allObs[i].Length;
                observationDistribs[i].ObservedValue = allObs[i];
            }

            // (b) Request the marginals
            ClickModelMarginals marginals = new ClickModelMarginals(numLabels);
            marginals.marginalScoreMean = engine.Infer<Gaussian>(scoreMean);
            marginals.marginalScorePrec = engine.Infer<Gamma>(scorePrec);
            marginals.marginalJudgePrec = engine.Infer<Gamma>(judgePrec);
            marginals.marginalClickPrec = engine.Infer<Gamma>(clickPrec);
            for (int i = 0; i < numThresholds; i++)
            {
                marginals.marginalThresh[i] = engine.Infer<Gaussian>(thresholds[i]);
            }

            Console.WriteLine("Training: sample size: " + labels.Length + "\n");
            Console.WriteLine("scoreMean = {0}", marginals.marginalScoreMean);
            Console.WriteLine("scorePrec = {0}", marginals.marginalScorePrec);
            Console.WriteLine("judgePrec = {0}", marginals.marginalJudgePrec);
            Console.WriteLine("clickPrec = {0}", marginals.marginalClickPrec);
            for (int t = 0; t < numThresholds; t++)
            {
                Console.WriteLine("threshMean {0} = {1}", t, marginals.marginalThresh[t]);
            }

            return marginals;
        }

        static private ClickModelMarginals Model2(int numLabels, bool allowNoExams)
        {
            // Inference engine must be EP because of the ConstrainBetween constraint
            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is Algorithms.ExpectationPropagation))
            {
                Console.WriteLine("This example only runs with Expectation Propagation");
                return null;
            }

            engine.NumberOfIterations = 10;

            // Includes lower and upper bounds
            int numThresholds = numLabels + 1;

            // Partition the dat into chunks to improve the schedule
            int chunkSize = 200;

            // Maximum number of passes through the data
            int maxPasses = 5;

            // The marginals at any given stage.
            ClickModelMarginals marginals = new ClickModelMarginals(numLabels);

            // Compare the marginals with the previous marginals to create
            // a convergence criterion
            Gaussian prevMargScoreMean;
            Gamma prevMargJudgePrec;
            Gamma prevMargClickPrec;
            double convergenceThresh = 0.01;

            // Get the arrays of human judgement labels, clicks, and examinations
            int[] labels;
            int[] clicks;
            int[] exams;
            string fileName = Path.Combine(
#if NETCOREAPP
                    Path.GetDirectoryName(typeof(ClickModel).Assembly.Location), // work dir is not the one with Microsoft.ML.Probabilistic.Tests.dll on netcore and neither is .Location on netfull
#endif
                "TutorialData", "ClickModel.txt");
            if (!File.Exists(fileName))
            {
                fileName = Path.Combine(
#if NETCOREAPP
                    Path.GetDirectoryName(typeof(ClickModel).Assembly.Location), // work dir is not the one with Microsoft.ML.Probabilistic.Tests.dll on netcore and neither is .Location on netfull
#endif
                    "..", "Samples", "C#", "ExamplesBrowser", "TutorialData", "ClickModel.txt");
            }

            LoadData(fileName, allowNoExams, out labels, out clicks, out exams);

            // Convert the raw click data into uncertain Gaussian observations chunk-by-chunk
            Gaussian[][][] allObs = getClickObservations(numLabels, chunkSize, labels, clicks, exams);
            int numChunks = allObs.Length;

            //-------------------------------------------------------------
            // Specify prior distributions
            //-------------------------------------------------------------
            Gaussian priorScoreMean = Gaussian.FromMeanAndVariance(0.5, 1.0);
            Gamma priorScorePrec = Gamma.FromMeanAndVariance(2.0, 0.0);
            Gamma priorJudgePrec = Gamma.FromMeanAndVariance(2.0, 1.0);
            Gamma priorClickPrec = Gamma.FromMeanAndVariance(2.0, 1.0);
            Gaussian[] priorThresholds;
            CreateThresholdPriors(numLabels, out priorThresholds);

            //-----------------------------------------------------
            // Create shared variables - these are the variables
            // which are shared between all chunks
            //-----------------------------------------------------
            Model model = new Model(numChunks);
            SharedVariable<double> scoreMean = SharedVariable<double>.Random(priorScoreMean).Named("scoreMean");
            SharedVariable<double> scorePrec = SharedVariable<double>.Random(priorScorePrec).Named("scorePrec");
            SharedVariable<double> judgePrec = SharedVariable<double>.Random(priorJudgePrec).Named("judgePrec");
            SharedVariable<double> clickPrec = SharedVariable<double>.Random(priorClickPrec).Named("clickPrec");
            SharedVariable<double>[] thresholds = new SharedVariable<double>[numThresholds];
            for (int t = 0; t < numThresholds; t++)
            {
                thresholds[t] = SharedVariable<double>.Random(priorThresholds[t]).Named("threshold" + t);
            }

            //----------------------------------------------------------------------------------
            // The model
            //----------------------------------------------------------------------------------

            // Gaussian click observations are given to the model - one set of observations
            // per label class. Also the number of observations per label class is given to the model
            VariableArray<Gaussian>[] observationDistribs = new VariableArray<Gaussian>[numLabels];
            Variable<int>[] numberOfObservations = new Variable<int>[numLabels];

            // For each label, and each observation (consisting of a human judgement and
            // a Gaussian click observation), there is a latent score variable, a judgement
            // score variable, and a click score variable
            for (int i = 0; i < numLabels; i++)
            {
                numberOfObservations[i] = Variable.New<int>().Named("NumObs" + i);
                Range r = new Range(numberOfObservations[i]).Named("N" + i);
                observationDistribs[i] = Variable.Array<Gaussian>(r).Named("Obs" + i);
                VariableArray<double> scores = Variable.Array<double>(r).Named("Scores" + i);
                VariableArray<double> scoresJ = Variable.Array<double>(r).Named("ScoresJ" + i);
                VariableArray<double> scoresC = Variable.Array<double>(r).Named("ScoresC" + i);
                scores[r] = Variable.GaussianFromMeanAndPrecision(scoreMean.GetCopyFor(model), scorePrec.GetCopyFor(model)).ForEach(r);
                scoresJ[r] = Variable.GaussianFromMeanAndPrecision(scores[r], judgePrec.GetCopyFor(model));
                scoresC[r] = Variable.GaussianFromMeanAndPrecision(scores[r], clickPrec.GetCopyFor(model));
                Variable.ConstrainEqualRandom(scoresC[r], observationDistribs[i][r]);
                Variable.ConstrainBetween(scoresJ[r], thresholds[i].GetCopyFor(model), thresholds[i + 1].GetCopyFor(model));
            }

            //----------------------------------------------------------
            // Outer loop iterates over a number of passes
            // Inner loop iterates over the unique labels
            //----------------------------------------------------------
            Console.WriteLine("Training: sample size: " + labels.Length + "\n");
            for (int pass = 0; pass < maxPasses; pass++)
            {
                prevMargScoreMean = marginals.marginalScoreMean;
                prevMargJudgePrec = marginals.marginalJudgePrec;
                prevMargClickPrec = marginals.marginalClickPrec;
                for (int c = 0; c < numChunks; c++)
                {
                    for (int i = 0; i < numLabels; i++)
                    {
                        numberOfObservations[i].ObservedValue = allObs[c][i].Length;
                        observationDistribs[i].ObservedValue = allObs[c][i];
                    }

                    model.InferShared(engine, c);

                    // Retrieve marginals
                    marginals.marginalScoreMean = scoreMean.Marginal<Gaussian>();
                    marginals.marginalScorePrec = scorePrec.Marginal<Gamma>();
                    marginals.marginalJudgePrec = judgePrec.Marginal<Gamma>();
                    marginals.marginalClickPrec = clickPrec.Marginal<Gamma>();
                    for (int i = 0; i < numThresholds; i++)
                    {
                        marginals.marginalThresh[i] = thresholds[i].Marginal<Gaussian>();
                    }

                    Console.WriteLine("\n****** Pass {0}, chunk {1} ******", pass, c);
                    Console.WriteLine("----- Marginals -----");
                    Console.WriteLine("scoreMean = {0}", marginals.marginalScoreMean);
                    Console.WriteLine("scorePrec = {0}", marginals.marginalScorePrec);
                    Console.WriteLine("judgePrec = {0}", marginals.marginalJudgePrec);
                    Console.WriteLine("clickPrec = {0}", marginals.marginalClickPrec);
                    for (int t = 0; t < numThresholds; t++)
                    {
                        Console.WriteLine("threshMean {0} = {1}", t, marginals.marginalThresh[t]);
                    }
                }

                // Test for convergence
                if (marginals.marginalScoreMean.MaxDiff(prevMargScoreMean) < convergenceThresh &&
                        marginals.marginalJudgePrec.MaxDiff(prevMargJudgePrec) < convergenceThresh &&
                        marginals.marginalClickPrec.MaxDiff(prevMargClickPrec) < convergenceThresh)
                {
                    Console.WriteLine("\n****** Inference converged ******\n");
                    break;
                }
            }

            return marginals;
        }

        // Method to read click data. This assumes a header row
        // followed by data rows with tab or comma separated text
        static private void LoadData(
            string ifn,         // The file name
            bool allowNoExams,  // Allow records with no examinations
            out int[] labels,   // Labels
            out int[] clicks,   // Clicks
            out int[] exams)    // Examinations
        {
            // File is assumed to have a header row, followed by
            // tab or comma separated label, clicks, exams
            labels = null;
            clicks = null;
            exams = null;
            int totalDocs = 0;
            string myStr;
            char[] sep = { '\t', ',' };

            for (int pass = 0; pass < 2; pass++)
            {
                if (1 == pass)
                {
                    labels = new int[totalDocs];
                    clicks = new int[totalDocs];
                    exams = new int[totalDocs];
                    totalDocs = 0;
                }

                using (var mySR = new StreamReader(ifn))
                {
                    mySR.ReadLine(); // Skip over header line
                    while ((myStr = mySR.ReadLine()) != null)
                    {
                        string[] mySplitStr = myStr.Split(sep);
                        int exm = int.Parse(mySplitStr[2]);

                        // Only include data with non-zero examinations
                        if (0 != exm || allowNoExams)
                        {
                            if (1 == pass)
                            {
                                int lab = int.Parse(mySplitStr[0]);
                                int clk = int.Parse(mySplitStr[1]);
                                labels[totalDocs] = lab;
                                clicks[totalDocs] = clk;
                                exams[totalDocs] = exm;
                            }

                            totalDocs++;
                        }
                    }
                }
            }
        }

        // Count the number of documents for each label
        static private int[] getLabelCounts(int numLabs, int[] labels)
        {
            return getLabelCounts(numLabs, labels, 0, labels.Length);
        }

        // Count the number of documents for each label for a given chunk
        static private int[] getLabelCounts(int numLabs, int[] labels, int startX, int endX)
        {
            int[] cnt = new int[numLabs];
            for (int l = 0; l < numLabs; l++)
            {
                cnt[l] = 0;
            }

            if (startX < 0)
            {
                startX = 0;
            }

            if (startX >= labels.Length)
            {
                startX = labels.Length - 1;
            }

            if (endX < 0)
            {
                endX = 0;
            }

            if (endX > labels.Length)
            {
                endX = labels.Length;
            }

            for (int d = startX; d < endX; d++)
            {
                cnt[labels[d]]++;
            }

            return cnt;
        }

        // Get click observations for each label class
        static private Gaussian[][] getClickObservations(int numLabs, int[] labels, int[] clicks, int[] exams)
        {
            Gaussian[][][] obs = getClickObservations(numLabs, labels.Length, labels, clicks, exams);
            return obs[0];
        }

        // Get click observations for each chunk and label class
        static private Gaussian[][][] getClickObservations(int numLabs, int chunkSize, int[] labels, int[] clicks, int[] exams)
        {
            int nData = labels.Length;
            int numChunks = (nData + chunkSize - 1) / chunkSize;
            Gaussian[][][] chunks = new Gaussian[numChunks][][];
            int[] obsX = new int[numLabs];

            int startChunk = 0;
            int endChunk = 0;
            for (int c = 0; c < numChunks; c++)
            {
                startChunk = endChunk;
                endChunk = startChunk + chunkSize;
                if (endChunk > nData)
                {
                    endChunk = nData;
                }

                int[] labCnts = getLabelCounts(numLabs, labels, startChunk, endChunk);
                chunks[c] = new Gaussian[numLabs][];
                Gaussian[][] currChunk = chunks[c];
                for (int l = 0; l < numLabs; l++)
                {
                    currChunk[l] = new Gaussian[labCnts[l]];
                    obsX[l] = 0;
                }

                for (int d = startChunk; d < endChunk; d++)
                {
                    int lab = labels[d];
                    int nC = clicks[d];
                    int nE = exams[d];
                    int nNC = nE - nC;
                    double b0 = 1.0 + nC;  // Observations of clicks
                    double b1 = 1.0 + nNC;   // Observations of no clicks
                    Beta b = new Beta(b0, b1);
                    double m, v;
                    b.GetMeanAndVariance(out m, out v);
                    Gaussian g = new Gaussian();
                    g.SetMeanAndVariance(m, v);
                    currChunk[lab][obsX[lab]++] = g;
                }
            }

            return chunks;
        }

        // This method creates threshold priors - they are
        // set at regular intervals between 0 and 1 with overlapping
        // distributions. The lower and upper bounds are fixed to
        // 0 and 1 respectively
        static private void CreateThresholdPriors(
                int numLabels,
                out Gaussian[] priorThresholds)
        {
            double invNumLabs = 1.0 / ((double)numLabels);
            double prec = (double)(numLabels * numLabels);
            double mean = invNumLabs;
            int numThresholds = numLabels + 1;
            priorThresholds = new Gaussian[numThresholds];
            priorThresholds[0] = Gaussian.PointMass(0);
            for (int t = 1; t < numThresholds - 1; t++)
            {
                priorThresholds[t] = new Gaussian();
                priorThresholds[t].SetMeanAndPrecision(mean, prec);
                mean += invNumLabs;
            }

            priorThresholds[numThresholds - 1] = Gaussian.PointMass(1);
        }
    }

    public class ClickModelMarginals
    {
        public Gaussian marginalScoreMean;
        public Gamma marginalScorePrec;
        public Gamma marginalJudgePrec;
        public Gamma marginalClickPrec;
        public Gaussian[] marginalThresh;

        public ClickModelMarginals(int numLabels)
        {
            marginalScoreMean = new Gaussian();
            marginalScorePrec = new Gamma();
            marginalJudgePrec = new Gamma();
            marginalClickPrec = new Gamma();
            marginalThresh = new Gaussian[numLabels + 1];
        }
    }
}
