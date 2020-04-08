// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#define multi_plate_model

using System;
using System.Collections.Generic;
using System.IO;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    /// <summary>
    /// Summary description for ClickModel
    /// </summary>
    public class ClickTest
    {
        // Count the number of documents for each label
        private int[] getLabelCounts(int numLabs, int[] labels)
        {
            return getLabelCounts(numLabs, labels, 0, labels.Length);
        }

        // Count the number of documents for each label for a given chunk
        private int[] getLabelCounts(int numLabs, int[] labels, int startX, int endX)
        {
            int[] cnt = new int[numLabs];
            for (int l = 0; l < numLabs; l++)
                cnt[l] = 0;

            if (startX < 0)
                startX = 0;
            if (startX >= labels.Length)
                startX = labels.Length - 1;
            if (endX < 0)
                endX = 0;
            if (endX > labels.Length)
                endX = labels.Length;

            for (int d = startX; d < endX; d++)
            {
                cnt[labels[d]]++;
            }
            return cnt;
        }

        // Get click observations for each chunk
        private Gaussian[][][] getClickObservations(int numLabs, int chunkSize, int[] labels, int[] clicks, int[] exams)
        {
            int nData = labels.Length;
            int numChunks = (nData + chunkSize - 1)/chunkSize;
            Gaussian[][][] chunks = new Gaussian[numChunks][][];
            int[] obsX = new int[numLabs];

            int startChunk = 0;
            int endChunk = 0;
            for (int c = 0; c < numChunks; c++)
            {
                startChunk = endChunk;
                endChunk = startChunk + chunkSize;
                if (endChunk > nData)
                    endChunk = nData;

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
                    double b0 = 1.0 + nNC; // Observations of no clicks
                    double b1 = 1.0 + nC; // Observations of clicks
                    Beta b = new Beta(b1, b0);
                    Gaussian g = new Gaussian();
                    g.SetMeanAndPrecision(b.GetMean(), b.TotalCount);
                    currChunk[lab][obsX[lab]++] = g;
                }
            }
            return chunks;
        }

        // Some real data
        private int[] fixedLabels =
            {
                2, 2, 4, 3, 2, 0, 3, 0, 3, 0, 4, 1, 1, 0, 3, 0, 0, 1, 2, 0,
                0, 2, 2, 0, 0, 2, 1, 1, 1, 2, 1, 1, 0, 0, 2, 2, 2, 1, 0, 0,
                0, 2, 0, 0, 0, 0, 4, 2, 3, 4, 4, 3, 3, 0, 2, 2, 2, 1, 2, 2,
                2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2,
                2, 1, 2, 1, 2, 2, 1, 2
            };

        private int[] fixedClicks =
            {
                0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 39, 39, 0, 0, 2, 0, 0, 8,
                0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 54, 1, 0, 0, 0, 0, 0, 0, 3, 4,
                0, 0, 0, 0, 0, 0, 1, 0, 14, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0,
                0, 3, 0, 3, 0, 0, 3, 0, 14
            };

        private int[] fixedExams =
            {
                0, 0, 0, 34, 0, 0, 0, 0, 0, 0, 2, 75, 75, 0, 0, 4, 0, 0, 17,
                0, 0, 17, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 8, 0, 0, 0, 1, 0, 0, 58, 2, 2, 2, 0, 0, 0, 0, 31, 50,
                0, 0, 0, 0, 0, 0, 14, 0, 24, 0, 0, 0, 0, 59, 0, 0, 0, 0, 0, 0,
                0, 27, 0, 5, 0, 0, 46, 0, 15
            };

        [Fact]
        [Trait("Category", "CsoftModel")]
        public void ClickModelSmall()
        {
            int numLabels = 5;
            bool learnScoreMean = true;
            bool learnScorePrec = false;
            bool learnJudgePrec = true;
            bool learnClickPrec = true;
            bool learnThresholds = true;
            double nominalScoreMean = 0.5;
            double nominalScorePrec = 2.0;
            double nominalJudgePrec = 2.0;
            double nominalClickPrec = 2.0;
            bool printToConsole = false;
            Gaussian margScoreMean;
            Gamma margScorePrec;
            Gamma margJudgePrec;
            Gamma margClickPrec;
            Gaussian[] margThresh;
            int chunkSize = 200;
            int nPasses = 5;

#if multi_plate_model
            LearnClick5LabelModel(
#else
            LearnClickModel(
#endif
                numLabels, learnScoreMean, learnScorePrec, learnJudgePrec, learnClickPrec, learnThresholds,
                nominalScoreMean, nominalScorePrec, nominalJudgePrec, nominalClickPrec,
                fixedLabels, fixedClicks, fixedExams, chunkSize, nPasses, printToConsole,
                out margScoreMean, out margScorePrec, out margJudgePrec, out margClickPrec, out margThresh);

            Console.WriteLine("****** Final Marginals ******");
            Console.WriteLine("scoreMean = {0}", margScoreMean);
            Console.WriteLine("scorePrec = {0}", margScorePrec);
            Console.WriteLine("judgePrec = {0}", margJudgePrec);
            Console.WriteLine("clickPrec = {0}", margClickPrec);
            for (int t = 0; t < margThresh.Length; t++)
                Console.WriteLine("threshMean {0} = {1}", t, margThresh[t]);
            Console.WriteLine("*****************************");
            Assert.True(true);
        }

        [Fact]
        public void ClickModelTest()
        {
            int numLabels = 5;
            bool learnScoreMean = true;
            bool learnScorePrec = false;
            bool learnJudgePrec = true;
            bool learnClickPrec = true;
            bool learnThresholds = true;
            double nominalScoreMean = 0.5;
            double nominalScorePrec = 2.0;
            double nominalJudgePrec = 2.0;
            double nominalClickPrec = 2.0;
            bool printToConsole = true;
            Gaussian margScoreMean;
            Gamma margScorePrec;
            Gamma margJudgePrec;
            Gamma margClickPrec;
            Gaussian[] margThresh;
            int chunkSize = 20; // was 200
            int nPasses = 5;

            int[] lb;
            int[] ck;
            int[] ex;

            // limit numDocs to 100 to make the test quicker
            LoadData(Path.Combine(TestUtils.DataFolderPath, "ClickModel.txt"), out lb, out ck, out ex, 100);
            Console.WriteLine("Total number of documents: {0}", lb.Length);

#if multi_plate_model
            LearnAPIClick5LabelModel(
#else
            LearnClickModel(
#endif
                numLabels, learnScoreMean, learnScorePrec, learnJudgePrec, learnClickPrec, learnThresholds,
                nominalScoreMean, nominalScorePrec, nominalJudgePrec, nominalClickPrec,
                lb, ck, ex, chunkSize, nPasses, printToConsole,
                out margScoreMean, out margScorePrec, out margJudgePrec, out margClickPrec, out margThresh);

            Console.WriteLine("****** Final Marginals ******");
            Console.WriteLine("scoreMean = {0}", margScoreMean);
            Console.WriteLine("scorePrec = {0}", margScorePrec);
            Console.WriteLine("judgePrec = {0}", margJudgePrec);
            Console.WriteLine("clickPrec = {0}", margClickPrec);
            for (int t = 0; t < margThresh.Length; t++)
                Console.WriteLine("threshMean {0} = {1}", t, margThresh[t]);
            Console.WriteLine("*****************************");
        }

        private void LoadData(string filename, out int[] labels, out int[] clicks, out int[] exams, int maxDocs = -1)
        {
            // File is assumed to have tab separated label, then clicks, then exams
            List<int> labelList = new List<int>();
            List<int> clickList = new List<int>();
            List<int> examList = new List<int>();
            char[] sep = {'\t'};

            foreach (string line in File.ReadLines(filename))
            {
                if (maxDocs >= 0 && labelList.Count == maxDocs) break;
                string[] split = line.Split(sep);
                labelList.Add(int.Parse(split[0]));
                clickList.Add(int.Parse(split[1]));
                examList.Add(int.Parse(split[2]));
            }
            labels = labelList.ToArray();
            clicks = clickList.ToArray();
            exams = examList.ToArray();
        }

        private void CalculatePriors(
            bool learnThresholds,
            int numLabels,
            out Gaussian[] priorThreshMean)
        {
            double invNumLabs = 1.0/((double) numLabels);
            double prec = (double) (numLabels*numLabels);
            double mean = invNumLabs;
            int numThresholds = numLabels + 1;
            priorThreshMean = new Gaussian[numThresholds];
            priorThreshMean[0] = Gaussian.PointMass(0);
            for (int t = 1; t < numThresholds - 1; t++)
            {
                priorThreshMean[t] = new Gaussian();
                if (learnThresholds)
                    priorThreshMean[t].SetMeanAndPrecision(mean, prec);
                else
                    priorThreshMean[t].Point = mean;

                mean += invNumLabs;
            }
            priorThreshMean[numThresholds - 1] = Gaussian.PointMass(1);
        }

        internal void LearnClickModel(
            int numLabels,
            bool learnScoreMean,
            bool learnScorePrec,
            bool learnJudgePrec,
            bool learnClickPrec,
            bool learnThresholds,
            double nominalScoreMean,
            double nominalScorePrec,
            double nominalJudgePrec,
            double nominalClickPrec,
            int[] labels,
            int[] clicks,
            int[] exams,
            int chunkSize,
            int nPasses,
            bool printToConsole,
            out Gaussian margScoreMean,
            out Gamma margScorePrec,
            out Gamma margJudgePrec,
            out Gamma margClickPrec,
            out Gaussian[] margThresh)
        {
            InferenceEngine engine = new InferenceEngine();
            int numThresholds = numLabels + 1; // Includes end-points

            //-------------------------------------------------------------
            // Priors
            //-------------------------------------------------------------
            Gaussian priorScoreMean = Gaussian.FromMeanAndVariance(nominalScoreMean, learnScoreMean ? 1 : 0);
            Gamma priorScorePrec = Gamma.FromMeanAndVariance(nominalScorePrec, learnScorePrec ? 1 : 0);
            Gamma priorJudgePrec = Gamma.FromMeanAndVariance(nominalJudgePrec, learnJudgePrec ? 1 : 0);
            Gamma priorClickPrec = Gamma.FromMeanAndVariance(nominalClickPrec, learnClickPrec ? 1 : 0);
            Gaussian[] priorThreshMean;
            CalculatePriors(learnThresholds, numLabels, out priorThreshMean);

            //------------------------------------------------------
            // Observations
            //------------------------------------------------------
            Gaussian[][][] clickObs = getClickObservations(numLabels, chunkSize, labels, clicks, exams);
            int numChunks = clickObs.Length;

            //-----------------------------------------------------
            // Place to maintain marginals
            //-----------------------------------------------------
            margScoreMean = new Gaussian(priorScoreMean);
            margScorePrec = new Gamma(priorScorePrec);
            margJudgePrec = new Gamma(priorJudgePrec);
            margClickPrec = new Gamma(priorClickPrec);
            margThresh = new Gaussian[numThresholds];
            for (int t = 0; t < numThresholds; t++)
            {
                margThresh[t] = new Gaussian(priorThreshMean[t]);
            }

            //-----------------------------------------------------
            // Model communication
            //-----------------------------------------------------
            Gaussian[,] outputScoreMean = new Gaussian[numChunks,numLabels];
            Gamma[,] outputScorePrec = new Gamma[numChunks,numLabels];
            Gamma[,] outputJudgePrec = new Gamma[numChunks,numLabels];
            Gamma[,] outputClickPrec = new Gamma[numChunks,numLabels];
            Gaussian[,] outputLowerThresh = new Gaussian[numChunks,numLabels];
            Gaussian[,] outputUpperThresh = new Gaussian[numChunks,numLabels];
            for (int c = 0; c < numChunks; c++)
            {
                for (int l = 0; l < numLabels; l++)
                {
                    outputScoreMean[c, l] = new Gaussian();
                    outputScorePrec[c, l] = new Gamma();
                    outputJudgePrec[c, l] = new Gamma();
                    outputClickPrec[c, l] = new Gamma();
                    outputLowerThresh[c, l] = new Gaussian();
                    outputUpperThresh[c, l] = new Gaussian();
                }
            }
            Gaussian inputScoreMean = new Gaussian();
            Gamma inputScorePrec = new Gamma();
            Gamma inputJudgePrec = new Gamma();
            Gamma inputClickPrec = new Gamma();
            Gaussian inputLowerThreshMean = new Gaussian();
            Gaussian inputUpperThreshMean = new Gaussian();

            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var ca = engine.Compiler.CompileWithoutParams(GetType().GetMethod("ClickModel"));

            //----------------------------------------------------------
            // Outer loop iterates over a number of passes
            // Inner loop iterates over the unique labels
            //----------------------------------------------------------
            for (int pass = 0; pass < nPasses; pass++)
            {
                for (int c = 0; c < numChunks; c++)
                {
                    for (int l = 0; l < numLabels; l++)
                    {
                        // Set up inputs - these include priors + messages
                        // from communicating models

                        // (a) Initialise all as Uniform
                        inputScoreMean.SetToUniform();
                        inputScorePrec.SetToUniform();
                        inputJudgePrec.SetToUniform();
                        inputClickPrec.SetToUniform();
                        inputLowerThreshMean.SetToUniform();
                        inputUpperThreshMean.SetToUniform();

                        // (b) Roll in output messages from all other models
                        for (int c1 = 0; c1 < numChunks; c1++)
                        {
                            for (int l1 = 0; l1 < numLabels; l1++)
                            {
                                if (l1 == l - 1)
                                {
                                    inputLowerThreshMean.SetToProduct(inputLowerThreshMean, outputUpperThresh[c1, l1]);
                                }
                                else if (l1 == l + 1)
                                {
                                    inputUpperThreshMean.SetToProduct(inputUpperThreshMean, outputLowerThresh[c1, l1]);
                                }
                                else if (l1 == l && c1 != c)
                                {
                                    inputLowerThreshMean.SetToProduct(inputLowerThreshMean, outputLowerThresh[c1, l1]);
                                    inputUpperThreshMean.SetToProduct(inputUpperThreshMean, outputUpperThresh[c1, l1]);
                                }

                                if (l1 == l && c1 == c)
                                    continue;

                                inputScoreMean.SetToProduct(inputScoreMean, outputScoreMean[c1, l1]);
                                inputScorePrec.SetToProduct(inputScorePrec, outputScorePrec[c1, l1]);
                                inputJudgePrec.SetToProduct(inputJudgePrec, outputJudgePrec[c1, l1]);
                                inputClickPrec.SetToProduct(inputClickPrec, outputClickPrec[c1, l1]);
                            }
                        }

                        // (c) Roll in the priors
                        inputScoreMean.SetToProduct(inputScoreMean, priorScoreMean);
                        inputScorePrec.SetToProduct(inputScorePrec, priorScorePrec);
                        inputJudgePrec.SetToProduct(inputJudgePrec, priorJudgePrec);
                        inputClickPrec.SetToProduct(inputClickPrec, priorClickPrec);
                        inputLowerThreshMean.SetToProduct(inputLowerThreshMean, priorThreshMean[l]);
                        inputUpperThreshMean.SetToProduct(inputUpperThreshMean, priorThreshMean[l + 1]);

                        ca.SetObservedValue("inputScoreMean", inputScoreMean);
                        ca.SetObservedValue("inputScorePrec", inputScorePrec);
                        ca.SetObservedValue("inputJudgePrec", inputJudgePrec);
                        ca.SetObservedValue("inputClickPrec", inputClickPrec);
                        ca.SetObservedValue("inputLowerThreshMean", inputLowerThreshMean);
                        ca.SetObservedValue("inputUpperThreshMean", inputUpperThreshMean);
                        ca.SetObservedValue("clickObservations", clickObs[c][l]);
                        ca.Execute(10);

                        // Retrieve marginals
                        margThresh[l] = ca.Marginal<Gaussian>("lowerThresh");
                        margThresh[l + 1] = ca.Marginal<Gaussian>("upperThresh");
                        margScoreMean = ca.Marginal<Gaussian>("scoreMean");
                        margScorePrec = ca.Marginal<Gamma>("scorePrec");
                        margJudgePrec = ca.Marginal<Gamma>("judgePrec");
                        margClickPrec = ca.Marginal<Gamma>("clickPrec");

                        // Retrieve the output messages
                        outputLowerThresh[c, l] = ca.Marginal<Gaussian>("lowerThresh", QueryTypes.MarginalDividedByPrior.Name);
                        outputUpperThresh[c, l] = ca.Marginal<Gaussian>("upperThresh", QueryTypes.MarginalDividedByPrior.Name);
                        outputScoreMean[c, l] = ca.Marginal<Gaussian>("scoreMean", QueryTypes.MarginalDividedByPrior.Name);
                        outputScorePrec[c, l] = ca.Marginal<Gamma>("scorePrec", QueryTypes.MarginalDividedByPrior.Name);
                        outputJudgePrec[c, l] = ca.Marginal<Gamma>("judgePrec", QueryTypes.MarginalDividedByPrior.Name);
                        outputClickPrec[c, l] = ca.Marginal<Gamma>("clickPrec", QueryTypes.MarginalDividedByPrior.Name);
                    }
                    if (printToConsole)
                    {
                        Console.WriteLine("****** Pass {0}, chunk {1} ******", pass, c);
                        Console.WriteLine("----- Marginals -----");
                        Console.WriteLine("scoreMean = {0}", margScoreMean);
                        Console.WriteLine("scorePrec = {0}", margScorePrec);
                        Console.WriteLine("judgePrec = {0}", margJudgePrec);
                        Console.WriteLine("clickPrec = {0}", margClickPrec);
                        for (int t = 0; t < numThresholds; t++)
                            Console.WriteLine("threshMean {0} = {1}", t, margThresh[t]);
                    }
                }
            }
        }

        internal void ClickModel(
            Gaussian inputScoreMean,
            Gamma inputScorePrec,
            Gamma inputJudgePrec,
            Gamma inputClickPrec,
            Gaussian inputLowerThreshMean,
            Gaussian inputUpperThreshMean,
            Gaussian[] clickObservations) // Click observations for this label
        {
            int n = clickObservations.Length;

            // Add variables outside plate - all variables outside the plate should
            // be marked as outputs because we will be communicating their messages
            // across labels
            double scoreMean = Factor.Random(inputScoreMean);
            double scorePrec = Factor.Random(inputScorePrec);
            double judgePrec = Factor.Random(inputJudgePrec);
            double clickPrec = Factor.Random(inputClickPrec);
            double lowerThresh = Factor.Random(inputLowerThreshMean);
            double upperThresh = Factor.Random(inputUpperThreshMean);

            // Now set up the plate
            double[] scores = new double[n];
            double[] scoresJ = new double[n];
            double[] scoresC = new double[n];

            bool b = true;
            for (int i = 0; i < n; i++)
            {
                scores[i] = Factor.Gaussian(scoreMean, 2.0);

                // click-based score
                scoresC[i] = Factor.Gaussian(scores[i], clickPrec);
                Constrain.EqualRandom(scoresC[i], clickObservations[i]);

                // judged score
                scoresJ[i] = Factor.Gaussian(scores[i], judgePrec);
                b = Factor.IsBetween(scoresJ[i], lowerThresh, upperThresh);
            }
            InferNet.Infer(scoreMean, nameof(scoreMean));
            InferNet.Infer(scorePrec, nameof(scorePrec));
            InferNet.Infer(judgePrec, nameof(judgePrec));
            InferNet.Infer(clickPrec, nameof(clickPrec));
            InferNet.Infer(lowerThresh, nameof(lowerThresh));
            InferNet.Infer(upperThresh, nameof(upperThresh));
        }


        private void LearnClick5LabelModel(
            int numLabels,
            bool learnScoreMean,
            bool learnScorePrec,
            bool learnJudgePrec,
            bool learnClickPrec,
            bool learnThresholds,
            double nominalScoreMean,
            double nominalScorePrec,
            double nominalJudgePrec,
            double nominalClickPrec,
            int[] labels,
            int[] clicks,
            int[] exams,
            int chunkSize,
            int nPasses,
            bool printToConsole,
            out Gaussian margScoreMean,
            out Gamma margScorePrec,
            out Gamma margJudgePrec,
            out Gamma margClickPrec,
            out Gaussian[] margThresh)
        {
            InferenceEngine engine = new InferenceEngine();
            int numThresholds = numLabels + 1; // Includes end-points

            //-------------------------------------------------------------
            // Priors
            //-------------------------------------------------------------
            Gaussian priorScoreMean = Gaussian.FromMeanAndVariance(nominalScoreMean, learnScoreMean ? 1 : 0);
            Gamma priorScorePrec = Gamma.FromMeanAndVariance(nominalScorePrec, learnScorePrec ? 1 : 0);
            Gamma priorJudgePrec = Gamma.FromMeanAndVariance(nominalJudgePrec, learnJudgePrec ? 1 : 0);
            Gamma priorClickPrec = Gamma.FromMeanAndVariance(nominalClickPrec, learnClickPrec ? 1 : 0);
            Gaussian[] priorThreshMean;
            CalculatePriors(learnThresholds, numLabels, out priorThreshMean);

            //------------------------------------------------------
            // Observations
            //------------------------------------------------------
            Gaussian[][][] clickObs = getClickObservations(numLabels, chunkSize, labels, clicks, exams);
            int numChunks = clickObs.Length;

            //-----------------------------------------------------
            // Create an array of batch variables
            //-----------------------------------------------------
            Model model = new Model(numChunks);
            SharedVariable<double, Gaussian> scoreMean = (SharedVariable<double, Gaussian>) SharedVariable<double>.Random(priorScoreMean).Named("scoreMean");
            SharedVariable<double, Gamma> scorePrec = (SharedVariable<double, Gamma>) SharedVariable<double>.Random(priorScorePrec).Named("scorePrec");
            SharedVariable<double, Gamma> judgePrec = (SharedVariable<double, Gamma>) SharedVariable<double>.Random(priorJudgePrec).Named("judgePrec");
            SharedVariable<double, Gamma> clickPrec = (SharedVariable<double, Gamma>) SharedVariable<double>.Random(priorClickPrec).Named("clickPrec");
            SharedVariable<double, Gaussian>[] threshMeans = new SharedVariable<double, Gaussian>[numThresholds];
            for (int t = 0; t < numThresholds; t++)
            {
                threshMeans[t] = (SharedVariable<double, Gaussian>) SharedVariable<double, Gaussian>.Random(priorThreshMean[t]).Named("threshMeans" + t);
            }

            engine.Compiler.DeclarationProvider = Microsoft.ML.Probabilistic.Compiler.RoslynDeclarationProvider.Instance;
            var info = GetType().GetMethod("Click5LabelModel", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var ca = engine.Compiler.CompileWithoutParams(info);
      
            //----------------------------------------------------------
            // Outer loop iterates over a number of passes
            // Inner loop iterates over the unique labels
            //----------------------------------------------------------
            for (int pass = 0; pass < nPasses; pass++)
            {
                for (int c = 0; c < numChunks; c++)
                {
                    Gaussian[] threshInputs = new Gaussian[threshMeans.Length];
                    for (int i = 0; i < threshInputs.Length; i++) threshInputs[i] = ((SharedVariable<double, Gaussian>) threshMeans[i]).MessageToBatch(model, c);
                    ca.SetObservedValue("inputScoreMean", scoreMean.MessageToBatch(model, c));
                    ca.SetObservedValue("inputScorePrec", scorePrec.MessageToBatch(model, c));
                    ca.SetObservedValue("inputJudgePrec", judgePrec.MessageToBatch(model, c));
                    ca.SetObservedValue("inputClickPrec", clickPrec.MessageToBatch(model, c));
                    ca.SetObservedValue("inputThresh", threshInputs);
                    ca.SetObservedValue("clickObservations", clickObs[c]);
                    ca.Execute(10);

                    // Retrieve the output messages
                    model.InferShared(ca, c);

                    if (printToConsole)
                    {
                        Console.WriteLine("****** Pass {0}, chunk {1} ******", pass, c);
                        Console.WriteLine("----- Marginals -----");
                        Console.WriteLine("scoreMean = {0}", scoreMean.Marginal<Gaussian>());
                        Console.WriteLine("scorePrec = {0}", scorePrec.Marginal<Gamma>());
                        Console.WriteLine("judgePrec = {0}", judgePrec.Marginal<Gamma>());
                        Console.WriteLine("clickPrec = {0}", clickPrec.Marginal<Gamma>());
                        for (int t = 0; t < numThresholds; t++)
                            Console.WriteLine("threshMean {0} = {1}", t, threshMeans[t].Marginal<Gaussian>());
                    }
                }
            }
            margScoreMean = scoreMean.Marginal<Gaussian>();
            margScorePrec = scorePrec.Marginal<Gamma>();
            margJudgePrec = judgePrec.Marginal<Gamma>();
            margClickPrec = clickPrec.Marginal<Gamma>();
            margThresh = new Gaussian[numThresholds];
            for (int i = 0; i < margThresh.Length; i++) margThresh[i] = threshMeans[i].Marginal<Gaussian>();
        }

        private void LearnAPIClick5LabelModel(
            int numLabels,
            bool learnScoreMean,
            bool learnScorePrec,
            bool learnJudgePrec,
            bool learnClickPrec,
            bool learnThresholds,
            double nominalScoreMean,
            double nominalScorePrec,
            double nominalJudgePrec,
            double nominalClickPrec,
            int[] labels,
            int[] clicks,
            int[] exams,
            int chunkSize,
            int nPasses,
            bool printToConsole,
            out Gaussian margScoreMean,
            out Gamma margScorePrec,
            out Gamma margJudgePrec,
            out Gamma margClickPrec,
            out Gaussian[] margThresh)
        {
            //------------------------------------------------------
            // Observations
            //------------------------------------------------------
            Gaussian[][][] allObs = getClickObservations(numLabels, chunkSize, labels, clicks, exams);
            int numChunks = allObs.Length;

            ////-------------------------------------------------------------
            //// Prior distributions
            ////-------------------------------------------------------------
            Gaussian priorScoreMean = Gaussian.FromMeanAndVariance(nominalScoreMean, learnScoreMean ? 1 : 0);
            Gamma priorScorePrec = Gamma.FromMeanAndVariance(nominalScorePrec, learnScorePrec ? 1 : 0);
            Gamma priorJudgePrec = Gamma.FromMeanAndVariance(nominalJudgePrec, learnJudgePrec ? 1 : 0);
            Gamma priorClickPrec = Gamma.FromMeanAndVariance(nominalClickPrec, learnClickPrec ? 1 : 0);
            Gaussian[] priorThreshMean;
            CalculatePriors(learnThresholds, numLabels, out priorThreshMean);

            ////-----------------------------------------------------
            //// Creates shared variables
            ////-----------------------------------------------------
            int numThresholds = numLabels + 1;
            SharedVariable<double> scoreMean = SharedVariable<double>.Random(priorScoreMean).Named("scoreMean");
            SharedVariable<double> scorePrec = SharedVariable<double>.Random(priorScorePrec).Named("scorePrec");
            SharedVariable<double> judgePrec = SharedVariable<double>.Random(priorJudgePrec).Named("judgePrec");
            SharedVariable<double> clickPrec = SharedVariable<double>.Random(priorClickPrec).Named("clickPrec");
            SharedVariable<double>[] thresholds = new SharedVariable<double>[numThresholds];
            for (int t = 0; t < numThresholds; t++)
            {
                thresholds[t] = SharedVariable<double>.Random(priorThreshMean[t]).Named("threshMeans" + t);
            }

            //----------------------------------------------------------------------------------
            // The model
            //----------------------------------------------------------------------------------

            Model model = new Model(numChunks);
            VariableArray<Gaussian>[] clickObs = new VariableArray<Gaussian>[numLabels];
            Variable<int>[] clickObsLength = new Variable<int>[numLabels];

            for (int i = 0; i < numLabels; i++)
            {
                clickObsLength[i] = Variable.New<int>().Named("clickObsLength" + i);
                Range r = new Range(clickObsLength[i]).Named("dataCount" + i);
                clickObs[i] = Variable.Array<Gaussian>(r).Named("Obs" + i);
                VariableArray<double> scores = Variable.Array<double>(r).Named("scores" + i);
                VariableArray<double> scoresJ = Variable.Array<double>(r).Named("scoresJ" + i);
                VariableArray<double> scoresC = Variable.Array<double>(r).Named("scoresC" + i);
                scores[r] = Variable<double>.GaussianFromMeanAndPrecision(scoreMean.GetCopyFor(model), scorePrec.GetCopyFor(model)).ForEach(r);
                scoresJ[r] = Variable<double>.GaussianFromMeanAndPrecision(scores[r], judgePrec.GetCopyFor(model));
                scoresC[r] = Variable<double>.GaussianFromMeanAndPrecision(scores[r], clickPrec.GetCopyFor(model));
                Variable.ConstrainBetween(scoresJ[r], thresholds[i].GetCopyFor(model), thresholds[i + 1].GetCopyFor(model));
                Variable.ConstrainEqualRandom<double, Gaussian>(scoresC[r], clickObs[i][r]);
                r.AddAttribute(new Sequential());
            }

            InferenceEngine engine = new InferenceEngine();

            //----------------------------------------------------------
            // Outer loop iterates over a number of passes
            // Inner loop iterates over the unique labels
            //----------------------------------------------------------
            for (int pass = 0; pass < nPasses; pass++)
            {
                for (int c = 0; c < numChunks; c++)
                {
                    for (int i = 0; i < numLabels; i++)
                    {
                        clickObsLength[i].ObservedValue = allObs[c][i].Length;
                        clickObs[i].ObservedValue = allObs[c][i];
                    }

                    // Infer the output messages
                    model.InferShared(engine, c);

                    if (printToConsole)
                    {
                        margScoreMean = scoreMean.Marginal<Gaussian>();
                        margScorePrec = scorePrec.Marginal<Gamma>();
                        margJudgePrec = judgePrec.Marginal<Gamma>();
                        margClickPrec = clickPrec.Marginal<Gamma>();
                        margThresh = new Gaussian[numThresholds];
                        for (int i = 0; i < numThresholds; i++)
                        {
                            margThresh[i] = thresholds[i].Marginal<Gaussian>();
                        }
                        Console.WriteLine("****** Pass {0}, chunk {1} ******", pass, c);
                        Console.WriteLine("----- Marginals -----");
                        Console.WriteLine("scoreMean = {0}", margScoreMean);
                        Console.WriteLine("scorePrec = {0}", margScorePrec);
                        Console.WriteLine("judgePrec = {0}", margJudgePrec);
                        Console.WriteLine("clickPrec = {0}", margClickPrec);
                        for (int t = 0; t < numThresholds; t++)
                            Console.WriteLine("threshMean {0} = {1}", t, margThresh[t]);
                    }
                }
            }
            margScoreMean = scoreMean.Marginal<Gaussian>();
            margScorePrec = scorePrec.Marginal<Gamma>();
            margJudgePrec = judgePrec.Marginal<Gamma>();
            margClickPrec = clickPrec.Marginal<Gamma>();
            margThresh = new Gaussian[numThresholds];
            for (int i = 0; i < numThresholds; i++)
            {
                margThresh[i] = thresholds[i].Marginal<Gaussian>();
            }
        }

        internal void Click5LabelModel(
            Gaussian inputScoreMean,
            Gamma inputScorePrec,
            Gamma inputJudgePrec,
            Gamma inputClickPrec,
            Gaussian[] inputThresh,
            Gaussian[][] clickObservations)
        {
            // Add variables outside plate - all variables outside the plate should
            // be marked as outputs because we will be communicating their messages
            // across chunks
            double scoreMean = Factor.Random(inputScoreMean);
            double scorePrec = Factor.Random(inputScorePrec);
            double judgePrec = Factor.Random(inputJudgePrec);
            double clickPrec = Factor.Random(inputClickPrec);
            double thresh0 = Factor.Random(inputThresh[0]);
            double thresh1 = Factor.Random(inputThresh[1]);
            double thresh2 = Factor.Random(inputThresh[2]);
            double thresh3 = Factor.Random(inputThresh[3]);
            double thresh4 = Factor.Random(inputThresh[4]);
            double thresh5 = Factor.Random(inputThresh[5]);
            //Attrib.AllVars(new DivideMessages(false), thresh0, thresh1, thresh2, thresh3, thresh4, thresh5);
            //Attrib.AllVars(new DivideMessages(false), scoreMean, scorePrec, judgePrec, clickPrec);

            // Plate 1
            int n1 = clickObservations[0].Length;
            double[] scores1 = new double[n1];
            double[] scoresJ1 = new double[n1];
            double[] scoresC1 = new double[n1];
            for (int i1 = 0; i1 < n1; i1++)
            {
                Attrib.Var(i1, new Sequential());
                scores1[i1] = Factor.Gaussian(scoreMean, scorePrec);

                // click-based score
                scoresC1[i1] = Factor.Gaussian(scores1[i1], clickPrec);
                Constrain.EqualRandom(scoresC1[i1], clickObservations[0][i1]);

                // judged score
                scoresJ1[i1] = Factor.Gaussian(scores1[i1], judgePrec);
                bool h1 = Factor.IsBetween(scoresJ1[i1], thresh0, thresh1);
                Constrain.Equal(true, h1);
            }

            // Plate 2
            int n2 = clickObservations[1].Length;
            double[] scores2 = new double[n2];
            double[] scoresJ2 = new double[n2];
            double[] scoresC2 = new double[n2];
            for (int i2 = 0; i2 < n2; i2++)
            {
                Attrib.Var(i2, new Sequential());
                scores2[i2] = Factor.Gaussian(scoreMean, scorePrec);

                // click-based score
                scoresC2[i2] = Factor.Gaussian(scores2[i2], clickPrec);
                Constrain.EqualRandom(scoresC2[i2], clickObservations[1][i2]);

                // judged score
                scoresJ2[i2] = Factor.Gaussian(scores2[i2], judgePrec);
                bool h2 = Factor.IsBetween(scoresJ2[i2], thresh1, thresh2);
                Constrain.Equal(true, h2);
            }

            // Plate 3
            int n3 = clickObservations[2].Length;
            double[] scores3 = new double[n3];
            double[] scoresJ3 = new double[n3];
            double[] scoresC3 = new double[n3];
            for (int i3 = 0; i3 < n3; i3++)
            {
                Attrib.Var(i3, new Sequential());
                scores3[i3] = Factor.Gaussian(scoreMean, scorePrec);

                // click-based score
                scoresC3[i3] = Factor.Gaussian(scores3[i3], clickPrec);
                Constrain.EqualRandom(scoresC3[i3], clickObservations[2][i3]);

                // judged score
                scoresJ3[i3] = Factor.Gaussian(scores3[i3], judgePrec);
                bool h3 = Factor.IsBetween(scoresJ3[i3], thresh2, thresh3);
                Constrain.Equal(true, h3);
            }

            // Plate 4
            int n4 = clickObservations[3].Length;
            double[] scores4 = new double[n4];
            double[] scoresJ4 = new double[n4];
            double[] scoresC4 = new double[n4];
            for (int i4 = 0; i4 < n4; i4++)
            {
                Attrib.Var(i4, new Sequential());
                scores4[i4] = Factor.Gaussian(scoreMean, scorePrec);

                // click-based score
                scoresC4[i4] = Factor.Gaussian(scores4[i4], clickPrec);
                Constrain.EqualRandom(scoresC4[i4], clickObservations[3][i4]);

                // judged score
                scoresJ4[i4] = Factor.Gaussian(scores4[i4], judgePrec);
                bool h4 = Factor.IsBetween(scoresJ4[i4], thresh3, thresh4);
                Constrain.Equal(true, h4);
            }

            // Plate 5
            int n5 = clickObservations[4].Length;
            double[] scores5 = new double[n5];
            double[] scoresJ5 = new double[n5];
            double[] scoresC5 = new double[n5];
            for (int i5 = 0; i5 < n5; i5++)
            {
                Attrib.Var(i5, new Sequential());
                scores5[i5] = Factor.Gaussian(scoreMean, scorePrec);

                // click-based score
                scoresC5[i5] = Factor.Gaussian(scores5[i5], clickPrec);
                Constrain.EqualRandom(scoresC5[i5], clickObservations[4][i5]);

                // judged score
                scoresJ5[i5] = Factor.Gaussian(scores5[i5], judgePrec);
                bool h5 = Factor.IsBetween(scoresJ5[i5], thresh4, thresh5);
                Constrain.Equal(true, h5);
            }

            //Attrib.AllVars(new DivideMessages(false), scores1, scores2, scores3, scores4, scores5);
            //Attrib.AllVars(new DivideMessages(false), scoresC1, scoresC2, scoresC3, scoresC4, scoresC5);
            //Attrib.AllVars(new DivideMessages(false), scoresJ1, scoresJ2, scoresJ3, scoresJ4, scoresJ5);

            InferNet.Infer(scoreMean, nameof(scoreMean));
            InferNet.Infer(scorePrec, nameof(scorePrec));
            InferNet.Infer(judgePrec, nameof(judgePrec));
            InferNet.Infer(clickPrec, nameof(clickPrec));
            InferNet.Infer(thresh0, nameof(thresh0));
            InferNet.Infer(thresh1, nameof(thresh1));
            InferNet.Infer(thresh2, nameof(thresh2));
            InferNet.Infer(thresh3, nameof(thresh3));
            InferNet.Infer(thresh4, nameof(thresh4));
            InferNet.Infer(thresh5, nameof(thresh5));
        }
    }
}