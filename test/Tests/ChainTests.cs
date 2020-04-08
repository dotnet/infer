// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    public class ChainTests
    {
        public void RolledUpFadingGrid(int length)
        {
            Range m = new Range(length);
            Range n = new Range(length);

            VariableArray2D<double> state = Variable.Array<double>(m, n);
            state[m, n] = Variable.GaussianFromMeanAndVariance(0, 10000).ForEach(m, n);

            VariableArray2D<bool> symbol = Variable.Array<bool>(m, n);
            symbol[m, n] = Variable.Bernoulli(.5).ForEach(m, n);

            double[,] observationValues = new double[length, length];
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < length; j++)
                {
                    observationValues[i, j] = 1.0;
                }
            }
            VariableArray2D<double> observation = Variable.Array<double>(m, n);
            observation.ObservedValue = observationValues;

            using (Variable.ForEach(m))
            {
                using (Variable.ForEach(n))
                {
                    Variable.ConstrainEqualRandom(state[m, n], new Gaussian(0, 1));

                    using (Variable.If(symbol[m, n]))
                    {
                        observation[m, n].SetTo(Variable.GaussianFromMeanAndVariance(state[m, n], .001));
                    }
                    using (Variable.IfNot(symbol[m, n]))
                    {
                        observation[m, n].SetTo(Variable.GaussianFromMeanAndVariance(-state[m, n], .001));
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            var result = engine.Infer<Gaussian[,]>(state);

            Console.WriteLine(result[length - 1, length - 1]);
        }


        public void SimplestChainTest(int numNodes)
        {
            MarkovChain mc = new MarkovChain();

            Gaussian[] result = mc.Infer2(numNodes);
            for (int i = 0; i < numNodes; i++)
            {
                Console.WriteLine(result[i]);
            }
        }

        public static double[,] MessageGridPrecisions(Variable<Gaussian>[,] messageGrid)
        {
            int gridSizeI = 1 + messageGrid.GetUpperBound(0);
            int gridSizeJ = 1 + messageGrid.GetUpperBound(1);

            double[,] grid = new double[gridSizeI, gridSizeJ];

            for (int i = 0; i < gridSizeI; i++)
            {
                for (int j = 0; j < gridSizeJ; j++)
                {
                    try
                    {
                        grid[i, j] = messageGrid[i, j].ObservedValue.Precision;
                    }
                    catch (ImproperDistributionException)
                    {
                        grid[i, j] = -1;
                    }
                }
            }

            return grid;
        }

        public static double[,] MessageGridMeans(Variable<Gaussian>[,] messageGrid)
        {
            int gridSizeI = 1 + messageGrid.GetUpperBound(0);
            int gridSizeJ = 1 + messageGrid.GetUpperBound(1);

            double[,] grid = new double[gridSizeI, gridSizeJ];

            for (int i = 0; i < gridSizeI; i++)
            {
                for (int j = 0; j < gridSizeJ; j++)
                {
                    try
                    {
                        grid[i, j] = messageGrid[i, j].ObservedValue.GetMean();
                    }
                    catch (ImproperDistributionException)
                    {
                        grid[i, j] = -1;
                    }
                }
            }

            return grid;
        }

        // Extract  state marginals, but catch negative variance exceptions
        // and report back a mean of -1 when this happens.
        public static double[,] SafeStateMarginalMeans(Variable<Gaussian>[,] messageGrid)
        {
            int gridSizeI = 1 + messageGrid.GetUpperBound(0);
            int gridSizeJ = 1 + messageGrid.GetUpperBound(1);

            double[,] grid = new double[gridSizeI, gridSizeJ];

            for (int i = 0; i < gridSizeI; i++)
            {
                for (int j = 0; j < gridSizeJ; j++)
                {
                    try
                    {
                        grid[i, j] = messageGrid[i, j].ObservedValue.GetMean();
                    }
                    catch (ImproperDistributionException)
                    {
                        grid[i, j] = -1;
                    }
                }
            }

            return grid;
        }

        // An assertion-like construct that can be set to throw or not throw exceptions
        public static bool checksThrowExceptions = false;

        public static bool CHECK_IS_PROPER(Gaussian v)
        {
            if (v.Precision < 0)
            {
                if (checksThrowExceptions)
                {
                    throw new ImproperDistributionException(v);
                }
                else
                {
                    return false;
                }
            }
            return true;
        }


        // A function used to find the "danger zone"
        public void BreakObservation(double priorMean, double priorVariance)
        {
            double observationVariance = .001;
            double observationMean = 1; // misnomer: actually a mixture of gaussians with means at 1 and -1

            Variable<double> state = new Variable<double>();
            Variable<Gaussian> downwardMessage = new Variable<Gaussian>();
            downwardMessage = Variable.New<Gaussian>();

            downwardMessage.ObservedValue = new Gaussian(priorMean, priorVariance);
            state = Variable.Random<double, Gaussian>(downwardMessage).Named("state");

            Variable<bool> symbol = Variable.Bernoulli(.5);

            Variable<double> observation = new Variable<double>();
            using (Variable.If(symbol))
            {
                observation.SetTo(Variable.GaussianFromMeanAndVariance(state, observationVariance));
            }
            using (Variable.IfNot(symbol))
            {
                observation.SetTo(Variable.GaussianFromMeanAndVariance(-state, observationVariance));
            }
            observation.ObservedValue = observationMean;

            InferenceEngine engine = new InferenceEngine();
            Gaussian result = engine.Infer<Gaussian>(state) / downwardMessage.ObservedValue;

            Console.WriteLine("{0}, {1}", result.GetMean(), result.GetVariance());
        }


        public ExperimentResult FadingChannelGridTest(string priorForm, int gridSizeI, int gridSizeJ, double priorMean, double priorVariance, int iterations,
                                                      string[] schedules)
        {
            double observationMean = 1;
            double observationVariance = .001;
            double transitionVariance = priorVariance * priorVariance / (priorVariance + observationVariance);

            int gridPassesPerIteration = 1;

            // Only used by "xcorner" priors
            //double priorVariance = .3;
            //double priorMean = 1;

            // Only used by "random" priors
            double singletonHyperpriorMeanMean = 1;
            double singletonHyperpriorMeanVariance = 1;
            double singletonHyperpriorVarianceShape = .4;
            double singletonHyperpriorVarianceScale = .5;

            // Priors over individual variables
            double[,] priorMeans = new double[gridSizeI, gridSizeJ];
            double[,] priorVariances = new double[gridSizeI, gridSizeJ];

            for (int i = 0; i < gridSizeI; i++)
            {
                for (int j = 0; j < gridSizeJ; j++)
                {
                    priorMeans[i, j] = 0;
                    priorVariances[i, j] = 1000000000;
                }
            }

            int numInfoSources = 0;
            double informativeObservationProbability = .25; // only applicable for spottyRandom.  chance a prior is informative

            if (priorForm == "1Corner" || priorForm == "NWCorner")
            {
                priorMeans[0, 0] = priorMean;
                priorVariances[0, 0] = priorVariance;
            }
            else if (priorForm == "SWCorner")
            {
                int ipos = gridSizeI - 1;
                int jpos = 0;

                priorMeans[ipos, jpos] = priorMean;
                priorVariances[ipos, jpos] = priorVariance;
            }
            else if (priorForm == "NECorner")
            {
                int ipos = 0;
                int jpos = gridSizeJ - 1;

                priorMeans[ipos, jpos] = priorMean;
                priorVariances[ipos, jpos] = priorVariance;
            }
            else if (priorForm == "SECorner")
            {
                int ipos = gridSizeI - 1;
                int jpos = gridSizeJ - 1;

                priorMeans[ipos, jpos] = priorMean;
                priorVariances[ipos, jpos] = priorVariance;
            }
            else if (priorForm == "random1Corner")
            {
                priorMeans[0, 0] = Gaussian.Sample(singletonHyperpriorMeanMean, singletonHyperpriorMeanVariance);
                priorVariances[0, 0] = Gamma.Sample(singletonHyperpriorVarianceShape, singletonHyperpriorVarianceScale);

                transitionVariance = Gamma.Sample(singletonHyperpriorVarianceShape, singletonHyperpriorVarianceScale);
                Console.WriteLine("*** 1 corner prior: N({0}, {1}), transitionVariance={2} ***", priorMeans[0, 0], priorVariances[0, 0], transitionVariance);
            }
            else if (priorForm == "random2Corner")
            {
                priorMeans[0, 0] = Gaussian.Sample(singletonHyperpriorMeanMean, singletonHyperpriorMeanVariance);
                priorVariances[0, 0] = Gamma.Sample(singletonHyperpriorVarianceShape, singletonHyperpriorVarianceScale);

                priorMeans[gridSizeI - 1, gridSizeJ - 1] = Gaussian.Sample(singletonHyperpriorMeanMean, singletonHyperpriorMeanVariance);
                priorVariances[gridSizeI - 1, gridSizeJ - 1] = Gamma.Sample(singletonHyperpriorVarianceShape, singletonHyperpriorVarianceScale);

                transitionVariance = Gamma.Sample(singletonHyperpriorVarianceShape, singletonHyperpriorVarianceScale);
                Console.WriteLine("*** 2 corner prior: N({0}, {1}), N({2},{3}), transitionVariance={4} ***", priorMeans[0, 0], priorVariances[0, 0],
                                  priorMeans[gridSizeI - 1, gridSizeJ - 1], priorVariances[gridSizeI - 1, gridSizeJ - 1], transitionVariance);
            }
            else if (priorForm == "2CornerA") // Priors from NW and SE
            {
                priorMeans[0, 0] = priorMean;
                priorVariances[0, 0] = priorVariance;
                priorMeans[gridSizeI - 1, gridSizeJ - 1] = -priorMean;
                priorVariances[gridSizeI - 1, gridSizeJ - 1] = priorVariance;
            }
            else if (priorForm == "2CornerB") // Priors from NE and SW
            {
                priorMeans[0, gridSizeJ - 1] = priorMean;
                priorVariances[0, gridSizeJ - 1] = priorVariance;
                priorMeans[gridSizeI - 1, 0] = -priorMean;
                priorVariances[gridSizeI - 1, 0] = priorVariance;
            }
            else if (priorForm == "spottyRandom")
            {
                for (int i = 0; i < gridSizeI; i++)
                {
                    for (int j = 0; j < gridSizeJ; j++)
                    {
                        if (Bernoulli.Sample(informativeObservationProbability))
                        {
                            priorMeans[i, j] = Gaussian.Sample(singletonHyperpriorMeanMean, singletonHyperpriorMeanVariance);
                            priorVariances[i, j] = Gamma.Sample(singletonHyperpriorVarianceShape, singletonHyperpriorVarianceScale);
                            numInfoSources++;
                        }
                    }
                }
                transitionVariance = Gamma.Sample(singletonHyperpriorVarianceShape, singletonHyperpriorVarianceScale);
            }
            else
            {
                Console.WriteLine("No prior chosen");
                throw new Exception();
            }

            ExperimentResult experimentResult = new ExperimentResult();
            experimentResult.parameters["priorForm"] = priorForm;

            experimentResult.parameters["priorMean"] = priorMean.ToString(CultureInfo.InvariantCulture);
            experimentResult.parameters["priorVariance"] = priorVariance.ToString(CultureInfo.InvariantCulture);


            //experimentResult.parameters["singletonHyperpriorMeanMean"] = singletonHyperpriorMeanMean.ToString();
            //experimentResult.parameters["singletonHyperpriorMeanVariance"] = singletonHyperpriorMeanVariance.ToString();
            //experimentResult.parameters["varianceHyperPriorScale"] = singletonHyperpriorVarianceScale.ToString();
            //experimentResult.parameters["varianceHyperPriorShape"] = singletonHyperpriorVarianceShape.ToString();

            if (priorForm == "random1Corner")
            {
                experimentResult.parameters["priorMean[0,0]"] = priorMeans[0, 0].ToString(CultureInfo.InvariantCulture);
                experimentResult.parameters["priorVariances[0,0]"] = priorVariances[0, 0].ToString(CultureInfo.InvariantCulture);
            }
            if (priorForm == "random2Corner")
            {
                experimentResult.parameters["priorMean[0,0]"] = priorMeans[0, 0].ToString(CultureInfo.InvariantCulture);
                experimentResult.parameters["priorVariances[0,0]"] = priorVariances[0, 0].ToString(CultureInfo.InvariantCulture);
                experimentResult.parameters["priorMean[Ni-1,Nj-1]"] = priorMeans[gridSizeI - 1, gridSizeJ - 1].ToString(CultureInfo.InvariantCulture);
                experimentResult.parameters["priorVariances[Ni-1,Nj-1]"] = priorVariances[gridSizeI - 1, gridSizeJ - 1].ToString(CultureInfo.InvariantCulture);
            }
            if (priorForm == "spottyRandom")
            {
                experimentResult.parameters["informativeObservationProbability"] = informativeObservationProbability.ToString(CultureInfo.InvariantCulture);
            }
            experimentResult.parameters["observationMean"] = observationMean.ToString(CultureInfo.InvariantCulture);
            experimentResult.parameters["observationVariance"] = observationVariance.ToString(CultureInfo.InvariantCulture);
            experimentResult.parameters["transitionVariance"] = transitionVariance.ToString(CultureInfo.InvariantCulture);
            experimentResult.parameters["numInfoSources"] = numInfoSources.ToString(CultureInfo.InvariantCulture);


            Dictionary<string, double[]> resultMeans = new Dictionary<string, double[]>();
            Dictionary<string, double[]> residuals = new Dictionary<string, double[]>();

            //Glo.GloBrowser.Add("resultMeans", resultMeans);
            //Glo.GloBrowser.Add("residuals", residuals);
            ////Glo.GloBrowser.Browser.Settings.AlternativePluginFolder = @"C:\Users\t-dtarlo\AppData\Local\Apps\2.0\EYR7K7R8.E1V\W60L5PCE.H7L\glo...tion_1e483dbaadd1a5a6_0001.0000_25fb220e3f161d00\Plugins";

            List<string> schedulesList = new List<string>(schedules);

            for (int s = 0; s < schedulesList.Count; s++)
            {
                bool serial;
                string scheduleName = schedulesList[s];
                Console.WriteLine(scheduleName);

                for (int numDownUpsPerIteration = 1; numDownUpsPerIteration <= 2; numDownUpsPerIteration++)
                {
                    for (int serialCounter = 1; serialCounter < 2; serialCounter++)
                    {
                        serial = (serialCounter == 1);
                        if (scheduleName == "Default" && serial)
                            continue;
                        if (scheduleName == "Default" && numDownUpsPerIteration > 1)
                            continue;
                        if (!serial && numDownUpsPerIteration > 1)
                            continue;
                        if (scheduleName == "SE-NW" && numDownUpsPerIteration <= 1)
                            continue;

                        try
                        //if (true)
                        {
                            InferenceResult result;
                            if (scheduleName == "Default")
                            {
                                result = FadingGridDefaultSchedule(iterations, priorMeans, priorVariances, transitionVariance,
                                                                   observationMean, observationVariance, gridSizeI, gridSizeJ);
                            }
                            else if (scheduleName == "E-W-S-N")
                            {
                                result = FadingGridRowsColsGridSchedule(iterations, gridPassesPerIteration, numDownUpsPerIteration, serial, priorMeans, priorVariances,
                                                                        transitionVariance,
                                                                        observationMean, observationVariance, gridSizeI, gridSizeJ);
                            }
                            else if (scheduleName == "SE-NW")
                            {
                                if (!serial)
                                {
                                    result = FadingGridTwoPhaseJohnGridSchedule(iterations, gridPassesPerIteration, numDownUpsPerIteration, priorMeans, priorVariances,
                                                                                transitionVariance, observationMean, observationVariance, gridSizeI, gridSizeJ);
                                }
                                else
                                {
                                    result = FadingGridRecommendedJohnGridSchedule(iterations, gridPassesPerIteration, numDownUpsPerIteration, priorMeans, priorVariances,
                                                                                   transitionVariance, observationMean, observationVariance, gridSizeI, gridSizeJ);
                                }
                            }
                            else if (scheduleName == "Approach 2")
                            {
                                result = FadingGridApproach2Schedule(iterations, gridPassesPerIteration, numDownUpsPerIteration, serial, priorMeans, priorVariances,
                                                                     transitionVariance, observationMean, observationVariance, gridSizeI, gridSizeJ);
                            }
                            else
                            {
                                Console.WriteLine("Don't know how to handle {0} {1}", scheduleName, serial);
                                result = null;
                            }

                            string displayScheduleName;
                            if (scheduleName == "Default")
                            {
                                displayScheduleName = scheduleName;
                            }
                            else
                            {
                                if (serial)
                                {
                                    displayScheduleName = "Serial " + scheduleName;
                                }
                                else
                                {
                                    displayScheduleName = "Two Phase " + scheduleName;
                                }

                                if (numDownUpsPerIteration > 1)
                                {
                                    displayScheduleName = String.Format("{0} {1} Up Downs/it", displayScheduleName, numDownUpsPerIteration);
                                }
                            }

                            if (result != null)
                            {
                                experimentResult.AddInferenceResult(displayScheduleName, result);
                                resultMeans.Add(displayScheduleName, result.resultMeans[0]);
                                residuals.Add(displayScheduleName, result.residuals);
                            }
                        }
                        catch (ControlledExperimentException)
                        //if (false)
                        {
                            string displayScheduleName;
                            if (scheduleName == "Default")
                            {
                                displayScheduleName = scheduleName;
                            }
                            else
                            {
                                if (serial)
                                {
                                    displayScheduleName = "Serial " + scheduleName;
                                }
                                else
                                {
                                    displayScheduleName = "Two Phase " + scheduleName;
                                }

                                if (numDownUpsPerIteration > 1)
                                {
                                    displayScheduleName = String.Format("{0} {1} Up Downs/it", displayScheduleName, numDownUpsPerIteration);
                                }
                            }
                            displayScheduleName += " (failed)";

                            Console.WriteLine("Exception in {0}", displayScheduleName);
                            //Console.WriteLine(e);
                            //throw e;
                            double[] emptyResults = new double[0];

                            resultMeans.Add(displayScheduleName, emptyResults);
                            residuals.Add(displayScheduleName, emptyResults);
                        }
                    }
                }
            }

            return experimentResult;
        }

        // unrolled model, automatic schedule
        public static InferenceResult FadingGridDefaultSchedule(int iterations, double[,] priorMeans, double[,] priorVariances, double transitionVariance,
                                                                double observationMean, double observationVariance, int gridSizeI, int gridSizeJ)
        {
            // this is a toy version of the model used by, extended to be a grid rather than chain:
            // "Window-Based Expectation Propagation for Adaptive Signal Detection in Flat-Fading Channels"
            // Yuan Qi and Thomas P. Minka
            // IEEE TRANSACTIONS ON WIRELESS COMMUNICATIONS, VOL. 6, NO. 1, JANUARY 2007

            Variable<double>[,] state = new Variable<double>[gridSizeI, gridSizeJ];
            Variable<double>[,] observation = new Variable<double>[gridSizeI, gridSizeJ];
            double[,] lastMeans = new double[gridSizeI, gridSizeJ];

            for (int i = 0; i < gridSizeI; i++)
            {
                for (int j = 0; j < gridSizeJ; j++)
                {
                    if (i == 0 && j == 0)
                    {
                        state[i, j] = Variable.GaussianFromMeanAndVariance(priorMeans[i, j], priorVariances[i, j]);
                    }
                    else
                    {
                        state[i, j] = Variable.GaussianFromMeanAndVariance(priorMeans[i, j], priorVariances[i, j]);
                    }

                    if (i > 0)
                    {
                        Variable.ConstrainEqualRandom(state[i, j] - state[i - 1, j], new Gaussian(0, transitionVariance));
                    }
                    if (j > 0)
                    {
                        Variable.ConstrainEqualRandom(state[i, j] - state[i, j - 1], new Gaussian(0, transitionVariance));
                    }

                    state[i, j].Name = "state_" + i + "_" + j;
                    observation[i, j] = Variable.New<double>().Named("observation" + i + "_" + j);

                    Variable<bool> symbol = Variable.Bernoulli(.5).Named("symbol" + i + "_" + j);
                    using (Variable.If(symbol))
                    {
                        observation[i, j].SetTo(Variable.GaussianFromMeanAndVariance(state[i, j], observationVariance));
                    }
                    using (Variable.IfNot(symbol))
                    {
                        observation[i, j].SetTo(Variable.GaussianFromMeanAndVariance(-state[i, j], observationVariance));
                    }
                    observation[i, j].ObservedValue = observationMean;
                }
            }

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;

            InferenceResult result = new InferenceResult(iterations, gridSizeI, gridSizeJ);

            try
            {
                for (int i = 0; i < gridSizeI * gridSizeJ; i++)
                {
                    result.resultMeans.Add(new double[iterations]);
                }

                for (int numIters = 0; numIters < iterations; numIters++)
                {
                    engine.NumberOfIterations = numIters;
                    Gaussian firstState = engine.Infer<Gaussian>(state[0, 0]);
                    Gaussian middleState = engine.Infer<Gaussian>(state[gridSizeI / 2, gridSizeJ / 2]);
                    Gaussian lastState = engine.Infer<Gaussian>(state[gridSizeI - 1, gridSizeJ - 1]);
                    //Console.WriteLine("[default] {0}: {1}, {2}, {3}", numIters + 1, firstState, middleState, lastState);
                    result.resultMeans[0][numIters] = lastState.GetMean();

                    double residual = 0;
                    for (int i = 0; i < gridSizeI; i++)
                    {
                        for (int j = 0; j < gridSizeJ; j++)
                        {
                            double newMean = engine.Infer<Gaussian>(state[i, j]).GetMean();
                            if (numIters > 0)
                            {
                                residual += System.Math.Pow((lastMeans[i, j] - newMean), 2);
                            }
                            lastMeans[i, j] = newMean;
                        }
                    }
                    if (numIters > 0)
                    {
                        result.residuals[numIters] = residual;
                        /*
                        if (residual < 1e-5)
                        {
                            for (int i = numIters; i < iterations; i++)
                            {
                                result.residuals[i] = 0;
                                result.resultMeans[0][i] = lastState.GetMean();
                            }
                            return result;
                        }*/
                    }
                }
            }
            catch (Exception)
            {
                result.crashed = true;
            }
            return result;
        }

        public static InferenceResult FadingGridRowsColsGridSchedule(int iterations, int gridPassesPerIteration, int numDownUpsPerIteration, bool serialUpDown,
                                                                     double[,] priorMeans, double[,] priorVariances,
                                                                     double transitionVariance, double observationMean, double observationVariance, int gridSizeI,
                                                                     int gridSizeJ)
        {
            Variable<double>[,] stateCopy = new Variable<double>[gridSizeI, gridSizeJ];

            Variable<Gaussian>[,] upwardMessage = new Variable<Gaussian>[gridSizeI, gridSizeJ];
            Variable<Gaussian>[,] downwardMessage = new Variable<Gaussian>[gridSizeI, gridSizeJ];

            Variable<double>[,] state = new Variable<double>[gridSizeI, gridSizeJ];
            Variable<double>[,] observation = new Variable<double>[gridSizeI, gridSizeJ];
            double[,] lastMeans = new double[gridSizeI, gridSizeJ];

            // Need these so we can call InferAll
            Variable<double>[] allStates = new Variable<double>[gridSizeI * gridSizeJ];
            Variable<double>[] allStateCopies = new Variable<double>[gridSizeI * gridSizeJ];

            for (int i = 0; i < gridSizeI; i++)
            {
                for (int j = 0; j < gridSizeJ; j++)
                {
                    if (i == 0 && j == 0)
                    {
                        state[i, j] = Variable.GaussianFromMeanAndVariance(priorMeans[i, j], priorVariances[i, j]);
                    }
                    else
                    {
                        state[i, j] = Variable.GaussianFromMeanAndVariance(priorMeans[i, j], priorVariances[i, j]);
                    }
                    allStates[i * gridSizeJ + j] = state[i, j];

                    state[i, j].Name = "state_" + i + "_" + j;
                    observation[i, j] = Variable.New<double>().Named("observation_" + i + "_" + j);

                    upwardMessage[i, j] = Variable.New<Gaussian>().Named("upward_" + i + "_" + j);
                    Variable.ConstrainEqualRandom(state[i, j], upwardMessage[i, j]);

                    downwardMessage[i, j] = Variable.New<Gaussian>().Named("downward_" + i + "_" + j);
                    stateCopy[i, j] = Variable.Random<double, Gaussian>(downwardMessage[i, j]).Named("stateCopy" + i + "_" + j);
                    allStateCopies[i * gridSizeJ + j] = stateCopy[i, j];


                    Variable<bool> symbol = Variable.Bernoulli(.5).Named("symbol_" + i + "_" + j);
                    using (Variable.If(symbol))
                    {
                        observation[i, j].SetTo(Variable.GaussianFromMeanAndVariance(stateCopy[i, j], observationVariance));
                    }
                    using (Variable.IfNot(symbol))
                    {
                        observation[i, j].SetTo(Variable.GaussianFromMeanAndVariance(-stateCopy[i, j], observationVariance));
                    }
                    observation[i, j].ObservedValue = observationMean;
                }
            }

            InferenceResult result = new InferenceResult(iterations, gridSizeI, gridSizeJ);

            try
            {
                for (int i = 0; i < gridSizeI * gridSizeJ; i++)
                {
                    result.resultMeans.Add(new double[iterations]);
                }

                InferenceEngine engine2 = new InferenceEngine();
                Gaussian[,] prior = new Gaussian[gridSizeI, gridSizeJ];
                for (int i = 0; i < gridSizeI; i++)
                {
                    for (int j = 0; j < gridSizeJ; j++)
                    {
                        prior[i, j] = new Gaussian(priorMeans[i, j], priorVariances[i, j]);
                    }
                }


                Gaussian[,] stateMarginal = new Gaussian[gridSizeI, gridSizeJ];

                // Cardinal directions are used for the grid directions, while "up" and "down" are
                // reserved for observations.
                // We use the convention that messages are indexed based on where the message is going,
                // so e.g. eastMessage[0, 1] is from state[0, 0] to state[0, 1].
                Gaussian[,] eastMessage = new Gaussian[gridSizeI, gridSizeJ];
                Gaussian[,] westMessage = new Gaussian[gridSizeI, gridSizeJ];
                Gaussian[,] northMessage = new Gaussian[gridSizeI, gridSizeJ];
                Gaussian[,] southMessage = new Gaussian[gridSizeI, gridSizeJ];

                for (int i = 0; i < gridSizeI; i++)
                {
                    for (int j = 0; j < gridSizeJ; j++)
                    {
                        upwardMessage[i, j].ObservedValue = Gaussian.Uniform();
                        downwardMessage[i, j].ObservedValue = Gaussian.Uniform();
                        eastMessage[i, j] = Gaussian.Uniform();
                        westMessage[i, j] = Gaussian.Uniform();
                        northMessage[i, j] = Gaussian.Uniform();
                        southMessage[i, j] = Gaussian.Uniform();
                    }
                }

                for (int iter = 0; iter < iterations; iter++)
                {
                    // East pass
                    for (int i = 0; i < gridSizeI; i++)
                    {
                        for (int j = 0; j < gridSizeJ; j++)
                        {
                            downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                            if (serialUpDown)
                            {
                                upwardMessage[i, j].ObservedValue = engine2.Infer<Gaussian>(stateCopy[i, j]) / downwardMessage[i, j].ObservedValue;
                                //double newMean2 = upwardMessage[i, j].ObservedValue.GetMean();
                            }
                            stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                            CHECK_IS_PROPER(stateMarginal[i, j]);

                            if (j < gridSizeJ - 1)
                            {
                                eastMessage[i, j + 1] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / westMessage[i, j], 1 / transitionVariance);
                            }
                        }
                    }
                    // West pass
                    for (int i = gridSizeI - 1; i >= 0; i--)
                    {
                        for (int j = gridSizeJ - 1; j >= 0; j--)
                        {
                            downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                            stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                            CHECK_IS_PROPER(stateMarginal[i, j]);

                            if (j > 0)
                            {
                                westMessage[i, j - 1] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / eastMessage[i, j], 1 / transitionVariance);
                            }
                        }
                    }

                    // South pass
                    for (int j = 0; j < gridSizeJ; j++)
                    {
                        for (int i = 0; i < gridSizeI; i++)
                        {
                            downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                            if (serialUpDown && numDownUpsPerIteration > 1)
                            {
                                upwardMessage[i, j].ObservedValue = engine2.Infer<Gaussian>(stateCopy[i, j]) / downwardMessage[i, j].ObservedValue;
                            }
                            stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;

                            CHECK_IS_PROPER(stateMarginal[i, j]);
                            /*
                            if (i == 2 && j == 3 && iter == 1)
                            {
                                Console.WriteLine("Interesting break point");
                            }
                            */

                            if (i < gridSizeI - 1)
                            {
                                southMessage[i + 1, j] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / northMessage[i, j], 1 / transitionVariance);
                            }
                        }
                    }

                    // North pass
                    for (int j = gridSizeJ - 1; j >= 0; j--)
                    {
                        for (int i = gridSizeI - 1; i >= 0; i--)
                        {
                            downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                            stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                            CHECK_IS_PROPER(stateMarginal[i, j]);

                            if (i > 0)
                            {
                                northMessage[i - 1, j] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / southMessage[i, j], 1 / transitionVariance);
                            }
                        }
                    }

                    // Second phase
                    if (!serialUpDown)
                    {
                        for (int i = 0; i < gridSizeI; i++)
                        {
                            for (int j = 0; j < gridSizeJ; j++)
                            {
                                downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                                upwardMessage[i, j].ObservedValue = engine2.Infer<Gaussian>(stateCopy[i, j]) / downwardMessage[i, j].ObservedValue;
                                stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                                CHECK_IS_PROPER(stateMarginal[i, j]);
                            }
                        }
                    }

                    result.scheduleBadness.Add(MessageGridPrecisions(upwardMessage));
                    result.intermediateDownwardMeans.Add(MessageGridMeans(downwardMessage));
                    result.intermediateDownwardPrecisions.Add(MessageGridPrecisions(downwardMessage));
                    //Glo.GloBrowser.Add(String.Format("iteration {0}", iter), UpwardMessagePrecisions(upwardMessage));

                    Gaussian firstState = stateMarginal[0, 0];
                    Gaussian middleState = stateMarginal[gridSizeI / 2, gridSizeJ / 2];
                    Gaussian lastState = stateMarginal[gridSizeI - 1, gridSizeJ - 1];
                    //Console.WriteLine("{0}: {1}, {2}, {3}", numIters, firstState, middleState, lastState);
                    result.resultMeans[0][iter] = lastState.GetMean();
                    double residual = 0;
                    for (int i = 0; i < gridSizeI; i++)
                    {
                        for (int j = 0; j < gridSizeJ; j++)
                        {
                            double newMean = 0;
                            try
                            {
                                newMean = stateMarginal[i, j].GetMean();
                            }
                            catch (Exception)
                            {
                                Console.WriteLine("Exception at {0}, {1}", i, j);
                                return result;
                            }
                            if (iter > 0)
                            {
                                residual += System.Math.Pow((lastMeans[i, j] - newMean), 2);
                            }
                            lastMeans[i, j] = newMean;
                        }
                    }
                    if (iter > 0)
                    {
                        result.residuals[iter] = residual;
                        /*
                        if (residual < 1e-5)
                        {
                            for (int i = iter; i < iterations; i++)
                            {
                                result.residuals[i] = 0;
                                result.resultMeans[0][i] = lastState.GetMean();
                            }
                            return result;
                        }*/
                    }
                }
            }
            catch (ControlledExperimentException)
            {
                result.crashed = true;
            }
            return result;
        }

        public static InferenceResult FadingGridApproach2Schedule(int iterations, int gridPassesPerIteration, int numDownUpsPerIteration, bool serialUpDown,
                                                                  double[,] priorMeans, double[,] priorVariances,
                                                                  double transitionVariance, double observationMean, double observationVariance, int gridSizeI, int gridSizeJ)
        {
            Variable<double>[,] stateCopy = new Variable<double>[gridSizeI, gridSizeJ];

            Variable<Gaussian>[,] upwardMessage = new Variable<Gaussian>[gridSizeI, gridSizeJ];
            Variable<Gaussian>[,] downwardMessage = new Variable<Gaussian>[gridSizeI, gridSizeJ];

            Variable<double>[,] state = new Variable<double>[gridSizeI, gridSizeJ];
            Variable<double>[,] observation = new Variable<double>[gridSizeI, gridSizeJ];
            double[,] lastMeans = new double[gridSizeI, gridSizeJ];

            // Need these so we can call InferAll
            Variable<double>[] allStates = new Variable<double>[gridSizeI * gridSizeJ];
            Variable<double>[] allStateCopies = new Variable<double>[gridSizeI * gridSizeJ];

            for (int i = 0; i < gridSizeI; i++)
            {
                for (int j = 0; j < gridSizeJ; j++)
                {
                    if (i == 0 && j == 0)
                    {
                        state[i, j] = Variable.GaussianFromMeanAndVariance(priorMeans[i, j], priorVariances[i, j]);
                    }
                    else
                    {
                        state[i, j] = Variable.GaussianFromMeanAndVariance(priorMeans[i, j], priorVariances[i, j]);
                    }
                    allStates[i * gridSizeJ + j] = state[i, j];

                    state[i, j].Name = "state_" + i + "_" + j;
                    observation[i, j] = Variable.New<double>().Named("observation_" + i + "_" + j);

                    upwardMessage[i, j] = Variable.New<Gaussian>().Named("upward_" + i + "_" + j);
                    Variable.ConstrainEqualRandom(state[i, j], upwardMessage[i, j]);

                    downwardMessage[i, j] = Variable.New<Gaussian>().Named("downward_" + i + "_" + j);
                    stateCopy[i, j] = Variable.Random<double, Gaussian>(downwardMessage[i, j]).Named("stateCopy" + i + "_" + j);
                    allStateCopies[i * gridSizeJ + j] = stateCopy[i, j];


                    Variable<bool> symbol = Variable.Bernoulli(.5).Named("symbol_" + i + "_" + j);
                    using (Variable.If(symbol))
                    {
                        observation[i, j].SetTo(Variable.GaussianFromMeanAndVariance(stateCopy[i, j], observationVariance));
                    }
                    using (Variable.IfNot(symbol))
                    {
                        observation[i, j].SetTo(Variable.GaussianFromMeanAndVariance(-stateCopy[i, j], observationVariance));
                    }
                    observation[i, j].ObservedValue = observationMean;
                }
            }

            InferenceResult result = new InferenceResult(iterations, gridSizeI, gridSizeJ);

            try
            {
                for (int i = 0; i < gridSizeI * gridSizeJ; i++)
                {
                    result.resultMeans.Add(new double[iterations]);
                }

                InferenceEngine engine2 = new InferenceEngine();
                Gaussian[,] prior = new Gaussian[gridSizeI, gridSizeJ];
                for (int i = 0; i < gridSizeI; i++)
                {
                    for (int j = 0; j < gridSizeJ; j++)
                    {
                        prior[i, j] = new Gaussian(priorMeans[i, j], priorVariances[i, j]);
                    }
                }

                Gaussian[,] stateMarginal = new Gaussian[gridSizeI, gridSizeJ];

                // Cardinal directions are used for the grid directions, while "up" and "down" are
                // reserved for observations.
                // We use the convention that messages are indexed based on where the message is going,
                // so e.g. eastMessage[0, 1] is from state[0, 0] to state[0, 1].
                Gaussian[,] eastMessage = new Gaussian[gridSizeI, gridSizeJ];
                Gaussian[,] westMessage = new Gaussian[gridSizeI, gridSizeJ];
                Gaussian[,] northMessage = new Gaussian[gridSizeI, gridSizeJ];
                Gaussian[,] southMessage = new Gaussian[gridSizeI, gridSizeJ];

                for (int i = 0; i < gridSizeI; i++)
                {
                    for (int j = 0; j < gridSizeJ; j++)
                    {
                        upwardMessage[i, j].ObservedValue = Gaussian.Uniform();
                        downwardMessage[i, j].ObservedValue = Gaussian.Uniform();
                        eastMessage[i, j] = Gaussian.Uniform();
                        westMessage[i, j] = Gaussian.Uniform();
                        northMessage[i, j] = Gaussian.Uniform();
                        southMessage[i, j] = Gaussian.Uniform();
                    }
                }

                for (int iter = 0; iter < iterations; iter++)
                {
                    // East-west-south
                    for (int i = 0; i < gridSizeI; i++)
                    {
                        // East pass
                        for (int j = 0; j < gridSizeJ; j++)
                        {
                            downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                            if (serialUpDown)
                            {
                                upwardMessage[i, j].ObservedValue = engine2.Infer<Gaussian>(stateCopy[i, j]) / downwardMessage[i, j].ObservedValue;
                            }
                            stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                            CHECK_IS_PROPER(stateMarginal[i, j]);

                            if (j < gridSizeJ - 1)
                            {
                                eastMessage[i, j + 1] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / westMessage[i, j], 1 / transitionVariance);
                            }
                        }

                        // West pass
                        for (int j = gridSizeJ - 1; j >= 0; j--)
                        {
                            downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                            stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                            CHECK_IS_PROPER(stateMarginal[i, j]);

                            if (j > 0)
                            {
                                westMessage[i, j - 1] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / eastMessage[i, j], 1 / transitionVariance);
                            }
                        }

                        // South one step pass
                        if (i < gridSizeI - 1)
                        {
                            for (int j = 0; j < gridSizeJ; j++)
                            {
                                southMessage[i + 1, j] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / northMessage[i, j], 1 / transitionVariance);
                            }
                        }
                    }

                    if (serialUpDown && numDownUpsPerIteration == 2)
                    {
                        // Do a full West-East-North pass
                        for (int i = gridSizeI - 1; i >= 0; i--)
                        {
                            // West pass
                            for (int j = gridSizeJ - 1; j >= 0; j--)
                            {
                                downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                                upwardMessage[i, j].ObservedValue = engine2.Infer<Gaussian>(stateCopy[i, j]) / downwardMessage[i, j].ObservedValue;
                                stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                                CHECK_IS_PROPER(stateMarginal[i, j]);

                                if (j > 0)
                                {
                                    westMessage[i, j - 1] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / eastMessage[i, j], 1 / transitionVariance);
                                }
                            }

                            // East pass
                            for (int j = 0; j < gridSizeJ; j++)
                            {
                                downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                                stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                                CHECK_IS_PROPER(stateMarginal[i, j]);

                                if (j < gridSizeJ - 1)
                                {
                                    eastMessage[i, j + 1] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / westMessage[i, j], 1 / transitionVariance);
                                }
                            }

                            // North one step pass
                            if (i > 0)
                            {
                                for (int j = 0; j < gridSizeJ; j++)
                                {
                                    northMessage[i - 1, j] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / southMessage[i, j], 1 / transitionVariance);
                                }
                            }
                        }
                    }
                    else
                    {
                        // North pass
                        for (int j = gridSizeJ - 1; j >= 0; j--)
                        {
                            for (int i = gridSizeI - 1; i >= 0; i--)
                            {
                                downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                                stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                                CHECK_IS_PROPER(stateMarginal[i, j]);

                                if (i > 0)
                                {
                                    northMessage[i - 1, j] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / southMessage[i, j], 1 / transitionVariance);
                                }
                            }
                        }
                    }
                    // Second phase
                    if (!serialUpDown)
                    {
                        for (int i = 0; i < gridSizeI; i++)
                        {
                            for (int j = 0; j < gridSizeJ; j++)
                            {
                                downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                                upwardMessage[i, j].ObservedValue = engine2.Infer<Gaussian>(stateCopy[i, j]) / downwardMessage[i, j].ObservedValue;
                                stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                                CHECK_IS_PROPER(stateMarginal[i, j]);
                            }
                        }
                    }
                    result.scheduleBadness.Add(MessageGridPrecisions(upwardMessage));
                    result.intermediateDownwardMeans.Add(MessageGridMeans(downwardMessage));
                    result.intermediateDownwardPrecisions.Add(MessageGridMeans(downwardMessage));
                    Gaussian firstState = stateMarginal[0, 0];
                    Gaussian middleState = stateMarginal[gridSizeI / 2, gridSizeJ / 2];
                    Gaussian lastState = stateMarginal[gridSizeI - 1, gridSizeJ - 1];
                    //Console.WriteLine("{0}: {1}, {2}, {3}", numIters, firstState, middleState, lastState);
                    result.resultMeans[0][iter] = lastState.GetMean();

                    double residual = 0;
                    for (int i = 0; i < gridSizeI; i++)
                    {
                        for (int j = 0; j < gridSizeJ; j++)
                        {
                            double newMean;
                            try
                            {
                                newMean = stateMarginal[i, j].GetMean();
                                if (iter > 0)
                                {
                                    residual += System.Math.Pow((lastMeans[i, j] - newMean), 2);
                                }
                                lastMeans[i, j] = newMean;
                            }
                            catch (ImproperDistributionException)
                            {
                                lastMeans[i, j] = -1;
                            }
                        }
                    }
                    if (iter > 0)
                    {
                        result.residuals[iter] = residual;
                        /*
                        if (residual < 1e-5)
                        {
                            for (int i = iter; i < iterations; i++)
                            {
                                result.residuals[i] = 0;
                                result.resultMeans[0][i] = lastState.GetMean();
                            }
                            return result;
                        }*/
                    }
                }
            }
            catch (ControlledExperimentException)
            {
                result.crashed = true;
            }
            catch (ImproperMessageException)
            {
                result.crashed = true;
            }
            return result;
        }

        public static InferenceResult FadingGridTwoPhaseJohnGridSchedule(int iterations, int gridPassesPerIteration, int numDownUpsPerIteration, double[,] priorMeans,
                                                                         double[,] priorVariances, double transitionVariance,
                                                                         double observationMean, double observationVariance, int gridSizeI, int gridSizeJ)
        {
            Variable<double>[,] stateCopy = new Variable<double>[gridSizeI, gridSizeJ];

            Variable<Gaussian>[,] upwardMessage = new Variable<Gaussian>[gridSizeI, gridSizeJ];
            Variable<Gaussian>[,] downwardMessage = new Variable<Gaussian>[gridSizeI, gridSizeJ];

            Variable<double>[,] state = new Variable<double>[gridSizeI, gridSizeJ];
            Variable<double>[,] observation = new Variable<double>[gridSizeI, gridSizeJ];
            double[,] lastMeans = new double[gridSizeI, gridSizeJ];

            // Need these so we can call InferAll
            Variable<double>[] allStates = new Variable<double>[gridSizeI * gridSizeJ];
            Variable<double>[] allStateCopies = new Variable<double>[gridSizeI * gridSizeJ];

            for (int i = 0; i < gridSizeI; i++)
            {
                for (int j = 0; j < gridSizeJ; j++)
                {
                    if (i == 0 && j == 0)
                    {
                        state[i, j] = Variable.GaussianFromMeanAndVariance(priorMeans[i, j], priorVariances[i, j]);
                    }
                    else
                    {
                        state[i, j] = Variable.GaussianFromMeanAndVariance(priorMeans[i, j], priorVariances[i, j]);
                    }
                    allStates[i * gridSizeJ + j] = state[i, j];

                    state[i, j].Name = "state_" + i + "_" + j;
                    observation[i, j] = Variable.New<double>().Named("observation_" + i + "_" + j);

                    upwardMessage[i, j] = Variable.New<Gaussian>().Named("upward_" + i + "_" + j);
                    Variable.ConstrainEqualRandom(state[i, j], upwardMessage[i, j]);

                    downwardMessage[i, j] = Variable.New<Gaussian>().Named("downward_" + i + "_" + j);
                    stateCopy[i, j] = Variable.Random<double, Gaussian>(downwardMessage[i, j]).Named("stateCopy" + i + "_" + j);
                    allStateCopies[i * gridSizeJ + j] = stateCopy[i, j];


                    Variable<bool> symbol = Variable.Bernoulli(.5).Named("symbol_" + i + "_" + j);
                    using (Variable.If(symbol))
                    {
                        observation[i, j].SetTo(Variable.GaussianFromMeanAndVariance(stateCopy[i, j], observationVariance));
                    }
                    using (Variable.IfNot(symbol))
                    {
                        observation[i, j].SetTo(Variable.GaussianFromMeanAndVariance(-stateCopy[i, j], observationVariance));
                    }
                    observation[i, j].ObservedValue = observationMean;
                }
            }

            InferenceResult result = new InferenceResult(iterations, gridSizeI, gridSizeJ);
            try
            {
                for (int i = 0; i < gridSizeI * gridSizeJ; i++)
                {
                    result.resultMeans.Add(new double[iterations]);
                }

                InferenceEngine engine2 = new InferenceEngine();
                Gaussian[,] prior = new Gaussian[gridSizeI, gridSizeJ];
                for (int i = 0; i < gridSizeI; i++)
                {
                    for (int j = 0; j < gridSizeJ; j++)
                    {
                        prior[i, j] = new Gaussian(priorMeans[i, j], priorVariances[i, j]);
                    }
                }

                Gaussian[,] stateMarginal = new Gaussian[gridSizeI, gridSizeJ];

                // Cardinal directions are used for the grid directions, while "up" and "down" are
                // reserved for observations.
                // We use the convention that messages are indexed based on where the message is going,
                // so e.g. eastMessage[0, 1] is from state[0, 0] to state[0, 1].
                Gaussian[,] eastMessage = new Gaussian[gridSizeI, gridSizeJ];
                Gaussian[,] westMessage = new Gaussian[gridSizeI, gridSizeJ];
                Gaussian[,] northMessage = new Gaussian[gridSizeI, gridSizeJ];
                Gaussian[,] southMessage = new Gaussian[gridSizeI, gridSizeJ];

                for (int i = 0; i < gridSizeI; i++)
                {
                    for (int j = 0; j < gridSizeJ; j++)
                    {
                        upwardMessage[i, j].ObservedValue = Gaussian.Uniform();
                        downwardMessage[i, j].ObservedValue = Gaussian.Uniform();
                        eastMessage[i, j] = Gaussian.Uniform();
                        westMessage[i, j] = Gaussian.Uniform();
                        northMessage[i, j] = Gaussian.Uniform();
                        southMessage[i, j] = Gaussian.Uniform();
                    }
                }

                for (int iter = 0; iter < iterations; iter++)
                {
                    // South East pass
                    for (int i = 0; i < gridSizeI; i++)
                    {
                        for (int j = 0; j < gridSizeJ; j++)
                        {
                            downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                            stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                            CHECK_IS_PROPER(stateMarginal[i, j]);

                            //Console.WriteLine("   --> SE downwardMessage[{0}, {1}] = {2}", i, j, downwardMessage[i, j].ObservedValue);
                            //Console.WriteLine("   --> SE inferredStateCopy[{0}, {1}] = {2}", i, j, engine2.Infer<Gaussian>(stateCopy[i, j]));
                            //Console.WriteLine("   --> SE upwardMessage[{0}, {1}] = {2}", i, j, upwardMessage[i, j].ObservedValue);
                            //Console.WriteLine("   --> SE stateMarginal[{0}, {1}] = {2}", i, j, stateMarginal[i, j]);

                            if (j < gridSizeJ - 1)
                            {
                                eastMessage[i, j + 1] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / westMessage[i, j], 1 / transitionVariance);
                            }
                            if (i < gridSizeI - 1)
                            {
                                southMessage[i + 1, j] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / northMessage[i, j], 1 / transitionVariance);
                            }
                        }
                    }

                    for (int i = gridSizeI - 1; i >= 0; i--)
                    {
                        for (int j = gridSizeJ - 1; j >= 0; j--)
                        {
                            downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                            //upwardMessage[i, j].ObservedValue = engine2.Infer<Gaussian>(stateCopy[i, j]) / downwardMessage[i, j].ObservedValue;
                            stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                            CHECK_IS_PROPER(stateMarginal[i, j]);

                            //Console.WriteLine("   --> NW downwardMessage[{0}, {1}] = {2}", i, j, downwardMessage[i, j].ObservedValue);
                            //Console.WriteLine("   --> NW upwardMessage[{0}, {1}] = {2}", i, j, upwardMessage[i, j].ObservedValue); 
                            //Console.WriteLine("   --> NW stateMarginal[{0}, {1}] = {2}", i, j, stateMarginal[i, j]);

                            if (j > 0)
                            {
                                westMessage[i, j - 1] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / eastMessage[i, j], 1 / transitionVariance);
                            }
                            if (i > 0)
                            {
                                northMessage[i - 1, j] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / southMessage[i, j], 1 / transitionVariance);
                            }
                        }
                    }

                    // Second phase
                    for (int i = 0; i < gridSizeI; i++)
                    {
                        for (int j = 0; j < gridSizeJ; j++)
                        {
                            downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                            upwardMessage[i, j].ObservedValue = engine2.Infer<Gaussian>(stateCopy[i, j]) / downwardMessage[i, j].ObservedValue;
                            stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                            CHECK_IS_PROPER(stateMarginal[i, j]);
                        }
                    }
                    result.scheduleBadness.Add(MessageGridPrecisions(upwardMessage));
                    result.intermediateDownwardMeans.Add(MessageGridMeans(downwardMessage));
                    result.intermediateDownwardPrecisions.Add(MessageGridMeans(downwardMessage));

                    Gaussian firstState = stateMarginal[0, 0];
                    Gaussian middleState = stateMarginal[gridSizeI / 2, gridSizeJ / 2];
                    Gaussian lastState = stateMarginal[gridSizeI - 1, gridSizeJ - 1];
                    //Console.WriteLine("{0}: {1}, {2}, {3}", numIters, firstState, middleState, lastState);
                    result.resultMeans[0][iter] = lastState.GetMean();
                    double residual = 0;
                    for (int i = 0; i < gridSizeI; i++)
                    {
                        for (int j = 0; j < gridSizeJ; j++)
                        {
                            double newMean = stateMarginal[i, j].GetMean();
                            if (iter > 0)
                            {
                                residual += System.Math.Pow((lastMeans[i, j] - newMean), 2);
                            }
                            lastMeans[i, j] = newMean;
                        }
                    }
                    if (iter > 0)
                    {
                        result.residuals[iter] = residual;
                        /*
                        if (residual < 1e-5)
                        {
                            for (int i = iter; i < iterations; i++)
                            {
                                result.residuals[i] = 0;
                                result.resultMeans[0][i] = lastState.GetMean();
                            }
                            return result;
                        }*/
                    }
                }
            }
            catch (ControlledExperimentException)
            {
                result.crashed = true;
            }

            return result;
        }

        public static InferenceResult FadingGridRecommendedJohnGridSchedule(int iterations, int gridPassesPerIteration, int numDownUpsPerIteration, double[,] priorMeans,
                                                                            double[,] priorVariances, double transitionVariance,
                                                                            double observationMean, double observationVariance, int gridSizeI, int gridSizeJ)
        {
            Variable<double>[,] stateCopy = new Variable<double>[gridSizeI, gridSizeJ];

            Variable<Gaussian>[,] upwardMessage = new Variable<Gaussian>[gridSizeI, gridSizeJ];
            Variable<Gaussian>[,] downwardMessage = new Variable<Gaussian>[gridSizeI, gridSizeJ];

            Variable<double>[,] state = new Variable<double>[gridSizeI, gridSizeJ];
            Variable<double>[,] observation = new Variable<double>[gridSizeI, gridSizeJ];
            double[,] lastMeans = new double[gridSizeI, gridSizeJ];

            // Need these so we can call InferAll
            Variable<double>[] allStates = new Variable<double>[gridSizeI * gridSizeJ];
            Variable<double>[] allStateCopies = new Variable<double>[gridSizeI * gridSizeJ];

            for (int i = 0; i < gridSizeI; i++)
            {
                for (int j = 0; j < gridSizeJ; j++)
                {
                    if (i == 0 && j == 0)
                    {
                        state[i, j] = Variable.GaussianFromMeanAndVariance(priorMeans[i, j], priorVariances[i, j]);
                    }
                    else
                    {
                        state[i, j] = Variable.GaussianFromMeanAndVariance(priorMeans[i, j], priorVariances[i, j]);
                    }
                    allStates[i * gridSizeJ + j] = state[i, j];

                    state[i, j].Name = "state_" + i + "_" + j;
                    observation[i, j] = Variable.New<double>().Named("observation_" + i + "_" + j);

                    upwardMessage[i, j] = Variable.New<Gaussian>().Named("upward_" + i + "_" + j);
                    Variable.ConstrainEqualRandom(state[i, j], upwardMessage[i, j]);

                    downwardMessage[i, j] = Variable.New<Gaussian>().Named("downward_" + i + "_" + j);
                    stateCopy[i, j] = Variable.Random<double, Gaussian>(downwardMessage[i, j]).Named("stateCopy" + i + "_" + j);
                    allStateCopies[i * gridSizeJ + j] = stateCopy[i, j];


                    Variable<bool> symbol = Variable.Bernoulli(.5).Named("symbol_" + i + "_" + j);
                    using (Variable.If(symbol))
                    {
                        observation[i, j].SetTo(Variable.GaussianFromMeanAndVariance(stateCopy[i, j], observationVariance));
                    }
                    using (Variable.IfNot(symbol))
                    {
                        observation[i, j].SetTo(Variable.GaussianFromMeanAndVariance(-stateCopy[i, j], observationVariance));
                    }
                    observation[i, j].ObservedValue = observationMean;
                }
            }

            InferenceResult result = new InferenceResult(iterations, gridSizeI, gridSizeJ);
            try
            {
                for (int i = 0; i < gridSizeI * gridSizeJ; i++)
                {
                    result.resultMeans.Add(new double[iterations]);
                }

                InferenceEngine engine2 = new InferenceEngine();
                Gaussian[,] prior = new Gaussian[gridSizeI, gridSizeJ];
                for (int i = 0; i < gridSizeI; i++)
                {
                    for (int j = 0; j < gridSizeJ; j++)
                    {
                        prior[i, j] = new Gaussian(priorMeans[i, j], priorVariances[i, j]);
                    }
                }

                Gaussian[,] stateMarginal = new Gaussian[gridSizeI, gridSizeJ];

                // Cardinal directions are used for the grid directions, while "up" and "down" are
                // reserved for observations.
                // We use the convention that messages are indexed based on where the message is going,
                // so e.g. eastMessage[0, 1] is from state[0, 0] to state[0, 1].
                Gaussian[,] eastMessage = new Gaussian[gridSizeI, gridSizeJ];
                Gaussian[,] westMessage = new Gaussian[gridSizeI, gridSizeJ];
                Gaussian[,] northMessage = new Gaussian[gridSizeI, gridSizeJ];
                Gaussian[,] southMessage = new Gaussian[gridSizeI, gridSizeJ];

                for (int i = 0; i < gridSizeI; i++)
                {
                    for (int j = 0; j < gridSizeJ; j++)
                    {
                        upwardMessage[i, j].ObservedValue = Gaussian.Uniform();
                        downwardMessage[i, j].ObservedValue = Gaussian.Uniform();
                        eastMessage[i, j] = Gaussian.Uniform();
                        westMessage[i, j] = Gaussian.Uniform();
                        northMessage[i, j] = Gaussian.Uniform();
                        southMessage[i, j] = Gaussian.Uniform();
                    }
                }

                for (int iter = 0; iter < iterations; iter++)
                {
                    // South East pass
                    for (int i = 0; i < gridSizeI; i++)
                    {
                        for (int j = 0; j < gridSizeJ; j++)
                        {
                            downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                            upwardMessage[i, j].ObservedValue = engine2.Infer<Gaussian>(stateCopy[i, j]) / downwardMessage[i, j].ObservedValue;
                            stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                            CHECK_IS_PROPER(stateMarginal[i, j]);

                            //Console.WriteLine("   --> SE downwardMessage[{0}, {1}] = {2}", i, j, downwardMessage[i, j].ObservedValue);
                            //Console.WriteLine("   --> SE inferredStateCopy[{0}, {1}] = {2}", i, j, engine2.Infer<Gaussian>(stateCopy[i, j]));
                            //Console.WriteLine("   --> SE upwardMessage[{0}, {1}] = {2}", i, j, upwardMessage[i, j].ObservedValue);
                            //Console.WriteLine("   --> SE stateMarginal[{0}, {1}] = {2}", i, j, stateMarginal[i, j]);

                            if (j < gridSizeJ - 1)
                            {
                                eastMessage[i, j + 1] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / westMessage[i, j], 1 / transitionVariance);
                            }
                            if (i < gridSizeI - 1)
                            {
                                southMessage[i + 1, j] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / northMessage[i, j], 1 / transitionVariance);
                            }
                        }
                    }

                    if (iter == 1)
                    {
                        Console.WriteLine("Break");
                    }

                    for (int i = gridSizeI - 1; i >= 0; i--)
                    {
                        for (int j = gridSizeJ - 1; j >= 0; j--)
                        {
                            downwardMessage[i, j].ObservedValue = prior[i, j] * eastMessage[i, j] * westMessage[i, j] * southMessage[i, j] * northMessage[i, j];
                            //upwardMessage[i, j].ObservedValue = engine2.Infer<Gaussian>(stateCopy[i, j]) / downwardMessage[i, j].ObservedValue;
                            if (numDownUpsPerIteration > 1) // || (i == 0 && j == 0 && iter == 1))
                            {
                                upwardMessage[i, j].ObservedValue = engine2.Infer<Gaussian>(stateCopy[i, j]) / downwardMessage[i, j].ObservedValue;
                            }
                            stateMarginal[i, j] = downwardMessage[i, j].ObservedValue * upwardMessage[i, j].ObservedValue;
                            if (i == 0 && j == 0)
                            {
                                Console.WriteLine("Interesting break point");
                            }
                            CHECK_IS_PROPER(stateMarginal[i, j]);

                            //Console.WriteLine("   --> NW downwardMessage[{0}, {1}] = {2}", i, j, downwardMessage[i, j].ObservedValue);
                            //Console.WriteLine("   --> NW upwardMessage[{0}, {1}] = {2}", i, j, upwardMessage[i, j].ObservedValue); 
                            //Console.WriteLine("   --> NW stateMarginal[{0}, {1}] = {2}", i, j, stateMarginal[i, j]);

                            if (j > 0)
                            {
                                westMessage[i, j - 1] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / eastMessage[i, j], 1 / transitionVariance);
                            }
                            if (i > 0)
                            {
                                northMessage[i - 1, j] = GaussianOp.SampleAverageConditional(stateMarginal[i, j] / southMessage[i, j], 1 / transitionVariance);
                            }
                        }
                    }
                    result.scheduleBadness.Add(MessageGridPrecisions(upwardMessage));
                    result.intermediateDownwardMeans.Add(MessageGridMeans(downwardMessage));
                    result.intermediateDownwardPrecisions.Add(MessageGridMeans(downwardMessage));

                    Gaussian firstState = stateMarginal[0, 0];
                    Gaussian middleState = stateMarginal[gridSizeI / 2, gridSizeJ / 2];
                    Gaussian lastState = stateMarginal[gridSizeI - 1, gridSizeJ - 1];
                    //Console.WriteLine("{0}: {1}, {2}, {3}", numIters, firstState, middleState, lastState);
                    result.resultMeans[0][iter] = lastState.GetMean();

                    double residual = 0;
                    for (int i = 0; i < gridSizeI; i++)
                    {
                        for (int j = 0; j < gridSizeJ; j++)
                        {
                            double newMean;
                            try
                            {
                                newMean = stateMarginal[i, j].GetMean();
                                if (iter > 0)
                                {
                                    residual += System.Math.Pow((lastMeans[i, j] - newMean), 2);
                                }
                                lastMeans[i, j] = newMean;
                            }
                            catch (ImproperDistributionException)
                            {
                                lastMeans[i, j] = -1;
                            }
                        }
                    }
                    if (iter > 0)
                    {
                        result.residuals[iter] = residual;
                        /*
                        if (residual < 1e-5)
                        {
                            for (int i = iter; i < iterations; i++)
                            {
                                result.residuals[i] = 0;
                                result.resultMeans[0][i] = lastState.GetMean();
                            }
                            return result;
                        }*/
                    }
                }
            }
            catch (ControlledExperimentException)
            {
                result.crashed = true;
            }
            catch (ImproperMessageException)
            {
                result.crashed = true;
            }

            return result;
        }


        // Potts model grids with 4-neighbor connectivity tests
        public void PottsTestCSharp4GridDiscrete()
        {
            // A 4-connected grid using C# loops to define variables and connectivity (unrolled)
            int sideLengthI = 5;
            int sideLengthJ = 5;
            bool useUniformSingletons = false;

            Variable<int>[,] pixels = new Variable<int>[sideLengthI, sideLengthJ];
            for (int i = 0; i < sideLengthI; i++)
            {
                for (int j = 0; j < sideLengthJ; j++)
                {
                    if (useUniformSingletons)
                    {
                        pixels[i, j] = Variable.DiscreteUniform(2);
                    }
                    else
                    {
                        if (i % 2 == 0)
                        {
                            pixels[i, j] = Variable.Discrete(new double[] { .51, .49 }).Named(String.Format("x_{0}{1}", i, j));
                        }
                        else
                        {
                        }

                        if ((i + j) % 3 == 0)
                        {
                            pixels[i, j] = Variable.Discrete(new double[] { .1, .9 }).Named(String.Format("x_{0}{1}", i, j));
                        }
                        else if ((i + j) % 3 == 1)
                        {
                            pixels[i, j] = Variable.Discrete(new double[] { .6, .4 }).Named(String.Format("x_{0}{1}", i, j));
                        }
                        else if ((i + j) % 3 == 2)
                        {
                            pixels[i, j] = Variable.Discrete(new double[] { .8, .2 }).Named(String.Format("x_{0}{1}", i, j));
                        }
                        else
                        {
                            throw new Exception("Should not reach here.");
                        }
                    }
                }
            }

            for (int i = 0; i < sideLengthI; i++)
            {
                for (int j = 0; j < sideLengthJ; j++)
                {
                    Variable<int> variableIJ = pixels[i, j];
                    if (j < sideLengthJ - 1)
                    {
                        Variable<int> rightNeighbor = pixels[i, j + 1];
                        //Variable.Constrain(Undirected.Potts, variableIJ, rightNeighbor, Variable.Constant(.6));
                        Variable<bool> tmpVar = (variableIJ == rightNeighbor);
                        Variable.ConstrainEqualRandom(tmpVar, new Bernoulli(.1));
                    }

                    if (i < sideLengthI - 1)
                    {
                        Variable<int> belowNeighbor = pixels[i + 1, j];
                        //Variable.Constrain(Undirected.Potts, variableIJ, belowNeighbor, Variable.Constant(.6));
                        //Variable<bool> tmpVar = (variableIJ == rightNeighbor);
                        Variable.ConstrainEqualRandom(variableIJ == belowNeighbor, new Bernoulli(.1));
                    }
                }
            }

            //InferenceEngine ie = new InferenceEngine { Algorithm = new MaxProductBeliefPropagation() };
            // These don't work
            //InferenceEngine ie = new InferenceEngine { Algorithm = new VariationalMessagePassing() };
            InferenceEngine ie = new InferenceEngine
            {
                Algorithm = new ExpectationPropagation()
            };

            for (int i = 0; i < sideLengthI; i++)
            {
                for (int j = 0; j < sideLengthJ; j++)
                {
                    Console.WriteLine("Dist over pixels[" + i + ", " + j + "] = " + ie.Infer<Discrete>(pixels[i, j]));
                }
            }
        }

        public void PottsTestCSharp4GridBool()
        {
            // A 4-connected grid using C# loops to define variables and connectivity (unrolled)
            int sideLengthI = 3;
            int sideLengthJ = 3;
            bool useUniformSingletons = false;

            Variable<bool>[,] pixels = new Variable<bool>[sideLengthI, sideLengthJ];
            for (int i = 0; i < sideLengthI; i++)
            {
                for (int j = 0; j < sideLengthJ; j++)
                {
                    if (useUniformSingletons)
                    {
                        //pixels[i, j] = Variable.DiscreteUniform(2);
                    }
                    else
                    {
                        if ((i + j) % 3 == 0)
                        {
                            pixels[i, j] = Variable.Bernoulli(.9).Named(String.Format("x_{0}{1}", i, j));
                        }
                        else if ((i + j) % 3 == 1)
                        {
                            pixels[i, j] = Variable.Bernoulli(.4).Named(String.Format("x_{0}{1}", i, j));
                        }
                        else if ((i + j) % 3 == 2)
                        {
                            pixels[i, j] = Variable.Bernoulli(.2).Named(String.Format("x_{0}{1}", i, j));
                        }
                        else
                        {
                            throw new Exception("Should not reach here.");
                        }
                    }
                }
            }

            for (int i = 0; i < sideLengthI; i++)
            {
                for (int j = 0; j < sideLengthJ; j++)
                {
                    Variable<bool> variableIJ = pixels[i, j];
                    if (j < sideLengthJ - 1)
                    {
                        Variable<bool> rightNeighbor = pixels[i, j + 1];
                        //Variable.Constrain(Undirected.Potts, variableIJ, rightNeighbor, Variable.Constant(.6));
                        Variable.ConstrainEqualRandom(variableIJ & rightNeighbor, new Bernoulli(.9));
                    }

                    if (i < sideLengthI - 1)
                    {
                        Variable<bool> belowNeighbor = pixels[i + 1, j];
                        //Variable.Constrain(Undirected.Potts, variableIJ, belowNeighbor, Variable.Constant(.6));
                        //Variable<bool> tmpVar = (variableIJ == rightNeighbor);
                        Variable.ConstrainEqualRandom(variableIJ & belowNeighbor, new Bernoulli(.9));
                    }
                }
            }

            //InferenceEngine ie = new InferenceEngine { Algorithm = new MaxProductBeliefPropagation() };
            // These don't work
            //InferenceEngine ie = new InferenceEngine { Algorithm = new VariationalMessagePassing() };
            InferenceEngine ie = new InferenceEngine
            {
                Algorithm = new ExpectationPropagation()
            };

            for (int i = 0; i < sideLengthI; i++)
            {
                for (int j = 0; j < sideLengthJ; j++)
                {
                    Console.WriteLine("Dist over pixels[" + i + ", " + j + "] = " + ie.Infer<Discrete>(pixels[i, j]));
                }
            }
        }

        // The simplest test of synchronous vs. asynchronous schedules we can think up
        public void OneVariableTest()
        {
            // Default schedule confused
            // timesteps = 4
            // observationVariance = .01;
            // var0 = 2
            // mu0 = 1
            // mu1 = .75
            // mu2 = 1

            // Synchronous schedule crashes
            // int timesteps = 6;
            // double observationVariance = .01;
            // double var0 = .25;
            // double mu0 = 1;
            // double mu1 = .7;
            // double mu2 = .3;

            // Synchronous schedule oscillates
            // int timesteps = 6;
            // double observationVariance = .1;
            // double var0 = .5;
            // double mu0 = .5;
            // double mu1 = .7;
            // double mu2 = 1;

            // Lots of confusion
            // int timesteps = 5;
            // double observationVariance = .1;
            // double var0 = .5;
            // double mu0 = .5;
            // double mu1 = .7;
            // double mu2 = 1;

            // Typical results
            // int timesteps = 4;
            // double observationVariance = .1;
            // double var0 = .5;
            // double mu0 = 1;
            // double mu1 = 2;
            // double mu2 = .1;

            // Default is fastest
            // int timesteps = 4;
            // double observationVariance = .01;
            // double var0 = 4;
            // double mu0 = .1;
            // double mu1 = 1;
            // double mu2 = .5;

            int timesteps = 5;
            double observationVariance = .001;
            double var0 = .1;
            double mu0 = 1;
            double mu1 = 1;
            double mu2 = 1;

            int iterations = 20;

            Dictionary<string, double[]> resultMeans = new Dictionary<string, double[]>();
            //Glo.GloBrowser.Start("");
            //Glo.GloBrowser.Add("resultMeans", resultMeans);
            try
            {
                double exactMean = OneVariableExactMeanBruteForce(timesteps, observationVariance, var0, mu0, mu1, mu2);
                double[] exactMeans = new double[iterations];
                for (int i = 0; i < iterations; i++)
                {
                    exactMeans[i] = exactMean;
                }

                Console.WriteLine("Exact mean: {0}", exactMean);
                resultMeans.Add("Exact (Discretized)", exactMeans);
            }
            catch (Exception)
            {
                Console.WriteLine("Exception in exact calculation");
            }

            OneVariableDefaultSchedule(timesteps, iterations, observationVariance, var0, mu0, mu1, mu2);
            try
            {
                Console.WriteLine("Default Schedule");
                double[] defaultScheduleMeans = OneVariableDefaultSchedule(timesteps, iterations, observationVariance, var0, mu0, mu1, mu2);
                resultMeans.Add("Default Schedule", defaultScheduleMeans);
            }
            catch (Exception)
            {
                Console.WriteLine("Exception in default schedule calculation");
            }

            try
            {
                Console.WriteLine("Synchronous");
                double[] synchronousScheduleMeans = OneVariableSynchronous(timesteps, iterations, observationVariance, var0, mu0, mu1, mu2);
                resultMeans.Add("Synchronous Schedule", synchronousScheduleMeans);

                /*
                // Running average of synchronous results
                 * double[] runningSynchronousMean = new double[iterations];
                double totalSum = 0;
                for (int i = 0; i < iterations; i++)
                {
                    totalSum += synchronousScheduleMeans[i];
                    runningSynchronousMean[i] = totalSum / (i + 1);
                }

                resultMeans.Add("Synchronous Running Average", runningSynchronousMean);
                */
            }
            catch (Exception)
            {
                Console.WriteLine("Exception in synchronous schedule calculation");
            }


            try
            {
                Console.WriteLine("Asynchronous");
                double[] asynchronousScheduleMeans = OneVariableAsynchronous(timesteps, iterations, observationVariance, var0, mu0, mu1, mu2);
                resultMeans.Add("Asynchronous Schedule", asynchronousScheduleMeans);
            }
            catch (Exception)
            {
                Console.WriteLine("Exception in synchronous schedule calculation");
            }

            //Glo.GloBrowser.Show("resultMeans", resultMeans);
        }

        public static double OneVariableExactMeanBruteForce(int timesteps, double observationVariance, double var0, double mu0, double mu1, double mu2)
        {
            double step = .00001;
            double minSlice = -5;
            double maxSlice = 5;
            int numSlices = (int)((maxSlice - minSlice) / step);

            double[] xVals = new double[numSlices];

            double[] f0Vals = new double[numSlices];
            double[] f1Vals = new double[numSlices];
            double[,] f2Vals = new double[numSlices, timesteps - 1];
            double[] fTotVals = new double[numSlices];

            double sumFX = 0;
            double sumF = 0;

            for (int t = 0; t < numSlices; t++)
            {
                xVals[t] = minSlice + t * step;
                f0Vals[t] = System.Math.Exp(Gaussian.GetLogProb(xVals[t], mu0, var0));
                f1Vals[t] = System.Math.Exp(Gaussian.GetLogProb(xVals[t], -mu1, observationVariance)) + System.Math.Exp(Gaussian.GetLogProb(xVals[t], mu1, observationVariance));
                fTotVals[t] = f0Vals[t] * f1Vals[t];

                for (int timestep = 0; timestep < timesteps - 1; timestep++)
                {
                    f2Vals[t, timestep] = System.Math.Exp(Gaussian.GetLogProb(xVals[t], -mu2, observationVariance)) +
                                          System.Math.Exp(Gaussian.GetLogProb(xVals[t], mu2, observationVariance));
                    fTotVals[t] *= f2Vals[t, timestep];
                }

                sumFX += fTotVals[t] * xVals[t];
                sumF += fTotVals[t];
            }
            Console.WriteLine("{0} / {1}", sumFX, sumF);
            return sumFX / sumF;
        }

        public static double OneVariableExactMean(int timesteps, double observationVariance, double var0, double mu0, double mu1, double mu2)
        {
            // This comes from a closed form computation that only considers the two observation case

            // Doesn't appear to be working right now.  -dt
            Console.WriteLine("OneVariableExactMean is buggy, do not use");
            throw new Exception();


            if (timesteps != 2)
            {
                Console.WriteLine("Exact computation only valid for timesteps = 2 (not {0})", timesteps);
                throw new Exception();
            }

            double mu_12 = (mu1 + mu2) / 2; // mu_12
            double mu_1m2 = (mu1 - mu2) / 2; // mu_1-2
            double mu_m12 = (-mu1 + mu2) / 2; // mu_-12
            double mu_m1m2 = (-mu1 - mu2) / 2; // mu_-1-2

            double c1 = System.Math.Exp(-System.Math.Pow(mu1 - mu2, 2) / (2 * observationVariance));
            double c2 = System.Math.Exp(-System.Math.Pow(mu1 + mu2, 2) / (2 * observationVariance));
            double c3 = System.Math.Exp(-System.Math.Pow(-mu1 - mu2, 2) / (2 * observationVariance));
            double c4 = System.Math.Exp(-System.Math.Pow(-mu1 + mu2, 2) / (2 * observationVariance));
            double normalizerC = c1 + c2 + c3 + c4;

            if (normalizerC == 0)
            {
                Console.WriteLine("ERROR: Underflow in exact computation");
                throw new Exception();
            }

            c1 = c1 / normalizerC;
            c2 = c2 / normalizerC;
            c3 = c3 / normalizerC;
            c4 = c4 / normalizerC;

            double varFinal = 1 / (2.0 / observationVariance + 1.0 / var0);

            double componentVariance = observationVariance / 2.0 + var0;
            double d1 = c1 * System.Math.Exp(Gaussian.GetLogProb(mu0, mu_12, componentVariance));
            double d2 = c2 * System.Math.Exp(Gaussian.GetLogProb(mu0, mu_1m2, componentVariance));
            double d3 = c3 * System.Math.Exp(Gaussian.GetLogProb(mu0, mu_m12, componentVariance));
            double d4 = c4 * System.Math.Exp(Gaussian.GetLogProb(mu0, mu_m1m2, componentVariance));
            double normalizerD = d1 + d2 + d3 + d4;
            d1 = d1 / normalizerD;
            d2 = d2 / normalizerD;
            d3 = d3 / normalizerD;
            d4 = d4 / normalizerD;

            if (normalizerD == 0)
            {
                Console.WriteLine("ERROR: Underflow in exact computation");
                throw new Exception();
            }

            double mu_012 = varFinal * (mu0 / var0 + (2 * mu_12) / observationVariance);
            double mu_01m2 = varFinal * (mu0 / var0 + (2 * mu_1m2) / observationVariance);
            double mu_0m12 = varFinal * (mu0 / var0 + (2 * mu_m12) / observationVariance);
            double mu_0m1m2 = varFinal * (mu0 / var0 + (2 * mu_m1m2) / observationVariance);

            double muFinal = d1 * mu_012 + d2 * mu_01m2 + d3 * mu_0m12 + d4 * mu_0m1m2;

            Console.WriteLine("mu_12s: {0} {1} {2} {3}", mu_12, mu_1m2, mu_m12, mu_m1m2);
            Console.WriteLine("cs: {0} {1} {2} {3}", c1, c2, c3, c4);
            Console.WriteLine("ds: {0} {1} {2} {3}", d1, d2, d3, d4);
            Console.WriteLine("mu_012s: {0} {1} {2} {3}", mu_012, mu_01m2, mu_0m12, mu_0m1m2);

            Console.WriteLine("Exact Mean: " + muFinal);
            return muFinal;
        }

        public static double[] OneVariableDefaultSchedule(int timesteps, int iterations, double observationVariance, double var0,
                                                          double mu0, double mu1, double mu2)
        {
            Variable<double> state = new Variable<double>();
            state = Variable.GaussianFromMeanAndVariance(mu0, var0);

            state.Name = "state";
            Variable<double>[] observation = new Variable<double>[timesteps];

            for (int time = 0; time < timesteps; time++)
            {
                observation[time] = Variable.New<double>().Named("observation" + time);

                Variable<bool> symbol = Variable.Bernoulli(.5).Named("symbol" + time);
                using (Variable.If(symbol))
                {
                    observation[time] = Variable.GaussianFromMeanAndVariance(state, observationVariance);
                }
                using (Variable.IfNot(symbol))
                {
                    observation[time] = Variable.GaussianFromMeanAndVariance(-state, observationVariance);
                }

                if (time <= 0)
                    observation[time].ObservedValue = mu1;
                else
                    observation[time].ObservedValue = mu2;
            }

            InferenceEngine engine = new InferenceEngine();

            double[] posteriorMeans = new double[iterations];

            for (int iteration = 0; iteration < iterations; iteration++)
            {
                engine.NumberOfIterations = iteration + 1;
                Gaussian result = engine.Infer<Gaussian>(state);
                Console.WriteLine("{0}", result);
                posteriorMeans[iteration] = result.GetMean();
            }
            return posteriorMeans;
        }

        // Synchronous schedule
        public static double[] OneVariableSynchronous(int timesteps, int iterations, double observationVariance, double var0, double mu0, double mu1, double mu2)
        {
            Variable<double>[] state = new Variable<double>[1];
            Variable<double>[] stateCopy = new Variable<double>[timesteps];
            Variable<Gaussian>[] upwardMessage = new Variable<Gaussian>[timesteps];
            Variable<Gaussian>[] downwardMessage = new Variable<Gaussian>[timesteps];
            Variable<double>[] observation = new Variable<double>[timesteps];

            for (int time = 0; time < timesteps; time++)
            {
                if (time == 0)
                {
                    state[0] = Variable.GaussianFromMeanAndVariance(mu0, var0);
                    state[0].Name = "state" + time;
                }

                upwardMessage[time] = Variable.New<Gaussian>().Named("upward" + time);
                Variable.ConstrainEqualRandom(state[0], upwardMessage[time]);
                downwardMessage[time] = Variable.New<Gaussian>().Named("downward" + time);
                stateCopy[time] = Variable.Random<double, Gaussian>(downwardMessage[time]).Named("stateCopy" + time);
                observation[time] = Variable.New<double>().Named("observation" + time);
                Variable<bool> symbol = Variable.Bernoulli(.5).Named("symbol" + time);
                using (Variable.If(symbol))
                {
                    observation[time] = Variable.GaussianFromMeanAndVariance(stateCopy[time], observationVariance);
                }
                using (Variable.IfNot(symbol))
                {
                    observation[time] = Variable.GaussianFromMeanAndVariance(-stateCopy[time], observationVariance);
                }
                if (time <= 0)
                    observation[time].ObservedValue = mu1;
                else
                    observation[time].ObservedValue = mu2;
                //else observation[time].ObservedValue = 1;
            }
            InferenceEngine engine = new InferenceEngine();
            InferenceEngine engine2 = new InferenceEngine();
            double[] resultMeans = new double[iterations];
            for (int numIters = 0; numIters < iterations; numIters++)
            {
                for (int t = 0; t < timesteps; t++)
                {
                    upwardMessage[t].ObservedValue = Gaussian.Uniform();
                    downwardMessage[t].ObservedValue = Gaussian.Uniform();
                }
                for (int iter = 0; iter < numIters; iter++)
                {
                    // phase 1
                    for (int t = 0; t < timesteps; t++)
                    {
                        downwardMessage[t].ObservedValue = engine.Infer<Gaussian>(state[0]) / upwardMessage[t].ObservedValue;
                    }
                    // phase 2
                    for (int t = 0; t < timesteps; t++)
                    {
                        upwardMessage[t].ObservedValue = engine2.Infer<Gaussian>(stateCopy[t]) / downwardMessage[t].ObservedValue;
                    }
                }
                Gaussian firstState = engine.Infer<Gaussian>(state[0]);
                Console.WriteLine("[2 phase] {0}: {1}", numIters, firstState);
                resultMeans[numIters] = firstState.GetMean();
            }
            return resultMeans;
        }

        // same model but using the recommended schedule from the paper.
        public static double[] OneVariableAsynchronous(int timesteps, int iterations, double observationVariance, double var0, double mu0, double mu1, double mu2)
        {
            Variable<double>[] stateCopy = new Variable<double>[timesteps];
            Variable<Gaussian>[] upwardMessage = new Variable<Gaussian>[timesteps];
            Variable<Gaussian>[] downwardMessage = new Variable<Gaussian>[timesteps];
            Variable<double>[] observation = new Variable<double>[timesteps];

            for (int time = 0; time < timesteps; time++)
            {
                upwardMessage[time] = Variable.New<Gaussian>().Named("upward" + time);
                downwardMessage[time] = Variable.New<Gaussian>().Named("downward" + time);
                stateCopy[time] = Variable.Random<double, Gaussian>(downwardMessage[time]).Named("stateCopy" + time);
                observation[time] = Variable.New<double>().Named("observation" + time);
                Variable<bool> symbol = Variable.Bernoulli(.5).Named("symbol" + time);
                using (Variable.If(symbol))
                {
                    observation[time] = Variable.GaussianFromMeanAndVariance(stateCopy[time], observationVariance);
                }
                using (Variable.IfNot(symbol))
                {
                    observation[time] = Variable.GaussianFromMeanAndVariance(-stateCopy[time], observationVariance);
                }
                if (time <= 0)
                    observation[time].ObservedValue = mu1;
                else
                    observation[time].ObservedValue = mu2;
                Console.Write("{0} ", observation[time].ObservedValue);
                //else observation[time].ObservedValue = 1;
            }
            Console.WriteLine();
            InferenceEngine engine2 = new InferenceEngine();
            Gaussian prior = new Gaussian(mu0, var0);
            double[] resultMeans = new double[iterations];

            for (int numIters = 0; numIters < iterations; numIters++)
            {
                Gaussian[] stateMarginal = new Gaussian[timesteps];
                Gaussian[] forwardMessage = new Gaussian[timesteps];
                Gaussian[] backwardMessage = new Gaussian[timesteps];
                for (int t = 0; t < timesteps; t++)
                {
                    upwardMessage[t].ObservedValue = Gaussian.Uniform();
                    downwardMessage[t].ObservedValue = Gaussian.Uniform();
                    forwardMessage[t] = Gaussian.Uniform();
                    backwardMessage[t] = Gaussian.Uniform();
                }
                forwardMessage[0] = prior;
                for (int iter = 0; iter < numIters; iter++)
                {
                    // forward pass
                    for (int t = 0; t < timesteps; t++)
                    {
                        downwardMessage[t].ObservedValue = forwardMessage[t] * backwardMessage[t];
                        upwardMessage[t].ObservedValue = engine2.Infer<Gaussian>(stateCopy[t]) / downwardMessage[t].ObservedValue;
                        stateMarginal[t] = downwardMessage[t].ObservedValue * upwardMessage[t].ObservedValue;
                        if (t < timesteps - 1)
                            forwardMessage[t + 1] = GaussianOp.SampleAverageConditional(stateMarginal[t] / backwardMessage[t], 100000000000);
                    }
                    // backward pass
                    for (int t = timesteps - 1; t >= 0; t--)
                    {
                        downwardMessage[t].ObservedValue = forwardMessage[t] * backwardMessage[t];
                        // Recomputing the upwardMessage is optional here, but helps
                        //upwardMessage[t].ObservedValue = engine2.Infer<Gaussian>(stateCopy[t]) / downwardMessage[t].ObservedValue;
                        stateMarginal[t] = downwardMessage[t].ObservedValue * upwardMessage[t].ObservedValue;
                        if (t > 0)
                            backwardMessage[t - 1] = GaussianOp.SampleAverageConditional(stateMarginal[t] / forwardMessage[t], 100000000000);
                    }
                }
                Gaussian firstState = stateMarginal[0];
                Gaussian middleState = stateMarginal[timesteps / 2];
                Gaussian lastState = stateMarginal[timesteps - 1];
                Console.WriteLine("[recommended] {0}: {1}", numIters, firstState);
                resultMeans[numIters] = firstState.GetMean();
            }

            return resultMeans;

            /*
            Variable<double>[] stateCopy = new Variable<double>[timesteps];
            Variable<Gaussian>[] upwardMessage = new Variable<Gaussian>[timesteps];
            Variable<Gaussian>[] downwardMessage = new Variable<Gaussian>[timesteps];
            Variable<double>[] observation = new Variable<double>[timesteps];

            Gaussian prior = new Gaussian(mu0, var0);
            Gaussian state = new Gaussian(mu0, var0);

            for (int time = 0; time < timesteps; time++)
            {
                upwardMessage[time] = Variable.New<Gaussian>().Named("upward" + time);
                downwardMessage[time] = Variable.New<Gaussian>().Named("downward" + time);
                stateCopy[time] = Variable.Random<double, Gaussian>(downwardMessage[time]).Named("stateCopy" + time);
                observation[time] = Variable.New<double>().Named("observation" + time);

                Variable<bool> symbol = Variable.Bernoulli(.5).Named("symbol" + time);
                using (Variable.If(symbol))
                {
                    observation[time] = Variable.GaussianFromMeanAndVariance(stateCopy[time], eps);
                }
                using (Variable.IfNot(symbol))
                {
                    observation[time] = Variable.GaussianFromMeanAndVariance(-stateCopy[time], eps);
                }
                if (time <= 0) observation[time].ObservedValue = 3;
                else observation[time].ObservedValue = .1;
                Console.Write("{0} ", observation[time].ObservedValue);
            }
            Console.WriteLine();
            InferenceEngine engine2 = new InferenceEngine();
                    
            for (int numIters = 1; numIters <= 10; numIters++)
            {
                Gaussian[] stateMarginal = new Gaussian[timesteps];

                for (int t = 0; t < timesteps; t++)
                {
                    upwardMessage[t].ObservedValue = Gaussian.Uniform();
                    downwardMessage[t].ObservedValue = Gaussian.Uniform();
                }

                engine2.InferAll(stateCopy);
                for (int iter = 0; iter < numIters; iter++)
                {
                    // forward pass
                    for (int t = 0; t < timesteps; t++)
                    {
                        downwardMessage[t].ObservedValue = prior;
                        upwardMessage[t].ObservedValue = engine2.Infer<Gaussian>(stateCopy[t]) / downwardMessage[t].ObservedValue;
                        stateMarginal[t] = downwardMessage[t].ObservedValue * upwardMessage[t].ObservedValue;
                   }
                    // backward pass
                    for (int t = timesteps - 1; t >= 0; t--)
                    {
                        downwardMessage[t].ObservedValue = forwardMessage[0] * backwardMessage[t];
                        // Recomputing the upwardMessage is optional here, but helps
                        upwardMessage[t].ObservedValue = engine2.Infer<Gaussian>(stateCopy[t]) / downwardMessage[t].ObservedValue;
                        stateMarginal[t] = downwardMessage[t].ObservedValue * upwardMessage[t].ObservedValue;
                        if (t > 0) backwardMessage[t - 1] = GaussianOp.SampleAverageConditional(stateMarginal[t] / forwardMessage[t], 1 / transitionVariance);
                    }
                }
                Gaussian firstState = stateMarginal[0];
                Gaussian lastState = stateMarginal[timesteps - 1];
                Console.WriteLine("{0}: {1}, {2}", numIters, firstState.GetMean().ToString("g4"), lastState.GetMean().ToString("g4"));
            }*/
        }
    }

    // Used for exceptions that we expect to cause when running an experiment.
    // These will get thrown by the individual calls to inference runs, then the
    // higher level experiment functions can handle these gracefully, reporting
    // partial results.
    // We don't want to just catch ImproperMessage or ImproperDistribution messages,
    // because sometimes we want them to send us into debug mode, and sometimes
    // we want to ignore them.  See uses of CHECK_IS_PROPER function for one example
    // of where we don't want to catch improper exceptions.
    public class ControlledExperimentException : Exception
    {
    }


    // A simple class for storing a set of experiment results
    public class ExperimentResult
    {
        public ExperimentResult()
        {
            results = new Dictionary<string, InferenceResult>();
            parameters = new Dictionary<string, string>();
        }


        public void AddInferenceResult(string name, InferenceResult result)
        {
            results.Add(name, result);
        }

        // Construct a dictionary mapping schedule names to residuals, so that
        // Glo will display them nicely.
        public Dictionary<string, double[]> Residuals()
        {
            Dictionary<string, double[]> residuals = new Dictionary<string, double[]>();

            foreach (var entry in results)
            {
                string scheduleName = entry.Key;
                InferenceResult result = entry.Value;

                residuals.Add(scheduleName, result.residuals);
            }
            return residuals;
        }

        // Construct a dictionary mapping schedule names to result means, so that
        // Glo will display them nicely.
        public Dictionary<string, double[]> ResultMeans(int varIndex)
        {
            Dictionary<string, double[]> means = new Dictionary<string, double[]>();

            foreach (var entry in results)
            {
                string scheduleName = entry.Key;
                InferenceResult result = entry.Value;

                means.Add(scheduleName, result.resultMeans[varIndex]);
            }
            return means;
        }

        public Dictionary<string, bool> CrashReport()
        {
            Dictionary<string, bool> crashed = new Dictionary<string, bool>();

            foreach (var entry in results)
            {
                string scheduleName = entry.Key;
                InferenceResult result = entry.Value;

                crashed.Add(scheduleName, result.crashed);
            }
            return crashed;
        }

        public Dictionary<string, List<double[,]>> ScheduleBadnesses()
        {
            Dictionary<string, List<double[,]>> scheduleBadnesses = new Dictionary<string, List<double[,]>>();
            foreach (var entry in results)
            {
                string scheduleName = entry.Key;
                InferenceResult result = entry.Value;

                scheduleBadnesses.Add(scheduleName, result.scheduleBadness);
            }

            return scheduleBadnesses;
        }

        public Dictionary<string, List<double[,]>> IntermediateDownwardMeans()
        {
            Dictionary<string, List<double[,]>> intermediateDownwardMeans = new Dictionary<string, List<double[,]>>();
            foreach (var entry in results)
            {
                string scheduleName = entry.Key;
                InferenceResult result = entry.Value;

                intermediateDownwardMeans.Add(scheduleName, result.intermediateDownwardMeans);
            }

            return intermediateDownwardMeans;
        }

        public Dictionary<string, List<double[,]>> IntermediateDownwardPrecisions()
        {
            Dictionary<string, List<double[,]>> intermediateDownwardPrecisions = new Dictionary<string, List<double[,]>>();
            foreach (var entry in results)
            {
                string scheduleName = entry.Key;
                InferenceResult result = entry.Value;

                intermediateDownwardPrecisions.Add(scheduleName, result.intermediateDownwardPrecisions);
            }

            return intermediateDownwardPrecisions;
        }

        public Dictionary<string, double> ScheduleScores()
        {
            Dictionary<string, double> scheduleScores = new Dictionary<string, double>();
            foreach (var entry in results)
            {
                string scheduleName = entry.Key;
                InferenceResult result = entry.Value;

                double minPrecision = 10000;
                for (int it = 0; it < result.scheduleBadness.Count; it++)
                {
                    for (int i = 0; i <= result.scheduleBadness[it].GetUpperBound(0); i++)
                    {
                        for (int j = 0; j < result.scheduleBadness[it].GetUpperBound(1); j++)
                        {
                            if (minPrecision > result.scheduleBadness[it][i, j])
                            {
                                minPrecision = result.scheduleBadness[it][i, j];
                            }
                        }
                    }
                }
                scheduleScores.Add(scheduleName, minPrecision);
            }

            return scheduleScores;
        }


        public Dictionary<string, bool> ConvergenceReport(int byIteration)
        {
            Dictionary<string, bool> convergenceResults = new Dictionary<string, bool>();
            foreach (var entry in results)
            {
                string scheduleName = entry.Key;
                InferenceResult result = entry.Value;

                convergenceResults.Add(scheduleName, result.DecideIfConvergedAfterIteration(byIteration));
            }
            return convergenceResults;
        }


        public List<string> ScheduleNames()
        {
            List<string> scheduleNames = new List<string>();
            foreach (var entry in results)
            {
                scheduleNames.Add(entry.Key);
            }

            return scheduleNames;
        }


        public string Summary()
        {
            string resultString = "";

            foreach (var entry in results)
            {
                bool converged = entry.Value.DecideIfConverged();
                resultString += entry.Key;
                if (converged)
                {
                    resultString += " converged";
                }
                resultString += ", ";
            }

            return resultString;
        }

        internal Dictionary<string, InferenceResult> results;

        public Dictionary<string, string> parameters;
    }


    // A simple class for storing the results of a single inference run
    public class InferenceResult
    {
        public InferenceResult(int iterations, int gridSizeI, int gridSizeJ)
        {
            numIterations = iterations;
            converged = false;
            resultMeans = new List<double[]>();
            residuals = new double[iterations];
            for (int i = 0; i < iterations; i++) residuals[i] = 1;
            crashed = false;
            convergenceThreshold = 1e-5;
            scheduleBadness = new List<double[,]>();
            intermediateDownwardMeans = new List<double[,]>();
            intermediateDownwardPrecisions = new List<double[,]>();

            dimI = gridSizeI;
            dimJ = gridSizeJ;
        }

        public bool DecideIfConverged()
        {
            if (crashed) return false;
            converged = (residuals[numIterations - 1] < convergenceThreshold);
            return converged;
        }

        public bool DecideIfConvergedAfterIteration(int iteration)
        {
            if (crashed) return false;
            if (iteration == 0) return false;

            bool converged = (residuals[iteration] < convergenceThreshold);

            for (int i = iteration; i < numIterations; i++)
            {
                if (residuals[i] > convergenceThreshold)
                {
                    converged = false;
                }
            }
            return converged;
        }

        private int dimI;
        private int dimJ;


        public string scheduleName;

        public int numIterations;

        public List<double[]> resultMeans;

        public double[] residuals;

        public bool converged;

        public double convergenceThreshold;

        public bool crashed;

        public List<double[,]> scheduleBadness;

        public List<double[,]> intermediateDownwardMeans;

        public List<double[,]> intermediateDownwardPrecisions;
    }

    // A stripped down version of John G's Markov chain class from the forums
    internal class MarkovChain
    {
        public MarkovChain()
        {
        }


        public Gaussian[] Infer3(int numNodesI, int numNodesJ)
        {
            Range nI = new Range(numNodesI);
            Range nJ = new Range(numNodesJ);

            VariableArray2D<double> Nodes = Variable.Array<double>(nI, nJ);

            for (int i = 0; i < numNodesI; i++)
            {
                for (int j = 0; j < numNodesJ; j++)
                {
                }
            }

            return null;
        }


        public Gaussian[] Infer2(int numNodes)
        {
            VariableArray<int> PrevIndices;
            VariableArray<int> NextIndices;
            VariableArray<double> Nodes;

            int numNodes1 = numNodes - 1;

            int[] prevIndices = new int[numNodes1];
            int[] nextIndices = new int[numNodes1];

            for (int i = 0; i < numNodes1; i++)
            {
                prevIndices[i] = i;
                nextIndices[i] = i + 1;
            }

            double priorVariance = 1;

            Range n = new Range(numNodes);

            // Would be nice to let n1 be defined as Range(2, numNodes)
            Range n1 = new Range(numNodes - 1);

            Nodes = Variable.Array<double>(n).Named("Nodes");
            Nodes[n] = Variable.GaussianFromMeanAndVariance(0, 100000).ForEach(n);

            PrevIndices = Variable.Array<int>(n1).Named("prevIndices");
            NextIndices = Variable.Array<int>(n1).Named("nextIndices");
            PrevIndices.ObservedValue = prevIndices;
            NextIndices.ObservedValue = nextIndices;

            // Would be nice not to need these.
            var prev = Variable.Subarray(Nodes, PrevIndices);
            var next = Variable.Subarray(Nodes, NextIndices);

            // Set the prior
            Variable.ConstrainEqualRandom(Nodes[Variable.Constant(0).Named("const0A")], new Gaussian(1, priorVariance));

            // Set the transitions
            using (Variable.ForEach(n1))
            {
                // Would be nice to reference prev[n1] as Nodes[n1].
                // Would be nice to reference next[n1] as Nodes[n1 + 1]

                // Then we could write:
                //    Variable.ConstrainEqualRandom(Nodes[n1] - Nodes[n1 - 1], new Gaussian(0, 1));
                Variable.ConstrainEqualRandom(prev[n1] - next[n1], new Gaussian(0, 1));

                // 
                // If we implemented inequality on ranges:
                //
                // using (Variable.If(n1 < 1)) {
                //    Variable.ConstrainEqualRandom(Nodes[n1], new Gaussian(1, priorVariance));
                // }
                // using (Variable.If(n1 >= 1) {
                //    Variable.ConstrainEqualRandom(Nodes[n1] - Nodes[n1 - 1], new Gaussian(0, 1));
                // }
            }

            // Make a loop
            Variable.ConstrainEqualRandom(Nodes[Variable.Constant(0).Named("const0B")] - Nodes[Variable.Constant(numNodes - 1).Named("constN-1")], new Gaussian(0, 1));

            InferenceEngine engine = new InferenceEngine();
            return engine.Infer<Gaussian[]>(Nodes);
        }


        public Gaussian[] Infer(int numNodes)
        {
            VariableArray<int> PrevIndicesI;
            VariableArray<int> NextIndicesI;
            VariableArray<int> PrevIndicesJ;
            VariableArray<int> NextIndicesJ;
            VariableArray2D<double> Nodes;

            int numNodesI = numNodes;
            int numNodesJ = numNodes;

            int numNodesI1 = numNodesI - 1;
            int numNodesJ1 = numNodesJ - 1;

            int[] prevIndicesI = new int[numNodesI1];
            int[] nextIndicesI = new int[numNodesI1];
            int[] prevIndicesJ = new int[numNodesJ1];
            int[] nextIndicesJ = new int[numNodesJ1];

            for (int i = 0; i < numNodesI1; i++)
            {
                prevIndicesI[i] = i;
                nextIndicesI[i] = i + 1;
            }
            for (int j = 0; j < numNodesJ1; j++)
            {
                prevIndicesJ[j] = j;
                nextIndicesJ[j] = j + 1;
            }

            double priorVariance = 1;

            Range nI = new Range(numNodesI).Named("nI");
            Range nJ = new Range(numNodesJ).Named("nJ");

            // Would be nice to let n1 be defined as Range(2, numNodes)
            Range nI1 = new Range(numNodesI - 1).Named("nI1");
            Range nJ1 = new Range(numNodesJ - 1).Named("nJ1");

            Nodes = Variable.Array<double>(nI, nJ);
            Nodes[nI, nJ] = Variable.GaussianFromMeanAndVariance(0, 100000).ForEach(nI, nJ);

            PrevIndicesI = Variable.Array<int>(nI1);
            NextIndicesI = Variable.Array<int>(nI1);
            PrevIndicesJ = Variable.Array<int>(nJ1);
            NextIndicesJ = Variable.Array<int>(nJ1);

            PrevIndicesI.ObservedValue = prevIndicesI;
            NextIndicesI.ObservedValue = nextIndicesI;
            PrevIndicesJ.ObservedValue = prevIndicesJ;
            NextIndicesJ.ObservedValue = nextIndicesJ;

            //Variable.Subarray(Nodes, new int[] { });

            // Would be nice not to need these.
            //var prevI = Variable.Subarray(Nodes, PrevIndicesI);
            //var nextI = Variable.Subarray(Nodes, NextIndicesI);
            //var prevJ = Variable.Subarray(Nodes, PrevIndicesJ);
            //var nextJ = Variable.Subarray(Nodes, NextIndicesJ);

            // Set the prior
            Variable.ConstrainEqualRandom(Nodes[Variable.Constant(0), Variable.Constant(0)], new Gaussian(1, priorVariance));

            // Set the transitions
            using (Variable.ForEach(nI1))
            {
                using (Variable.ForEach(nJ1))
                {
                    // Would be nice to reference prev[n1] as Nodes[n1].
                    // Would be nice to reference next[n1] as Nodes[n1 + 1]
                    //
                    // Then we could write:
                    //    Variable.ConstrainEqualRandom(Nodes[n1] - Nodes[n1 - 1], new Gaussian(0, 1));
                    //Variable.ConstrainEqualRandom(Nodes[nI1] - Nodes[nI1], new Gaussian(0, 1));


                    // If we implemented inequality on ranges:
                    //
                    // using (Variable.If(n1 < 1)) {
                    //    Variable.ConstrainEqualRandom(Nodes[n1], new Gaussian(1, priorVariance));
                    // }
                    // using (Variable.If(n1 >= 1) {
                    //    Variable.ConstrainEqualRandom(Nodes[n1] - Nodes[n1 - 1], new Gaussian(0, 1));
                    // }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            return engine.Infer<Gaussian[]>(Nodes);
        }
    }


#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}