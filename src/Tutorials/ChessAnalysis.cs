// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tutorials
{
    [Example("Applications", "Comparing chess players through time")]
    public class ChessAnalysis
    {
        public void Run()
        {
            // This example requires EP
            InferenceEngine engine = new InferenceEngine();
            if (!(engine.Algorithm is Algorithms.ExpectationPropagation))
            {
                Console.WriteLine("This example only runs with Expectation Propagation");
                return;
            }

            int nPlayers = 10;
            int nYears = 10;
            Rand.Restart(1);

            var skillPrior = new Gaussian(1200, 800 * 800);
            var drawMarginMeanPrior = new Gaussian(700, 500 * 500);
            var drawMarginPrecisionPrior = Gamma.FromShapeAndRate(2, 500 * 500);
            var performancePrecisionPrior = Gamma.FromShapeAndRate(2, 800 * 800);
            var skillChangePrecisionPrior = Gamma.FromShapeAndRate(2, 26 * 26);
            var drawMarginChangePrecisionPrior = Gamma.FromShapeAndRate(2, 10 * 10);
            var whiteAdvantagePrior = new Gaussian(0, 200 * 200);

            var drawMarginMean = Variable.Random(drawMarginMeanPrior).Named("drawMarginMean");
            var drawMarginPrecision = Variable.Random(drawMarginPrecisionPrior).Named("drawMarginPrecision");
            var performancePrecision = Variable.Random(performancePrecisionPrior).Named("performancePrecision");
            var skillChangePrecision = Variable.Random(skillChangePrecisionPrior).Named("skillChangePrecision");
            var drawMarginChangePrecision = Variable.Random(drawMarginChangePrecisionPrior).Named("drawMarginChangePrecision");
            var whiteAdvantage = Variable.Random(whiteAdvantagePrior).Named("whiteAdvantage");

            Range player = new Range(nPlayers).Named("player");
            Range year = new Range(nYears).Named("year");
            VariableArray<int> firstYear = Variable.Array<int>(player).Named("firstYear");
            var skill = Variable.Array(Variable.Array<double>(player), year).Named("skill");
            var drawMargin = Variable.Array(Variable.Array<double>(player), year).Named("drawMargin");

            using (var yearBlock = Variable.ForEach(year))
            {
                var y = yearBlock.Index;
                using (Variable.If(y == 0))
                {
                    skill[year][player] = Variable.Random(skillPrior).ForEach(player);
                    drawMargin[year][player] = Variable.GaussianFromMeanAndPrecision(drawMarginMean, drawMarginPrecision).ForEach(player);
                }

                using (Variable.If(y > 0))
                {
                    using (Variable.ForEach(player))
                    {
                        Variable<bool> isFirstYear = (firstYear[player] >= y).Named("isFirstYear");
                        using (Variable.If(isFirstYear))
                        {
                            skill[year][player] = Variable.Random(skillPrior);
                            drawMargin[year][player] = Variable.GaussianFromMeanAndPrecision(drawMarginMean, drawMarginPrecision);
                        }

                        using (Variable.IfNot(isFirstYear))
                        {
                            skill[year][player] = Variable.GaussianFromMeanAndPrecision(skill[y - 1][player], skillChangePrecision);
                            drawMargin[year][player] = Variable.GaussianFromMeanAndPrecision(drawMargin[y - 1][player], drawMarginChangePrecision);
                        }
                    }
                }
            }

            // Sample parameter values according to the above model
            firstYear.ObservedValue = Util.ArrayInit(nPlayers, i => Rand.Int(nYears));
            Parameters parameters = new Parameters();
            parameters.drawMarginMean = drawMarginMeanPrior.Sample();
            parameters.drawMarginPrecision = drawMarginPrecisionPrior.Sample();
            parameters.performancePrecision = performancePrecisionPrior.Sample();
            parameters.skillChangePrecision = skillChangePrecisionPrior.Sample();
            parameters.drawMarginChangePrecision = drawMarginChangePrecisionPrior.Sample();
            parameters.whiteAdvantage = whiteAdvantagePrior.Sample();
            parameters.skill = Util.ArrayInit(nYears, y => Util.ArrayInit(nPlayers, i => skillPrior.Sample()));
            parameters.drawMargin = Util.ArrayInit(nYears, y => Util.ArrayInit(nPlayers, i => Gaussian.Sample(parameters.drawMarginMean, parameters.drawMarginPrecision)));
            for (int y = 0; y < nYears; y++)
            {
                for (int i = 0; i < nPlayers; i++)
                {
                    if (y > firstYear.ObservedValue[i])
                    {
                        parameters.skill[y][i] = Gaussian.Sample(parameters.skill[y - 1][i], parameters.skillChangePrecision);
                        parameters.drawMargin[y][i] = Gaussian.Sample(parameters.drawMargin[y - 1][i], parameters.drawMarginChangePrecision);
                    }
                }
            }

            // Sample game outcomes
            int[][] whiteData, blackData, outcomeData;
            GenerateData(parameters, firstYear.ObservedValue, out whiteData, out blackData, out outcomeData);

            bool inferParameters = false;  // make this true to infer additional parameters
            if (!inferParameters)
            {
                // fix the true parameters
                drawMarginMean.ObservedValue = parameters.drawMarginMean;
                drawMarginPrecision.ObservedValue = parameters.drawMarginPrecision;
                performancePrecision.ObservedValue = parameters.performancePrecision;
                skillChangePrecision.ObservedValue = parameters.skillChangePrecision;
                drawMarginChangePrecision.ObservedValue = parameters.drawMarginChangePrecision;
            }

            // Learn the skills from the data
            int[] nGamesData = Util.ArrayInit(nYears, y => outcomeData[y].Length);
            var nGames = Variable.Observed(nGamesData, year).Named("nGames");
            Range game = new Range(nGames[year]).Named("game");
            var whitePlayer = Variable.Observed(whiteData, year, game).Named("whitePlayer");
            var blackPlayer = Variable.Observed(blackData, year, game).Named("blackPlayer");
            var outcome = Variable.Observed(outcomeData, year, game).Named("outcome");
            using (Variable.ForEach(year))
            {
                using (Variable.ForEach(game))
                {
                    var w = whitePlayer[year][game];
                    var b = blackPlayer[year][game];
                    Variable<double> white_performance = Variable.GaussianFromMeanAndPrecision(skill[year][w], performancePrecision).Named("white_performance");
                    Variable<double> black_performance = Variable.GaussianFromMeanAndPrecision(skill[year][b], performancePrecision).Named("black_performance");
                    Variable<double> white_drawMargin = Variable.Copy(drawMargin[year][w]).Named("white_drawMargin");
                    Variable<double> black_drawMargin = Variable.Copy(drawMargin[year][b]).Named("black_drawMargin");
                    Variable<double> white_delta = (white_performance - black_performance + whiteAdvantage).Named("white_delta");
                    using (Variable.Case(outcome[year][game], 0))
                    { // black wins
                        Variable.ConstrainTrue(white_delta + white_drawMargin < 0);
                    }

                    using (Variable.Case(outcome[year][game], 1))
                    { // draw
                        Variable.ConstrainBetween(white_delta, -white_drawMargin, black_drawMargin);
                    }

                    using (Variable.Case(outcome[year][game], 2))
                    { // white wins
                        Variable.ConstrainTrue(white_delta - black_drawMargin > 0);
                    }
                }
            }

            year.AddAttribute(new Models.Attributes.Sequential());   // helps inference converge faster

            engine.NumberOfIterations = 10;
            var skillPost = engine.Infer<Gaussian[][]>(skill);
            var drawMarginPost = engine.Infer<Gaussian[][]>(drawMargin);

            // compare estimates to the true values
            if (inferParameters)
            {
                Console.WriteLine("drawMargin mean = {0} (truth = {1})", engine.Infer<Gaussian>(drawMarginMean), parameters.drawMarginMean);
                Console.WriteLine("drawMargin precision = {0} (truth = {1})", engine.Infer<Gamma>(drawMarginPrecision).GetMean(), parameters.drawMarginPrecision);
                Console.WriteLine("performancePrecision = {0} (truth = {1})", engine.Infer<Gamma>(performancePrecision).GetMean(), parameters.performancePrecision);
                Console.WriteLine("skillChangePrecision = {0} (truth = {1})", engine.Infer<Gamma>(skillChangePrecision).GetMean(), parameters.skillChangePrecision);
                Console.WriteLine("drawMarginChangePrecision = {0} (truth = {1})", engine.Infer<Gamma>(drawMarginChangePrecision).GetMean(), parameters.drawMarginChangePrecision);
            }

            Console.WriteLine("white advantage = {0} (truth = {1})", engine.Infer<Gaussian>(whiteAdvantage), parameters.whiteAdvantage);
            int countPrinted = 0;
            for (int y = 0; y < nYears; y++)
            {
                for (int p = 0; p < nPlayers; p++)
                {
                    if (y >= firstYear.ObservedValue[p])
                    {
                        if (++countPrinted > 3)
                        {
                            break;
                        }

                        Console.WriteLine("skill[{0}][{1}] = {2} (truth = {3:g4})", y, p, skillPost[y][p], parameters.skill[y][p]);
                        Console.WriteLine("drawMargin[{0}][{1}] = {2} (truth = {3:g4})", y, p, drawMarginPost[y][p], parameters.drawMargin[y][p]);
                    }
                }
            }
        }

        public class Parameters
        {
            public double drawMarginMean, drawMarginPrecision, performancePrecision, skillChangePrecision, drawMarginChangePrecision, whiteAdvantage;
            public double[][] skill, drawMargin;
        }

        public void GenerateData(Parameters parameters, int[] firstYear, out int[][] whiteData, out int[][] blackData, out int[][] outcomeData)
        {
            int nYears = parameters.skill.Length;
            int nPlayers = parameters.skill[0].Length;
            int nGames = 1000;
            var whitePlayer = Util.ArrayInit(nYears, year => new List<int>());
            var blackPlayer = Util.ArrayInit(nYears, year => new List<int>());
            var outcomes = Util.ArrayInit(nYears, year => new List<int>());
            for (int game = 0; game < nGames; game++)
            {
                while (true)
                {
                    int w = Rand.Int(nPlayers);
                    int b = Rand.Int(nPlayers);
                    if (w == b)
                    {
                        continue;
                    }

                    int minYear = System.Math.Max(firstYear[w], firstYear[b]);
                    int year = Rand.Int(minYear, nYears);
                    double white_delta = parameters.whiteAdvantage + Gaussian.Sample(parameters.skill[year][w], parameters.performancePrecision)
                        - Gaussian.Sample(parameters.skill[year][b], parameters.performancePrecision);
                    double white_drawMargin = parameters.drawMargin[year][w];
                    double black_drawMargin = parameters.drawMargin[year][b];
                    int outcome;
                    if (white_delta > black_drawMargin)
                    {
                        outcome = 2;  // white wins
                    }
                    else if (white_delta < -white_drawMargin)
                    {
                        outcome = 0;  // black wins
                    }
                    else
                    {
                        outcome = 1;  // draw
                    }
                    whitePlayer[year].Add(w);
                    blackPlayer[year].Add(b);
                    outcomes[year].Add(outcome);
                    break;
                }
            }

            whiteData = Util.ArrayInit(nYears, year => whitePlayer[year].ToArray());
            blackData = Util.ArrayInit(nYears, year => blackPlayer[year].ToArray());
            outcomeData = Util.ArrayInit(nYears, year => outcomes[year].ToArray());
        }
    }
}
