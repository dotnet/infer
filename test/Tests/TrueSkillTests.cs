// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Serialization;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{
    using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
    using GaussianArray = DistributionStructArray<Gaussian, double>;

    public class TrueSkillTests
    {
        [Fact]
        [Trait("Category", "OpenBug")]
        public void RaterDrawMarginPrecisionAndThresholdsModel()
        {
            const double SkillMean = 25.0;
            //const double SkillPrecision = 1;

            var engine = new InferenceEngine();
            engine.ShowProgress = false;
            engine.Compiler.ReturnCopies = true;
            engine.Compiler.FreeMemory = false;
            //engine.Compiler.UseExistingSourceFiles = true;
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.ModelName = "RaterDrawMarginPrecisionAndThresholdsModel";

            var ModelSelector = Variable.Bernoulli(0.5);
            ModelSelector.Name = nameof(ModelSelector);
            //var modelSelectorBlock = Variable.If(ModelSelector);

            // Define the number of games
            var GameCount = Variable.Observed(default(int));
            GameCount.Name = nameof(GameCount);
            var game = new Range(GameCount).Named("game");
            game.AddAttribute(new Sequential());

            // Number of players
            var PlayerCount = Variable.Observed(default(int));
            PlayerCount.Name = nameof(PlayerCount);
            var player = new Range(PlayerCount).Named("player");
            var GamePlayerCount = Variable.Observed(default(int[]), game);
            GamePlayerCount.Name = nameof(GamePlayerCount);
            var gamePlayer = new Range(GamePlayerCount[game]).Named("gamePlayer");

            // Prior skill
            var PlayerSkills = Variable.Array<double>(player);
            PlayerSkills.Name = nameof(PlayerSkills);
            var playerSkillsPriorPrecision = Variable.Array<double>(player).Named("PlayerSkillsPriorPrecision");
            var PlayerSkillsPriorPrecisionPrior = Variable.Observed(default(Gamma[]), player);
            PlayerSkillsPriorPrecisionPrior.Name = nameof(PlayerSkillsPriorPrecisionPrior);
            playerSkillsPriorPrecision[player] = Variable<double>.Random(PlayerSkillsPriorPrecisionPrior[player]);
            PlayerSkills[player] = Variable.GaussianFromMeanAndPrecision(SkillMean, playerSkillsPriorPrecision[player]);
            //PlayerSkills[player] = Variable.GaussianFromMeanAndPrecision(SkillMean, SkillPrecision).ForEach(player);
            bool workaround = false;
            VariableArray<double> PlayerSkills2;
            if (workaround)
            {
                PlayerSkills = Variable.SequentialCopy(PlayerSkills, out PlayerSkills2);
                PlayerSkills.Name = nameof(PlayerSkills) + "Primary";
                PlayerSkills2.Name = nameof(PlayerSkills2);
            }
            else
            {
                PlayerSkills2 = PlayerSkills;
            }

            // Game outcomes
            var PlayerIndices = Variable.Observed(default(int[][]), game, gamePlayer);
            PlayerIndices.Name = nameof(PlayerIndices);
            var PlayerRanks = Variable.Observed(default(int[][]), game, gamePlayer);
            PlayerRanks.Name = nameof(PlayerRanks);

            // Rater count
            var RaterCount = Variable.Observed(default(int));
            RaterCount.Name = nameof(RaterCount);
            var rater = new Range(RaterCount).Named("rater");

            // Rater precision
            var RaterPrecisions = Variable.Array<double>(rater);
            RaterPrecisions.Name = nameof(RaterPrecisions);
            RaterPrecisions[rater] = Variable.Random(Gamma.FromShapeAndRate(10, 1)).ForEach(rater);

            // Raters of ranking
            var RaterIndices = Variable.Observed(default(int[]), game);
            RaterIndices.Name = nameof(RaterIndices);

            // Rater draw margin
            var RaterDrawMargins = Variable.Array<double>(rater);
            RaterDrawMargins.Name = nameof(RaterDrawMargins);
            var RaterDrawMarginsPrior = Variable.Observed(default(Gaussian[]), rater);
            RaterDrawMarginsPrior.Name = nameof(RaterDrawMarginsPrior);
            RaterDrawMargins[rater].SetTo(Variable<double>.Random(RaterDrawMarginsPrior[rater]));
            bool useDrawMargin = false;
            if (useDrawMargin)
            {
                Variable.ConstrainPositive(RaterDrawMargins[rater]);
            }

            // Decks
            using (Variable.ForEach(game))
            {
                var gamePlayerSkills = Variable.Subarray(PlayerSkills, PlayerIndices[game]).Named("GamePlayerSkills");
                var gamePlayerPerformances = Variable.Array<double>(gamePlayer).Named("GamePlayerPerformances");
                gamePlayerPerformances.AddAttribute(new DivideMessages(false));
                gamePlayerPerformances[gamePlayer] = Variable.GaussianFromMeanAndPrecision(
                    gamePlayerSkills[gamePlayer], RaterPrecisions[RaterIndices[game]]);

                var drawMargin = RaterDrawMargins[RaterIndices[game]];
                using (var playerInGame = Variable.ForEach(gamePlayer))
                {
                    using (Variable.If(playerInGame.Index > 0))
                    {
                        var performanceDifference = (gamePlayerPerformances[playerInGame.Index - 1] - gamePlayerPerformances[playerInGame.Index]).Named("PerformanceDifference");

                        if (useDrawMargin)
                        {
                            var isDraw = (PlayerRanks[game][playerInGame.Index] == PlayerRanks[game][playerInGame.Index - 1]).Named("IsDraw");

                            using (Variable.If(isDraw))
                            {
                                Variable.ConstrainBetween(performanceDifference, -drawMargin, drawMargin);
                            }

                            using (Variable.IfNot(isDraw))
                            {
                                Variable.ConstrainTrue(performanceDifference > drawMargin);
                            }
                        }
                        else
                        {
                            Variable.ConstrainTrue(performanceDifference > 0);
                        }
                    }
                }
            }

            // Rater thresholds
            var RaterThresholdCount = Variable.Observed(default(int));
            RaterThresholdCount.Name = nameof(RaterThresholdCount);
            var raterThreshold = new Range(RaterThresholdCount);
            raterThreshold.Name = nameof(raterThreshold);
            var RaterThresholds = Variable.Array(Variable.Array<double>(raterThreshold), rater);
            RaterThresholds.Name = nameof(RaterThresholds);
            var RaterThresholdsPrior = Variable.Observed(default(Gaussian[][]), rater, raterThreshold);
            RaterThresholdsPrior.Name = nameof(RaterThresholdsPrior);
            RaterThresholds[rater][raterThreshold] = Variable<double>.Random(RaterThresholdsPrior[rater][raterThreshold]);

            // Review count
            var ReviewCount = Variable.Observed(default(int));
            ReviewCount.Name = nameof(ReviewCount);
            var review = new Range(ReviewCount).Named("review");
            review.AddAttribute(new Sequential());

            // Review data
            var ReviewRepetitionIndices = Variable.Observed(default(int[]), review);
            ReviewRepetitionIndices.Name = nameof(ReviewRepetitionIndices);
            var ReviewRaterIndices = Variable.Observed(default(int[]), review);
            ReviewRaterIndices.Name = nameof(ReviewRaterIndices);
            var repetitionRatingValue = new Range(RaterThresholdCount - 1);
            repetitionRatingValue.Name = nameof(repetitionRatingValue);
            var ReviewRepetitionRatings = Variable.Observed(default(int[]), review);
            ReviewRepetitionRatings.Name = nameof(ReviewRepetitionRatings);
            ReviewRepetitionRatings.SetValueRange(repetitionRatingValue);

            bool useThresholds = false;
            if (useThresholds)
            {
                // Ordered rater thresholds
                using (Variable.ForEach(rater))
                using (var thresholdForRater = Variable.ForEach(raterThreshold))
                using (Variable.If(thresholdForRater.Index > 0))
                {
                    var thresholdDifference = RaterThresholds[rater][thresholdForRater.Index] - RaterThresholds[rater][thresholdForRater.Index - 1];
                    Variable.ConstrainPositive(thresholdDifference);
                }
            }

            // Reviews
            using (Variable.ForEach(review))
            {
                var repetitionIndex = ReviewRepetitionIndices[review];
                var raterIndex = ReviewRaterIndices[review];
                var repetitionRating = ReviewRepetitionRatings[review];

                var continuousRating = Variable.GaussianFromMeanAndPrecision(
                    PlayerSkills2[repetitionIndex],
                    RaterPrecisions[raterIndex]);

                if (useThresholds)
                {
                    var raterThresholds = Variable.Copy(RaterThresholds[raterIndex]).Named("raterThresholds");
                    using (Variable.Switch(repetitionRating))
                    {
                        // This hack allows indexing the thresholds with the repetitionRatingValue range instead of the raterThreshold range
                        var currentRating = (repetitionRating + 0).Named("CurrentRating");
                        var nextRating = (repetitionRating + 1).Named("NextRating");

                        Variable.ConstrainBetween(continuousRating, raterThresholds[currentRating], raterThresholds[nextRating]);
                    }
                }
                else
                {
                    Variable.ConstrainPositive(continuousRating);
                }
            }
            //modelSelectorBlock.CloseBlock();

            var trainingVariablesToInfer = new IVariable[] { PlayerSkills };
            var TrainingInferenceAlgorithm = engine.GetCompiledInferenceAlgorithm(trainingVariablesToInfer);

            // Data
            TrainingInferenceAlgorithm.SetObservedValue(GameCount.Name, 1);
            TrainingInferenceAlgorithm.SetObservedValue(PlayerCount.Name, 2);
            TrainingInferenceAlgorithm.SetObservedValue(RaterCount.Name, 1);
            TrainingInferenceAlgorithm.SetObservedValue(PlayerSkillsPriorPrecisionPrior.Name, Util.ArrayInit(2, r => Gamma.PointMass(1)));

            TrainingInferenceAlgorithm.SetObservedValue(PlayerIndices.Name, new int[][] { new int[] { 0, 1 } });
            TrainingInferenceAlgorithm.SetObservedValue(GamePlayerCount.Name, new int[] { 2 });
            TrainingInferenceAlgorithm.SetObservedValue(RaterIndices.Name, new int[] { 0 });
            if (useDrawMargin)
            {
                TrainingInferenceAlgorithm.SetObservedValue(RaterDrawMarginsPrior.Name, Util.ArrayInit(1, r => Gaussian.FromMeanAndPrecision(1, 10)));
                TrainingInferenceAlgorithm.SetObservedValue(PlayerRanks.Name, new int[][] { new int[] { 0, 1 } });
            }

            TrainingInferenceAlgorithm.SetObservedValue(ReviewCount.Name, 0);
            TrainingInferenceAlgorithm.SetObservedValue(ReviewRaterIndices.Name, new int[0]);
            TrainingInferenceAlgorithm.SetObservedValue(ReviewRepetitionIndices.Name, new int[0]);
            if (useThresholds)
            {
                TrainingInferenceAlgorithm.SetObservedValue(ReviewRepetitionRatings.Name, new int[0]);
            }

            // Inference
            for (int i = 1; i <= 1; ++i)
            {
                TrainingInferenceAlgorithm.Execute(i);
            }

            // Posteriors
            var repetitionScoresPosterior = TrainingInferenceAlgorithm.Marginal<Gaussian[]>(PlayerSkills.Name);
            foreach (var score in repetitionScoresPosterior)
            {
                Assert.True(score.GetMean() > SkillMean - 5, $"score = {score}");
            }
        }

        [Fact]
        public void DuplicateUsesArrayTest()
        {
            var nPlayers = new Range(2).Named("nPlayers");
            var matchCounts = new Range(1).Named("matchCounts");

            var skills = Variable.Array(Variable.Array<double>(matchCounts), nPlayers).Named("skills");
            using (Variable.ForEach(nPlayers))
            {
                using (var matchBlock = Variable.ForEach(matchCounts))
                {
                    using (Variable.If(matchBlock.Index == 1))
                    {
                        skills[nPlayers][matchBlock.Index] = Variable.GaussianFromMeanAndVariance(0, 1);
                    }

                    using (Variable.If(matchBlock.Index < 1))
                    {
                        skills[nPlayers][matchBlock.Index] = Variable.GaussianFromMeanAndPrecision(skills[nPlayers][matchBlock.Index + 1], 1.0);
                    }

                    using (Variable.If(matchBlock.Index > 1))
                    {
                        skills[nPlayers][matchBlock.Index] = Variable.GaussianFromMeanAndPrecision(skills[nPlayers][matchBlock.Index - 1], 1.0);
                    }
                }
            }

            var playerPerformance = Variable.Array<double>(nPlayers).Named("playerPerformance");
            using (Variable.ForEach(nPlayers))
            {
                playerPerformance[nPlayers] = Variable.GaussianFromMeanAndPrecision(skills[nPlayers][0], 1.0);
            }
            Variable.ConstrainTrue(playerPerformance[0] > playerPerformance[1]);

            var inferenceEngine = new InferenceEngine
            {
                Compiler =
                {
                    UseSerialSchedules = false
                },
            };

            var inferredSkills = inferenceEngine.Infer<Gaussian[][]>(skills);
        }

        /// <summary>
        /// Test a model where inference can fail due to incorrect initial messages.
        /// Failure only happens when ProductOp throws on uniform inputs.
        /// Fails with "The distribution is improper" because Team1Perf_uses_F is uniform.
        /// This happens because skill_uses_F[player][gameOfPlayer][1] is uniform for gameOfPlayer>0,
        /// due to never being initialized in the initialization schedule.
        /// </summary>
        [Fact]
        public void TrueSkill2Test()
        {
            var performancePrecision = Variable.GammaFromMeanAndVariance(1, 1e-3);
            performancePrecision.Name = nameof(performancePrecision);
            //performancePrecision.ObservedValue = 1;
            var skillChangeWithMatchPrecision = Variable.GammaFromMeanAndVariance(1, 1e-3);
            skillChangeWithMatchPrecision.Name = nameof(skillChangeWithMatchPrecision);
            //skillChangeWithMatchPrecision.ObservedValue = 1;

            var pKillW = Variable.GaussianFromMeanAndPrecision(0, 1);
            pKillW.Name = nameof(pKillW);
            pKillW.ObservedValue = 0;
            var oKillW = Variable.GaussianFromMeanAndPrecision(0, 1);
            oKillW.Name = nameof(oKillW);
            //oKillW.ObservedValue = 0;
            var killV = Variable.GammaFromMeanAndVariance(1, 1e-1);
            killV.Name = nameof(killV);
            killV.ObservedValue = 1;
            var pDeathW = Variable.GaussianFromMeanAndPrecision(0, 1);
            pDeathW.Name = nameof(pDeathW);
            //pDeathW.ObservedValue = 0;
            var oDeathW = Variable.GaussianFromMeanAndPrecision(0, 1);
            oDeathW.Name = nameof(oDeathW);
            //oDeathW.ObservedValue = 0;
            var deathV = Variable.GammaFromMeanAndVariance(1, 1e-1);
            deathV.Name = nameof(deathV);
            //deathV.ObservedValue = 1;

            Range player = new Range(4);
            player.Name = nameof(player);

            var gameCountOfPlayer = Variable.Observed(new int[] { 3, 1, 1, 1 }, player);
            gameCountOfPlayer.Name = nameof(gameCountOfPlayer);
            Range gameOfPlayer = new Range(gameCountOfPlayer[player]);
            gameOfPlayer.Name = nameof(gameOfPlayer);

            var priorSkill = Variable.Observed(new Gaussian[] { new Gaussian(0, 1), new Gaussian(0, 1), new Gaussian(0, 1), new Gaussian(0, 1) }, player);
            var skill = Variable.Array(Variable.Array<double>(gameOfPlayer), player);
            skill.Name = nameof(skill);
            using (var playerBlock = Variable.ForEach(player))
            {
                var p = playerBlock.Index;
                using (var gameBlock = Variable.ForEach(gameOfPlayer))
                {
                    var g = gameBlock.Index;
                    using (Variable.If(g == 0))
                    {
                        skill[player][gameOfPlayer] = Variable<double>.Random(priorSkill[player]);
                    }
                    using (Variable.If(g > 0))
                    {
                        skill[player][gameOfPlayer] = Variable.GaussianFromMeanAndPrecision(skill[player][g - 1], skillChangeWithMatchPrecision);
                    }
                }
            }

            Range game = new Range(3);
            game.Name = nameof(game);
            Range member = new Range(1);
            member.Name = nameof(member);

            var Team1Player = Variable.Observed(new int[][] { new int[] { 0 }, new int[] { 0 }, new int[] { 0 } }, game, member);
            Team1Player.Name = nameof(Team1Player);
            var Team1Kill = Variable.Observed(new double[][] { new double[] { 1 }, new double[] { 1 }, new double[] { 1 } }, game, member);
            var Team1Death = Variable.Observed(new double[][] { new double[] { 1 }, new double[] { 1 }, new double[] { 1 } }, game, member);
            var Team1GameIndex = Variable.Observed(new int[][] { new int[] { 0 }, new int[] { 1 }, new int[] { 2 } }, game, member);
            Team1GameIndex.Name = nameof(Team1GameIndex);

            var Team2Player = Variable.Observed(new int[][] { new int[] { 1 }, new int[] { 2 }, new int[] { 3 } }, game, member);
            Team2Player.Name = nameof(Team2Player);
            var Team2GameIndex = Variable.Observed(new int[][] { new int[] { 0 }, new int[] { 0 }, new int[] { 0 } }, game, member);
            Team2GameIndex.Name = nameof(Team2GameIndex);

            var Team1Perf = Variable.Array(Variable.Array<double>(member), game);
            Team1Perf.Name = nameof(Team1Perf);
            var Team2Perf = Variable.Array(Variable.Array<double>(member), game);
            Team2Perf.Name = nameof(Team2Perf);

            using (Variable.ForEach(game))
            {
                using (var playerBlock = Variable.ForEach(member))
                {
                    var PlayerIndex1 = Team1Player[game][member];
                    var GameIndex1 = Team1GameIndex[game][member];
                    Team1Perf[game][member] = Variable.GaussianFromMeanAndPrecision(skill[PlayerIndex1][GameIndex1], performancePrecision);

                    var PlayerIndex2 = Team2Player[game][member];
                    var GameIndex2 = Team2GameIndex[game][member];
                    Team2Perf[game][member] = Variable.GaussianFromMeanAndPrecision(skill[PlayerIndex2][GameIndex2], performancePrecision);
                }

                Variable<double> Team1PerfSum = Variable.Sum(Team1Perf[game]);
                Team1PerfSum.Name = nameof(Team1PerfSum);

                Variable<double> Team2PerfSum = Variable.Sum(Team2Perf[game]);
                Team2PerfSum.Name = nameof(Team2PerfSum);

                using (var playerBlock = Variable.ForEach(member))
                {
                    Team1Kill[game][member] = Variable.Max(0.0, Variable.GaussianFromMeanAndPrecision(pKillW * Team1Perf[game][member] + oKillW * Team2PerfSum, killV));
                    Team1Death[game][member] = Variable.Max(0.0, Variable.GaussianFromMeanAndPrecision(pDeathW * Team1Perf[game][member] + oDeathW * Team2PerfSum, deathV));
                }
            }
            performancePrecision.AddAttribute(new PointEstimate());
            performancePrecision.InitialiseTo(Gamma.PointMass(1));
            skillChangeWithMatchPrecision.AddAttribute(new PointEstimate());
            skillChangeWithMatchPrecision.InitialiseTo(Gamma.PointMass(1));
            pKillW.AddAttribute(new PointEstimate());
            pKillW.InitialiseTo(Gaussian.PointMass(0));
            oKillW.AddAttribute(new PointEstimate());
            oKillW.InitialiseTo(Gaussian.PointMass(0));
            killV.AddAttribute(new PointEstimate());
            killV.InitialiseTo(Gamma.PointMass(1));
            pDeathW.AddAttribute(new PointEstimate());
            pDeathW.InitialiseTo(Gaussian.PointMass(0));
            oDeathW.AddAttribute(new PointEstimate());
            oDeathW.InitialiseTo(Gaussian.PointMass(0));
            deathV.AddAttribute(new PointEstimate());
            deathV.InitialiseTo(Gamma.PointMass(1));
            game.AddAttribute(new Sequential());   // helps inference converge faster

            InferenceEngine engine = new InferenceEngine();
            engine.Compiler.InitialisationAffectsSchedule = true;
            engine.Infer(performancePrecision);
        }

        /// <summary>
        /// Gives "not sequential" error if the two sequential loops are not ordered correctly
        /// </summary>
        [Fact]
        public void TrueSkillChainTest()
        {
            TrueSkillChain(false);
        }

        /// <summary>
        /// Test that the translation symmetry in skills is resolved by the prior.
        /// At the moment, it is because PlayerSkills_B is not being updated.
        /// MarginalIncrementArray has a Cancels dependency that Iteration ignores.
        /// this doesn't seem possible to fix under the way Increments are handled.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void TrueSkillChainWithInitialisationTest()
        {
            TrueSkillChain(true);
        }

        private void TrueSkillChain(bool initialise)
        {
            var playerCount = 3;
            var playerIndices = new[]
                       {
                           new[] { 0, 1, 2 },
                           new[] { 0, 1 },
                           new[] { 1, 2 },
                           new[] { 2, 0 }
                       };
            var playerRanks = new[]
                       {
                           new[] { 1, 2, 2 },
                           new[] { 1, 2 },
                           new[] { 1, 2 },
                           new[] { 1, 2 }
                       };

            /// <summary>
            /// The prior mean of the player skill.
            /// </summary>
            /// <remarks>This is the value used in Xbox Live.</remarks>
            const double SkillMean = 25.0;

            /// <summary>
            /// The prior standard deviation of the player skill.
            /// </summary>
            /// <remarks>This is the value used in Xbox Live.</remarks>
            const double SkillStandardDeviation = SkillMean / 3.0;

            const double performanceNoiseVariance = 1.0;
            const double drawMargin = 0.7;

            // Define the number of games
            var GameCount = Variable.Observed(default(int)).Named("GameCount");
            var game = new Range(GameCount).Named("game");
            game.AddAttribute(new Sequential());

            // Define the number of players
            var PlayerCount = Variable.Observed(playerCount).Named("PlayerCount");
            var player = new Range(PlayerCount).Named("player");
            var GamePlayerCount = Variable.Observed(default(int[]), game).Named("GamePlayerCount");
            var gamePlayer = new Range(GamePlayerCount[game]).Named("gamePlayer");

            // Define the prior skill of each player
            var PlayerSkills = Variable.Array<double>(player).Named("PlayerSkills");
            var PlayerSkillsPrior = Variable.Observed(default(Gaussian[]), player).Named("PlayerSkillsPrior");
            PlayerSkills[player] = Variable<double>.Random(PlayerSkillsPrior[player]);
            if (initialise)
                PlayerSkills[player].InitialiseTo(Gaussian.FromMeanAndVariance(0, 1));

            // Define the draw margin
            var DrawMargin = Variable.Observed(default(double)).Named("DrawMargin");

            // Define the observed game outcomes
            var PlayerIndices = Variable.Observed(default(int[][]), game, gamePlayer).Named("PlayerIndices");
            var PlayerRanks = Variable.Observed(default(int[][]), game, gamePlayer).Named("PlayerRanks");

            // Define the performances
            var PerformanceNoiseVariance = Variable.Observed(default(double)).Named("PerformanceNoiseVariance");

            using (Variable.ForEach(game))
            {
                var gamePlayerSkills = Variable.Subarray(PlayerSkills, PlayerIndices[game]).Named("GamePlayerSkills");
                var gamePlayerPerformances = Variable.Array<double>(gamePlayer).Named("GamePlayerPerformances");
                gamePlayerPerformances[gamePlayer] = Variable.GaussianFromMeanAndVariance(gamePlayerSkills[gamePlayer], PerformanceNoiseVariance);

                using (var gp = Variable.ForEach(gamePlayer))
                {
                    using (Variable.If(gp.Index > 0))
                    {
                        var performanceDifference = (gamePlayerPerformances[gp.Index - 1] - gamePlayerPerformances[gp.Index]).Named("PerformanceDifference");

                        var isDraw = (PlayerRanks[game][gp.Index] == PlayerRanks[game][gp.Index - 1]).Named("IsDraw");

                        using (Variable.If(isDraw))
                        {
                            Variable.ConstrainBetween(performanceDifference, -DrawMargin, DrawMargin);
                        }

                        using (Variable.IfNot(isDraw))
                        {
                            Variable.ConstrainTrue(performanceDifference > DrawMargin);
                        }
                    }
                }
            }

            var engine = new InferenceEngine();

            engine.ShowProgress = false;
            engine.Compiler.ReturnCopies = false;
            engine.Compiler.FreeMemory = false;
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.Compiler.TreatWarningsAsErrors = true;
            engine.Compiler.AllowSerialInitialisers = false; // not needed, but a useful test
            engine.ModelName = "TrueSkill";

            var InferenceAlgorithm = engine.GetCompiledInferenceAlgorithm(PlayerSkills);
            // Set the training data
            InferenceAlgorithm.SetObservedValue(GameCount.Name, playerIndices.Length);
            InferenceAlgorithm.SetObservedValue(PlayerIndices.Name, playerIndices);
            InferenceAlgorithm.SetObservedValue(PlayerRanks.Name, playerRanks);
            InferenceAlgorithm.SetObservedValue(GamePlayerCount.Name, Util.ArrayInit(playerIndices.Length, i => playerIndices[i].Length));

            // Set the priors
            var playerSkillPrior = Util.ArrayInit(playerCount, p => Gaussian.FromMeanAndVariance(SkillMean, SkillStandardDeviation * SkillStandardDeviation));
            InferenceAlgorithm.SetObservedValue(PlayerSkillsPrior.Name, playerSkillPrior);

            // Set the hyper-parameters
            InferenceAlgorithm.SetObservedValue(PerformanceNoiseVariance.Name, performanceNoiseVariance);
            InferenceAlgorithm.SetObservedValue(DrawMargin.Name, drawMargin);

            GaussianArray skillsExpected = new GaussianArray(new Gaussian[] {
                new Gaussian(25.63, 1.505),
                new Gaussian(24.66, 1.372),
                new Gaussian(24.71, 1.347)
            });

            bool verbose = false;
            if (verbose)
            {
                // Run inference
                for (int iter = 1; iter < 200; iter++)
                {
                    InferenceAlgorithm.Execute(iter);
                    var learnedSkills = InferenceAlgorithm.Marginal<IList<Gaussian>>(PlayerSkills.Name);
                    Console.Write(iter);
                    for (int i = 0; i < playerCount; i++)
                    {
                        Console.Write(" ");
                        Console.Write(learnedSkills[i]);
                    }
                    double maxdiff = skillsExpected.MaxDiff(learnedSkills);
                    Console.Write(" ");
                    Console.WriteLine(maxdiff);
                    Console.WriteLine();
                }
            }

            InferenceAlgorithm.Execute(120);
            var skillsActual = InferenceAlgorithm.Marginal<IList<Gaussian>>(PlayerSkills.Name);
            Assert.True(skillsExpected.MaxDiff(skillsActual) < 1e-2);
        }

        public static int[] GetRanks(IReadOnlyList<double> scores, double drawMargin)
        {
            int[] indices = Util.ArrayInit(scores.Count, i => i);
            double[] scoresSorted = new double[scores.Count];
            scores.CopyTo(scoresSorted, 0);
            Array.Sort(scoresSorted, indices);
            int[] ranksSorted = GetRanksSorted(scoresSorted, drawMargin);
            int[] ranks = new int[scores.Count];
            for (int i = 0; i < indices.Length; i++)
            {
                ranks[indices[i]] = ranksSorted[i];
            }
            return ranks;
        }

        public static int[] GetRanksSorted(IReadOnlyList<double> scores, double drawMargin)
        {
            int[] ranks = new int[scores.Count];
            ranks[0] = 0;
            for (int i = 1; i < scores.Count; i++)
            {
                if (scores[i] - scores[i - 1] < drawMargin)
                    ranks[i] = ranks[i - 1];
                else
                    ranks[i] = ranks[i - 1] + 1;
            }
            return ranks;
        }

        public static double Spearman(int[] ranks1, int[] ranks2)
        {
            int n = ranks1.Length;
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = ranks1[i] - ranks2[i];
                sum += diff * diff;
            }
            return 1 - 6 * sum / (n * (n * n - 1));
        }

        public static double RankRmse(int[] ranks1, int[] ranks2)
        {
            int n = ranks1.Length;
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = ranks1[i] - ranks2[i];
                sum += diff * diff;
            }
            return System.Math.Sqrt(sum / n);
        }

        /// <summary>
        /// Test convergence of TrueSkill for Cecily Morrison
        /// </summary>
        internal void TrueSkillChainTest2()
        {
            /// <summary>
            /// The prior mean of the player skill.
            /// </summary>
            /// <remarks>This is the value used in Xbox Live.</remarks>
            const double SkillMean = 25.0;

            /// <summary>
            /// The prior standard deviation of the player skill.
            /// </summary>
            /// <remarks>This is the value used in Xbox Live.</remarks>
            const double SkillStandardDeviation = SkillMean / 3.0;

            const double beta = 25.0 / 6;
            const double performanceNoiseVariance = beta * beta;
            const double drawMargin = 0.74;

            // Define the number of games
            var GameCount = Variable.Observed(default(int)).Named("GameCount");
            var game = new Range(GameCount).Named("game");
            game.AddAttribute(new Sequential());

            // Define the number of players
            var PlayerCount = Variable.Observed(default(int)).Named("PlayerCount");
            var player = new Range(PlayerCount).Named("player");
            var GamePlayerCount = Variable.Observed(default(int[]), game).Named("GamePlayerCount");
            var gamePlayer = new Range(GamePlayerCount[game]).Named("gamePlayer");

            // Define the prior skill of each player
            var PlayerSkills = Variable.Array<double>(player).Named("PlayerSkills");
            var PlayerSkillsPrior = Variable.Observed(default(Gaussian[]), player).Named("PlayerSkillsPrior");
            PlayerSkills[player] = Variable<double>.Random(PlayerSkillsPrior[player]);

            // Define the draw margin
            var DrawMargin = Variable.Observed(default(double)).Named("DrawMargin");

            // Define the observed game outcomes
            var PlayerIndices = Variable.Observed(default(int[][]), game, gamePlayer).Named("PlayerIndices");
            var PlayerRanks = Variable.Observed(default(int[][]), game, gamePlayer).Named("PlayerRanks");

            // Define the performances
            var PerformanceNoiseVariance = Variable.Observed(default(double)).Named("PerformanceNoiseVariance");

            using (Variable.ForEach(game))
            {
                var gamePlayerSkills = Variable.Subarray(PlayerSkills, PlayerIndices[game]).Named("GamePlayerSkills");
                var gamePlayerPerformances = Variable.Array<double>(gamePlayer).Named("GamePlayerPerformances");
                gamePlayerPerformances[gamePlayer] = Variable.GaussianFromMeanAndVariance(gamePlayerSkills[gamePlayer], PerformanceNoiseVariance);
                gamePlayerPerformances.AddAttribute(new DivideMessages(false));

                using (var gp = Variable.ForEach(gamePlayer))
                {
                    using (Variable.If(gp.Index > 0))
                    {
                        var performanceDifference = (gamePlayerPerformances[gp.Index - 1] - gamePlayerPerformances[gp.Index]).Named("PerformanceDifference");

                        var isDraw = (PlayerRanks[game][gp.Index] == PlayerRanks[game][gp.Index - 1]).Named("IsDraw");

                        using (Variable.If(isDraw))
                        {
                            Variable.ConstrainBetween(performanceDifference, -DrawMargin, DrawMargin);
                        }

                        using (Variable.IfNot(isDraw))
                        {
                            Variable.ConstrainTrue(performanceDifference > DrawMargin);
                        }
                    }
                }
            }

            var engine = new InferenceEngine();

            engine.ShowProgress = false;
            //engine.Compiler.ReturnCopies = false;
            engine.Compiler.FreeMemory = false;
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.Compiler.TreatWarningsAsErrors = true;
            engine.ModelName = "TrueSkill";

            var InferenceAlgorithm = engine.GetCompiledInferenceAlgorithm(PlayerSkills);

            // Load the training data
            //TODO: change path
            var dict = MatlabReader.Read(@"c:\Users\minka\Downloads\trueSkillData.mat");
            Matrix results = (Matrix)dict["trueSkillResults"];
            int rater = 2;
            int playerCount = results.Rows;
            double[] trueSkills = Util.ArrayInit(playerCount, i => results[i, rater]);
            int[] trueRanks = GetRanks(trueSkills, drawMargin);
            int[] trueRanks0 = GetRanks(trueSkills, 0);

            // Generate outcomes
            Rand.Restart(0);
            int playersPerGame = 8;
            int maxGamesPerPlayer = 100;
            int[] maxGamesChoices = { 5, 10, 15, 20, 25, 50, 75, 100, 150, 200 };
            foreach (var maxGames in maxGamesChoices)
            {
                MeanVarianceAccumulator maxGamesMva = new MeanVarianceAccumulator();
                for (int maxGamesTrial = 0; maxGamesTrial < 100; maxGamesTrial++)
                {
                    List<int> players = new List<int>();
                    for (int i = 0; i < playerCount; i++)
                    {
                        players.Add(i);
                    }
                    List<int[]> indices = new List<int[]>();
                    List<int[]> ranks = new List<int[]>();
                    Dictionary<int, int> counts = new Dictionary<int, int>();
                    MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
                    bool estimateEntropy = false;
                    bool useRandom = true;
                    for (int gameIndex = 0; gameIndex < maxGames; gameIndex++)
                    {
                        if (players.Count < playersPerGame)
                            break;
                        HashSet<int> set;
                        if (useRandom)
                        {
                            set = Rand.SampleWithoutReplacement(players, playersPerGame);
                        }
                        else
                        {
                            // pick a rank at random
                            int j = Rand.Int(playerCount);
                            int r = trueRanks0[j];
                            // take all players near that rank
                            int lowerRank = r - playersPerGame / 2;
                            int upperRank = r + playersPerGame / 2 - 1;
                            if (lowerRank < 0)
                            {
                                upperRank -= lowerRank;
                                lowerRank = 0;
                            }
                            if (upperRank >= playerCount)
                            {
                                lowerRank -= (upperRank - playerCount + 1);
                                upperRank = playerCount - 1;
                            }
                            set = new HashSet<int>();
                            for (int i = 0; i < trueRanks0.Length; i++)
                            {
                                r = trueRanks0[i];
                                if (r >= lowerRank && r <= upperRank)
                                    set.Add(i);
                            }
                            if (set.Count != playersPerGame)
                                throw new Exception();
                        }
                        int[] playersInGame = new int[playersPerGame];
                        set.CopyTo(playersInGame, 0);
                        double[] scores = Util.ArrayInit(playersPerGame, i => -(trueSkills[playersInGame[i]] + Rand.Normal() * beta));
                        Array.Sort(scores, playersInGame);
                        indices.Add(playersInGame);
                        ranks.Add(GetRanksSorted(scores, drawMargin));
                        foreach (int i in playersInGame)
                        {
                            int count;
                            counts.TryGetValue(i, out count);
                            counts[i] = count + 1;
                            if (counts[i] == maxGamesPerPlayer)
                                players.Remove(i);
                        }
                        if (estimateEntropy)
                        {
                            // estimate the entropy of the outcome distribution
                            Dictionary<int[], int> frequency = new Dictionary<int[], int>(new RankComparer());
                            int ntrials = 2000;
                            for (int trial = 0; trial < ntrials; trial++)
                            {
                                double[] scoresTemp = Util.ArrayInit(playersPerGame, i => -(trueSkills[playersInGame[i]] + Rand.Normal() * beta));
                                int[] ranksTemp = GetRanks(scoresTemp, 0);
                                int count;
                                frequency.TryGetValue(ranksTemp, out count);
                                frequency[ranksTemp] = count + 1;
                            }
                            double entropy = 0;
                            foreach (int count in frequency.Values)
                            {
                                double p = (double)count / ntrials;
                                entropy -= p * System.Math.Log(p);
                            }
                            mva.Add(entropy);
                        }
                    }
                    //Console.WriteLine("{0} games", indices.Count);
                    if (estimateEntropy)
                    {
                        double initialEntropy = MMath.GammaLn(playersPerGame + 1);
                        double avgEntropy = mva.Mean;
                        Console.WriteLine("average entropy = {0}, information gained = {1}", avgEntropy, initialEntropy - avgEntropy);
                    }
                    var playerIndices = indices.ToArray();
                    var playerRanks = ranks.ToArray();

                    // Set the training data
                    InferenceAlgorithm.SetObservedValue(GameCount.Name, playerIndices.Length);
                    InferenceAlgorithm.SetObservedValue(PlayerCount.Name, playerCount);
                    InferenceAlgorithm.SetObservedValue(PlayerIndices.Name, playerIndices);
                    InferenceAlgorithm.SetObservedValue(PlayerRanks.Name, playerRanks);
                    InferenceAlgorithm.SetObservedValue(GamePlayerCount.Name, Util.ArrayInit(playerIndices.Length, i => playerIndices[i].Length));

                    // Set the priors
                    var playerSkillPrior = Util.ArrayInit(playerCount, p => Gaussian.FromMeanAndVariance(SkillMean, SkillStandardDeviation * SkillStandardDeviation));
                    InferenceAlgorithm.SetObservedValue(PlayerSkillsPrior.Name, playerSkillPrior);

                    // Set the hyper-parameters
                    InferenceAlgorithm.SetObservedValue(PerformanceNoiseVariance.Name, performanceNoiseVariance);
                    InferenceAlgorithm.SetObservedValue(DrawMargin.Name, drawMargin);

                    // Run inference
                    IList<Gaussian> learnedSkills = null;
                    bool verbose = false;
                    for (int iter = 1; iter < 2000; iter++)
                    {
                        InferenceAlgorithm.Execute(iter);
                        var learnedSkillsNew = InferenceAlgorithm.Marginal<IList<Gaussian>>(PlayerSkills.Name);
                        if (learnedSkills != null && ((Diffable)learnedSkills).MaxDiff(learnedSkillsNew) < 1e-3)
                            break;
                        learnedSkills = (IList<Gaussian>)((ICloneable)learnedSkillsNew).Clone();
                        if (verbose)
                        {
                            Console.Write(iter);
                            Console.Write(" ");
                            Console.Write(learnedSkills[0]);
                            Console.WriteLine();
                        }
                    }
                    double[] skillMeans = Util.ArrayInit(learnedSkills.Count, i => learnedSkills[i].GetMean());
                    int[] learnedRanks = GetRanks(skillMeans, drawMargin);
                    if (verbose)
                    {
                        Console.WriteLine("trueRank\tlearnedRank");
                        for (int i = 0; i < trueRanks.Length; i++)
                        {
                            Console.WriteLine("{0}\t{1}", trueRanks[i], learnedRanks[i]);
                        }
                    }
                    double rmse = RankRmse(trueRanks, learnedRanks);
                    //Console.WriteLine("RMSE in ranks = {0}", rmse);
                    //Console.WriteLine("Spearman correlation = {0}", Spearman(trueRanks, learnedRanks));
                    maxGamesMva.Add(rmse);
                }
                Console.WriteLine("{0} games, average RMSE = {1}", maxGames, maxGamesMva.Mean);
            }
        }

        class RankComparer : IEqualityComparer<int[]>
        {
            public bool Equals(int[] x, int[] y)
            {
                if (x.Length != y.Length)
                    return false;
                for (int i = 0; i < x.Length; i++)
                {
                    if (x[i] != y[i])
                        return false;
                }
                return true;
            }

            public int GetHashCode(int[] obj)
            {
                int hash = Hash.Start;
                for (int i = 0; i < obj.Length; i++)
                {
                    hash = Hash.Combine(hash, obj[i]);
                }
                return hash;
            }
        }

        private int[][][] GetPlayerIndicesByRank(int[][] playerRanks, int[][] playerIndices)
        {
            return Util.ArrayInit(playerRanks.Length, gameIndex =>
            {
                var playerRanksThisGame = playerRanks[gameIndex];
                var playerIndicesThisGame = playerIndices[gameIndex];
                var distinctRanks = playerRanksThisGame.Distinct().OrderByDescending(r => r).ToList();
                return Util.ArrayInit(distinctRanks.Count, rankIndex =>
                    playerRanksThisGame
                        .IndexOfAll(distinctRanks[rankIndex])
                        .Select(index => playerIndicesThisGame[index])
                        .ToArray());
            });
        }

        [Fact]
        public void TrueSkillChainTest3()
        {
            var playerCount = 3;
            var playerIndices = new[]
                       {
                           new[] { 0, 1, 2 },
                           new[] { 0, 1 },
                           new[] { 1, 2 },
                           new[] { 2, 0 }
                       };
            var playerRanks = new[]
                       {
                           new[] { 1, 2, 2 },
                           new[] { 1, 2 },
                           new[] { 1, 2 },
                           new[] { 1, 2 }
                       };
            var playerIndicesByRank = GetPlayerIndicesByRank(playerRanks, playerIndices);

            /// <summary>
            /// The prior mean of the player skill.
            /// </summary>
            /// <remarks>This is the value used in Xbox Live.</remarks>
            const double SkillMean = 25.0;

            /// <summary>
            /// The prior standard deviation of the player skill.
            /// </summary>
            /// <remarks>This is the value used in Xbox Live.</remarks>
            const double SkillStandardDeviation = SkillMean / 3.0;

            // Define the number of games
            var GameCount = Variable.Observed(default(int)).Named("GameCount");
            var game = new Range(GameCount).Named("game");
            game.AddAttribute(new Sequential());

            // Define the number of players
            var PlayerCount = Variable.Observed(playerCount).Named("PlayerCount");
            var player = new Range(PlayerCount).Named("player");
            var RankCount = Variable.Observed(default(int[]), game).Named("RankCount");
            var rank = new Range(RankCount[game]).Named("rank");
            var playerAtRankCount = Variable.Observed(default(int[][]), game, rank).Named("playerAtRankCount");
            var playerAtRank = new Range(playerAtRankCount[game][rank]).Named("playerAtRank");

            // Define the prior skill of each player
            var PlayerSkills = Variable.Array<double>(player).Named("PlayerSkills");
            var PlayerSkillsPrior = Variable.Observed(default(Gaussian[]), player).Named("PlayerSkillsPrior");
            PlayerSkills[player] = Variable<double>.Random(PlayerSkillsPrior[player]);

            // Define the draw margin
            var DrawMargin = Variable.Observed(default(double)).Named("DrawMargin");

            // Define the observed game outcomes
            var PlayerIndicesByRank = Variable.Observed(default(int[][][]), game, rank, playerAtRank).Named("PlayerIndices");

            // Define the performances
            var PerformanceNoiseVariance = Variable.Observed(default(double)).Named("PerformanceNoiseVariance");

            using (Variable.ForEach(game))
            {
                // In this model, each player is assumed to have the smallest rank consistent with the following constraint:
                // If performance[i] > performance[j] + drawMargin, then rank[i] > rank[j].  (Here rank increases with performance.)
                // This means that for a player i with rank r>0, there exists a player j at rank r-1 such that performance[i] > performance[j] + drawMargin.
                // In particular, this must be true for the j with smallest performance at rank r-1.
                // When two players have the same rank, we know that their performance gap must be less than the draw margin.
                // In particular, this must be true for the j with smallest performance at rank r.

                var minPerformanceAtRank = Variable.Array<double>(rank).Named("minPerformanceAtRank");
                using (var rankBlock = Variable.ForEach(rank))
                {
                    var skills = Variable.Subarray(PlayerSkills, PlayerIndicesByRank[game][rank]);
                    skills.Name = nameof(skills);
                    var performances = Variable.Array<double>(playerAtRank);
                    performances.Name = nameof(performances);
                    performances[playerAtRank] = Variable.GaussianFromMeanAndVariance(skills[playerAtRank], PerformanceNoiseVariance);
                    var indexOfMinimumPerformance = Variable.DiscreteUniform(playerAtRank);
                    indexOfMinimumPerformance.Name = nameof(indexOfMinimumPerformance);
                    using (Variable.Switch(indexOfMinimumPerformance))
                    {
                        minPerformanceAtRank[rank] = performances[indexOfMinimumPerformance];
                        Range playerAtRank2 = playerAtRank.Clone();
                        playerAtRank2.Name = nameof(playerAtRank2);
                        using (var playerBlock = Variable.ForEach(playerAtRank2))
                        {
                            var playerIsMinimum = (playerBlock.Index == indexOfMinimumPerformance);
                            playerIsMinimum.Name = nameof(playerIsMinimum);
                            using (Variable.IfNot(playerIsMinimum))
                            {
                                var performanceDifferenceAtRank = (performances[playerBlock.Index] - minPerformanceAtRank[rank]);
                                performanceDifferenceAtRank.Name = nameof(performanceDifferenceAtRank);
                                Variable.ConstrainBetween(performanceDifferenceAtRank, 0, DrawMargin);
                            }
                        }
                    }

                    using (Variable.If(rankBlock.Index > 0))
                    {
                        var performanceDifference = (minPerformanceAtRank[rank] - minPerformanceAtRank[rankBlock.Index - 1]);
                        performanceDifference.Name = nameof(performanceDifference);
                        Variable.ConstrainTrue(performanceDifference > DrawMargin);
                    }
                }
            }

            var engine = new InferenceEngine();

            engine.ShowProgress = false;
            engine.Compiler.ReturnCopies = false;
            engine.Compiler.FreeMemory = false;
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;
            engine.Compiler.TreatWarningsAsErrors = true;
            //engine.Compiler.AllowSerialInitialisers = false; // not needed, but a useful test
            engine.ModelName = "TrueSkill";

            var InferenceAlgorithm = engine.GetCompiledInferenceAlgorithm(PlayerSkills);
            // Set the training data
            InferenceAlgorithm.SetObservedValue(GameCount.Name, playerIndicesByRank.Length);
            InferenceAlgorithm.SetObservedValue(PlayerIndicesByRank.Name, playerIndicesByRank);
            InferenceAlgorithm.SetObservedValue(playerAtRankCount.Name, Util.ArrayInit(playerIndicesByRank.Length, i =>
                Util.ArrayInit(playerIndicesByRank[i].Length, j => playerIndicesByRank[i][j].Length)));
            InferenceAlgorithm.SetObservedValue(RankCount.Name, Util.ArrayInit(playerIndicesByRank.Length, i => playerIndicesByRank[i].Length));

            // Set the priors
            var playerSkillPrior = Util.ArrayInit(playerCount, p => Gaussian.FromMeanAndVariance(SkillMean, SkillStandardDeviation * SkillStandardDeviation));
            InferenceAlgorithm.SetObservedValue(PlayerSkillsPrior.Name, playerSkillPrior);

            const double performanceNoiseVariance = 1.0;
            const double drawMargin = 0.7;

            // Set the hyper-parameters
            InferenceAlgorithm.SetObservedValue(PerformanceNoiseVariance.Name, performanceNoiseVariance);
            InferenceAlgorithm.SetObservedValue(DrawMargin.Name, drawMargin);

            GaussianArray skillsExpected = new GaussianArray(new Gaussian[] {
                new Gaussian(25.59, 1.512),
                new Gaussian(24.69, 1.386),
                new Gaussian(24.71, 1.345)
            });

            bool verbose = false;
            if (verbose)
            {
                // Run inference
                for (int iter = 1; iter < 200; iter++)
                {
                    InferenceAlgorithm.Execute(iter);
                    var learnedSkills = InferenceAlgorithm.Marginal<IList<Gaussian>>(PlayerSkills.Name);
                    Console.Write(iter);
                    for (int i = 0; i < playerCount; i++)
                    {
                        Console.Write(" ");
                        Console.Write(learnedSkills[i]);
                    }
                    double maxdiff = skillsExpected.MaxDiff(learnedSkills);
                    Console.Write(" ");
                    Console.WriteLine(maxdiff);
                    Console.WriteLine();
                }
            }

            InferenceAlgorithm.Execute(200);
            var skillsActual = InferenceAlgorithm.Marginal<IList<Gaussian>>(PlayerSkills.Name);
            Assert.True(skillsExpected.MaxDiff(skillsActual) < 1e-2);
        }

        // Crashes due to an invalid init schedule when UseSerialSchedules = false
        // skill_uses_F[year][player][1] is not initialized
        // skill_depth1_F has a SkipIfUniform Any dependency 
        // SkipIfUniform dependency of white_delta_use_B on white_delta_uses_B is incorrect - should be All not Any since the cases do not overlap
        // - this suggests a bug in CreateDummyStmts - it shows why we need to detect control flow joins
        // during scheduling, the dependency on skill_uses_F[year-1] should not require skill_uses_F[year] if they are in the same loop
        [Fact]
        [Trait("Category", "OpenBug")]
        public void ChessTest()
        {
            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());
            engine.Compiler.UseSerialSchedules = false;

            int nPlayers = 2;
            int nYears = 2;
            Rand.Restart(1);

            var skillPrior = new Gaussian(0, 2 * 2);
            var skillChangePrecisionPrior = Gamma.FromShapeAndRate(2, 2 * 0.1 * 0.1);
            var skillChangePrecision = Variable.Random(skillChangePrecisionPrior).Named("skillChangePrecision");

            Range player = new Range(nPlayers).Named("player");
            Range year = new Range(nYears).Named("year");
            var skill = Variable.Array(Variable.Array<double>(player), year).Named("skill");

            using (var yearBlock = Variable.ForEach(year))
            {
                var y = yearBlock.Index;
                using (Variable.If(y == 0))
                {
                    skill[year][player] = Variable.Random(skillPrior).ForEach(player);
                }
                using (Variable.If(y > 0))
                {
                    skill[year][player] = Variable.GaussianFromMeanAndPrecision(skill[y - 1][player], skillChangePrecision);
                }
            }

            // Sample game outcomes
            int[][] whiteData, blackData, outcomeData;
            whiteData = new int[][] { new int[0], new int[1] { 1 } };
            blackData = new int[][] { new int[0], new int[1] { 0 } };
            outcomeData = new int[][] { new int[0], new int[1] { 0 } };

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
                    var w = whitePlayer[year][game].Named("w");
                    var b = blackPlayer[year][game].Named("b");
                    var white_performance = skill[year][w];
                    var black_performance = skill[year][b];
                    Variable<bool> white_delta = Variable.Bernoulli(Variable.Logistic(white_performance - black_performance)).Named("white_delta");
                    using (Variable.Case(outcome[year][game], 0))
                    { // black wins
                        Variable.ConstrainFalse(white_delta);
                    }
                    using (Variable.Case(outcome[year][game], 1))
                    { // white wins
                        Variable.ConstrainTrue(white_delta);
                    }
                }
            }
            //year.AddAttribute(new Sequential());   // helps inference converge faster

            engine.NumberOfIterations = 50;
            var skillChangePrecPost = engine.Infer<Gamma>(skillChangePrecision);
            Console.WriteLine(skillChangePrecPost);
        }
    }
}