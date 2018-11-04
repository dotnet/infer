// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tutorials
{
    using System;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Distributions;

    [Example("Applications", "A simplified version of the TrueSkill rating system")]
    public class TrueSkill
    {
        public void Run()
        {
            // Compile model
            var trueSkill = new Model();

            // Load data
            var players1 = new[] { 0, 1, 3 };
            var players2 = new[] { 1, 2, 2 };
            var player1Wins = new[] { true, true, true };

            // Training
            trueSkill.Train(players1, players2, player1Wins);

            // Prediction
            var player0Vs3 = trueSkill.PredictDistribution(0, 3);

            // Both players 0 and 3 have one win and no losses, but player 0 is more likely to win
            Console.WriteLine(player0Vs3);
        }

        private class Model
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="Model"/> class.
            /// </summary>
            public Model()
            {
                var trainingVariableToInfer = this.CreateModel(isTrainingModel: true);
                var trainingInferenceEngine = GetInferenceEngine();
                trainingInferenceEngine.ModelName = "Training";
                this.TrainingInferenceAlgorithm =
                    trainingInferenceEngine.GetCompiledInferenceAlgorithm(trainingVariableToInfer);

                var predictionVariableToInfer = this.CreateModel(isTrainingModel: false);
                if (predictionVariableToInfer != null)
                {
                    var predictionInferenceEngine = GetInferenceEngine();
                    predictionInferenceEngine.ModelName = "Prediction";
                    this.PredictionInferenceAlgorithm =
                        predictionInferenceEngine.GetCompiledInferenceAlgorithm(predictionVariableToInfer);
                }
            }

            /// <summary>
            /// Gets the training inference algorithm.
            /// </summary>
            private IGeneratedAlgorithm TrainingInferenceAlgorithm { get; }

            /// <summary>
            /// Gets the prediction inference algorithm.
            /// </summary>
            private IGeneratedAlgorithm PredictionInferenceAlgorithm { get; }

            #region Locals

            /// <summary>
            /// The prior mean of the player skill.
            /// </summary>
            /// <remarks>Value learned on Halo 5 data.</remarks>
            private const double SkillMean = 6.0;

            /// <summary>
            /// The prior standard deviation of the player skill.
            /// </summary>
            /// <remarks>Value learned on Halo 5 data.</remarks>
            private const double SkillStandardDeviation = 3.0;

            /// <summary>
            /// The performance noise.
            /// </summary>
            private const double PerformanceNoiseVariance = 1.0;

            /// <summary>
            /// Gets or sets the number of games played.
            /// </summary>
            private Variable<int> GameCount { get; set; }

            /// <summary>
            /// Gets or sets the number of unique players.
            /// </summary>
            private Variable<int> PlayerCount { get; set; }

            /// <summary>
            /// Gets or sets the first players in each game.
            /// </summary>
            private VariableArray<int> Players1 { get; set; }

            /// <summary>
            /// Gets or sets the second players in each game.
            /// </summary>
            private VariableArray<int> Players2 { get; set; }

            /// <summary>
            /// Gets or sets a value indicating whether player 1 beat player 2 on a given game.
            /// </summary>
            private VariableArray<bool> Player1Wins { get; set; }

            /// <summary>
            /// Gets or sets the skills of the players.
            /// </summary>
            private VariableArray<double> PlayerSkills { get; set; }

            /// <summary>
            /// Gets or sets the prior of the player skills.
            /// </summary>
            private VariableArray<Gaussian> PlayerSkillsPrior { get; set; }

            #endregion

            /// <summary>
            /// Trains the model.
            /// </summary>
            /// <remarks>
            /// Calling this method multiple times will re-train the model
            /// (as opposed to incrementally learn).
            /// </remarks>        
            public void Train(int[] players1, int[] players2, bool[] player1Wins)
            {
                // Check input data
                if (players1.Length != players2.Length && players2.Length != player1Wins.Length)
                {
                    throw new ArgumentException("The number of elements in the training data arrays must be the same.");
                }

                // Player identifiers are assumed to be zero-based consecutive integers
                var playerCount = players1.Concat(players2).Max() + 1;

                // Set training data
                this.TrainingInferenceAlgorithm.SetObservedValue(this.GameCount.Name, players1.Length);
                this.TrainingInferenceAlgorithm.SetObservedValue(this.PlayerCount.Name, playerCount);
                this.TrainingInferenceAlgorithm.SetObservedValue(this.Players1.Name, players1);
                this.TrainingInferenceAlgorithm.SetObservedValue(this.Players2.Name, players2);
                this.TrainingInferenceAlgorithm.SetObservedValue(this.Player1Wins.Name, player1Wins);

                // Set the priors
                this.TrainingInferenceAlgorithm.SetObservedValue(
                    this.PlayerSkillsPrior.Name,
                    Util.ArrayInit(playerCount,
                        p => new Gaussian(SkillMean, SkillStandardDeviation * SkillStandardDeviation)));

                // Run inference
                this.TrainingInferenceAlgorithm.Execute(InferenceEngine.DefaultEngine.NumberOfIterations);

                // Set up the prediction algorithm
                this.PredictionInferenceAlgorithm.SetObservedValue(this.PlayerCount.Name, playerCount);
                this.PredictionInferenceAlgorithm.SetObservedValue(
                    this.PlayerSkillsPrior.Name,
                    this.TrainingInferenceAlgorithm.Marginal<Gaussian[]>(this.PlayerSkills.Name));
            }

            /// <summary>
            /// Predicts the probability of the player passed as a first argument 
            /// to will win over the player passed as a second argument.
            /// </summary>
            /// <returns>
            /// The probability of the player passed as a first argument 
            /// to win over the player passed as a second argument.
            /// </returns>
            public Bernoulli PredictDistribution(int winner, int loser)
            {
                // Set observed values
                this.PredictionInferenceAlgorithm.SetObservedValue(this.GameCount.Name, 1);
                this.PredictionInferenceAlgorithm.SetObservedValue(this.Players1.Name, new[] { winner });
                this.PredictionInferenceAlgorithm.SetObservedValue(this.Players2.Name, new[] { loser });

                // Inference
                this.PredictionInferenceAlgorithm.Execute(InferenceEngine.DefaultEngine.NumberOfIterations);
                return this.PredictionInferenceAlgorithm.Marginal<Bernoulli[]>(this.Player1Wins.Name)[0];
            }

            /// <summary>
            /// Creates the Infer.NET model.
            /// </summary>
            /// <param name="isTrainingModel">
            /// True if the model is a training model, 
            /// false if the model is a prediction model.
            /// </param>
            /// <returns>The variables to infer.</returns>
            private IVariable[] CreateModel(bool isTrainingModel)
            {
                // Define the number of games
                this.GameCount = Variable.Observed(default(int)).Named("GameCount");
                var game = new Range(this.GameCount).Named("game");
                //// game.AddAttribute(new Sequential());

                // Define the number of players
                this.PlayerCount = Variable.Observed(default(int)).Named("PlayerCount");
                var player = new Range(this.PlayerCount).Named("player");

                // Define the prior skill of each player
                this.PlayerSkills = Variable.Array<double>(player).Named("PlayerSkills");
                this.PlayerSkillsPrior = Variable.Observed(default(Gaussian[]), player).Named("PlayerSkillsPrior");
                this.PlayerSkills[player] = Variable<double>.Random(this.PlayerSkillsPrior[player]);

                // Define the observed game outcomes
                this.Players1 = Variable.Observed(default(int[]), game).Named("Players1");
                this.Players2 = Variable.Observed(default(int[]), game).Named("Players2");
                var performanceNoiseVariance = Variable.Observed(PerformanceNoiseVariance)
                    .Named("PerformanceNoiseVariance");
                this.Player1Wins = isTrainingModel
                    ? Variable.Observed(default(bool[]), game)
                    : Variable.Array<bool>(game);
                this.Player1Wins.Name = "Player1Wins";

                using (Variable.ForEach(game))
                {
                    // Define the player performance as a noisy version of their skill
                    var player1Performance = Variable.GaussianFromMeanAndVariance(
                        this.PlayerSkills[this.Players1[game]], performanceNoiseVariance).Named("Player1Performance");
                    var player2Performance = Variable.GaussianFromMeanAndVariance(
                        this.PlayerSkills[this.Players2[game]], performanceNoiseVariance).Named("Player2Performance");

                    this.Player1Wins[game] = player1Performance > player2Performance;

                    //// The following code demonstrates how to model ties.
                    //// It requires integer labels: 0 (win), 1 (draw), 2 (lose)
                    ////var playerSkillDiff = (player1Performance - player2Performance).Named("PlayerSkillDiff");
                    ////this.Outcomes[game] = Variable.DiscreteUniform(3).Named("Outcome");
                    ////using (Variable.Case(this.Outcomes[game], 0))
                    ////{
                    ////    Variable.ConstrainTrue(playerSkillDiff > this.DrawMargin);
                    ////}
                    ////using (Variable.Case(this.Outcomes[game], 1))
                    ////{
                    ////    Variable.ConstrainBetween(playerSkillDiff, -this.DrawMargin, this.DrawMargin);
                    ////}
                    ////using (Variable.Case(this.Outcomes[game], 2))
                    ////{
                    ////    Variable.ConstrainTrue(playerSkillDiff < -this.DrawMargin);
                    ////}
                }

                return isTrainingModel
                    ? new IVariable[] { this.PlayerSkills }
                    : new IVariable[] { this.Player1Wins };
            }

            /// <summary>
            /// Creates an Infer.NET inference engine and sets its options.
            /// </summary>
            /// <returns>Infer.NET inference engine.</returns>
            private static InferenceEngine GetInferenceEngine()
            {
                var engine = new InferenceEngine();

                // engine.ShowProgress = false;
                engine.Compiler.ReturnCopies = false;
                engine.Compiler.FreeMemory = false;
                engine.Compiler.GenerateInMemory = false;
                engine.Compiler.WriteSourceFiles = true;

                return engine;
            }
        }
    }
}
