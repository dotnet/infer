// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.Collections.Generic;

    using Distributions;
    using Factors;
    using Factors.Attributes;
    using Models;

    /// <summary>
    /// Generates the inference algorithms for the Bayes point machine classifier models.
    /// </summary>
    internal static class AlgorithmFactories
    {
        #region Algorithm factory collections

        /// <summary>
        /// The training algorithm factories.
        /// </summary>
        public static readonly IEnumerable<Action<string, bool, bool>> TrainingAlgorithmFactories = 
            new Action<string, bool, bool>[]
                {
                    (generatedSourceFolder, computeModelEvidence, useCompoundWeightPriorDistribution) 
                        => CreateDenseBinaryTrainingAlgorithm(generatedSourceFolder, computeModelEvidence, useCompoundWeightPriorDistribution),
                    (generatedSourceFolder, computeModelEvidence, useCompoundWeightPriorDistribution) 
                        => CreateSparseBinaryTrainingAlgorithm(generatedSourceFolder, computeModelEvidence, useCompoundWeightPriorDistribution),
                    (generatedSourceFolder, computeModelEvidence, useCompoundWeightPriorDistribution) 
                        => CreateDenseMulticlassTrainingAlgorithm(generatedSourceFolder, computeModelEvidence, useCompoundWeightPriorDistribution),
                    (generatedSourceFolder, computeModelEvidence, useCompoundWeightPriorDistribution) 
                        => CreateSparseMulticlassTrainingAlgorithm(generatedSourceFolder, computeModelEvidence, useCompoundWeightPriorDistribution)
                };

        /// <summary>
        /// The prediction algorithm factories.
        /// </summary>
        public static readonly IEnumerable<Action<string>> PredictionAlgorithmFactories = 
            new Action<string>[]
                {
                    generatedSourceFolder => CreateDenseBinaryPredictionAlgorithm(generatedSourceFolder),
                    generatedSourceFolder => CreateSparseBinaryPredictionAlgorithm(generatedSourceFolder),
                    generatedSourceFolder => CreateDenseMulticlassPredictionAlgorithm(generatedSourceFolder),
                    generatedSourceFolder => CreateSparseMulticlassPredictionAlgorithm(generatedSourceFolder)
                };

        #endregion

        #region Training algorithms

        /// <summary>
        /// Creates an Infer.NET inference algorithm which trains a binary Bayes point machine classifier 
        /// with factorized weight distributions on features in a dense representation.
        /// </summary>
        /// <param name="generatedSourceFolder">The folder to drop the generated training algorithm to.</param>
        /// <param name="computeModelEvidence">If true, the generated training algorithm computes evidence.</param>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the generated training algorithm uses compound prior distributions over weights. Otherwise
        /// <see cref="Gaussian"/> prior distributions are used.
        /// </param>
        public static void CreateDenseBinaryTrainingAlgorithm(
            string generatedSourceFolder,
            bool computeModelEvidence,
            bool useCompoundWeightPriorDistributions)
        {
            // Create the model
            var model = new DenseBinaryModel(computeModelEvidence, useCompoundWeightPriorDistributions);

            // Add the observed variables
            model.Labels.ObservedValue = default(bool[]);

            // Add the inferred variables
            var queryVariables = new List<IVariable> { model.Weights };
            model.Weights.AddAttribute(QueryTypes.Marginal);
            model.Weights.AddAttribute(QueryTypes.MarginalDividedByPrior);

            if (useCompoundWeightPriorDistributions)
            {
                queryVariables.Add(model.WeightPrecisionRates);
                model.WeightPrecisionRates.AddAttribute(QueryTypes.MarginalDividedByPrior);

                if (computeModelEvidence)
                {
                    // This is required to compute evidence corrections
                    model.WeightPrecisionRates.AddAttribute(QueryTypes.Marginal);
                }
            }

            if (computeModelEvidence)
            {
                queryVariables.Add(model.ModelSelector);
                model.ModelSelector.AddAttribute(QueryTypes.Marginal);
            }

            // Apply the query to the model and compile the inference algorithm
            string queryName = 
                (useCompoundWeightPriorDistributions ? "Compound" : "Gaussian") 
                + "DenseBinaryBpmTraining"
                + (computeModelEvidence ? "Evidence" : string.Empty);

            GetCompiledInferenceAlgorithm(queryName, generatedSourceFolder, queryVariables.ToArray());
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm which trains a binary Bayes point machine classifier 
        /// with factorized weight distributions on features in a sparse representation.
        /// </summary>
        /// <param name="generatedSourceFolder">The folder to drop the generated training algorithm to.</param>
        /// <param name="computeModelEvidence">If true, the generated training algorithm computes evidence.</param>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the generated training algorithm uses compound prior distributions over weights. Otherwise
        /// <see cref="Gaussian"/> prior distributions are used.
        /// </param>
        public static void CreateSparseBinaryTrainingAlgorithm(
            string generatedSourceFolder,
            bool computeModelEvidence,
            bool useCompoundWeightPriorDistributions)
        {
            // Create the model
            var model = new SparseBinaryModel(computeModelEvidence, useCompoundWeightPriorDistributions);

            // Add the observed variables
            model.Labels.ObservedValue = default(bool[]);

            // Add the inferred variables
            var queryVariables = new List<IVariable> { model.Weights };
            model.Weights.AddAttribute(QueryTypes.Marginal);
            model.Weights.AddAttribute(QueryTypes.MarginalDividedByPrior);

            if (useCompoundWeightPriorDistributions)
            {
                queryVariables.Add(model.WeightPrecisionRates);
                model.WeightPrecisionRates.AddAttribute(QueryTypes.MarginalDividedByPrior);

                if (computeModelEvidence)
                {
                    // This is required to compute evidence corrections
                    model.WeightPrecisionRates.AddAttribute(QueryTypes.Marginal);
                }
            }

            if (computeModelEvidence)
            {
                queryVariables.Add(model.ModelSelector);
                model.ModelSelector.AddAttribute(QueryTypes.Marginal);
            }

            // Apply the query to the model and compile the inference algorithm
            string queryName = 
                (useCompoundWeightPriorDistributions ? "Compound" : "Gaussian") 
                + "SparseBinaryBpmTraining"
                + (computeModelEvidence ? "Evidence" : string.Empty);

            GetCompiledInferenceAlgorithm(queryName, generatedSourceFolder, queryVariables.ToArray());
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm which trains a multi-class Bayes point machine classifier 
        /// with factorized weight distributions on features in a dense representation.
        /// </summary>
        /// <param name="generatedSourceFolder">The folder to drop the generated training algorithm to.</param>
        /// <param name="computeModelEvidence">If true, the generated training algorithm computes evidence.</param>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the generated training algorithm uses compound prior distributions over weights. Otherwise
        /// <see cref="Gaussian"/> prior distributions are used.
        /// </param>
        public static void CreateDenseMulticlassTrainingAlgorithm(
            string generatedSourceFolder,
            bool computeModelEvidence,
            bool useCompoundWeightPriorDistributions)
        {
            // Create the model
            var model = new DenseMulticlassModel(computeModelEvidence, useCompoundWeightPriorDistributions, breakSymmetries: true);

            // Add the observed variables
            model.Labels.ObservedValue = default(int[]);

            // Add the inferred variables
            var queryVariables = new List<IVariable> { model.Weights };
            model.Weights.AddAttribute(QueryTypes.Marginal);
            model.Weights.AddAttribute(QueryTypes.MarginalDividedByPrior);

            if (useCompoundWeightPriorDistributions)
            {
                queryVariables.Add(model.WeightPrecisionRates);
                model.WeightPrecisionRates.AddAttribute(QueryTypes.MarginalDividedByPrior);

                if (computeModelEvidence)
                {
                    // This is required to compute evidence corrections
                    model.WeightPrecisionRates.AddAttribute(QueryTypes.Marginal);
                }
            }

            if (computeModelEvidence)
            {
                queryVariables.Add(model.ModelSelector);
                model.ModelSelector.AddAttribute(QueryTypes.Marginal);
            }

            // Apply the query to the model and compile the inference algorithm
            string queryName = 
                (useCompoundWeightPriorDistributions ? "Compound" : "Gaussian") 
                + "DenseMulticlassBpmTraining" 
                + (computeModelEvidence ? "Evidence" : string.Empty);

            GetCompiledInferenceAlgorithm(queryName, generatedSourceFolder, queryVariables.ToArray());
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm which trains a multi-class Bayes point machine classifier 
        /// with factorized weight distributions on features in a sparse representation.
        /// </summary>
        /// <param name="generatedSourceFolder">The folder to drop the generated training algorithm to.</param>
        /// <param name="computeModelEvidence">If true, the generated training algorithm computes evidence.</param>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the generated training algorithm uses compound prior distributions over weights. Otherwise
        /// <see cref="Gaussian"/> prior distributions are used.
        /// </param>
        public static void CreateSparseMulticlassTrainingAlgorithm(
            string generatedSourceFolder,
            bool computeModelEvidence,
            bool useCompoundWeightPriorDistributions)
        {
            // Create the model
            var model = new SparseMulticlassModel(computeModelEvidence, useCompoundWeightPriorDistributions, breakSymmetries: true);

            // Add the observed variables
            model.Labels.ObservedValue = default(int[]);

            // Add the inferred variables
            var queryVariables = new List<IVariable> { model.Weights };
            model.Weights.AddAttribute(QueryTypes.Marginal);
            model.Weights.AddAttribute(QueryTypes.MarginalDividedByPrior);

            if (useCompoundWeightPriorDistributions)
            {
                queryVariables.Add(model.WeightPrecisionRates);
                model.WeightPrecisionRates.AddAttribute(QueryTypes.MarginalDividedByPrior);

                if (computeModelEvidence)
                {
                    // This is required to compute evidence corrections
                    model.WeightPrecisionRates.AddAttribute(QueryTypes.Marginal);
                }
            }

            if (computeModelEvidence)
            {
                queryVariables.Add(model.ModelSelector);
                model.ModelSelector.AddAttribute(QueryTypes.Marginal);
            }

            // Apply the query to the model and compile the inference algorithm
            string queryName = 
                (useCompoundWeightPriorDistributions ? "Compound" : "Gaussian") 
                + "SparseMulticlassBpmTraining"
                + (computeModelEvidence ? "Evidence" : string.Empty);

            GetCompiledInferenceAlgorithm(queryName, generatedSourceFolder, queryVariables.ToArray());
        }

        #endregion

        #region Prediction algorithms

        /// <summary>
        /// Creates an Infer.NET inference algorithm for making predictions from a binary Bayes point machine classifier
        /// with <see cref="Gaussian"/> prior distributions over factorized weights and features in a dense representation.
        /// </summary>
        /// <param name="generatedSourceFolder">The folder to drop the generated prediction algorithm to.</param>
        public static void CreateDenseBinaryPredictionAlgorithm(string generatedSourceFolder)
        {
            // Create the model
            var model = new DenseBinaryModel(computeModelEvidence: false, useCompoundWeightPriorDistributions: false);

            // Add the inferred variables
            model.Labels.AddAttribute(QueryTypes.Marginal);

            // Apply the query to the model and compile the inference algorithm
            GetCompiledInferenceAlgorithm("GaussianDenseBinaryBpmPrediction", generatedSourceFolder, model.Labels);
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm for making predictions from a binary Bayes point machine classifier
        /// with <see cref="Gaussian"/> prior distributions over factorized weights and features in a dense representation.
        /// </summary>
        /// <param name="generatedSourceFolder">The folder to drop the generated prediction algorithm to.</param>
        public static void CreateSparseBinaryPredictionAlgorithm(string generatedSourceFolder)
        {
            // Create the model
            var model = new SparseBinaryModel(computeModelEvidence: false, useCompoundWeightPriorDistributions: false);

            // Add the inferred variables
            model.Labels.AddAttribute(QueryTypes.Marginal);

            // Apply the query to the model and compile the inference algorithm
            GetCompiledInferenceAlgorithm("GaussianSparseBinaryBpmPrediction", generatedSourceFolder, model.Labels);
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm for making predictions from a multi-class Bayes point machine classifier
        /// with <see cref="Gaussian"/> prior distributions over factorized weights and features in a dense representation.
        /// </summary>
        /// <param name="generatedSourceFolder">The folder to drop the generated prediction algorithm to.</param>
        public static void CreateDenseMulticlassPredictionAlgorithm(string generatedSourceFolder)
        {
            // Create the model
            var model = new DenseMulticlassModel(
                computeModelEvidence: false, useCompoundWeightPriorDistributions: false, breakSymmetries: false);

            // Add the inferred variables
            model.Labels.AddAttribute(QueryTypes.Marginal);

            // Apply the query to the model and compile the inference algorithm
            GetCompiledInferenceAlgorithm("GaussianDenseMulticlassBpmPrediction", generatedSourceFolder, model.Labels);
        }

        /// <summary>
        /// Creates an Infer.NET inference algorithm for making predictions from a multi-class Bayes point machine classifier
        /// with <see cref="Gaussian"/> prior distributions over factorized weights and features in a sparse representation.
        /// </summary>
        /// <param name="generatedSourceFolder">The folder to drop the generated prediction algorithm to.</param>
        public static void CreateSparseMulticlassPredictionAlgorithm(string generatedSourceFolder)
        {
            // Create the model
            var model = new SparseMulticlassModel(
                computeModelEvidence: false, useCompoundWeightPriorDistributions: false, breakSymmetries: false);

            // Add the inferred variables
            model.Labels.AddAttribute(QueryTypes.Marginal);

            // Apply the query to the model and compile the inference algorithm
            GetCompiledInferenceAlgorithm("GaussianSparseMulticlassBpmPrediction", generatedSourceFolder, model.Labels);
        }

        #endregion

        #region Helper methods

        /// <summary>
        /// Compiles a model for the specified query variables and returns the generated inference algorithm.
        /// </summary>
        /// <param name="name">The name of the generated inference algorithm.</param>
        /// <param name="generatedSourceFolder">The folder to drop the generated inference algorithm to.</param>
        /// <param name="queryVariables">The query variables.</param>
        private static void GetCompiledInferenceAlgorithm(string name, string generatedSourceFolder, params IVariable[] queryVariables)
        {
            const string modelNamespace = "Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal";
            // Create the inference engine
            var engine = new InferenceEngine
            {
                ModelNamespace = modelNamespace,
                ModelName = name,
                ShowProgress = false
            };

            engine.Compiler.AddComments = false;  // avoid irrelevant code changes
            engine.Compiler.FreeMemory = false;
            engine.Compiler.GeneratedSourceFolder = generatedSourceFolder;
            engine.Compiler.GenerateInMemory = true;
            engine.Compiler.GivePriorityTo(typeof(GammaFromShapeAndRateOp_Laplace));
            engine.Compiler.RecommendedQuality = QualityBand.Experimental; // bug
            engine.Compiler.ReturnCopies = false;
            engine.Compiler.ShowWarnings = true;
            engine.Compiler.UseSerialSchedules = true;
            engine.Compiler.WriteSourceFiles = true;

            // Generate the inference algorithm
            engine.GetCompiledInferenceAlgorithm(queryVariables);
        }

        #endregion
    }
}
