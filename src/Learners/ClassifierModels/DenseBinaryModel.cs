// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Models.Attributes;
    using GammaArray = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray<Distributions.Gamma, double>;
    using GaussianArray = Microsoft.ML.Probabilistic.Distributions.DistributionStructArray<Distributions.Gaussian, double>;

    /// <summary>
    /// An Infer.NET model of a binary Bayes point machine classifier with factorized weight distributions 
    /// and features in a dense representation.
    /// <para>
    /// The factorized weight distributions can be interpreted as a <see cref="VectorGaussian"/> distribution 
    /// with diagonal covariance matrix, which ignores possible correlations between weights.
    /// </para>
    /// </summary>
    internal class DenseBinaryModel
    {
        #region Fields and constructor

        /// <summary>
        /// The range over instances.
        /// </summary>
        private Range instanceRange;

        /// <summary>
        /// The range over features.
        /// </summary>
        private Range featureRange;

        /// <summary>
        /// The variables for the observed feature values.
        /// </summary>
        private VariableArray<VariableArray<double>, double[][]> featureValues;

        /// <summary>
        /// Initializes a new instance of the <see cref="DenseBinaryModel"/> class.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the model computes evidence.</param>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the model uses compound prior distributions over weights. Otherwise, 
        /// <see cref="Gaussian"/> prior distributions are used.
        /// </param>
        public DenseBinaryModel(
            bool computeModelEvidence, 
            bool useCompoundWeightPriorDistributions)
        {
            if (computeModelEvidence)
            {
                // Create Bayes point machine classifier model with surrounding evidence block
                this.ModelSelector = Variable.Bernoulli(0.5).Named("ModelSelector");
                using (Variable.If(this.ModelSelector))
                {
                    this.DefineModel(useCompoundWeightPriorDistributions);
                }
            }
            else
            {
                // Create Bayes point machine classifier model without evidence block
                this.DefineModel(useCompoundWeightPriorDistributions);
            }
        }

        #endregion

        #region Variables accessible in inference queries

        /// <summary>
        /// Gets the random variables for the labels.
        /// </summary>
        public VariableArray<bool> Labels { get; private set; }

        /// <summary>
        /// Gets the random variables for the weights.
        /// </summary>
        public VariableArray<double> Weights { get; private set; }

        /// <summary>
        /// Gets the random variables for the weight precision rates.
        /// </summary>
        public VariableArray<double> WeightPrecisionRates { get; private set; }

        /// <summary>
        /// Gets the random variable for the model selector.
        /// </summary>
        public Variable<bool> ModelSelector { get; private set; }

        #endregion

        #region Model implementation

        /// <summary>
        /// Defines the binary Bayes point machine classifier model with factorized weight distributions.
        /// </summary>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the model uses compound prior distributions over weights. Otherwise, <see cref="Gaussian"/> prior distributions are used.
        /// </param>
        private void DefineModel(bool useCompoundWeightPriorDistributions)
        {
            this.DefineRanges();
            this.DefinePrior(useCompoundWeightPriorDistributions);
            this.DefineLikelihood();
        }

        /// <summary>
        /// Defines the ranges of plates used in the binary Bayes point machine classifier model.
        /// </summary>
        private void DefineRanges()
        {
            // Define the range over instances
            var instanceCount = Variable.Observed(default(int)).Named("InstanceCount");
            this.instanceRange = new Range(instanceCount).Named("InstanceRange");

            // Generate a schedule which is sequential over instances
            this.instanceRange.AddAttribute(new Sequential());

            // Define the range over features
            var featureCount = Variable.Observed(default(int)).Named("FeatureCount");
            this.featureRange = new Range(featureCount).Named("FeatureRange");
        }

        /// <summary>
        /// Defines the prior distributions used in the binary Bayes point machine classifier model.
        /// </summary>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the model uses compound prior distributions over weights. Otherwise, <see cref="Gaussian"/> prior distributions are used.
        /// </param>
        private void DefinePrior(bool useCompoundWeightPriorDistributions)
        {
            // Define the observed feature values
            this.featureValues = 
                Variable.Observed(default(double[][]), this.instanceRange, this.featureRange).Named("FeatureValues");

            // Define the weights
            this.Weights = Variable.Array<double>(this.featureRange).Named("Weights");

            // Define the prior distributions over weights
            if (useCompoundWeightPriorDistributions)
            {
                this.DefineCompoundWeightPrior();
            }
            else
            {
                this.DefineGaussianWeightPrior();
            }
        }

        /// <summary>
        /// Defines the likelihood of the binary Bayes point machine classifier model.
        /// </summary>
        private void DefineLikelihood()
        {
            // Define the class labels
            this.Labels = Variable.Array<bool>(this.instanceRange).Named("Labels");

            // For all instances...
            using (Variable.ForEach(this.instanceRange))
            {
                this.Labels[this.instanceRange] = this.ComputeNoisyScore(this.Weights, this.featureValues[this.instanceRange]) > 0;
            }
        }

        /// <summary>
        /// Defines the compound prior distributions over weights used in the binary Bayes point machine classifier model.
        /// </summary>
        private void DefineCompoundWeightPrior()
        {
            var weightPrecisionRateRates = Variable.Array<double>(this.featureRange).Named("WeightPrecisionRateRates");
            this.WeightPrecisionRates = Variable.Array<double>(this.featureRange).Named("WeightPrecisionRates");
            var sharedWeightPrecisionRates = Variable.Array<double>(this.featureRange).Named("SharedWeightPrecisionRates");
            var weightPrecisions = Variable.Array<double>(this.featureRange).Named("WeightPrecisions");
            var commonWeightPrecision = Variable.GammaFromShapeAndRate(1, 1).Named("CommonWeightPrecision");

            // Define the observed weight constraints for incremental and batched training
            var weightConstraints = Variable.Observed(default(GaussianArray)).Named("WeightConstraints");

            // Define the observed weight precision rate constraints for incremental and batched training
            var weightPrecisionRateConstraints = Variable.Observed(default(GammaArray)).Named("WeightPrecisionRateConstraints");

            // The compound Gamma distributions over weight precisions
            weightPrecisionRateRates[this.featureRange] = Variable.GammaFromShapeAndRate(1.0, 1.0).ForEach(this.featureRange);
            sharedWeightPrecisionRates[this.featureRange] = Variable.GammaFromShapeAndRate(1.0, weightPrecisionRateRates[this.featureRange]);
            weightPrecisions[this.featureRange] = commonWeightPrecision / sharedWeightPrecisionRates[this.featureRange];

            // The distributions over weights
            this.Weights[this.featureRange] = Variable.GaussianFromMeanAndPrecision(0.0, weightPrecisions[this.featureRange]);

            // Constrain weights to their marginals divided by their priors to allow for incremental and batched training
            Variable.ConstrainEqualRandom<double[], GaussianArray>(this.Weights, weightConstraints);

            // By switching to the ReplicateOp_NoDivide operators, weight precision rates are initialized before the computation of 
            // the likelihood. This results in the weights having a range similar to the data, avoiding numerical instabilities.
            sharedWeightPrecisionRates.AddAttribute(new DivideMessages(false));

            // Constrain weight precision rates to their marginals divided by their priors to allow for incremental and batched training
            this.WeightPrecisionRates[this.featureRange] = Variable.Copy(sharedWeightPrecisionRates[this.featureRange]);
            Variable.ConstrainEqualRandom<double[], GammaArray>(this.WeightPrecisionRates, weightPrecisionRateConstraints);

            // Anchor weight precision rates at feature values (required for compound prior distributions)
            this.featureValues[this.instanceRange][this.featureRange] = 
                Variable.GaussianFromMeanAndPrecision(0.0, this.WeightPrecisionRates[this.featureRange]).ForEach(this.instanceRange);
        }

        /// <summary>
        /// Defines the <see cref="Gaussian"/> prior distributions over weights used in the binary Bayes point machine classifier model.
        /// </summary>
        private void DefineGaussianWeightPrior()
        {
            // Define the observed Gaussian prior distributions
            var weightPriors = Variable.Observed(default(GaussianArray)).Named("WeightPriors");

            // Define the observed weight constraints for incremental and batched training
            var weightConstraints = Variable.Observed(default(GaussianArray)).Named("WeightConstraints");

            // The distributions over weights
            this.Weights.SetTo(Variable<double[]>.Random(weightPriors));

            // Constrain weights to their marginals divided by their priors to allow for incremental and batched training
            Variable.ConstrainEqualRandom<double[], GaussianArray>(this.Weights, weightConstraints);
        }

        /// <summary>
        /// Computes the noisy class score for given feature values and weights.
        /// </summary>
        /// <param name="weights">The weights.</param>
        /// <param name="values">The feature values of a single instance.</param>
        /// <returns>The noisy score.</returns>
        private Variable<double> ComputeNoisyScore(VariableArray<double> weights, VariableArray<double> values)
        {
            var featureScores = Variable.Array<double>(this.featureRange).Named("FeatureScores");
            featureScores[this.featureRange] = values[this.featureRange] * weights[this.featureRange];
            var score = Variable.Sum(featureScores).Named("Score");
            var noisyScore = Variable.GaussianFromMeanAndVariance(score, 1.0).Named("NoisyScore");
            return noisyScore;
        }

        #endregion
    }
}