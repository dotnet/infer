// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Models.Attributes;

    /// <summary>
    /// An Infer.NET model of a binary Bayes point machine classifier with non-factorized weight distribution
    /// and features in a dense representation.
    /// </summary>
    internal class DenseBinaryVectorModel
    {
        #region Fields and constructor

        /// <summary>
        /// The range over instances.
        /// </summary>
        private Range instanceRange;

        /// <summary>
        /// Initializes a new instance of the <see cref="DenseBinaryVectorModel"/> class.
        /// </summary>
        /// <param name="computeModelEvidence">If true, the model computes evidence.</param>
        /// <param name="useCompoundWeightPriorDistributions">
        /// If true, the model uses compound prior distributions over weights. Otherwise, 
        /// <see cref="Gaussian"/> prior distributions are used.
        /// </param>
        public DenseBinaryVectorModel(
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
        public Variable<Vector> Weights { get; private set; }

        /// <summary>
        /// Gets the random variables for the weight priors.
        /// </summary>
        public Variable<VectorGaussian> WeightPriors { get; private set; }

        /// <summary>
        /// Gets the random variables for the weight precision rates.
        /// </summary>
        public Variable<PositiveDefiniteMatrix> WeightPrecisionRate { get; private set; }

        /// <summary>
        /// Gets the random variable for the model selector.
        /// </summary>
        public Variable<bool> ModelSelector { get; private set; }

        /// <summary>
        /// The variables for the observed feature values.
        /// </summary>
        public VariableArray<Vector> FeatureValues { get; private set; }

        public Variable<int> InstanceCount { get; private set; }

        public Variable<int> FeatureCount { get; private set; }

        #endregion

        #region Model implementation

        /// <summary>
        /// Defines the binary Bayes point machine classifier model with factorized weight distributions.
        /// </summary>
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
            this.InstanceCount = Variable.Observed(default(int)).Named("InstanceCount");
            this.instanceRange = new Range(this.InstanceCount).Named("InstanceRange");

            // Generate a schedule which is sequential over instances
            this.instanceRange.AddAttribute(new Sequential());

            // Define the range over features
            this.FeatureCount = Variable.Observed(default(int)).Named("FeatureCount");
        }

        /// <summary>
        /// Defines the prior distributions used in the binary Bayes point machine classifier model.
        /// </summary>
        private void DefinePrior(bool useCompoundWeightPriorDistributions)
        {
            // Define the observed feature values
            this.FeatureValues =
                Variable.Observed(default(Vector[]), this.instanceRange).Named("FeatureValues");

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
                this.Labels[this.instanceRange] = this.ComputeNoisyScore(this.Weights, this.FeatureValues[this.instanceRange]) > 0;
            }
        }

        /// <summary>
        /// Defines the compound prior distributions over weights used in the binary Bayes point machine classifier model.
        /// </summary>
        private void DefineCompoundWeightPrior()
        {
            var commonWeightPrecision = Variable.GammaFromShapeAndRate(1, 1).Named("CommonWeightPrecision");

            // Define the observed weight constraints for incremental and batched training
            var weightConstraints = Variable.Observed(default(VectorGaussian)).Named("WeightConstraints");

            // Define the observed weight precision rate constraints for incremental and batched training
            var weightPrecisionRateConstraints = Variable.Observed(default(Wishart)).Named("WeightPrecisionRateConstraints");

            // The compound Gamma distributions over weight precisions
            var eye = Variable<PositiveDefiniteMatrix>.Factor(PositiveDefiniteMatrix.Identity, this.FeatureCount).Named("eye");
            var weightPrecisionRateRate = Variable.WishartFromShapeAndRate(1.0, eye);
            weightPrecisionRateRate.Name = nameof(weightPrecisionRateRate);
            var sharedWeightPrecisionRate = Variable.WishartFromShapeAndRate(1.0, weightPrecisionRateRate);
            sharedWeightPrecisionRate.Name = nameof(sharedWeightPrecisionRate);
            // TODO: This needs to be a ratio not a product.  The product is left as a placeholder.
            // var weightPrecision = commonWeightPrecision / sharedWeightPrecisionRate;
            var weightPrecision = Variable<PositiveDefiniteMatrix>.Factor(Factors.Factor.Product, commonWeightPrecision, sharedWeightPrecisionRate);
            weightPrecision.Name = nameof(weightPrecision);

            // The distributions over weights
            var zero = Variable<Vector>.Factor(Vector.Zero, this.FeatureCount).Named("zero");
            this.Weights = Variable.VectorGaussianFromMeanAndPrecision(zero, weightPrecision).Named("Weights");

            // Constrain weights to their marginals divided by their priors to allow for incremental and batched training
            Variable.ConstrainEqualRandom(this.Weights, weightConstraints);

            // By switching to the ReplicateOp_NoDivide operators, weight precision rates are initialized before the computation of
            // the likelihood. This results in the weights having a range similar to the data, avoiding numerical instabilities.
            sharedWeightPrecisionRate.AddAttribute(new DivideMessages(false));

            // Constrain weight precision rates to their marginals divided by their priors to allow for incremental and batched training
            this.WeightPrecisionRate = Variable.Copy(sharedWeightPrecisionRate);
            this.WeightPrecisionRate.Name = nameof(this.WeightPrecisionRate);
            Variable.ConstrainEqualRandom(this.WeightPrecisionRate, weightPrecisionRateConstraints);

            // Anchor weight precision rates at feature values (required for compound prior distributions)
            this.FeatureValues[this.instanceRange] =
                Variable.VectorGaussianFromMeanAndPrecision(zero, this.WeightPrecisionRate).ForEach(this.instanceRange);
        }

        /// <summary>
        /// Defines the <see cref="Gaussian"/> prior distributions over weights used in the binary Bayes point machine classifier model.
        /// </summary>
        private void DefineGaussianWeightPrior()
        {
            // Define the observed Gaussian prior distributions
            var weightPriors = Variable.Observed(default(VectorGaussian)).Named("WeightPriors");

            // Define the observed weight constraints for incremental and batched training
            var weightConstraints = Variable.Observed(default(VectorGaussian)).Named("WeightConstraints");

            // The distributions over weights
            this.Weights = Variable<Vector>.Random(weightPriors);
            this.Weights.Name = nameof(this.Weights);

            // Constrain weights to their marginals divided by their priors to allow for incremental and batched training
            Variable.ConstrainEqualRandom(this.Weights, weightConstraints);
        }

        /// <summary>
        /// Computes the noisy class score for given feature values and weights.
        /// </summary>
        /// <param name="weights">The weights.</param>
        /// <param name="values">The feature values of a single instance.</param>
        /// <returns>The noisy score.</returns>
        private Variable<double> ComputeNoisyScore(Variable<Vector> weights, Variable<Vector> values)
        {
            var score = Variable.InnerProduct(weights, values).Named("Score");
            var noisyScore = Variable.GaussianFromMeanAndVariance(score, 1.0).Named("NoisyScore");
            return noisyScore;
        }

        #endregion
    }
}
