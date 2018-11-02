// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;

    /// <summary>
    /// The loss function which determines how a prediction in the form of a distribution is converted into a point prediction.
    /// </summary>
    public enum LossFunction
    {
        /// <summary>
        /// The zero-one loss function is equivalent to choosing the mode of the predicted distribution as point
        /// estimate. Use this loss function to minimize mean classification error.
        /// </summary>
        ZeroOne,

        /// <summary>
        /// The squared or quadratic loss function is equivalent to choosing the mean of the predicted distribution 
        /// as point estimate. Use this loss function to minimize mean squared error.
        /// </summary>
        Squared,

        /// <summary>
        /// The absolute loss function is equivalent to choosing the median of the predicted distribution as point
        /// estimate. Use this loss function to minimize mean absolute error.
        /// </summary>
        Absolute,

        /// <summary>
        /// The custom loss function allows to provide a user-defined loss function when converting a prediction in
        /// the form of a distribution into a point prediction.
        /// </summary>
        Custom,
    }

    /// <summary>
    /// Implements point estimators.
    /// </summary>
    public static class PointEstimator
    {
        /// <summary>
        /// Gets a method which converts a <see cref="Bernoulli"/> distribution into a point estimate.
        /// </summary>
        /// <param name="lossFunction"> The <see cref="LossFunction"/>, which determines the loss to minimize.</param>
        /// <returns>The point estimator.</returns>
        public static Func<Bernoulli, bool> ForBernoulli(LossFunction lossFunction)
        {
            switch (lossFunction)
            {
                // Zero-one loss
                case LossFunction.ZeroOne:
                    return distribution => distribution.GetMode();

                // Quadratic loss
                case LossFunction.Squared:
                    return distribution => distribution.GetMean() >= 0.5;

                // Absolute loss
                case LossFunction.Absolute:
                    return distribution => distribution.GetMode(); // For Bernoulli distributions in Infer.NET, the median equals the mode!

                // Custom loss
                case LossFunction.Custom:
                    throw new InvalidOperationException("Call PointEstimator.ForBernoulli with actual custom loss function instead.");
            }

            // Should never be reached
            Debug.Fail(string.Format("Loss function {0} not supported", lossFunction));
            return null;
        }


        /// <summary>
        /// Gets a method which converts a <see cref="Discrete"/> distribution into a point estimate.
        /// </summary>
        /// <param name="lossFunction"> The <see cref="LossFunction"/>, which determines the loss to minimize.</param>
        /// <returns>The point estimator.</returns>
        public static Func<Discrete, int> ForDiscrete(LossFunction lossFunction)
        {
            switch (lossFunction)
            {
                // Zero-one loss
                case LossFunction.ZeroOne:
                    return distribution => distribution.GetMode();

                // Quadratic loss
                case LossFunction.Squared:
                    return distribution => Convert.ToInt32(distribution.GetMean());

                // Absolute loss
                case LossFunction.Absolute:
                    return distribution => distribution.GetMedian();

                // Custom loss
                case LossFunction.Custom:
                    throw new InvalidOperationException("Call PointEstimator.ForDiscrete with actual custom loss function instead.");
            }

            // Should never be reached
            Debug.Fail(string.Format("Loss function {0} not supported", lossFunction));
            return null;
        }

        /// <summary>
        /// Gets a method which converts a <see cref="Discrete"/> distribution into a point estimate.
        /// </summary>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="lossFunction"> The <see cref="LossFunction"/>, which determines the loss to minimize.</param>
        /// <returns>The point estimator.</returns>
        public static Func<IDictionary<TLabel, double>, TLabel> ForDiscrete<TLabel>(LossFunction lossFunction)
        {
            switch (lossFunction)
            {
                // Zero-one loss
                case LossFunction.ZeroOne:
                    return GetMode;

                // Quadratic loss
                case LossFunction.Squared:
                    throw new InvalidOperationException("Call PointEstimator.ForDiscrete with actual custom loss function instead.");

                // Absolute loss
                case LossFunction.Absolute:
                    throw new InvalidOperationException("Call PointEstimator.ForDiscrete with actual custom loss function instead.");

                // Custom loss
                case LossFunction.Custom:
                    throw new InvalidOperationException("Call PointEstimator.ForDiscrete with actual custom loss function instead.");
            }

            // Should never be reached
            Debug.Fail(string.Format("Loss function {0} not supported", lossFunction));
            return null;
        }


        /// <summary>
        /// Gets a method which converts a <see cref="Bernoulli"/> distribution into a point estimate.
        /// </summary>
        /// <param name="customLossFunction">The custom loss function used to compute the point estimate.</param>
        /// <returns>The point estimator.</returns>
        public static Func<Bernoulli, bool> ForBernoulli(Func<bool, bool, double> customLossFunction)
        {
            return distribution => GetEstimate(distribution, customLossFunction);
        }

        /// <summary>
        /// Gets a method which converts a <see cref="Discrete"/> distribution into a point estimate.
        /// </summary>
        /// <param name="customLossFunction">The custom loss function used to compute the point estimate.</param>
        /// <returns>The point estimator.</returns>
        public static Func<Discrete, int> ForDiscrete(Func<int, int, double> customLossFunction)
        {
            return distribution => GetEstimate(distribution, customLossFunction);
        }

        /// <summary>
        /// Gets a method which converts a <see cref="Discrete"/> distribution into a point estimate.
        /// </summary>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="customLossFunction">The custom loss function used to compute the point estimate.</param>
        /// <returns>The point estimator.</returns>
        public static Func<IDictionary<TLabel, double>, TLabel> ForDiscrete<TLabel>(Func<TLabel, TLabel, double> customLossFunction)
        {
            return distribution => GetEstimate(distribution, customLossFunction);
        }

        /// <summary>
        /// Computes the point estimate for a <see cref="Bernoulli"/> distribution using a specified loss function.
        /// </summary>
        /// <param name="distribution">The <see cref="Bernoulli"/> distribution.</param>
        /// <param name="lossFunction">The loss function.</param>
        /// <returns>The point estimate.</returns>
        public static bool GetEstimate(Bernoulli distribution, Func<bool, bool, double> lossFunction)
        {
            if (lossFunction == null)
            {
                throw new ArgumentNullException(nameof(lossFunction));
            }

            bool argminRisk = false;
            double minRisk = double.PositiveInfinity;
            double probTrue = distribution.GetProbTrue();
            double probFalse = 1 - probTrue;

            foreach (bool truth in new[] { true, false })
            {
                double risk = 0;
                foreach (bool estimate in new[] { true, false })
                {
                    risk += (estimate ? probTrue : probFalse) * lossFunction(truth, estimate);

                    // Early bailout
                    if (risk > minRisk)
                    {
                        break;
                    }
                }

                if (risk < minRisk)
                {
                    minRisk = risk;
                    argminRisk = truth;
                }
            }

            return argminRisk;
        }

        /// <summary>
        /// Computes the point estimate for a <see cref="Discrete"/> distribution using a specified loss function.
        /// </summary>
        /// <param name="distribution">The <see cref="Discrete"/> distribution.</param>
        /// <param name="lossFunction">The loss function.</param>
        /// <returns>The point estimate.</returns>
        public static int GetEstimate(Discrete distribution, Func<int, int, double> lossFunction)
        {
            if (distribution == null)
            {
                throw new ArgumentNullException(nameof(distribution));
            }

            if (lossFunction == null)
            {
                throw new ArgumentNullException(nameof(lossFunction));
            }

            int argminRisk = 0;
            double minRisk = double.PositiveInfinity;

            for (int truth = 0; truth < distribution.Dimension; truth++)
            {
                double risk = 0;
                for (int estimate = 0; estimate < distribution.Dimension; estimate++)
                {
                    risk += distribution[estimate] * lossFunction(truth, estimate);

                    // Early bailout
                    if (risk > minRisk)
                    {
                        break;
                    }
                }

                if (risk < minRisk)
                {
                    minRisk = risk;
                    argminRisk = truth;
                }
            }

            return argminRisk;
        }

        /// <summary>
        /// Computes the point estimate for a <see cref="Discrete"/> distribution using a specified loss function.
        /// </summary>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="distribution">The predictive distribution.</param>
        /// <param name="lossFunction">The loss function.</param>
        /// <returns>The point estimate.</returns>
        public static TLabel GetEstimate<TLabel>(IDictionary<TLabel, double> distribution, Func<TLabel, TLabel, double> lossFunction)
        {
            if (distribution == null)
            {
                throw new ArgumentNullException(nameof(distribution));
            }

            if (lossFunction == null)
            {
                throw new ArgumentNullException(nameof(lossFunction));
            }

            if (distribution.Count < 2)
            {
                throw new ArgumentException("The distribution must contain at least two elements", nameof(distribution));
            }

            TLabel argminRisk = distribution.Keys.First();
            double minRisk = double.PositiveInfinity;

            foreach (var truth in distribution)
            {
                double risk = 0;
                foreach (var estimate in distribution)
                {
                    risk += estimate.Value * lossFunction(truth.Key, estimate.Key);

                    // Early bailout
                    if (risk > minRisk)
                    {
                        break;
                    }
                }

                if (risk < minRisk)
                {
                    minRisk = risk;
                    argminRisk = truth.Key;
                }
            }

            return argminRisk;
        }

        /// <summary>
        /// Gets the mode of the specified generic discrete distribution.
        /// </summary>
        /// <typeparam name="TLabel">The type of a label.</typeparam>
        /// <param name="distribution">The distribution.</param>
        /// <returns>The mode of the distribution.</returns>
        public static TLabel GetMode<TLabel>(this IDictionary<TLabel, double> distribution)
        {
            if (distribution == null)
            {
                throw new ArgumentNullException(nameof(distribution));
            }

            if (distribution.Count < 2)
            {
                throw new ArgumentException("The distribution must contain at least two elements", nameof(distribution));
            }

            TLabel modeLabel = distribution.Keys.First();
            double mode = double.NegativeInfinity;
            foreach (var element in distribution)
            {
                if (element.Value > mode)
                {
                    mode = element.Value;
                    modeLabel = element.Key;
                }
            }

            return modeLabel;
        }
    }
}
