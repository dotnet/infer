// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Math
{
    using System;

    /// <summary>
    /// A class for evaluating continued fractions
    /// </summary>
    public abstract class ContinuedFraction
    {
        /// <summary>
        /// Gets the numerator for the current term.
        /// </summary>
        /// <param name="n">The iteration - must be greater than 0.</param>
        /// <returns>The numerator for the current term.</returns>
        public abstract double GetNumerator(int n);

        /// <summary>
        /// Gets the denominator for the current term.
        /// </summary>
        /// <param name="n">The iteration - must be greater than 0.</param>
        /// <returns>The denominator for the current term.</returns>
        public abstract double GetDenominator(int n);

        /// <summary>
        /// Evaluates the continued fraction.
        /// </summary>
        /// <param name="epsilon">The convergence tolerance.</param>
        /// <returns>The value of the fraction</returns>
        protected double Evaluate(double epsilon)
        {
            double previousA = 0.0;
            double currentA = GetNumerator(1);
            double previousB = 1.0;
            double currentB = GetDenominator(1);

            double currFrac = currentA / currentB;

            for (int n = 2; ; n++)
            {
                double an = GetNumerator(n);
                double bn = GetDenominator(n);
                double aNext = bn * currentA + an * previousA;
                double bNext = bn * currentB + an * previousB;
                previousA = currentA;
                previousB = currentB;
                currentA = aNext;
                currentB = bNext;
                double nextFrac = currentA / currentB;
                double delta = nextFrac - currFrac;
                currFrac = nextFrac;

                if (Math.Abs(delta) < epsilon * Math.Abs(currFrac))
                {
                    break;
                }

                if (n > 1000)
                {
                    throw new Exception("Continued fraction does not converge");
                }
            }

            return currFrac;
        }
    }
}
