// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors.Attributes;

namespace Microsoft.ML.Probabilistic.Factors
{
    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/doc/*'/>
    [FactorMethod(typeof(System.Math), "Pow", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class PowerOp
    {
        // GammaPower = GammaPower ^ y  /////////////////////////////////////////////////////////

        public static GammaPower PowAverageConditional([SkipIfUniform] GammaPower x, double y)
        {
            return GammaPower.FromShapeAndRate(x.Shape, x.Rate, y*x.Power);
        }

        public static GammaPower XAverageConditional([SkipIfUniform] GammaPower pow, double y, GammaPower result)
        {
            double power = pow.Power / y;
            if (power != result.Power)
                throw new NotSupportedException("Outgoing message power (" + power + ") does not match the desired power (" + result.Power + ")");
            return GammaPower.FromShapeAndRate(pow.Shape, pow.Rate, pow.Power / y);
        }

        public static GammaPower PowAverageLogarithm([SkipIfUniform] GammaPower x, double y)
        {
            return PowAverageConditional(x, y);
        }

        public static GammaPower XAverageLogarithm([SkipIfUniform] GammaPower pow, double y, GammaPower result)
        {
            return XAverageConditional(pow, y, result);
        }

        // GammaPower = Gamma ^ y //////////////////////////////////////////////////////////////

        public static GammaPower PowAverageConditional([SkipIfUniform] Gamma x, double y)
        {
            return GammaPower.FromShapeAndRate(x.Shape, x.Rate, y);
        }

        public static Gamma XAverageConditional([SkipIfUniform] GammaPower pow, double y)
        {
            if (y != pow.Power)
                throw new NotSupportedException("Incoming message " + pow + " does not match the exponent (" + y + ")");
            return Gamma.FromShapeAndRate(pow.Shape, pow.Rate);
        }

        public static GammaPower PowAverageLogarithm([SkipIfUniform] Gamma x, double y)
        {
            return PowAverageConditional(x, y);
        }

        public static Gamma XAverageLogarithm([SkipIfUniform] GammaPower pow, double y)
        {
            return XAverageConditional(pow, y);
        }

        // Pareto = Pareto ^ y ////////////////////////////////////////////////////////////////////

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="PowAverageConditional(Pareto, double)"]/*'/>
        public static Pareto PowAverageConditional(Pareto x, double y)
        {
            if (y < 0)
                throw new NotSupportedException("Pareto raised to a negative power (" + y + ") cannot be represented by a Pareto distribution");
            // p(x) =propto 1/x^(s+1)
            // z = x^y implies
            // p(z) = p(x = z^(1/y)) (1/y) z^(1/y-1)
            //      =propto z^(-(s+1)/y+1/y-1) = z^(-s/y-1)
            return new Pareto(x.Shape / y, System.Math.Pow(x.LowerBound, y));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="XAverageConditional(Pareto, Gamma, double)"]/*'/>
        public static Gamma XAverageConditional(Pareto pow, Gamma x, double y)
        {
            // factor is delta(pow - x^y)
            // marginal for x is Pareto(x^y; s,L) Ga(x; a,b)
            // =propto x^(a-1-y(s+1)) exp(-bx)   for x >= L^(1/y)
            // we can compute moments via the incomplete Gamma function.
            // change variables to: z=bx,  dz=b dx
            // int_LL^inf x^(c-1) exp(-bx) dx = int_(bLL)^inf z^(c-1)/b^c exp(-z) dz = gammainc(c,bLL,inf)/b^c
            double lowerBound = System.Math.Pow(pow.LowerBound, 1 / y);
            double b = x.Rate;
            double c = x.Shape - y * (pow.Shape + 1);
            double bL = b * lowerBound;
            double m, m2;
            if (y > 0)
            {
                // note these ratios can be simplified
                double z = MMath.GammaUpper(c, bL);
                m = MMath.GammaUpper(c + 1, bL) * c / z / b;
                m2 = MMath.GammaUpper(c + 2, bL) * c * (c + 1) / z / (b * b);
            }
            else
            {
                double z = MMath.GammaLower(c, bL);
                m = MMath.GammaLower(c + 1, bL) * c / z / b;
                m2 = MMath.GammaLower(c + 2, bL) * c * (c + 1) / z / (b * b);
            }
            double v = m2 - m * m;
            Gamma xPost = Gamma.FromMeanAndVariance(m, v);
            Gamma result = new Gamma();
            result.SetToRatio(xPost, x, true);
            return result;
        }
    }
}
