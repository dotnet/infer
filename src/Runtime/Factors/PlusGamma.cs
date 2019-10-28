// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaVmpOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Plus", typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Experimental)]
    public static class PlusGammaOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaOp"]/message_doc[@name="SumAverageConditional(GammaPower, double)"]/*'/>
        public static GammaPower SumAverageConditional([SkipIfUniform] GammaPower a, double b)
        {
            if (double.IsInfinity(b) || double.IsNaN(b)) throw new ArgumentOutOfRangeException(nameof(b), b, $"Argument is outside the range of supported values.");
            if (a.IsUniform()) return a;
            else if (a.Power == 0 && b != 0) throw new ArgumentException("Cannot add {b} to {a}");
            else if (a.IsPointMass) return GammaPower.PointMass(a.Point + b, a.Power);
            else if (a.Power < 0)
            {                
                if (a.Shape <= a.Power) return a; // mode is at infinity
                // The mode is ((Shape - Power)/Rate)^Power
                // We want to shift the mode by b, preserving the Shape and Power.
                // This implies ((Shape - Power)/newRate)^Power = newMode
                // newRate = (Shape - Power)/newMode^(1/Power)
                return GammaPower.FromShapeAndRate(a.Shape, (a.Shape - a.Power) * Math.Pow(a.GetMode() + b, -1 / a.Power), a.Power);
            }
            else if (!a.IsProper()) throw new ImproperDistributionException(a);
            else
            {
                // The mean is Math.Exp(MMath.GammaLn(Shape + Power) - MMath.GammaLn(Shape) - Power * Math.Log(Rate))
                // We want to shift the mean by b, preserving the Shape and Power.
                // This implies s*newRate^(-Power) = newMean
                // newRate = (newMean/s)^(-1/Power)
                // If power == 1, mean is shape/rate, newRate = shape/newMean.
                double s = (a.Power == 1) ? a.Shape : Math.Exp(MMath.GammaLn(a.Shape + a.Power) - MMath.GammaLn(a.Shape));
                double r = Math.Pow(a.Rate, -a.Power);
                return GammaPower.FromShapeAndRate(a.Shape, Math.Pow(r + b / s, -1 / a.Power), a.Power);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaOp"]/message_doc[@name="SumAverageConditional(GammaPower, double)"]/*'/>
        public static GammaPower SumAverageConditional(double a, [SkipIfUniform] GammaPower b)
        {
            return SumAverageConditional(b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaOp"]/message_doc[@name="AAverageConditional(GammaPower, double)"]/*'/>
        public static GammaPower AAverageConditional([SkipIfUniform] GammaPower sum, double b)
        {
            return SumAverageConditional(sum, -b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaOp"]/message_doc[@name="BAverageConditional(GammaPower, double)"]/*'/>
        public static GammaPower BAverageConditional([SkipIfUniform] GammaPower sum, double a)
        {
            return AAverageConditional(sum, a);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaVmpOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Plus", typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Experimental)]
    public static class PlusGammaVmpOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaVmpOp"]/message_doc[@name="SumAverageLogarithm(Gamma, Gamma)"]/*'/>
        public static Gamma SumAverageLogarithm([Proper] Gamma a, [Proper] Gamma b)
        {
            // return a Gamma with the correct moments
            double ma, va, mb, vb;
            a.GetMeanAndVariance(out ma, out va);
            b.GetMeanAndVariance(out mb, out vb);
            return Gamma.FromMeanAndVariance(ma + mb, va + vb);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaVmpOp"]/message_doc[@name="AAverageLogarithm(Gamma, Gamma, Gamma)"]/*'/>
        public static Gamma AAverageLogarithm([SkipIfUniform] Gamma sum, [Proper] Gamma a, [Proper] Gamma b)
        {
            // f = int_sum sum^(ss-1)*exp(-sr*sum)*delta(sum = a+b) dsum
            //   = (a+b)^(ss-1)*exp(-sr*(a+b))
            // log(f) = (ss-1)*log(a+b) - sr*(a+b)
            // apply a lower bound:
            // log(a+b) >= q*log(a) + (1-q)*log(b) - q*log(q) - (1-q)*log(1-q)
            // optimal q = exp(a)/(exp(a)+exp(b)) if (a,b) are fixed
            // optimal q = exp(E[log(a)])/(exp(E[log(a)])+exp(E[log(b)]))  if (a,b) are random
            // This generalizes the bound used by Cemgil (2008).
            // The message to A has shape (ss-1)*q + 1 and rate sr.
            double x = sum.Shape - 1;
            double ma = Math.Exp(a.GetMeanLog());
            double mb = Math.Exp(b.GetMeanLog());
            double p = ma / (ma + mb);
            double m = x * p;
            return Gamma.FromShapeAndRate(m + 1, sum.Rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaVmpOp"]/message_doc[@name="BAverageLogarithm(Gamma, Gamma, Gamma)"]/*'/>
        public static Gamma BAverageLogarithm([SkipIfUniform] Gamma sum, [Proper] Gamma a, [Proper] Gamma b)
        {
            return AAverageLogarithm(sum, b, a);
        }
    }
}
