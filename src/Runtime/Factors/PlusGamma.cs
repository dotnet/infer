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
            if (a.IsUniform() || b == 0) return a;
            else if (a.Power == 0) throw new ArgumentException($"Cannot add {b} to {a}");
            else if (a.IsPointMass) return GammaPower.PointMass(a.Point + b, a.Power);
            else if (a.Power < 0)
            {
                if (a.Shape <= a.Power) return a; // mode is at zero
                // The mode is ((Shape - Power)/Rate)^Power
                // We want to shift the mode by b, preserving the Shape and Power.
                // This implies ((Shape - Power)/newRate)^Power = newMode
                // newRate = (Shape - Power)/newMode^(1/Power)
                //         = (a.Shape - a.Power) * Math.Pow(a.GetMode() + b, -1 / a.Power);
                //double logMode = a.Power * (Math.Log(Math.Max(0, a.Shape - a.Power)) - Math.Log(a.Rate));
                //if (logMode > double.MaxValue) return a; // mode is at infinity
                double logShapeMinusPower = Math.Log(a.Shape - a.Power);
                double mode = a.GetMode();
                if (mode > double.MaxValue) return a; // mode is at infinity
                double newMode = mode + b;
                double newLogMode = Math.Log(newMode);
                // Find newLogRate to satisfy a.Power*(logShapeMinusPower - newLogRate) <= newLogMode
                // logShapeMinusPower - newLogRate >= newLogMode/a.Power
                // newLogRate - logShapeMinusPower <= -newLogMode/a.Power
                double newLogModeOverPower = MMath.LargestDoubleRatio(-a.Power, newLogMode);
                double newLogRate = MMath.LargestDoubleSum(logShapeMinusPower, newLogModeOverPower);
                if ((logShapeMinusPower - newLogRate) * a.Power > newLogMode) throw new Exception();
                // Ideally this would find largest newRate such that log(newRate) <= newLogRate
                double newRate = Math.Exp(newLogRate);
                if (logShapeMinusPower == newLogRate) newRate = a.Shape - a.Power;
                if (a.Rate > 0) newRate = Math.Max(double.Epsilon, newRate);
                if (!double.IsPositiveInfinity(a.Rate)) newRate = Math.Min(double.MaxValue, newRate);
                return GammaPower.FromShapeAndRate(a.Shape, newRate, a.Power);
            }
            else if (!a.IsProper()) throw new ImproperDistributionException(a);
            else
            {
                // The mean is Math.Exp(Power * (MMath.RisingFactorialLnOverN(Shape, Power) - Math.Log(Rate)))
                // We want to shift the mean by b, preserving the Shape and Power.
                // This implies log(newRate) = MMath.RisingFactorialLnOverN(Shape, Power) - log(newMean)/Power
                double logShape = MMath.RisingFactorialLnOverN(a.Shape, a.Power);
                //double logMean = a.GetLogMeanPower(1);
                //double newLogMean = (b > 0) ? 
                //    MMath.LogSumExp(logMean, Math.Log(b)) :
                //    MMath.LogDifferenceOfExp(logMean, Math.Log(-b));
                double newLogMean = Math.Log(a.GetMean() + b);
                // If logShape is big, this difference can lose accuracy
                // Find newLogRate to satisfy logShape - newLogRate <= newLogMean/a.Power
                double newLogMeanOverPower = MMath.LargestDoubleRatio(a.Power, newLogMean);
                double newLogRate = -MMath.LargestDoubleSum(-logShape, newLogMeanOverPower);
                // check: (logShape - newLogRate)*a.Power <= newLogMean
                if ((logShape - newLogRate) * a.Power > newLogMean) throw new Exception();
                double newRate = Math.Exp(newLogRate);
                newRate = Math.Max(double.Epsilon, newRate);
                if (!double.IsPositiveInfinity(a.Rate)) newRate = Math.Min(double.MaxValue, newRate);
                return GammaPower.FromShapeAndRate(a.Shape, newRate, a.Power);
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
