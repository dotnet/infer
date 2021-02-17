// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using System.Diagnostics;

namespace Microsoft.ML.Probabilistic.Factors
{
    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/doc/*'/>
    [FactorMethod(typeof(System.Math), "Pow", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class PowerOp
    {
        public static Gamma GammaFromMeanAndMeanInverse(double mean, double meanInverse)
        {
            if (mean < 0) throw new ArgumentOutOfRangeException(nameof(mean), mean, "mean < 0");
            if (meanInverse < 0) throw new ArgumentOutOfRangeException(nameof(meanInverse), meanInverse, "meanInverse < 0");
            // mean = a/b
            // meanInverse = b/(a-1)
            // a = mean*meanInverse / (mean*meanInverse - 1)
            // b = a/mean
            double rate = meanInverse / (mean * meanInverse - 1);
            if (rate < 0 || rate > double.MaxValue) return Gamma.PointMass(mean);
            double shape = mean * rate;
            if (shape > double.MaxValue)
                return Gamma.PointMass(mean);
            else
                return Gamma.FromShapeAndRate(shape, rate);
        }

        public static Gamma GammaFromGammaPower(GammaPower message)
        {
            if (message.Power == 1) return Gamma.FromShapeAndRate(message.Shape, message.Rate); // same as below, but faster
            if (message.IsUniform()) return Gamma.Uniform();
            message.GetMeanAndVariance(out double mean, out double variance);
            return Gamma.FromMeanAndVariance(mean, variance);
        }

        public static Gamma FromMeanPowerAndMeanLog(double meanPower, double meanLog, double power)
        {
            // We want E[log(x)] = meanLog but this sets E[log(x^power)] = meanLog, so we scale meanLog
            var gammaPower = GammaPower.FromMeanAndMeanLog(meanPower, meanLog * power, power);
            return Gamma.FromShapeAndRate(gammaPower.Shape, gammaPower.Rate);
        }

        // Gamma = TruncatedGamma ^ y  /////////////////////////////////////////////////////////

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="PowAverageConditional(TruncatedGamma, double)"]/*'/>
        public static Gamma PowAverageConditional([SkipIfUniform] TruncatedGamma x, double y)
        {
            double mean = x.GetMeanPower(y);
            if (x.LowerBound > 0)
            {
                double meanInverse = x.GetMeanPower(-y);
                return GammaFromMeanAndMeanInverse(mean, meanInverse);
            }
            else
            {
                double variance = x.GetMeanPower(2 * y) - mean * mean;
                return Gamma.FromMeanAndVariance(mean, variance);
            }
        }

        public static TruncatedGamma XAverageConditional([SkipIfUniform] Gamma pow, TruncatedGamma x, double y)
        {
            // message computed below should be uniform when pow is uniform, but may not due to roundoff error.
            if (pow.IsUniform()) return TruncatedGamma.Uniform();
            // Factor is (x^y)^(pow.Shape/pow.Power - 1) * exp(-pow.Rate*(x^y)^1/pow.Power)
            // =propto x^(pow.Shape/(pow.Power/y) - y) * exp(-pow.Rate*x^y/pow.Power)
            // newShape/(pow.Power/y) - 1 = pow.Shape/(pow.Power/y) - y
            // newShape = pow.Shape + (1-y)*(pow.Power/y)
            double power = 1 / y;
            var toPow = PowAverageConditional(x, y);
            var powMarginal = pow * toPow;
            // xMarginal2 is the exact distribution of pow^(1/y) where pow has distribution powMarginal
            GammaPower xMarginal2 = GammaPower.FromShapeAndRate(powMarginal.Shape, powMarginal.Rate, power);
            var xMarginal = new TruncatedGamma(GammaFromGammaPower(xMarginal2), x.LowerBound, x.UpperBound);
            var result = xMarginal;
            result.SetToRatio(xMarginal, x, GammaProductOp_Laplace.ForceProper);
            return result;
        }

        // GammaPower = TruncatedGamma ^ y  /////////////////////////////////////////////////////////

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="XAverageConditional(TruncatedGamma, double, GammaPower)"]/*'/>
        public static GammaPower PowAverageConditional([SkipIfUniform] TruncatedGamma x, double y, GammaPower result)
        {
            if (result.Power == -1) y = -y;
            else if (result.Power != 1) throw new ArgumentException($"result.Power ({result.Power}) is not 1 or -1", nameof(result));
            double mean = x.GetMeanPower(y);
            if (x.LowerBound > 0)
            {
                double meanInverse = x.GetMeanPower(-y);
                Gamma result1 = GammaFromMeanAndMeanInverse(mean, meanInverse);
                return GammaPower.FromShapeAndRate(result1.Shape, result1.Rate, result.Power);
            }
            else
            {
                double variance = x.GetMeanPower(2 * y) - mean * mean;
                Gamma result1 = Gamma.FromMeanAndVariance(mean, variance);
                return GammaPower.FromShapeAndRate(result1.Shape, result1.Rate, result.Power);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="XAverageConditional(GammaPower, TruncatedGamma, double)"]/*'/>
        public static TruncatedGamma XAverageConditional([SkipIfUniform] GammaPower pow, TruncatedGamma x, double y)
        {
            // message computed below should be uniform when pow is uniform, but may not due to roundoff error.
            if (pow.IsUniform()) return TruncatedGamma.Uniform();
            // Factor is (x^y)^(pow.Shape/pow.Power - 1) * exp(-pow.Rate*(x^y)^1/pow.Power)
            // =propto x^(pow.Shape/(pow.Power/y) - y) * exp(-pow.Rate*x^y/pow.Power)
            // newShape/(pow.Power/y) - 1 = pow.Shape/(pow.Power/y) - y
            // newShape = pow.Shape + (1-y)*(pow.Power/y)
            double power = pow.Power / y;
            var toPow = PowAverageConditional(x, y, pow);
            var powMarginal = pow * toPow;
            // xMarginal2 is the exact distribution of pow^(1/y) where pow has distribution powMarginal
            GammaPower xMarginal2 = GammaPower.FromShapeAndRate(powMarginal.Shape, powMarginal.Rate, power);
            var xMarginal = new TruncatedGamma(GammaFromGammaPower(xMarginal2), x.LowerBound, x.UpperBound);
            var result = xMarginal;
            result.SetToRatio(xMarginal, x, GammaProductOp_Laplace.ForceProper);
            return result;
        }

        // Gamma = Gamma ^ y  /////////////////////////////////////////////////////////

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="PowAverageConditional(Gamma, double, Gamma)"]/*'/>
        public static Gamma PowAverageConditional([SkipIfUniform] Gamma x, double y, Gamma result)
        {
            GammaPower message = PowAverageConditional(x, y);
            return GammaFromGammaPower(message);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="XAverageConditional(Gamma, Gamma, double, Gamma)"]/*'/>
        public static Gamma XAverageConditional([SkipIfUniform] Gamma pow, Gamma x, double y, Gamma result)
        {
            // message computed below should be uniform when pow is uniform, but may not due to roundoff error.
            if (pow.IsUniform()) return Gamma.Uniform();
            // Factor is (x^y)^(pow.Shape/pow.Power - 1) * exp(-pow.Rate*(x^y)^1/pow.Power)
            // =propto x^(pow.Shape/(pow.Power/y) - y) * exp(-pow.Rate*x^y/pow.Power)
            // newShape/(pow.Power/y) - 1 = pow.Shape/(pow.Power/y) - y
            // newShape = pow.Shape + (1-y)*(pow.Power/y)
            double power = 1 / y;
            var toPow = PowAverageConditional(x, y, pow);
            var powMarginal = pow * toPow;
            // xMarginal2 is the exact distribution of pow^(1/y) where pow has distribution powMarginal
            GammaPower xMarginal2 = GammaPower.FromShapeAndRate(powMarginal.Shape, powMarginal.Rate, power);
            Gamma xMarginal = GammaFromGammaPower(xMarginal2);
            result.SetToRatio(xMarginal, x, GammaProductOp_Laplace.ForceProper);
            return result;
        }

        // GammaPower = GammaPower ^ y  /////////////////////////////////////////////////////////

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="LogAverageFactor(GammaPower, GammaPower, double)"]/*'/>
        public static double LogAverageFactor(GammaPower pow, GammaPower x, double y)
        {
            // GetLogAverageOf = 
            // gammaln(shape1+shape2-power) - (shape1+shape2-power)*log(rate1+rate2) - log(|power|)
            // -gammaln(shape1) + shape1 * log(rate1)
            // -gammaln(shape2) + shape2 * log(rate2)
            // d/dshape2 = digamma(shape1+shape2-power) - digamma(shape2) - log(rate1/rate2 + 1)
            // d/drate2 = -(shape1+shape2-power)/(rate1+rate2) + shape2/rate2
            GammaPower toPow = PowAverageConditional(x, y, pow);
            return toPow.GetLogAverageOf(pow);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="PowAverageConditional(GammaPower, double, GammaPower)"]/*'/>
        public static GammaPower PowAverageConditional([SkipIfUniform] GammaPower x, double y, GammaPower result)
        {
            GammaPower message;
            if (x.IsPointMass) message = GammaPower.PointMass(System.Math.Pow(x.Point, y), y * x.Power);
            else message = GammaPower.FromShapeAndRate(x.Shape, x.Rate, y * x.Power);
            return GammaPowerFromDifferentPower(message, result.Power);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="XAverageConditional(GammaPower, GammaPower, double, GammaPower)"]/*'/>
        public static GammaPower XAverageConditional([SkipIfUniform] GammaPower pow, GammaPower x, double y, GammaPower result)
        {
            // message computed below should be uniform when pow is uniform, but may not due to roundoff error.
            if (pow.IsUniform()) return GammaPower.Uniform(result.Power);
            // Factor is (x^y)^(pow.Shape/pow.Power - 1) * exp(-pow.Rate*(x^y)^1/pow.Power)
            // =propto x^(pow.Shape/(pow.Power/y) - y) * exp(-pow.Rate*x^y/pow.Power)
            // newShape/(pow.Power/y) - 1 = pow.Shape/(pow.Power/y) - y
            // newShape = pow.Shape + (1-y)*(pow.Power/y)
            double power = pow.Power / y;
            var toPow = PowAverageConditional(x, y, pow);
            GammaPower powMarginal = pow * toPow;
            // xMarginal2 is the exact distribution of pow^(1/y) where pow has distribution powMarginal
            GammaPower xMarginal2 = GammaPower.FromShapeAndRate(powMarginal.Shape, powMarginal.Rate, power);
            GammaPower xMarginal = GammaPowerFromDifferentPower(xMarginal2, result.Power);
            result.SetToRatio(xMarginal, x, GammaProductOp_Laplace.ForceProper);
            return result;
        }

        public static GammaPower GammaPowerFromDifferentPower(GammaPower message, double newPower)
        {
            if (message.Power == newPower) return message; // same as below, but faster
            if (message.IsUniform()) return GammaPower.Uniform(newPower);
            // Making two hops ensures that the desired mean powers are finite.
            if (message.Power > 0 && newPower < 0 && newPower != -1) return GammaPowerFromDifferentPower(GammaPowerFromDifferentPower(message, -1), newPower);
            if (message.Power < 0 && newPower > 0 && newPower != 1) return GammaPowerFromDifferentPower(GammaPowerFromDifferentPower(message, 1), newPower);
            // Project the message onto the desired power
            if (newPower == 1 || newPower == -1 || newPower == 2)
            {
                message.GetMeanAndVariance(out double mean, out double variance);
                if (!double.IsPositiveInfinity(mean))
                    return GammaPower.FromMeanAndVariance(mean, variance, newPower);
                // Fall through
            }
            bool useMean = false;
            if(useMean)
            {
                // Constraints:
                // mean = Gamma(Shape + newPower)/Gamma(Shape)/Rate^newPower =approx (Shape/Rate)^newPower
                // mean2 = Gamma(Shape + 2*newPower)/Gamma(Shape)/Rate^(2*newPower) =approx ((Shape + newPower)/Rate)^newPower * (Shape/Rate)^newPower
                // mean2/mean^2 = Gamma(Shape + 2*newPower)*Gamma(Shape)/Gamma(Shape + newPower)^2 =approx ((Shape + newPower)/Shape)^newPower
                // Shape =approx newPower/((mean2/mean^2)^(1/newPower) - 1)
                // Rate = Shape/mean^(1/newPower)
                message.GetMeanAndVariance(out double mean, out double variance);
                double meanp = System.Math.Pow(mean, 1 / newPower);
                double mean2p = System.Math.Pow(variance + mean * mean, 1 / newPower);
                double shape = newPower / (mean2p / meanp / meanp - 1);
                if (double.IsInfinity(shape)) return GammaPower.PointMass(mean, newPower);
                double rate = shape / meanp;
                return GammaPower.FromShapeAndRate(shape, rate, newPower);
            }
            else
            {
                // Compute the mean and variance of x^1/newPower
                double mean = message.GetMeanPower(1 / newPower);
                double mean2 = message.GetMeanPower(2 / newPower);
                double variance = System.Math.Max(0, mean2 - mean * mean);
                if (double.IsPositiveInfinity(mean*mean)) variance = mean;
                return GammaPower.FromGamma(Gamma.FromMeanAndVariance(mean, variance), newPower);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="PowAverageLogarithm(GammaPower, double, GammaPower)"]/*'/>
        public static GammaPower PowAverageLogarithm([SkipIfUniform] GammaPower x, double y, GammaPower result)
        {
            return PowAverageConditional(x, y, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="XAverageLogarithm(GammaPower, GammaPower, double, GammaPower)"]/*'/>
        public static GammaPower XAverageLogarithm([SkipIfUniform] GammaPower pow, GammaPower x, double y, GammaPower result)
        {
            return XAverageConditional(pow, x, y, result);
        }

        // GammaPower = Gamma ^ y //////////////////////////////////////////////////////////////

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="PowAverageConditional(Gamma, double)"]/*'/>
        public static GammaPower PowAverageConditional([SkipIfUniform] Gamma x, double y)
        {
            if (x.IsPointMass) return GammaPower.PointMass(System.Math.Pow(x.Point, y), y);
            else return GammaPower.FromShapeAndRate(x.Shape, x.Rate, y);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="XAverageConditional(GammaPower, double)"]/*'/>
        public static Gamma XAverageConditional([SkipIfUniform] GammaPower pow, double y)
        {
            if (pow.IsPointMass) return Gamma.PointMass(System.Math.Pow(pow.Point, 1 / y));
            if (y != pow.Power)
                throw new NotSupportedException("Incoming message " + pow + " does not match the exponent (" + y + ")");
            return Gamma.FromShapeAndRate(pow.Shape + (1 - y), pow.Rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="PowAverageLogarithm(Gamma, double)"]/*'/>
        public static GammaPower PowAverageLogarithm([SkipIfUniform] Gamma x, double y)
        {
            return PowAverageConditional(x, y);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PowerOp"]/message_doc[@name="XAverageLogarithm(GammaPower, double)"]/*'/>
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
