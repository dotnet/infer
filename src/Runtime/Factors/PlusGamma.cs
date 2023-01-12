// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Plus", typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Experimental)]
    public static class PlusGammaOp
    {
        const bool ForceProper = true;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaOp"]/message_doc[@name="SumAverageConditional(GammaPower, GammaPower)"]/*'/>
        public static GammaPower SumAverageConditional([SkipIfUniform] GammaPower a, [SkipIfUniform] GammaPower b, GammaPower result)
        {
            return PlusGammaVmpOp.SumAverageLogarithm(a, b, result);
        }

        public static GammaPower AAverageConditional([SkipIfUniform] GammaPower sum, [SkipIfUniform] GammaPower a, [SkipIfUniform] GammaPower b, GammaPower to_a, GammaPower to_b)
        {
            if (sum.IsUniform()) return sum;
            return PlusGammaVmpOp.AAverageLogarithm(sum, a, b, to_a, to_b);
        }

        public static double TruncatedGammaPowerGetMean(GammaPower gammaPower, double lowerBound, double upperBound)
        {
            // int_(L^(1/p))^(U^(1/p)) x^p p(x) dx
            TruncatedGamma truncatedGamma;
            if (gammaPower.Power < 0)
            {
                truncatedGamma = new TruncatedGamma(Gamma.FromShapeAndRate(gammaPower.Shape, gammaPower.Rate), Math.Pow(upperBound, 1 / gammaPower.Power), Math.Pow(lowerBound, 1 / gammaPower.Power));
            }
            else
            {
                truncatedGamma = new TruncatedGamma(Gamma.FromShapeAndRate(gammaPower.Shape, gammaPower.Rate), Math.Pow(lowerBound, 1 / gammaPower.Power), Math.Pow(upperBound, 1 / gammaPower.Power));
            }
            //return Math.Pow(truncatedGamma.GetMean(), gammaPower.Power);
            return truncatedGamma.GetMeanPower(gammaPower.Power);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaOp"]/message_doc[@name="AAverageConditional(GammaPower, GammaPower)"]/*'/>
        public static GammaPower AAverageConditional2([SkipIfUniform] GammaPower sum, [SkipIfUniform] GammaPower a, [SkipIfUniform] GammaPower b, GammaPower result)
        {
            if (sum.IsUniform() || b.IsUniform())
            {
                result.SetToUniform();
                return result;
            }
            if (sum.IsPointMass && sum.Point == 0)
            {
                if (b.IsPointMass || a.Shape / a.Power <= b.Shape / b.Power)
                {
                    result.Point = 0;
                    return result;
                }
                else
                {
                    result.SetToUniform();
                    return result;
                }
            }
            if (sum.IsProper() && b.IsProper())
            {
                sum.GetMeanAndVariance(out double sumMean, out double sumVariance);
                b.GetMeanAndVariance(out double bMean, out double bVariance);
                double rMean = sumMean - bMean;
                double rVariance = sumVariance + bVariance;
                double aVariance = a.GetVariance();
                if (aVariance >= rVariance)
                {
                    if (rMean < 0)
                    {
                        // If b cannot be less than x, then sum cannot be less than x.
                        double tailProbability = 0.1;
                        double bLowerBound = b.GetQuantile(tailProbability);
                        // If sum cannot be more than x, then b cannot be more than x.
                        double sumUpperBound = sum.GetQuantile(1 - tailProbability);
                        if (sumUpperBound > bLowerBound)
                        {
                            // Compute the mean of the truncated distributions.
                            // sum is truncated to [bLowerBound, infinity]
                            sumMean = TruncatedGammaPowerGetMean(sum, bLowerBound, double.PositiveInfinity);
                            // b is truncated to [0, sumUpperBound]
                            bMean = TruncatedGammaPowerGetMean(b, 0, sumUpperBound);
                            rMean = sumMean - bMean;
                        }
                        else
                        {
                            //result.Point = 0;
                            //return result;
                        }
                    }
                    if (rMean > 0)
                    {
                        result = GammaPower.FromMeanAndVariance(rMean, rVariance, result.Power);
                        if (ForceProper && result.Power == 1 && result.Shape < 1)
                        {
                            result.Shape = 1;
                            // Set rate such that shape/rate = rMean
                            result.Rate = result.Shape / rMean;
                        }
                        return result;
                    }
                }
            }
            // logZ = sum.GetLogAverageOf(toSum)
            // where toSum = SumAverageConditional(a, b)
            // Compute the derivatives wrt a.Rate to get the message to A.
            if (sum.Power == 1)
            {
                GetGammaMomentDerivs(a, out double mean, out double dmean, out double ddmean, out double variance, out double dvariance, out double ddvariance);
                mean += b.GetMean();
                variance += b.GetVariance();
                GetGammaDerivs(mean, dmean, ddmean, variance, dvariance, ddvariance, out double ds, out double dds, out double dr, out double ddr);
                GetDerivLogZ(sum, GammaPower.FromMeanAndVariance(mean, variance, sum.Power), ds, dds, dr, ddr, out double dlogZ, out double ddlogZ);
                return GammaPowerFromDerivLogZ(a, dlogZ, ddlogZ);
            }
            else if (sum.Power == -1)
            {
                bool aHasInfiniteMean = (a.Power == -1 && a.Shape <= 1);
                bool bHasInfiniteMean = (b.Power == -1 && b.Shape <= 1);
                if (aHasInfiniteMean)
                {
                    if (bHasInfiniteMean)
                    {
                        if (a.Shape <= b.Shape) return sum;
                        else
                        {
                            result.SetToUniform();
                            return result;
                        }
                    }
                    else return sum;
                }
                else if (bHasInfiniteMean)
                {
                    result.SetToUniform();
                    return result;
                }

                GetInverseGammaMomentDerivs(a, out double mean, out double dmean, out double ddmean, out double variance, out double dvariance, out double ddvariance);
                mean += b.GetMean();
                variance += b.GetVariance();
                if (variance > double.MaxValue) return GammaPower.Uniform(a.Power); //throw new NotSupportedException();
                GetInverseGammaDerivs(mean, dmean, ddmean, variance, dvariance, ddvariance, out double ds, out double dds, out double dr, out double ddr);
                if (sum.IsPointMass && sum.Point == 0) return GammaPower.PointMass(0, a.Power);
                GetDerivLogZ(sum, GammaPower.FromMeanAndVariance(mean, variance, sum.Power), ds, dds, dr, ddr, out double dlogZ, out double ddlogZ);
                return GammaPowerFromDerivLogZ(a, dlogZ, ddlogZ);
            }
            else throw new NotImplementedException($"sum.Power == {sum.Power}");
        }

        public static GammaPower GammaPowerFromDerivLogZ(GammaPower a, double dlogZ, double ddlogZ)
        {
            bool method1 = false;
            if (method1)
            {
                GetPosteriorMeanAndVariance(Gamma.FromShapeAndRate(a.Shape, a.Rate), dlogZ, ddlogZ, out double iaMean, out double iaVariance);
                Gamma ia = Gamma.FromMeanAndVariance(iaMean, iaVariance);
                return GammaPower.FromShapeAndRate(ia.Shape, ia.Rate, a.Power) / a;
            }
            else
            {
                double alpha = -a.Rate * dlogZ;
                // dalpha/dr = -dlogZ - r*ddlogZ
                // beta = -r * dalpha/dr
                double beta = a.Rate * (dlogZ + a.Rate * ddlogZ);
                Gamma prior = Gamma.FromShapeAndRate(a.Shape, a.Rate);
                // ia is the marginal of a^(1/a.Power)
                Gamma ia = GaussianOp.GammaFromAlphaBeta(prior, alpha, beta, ForceProper) * prior;
                return GammaPower.FromShapeAndRate(ia.Shape, ia.Rate, a.Power) / a;
            }
        }

        /// <summary>
        /// Gets first and second derivatives of the moments with respect to the rate parameter of the distribution.
        /// </summary>
        /// <param name="gammaPower"></param>
        /// <param name="mean"></param>
        /// <param name="dmean"></param>
        /// <param name="ddmean"></param>
        /// <param name="variance"></param>
        /// <param name="dvariance"></param>
        /// <param name="ddvariance"></param>
        public static void GetInverseGammaMomentDerivs(GammaPower gammaPower, out double mean, out double dmean, out double ddmean, out double variance, out double dvariance, out double ddvariance)
        {
            if (gammaPower.Power != -1) throw new ArgumentException();
            if (gammaPower.Shape <= 2) throw new ArgumentOutOfRangeException($"gammaPower.Shape <= 2");
            mean = gammaPower.Rate / (gammaPower.Shape - 1);
            dmean = 1 / (gammaPower.Shape - 1);
            ddmean = 0;
            variance = mean * mean / (gammaPower.Shape - 2);
            dvariance = 2 * mean * dmean / (gammaPower.Shape - 2);
            ddvariance = 2 * dmean * dmean / (gammaPower.Shape - 2);
        }

        /// <summary>
        /// Gets the derivatives of the function that converts moments into gamma parameters.
        /// </summary>
        public static void GetInverseGammaDerivs(double mean, double dmean, double ddmean, double variance, double dvariance, double ddvariance, out double ds, out double dds, out double dr, out double ddr)
        {
            double shape = 2 + mean * mean / variance;
            double v2 = variance * variance;
            ds = 2 * mean * dmean / variance - mean * mean / v2 * dvariance;
            dds = 2 * mean * ddmean / variance - mean * mean / v2 * ddvariance + 2 * dmean * dmean / variance - 4 * mean * dmean / v2 * dvariance + 2 * mean * mean / (v2 * variance) * dvariance * dvariance;
            dr = dmean * (shape - 1) + mean * ds;
            ddr = ddmean * (shape - 1) + 2 * dmean * ds + mean * dds;
        }

        /// <summary>
        /// Gets first and second derivatives of the moments with respect to the rate parameter of the distribution.
        /// </summary>
        /// <param name="gammaPower"></param>
        /// <param name="mean"></param>
        /// <param name="dmean"></param>
        /// <param name="ddmean"></param>
        /// <param name="variance"></param>
        /// <param name="dvariance"></param>
        /// <param name="ddvariance"></param>
        public static void GetGammaMomentDerivs(GammaPower gammaPower, out double mean, out double dmean, out double ddmean, out double variance, out double dvariance, out double ddvariance)
        {
            if (gammaPower.Power != 1) throw new ArgumentException();
            mean = gammaPower.Shape / gammaPower.Rate;
            variance = mean / gammaPower.Rate;
            dmean = -variance;
            ddmean = 2 * variance / gammaPower.Rate;
            dvariance = -ddmean;
            ddvariance = -3 * dvariance / gammaPower.Rate;
        }

        /// <summary>
        /// Gets the derivatives of the function that converts moments into gamma parameters.
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="dmean"></param>
        /// <param name="ddmean"></param>
        /// <param name="variance"></param>
        /// <param name="dvariance"></param>
        /// <param name="ddvariance"></param>
        /// <param name="ds"></param>
        /// <param name="dds"></param>
        /// <param name="dr"></param>
        /// <param name="ddr"></param>
        public static void GetGammaDerivs(double mean, double dmean, double ddmean, double variance, double dvariance, double ddvariance, out double ds, out double dds, out double dr, out double ddr)
        {
            double rate = mean / variance;
            //double shape = mean * rate;
            double v2 = variance * variance;
            dr = dmean / variance - mean / v2 * dvariance;
            ddr = ddmean / variance - mean / v2 * ddvariance - 2 * dmean / v2 * dvariance + 2 * mean / (v2 * variance) * dvariance * dvariance;
            ds = dmean * rate + mean * dr;
            dds = ddmean * rate + 2 * dmean * dr + mean * ddr;
        }

        /// <summary>
        /// Gets the derivatives of the function that converts gamma parameters of toSum into the integral with sum.
        /// </summary>
        /// <param name="sum"></param>
        /// <param name="toSum"></param>
        /// <param name="ds"></param>
        /// <param name="dds"></param>
        /// <param name="dr"></param>
        /// <param name="ddr"></param>
        /// <param name="dlogZ"></param>
        /// <param name="ddlogZ"></param>
        public static void GetDerivLogZ(GammaPower sum, GammaPower toSum, double ds, double dds, double dr, double ddr, out double dlogZ, out double ddlogZ)
        {
            if (sum.Power != toSum.Power) throw new ArgumentException($"sum.Power ({sum.Power}) != toSum.Power ({toSum.Power})");
            if (toSum.IsPointMass) throw new NotSupportedException();
            if (toSum.IsUniform())
            {
                dlogZ = 0;
                ddlogZ = 0;
                return;
            }
            if (sum.IsPointMass)
            {
                // Z = toSum.GetLogProb(sum.Point)
                // log(Z) = (toSum.Shape/toSum.Power - 1)*log(sum.Point) - toSum.Rate*sum.Point^(1/toSum.Power) + toSum.Shape*log(toSum.Rate) - GammaLn(toSum.Shape)
                if (sum.Point == 0) throw new NotSupportedException();
                double logSumOverPower = Math.Log(sum.Point) / toSum.Power;
                double powSum = Math.Exp(logSumOverPower);
                double logRate = Math.Log(toSum.Rate);
                double digammaShape = MMath.Digamma(toSum.Shape);
                double shapeOverRate = toSum.Shape / toSum.Rate;
                dlogZ = ds * logSumOverPower - dr * powSum + ds * logRate + shapeOverRate * dr - digammaShape * ds;
                ddlogZ = dds * logSumOverPower - ddr * powSum + dds * logRate + 2 * ds * dr / toSum.Rate + shapeOverRate * ddr - MMath.Trigamma(toSum.Shape) * ds - digammaShape * dds;
            }
            else
            {
                GammaPower product = sum * toSum;
                double cs = (MMath.Digamma(product.Shape) - Math.Log(product.Shape)) - (MMath.Digamma(toSum.Shape) - Math.Log(toSum.Shape));
                double cr = toSum.Shape / toSum.Rate - product.Shape / product.Rate;
                double css = MMath.Trigamma(product.Shape) - MMath.Trigamma(toSum.Shape);
                double csr = 1 / toSum.Rate - 1 / product.Rate;
                double crr = product.Shape / (product.Rate * product.Rate) - toSum.Shape / (toSum.Rate * toSum.Rate);
                dlogZ = cs * ds + cr * dr;
                ddlogZ = cs * dds + cr * ddr + css * ds * ds + 2 * csr * ds * dr + crr * dr * dr;
            }
        }

        public static void GetPosteriorMeanAndVariance(Gamma prior, double dlogZ, double ddlogZ, out double mean, out double variance)
        {
            // dlogZ is derivative of log(Z) wrt prior rate parameter
            prior.GetMeanAndVariance(out double priorMean, out double priorVariance);
            mean = priorMean - dlogZ;
            variance = priorVariance + ddlogZ;
        }

        public static GammaPower BAverageConditional([SkipIfUniform] GammaPower sum, [SkipIfUniform] GammaPower a, [SkipIfUniform] GammaPower b, GammaPower to_a, GammaPower to_b)
        {
            return AAverageConditional(sum, b, a, to_b, to_a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaOp"]/message_doc[@name="BAverageConditional(GammaPower, GammaPower)"]/*'/>
        public static GammaPower BAverageConditional2([SkipIfUniform] GammaPower sum, [SkipIfUniform] GammaPower a, [SkipIfUniform] GammaPower b, GammaPower result)
        {
            return AAverageConditional2(sum, b, a, result);
        }

        public static double LogAverageFactor([SkipIfUniform] GammaPower sum, [SkipIfUniform] GammaPower a, [SkipIfUniform] GammaPower b)
        {
            GammaPower toSum = SumAverageConditional(a, b, sum);
            return toSum.GetLogAverageOf(sum);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaOp"]/message_doc[@name="LogEvidenceRatio(GammaPower, GammaPower, GammaPower)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] GammaPower sum, [SkipIfUniform] GammaPower a, [SkipIfUniform] GammaPower b)
        {
            return 0.0;
        }

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
                double newMode = Math.Max(0, mode + b);
                double newLogMode = Math.Log(newMode);
                // Find newLogRate to satisfy a.Power*(logShapeMinusPower - newLogRate) <= newLogMode
                // logShapeMinusPower - newLogRate >= newLogMode/a.Power
                // newLogRate - logShapeMinusPower <= -newLogMode/a.Power
                double newLogModeOverPower = MMath.LargestDoubleRatio(newLogMode, -a.Power);
                double newLogRate = MMath.LargestDoubleSum(logShapeMinusPower, newLogModeOverPower);
                if ((double)((logShapeMinusPower - newLogRate) * a.Power) > newLogMode) throw new Exception();
                // Ideally this would find largest newRate such that log(newRate) <= newLogRate
                double newRate = Math.Exp(newLogRate);
                if (logShapeMinusPower == newLogRate) newRate = a.Shape - a.Power;
                if (a.Rate > 0) newRate = Math.Max(double.Epsilon, newRate);
                if (!double.IsPositiveInfinity(a.Rate)) newRate = Math.Min(double.MaxValue, newRate);
                return GammaPower.FromShapeAndRate(a.Shape, newRate, a.Power);
            }
            else if (!a.IsProper()) return a;
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
                double newMean = Math.Max(0, a.GetMean() + b);
                if (b < 0 && false)
                {
                    newMean = Math.Max(0, TruncatedGammaPowerGetMean(a, -b, double.PositiveInfinity) + b);
                }
                double newLogMean = Math.Log(newMean);
                // If logShape is big, this difference can lose accuracy
                // Find newLogRate to satisfy logShape - newLogRate <= newLogMean/a.Power
                double newLogRate;
                if (a.Power < 0)
                {
                    double newLogMeanOverPower = MMath.LargestDoubleRatio(newLogMean, -a.Power);
                    newLogRate = MMath.LargestDoubleSum(logShape, newLogMeanOverPower);
                }
                else
                {
                    double newLogMeanOverPower = MMath.LargestDoubleRatio(newLogMean, a.Power);
                    newLogRate = -MMath.LargestDoubleSum(-logShape, newLogMeanOverPower);
                }
                // check: (logShape - newLogRate)*a.Power <= newLogMean
                if ((double)((logShape - newLogRate) * a.Power) > newLogMean) throw new Exception();
                double newRate = Math.Exp(newLogRate);
                newRate = Math.Max(double.Epsilon, newRate);
                if (!double.IsPositiveInfinity(a.Rate)) newRate = Math.Min(double.MaxValue, newRate);
                return GammaPower.FromShapeAndRate(a.Shape, newRate, a.Power);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaOp"]/message_doc[@name="SumAverageConditional(double, GammaPower)"]/*'/>
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaVmpOp"]/message_doc[@name="SumAverageLogarithm(GammaPower, GammaPower, GammaPower)"]/*'/>
        public static GammaPower SumAverageLogarithm([SkipIfUniform] GammaPower a, [SkipIfUniform] GammaPower b, GammaPower result)
        {
            if (a.IsUniform() || b.IsUniform())
            {
                result.SetToUniform();
                return result;
            }
            if (result.Power == -1)
            {
                bool aHasInfiniteMean = (a.Power == -1 && a.Shape <= 1);
                bool bHasInfiniteMean = (b.Power == -1 && b.Shape <= 1);
                if (aHasInfiniteMean)
                {
                    if (bHasInfiniteMean)
                    {
                        return InvGammaFromShapeAndMeanInverse(Math.Min(a.Shape, b.Shape), MeanInverseOfSum(a, b));
                    }
                    else return InvGammaFromShapeAndMeanInverse(a.Shape, MeanInverseOfSum(a, b));
                }
                else if (bHasInfiniteMean) return InvGammaFromShapeAndMeanInverse(b.Shape, MeanInverseOfSum(a, b));
            }
            a.GetMeanAndVariance(out double aMean, out double aVariance);
            b.GetMeanAndVariance(out double bMean, out double bVariance);
            double mean = aMean + bMean;
            double variance = aVariance + bVariance;
            if (result.Power == -1 && variance > double.MaxValue && false)
            {
                // mean is finite
                return InvGammaFromMeanAndMeanInverse(mean, MeanInverseOfSum(a, b));
            }
            return GammaPower.FromMeanAndVariance(mean, variance, result.Power);
        }

        public static double MeanInverseOfSum(GammaPower a, GammaPower b)
        {
            double aMeanReciprocal = a.GetMeanPower(-1);
            double bMeanReciprocal = b.GetMeanPower(-1);
            // 1/(a+b) = 1/(q*a/q + (1-q)*b/(1-q)) 
            //         <= q/(a/q) + (1-q)/(b/(1-q)) 
            //         = q^2/a + (1-q)^2/b
            // d/dq = 2*q*E[1/a] - 2*(1-q)*E[1/b] = 0
            // optimal q = E[1/b]/(E[1/a] + E[1/b])
            // which gives (E[1/b]^2*E[1/a] + E[1/a]^2*E[1/b])/(E[1/a] + E[1/b])^2 = E[1/a]*E[1/b]/(E[1/a] + E[1/b]) = 1/(1/E[1/a] + 1/E[1/b])
            double upperBound = 1 / (1 / aMeanReciprocal + 1 / bMeanReciprocal);
            return upperBound;
        }

        public static GammaPower InvGammaFromShapeAndMeanInverse(double shape, double meanInverse)
        {
            return GammaPower.FromShapeAndRate(shape, shape / meanInverse, -1);
        }

        public static GammaPower InvGammaFromMeanAndMeanInverse(double mean, double meanInverse)
        {
            double shape = 1 / (1 - 1 / mean / meanInverse);
            double rate = mean * (shape - 1);
            return GammaPower.FromShapeAndRate(shape, rate, -1);
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
            // If sum is a point mass, then the message to A is pointmass(s*q)
            double x = sum.Shape - 1;
            double ma = Math.Exp(a.GetMeanLog());
            double mb = Math.Exp(b.GetMeanLog());
            double p = ma / (ma + mb);
            if (sum.IsPointMass) return Gamma.PointMass(sum.Point * p);
            double m = x * p;
            return Gamma.FromShapeAndRate(m + 1, sum.Rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaVmpOp"]/message_doc[@name="AAverageLogarithm(GammaPower, GammaPower, GammaPower)"]/*'/>
        public static GammaPower AAverageLogarithm([SkipIfUniform] GammaPower sum, [Proper] GammaPower a, [Proper] GammaPower b, GammaPower to_a, GammaPower to_b)
        {
            // f = int_sum sum^(ss/c -1)*exp(-sr*sum^(1/c))*delta(sum = a+b) dsum
            //   = (a+b)^(ss/c -1)*exp(-sr*(a+b)^(1/c))
            // log(f) = (ss/c -1)*log(a+b) - sr*(a+b)^(1/c)
            // apply a lower bound:
            // log(a+b) >= p*log(a) + (1-p)*log(b) - p*log(p) - (1-p)*log(1-p)
            // optimal p = exp(a)/(exp(a)+exp(b)) if (a,b) are fixed
            // optimal p = exp(E[log(a)])/(exp(E[log(a)])+exp(E[log(b)]))  if (a,b) are random
            // This generalizes the bound used by Cemgil (2008).
            // (a+b)^(1/c) = (a*q/q + b*(1-q)/(1-q))^(1/c) 
            //             <= q*(a/q)^(1/c) + (1-q)*(b/(1-q))^(1/c)  if c < 0
            //             = q^(1-1/c)*a^(1/c) + (1-q)^(1-1/c)*b^(1/c)
            // d/dq = (1-1/c)*(q^(-1/c)*a^(1/c) - (1-q)^(-1/c)*b^(1/c))
            // optimal q = a/(a + b) if (a,b) are fixed
            // optimal q = E[a^(1/c)]^c/(E[a^(1/c)]^c + E[b^(1/c)]^c) if (a,b) are random
            // The message to A has shape (ss-c)*p + c and rate sr*q^(1-1/c).
            // If sum is a point mass, then the message to A is pointmass(s*p^c*q^(1-c))
            if (a.Power != sum.Power) throw new NotSupportedException($"a.Power ({a.Power}) != sum.Power ({sum.Power})");
            double x = sum.Shape - sum.Power;
            GammaPower aPost = a * to_a;
            if (aPost.IsUniform())
            {
                return sum;
            }
            GammaPower bPost = b * to_b;
            if (bPost.IsUniform())
            {
                if (sum.IsPointMass) return sum;
                return GammaPower.FromShapeAndRate(sum.Power, sum.Rate, sum.Power);
            }
            double ma = Math.Exp(aPost.GetMeanLog());
            double mb = Math.Exp(bPost.GetMeanLog());
            double denom = ma + mb;
            double p = (denom == 0) ? 0.5 : ma / (ma + mb);
            double m = x * p;
            double mac = Math.Exp(a.Power * aPost.GetLogMeanPower(1 / a.Power));
            double mbc = Math.Exp(b.Power * bPost.GetLogMeanPower(1 / b.Power));
            double denom2 = mac + mbc;
            double q = (denom2 == 0) ? 0.5 : mac / (mac + mbc);
            if (sum.IsPointMass) return GammaPower.PointMass(sum.Point * Math.Pow(p / q, sum.Power) * q, sum.Power);
            return GammaPower.FromShapeAndRate(m + sum.Power, sum.Rate * Math.Pow(q, 1 - 1 / sum.Power), sum.Power);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaVmpOp"]/message_doc[@name="BAverageLogarithm(Gamma, Gamma, Gamma)"]/*'/>
        public static Gamma BAverageLogarithm([SkipIfUniform] Gamma sum, [Proper] Gamma a, [Proper] Gamma b)
        {
            return AAverageLogarithm(sum, b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="PlusGammaVmpOp"]/message_doc[@name="BAverageLogarithm(GammaPower, GammaPower, GammaPower)"]/*'/>
        public static GammaPower BAverageLogarithm([SkipIfUniform] GammaPower sum, [Proper] GammaPower a, [Proper] GammaPower b, GammaPower to_a, GammaPower to_b)
        {
            return AAverageLogarithm(sum, b, a, to_b, to_a);
        }
    }
}
