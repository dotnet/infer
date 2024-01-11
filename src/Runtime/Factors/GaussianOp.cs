// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Distributions;
    using Math;
    using Attributes;
    using Utilities;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/doc/*'/>
    public class GaussianOpBase
    {
        //-- Easy cases ----------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="SampleAverageConditional(double, double)"]/*'/>
        public static Gaussian SampleAverageConditional(double mean, double precision)
        {
            return Gaussian.FromMeanAndPrecision(mean, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="MeanAverageConditional(double, double)"]/*'/>
        public static Gaussian MeanAverageConditional(double sample, double precision)
        {
            return SampleAverageConditional(sample, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="PrecisionAverageConditional(double, double)"]/*'/>
        public static Gamma PrecisionAverageConditional(double sample, double mean)
        {
            double diff = sample - mean;
            return Gamma.FromShapeAndRate(1.5, 0.5 * diff * diff);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="SampleAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian SampleAverageConditional([SkipIfUniform] Gaussian mean, double precision)
        {
            if (mean.IsPointMass)
                return SampleAverageConditional(mean.Point, precision);
            // if (precision < 0) throw new ArgumentException("The constant precision given to the Gaussian factor is negative", "precision");
            if (precision == 0)
            {
                return Gaussian.Uniform();
            }
            else if (double.IsPositiveInfinity(precision))
            {
                return mean;
            }
            else
            {
                if (mean.Precision <= -precision)
                    throw new ImproperMessageException(mean);
                // The formula is int_mean N(x;mean,1/prec) p(mean) = N(x; mm, mv + 1/prec)
                // sample.Precision = inv(mv + inv(prec)) = mprec*prec/(prec + mprec)
                // sample.MeanTimesPrecision = sample.Precision*mm = R*(mprec*mm)
                // R = Prec/(Prec + mean.Prec)
                // This code works for mean.IsUniform() since then mean.Precision = 0, mean.MeanTimesPrecision = 0
                Gaussian result = new Gaussian();
                double R = precision / (precision + mean.Precision);
                result.Precision = R * mean.Precision;
                result.MeanTimesPrecision = R * mean.MeanTimesPrecision;
                return result;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="MeanAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian MeanAverageConditional([SkipIfUniform] Gaussian sample, double precision)
        {
            return SampleAverageConditional(sample, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="LogAverageFactor(double, double, double)"]/*'/>
        public static double LogAverageFactor(double sample, double mean, double precision)
        {
            return Gaussian.GetLogProb(sample, mean, 1.0 / precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, double)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, double precision)
        {
            return GaussianFromMeanAndVarianceOp.LogAverageFactor(sample, mean, 1.0 / precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="LogAverageFactor(Gaussian, double, double)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gaussian sample, double mean, double precision)
        {
            return LogAverageFactor(sample, Gaussian.PointMass(mean), precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="LogAverageFactor(double, Gaussian, double)"]/*'/>
        public static double LogAverageFactor(double sample, [SkipIfUniform] Gaussian mean, double precision)
        {
            //if(mean.IsPointMass) return LogAverageFactor(sample,mean.Point,precision);
            return LogAverageFactor(Gaussian.PointMass(sample), mean, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="LogAverageFactor(double, double, Gamma)"]/*'/>
        public static double LogAverageFactor(double sample, double mean, [SkipIfUniform] Gamma precision)
        {
            if (precision.IsPointMass)
                return LogAverageFactor(sample, mean, precision.Point);
            if (precision.IsUniform())
                return Double.PositiveInfinity;
            return TPdfLn(sample - mean, 2 * precision.Rate, 2 * precision.Shape + 1);
        }

        /// <summary>
        /// Logarithm of Student T density.
        /// </summary>
        /// <param name="x">sample</param>
        /// <param name="v">variance parameter</param>
        /// <param name="n">degrees of freedom plus 1</param>
        /// <returns></returns>
        public static double TPdfLn(double x, double v, double n)
        {
            return MMath.GammaLn(n * 0.5) - MMath.GammaLn((n - 1) * 0.5) - 0.5 * Math.Log(v * Math.PI) - 0.5 * n * Math.Log(1 + x * x / v);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="LogEvidenceRatio(double, double, double)"]/*'/>
        public static double LogEvidenceRatio(double sample, double mean, double precision)
        {
            return LogAverageFactor(sample, mean, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gaussian sample, Gaussian mean, double precision)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="LogEvidenceRatio(Gaussian, double, double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gaussian sample, double mean, double precision)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="LogEvidenceRatio(double, Gaussian, double)"]/*'/>
        public static double LogEvidenceRatio(double sample, [SkipIfUniform] Gaussian mean, double precision)
        {
            return LogAverageFactor(sample, mean, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOpBase"]/message_doc[@name="LogEvidenceRatio(double, double, Gamma)"]/*'/>
        public static double LogEvidenceRatio(double sample, double mean, [SkipIfUniform] Gamma precision)
        {
            return LogAverageFactor(sample, mean, precision);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/doc/*'/>
    [FactorMethod(typeof(Gaussian), "Sample", typeof(double), typeof(double), Default = true)]
    [FactorMethod(new string[] { "sample", "mean", "precision" }, typeof(Factor), "Gaussian", Default = true)]
    [Quality(QualityBand.Mature)]
    public class GaussianOp : GaussianOpBase
    {
        //-- TruncatedGaussian ---------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="SampleAverageConditional(double, double, TruncatedGaussian)"]/*'/>
        public static TruncatedGaussian SampleAverageConditional(double mean, double precision, TruncatedGaussian result)
        {
            return TruncatedGaussian.FromGaussian(Gaussian.FromMeanAndPrecision(mean, precision));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="MeanAverageConditional(double, double, TruncatedGaussian)"]/*'/>
        public static TruncatedGaussian MeanAverageConditional(double sample, double precision, TruncatedGaussian result)
        {
            return SampleAverageConditional(sample, precision, result);
        }

        //-- Gibbs ---------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="LogFactorValue(double, double, double)"]/*'/>
        public static double LogFactorValue(double sample, double mean, double precision)
        {
            return Gaussian.GetLogProb(sample, mean, 1.0 / precision);
        }

        //-- EP ------------------------------------------------------------------------------------------------
        /// <summary>
        /// Static flag to force a proper distribution
        /// </summary>
        public static bool ForceProper = true;

        /// <summary>
        /// Number of quadrature nodes to use for computing the messages.
        /// Reduce this number to save time in exchange for less accuracy.
        /// </summary>
        public static int QuadratureNodeCount = 50;

        public static bool modified = false;

        /// <summary>
        /// Compute an EP message
        /// </summary>
        /// <param name="prior"></param>
        /// <param name="alpha">dlogZ/dm0</param>
        /// <param name="beta">ddlogZ/dm0^2</param>
        /// <param name="forceProper"></param>
        /// <returns></returns>
        public static Gaussian GaussianFromAlphaBeta(Gaussian prior, double alpha, double beta, bool forceProper)
        {
            if (prior.IsPointMass)
                return Gaussian.FromDerivatives(prior.Point, alpha, -beta, forceProper);
            double prec = prior.Precision;
            if (prec == beta && prec != 0)
            {
                return Gaussian.PointMass((prior.MeanTimesPrecision + alpha) / prec);
            }
            double tau = prior.MeanTimesPrecision;
            double weight = beta / (prec - beta);
            if (forceProper && weight < 0)
                weight = 0;
            if (prec == 0)
                weight = 0;
            // eq (31) in EP quickref; same as inv(inv(beta)-inv(prec))
            double resultPrecision = prec * weight;
            // eq (30) in EP quickref times above and simplified
            double resultMeanTimesPrecision = weight * (tau + alpha) + alpha;
            if (double.IsNaN(resultPrecision) || double.IsNaN(resultMeanTimesPrecision))
                throw new InferRuntimeException($"result is NaN.  prior={prior}, alpha={alpha}, beta={beta}, forceProper={forceProper}");
            return Gaussian.FromNatural(resultMeanTimesPrecision, resultPrecision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="SampleAverageConditional(Gaussian, Gaussian, Gamma, Gamma)"]/*'/>
        public static Gaussian SampleAverageConditional([NoInit] Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision, [NoInit] Gamma to_precision)
        {
            if (sample.Precision == 0 && precision.Shape <= 1.0)
                sample = Gaussian.FromNatural(1e-20, 1e-20);
            if (precision.IsPointMass)
            {
                return SampleAverageConditional(mean, precision.Point);
            }
            else if (sample.Precision == 0)
            {
                // for large vx, Z =approx N(mx; mm, vx+vm+E[1/prec])
                if (mean.Precision == 0) return mean;
                double mm, mv;
                mean.GetMeanAndVariance(out mm, out mv);
                // NOTE: this error may happen because sample didn't receive any message yet under the schedule.
                // Need to make the scheduler smarter to avoid this.
                if (precision.Shape <= 1.0)
                    throw new ArgumentException("The posterior has infinite variance due to precision distributed as " + precision +
                                                " (shape <= 1).  Try using a different prior for the precision, with shape > 1.");
                return Gaussian.FromMeanAndVariance(mm, mv + precision.GetMeanInverse());
            }
            else if (mean.IsUniform() /*|| precision.IsUniform() */)
            {
                return Gaussian.Uniform();
            }
            else if (!precision.IsProper())
            {
                throw new ImproperMessageException(precision);
            }
            else if (sample.IsPointMass && mean.IsPointMass)
            {
                // f(x) = int_r N(x;m,1/r) p(r) dr
                //      = 1/(rr + (x-m)^2/2)^(rs+0.5)
                // log f = -(rs+0.5)*log(rr + (x-m)^2/2)
                // dlogf = -(rs+0.5)*(x-m)/(rr + (x-m)^2/2)
                // ddlogf = -(rs+0.5)/(rr + (x-m)^2/2) + (rs+0.5)*(x-m)^2/(rr + (x-m)^2/2)^2
                double x = sample.Point;
                double m = mean.Point;
                double diff = x - m;
                double s = precision.Shape + 0.5;
                double denom = precision.Rate + diff * diff * 0.5;
                double ddenom = diff / denom;
                double dlogf = -s * ddenom;
                double ddlogf = -s / denom + s * ddenom * ddenom;
                return Gaussian.FromDerivatives(x, dlogf, ddlogf, ForceProper);
            }
            else
            {
                // The formula is int_prec int_mean N(x;mean,1/prec) p(x) p(mean) p(prec) =
                // int_prec N(x; mm, mv + 1/prec) p(x) p(prec) =
                // int_prec N(x; new xm, new xv) N(xm; mm, mv + xv + 1/prec) p(prec)
                // Let R = Prec/(Prec + mean.Prec)
                // new xv = inv(R*mean.Prec + sample.Prec)
                // new xm = xv*(R*mean.PM + sample.PM)
                //if (sample.Precision <= 0 || mean.Precision <= 0) return Gaussian.Uniform();

                // In the case where sample and mean are improper distributions, 
                // we must only consider values of prec for which (new xv > 0).
                // This happens when R*mean.Prec > -sample.Prec
                // As a function of Prec, R*mean.Prec has a singularity at Prec=-mean.Prec
                // This function is greater than a threshold when Prec is sufficiently small or sufficiently large.
                // Therefore we construct an interval of Precs to exclude from the integration.
                double xm, xv, mm, mv;
                sample.GetMeanAndVarianceImproper(out xm, out xv);
                mean.GetMeanAndVarianceImproper(out mm, out mv);
                double v = xv + mv;
                double lowerBound = 0;
                double upperBound = Double.PositiveInfinity;
                bool precisionIsBetween;
                if (mean.Precision >= 0)
                {
                    if (sample.Precision < -mean.Precision)
                        throw new ImproperMessageException(sample);
                    precisionIsBetween = true;
                    //lowerBound = -mean.Precision * sample.Precision / (mean.Precision + sample.Precision);
                    lowerBound = -1.0 / v;
                }
                else // mean.Precision < 0
                {
                    if (sample.Precision < 0)
                    {
                        precisionIsBetween = true;
                        lowerBound = -1.0 / v;
                        upperBound = -mean.Precision;
                    }
                    else if (sample.Precision < -mean.Precision)
                    {
                        precisionIsBetween = true;
                        lowerBound = 0;
                        upperBound = -mean.Precision;
                    }
                    else // sample.Precision >= -mean.Precision > 0
                    {
                        // mv < v < 0
                        // 0 < -v < -mv
                        // we want 1/(mv + 1/prec) > -sample.Precision
                        // If mv + 1/prec > 0 (1/prec > -mv) then -xv < mv + 1/prec, 1/prec > -v, prec < -1/v, prec < -1/mv (latter is stronger)
                        // If mv + 1/prec < 0 (1/prec < -mv) then -xv > mv + 1/prec, 1/prec < -v, prec > -1/v, prec > -1/mv (former is stronger)
                        // Therefore the integration region is (prec < -1/mv) and (prec > -1/v).
                        // in this case, the precision should NOT be in this interval.
                        precisionIsBetween = false;
                        lowerBound = -mean.Precision;
                        upperBound = -1.0 / v;
                    }
                }
                double[] nodes = new double[QuadratureNodeCount];
                double[] logWeights = new double[nodes.Length];
                Gamma precMarginal = precision * to_precision;
                if (precMarginal.IsPointMass) return SampleAverageConditional(mean, precMarginal.Point);
                precMarginal = GaussianOp_Laplace.Q(sample, mean, precision, precMarginal);
                QuadratureNodesAndWeights(precMarginal, nodes, logWeights);
                if (!to_precision.IsUniform())
                {
                    // modify the weights
                    for (int i = 0; i < logWeights.Length; i++)
                    {
                        logWeights[i] += precision.GetLogProb(nodes[i]) - precMarginal.GetLogProb(nodes[i]);
                    }
                }
                if (xv < 1 && mean.Precision > 0 && !modified)
                {
                    // Compute the message directly
                    // f(m) = int_r N(0;m,v+1/r) Ga(r;a,b) dr
                    // df/dm = int_r -m/(v+1/r) N(0;m,v+1/r) Ga(r;a,b) dr
                    // ddf/dm^2 = int_r (-1/(v+1/r) +m^2/(v+1/r)^2) N(0;m,v+1/r) Ga(r;a,b) dr
                    // This approach works even if sample.IsPointMass
                    double Z = 0;
                    double sum1 = 0;
                    double sum2 = 0;
                    double m = xm - mm;
                    double shift = 0;
                    for (int i = 0; i < nodes.Length; i++)
                    {
                        double prec = nodes[i];
                        Assert.IsTrue(prec > 0);
                        if ((prec > lowerBound && prec < upperBound) != precisionIsBetween)
                            continue;
                        double ivr = prec / (v * prec + 1);
                        double dlogf = -m * ivr;
                        double ddlogf = dlogf * dlogf - ivr;
                        double lp = 0.5 * Math.Log(ivr) - 0.5 * m * m * ivr;
                        if (double.IsPositiveInfinity(lp))
                            throw new Exception("overflow");
                        if (!double.IsNegativeInfinity(lp) && shift == 0)
                            shift = logWeights[i] + lp;
                        double f = Math.Exp(logWeights[i] + lp - shift);
                        Z += f;
                        sum1 += dlogf * f;
                        sum2 += ddlogf * f;
                    }
                    if (double.IsInfinity(Z))
                        throw new Exception("Z is infinite");
                    if (double.IsNaN(Z))
                        throw new Exception("Z is nan");
                    if (Z == 0.0)
                    {
                        throw new Exception("Quadrature found zero mass");
                    }
                    double alpha = sum1 / Z;
                    double beta = alpha * alpha - sum2 / Z;
                    return GaussianOp.GaussianFromAlphaBeta(sample, alpha, beta, GaussianOp.ForceProper);
                }
                else
                {
                    // Compute the marginal and use SetToRatio
                    GaussianEstimator est = new GaussianEstimator();
                    double shift = 0;
                    for (int i = 0; i < nodes.Length; i++)
                    {
                        double newVar, newMean;
                        double prec = nodes[i];
                        Assert.IsTrue(prec > 0);
                        if ((prec > lowerBound && prec < upperBound) != precisionIsBetween)
                            continue;
                        // the following works even if sample is uniform. (sample.Precision == 0)
                        if (mean.IsPointMass)
                        {
                            // take limit mean.Precision -> Inf
                            newVar = 1.0 / (prec + sample.Precision);
                            newMean = newVar * (prec * mean.Point + sample.MeanTimesPrecision);
                        }
                        else
                        {
                            // mean.Precision < Inf
                            double R = prec / (prec + mean.Precision);
                            newVar = 1.0 / (R * mean.Precision + sample.Precision);
                            newMean = newVar * (R * mean.MeanTimesPrecision + sample.MeanTimesPrecision);
                        }
                        double lp = Gaussian.GetLogProb(xm, mm, xv + mv + 1.0 / prec);
                        if (i == 0)
                            shift = logWeights[i] + lp;
                        double f = Math.Exp(logWeights[i] + lp - shift);
                        est.Add(Gaussian.FromMeanAndVariance(newMean, newVar), f);
                    }
                    double Z = est.mva.Count;
                    if (double.IsNaN(Z))
                        throw new Exception("Z is nan");
                    if (Z == 0.0)
                    {
                        throw new Exception("Quadrature found zero mass");
                    }
                    Gaussian result = est.GetDistribution(new Gaussian());
                    if (modified && !sample.IsUniform())
                    {
                        // heuristic method to avoid improper messages:
                        // the message's mean must be E[mean] (regardless of context) and the variance is chosen to match the posterior mean when multiplied by context
                        double sampleMean = result.GetMean();
                        if (sampleMean != mm)
                        {
                            result.Precision = (sample.MeanTimesPrecision - sampleMean * sample.Precision) / (sampleMean - mm);
                            if (result.Precision < 0)
                                throw new Exception("internal: sampleMean is not between sample.Mean and mean.Mean");
                            result.MeanTimesPrecision = result.Precision * mm;
                        }
                    }
                    else
                    {
                        if (result.IsPointMass)
                            throw new Exception("Quadrature found zero variance");
                        result.SetToRatio(result, sample, ForceProper);
                    }
                    return result;
                }
            }
        }

        public static Gaussian SampleAverageConditional_slow(Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision)
        {
            if (precision.IsUniform())
                return Gaussian.Uniform();
            Gamma to_precision = PrecisionAverageConditional_slow(sample, mean, precision);
            return SampleAverageConditional(sample, mean, precision, to_precision);
        }

        public static Gaussian MeanAverageConditional_slow([SkipIfUniform] Gaussian sample, Gaussian mean, [SkipIfUniform] Gamma precision)
        {
            return SampleAverageConditional_slow(mean, sample, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="SampleAverageConditional(Gaussian, double, Gamma, Gamma)"]/*'/>
        public static Gaussian SampleAverageConditional([NoInit] Gaussian sample, double mean, [SkipIfUniform] Gamma precision, [NoInit] Gamma to_precision)
        {
            return SampleAverageConditional(sample, Gaussian.PointMass(mean), precision, to_precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="SampleAverageConditionalInit()"]/*'/>
        [Skip]
        public static Gaussian SampleAverageConditionalInit()
        {
            return Gaussian.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="MeanAverageConditional(Gaussian, Gaussian, Gamma, Gamma)"]/*'/>
        public static Gaussian MeanAverageConditional([SkipIfUniform] Gaussian sample, [NoInit] Gaussian mean, [SkipIfUniform] Gamma precision, [NoInit] Gamma to_precision)
        {
            return SampleAverageConditional(mean, sample, precision, to_precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="MeanAverageConditional(double, Gaussian, Gamma, Gamma)"]/*'/>
        public static Gaussian MeanAverageConditional(double sample, [NoInit] Gaussian mean, [SkipIfUniform] Gamma precision, [NoInit] Gamma to_precision)
        {
            return SampleAverageConditional(mean, sample, precision, to_precision);
        }

        public static Gamma PrecisionAverageConditional_slow([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision)
        {
            return PrecisionAverageConditional(sample, mean, precision);
        }

        public static Gamma PrecisionAverageConditional_Point(double ym, double yv, double precision)
        {
            // xdlogf = 0.5*(1/(yv*r+1) - r*ym*ym/(yv*r+1)^2) = 0.5*(yv*r+1-r*ym*ym)/(yv*r+1)^2
            // x2ddlogf = (-yv*r-0.5+r*ym*ym)/(yv*r+1)^2 - r*ym*ym/(yv*r+1)^3
            // xdlogf + x2ddlogf = (-0.5*yv*r + 0.5*r*ym*ym)/(yv*r+1)^2 - r*ym*ym/(yv*r+1)^3
            // as r->0: -0.5*r*(yv + ym*ym)
            if (precision == 0 || yv == 0) return Gamma.FromShapeAndRate(1.5, 0.5 * (yv + ym * ym));
            // point mass case
            // f(r) = N(xm;mm, xv+mv+1/r)
            // log f(r) = -0.5*log(xv+mv+1/r) - 0.5*(xm-mm)^2/(xv+mv+1/r)
            // (log f)' = (-0.5/(yv + 1/r) + 0.5*ym^2/(yv+1/r)^2)*(-1/r^2)
            // (log f)'' = (-0.5/(yv + 1/r) + 0.5*ym^2/(yv+1/r)^2)*(2/r^3) + (0.5/(yv+1/r)^2 - ym^2/(yv+1/r)^3)*(1/r^4)
            // r (log f)' = 0.5/(yv*r + 1) - 0.5*r*ym^2/(yv*r+1)^2
            // r^2 (log f)'' = -1/(yv*r + 1) + r*ym^2/(yv*r+1)^2) + 0.5/(yv*r+1)^2 - r*ym^2/(yv*r+1)^3
            double vdenom = 1 / (yv * precision + 1);
            double ymvdenom = ym * vdenom;
            double ymvdenom2 = (precision > double.MaxValue) ? 0 : (precision * ymvdenom * ymvdenom);
            //dlogf = (-0.5 * denom + 0.5 * ym2denom2) * (-v2);
            //dlogf = 0.5 * (1 - ym * ymdenom) * denom * v2;
            //dlogf = 0.5 * (v - ym * ym/(yv*precision+1))/(yv*precision + 1);
            //double dlogf = 0.5 * (v * vdenom - ymvdenom2);
            //double xdlogf = precision * dlogf;
            double xdlogf = 0.5 * (vdenom - ymvdenom2);
            //double ddlogf = dlogf * (-2 * v) + (0.5 * denom - ym2denom2) * denom * v2 * v2;
            //double ddlogf = v * (-2 * dlogf + (0.5 * v * vdenom - ym*ym*vdenom*vdenom) * vdenom);
            double x2ddlogf = -2 * xdlogf + (0.5 * vdenom - ymvdenom2) * vdenom;
            bool checkDerivatives = false;
            if (checkDerivatives)
            {
                double delta = precision * 1e-6;
                double logf = Gaussian.GetLogProb(0, ym, yv + 1 / precision);
                double logfd = Gaussian.GetLogProb(0, ym, yv + 1 / (precision + delta));
                double logfd2 = Gaussian.GetLogProb(0, ym, yv + 1 / (precision - delta));
                double dlogf2 = (logfd - logfd2) / (2 * delta);
                double ddlogf2 = (logfd + logfd2 - 2 * logf) / (delta * delta);
                double ulp = MMath.Ulp(logf);
                if (logfd - logf > ulp && logf - logfd2 > ulp)
                {
                    double v = 1 / precision;
                    double dlogf = v * xdlogf;
                    double ddlogf = v * v * x2ddlogf;
                    Console.WriteLine($"dlogf={dlogf} check={dlogf2} ddlogf={ddlogf} check={ddlogf2}");
                    if (Math.Abs(dlogf2 - dlogf) > 1e-4) throw new Exception();
                    if (Math.Abs(ddlogf2 - ddlogf) > 1e-4) throw new Exception();
                }
            }
            return GammaFromDerivatives(precision, xdlogf, x2ddlogf, ForceProper);
        }

        /// <summary>
        /// Construct a Gamma distribution whose pdf has the given derivatives at a point.
        /// </summary>
        /// <param name="x">Must be positive</param>
        /// <param name="xdLogP">Desired derivative of log-density at x, times x</param>
        /// <param name="x2ddLogP">Desired second derivative of log-density at x, times x*x</param>
        /// <param name="forceProper">If true and both derivatives cannot be matched by a proper distribution, match only the first.</param>
        /// <returns></returns>
        internal static Gamma GammaFromDerivatives(double x, double xdLogP, double x2ddLogP, bool forceProper)
        {
            if (x <= 0)
                throw new ArgumentException("x <= 0");
            double a = -x2ddLogP;
            if (forceProper)
            {
                if (xdLogP < 0)
                {
                    if (a < 0)
                        a = 0;
                }
                else
                {
                    double amin = xdLogP;
                    if (a < amin)
                        a = amin;
                }
            }
            double b = (a - xdLogP) / x;
            if (forceProper)
            {
                // correct roundoff errors that might make b negative
                b = Math.Max(b, 0);
            }
            if (double.IsNaN(a) || double.IsNaN(b))
                throw new InferRuntimeException($"result is NaN.  x={x}, xdlogf={xdLogP}, x2ddlogf={x2ddLogP}");
            return Gamma.FromShapeAndRate(a + 1, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="PrecisionAverageConditional(Gaussian, Gaussian, Gamma)"]/*'/>
        public static Gamma PrecisionAverageConditional([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision)
        {
            if (sample.IsPointMass && mean.IsPointMass)
                return PrecisionAverageConditional(sample.Point, mean.Point);

            double xm, xv, mm, mv;
            sample.GetMeanAndVarianceImproper(out xm, out xv);
            mean.GetMeanAndVarianceImproper(out mm, out mv);
            double ym = xm - mm;
            double yv = xv + mv;

            Gamma result = new Gamma();
            if (double.IsPositiveInfinity(yv))
            {
                result.SetToUniform();
            }
            else if (precision.IsPointMass)
            {
                // must return a sensible value since precision could be initialized to a point mass.
                return PrecisionAverageConditional_Point(ym, yv, precision.Point);
            }
            else if (!precision.IsProper())
            {
                // improper prior
                throw new ImproperMessageException(precision);
            }
            else
            {
                // use quadrature to integrate over the precision
                // see LogAverageFactor
                double ym2 = ym * ym;
                double[] nodes = new double[QuadratureNodeCount];
                double[] logWeights = new double[nodes.Length];
                double[] logf = new double[nodes.Length];
                double shift = double.NegativeInfinity;
                double Z = 0;
                MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
                Gamma precMarginal = GaussianOp_Laplace.Q(sample, mean, precision, Gamma.Uniform());
                while (true)
                {
                    if (precMarginal.IsPointMass)
                        return PrecisionAverageConditional_Point(ym, yv, precMarginal.Point);
                    QuadratureNodesAndWeights(precMarginal, nodes, logWeights);
                    // modify the weights
                    for (int i = 0; i < logWeights.Length; i++)
                    {
                        logWeights[i] += precision.GetLogProb(nodes[i]) - precMarginal.GetLogProb(nodes[i]);
                    }
                    int argmax = -1;
                    shift = double.NegativeInfinity;
                    for (int i = 0; i < nodes.Length; i++)
                    {
                        double v = 1.0 / nodes[i] + yv;
                        if (v < 0)
                            continue;
                        double lp = -0.5 * Math.Log(v) - 0.5 * ym2 / v;
                        double logfi = logWeights[i] + lp;
                        if (logfi > shift)
                        {
                            shift = logfi;
                            argmax = i;
                        }
                        logf[i] = logfi;
                    }
                    mva.Clear();
                    for (int i = 0; i < nodes.Length; i++)
                    {
                        double f = Math.Exp(logf[i] - shift);
                        mva.Add(nodes[i], f);
                    }
                    Z = mva.Count;
                    if (double.IsNaN(Z))
                        throw new Exception("Z is nan");
                    if (Z > 2 && mva.Variance > 0)
                        break;
                    // one quadrature node dominates the answer.  must re-try with a different weight function.
                    double delta = (argmax == 0) ? (nodes[argmax + 1] - nodes[argmax]) : (nodes[argmax] - nodes[argmax - 1]);
                    precMarginal *= Gamma.FromMeanAndVariance(nodes[argmax], delta * delta);
                }
                if (ym2 - yv < precision.Rate)
                {
                    // This is a special approach that avoids SetToRatio, useful when precision.Rate is large.
                    // this approach is inaccurate if posterior mean << prior mean, since then alpha =approx -precision.Shape
                    // this happens when ym2-yv >> precision.Rate
                    // sum1 = Exf'
                    double sum1 = 0;
                    // sum2 = Ex^2f''
                    double sum2 = 0;
                    for (int i = 0; i < nodes.Length; i++)
                    {
                        double f = Math.Exp(logf[i] - shift);
                        if (nodes[i] * yv > 1)
                        {
                            double v = 1.0 / nodes[i];
                            double denom = 1 / (yv + v);
                            double denom2 = denom * denom;
                            double dlogf1 = -0.5 * denom + 0.5 * ym2 * denom2;
                            // dlogfr = r f'/f
                            double dlogfr = dlogf1 * (-v);
                            double dlogf2 = (0.5 - ym2 * denom) * denom2;
                            // ddfrr = r^2 f''/f
                            // f''/f = d(f'/f) + (f'/f)^2
                            double ddfrr = dlogfr * dlogfr + dlogf2 * v * v + (2 * v) * dlogf1;
                            sum1 += dlogfr * f;
                            sum2 += ddfrr * f;
                        }
                        else  // nodes[i] is small
                        {
                            double r = nodes[i];
                            double vdenom = 1 / (r * yv + 1);
                            double v2denom2 = vdenom * vdenom;
                            double vdlogf1 = -0.5 * vdenom + 0.5 * ym2 * v2denom2 * r;
                            // dlogfr = r f'/f
                            double dlogfr = -vdlogf1;
                            double v2dlogf2 = (0.5 - ym2 * vdenom * r) * v2denom2;
                            // ddfrr = r^2 f''/f
                            // f''/f = d(f'/f) + (f'/f)^2
                            double ddfrr = dlogfr * dlogfr + v2dlogf2 + 2 * vdlogf1;
                            sum1 += dlogfr * f;
                            sum2 += ddfrr * f;
                        }
                    }
                    double alpha = sum1 / Z;
                    double beta = (sum1 + sum2) / Z - alpha * alpha;
                    return GammaFromAlphaBeta(precision, alpha, beta, ForceProper);
                }
                else
                {
                    // SetToRatio method
                    double rmean = mva.Mean;
                    double rvariance = mva.Variance;
                    if (rvariance <= 0)
                        throw new Exception("Quadrature found zero variance");
                    if (Double.IsInfinity(rmean))
                    {
                        result.SetToUniform();
                    }
                    else
                    {
                        result.SetMeanAndVariance(rmean, rvariance);
                        result.SetToRatio(result, precision, ForceProper);
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// Gamma message computed directly from prior and expected derivatives of factor
        /// </summary>
        /// <param name="prior"></param>
        /// <param name="alpha">Exf'/Ef = -b dlogZ/db</param>
        /// <param name="beta">(Exf' + Ex^2f'')/Ef - alpha^2 = -b dalpha/db</param>
        /// <param name="forceProper"></param>
        /// <returns></returns>
        public static Gamma GammaFromAlphaBeta(Gamma prior, double alpha, double beta, bool forceProper)
        {
            double bv = (prior.Shape + alpha + beta) / prior.Rate;
            //if (bv <= 0)
            //    throw new Exception("Quadrature found zero variance");
            Gamma result = new Gamma();
            result.Rate = -beta / bv;
            // this is actually shape-1 until incremented below
            result.Shape = (prior.GetMean() * (alpha - beta) + alpha * alpha / prior.Rate) / bv;
            // same as result.Shape = (prior.Shape*(alpha - beta) + alpha*alpha)/(prior.Shape + alpha + beta)
            if (forceProper)
            {
                double rmean = (prior.Shape + alpha) / prior.Rate;
                if (rmean <= 0) return Gamma.PointMass(0);
                //Console.WriteLine("posterior mean = {0}", rmean);
                //Console.WriteLine("posterior variance = {0}", bv/prior.Rate);
                if (result.Rate < 0)
                {
                    if (result.Shape > 0)
                    {
                        // below is equivalent to: result.Rate = (prior.Shape + result.Shape) / rmean - prior.Rate;
                        result.Rate = (result.Shape - alpha) / rmean;
                    }
                }
                else if (result.Shape < 0)
                {
                    // Rate >= 0
                    // below is equivalent to: result.Shape = rmean * (prior.Rate + result.Rate) - prior.Shape;
                    result.Shape = alpha + rmean * result.Rate;
                }
                if (result.Shape < 0 || result.Rate < 0)
                {
                    if (alpha > 0)
                    {
                        // mean has increased
                        result.Rate = 0;
                        // below is equivalent to: result.Shape = rmean * prior.Rate - prior.Shape;
                        result.Shape = alpha;
                    }
                    else
                    {
                        // mean has decreased
                        result.Shape = 0;
                        // below is equivalent to: result.Rate = prior.Shape / rmean - prior.Rate;
                        result.Rate = -alpha / rmean;
                    }
                }
            }
            result.Shape++;
            return result;
        }

        /// <summary>
        /// Quadrature nodes for Gamma expectations
        /// </summary>
        /// <param name="precision">'precision' message</param>
        /// <param name="nodes">Place to put the nodes</param>
        /// <param name="logWeights">Place to put the log-weights</param>
        public static void QuadratureNodesAndWeights(Gamma precision, double[] nodes, double[] logWeights)
        {
            Quadrature.GammaNodesAndWeights(precision.Shape - 1, precision.Rate, nodes, logWeights);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="LogAverageFactor(double, Gaussian, Gamma, Gamma)"]/*'/>
        public static double LogAverageFactor(double sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision, Gamma to_precision)
        {
            if (mean.IsPointMass)
                return LogAverageFactor(sample, mean.Point, precision);
            if (precision.IsPointMass)
                return LogAverageFactor(sample, mean, precision.Point);
            if (precision.IsUniform())
                return Double.PositiveInfinity;
            return LogAverageFactor(Gaussian.PointMass(sample), mean, precision, to_precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="LogAverageFactor(Gaussian, double, Gamma, Gamma)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gaussian sample, double mean, [SkipIfUniform] Gamma precision, Gamma to_precision)
        {
            if (precision.IsPointMass)
                return LogAverageFactor(sample, mean, precision.Point);
            if (precision.IsUniform())
                return Double.PositiveInfinity;
            if (sample.IsPointMass)
                return LogAverageFactor(sample.Point, mean, precision);
            return LogAverageFactor(sample, Gaussian.PointMass(mean), precision, to_precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, Gamma, Gamma)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision, Gamma to_precision)
        {
            if (precision.IsPointMass)
                return LogAverageFactor(sample, mean, precision.Point);
            if (precision.IsUniform())
                return Double.PositiveInfinity;
            if (sample.IsPointMass && mean.IsPointMass)
                return LogAverageFactor(sample.Point, mean.Point, precision);
            if (sample.Precision == 0 || mean.Precision == 0)
                return 0.0;
            // this code works even if sample and mean are point masses, but not if any variable is uniform.
            double xm, xv, mm, mv;
            sample.GetMeanAndVariance(out xm, out xv);
            mean.GetMeanAndVariance(out mm, out mv);
            // use quadrature to integrate over the precision
            double[] nodes = new double[QuadratureNodeCount];
            double[] logWeights = new double[nodes.Length];
            Gamma precMarginal = precision * to_precision;
            QuadratureNodesAndWeights(precMarginal, nodes, logWeights);
            for (int i = 0; i < nodes.Length; i++)
            {
                logWeights[i] = logWeights[i] + Gaussian.GetLogProb(xm, mm, xv + mv + 1.0 / nodes[i]);
            }
            if (!to_precision.IsUniform())
            {
                // modify the weights
                for (int i = 0; i < logWeights.Length; i++)
                {
                    logWeights[i] += precision.GetLogProb(nodes[i]) - precMarginal.GetLogProb(nodes[i]);
                }
            }
            return MMath.LogSumExp(logWeights);
        }

        public static double LogAverageFactor_slow([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision)
        {
            if (precision.IsUniform())
                return Double.PositiveInfinity;
            Gamma to_precision = PrecisionAverageConditional_slow(sample, mean, precision);
            return LogAverageFactor(sample, mean, precision, to_precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="LogEvidenceRatio(double, Gaussian, Gamma, Gamma)"]/*'/>
        public static double LogEvidenceRatio(double sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision, Gamma to_precision)
        {
            return LogAverageFactor(sample, mean, precision, to_precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gamma, Gaussian, Gamma)"]/*'/>
        public static double LogEvidenceRatio(
            [SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision, [Fresh] Gaussian to_sample, Gamma to_precision)
        {
            if (precision.IsPointMass)
                return LogEvidenceRatio(sample, mean, precision.Point);
            //Gaussian to_Sample = SampleAverageConditional(sample, mean, precision);
            return LogAverageFactor(sample, mean, precision, to_precision)
              - sample.GetLogAverageOf(to_sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, double, Gamma, Gaussian, Gamma)"]/*'/>
        public static double LogEvidenceRatio(
            [SkipIfUniform] Gaussian sample, double mean, [SkipIfUniform] Gamma precision, [Fresh] Gaussian to_sample, Gamma to_precision)
        {
            if (precision.IsPointMass)
                return LogEvidenceRatio(sample, mean, precision.Point);
            //Gaussian to_Sample = SampleAverageConditional(sample, mean, precision);
            return LogAverageFactor(sample, mean, precision, to_precision)
              - sample.GetLogAverageOf(to_sample);
        }

        //-- VMP ------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="SampleAverageLogarithm(double, double)"]/*'/>
        public static Gaussian SampleAverageLogarithm(double mean, double precision)
        {
            return SampleAverageConditional(mean, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="SampleAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian SampleAverageLogarithm([Proper] Gaussian mean, double precision)
        {
            if (precision < 0.0)
                throw new ArgumentException("precision < 0 (" + precision + ")");
            Gaussian result = new Gaussian();
            result.Precision = precision;
            result.MeanTimesPrecision = precision * mean.GetMean();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="MeanAverageLogarithm(double, double)"]/*'/>
        public static Gaussian MeanAverageLogarithm(double sample, double precision)
        {
            return MeanAverageConditional(sample, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="MeanAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian MeanAverageLogarithm([Proper] Gaussian sample, double precision)
        {
            return SampleAverageLogarithm(sample, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="SampleAverageLogarithm(Gaussian, Gamma)"]/*'/>
        public static Gaussian SampleAverageLogarithm([Proper] Gaussian mean, [Proper] Gamma precision)
        {
            // The formula is exp(int_prec int_mean p(mean) p(prec) log N(x;mean,1/prec)) =
            // exp(-0.5 E[prec*(x-mean)^2] + const.) =
            // exp(-0.5 E[prec] (x^2 - 2 x E[mean]) + const.) =
            // N(x; E[mean], 1/E[prec])
            Gaussian result = new Gaussian();
            result.Precision = precision.GetMean();
            result.MeanTimesPrecision = result.Precision * mean.GetMean();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="MeanAverageLogarithm(Gaussian, Gamma)"]/*'/>
        public static Gaussian MeanAverageLogarithm([Proper] Gaussian sample, [Proper] Gamma precision)
        {
            return SampleAverageLogarithm(sample, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="SampleAverageLogarithm(double, Gamma)"]/*'/>
        public static Gaussian SampleAverageLogarithm(double mean, [Proper] Gamma precision)
        {
            Gaussian result = new Gaussian();
            result.Precision = precision.GetMean();
            result.MeanTimesPrecision = result.Precision * mean;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="MeanAverageLogarithm(double, Gamma)"]/*'/>
        public static Gaussian MeanAverageLogarithm(double sample, [Proper] Gamma precision)
        {
            return SampleAverageLogarithm(sample, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="PrecisionAverageLogarithm(double, double)"]/*'/>
        public static Gamma PrecisionAverageLogarithm(double sample, double mean)
        {
            return PrecisionAverageConditional(sample, mean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="PrecisionAverageLogarithm(Gaussian, Gaussian)"]/*'/>
        public static Gamma PrecisionAverageLogarithm([Proper] Gaussian sample, [Proper] Gaussian mean)
        {
            if (sample.IsUniform())
                throw new ImproperMessageException(sample);
            if (mean.IsUniform())
                throw new ImproperMessageException(mean);
            // The formula is exp(int_x int_mean p(x) p(mean) log N(x;mean,1/prec)) =
            // exp(-0.5 prec E[(x-mean)^2] + 0.5 log(prec)) =
            // Gamma(prec; 0.5, 0.5*E[(x-mean)^2])
            // E[(x-mean)^2] = E[x^2] - 2 E[x] E[mean] + E[mean^2] = var(x) + (E[x]-E[mean])^2 + var(mean)
            Gamma result = new Gamma();
            result.Shape = 1.5;
            double mx, vx, mm, vm;
            sample.GetMeanAndVariance(out mx, out vx);
            mean.GetMeanAndVariance(out mm, out vm);
            double diff = mx - mm;
            result.Rate = 0.5 * (vx + diff * diff + vm);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="PrecisionAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gamma PrecisionAverageLogarithm([Proper] Gaussian sample, double mean)
        {
            if (sample.IsUniform())
                throw new ImproperMessageException(sample);
            Gamma result = new Gamma();
            result.Shape = 1.5;
            double mx, vx;
            sample.GetMeanAndVariance(out mx, out vx);
            double diff = mx - mean;
            result.Rate = 0.5 * (vx + diff * diff);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="PrecisionAverageLogarithm(double, Gaussian)"]/*'/>
        public static Gamma PrecisionAverageLogarithm(double sample, [Proper] Gaussian mean)
        {
            return PrecisionAverageLogarithm(mean, sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="AverageLogFactor(Gaussian, Gaussian, Gamma)"]/*'/>
        public static double AverageLogFactor([Proper] Gaussian sample, [Proper] Gaussian mean, [Proper] Gamma precision)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, mean, precision);
            if (mean.IsPointMass)
                return AverageLogFactor(sample, mean.Point, precision);
            if (precision.IsPointMass)
                return AverageLogFactor(sample, mean, precision.Point);

            return ComputeAverageLogFactor(sample, mean, precision.GetMeanLog(), precision.GetMean());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="AverageLogFactor(double, double, Gamma)"]/*'/>
        public static double AverageLogFactor(double sample, double mean, [Proper] Gamma precision)
        {
            if (precision.IsPointMass)
                return AverageLogFactor(sample, mean, precision.Point);
            else
            {
                double diff = sample - mean;
                return -MMath.LnSqrt2PI + 0.5 * (precision.GetMeanLog() - precision.GetMean() * diff * diff);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="AverageLogFactor(double, double, double)"]/*'/>
        public static double AverageLogFactor(double sample, double mean, double precision)
        {
            double diff = sample - mean;
            if (double.IsPositiveInfinity(precision))
                return (diff == 0.0) ? 0.0 : double.NegativeInfinity;
            if (precision == 0.0)
                return 0.0;
            return -MMath.LnSqrt2PI + 0.5 * (Math.Log(precision) - precision * diff * diff);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="AverageLogFactor(Gaussian, double, double)"]/*'/>
        public static double AverageLogFactor([Proper] Gaussian sample, double mean, double precision)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, mean, precision);
            else if (double.IsPositiveInfinity(precision))
                return sample.GetLogProb(mean);
            else if (precision == 0.0)
                return 0.0;
            else
                return ComputeAverageLogFactor(sample, mean, Math.Log(precision), precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="AverageLogFactor(double, Gaussian, double)"]/*'/>
        public static double AverageLogFactor(double sample, [Proper] Gaussian mean, double precision)
        {
            return AverageLogFactor(mean, sample, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="AverageLogFactor(double, Gaussian, Gamma)"]/*'/>
        public static double AverageLogFactor(double sample, [Proper] Gaussian mean, [Proper] Gamma precision)
        {
            return AverageLogFactor(mean, sample, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="AverageLogFactor(Gaussian, double, Gamma)"]/*'/>
        public static double AverageLogFactor([Proper] Gaussian sample, double mean, [Proper] Gamma precision)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, mean, precision);
            if (precision.IsPointMass)
                return AverageLogFactor(sample, mean, precision.Point);

            return ComputeAverageLogFactor(sample, mean, precision.GetMeanLog(), precision.GetMean());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp"]/message_doc[@name="AverageLogFactor(Gaussian, Gaussian, double)"]/*'/>
        public static double AverageLogFactor([Proper] Gaussian sample, [Proper] Gaussian mean, double precision)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, mean, precision);
            else if (mean.IsPointMass)
                return AverageLogFactor(sample, mean.Point, precision);
            else if (double.IsPositiveInfinity(precision))
                return sample.GetLogAverageOf(mean);
            else if (precision == 0.0)
                return 0.0;
            else
                return ComputeAverageLogFactor(sample, mean, Math.Log(precision), precision);
        }

        /// <summary>
        /// Helper method for computing average log factor
        /// </summary>
        /// <param name="sample">Incoming message from 'sample'.</param>
        /// <param name="mean">Incoming message from 'mean'.</param>
        /// <param name="precision_Elogx">Expected log value of the incoming message from 'precision'</param>
        /// <param name="precision_Ex">Expected value of incoming message from 'precision'</param>
        /// <returns>Computed average log factor</returns>
        private static double ComputeAverageLogFactor(Gaussian sample, Gaussian mean, double precision_Elogx, double precision_Ex)
        {
            if (precision_Ex == 0.0)
                throw new ArgumentException("precision == 0");
            if (double.IsPositiveInfinity(precision_Ex))
                throw new ArgumentException("precision is infinite");
            double sampleMean, sampleVariance = 0;
            double meanMean, meanVariance = 0;
            sample.GetMeanAndVariance(out sampleMean, out sampleVariance);
            mean.GetMeanAndVariance(out meanMean, out meanVariance);
            double diff = sampleMean - meanMean;
            return -MMath.LnSqrt2PI + 0.5 * (precision_Elogx
                            - precision_Ex * (diff * diff + sampleVariance + meanVariance));
        }

        /// <summary>
        /// Helper method for computing average log factor
        /// </summary>
        /// <param name="sample">Incoming message from 'sample'.</param>
        /// <param name="mean">Constant value for 'mean'.</param>
        /// <param name="precision_Elogx">Expected log value of the incoming message from 'precision'</param>
        /// <param name="precision_Ex">Expected value of incoming message from 'precision'</param>
        /// <returns>Computed average log factor</returns>
        private static double ComputeAverageLogFactor(Gaussian sample, double mean, double precision_Elogx, double precision_Ex)
        {
            if (double.IsPositiveInfinity(precision_Ex))
                throw new ArgumentException("precision is infinite");
            double sampleMean, sampleVariance;
            sample.GetMeanAndVariance(out sampleMean, out sampleVariance);
            double diff = sampleMean - mean;
            return -MMath.LnSqrt2PI + 0.5 * (precision_Elogx
                            - precision_Ex * (diff * diff + sampleVariance));
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Slow"]/doc/*'/>
    [FactorMethod(typeof(Gaussian), "Sample", typeof(double), typeof(double), Default = false)]
    [FactorMethod(new string[] { "sample", "mean", "precision" }, typeof(Factor), "Gaussian", Default = false)]
    [Quality(QualityBand.Experimental)]
    public class GaussianOp_Slow : GaussianOpBase
    {
        public static int QuadratureNodeCount = 20000;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Slow"]/message_doc[@name="SampleAverageConditional(Gaussian, Gaussian, Gamma)"]/*'/>
        public static Gaussian SampleAverageConditional([NoInit] Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision)
        {
            if (sample.IsUniform() && precision.Shape <= 1.0)
                sample = Gaussian.FromNatural(1e-20, 1e-20);
            if (precision.IsPointMass)
            {
                return SampleAverageConditional(mean, precision.Point);
            }
            else if (sample.IsUniform())
            {
                // for large vx, Z =approx N(mx; mm, vx+vm+E[1/prec])
                double mm, mv;
                mean.GetMeanAndVariance(out mm, out mv);
                // NOTE: this error may happen because sample didn't receive any message yet under the schedule.
                // Need to make the scheduler smarter to avoid this.
                if (precision.Shape <= 1.0)
                    throw new ArgumentException("The posterior has infinite variance due to precision distributed as " + precision +
                                                " (shape <= 1).  Try using a different prior for the precision, with shape > 1.");
                return Gaussian.FromMeanAndVariance(mm, mv + precision.GetMeanInverse());
            }
            else if (mean.IsUniform() || precision.IsUniform())
            {
                return Gaussian.Uniform();
            }
            else if (!precision.IsProper())
            {
                throw new ImproperMessageException(precision);
            }
            else
            {
                double mx, vx;
                sample.GetMeanAndVariance(out mx, out vx);
                double mm, vm;
                mean.GetMeanAndVariance(out mm, out vm);
                double m = mx - mm;
                double v = vx + vm;
                if (double.IsPositiveInfinity(v))
                    return Gaussian.Uniform();
                double m2 = m * m;
                double a = precision.Shape;
                double b = precision.Rate;
                double logr0;
                double logrmin, logrmax;
                GetIntegrationBoundsForPrecision(m, v, a, b, out logrmin, out logrmax, out logr0);
                if (logrmin == logr0 || logrmax == logr0)
                    return SampleAverageConditional(mean, Math.Exp(logr0));
                int n = QuadratureNodeCount;
                double inc = (logrmax - logrmin) / (n - 1);
                if (vx < 1 && mean.Precision > 0)
                {
                    // Compute the message directly
                    // f(m) = int_r N(0;m,v+1/r) Ga(r;a,b) dr
                    // df/dm = int_r -m/(v+1/r) N(0;m,v+1/r) Ga(r;a,b) dr
                    // ddf/dm^2 = int_r (-1/(v+1/r) +m^2/(v+1/r)^2) N(0;m,v+1/r) Ga(r;a,b) dr
                    // This approach works even if sample.IsPointMass
                    double Z = 0;
                    double sum1 = 0;
                    double sum2 = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double logr = logrmin + i * inc;
                        double ivr = 1 / (v + Math.Exp(-logr));
                        double dlogf = -m * ivr;
                        double ddlogf = dlogf * dlogf - ivr;
                        double diff = LogLikelihoodRatio(logr, logr0, m, v, a, b);
                        if ((i == 0 || i == n - 1) && (diff > -49))
                            throw new InferRuntimeException("invalid integration bounds");
                        double f = Math.Exp(diff);
                        if (double.IsPositiveInfinity(f))
                        {
                            // this can happen if the likelihood is extremely sharp
                            //throw new Exception("overflow");
                            return SampleAverageConditional(mean, Math.Exp(logr0));
                        }
                        Z += f;
                        sum1 += dlogf * f;
                        sum2 += ddlogf * f;
                    }
                    double alpha = sum1 / Z;
                    double beta = alpha * alpha - sum2 / Z;
                    return GaussianOp.GaussianFromAlphaBeta(sample, alpha, beta, GaussianOp.ForceProper);
                }
                else
                {
                    // Compute the marginal and use SetToRatio
                    GaussianEstimator est = new GaussianEstimator();
                    for (int i = 0; i < n; i++)
                    {
                        double logr = logrmin + i * inc;
                        double prec = Math.Exp(logr);
                        double newVar, newMean;
                        // the following works even if sample is uniform. (sample.Precision == 0)
                        if (mean.IsPointMass)
                        {
                            // take limit mean.Precision -> Inf
                            newVar = 1.0 / (prec + sample.Precision);
                            newMean = newVar * (prec * mean.Point + sample.MeanTimesPrecision);
                        }
                        else
                        {
                            // mean.Precision < Inf
                            double R = prec / (prec + mean.Precision);
                            newVar = 1.0 / (R * mean.Precision + sample.Precision);
                            newMean = newVar * (R * mean.MeanTimesPrecision + sample.MeanTimesPrecision);
                        }
                        double diff = LogLikelihoodRatio(logr, logr0, m, v, a, b);
                        if ((i == 0 || i == n - 1) && ((diff > -49) || (diff < -51)))
                            throw new InferRuntimeException("invalid integration bounds");
                        double p = Math.Exp(diff);
                        if (double.IsPositiveInfinity(p))
                            throw new InferRuntimeException("overflow");
                        est.Add(Gaussian.FromMeanAndVariance(newMean, newVar), p);
                    }
                    if (est.mva.Count == 0)
                        throw new InferRuntimeException("Quadrature found zero mass");
                    if (double.IsNaN(est.mva.Count))
                        throw new InferRuntimeException("count is nan");
                    Gaussian sampleMarginal = est.GetDistribution(new Gaussian());
                    Gaussian result = new Gaussian();
                    result.SetToRatio(sampleMarginal, sample, GaussianOp.ForceProper);
                    if (double.IsNaN(result.Precision))
                        throw new InferRuntimeException($"result is NaN. sample={sample}, mean={mean}, precision={precision}");
                    return result;
                }
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Slow"]/message_doc[@name="PrecisionAverageConditional(Gaussian, Gaussian, Gamma)"]/*'/>
        public static Gamma PrecisionAverageConditional([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision)
        {
            if (sample.IsPointMass && mean.IsPointMass)
                return PrecisionAverageConditional(sample.Point, mean.Point);
            else if (precision.IsPointMass || precision.GetVariance() < 1e-20)
            {
                return GaussianOp.PrecisionAverageConditional(sample, mean, precision);
            }
            else if (sample.IsUniform() || mean.IsUniform())
            {
                return Gamma.Uniform();
            }
            else if (!precision.IsProper())
            {
                // improper prior
                throw new ImproperMessageException(precision);
            }
            else
            {
                double mx, vx;
                sample.GetMeanAndVariance(out mx, out vx);
                double mm, vm;
                mean.GetMeanAndVariance(out mm, out vm);
                double m = mx - mm;
                double v = vx + vm;
                if (double.IsPositiveInfinity(v))
                    return Gamma.Uniform();
                double m2 = m * m;
                double a = precision.Shape;
                double b = precision.Rate;
                double logr0;
                double logrmin, logrmax;
                GetIntegrationBoundsForPrecision(m, v, a, b, out logrmin, out logrmax, out logr0);
                if (logrmin == logr0 || logrmax == logr0)
                    return PrecisionAverageConditional(sample, mean, Gamma.PointMass(Math.Exp(logr0)));
                int n = QuadratureNodeCount;
                double inc = (logrmax - logrmin) / (n - 1);
                if (m2 - v < precision.Rate)
                {
                    // This is a special approach that avoids SetToRatio, useful when precision.Rate is large.
                    // this approach is inaccurate if posterior mean << prior mean, since then alpha =approx -precision.Shape
                    // this happens when ym2-yv >> precision.Rate
                    // sum1 = Exf'
                    double sum1 = 0;
                    // sum2 = Ex^2f''
                    double sum2 = 0;
                    double Z = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double logr = logrmin + i * inc;
                        double r = Math.Exp(logr);
                        double ir = 1 / r;
                        double denom = 1 / (v + ir);
                        double ir2 = ir * ir;
                        double denom2 = denom * denom;
                        double dlogf1 = -0.5 * denom + 0.5 * m2 * denom2;
                        // dlogfr = r f'/f
                        double dlogfr = dlogf1 * (-ir);
                        double dlogf2 = (0.5 - m2 * denom) * denom2;
                        // ddfrr = r^2 f''/f
                        // f''/f = d(f'/f) + (f'/f)^2
                        double ddfrr = dlogfr * dlogfr + dlogf2 * ir2 + (2 * ir) * dlogf1;
                        double diff = LogLikelihoodRatio(logr, logr0, m, v, a, b);
                        if ((i == 0 || i == n - 1) && (diff > -49))
                            throw new InferRuntimeException("invalid integration bounds");
                        double f = Math.Exp(diff);
                        if (double.IsPositiveInfinity(f))
                        {
                            // this can happen if the likelihood is extremely sharp
                            //throw new Exception("overflow");
                            return PrecisionAverageConditional(sample, mean, Gamma.PointMass(Math.Exp(logr0)));
                        }
                        Z += f;
                        sum1 += dlogfr * f;
                        sum2 += ddfrr * f;
                    }
                    double alpha = sum1 / Z;
                    double beta = (sum1 + sum2) / Z - alpha * alpha;
                    return GaussianOp.GammaFromAlphaBeta(precision, alpha, beta, GaussianOp.ForceProper);
                }
                else
                {
                    // Compute the marginal and then divide
                    MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
                    for (int i = 0; i < n; i++)
                    {
                        double logr = logrmin + i * inc;
                        double r = Math.Exp(logr);
                        double diff = LogLikelihoodRatio(logr, logr0, m, v, a, b);
                        if ((i == 0 || i == n - 1) && (diff > -49))
                            throw new InferRuntimeException("invalid integration bounds");
                        double p = Math.Exp(diff);
                        if (double.IsPositiveInfinity(p))
                            throw new InferRuntimeException("overflow");
                        mva.Add(r, p);
                    }
                    if (mva.Count == 0)
                        throw new InferRuntimeException("Quadrature found zero mass");
                    if (double.IsNaN(mva.Count))
                        throw new InferRuntimeException("count is nan");
                    Gamma precMarginal = Gamma.FromMeanAndVariance(mva.Mean, mva.Variance);
                    Gamma result = new Gamma();
                    result.SetToRatio(precMarginal, precision, GaussianOp.ForceProper);
                    if (double.IsNaN(result.Rate))
                        throw new InferRuntimeException($"result is NaN.  sample={sample}, mean={mean}, precision={precision}");
                    return result;
                }
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Slow"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, Gamma)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision)
        {
            if (precision.IsPointMass)
                return LogAverageFactor(sample, mean, precision.Point);
            if (precision.IsUniform())
                return Double.PositiveInfinity;
            if (sample.IsPointMass && mean.IsPointMass)
                return LogAverageFactor(sample.Point, mean.Point, precision);
            if (sample.IsUniform() || mean.IsUniform())
                return 0.0;
            double mx, vx;
            sample.GetMeanAndVariance(out mx, out vx);
            double mm, vm;
            mean.GetMeanAndVariance(out mm, out vm);
            double m = mx - mm;
            double v = vx + vm;
            double m2 = m * m;
            double a = precision.Shape;
            double b = precision.Rate;
            double logr0;
            double logrmin, logrmax;
            GetIntegrationBoundsForPrecision(m, v, a, b, out logrmin, out logrmax, out logr0);
            int n = QuadratureNodeCount;
            double inc = (logrmax - logrmin) / (n - 1);
            double logZ = double.NegativeInfinity;
            for (int i = 0; i < n; i++)
            {
                double logr = logrmin + i * inc;
                double r = Math.Exp(logr);
                double vr = v + 1 / r;
                double logp = -0.5 * Math.Log(vr) - 0.5 * m2 / vr + a * logr - b * r;
                logZ = MMath.LogSumExp(logZ, logp);
            }
            logZ += -MMath.LnSqrt2PI - MMath.GammaLn(a) + a * Math.Log(b) + Math.Log(inc);
            return logZ;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Slow"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gamma, Gaussian)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision, [Fresh] Gaussian to_sample)
        {
            if (precision.IsPointMass)
                return LogEvidenceRatio(sample, mean, precision.Point);
            //Gaussian to_Sample = SampleAverageConditional(sample, mean, precision);
            return LogAverageFactor(sample, mean, precision)
              - sample.GetLogAverageOf(to_sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Slow"]/message_doc[@name="LogEvidenceRatio(double, Gaussian, Gamma)"]/*'/>
        public static double LogEvidenceRatio(double sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision)
        {
            if (precision.IsPointMass)
                return LogEvidenceRatio(sample, mean, precision.Point);
            //Gaussian to_Sample = SampleAverageConditional(sample, mean, precision);
            return LogAverageFactor(Gaussian.PointMass(sample), mean, precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Slow"]/message_doc[@name="MeanAverageConditional(Gaussian, Gaussian, Gamma)"]/*'/>
        public static Gaussian MeanAverageConditional([SkipIfUniform] Gaussian sample, [NoInit] Gaussian mean, [SkipIfUniform] Gamma precision)
        {
            return SampleAverageConditional(mean, sample, precision);
        }

        public static void GetIntegrationBoundsForPrecision(double m, double v, double a, double b, out double logrmin, out double logrmax, out double logrmode)
        {
            // compute stationary points and inflection points
            double v2 = v * v;
            double m2 = m * m;
            double[] coeffs = { -b * v2, (a * v2 - 2 * b * v), (0.5 * v - 0.5 * m2 + 2 * a * v - b), (0.5 + a) };
            List<double> stationaryPoints;
            Predicate<double> isPositive = (r => r > 0.0);
            GetRealRoots(coeffs, out stationaryPoints, isPositive);
            //Console.WriteLine("stationary points = {0}", StringUtil.CollectionToString(stationaryPoints, " "));
            stationaryPoints = stationaryPoints.ConvertAll(x => Math.Log(x));
            double v3 = v * v2;
            double[] coeffs2 = { -b * v3, -b * 3 * v2, (-b * 3 * v - 0.5 * v2 + 0.5 * m2 * v), (-0.5 * v - b - 0.5 * m2) };
            List<double> inflectionPoints;
            GetRealRoots(coeffs2, out inflectionPoints, isPositive);
            //Console.WriteLine("inflection points = {0}", StringUtil.CollectionToString(inflectionPoints, " "));
            inflectionPoints = inflectionPoints.ConvertAll(x => Math.Log(x));
            // find the maximum and the target value
            double like(double logx)
            {
                if (double.IsInfinity(logx))
                    return double.NegativeInfinity;
                double x = Math.Exp(logx);
                double vx = v + 1 / x;
                return a * logx - b * x - 0.5 * Math.Log(vx) - 0.5 * m2 / vx;
            }
            var stationaryValues = stationaryPoints.ConvertAll(logx => like(logx));
            double max = MMath.Max(stationaryValues);
            double logx0 = stationaryPoints[stationaryValues.IndexOf(max)];
            logrmode = logx0;
            double func(double logx)
            {
                return LogLikelihoodRatio(logx, logx0, m, v, a, b) + 50;
            }
            double deriv(double logx)
            {
                double x = Math.Exp(logx);
                double vx = v * x + 1;
                return a - b * x + 0.5 * (vx - m * m * x) / (vx * vx);
            }
            // find where the likelihood matches the bound value
            List<double> zeroes = FindZeroes(func, deriv, stationaryPoints, inflectionPoints);
            logrmin = MMath.Min(zeroes);
            logrmax = MMath.Max(zeroes);
            //Console.WriteLine("rmin = {0} rmax = {1} r0 = {2} f(rmin) = {3} f(rmax) = {4}",
            //    Math.Exp(logrmin), Math.Exp(logrmax), Math.Exp(logx0), func(logrmin), func(logrmax));
        }

        private static double LogLikelihoodRatio(double logx, double logx0, double m, double v, double a, double b)
        {
            if (double.IsInfinity(logx))
                return double.NegativeInfinity;
            double vx = v + Math.Exp(-logx);
            double vx0 = v + Math.Exp(-logx0);
            double diff = MMath.DifferenceOfExp(logx, logx0);
            return a * (logx - logx0) - b * diff - 0.5 * Math.Log(vx / vx0) - 0.5 * m * m * (1 / vx - 1 / vx0);
        }

        /// <summary>
        /// Find all zeroes of a function, given its stationary points and inflection points 
        /// </summary>
        /// <param name="func"></param>
        /// <param name="deriv"></param>
        /// <param name="stationaryPoints"></param>
        /// <param name="inflectionPoints"></param>
        /// <returns></returns>
        public static List<double> FindZeroes(Func<double, double> func, Func<double, double> deriv, IList<double> stationaryPoints, IList<double> inflectionPoints)
        {
            // To find the roots of a 1D function:
            // Find the set of stationary points and inflection points.
            // Between neighboring stationary points, the function must be monotonic, and there must be at least one inflection point.
            // Inflection points have the maximum derivative and thus will take the smallest Newton step.
            // At each inflection point, the first step of Newton's method will either go left or right.
            // If it stays within the neighboring stationary points, run Newton's method to convergence. It must find a root.
            // If it goes beyond the neighboring stationary point, skip the search.
            // Special consideration is needed for the smallest and largest stationary points, since there may be no inflection point to start at.
            // In this case, we find any point smaller than the smallest stationary point where the function has different sign from that stationary point.
            List<double> zeroes = new List<double>();
            // sort the stationary points
            var stationaryPointsSorted = new List<double>(stationaryPoints);
            stationaryPointsSorted.Add(double.NegativeInfinity);
            stationaryPointsSorted.Add(double.PositiveInfinity);
            stationaryPointsSorted.Sort();
            List<double> stationaryValues = new List<double>();
            foreach (double x in stationaryPointsSorted)
                stationaryValues.Add(func(x));
            for (int i = 1; i < stationaryPointsSorted.Count; i++)
            {
                bool lowerIsPositive = (stationaryValues[i - 1] > 0);
                bool upperIsPositive = (stationaryValues[i] > 0);
                if (lowerIsPositive != upperIsPositive)
                {
                    // found an interval with a zero
                    double lowerBound = stationaryPointsSorted[i - 1];
                    double upperBound = stationaryPointsSorted[i];
                    bool foundZero = false;
                    foreach (double inflection in inflectionPoints)
                    {
                        // Search from each valid inflection point in the interval
                        double derivStart = Math.Abs(deriv(inflection));
                        bool valid = (double.IsInfinity(lowerBound) || derivStart >= Math.Abs(deriv(lowerBound))) && 
                            (double.IsInfinity(upperBound) || derivStart >= Math.Abs(deriv(upperBound)));
                        if (valid && FindZeroNewton(func, deriv, inflection, lowerBound, upperBound, out double x))
                        {
                            foundZero = true;
                            zeroes.Add(x);
                            break;
                        }
                    }
                    // There are no inflection points in the interval.
                    if (!foundZero)
                    {
                        // try again starting from the edge
                        double start;
                        if (!double.IsNegativeInfinity(lowerBound))
                        {
                            if (!double.IsPositiveInfinity(upperBound))
                                start = (lowerBound + upperBound) / 2;
                            else
                            {
                                double delta = Math.Max(1, Math.Abs(lowerBound));
                                double f;
                                while (true)
                                {
                                    start = lowerBound + delta;
                                    f = func(start);
                                    upperIsPositive = (f > 0);
                                    if (lowerIsPositive != upperIsPositive) break;
                                    lowerBound = start;
                                    delta *= 2;
                                }
                                if (start == upperBound)
                                    continue;  // we're too close to double.maxValue to be able to find a root
                                if (double.IsInfinity(f))
                                {
                                    // we can't call FindZeroNewton with an infinite function value.
                                    // use bisection to search for a finite value with a different sign.
                                    // we know the root is in [lowerBound, start]
                                    upperBound = start;
                                    start = Bisection(func, lowerBound, upperBound, lowerIsPositive, upperIsPositive);
                                }
                            }
                        }
                        else if (!double.IsPositiveInfinity(upperBound))
                        {
                            double delta = Math.Max(1, Math.Abs(upperBound));
                            double f;
                            while (true)
                            {
                                start = upperBound - delta;
                                f = func(start);
                                lowerIsPositive = (f > 0);
                                if (lowerIsPositive != upperIsPositive) break;
                                upperBound = start;
                                delta *= 2;
                            }
                            if (start == lowerBound)
                                continue;  // we're too close to double.minValue to be able to find a root
                            if (double.IsInfinity(f))
                            {
                                lowerBound = start;
                                start = Bisection(func, lowerBound, upperBound, lowerIsPositive, lowerIsPositive);
                            }
                        }
                        else
                            throw new ArgumentException("no finite stationary points");
                        double x;
                        if (FindZeroNewton(func, deriv, start, lowerBound, upperBound, out x))
                        {
                            zeroes.Add(x);
                        }
                        else
                        {
                            if (x < lowerBound)
                                zeroes.Add(lowerBound);
                            else if (x > upperBound)
                                zeroes.Add(upperBound);
                            else
                                throw new Exception(string.Format("could not find a zero between {0} and {1}", lowerBound, upperBound));
                        }
                    }
                }
            }
            return zeroes;
        }

        /// <summary>
        /// Computes an x in [lowerBound,upperBound] where func(x) is finite and (func(x)>0) == wantPositive.
        /// </summary>
        /// <param name="func"></param>
        /// <param name="lowerBound"></param>
        /// <param name="upperBound"></param>
        /// <param name="lowerValueIsPositive"></param>
        /// <param name="wantPositive"></param>
        /// <returns></returns>
        private static double Bisection(Func<double, double> func, double lowerBound, double upperBound, bool lowerValueIsPositive, bool wantPositive)
        {
            if (double.IsInfinity(lowerBound) || double.IsInfinity(upperBound))
                throw new ArgumentException("infinite bound");
            while (true)
            {
                double x = MMath.Average(lowerBound, upperBound);
                double f = func(x);
                bool isPositive = (f > 0);
                if (isPositive == wantPositive)
                {
                    if (double.IsInfinity(f))
                    {
                        // move away from the desired end
                        if (lowerValueIsPositive == wantPositive)
                        {
                            if (MMath.AreEqual(lowerBound, x)) return x;
                            else lowerBound = x;
                        }
                        else
                        {
                            if (MMath.AreEqual(upperBound, x)) return x;
                            else upperBound = x;
                        }
                    }
                    else
                    {
                        return x;
                    }
                }
                else // (isPositive != wantPositive)
                {
                    // move toward the desired end
                    if (lowerValueIsPositive == wantPositive)
                        upperBound = x;
                    else
                        lowerBound = x;
                }
            }
        }

        // The function is assumed monotonic between the bounds, and start is such that Newton should never change direction.
        // As a result, we can check for convergence without using tolerances.
        private static bool FindZeroNewton(Func<double, double> func, Func<double, double> deriv, double start, double lowerBound, double upperBound, out double x)
        {
            x = start;
            bool prevDeltaPositive = false;
            double oldx = x;
            for (int iter = 0; iter < 1000; iter++)
            {
                if (x < lowerBound || x > upperBound || double.IsNaN(x))
                    return false;
                double fx = func(x);
                double dfx = deriv(x);
                //Console.WriteLine("{0}: x = {1} f = {2} df = {3}", iter, x, fx, dfx);
                double delta = fx / dfx;
                bool deltaPositive = (delta > 0);
                if (iter > 0 && deltaPositive != prevDeltaPositive)
                {
                    // changed direction
                    if (fx > 0)
                    {
                        double x2 = x - delta;
                        if (x2 == x)
                            x = oldx;
                        else
                            x = x2;
                    }
                    return true;
                }
                prevDeltaPositive = deltaPositive;
                oldx = x;
                x -= delta;
                if (x == oldx)
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Get the complex roots of a polynomial
        /// </summary>
        /// <param name="coeffs">Coefficients of the polynomial, starting from the highest degree monomial down to the constant term</param>
        /// <param name="rootsReal">On exit, the real part of the roots</param>
        /// <param name="rootsImag">On exit, the imaginary part of the roots</param>
        public static void GetRoots(IList<double> coeffs, out double[] rootsReal, out double[] rootsImag)
        {
            int firstNonZero = coeffs.Count;
            for (int i = 0; i < coeffs.Count; i++)
            {
                if (coeffs[i] != 0)
                {
                    firstNonZero = i;
                    break;
                }
            }
            int n = coeffs.Count - 1 - firstNonZero;
            if (n <= 0)
            {
                rootsReal = new double[0];
                rootsImag = new double[0];
                return;
            }
            double firstNonZeroCoeff = coeffs[firstNonZero];
            Matrix m = new Matrix(n, n);
            for (int i = 0; i < n - 1; i++)
            {
                m[i + 1, i] = firstNonZeroCoeff;
            }
            for (int i = 0; i < n; i++)
            {
                m[0, i] = -coeffs[i + 1 + firstNonZero];
            }
            rootsReal = new double[n];
            rootsImag = new double[n];
            // Note m is already in Hessenberg form, so there is some potential savings here.
            m.EigenvaluesInPlace(rootsReal, rootsImag);
            for (int i = 0; i < n; i++)
            {
                rootsReal[i] /= firstNonZeroCoeff;
                rootsImag[i] /= firstNonZeroCoeff;
            }
        }

        /// <summary>
        /// Get the real roots of a polynomial
        /// </summary>
        /// <param name="coeffs">Coefficients of the polynomial, starting from the highest degree monomial down to the constant term</param>
        /// <param name="roots">On exit, the real roots</param>
        /// <param name="filter">If not null, only roots where filter returns true are included</param>
        public static void GetRealRoots(IList<double> coeffs, out List<double> roots, Predicate<double> filter = null)
        {
            double[] rootsReal, rootsImag;
            GetRoots(coeffs, out rootsReal, out rootsImag);
            roots = new List<double>();
            for (int i = 0; i < rootsReal.Length; i++)
            {
                if (Math.Abs(rootsImag[i]) < 1e-15 && (filter == null || filter(rootsReal[i])))
                    roots.Add(rootsReal[i]);
            }
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Laplace"]/doc/*'/>
    [FactorMethod(typeof(Gaussian), "Sample", typeof(double), typeof(double), Default = false)]
    [FactorMethod(new string[] { "sample", "mean", "precision" }, typeof(Factor), "Gaussian", Default = false)]
    // Q's mean is the expansion point = the maximum of the integrand over precision
    [Buffers("Q")]
    [Quality(QualityBand.Experimental)]
    public class GaussianOp_Laplace : GaussianOpBase
    {
        // Laplace cases -----------------------------------------------------------------------------------------
        public static bool modified = true;
        public static bool modified2 = false;

        /// <summary>
        /// Approximate the mean and variance of a function g(x) where x is distributed according to a Gamma times f(x).
        /// </summary>
        /// <param name="q">The Gamma distribution that multiplies f</param>
        /// <param name="g">g[0] = g(xhat), g[1] = g'(xhat), g[2] = g''(xhat), and so on.</param>
        /// <param name="dlogf">dlogf[0] = (logf)'(xhat), dlogf[1] = (logf)''(xhat), and so on.</param>
        /// <param name="m"></param>
        /// <param name="v"></param>
        public static void LaplaceMoments(Gamma q, double[] g, double[] dlogf, out double m, out double v)
        {
            if (q.IsPointMass)
            {
                m = g[0];
                v = 0;
                return;
            }
            double a = q.Shape;
            double b = q.Rate;
            double x = a / b;
            double dg = g[1];
            double ddg = g[2];
            double ddlogf = dlogf[1];
            double dddlogf = dlogf[2];
            double dx = dg * x / b;
            double a1 = -2 * x * ddlogf - dddlogf * x * x;
            double da = -ddg * x * x + dx * a1;
            m = g[0] + (MMath.Digamma(a) - Math.Log(a)) * da;
            if (double.IsNaN(m)) throw new Exception("m is nan");
            if (da > double.MaxValue || da < double.MinValue)
            {
                v = double.PositiveInfinity;
            }
            else if (g.Length > 3)
            {
                double dddg = g[3];
                double d4logf = dlogf[3];
                double db = -dg + da / x; // da/x = -ddg*x + dg/b*a1
                double ddx = (dg + x * ddg) / b * dx - x * dg / b / b * db;
                double a2 = -2 * ddlogf - 4 * x * dddlogf - d4logf * x * x;
                double dda = (-2 * x * ddg - dddg * x * x) * dx + a2 * dx * dx + a1 * ddx;
                v = dg * dx + (MMath.Trigamma(a) - 1 / a) * da * da + (MMath.Digamma(a) - Math.Log(a)) * dda;
                //if (v < 0)
                //    throw new Exception("v < 0");
                if (double.IsNaN(v)) throw new Exception("v is nan");
            }
            else
            {
                v = 0;
            }
        }

        // same as LaplaceMoments but where the arguments have been multiplied by x (the mean of q)
        public static void LaplaceMoments2(Gamma q, double[] xg, double[] xdlogf, out double m, out double v)
        {
            if (q.IsPointMass)
            {
                m = xg[0];
                v = 0;
                return;
            }
            double a = q.Shape;
            double b = q.Rate;
            double x = a / b;
            double xdg = xg[1];
            double xxddg = xg[2];
            double xxddlogf = xdlogf[1];
            double xxxdddlogf = xdlogf[2];
            double dxix = xdg / a;
            double xa1 = -2 * xxddlogf - xxxdddlogf;
            double da = -xxddg + dxix * xa1;
            m = xg[0] + (MMath.Digamma(a) - Math.Log(a)) * da;
            if (xg.Length > 3)
            {
                double xxxdddg = xg[3];
                double x4d4logf = xdlogf[3];
                double xdb = da - xdg;
                double ddxix = (xdg + xxddg) / a * dxix - xdg / (a * a) * xdb;
                double x2a2 = -2 * xxddlogf - 4 * xxxdddlogf - x4d4logf;
                double dda = (-2 * xxddg - xxxdddg) * dxix + x2a2 * dxix * dxix + xa1 * ddxix;
                v = xdg * dxix + (MMath.Trigamma(a) - 1 / a) * da * da + (MMath.Digamma(a) - Math.Log(a)) * dda;
                if (v < 0)
                    throw new Exception("v < 0");
            }
            else
            {
                v = 0;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Laplace"]/message_doc[@name="QInit()"]/*'/>
        [Skip]
        public static Gamma QInit()
        {
            return Gamma.Uniform();
        }

        // Perform one update of Q
        private static Gamma QUpdate(Gaussian sample, Gaussian mean, Gamma precision, Gamma q)
        {
            if (sample.IsUniform() || mean.IsUniform() || precision.IsPointMass)
                return precision;
            double mx, vx;
            sample.GetMeanAndVariance(out mx, out vx);
            double mm, vm;
            mean.GetMeanAndVariance(out mm, out vm);
            double m = mx - mm;
            double v = vx + vm;
            double m2 = m * m;
            double a = q.Shape;
            double b = q.Rate;
            if (b == 0 || q.IsPointMass)
            {
                a = precision.Shape;
                // this guess comes from solving dlogf=0 for x
                double guess = m2 - v;
                b = Math.Max(precision.Rate, a * guess);
            }
            double x = a / b;
            double x2 = x * x;
            double x3 = x * x2;
            double x4 = x * x3;
            double p = 1 / (v + 1 / x);
            double dlogf1 = -0.5 * p + 0.5 * m2 * p * p;
            double dlogf = dlogf1 * (-1 / x2);
            double ddlogf1 = 0.5 * p * p - m2 * p * p * p;
            double ddlogf = dlogf1 * 2 / x3 + ddlogf1 / x4;
            b = precision.Rate - (dlogf + x * ddlogf);
            if (b < 0)
            {
                if (q.Rate == precision.Rate || true)
                    return QUpdate(sample, mean, precision, Gamma.FromShapeAndRate(precision.Shape, precision.Shape * (m2 - v)));
            }
            a = precision.Shape - x2 * ddlogf;
            if (a <= 0)
                a = b * precision.Shape / (precision.Rate - dlogf);
            if (a <= 0 || b <= 0)
                throw new InferRuntimeException("a <= 0 || b <= 0");
            if (double.IsNaN(a) || double.IsNaN(b))
                throw new InferRuntimeException($"result is NaN.  sample={sample}, mean={mean}, precision={precision}, q={q}");
            return Gamma.FromShapeAndRate(a, b);
        }
        private static double QReinitialize(Gaussian sample, Gaussian mean, Gamma precision, double x)
        {
            // there can be two local optima for q
            // each time we update q, check the function value near each optimum, and re-initialize q if any is higher
            double mx, vx;
            sample.GetMeanAndVariance(out mx, out vx);
            double mm, vm;
            mean.GetMeanAndVariance(out mm, out vm);
            double m = mx - mm;
            double v = vx + vm;
            double m2 = m * m;
            double a = precision.Shape;
            double b = precision.Rate;
            //double init0 = precision.GetMean();
            double init0 = (a + 0.5) / b;
            double init1 = (a + 0.5) / (b + 0.5 * (m2 + v));
            //double init1 = 1/(m2-v);
            if (v > m2)
            {
                // if v is large then
                // -0.5*log(v + 1/x) - 0.5*m2/(v + 1/x) + a*log(x) - b*x =approx -0.5/v/x + 0.5*m2/v^2/x + a*log(x) - b*x
                double c = 0.5 / v * (m2 / v - 1);
                // c < 0 so the Sqrt always succeeds
                init1 = (a + Math.Sqrt(a * a - 4 * b * c)) / (2 * b);
            }
            double logz0 = -0.5 * Math.Log(1 + 1 / init0 / v) - 0.5 * m2 / (v + 1 / init0) + a * Math.Log(init0) - b * init0;
            double logz1 = -0.5 * Math.Log(1 + 1 / init1 / v) - 0.5 * m2 / (v + 1 / init1) + a * Math.Log(init1) - b * init1;
            double logz = -0.5 * Math.Log(1 + 1 / x / v) - 0.5 * m2 / (v + 1 / x) + a * Math.Log(x) - b * x;
            if (double.IsNaN(logz))
                logz = double.NegativeInfinity;
            double logzMax = Math.Max(logz1, logz);
            // must use IsGreater here because round-off errors can cause logz1 > logz even though it is a worse solution according to the gradient
            if (IsGreater(logz0, logzMax))
                return init0;
            else if (IsGreater(logz1, logz))
                return init1;
            else
                return x;
        }
        private static bool IsGreater(double a, double b)
        {
            return (a > b) && (MMath.AbsDiff(a, b, 1e-14) > 1e-12);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        public static Gamma Q_Slow(Gaussian sample, Gaussian mean, Gamma precision)
        {
            if (precision.IsPointMass || sample.IsUniform() || mean.IsUniform())
                return precision;

            // Find the maximum of the integrand over precision
            double mx, vx;
            sample.GetMeanAndVariance(out mx, out vx);
            double mm, vm;
            mean.GetMeanAndVariance(out mm, out vm);
            double m = mx - mm;
            double v = vx + vm;
            if (double.IsInfinity(v))
                return precision;
            double a = precision.Shape;
            double b = precision.Rate;
            double logrmin, logrmax, logx;
            GaussianOp_Slow.GetIntegrationBoundsForPrecision(m, v, a, b, out logrmin, out logrmax, out logx);
            double x = Math.Exp(logx);
            double[] xdlogfss = xdlogfs(x, m, v);
            double xdlogf = xdlogfss[0];
            double xxddlogf = xdlogfss[1];
            a = precision.Shape - xxddlogf;
            b = precision.Rate - (xdlogf + xxddlogf) / x;
            if (a <= 0 || b <= 0)
                throw new InferRuntimeException();
            if (double.IsNaN(a) || double.IsNaN(b))
                throw new InferRuntimeException($"result is NaN.  sample={sample}, mean={mean}, precision={precision}");
            return Gamma.FromShapeAndRate(a, b);
        }

        // sample may need [NoInit] here
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Laplace"]/message_doc[@name="Q(Gaussian, Gaussian, Gamma, Gamma)"]/*'/>
        [Fresh]
        public static Gamma Q(Gaussian sample, Gaussian mean, Gamma precision, Gamma q)
        {
            if (precision.IsPointMass || sample.Precision == 0 || mean.Precision == 0)
                return precision;

            // Find the maximum of the integrand over precision
            double mx, vx;
            sample.GetMeanAndVariance(out mx, out vx);
            double mm, vm;
            mean.GetMeanAndVariance(out mm, out vm);
            double m = mx - mm;
            double v = vx + vm;
            if (double.IsInfinity(v))
                return precision;
            double a = precision.Shape;
            double b = precision.Rate;
            double m2 = m * m;
            double[] dlogfss;
            double x = q.GetMean();
            if (double.IsPositiveInfinity(x))
                x = (a + Math.Sqrt(a * a + 2 * b / v)) / (2 * b);
            // TODO: check initialization
            if (q.IsUniform())
                x = 0;
            //x = QReinitialize(sample, mean, precision, x);
            int maxIter = 100;
            for (int iter = 0; iter < maxIter; iter++)
            {
                double oldx = x;
                if (true)
                {
                    // want to find x that optimizes -0.5*log(v + 1/x) - 0.5*m2/(v + 1/x) + a*log(x) - b*x
                    // we fit a lower bound, then optimize the bound.
                    // this is guaranteed to improve x.
                    double logSlope = precision.Shape;
                    double slope = -precision.Rate;
                    double denom = v * x + 1;
                    if (v * x < 1)
                    {
                        // 1/(v+1/x) <= 1/(v+1/x0) + (x-x0)/(v*x0+1)^2
                        slope += -0.5 * m2 / (denom * denom);
                        // log(v+1/x)  = log(v*x+1) - log(x)
                        // log(v*x+1) <= log(v*x0+1) + (x-x0)*v/(v*x0+1)
                        logSlope += 0.5;
                        slope += -0.5 * v / denom;
                        x = -logSlope / slope;
                        // at x=0 the update is x = (a+0.5)/(b + 0.5*(m2+v))
                        // at x=inf the update is x = (a+0.5)/b
                    }
                    else
                    {
                        // if v*x > 1:
                        // log(v+1/x) <= log(v+1/x0) + (1/x - 1/x0)/(v + 1/x0)
                        double invSlope = -0.5 * x / denom;
                        bool found = false;
                        if (true)
                        {
                            // logf =approx c1/x + c2*x + c3  (c1<0, c2<0)
                            // dlogf = -c1/x^2 + c2
                            // ddlogf = 2*c1/x^3
                            double x2 = x * x;
                            dlogfss = dlogfs(x, m, v);
                            double c1 = 0.5 * x * x2 * (dlogfss[1] - a / x2);
                            double c2 = (dlogfss[0] + a / x - b) + c1 / x2;
                            if (c1 < 0 && c2 < 0 && c1 > double.MinValue && c2 > double.MinValue)
                            {
                                x = Math.Sqrt(c1 / c2);
                                found = true;
                            }
                        }
                        if (false)
                        {
                            double oldf = -0.5 * Math.Log(v + 1 / x) - 0.5 * m2 / (v + 1 / x) + a * Math.Log(x) - b * x;
                            // 1/(v+1/x) =approx 1/(v+1/x0) + (1/x0 - 1/x)/(v+1/x0)^2
                            double s = x / denom;
                            double invSlope2 = invSlope + 0.5 * m2 * s * s;
                            double c = 0.5 * logSlope / slope;
                            double d = invSlope2 / slope;
                            double x2 = Math.Sqrt(c * c + d) - c;
                            double newf = -0.5 * Math.Log(v + 1 / x2) - 0.5 * m2 / (v + 1 / x2) + a * Math.Log(x2) - b * x2;
                            if (IsGreater(newf, oldf))
                            {
                                x = x2;
                                found = true;
                            }
                        }
                        if (!found)
                        {
                            // 1/(v+1/x) <= 1/(v+1/x0) + (x-x0)/(v*x0+1)^2
                            slope += -0.5 * m2 / (denom * denom);
                            // solve for the maximum of logslope*log(r)+slope*r+invslope./r
                            // logslope/r + slope - invslope/r^2 = 0
                            // logslope*r + slope*r^2 - invslope = 0
                            //x = (-logSlope - Math.Sqrt(logSlope*logSlope + 4*invSlope*slope))/(2*slope);
                            double c = 0.5 * logSlope / slope;
                            double d = invSlope / slope;
                            // note c < 0 always
                            x = Math.Sqrt(c * c + d) - c;
                            // at x=inf, invSlope=-0.5/v so the update is x = (ax + sqrt(ax*ax + 2*b/v))/(2*b)
                        }
                    }
                    if (x < 0)
                        throw new Exception("x < 0");
                }
                else
                {
                    dlogfss = dlogfs(x, m, v);
                    if (true)
                    {
                        x = (precision.Shape - x * x * dlogfss[1]) / (precision.Rate - dlogfss[0] - x * dlogfss[1]);
                    }
                    else if (true)
                    {
                        double delta = dlogfss[0] - precision.Rate;
                        x *= Math.Exp(-(delta * x + precision.Shape) / (delta * x + dlogfss[1] * x));
                    }
                    else
                    {
                        x = precision.Shape / (precision.Rate - dlogfss[0]);
                    }
                    if (x < 0)
                        throw new Exception("x < 0");
                }
                if (double.IsNaN(x))
                    throw new Exception("x is nan");
                //System.Diagnostics.Trace.WriteLine($"{iter}: {x}");
                if (MMath.AbsDiff(oldx, x, 1e-10) < 1e-10)
                {
                    x = QReinitialize(sample, mean, precision, x);
                    if (MMath.AbsDiff(oldx, x, 1e-10) < 1e-10)
                        break;
                }
                if (iter == maxIter - 1)
                {
                    //throw new Exception("not converging");
                    double logrmin, logrmax, logx;
                    GaussianOp_Slow.GetIntegrationBoundsForPrecision(m, v, a, b, out logrmin, out logrmax, out logx);
                    x = Math.Exp(logx);
                }
            }
            //x = r0;
            double[] xdlogfss = xdlogfs(x, m, v);
            double xdlogf = xdlogfss[0];
            double xxddlogf = xdlogfss[1];
            a = precision.Shape - xxddlogf;
            if (x == 0)
            {
                // xdlogf = 0.5/(v*x+1) - 0.5*x*m*m/(v*x+1)^2
                // xxddlogf = -1/(v*x+1) + x*m*m/(v*x+1)^2 + 0.5/(v*x+1)^2 - x*m*m/(v*x+1)^3
                // xdlogf + xxddlogf = -0.5*x*v/(v*x+1)^2 + 0.5*x*m*m/(v*x+1)^2 - x*m*m/(v*x+1)^3
                // limit x->0: x*(-0.5*v - 0.5*m*m)
                b = precision.Rate + 0.5 * (v + m * m);
            }
            else
            {
                b = precision.Rate - (xdlogf + xxddlogf) / x;
            }
            if (a <= 0 || b <= 0)
                throw new InferRuntimeException();
            if (double.IsNaN(a) || double.IsNaN(b))
                throw new InferRuntimeException($"result is NaN.  sample={sample}, mean={mean}, precision={precision}, q={q}");
            return Gamma.FromShapeAndRate(a, b);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Laplace"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, Gamma, Gamma)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, Gamma precision, Gamma q)
        {
            double mx, vx;
            sample.GetMeanAndVariance(out mx, out vx);
            double mm, vm;
            mean.GetMeanAndVariance(out mm, out vm);
            double m = mx - mm;
            double v = vx + vm;
            double m2 = m * m;
            double x = q.GetMean();
            double logf = -MMath.LnSqrt2PI - 0.5 * Math.Log(v + 1 / x) - 0.5 * m2 / (v + 1 / x);
            double logz = logf + precision.GetLogProb(x) - q.GetLogProb(x);
            return logz;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Laplace"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gamma, Gaussian, Gamma)"]/*'/>
        public static double LogEvidenceRatio(
            [SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision, [Fresh] Gaussian to_sample, [Fresh] Gamma q)
        {
            return LogAverageFactor(sample, mean, precision, q)
              - sample.GetLogAverageOf(to_sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Laplace"]/message_doc[@name="LogEvidenceRatio(double, Gaussian, Gamma, Gamma)"]/*'/>
        public static double LogEvidenceRatio(double sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision, [Fresh] Gamma q)
        {
            return LogAverageFactor(Gaussian.PointMass(sample), mean, precision, q);
        }

        /// <summary>
        /// Derivatives of the factor wrt precision
        /// </summary>
        /// <param name="x"></param>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <returns></returns>
        public static double[] dlogfs(double x, double m, double v)
        {
            if (double.IsPositiveInfinity(v))
                return new double[4];
            if (x == 0)
                return new double[] { double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity };
            // log f(x) = -0.5*log(v+1/x) -0.5*m^2/(v+1/x)
            double x2 = x * x;
            double x3 = x * x2;
            double x4 = x * x3;
            double pix = 1 / (v * x + 1);
            double pix2 = pix / x;
            double p2ix2 = pix * pix;
            double p3ix3 = p2ix2 * pix;
            double m2p2ix2 = p2ix2 * m * m;
            double dlogf1ix2 = -0.5 * pix2 + 0.5 * m2p2ix2;
            double dlogf = dlogf1ix2 * (-1);
            double ddlogf1ix3 = 0.5 * pix * pix2 - m2p2ix2 * pix;
            double ddlogf = (dlogf1ix2 * 2 + ddlogf1ix3) / x;
            double dddlogf1ix4 = -p2ix2 * pix2 + 3 * m2p2ix2 * p2ix2;
            double dddlogf = (dlogf1ix2 * (-6) + ddlogf1ix3 * (-6) + dddlogf1ix4 * (-1)) / x2;
            double d4logf1ix5 = 3 * pix2 * p3ix3 - 12 * m2p2ix2 * p3ix3;
            double d4logf = (dlogf1ix2 * 24 + ddlogf1ix3 * 36 + dddlogf1ix4 * 12 + d4logf1ix5) / x3;
            return new double[] { dlogf, ddlogf, dddlogf, d4logf };
        }

        /// <summary>
        /// Derivatives of the factor wrt precision, times powers of x
        /// </summary>
        /// <param name="x"></param>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <returns></returns>
        public static double[] xdlogfs(double x, double m, double v)
        {
            if (double.IsPositiveInfinity(v))
                return new double[4];
            // log f(x) = -0.5*log(v+1/x) -0.5*m^2/(v+1/x)
            if (x * v > 1 && false)
            {
                double m2 = m * m;
                double x2 = x * x;
                double x3 = x * x2;
                double x4 = x * x3;
                double p = 1 / (v + 1 / x);
                double p2 = p * p;
                double p3 = p * p2;
                double dlogf1 = -0.5 * p + 0.5 * m2 * p2;
                double xdlogf = dlogf1 * (-1 / x);
                double ddlogf1 = 0.5 * p2 - m2 * p3;
                double xxddlogf = dlogf1 * 2 / x + ddlogf1 / x2;
                double dddlogf1 = -p3 + 3 * m2 * p * p3;
                double xxxdddlogf = dlogf1 * (-6) / x + ddlogf1 * (-6) / x2 + dddlogf1 * (-1) / x3;
                double d4logf1 = 3 * p * p3 - 12 * m2 * p2 * p3;
                double x4d4logf = dlogf1 * (24) / x + ddlogf1 * 36 / x2 + dddlogf1 * (12) / x3 + d4logf1 / x4;
                return new double[] { xdlogf, xxddlogf, xxxdddlogf, x4d4logf };
            }
            else  // x is small
            {
                double m2x = x * m * m;
                double x2 = x * x;
                double x3 = x * x2;
                double x4 = x * x3;
                double p = 1 / (v * x + 1);
                double p2 = p * p;
                double p3 = p * p2;
                // xdlogf = 0.5/(v*x+1) - 0.5*x*m*m/(v*x+1)^2
                double ixdlogf1 = -0.5 * p + 0.5 * m2x * p2;
                double xdlogf = -ixdlogf1;
                double ix2ddlogf1 = 0.5 * p2 - m2x * p3;
                // xxddlogf = -1/(v*x+1) + x*m*m/(v*x+1)^2 + 0.5/(v*x+1)^2 - x*m*m/(v*x+1)^3
                double xxddlogf = ixdlogf1 * 2 + ix2ddlogf1;
                double ix3dddlogf1 = -p3 + 3 * m2x * p * p3;
                double xxxdddlogf = ixdlogf1 * (-6) + ix2ddlogf1 * (-6) + ix3dddlogf1 * (-1);
                double ix4d4logf1 = 3 * p * p3 - 12 * m2x * p2 * p3;
                double x4d4logf = ixdlogf1 * (24) + ix2ddlogf1 * 36 + ix3dddlogf1 * (12) + ix4d4logf1;
                return new double[] { xdlogf, xxddlogf, xxxdddlogf, x4d4logf };
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Laplace"]/message_doc[@name="PrecisionAverageConditional(Gaussian, Gaussian, Gamma, Gamma)"]/*'/>
        public static Gamma PrecisionAverageConditional([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [Proper] Gamma precision, Gamma q)
        {
            if (sample.IsUniform() || mean.IsUniform())
                return Gamma.Uniform();
            double mx, vx;
            sample.GetMeanAndVariance(out mx, out vx);
            double mm, vm;
            mean.GetMeanAndVariance(out mm, out vm);
            double m = mx - mm;
            double v = vx + vm;
            double x = q.GetMean();
            double[] xg = new double[] { x, x, 0, 0 };
            double precMean, precVariance;
            LaplaceMoments2(q, xg, xdlogfs(x, m, v), out precMean, out precVariance);
            if (precMean < 0)
                throw new InferRuntimeException("precMean < 0");
            if (double.IsNaN(precMean) || double.IsNaN(precVariance))
                throw new InferRuntimeException("result is NaN");
            Gamma precMarginal = Gamma.FromMeanAndVariance(precMean, precVariance);
            Gamma result = new Gamma();
            result.SetToRatio(precMarginal, precision, GaussianOp.ForceProper);
            if (double.IsNaN(result.Rate))
                throw new InferRuntimeException("result is NaN");
            return result;
        }

        public static Gamma PrecisionAverageConditional_slow([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [Proper] Gamma precision)
        {
            Gamma q = QInit();
            q = Q(sample, mean, precision, q);
            return PrecisionAverageConditional(sample, mean, precision, q);
        }

        public static double[] dlogfxs(double x, Gamma precision)
        {
            if (precision.IsPointMass)
                throw new Exception("precision is point mass");
            // f(sample,mean) = tpdf(sample-mean, 0, 2b, 2a+1) N(mean; mm, vm)
            // f(x) = tpdf(x; 0, 2b, 2a+1)
            // log f(x) = -(a+0.5)*log(1 + x^2/2/b)  + const.
            double a = precision.Shape;
            double b = precision.Rate;
            double a2 = -(a + 0.5);
            double bdenom = b + x * x / 2;
            double bdenom2 = bdenom * bdenom;
            double bdenom3 = bdenom * bdenom2;
            double bdenom4 = bdenom * bdenom3;
            double dlogf = x / bdenom;
            double ddlogf = 1 / bdenom - x * x / bdenom2;
            double dddlogf = 2 * x * x * x / bdenom3 - 3 * x / bdenom2;
            double d4logf = 12 * x * x / bdenom3 - 6 * x * x * x * x / bdenom4 - 3 / bdenom2;
            return new double[] { a2 * dlogf, a2 * ddlogf, a2 * dddlogf, a2 * d4logf };
        }

        public static Gaussian Qx(Gaussian y, Gamma precision, Gaussian qx)
        {
            if (y.IsPointMass)
                return y;
            double x = QxReinitialize(y, precision, qx.GetMean());
            double r = 0;
            for (int iter = 0; iter < 1000; iter++)
            {
                double oldx = x;
                double[] dlogfs = dlogfxs(x, precision);
                double ddlogf = dlogfs[1];
                r = Math.Max(0, -ddlogf);
                double t = r * x + dlogfs[0];
                x = (t + y.MeanTimesPrecision) / (r + y.Precision);
                if (Math.Abs(oldx - x) < 1e-10)
                    break;
                //Console.WriteLine("{0}: {1}", iter, x);        
                if (iter == 1000 - 1)
                    throw new Exception("not converging");
                if (iter % 100 == 99)
                    x = QxReinitialize(y, precision, x);
            }
            return Gaussian.FromMeanAndPrecision(x, r + y.Precision);
        }
        private static double QxReinitialize(Gaussian y, Gamma precision, double x)
        {
            double init0 = 0;
            double init1 = y.GetMean();
            double a = precision.Shape;
            double b = precision.Rate;
            double a2 = -(a + 0.5);
            double logz0 = a2 * Math.Log(b + init0 * init0 / 2) + y.GetLogProb(init0);
            double logz1 = a2 * Math.Log(b + init1 * init1 / 2) + y.GetLogProb(init1);
            double logz = a2 * Math.Log(b + x * x / 2) + y.GetLogProb(x);
            if (logz0 > Math.Max(logz1, logz))
                return init0;
            else if (logz1 > logz)
                return init1;
            else
                return x;
        }

        public static void LaplaceMoments(Gaussian q, double[] dlogfx, out double m, out double v)
        {
            double vx = 1 / q.Precision;
            double delta = 0.5 * dlogfx[2] * vx * vx;
            m = q.GetMean() + delta;
            v = vx + 4 * delta * delta + 0.5 * dlogfx[3] * vx * vx * vx;
            if (v < 0)
                throw new Exception();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Laplace"]/message_doc[@name="SampleAverageConditional(Gaussian, Gaussian, Gamma, Gamma)"]/*'/>
        public static Gaussian SampleAverageConditional([NoInit] Gaussian sample, [SkipIfUniform] Gaussian mean, [Proper] Gamma precision, Gamma q)
        {
            if (mean.IsUniform() || sample.IsPointMass)
                return Gaussian.Uniform();
            if (precision.IsPointMass)
                return GaussianOp.SampleAverageConditional(mean, precision.Point);
            if (q.IsPointMass)
                throw new Exception();
            double mm, vm;
            mean.GetMeanAndVariance(out mm, out vm);
            if (sample.IsUniform())
            {
                if (precision.Shape <= 1.0)
                {
                    sample = Gaussian.FromNatural(1e-20, 1e-20);  // try to proceed instead of throwing an exception
                    //throw new ArgumentException("The posterior has infinite variance due to precision distributed as "+precision+" (shape <= 1).  Try using a different prior for the precision, with shape > 1.");
                }
                else
                    return Gaussian.FromMeanAndVariance(mm, vm + precision.GetMeanInverse());
            }
            double mx, vx;
            sample.GetMeanAndVariance(out mx, out vx);
            double m = mx - mm;
            double v = vx + vm;
            Gaussian sampleMarginal;
            bool useQx = true;
            if (useQx)
            {
                Gaussian qx = Qx(Gaussian.FromMeanAndVariance(m, v), precision, Gaussian.FromMeanAndPrecision(0, 1));
                double y = qx.GetMean();
                double[] dlogfx = dlogfxs(y, precision);
                double delta = 0.5 * dlogfx[2] / (qx.Precision * qx.Precision);
                // sampleMean can be computed directly as:
                //double dlogz = dlogfx[0] + delta/v;
                //double sampleMean = mx + vx*dlogz;
                double my = y + delta;
                double vy = 1 / qx.Precision + 4 * delta * delta + 0.5 * dlogfx[3] / (qx.Precision * qx.Precision * qx.Precision);
                if (vy < 0)
                    throw new Exception();
                Gaussian yMsg = new Gaussian(my, vy) / (new Gaussian(m, v));
                Gaussian sampleMsg = DoublePlusOp.SumAverageConditional(yMsg, mean);
                sampleMarginal = sampleMsg * sample;
                if (!sampleMarginal.IsProper())
                    throw new Exception();
                if (sampleMarginal.IsPointMass)
                    throw new Exception();
                //Console.WriteLine("{0} {1}", sampleMean, sampleMarginal);
            }
            else
            {
                double a = q.Shape;
                double b = q.Rate;
                double x = a / b;
                if (sample.IsPointMass || mean.IsPointMass)
                {
                    double denom = 1 + x * v;
                    double denom2 = denom * denom;
                    Gaussian y = sample * mean;
                    double my, vy;
                    y.GetMeanAndVariance(out my, out vy);
                    // 1-1/denom = x*v/denom
                    // sampleMean = E[ (x*(mx*vm + mm*vx) + mx)/denom ]
                    //            = E[ (mx*vm + mm*vx)/v*(1-1/denom) + mx/denom ]
                    //            = E[ (mx/vx + mm/vm)/(1/vx + 1/vm)*(1-1/denom) + mx/denom ]
                    // sampleVariance = var((my*(1-1/denom) + mx/denom)) + E[ (r*vx*vm + vx)/denom ]
                    //                = (mx-my)^2*var(1/denom) + E[ vx*vm/v*(1-1/denom) + vx/denom ]
                    double[] g = new double[] { 1 / denom, -v / denom2, 2 * v * v / (denom2 * denom), -6 * v * v * v / (denom2 * denom2) };
                    double[] dlogf = dlogfs(x, m, v);
                    double edenom, vdenom;
                    LaplaceMoments(q, g, dlogf, out edenom, out vdenom);
                    double sampleMean = mx * edenom + my * (1 - edenom);
                    double diff = mx - my;
                    double sampleVariance = vx * edenom + vy * (1 - edenom) + diff * diff * vdenom;
                    sampleMarginal = Gaussian.FromMeanAndVariance(sampleMean, sampleVariance);
                }
                else
                {
                    // 1 - samplePrec*mPrec/denom = x*yprec/denom
                    // sampleMean = E[ (x*ymprec + sampleMP*mPrec)/denom ]
                    //            = E[ (1 - samplePrec*mPrec/denom)*ymprec/yprec + sampleMP*mPrec/denom ]
                    // sampleVariance = var((1 - samplePrec*mPrec/denom)*ymprec/yprec + sampleMP*mPrec/denom) + E[ (x+mPrec)/denom ]
                    //                = (sampleMP*mPrec - samplePrec*mPrec*ymprec/yprec)^2*var(1/denom) + E[ (1-samplePrec*mPrec/denom)/yprec + mPrec/denom ]
                    double yprec = sample.Precision + mean.Precision;
                    double ymprec = sample.MeanTimesPrecision + mean.MeanTimesPrecision;
                    double denom = sample.Precision * mean.Precision + x * yprec;
                    double denom2 = denom * denom;
                    double[] g = new double[] { 1 / denom, -yprec / denom2, 2 * yprec * yprec / (denom2 * denom), -6 * yprec * yprec * yprec / (denom2 * denom2) };
                    double[] dlogf = dlogfs(x, m, v);
                    double edenom, vdenom;
                    LaplaceMoments(q, g, dlogf, out edenom, out vdenom);
                    double sampleMean = sample.MeanTimesPrecision * mean.Precision * edenom + ymprec / yprec * (1 - sample.Precision * mean.Precision * edenom);
                    double diff = sample.MeanTimesPrecision * mean.Precision - sample.Precision * mean.Precision * ymprec / yprec;
                    double sampleVariance = mean.Precision * edenom + (1 - sample.Precision * mean.Precision * edenom) / yprec + diff * diff * vdenom;
                    sampleMarginal = Gaussian.FromMeanAndVariance(sampleMean, sampleVariance);
                }
            }
            Gaussian result = new Gaussian();
            result.SetToRatio(sampleMarginal, sample, GaussianOp.ForceProper);
            if (modified2)
            {
                if (!mean.IsPointMass)
                    throw new Exception();
                result.SetMeanAndPrecision(mm, q.GetMean());
            }
            else if (modified && !sample.IsUniform())
            {
                // heuristic method to avoid improper messages:
                // the message's mean must be E[mean] (regardless of context) and the variance is chosen to match the posterior mean when multiplied by context
                double sampleMean = sampleMarginal.GetMean();
                if (sampleMean != mm)
                {
                    double newPrecision = (sample.MeanTimesPrecision - sampleMean * sample.Precision) / (sampleMean - mm);
                    if (newPrecision >= 0)
                    {
                        result.Precision = newPrecision;
                        result.MeanTimesPrecision = result.Precision * mm;
                    }
                }
            }
            //if (!result.IsProper()) throw new Exception();
            if (result.Precision < -0.001)
                throw new InferRuntimeException();
            if (double.IsNaN(result.MeanTimesPrecision) || double.IsNaN(result.Precision))
                throw new InferRuntimeException("result is nan");
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianOp_Laplace"]/message_doc[@name="MeanAverageConditional(Gaussian, Gaussian, Gamma, Gamma)"]/*'/>
        public static Gaussian MeanAverageConditional([SkipIfUniform] Gaussian sample, [NoInit] Gaussian mean, [Proper] Gamma precision, Gamma q)
        {
            return SampleAverageConditional(mean, sample, precision, q);
        }
    }

    [FactorMethod(typeof(Gaussian), "Sample", typeof(double), typeof(double), Default = false)]
    [FactorMethod(new string[] { "sample", "mean", "precision" }, typeof(Factor), "Gaussian", Default = false)]
    [Quality(QualityBand.Experimental)]
    public class GaussianOp_EM : GaussianOpBase
    {
        public static double LogEvidenceRatio(
            [SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision)
        {
            if (!precision.IsPointMass)
                throw new ArgumentException("precision is not a point mass");
            return GaussianOp.LogEvidenceRatio(sample, mean, precision.Point);
        }

        public static Gamma PrecisionAverageConditional([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [Proper] Gamma precision)
        {
            if (!precision.IsPointMass)
                throw new ArgumentException("precision is not a point mass");
            double r = precision.Point;
            // the factor is N(sample; mean, 1/precision)
            // the message to (sample - mean) is N(0, 1/precision)
            Gamma result = new Gamma();
            result.Shape = 1.5;
            double mx, vx, mm, vm;
            sample.GetMeanAndVariance(out mx, out vx);
            mean.GetMeanAndVariance(out mm, out vm);
            Gaussian diffPrior = new Gaussian(mx - mm, vx + vm);
            Gaussian diffLike = Gaussian.FromMeanAndPrecision(0, r);
            Gaussian diffPost = diffPrior * diffLike;
            double md, vd;
            diffPost.GetMeanAndVariance(out md, out vd);
            result.Rate = 0.5 * (vd + md * md);
            return result;
        }

        public static Gaussian SampleAverageConditional([SkipIfUniform] Gaussian mean, [Proper] Gamma precision)
        {
            if (!precision.IsPointMass)
                throw new ArgumentException("precision is not a point mass");
            return GaussianOpBase.SampleAverageConditional(mean, precision.Point);
        }

        public static Gaussian MeanAverageConditional([SkipIfUniform] Gaussian sample, [Proper] Gamma precision)
        {
            return SampleAverageConditional(sample, precision);
        }
    }

    /// <summary>
    /// This class defines specializations for the case where precision is a point mass.
    /// These methods have fewer inputs, allowing more efficient schedules.
    /// </summary>
    [FactorMethod(typeof(Gaussian), "Sample", typeof(double), typeof(double), Default = false)]
    [FactorMethod(new string[] { "sample", "mean", "precision" }, typeof(Factor), "Gaussian", Default = false)]
    [Quality(QualityBand.Preview)]
    public class GaussianOp_PointPrecision : GaussianOpBase
    {
        public static double LogEvidenceRatio(
            double sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision)
        {
            if (!precision.IsPointMass)
                throw new ArgumentException("precision is not a point mass");
            return GaussianOp.LogEvidenceRatio(sample, mean, precision.Point);
        }

        [Skip]
        public static double LogEvidenceRatio(
            [SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [SkipIfUniform] Gamma precision)
        {
            if (!precision.IsPointMass)
                throw new ArgumentException("precision is not a point mass");
            return GaussianOp.LogEvidenceRatio(sample, mean, precision.Point);
        }

        public static Gamma PrecisionAverageConditional([SkipIfUniform] Gaussian sample, [SkipIfUniform] Gaussian mean, [Proper] Gamma precision)
        {
            if (!precision.IsPointMass)
                throw new ArgumentException("precision is not a point mass");
            return GaussianOp.PrecisionAverageConditional(sample, mean, precision);
        }

        public static Gaussian SampleAverageConditional([SkipIfUniform] Gaussian mean, [Proper] Gamma precision)
        {
            if (!precision.IsPointMass)
                throw new ArgumentException("precision is not a point mass");
            return GaussianOpBase.SampleAverageConditional(mean, precision.Point);
        }

        public static Gaussian MeanAverageConditional([SkipIfUniform] Gaussian sample, [Proper] Gamma precision)
        {
            return SampleAverageConditional(sample, precision);
        }
    }
}
