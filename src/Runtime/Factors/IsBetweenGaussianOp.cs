// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Diagnostics;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "IsBetween", typeof(double), typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Mature)]
    [Buffers("logZ")]
    public static class IsBetweenGaussianOp
    {
        /// <summary>
        /// Static flag to force a proper distribution
        /// </summary>
        public static bool ForceProper = true;
        public static double LowPrecisionThreshold = 0; // 1e-8;
        public static double LargeMeanThreshold = 2.5e3;

        //-- TruncatedGaussian bounds ------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LogEvidenceRatio(bool, Gaussian, TruncatedGaussian, TruncatedGaussian)"]/*'/>
        public static double LogEvidenceRatio(bool isBetween, [RequiredArgument] Gaussian X, TruncatedGaussian lowerBound, TruncatedGaussian upperBound)
        {
            if (lowerBound.IsPointMass && upperBound.IsPointMass)
            {
                return LogEvidenceRatio(isBetween, X, lowerBound.Point, upperBound.Point);
            }
            else
                throw new NotImplementedException($"{nameof(lowerBound)} is not a point mass");
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LogEvidenceRatio(Bernoull, Gaussian, TruncatedGaussian, TruncatedGaussian)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, TruncatedGaussian lowerBound, TruncatedGaussian upperBound)
        {
            if (lowerBound.IsPointMass && upperBound.IsPointMass)
            {
                return LogEvidenceRatio(isBetween, X, lowerBound.Point, upperBound.Point);
            }
            else
                throw new NotImplementedException($"{nameof(lowerBound)} is not a point mass");
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LowerBoundAverageConditional(bool, double)"]/*'/>
        public static TruncatedGaussian LowerBoundAverageConditional(bool isBetween, double x)
        {
            if (!isBetween)
                throw new ArgumentException($"{nameof(TruncatedGaussian)} requires {nameof(isBetween)}=true", nameof(isBetween));
            return new TruncatedGaussian(0, Double.PositiveInfinity, Double.NegativeInfinity, x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="UpperBoundAverageConditional(bool, double)"]/*'/>
        public static TruncatedGaussian UpperBoundAverageConditional(bool isBetween, double x)
        {
            if (!isBetween)
                throw new ArgumentException($"{nameof(TruncatedGaussian)} requires {nameof(isBetween)}=true", nameof(isBetween));
            return new TruncatedGaussian(0, Double.PositiveInfinity, x, Double.PositiveInfinity);
        }

        public static Gaussian XAverageConditional([SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, TruncatedGaussian lowerBound, TruncatedGaussian upperBound)
        {
            if (lowerBound.IsPointMass && upperBound.IsPointMass)
            {
                return XAverageConditional(isBetween, X, lowerBound.Point, upperBound.Point);
            }
            else
                throw new NotImplementedException($"{nameof(lowerBound)} is not a point mass");
        }

        public static TruncatedGaussian LowerBoundAverageConditional([SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] TruncatedGaussian lowerBound, [RequiredArgument] TruncatedGaussian upperBound, double logZ)
        {
            if (lowerBound.IsPointMass && upperBound.IsPointMass)
            {
                Gaussian result1 = LowerBoundAverageConditional(isBetween, X, lowerBound.Gaussian, upperBound.Gaussian, logZ);
                return new TruncatedGaussian(result1, lowerBound.LowerBound, lowerBound.UpperBound);
            }
            else
                throw new NotImplementedException($"{nameof(lowerBound)} is not a point mass");
        }

        public static TruncatedGaussian UpperBoundAverageConditional([SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] TruncatedGaussian lowerBound, [RequiredArgument] TruncatedGaussian upperBound, double logZ)
        {
            if (upperBound.IsPointMass && lowerBound.IsPointMass)
            {
                Gaussian result1 = UpperBoundAverageConditional(isBetween, X, lowerBound.Gaussian, upperBound.Gaussian, logZ);
                return new TruncatedGaussian(result1, upperBound.LowerBound, upperBound.UpperBound);
            }
            else
                throw new NotImplementedException($"{nameof(lowerBound)} is not a point mass");
        }

        //-- Constant bounds --------------------------------------------------------------------------------

        /// <summary>
        /// The logarithm of the probability that L &lt;= X &lt; U.
        /// </summary>
        /// <param name="X"></param>
        /// <param name="L">Can be negative infinity.</param>
        /// <param name="U">Can be positive infinity.</param>
        /// <returns></returns>
        public static double LogProbBetween(Gaussian X, double L, double U)
        {
            if (L > U)
                throw new AllZeroException("low > high (" + L + " > " + U + ")");
            if (L == U) return double.NegativeInfinity;
            else if (double.IsPositiveInfinity(U) && double.IsNegativeInfinity(L)) return 0;
            else if (X.IsUniform())
            {
                if (double.IsNegativeInfinity(L))
                {
                    if (double.IsPositiveInfinity(U))
                        return 0.0; // always between
                    else
                        return -MMath.Ln2; // between half the time
                }
                else if (double.IsPositiveInfinity(U))
                    return -MMath.Ln2; // between half the time
                else
                    return double.NegativeInfinity; // never between two finite numbers
            }
            else if (double.IsNegativeInfinity(X.MeanTimesPrecision))
            {
                if (double.IsNegativeInfinity(L)) return 0;
                else return double.NegativeInfinity;
            }
            else if (double.IsPositiveInfinity(X.MeanTimesPrecision))
            {
                if (double.IsPositiveInfinity(U)) return 0;
                else return double.NegativeInfinity;
            }
            else if (X.IsPointMass)
            {
                return Factor.IsBetween(X.Point, L, U) ? 0.0 : double.NegativeInfinity;
            }
            else
            {
                double sqrtPrec = Math.Sqrt(X.Precision);
                double mx = X.GetMean();
                double zL = (L - mx) * sqrtPrec;
                double zU = (U - mx) * sqrtPrec;
                double diff = U - L;
                // diffs = zU - zL
                double diffs = sqrtPrec * diff;
                double deltaOverDiffs = (-zL - zU) / 2;
                double delta = diffs * deltaOverDiffs;
                // we can either compute as:  p(x <= U) - p(x <= L)
                // or:  p(x > L) - p(x > U)
                // depending on whether the bigger absolute value is negative or positive
                bool zLIsBigger = Math.Abs(zL) > Math.Abs(zU);
                bool biggerIsNegative;
                if (zLIsBigger)
                    biggerIsNegative = (zL < 0);
                else
                    biggerIsNegative = (zU < 0);
                if (biggerIsNegative)
                {
                    // compute p(x <= U) - p(x <= L)
                    // must have mx > center so delta > 0
                    if (diffs < 1e-3 || diffs < 0.7 * Math.Abs(zL))
                    {
                        // (Cdf(zU) - Cdf(zL))/N(zL) = R(zU)*exp(delta) - R(zL)
                        // (Cdf(zU) - Cdf(zL))/N(zU) = R(zU) - R(zL)*exp(-delta)
                        double expMinus1 = MMath.ExpMinus1(delta);
                        if (expMinus1 > 1e100) return MMath.NormalCdfLn(zU);
                        double rU = MMath.NormalCdfRatio(zU);
                        double drU = MMath.NormalCdfRatioDiff(zL, diffs);
                        double ZR = (drU + rU * expMinus1) / sqrtPrec;
                        // logProbL = X.GetLogProb(L)
                        double logProbL = -MMath.LnSqrt2PI + Math.Log(sqrtPrec) - zL * zL / 2;
                        return Math.Log(ZR) + logProbL;
                    }
                    else
                    {
                        // This approach loses accuracy when pu is near pl.
                        double pu = MMath.NormalCdfLn(zU); // log(p(x <= U))
                        double pl = MMath.NormalCdfLn(zL); // log(p(x <= L))
                        // since zU > zL, this is always <= 0
                        return MMath.LogDifferenceOfExp(pu, pl);
                    }
                }
                else
                {
                    // compute p(x > L) - p(x > U)
                    // must have mx <= center so delta <= 0
                    if (diffs < 1e-3 || diffs < 0.7 * Math.Abs(zU))
                    {
                        // (Cdf(-zL) - Cdf(-zU))/N(zU) = R(-zL)*exp(-delta) - R(-zU)
                        // (Cdf(-zL) - Cdf(-zU))/N(zL) = R(-zL) - R(-zU)*exp(delta)
                        double expMinus1 = MMath.ExpMinus1(-delta);
                        if (expMinus1 > 1e100) return MMath.NormalCdfLn(-zL);
                        double rU = MMath.NormalCdfRatio(-zL);
                        double drU = MMath.NormalCdfRatioDiff(-zU, diffs);
                        double ZR = (drU + rU * expMinus1) / sqrtPrec;
                        // logProbU = X.GetLogProb(U)
                        double logProbU = -MMath.LnSqrt2PI + Math.Log(sqrtPrec) - zU * zU / 2;
                        return Math.Log(ZR) + logProbU;
                    }
                    else
                    {
                        double pl = MMath.NormalCdfLn(-zL); // log(p(x > L))
                        double pu = MMath.NormalCdfLn(-zU); // log(p(x > U))
                        return MMath.LogDifferenceOfExp(pl, pu);
                    }
                }
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LogAverageFactor(Bernoulli, Gaussian, double, double)"]/*'/>
        public static double LogAverageFactor(Bernoulli isBetween, [RequiredArgument] Gaussian x, double lowerBound, double upperBound)
        {
#if true
            Bernoulli to_isBetween = IsBetweenAverageConditional(x, lowerBound, upperBound);
            return to_isBetween.GetLogAverageOf(isBetween);
#else
            if (isBetween.LogOdds == 0.0) return -MMath.Ln2;
            else {
                double logitProbBetween = MMath.LogitFromLog(LogProbBetween(x, lowerBound, upperBound));
                return Bernoulli.LogProbEqual(isBetween.LogOdds, logitProbBetween);
            }
#endif
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LogEvidenceRatio(bool, Gaussian, double, double)"]/*'/>
        public static double LogEvidenceRatio(bool isBetween, [RequiredArgument] Gaussian x, double lowerBound, double upperBound)
        {
            return LogAverageFactor(isBetween, x, lowerBound, upperBound);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LogAverageFactor(bool, Gaussian, double, double)"]/*'/>
        public static double LogAverageFactor(bool isBetween, [RequiredArgument] Gaussian x, double lowerBound, double upperBound)
        {
            return LogAverageFactor(Bernoulli.PointMass(isBetween), x, lowerBound, upperBound);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli, Gaussian, double, double)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli isBetween, Gaussian x, double lowerBound, double upperBound)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="IsBetweenAverageConditional(Gaussian, double, double)"]/*'/>
        public static Bernoulli IsBetweenAverageConditional([RequiredArgument] Gaussian X, double lowerBound, double upperBound)
        {
            Bernoulli result = new Bernoulli();
            result.SetLogProbTrue(LogProbBetween(X, lowerBound, upperBound));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="XAverageConditional(Bernoulli, Gaussian, double, double)"]/*'/>
        public static Gaussian XAverageConditional([SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, double lowerBound, double upperBound)
        {
            // TM: RequiredArgument is included to get good schedules
            // Note that SkipIfUniform would not be correct in this case
            if (upperBound == lowerBound)
            {
                if (isBetween.IsPointMass && isBetween.Point)
                    return Gaussian.PointMass(lowerBound);
                else
                    return Gaussian.Uniform();
            }
            if (double.IsPositiveInfinity(upperBound) && double.IsNegativeInfinity(lowerBound)) return Gaussian.Uniform();
            if (X.IsPointMass)
            {
                if (isBetween.IsPointMass)
                {
                    if (isBetween.Point)
                    {
                        if (X.Point < lowerBound) return Gaussian.PointMass(lowerBound);
                        else if (X.Point > upperBound) return Gaussian.PointMass(upperBound);
                        else return Gaussian.Uniform();
                    }
                    else
                    {
                        if (X.Point >= lowerBound && X.Point <= upperBound)
                        {
                            double center = MMath.Average(lowerBound, upperBound);
                            if (X.Point == center)
                            {
                                return Gaussian.Uniform();
                            }
                            else if (X.Point < center)
                            {
                                return Gaussian.PointMass(lowerBound);
                            }
                            else
                            {
                                return Gaussian.PointMass(upperBound);
                            }
                        }
                        else return Gaussian.Uniform();
                    }
                }
                return Gaussian.Uniform();
            }
            else if (X.IsUniform())
            {
                if (Double.IsInfinity(lowerBound) || Double.IsInfinity(upperBound) ||
                    !Double.IsPositiveInfinity(isBetween.LogOdds))
                {
                    return Gaussian.Uniform();
                }
                else
                {
                    double diff = upperBound - lowerBound;
                    return Gaussian.FromMeanAndVariance((lowerBound + upperBound) / 2, diff * diff / 12);
                }
            }
            else if (double.IsNegativeInfinity(X.MeanTimesPrecision))
            {
                if (isBetween.IsPointMass)
                {
                    if (isBetween.Point)
                    {
                        if (double.IsNegativeInfinity(lowerBound)) return Gaussian.Uniform();
                        else return Gaussian.PointMass(lowerBound);
                    }
                    else
                    {
                        if (!double.IsNegativeInfinity(lowerBound)) return Gaussian.Uniform();
                        else return Gaussian.PointMass(lowerBound);
                    }
                }
                else return Gaussian.Uniform();
            }
            else if (double.IsPositiveInfinity(X.MeanTimesPrecision))
            {
                if (isBetween.IsPointMass)
                {
                    if (isBetween.Point)
                    {
                        if (double.IsPositiveInfinity(upperBound)) return Gaussian.Uniform();
                        else return Gaussian.PointMass(lowerBound);
                    }
                    else
                    {
                        if (!double.IsPositiveInfinity(upperBound)) return Gaussian.Uniform();
                        else return Gaussian.PointMass(lowerBound);
                    }
                }
                else return Gaussian.Uniform();
            }
            else
            {
                // X is not a point mass or uniform
                double d_p = 2 * isBetween.GetProbTrue() - 1;
                double mx = X.GetMean();
                double center = MMath.Average(lowerBound, upperBound);
                double diff = upperBound - lowerBound;
                if (d_p == 1.0)
                {
                    double sqrtPrec = Math.Sqrt(X.Precision);
                    double diffs = diff * sqrtPrec;
                    // X.Prob(U) = X.Prob(L) * exp(delta)
                    // delta = diff*(mx - center)*prec = -diff*(zL+zU)/2*sqrtPrec = -diffs*(zL+zU)/2 = -(zU-zL)*(zL+zU)/2
                    // X.Prob(L) * exp(delta/2) = -l^2/2 -(u^2-l^2)/4 = -u^2/4 - l^2/4 = -(u^2+l^2)/4
                    // zU - zL = diffs
                    bool flip = false;
                    double zL = (lowerBound - mx) * sqrtPrec;
                    if (zL < double.MinValue && double.IsPositiveInfinity(upperBound))
                        return Gaussian.Uniform();
                    double zU = (upperBound - mx) * sqrtPrec;
                    if (zU > double.MaxValue && double.IsNegativeInfinity(lowerBound))
                        return Gaussian.Uniform();
                    double deltaOverDiffs = (-zL - zU) / 2;
                    if (deltaOverDiffs < 0)
                    {
                        // Flip symmetry:
                        // X' = 2*center - X
                        // should flip the mean of the message, leaving message precision unchanged.
                        // Therefore we can assume mx > center, and flip at the end.
                        // After flip, lowerBound - (2*center - mx) = mx - (lowerBound - 2*center) = mx - upperBound
                        zL = (mx - upperBound) * sqrtPrec;
                        zU = (mx - lowerBound) * sqrtPrec;
                        deltaOverDiffs = -deltaOverDiffs;
                        flip = true;
                    }
                    if (double.IsNaN(zL)) throw new Exception($"{nameof(zL)} is NaN when {nameof(X)}={X}, {nameof(lowerBound)}={lowerBound:r}, {nameof(upperBound)}={upperBound:r}");
                    if (double.IsNaN(zU)) throw new Exception($"{nameof(zU)} is NaN when {nameof(X)}={X}, {nameof(lowerBound)}={lowerBound:r}, {nameof(upperBound)}={upperBound:r}");
                    if (zU > 3.5)
                    {
                        // When zU > 0, X.GetMean() is inside the constraints and 
                        // zU is the distance to the closest boundary, measured in units of standard deviations of the prior.
                        // If zU > 10, then the boundaries are many standard deviations away, and therefore have little effect.
                        // In this case, alpha and beta will be very small.
                        double logZ = LogAverageFactor(isBetween, X, lowerBound, upperBound);
                        if (logZ == 0) return Gaussian.Uniform();
                        double logPhiL = X.GetLogProb(lowerBound);
                        double alphaL = d_p * Math.Exp(logPhiL - logZ);
                        double logPhiU = X.GetLogProb(upperBound);
                        double alphaU = d_p * Math.Exp(logPhiU - logZ);
                        double alphaX = alphaL - alphaU;
                        double betaX = alphaX * alphaX;
                        double betaU;
                        if (alphaU == 0.0) betaU = 0;
                        else betaU = (upperBound * X.Precision - X.MeanTimesPrecision) * alphaU;   // (upperBound - mx) / vx * alphaU;
                        double betaL;
                        if (alphaL == 0.0) betaL = 0;
                        else betaL = (X.MeanTimesPrecision - lowerBound * X.Precision) * alphaL; // -(lowerBound - mx) / vx * alphaL;
                        if (Math.Abs(betaU) > Math.Abs(betaL)) betaX = (betaX + betaL) + betaU;
                        else betaX = (betaX + betaU) + betaL;
                        return GaussianOp.GaussianFromAlphaBeta(X, alphaX, betaX, ForceProper);
                    }
                    // in the flipped case, (2*center-x - center) = (center - x) = abs(x - center)
                    //double delta = diff * Math.Abs(X.MeanTimesPrecision - X.Precision * center);
                    // this formula is more accurate than above
                    double delta = diffs * deltaOverDiffs;
                    double deltaSqrtVx = diff * deltaOverDiffs; // delta / sqrtPrec
                    if (delta < 0) throw new Exception($"{nameof(delta)} ({delta}) < 0");
                    if (delta < MMath.Ulp1 && (mx <= lowerBound || mx >= upperBound))
                    {
                        double variance = diff * diff / 12;
                        return new Gaussian(center, variance) / X;
                    }
                    double expMinus1 = MMath.ExpMinus1(delta);
                    double expMinus1RatioMinus1RatioMinusHalf = MMath.ExpMinus1RatioMinus1RatioMinusHalf(delta);
                    // expMinus1RatioMinus1 = expMinus1Ratio - 1;
                    double expMinus1RatioMinus1 = delta * (0.5 + expMinus1RatioMinus1RatioMinusHalf);
                    double expMinus1Ratio = 1 + expMinus1RatioMinus1;
                    // Z/X.Prob(U)*sqrtPrec = NormalCdfRatio(zU) - NormalCdfRatio(zL)*X.Prob(L)/X.Prob(U) 
                    //                      = NormalCdfRatio(zU) - NormalCdfRatio(zL)*exp(-delta)
                    //                      = (NormalCdfRatio(zU) - NormalCdfRatio(zL)) + NormalCdfRatio(zL)*(1-exp(-delta))
                    //                      = (NormalCdfRatio(zU) - NormalCdfRatio(zL)) + NormalCdfRatio(zL)/cU
                    // Z/(X.Prob(L)-X.Prob(U)) = Z/X.Prob(U)*(-cU) = -((NormalCdfRatio(zU) - NormalCdfRatio(zL))*cU + NormalCdfRatio(zL))/sqrtPrec
                    // MMath.NormalCdf(zU) =approx MMath.NormalCdf(z) + (zU-z)*N(z;0,1) + 0.5*(zU-z)^2*(-z)*N(z;0,1) + 1/6*(zU-z)^3*...
                    // MMath.NormalCdf(zL) =approx MMath.NormalCdf(z) + (zL-z)*N(z;0,1)
                    // Z =approx (zU-zL)*N(z;0,1)
                    // Z/(X.Prob(L)-X.Prob(U)) =approx (zU-zL)/(exp(-delta/2) - exp(delta/2))/sqrtPrec =approx (zU-zL)/(-delta)/sqrtPrec = 2/(zL+zU)/sqrtPrec
                    // Z/(X.Prob(L)-X.Prob(U)) = Z/X.Prob(L)/(1-exp(delta))
                    // Z = MMath.NormalCdf(zU) - MMath.NormalCdf(zL)
                    // Z/(X.Prob(L) - X.Prob(U)) = -NormalCdf(zU)/X.Prob(U)*cU - NormalCdf(zL)/X.Prob(L)*cL
                    //                           = -(NormalCdfRatio(zU)*cU + NormalCdfRatio(zL)*cL)/sqrtPrec
                    // since we know that |cU| > |cL|, we substitute cU = 1-cL
                    double rU = MMath.NormalCdfRatio(zU);
                    double r1U = MMath.NormalCdfMomentRatio(1, zU);
                    double r3U = MMath.NormalCdfMomentRatio(3, zU) * 6;
                    if (zU < -(MMath.Sqrt2 * MMath.Sqrt3) / MMath.SqrtUlp1)
                    {
                        // in this regime, rU = -1/zU, r1U = rU*rU
                        // because rU = -1/zU + 1/zU^3 + ...
                        // and r1U = 1/zU^2 - 3/zU^4 + ...
                        // The second term is smaller by a factor of 3/zU^2.
                        // The threshold satisfies 3/zU^2 < 0.5 * ulp(1) or zU < -sqrt(6 / ulp(1))
                        if (expMinus1 > 1e100)
                        {
                            double invzUs = 1 / (zU * sqrtPrec);
                            double mp2;
                            // r1U/rU = -1/zU = rU
                            if (!flip) mp2 = upperBound + invzUs;
                            else mp2 = lowerBound - invzUs;
                            double vp2 = invzUs * invzUs;
                            return new Gaussian(mp2, vp2) / X;
                        }
                        else
                        {
                            double mp2;
                            if (delta < 10)
                            {
                                double offset = (expMinus1RatioMinus1 * diff / 2 + delta * expMinus1RatioMinus1RatioMinusHalf / (zU * sqrtPrec)) / expMinus1Ratio;
                                if (flip) offset = -offset;
                                mp2 = center + offset;
                            }
                            else
                            {
                                // center = upperBound - diff/2 = lowerBound + diff/2
                                double offset = (-diff / 2 + delta * expMinus1RatioMinus1RatioMinusHalf / (zU * sqrtPrec)) / expMinus1Ratio;
                                if (flip)
                                {
                                    mp2 = lowerBound - offset;
                                }
                                else
                                {
                                    mp2 = upperBound + offset;
                                }
                            }
                            double c = expMinus1RatioMinus1RatioMinusHalf - delta / 2 * (expMinus1RatioMinus1RatioMinusHalf + 1);
                            double expMinus1RatioSqr = expMinus1Ratio * expMinus1Ratio;
                            double diffsSqrOver4 = diffs * diffs / 4;
                            // Abs is needed to avoid some 32-bit oddities.
                            double prec2 = expMinus1RatioSqr * X.Precision /
                                Math.Abs(r1U * expMinus1 * expMinus1RatioMinus1RatioMinusHalf
                                + rU * diffs * c
                                + diffsSqrOver4);
                            if (prec2 > double.MaxValue || diffsSqrOver4 < 1e-308)
                            {
                                // same as above but divide top and bottom by X.Precision, to avoid overflow
                                prec2 = expMinus1RatioSqr /
                                    Math.Abs(r1U / X.Precision * expMinus1 * expMinus1RatioMinus1RatioMinusHalf
                                    + rU / sqrtPrec * diff * c
                                    + diff * diff / 4);
                            }
                            return Gaussian.FromMeanAndPrecision(mp2, prec2) / X;
                        }
                    }
                    if (expMinus1 > 1e100) // cL < 1e-100, delta < 254
                    {
                        // if cL==0 then alphaX = -sqrtPrec / rU, betaX = alphaX^2 * r1U = prec * r1U/rU^2
                        //   beta/(prec - beta) = r1U/rU^2 / (1 - r1U/rU^2) = 1/(rU^2/r1U - 1)
                        // posterior mean = m - sqrt(v)/rU = sqrt(v)*(m*sqrtPrec - 1/rU) = sqrt(v)*(-zU + U*sqrtPrec - 1/rU)
                        //                = U - sqrt(v)*(zU + 1/rU) = U - sqrt(v)*r1U/rU
                        // posterior variance = v - v*r1U/rU^2
                        // in the flipped case:
                        // 2*center - mp2 = (2*center - upperBound) + r1U / rU / sqrtPrec = lowerBound + r1U / rU / sqrtPrec
                        double mp2;
                        if (!flip) mp2 = upperBound - r1U / rU / sqrtPrec;
                        else mp2 = lowerBound + r1U / rU / sqrtPrec;
                        // This approach loses accuracy when r1U/(rU*rU) < 1e-3, which is zU > 3.5
                        if (zU > 3.5) throw new Exception("zU > 3.5");
                        double prec2 = X.Precision * (rU * rU / NormalCdfRatioSqrMinusDerivative(zU, rU, r1U, r3U));
                        //Console.WriteLine($"z = {zU:r} r = {rU:r} r1 = {r1U:r} r1U/rU = {r1U / rU:r} r1U/rU/sqrtPrec = {r1U / rU / sqrtPrec:r} sqrtPrec = {sqrtPrec:r} mp = {mp2:r}");
                        return Gaussian.FromMeanAndPrecision(mp2, prec2) / X;
                    }
                    // TODO: compute these more efficiently
                    double rL = MMath.NormalCdfRatio(zL);
                    double r1L = MMath.NormalCdfMomentRatio(1, zL);
                    double r2L = MMath.NormalCdfMomentRatio(2, zL) * 2;
                    double r2U = MMath.NormalCdfMomentRatio(2, zU) * 2;
                    double r3L = MMath.NormalCdfMomentRatio(3, zL) * 6;
                    double drU, drU2, drU3, dr1U, dr1U2, dr1U3, dr2U, dr2U2;
                    if (diffs < Math.Abs(zL) * 0.7 || diffs <= 9.9)
                    {
                        drU3 = MMath.NormalCdfRatioDiff(zL, diffs, 3);
                        //drU2 = MMath.NormalCdfRatioDiff(zL, diffs, 2);
                        drU2 = diffs * (r2L / 2 + drU3);
                        //drU = MMath.NormalCdfRatioDiff(zL, diffs);
                        drU = diffs * (r1L + drU2);
                        dr1U3 = MMath.NormalCdfMomentRatioDiff(1, zL, diffs, 3);
                        //dr1U2 = MMath.NormalCdfMomentRatioDiff(1, zL, diffs, 2);
                        dr1U2 = diffs * (r3L / 2 + dr1U3);
                        //dr1U = MMath.NormalCdfMomentRatioDiff(1, zL, diffs);
                        dr1U = diffs * (r2L + dr1U2);
                        dr2U2 = MMath.NormalCdfMomentRatioDiff(2, zL, diffs, 2);
                        //dr2U = MMath.NormalCdfMomentRatioDiff(2, zL, diffs);
                        dr2U = diffs * (r3L + dr2U2);
                    }
                    else
                    {
                        drU = rU - rL;
                        drU2 = drU / diffs - r1L;
                        drU3 = drU2 / diffs - r2L / 2;
                        // dr1U = diffs*(r2L + dr1U2)
                        dr1U = r1U - r1L;
                        // dr1U2 = diffs*(r3L/2 + dr1U3)
                        dr1U2 = dr1U / diffs - r2L;
                        // dr1U3 = diffs*r4L/6 + ...
                        dr1U3 = dr1U2 / diffs - r3L / 2;
                        dr2U = r2U - r2L;
                        dr2U2 = dr2U / diffs - r3L;
                    }
                    double alphaXcLprecDiff = 1 / (rU * deltaOverDiffs * expMinus1Ratio + r1L + drU2);
                    double mp;
                    if (delta < 10)
                    {
                        // when delta is small, it is more accurate to compute the mean as an offset from the center.
                        //betaX = alphaX * alphaX * (r1U - cL * (dr1U + cU * diffs * drU));
                        // -(zL+zU)/2 = delta/diffs
                        // posterior mean = m + v*alpha = v*(m*prec + alpha) = center + v*((m-center)*prec + alpha)
                        //                = center + v*(delta/diff + alpha)
                        //                = center + sqrt(v)*(-(zL+zU)/2*Z + X.Prob(L)-X.Prob(U))/Z
                        //                = center + sqrt(v)*(delta/diffs*Z + X.Prob(L)*(1-exp(delta)))/Z
                        //                = center + sqrt(v)*(Z/diffs + X.Prob(L)*(1-exp(delta))/delta)*delta/Z
                        double numer = (drU3 - dr1U2 / 2) * expMinus1Ratio + (r1L + drU2) * (
                            (-zL) / 2 * expMinus1RatioMinus1
                            - deltaOverDiffs * expMinus1RatioMinus1RatioMinusHalf
                            + diffs / 4);
                        if (flip) numer = -numer;
                        mp = center + numer * deltaSqrtVx * alphaXcLprecDiff;
                    }
                    else
                    {
                        // when delta is large, it is more accurate to compute the mean as an offset from an endpoint.
                        double numerLargezL8 = (drU3 - dr1U2 / 2 - drU / 2 - r2L / 2 + drU2 * (-zL) / 2) * expMinus1Ratio
                            + (r1L + drU2) * (zL / 2
                            - deltaOverDiffs * expMinus1RatioMinus1RatioMinusHalf
                            - 0.5 / deltaOverDiffs
                            + diffs / 4);
                        if (flip)
                        {
                            mp = lowerBound - numerLargezL8 * deltaSqrtVx * alphaXcLprecDiff;
                        }
                        else
                        {
                            mp = upperBound + numerLargezL8 * deltaSqrtVx * alphaXcLprecDiff;
                        }
                    }
                    // double vp = (1 - betaX / X.Precision) / X.Precision;
                    double drU2r1U = NormalCdfRatioSqrMinusDerivative(zU, rU, r1U, r3U);
                    double qOverPrec =
                        deltaSqrtVx * deltaSqrtVx * (expMinus1Ratio * deltaOverDiffs * expMinus1RatioMinus1RatioMinusHalf * drU2r1U / diffs + (r2U / diffs - r1U / 2) * expMinus1RatioMinus1 / 2)
                        + deltaSqrtVx * deltaSqrtVx * (dr2U / diffs - r1U / 2) / 2
                        + rU * expMinus1 / X.Precision * (-(rU / 2 + (r1L + drU2) * zL) / 2 * delta - r1L / 2 * diffs / 2 + drU2 / diffs * delta + drU3)
                        + deltaSqrtVx * expMinus1RatioMinus1 / sqrtPrec * (dr2U / diffs - dr1U2 / diffs - r1U / 2)
                        - deltaSqrtVx / 2 * diff / 2 * r2L
                        + deltaSqrtVx / sqrtPrec / 2 * (-dr1U - drU2 + 2 * dr2U2 - 2 * dr1U3)
                        + (r1L + drU2) * (deltaSqrtVx * deltaSqrtVx * ((rU / diffs - 1) * expMinus1RatioMinus1RatioMinusHalf - rL * deltaOverDiffs / 2 + drU2 / 2)
                            + diff * (drU3 * (delta + 1) / sqrtPrec - diff / 4 + r2L / 2 * deltaSqrtVx
                            + rL / 2 * diffs / 2 * diff / 2
                            - rL / 2 * delta * deltaSqrtVx / 2));
                    if (delta == 0) // avoid 0*infinity
                        qOverPrec = (r1L + drU2) * diff * (drU3 / sqrtPrec - diff / 4 + rL / 2 * diffs / 2 * diff / 2);
                    double vp = qOverPrec * alphaXcLprecDiff * alphaXcLprecDiff;
                    if (double.IsNaN(qOverPrec) || 1 / vp < X.Precision) return Gaussian.FromMeanAndPrecision(mp, MMath.NextDouble(X.Precision)) / X;
                    return new Gaussian(mp, vp) / X;
                }
                else
                {
                    double logZ = LogAverageFactor(isBetween, X, lowerBound, upperBound);
                    Gaussian GetPointMessage()
                    {
                        if (mx == center)
                        {
                            // The posterior is two point masses.
                            // Compute the moment-matched Gaussian and divide.
                            Gaussian result = Gaussian.FromMeanAndVariance(center, diff * diff / 4);
                            result.SetToRatio(result, X, ForceProper);
                            return result;
                        }
                        else if (mx < center)
                        {
                            return Gaussian.PointMass(lowerBound);
                        }
                        else
                        {
                            return Gaussian.PointMass(upperBound);
                        }
                    }
                    if (d_p == -1.0 && logZ < double.MinValue) return GetPointMessage();
                    double logPhiL = X.GetLogProb(lowerBound);
                    double alphaL = d_p * Math.Exp(logPhiL - logZ);
                    double logPhiU = X.GetLogProb(upperBound);
                    double alphaU = d_p * Math.Exp(logPhiU - logZ);
                    double alphaX = alphaL - alphaU;
                    double betaU;
                    if (alphaU == 0.0) betaU = 0;
                    else betaU = (upperBound * X.Precision - X.MeanTimesPrecision - alphaX) * alphaU;   // (upperBound - mx) / vx * alphaU;
                    double betaL;
                    if (alphaL == 0.0) betaL = 0;
                    else betaL = (X.MeanTimesPrecision - lowerBound * X.Precision + alphaX) * alphaL; // -(lowerBound - mx) / vx * alphaL;
                    double betaX = betaU + betaL;
                    if (Math.Abs(betaX) > double.MaxValue)
                    {
                        if (d_p == -1.0) return GetPointMessage();
                        else return Gaussian.Uniform();
                    }
                    return GaussianOp.GaussianFromAlphaBeta(X, alphaX, betaX, ForceProper);
                }
            }
        }

        /// <summary>
        /// Returns (NormalCdfRatio(z)^2 - NormalCdfMomentRatio(1,z))
        /// </summary>
        /// <param name="z">A number &lt;= 20</param>
        /// <param name="r">NormalCdfRatio(z)</param>
        /// <param name="r1">NormalCdfMomentRatio(1,z)</param>
        /// <param name="r3">NormalCdfMomentRatio(3,z)*6</param>
        /// <returns></returns>
        public static double NormalCdfRatioSqrMinusDerivative(double z, double r, double r1, double r3)
        {
            if (z > 20) throw new ArgumentOutOfRangeException(nameof(z), "z > 20");
            else if (z > -1e-4)
            {
                return r * r - r1;
            }
            else if (z < -1e77) return 0;
            else
            {
                // r1 = z*r + 1
                // r = (r1-1)/z
                // r1 = (r2-r)/z = r2/z - r1/z^2 + 1/z^2
                // r1 = (r2/z + 1/z^2)/(1 + 1/z^2) = (r2*z + 1)/(1 + z^2)
                // r3 = z*r2 + 2*r1
                // r*r - r1 = (r*r + r*r*z^2 - r2*z - 1)/(1 + z^2) = (r*r + (r1-1)^2 - (r3 - 2*r1) - 1)/(1+z^2)
                return (r * r + r1 * r1 - r3) / (1 + z * z);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="XAverageConditional(bool, Gaussian, double, double)"]/*'/>
        public static Gaussian XAverageConditional(bool isBetween, [RequiredArgument] Gaussian X, double lowerBound, double upperBound)
        {
            return XAverageConditional(Bernoulli.PointMass(isBetween), X, lowerBound, upperBound);
        }

        //-- Random bounds --------------------------------------------------------------------------------

        // Compute the mean and variance of (X-L) and (U-X)
        // sqrtomr2 is sqrt(1-r*r) with high accuracy.
        internal static void GetDiffMeanAndVariance(Gaussian X, Gaussian L, Gaussian U, out double yl, out double yu, out double r,
            out double sqrtomr2,
            out double invSqrtVxl,
            out double invSqrtVxu)
        {
            double mx, vx, ml, vl, mu, vu;
            X.GetMeanAndVariance(out mx, out vx);
            L.GetMeanAndVariance(out ml, out vl);
            U.GetMeanAndVariance(out mu, out vu);
            if (X.IsPointMass && L.IsPointMass)
            {
                invSqrtVxl = Double.PositiveInfinity;
                yl = (X.Point >= L.Point) ? Double.PositiveInfinity : Double.NegativeInfinity;
            }
            else if (L.IsUniform())
            {
                invSqrtVxl = 0.0;
                yl = 0;
            }
            else
            {
                invSqrtVxl = 1.0 / Math.Sqrt(vx + vl);
                yl = (mx - ml) * invSqrtVxl;
            }
            if (X.IsPointMass && U.IsPointMass)
            {
                invSqrtVxu = Double.PositiveInfinity;
                yu = (X.Point < U.Point) ? Double.PositiveInfinity : Double.NegativeInfinity;
            }
            else if (U.IsUniform())
            {
                invSqrtVxu = 0.0;
                yu = 0;
            }
            else
            {
                invSqrtVxu = 1.0 / Math.Sqrt(vx + vu);
                yu = (mu - mx) * invSqrtVxu;
            }
            if (X.IsPointMass)
            {
                r = 0.0;
                sqrtomr2 = 1;
            }
            else
            {
                //r = -vx * invSqrtVxl * invSqrtVxu;
                // This formula ensures r is between -1 and 1.
                //r = -1 / Math.Sqrt(1 + vl / vx) / Math.Sqrt(1 + vu / vx);
                //r = -vx / Math.Sqrt((vx + vl) * (vx + vu));
                r = Math.Max(-1, Math.Min(1, -vx / Math.Sqrt(vx + vl) / Math.Sqrt(vx + vu)));
                if (r < -1 || r > 1)
                    throw new Exception("Internal: r is outside [-1,1]");
                double omr2 = ((vl + vu) * vx + vl * vu) / (vx + vl) / (vx + vu);
                sqrtomr2 = Math.Sqrt(omr2);
            }
        }

        /// <summary>
        /// The logarithm of the probability that L &lt;= X &lt; U.
        /// </summary>
        /// <param name="X"></param>
        /// <param name="L">Can be uniform.  Can be negative infinity.</param>
        /// <param name="U">Can be uniform.  Can be positive infinity.</param>
        /// <returns>A real number between -infinity and 0.</returns>
        public static double LogProbBetween(Gaussian X, Gaussian L, Gaussian U)
        {
            if (L.IsPointMass && U.IsPointMass)
                return LogProbBetween(X, L.Point, U.Point);
            if (X.IsUniform())
            {
                if (L.IsPointMass && Double.IsNegativeInfinity(L.Point))
                {
                    if (U.IsPointMass && Double.IsPositiveInfinity(U.Point))
                        return 0.0; // always between
                    else
                        return -MMath.Ln2; // between half the time
                }
                else if (U.IsPointMass && Double.IsPositiveInfinity(U.Point))
                    return -MMath.Ln2; // between half the time
                else if (L.IsUniform() || U.IsUniform())
                {
                    return -2 * MMath.Ln2; // log(0.25)
                }
                else
                {
                    return Double.NegativeInfinity;
                }
            }
            else if (!X.IsProper())
            {
                return double.NegativeInfinity;
            }
            else
            {
                // at this point, X is not uniform
                double yl, yu, r, sqrtomr2;
                GetDiffMeanAndVariance(X, L, U, out yl, out yu, out r, out sqrtomr2, out _, out _);
                //Trace.WriteLine($"yl={yl:r} yu={yu:r} r={r:r} sqrtomr2={sqrtomr2:r}");
                var prob = MMath.NormalCdf(yl, yu, r, sqrtomr2);
                double logProb = prob.Log();
                if (logProb > 0)
                    throw new Exception("LogProbBetween is positive");
                return logProb;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LogZ(Bernoulli, Gaussian, Gaussian, Gaussian)"]/*'/>
        [Fresh]
        public static double LogZ(Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            return LogAverageFactor(isBetween, X, lowerBound, upperBound);
        }

        [Fresh]
        public static double LogZ(Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] TruncatedGaussian lowerBound, [RequiredArgument] TruncatedGaussian upperBound)
        {
            if (lowerBound.IsPointMass && upperBound.IsPointMass)
                return LogAverageFactor(isBetween, X, lowerBound.Point, upperBound.Point);
            else
                throw new NotImplementedException("lowerBound is not a point mass");
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LogAverageFactor(Bernoulli, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static double LogAverageFactor(
            Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            if (isBetween.LogOdds == 0.0)
                return -MMath.Ln2;
            else
            {
                double logProb = LogProbBetween(X, lowerBound, upperBound);
                double logitProbBetween = MMath.LogitFromLog(logProb);
                return Bernoulli.LogProbEqual(isBetween.LogOdds, logitProbBetween);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LogEvidenceRatio(bool, Gaussian, Gaussian, Gaussian, double)"]/*'/>
        public static double LogEvidenceRatio(bool isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound, double logZ)
        {
            //return LogAverageFactor(Bernoulli.PointMass(isBetween), X, lowerBound, upperBound);
            return logZ;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli, Gaussian, Gaussian, Gaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="IsBetweenAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Bernoulli IsBetweenAverageConditional([SkipIfUniform] Gaussian X, [SkipIfUniform] Gaussian lowerBound, [SkipIfUniform] Gaussian upperBound)
        {
            Bernoulli result = new Bernoulli();
            result.SetLogProbTrue(LogProbBetween(X, lowerBound, upperBound));
            return result;
        }

        public static Gaussian LowerBoundAverageConditional_Slow(
            [SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            return LowerBoundAverageConditional(isBetween, X, lowerBound, upperBound, LogZ(isBetween, X, lowerBound, upperBound));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LowerBoundAverageConditional(Bernoulli, Gaussian, Gaussian, Gaussian, double)"]/*'/>
        [SkipIfAllUniform("X", "lowerBound")]
        [SkipIfAllUniform("X", "upperBound")]
        public static Gaussian LowerBoundAverageConditional(
            [SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound,
            double logZ)
        {
            Gaussian result = new Gaussian();
            if (isBetween.IsUniform())
                return result;
            if (X.Precision == 0)
            {
                if (upperBound.IsUniform() || lowerBound.IsUniform())
                {
                    result.SetToUniform();
                }
                else if (isBetween.IsPointMass && isBetween.Point)
                {
                    double ml, vl, mu, vu;
                    lowerBound.GetMeanAndVariance(out ml, out vl);
                    upperBound.GetMeanAndVariance(out mu, out vu);
                    double vlu = vl + vu;
                    double dmul = mu - ml;
                    double alpha = Math.Exp(Gaussian.GetLogProb(ml, mu, vlu) - MMath.NormalCdfLn(dmul / Math.Sqrt(vlu)));
                    double alphaU = 1.0 / (dmul + vlu * alpha);
                    double betaU = alphaU * (alphaU - alpha);
                    result.SetMeanAndVariance(ml - vl * alphaU, vl - vl * vl * betaU);
                    result.SetToRatio(result, lowerBound);
                }
                else
                    throw new NotImplementedException();
            }
            else if (lowerBound.IsUniform())
            {
                if (isBetween.IsPointMass && !isBetween.Point)
                {
                    // X < lowerBound < upperBound
                    // lowerBound is not a point mass so lowerBound==X is impossible
                    return XAverageConditional(Bernoulli.PointMass(true), lowerBound, X, upperBound, logZ);
                }
                else
                {
                    result.SetToUniform();
                }
            }
            else
            {
                double d_p = 2 * isBetween.GetProbTrue() - 1;
                if (lowerBound.IsPointMass)
                {
                    if (double.IsNegativeInfinity(lowerBound.Point)) return Gaussian.Uniform();
                    if (X.IsPointMass)
                    {
                        if (lowerBound.Point < X.Point) return Gaussian.Uniform();
                        else return X;
                    }
                    if (upperBound.IsPointMass)
                    {
                        // r = -1 case
                        // X is not uniform or point mass
                        // f(L) = d_p (NormalCdf((U-mx)*sqrtPrec) - NormalCdf((L-mx)*sqrtPrec)) + const.
                        // dlogf = -d_p N(L;mx,vx)/f
                        // ddlogf = -dlogf^2 + dlogf*(mx-L)/vx
                        double L = lowerBound.Point;
                        double U = upperBound.Point;
                        double mx = X.GetMean();
                        if (mx < L && d_p == 1)
                        {
                            // Z = d_p (MMath.NormalCdf(sqrtPrec*(mx-L)) - MMath.NormalCdf(sqrtPrec*(mx-U)))
                            // Z/X.GetProb(L)*sqrtPrec = MMath.NormalCdfRatio(sqrtPrec*(mx-L)) - MMath.NormalCdfRatio(sqrtPrec*(mx-U))*X.GetProb(U)/X.GetProb(L)
                            // X.GetProb(U)/X.GetProb(L) = Math.Exp(X.MeanTimesPrecision*(U-L) - 0.5*(U*U - L*L)*X.Precision) =approx 0
                            double sqrtPrec = Math.Sqrt(X.Precision);
                            double mxL = sqrtPrec * (mx - L);
                            double LCdfRatio = MMath.NormalCdfRatio(mxL);
                            double UCdfRatio = MMath.NormalCdfRatio(sqrtPrec * (mx - U));
                            double UPdfRatio = Math.Exp((U - L) * (X.MeanTimesPrecision - 0.5 * (U + L) * X.Precision));
                            double CdfRatioDiff = LCdfRatio - UCdfRatio * UPdfRatio;
                            double dlogf = -d_p / CdfRatioDiff * sqrtPrec;
                            // ddlogf = dlogf * (d_p * sqrtPrec / CdfRatioDiff + sqrtPrec*mxL)
                            //        = dlogf * sqrtPrec * (d_p / CdfRatioDiff + mxL)
                            //        = dlogf * sqrtPrec * (d_p + mxL * CdfRatioDiff) / CdfRatioDiff
                            //        = dlogf * sqrtPrec * (d_p - 1 - mxL * UCdfRatio * UPdfRatio + NormalCdfMomentRatio(1,mxL)) / CdfRatioDiff
                            //double ddlogf = dlogf * (-dlogf + (X.MeanTimesPrecision - L * X.Precision));
                            double ddlogf = -dlogf * dlogf * (d_p - 1 + mxL * (CdfRatioDiff - LCdfRatio) + MMath.NormalCdfMomentRatio(1, mxL)) / d_p;
                            return Gaussian.FromDerivatives(L, dlogf, ddlogf, ForceProper);
                            // this is equivalent
                            // L - dlogf/ddlogf = L - 1/(-dlogf + (X.MeanTimesPrecision - L * X.Precision)) =approx mx
                            //return Gaussian.FromMeanAndPrecision(L - dlogf / ddlogf, -ddlogf);
                        }
                        else if (mx > U && d_p == 1)
                        {
                            double sqrtPrec = Math.Sqrt(X.Precision);
                            double zL = (L - mx) * sqrtPrec;
                            double zU = (U - mx) * sqrtPrec;
                            double diff = U - L;
                            // diffs = zU - zL
                            double diffs = sqrtPrec * diff;
                            double deltaOverDiffs = (-zL - zU) / 2;
                            double delta = diffs * deltaOverDiffs;
                            // mx > U implies abs(zL) > abs(zU) and -zL > diffs
                            // (Cdf(zU) - Cdf(zL))/N(zL) = R(zU)*exp(delta) - R(zL)
                            double expMinus1 = MMath.ExpMinus1(delta);
                            //if (expMinus1 > 1e100) return MMath.NormalCdfLn(zU);
                            double rU = MMath.NormalCdfRatio(zU);
                            double drU = MMath.NormalCdfRatioDiff(zL, diffs);
                            double CdfRatioDiff = drU + rU * expMinus1;
                            double dlogf = -d_p / CdfRatioDiff * sqrtPrec;
                            double ddlogf = dlogf * (-dlogf + (X.MeanTimesPrecision - L * X.Precision));
                            return Gaussian.FromDerivatives(L, dlogf, ddlogf, ForceProper);
                        }
                        else
                        {
                            double dlogf = -d_p * Math.Exp(X.GetLogProb(L) - logZ);
                            double ddlogf = dlogf * (-dlogf + (X.MeanTimesPrecision - L * X.Precision));
                            return Gaussian.FromDerivatives(L, dlogf, ddlogf, ForceProper);
                        }
                    }
                }
                if (X.IsPointMass)
                {
                    if (upperBound.IsPointMass && upperBound.Point < X.Point)
                    {
                        // The constraint reduces to (lowerBound < upperBound), which is (lowerBound - upperBound < 0)
                        Gaussian shifted_F = DoublePlusOp.AAverageConditional(lowerBound, upperBound.Point);
                        Gaussian shifted_B = IsPositiveOp.XAverageConditional(false, shifted_F);
                        return DoublePlusOp.SumAverageConditional(shifted_B, upperBound.Point);
                    }
                    else
                    {
                        // The constraint reduces to (lowerBound < X), which is (lowerBound - X < 0)
                        Gaussian shifted_F = DoublePlusOp.AAverageConditional(lowerBound, X.Point);
                        Gaussian shifted_B = IsPositiveOp.XAverageConditional(false, shifted_F);
                        return DoublePlusOp.SumAverageConditional(shifted_B, X.Point);
                    }
                }
                bool precisionWasZero = AdjustXPrecision(isBetween, ref X, lowerBound, upperBound, ref logZ, 1e-0);
                if (Double.IsNegativeInfinity(logZ))
                    throw new AllZeroException();
                GetDiffMeanAndVariance(X, lowerBound, upperBound, out double yl, out double yu, out double r, out double sqrtomr2, out double invSqrtVxl, out double invSqrtVxu);
                GetAlpha(X, lowerBound, upperBound, logZ, out double? logZRatio, d_p, yl, yu, r, sqrtomr2, invSqrtVxl, invSqrtVxu, true, out double alphaL, false, out double alphaU, out double alphaX, out double ylInvSqrtVxlPlusAlphaX, out double yuInvSqrtVxuMinusAlphaX);
                // if we get here, we know that -1 < r <= 0 and invSqrtVxl is finite
                // since lowerBound is not uniform and X is not uniform, invSqrtVxl > 0
                // yl is always finite.  yu may be +/-infinity.
                double betaL;
                if (d_p == 1 && logZRatio.HasValue && yl < 0 && yu < 0)
                {
                    double invZRatio = Math.Exp(-logZRatio.Value);
                    double yuryl = (yu - r * yl) / sqrtomr2;
                    double Ryuryl = MMath.NormalCdfRatio(yuryl);
                    double ylryu = (yl - r * yu) / sqrtomr2;
                    double Rylryu = MMath.NormalCdfRatio(ylryu);
                    // alphaL = -invSqrtVxl*R(yuryl)/ZRatio
                    // alphaL*(alphaL - yl*invSqrtVxl) = invSqtVxl^2*R(yuryl)/ZRatio*(R(yuryl)/ZRatio + yl)
                    // beta = q * invSqrtVxl^2 / ZRatio
                    //double q = Ryuryl * (Ryuryl * invZRatio + yl) + r / Math.Sqrt(omr2);
                    double R1yuryl = MMath.NormalCdfMomentRatio(1, yuryl);
                    double R1ylryu = MMath.NormalCdfMomentRatio(1, ylryu);
                    // This is an asymptotic approximation of phi1(x,y,r)/phi_r(x,y,r)/sqrtomr2 where phi1 is the integral of phi wrt x
                    double intZRatio = (r * Rylryu + Ryuryl - r * sqrtomr2 * R1ylryu * yl) / (yl * yl + 1);
                    // Substitute Math.Exp(logZRatio) = (intZRatio - r * Rylryu - Ryuryl) / yl
                    //double u2 = Ryuryl * (intZRatio - r * Rylryu) * invZRatio + r / Math.Sqrt(omr2);
                    // r*yuryl + ylryu = yl*sqrtomr2
                    // yuryl = (R2 - 1)/Ryuryl
                    // Substitute r/sqrtomr2 = r*yl/(r*yuryl + ylryu) = r*yl/(r*(R2yuryl - 1)/Ryuryl + (R2ylryu - 1)/Rylryu)
                    // = r*yl*Ryuryl*Rylryu/(r*(R2yuryl - 1)*Rylryu + (R2ylryu - 1)*Ryuryl)
                    double w = r * R1yuryl * Rylryu + R1ylryu * Ryuryl;
                    //double u3 = Ryuryl * (intZRatio - r * Rylryu) * invZRatio + r * yl * Ryuryl * Rylryu / (w - (r * Rylryu + Ryuryl));
                    //double u4 = Ryuryl * (intZRatio - r * Rylryu + r * yl * Rylryu * Math.Exp(logZRatio) / (w - (r * Rylryu + Ryuryl))) * invZRatio;
                    //u4 = Ryuryl * (intZRatio - r * Rylryu + r * Rylryu * (intZRatio - r * Rylryu - Ryuryl) / (w - (r * Rylryu + Ryuryl))) * invZRatio;
                    //u4 = Ryuryl * (intZRatio + r * Rylryu * (intZRatio - w) / (w - (r * Rylryu + Ryuryl))) * invZRatio;
                    double q = Ryuryl * (intZRatio * (w - Ryuryl) - r * Rylryu * w) / (w - (r * Rylryu + Ryuryl)) * invZRatio;
                    betaL = q * invSqrtVxl * invSqrtVxl * invZRatio;
                    if (double.IsNaN(betaL)) throw new Exception("betaL is NaN");
                }
                else
                {
                    // (mx - ml) / (vl + vx) = yl*invSqrtVxl
                    betaL = alphaL * (alphaL - yl * invSqrtVxl);
                    // This formula is slightly more accurate when r == -1 && yl > 0 && yu < 0:
                    // betaU = alphaU * invSqrtVxu * (MMath.NormalCdfMomentRatio(1, yu) - yu * ncrl / delta) / ZoverPhiU;
                    // because yu * ZoverPhiU + 1 = ncru*yu+1 - yu*ncrl / delta
                    if (r > -1 && r != 0 && !precisionWasZero)
                    {
                        double omr2 = sqrtomr2 * sqrtomr2;
                        double logPhiR = GetLogPhiR(X, lowerBound, upperBound, yl, yu, r, omr2, logZ, logZRatio);
                        double c = d_p * r * Math.Exp(logPhiR);
                        betaL += c * invSqrtVxl * invSqrtVxl;
                        if (double.IsNaN(betaL)) throw new Exception("betaL is NaN");
                    }
                }
                //Trace.WriteLine($"alpha = {alphaL} beta = {betaL} yl = {yl} yu = {yu} r = {r}");
                return GaussianOp.GaussianFromAlphaBeta(lowerBound, alphaL, betaL, ForceProper);
            }
            if (Double.IsNaN(result.Precision) || Double.IsNaN(result.MeanTimesPrecision))
                throw new InferRuntimeException($"{nameof(result)} is NaN.  {nameof(isBetween)}={isBetween}, {nameof(X)}={X}, {nameof(lowerBound)}={lowerBound}, {nameof(upperBound)}={upperBound}, {nameof(logZ)}={logZ}");
            return result;
        }

        public static Gaussian UpperBoundAverageConditional_Slow(
            [SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            return UpperBoundAverageConditional(isBetween, X, lowerBound, upperBound, LogZ(isBetween, X, lowerBound, upperBound));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="UpperBoundAverageConditional(Bernoulli, Gaussian, Gaussian, Gaussian, double)"]/*'/>
        [SkipIfAllUniform("X", "lowerBound")]
        [SkipIfAllUniform("X", "upperBound")]
        public static Gaussian UpperBoundAverageConditional(
            [SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound,
            double logZ)
        {
            Gaussian result = LowerBoundAverageConditional(isBetween,
                Gaussian.FromNatural(-X.MeanTimesPrecision, X.Precision),
                Gaussian.FromNatural(-upperBound.MeanTimesPrecision, upperBound.Precision),
                Gaussian.FromNatural(-lowerBound.MeanTimesPrecision, lowerBound.Precision),
                logZ);
            result.MeanTimesPrecision *= -1;
            return result;
        }

        public static Gaussian XAverageConditional_Slow(
            [SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            return XAverageConditional(isBetween, X, lowerBound, upperBound, LogZ(isBetween, X, lowerBound, upperBound));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="XAverageConditional(Bernoulli, Gaussian, Gaussian, Gaussian, double)"]/*'/>
        [SkipIfAllUniform("lowerBound", "upperBound")]
        [SkipIfAllUniform("X", "lowerBound")]
        [SkipIfAllUniform("X", "upperBound")]
        public static Gaussian XAverageConditional(
            [SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound,
            double logZ)
        {
            if (lowerBound.IsPointMass && upperBound.IsPointMass)
                return XAverageConditional(isBetween, X, lowerBound.Point, upperBound.Point);
            Gaussian result = new Gaussian();
            if (isBetween.IsUniform())
                return result;
            if (X.IsPointMass && lowerBound.IsPointMass && X.Point < lowerBound.Point)
            {
                return lowerBound;
            }
            else if (X.IsPointMass && upperBound.IsPointMass && X.Point > upperBound.Point)
            {
                return upperBound;
            }
            else if (X.Precision == 0)
            {
                if (lowerBound.IsUniform() || upperBound.IsUniform() ||
                    (lowerBound.IsPointMass && Double.IsInfinity(lowerBound.Point)) ||
                    (upperBound.IsPointMass && Double.IsInfinity(upperBound.Point)) ||
                    !Double.IsPositiveInfinity(isBetween.LogOdds))
                {
                    return result;
                }
                else if (isBetween.IsPointMass && isBetween.Point)
                {
                    double ml, vl, mu, vu;
                    lowerBound.GetMeanAndVariance(out ml, out vl);
                    upperBound.GetMeanAndVariance(out mu, out vu);
                    double vlu = vl + vu; // vlu > 0
                    double dmul = mu - ml;
                    double alpha = Math.Exp(Gaussian.GetLogProb(ml, mu, vlu) - MMath.NormalCdfLn(dmul / Math.Sqrt(vlu)));
                    double alphaU = 1.0 / (dmul + vlu * alpha);
                    double betaU = alphaU * (alphaU - alpha);
                    double munew = mu + vu * alphaU;
                    double mlnew = ml - vl * alphaU;
                    double vunew = vu - vu * vu * betaU;
                    double vlnew = vl - vl * vl * betaU;
                    double diff = dmul + vlu * alphaU; // munew - mlnew;
                    return Gaussian.FromMeanAndVariance((munew + mlnew) / 2, diff * diff / 12 + (vunew + vlnew + vu * vl * betaU) / 3);
                }
                else
                    throw new NotImplementedException();
            }
            else
            {
                // X is not uniform
                bool precisionWasZero = AdjustXPrecision(isBetween, ref X, lowerBound, upperBound, ref logZ);
                if (Double.IsNegativeInfinity(logZ))
                    throw new AllZeroException();
                double d_p = 2 * isBetween.GetProbTrue() - 1;
                GetDiffMeanAndVariance(X, lowerBound, upperBound, out double yl, out double yu, out double r, out double sqrtomr2, out double invSqrtVxl, out double invSqrtVxu);
                GetAlpha(X, lowerBound, upperBound, logZ, out double? logZRatio, d_p, yl, yu, r, sqrtomr2, invSqrtVxl, invSqrtVxu, true, out double alphaL, true, out double alphaU, out double alphaX, out double ylInvSqrtVxlPlusAlphaX, out double yuInvSqrtVxuMinusAlphaX);
                //alphaX = -alphaL - alphaU;
                // (mx - ml) / (vl + vx) = yl*invSqrtVxl
                // To improve accuracy here, could rewrite betaX to be relative to X.Precision, or solve for posterior on X directly as in uniform case.
                // betaX = alphaX * alphaX - alphaL * (yl * invSqrtVxl) + alphaU * (yu * invSqrtVxu)
                //       = (-alphaL - alphaU)*alphaX - alphaL * (yl * invSqrtVxl) + alphaU * (yu * invSqrtVxu)
                //       = - alphaL * (yl * invSqrtVxl + alphaX) + alphaU * (yu * invSqrtVxu - alphaX)
                double betaX = 0;
                if (!Double.IsInfinity(yl))
                {
                    // if yl is infinity then alphaL == 0
                    betaX -= alphaL * ylInvSqrtVxlPlusAlphaX;
                }
                if (!Double.IsInfinity(yu))
                {
                    // if yu is infinity then alphaU == 0
                    betaX += alphaU * yuInvSqrtVxuMinusAlphaX;
                }
                if (r > -1 && r != 0 && !Double.IsInfinity(yl) && !Double.IsInfinity(yu) && !precisionWasZero)
                {
                    double omr2 = sqrtomr2 * sqrtomr2;
                    double logPhiR = GetLogPhiR(X, lowerBound, upperBound, yl, yu, r, omr2, logZ, logZRatio);
                    double c = d_p * r * Math.Exp(logPhiR);
                    betaX += c * (-2 * X.Precision + invSqrtVxl * invSqrtVxl + invSqrtVxu * invSqrtVxu);
                }
                if (TraceAlpha)
                    Trace.WriteLine($"yu = {yu} yl = {yl} r = {r} alphaX={alphaX}, alphaL={alphaL}, alphaU={alphaU}, betaX={betaX}, ylInvSqrtVxlPlusAlphaX = {ylInvSqrtVxlPlusAlphaX}, yuInvSqrtVxuMinusAlphaX = {yuInvSqrtVxuMinusAlphaX}");
                return GaussianOp.GaussianFromAlphaBeta(X, alphaX, betaX, ForceProper);
            }
        }

        public static bool TraceAlpha;
        private const double smallR = -1;
        private const double smallLogZ = -1e4;

        // Invariant to swapping yu and yl
        private static double GetLogPhiR(Gaussian X, Gaussian lowerBound, Gaussian upperBound, double yl, double yu, double r, double omr2, double logZ, double? logZRatio)
        {
            double logPhiR = -0.5 * Math.Log(omr2);
            if (logZRatio.HasValue) return logPhiR - logZRatio.Value;
            logPhiR += -2 * MMath.LnSqrt2PI;
            if (r > smallR)
            {
                if (Math.Abs(yu) > Math.Abs(yl))
                    logPhiR -= 0.5 * (yl * yl + yu * (yu - 2 * r * yl)) / omr2;
                else
                    logPhiR -= 0.5 * (yu * yu + yl * (yl - 2 * r * yu)) / omr2;
            }
            else
            {
                // yu = (mu-mx)*invSqrtVxu
                // r = -vx*invSqrtVxu*invSqrtVxl
                // (yu * yu + yl * (yl - 2 * r * yu)) =
                // ((mu-mx)*(mu-mx)/(vx+vu) + (mx-ml)*invSqrtVxl * ((mx-ml)*invSqrtVxl - 2 * r * (mu-mx)*invSqrtVxu))
                // ((mu-mx)*(mu-mx)/(vx+vu) + (mx-ml) * ((mx-ml) + 2 *vx * invSqrtVxu* (mu-mx)*invSqrtVxu)/(vx+vl))
                // ((mu-mx)*(mu-mx)/(vx+vu) + (mx-ml) * ((mx-ml) + 2 *vx * (mu-mx)/(vx+vu))/(vx+vl))
                // (mu-mx)*(mu-mx)/(vx+vu) + (mx-ml) * ((mx-ml)*(vx+vu) + (2*vx*mu-2*vx*mx))/(vx+vu)/(vx+vl)
                // ((mu-mx)*(mu-mx) + (mx-ml) * ((mx-ml)*(vx+vu) + (2*vx*mu-2*vx*mx))/(vx+vl))/(vx+vu)
                // ((mu-mx)*(mu-mx) + (mx-ml) * ((mx-ml)*vu + (2*mu-ml-mx)*vx)/(vx+vl))/(vx+vu)
                // ((mu-mx)*(mu-mx)*(vx+vl) + (mx-ml) * ((mx-ml)*vu + (2*mu-ml-mx)*vx))/(vx+vl)/(vx+vu)
                // (((mu-mx)*(mu-mx) + (mx-ml)*(2*mu-ml-mx))*vx + (mu-mx)*(mu-mx)*vl + (mx-ml)*(mx-ml)*vu)/(vx+vl)/(vx+vu)
                // ((mu-ml)*(mu-ml)*vx + (mu-mx)*(mu-mx)*vl + (mx-ml)*(mx-ml)*vu)/(vx+vl)/(vx+vu)
                // (1-r*r) = 1 - vx*vx/(vx+vl)/(vx+vu) = (vx*vl + vx*vu + vl*vu)/(vx+vl)/(vx+vu)
                double mx, vx, ml, vl, mu, vu;
                X.GetMeanAndVariance(out mx, out vx);
                lowerBound.GetMeanAndVariance(out ml, out vl);
                upperBound.GetMeanAndVariance(out mu, out vu);
                double dmul = mu - ml;
                double dmux = mu - mx; // yu/invSqrtVxu
                double dmxl = mx - ml;
                logPhiR -= 0.5 * (dmul * dmul * vx + dmux * dmux * vl + dmxl * dmxl * vu) / (vx * vl + vx * vu + vl * vu);
            }
            if (double.IsNaN(logPhiR)) throw new Exception($"logPhiR is NaN for X={X}, lowerBound={lowerBound}, upperBound={upperBound}, yl={yl}, yu={yu}, r={r}");
            return logPhiR - logZ;
        }

        private static bool AdjustXPrecision(Bernoulli isBetween, ref Gaussian X, Gaussian lowerBound, Gaussian upperBound, ref double logZ, double precisionScaling = 1)
        {
            bool precisionWasZero = false;
            double minPrecision = precisionScaling * LowPrecisionThreshold * Math.Min(lowerBound.Precision, upperBound.Precision);
            // minPrecision cannot be infinity here
            if (double.IsPositiveInfinity(minPrecision)) throw new Exception("minPrecision is infinity");
            //Trace.WriteLine($"X.Precision={X.Precision}, minPrecision={minPrecision}");
            // When X.Precision is small, r is close to -1.
            if (X.Precision < minPrecision)
            {
                double mx, vx, ml, vl, mu, vu;
                X.GetMeanAndVariance(out mx, out vx);
                lowerBound.GetMeanAndVariance(out ml, out vl);
                upperBound.GetMeanAndVariance(out mu, out vu);
                double maxMean = Math.Max(Math.Abs(ml), Math.Abs(mu)) * LargeMeanThreshold;
                //Trace.WriteLine($"mx={Math.Abs(mx)}, maxMean={maxMean}");
                if (Math.Abs(mx) > maxMean)
                {
                    // in this case, logZ is inaccurate, making all further computations inaccurate.
                    // To prevent this, we make X.Precision artificially larger.
                    //Trace.WriteLine("precisionWasZero");
                    precisionWasZero = true;
                    //X.SetMeanAndPrecision(mx, minPrecision);
                    X.Precision = minPrecision;
                    logZ = LogAverageFactor(isBetween, X, lowerBound, upperBound);
                }
            }

            return precisionWasZero;
        }

        private static void GetAlpha(Gaussian X, Gaussian lowerBound, Gaussian upperBound, double logZ, out double? logZRatio,
            double d_p, double yl, double yu, double r, double sqrtomr2, double invSqrtVxl, double invSqrtVxu,
            bool getAlphaL, out double alphaL, bool getAlphaU, out double alphaU, out double alphaX, out double ylInvSqrtVxlPlusAlphaX, out double yuInvSqrtVxuMinusAlphaX)
        {
            // NormalCdfRatioLn should not be used when r == -1 or yl is infinite or yu is infinite.
            // yl = -inf or yu = -inf lead to logZ = -inf which is excluded above.
            bool useLogZRatio = (r > smallR) && (logZ < smallLogZ) && (yl + yu <= 1e-4);
            logZRatio = useLogZRatio ? (double?)MMath.NormalCdfRatioLn(yl, yu, r, sqrtomr2) : null;
            double mx, vx, ml, vl, mu, vu;
            X.GetMeanAndVariance(out mx, out vx);
            lowerBound.GetMeanAndVariance(out ml, out vl);
            upperBound.GetMeanAndVariance(out mu, out vu);
            double rPlus1 = MMath.GetRPlus1(r, sqrtomr2);
            double omr2 = sqrtomr2 * sqrtomr2;
            double r1yuryl = 0;
            alphaL = 0.0;
            if (r == 0 && !Double.IsInfinity(yl))
            {
                alphaL = -d_p * invSqrtVxl / MMath.NormalCdfRatio(yl);
            }
            else if (!lowerBound.IsUniform() && !Double.IsInfinity(yl))
            {
                // since X and lowerBound are not both uniform, invSqrtVxl > 0
                double logPhiL = 0;
                if (logZRatio == null)
                    logPhiL += Gaussian.GetLogProb(yl, 0, 1);
                if (r > -1)
                {
                    double yuryl;
                    if (r > smallR)
                    {
                        yuryl = MMath.GetXMinusRY(yu, yl, r, omr2) / sqrtomr2;
                    }
                    else
                    {
                        // yu - r * yl
                        // = (mu-mx)*invSqrtVxu + vx*invSqrtVxu*invSqrtVxl*(mx-ml)*invSqrtVxl
                        // = (mu-mx + (mx-ml)*vx/(vx+vl))*invSqrtVxu
                        // = ((mu-mx)*vl + (mu-ml)*vx)/(vx+vl)*invSqrtVxu
                        // Sqrt(1 - r * r) = Sqrt(vx*vl + vx*vu + vl*vu)*invSqrtVxl*invSqrtVxu
                        yuryl = ((mu - mx) * vl + (mu - ml) * vx) * invSqrtVxl / Math.Sqrt(vx * vl + vx * vu + vl * vu);
                    }
                    if (logZRatio.HasValue)
                    {
                        // in this case, PhiL is divided by N(yl;0,1)*N(yuryl;0,1)
                        logPhiL = MMath.NormalCdfRatioLn(yuryl);
                        r1yuryl = MMath.NormalCdfMomentRatio(1, yuryl);
                    }
                    else
                    {
                        logPhiL += MMath.NormalCdfLn(yuryl);
                    }
                    if (logPhiL > double.MaxValue) throw new Exception();
                    //Trace.WriteLine($"yuryl = {yuryl}, invSqrtVxl = {invSqrtVxl} useLogZRatio = {useLogZRatio}");
                }
                alphaL = -d_p * invSqrtVxl * Math.Exp(logPhiL - (logZRatio ?? logZ));
            }
            alphaU = 0.0;
            double r1ylryu = 0;
            if (r == 0 && !Double.IsInfinity(yu))
            {
                alphaU = d_p * invSqrtVxu / MMath.NormalCdfRatio(yu);
            }
            else if (!upperBound.IsUniform() && !Double.IsInfinity(yu))
            {
                // since X and upperBound are not both uniform, invSqrtVxu > 0
                double logPhiU = 0;
                if (logZRatio == null)
                    logPhiU = Gaussian.GetLogProb(yu, 0, 1);
                if (r > -1)
                {
                    double ylryu;
                    if (r > smallR)
                    {
                        ylryu = MMath.GetXMinusRY(yl, yu, r, omr2) / sqrtomr2;
                    }
                    else
                    {
                        // yl - r * yu = 
                        // (mx-ml)*invSqrtVxl + vx*invSqrtVxu*invSqrtVxl*(mu-mx)*invSqrtVxu =
                        // (mx-ml + (mu-mx)*vx/(vx+vu))*invSqrtVxl =
                        // ((mx-ml)*vu + (mu-ml)*vx)/(vx+vu)*invSqrtVxl
                        // Sqrt(1 - r * r) = Sqrt(vx*vl + vx*vu + vl*vu)*invSqrtVxl*invSqrtVxu
                        ylryu = ((mx - ml) * vu + (mu - ml) * vx) * invSqrtVxu / Math.Sqrt(vx * vl + vx * vu + vl * vu);
                    }
                    if (logZRatio.HasValue)
                    {
                        // in this case, PhiL is divided by N(yu;0,1)*N(ylryu;0,1)
                        logPhiU = MMath.NormalCdfRatioLn(ylryu);
                        r1ylryu = MMath.NormalCdfMomentRatio(1, ylryu);
                    }
                    else
                        logPhiU += MMath.NormalCdfLn(ylryu);
                    //Trace.WriteLine($"ylryu = {ylryu}, invSqrtVxu = {invSqrtVxu}");
                }
                alphaU = d_p * invSqrtVxu * Math.Exp(logPhiU - (logZRatio ?? logZ));
            }
            if (logZRatio.HasValue && r != 0 && (r1yuryl < 0.5) && (r1ylryu < 0.5))
            {
                // NormalCdfRatio(x) = (NormalCdfMomentRatio(1,x) - 1)/x
                // alphaX = -alphaL - alphaU
                // = d_p * Math.Exp(-logZRatio) * (invSqrtVxl * NormalCdfRatio(yuryl) - invSqrtVxu * NormalCdfRatio(ylryu))
                // invSqrtVxl * NormalCdfRatio(yuryl) - invSqrtVxu * NormalCdfRatio(ylryu)
                // = invSqrtVxl * (NormalCdfMomentRatio(1,yuryl) - 1) / yuryl - invSqrtVxu * (NormalCdfMomentRatio(1,ylryu) - 1) / ylryu
                // = Math.Sqrt(vx * vl + vx * vu + vl * vu) * ((NormalCdfMomentRatio(1,yuryl) - 1) / ((mu - mx) * vl + (mu - ml) * vx) - (NormalCdfMomentRatio(1,ylryu) - 1) / ((mx - ml) * vu + (mu - ml) * vx))
                // -1 / ((mu - mx) * vl + (mu - ml) * vx) + 1 / ((mx - ml) * vu + (mu - ml) * vx)
                // = ((ml - mx) * vu + (mu - mx) * vl) / ((mu - mx) * vl + (mu - ml) * vx) / ((mx - ml) * vu + (mu - ml) * vx)
                double dmulvx = (mu - ml) * vx;
                double dmuxvl = (mu - mx) * vl;
                double cyuryl = dmuxvl + dmulvx;
                double dmxlvu = (mx - ml) * vu;
                double cylryu = dmxlvu + dmulvx;
                double q2 = (dmuxvl - dmxlvu) / cyuryl / cylryu;
                alphaX = d_p * Math.Exp(-logZRatio.Value) * Math.Sqrt(vx * vl + vx * vu + vl * vu) * (q2 + r1yuryl / cyuryl - r1ylryu / cylryu);
            }
            else
            {
                alphaX = -alphaL - alphaU;
            }
            ylInvSqrtVxlPlusAlphaX = yl * invSqrtVxl + alphaX;
            yuInvSqrtVxuMinusAlphaX = yu * invSqrtVxu - alphaX;
            double invSqrtVxlMinusInvSqrtVxu = invSqrtVxl - invSqrtVxu;
            if (Math.Abs(invSqrtVxlMinusInvSqrtVxu) < invSqrtVxl * 1e-10)
            {
                // Use Taylor expansion:
                // f(y) = 1/sqrt(vx+y)
                // f'(y) = -0.5/(vx+y)^(3/2)
                // f(vl) - f(vu) =approx f(vu) + (vl-vu)*f'(vu)
                invSqrtVxlMinusInvSqrtVxu = (vl - vu) * (-0.5) * (invSqrtVxu * invSqrtVxu * invSqrtVxu);
            }
            if (d_p == 1)
            {
                double ylInvSqrtVxlPlusAlphaX2 = ylInvSqrtVxlPlusAlphaX;
                //   yl * invSqrtVxl + alphaX 
                // = yl * invSqrtVxl - alphaL - alphaU
                // = invSqrtVxl * (yl + Zx / Z) - invSqrtVxu * Zy / Z
                // = invSqrtVxl * (yl + Zx / Z - Zy / Z) + (invSqrtVxl - invSqrtVxu) * Zy /Z
                //   yl + Zx / Z - Zy / Z 
                // = (yl*Z + Zx - Zy)/Z
                // = (intZ - Zx - r*Zy + Zx - Zy)/Z
                // = intZ/Z - (1+r)*Zy/Z
                double intZOverZ = MMath.NormalCdfIntegralRatio(yl, yu, r, sqrtomr2);
                if (r == 0)
                    ylInvSqrtVxlPlusAlphaX = invSqrtVxl * intZOverZ - alphaU;
                else
                    ylInvSqrtVxlPlusAlphaX = invSqrtVxl * intZOverZ + (invSqrtVxlMinusInvSqrtVxu - rPlus1 * invSqrtVxl) / invSqrtVxu * alphaU;
                //ylInvSqrtVxlPlusAlphaX = -invSqrtVxl / yl;
                if (TraceAlpha)
                    Trace.WriteLine($"ylInvSqrtVxlPlusAlphaX = {ylInvSqrtVxlPlusAlphaX} replaces {ylInvSqrtVxlPlusAlphaX2} (intZOverZ = {intZOverZ}, alphaU = {alphaU}, r = {r}, yl = {yl})");
                if (double.IsNaN(ylInvSqrtVxlPlusAlphaX)) throw new Exception("ylInvSqrtVxlPlusAlphaX is NaN");
                // alphaL = -invSqrtVxl * Zx / Z
                // yuInvSqrtVxuMinusAlphaX
                // = yu * invSqrtVxu - alphaX
                // = yu * invSqrtVxu + alphaU + alphaL
                // = invSqrtVxu * (yu + Zy / Z) - invSqrtVxl * Zx / Z
                // = invSqrtVxu * (yu + Zy / Z - Zx / Z) - (invSqrtVxl - invSqrtVxu) * Zx / Z
                //   yu + Zy / Z - Zx / Z 
                // = (yu*Z + Zy - Zx)/Z
                // = (intZ2 - Zy - r*Zx + Zy - Zx)/Z
                // = intZ2/Z - (1+r)*Zx/Z
                double yuInvSqrtVxuMinusAlphaX2 = yuInvSqrtVxuMinusAlphaX;
                double intZ2OverZ = MMath.NormalCdfIntegralRatio(yu, yl, r, sqrtomr2);
                if (r == 0)
                    yuInvSqrtVxuMinusAlphaX = invSqrtVxu * intZ2OverZ + alphaL;
                else
                    yuInvSqrtVxuMinusAlphaX = invSqrtVxu * intZ2OverZ + (invSqrtVxlMinusInvSqrtVxu + rPlus1 * invSqrtVxu) / invSqrtVxl * alphaL;
                if (TraceAlpha)
                    Trace.WriteLine($"yuInvSqrtVxuMinusAlphaX = {yuInvSqrtVxuMinusAlphaX} replaces {yuInvSqrtVxuMinusAlphaX2} (intZ2OverZ = {intZ2OverZ}, alphaL = {alphaL})");
                //if (Math.Abs(yuInvSqrtVxuMinusAlphaX - yuInvSqrtVxuMinusAlphaX2) > 1e-2) throw new Exception();
                if (double.IsNaN(yuInvSqrtVxuMinusAlphaX)) throw new Exception("yuInvSqrtVxuMinusAlphaX is NaN");
            }
            // TODO: make this case smoothly blend into the X.IsPointMass case
            if (string.Empty.Length > 0)//(r == -1 || (!Double.IsInfinity(yl) && !Double.IsInfinity(yu) && logZ == MMath.NormalCdfLn(yl, yu, -1)))
            {
                if ((yl > 0 && yu < 0) || (yl < 0 && yu > 0))
                {
                    Trace.WriteLine("special case for r == -1");
                    // yu + yl = (mu-mx)/sqrt(vx+vu) + (mx-ml)/sqrt(vx+vl) = mu/sqrt(vx+vu) - ml/sqrt(vx+vl) + mx*(1/sqrt(vx+vl) - 1/sqrt(vx+vu))
                    // This is more accurate than (yu+yl)
                    double yuPlusyl = mu * invSqrtVxu - ml * invSqrtVxl + mx * invSqrtVxlMinusInvSqrtVxu;
                    // double delta = Math.Exp(-0.5 * (yu * yu - yl * yl));
                    // This is more accurate than above.
                    double deltaMinus1 = MMath.ExpMinus1(-0.5 * yuPlusyl * (yu - yl));
                    double delta = 1 + deltaMinus1;
                    bool deltaMinus1IsSmall = Math.Abs(deltaMinus1) < 1e-8;
                    double ZoverPhiL, ZoverPhiU;
                    if (yl > 0 && yu < 0)
                    {
                        // This code inlines the following formula for Z:
                        // Z = NormalCdf(yu) - NormalCdf(-yl)
                        double ncru = MMath.NormalCdfRatio(yu);
                        double ncrl = MMath.NormalCdfRatio(-yl);
                        if (deltaMinus1IsSmall)
                        {
                            double ncruMinusNcrl = MMath.NormalCdfRatioDiff(-yl, yuPlusyl);
                            ZoverPhiL = ncruMinusNcrl + ncru * deltaMinus1;
                            ZoverPhiU = ZoverPhiL / delta;
                        }
                        else
                        {
                            ZoverPhiL = ncru * delta - ncrl;
                            ZoverPhiU = ncru - ncrl / delta;
                        }
                    }
                    else // (yl < 0 && yu > 0)
                    {
                        // This code inlines the following formula for Z:
                        // Z = NormalCdf(yl) - NormalCdf(-yu)
                        double ncru = MMath.NormalCdfRatio(-yu);
                        double ncrl = MMath.NormalCdfRatio(yl);
                        if (deltaMinus1IsSmall)
                        {
                            double ncrlMinusNcru = MMath.NormalCdfRatioDiff(-yu, yuPlusyl);
                            ZoverPhiL = ncrlMinusNcru - ncru * deltaMinus1;
                            ZoverPhiU = ZoverPhiL / delta;
                        }
                        else
                        {
                            ZoverPhiL = ncrl - ncru * delta;
                            ZoverPhiU = ncrl / delta - ncru;
                        }
                    }
                    alphaL = -d_p * invSqrtVxl / ZoverPhiL;
                    //alphaU = d_p * invSqrtVxu * delta / ZoverPhiL;
                    alphaU = d_p * invSqrtVxu / ZoverPhiU;
                    if (deltaMinus1IsSmall)
                    {
                        // alphaX = d_p/ZoverPhiL * (invSqrtVxl - invSqrtVxu * delta)
                        alphaX = d_p / ZoverPhiL * (invSqrtVxlMinusInvSqrtVxu - invSqrtVxu * deltaMinus1);
                    }
                    else
                    {
                        alphaX = -alphaL - alphaU;
                    }
                    if (d_p == 1)
                    {
                        //   yl * invSqrtVxl + alphaX 
                        // = yl * invSqrtVxl - alphaL - alphaU
                        // = invSqrtVxl * (yl + 1/ZoverPhiL) - invSqrtVxu / ZoverPhiU
                        // = invSqrtVxl * (yl + 1/ZoverPhiL - 1/ZoverPhiU) + (invSqrtVxl - invSqrtVxu) / ZoverPhiU
                        // yl + 1/ZoverPhiL - 1/ZoverPhiU = (yl*ZRatio + Rylryu - Ryuryl)/ZRatio
                        // = (intZRatio - Rylryu - r*Ryuryl + Rylryu - Ryuryl)/ZRatio
                        // = intZRatio/ZRatio = intZ/Z
                        double ylInvSqrtVxlPlusAlphaX2 = ylInvSqrtVxlPlusAlphaX;
                        double intZOverZ = MMath.NormalCdfIntegralRatio(yl, yu, -1);
                        ylInvSqrtVxlPlusAlphaX = invSqrtVxl * intZOverZ + invSqrtVxlMinusInvSqrtVxu / ZoverPhiU;
                        Trace.WriteLine($"ylInvSqrtVxlPlusAlphaX = {ylInvSqrtVxlPlusAlphaX} replaces {ylInvSqrtVxlPlusAlphaX2} (intZOverZ = {intZOverZ}, alphaU = {alphaU})");
                        if (double.IsNaN(ylInvSqrtVxlPlusAlphaX)) throw new Exception("ylInvSqrtVxlPlusAlphaX is NaN");
                        //   yu * invSqrtVxu - alphaX
                        // = yu * invSqrtVxu + alphaL + alphaU
                        // = invSqrtVxu * (yu + 1/ZoverPhiU) - invSqrtVxl / ZoverPhiL
                        // = invSqrtVxu * (yu + 1/ZoverPhiU - 1/ZoverPhiL) - (invSqrtVxl - invSqrtVxu) / ZoverPhiL
                        // yu + 1/ZoverPhiU - 1/ZoverPhiL = (yu*ZRatio + Ryuryl - Rylryu)/ZRatio
                        // = (intZ2Ratio - r*Rylryu - Ryuryl + Ryuryl - Rylryu)/ZRatio
                        // = intZ2/Z
                        double yuInvSqrtVxuMinusAlphaX2 = yuInvSqrtVxuMinusAlphaX;
                        double intZ2OverZ = MMath.NormalCdfIntegralRatio(yu, yl, -1);
                        yuInvSqrtVxuMinusAlphaX = invSqrtVxu * intZ2OverZ - invSqrtVxlMinusInvSqrtVxu / ZoverPhiL;
                        Trace.WriteLine($"yuInvSqrtVxuMinusAlphaX = {yuInvSqrtVxuMinusAlphaX} replaces {yuInvSqrtVxuMinusAlphaX2} (intZ2OverZ = {intZ2OverZ}, alphaL = {alphaL})");
                        if (double.IsNaN(yuInvSqrtVxuMinusAlphaX)) throw new Exception("yuInvSqrtVxuMinusAlphaX is NaN");
                    }
                    if (double.IsNaN(alphaL)) throw new Exception("alphaL is NaN");
                    if (double.IsNaN(alphaU)) throw new Exception("alphaU is NaN");
                }
            }
        }

#if false
    /// <summary>
    /// EP message to 'isBetween'
    /// </summary>
    /// <param name="X">Incoming message from 'x'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
    /// <param name="lowerBound">Constant value for 'lowerBound'.</param>
    /// <param name="upperBound">Incoming message from 'upperBound'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
    /// <returns>The outgoing EP message to the 'isBetween' argument</returns>
    /// <remarks><para>
    /// The outgoing message is a distribution matching the moments of 'isBetween' as the random arguments are varied.
    /// The formula is <c>proj[p(isBetween) sum_(x,upperBound) p(x,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(isBetween)</c>.
    /// </para></remarks>
    /// <exception cref="ImproperMessageException"><paramref name="X"/> is not a proper distribution</exception>
    /// <exception cref="ImproperMessageException"><paramref name="upperBound"/> is not a proper distribution</exception>
        public static Bernoulli IsBetweenAverageConditional([SkipIfUniform] Gaussian X, double lowerBound, [SkipIfUniform] Gaussian upperBound)
        {
            return IsBetweenAverageConditional(X, Gaussian.PointMass(lowerBound), upperBound);
        }

        /// <summary>
        /// EP message to 'isBetween'
        /// </summary>
        /// <param name="X">Incoming message from 'x'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="upperBound">Constant value for 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'isBetween' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'isBetween' as the random arguments are varied.
        /// The formula is <c>proj[p(isBetween) sum_(x,lowerBound) p(x,lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(isBetween)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="X"/> is not a proper distribution</exception>
        /// <exception cref="ImproperMessageException"><paramref name="lowerBound"/> is not a proper distribution</exception>
        public static Bernoulli IsBetweenAverageConditional([SkipIfUniform] Gaussian X, [SkipIfUniform] Gaussian lowerBound, double upperBound)
        {
            return IsBetweenAverageConditional(X, lowerBound, Gaussian.PointMass(upperBound));
        }

        /// <summary>
        /// EP message to 'isBetween'
        /// </summary>
        /// <param name="X">Constant value for 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the 'isBetween' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'isBetween' as the random arguments are varied.
        /// The formula is <c>proj[p(isBetween) sum_(lowerBound,upperBound) p(lowerBound,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(isBetween)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="lowerBound"/> is not a proper distribution</exception>
        /// <exception cref="ImproperMessageException"><paramref name="upperBound"/> is not a proper distribution</exception>
        public static Bernoulli IsBetweenAverageConditional(double X, [SkipIfUniform] Gaussian lowerBound, [SkipIfUniform] Gaussian upperBound)
        {
            return IsBetweenAverageConditional(Gaussian.PointMass(X), lowerBound, upperBound);
        }

        /// <summary>
        /// EP message to 'isBetween'
        /// </summary>
        /// <param name="X">Constant value for 'x'.</param>
        /// <param name="lowerBound">Constant value for 'lowerBound'.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the 'isBetween' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'isBetween' as the random arguments are varied.
        /// The formula is <c>proj[p(isBetween) sum_(upperBound) p(upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(isBetween)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="upperBound"/> is not a proper distribution</exception>
        public static Bernoulli IsBetweenAverageConditional(double X, double lowerBound, [SkipIfUniform] Gaussian upperBound)
        {
            return IsBetweenAverageConditional(Gaussian.PointMass(X), Gaussian.PointMass(lowerBound), upperBound);
        }

        /// <summary>
        /// EP message to 'isBetween'
        /// </summary>
        /// <param name="X">Constant value for 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="upperBound">Constant value for 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'isBetween' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'isBetween' as the random arguments are varied.
        /// The formula is <c>proj[p(isBetween) sum_(lowerBound) p(lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(isBetween)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="lowerBound"/> is not a proper distribution</exception>
        public static Bernoulli IsBetweenAverageConditional(double X, [SkipIfUniform] Gaussian lowerBound, double upperBound)
        {
            return IsBetweenAverageConditional(Gaussian.PointMass(X), lowerBound, Gaussian.PointMass(upperBound));
        }

        /// <summary>
        /// EP message to 'x'
        /// </summary>
        /// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="X">Incoming message from 'x'.</param>
        /// <param name="lowerBound">Constant value for 'lowerBound'.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'x' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'x' as the random arguments are varied.
        /// The formula is <c>proj[p(x) sum_(isBetween,upperBound) p(isBetween,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(x)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
        [SkipIfAllUniform("X", "upperBound")]
    public static Gaussian XAverageConditional([SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, double lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            return XAverageConditional(isBetween, X, Gaussian.PointMass(lowerBound), upperBound);
        }

        /// <summary>
        /// EP message to 'x'
        /// </summary>
        /// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="X">Incoming message from 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
        /// <param name="upperBound">Constant value for 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'x' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'x' as the random arguments are varied.
        /// The formula is <c>proj[p(x) sum_(isBetween,lowerBound) p(isBetween,lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(x)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
        [SkipIfAllUniform("X", "lowerBound")]
    public static Gaussian XAverageConditional([SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, double upperBound)
        {
            return XAverageConditional(isBetween, X, lowerBound, Gaussian.PointMass(upperBound));
        }

        /// <summary>
        /// EP message to 'x'
        /// </summary>
        /// <param name="isBetween">Constant value for 'isBetween'.</param>
        /// <param name="X">Incoming message from 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'x' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'x' as the random arguments are varied.
        /// The formula is <c>proj[p(x) sum_(lowerBound,upperBound) p(lowerBound,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(x)</c>.
        /// </para></remarks>
        [SkipIfAllUniform("lowerBound", "upperBound")]
        [SkipIfAllUniform("X", "lowerBound")]
        [SkipIfAllUniform("X", "upperBound")]
    public static Gaussian XAverageConditional(bool isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            return XAverageConditional(Bernoulli.PointMass(isBetween), X, lowerBound, upperBound);
        }

        /// <summary>
        /// EP message to 'x'
        /// </summary>
        /// <param name="isBetween">Constant value for 'isBetween'.</param>
        /// <param name="X">Incoming message from 'x'.</param>
        /// <param name="lowerBound">Constant value for 'lowerBound'.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'x' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'x' as the random arguments are varied.
        /// The formula is <c>proj[p(x) sum_(upperBound) p(upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(x)</c>.
        /// </para></remarks>
        [SkipIfAllUniform("X", "upperBound")]
    public static Gaussian XAverageConditional(bool isBetween, [RequiredArgument] Gaussian X, double lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            return XAverageConditional(Bernoulli.PointMass(isBetween), X, Gaussian.PointMass(lowerBound), upperBound);
        }

        /// <summary>
        /// EP message to 'x'
        /// </summary>
        /// <param name="isBetween">Constant value for 'isBetween'.</param>
        /// <param name="X">Incoming message from 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
        /// <param name="upperBound">Constant value for 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'x' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'x' as the random arguments are varied.
        /// The formula is <c>proj[p(x) sum_(lowerBound) p(lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(x)</c>.
        /// </para></remarks>
        [SkipIfAllUniform("X", "lowerBound")]
    public static Gaussian XAverageConditional(bool isBetween, [RequiredArgument] Gaussian X, [RequiredArgument] Gaussian lowerBound, double upperBound)
        {
            return XAverageConditional(Bernoulli.PointMass(isBetween), X, lowerBound, Gaussian.PointMass(upperBound));
        }

        /// <summary>
        /// EP message to 'upperBound'
        /// </summary>
        /// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="X">Constant value for 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'upperBound' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'upperBound' as the random arguments are varied.
        /// The formula is <c>proj[p(upperBound) sum_(isBetween,lowerBound) p(isBetween,lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(upperBound)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
    public static Gaussian UpperBoundAverageConditional([SkipIfUniform] Bernoulli isBetween, double X, [RequiredArgument] Gaussian lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            return UpperBoundAverageConditional(isBetween, Gaussian.PointMass(X), lowerBound, upperBound);
        }

        /// <summary>
        /// EP message to 'upperBound'
        /// </summary>
        /// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="X">Incoming message from 'x'.</param>
        /// <param name="lowerBound">Constant value for 'lowerBound'.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'upperBound' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'upperBound' as the random arguments are varied.
        /// The formula is <c>proj[p(upperBound) sum_(isBetween,x) p(isBetween,x) factor(isBetween,x,lowerBound,upperBound)]/p(upperBound)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
        [SkipIfAllUniform("X", "upperBound")]
    public static Gaussian UpperBoundAverageConditional([SkipIfUniform] Bernoulli isBetween, [RequiredArgument] Gaussian X, double lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            return UpperBoundAverageConditional(isBetween, X, Gaussian.PointMass(lowerBound), upperBound);
        }

        /// <summary>
        /// EP message to 'upperBound'
        /// </summary>
        /// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="X">Constant value for 'x'.</param>
        /// <param name="lowerBound">Constant value for 'lowerBound'.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'upperBound' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'upperBound' as the random arguments are varied.
        /// The formula is <c>proj[p(upperBound) sum_(isBetween) p(isBetween) factor(isBetween,x,lowerBound,upperBound)]/p(upperBound)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
    public static Gaussian UpperBoundAverageConditional([SkipIfUniform] Bernoulli isBetween, double X, double lowerBound, [RequiredArgument] Gaussian upperBound)
        {
            return UpperBoundAverageConditional(isBetween, Gaussian.PointMass(X), Gaussian.PointMass(lowerBound), upperBound);
        }

        /// <summary>
        /// EP message to 'upperBound'
        /// </summary>
        /// <param name="isBetween">Constant value for 'isBetween'.</param>
        /// <param name="X">Incoming message from 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'upperBound' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'upperBound' as the random arguments are varied.
        /// The formula is <c>proj[p(upperBound) sum_(x,lowerBound) p(x,lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(upperBound)</c>.
        /// </para></remarks>
        [SkipIfAllUniform("X", "lowerBound")]
        [SkipIfAllUniform("X", "upperBound")]
        public static Gaussian UpperBoundAverageConditional(bool isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            return UpperBoundAverageConditional(Bernoulli.PointMass(isBetween), X, lowerBound, upperBound);
        }

        /// <summary>
        /// EP message to 'upperBound'
        /// </summary>
        /// <param name="isBetween">Constant value for 'isBetween'.</param>
        /// <param name="X">Constant value for 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'upperBound' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'upperBound' as the random arguments are varied.
        /// The formula is <c>proj[p(upperBound) sum_(lowerBound) p(lowerBound) factor(isBetween,x,lowerBound,upperBound)]/p(upperBound)</c>.
        /// </para></remarks>
        public static Gaussian UpperBoundAverageConditional(bool isBetween, double X, Gaussian lowerBound, Gaussian upperBound)
        {
            return UpperBoundAverageConditional(Bernoulli.PointMass(isBetween), Gaussian.PointMass(X), lowerBound, upperBound);
        }

        /// <summary>
        /// EP message to 'upperBound'
        /// </summary>
        /// <param name="isBetween">Constant value for 'isBetween'.</param>
        /// <param name="X">Incoming message from 'x'.</param>
        /// <param name="lowerBound">Constant value for 'lowerBound'.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'upperBound' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'upperBound' as the random arguments are varied.
        /// The formula is <c>proj[p(upperBound) sum_(x) p(x) factor(isBetween,x,lowerBound,upperBound)]/p(upperBound)</c>.
        /// </para></remarks>
        [SkipIfAllUniform("X", "upperBound")]
        public static Gaussian UpperBoundAverageConditional(bool isBetween, Gaussian X, double lowerBound, Gaussian upperBound)
        {
            return UpperBoundAverageConditional(Bernoulli.PointMass(isBetween), X, Gaussian.PointMass(lowerBound), upperBound);
        }

        /// <summary>
        /// EP message to 'upperBound'
        /// </summary>
        /// <param name="isBetween">Constant value for 'isBetween'.</param>
        /// <param name="X">Constant value for 'x'.</param>
        /// <param name="lowerBound">Constant value for 'lowerBound'.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'upperBound' argument</returns>
        /// <remarks><para>
        /// The outgoing message is the factor viewed as a function of 'upperBound' conditioned on the given values.
        /// </para></remarks>
        public static Gaussian UpperBoundAverageConditional(bool isBetween, double X, double lowerBound, Gaussian upperBound)
        {
            return UpperBoundAverageConditional(Bernoulli.PointMass(isBetween), Gaussian.PointMass(X), Gaussian.PointMass(lowerBound), upperBound);
        }

        /// <summary>
        /// EP message to 'lowerBound'
        /// </summary>
        /// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="X">Constant value for 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'lowerBound' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'lowerBound' as the random arguments are varied.
        /// The formula is <c>proj[p(lowerBound) sum_(isBetween,upperBound) p(isBetween,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(lowerBound)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
        public static Gaussian LowerBoundAverageConditional([SkipIfUniform] Bernoulli isBetween, double X, Gaussian lowerBound, Gaussian upperBound)
        {
            return LowerBoundAverageConditional(isBetween, Gaussian.PointMass(X), lowerBound, upperBound);
        }

        /// <summary>
        /// EP message to 'lowerBound'
        /// </summary>
        /// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="X">Incoming message from 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
        /// <param name="upperBound">Constant value for 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'lowerBound' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'lowerBound' as the random arguments are varied.
        /// The formula is <c>proj[p(lowerBound) sum_(isBetween,x) p(isBetween,x) factor(isBetween,x,lowerBound,upperBound)]/p(lowerBound)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
        [SkipIfAllUniform("X", "lowerBound")]
        public static Gaussian LowerBoundAverageConditional([SkipIfUniform] Bernoulli isBetween, Gaussian X, Gaussian lowerBound, double upperBound)
        {
            return LowerBoundAverageConditional(isBetween, X, lowerBound, Gaussian.PointMass(upperBound));
        }

        /// <summary>
        /// EP message to 'lowerBound'
        /// </summary>
        /// <param name="isBetween">Incoming message from 'isBetween'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="X">Constant value for 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
        /// <param name="upperBound">Constant value for 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'lowerBound' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'lowerBound' as the random arguments are varied.
        /// The formula is <c>proj[p(lowerBound) sum_(isBetween) p(isBetween) factor(isBetween,x,lowerBound,upperBound)]/p(lowerBound)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="isBetween"/> is not a proper distribution</exception>
        public static Gaussian LowerBoundAverageConditional([SkipIfUniform] Bernoulli isBetween, double X, Gaussian lowerBound, double upperBound)
        {
            return LowerBoundAverageConditional(isBetween, Gaussian.PointMass(X), lowerBound, Gaussian.PointMass(upperBound));
        }

        /// <summary>
        /// EP message to 'lowerBound'
        /// </summary>
        /// <param name="isBetween">Constant value for 'isBetween'.</param>
        /// <param name="X">Incoming message from 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'lowerBound' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'lowerBound' as the random arguments are varied.
        /// The formula is <c>proj[p(lowerBound) sum_(x,upperBound) p(x,upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(lowerBound)</c>.
        /// </para></remarks>
        [SkipIfAllUniform("X", "lowerBound")]
        [SkipIfAllUniform("X", "upperBound")]
        public static Gaussian LowerBoundAverageConditional(bool isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            return LowerBoundAverageConditional(Bernoulli.PointMass(isBetween), X, lowerBound, upperBound);
        }

        /// <summary>
        /// EP message to 'lowerBound'
        /// </summary>
        /// <param name="isBetween">Constant value for 'isBetween'.</param>
        /// <param name="X">Constant value for 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
        /// <param name="upperBound">Incoming message from 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'lowerBound' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'lowerBound' as the random arguments are varied.
        /// The formula is <c>proj[p(lowerBound) sum_(upperBound) p(upperBound) factor(isBetween,x,lowerBound,upperBound)]/p(lowerBound)</c>.
        /// </para></remarks>
        public static Gaussian LowerBoundAverageConditional(bool isBetween, double X, Gaussian lowerBound, Gaussian upperBound)
        {
            return LowerBoundAverageConditional(Bernoulli.PointMass(isBetween), Gaussian.PointMass(X), lowerBound, upperBound);
        }

        /// <summary>
        /// EP message to 'lowerBound'
        /// </summary>
        /// <param name="isBetween">Constant value for 'isBetween'.</param>
        /// <param name="X">Incoming message from 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
        /// <param name="upperBound">Constant value for 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'lowerBound' argument</returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'lowerBound' as the random arguments are varied.
        /// The formula is <c>proj[p(lowerBound) sum_(x) p(x) factor(isBetween,x,lowerBound,upperBound)]/p(lowerBound)</c>.
        /// </para></remarks>
        [SkipIfAllUniform("X", "lowerBound")]
        public static Gaussian LowerBoundAverageConditional(bool isBetween, Gaussian X, Gaussian lowerBound, double upperBound)
        {
            return LowerBoundAverageConditional(Bernoulli.PointMass(isBetween), X, lowerBound, Gaussian.PointMass(upperBound));
        }

        /// <summary>
        /// EP message to 'lowerBound'
        /// </summary>
        /// <param name="isBetween">Constant value for 'isBetween'.</param>
        /// <param name="X">Constant value for 'x'.</param>
        /// <param name="lowerBound">Incoming message from 'lowerBound'.</param>
        /// <param name="upperBound">Constant value for 'upperBound'.</param>
        /// <returns>The outgoing EP message to the 'lowerBound' argument</returns>
        /// <remarks><para>
        /// The outgoing message is the factor viewed as a function of 'lowerBound' conditioned on the given values.
        /// </para></remarks>
        public static Gaussian LowerBoundAverageConditional(bool isBetween, double X, Gaussian lowerBound, double upperBound)
        {
            return LowerBoundAverageConditional(Bernoulli.PointMass(isBetween), Gaussian.PointMass(X), lowerBound, Gaussian.PointMass(upperBound));
        }
#endif

        // ------------------AverageLogarithm -------------------------------------
        private const string NotSupportedMessage =
            "Variational Message Passing does not support the IsBetween factor with Gaussian distributions, since the factor is not conjugate to the Gaussian.";


        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="IsBetweenAverageLogarithm(Gaussian, Gaussian, Gaussian)"]/*'/>
        [NotSupported(IsBetweenGaussianOp.NotSupportedMessage)]
        public static Bernoulli IsBetweenAverageLogarithm(Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            throw new NotSupportedException(IsBetweenGaussianOp.NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="XAverageLogarithm(Bernoulli, Gaussian, Gaussian, Gaussian)"]/*'/>
        [NotSupported(IsBetweenGaussianOp.NotSupportedMessage)]
        public static Gaussian XAverageLogarithm(Bernoulli isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            throw new NotSupportedException(IsBetweenGaussianOp.NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LowerBoundAverageLogarithm(Bernoulli, Gaussian, Gaussian, Gaussian)"]/*'/>
        [NotSupported(IsBetweenGaussianOp.NotSupportedMessage)]
        public static Gaussian LowerBoundAverageLogarithm(Bernoulli isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            throw new NotSupportedException(IsBetweenGaussianOp.NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="UpperBoundAverageLogarithm(Bernoulli, Gaussian, Gaussian, Gaussian)"]/*'/>
        [NotSupported(IsBetweenGaussianOp.NotSupportedMessage)]
        public static Gaussian UpperBoundAverageLogarithm(Bernoulli isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            throw new NotSupportedException(IsBetweenGaussianOp.NotSupportedMessage);
        }

        private const string RandomBoundsNotSupportedMessage = "VMP does not support truncation with stochastic bounds.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="XAverageLogarithm(bool, Gaussian, double, double, Gaussian)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static Gaussian XAverageLogarithm(bool isBetween, [Stochastic] Gaussian X, double lowerBound, double upperBound, Gaussian to_X)
        {
            if (!isBetween)
                throw new ArgumentException($"{nameof(TruncatedGaussian)} requires {nameof(isBetween)}=true", nameof(isBetween));
            var prior = X / to_X;
            var tg = new TruncatedGaussian(prior);
            tg.LowerBound = lowerBound;
            tg.UpperBound = upperBound;
            return tg.ToGaussian() / prior;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="AverageLogFactor(bool, Gaussian, double, double, Gaussian)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static double AverageLogFactor(bool isBetween, [Stochastic] Gaussian X, double lowerBound, double upperBound, Gaussian to_X)
        {
            if (!isBetween)
                throw new ArgumentException($"{nameof(TruncatedGaussian)} requires {nameof(isBetween)}=true", nameof(isBetween));
            var prior = X / to_X;
            var tg = new TruncatedGaussian(prior);
            tg.LowerBound = lowerBound;
            tg.UpperBound = upperBound;
            return X.GetAverageLog(X) - tg.GetAverageLog(tg);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LowerBoundAverageLogarithm()"]/*'/>
        /// <remarks><para>
        /// Variational Message Passing does not support ConstrainBetween with Gaussian distributions, since the factor is not conjugate to the Gaussian.
        /// This method will throw an exception.
        /// </para></remarks>
        [NotSupported(RandomBoundsNotSupportedMessage)]
        public static Gaussian LowerBoundAverageLogarithm()
        {
            throw new NotSupportedException(RandomBoundsNotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="UpperBoundAverageLogarithm()"]/*'/>
        /// <remarks><para>
        /// Variational Message Passing does not support ConstrainBetween with Gaussian distributions, since the factor is not conjugate to the Gaussian.
        /// This method will throw an exception.
        /// </para></remarks>
        [NotSupported(RandomBoundsNotSupportedMessage)]
        public static Gaussian UpperBoundAverageLogarithm()
        {
            throw new NotSupportedException(RandomBoundsNotSupportedMessage);
        }
    }
}
