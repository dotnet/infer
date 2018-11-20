// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "IsBetween", typeof(double), typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Mature)]
    [Buffers("logZ")]
    public static class DoubleIsBetweenOp
    {
        /// <summary>
        /// Static flag to force a proper distribution
        /// </summary>
        public static bool ForceProper = true;
        public static double LowPrecisionThreshold = 1e-10;

        //-- TruncatedGaussian bounds ------------------------------------------------------------------------------

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
                double mx, vx;
                X.GetMeanAndVariance(out mx, out vx);
                double center = MMath.Average(lowerBound, upperBound);
                if (d_p == 1.0)
                {
                    double diff = upperBound - lowerBound;
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
                        double logPhiL = Gaussian.GetLogProb(lowerBound, mx, vx);
                        double alphaL = d_p * Math.Exp(logPhiL - logZ);
                        double logPhiU = Gaussian.GetLogProb(upperBound, mx, vx);
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
                    if (delta < 0) throw new Exception($"{nameof(delta)} < 0");
                    if (delta < 1e-16 && (mx <= lowerBound || mx >= upperBound))
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
                    if (zU < -1e20)
                    {
                        // in this regime, rU = -1/zU, r1U = rU*rU
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
                            // Abs is needed to avoid some 32-bit oddities.
                            double prec2 = (expMinus1Ratio * expMinus1Ratio) /
                                Math.Abs(r1U / X.Precision * expMinus1 * expMinus1RatioMinus1RatioMinusHalf
                                + rU / sqrtPrec * diff * (expMinus1RatioMinus1RatioMinusHalf - delta / 2 * (expMinus1RatioMinus1RatioMinusHalf + 1))
                                + diff * diff / 4);
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
                        double prec2 = rU * rU * X.Precision;
                        if (prec2 != 0) // avoid 0/0
                            prec2 /= NormalCdfRatioSqrMinusDerivative(zU, rU, r1U, r3U);
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
                    return new Gaussian(mp, vp) / X;
                }
                else
                {
                    double logZ = LogAverageFactor(isBetween, X, lowerBound, upperBound);
                    if (d_p == -1.0 && logZ < double.MinValue)
                    {
                        if (mx == center)
                        {
                            // The posterior is two point masses.
                            // Compute the moment-matched Gaussian and divide.
                            double diff = upperBound - lowerBound;
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
                    double logPhiL = Gaussian.GetLogProb(lowerBound, mx, vx);
                    double alphaL = d_p * Math.Exp(logPhiL - logZ);
                    double logPhiU = Gaussian.GetLogProb(upperBound, mx, vx);
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
                // r*r = (r1*r1 - 2r1 +1)/z^2
                // r*r - r1 = (r1*r1 - 2r1 + 1 - z^2*r1)/z^2 = (r1*r1 + 1 - r3 + z*r)/z^2 = (r1*r1 + r1 - r3)/z^2
                // r3 = z*r2 + 2*r1 = z^2*r1 + z*r + 2*r1
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
        internal static void GetDiffMeanAndVariance(Gaussian X, Gaussian L, Gaussian U, out double yl, out double yu, out double r, out double invSqrtVxl,
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
            }
            else
            {
                //r = -vx * invSqrtVxl * invSqrtVxu;
                // This is a more accurate way to compute the above.
                r = -1 / Math.Sqrt(1 + vl / vx) / Math.Sqrt(1 + vu / vx);
                if (r < -1 || r > 1)
                    throw new Exception("Internal: r is outside [-1,1]");
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
                double yl, yu, r, invSqrtVxl, invSqrtVxu;
                GetDiffMeanAndVariance(X, L, U, out yl, out yu, out r, out invSqrtVxl, out invSqrtVxu);
                double logp = MMath.NormalCdfLn(yl, yu, r);
                if (logp > 0)
                    throw new Exception("LogProbBetween is positive");
                return logp;
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
            if (X.IsUniform())
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
                    double alpha = Math.Exp(Gaussian.GetLogProb(ml, mu, vlu) - MMath.NormalCdfLn((mu - ml) / Math.Sqrt(vlu)));
                    double alphaU = 1.0 / (mu - ml + vlu * alpha);
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
                bool precisionWasZero = false;
                if (Double.IsNegativeInfinity(logZ))
                {
                    if (X.Precision < LowPrecisionThreshold)
                    {
                        precisionWasZero = true;
                        X.Precision = LowPrecisionThreshold;
                        logZ = LogAverageFactor(isBetween, X, lowerBound, upperBound);
                    }
                }
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
                        double mx = X.GetMean();
                        if (mx < L && d_p == 1)
                        {
                            // Z = d_p (MMath.NormalCdf(sqrtPrec*(mx-L)) - MMath.NormalCdf(sqrtPrec*(mx-U)))
                            // Z/X.GetProb(L)*sqrtPrec = MMath.NormalCdfRatio(sqrtPrec*(mx-L)) - MMath.NormalCdfRatio(sqrtPrec*(mx-U))*X.GetProb(U)/X.GetProb(L)
                            // X.GetProb(U)/X.GetProb(L) = Math.Exp(X.MeanTimesPrecision*(U-L) - 0.5*(U*U - L*L)*X.Precision) =approx 0
                            double sqrtPrec = Math.Sqrt(X.Precision);
                            double U = upperBound.Point;
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
                        return IsPositiveOp.XAverageConditional(false, DoublePlusOp.AAverageConditional(lowerBound, upperBound.Point));
                    }
                    else
                    {
                        return IsPositiveOp.XAverageConditional(false, DoublePlusOp.AAverageConditional(lowerBound, X.Point));
                    }
                }
                double yl, yu, r, invSqrtVxl, invSqrtVxu;
                GetDiffMeanAndVariance(X, lowerBound, upperBound, out yl, out yu, out r, out invSqrtVxl, out invSqrtVxu);
                // if we get here, we know that -1 < r <= 0 and invSqrtVxl is finite
                // since lowerBound is not uniform and X is not uniform, invSqrtVxl > 0
                // yl is always finite.  yu may be +/-infinity, in which case r = 0.
                double alphaL;
                if (r == 0)
                {
                    alphaL = -d_p * invSqrtVxl / MMath.NormalCdfRatio(yl);
                }
                else
                {
                    double logPhiL = Gaussian.GetLogProb(yl, 0, 1) + MMath.NormalCdfLn((yu - r * yl) / Math.Sqrt(1 - r * r));
                    alphaL = -d_p * invSqrtVxl * Math.Exp(logPhiL - logZ);
                }
                // (mx - ml) / (vl + vx) = yl*invSqrtVxl
                double betaL = alphaL * (alphaL - yl * invSqrtVxl);
                if (r > -1 && r != 0 && !precisionWasZero)
                {
                    double logPhiR = -2 * MMath.LnSqrt2PI - 0.5 * Math.Log(1 - r * r) - 0.5 * (yl * yl + yu * (yu - 2 * r * yl)) / (1 - r * r);
                    double c = d_p * r * Math.Exp(logPhiR - logZ);
                    betaL += c * invSqrtVxl * invSqrtVxl;
                }
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
            Gaussian result = new Gaussian();
            if (isBetween.IsUniform())
                return result;
            if (X.IsUniform())
            {
                if (lowerBound.IsUniform() || upperBound.IsUniform())
                {
                    result.SetToUniform();
                }
                else if (isBetween.IsPointMass && isBetween.Point)
                {
                    double ml, vl, mu, vu;
                    lowerBound.GetMeanAndVariance(out ml, out vl);
                    upperBound.GetMeanAndVariance(out mu, out vu);
                    double vlu = vl + vu;
                    double alpha = Math.Exp(Gaussian.GetLogProb(ml, mu, vlu) - MMath.NormalCdfLn((mu - ml) / Math.Sqrt(vlu)));
                    double alphaU = 1.0 / (mu - ml + vlu * alpha);
                    double betaU = alphaU * (alphaU - alpha);
                    result.SetMeanAndVariance(mu + vu * alphaU, vu - vu * vu * betaU);
                    result.SetToRatio(result, upperBound);
                }
                else
                    throw new NotImplementedException();
            }
            else if (upperBound.IsUniform())
            {
                if (isBetween.IsPointMass && !isBetween.Point)
                {
                    // lowerBound <= upperBound <= X
                    // upperBound is not a point mass so upperBound==X is impossible
                    return XAverageConditional(Bernoulli.PointMass(true), upperBound, lowerBound, X, logZ);
                }
                else
                {
                    result.SetToUniform();
                }
            }
            else
            {
                bool precisionWasZero = false;
                if (Double.IsNegativeInfinity(logZ))
                {
                    if (X.Precision < LowPrecisionThreshold)
                    {
                        precisionWasZero = true;
                        X.Precision = LowPrecisionThreshold;
                        logZ = LogAverageFactor(isBetween, X, lowerBound, upperBound);
                    }
                }
                double d_p = 2 * isBetween.GetProbTrue() - 1;
                if (upperBound.IsPointMass)
                {
                    if (double.IsPositiveInfinity(upperBound.Point)) return Gaussian.Uniform();
                    if (X.IsPointMass)
                    {
                        if (upperBound.Point > X.Point) return Gaussian.Uniform();
                        else return X;
                    }
                    if (lowerBound.IsPointMass)
                    {
                        // r = -1 case
                        // X is not uniform or point mass
                        // f(U) = d_p (NormalCdf((U-mx)*sqrtPrec) - NormalCdf((L-mx)*sqrtPrec)) + const.
                        // dlogf/dU = d_p N(U;mx,vx)/f
                        // ddlogf = -dlogf^2 + dlogf*(mx-U)/vx
                        double U = upperBound.Point;
                        double mx = X.GetMean();
                        if (mx > U && d_p == 1)
                        {
                            // Z = MMath.NormalCdf(sqrtPrec*(U-mx)) - MMath.NormalCdf(sqrtPrec*(L-mx))
                            // Z/X.GetProb(U)*sqrtPrec = MMath.NormalCdfRatio(sqrtPrec*(U-mx)) - MMath.NormalCdfRatio(sqrtPrec*(L-mx))*X.GetProb(L)/X.GetProb(U)
                            // X.GetProb(L)/X.GetProb(U) = Math.Exp(-X.MeanTimesPrecision*(U-L) + 0.5*(U*U - L*L)*X.Precision) =approx 0
                            double sqrtPrec = Math.Sqrt(X.Precision);
                            double L = lowerBound.Point;
                            double Umx = sqrtPrec * (U - mx);
                            double UCdfRatio = MMath.NormalCdfRatio(Umx);
                            double LCdfRatio = MMath.NormalCdfRatio(sqrtPrec * (L - mx));
                            double LPdfRatio = Math.Exp((U - L) * (-X.MeanTimesPrecision + 0.5 * (U + L) * X.Precision));
                            double CdfRatioDiff = UCdfRatio - LCdfRatio * LPdfRatio;
                            double dlogf = d_p / CdfRatioDiff * sqrtPrec;
                            // ddlogf = -dlogf * (d_p * sqrtPrec / CdfRatioDiff + sqrtPrec*Umx)
                            //        = -dlogf * sqrtPrec * (d_p / CdfRatioDiff + Umx)
                            //        = -dlogf * sqrtPrec * (d_p + Umx * CdfRatioDiff) / CdfRatioDiff
                            //        = -dlogf * sqrtPrec * (d_p - 1 - Umx * LCdfRatio * LPdfRatio + NormalCdfMomentRatio(1,Umx)) / CdfRatioDiff
                            double ddlogf = -dlogf * dlogf * (Umx * (CdfRatioDiff - UCdfRatio) + MMath.NormalCdfMomentRatio(1, Umx));
                            return Gaussian.FromDerivatives(U, dlogf, ddlogf, ForceProper);
                            // this is equivalent
                            // U - dlogf/ddlogf = U - 1/(-dlogf + (X.MeanTimesPrecision - U * X.Precision)) =approx mx
                            //return Gaussian.FromMeanAndPrecision(U - dlogf / ddlogf, -ddlogf);
                        }
                        else
                        {
                            double dlogf = d_p * Math.Exp(X.GetLogProb(U) - logZ);
                            double ddlogf = dlogf * (-dlogf + (X.MeanTimesPrecision - U * X.Precision));
                            return Gaussian.FromDerivatives(U, dlogf, ddlogf, ForceProper);
                        }
                    }
                }
                if (X.IsPointMass)
                {
                    if (lowerBound.IsPointMass && lowerBound.Point > X.Point)
                    {
                        return IsPositiveOp.XAverageConditional(true, DoublePlusOp.AAverageConditional(upperBound, lowerBound.Point));
                    }
                    else
                    {
                        return IsPositiveOp.XAverageConditional(true, DoublePlusOp.AAverageConditional(upperBound, X.Point));
                    }
                }
                double yl, yu, r, invSqrtVxl, invSqrtVxu;
                GetDiffMeanAndVariance(X, lowerBound, upperBound, out yl, out yu, out r, out invSqrtVxl, out invSqrtVxu);
                // if we get here, -1 < r <= 0 and invSqrtVxu is finite
                // since upperBound is not uniform and X is not uniform, invSqrtVxu > 0
                // yu is always finite.  yl may be infinity, in which case r = 0.
                double alphaU;
                if (r == 0)
                {
                    alphaU = d_p * invSqrtVxu / MMath.NormalCdfRatio(yu);
                }
                else
                {
                    double logPhiU = Gaussian.GetLogProb(yu, 0, 1) + MMath.NormalCdfLn((yl - r * yu) / Math.Sqrt(1 - r * r));
                    alphaU = d_p * invSqrtVxu * Math.Exp(logPhiU - logZ);
                }
                // (mu - mx) / (vx + vu) = yu*invSqrtVxu
                double betaU = alphaU * (alphaU + yu * invSqrtVxu);
                if (r > -1 && r != 0 && !precisionWasZero)
                {
                    // (yu * yu + yl * (yl - 2 * r * yu))
                    // ((mu-mx)*(mu-mx)/(vx+vu) + (mx-ml)*invSqrtVxl * ((mx-ml)*invSqrtVxl - 2 * r * (mu-mx)*invSqrtVxu))
                    // ((mu-mx)*(mu-mx)/(vx+vu) + (mx-ml) * ((mx-ml) + 2 *vx * invSqrtVxu* (mu-mx)*invSqrtVxu)/(vx+vl))
                    // ((mu-mx)*(mu-mx)/(vx+vu) + (mx-ml) * ((mx-ml) + 2 *vx * (mu-mx)/(vx+vu))/(vx+vl))
                    // (mu-mx)*(mu-mx)/(vx+vu) + (mx-ml) * ((mx-ml)*(vx+vu) + (2*vx*mu-2*vx*mx))/(vx+vu)/(vx+vl)
                    // (mu-mx)*(mu-mx) + (mx-ml) * ((mx-ml)*(vx+vu) + (2*vx*mu-2*vx*mx))/(vx+vl)
                    // (mu-mx)*(mu-mx) + (mx-ml) * ((mx-ml)*vu + (2*mu-ml-mx)*vx)/(vx+vl)
                    // (mu-mx)*(mu-mx)*(vx+vl) + (mx-ml) * ((mx-ml)*vu + (2*mu-ml-mx)*vx)
                    // ((mu-mx)*(mu-mx) + (mx-ml)*(2*mu-ml-mx))*vx + (mu-mx)*(mu-mx)*vl + (mx-ml)*(mx-ml)*vu
                    // (mu-ml)*(mu-ml)*vx + (mu-mx)*(mu-mx)*vl + (mx-ml)*(mx-ml)*vu
                    double logPhiR = -2 * MMath.LnSqrt2PI - 0.5 * Math.Log(1 - r * r);
                    if (r > -0.99)
                    {
                        logPhiR -= 0.5 * (yu * yu + yl * (yl - 2 * r * yu)) / (1 - r * r);
                    }
                    else
                    {
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
                        logPhiR -= 0.5 * (dmul * dmul * vx + dmux * dmux * vl + dmxl * dmxl * vu) / (vx*vl + vx*vu + vl*vu);
                    }
                    double c = d_p * r * Math.Exp(logPhiR - logZ);
                    betaU += c * invSqrtVxu * invSqrtVxu;
                }
                return GaussianOp.GaussianFromAlphaBeta(upperBound, alphaU, betaU, ForceProper);
            }
            if (Double.IsNaN(result.Precision) || Double.IsNaN(result.MeanTimesPrecision))
                throw new InferRuntimeException($"{nameof(result)} is NaN.  {nameof(isBetween)}={isBetween}, {nameof(X)}={X}, {nameof(lowerBound)}={lowerBound}, {nameof(upperBound)}={upperBound}, {nameof(logZ)}={logZ}");
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
            else if (X.IsUniform())
            {
                if (lowerBound.IsUniform() || upperBound.IsUniform() ||
                    (lowerBound.IsPointMass && Double.IsInfinity(lowerBound.Point)) ||
                    (upperBound.IsPointMass && Double.IsInfinity(upperBound.Point)) ||
                    !Double.IsPositiveInfinity(isBetween.LogOdds))
                {
                    result.SetToUniform();
                }
                else if (isBetween.IsPointMass && isBetween.Point)
                {
                    double ml, vl, mu, vu;
                    lowerBound.GetMeanAndVariance(out ml, out vl);
                    upperBound.GetMeanAndVariance(out mu, out vu);
                    double vlu = vl + vu; // vlu > 0
                    double alpha = Math.Exp(Gaussian.GetLogProb(ml, mu, vlu) - MMath.NormalCdfLn((mu - ml) / Math.Sqrt(vlu)));
                    double alphaU = 1.0 / (mu - ml + vlu * alpha);
                    double betaU = alphaU * (alphaU - alpha);
                    double munew = mu + vu * alphaU;
                    double mlnew = ml - vl * alphaU;
                    double vunew = vu - vu * vu * betaU;
                    double vlnew = vl - vl * vl * betaU;
                    double diff = munew - mlnew;
                    result.SetMeanAndVariance((munew + mlnew) / 2, diff * diff / 12 + (vunew + vlnew + vu * vl * betaU) / 3);
                }
                else
                    throw new NotImplementedException();
            }
            else
            {
                // X is not a point mass or uniform
                bool precisionWasZero = false;
                if (Double.IsNegativeInfinity(logZ))
                {
                    if (X.Precision < LowPrecisionThreshold)
                    {
                        precisionWasZero = true;
                        X.Precision = LowPrecisionThreshold;
                        logZ = LogAverageFactor(isBetween, X, lowerBound, upperBound);
                    }
                    if (Double.IsNegativeInfinity(logZ))
                        throw new AllZeroException();
                }
                double d_p = 2 * isBetween.GetProbTrue() - 1;
                double yl, yu, r, invSqrtVxl, invSqrtVxu;
                GetDiffMeanAndVariance(X, lowerBound, upperBound, out yl, out yu, out r, out invSqrtVxl, out invSqrtVxu);
                // r == -1 iff lowerBound and upperBound are point masses
                // since X is not a point mass, invSqrtVxl is finite, invSqrtVxu is finite
                double alphaL = 0.0;
                if (X.IsPointMass && !Double.IsInfinity(yl))
                {
                    alphaL = -d_p * invSqrtVxl / MMath.NormalCdfRatio(yl);
                }
                else if (!lowerBound.IsUniform() && !Double.IsInfinity(yl))
                {
                    // since X and lowerBound are not both uniform, invSqrtVxl > 0
                    double logPhiL = Gaussian.GetLogProb(yl, 0, 1);
                    if (r > -1)
                        logPhiL += MMath.NormalCdfLn((yu - r * yl) / Math.Sqrt(1 - r * r));
                    alphaL = -d_p * invSqrtVxl * Math.Exp(logPhiL - logZ);
                    // TODO: make this case smoothly blend into the X.IsPointMass case
                }
                double alphaU = 0.0;
                if (X.IsPointMass && !Double.IsInfinity(yu))
                {
                    alphaU = d_p * invSqrtVxu / MMath.NormalCdfRatio(yu);
                }
                else if (!upperBound.IsUniform() && !Double.IsInfinity(yu))
                {
                    // since X and upperBound are not both uniform, invSqrtVxu > 0
                    double logPhiU = Gaussian.GetLogProb(yu, 0, 1);
                    if (r > -1)
                        logPhiU += MMath.NormalCdfLn((yl - r * yu) / Math.Sqrt(1 - r * r));
                    alphaU = d_p * invSqrtVxu * Math.Exp(logPhiU - logZ);
                }
                double alphaX = -alphaL - alphaU;
                // (mx - ml) / (vl + vx) = yl*invSqrtVxl
                double betaX = alphaX * alphaX;
                if (!Double.IsInfinity(yl))
                {
                    betaX -= alphaL * (yl * invSqrtVxl);
                }
                if (!Double.IsInfinity(yu))
                {
                    betaX += alphaU * (yu * invSqrtVxu);
                }
                if (r > -1 && r != 0 && !Double.IsInfinity(yl) && !Double.IsInfinity(yu) && !precisionWasZero)
                {
                    double logPhiR = -2 * MMath.LnSqrt2PI - 0.5 * Math.Log(1 - r * r) - 0.5 * (yl * yl + yu * yu - 2 * r * yl * yu) / (1 - r * r);
                    double c = d_p * r * Math.Exp(logPhiR - logZ);
                    betaX += c * (-2 * X.Precision + invSqrtVxl * invSqrtVxl + invSqrtVxu * invSqrtVxu);
                }
                return GaussianOp.GaussianFromAlphaBeta(X, alphaX, betaX, ForceProper);
            }
            return result;
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
        [NotSupported(DoubleIsBetweenOp.NotSupportedMessage)]
        public static Bernoulli IsBetweenAverageLogarithm(Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            throw new NotSupportedException(DoubleIsBetweenOp.NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="XAverageLogarithm(Bernoulli, Gaussian, Gaussian, Gaussian)"]/*'/>
        [NotSupported(DoubleIsBetweenOp.NotSupportedMessage)]
        public static Gaussian XAverageLogarithm(Bernoulli isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            throw new NotSupportedException(DoubleIsBetweenOp.NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="LowerBoundAverageLogarithm(Bernoulli, Gaussian, Gaussian, Gaussian)"]/*'/>
        [NotSupported(DoubleIsBetweenOp.NotSupportedMessage)]
        public static Gaussian LowerBoundAverageLogarithm(Bernoulli isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            throw new NotSupportedException(DoubleIsBetweenOp.NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/message_doc[@name="UpperBoundAverageLogarithm(Bernoulli, Gaussian, Gaussian, Gaussian)"]/*'/>
        [NotSupported(DoubleIsBetweenOp.NotSupportedMessage)]
        public static Gaussian UpperBoundAverageLogarithm(Bernoulli isBetween, Gaussian X, Gaussian lowerBound, Gaussian upperBound)
        {
            throw new NotSupportedException(DoubleIsBetweenOp.NotSupportedMessage);
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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleIsBetweenOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "IsBetween", typeof(double), typeof(double), typeof(double))]
    [Quality(QualityBand.Mature)]
    public static class TruncatedGaussianIsBetweenOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="TruncatedGaussianIsBetweenOp"]/message_doc[@name="AverageLogFactor(TruncatedGaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor([IgnoreDependency] TruncatedGaussian X)
        {
            return 0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="TruncatedGaussianIsBetweenOp"]/message_doc[@name="LogEvidenceRatio(bool, TruncatedGaussian, double, double)"]/*'/>
        public static double LogEvidenceRatio(bool isBetween, [SkipIfUniform] TruncatedGaussian x, double lowerBound, double upperBound)
        {
            return x.GetLogAverageOf(XAverageConditional(isBetween, lowerBound, upperBound));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="TruncatedGaussianIsBetweenOp"]/message_doc[@name="XAverageConditional(bool, double, double)"]/*'/>
        public static TruncatedGaussian XAverageConditional(bool isBetween, double lowerBound, double upperBound)
        {
            if (!isBetween)
                throw new ArgumentException($"{nameof(TruncatedGaussian)} requires {nameof(isBetween)}=true", nameof(isBetween));
            return new TruncatedGaussian(0, Double.PositiveInfinity, lowerBound, upperBound);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="TruncatedGaussianIsBetweenOp"]/message_doc[@name="XAverageLogarithm(bool, double, double)"]/*'/>
        public static TruncatedGaussian XAverageLogarithm(bool isBetween, double lowerBound, double upperBound)
        {
            return XAverageConditional(isBetween, lowerBound, upperBound);
        }
    }
}
