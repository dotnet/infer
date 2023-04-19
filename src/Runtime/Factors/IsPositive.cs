// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Distributions;
    using Math;
    using Attributes;
    using Utilities;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "IsPositive", typeof(double))]
    [Quality(QualityBand.Mature)]
    public static class IsPositiveOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="LogAverageFactor(bool, double)"]/*'/>
        public static double LogAverageFactor(bool isPositive, double x)
        {
            return (isPositive == Factor.IsPositive(x)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="LogEvidenceRatio(bool, double)"]/*'/>
        public static double LogEvidenceRatio(bool isPositive, double x)
        {
            return LogAverageFactor(isPositive, x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="AverageLogFactor(bool, double)"]/*'/>
        public static double AverageLogFactor(bool isPositive, double x)
        {
            return LogAverageFactor(isPositive, x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="LogAverageFactor(bool, Gaussian)"]/*'/>
        public static double LogAverageFactor(bool isPositive, Gaussian x)
        {
            return LogAverageFactor(Bernoulli.PointMass(isPositive), x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="LogAverageFactor(Bernoulli, Gaussian)"]/*'/>
        public static double LogAverageFactor(Bernoulli isPositive, Gaussian x)
        {
            if (isPositive.IsPointMass && x.Precision == 0)
            {
                double tau = x.MeanTimesPrecision;
                if (isPositive.Point && tau < 0)
                {
                    // int I(x>0) exp(tau*x) dx = -1/tau
                    return -Math.Log(-tau);
                }
                if (!isPositive.Point && tau > 0)
                {
                    // int I(x<0) exp(tau*x) dx = 1/tau
                    return -Math.Log(tau);
                }
            }
            Bernoulli to_isPositive = IsPositiveAverageConditional(x);
            return isPositive.GetLogAverageOf(to_isPositive);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="LogEvidenceRatio(Bernoulli, Gaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli isPositive, Gaussian x)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="LogEvidenceRatio(bool, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(bool isPositive, Gaussian x)
        {
            return LogAverageFactor(isPositive, x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="IsPositiveAverageConditional(Gaussian)"]/*'/>
        public static Bernoulli IsPositiveAverageConditional([SkipIfUniform, Proper] Gaussian x)
        {
            Bernoulli result = new Bernoulli();
            if (x.IsPointMass)
            {
                result.Point = Factor.IsPositive(x.Point);
            }
            else if (x.IsUniform())
            {
                result.LogOdds = 0.0;
            }
            else if (!x.IsProper())
            {
                throw new ImproperMessageException(x);
            }
            else
            {
                // m/sqrt(v) = (m/v)/sqrt(1/v)
                double z = x.MeanTimesPrecision / Math.Sqrt(x.Precision);
                // p(true) = NormalCdf(z)
                // log(p(true)/p(false)) = NormalCdfLogit(z)
                result.LogOdds = MMath.NormalCdfLogit(z);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="IsPositiveAverageConditionalInit()"]/*'/>
        [Skip]
        public static Bernoulli IsPositiveAverageConditionalInit()
        {
            return new Bernoulli();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="XAverageConditional(Bernoulli, Gaussian)"]/*'/>
        public static Gaussian XAverageConditional([SkipIfUniform] Bernoulli isPositive, [SkipIfUniform, Proper] Gaussian x)
        {
            return XAverageConditional_Helper(isPositive, x, false);
        }

        public static Gaussian XAverageConditional_Helper([SkipIfUniform] Bernoulli isPositive, [SkipIfUniform, Proper] Gaussian x, bool forceProper)
        {
            if (x.IsPointMass)
            {
                if (isPositive.IsPointMass && (isPositive.Point != (x.Point > 0)))
                    return Gaussian.PointMass(0);
                else
                    return Gaussian.Uniform();
            }
            double tau = x.MeanTimesPrecision;
            double prec = x.Precision;
            if (prec == 0.0)
            {
                if (isPositive.IsPointMass)
                {
                    if ((isPositive.Point && tau < 0) ||
                        (!isPositive.Point && tau > 0))
                    {
                        // posterior is proportional to I(x>0) exp(tau*x)
                        //double mp = -1 / tau;
                        //double vp = mp * mp;
                        return Gaussian.FromNatural(-tau, tau * tau) / x;
                    }
                }
                return Gaussian.Uniform();
            }
            else if (prec < 0)
            {
                throw new ImproperMessageException(x);
            }
            double sqrtPrec = Math.Sqrt(prec);
            // m/sqrt(v) = (m/v)/sqrt(1/v)
            double z = tau / sqrtPrec;
            // epsilon = p(b=F)
            // eq (51) in EP quickref
            double alpha;
            if (isPositive.IsPointMass)
            {
                if ((isPositive.Point && z < -10) || (!isPositive.Point && z > 10))
                {
                    if (z > 10) z = -z;
                    //double Y = MMath.NormalCdfRatio(z);
                    // dY = z*Y + 1
                    // d2Y = z*dY + Y
                    // posterior mean = m + sqrt(v)/Y = sqrt(v)*(z + 1/Y) = sqrt(v)*dY/Y 
                    //                = sqrt(v)*(d2Y/Y - 1)/z = (d2Y/Y - 1)/tau
                    //                =approx sqrt(v)*(-1/z) = -1/tau
                    // posterior variance = v - v*dY/Y^2 =approx v/z^2
                    // posterior E[x^2] = v - v*dY/Y^2 + v*dY^2/Y^2 = v - v*dY/Y^2*(1 - dY) = v + v*z*dY/Y = v*d2Y/Y
                    // d3Y = z*d2Y + 2*dY
                    //     = z*(z*dY + Y) + 2*dY 
                    //     = z^2*dY + z*Y + 2*dY
                    //     = z^2*dY + 3*dY - 1
                    double d3Y = 6 * MMath.NormalCdfMomentRatio(3, z);
                    //double dY = MMath.NormalCdfMomentRatio(1, z);
                    double dY = (d3Y + 1) / (z * z + 3);
                    if (MMath.AreEqual(dY, 0))
                    {
                        double tau2 = tau * tau;
                        if (tau2 > double.MaxValue) return Gaussian.PointMass(-1 / tau);
                        else return Gaussian.FromNatural(-2 * tau, tau2);
                    }
                    // Y = (dY-1)/z
                    // alpha = sqrtPrec*z/(dY-1) = tau/(dY-1)
                    // alpha+tau = tau*(1 + 1/(dY-1)) = tau*dY/(dY-1) = alpha*dY
                    // beta = alpha * (alpha + tau)
                    //      = alpha * tau * dY / (dY - 1)
                    //      = prec * z^2 * dY/(dY-1)^2
                    // prec/beta = (dY-1)^2/(dY*z^2)
                    // prec/beta - 1 = ((dY-1)^2 - dY*z^2)/(dY*z^2)
                    // weight = beta/(prec - beta) 
                    //        = dY*z^2/((dY-1)^2 - dY*z^2)
                    //        = dY*z^2/(dY^2 -2*dY + 1 - dY*z^2)
                    //        = dY*z^2/(dY^2 + 1 + z*Y - d3Y)
                    //        = dY*z^2/(dY^2 + dY - d3Y)
                    //        = z^2/(dY + 1 - d3Y/dY)
                    double d3YidY = d3Y / dY;
                    double denom = dY - d3YidY + 1;
                    double msgPrec = tau * tau / denom;
                    // weight * (tau + alpha) + alpha 
                    // = z^2/(dY + 1 - d3Y/dY) * alpha*dY + alpha
                    // = alpha*(z^2*dY/(dY + 1 - d3Y/dY) + 1)
                    // = alpha*(z^2*dY + dY + 1 - d3Y/dY)/(dY + 1 - d3Y/dY)
                    // = alpha*(z^2*dY + dY + 1 - d3Y/dY)/(dY + 1 - d3Y/dY)
                    // = alpha*(d3Y - 2*dY + 2 - d3Y/dY)/(dY + 1 - d3Y/dY)
                    // z^2*dY = d3Y - 3*dY + 1
                    double numer = (d3Y - 2 * dY - d3YidY + 2) / (dY - 1);
                    double msgMeanTimesPrec = tau * numer / denom;
                    if (msgPrec > double.MaxValue)
                    {
                        // In this case, the message should be the posterior.
                        // posterior mean = (msgMeanTimesPrec + tau)*denom/(tau*tau)
                        // = (numer + denom)/tau
                        return Gaussian.PointMass((numer + denom) / tau);
                    }
                    else return Gaussian.FromNatural(msgMeanTimesPrec, msgPrec);
                }
                else if (isPositive.Point)
                {
                    alpha = sqrtPrec / MMath.NormalCdfRatio(z);
                }
                else
                {
                    alpha = -sqrtPrec / MMath.NormalCdfRatio(-z);
                }
            }
            else
            {
                //double v = MMath.LogSumExp(isPositive.GetLogProbTrue() + MMath.NormalCdfLn(z), isPositive.GetLogProbFalse() + MMath.NormalCdfLn(-z));
                double v = LogAverageFactor(isPositive, x);
                alpha = sqrtPrec * Math.Exp(-z * z * 0.5 - MMath.LnSqrt2PI - v) * (2 * isPositive.GetProbTrue() - 1);
            }
            // eq (52) in EP quickref (where tau = mnoti/Vnoti)
            double beta;
            if (alpha == 0)
                beta = 0;  // avoid 0 * infinity
            else
                beta = alpha * (alpha + tau);
            double weight = beta / (prec - beta);
            if (forceProper && weight < 0)
                weight = 0;
            Gaussian result = new Gaussian();
            if (weight == 0)
            {
                // avoid 0 * infinity
                result.MeanTimesPrecision = alpha;
            }
            else
            {
                // eq (31) in EP quickref; same as inv(inv(beta)-inv(prec))
                result.Precision = prec * weight;
                // eq (30) in EP quickref times above and simplified
                result.MeanTimesPrecision = weight * (tau + alpha) + alpha;
            }
            if (double.IsNaN(result.Precision) || double.IsNaN(result.MeanTimesPrecision))
                throw new InferRuntimeException($"result is NaN.  isPositive={isPositive}, x={x}, forceProper={forceProper}");
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="XAverageConditional(Bernoulli, double)"]/*'/>
        public static Gaussian XAverageConditional([SkipIfUniform] Bernoulli isPositive, double x)
        {
            return XAverageConditional(isPositive, Gaussian.PointMass(x));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="XAverageConditional(bool, Gaussian)"]/*'/>
        public static Gaussian XAverageConditional(bool isPositive, [SkipIfUniform] Gaussian x)
        {
            return XAverageConditional(Bernoulli.PointMass(isPositive), x);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="XAverageConditionalInit()"]/*'/>
        [Skip]
        public static Gaussian XAverageConditionalInit()
        {
            return Gaussian.Uniform();
        }

        //-- VMP ---------------------------------------------------------------------------------------------
        private const string NotSupportedMessage =
            "Variational Message Passing does not support the IsPositive factor with Gaussian distributions, since the factor is not conjugate to the Gaussian.";

        private const string NotSupportedMessage2 =
            "Variational Message Passing does not support the IsPositive factor with stochastic output and Gaussian distributions, since the factor is not conjugate to the Gaussian.";

#if true
        //-----------------------------------------------------------------------------------------------------------
        // We can calculate VMP messages by implicitly modelling X as a TruncatedGaussian. This factor hides this
        // details of the implementation from the user. Note we cannot have multiple uses of this variable
        // in this case. To do this requires using truncated Gaussians explicitly. 

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="XAverageLogarithm(bool, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static Gaussian XAverageLogarithm(bool isPositive, [SkipIfUniform, Stochastic] Gaussian x, Gaussian to_X)
        {
            var prior = x / to_X;
            return XAverageConditional(isPositive, prior);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="XAverageLogarithm(Bernoulli, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static Gaussian XAverageLogarithm([SkipIfUniform] Bernoulli isPositive, [SkipIfUniform, Stochastic] Gaussian x, Gaussian to_X)
        {
            if (isPositive.IsPointMass)
                return XAverageLogarithm(isPositive.Point, x, to_X);
            if (isPositive.IsUniform())
                return Gaussian.Uniform();
            throw new NotSupportedException(NotSupportedMessage2);
            var prior = x / to_X;
            return XAverageConditional(isPositive, prior);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="AverageLogFactor(Bernoulli, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static double AverageLogFactor(Bernoulli isPositive, [SkipIfUniform] Gaussian X, Gaussian to_X)
        {
            if (isPositive.IsPointMass)
                return AverageLogFactor(isPositive.Point, X, to_X);
            var prior = X / to_X;
            //var tg = new TruncatedGaussian(prior);
            //tg.LowerBound = 0;
            // Remove the incorrect Gaussian entropy contribution and add the correct
            // truncated Gaussian entropy. Log(1)=0 so the factor itself does not contribute. 
            return X.GetAverageLog(X) - X.GetAverageLog(prior) + LogAverageFactor(isPositive, prior);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="AverageLogFactor(bool, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static double AverageLogFactor(bool isPositive, [SkipIfUniform] Gaussian x, Gaussian to_X)
        {
            if (isPositive)
                return AverageLogFactor_helper(x, to_X);
            else
            {
                x.MeanTimesPrecision *= -1.0;
                to_X.MeanTimesPrecision *= -1.0;
                return AverageLogFactor_helper(x, to_X);
            }
        }
#else
        
    /// <summary>
    /// Evidence message for VMP
    /// </summary>
    /// <param name="isPositive">Constant value for 'isPositive'.</param>
    /// <param name="x">Incoming message from 'x'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
    /// <returns>Zero</returns>
    /// <remarks><para>
    /// In Variational Message Passing, the evidence contribution of a deterministic factor is zero.
    /// Adding up these values across all factors and variables gives the log-evidence estimate for VMP.
    /// </para></remarks>
    /// <exception cref="ImproperMessageException"><paramref name="x"/> is not a proper distribution</exception>
        [Skip]
        public static double AverageLogFactor(bool isPositive, [SkipIfUniform] TruncatedGaussian x)
        {
            return 0.0;
        }

        [NotSupported(NotSupportedMessage)]
        public static Gaussian XAverageLogarithm(bool isPositive, [SkipIfUniform] Gaussian x)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }
        [NotSupported(NotSupportedMessage)]
        public static double AverageLogFactor(bool isPositive, [SkipIfUniform] Gaussian x)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>
        /// VMP message to 'x'
        /// </summary>
        /// <param name="isPositive">Incoming message from 'isPositive'.</param>
        /// <param name="x">Incoming message from 'x'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is the factor viewed as a function of 'x' with 'isPositive' integrated out.
        /// The formula is <c>sum_isPositive p(isPositive) factor(isPositive,x)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="x"/> is not a proper distribution</exception>
        [NotSupported(NotSupportedMessage2)]
        public static Gaussian XAverageLogarithm(Bernoulli isPositive, [SkipIfUniform] Gaussian x)
        {
            throw new NotSupportedException(NotSupportedMessage2);
        }

#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="XAverageLogarithm(bool)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static TruncatedGaussian XAverageLogarithm(bool isPositive)
        {
            return XAverageConditional(isPositive);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="XAverageConditional(bool)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static TruncatedGaussian XAverageConditional(bool isPositive)
        {
            if (isPositive)
            {
                return new TruncatedGaussian(0, double.PositiveInfinity, 0, double.PositiveInfinity);
            }
            else
            {
                return new TruncatedGaussian(0, double.PositiveInfinity, double.NegativeInfinity, 0);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="XAverageConditional(Bernoulli)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static TruncatedGaussian XAverageConditional([SkipIfUniform] Bernoulli isPositive)
        {
            if (isPositive.IsUniform())
                return TruncatedGaussian.Uniform();
            if (isPositive.IsPointMass)
                return XAverageConditional(isPositive.Point);
            throw new NotSupportedException("Cannot return a TruncatedGaussian when isPositive is random");
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="IsPositiveAverageLogarithm(Gaussian)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static Bernoulli IsPositiveAverageLogarithm([SkipIfUniform] Gaussian x)
        {
            // same as BP if you use John Winn's rule.
            return IsPositiveAverageConditional(x);
        }

        /// <summary>
        /// Evidence message for VMP
        /// </summary>
        /// <param name="X">Incoming message from 'x'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="to_X">Previous outgoing message to 'X'.</param>
        /// <returns>Zero</returns>
        /// <remarks><para>
        /// In Variational Message Passing, the evidence contribution of a deterministic factor is zero.
        /// Adding up these values across all factors and variables gives the log-evidence estimate for VMP.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="X"/> is not a proper distribution</exception>
        [Quality(QualityBand.Preview)]
        public static double AverageLogFactor_helper([SkipIfUniform] Gaussian X, Gaussian to_X)
        {
            //if (!isPositive) throw new ArgumentException("VariationalMessagePassing requires isPositive=true", "isPositive");
            var prior = X / to_X;
            if (!prior.IsProper())
                throw new ImproperMessageException(prior);
            var tg = new TruncatedGaussian(prior);
            tg.LowerBound = 0;
            // Remove the incorrect Gaussian entropy contribution and add the correct
            // truncated Gaussian entropy. Log(1)=0 so the factor itself does not contribute. 
            return X.GetAverageLog(X) - tg.GetAverageLog(tg);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp"]/message_doc[@name="AverageLogFactor(TruncatedGaussian)"]/*'/>
        [Skip]
        [Quality(QualityBand.Preview)]
        public static double AverageLogFactor(TruncatedGaussian X)
        {
            return 0;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp_Proper"]/doc/*'/>
    [FactorMethod(typeof(Factor), "IsPositive", typeof(double), Default = true)]
    [Quality(QualityBand.Mature)]
    public static class IsPositiveOp_Proper
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="IsPositiveOp_Proper"]/message_doc[@name="XAverageConditional(Bernoulli, Gaussian)"]/*'/>
        public static Gaussian XAverageConditional([SkipIfUniform] Bernoulli isPositive, [SkipIfUniform, Proper] Gaussian x)
        {
            return IsPositiveOp.XAverageConditional_Helper(isPositive, x, forceProper: true);
        }
    }
}
