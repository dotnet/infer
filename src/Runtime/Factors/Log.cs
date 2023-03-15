// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/doc/*'/>
    [FactorMethod(typeof(Math), "Log", typeof(double))]
    [Quality(QualityBand.Preview)]
    public class LogOp_EP
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/message_doc[@name="LogAverageFactor(double, double)"]/*'/>
        public static double LogAverageFactor(double log, double d)
        {
            return (log == Math.Log(d)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/message_doc[@name="LogEvidenceRatio(double, double)"]/*'/>
        public static double LogEvidenceRatio(double log, double d)
        {
            return LogAverageFactor(log, d);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/message_doc[@name="AverageLogFactor(double, double)"]/*'/>
        public static double AverageLogFactor(double log, double d)
        {
            return LogAverageFactor(log, d);
        }

        //-- EP -------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/message_doc[@name="LogAverageFactor(double, Gamma)"]/*'/>
        public static double LogAverageFactor(double log, Gamma d)
        {
            double exp = Math.Exp(log);
            return d.GetLogProb(exp) + log;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/message_doc[@name="LogAverageFactor(Gaussian, double)"]/*'/>
        public static double LogAverageFactor(Gaussian log, double d)
        {
            return log.GetLogProb(Math.Log(d));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/message_doc[@name="LogAverageFactor(Gaussian, Gamma, Gaussian)"]/*'/>
        public static double LogAverageFactor(Gaussian log, Gamma d, [Fresh] Gaussian to_log)
        {
            Gamma g = Gamma.FromShapeAndRate(d.Shape + 1, d.Rate);
            return d.Shape / d.Rate * ExpOp.LogAverageFactor(g, log, to_log);
        }


        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gamma, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(Gaussian log, Gamma d, [Fresh] Gaussian to_log)
        {
            return LogAverageFactor(log, d, to_log) - to_log.GetAverageLog(log);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/message_doc[@name="LogEvidenceRatio(double, Gamma)"]/*'/>
        public static double LogEvidenceRatio(double log, Gamma d)
        {
            return LogAverageFactor(log, d);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/message_doc[@name="DAverageConditional(Gaussian, double)"]/*'/>
        public static Gamma DAverageConditional(Gaussian log, double d)
        {
            // Factor is N(log(d); m, v) = exp(m/v*log(d) - 0.5/v*log(d)^2)
            double logd = Math.Log(d);
            double xdlogp = log.MeanTimesPrecision - log.Precision * logd;
            double xxddlogp = -log.Precision - xdlogp;
            return Gamma.FromDerivatives(d, xdlogp / d, xdlogp, xxddlogp, false);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/message_doc[@name="DAverageConditional(Gaussian, Gamma, Gaussian)"]/*'/>
        public static Gamma DAverageConditional([Proper] Gaussian log, Gamma d, Gaussian to_log)
        {
            // Factor is N(log(d); m, v)
            if (d.IsPointMass)
            {
                return DAverageConditional(log, d.Point);
            }
            var g = Gamma.FromShapeAndRate(d.Shape + 1, d.Rate);
            return ExpOp.ExpAverageConditional(g, log, to_log);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/message_doc[@name="DAverageConditional(double)"]/*'/>
        public static Gamma DAverageConditional(double log)
        {
            return Gamma.PointMass(Math.Exp(log));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/message_doc[@name="LogAverageConditional(double)"]/*'/>
        public static Gaussian LogAverageConditional(double d)
        {
            return Gaussian.PointMass(Math.Log(d));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/message_doc[@name="LogAverageConditional(Gaussian, Gamma, Gaussian)"]/*'/>
        public static Gaussian LogAverageConditional([Proper] Gaussian log, [SkipIfUniform] Gamma d, Gaussian result)
        {
            var g = Gamma.FromShapeAndRate(d.Shape + 1, d.Rate);
            return ExpOp.DAverageConditional(g, log, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_EP"]/message_doc[@name="LogAverageConditionalInit()"]/*'/>
        [Skip]
        public static Gaussian LogAverageConditionalInit()
        {
            return Gaussian.Uniform();
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_VMP"]/doc/*'/>
    [FactorMethod(typeof(Math), "Log", typeof(double))]
    [Quality(QualityBand.Preview)]
    public class LogOp_VMP
    {
        //-- VMP -------------------------------------------------------------------------------------------

        /// <summary>
        /// Determines the amount of damping to use on the VMP updates for D. 
        /// </summary>
        public static double damping = 0.3;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_VMP"]/message_doc[@name="LogAverageLogarithm(Gaussian, double, Gaussian)"]/*'/>
        public static Gaussian LogAverageLogarithm(Gaussian log, double d, Gaussian result)
        {
            result.Point = Math.Log(d);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_VMP"]/message_doc[@name="DAverageLogarithm(double, Gamma, Gamma)"]/*'/>
        public static Gamma DAverageLogarithm(double log, Gamma d, Gamma result)
        {
            result.Point = Math.Exp(log);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_VMP"]/message_doc[@name="LogAverageLogarithm(Gaussian, Gamma, Gaussian)"]/*'/>
        public static Gaussian LogAverageLogarithm(Gaussian log, [SkipIfUniform] Gamma d, Gaussian result)
        {
            if (d.IsPointMass)
                return LogAverageLogarithm(log, d.Point, result);
            result.SetMeanAndVariance(d.GetMeanLog(), MMath.Trigamma(d.Shape));
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_VMP"]/message_doc[@name="DAverageLogarithm(Gaussian, Gamma, Gamma)"]/*'/>
        public static Gamma DAverageLogarithm([SkipIfUniform] Gaussian log, [SkipIfUniform] Gamma d, Gamma to_D)
        {
            if (log.IsPointMass)
                return DAverageLogarithm(log.Point, d, to_D);
            Vector grad = Vector.Zero(2);
            double meanLog = d.GetMeanLog();
            double m, v;
            log.GetMeanAndVariance(out m, out v);
            grad[0] = -MMath.Tetragamma(d.Shape) / (2 * v) - MMath.Trigamma(d.Shape) / v * (meanLog - m);
            grad[1] = (meanLog - m) / (v * d.Rate);
            Gamma approximateFactor = GammaFromShapeAndRateOp.NonconjugateProjection(d, grad);
            if (damping == 0.0)
                return approximateFactor;
            else
                return (approximateFactor ^ (1 - damping)) * (to_D ^ damping);
        }

#if false
        /// <summary>
        /// Evidence message for VMP
        /// </summary>
        /// <param name="log">Incoming message from 'log'.</param>
        /// <param name="d">Incoming message from 'd'.</param>
        /// <returns>Zero</returns>
        /// <remarks><para>
        /// In Variational Message Passing, the evidence contribution of a deterministic factor is zero.
        /// Adding up these values across all factors and variables gives the log-evidence estimate for VMP.
        /// </para></remarks>
        public static double AverageLogFactor(Gaussian log, Gamma d)
        {
            double m, v;
            log.GetMeanAndVariance(out m, out v);
            double Elogd=d.GetMeanLog();
            double Elogd2;
            if (!d.IsPointMass)
                Elogd2 = MMath.Trigamma(d.Shape) + Elogd * Elogd;
            else
                Elogd2 = Math.Log(d.Point) * Math.Log(d.Point);
            return -Elogd2/(2*v)+m*Elogd/v-m*m/(2*v)-MMath.LnSqrt2PI-.5*Math.Log(v);
        }
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_VMP"]/message_doc[@name="AverageLogFactor()"]/*'/>
        public static double AverageLogFactor()
        {
            return 0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_VMP"]/message_doc[@name="LogAverageLogarithm(double)"]/*'/>
        public static Gamma LogAverageLogarithm(double d)
        {
            return Gamma.PointMass(Math.Log(d));
        }

        private const string NotSupportedMessage = "VMP cannot support deterministic factors such as Log with fixed outputs";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="LogOp_VMP"]/message_doc[@name="DAverageLogarithm(double)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gaussian DAverageLogarithm(double log)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }
    }
}
