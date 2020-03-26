// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/doc/*'/>
    [FactorMethod(new string[] { "max", "a", "b" }, typeof(Math), "Max", typeof(double), typeof(double))]
    [Quality(QualityBand.Stable)]
    public static class MaxGaussianOp
    {
        /// <summary>
        /// Static flag to force a proper distribution
        /// </summary>
        public static bool ForceProper = true;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="LogAverageFactor(double, double, double)"]/*'/>
        public static double LogAverageFactor(double max, double a, double b)
        {
            return (max == Math.Max(a, b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="LogEvidenceRatio(double, double, double)"]/*'/>
        public static double LogEvidenceRatio(double max, double a, double b)
        {
            return LogAverageFactor(max, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="AverageLogFactor(double, double, double)"]/*'/>
        public static double AverageLogFactor(double max, double a, double b)
        {
            return LogAverageFactor(max, a, b);
        }

        // logz is the log-expectation of the factor value.
        // logw1 is the log-probability that max==a
        // exp(logw1) = N(mx;m1,vx+v1) phi((mx1 - m2)/sqrt(vx1+v2))
        // alpha1 is the correction to the mean of a due to b given that max==a
        // alpha1 = N(mx1;m2,vx1+v2)/phi
        // vx1 is the variance of a given that max==a (ignoring b)
        // mx1 is the mean of a given that max==a (ignoring b)
        internal static void ComputeStats(Gaussian max, Gaussian a, Gaussian b, out double logz,
                                          out double logw1, out double alpha1, out double vx1, out double mx1,
                                          out double logw2, out double alpha2, out double vx2, out double mx2)
        {
            double logPhi1, logPhi2;
            if (max.IsPointMass)
            {
                vx1 = 0.0;
                mx1 = max.Point;
                vx2 = 0.0;
                mx2 = max.Point;
                if (b.IsPointMass)
                {
                    if (b.Point > max.Point)
                        throw new AllZeroException();
                    else if (b.Point == max.Point)
                    {
                        // the factor reduces to the constraint (max.Point >= a)
                        if (a.IsPointMass)
                        {
                            if (a.Point > max.Point)
                                throw new AllZeroException();
                            if (a.Point == max.Point)
                            {
                                logw1 = -MMath.Ln2;
                                logw2 = -MMath.Ln2;
                                logz = 0;
                                alpha1 = 0;
                                alpha2 = 0;
                                return;
                            }
                            else
                            {
                                logw2 = 0;
                            }
                        }
                        else if (a.Precision == 0)
                        {
                            logw1 = double.NegativeInfinity;
                            logw2 = -MMath.Ln2;
                            logz = logw2;
                            alpha1 = 0;
                            alpha2 = 0;
                            return;
                        }
                        else
                        {
                            logw2 = MMath.NormalCdfLn((max.Point * a.Precision - a.MeanTimesPrecision) / Math.Sqrt(a.Precision));
                        }
                        logw1 = double.NegativeInfinity;
                        logz = logw2;
                        alpha1 = 0;
                        alpha2 = Math.Exp(a.GetLogProb(max.Point) - logw2);
                        return;
                    }
                    else // b.Point < max.Point
                    {
                        // the factor reduces to the constraint (a == max.Point)
                        logw1 = a.GetLogProb(max.Point);
                        logw2 = double.NegativeInfinity;
                        logz = logw1;
                        alpha1 = 0;
                        alpha2 = 0;
                        return;
                    }
                }
                else if (a.IsPointMass) // !b.IsPointMass
                {
                    if (a.Point > max.Point)
                        throw new AllZeroException();
                    else if (a.Point == max.Point)
                    {
                        // the factor reduces to the constraint (max.Point > b)
                        if (b.Precision == 0)
                        {
                            logw2 = double.NegativeInfinity;
                            logw1 = -MMath.Ln2;
                            logz = logw1;
                            alpha1 = 0;
                            alpha2 = 0;
                            return;
                        }
                        else
                        {
                            logw1 = MMath.NormalCdfLn((max.Point * b.Precision - b.MeanTimesPrecision) / Math.Sqrt(b.Precision));
                            logw2 = double.NegativeInfinity;
                            logz = logw1;
                            alpha2 = 0;
                            alpha1 = Math.Exp(b.GetLogProb(max.Point) - logw1);
                            return;
                        }
                    }
                    else // a.Point < max.Point
                    {
                        // the factor reduces to the constraint (b == max.Point)
                        logw2 = b.GetLogProb(max.Point);
                        logw1 = double.NegativeInfinity;
                        logz = logw2;
                        alpha1 = 0;
                        alpha2 = 0;
                        return;
                    }
                }
                else // !a.IsPointMass && !b.IsPointMass
                {
                    double z1 = (mx1 * b.Precision - b.MeanTimesPrecision) / Math.Sqrt(b.Precision);
                    logPhi1 = MMath.NormalCdfLn(z1);
                    alpha1 = Math.Sqrt(b.Precision) / MMath.NormalCdfRatio(z1);
                    double z2 = (mx2 * a.Precision - a.MeanTimesPrecision) / Math.Sqrt(a.Precision);
                    logPhi2 = MMath.NormalCdfLn(z2);
                    alpha2 = Math.Sqrt(a.Precision) / MMath.NormalCdfRatio(z2);
                }
                // fall through
            }
            else // !max.IsPointMass
            {
                if (a.IsPointMass)
                {
                    vx1 = 0.0;
                    mx1 = a.Point;
                    if (b.IsPointMass)
                    {
                        vx2 = 0.0;
                        mx2 = b.Point;
                        if (a.Point > b.Point)
                        {
                            logw1 = max.GetLogAverageOf(a);
                            logw2 = double.NegativeInfinity;
                            logz = logw1;
                        }
                        else if (a.Point < b.Point)
                        {
                            logw2 = max.GetLogAverageOf(b);
                            logw1 = double.NegativeInfinity;
                            logz = logw2;
                        }
                        else // a.Point == b.Point
                        {
                            logw1 = -MMath.Ln2;
                            logw2 = -MMath.Ln2;
                            logz = 0;
                            alpha1 = double.PositiveInfinity;
                            alpha2 = double.PositiveInfinity;
                            return;
                        }
                        alpha1 = 0;
                        alpha2 = 0;
                        return;
                    }
                    double z1 = (a.Point * b.Precision - b.MeanTimesPrecision) / Math.Sqrt(b.Precision);
                    logPhi1 = MMath.NormalCdfLn(z1);
                    alpha1 = Math.Sqrt(b.Precision) / MMath.NormalCdfRatio(z1);
                }
                else // !a.IsPointMass
                {
                    vx1 = 1.0 / (max.Precision + a.Precision);
                    mx1 = vx1 * (max.MeanTimesPrecision + a.MeanTimesPrecision);
                    double m2, v2;
                    b.GetMeanAndVariance(out m2, out v2);
                    double s = Math.Sqrt(vx1 + v2);
                    // This approach is more accurate for large max.Precision
                    double z1 = (max.MeanTimesPrecision + a.MeanTimesPrecision - m2 * (max.Precision + a.Precision)) * vx1 / s;
                    logPhi1 = MMath.NormalCdfLn(z1);
                    alpha1 = 1 / (s * MMath.NormalCdfRatio(z1));
                }
                if (b.IsPointMass)
                {
                    vx2 = 0.0;
                    mx2 = b.Point;
                    double z2 = (b.Point * a.Precision - a.MeanTimesPrecision) / Math.Sqrt(a.Precision);
                    logPhi2 = MMath.NormalCdfLn(z2);
                    alpha2 = Math.Sqrt(a.Precision) / MMath.NormalCdfRatio(z2);
                }
                else // !b.IsPointMass
                {
                    vx2 = 1.0 / (max.Precision + b.Precision);
                    mx2 = vx2 * (max.MeanTimesPrecision + b.MeanTimesPrecision);
                    double m1, v1;
                    a.GetMeanAndVariance(out m1, out v1);
                    double s = Math.Sqrt(vx2 + v1);
                    // This approach is more accurate for large max.Precision
                    double z2 = (max.MeanTimesPrecision + b.MeanTimesPrecision - m1 * (max.Precision + b.Precision)) * vx2 / s;
                    logPhi2 = MMath.NormalCdfLn(z2);
                    //double logPhi2b = MMath.NormalCdfLn((mx2 - m1) / Math.Sqrt(vx2 + v1));
                    alpha2 = 1 / (s * MMath.NormalCdfRatio(z2));
                }
            }
            // !(max.IsPointMass && a.IsPointMass)
            // !(max.IsPointMass && b.IsPointMass)
            // !(a.IsPointMass && b.IsPointMass)
            logw1 = max.GetLogAverageOf(a);
            logw1 += logPhi1;

            logw2 = max.GetLogAverageOf(b);
            logw2 += logPhi2;

            logz = MMath.LogSumExp(logw1, logw2);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gaussian max, [Proper] Gaussian a, [Proper] Gaussian b)
        {
            double logw1, a1, vx1, mx1;
            double logw2, a2, vx2, mx2;
            double logz;
            ComputeStats(max, a, b, out logz, out logw1, out a1, out vx1, out mx1,
                         out logw2, out a2, out vx2, out mx2);
            return logz;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="LogAverageFactor(Gaussian, double, Gaussian)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gaussian max, double a, [Proper] Gaussian b)
        {
            return LogAverageFactor(max, Gaussian.PointMass(a), b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, double)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gaussian max, [Proper] Gaussian a, double b)
        {
            return LogAverageFactor(max, a, Gaussian.PointMass(b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="LogAverageFactor(double, Gaussian, Gaussian)"]/*'/>
        public static double LogAverageFactor(double max, [Proper] Gaussian a, [Proper] Gaussian b)
        {
            return LogAverageFactor(Gaussian.PointMass(max), a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="LogAverageFactor(double, double, Gaussian)"]/*'/>
        public static double LogAverageFactor(double max, double a, [Proper] Gaussian b)
        {
            return LogAverageFactor(Gaussian.PointMass(max), Gaussian.PointMass(a), b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="LogAverageFactor(double, Gaussian, double)"]/*'/>
        public static double LogAverageFactor(double max, [Proper] Gaussian a, double b)
        {
            return LogAverageFactor(Gaussian.PointMass(max), a, Gaussian.PointMass(b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian max, [Proper] Gaussian a, [Proper] Gaussian b)
        {
            Gaussian to_max = MaxAverageConditional(max, a, b);
            return LogAverageFactor(max, a, b)
                   - to_max.GetLogAverageOf(max);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, double, Gaussian)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian max, double a, [Proper] Gaussian b)
        {
            Gaussian to_max = MaxAverageConditional(max, a, b);
            return LogAverageFactor(max, a, b)
                   - to_max.GetLogAverageOf(max);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, double)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian max, [Proper] Gaussian a, double b)
        {
            Gaussian to_max = MaxAverageConditional(max, a, b);
            return LogAverageFactor(max, a, b)
                   - to_max.GetLogAverageOf(max);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="LogEvidenceRatio(double, Gaussian, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(double max, [Proper] Gaussian a, [Proper] Gaussian b)
        {
            return LogAverageFactor(max, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="LogEvidenceRatio(double, double, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(double max, double a, [Proper] Gaussian b)
        {
            return LogAverageFactor(max, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="LogEvidenceRatio(double, Gaussian, double)"]/*'/>
        public static double LogEvidenceRatio(double max, [Proper] Gaussian a, double b)
        {
            return LogAverageFactor(max, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="MaxAverageConditionalInit()"]/*'/>
        [Skip]
        public static Gaussian MaxAverageConditionalInit()
        {
            return Gaussian.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="MaxAverageConditional(Gaussian, double, Gaussian)"]/*'/>
        public static Gaussian MaxAverageConditional(Gaussian max, double a, [Proper] Gaussian b)
        {
            return MaxAverageConditional(max, Gaussian.PointMass(a), b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="MaxAverageConditional(Gaussian, Gaussian, double)"]/*'/>
        public static Gaussian MaxAverageConditional(Gaussian max, [Proper] Gaussian a, double b)
        {
            return MaxAverageConditional(max, a, Gaussian.PointMass(b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="MaxAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian MaxAverageConditional(Gaussian max, [Proper] Gaussian a, [Proper] Gaussian b)
        {
            // the following code works correctly even if max is uniform or improper.
            if (!a.IsProper())
                throw new ImproperMessageException(a);
            if (!b.IsProper())
                throw new ImproperMessageException(b);
            double m1, v1, m2, v2;
            a.GetMeanAndVariance(out m1, out v1);
            b.GetMeanAndVariance(out m2, out v2);

            double logw1, alpha1, vx1, mx1;
            double logw2, alpha2, vx2, mx2;
            double logz;
            ComputeStats(max, a, b, out logz, out logw1, out alpha1, out vx1, out mx1,
                         out logw2, out alpha2, out vx2, out mx2);
            double w1 = Math.Exp(logw1 - logz);
            double w2 = Math.Exp(logw2 - logz);
            if (max.IsPointMass)
            {
                double z1 = a.MeanTimesPrecision - max.Point * a.Precision;
                double z2 = b.MeanTimesPrecision - max.Point * b.Precision;
                if (a.IsPointMass)
                {
                    if (max.Point == a.Point)
                    {
                        return Gaussian.PointMass(a.Point);
                    }
                    else
                    {
                        z1 = 0;
                    }
                }
                if (b.IsPointMass)
                {
                    if (max.Point == b.Point)
                    {
                        return Gaussian.PointMass(b.Point);
                    }
                    else
                    {
                        z2 = 0;
                    }
                }
                double alpha = (z1 + alpha1) * w1 + (z2 + alpha2) * w2;
                double beta = -alpha * alpha;
                if (w1 > 0) beta += (z1 * z1 - a.Precision + (2 * z1 + z2) * alpha1) * w1;
                if (w2 > 0) beta += (z2 * z2 - b.Precision + (2 * z2 + z1) * alpha2) * w2;
                //Console.WriteLine($"z1={z1} w1={w1} alpha1={alpha1} z2={z2} w2={w2} alpha2={alpha2} alpha={alpha} beta={beta:r}");
                return GaussianOp.GaussianFromAlphaBeta(max, alpha, -beta, ForceProper);
            }
            bool useMessage = max.Precision > 10;
            if (useMessage)
            {
                // We want to avoid computing 1/max.Precision or max.MeanTimesPrecision/max.Precision, since these lead to roundoff errors.
                double z1 = (m1 * max.Precision - max.MeanTimesPrecision) / (v1 * max.Precision + 1);
                double z2 = (m2 * max.Precision - max.MeanTimesPrecision) / (v2 * max.Precision + 1);
                double alpha = 0;
                if (w1 > 0) alpha += (z1 + vx1 * max.Precision * alpha1) * w1;
                if (w2 > 0) alpha += (z2 + vx2 * max.Precision * alpha2) * w2;
                double beta = -alpha * alpha;
                if (w1 > 0)
                {
                    double diff;
                    // compute diff to avoid roundoff errors
                    if (a.IsPointMass) diff = m2 - mx1;
                    else diff = (m2 * (max.Precision + a.Precision) - (max.MeanTimesPrecision + a.MeanTimesPrecision)) * vx1;
                    if (!b.IsPointMass) diff *= vx1 / (v2 + vx1);
                    beta += (z1 * z1 - max.Precision / (v1 * max.Precision + 1) + (2 * z1 + diff * max.Precision) * vx1 * max.Precision * alpha1) * w1;
                }
                if (w2 > 0)
                {
                    double diff;
                    if (b.IsPointMass) diff = m1 - mx2;
                    else diff = (m1 * (max.Precision + b.Precision) - (max.MeanTimesPrecision + b.MeanTimesPrecision)) * vx2;
                    if (!a.IsPointMass) diff *= vx2 / (v1 + vx2);
                    beta += (z2 * z2 - max.Precision / (v2 * max.Precision + 1) + (2 * z2 + diff * max.Precision) * vx2 * max.Precision * alpha2) * w2;
                }
                bool check = false;
                if (check)
                {
                    double mx = max.GetMean();
                    double delta = mx * 1e-4;
                    double logzd, logw1d, alpha1d, vx1d, mx1d, logw2d, alpha2d, vx2d, mx2d;
                    ComputeStats(Gaussian.FromMeanAndPrecision(mx + delta, max.Precision), a, b, out logzd, out logw1d, out alpha1d, out vx1d, out mx1d, out logw2d, out alpha2d, out vx2d, out mx2d);
                    double logzd2;
                    ComputeStats(Gaussian.FromMeanAndPrecision(mx - delta, max.Precision), a, b, out logzd2, out logw1d, out alpha1d, out vx1d, out mx1d, out logw2d, out alpha2d, out vx2d, out mx2d);
                    double alphaCheck = (logzd - logzd2) / (2 * delta);
                    double alphaError = Math.Abs(alpha - alphaCheck);
                    double betaCheck = (logzd + logzd2 - 2 * logz) / (delta * delta);
                    double betaError = Math.Abs(beta - betaCheck);
                    Console.WriteLine($"alpha={alpha} check={alphaCheck} error={alphaError} beta={beta} check={betaCheck} error={betaError}");
                }
                //Console.WriteLine($"z1={z1} w1={w1} vx1={vx1} alpha1={alpha1} z2={z2} w2={w2} vx2={vx2} alpha2={alpha2} alpha={alpha} beta={beta:r} {max.Precision:r}");
                return GaussianOp.GaussianFromAlphaBeta(max, alpha, -beta, ForceProper);
            }
            // the posterior is a mixture model with weights exp(logw1-logz), exp(logw2-logz) and distributions
            // N(x; mx1, vx1) phi((x - m2)/sqrt(v2)) / phi((mx1 - m2)/sqrt(vx1 + v2))
            // the moments of the posterior are computed via the moments of these two components.
            if (vx1 == 0) alpha1 = 0;
            if (vx2 == 0) alpha2 = 0;
            double mc1 = mx1 + alpha1 * vx1;
            double mc2 = mx2 + alpha2 * vx2;
            double m, v;
            if (w1 == 0)
            {
                // avoid dealing with infinities.
                m = mx2;
                v = vx2;
            }
            else if (w2 == 0)
            {
                // avoid dealing with infinities.
                m = mx1;
                v = vx1;
            }
            else
            {
                m = w1 * mc1 + w2 * mc2;
                double beta1;
                if (alpha1 == 0)
                    beta1 = 0;
                else
                {
                    double r1 = (mx1 - m2) / (vx1 + v2);
                    beta1 = alpha1 * (alpha1 + r1);
                }
                double vc1 = vx1 * (1 - vx1 * beta1);
                double beta2;
                if (alpha2 == 0)
                    beta2 = 0;
                else
                {
                    double r2 = (mx2 - m1) / (vx2 + v1);
                    beta2 = alpha2 * (alpha2 + r2);
                }
                double vc2 = vx2 * (1 - vx2 * beta2);
                double diff = mc1 - mc2;
                v = w1 * vc1 + w2 * vc2 + w1 * w2 * diff * diff;
            }
            Gaussian result = new Gaussian(m, v);
            result.SetToRatio(result, max, ForceProper);
            if (Double.IsNaN(result.Precision) || Double.IsNaN(result.MeanTimesPrecision))
                throw new InferRuntimeException($"result is NaN.  max={max}, a={a}, b={b}");
            return result;
        }

#if false
        public static void ComputeStats(Gaussian max, Gaussian a, Gaussian b, out double logz,
            out double logw1, out double logPhi1, out double logu1, out double vx1, out double mx1,
            out double logw2, out double logPhi2, out double logu2, out double vx2, out double mx2)
        {
            double mx, vx, ma, va, mb, vb;
            max.GetMeanAndVariance(out mx, out vx);
            a.GetMeanAndVariance(out ma, out va);
            b.GetMeanAndVariance(out mb, out vb);
            if (false) {
                vx1 = 1.0 / (1.0 / vx + 1.0 / va);
                mx1 = vx1 * (mx / vx + ma / va);
                vx2 = 1.0 / (1.0 / vx + 1.0 / vb);
                mx2 = vx2 * (mx / vx + mb / vb);
            } else {
                if(max.IsPointMass || a.IsPointMass || b.IsPointMass) throw new NotImplementedException();
                vx1 = 1.0/(max.Precision + a.Precision);
                mx1 = vx1*(max.MeanTimesPrecision + a.MeanTimesPrecision);
                vx2 = 1.0/(max.Precision + b.Precision);
                mx2 = vx2*(max.MeanTimesPrecision + b.MeanTimesPrecision);
            }
            logw1 = max.GetLogAverageOf(a);
            logPhi1 = MMath.NormalCdfLn((mx1 - mb) / Math.Sqrt(vx1 + vb));
            logu1 = Gaussian.GetLogProb(mx1, mb, vx1 + vb);

            logw2 = max.GetLogAverageOf(b);
            logPhi2 = MMath.NormalCdfLn((mx2 - ma) / Math.Sqrt(vx2 + va));
            logu2 = Gaussian.GetLogProb(mx2, ma, vx2 + va);

            logz = MMath.LogSumExp(logw1 + logPhi1, logw2 + logPhi2);
        }
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian max, [Proper] Gaussian a, [Proper] Gaussian b)
        {
            if (max.IsUniform())
                return Gaussian.Uniform();
            if (!b.IsProper())
                throw new ImproperMessageException(b);

            double logw1, alpha1, vx1, mx1;
            double logw2, alpha2, vx2, mx2;
            double logz;
            ComputeStats(max, a, b, out logz, out logw1, out alpha1, out vx1, out mx1,
                         out logw2, out alpha2, out vx2, out mx2);
            double w1 = Math.Exp(logw1 - logz);
            double w2 = Math.Exp(logw2 - logz);
            bool checkDerivatives = false;
            if (a.IsPointMass)
            {
                // f(a) = int p(x = max(a,b)) p(b) db
                // f(a) = p(x = a) p(a >= b) + p(x = b) p(a < b | x = b)
                // f(a) = N(a; mx, vx) phi((a - m2)/sqrt(v2)) + N(m2; mx, vx + v2) phi((mx2 - a)/sqrt(vx2))
                // f'(a) = (mx-a)/vx N(a; mx, vx) phi((a - m2)/sqrt(v2)) + N(a; mx, vx) N(a; m2, v2) - N(m2; mx, vx + v2) N(a; mx2, vx2)
                //       = (mx-a)/vx N(a; mx, vx) phi((a - m2)/sqrt(v2))
                // f''(a) = -1/vx N(a; mx, vx) phi((a - m2)/sqrt(v2)) + (mx-a)^2/vx^2 N(a; mx, vx) phi((a - m2)/sqrt(v2)) + (mx-a)/vx N(a; mx, vx) N(a; m2, v2)
                // ddlogf = f''(a)/f(a) - (f'(a)/f(a))^2 = (f''(a) f(a) - f'(a)^2)/f(a)^2
                double aPoint = a.Point;
                if (max.IsPointMass)
                {
                    return max;
                }
                double z = max.MeanTimesPrecision - aPoint * max.Precision;
                double alpha = z * w1;
                double beta = (z * alpha1 - max.Precision + z * z) * w1 - alpha * alpha;
                if (b.IsPointMass && b.Point != aPoint)
                    beta -= max.Precision * w2 * alpha2 * (b.Point - aPoint);
                if (checkDerivatives)
                {
                    double m1 = a.GetMean();
                    double delta = m1 * 1e-6;
                    double logzd, logw1d, alpha1d, vx1d, mx1d, logw2d, alpha2d, vx2d, mx2d;
                    ComputeStats(max, Gaussian.FromMeanAndPrecision(m1 + delta, a.Precision), b, out logzd, out logw1d, out alpha1d, out vx1d, out mx1d, out logw2d, out alpha2d, out vx2d, out mx2d);
                    double logzd2;
                    ComputeStats(max, Gaussian.FromMeanAndPrecision(m1 - delta, a.Precision), b, out logzd2, out logw1d, out alpha1d, out vx1d, out mx1d, out logw2d, out alpha2d, out vx2d, out mx2d);
                    double alphaCheck = (logzd - logzd2) / (2 * delta);
                    double alphaError = Math.Abs(alpha - alphaCheck);
                    double betaCheck = (logzd + logzd2 - 2 * logz) / (delta * delta);
                    double betaError = Math.Abs(beta - betaCheck);
                    Console.WriteLine($"alpha={alpha} check={alphaCheck} error={alphaError} beta={beta} check={betaCheck} error={betaError}");
                }
                return Gaussian.FromDerivatives(aPoint, alpha, beta, ForceProper);
            }
            bool useMessage = (w1>0) || (logw2 > -100);
            if (useMessage)
            {
                // vx1 = 1/(1/vx + 1/v1) = vx*v1/(vx + v1)
                double z, alpha, beta;
                if (max.IsPointMass)
                {
                    if (w2 == 0)
                    {
                        return max;
                    }
                    z = max.Point * a.Precision - a.MeanTimesPrecision;
                    alpha = z * w1 - w2 * alpha2;
                    beta = (z * z - a.Precision) * w1 - z * alpha2 * w2 - alpha * alpha;
                }
                else
                {
                    //z = vx1 * (max.MeanTimesPrecision * a.Precision - a.MeanTimesPrecision * max.Precision);
                    z = (max.MeanTimesPrecision - a.GetMean() * max.Precision)/(max.Precision/a.Precision + 1);
                    alpha = z * w1 - vx1 * max.Precision * w2 * alpha2;
                    if (w2 == 0)
                    {
                        //beta = (z * alpha1 - max.Precision) / (max.Precision / a.Precision + 1);
                        //double resultPrecision = a.Precision * beta / (-a.Precision - beta);
                        //resultPrecision = beta / (-1 - beta/a.Precision);
                        //resultPrecision = 1 / (-1/beta - 1 / a.Precision);
                        //resultPrecision = a.Precision / (-a.Precision / beta - 1);
                        //resultPrecision = a.Precision / (-(a.Precision + max.Precision) / (z * alpha1 - max.Precision) - 1);
                        //resultPrecision = -a.Precision / ((a.Precision + z*alpha1) / (z * alpha1 - max.Precision));
                        double zalpha1 = z * alpha1;
                        double denom = (1 + zalpha1 / a.Precision);
                        double resultPrecision = (max.Precision - zalpha1) / denom;
                        //double weight = (max.Precision - z * alpha1) / (a.Precision + z * alpha1);
                        //double weightPlus1 = (max.Precision + a.Precision) / (a.Precision + z * alpha1);
                        //double resultMeanTimesPrecision = weight * (a.MeanTimesPrecision + alpha) + alpha;
                        //resultMeanTimesPrecision = weight * a.MeanTimesPrecision + weightPlus1 * alpha;
                        //double resultMeanTimesPrecision = (a.Precision * max.MeanTimesPrecision - z * alpha1 * a.MeanTimesPrecision) / (a.Precision + z * alpha1);
                        double resultMeanTimesPrecision = (max.MeanTimesPrecision - zalpha1 * a.GetMean()) / denom;
                        return Gaussian.FromNatural(resultMeanTimesPrecision, resultPrecision);
                    }
                    else
                    {
                        beta = ((z * alpha1 - max.Precision) * vx1 * a.Precision + z * z) * w1
                            - max.Precision * vx1 * w2 * alpha2 * (mx2 * a.Precision - a.MeanTimesPrecision) / (vx2 * a.Precision + 1) - alpha * alpha;
                    }
                }
                //Console.WriteLine($"z={z} w1={w1:r} w2={w2:r} logw2={logw2} alpha1={alpha1} alpha2={alpha2} alpha={alpha:r} beta={beta:r}");
                if (checkDerivatives)
                {
                    double m1 = a.GetMean();
                    double delta = m1 * 1e-6;
                    double logzd, logw1d, alpha1d, vx1d, mx1d, logw2d, alpha2d, vx2d, mx2d;
                    ComputeStats(max, Gaussian.FromMeanAndPrecision(m1 + delta, a.Precision), b, out logzd, out logw1d, out alpha1d, out vx1d, out mx1d, out logw2d, out alpha2d, out vx2d, out mx2d);
                    double logzd2;
                    ComputeStats(max, Gaussian.FromMeanAndPrecision(m1 - delta, a.Precision), b, out logzd2, out logw1d, out alpha1d, out vx1d, out mx1d, out logw2d, out alpha2d, out vx2d, out mx2d);
                    double alphaCheck = (logzd - logzd2) / (2 * delta);
                    double alphaError = Math.Abs(alpha - alphaCheck);
                    double betaCheck = (logzd + logzd2 - 2 * logz) / (delta * delta);
                    double betaError = Math.Abs(beta - betaCheck);
                    Console.WriteLine($"alpha={alpha} check={alphaCheck} error={alphaError} beta={beta} check={betaCheck} error={betaError}");
                }
                return GaussianOp.GaussianFromAlphaBeta(a, alpha, -beta, ForceProper);
            }
            else
            {
                double m1, v1, m2, v2;
                a.GetMeanAndVariance(out m1, out v1);
                b.GetMeanAndVariance(out m2, out v2);
                // the posterior is a mixture model with weights exp(logw1-logz), exp(logw2-logz) and distributions
                // N(a; mx1, vx1) phi((a - m2)/sqrt(v2)) / phi((mx1 - m2)/sqrt(vx1 + v2))
                // N(a; m1, v1) phi((mx2 - a)/sqrt(vx2)) / phi((mx2 - m1)/sqrt(vx2 + v1))
                // the moments of the posterior are computed via the moments of these two components.
                if (vx1 == 0) alpha1 = 0;
                if (vx2 == 0) alpha2 = 0;
                double mc1 = mx1;
                if (alpha1 != 0) // avoid 0*infinity
                    mc1 += alpha1 * vx1;
                alpha2 = -alpha2;
                double mc2 = m1;
                if (alpha2 != 0) // avoid 0*infinity
                    mc2 += alpha2 * v1;
                // z2 = (mx2 - m1) / Math.Sqrt(vx2 + v1)
                // logw2 = MMath.NormalCdfLn(z2);
                // alpha2 = -Math.Exp(Gaussian.GetLogProb(mx2, m1, vx2 + v1) - logw2);
                //        = -1/sqrt(vx2+v1)/NormalCdfRatio(z2)
                // m1 + alpha2*v1 = sqrt(vx2+v1)*(m1/sqrt(vx2+v1) - v1/(vx2+v1)/NormalCdfRatio(z2))
                //                = sqrt(vx2+v1)*(mx2/sqrt(vx2+v1) - z2 - v1/(vx2+v1)/NormalCdfRatio(z2))
                //                = sqrt(vx2+v1)*(mx2/sqrt(vx2+v1) - dY/Y)  if vx2=0
                double z2 = 0, Y = 0, dY = 0;
                const double z2small = 0;
                if (vx2 == 0)
                {
                    z2 = (mx2 - m1) / Math.Sqrt(vx2 + v1);
                    if (z2 < z2small)
                    {
                        Y = MMath.NormalCdfRatio(z2);
                        dY = MMath.NormalCdfMomentRatio(1, z2);
                        mc2 = mx2 - Math.Sqrt(v1) * dY / Y;
                    }
                }
                double m = w1 * mc1 + w2 * mc2;
                double beta1;
                if (alpha1 == 0)
                    beta1 = 0;  // avoid 0*infinity
                else
                {
                    double r1 = (mx1 - m2) / (vx1 + v2);
                    beta1 = alpha1 * (alpha1 + r1);
                }
                double beta2;
                if (alpha2 == 0)
                    beta2 = 0;  // avoid 0*infinity
                else
                {
                    double r2 = (mx2 - m1) / (vx2 + v1);
                    beta2 = alpha2 * (alpha2 - r2);
                }
                double vc1 = vx1 * (1 - vx1 * beta1);
                double vc2;
                if (vx2 == 0 && z2 < z2small)
                {
                    // beta2 = alpha2 * (alpha2 - z2/sqrt(v1))
                    // vc2 = v1 - v1^2 * alpha2 * (alpha2 - z2/sqrt(v1))
                    //     = v1 - v1*dY/Y^2
                    //     =approx v1/z2^2
                    // posterior E[x^2] = v - v*dY/Y^2 + v*dY^2/Y^2 = v - v*dY/Y^2*(1 - dY) = v + v*z*dY/Y = v*d2Y/Y
                    //vc2 = v1 * (1 - dY / (Y * Y));
                    double d2Y = 2 * MMath.NormalCdfMomentRatio(2, z2);
                    double dYiY = dY / Y;
                    vc2 = v1 * (d2Y / Y - dYiY * dYiY);
                }
                else if (beta2 == 0)
                {
                    vc2 = v1;
                }
                else
                {
                    vc2 = v1 * (1 - v1 * beta2);
                }
                double diff = mc1 - mc2;
                double v = w1 * vc1 + w2 * vc2 + w1 * w2 * diff * diff;
                Gaussian result = new Gaussian(m, v);
                //Console.WriteLine($"z2={z2} m={m} v={v} vc2={vc2} diff={diff}");
                result.SetToRatio(result, a, ForceProper);
                if (Double.IsNaN(result.Precision) || Double.IsNaN(result.MeanTimesPrecision))
                    throw new InferRuntimeException($"result is NaN.  max={max}, a={a}, b={b}");
                return result;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="AAverageConditional(double, Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageConditional(double max, [Proper] Gaussian a, [Proper] Gaussian b)
        {
            return AAverageConditional(Gaussian.PointMass(max), a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, double)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian max, [Proper] Gaussian a, double b)
        {
            return AAverageConditional(max, a, Gaussian.PointMass(b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="AAverageConditional(double, Gaussian, double)"]/*'/>
        public static Gaussian AAverageConditional(double max, [Proper] Gaussian a, double b)
        {
            return AAverageConditional(Gaussian.PointMass(max), a, Gaussian.PointMass(b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="BAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian max, [Proper] Gaussian a, [Proper] Gaussian b)
        {
            return AAverageConditional(max, b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="BAverageConditional(double, Gaussian, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional(double max, [Proper] Gaussian a, [Proper] Gaussian b)
        {
            return AAverageConditional(max, b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="BAverageConditional(Gaussian, double, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian max, double a, [Proper] Gaussian b)
        {
            return AAverageConditional(max, b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxGaussianOp"]/message_doc[@name="BAverageConditional(double, double, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional(double max, double a, [Proper] Gaussian b)
        {
            return AAverageConditional(max, b, a);
        }
    }
}
