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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaPowerProductOp_Laplace"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Buffers("Q")]
    [Quality(QualityBand.Experimental)]
    public static class GammaPowerProductOp_Laplace
    {
        // derivatives of the factor marginalized over Product and A
        internal static double[] dlogfs(double b, GammaPower product, GammaPower A)
        {
            if (A.Power != product.Power) throw new NotSupportedException($"A.Power ({A.Power}) != product.Power ({product.Power})");
            if (product.IsPointMass)
            {
                double productPointPower = Math.Pow(product.Point, 1 / A.Power);
                if (productPointPower > double.MaxValue) return new double[] { 0, 0, 0, 0 };
                // int delta(a^pa * b^pb - y) Ga(a; s, r) da 
                // = int delta(a' - y) Ga(a'^(1/pa)/b^(pb/pa); s, r) a'^(1/pa-1)/b^(pb/pa)/pa da' 
                // = Ga(y^(1/pa)/b^(pb/pa); s, r) y^(1/pa-1)/b^(pb/pa)/pa
                // logf = -s*pb/pa*log(b) - r*y^(1/pa)/b^(pb/pa)
                double ib = 1 / b;
                double ib2 = ib * ib;
                double s = A.Shape;
                double c = A.Rate * productPointPower / b;
                double dlogf = -s * ib + c * ib;
                double ddlogf = s * ib2 - 2 * c * ib2;
                double dddlogf = -2 * s * ib * ib2 + 6 * c * ib2 * ib;
                double d4logf = 6 * s * ib2 * ib2 - 24 * c * ib2 * ib2;
                return new double[] { dlogf, ddlogf, dddlogf, d4logf };
            }
            else if (A.IsPointMass)
            {
                // (a * b^pb)^(y_s/py - 1) exp(-y_r*(a*b^pb)^(1/py))
                // logf = (y_s/py-1)*pb*log(b) - y_r*a^(1/py)*b^(pb/py)
                double ib = 1 / b;
                double ib2 = ib * ib;
                double s = product.Shape - product.Power;
                double c = product.Rate * Math.Pow(A.Point, 1 / product.Power);
                double dlogf = s * ib - c;
                double ddlogf = -s * ib2;
                double dddlogf = 2 * s * ib * ib2;
                double d4logf = -6 * s * ib2 * ib2;
                return new double[] { dlogf, ddlogf, dddlogf, d4logf };
            }
            else
            {
                // int (a^pa * b^pb)^(y_s/y_p - 1) exp(-y_r*(a^pa*b^pb)^(1/y_p)) Ga(a; s, r) da 
                // = int a^(pa*(y_s/y_p-1) + s-1) b^(pb*(y_s/y_p-1)) exp(-y_r a^(pa/y_p) b^(pb/y_p) -r*a) da
                // where pa = pb = y_p:
                // = int a^(y_s-pa + s-1) b^(y_s-y_p) exp(-(y_r b + r)*a) da
                // = b^(y_s-y_p) / (r + b y_r)^(y_s-pa + s) 
                // logf = (y_s-y_p)*log(b) - (s+y_s-pa)*log(r + b*y_r)
                double b2 = b * b;
                double s = product.Shape - product.Power;
                double c = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(product.Shape, A.Shape) + (1 - A.Power);
                double denom = A.Rate / product.Rate + b;
                if (product.Rate == 0)
                {
                    c = 0;
                    denom = double.PositiveInfinity;
                }
                if (b < 1e-77)
                {
                    double bOverDenom = b / denom;
                    double bOverDenom2 = bOverDenom * bOverDenom;
                    double dlogf = (s - c * bOverDenom) / b;
                    double ddlogf = (-s + c * bOverDenom2) / b / b;
                    double dddlogf = (2 * s - 2 * c * bOverDenom * bOverDenom2) / b / b / b;
                    double d4logf = (-6 * s + 6 * c * bOverDenom2 * bOverDenom2) / b / b / b / b;
                    return new double[] { dlogf, ddlogf, dddlogf, d4logf };
                }
                else
                {
                    double denom2 = denom * denom;
                    double dlogf = s / b - c / denom;
                    double ddlogf = -s / b2 + c / denom2;
                    double dddlogf = 2 * s / (b * b2) - 2 * c / (denom * denom2);
                    double d4logf = -6 * s / (b2 * b2) + 6 * c / (denom2 * denom2);
                    return new double[] { dlogf, ddlogf, dddlogf, d4logf };
                }
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaPowerProductOp_Laplace"]/message_doc[@name="QInit()"]/*'/>
        [Skip]
        public static Gamma QInit()
        {
            return Gamma.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaPowerProductOp_Laplace"]/message_doc[@name="Q(GammaPower, GammaPower, GammaPower)"]/*'/>
        [Fresh]
        public static Gamma Q(GammaPower product, [Proper] GammaPower A, [Proper] GammaPower B)
        {
            // ensure B has the larger shape
            if (B.Shape < A.Shape) return Q(product, B, A);
            if (B.IsPointMass)
                return Gamma.PointMass(B.Point);
            if (A.IsPointMass)
                return Gamma.FromShapeAndRate(B.Shape, B.Rate);
            if (A.Power != product.Power) throw new NotSupportedException($"A.Power ({A.Power}) != product.Power ({product.Power})");
            if (B.Power != product.Power) throw new NotSupportedException($"B.Power ({B.Power}) != product.Power ({product.Power})");
            double x;
            if (product.IsPointMass)
            {
                if (product.Point == 0) return Gamma.PointMass(0);
                double productPointPower = Math.Pow(product.Point, 1 / A.Power);
                // y = product^(1/power)
                // logf = -a_s*log(b) - y*a_r/b
                // logp = b_s*log(b) - b_r*b
                // dlogfp = (b_s-a_s)/b - b_r + y*a_r/b^2 = 0
                // -b_r b^2 + (b_s-a_s) b + y*a_r = 0
                double shape = B.Shape - A.Shape;
                x = (Math.Sqrt(shape * shape + 4 * B.Rate * A.Rate * productPointPower) + shape) / 2 / B.Rate;
            }
            else
            {
                double shape1 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(B.Shape, product.Shape) + (1 - product.Power);
                if (product.Rate == 0)
                {
                    x = GammaFromShapeAndRateOp_Slow.FindMaximum(shape1, 0, A.Rate, B.Rate);
                }
                else
                {
                    double shape2 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(A.Shape, product.Shape) + (1 - A.Power);
                    // find the maximum of the factor marginalized over Product and A, times B
                    // From above:
                    // logf = (y_s/y_p-1)*pb*log(b) - (s+y_s-pa)*log(r + b^(pb/y_p)*y_r)
                    x = GammaFromShapeAndRateOp_Slow.FindMaximum(shape1, shape2, A.Rate / product.Rate, B.Rate);
                }
                if (x == 0)
                    x = 1e-100;
            }
            double[] dlogfss = dlogfs(x, product, A);
            double dlogf = dlogfss[0];
            double ddlogf = dlogfss[1];
            return GammaFromShapeAndRateOp_Laplace.GammaFromDerivatives(Gamma.FromShapeAndRate(B.Shape, B.Rate), x, dlogf, ddlogf);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaPowerProductOp_Laplace"]/message_doc[@name="LogAverageFactor(GammaPower, GammaPower, GammaPower, Gamma)"]/*'/>
        public static double LogAverageFactor(GammaPower product, GammaPower A, GammaPower B, Gamma q)
        {
            if (B.Shape < A.Shape) return LogAverageFactor(product, B, A, q);
            if (B.IsPointMass)
                return GammaProductOp.LogAverageFactor(product, A, B.Point);
            if (A.IsPointMass)
                return GammaProductOp.LogAverageFactor(product, A.Point, B);
            double qPoint = q.GetMean();
            double logf;
            if (product.IsPointMass)
            {
                // Ga(y/q; s, r)/q
                if (qPoint == 0)
                {
                    if (product.Point == 0)
                        logf = A.GetLogProb(0);
                    else
                        logf = double.NegativeInfinity;
                }
                else logf = A.GetLogProb(product.Point / qPoint) - Math.Log(qPoint);
            }
            else
            {
                // int Ga^y_p(a^pa b^pb; y_s, y_r) Ga(a; s, r) da = q^(y_s-y_p) / (r + q y_r)^(y_s + s-pa)  Gamma(y_s+s-pa)
                double shape = product.Shape - product.Power;
                double shape2 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(product.Shape, A.Shape) + (1 - A.Power);
                if (IsProper(product) && product.Shape > A.Shape)
                {
                    // same as below but product.GetLogNormalizer() is inlined and combined with other terms
                    double AShapeMinusPower = A.Shape - A.Power;
                    logf = shape * Math.Log(qPoint)
                        - Gamma.FromShapeAndRate(A.Shape, A.Rate).GetLogNormalizer()
                        - product.Shape * Math.Log(A.Rate / product.Rate + qPoint)
                        - Math.Log(Math.Abs(product.Power));
                    if (AShapeMinusPower != 0)
                        logf += AShapeMinusPower * (MMath.RisingFactorialLnOverN(product.Shape, AShapeMinusPower) - Math.Log(A.Rate + qPoint * product.Rate));
                }
                else
                {
                    logf = shape * Math.Log(qPoint)
                        - shape2 * Math.Log(A.Rate + qPoint * product.Rate)
                        + MMath.GammaLn(shape2)
                        - Gamma.FromShapeAndRate(A.Shape, A.Rate).GetLogNormalizer()
                        - product.GetLogNormalizer();
                    // normalizer is -MMath.GammaLn(Shape) + Shape * Math.Log(Rate) - Math.Log(Math.Abs(Power))
                }
            }
            double logz = logf + Gamma.FromShapeAndRate(B.Shape, B.Rate).GetLogProb(qPoint) - q.GetLogProb(qPoint);
            return logz;
        }

        private static bool IsProper(GammaPower gammaPower)
        {
            return (gammaPower.Shape > 0) && (gammaPower.Rate > 0) && !double.IsInfinity(gammaPower.Power);
        }

        private static double LogAverageFactor_slow(GammaPower product, GammaPower A, [Proper] GammaPower B)
        {
            return LogAverageFactor(product, A, B, Q(product, A, B));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaPowerProductOp_Laplace"]/message_doc[@name="LogEvidenceRatio(GammaPower, GammaPower, GammaPower, GammaPower, Gamma)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] GammaPower product, GammaPower A, GammaPower B, GammaPower to_product, Gamma q)
        {
            //if (double.IsPositiveInfinity(product.Rate)) return double.NegativeInfinity;
            return LogAverageFactor(product, A, B, q) - to_product.GetLogAverageOf(product);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaPowerProductOp_Laplace"]/message_doc[@name="LogEvidenceRatio(double, GammaPower, GammaPower, Gamma)"]/*'/>
        public static double LogEvidenceRatio(double product, GammaPower A, GammaPower B, Gamma q)
        {
            return LogAverageFactor(GammaPower.PointMass(product, 1), A, B, q);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaPowerProductOp_Laplace"]/message_doc[@name="BAverageConditional(GammaPower, GammaPower, GammaPower, Gamma, GammaPower)"]/*'/>
        public static GammaPower BAverageConditional([SkipIfUniform] GammaPower product, [Proper] GammaPower A, [Proper] GammaPower B, Gamma q, GammaPower result)
        {
            if (B.Shape < A.Shape) return AAverageConditional(product, B, A, q, result);
            if (A.IsPointMass)
                return GammaProductOp.BAverageConditional(product, A.Point, result);
            if (B.IsPointMass)
                return GammaPower.Uniform(result.Power); // TODO
            if (product.IsUniform()) return product;
            if (q.IsUniform())
                q = Q(product, A, B);
            double qPoint = q.GetMean();
            GammaPower bMarginal;
            // threshold ensures 6/qPoint^4 does not overflow
            double threshold = Math.Sqrt(Math.Sqrt(6 / double.MaxValue));
            if (result.Power < 0 && qPoint > threshold)
            {
                double iqMean, iqVariance;
                GetIQMoments(product, A, q, qPoint, out iqMean, out iqVariance);
                GammaPower iqMarginal = GammaPower.FromMeanAndVariance(iqMean, iqVariance, -1);
                bMarginal = GammaPower.FromShapeAndRate(iqMarginal.Shape, iqMarginal.Rate, result.Power);
            }
            else
            {
                // B.Shape >= A.Shape therefore Q is the approximate distribution of B^(1/B.Power).
                // We compute the approximate moments of q = b^(1/b.Power) to get a Gamma distribution and then raise to B.Power.
                double qMean, qVariance;
                GetQMoments(product, A, q, qPoint, out qMean, out qVariance);
                bMarginal = GammaPower.FromGamma(Gamma.FromMeanAndVariance(qMean, qVariance), result.Power);
            }
            result.SetToRatio(bMarginal, B, GammaProductOp_Laplace.ForceProper);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaPowerProductOp_Laplace"]/message_doc[@name="ProductAverageConditional(GammaPower, GammaPower, GammaPower, Gamma, GammaPower)"]/*'/>
        public static GammaPower ProductAverageConditional(GammaPower product, [Proper] GammaPower A, [SkipIfUniform] GammaPower B, Gamma q, GammaPower result)
        {
            if (B.Shape < A.Shape) return ProductAverageConditional(product, B, A, q, result);
            if (B.IsPointMass)
                return GammaProductOp.ProductAverageConditional(A, B.Point);
            if (B.IsUniform())
                return GammaPower.Uniform(result.Power);
            if (A.IsPointMass)
                return GammaProductOp.ProductAverageConditional(A.Point, B);
            if (product.IsPointMass)
                return GammaPower.Uniform(result.Power); // TODO
            if (A.Power != product.Power) throw new NotSupportedException($"A.Power ({A.Power}) != product.Power ({product.Power})");
            if (B.Power != product.Power) throw new NotSupportedException($"B.Power ({B.Power}) != product.Power ({product.Power})");
            if (A.Rate == 0)
            {
                if (B.Rate == 0) return GammaPower.FromShapeAndRate(Math.Min(A.Shape, B.Shape), 0, result.Power);
                else return A;
            }
            if (B.Rate == 0) return B;

            double qPoint = q.GetMean();
            double r = product.Rate;
            double shape2 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(product.Shape, A.Shape) + (1 - A.Power);
            GammaPower productMarginal;
            // threshold ensures 6/qPoint^4 does not overflow
            double threshold = Math.Sqrt(Math.Sqrt(6 / double.MaxValue));
            if (shape2 > 2 && result.Power < 0 && qPoint > threshold) 
            {
                // Compute the moments of product^(-1/product.Power)
                // Here q = b^(1/b.Power)
                // E[a^(-1/a.Power) b^(-1/b.Power)] = E[(q r + a_r)/(shape2-1)/q]
                // var(a^(-1/a.Power) b^(-1/b.Power)) = E[(q r + a_r)^2/(shape2-1)/(shape2-2)/q^2] - E[a^(-1/a.Power) b^(-1/b.Power)]^2
                //          = (var((q r + a_r)/q) + E[(q r + a_r)/q]^2)/(shape2-1)/(shape2-2) - E[(q r + a_r)/q]^2/(shape2-1)^2
                //          = var((q r + a_r)/q)/(shape2-1)/(shape2-2) + E[(q r + a_r)/(shape2-1)/q]^2/(shape2-2)
                double iqMean, iqVariance;
                GetIQMoments(product, A, q, qPoint, out iqMean, out iqVariance);
                double ipMean = (r + A.Rate * iqMean) / (shape2 - 1);
                double ipVariance = (iqVariance * A.Rate * A.Rate / (shape2 - 1) + ipMean * ipMean) / (shape2 - 2);
                // TODO: use ipVarianceOverMeanSquared
                GammaPower ipMarginal = GammaPower.FromMeanAndVariance(ipMean, ipVariance, -1);
                if (ipMarginal.IsUniform())
                {
                    return GammaPower.Uniform(result.Power);
                }
                else
                    productMarginal = GammaPower.FromShapeAndRate(ipMarginal.Shape, ipMarginal.Rate, result.Power);
                bool check = false;
                if (check)
                {
                    // Importance sampling
                    MeanVarianceAccumulator mvaInvQ = new MeanVarianceAccumulator();
                    MeanVarianceAccumulator mvaInvProduct = new MeanVarianceAccumulator();
                    Gamma qPrior = Gamma.FromShapeAndRate(B.Shape, B.Rate);
                    double shift = (product.Shape - product.Power) * Math.Log(qPoint) - shape2 * Math.Log(A.Rate + qPoint * r) + qPrior.GetLogProb(qPoint) - q.GetLogProb(qPoint);
                    for (int i = 0; i < 1000000; i++)
                    {
                        double qSample = q.Sample();
                        // logf = (y_s-y_p)*log(b) - (s+y_s-pa)*log(r + b*y_r)
                        double logf = (product.Shape - product.Power) * Math.Log(qSample) - shape2 * Math.Log(A.Rate + qSample * r) + qPrior.GetLogProb(qSample) - q.GetLogProb(qSample);
                        double weight = Math.Exp(logf - shift);
                        mvaInvQ.Add(1 / qSample, weight);
                        double invProduct = (r + A.Rate / qSample) / (shape2 - 1);
                        mvaInvProduct.Add(invProduct, weight);
                    }
                    Trace.WriteLine($"invQ = {mvaInvQ}, {iqMean}, {iqVariance}");
                    Trace.WriteLine($"invProduct = {mvaInvProduct}");
                    Trace.WriteLine($"invA = {mvaInvProduct.Variance * (shape2 - 1) / (shape2 - 2) + mvaInvProduct.Mean * mvaInvProduct.Mean / (shape2 - 2)}, {ipMean}, {ipVariance}");
                    Trace.WriteLine($"productMarginal = {productMarginal}");
                }
            }
            else
            {
                // Compute the moments of y = product^(1/product.Power)
                // yMean = E[shape2*b/(b y_r + a_r)]
                // yVariance = E[shape2*(shape2+1)*b^2/(b y_r + a_r)^2] - yMean^2 
                //           = var(shape2*b/(b y_r + a_r)) + E[shape2*b^2/(b y_r + a_r)^2]
                //           = shape2^2*var(b/(b y_r + a_r)) + shape2*(var(b/(b y_r + a_r)) + (yMean/shape2)^2)
                // Let g = b/(b y_r + a_r)
                double denom = qPoint * r + A.Rate;
                double denom2 = denom * denom;
                double rOverDenom = r / denom;
                double[] gDerivatives = (denom == 0)
                    ? new double[] { 0, 0, 0, 0 }
                    : new double[] { qPoint / denom, A.Rate / denom2, -2 * A.Rate / denom2 * rOverDenom, 6 * A.Rate / denom2 * rOverDenom * rOverDenom };
                double gMean, gVariance;
                GaussianOp_Laplace.LaplaceMoments(q, gDerivatives, dlogfs(qPoint, product, A), out gMean, out gVariance);
                double yMean = shape2 * gMean;
                double yVariance = shape2 * shape2 * gVariance + shape2 * (gVariance + gMean * gMean);
                productMarginal = GammaPower.FromGamma(Gamma.FromMeanAndVariance(yMean, yVariance), result.Power);
            }

            result.SetToRatio(productMarginal, product, GammaProductOp_Laplace.ForceProper);
            if (double.IsNaN(result.Shape) || double.IsNaN(result.Rate))
                throw new InferRuntimeException("result is nan");
            return result;
        }


        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaPowerProductOp_Laplace"]/message_doc[@name="AAverageConditional(GammaPower, GammaPower, GammaPower, Gamma, GammaPower)"]/*'/>
        public static GammaPower AAverageConditional([SkipIfUniform] GammaPower product, GammaPower A, [SkipIfUniform] GammaPower B, Gamma q, GammaPower result)
        {
            if (B.Shape < A.Shape) return BAverageConditional(product, B, A, q, result);
            if (B.IsPointMass)
                return GammaProductOp.AAverageConditional(product, B.Point, result);
            if (A.IsPointMass)
                return GammaPower.Uniform(A.Power); // TODO
            if (product.IsUniform()) return product;
            double qPoint = q.GetMean();
            GammaPower aMarginal;
            if (product.IsPointMass)
            {
                // Z = int Ga(y/q; s, r)/q Ga(q; q_s, q_r) dq
                // E[a] = E[product/q]
                // E[a^2] = E[product^2/q^2]
                // aVariance = E[a^2] - aMean^2
                double productPoint = product.Point;
                if (productPoint == 0) aMarginal = GammaPower.PointMass(0, result.Power);
                else
                {
                    double iqMean, iqVariance;
                    GetIQMoments(product, A, q, qPoint, out iqMean, out iqVariance);
                    double aMean = productPoint * iqMean;
                    double aVariance = productPoint * productPoint * iqVariance;
                    aMarginal = GammaPower.FromGamma(Gamma.FromMeanAndVariance(aMean, aVariance), result.Power);
                }
            }
            else
            {
                if (double.IsPositiveInfinity(product.Rate))
                {
                    return GammaPower.PointMass(0, result.Power);
                }
                if (A.Power != product.Power) throw new NotSupportedException($"A.Power ({A.Power}) != product.Power ({product.Power})");
                if (B.Power != product.Power) throw new NotSupportedException($"B.Power ({B.Power}) != product.Power ({product.Power})");
                double r = product.Rate;
                double g = 1 / (qPoint * r + A.Rate);
                double g2 = g * g;
                double shape2 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(product.Shape, A.Shape) + (1 - A.Power);
                // From above:
                // a^(y_s-pa + a_s-1) exp(-(y_r b + a_r)*a)
                if (shape2 > 2)
                {
                    // Compute the moments of a^(-1/a.Power)
                    // Here q = b^(1/b.Power)
                    // E[a^(-1/a.Power)] = E[(q r + a_r)/(shape2-1)]
                    // var(a^(-1/a.Power)) = E[(q r + a_r)^2/(shape2-1)/(shape2-2)] - E[a^(-1/a.Power)]^2
                    //          = (var(q r + a_r) + E[(q r + a_r)]^2)/(shape2-1)/(shape2-2) - E[(q r + a_r)]^2/(shape2-1)^2
                    //          = var(q r + a_r)/(shape2-1)/(shape2-2) + E[(q r + a_r)/(shape2-1)]^2/(shape2-2)
                    // TODO: share this computation with BAverageConditional
                    double qMean, qVariance;
                    GetQMoments(product, A, q, qPoint, out qMean, out qVariance);
                    double iaMean = (qMean * r + A.Rate) / (shape2 - 1);
                    //double iaVariance = (qVariance * r2 / (shape2 - 1) + iaMean * iaMean) / (shape2 - 2);
                    // shape = mean^2/variance + 2
                    //double iaVarianceOverMeanSquared = (qVariance / (shape2 - 1) * r / iaMean * r / iaMean + 1) / (shape2 - 2);
                    double iaVarianceOverMeanSquared = (qVariance * (shape2 - 1) / (qMean + A.Rate / r) / (qMean + A.Rate / r) + 1) / (shape2 - 2);
                    //GammaPower iaMarginal = GammaPower.FromMeanAndVariance(iaMean, iaVariance, -1);
                    GammaPower iaMarginal = InverseGammaFromMeanAndVarianceOverMeanSquared(iaMean, iaVarianceOverMeanSquared);
                    if (iaMarginal.IsUniform())
                    {
                        if (result.Power > 0)
                            return GammaPower.PointMass(0, result.Power);
                        else
                            return GammaPower.Uniform(result.Power);
                    }
                    else
                        aMarginal = GammaPower.FromShapeAndRate(iaMarginal.Shape, iaMarginal.Rate, result.Power);
                    bool check = false;
                    if (check)
                    {
                        // Importance sampling
                        MeanVarianceAccumulator mvaB = new MeanVarianceAccumulator();
                        MeanVarianceAccumulator mvaInvA = new MeanVarianceAccumulator();
                        Gamma bPrior = Gamma.FromShapeAndRate(B.Shape, B.Rate);
                        q = bPrior;
                        double shift = (product.Shape - product.Power) * Math.Log(qPoint) - shape2 * Math.Log(A.Rate + qPoint * r) + bPrior.GetLogProb(qPoint) - q.GetLogProb(qPoint);
                        for (int i = 0; i < 1000000; i++)
                        {
                            double bSample = q.Sample();
                            // logf = (y_s-y_p)*log(b) - (s+y_s-pa)*log(r + b*y_r)
                            double logf = (product.Shape - product.Power) * Math.Log(bSample) - shape2 * Math.Log(A.Rate + bSample * r) + bPrior.GetLogProb(bSample) - q.GetLogProb(bSample);
                            double weight = Math.Exp(logf - shift);
                            mvaB.Add(bSample, weight);
                            double invA = (bSample * r + A.Rate) / (shape2 - 1);
                            mvaInvA.Add(invA, weight);
                        }
                        Trace.WriteLine($"b = {mvaB}, {qMean}, {qVariance}");
                        Trace.WriteLine($"invA = {mvaInvA} {mvaInvA.Variance * (shape2 - 1) / (shape2 - 2) + mvaInvA.Mean * mvaInvA.Mean / (shape2 - 2)}, {iaMean}, {iaVarianceOverMeanSquared * iaMean * iaMean}");
                        Trace.WriteLine($"aMarginal = {aMarginal}");
                    }
                }
                else
                {
                    // Compute the moments of a^(1/a.Power)
                    // aMean = shape2/(b y_r + a_r)
                    // aVariance = E[shape2*(shape2+1)/(b y_r + a_r)^2] - aMean^2 = var(shape2/(b y_r + a_r)) + E[shape2/(b y_r + a_r)^2]
                    //           = shape2^2*var(1/(b y_r + a_r)) + shape2*(var(1/(b y_r + a_r)) + (aMean/shape2)^2)
                    double r2 = r * r;
                    double[] gDerivatives = new double[] { g, -r * g2, 2 * g2 * g * r2, -6 * g2 * g2 * r2 * r };
                    double gMean, gVariance;
                    GaussianOp_Laplace.LaplaceMoments(q, gDerivatives, dlogfs(qPoint, product, A), out gMean, out gVariance);
                    double aMean = shape2 * gMean;
                    double aVariance = shape2 * shape2 * gVariance + shape2 * (gVariance + gMean * gMean);
                    aMarginal = GammaPower.FromGamma(Gamma.FromMeanAndVariance(aMean, aVariance), result.Power);
                }
            }
            result.SetToRatio(aMarginal, A, GammaProductOp_Laplace.ForceProper);
            if (double.IsNaN(result.Shape) || double.IsNaN(result.Rate))
                throw new InferRuntimeException("result is nan");
            return result;
        }

        private static GammaPower InverseGammaFromMeanAndVarianceOverMeanSquared(double mean, double varianceOverMeanSquared)
        {
            double shape = 1 / varianceOverMeanSquared + 2;
            return GammaPower.FromShapeAndRate(shape, mean * (shape - 1), -1);
        }

        private static void GetQMoments(GammaPower product, GammaPower A, Gamma q, double qPoint, out double qMean, out double qVariance)
        {
            double[] qDerivatives = new double[] { qPoint, 1, 0, 0 };
            GaussianOp_Laplace.LaplaceMoments(q, qDerivatives, dlogfs(qPoint, product, A), out qMean, out qVariance);
        }

        private static void GetIQMoments(GammaPower product, GammaPower A, Gamma q, double qPoint, out double iqMean, out double iqVariance)
        {
            double iq = 1 / qPoint;
            double iq2 = iq * iq;
            double[] iqDerivatives = new double[] { iq, -iq2, 2 * iq2 * iq, -6 * iq2 * iq2 };
            GaussianOp_Laplace.LaplaceMoments(q, iqDerivatives, dlogfs(qPoint, product, A), out iqMean, out iqVariance);
        }
    }
}
