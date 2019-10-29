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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaProductOp_Laplace"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Buffers("Q")]
    [Quality(QualityBand.Experimental)]
    public static class GammaPowerProductOp_Laplace
    {
        // derivatives of the factor marginalized over Product and A
        private static double[] dlogfs(double b, GammaPower product, GammaPower A)
        {
            if (A.Power != product.Power) throw new NotSupportedException($"A.Power ({A.Power}) != product.Power ({product.Power})");
            if (product.IsPointMass)
            {
                // int delta(a^pa * b^pb - y) Ga(a; s, r) da 
                // = int delta(a' - y) Ga(a'^(1/pa)/b^(pb/pa); s, r) a'^(1/pa-1)/b^(pb/pa)/pa da' 
                // = Ga(y^(1/pa)/b^(pb/pa); s, r) y^(1/pa-1)/b^(pb/pa)/pa
                // logf = -s*pb/pa*log(b) - r*y^(1/pa)/b^(pb/pa)
                double ib = 1 / b;
                double ib2 = ib * ib;
                double s = A.Shape;
                double c = A.Rate * Math.Pow(product.Point, 1 / A.Power) / b;
                double dlogf = -s * ib + c * ib;
                double ddlogf = s * ib2 - 2 * c * ib2 * ib;
                double dddlogf = -2 * s * ib * ib2 + 6 * c * ib2 * ib2;
                double d4logf = 6 * s * ib2 * ib2 - 24 * c * ib2 * ib2 * ib;
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
                double r = product.Rate;
                double denom = 1 / (A.Rate + b * r);
                double denom2 = denom * denom;
                double b2 = b * b;
                double s = product.Shape - product.Power;
                double c = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(product.Shape, A.Shape) + (1 - A.Power);
                double ddenom = r;
                double dlogf = s / b - c * denom * ddenom;
                double ddenom2 = ddenom * ddenom;
                double ddlogf = -s / b2 + c * denom2 * ddenom2;
                double dddlogf = 2 * s / (b * b2) - 2 * c * denom * denom2 * ddenom * ddenom2;
                double d4logf = -6 * s / (b2 * b2) + 6 * c * denom2 * denom2 * ddenom2 * ddenom2;
                return new double[] { dlogf, ddlogf, dddlogf, d4logf };
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
            if (B.IsPointMass)
                return Gamma.PointMass(B.Point);
            if (A.IsPointMass)
                throw new NotImplementedException();
            if (product.Rate == 0)
                return Gamma.FromShapeAndRate(B.Shape, B.Rate);
            double x;
            if (product.IsPointMass)
            {
                // logf = -a_s*log(b) - y*a_r/b
                // logp = b_s*log(b) - b_r*b
                // dlogfp = (b_s-a_s)/b - b_r + y*a_r/b^2 = 0
                // -b_r b^2 + (b_s-a_s) b + y*a_r = 0
                double shape = B.Shape - A.Shape;
                double y = product.Point;
                x = (Math.Sqrt(shape * shape + 4 * B.Rate * A.Rate * y) + shape) / 2 / B.Rate;
            }
            else
            {
                if (A.Power != product.Power) throw new NotSupportedException($"A.Power ({A.Power}) != product.Power ({product.Power})");
                if (B.Power != product.Power) throw new NotSupportedException($"B.Power ({B.Power}) != product.Power ({product.Power})");
                double shape1 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(B.Shape, product.Shape) + (1 - product.Power);
                double shape2 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(A.Shape, product.Shape) + (1 - A.Power);
                // find the maximum of the factor marginalized over Product and A, times B
                // From above:
                // logf = (y_s/y_p-1)*pb*log(b) - (s+y_s-pa)*log(r + b^(pb/y_p)*y_r)
                // let b' = b^(pb/y_p)*y_r and maximize over b'
                // logf = (y_s/y_p-1)*y_p*log(b') - (s+y_s-pa)*log(r + b')
                x = GammaFromShapeAndRateOp_Slow.FindMaximum(shape1, shape2, A.Rate, B.Rate / product.Rate);
                if (x == 0)
                    x = 1e-100;
                x /= product.Rate;
            }
            double[] dlogfss = dlogfs(x, product, A);
            double dlogf = dlogfss[0];
            double ddlogf = dlogfss[1];
            return GammaFromShapeAndRateOp_Laplace.GammaFromDerivatives(Gamma.FromShapeAndRate(B.Shape, B.Rate), x, dlogf, ddlogf);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaPowerProductOp_Laplace"]/message_doc[@name="LogAverageFactor(GammaPower, GammaPower, GammaPower, Gamma)"]/*'/>
        public static double LogAverageFactor(GammaPower product, GammaPower A, GammaPower B, Gamma q)
        {
            if (B.IsPointMass)
                return GammaProductOp.LogAverageFactor(product, A, B.Point);
            if (A.IsPointMass)
                return GammaProductOp.LogAverageFactor(product, A.Point, B);
            double x = q.GetMean();
            double logf;
            if (product.IsPointMass)
            {
                // Ga(y/b; s, r)/b
                double y = product.Point;
                logf = (A.Shape - 1) * Math.Log(y) - A.Shape * Math.Log(x) - A.Rate * y / x - A.GetLogNormalizer();
            }
            else
            {
                // int Ga^y_p(a^pa b^pb; y_s, y_r) Ga(a; s, r) da = b^(y_s-y_p) / (r + b y_r)^(y_s + s-pa)  Gamma(y_s+s-pa)
                double shape = product.Shape - product.Power;
                double shape2 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(product.Shape, A.Shape) + (1-A.Power);
                logf = shape * Math.Log(x) - shape2 * Math.Log(A.Rate + x * product.Rate) +
                  MMath.GammaLn(shape2) - Gamma.FromShapeAndRate(A.Shape, A.Rate).GetLogNormalizer() - product.GetLogNormalizer();
            }
            double logz = logf + Gamma.FromShapeAndRate(B.Shape, B.Rate).GetLogProb(x) - q.GetLogProb(x);
            return logz;
        }

        private static double LogAverageFactor_slow(GammaPower product, GammaPower A, [Proper] GammaPower B)
        {
            return LogAverageFactor(product, A, B, Q(product, A, B));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaPowerProductOp_Laplace"]/message_doc[@name="LogEvidenceRatio(GammaPower, GammaPower, GammaPower, GammaPower, Gamma)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] GammaPower product, GammaPower A, GammaPower B, GammaPower to_product, Gamma q)
        {
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
            if (A.IsPointMass)
                return GammaProductOp.BAverageConditional(product, A.Point, result);
            if (B.IsPointMass)
                throw new NotImplementedException();
            if (product.IsUniform()) return product;
            if (q.IsUniform())
                q = Q(product, A, B);
            double bPoint = q.GetMean();
            // derivatives of b^b_power
            double g = Math.Pow(bPoint, B.Power);
            double[] bDerivatives = new double[] { g, 0, 0, 0 };
            bDerivatives[1] = bDerivatives[0] * B.Power / bPoint;
            bDerivatives[2] = bDerivatives[1] * (B.Power - 1) / bPoint;
            bDerivatives[3] = bDerivatives[2] * (B.Power - 2) / bPoint;
            double bMean, bVariance;
            GaussianOp_Laplace.LaplaceMoments(q, bDerivatives, dlogfs(bPoint, product, A), out bMean, out bVariance);
            GammaPower bMarginal = GammaPower.FromMeanAndVariance(bMean, bVariance, result.Power);
            result.SetToRatio(bMarginal, B, true);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaPowerProductOp_Laplace"]/message_doc[@name="ProductAverageConditional(GammaPower, GammaPower, GammaPower, Gamma, GammaPower)"]/*'/>
        public static GammaPower ProductAverageConditional(GammaPower product, [Proper] GammaPower A, [SkipIfUniform] GammaPower B, Gamma q, GammaPower result)
        {
            if (B.IsPointMass)
                return GammaProductOp.ProductAverageConditional(A, B.Point);
            if (A.IsPointMass)
                return GammaProductOp.ProductAverageConditional(A.Point, B);
            if (product.IsPointMass)
                throw new NotImplementedException();

            if (A.Power != product.Power) throw new NotSupportedException($"A.Power ({A.Power}) != product.Power ({product.Power})");
            if (B.Power != product.Power) throw new NotSupportedException($"B.Power ({B.Power}) != product.Power ({product.Power})");
            double productMean, productVariance;
            double bPoint = q.GetMean();
            double r = product.Rate;
            double r2 = r * r;
            double denom = 1 / (bPoint * r + A.Rate);
            double denom2 = denom * denom;
            double shape2 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(product.Shape, A.Shape) + (1 - A.Power);
            // yMean = shape2*b^(pb)/(b y_r + a_r)
            // yVariance = E[shape2*(shape2+1)*b^2/(b y_r + a_r)^2] - yMean^2 
            //           = var(shape2*b/(b y_r + a_r)) + E[shape2*b^2/(b y_r + a_r)^2]
            //           = shape2^2*var(b/(b y_r + a_r)) + shape2*(var(b/(b y_r + a_r)) + (yMean/shape2)^2)
            double[] gDerivatives = new double[] { bPoint * denom, A.Rate * denom2, -2 * denom2 * denom * A.Rate * r, 6 * denom2 * denom2 * A.Rate * r2 };
            double gMean, gVariance;
            GaussianOp_Laplace.LaplaceMoments(q, gDerivatives, dlogfs(bPoint, product, A), out gMean, out gVariance);
            productMean = shape2 * gMean;
            productVariance = shape2 * shape2 * gVariance + shape2 * (gVariance + gMean * gMean);

            GammaPower productMarginal = GammaPower.FromGamma(Gamma.FromMeanAndVariance(productMean, productVariance), result.Power);
            result.SetToRatio(productMarginal, product, true);
            if (double.IsNaN(result.Shape) || double.IsNaN(result.Rate))
                throw new InferRuntimeException("result is nan");
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GammaPowerProductOp_Laplace"]/message_doc[@name="AAverageConditional(GammaPower, GammaPower, GammaPower, Gamma, GammaPower)"]/*'/>
        public static GammaPower AAverageConditional([SkipIfUniform] GammaPower product, GammaPower A, [SkipIfUniform] GammaPower B, Gamma q, GammaPower result)
        {
            if (B.IsPointMass)
                return GammaProductOp.AAverageConditional(product, B.Point, result);
            if (A.IsPointMass)
                throw new NotImplementedException();
            if (product.IsUniform()) return product;
            double aMean, aVariance;
            double bPoint = q.GetMean();
            if (product.IsPointMass)
            {
                // Z = int Ga(y/b; s, r)/b Ga(b; b_s, b_r) db
                // E[a] = E[y/b]
                // E[a^2] = E[y^2/b^2]
                // aVariance = E[a^2] - aMean^2
                double y = product.Point;
                double ib = 1 / bPoint;
                double ib2 = ib * ib;
                double[] g = new double[] { ib, -ib2, 2 * ib2 * ib, -6 * ib2 * ib2 };
                double pMean, pVariance;
                GaussianOp_Laplace.LaplaceMoments(q, g, dlogfs(bPoint, product, A), out pMean, out pVariance);
                aMean = y * pMean;
                aVariance = y * y * pVariance;
            }
            else
            {
                if (A.Power != product.Power) throw new NotSupportedException($"A.Power ({A.Power}) != product.Power ({product.Power})");
                if (B.Power != product.Power) throw new NotSupportedException($"B.Power ({B.Power}) != product.Power ({product.Power})");
                double r = product.Rate;
                double r2 = r * r;
                double g = 1 / (bPoint * r + A.Rate);
                double g2 = g * g;
                double shape2 = GammaFromShapeAndRateOp_Slow.AddShapesMinus1(product.Shape, A.Shape) + (1 - A.Power);
                // From above:
                // a^(y_s-pa + a_s-1) exp(-(y_r b + a_r)*a)
                // aMean = shape2/(b y_r + a_r)
                // aVariance = E[shape2*(shape2+1)/(b y_r + a_r)^2] - aMean^2 = var(shape2/(b y_r + a_r)) + E[shape2/(b y_r + a_r)^2]
                //           = shape2^2*var(1/(b y_r + a_r)) + shape2*(var(1/(b y_r + a_r)) + (aMean/shape2)^2)
                double[] gDerivatives = new double[] { g, -r * g2, 2 * g2 * g * r2, -6 * g2 * g2 * r2 * r };
                double gMean, gVariance;
                GaussianOp_Laplace.LaplaceMoments(q, gDerivatives, dlogfs(bPoint, product, A), out gMean, out gVariance);
                aMean = shape2 * gMean;
                aVariance = shape2 * shape2 * gVariance + shape2 * (gVariance + gMean * gMean);
            }
            GammaPower aMarginal = GammaPower.FromGamma(Gamma.FromMeanAndVariance(aMean, aVariance), result.Power);
            result.SetToRatio(aMarginal, A, true);
            if (double.IsNaN(result.Shape) || double.IsNaN(result.Rate))
                throw new InferRuntimeException("result is nan");
            return result;
        }
    }
}
