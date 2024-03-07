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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpBase"]/doc/*'/>
    public class GaussianProductOpBase
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpBase"]/message_doc[@name="ProductAverageConditional(double, Gaussian)"]/*'/>
        public static Gaussian ProductAverageConditional(double A, [SkipIfUniform] Gaussian B)
        {
            return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpBase"]/message_doc[@name="ProductAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, double B)
        {
            return ProductAverageConditional(B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpBase"]/message_doc[@name="ProductAverageConditional(double, double)"]/*'/>
        public static Gaussian ProductAverageConditional(double a, double b)
        {
            return Gaussian.PointMass(a * b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpBase"]/message_doc[@name="AAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, double B)
        {
            if (Product.IsPointMass)
                return AAverageConditional(Product.Point, B);
            if (Product.IsUniform()) 
                return Product;
            // (m - ab)^2/v = (a^2 b^2 - 2abm + m^2)/v
            // This code works correctly even if B=0 or Product is uniform (and B is finite). 
            Gaussian result = new Gaussian();
            result.Precision = Product.Precision * B * B;
            result.MeanTimesPrecision = Product.MeanTimesPrecision * B;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpBase"]/message_doc[@name="AAverageConditional(double, double)"]/*'/>
        public static Gaussian AAverageConditional(double Product, double B)
        {
            if (B == 0)
            {
                if (Product != 0)
                    throw new AllZeroException();
                return Gaussian.Uniform();
            }
            else
                return Gaussian.PointMass(Product / B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpBase"]/message_doc[@name="BAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, double A)
        {
            return AAverageConditional(Product, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpBase"]/message_doc[@name="BAverageConditional(double, double)"]/*'/>
        public static Gaussian BAverageConditional(double Product, double A)
        {
            return AAverageConditional(Product, A);
        }

        // TruncatedGaussian //////////////////////////////////////////////////////////////////////////////////////////////

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpBase"]/message_doc[@name="ProductAverageConditional(double, TruncatedGaussian)"]/*'/>
        public static TruncatedGaussian ProductAverageConditional(double A, [SkipIfUniform] TruncatedGaussian B)
        {
            return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpBase"]/message_doc[@name="ProductAverageConditional(TruncatedGaussian, double)"]/*'/>
        public static TruncatedGaussian ProductAverageConditional([SkipIfUniform] TruncatedGaussian A, double B)
        {
            return ProductAverageConditional(B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpBase"]/message_doc[@name="AAverageConditional(TruncatedGaussian, double)"]/*'/>
        public static TruncatedGaussian AAverageConditional([SkipIfUniform] TruncatedGaussian Product, double B)
        {
            if (Product.IsUniform()) return Product;
            return new TruncatedGaussian(AAverageConditional(Product.Gaussian, B), ((B >= 0) ? Product.LowerBound : Product.UpperBound) / B, ((B >= 0) ? Product.UpperBound : Product.LowerBound) / B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpBase"]/message_doc[@name="BAverageConditional(TruncatedGaussian, double)"]/*'/>
        public static TruncatedGaussian BAverageConditional([SkipIfUniform] TruncatedGaussian Product, double A)
        {
            return AAverageConditional(Product, A);
        }
    }

    public class GaussianProductOpEvidenceBase : GaussianProductOpBase
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpEvidenceBase"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, double, Gaussian)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogAverageFactor(Gaussian product, Gaussian a, double b, [Fresh] Gaussian to_product)
        {
            //Gaussian to_product = GaussianProductOp.ProductAverageConditional(a, b);
            return to_product.GetLogAverageOf(product);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpEvidenceBase"]/message_doc[@name="LogAverageFactor(Gaussian, double, Gaussian, Gaussian)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogAverageFactor(Gaussian product, double a, Gaussian b, [Fresh] Gaussian to_product)
        {
            return LogAverageFactor(product, b, a, to_product);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpEvidenceBase"]/message_doc[@name="LogAverageFactor(double, Gaussian, double)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogAverageFactor(double product, Gaussian a, double b)
        {
            Gaussian to_product = GaussianProductOp.ProductAverageConditional(a, b);
            return to_product.GetLogProb(product);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpEvidenceBase"]/message_doc[@name="LogAverageFactor(double, double, Gaussian)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogAverageFactor(double product, double a, Gaussian b)
        {
            return LogAverageFactor(product, b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpEvidenceBase"]/message_doc[@name="LogAverageFactor(double, double, double)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogAverageFactor(double product, double a, double b)
        {
            return (product == Factor.Product(a, b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpEvidenceBase"]/message_doc[@name="LogEvidenceRatio(double, double, double)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogEvidenceRatio(double product, double a, double b)
        {
            return LogAverageFactor(product, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpEvidenceBase"]/message_doc[@name="LogAverageFactor(Gaussian, double, double)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogAverageFactor(Gaussian product, double a, double b)
        {
            return product.GetLogProb(Factor.Product(a, b));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpEvidenceBase"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, double)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        [Skip]
        public static double LogEvidenceRatio(Gaussian product, Gaussian a, double b)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpEvidenceBase"]/message_doc[@name="LogEvidenceRatio(Gaussian, double, Gaussian)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        [Skip]
        public static double LogEvidenceRatio(Gaussian product, double a, Gaussian b)
        {
            return LogEvidenceRatio(product, b, a);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpEvidenceBase"]/message_doc[@name="LogEvidenceRatio(Gaussian, double, double)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        [Skip]
        public static double LogEvidenceRatio(Gaussian product, double a, double b)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpEvidenceBase"]/message_doc[@name="LogEvidenceRatio(double, Gaussian, double)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogEvidenceRatio(double product, Gaussian a, double b)
        {
            return LogAverageFactor(product, a, b);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOpEvidenceBase"]/message_doc[@name="LogEvidenceRatio(double, double, Gaussian)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogEvidenceRatio(double product, double a, Gaussian b)
        {
            return LogEvidenceRatio(product, b, a);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double), Default = true)]
    //[FactorMethod(new string[] { "A", "Product", "B" }, typeof(Factor), "Ratio", typeof(double), typeof(double), Default=true)]
    [Quality(QualityBand.Mature)]
    public class GaussianProductOp : GaussianProductOpEvidenceBase
    {
        /// <summary>
        /// The number of quadrature nodes used to compute the messages.
        /// Reduce this number to save time in exchange for less accuracy.
        /// Must be an odd number.
        /// </summary>
        public static int QuadratureNodeCount = 1001; // must be odd to avoid A=0

        /// <summary>
        /// Force proper messages
        /// </summary>
        public static bool ForceProper = true;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="ProductAverageConditionalInit(Gaussian, Gaussian)"]/*'/>
        [Skip]
        public static Gaussian ProductAverageConditionalInit(Gaussian A, Gaussian B)
        {
            return Gaussian.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="ProductAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gaussian ProductAverageConditional([NoInit] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (A.IsUniform() || B.IsUniform()) return Gaussian.Uniform();
            if (A.IsPointMass)
                return ProductAverageConditional(A.Point, B);
            if (B.IsPointMass)
                return ProductAverageConditional(A, B.Point);
            if (Product.IsPointMass || Product.Precision > 10)
                return GaussianProductOp_Slow.ProductAverageConditional(Product, A, B);
            if (Product.Precision < 1e-100)
                return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
            double mA, vA;
            A.GetMeanAndVariance(out mA, out vA);
            double mB, vB;
            B.GetMeanAndVariance(out mB, out vB);
            double mProduct, vProduct;
            Product.GetMeanAndVariance(out mProduct, out vProduct);
            // algorithm: quadrature on A from -1 to 1, plus quadrature on 1/A from -1 to 1.
            double z = 0, sumX = 0, sumX2 = 0;
            for (int i = 0; i <= QuadratureNodeCount; i++)
            {
                double a = (2.0 * i) / QuadratureNodeCount - 1;
                double logfA = Gaussian.GetLogProb(mProduct, a * mB, vProduct + a * a * vB) + Gaussian.GetLogProb(a, mA, vA);
                double fA = Math.Exp(logfA);

                z += fA;
                double b = (mB * vProduct + a * mProduct * vB) / (vProduct + a * a * vB);
                double b2 = b * b + (vProduct * vB) / (vProduct + a * a * vB);
                double x = a * b;
                double x2 = a * a * b2;
                sumX += x * fA;
                sumX2 += x2 * fA;

                double invA = a;
                a = 1.0 / invA;
                double logfInvA = Gaussian.GetLogProb(mProduct * invA, mB, vProduct * invA * invA + vB) + Gaussian.GetLogProb(a, mA, vA) - Math.Log(Math.Abs(invA + Double.Epsilon));
                double fInvA = Math.Exp(logfInvA);
                z += fInvA;
                b = (mB * vProduct + a * mProduct * vB) / (vProduct + a * a * vB);
                b2 = b * b + (vProduct * vB) / (vProduct + a * a * vB);
                x = a * b;
                x2 = a * a * b2;
                sumX += x * fInvA;
                sumX2 += x2 * fInvA;
            }
            if (z < 1e-310)
            {
                return GaussianProductOp_Slow.ProductAverageConditional(Product, A, B);
                //throw new Exception("quadrature found zero mass");
            }
            double mean = sumX / z;
            double var = sumX2 / z - mean * mean;
            Gaussian result = Gaussian.FromMeanAndVariance(mean, var);
            result.SetToRatio(result, Product, ForceProper);
            return result;
        }

        /// <summary>
        /// Compute the derivatives of the logarithm of the marginalized factor.
        /// </summary>
        /// <param name="a">The point to evaluate the derivatives</param>
        /// <param name="mB"></param>
        /// <param name="vB"></param>
        /// <param name="Product"></param>
        /// <param name="dlogf"></param>
        /// <param name="ddlogf"></param>
        public static void ADerivatives(double a, double mB, double vB, Gaussian Product, out double dlogf, out double ddlogf)
        {
            // f(a) = int_b N(mp; ab, vp) p(b) db
            //      = N(mp; a*mb, vp + a^2*vb)
            // log f(a) = -0.5*log(vp + a^2*vb) -0.5*(mp - a*mb)^2/(vp + a^2*vb)
            // dlogf = -a*vb/(vp + a^2*vb) + mb*(mp - a*mb)/(vp + a^2*vb) + a*vb*(mp - a*mb)^2/(vp + a^2*vb)^2
            // ddlogf = -vb/(vp + a^2*vb) + 2*a^2*vb^2/(vp + a^2*vb)^2 + mb*(- mb)/(vp + a^2*vb) - 2*a*vb*mb*(mp - a*mb)/(vp + a^2*vb)^2 + 
            //          vb*(mp - a*mb)^2/(vp + a^2*vb)^2 - 2*a*vb*mb*(mp - a*mb)/(vp + a^2*vb)^2 - (2*a*vb)^2*(mp - a*mb)^2/(vp + a^2*vb)^3
            // (-vb-mb^2)(vp+a^2*vb)/(vp+a^2*vb)^2 = ((-vb-mb^2)vp - a^2*vb^2 - a^2*vb*mb^2)/(vp+a^2*vb)^2
            // (2*a^2*vb^2 - 4*a*vb*mb*(mp - a*mb) + vb*(mp - a*mb)^2)/(vp + a^2*vb)^2 
            // = (2*a^2*vb^2 - 4*a*vb*mb*mp + 4*a^2*vb*mb^2 + vb*mp^2 - 2*a*vb*mb*mp + vb*a^2*mb^2)/(vp + a^2*vb)^2
            // = (2*a^2*vb^2 + 5*a^2*vb*mb^2 + vb*mp^2 - 6*a*vb*mb*mp)/(vp + a^2*vb)^2
            // together = (a^2*vb^2 + 4*a^2*vb*mb^2 + vb*mp^2 - 6*a*vb*mb*mp - vp*(vb+mb^2))/(vp + a^2*vb)^2
            if (double.IsInfinity(vB))
                throw new ArgumentException("vB is infinite");
            if (Product.IsPointMass)
                throw new ArgumentException("Product is a point mass");
            // v*Product.Precision
            double v = 1 + a * a * vB * Product.Precision;
            // diff*Product.Precision
            double diff = Product.MeanTimesPrecision - a * mB * Product.Precision;
            double diff2 = diff * diff;
            double v2 = v * v;
            double avb = a * vB;
            double avbPrec = avb * Product.Precision;
            dlogf = (mB * diff - avbPrec) / v + avb * diff2 / v2;
            ddlogf = (-mB * mB - vB) * Product.Precision / v + (2 * avbPrec * (avbPrec - 2 * mB * diff) + vB * diff2) / v2 - (4 * avb * avbPrec * diff2) / (v2 * v);
            //ddlogf = (avb*avb + 4*avb*a*mB*mB + vB*mProduct*mProduct - 6*avb*mB*mProduct - vProduct*(vB + mB*mB)) / denom2 - (4 * avb * avb * diff2) / (denom2 * denom);
            bool check = false;
            if (check)
            {
                double delta = 1e-6;
                double mProduct, vProduct;
                Product.GetMeanAndVariance(out mProduct, out vProduct);
                // logf1 =approx logf0 - delta*dlogf0 + 0.5*delta^2*ddlogf0 - 1/6*delta^3*dddlogf0
                double logf1 = GaussianProductOp_Slow.LogLikelihood(a - delta, mProduct, vProduct, 0, 0, mB, vB);
                // logf2 =approx logf0 + delta*dlogf0 + 0.5*delta^2*ddlogf0 + 1/6*delta^3*dddlogf0
                double logf2 = GaussianProductOp_Slow.LogLikelihood(a + delta, mProduct, vProduct, 0, 0, mB, vB);
                // error should be 1/3*delta^3*dddlogf0
                double dlogf0 = (logf2 - logf1) / delta / 2;
                double logf0 = GaussianProductOp_Slow.LogLikelihood(a, mProduct, vProduct, 0, 0, mB, vB);
                double ddlogf0 = (logf2 + logf1 - 2 * logf0) / delta / delta;
                double dlogfError = MMath.AbsDiff(dlogf, dlogf0, 1e-8);
                double ddlogfError = MMath.AbsDiff(ddlogf, ddlogf0, 1e-8);
                if (Math.Abs(ddlogf0 / dlogf0) < 1000)
                {
                    //Console.WriteLine("{0} should be {1} (error {2})", dlogf, dlogf0, dlogfError);
                    //Console.WriteLine("{0} should be {1} (error {2})", ddlogf, ddlogf0, ddlogfError);
                    if (dlogfError > delta * 10 || ddlogfError > delta * 1000)
                        throw new Exception("wrong derivative");
                }
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, [NoInit] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (B.IsPointMass)
                return AAverageConditional(Product, B.Point);
            if (Product.IsUniform() || B.IsUniform())
                return Gaussian.Uniform();
            double mA, vA;
            A.GetMeanAndVariance(out mA, out vA);
            double mB, vB;
            B.GetMeanAndVariance(out mB, out vB);
            if (A.IsPointMass)
            {
                ADerivatives(mA, mB, vB, Product, out double dlogf, out double ddlogf);
                return Gaussian.FromDerivatives(mA, dlogf, ddlogf, ForceProper);
            }
            else
            {
                // algorithm: quadrature on A from -1 to 1, plus quadrature on 1/A from -1 to 1.
                double mProduct, vProduct;
                Product.GetMeanAndVariance(out mProduct, out vProduct);
                double z = 0, sumA = 0, sumA2 = 0;
                for (int i = 0; i <= QuadratureNodeCount; i++)
                {
                    double a = (2.0 * i) / QuadratureNodeCount - 1;
                    double logfA = Gaussian.GetLogProb(mProduct, a * mB, vProduct + a * a * vB) + Gaussian.GetLogProb(a, mA, vA);
                    double fA = Math.Exp(logfA);
                    z += fA;
                    sumA += a * fA;
                    sumA2 += a * a * fA;

                    double invA = a;
                    a = 1.0 / invA;
                    double logfInvA = Gaussian.GetLogProb(mProduct * invA, mB, vProduct * invA * invA + vB) + Gaussian.GetLogProb(a, mA, vA) -
                                      Math.Log(Math.Abs(invA + Double.Epsilon));
                    double fInvA = Math.Exp(logfInvA);
                    z += fInvA;
                    sumA += a * fInvA;
                    sumA2 += a * a * fInvA;
                }
                double mean = sumA / z;
                double variance = sumA2 / z - mean * mean;
                if (z == 0 || variance <= 0 || variance >= vA)
                {
                    return GaussianProductOp_Slow.AAverageConditional(Product, A, B);
                    //throw new Exception("quadrature failed");
                }
                Gaussian result = new Gaussian();
                result.SetMeanAndVariance(mean, variance);
                result.SetToRatio(result, A, ForceProper);
                return result;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="AAverageConditional(double, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gaussian AAverageConditional(double Product, [NoInit] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            return AAverageConditional(Gaussian.PointMass(Product), A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="BAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [NoInit] Gaussian B)
        {
            return AAverageConditional(Product, B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="BAverageConditional(double, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gaussian BAverageConditional(double Product, [SkipIfUniform] Gaussian A, [NoInit] Gaussian B)
        {
            return AAverageConditional(Product, B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogAverageFactor([SkipIfUniform] Gaussian Product, Gaussian A, Gaussian B, Gaussian to_product)
        {
            if (A.IsPointMass)
                return LogAverageFactor(Product, B, A.Point, to_product);
            if (B.IsPointMass)
                return LogAverageFactor(Product, A, B.Point, to_product);
            if (Product.IsUniform())
                return 0.0;
            double mA, vA;
            A.GetMeanAndVariance(out mA, out vA);
            double mB, vB;
            B.GetMeanAndVariance(out mB, out vB);
            double mProduct, vProduct;
            Product.GetMeanAndVariance(out mProduct, out vProduct);
            // algorithm: quadrature on A from -1 to 1, plus quadrature on 1/A from -1 to 1.
            double z = 0;
            for (int i = 0; i <= GaussianProductOp.QuadratureNodeCount; i++)
            {
                double a = (2.0 * i) / GaussianProductOp.QuadratureNodeCount - 1;
                double logfA = Gaussian.GetLogProb(mProduct, a * mB, vProduct + a * a * vB) + Gaussian.GetLogProb(a, mA, vA);
                double fA = Math.Exp(logfA);
                z += fA;

                double invA = a;
                a = 1.0 / invA;
                double logfInvA = Gaussian.GetLogProb(mProduct * invA, mB, vProduct * invA * invA + vB) + Gaussian.GetLogProb(a, mA, vA) - Math.Log(Math.Abs(invA + Double.Epsilon));
                double fInvA = Math.Exp(logfInvA);
                z += fInvA;
            }
            double inc = 2.0 / GaussianProductOp.QuadratureNodeCount;
            return Math.Log(z * inc);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, Gaussian A, Gaussian B, Gaussian to_product)
        {
            return LogAverageFactor(Product, A, B, to_product) - to_product.GetLogAverageOf(Product);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double), Default = true)]
    //[FactorMethod(new string[] { "A", "Product", "B" }, typeof(Factor), "Ratio", typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Mature)]
    public class GaussianProductOp_Slow : GaussianProductOpEvidenceBase
    {
        public static int QuadratureNodeCount = 20000;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="ProductAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gaussian ProductAverageConditional([NoInit] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (A.IsPointMass)
                return GaussianProductOp.ProductAverageConditional(A.Point, B);
            if (B.IsPointMass)
                return GaussianProductOp.ProductAverageConditional(A, B.Point);
            if (Product.IsUniform() || Product.Precision < 1e-100 || A.IsUniform() || B.IsUniform())
                return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
            double mA, vA;
            A.GetMeanAndVariance(out mA, out vA);
            double mB, vB;
            B.GetMeanAndVariance(out mB, out vB);
            double mProduct, vProduct;
            Product.GetMeanAndVariance(out mProduct, out vProduct);
            bool oldMethod = false;
            if (oldMethod)
            {
                // algorithm: quadrature on A from -1 to 1, plus quadrature on 1/A from -1 to 1.
                double z = 0, sumX = 0, sumX2 = 0;
                for (int i = 0; i <= QuadratureNodeCount; i++)
                {
                    double a = (2.0 * i) / QuadratureNodeCount - 1;
                    double logfA = Gaussian.GetLogProb(mProduct, a * mB, vProduct + a * a * vB) + Gaussian.GetLogProb(a, mA, vA);
                    double fA = Math.Exp(logfA);

                    z += fA;
                    double b = (mB * vProduct + a * mProduct * vB) / (vProduct + a * a * vB);
                    double b2 = b * b + (vProduct * vB) / (vProduct + a * a * vB);
                    double x = a * b;
                    double x2 = a * a * b2;
                    sumX += x * fA;
                    sumX2 += x2 * fA;

                    double invA = a;
                    a = 1.0 / invA;
                    double logfInvA = Gaussian.GetLogProb(mProduct * invA, mB, vProduct * invA * invA + vB) + Gaussian.GetLogProb(a, mA, vA) - Math.Log(Math.Abs(invA + Double.Epsilon));
                    double fInvA = Math.Exp(logfInvA);
                    z += fInvA;
                    b = (mB * vProduct + a * mProduct * vB) / (vProduct + a * a * vB);
                    b2 = b * b + (vProduct * vB) / (vProduct + a * a * vB);
                    x = a * b;
                    x2 = a * a * b2;
                    sumX += x * fInvA;
                    sumX2 += x2 * fInvA;
                }
                double mean = sumX / z;
                double var = sumX2 / z - mean * mean;
                Gaussian result = Gaussian.FromMeanAndVariance(mean, var);
                result.SetToRatio(result, Product, GaussianProductOp.ForceProper);
                return result;
            }
            else
            {
                double pA = A.Precision;
                double a0, amin, amax;
                GetIntegrationBoundsForA(mProduct, vProduct, mA, pA, mB, vB, out a0, out amin, out amax);
                if (amin == a0 || amax == a0)
                    return AAverageConditional(Product, Gaussian.PointMass(a0), B);
                int n = QuadratureNodeCount;
                double inc = (amax - amin) / (n - 1);
                if (vProduct < 1)
                {
                    // Compute the message directly
                    double Z = 0;
                    double sum1 = 0;
                    double sum2 = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double a = amin + i * inc;
                        double logf = LogLikelihoodRatio(a, a0, mProduct, vProduct, mA, pA, mB, vB);
                        double v = vProduct + a * a * vB;
                        double diff = mProduct - a * mB;
                        double diffv = diff / v;
                        double diffv2 = diffv * diffv;
                        double dlogf = -diffv;
                        double ddlogf = diffv2 -1/v;
                        if ((i == 0 || i == n - 1) && (logf > -49))
                            throw new Exception("invalid integration bounds");
                        double f = Math.Exp(logf);
                        Z += f;
                        sum1 += dlogf * f;
                        sum2 += ddlogf * f;
                    }
                    if (double.IsPositiveInfinity(Z))
                    {
                        // this can happen if the likelihood is extremely sharp
                        //throw new Exception("overflow");
                        return ProductAverageConditional(Product, Gaussian.PointMass(a0), B);
                    }
                    double alpha = sum1 / Z;
                    double beta = alpha*alpha - sum2 / Z;
                    return GaussianOp.GaussianFromAlphaBeta(Product, alpha, beta, GaussianProductOp.ForceProper);
                }
                else
                {
                    // Compute the marginal and then divide
                    double rmin = Math.Sign(amin) * Math.Pow(Math.Abs(amin), 1.0 / 3);
                    double rmax = Math.Sign(amax) * Math.Pow(Math.Abs(amax), 1.0 / 3);
                    double rinc = (rmax - rmin) / (n - 1);
                    bool useCube = 1000 * vB > vProduct;
                    MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
                    for (int i = 0; i < n; i++)
                    {
                        double a, r = default;
                        if (useCube)
                        {
                            r = rmin + i * rinc;
                            a = Math.Pow(r, 3);
                        }
                        else
                        {
                            a = amin + i * inc;
                        }
                        double logfA = LogLikelihoodRatio(a, a0, mProduct, vProduct, mA, pA, mB, vB);
                        double fA = Math.Exp(logfA);
                        if (useCube)
                        {
                            fA *= 3 * r * r;
                        }
                        double avB = a * vB;
                        double v = vProduct + a * avB;
                        double mX = a * (mProduct * avB + vProduct * mB) / v;
                        double vX = a * avB * vProduct / v;
                        mva.Add(mX, vX, fA);
                    }
                    double mean = mva.Mean;
                    double variance = mva.Variance;
                    if (variance <= 0)
                        throw new Exception("quadrature failed");
                    Gaussian result = new Gaussian();
                    result.SetMeanAndVariance(mean, variance);
                    result.SetToRatio(result, Product, GaussianProductOp.ForceProper);
                    return result;
                }
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, [NoInit] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (B.IsPointMass)
                return GaussianProductOp.AAverageConditional(Product, B.Point);
            if (Product.IsUniform() || B.IsUniform() || Product.Precision < 1e-100)
                return Gaussian.Uniform();
            if (A.IsPointMass)
            {
                return GaussianProductOp.AAverageConditional(Product, A, B);
            }
            else
            {
                double mProduct, vProduct;
                Product.GetMeanAndVariance(out mProduct, out vProduct);
                double mA, vA;
                A.GetMeanAndVariance(out mA, out vA);
                double mB, vB;
                B.GetMeanAndVariance(out mB, out vB);
                double pA = A.Precision;
                double a0, amin, amax;
                GetIntegrationBoundsForA(mProduct, vProduct, mA, pA, mB, vB, out a0, out amin, out amax);
                if (amin == a0 || amax == a0)
                    return AAverageConditional(Product, Gaussian.PointMass(a0), B);
                int n = QuadratureNodeCount;
                double inc = (amax - amin) / (n - 1);
                if (vA < 1)
                {
                    // Compute the message directly
                    // alpha = dlogZ/dmA = (1/Z) int f'(a) p(a) da = (1/Z) int dlogf(a) f(a) p(a) da
                    // beta = -dalpha/dmA = alpha^2 - (1/Z) int f''(a) p(a) da
                    //                    = alpha^2 - (1/Z) int (dlogf(a)^2 + ddlogf(a)) f(a) p(a) da
                    // if p(a) approaches a point mass, then alpha -> dlogf(a), beta -> -ddlogf(a)
                    // and these will be the derivatives of the message.
                    double Z = 0;
                    double sum1 = 0;
                    double sum2 = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double a = amin + i * inc;
                        double logf = LogLikelihoodRatio(a, a0, mProduct, vProduct, mA, pA, mB, vB);
                        double dlogf, ddlogf;
                        GaussianProductOp.ADerivatives(a, mB, vB, Product, out dlogf, out ddlogf);
                        if ((i == 0 || i == n - 1) && (logf > -49))
                            throw new Exception("invalid integration bounds");
                        double f = Math.Exp(logf);
                        if (double.IsPositiveInfinity(f))
                        {
                            // this can happen if the likelihood is extremely sharp
                            //throw new Exception("overflow");
                            return AAverageConditional(Product, Gaussian.PointMass(a0), B);
                        }
                        Z += f;
                        sum1 += dlogf * f;
                        sum2 += (dlogf*dlogf + ddlogf) * f;
                    }
                    double alpha = sum1 / Z;
                    double beta = alpha * alpha - sum2 / Z;
                    return GaussianOp.GaussianFromAlphaBeta(A, alpha, beta, GaussianProductOp.ForceProper);
                }
                else
                {
                    // Compute the marginal and then divide
                    MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
                    for (int i = 0; i < n; i++)
                    {
                        double a = amin + i * inc;
                        double logfA = LogLikelihoodRatio(a, a0, mProduct, vProduct, mA, pA, mB, vB);
                        double fA = Math.Exp(logfA);
                        mva.Add(a, fA);
                    }
                    double mean = mva.Mean;
                    double variance = mva.Variance;
                    if (variance <= 0)
                        throw new Exception("quadrature failed");
                    Gaussian result = new Gaussian();
                    result.SetMeanAndVariance(mean, variance);
                    result.SetToRatio(result, A, GaussianProductOp.ForceProper);
                    return result;
                }
            }
        }

        public static void GetIntegrationBoundsForA(double mProduct, double vProduct, double mA, double pA, 
            double mB, double vB, out double amode, out double amin, out double amax)
        {
            if (double.IsInfinity(vProduct)) throw new ArgumentException("vProduct is infinity");
            if (double.IsInfinity(vB)) throw new ArgumentException("vB is infinity");
            // this code works even if vA = infinity
            double mpA = mA*pA;
            double vB2 = vB*vB;
            double vProduct2 = vProduct*vProduct;
            // coefficients of polynomial for derivative
            double[] coeffs = { -pA*vB2, mpA*vB2, -pA*2*vProduct*vB-vB2, mpA*2*vProduct*vB -vB*mB*mProduct, 
                                 -pA*vProduct2+vB*mProduct*mProduct-vB*vProduct-vProduct*mB*mB, mpA*vProduct2+vProduct*mB*mProduct };
            //double[] coeffs = { -pA*vB2, mpA*vB2, -pA*2*vProduct*vB, mpA*2*vProduct*vB, 
            //                     -pA*vProduct2, mpA*vProduct2 };
            //Console.WriteLine(StringUtil.CollectionToString(coeffs, " "));
            List<double> stationaryPoints;
            GaussianOp_Slow.GetRealRoots(coeffs, out stationaryPoints);
            // coefficients of polynomial for 2nd derivative
            double[] coeffs2 = new double[7];
            for (int i = 0; i < coeffs2.Length; i++)
            {
                double c = 0;
                if (i >= 2)
                {
                    c += vProduct * coeffs[i - 2] * (5 - (i - 2));
                }
                if (i <= 5)
                {
                    c += vB * coeffs[i] * (5 - i - 4);
                }
                coeffs2[i] = c;
            }
            //Console.WriteLine(StringUtil.CollectionToString(coeffs2, " "));
            List<double> inflectionPoints;
            GaussianOp_Slow.GetRealRoots(coeffs2, out inflectionPoints);
            double like(double a) => LogLikelihood(a, mProduct, vProduct, mA, pA, mB, vB);
            var stationaryValues = stationaryPoints.ConvertAll(a => like(a));
            double max = MMath.Max(stationaryValues);
            double a0 = stationaryPoints[stationaryValues.IndexOf(max)];
            amode = a0;
            double func(double a)
            {
                return LogLikelihoodRatio(a, a0, mProduct, vProduct, mA, pA, mB, vB) + 50;
            }
            double deriv(double a)
            {
                if (double.IsInfinity(a))
                    return -a;
                double v = vProduct + vB * a * a;
                double diffv = (mProduct - a * mB) / v;
                double diffa = a - mA;
                return a * vB * (diffv * diffv - 1 / v) + mB * diffv - diffa * pA;
            }
            // find where the likelihood matches the bound value
            List<double> zeroes = GaussianOp_Slow.FindZeroes(func, deriv, stationaryPoints, inflectionPoints);
            amin = MMath.Min(zeroes);
            amax = MMath.Max(zeroes);
            Assert.IsTrue(amin <= amax);
            //Console.WriteLine("amin = {0} amode = {1} amax = {2}", amin, amode, amax);
        }

        internal static double LogLikelihood(double a, double mProduct, double vProduct, double mA, double pA, double mB, double vB)
        {
            if (double.IsInfinity(a))
                return double.NegativeInfinity;
            double v = vProduct + vB * a * a;
            double diff = mProduct - a * mB;
            double diffa = a - mA;
            return -0.5 * (Math.Log(v) + diff * diff / v + diffa * diffa * pA);
        }

        internal static double LogLikelihoodRatio(double a, double a0, double mProduct, double vProduct, double mA, double pA, double mB, double vB)
        {
            if (double.IsInfinity(a))
                return double.NegativeInfinity;
            double v = vProduct + vB * a * a;
            double diff = mProduct - a * mB;
            double diffa = a - mA;
            double v0 = vProduct + vB * a0 * a0;
            double diff0 = mProduct - a0 * mB;
            double diffa0 = a0 - mA;
            return -0.5 * (Math.Log(v / v0) + diff * diff / v + diffa * diffa * pA) + 0.5 * (diff0 * diff0 / v0 + diffa0 * diffa0 * pA);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="AAverageConditional(double, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gaussian AAverageConditional(double Product, [NoInit] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            return AAverageConditional(Gaussian.PointMass(Product), A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="BAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [NoInit] Gaussian B)
        {
            return AAverageConditional(Product, B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="BAverageConditional(double, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Experimental)]
        public static Gaussian BAverageConditional(double Product, [SkipIfUniform] Gaussian A, [NoInit] Gaussian B)
        {
            return AAverageConditional(Product, B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogAverageFactor([SkipIfUniform] Gaussian Product, [Proper] Gaussian A, [Proper] Gaussian B, Gaussian to_product)
        {
            if (A.IsPointMass)
                return LogAverageFactor(Product, B, A.Point, to_product);
            if (B.IsPointMass)
                return LogAverageFactor(Product, A, B.Point, to_product);
            if (Product.IsUniform())
                return 0.0;
            double mA, vA;
            A.GetMeanAndVariance(out mA, out vA);
            double mB, vB;
            B.GetMeanAndVariance(out mB, out vB);
            double mProduct, vProduct;
            Product.GetMeanAndVariance(out mProduct, out vProduct);
            double pA = A.Precision;
            double a0, amin, amax;
            GaussianProductOp_Slow.GetIntegrationBoundsForA(mProduct, vProduct, mA, pA, mB, vB, out a0, out amin, out amax);
            if (amin == a0 || amax == a0)
                return LogAverageFactor(Product, B, a0, to_product);
            int n = GaussianProductOp_Slow.QuadratureNodeCount;
            double inc = (amax - amin) / (n - 1);
            double logZ = GaussianProductOp_Slow.LogLikelihood(a0, mProduct, vProduct, mA, pA, mB, vB);
            logZ += 0.5 * Math.Log(pA) - 2 * MMath.LnSqrt2PI;
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                double a = amin + i * inc;
                double logfA = GaussianProductOp_Slow.LogLikelihoodRatio(a, a0, mProduct, vProduct, mA, pA, mB, vB);
                double fA = Math.Exp(logfA);
                sum += fA;
            }
            return logZ + Math.Log(sum * inc);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [Proper] Gaussian A, [Proper] Gaussian B, Gaussian to_product)
        {
            return LogAverageFactor(Product, A, B, to_product) - to_product.GetLogAverageOf(Product);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/doc/*'/>
    /// <remarks>
    /// This class allows EP to process the product factor as if running VMP, as required by Stern's algorithm.
    /// The algorithm comes from "Matchbox: Large Scale Online Bayesian Recommendations" by David Stern, Ralf Herbrich, and Thore Graepel, WWW 2009.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product_SHG09", typeof(double), typeof(double))]
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Preview)]
    public static class GaussianProductOp_SHG09
    {
        public static Gaussian ProductAverageConditional2(
            Gaussian product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, [NoInit] Gaussian to_A, [NoInit] Gaussian to_B)
        {
            // this version divides out the message from product, so the marginal for Product is correct and the factor is composable.
            return GaussianProductVmpOp.ProductAverageLogarithm(A * to_A, B * to_B) / product;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="ProductAverageConditional(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, [NoInit] Gaussian to_A, [NoInit] Gaussian to_B)
        {
            // note we are not dividing out the message from Product.
            // this means that the marginal for Product will not be correct, and the factor is not composable.
            return GaussianProductVmpOp.ProductAverageLogarithm(A * to_A, B * to_B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, [Proper /*, Fresh*/] Gaussian B, [NoInit] Gaussian to_B)
        {
            return GaussianProductVmpOp.AAverageLogarithm(Product, B * to_B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="BAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, [Proper /*, Fresh*/] Gaussian A, [NoInit] Gaussian to_A)
        {
            //return BAverageConditional(Product, A.GetMean());
            return AAverageConditional(Product, A, to_A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(
            [SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, Gaussian to_A, Gaussian to_B, Gaussian to_product)
        {
            // The SHG paper did not define how to compute evidence.
            // The formula below comes from matching the VMP evidence for a model with a single product factor.
            Gaussian qA = A * to_A;
            Gaussian qB = B * to_B;
            Gaussian qProduct = to_product;
            double aRatio = A.GetLogAverageOf(to_A) - qA.GetAverageLog(to_A);
            double bRatio = B.GetLogAverageOf(to_B) - qB.GetAverageLog(to_B);
            double productRatio = qProduct.GetAverageLog(Product) - to_product.GetLogAverageOf(Product);
            return aRatio + bRatio + productRatio;
        }

#if true
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="ProductAverageConditional(double, Gaussian, Gaussian)"]/*'/>
        public static Gaussian ProductAverageConditional(double A, [SkipIfUniform] Gaussian B, [NoInit] Gaussian to_B)
        {
            return GaussianProductVmpOp.ProductAverageLogarithm(A, B * to_B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="ProductAverageConditional(Gaussian, double, Gaussian)"]/*'/>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, double B, [NoInit] Gaussian to_A)
        {
            return GaussianProductVmpOp.ProductAverageLogarithm(A * to_A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="AAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, double B)
        {
            return GaussianProductVmpOp.AAverageLogarithm(Product, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="BAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, double A)
        {
            return AAverageConditional(Product, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, double, Gaussian, Gaussian)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, double B, Gaussian to_A, Gaussian to_product)
        {
            // The SHG paper did not define how to compute evidence.
            // The formula below comes from matching the VMP evidence for a model with a single product factor.
            Gaussian qA = A * to_A;
            Gaussian qProduct = to_product;
            double aRatio = A.GetLogAverageOf(to_A) - qA.GetAverageLog(to_A);
            double productRatio = qProduct.GetAverageLog(Product) - to_product.GetLogAverageOf(Product);
            return aRatio + productRatio;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="LogEvidenceRatio(Gaussian, double, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, double A, [SkipIfUniform] Gaussian B, Gaussian to_B, Gaussian to_product)
        {
            // The SHG paper did not define how to compute evidence.
            // The formula below comes from matching the VMP evidence for a model with a single product factor.
            Gaussian qB = B * to_B;
            Gaussian qProduct = to_product;
            double bRatio = B.GetLogAverageOf(to_B) - qB.GetAverageLog(to_B);
            double productRatio = qProduct.GetAverageLog(Product) - to_product.GetLogAverageOf(Product);
            return bRatio + productRatio;
        }
#else
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="ProductAverageConditional(double, Gaussian)"]/*'/>
        public static Gaussian ProductAverageConditional(double A, [SkipIfUniform] Gaussian B)
        {
            return GaussianProductOp.ProductAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="ProductAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, double B)
        {
            return GaussianProductOp.ProductAverageConditional(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="AAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, double B)
        {
            return GaussianProductOp.AAverageConditional(Product, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="BAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, double A)
        {
            return GaussianProductOp.BAverageConditional(Product, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, double)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, double B)
        {
            return GaussianProductEvidenceOp.LogEvidenceRatio(Product, A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_SHG09"]/message_doc[@name="LogEvidenceRatio(Gaussian, double, Gaussian)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, double A, [SkipIfUniform] Gaussian B)
        {
            return GaussianProductEvidenceOp.LogEvidenceRatio(Product, A, B);
        }
#endif
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_LaplaceProp"]/doc/*'/>
    /// <remarks>
    /// This class allows EP to process the product factor using Laplace Propagation with other variables marginalized out.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    //[FactorMethod(new string[] { "A", "Product", "B" }, typeof(Factor), "Ratio", typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Experimental)]
    public class GaussianProductOp_LaplaceProp : GaussianProductOpEvidenceBase
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_LaplaceProp"]/message_doc[@name="ProductAverageConditional(Gaussian, Gaussian)"]/*'/>
        public static Gaussian ProductAverageConditional(Gaussian A, Gaussian B)
        {
            return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
        }

        public static Gaussian ProductAverageConditional2(Gaussian Product, Gaussian A, Gaussian B, Gaussian to_A)
        {
            if (Product.IsUniform())
                return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
            double mx, vx;
            Product.GetMeanAndVariance(out mx, out vx);
            double ma, va;
            A.GetMeanAndVariance(out ma, out va);
            Gaussian Apost = A * to_A;
            double ahat = Apost.GetMean();
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            double denom = 1 / (vx + ahat * ahat * vb);
            double diff = mx - ahat * mb;
            double dlogz = -diff * denom;
            double dd = ahat * vb;
            double q = denom * (mb + 2 * dd * diff * denom);
            double ddd = vb;
            double n = diff * diff;
            double dn = -diff * mb;
            double ddn = mb * mb;
            double dda = denom * (-(ddd + ddn) + denom * (2 * dd * (dd + 2 * dn) + n * ddd - denom * 4 * n * dd * dd));
            double da = va * q / (1 - va * dda);
            double ddlogz = -denom + q * da;
            return GaussianOp.GaussianFromAlphaBeta(Product, dlogz, -ddlogz, true);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_LaplaceProp"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageConditional(Gaussian Product, Gaussian A, Gaussian B, Gaussian to_A)
        {
            // The factor is N(mx; ahat * mb, vx + ahat * ahat * vb)
            Gaussian Apost = A * to_A;
            double mx, vx;
            Product.GetMeanAndVariance(out mx, out vx);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            double ahat = Apost.GetMean();
            double denom = 1 / (vx + ahat * ahat * vb);
            double diff = mx - ahat * mb;
            double dd = ahat * vb;
            double ddd = vb;
            double n = diff * diff;
            double dn = -diff * mb;
            double ddn = mb * mb;
            double dlogf = denom * (-(dd + dn) + denom * n * dd);
            double ddlogf = denom * (-(ddd + ddn) + denom * (2 * dd * (dd + 2 * dn) + n * ddd - denom * 4 * n * dd * dd));
            double r = Math.Max(0, -ddlogf);
            return Gaussian.FromNatural(r * ahat + dlogf, r);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_LaplaceProp"]/message_doc[@name="BAverageConditional(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional(Gaussian Product, Gaussian A, Gaussian B, Gaussian to_B)
        {
            return AAverageConditional(Product, B, A, to_B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_LaplaceProp"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogAverageFactor([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, Gaussian to_A)
        {
            double mx, vx;
            Product.GetMeanAndVariance(out mx, out vx);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            double ma, va;
            A.GetMeanAndVariance(out ma, out va);
            Gaussian Apost = A * to_A;
            double ahat = Apost.GetMean();
            return Gaussian.GetLogProb(mx, ahat * mb, vx + ahat * ahat * vb) + A.GetLogProb(ahat) - Apost.GetLogProb(ahat);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_LaplaceProp"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, Gaussian to_A, Gaussian to_product)
        {
            return LogAverageFactor(Product, A, B, to_A) - to_product.GetLogAverageOf(Product);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_Laplace"]/doc/*'/>
    /// <remarks>
    /// This class allows EP to process the product factor using Laplace's method.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Buffers("ahat")]  // mode of A
    [Quality(QualityBand.Experimental)]
    public class GaussianProductOp_Laplace : GaussianProductOpEvidenceBase
    {
        public static double AhatInit(Gaussian a)
        {
            return a.GetMean();
        }

        public static double Ahat(Gaussian Product, Gaussian A, Gaussian B, double ahat)
        {
            if (B.GetMean() == 0.0) return A.GetMean();
            if (ahat == 0)
                ahat = -1;
            for (int iter = 0; iter < 100; iter++)
            {
                double olda = ahat;
                double bhat = (B.MeanTimesPrecision + ahat * Product.MeanTimesPrecision) / (B.Precision + ahat * ahat * Product.Precision);
                ahat = (A.MeanTimesPrecision + bhat * Product.MeanTimesPrecision) / (A.Precision + bhat * bhat * Product.Precision);
                //Console.WriteLine("{0} ahat = {1}", iter, ahat);
                if (MMath.AbsDiff(olda, ahat, 1e-10) < 1e-10)
                    break;
            }
            return ahat;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_Laplace"]/message_doc[@name="ProductAverageConditional(Gaussian, Gaussian, Gaussian, double)"]/*'/>
        public static Gaussian ProductAverageConditional([NoInit] Gaussian Product, Gaussian A, Gaussian B, [Fresh] double ahat)
        {
            if (Product.IsUniform())
            {
                double ma, va;
                A.GetMeanAndVariance(out ma, out va);
                double mb, vb;
                B.GetMeanAndVariance(out mb, out vb);
                // this is the limiting value of below - it excludes va*vb
                return Gaussian.FromMeanAndVariance(ma * mb, ma * ma * vb + mb * mb * va);
            }
            double bhat = (B.MeanTimesPrecision + ahat * Product.MeanTimesPrecision) / (B.Precision + ahat * ahat * Product.Precision);
            double ab = ahat * bhat;
            PositiveDefiniteMatrix hessian = new PositiveDefiniteMatrix(2, 2);
            hessian[0, 0] = A.Precision + Product.Precision * bhat * bhat;
            hessian[0, 1] = Product.Precision * 2 * ab - Product.MeanTimesPrecision;
            hessian[1, 0] = hessian[0, 1];
            hessian[1, 1] = B.Precision + Product.Precision * ahat * ahat;
            PositiveDefiniteMatrix inverseH = new PositiveDefiniteMatrix(2, 2);
            inverseH.SetToInverse(hessian);
            Vector gderiv = Vector.FromArray(new double[] { bhat, ahat });
            Matrix gderiv2 = new Matrix(new double[,] { { 0, 1 }, { 1, 0 } });
            Matrix deriv3 = new Matrix(new double[,] { 
                { 0, bhat }, 
                { bhat, ahat },
                { bhat, ahat },
                { ahat, 0 }
            });
            deriv3.Scale(-2 * Product.Precision);
            Vector dx = Vector.Zero(2);
            dx.SetToProduct(inverseH, gderiv);
            // this is actually -dH
            Matrix dH = new Matrix(2, 2);
            for (int i = 0; i < 4; i++)
            {
                dH[i] = deriv3[i, 0] * dx[0] + deriv3[i, 1] * dx[1];
            }
            dH.SetToSum(dH, gderiv2);
            double mpost = ab + 0.5 * Matrix.TraceOfProduct(inverseH, dH);
            Matrix deriv4 = new Matrix(new double[,] {
                { 0, 0 },
                { 0, 1 },
                { 0, 1 },
                { 1, 0 },
                { 0, 1 },
                { 1, 0 },
                { 1, 0 },
                { 0, 0 }
            });
            deriv4.Scale(-2 * Product.Precision);
            Matrix deriv4Product = new Matrix(2, 2);
            for (int i = 0; i < 4; i++)
            {
                double c1 = deriv4[i, 0] * dx[0] + deriv4[i, 1] * dx[1];
                int i2 = i + 4;
                double c2 = deriv4[i2, 0] * dx[0] + deriv4[i2, 1] * dx[1];
                deriv4Product[i] = c1 * dx[0] + c2 * dx[1];
            }
            // H^(-1) dH
            Matrix invHdH = inverseH * dH;
            Matrix deriv3dx2 = deriv3 * inverseH * (gderiv2 + dH);
            Matrix deriv3Product = new Matrix(2, 2);
            for (int i = 0; i < 4; i++)
            {
                deriv3Product[i] = deriv3dx2[i, 0] * dx[0] + deriv3dx2[i, 1] * dx[1];
            }
            double vpost = gderiv.Inner(dx) 
                + 0.5 * Matrix.TraceOfProduct(invHdH, invHdH)
                + 0.5 * Matrix.TraceOfProduct(inverseH, deriv4Product)
                + 0.5 * Matrix.TraceOfProduct(inverseH, deriv3Product);
            if (vpost < 0)
                vpost = double.NaN;
            Gaussian result = Gaussian.FromMeanAndVariance(mpost, vpost);
            result.SetToRatio(result, Product, GaussianProductOp.ForceProper);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_Laplace"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, Gaussian, double)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, [NoInit] Gaussian A, Gaussian B, [Fresh] double ahat)
        {
            if (Product.IsUniform()) return Product;
            Gaussian result = GetMoments(0, Product, A, B, ahat);
            result.SetToRatio(result, A, GaussianProductOp.ForceProper);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_Laplace"]/message_doc[@name="BAverageConditional(Gaussian, Gaussian, Gaussian, double)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, [NoInit] Gaussian B, [Fresh] double ahat)
        {
            if (Product.IsUniform()) return Product;
            Gaussian result = GetMoments(1, Product, A, B, ahat);
            result.SetToRatio(result, B, GaussianProductOp.ForceProper);
            return result;
        }

        public static Gaussian GetMoments(int index, Gaussian Product, Gaussian A, Gaussian B, [Fresh] double ahat)
        {
            double bhat = (B.MeanTimesPrecision + ahat * Product.MeanTimesPrecision) / (B.Precision + ahat * ahat * Product.Precision);
            double ab = ahat * bhat;
            PositiveDefiniteMatrix hessian = new PositiveDefiniteMatrix(2, 2);
            hessian[0, 0] = A.Precision + Product.Precision * bhat*bhat;
            hessian[0, 1] = Product.Precision * 2*ab - Product.MeanTimesPrecision;
            hessian[1, 0] = hessian[0, 1];
            hessian[1, 1] = B.Precision + Product.Precision * ahat*ahat;
            PositiveDefiniteMatrix inverseH = new PositiveDefiniteMatrix(2,2);
            inverseH.SetToInverse(hessian);
            Matrix deriv3 = new Matrix(new double[,] { 
                { 0, bhat }, 
                { bhat, ahat },
                { bhat, ahat },
                { ahat, 0 }
            });
            deriv3.Scale(-2 * Product.Precision);
            // this is actually -dH
            Matrix dH = new Matrix(2,2);
            for (int i = 0; i < 4; i++)
            {
                dH[i] = deriv3[i, 0] * inverseH[index, 0] + deriv3[i, 1] * inverseH[index, 1];
            }
            double mpost = (index == 0 ? ahat : bhat) + 0.5 * Matrix.TraceOfProduct(inverseH, dH);
            Matrix deriv4 = new Matrix(new double[,] {
                { 0, 0 },
                { 0, 1 },
                { 0, 1 },
                { 1, 0 },
                { 0, 1 },
                { 1, 0 },
                { 1, 0 },
                { 0, 0 }
            });
            deriv4.Scale(-2 * Product.Precision);
            Matrix deriv4Product = new Matrix(2, 2);
            for (int i = 0; i < 4; i++)
            {
                double c1 = deriv4[i, 0] * inverseH[index, 0] + deriv4[i, 1] * inverseH[index, 1];
                int i2 = i+4;
                double c2 = deriv4[i2, 0] * inverseH[index, 0] + deriv4[i2, 1] * inverseH[index, 1];
                deriv4Product[i] = c1 * inverseH[index, 0] + c2 * inverseH[index, 1];
            }
            // H^(-1) dH
            Matrix invHdH = inverseH * dH;
            Matrix deriv3invHdH = deriv3 * invHdH;
            Matrix deriv3Product = new Matrix(2, 2);
            for (int i = 0; i < 4; i++)
            {
                deriv3Product[i] = deriv3invHdH[i, 0] * inverseH[index, 0] + deriv3invHdH[i, 1] * inverseH[index, 1];
            }
            double vpost = inverseH[index, index]
                + 0.5 * Matrix.TraceOfProduct(invHdH, invHdH)
                + 0.5 * Matrix.TraceOfProduct(inverseH, deriv4Product)
                + 0.5 * Matrix.TraceOfProduct(inverseH, deriv3Product);
            return Gaussian.FromMeanAndVariance(mpost, vpost);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_Laplace"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, Gaussian to_A)
        {
            double mx, vx;
            Product.GetMeanAndVariance(out mx, out vx);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            double ma, va;
            A.GetMeanAndVariance(out ma, out va);
            Gaussian Apost = A * to_A;
            double ahat = Apost.GetMean();
            return Gaussian.GetLogProb(mx, ahat * mb, vx + ahat * ahat * vb) + A.GetLogProb(ahat) - Apost.GetLogProb(ahat);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_Laplace"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            // TODO
            return 0.0;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_Laplace"]/doc/*'/>
    /// <remarks>
    /// This class allows EP to process the product factor using Laplace's method.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public class GaussianProductOp_Laplace2 : GaussianProductOpEvidenceBase
    {
        public static bool modified = true;
        public static bool offDiagonal = false;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_Laplace2"]/message_doc[@name="ProductAverageConditional(Gaussian, Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian ProductAverageConditional([NoInit] Gaussian Product, Gaussian A, Gaussian B, Gaussian to_A, Gaussian to_B)
        {
            if (Product.IsUniform())
            {
                double ma, va;
                A.GetMeanAndVariance(out ma, out va);
                double mb, vb;
                B.GetMeanAndVariance(out mb, out vb);
                // this is the limiting value of below - it excludes va*vb
                return Gaussian.FromMeanAndVariance(ma * mb, ma * ma * vb + mb * mb * va);
            }
            if (A.IsPointMass)
                return GaussianProductOp.ProductAverageConditional(A.Point, B);
            if (B.IsPointMass)
                return GaussianProductOp.ProductAverageConditional(A, B.Point);
            Gaussian Apost = A * to_A;
            Gaussian Bpost = B * to_B;
            double ahat = Apost.GetMean();
            double bhat = Bpost.GetMean();
            double gx = -(Product.MeanTimesPrecision - ahat * bhat * Product.Precision);
            double gxx = -Product.Precision;
            double gax = bhat * Product.Precision;
            double gbx = ahat * Product.Precision;
            double gaa = -bhat * bhat * Product.Precision;
            double gbb = -ahat * ahat * Product.Precision;
            double gab = Product.MeanTimesPrecision - 2 * ahat * bhat * Product.Precision;
            if (modified)
            {
                double gaaa = 0;
                double gaaab = 0;
                double gaaaa = 0;
                double gaab = -2 * bhat * Product.Precision;
                double gaabb = -2 * Product.Precision;
                double gabb = -2 * ahat * Product.Precision;
                double gabbb = 0;
                double gbbb = 0;
                double gbbbb = 0;
                double adiff = A.Precision - gaa;
                double bdiff = B.Precision - gbb;
                double h = adiff * bdiff;
                double ha = -gaaa * bdiff - gabb * adiff;
                double hb = -gaab * bdiff - gbbb * adiff;
                double haa = -gaaaa * bdiff - gaabb * adiff + 2 * gaaa * gabb;
                double hab = -gaaab * bdiff - gabbb * adiff + gaaa * gbbb + gabb * gaab;
                double hbb = -gaabb * bdiff - gbbbb * adiff + 2 * gaab * gbbb;
                if (offDiagonal)
                {
                    h += -gab * gab;
                    ha += -2 * gab * gaab;
                    hb += -2 * gab * gabb;
                    haa += -2 * gaab * gaab - 2 * gab * gaaab;
                    hab += -2 * gabb * gaab - 2 * gab * gaabb;
                    hbb += -2 * gabb * gabb - 2 * gab * gabbb;
                }
                double logha = ha / h;
                double loghb = hb / h;
                double loghaa = haa / h - logha * logha;
                double loghab = hab / h - logha * loghb;
                double loghbb = hbb / h - loghb * loghb;
                gaa -= 0.5 * loghaa;
                gab -= 0.5 * loghab;
                gbb -= 0.5 * loghbb;
            }
            double cb = gab / (B.Precision - gbb);
            double dax = (gax + gbx * cb) / (A.Precision - (gaa + gab * cb));
            double dbx = gbx / (B.Precision - gbb) + cb * dax;
            double dlogz = gx;
            double ddlogz = gxx + gax * dax + gbx * dbx;
            return GaussianOp.GaussianFromAlphaBeta(Product, dlogz, -ddlogz, true);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_Laplace2"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, [NoInit] Gaussian A, Gaussian B, Gaussian to_A)
        {
            if (A.IsPointMass)
                return GaussianProductOp.AAverageConditional(Product, A, B);
            Gaussian Apost = A * to_A;
            double ahat = Apost.GetMean();
            double bhat = (ahat * Product.MeanTimesPrecision + B.MeanTimesPrecision) / (ahat * ahat * Product.Precision + B.Precision);
            double ga = (Product.MeanTimesPrecision - ahat * bhat * Product.Precision) * bhat;
            double gaa = -bhat * bhat * Product.Precision;
            double gbb = -ahat * ahat * Product.Precision;
            double gab = Product.MeanTimesPrecision - 2 * ahat * bhat * Product.Precision;
            if (modified)
            {
                double gaaa = 0;
                double gaaab = 0;
                double gaaaa = 0;
                double gaab = -2 * bhat * Product.Precision;
                double gaabb = -2 * Product.Precision;
                double gabb = -2 * ahat * Product.Precision;
                double gabbb = 0;
                double gbbb = 0;
                double gbbbb = 0;
                double adiff = A.Precision - gaa;
                double bdiff = B.Precision - gbb;
                double h = adiff * bdiff;
                double ha = -gaaa * bdiff - gabb * adiff;
                double hb = -gaab * bdiff - gbbb * adiff;
                double haa = -gaaaa * bdiff - gaabb * adiff + 2 * gaaa * gabb;
                double hab = -gaaab * bdiff - gabbb * adiff + gaaa * gbbb + gabb * gaab;
                double hbb = -gaabb * bdiff - gbbbb * adiff + 2 * gaab * gbbb;
                if (offDiagonal)
                {
                    // this can cause h to be negative
                    h += -gab * gab;
                    ha += -2 * gab * gaab;
                    hb += -2 * gab * gabb;
                    haa += -2 * gaab * gaab - 2 * gab * gaaab;
                    hab += -2 * gabb * gaab - 2 * gab * gaabb;
                    hbb += -2 * gabb * gabb - 2 * gab * gabbb;
                }
                double logha = ha / h;
                double loghb = hb / h;
                double loghaa = haa / h - logha * logha;
                double loghab = hab / h - logha * loghb;
                double loghbb = hbb / h - loghb * loghb;
                ga -= 0.5 * logha;
                gaa -= 0.5 * loghaa;
                gab -= 0.5 * loghab;
                gbb -= 0.5 * loghbb;
            }
            double cb = gab / (B.Precision - gbb);
            double ddlogf = gaa + cb * gab;
            double r = Math.Max(0, -ddlogf);
            return Gaussian.FromNatural(r * ahat + ga, r);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_Laplace2"]/message_doc[@name="BAverageConditional(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, [NoInit] Gaussian B, Gaussian to_B)
        {
            return AAverageConditional(Product, B, A, to_B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_Laplace2"]/message_doc[@name="LogAverageFactor(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static double LogAverageFactor([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, Gaussian to_A)
        {
            double mx, vx;
            Product.GetMeanAndVariance(out mx, out vx);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            double ma, va;
            A.GetMeanAndVariance(out ma, out va);
            Gaussian Apost = A * to_A;
            double ahat = Apost.GetMean();
            return Gaussian.GetLogProb(mx, ahat * mb, vx + ahat * ahat * vb) + A.GetLogProb(ahat) - Apost.GetLogProb(ahat);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp_Laplace2"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            // TODO
            return 0.0;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp6"]/doc/*'/>
    /// <remarks>
    /// This class allows EP to process the product factor using modified Laplace's method.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    internal static class GaussianProductOp6
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp6"]/message_doc[@name="ProductAverageConditional(Gaussian, Gaussian)"]/*'/>
        public static Gaussian ProductAverageConditional(Gaussian A, Gaussian B)
        {
            return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp6"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageConditional(Gaussian Product, Gaussian A, Gaussian B, Gaussian to_A)
        {
            Gaussian Apost = A * to_A;
            double mx, vx;
            Product.GetMeanAndVariance(out mx, out vx);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            double ma, va;
            bool usePost = false;
            if (usePost)
                Apost.GetMeanAndVariance(out ma, out va);
            else
                A.GetMeanAndVariance(out ma, out va);
            double ahat = ma;
            ahat = Apost.GetMean();
            double denom = vx + ahat * ahat * vb;
            double diff = mx - ahat * mb;
            double dd = 2 * ahat * vb;
            double ddd = 2 * vb;
            double n = diff * diff;
            double dn = -2 * diff * mb;
            double ddn = 2 * mb * mb;
            double dlogf = -0.5 * (dd + dn) / denom + 0.5 * n * dd / (denom * denom);
            double ddlogf = -0.5 * (ddd + ddn) / denom + 0.5 * (dd * dd + 2 * dd * dn + n * ddd) / (denom * denom) - n * dd * dd / (denom * denom * denom);
            if (usePost)
            {
                double m0, v0;
                A.GetMeanAndVariance(out m0, out v0);
                dlogf += (m0 - ahat) / v0;
                ddlogf += 1 / va - 1 / v0;
                // at a fixed point, dlogf2 = 0 so (ahat - m0)/v0 = dlogf, ahat = m0 + v0*dlogf
                // this is the same fixed point condition as Laplace
            }
            // Ef'/Ef =approx dlogf*f(ahat)/f(ahat) = dlogf
            double mnew = ma + va * dlogf;
            double vnew = va * (1 + va * (ddlogf - dlogf * dlogf));
            double rnew = Math.Max(1 / vnew, A.Precision);
            return Gaussian.FromNatural(rnew * mnew - A.MeanTimesPrecision, rnew - A.Precision);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp6"]/message_doc[@name="BAverageConditional(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional(Gaussian Product, Gaussian A, Gaussian B, Gaussian to_B)
        {
            return AAverageConditional(Product, B, A, to_B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp6"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            // TODO
            return 0.0;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp3"]/doc/*'/>
    /// <remarks>
    /// This class allows EP to process the product factor as a linear factor.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    [Buffers("weights")]
    internal static class GaussianProductOp3
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp3"]/message_doc[@name="Weights(double, Gaussian)"]/*'/>
        public static Vector Weights(double A, Gaussian B)
        {
            Vector weights = Vector.Zero(4);
            weights[1] = A;
            return weights;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp3"]/message_doc[@name="Weights(Gaussian, double)"]/*'/>
        public static Vector Weights(Gaussian A, double B)
        {
            Vector weights = Vector.Zero(4);
            weights[0] = B;
            return weights;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp3"]/message_doc[@name="Weights(Gaussian, Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Vector Weights(Gaussian A, Gaussian B, Gaussian to_A, Gaussian to_B)
        {
            if (A.IsPointMass)
                return Weights(A.Point, B);
            if (B.IsPointMass)
                return Weights(A, B.Point);
            A *= to_A;
            B *= to_B;
            double ma, va, mb, vb;
            A.GetMeanAndVariance(out ma, out va);
            B.GetMeanAndVariance(out mb, out vb);
            double ma2 = va + ma * ma;
            double mb2 = vb + mb * mb;
            Vector w = Vector.Zero(3);
            w[0] = ma2 * mb;
            w[1] = mb2 * ma;
            w[2] = ma * mb;
            PositiveDefiniteMatrix M = new PositiveDefiniteMatrix(3, 3);
            M[0, 0] = ma2;
            M[0, 1] = ma * mb;
            M[0, 2] = ma;
            M[1, 0] = ma * mb;
            M[1, 1] = mb2;
            M[1, 2] = mb;
            M[2, 0] = ma;
            M[2, 1] = mb;
            M[2, 2] = 1;
            w = w.PredivideBy(M);
            Vector weights = Vector.Zero(4);
            weights[0] = w[0];
            weights[1] = w[1];
            weights[2] = w[2];
            weights[3] = ma2 * mb2 - w[0] * ma2 * mb - w[1] * mb2 * ma - w[2] * ma * mb;
            if (weights[3] < 0)
                weights[3] = 0;
            if (false)
            {
                // debugging
                GaussianEstimator est = new GaussianEstimator();
                for (int i = 0; i < 10000; i++)
                {
                    double sa = A.Sample();
                    double sb = B.Sample();
                    double f = sa * sb;
                    double g = sa * weights[0] + sb * weights[1] + weights[2];
                    est.Add(f - g);
                }
                Console.WriteLine(weights);
                Console.WriteLine(est.GetDistribution(new Gaussian()));
            }
            return weights;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp3"]/message_doc[@name="ProductAverageConditional(Gaussian, Gaussian, Vector)"]/*'/>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, [Fresh] Vector weights)
        {
            // factor is product = N(w[0]*a + w[1]*b + w[2], w[3])
            double v = weights[3];
            Gaussian m = DoublePlusOp.SumAverageConditional(GaussianProductOp.ProductAverageConditional(weights[0], A),
                                                            GaussianProductOp.ProductAverageConditional(weights[1], B));
            m = DoublePlusOp.SumAverageConditional(m, weights[2]);
            return GaussianFromMeanAndVarianceOp.SampleAverageConditional(m, v);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp3"]/message_doc[@name="ProductAverageConditional(double, Gaussian, Vector)"]/*'/>
        public static Gaussian ProductAverageConditional(double A, [SkipIfUniform] Gaussian B, [Fresh] Vector weights)
        {
            return ProductAverageConditional(Gaussian.PointMass(A), B, weights);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp3"]/message_doc[@name="ProductAverageConditional(Gaussian, double, Vector)"]/*'/>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, double B, [Fresh] Vector weights)
        {
            return ProductAverageConditional(A, Gaussian.PointMass(B), weights);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp3"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, Vector)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian B, [Fresh] Vector weights)
        {
            // factor is product = N(w[0]*a + w[1]*b + w[2], w[3])
            double v = weights[3];
            Gaussian sum_B = GaussianFromMeanAndVarianceOp.MeanAverageConditional(Product, v);
            sum_B = DoublePlusOp.AAverageConditional(sum_B, weights[2]);
            Gaussian scale_B = DoublePlusOp.AAverageConditional(sum_B, GaussianProductOp.ProductAverageConditional(weights[1], B));
            return GaussianProductOp.AAverageConditional(scale_B, weights[0]);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp3"]/message_doc[@name="AAverageConditional(Gaussian, double, Vector)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, double B, [Fresh] Vector weights)
        {
            return AAverageConditional(Product, Gaussian.PointMass(B), weights);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp3"]/message_doc[@name="BAverageConditional(Gaussian, Gaussian, Vector)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [Fresh] Vector weights)
        {
            // factor is product = N(w[0]*a + w[1]*b + w[2], w[3])
            double v = weights[3];
            Gaussian sum_B = GaussianFromMeanAndVarianceOp.MeanAverageConditional(Product, v);
            sum_B = DoublePlusOp.AAverageConditional(sum_B, weights[2]);
            Gaussian scale_B = DoublePlusOp.AAverageConditional(sum_B, GaussianProductOp.ProductAverageConditional(weights[0], A));
            return GaussianProductOp.AAverageConditional(scale_B, weights[1]);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp3"]/message_doc[@name="BAverageConditional(Gaussian, double, Vector)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, double A, [Fresh] Vector weights)
        {
            return BAverageConditional(Product, Gaussian.PointMass(A), weights);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp3"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            return 0.0;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp4"]/doc/*'/>
    /// <remarks>
    /// This class allows EP to process the product factor using an approximation to the integral Z.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    internal static class GaussianProductOp4
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp4"]/message_doc[@name="ProductAverageConditional(Gaussian, Gaussian)"]/*'/>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp4"]/message_doc[@name="ProductAverageConditional(double, Gaussian)"]/*'/>
        public static Gaussian ProductAverageConditional(double A, [SkipIfUniform] Gaussian B)
        {
            return ProductAverageConditional(Gaussian.PointMass(A), B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp4"]/message_doc[@name="ProductAverageConditional(Gaussian, double)"]/*'/>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, double B)
        {
            return ProductAverageConditional(A, Gaussian.PointMass(B));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp4"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (A.IsPointMass)
                return GaussianProductOp.AAverageConditional(Product, A, B);
            double ma, va;
            A.GetMeanAndVariance(out ma, out va);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            double mx, vx;
            Product.GetMeanAndVariance(out mx, out vx);
            double diff = mx - ma * mb;
            double prec = 1 / (vx + va * vb + va * mb * mb + vb * ma * ma);
            //if (prec < 1e-14) return Gaussian.Uniform();
            double alpha = prec * (-vb * ma + prec * diff * diff * ma * vb + diff * mb);
            double beta = alpha * alpha + prec * (1 - diff * diff * prec) * (vb + mb * mb);
            //if (beta == 0) return Gaussian.Uniform();
            if (double.IsNaN(alpha) || double.IsNaN(beta))
                throw new InferRuntimeException($"alpha or beta is nan.  product={Product}, a={A}, b={B}");
            double r = beta / (A.Precision - beta);
            Gaussian result = new Gaussian();
            result.Precision = r * A.Precision;
            result.MeanTimesPrecision = r * (alpha + A.MeanTimesPrecision) + alpha;
            //Gaussian result = new Gaussian(ma + alpha/beta, 1/beta - va);
            if (double.IsNaN(result.Precision) || double.IsNaN(result.MeanTimesPrecision))
                throw new InferRuntimeException($"result is NaN.  product={Product}, a={A}, b={B}");
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp4"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, double)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, double B)
        {
            return AAverageConditional(Product, A, Gaussian.PointMass(B));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp4"]/message_doc[@name="BAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, Gaussian B)
        {
            return AAverageConditional(Product, B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp4"]/message_doc[@name="BAverageConditional(Gaussian, double, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, double A, Gaussian B)
        {
            return BAverageConditional(Product, B, Gaussian.PointMass(A));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp4"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            return 0.0;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp5"]/doc/*'/>
    /// <remarks>
    /// This class allows EP to process the product factor using a log-normal approximation to the input distributions
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    internal static class GaussianProductOp5
    {
        public static Gaussian GetExpMoments(Gaussian x)
        {
            double m, v;
            x.GetMeanAndVariance(out m, out v);
            return new Gaussian(Math.Exp(m + v / 2), Math.Exp(2 * m + v) * (Math.Exp(v) - 1));
        }

        public static Gaussian GetLogMoments(Gaussian x)
        {
            double m, v;
            x.GetMeanAndVariance(out m, out v);
            double lv = Math.Log(v / (m * m) + 1);
            double lm = Math.Log(Math.Abs(m)) - lv / 2;
            return new Gaussian(lm, lv);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp5"]/message_doc[@name="ProductAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian ProductAverageConditional(Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            Gaussian logA = GetLogMoments(A);
            Gaussian logB = GetLogMoments(B);
            Gaussian logMsg = DoublePlusOp.SumAverageConditional(logA, logB);
            if (false)
            {
                Gaussian logProduct = GetLogMoments(Product);
                Gaussian logPost = logProduct * logMsg;
                return GetExpMoments(logPost) / Product;
            }
            else
            {
                return GetExpMoments(logMsg);
            }
            //return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp5"]/message_doc[@name="ProductAverageConditional(Gaussian, double, Gaussian)"]/*'/>
        public static Gaussian ProductAverageConditional(Gaussian Product, double A, [SkipIfUniform] Gaussian B)
        {
            return ProductAverageConditional(Product, Gaussian.PointMass(A), B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp5"]/message_doc[@name="ProductAverageConditional(Gaussian, Gaussian, double)"]/*'/>
        public static Gaussian ProductAverageConditional(Gaussian Product, [SkipIfUniform] Gaussian A, double B)
        {
            return ProductAverageConditional(Product, A, Gaussian.PointMass(B));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp5"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (A.IsPointMass)
                return GaussianProductOp.AAverageConditional(Product, A, B);
            Gaussian logA = GetLogMoments(A);
            Gaussian logB = GetLogMoments(B);
            Gaussian logProduct = GetLogMoments(Product);
            Gaussian logMsg = DoublePlusOp.AAverageConditional(logProduct, logB);
            return GetExpMoments(logMsg);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp5"]/message_doc[@name="AAverageConditional(Gaussian, Gaussian, double)"]/*'/>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, double B)
        {
            return AAverageConditional(Product, A, Gaussian.PointMass(B));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp5"]/message_doc[@name="BAverageConditional(Gaussian, Gaussian, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, Gaussian B)
        {
            return AAverageConditional(Product, B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp5"]/message_doc[@name="BAverageConditional(Gaussian, double, Gaussian)"]/*'/>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, double A, Gaussian B)
        {
            return BAverageConditional(Product, B, Gaussian.PointMass(A));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductOp5"]/message_doc[@name="LogEvidenceRatio(Gaussian, Gaussian, Gaussian)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            return 0.0;
        }
    }

    /// <remarks>
    /// This class allows EP to process the product factor using Laplace's method.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public class GaussianProductOp_EM : GaussianProductOpEvidenceBase
    {
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (A.IsPointMass)
                return GaussianProductOp.ProductAverageConditional(A.Point, B);
            else if (B.IsPointMass)
                return GaussianProductOp.ProductAverageConditional(A, B.Point);
            else
                throw new ArgumentException("Neither A nor B is a point mass");
        }

        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, Gaussian B)
        {
            if (!B.IsPointMass)
                throw new ArgumentException("B is not a point mass");
            return GaussianProductOp.AAverageConditional(Product, B.Point);
        }

        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, [Fresh] Gaussian to_A)
        {
            Gaussian Apost = A * to_A;
            return GaussianProductVmpOp.BAverageLogarithm(Product, Apost);
        }

        public static Gaussian AAverageConditional2([SkipIfUniform] Gaussian Product, [NoInit] Gaussian A, Gaussian B, Gaussian to_B)
        {
            if (A.IsPointMass)
            {
                Gaussian Bpost = B * to_B;
                return GaussianProductVmpOp.AAverageLogarithm(Product, Bpost);
            }
            return GaussianProductOp.AAverageConditional(Product, A, B);
        }

        public static Gaussian BAverageConditional2([SkipIfUniform] Gaussian Product, Gaussian A, [NoInit] Gaussian B, Gaussian to_A)
        {
            return AAverageConditional2(Product, B, A, to_A);
        }

        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, Gaussian to_product)
        {
            return GaussianProductOp.LogEvidenceRatio(Product, A, B, to_product);
        }
    }

    /// <remarks>
    /// This class requires B to be a point mass distribution.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Preview)]
    public class GaussianProductOp_PointB : GaussianProductOpEvidenceBase
    {
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (B.IsPointMass)
                return GaussianProductOp.ProductAverageConditional(A, B.Point);
            else if (A.IsPointMass)
                return GaussianProductOp.ProductAverageConditional(A.Point, B);
            else
                throw new ArgumentException("Neither A nor B is a point mass");
        }

        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian B)
        {
            if (!B.IsPointMass)
                throw new ArgumentException("B is not a point mass");
            return GaussianProductOp.AAverageConditional(Product, B.Point);
        }

        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [NoInit] Gaussian B)
        {
            return GaussianProductOp.BAverageConditional(Product, A, B);
        }

        public static double LogEvidenceRatio(double Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (!B.IsPointMass)
                throw new ArgumentException("B is not a point mass");
            return GaussianProductOp.LogEvidenceRatio(Product, A, B.Point);
        }

        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (!B.IsPointMass)
                throw new ArgumentException("B is not a point mass");
            return GaussianProductOp.LogEvidenceRatio(Product, A, B.Point);
        }

        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, [SkipIfUniform] TruncatedGaussian B)
        {
            if (B.IsPointMass)
                return GaussianProductOp.ProductAverageConditional(A, B.Point);
            else
                throw new ArgumentException("B is not a point mass");
        }

        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] TruncatedGaussian B)
        {
            if (!B.IsPointMass)
                throw new ArgumentException("B is not a point mass");
            return GaussianProductOp.AAverageConditional(Product, B.Point);
        }

        public static TruncatedGaussian BAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [NoInit] TruncatedGaussian B)
        {
            return TruncatedGaussian.FromGaussian(GaussianProductOp.BAverageConditional(Product, A, B.Gaussian));
        }

        public static double LogEvidenceRatio(double Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] TruncatedGaussian B)
        {
            if (!B.IsPointMass)
                throw new ArgumentException("B is not a point mass");
            return GaussianProductOp.LogEvidenceRatio(Product, A, B.Point);
        }

        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] TruncatedGaussian B)
        {
            if (!B.IsPointMass)
                throw new ArgumentException("B is not a point mass");
            return GaussianProductOp.LogEvidenceRatio(Product, A, B.Point);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Mature)]
    public static class GaussianProductVmpOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/message_doc[@name="AverageLogFactor(Gaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor(Gaussian product)
        {
            return 0.0;
        }

        internal const string NotSupportedMessage = "Variational Message Passing does not support a Product factor with fixed output and two random inputs.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/message_doc[@name="ProductAverageLogarithm(Gaussian, Gaussian)"]/*'/>
        public static Gaussian ProductAverageLogarithm([SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            // p(x|a,b) = N(E[a]*E[b], E[b]^2*var(a) + E[a]^2*var(b) + var(a)*var(b))
            double ma, va, mb, vb;
            A.GetMeanAndVariance(out ma, out va);
            B.GetMeanAndVariance(out mb, out vb);
            if (double.IsInfinity(va))
            {
                // avoid multiplying Inf*0
                if (vb == 0 && mb == 0)
                    return Gaussian.PointMass(0);
                else
                    return Gaussian.Uniform();
            }
            if (double.IsInfinity(vb))
            {
                // avoid multiplying Inf*0
                if (va == 0 && ma == 0)
                    return Gaussian.PointMass(0);
                else
                    return Gaussian.Uniform();
            }
            // Uses John Winn's rule for deterministic factors.
            // Strict variational inference would set the variance to 0.
            return Gaussian.FromMeanAndVariance(ma * mb, mb * mb * va + ma * ma * vb + va * vb);
        }

        public static Gaussian ProductDeriv(Gaussian Product, [SkipIfUniform, Stochastic] Gaussian A, [SkipIfUniform, Stochastic] Gaussian B, Gaussian to_A, Gaussian to_B)
        {
            if (A.IsPointMass)
                return ProductDeriv(Product, A.Point, B, to_B);
            if (B.IsPointMass)
                return ProductDeriv(Product, A, B.Point, to_A);
            double ma, va, mb, vb;
            A.GetMeanAndVariance(out ma, out va);
            B.GetMeanAndVariance(out mb, out vb);
            //Console.WriteLine("ma = {0}, va = {1}, mb = {2}, vb = {3}", ma, va, mb, vb);
            double ma0, va0, mb0, vb0;
            (A / to_A).GetMeanAndVariance(out ma0, out va0);
            (B / to_B).GetMeanAndVariance(out mb0, out vb0);
            Gaussian to_A2 = AAverageLogarithm(Product, B);
            double va2 = 1 / (1 / va0 + to_A2.Precision);
            double ma2 = va2 * (ma0 / va0 + to_A2.MeanTimesPrecision);
            Gaussian to_B2 = BAverageLogarithm(Product, A);
            double vb2 = 1 / (1 / vb0 + to_B2.Precision);
            double mb2 = vb2 * (mb0 / vb0 + to_B2.MeanTimesPrecision);
            double dva2 = 0;
            double dma2 = va2 * mb + dva2 * ma2 / va2;
            double dvb2 = 0;
            // this doesn't seem to help
            //dvb2 = -vb2*vb2*2*ma2*dma2*Product.Precision;
            double dmb2 = vb2 * ma + dvb2 * mb2 / vb2;
            double pPrec2 = 1 / (va2 * vb2 + va2 * mb2 * mb2 + vb2 * ma2 * ma2);
            double dpPrec2 = -(dva2 * vb2 + va2 * dvb2 + dva2 * mb2 * mb2 + va2 * 2 * mb2 * dmb2 + dvb2 * ma2 * ma2 + vb2 * 2 * ma2 * dma2) * pPrec2 * pPrec2;
            double pMeanTimesPrec2 = ma2 * mb2 * pPrec2;
            double pMeanTimesPrec2Deriv = dma2 * mb2 * pPrec2 + ma2 * dmb2 * pPrec2 + ma2 * mb2 * dpPrec2;
            return Gaussian.FromNatural(pMeanTimesPrec2Deriv - 1, 0);
        }

        [Skip]
        public static Gaussian ProductDeriv(Gaussian Product, [SkipIfUniform, Stochastic] Gaussian A, double B, Gaussian to_A)
        {
            return Gaussian.Uniform();
        }

        [Skip]
        public static Gaussian ProductDeriv(Gaussian Product, double A, [SkipIfUniform, Stochastic] Gaussian B, Gaussian to_B)
        {
            return ProductDeriv(Product, B, A, to_B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/message_doc[@name="ProductAverageLogarithm(double, Gaussian)"]/*'/>
        public static Gaussian ProductAverageLogarithm(double A, [SkipIfUniform] Gaussian B)
        {
            if (B.IsPointMass)
                return Gaussian.PointMass(A * B.Point);
            if (A == 0)
                return Gaussian.PointMass(A);
            // m = A*mb
            // v = A*A*vb
            // 1/v = (1/vb)/(A*A)
            // m/v = (mb/vb)/A
            double meanTimesPrecision = B.MeanTimesPrecision / A;
            double precision = B.Precision / A / A;
            if (precision > double.MaxValue || Math.Abs(meanTimesPrecision) > double.MaxValue)
            {
                return Gaussian.FromMeanAndPrecision(A * B.GetMean(), precision);
            }
            else
            {
                return Gaussian.FromNatural(meanTimesPrecision, precision);
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/message_doc[@name="ProductAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian ProductAverageLogarithm([SkipIfUniform] Gaussian A, double B)
        {
            return ProductAverageLogarithm(B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/message_doc[@name="ProductAverageLogarithm(double, TruncatedGaussian)"]/*'/>
        public static TruncatedGaussian ProductAverageLogarithm(double A, [SkipIfUniform] TruncatedGaussian B)
        {
            if (B.IsUniform()) return B;
            return new TruncatedGaussian(ProductAverageLogarithm(A, B.Gaussian), ((A >= 0) ? B.LowerBound : B.UpperBound) * A, ((A >= 0) ? B.UpperBound : B.LowerBound) * A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/message_doc[@name="ProductAverageLogarithm(TruncatedGaussian, double)"]/*'/>
        public static TruncatedGaussian ProductAverageLogarithm([SkipIfUniform] TruncatedGaussian A, double B)
        {
            return ProductAverageLogarithm(B, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/message_doc[@name="AAverageLogarithm(Gaussian, Gaussian)"]/*'/>
        public static Gaussian AAverageLogarithm([SkipIfUniform] Gaussian Product, [Proper] Gaussian B)
        {
            if (B.IsPointMass)
                return AAverageLogarithm(Product, B.Point);
            if (Product.IsPointMass)
                return AAverageLogarithm(Product.Point, B);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            // note this is exact if B is a point mass (vb=0).
            Gaussian result = new Gaussian();
            if (double.IsPositiveInfinity(vb) && Product.Precision == 0)
                result.Precision = 0;
            else
                result.Precision = Product.Precision * (vb + mb * mb);
            result.MeanTimesPrecision = Product.MeanTimesPrecision * mb;
            return result;
        }


        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/message_doc[@name="AAverageLogarithm(double, Gaussian)"]/*'/>
        [NotSupported(GaussianProductVmpOp.NotSupportedMessage)]
        public static Gaussian AAverageLogarithm(double Product, [Proper] Gaussian B)
        {
            // Throw an exception rather than return a meaningless point mass.
            throw new NotSupportedException(GaussianProductVmpOp.NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/message_doc[@name="AAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian AAverageLogarithm([SkipIfUniform] Gaussian Product, double B)
        {
            if (Product.IsPointMass)
                return AAverageLogarithm(Product.Point, B);
            return GaussianProductOp.AAverageConditional(Product, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/message_doc[@name="AAverageLogarithm(double, double)"]/*'/>
        public static Gaussian AAverageLogarithm(double Product, double B)
        {
            return GaussianProductOp.AAverageConditional(Product, B);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/message_doc[@name="BAverageLogarithm(Gaussian, Gaussian)"]/*'/>
        public static Gaussian BAverageLogarithm([SkipIfUniform] Gaussian Product, [Proper] Gaussian A)
        {
            return AAverageLogarithm(Product, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/message_doc[@name="BAverageLogarithm(double, Gaussian)"]/*'/>
        [NotSupported(GaussianProductVmpOp.NotSupportedMessage)]
        public static Gaussian BAverageLogarithm(double Product, [Proper] Gaussian A)
        {
            return AAverageLogarithm(Product, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/message_doc[@name="BAverageLogarithm(Gaussian, double)"]/*'/>
        public static Gaussian BAverageLogarithm([SkipIfUniform] Gaussian Product, double A)
        {
            return AAverageLogarithm(Product, A);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianProductVmpOp"]/message_doc[@name="BAverageLogarithm(double, double)"]/*'/>
        public static Gaussian BAverageLogarithm(double Product, double A)
        {
            return AAverageLogarithm(Product, A);
        }
    }
}
