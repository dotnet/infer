// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductGaussianBetaVmpOp"]/doc/*'/>
    /// <remarks>
    /// Implements nonconjugate VMP messages for multiplying a Gaussian variable (a) with a Beta variable (b).
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class ProductGaussianBetaVmpOp
    {
        /// <summary>
        /// How much damping to use to prevent improper messages. 
        /// </summary>
        public static double damping = 0.5;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductGaussianBetaVmpOp"]/message_doc[@name="ProductAverageLogarithm(Gaussian, Beta)"]/*'/>
        public static Gaussian ProductAverageLogarithm([SkipIfUniform] Gaussian A, [SkipIfUniform] Beta B)
        {
            double ma = A.GetMean(), mb = B.GetMean();
            double va = A.GetVariance(), vb = B.GetVariance();
            Gaussian result = Gaussian.FromMeanAndVariance(ma*mb, va*vb + va*mb*mb + vb*ma*ma);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductGaussianBetaVmpOp"]/message_doc[@name="ProductAverageLogarithm(double, Beta)"]/*'/>
        public static Gaussian ProductAverageLogarithm(double A, [SkipIfUniform] Beta B)
        {
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            Gaussian result = new Gaussian();
            result.SetMeanAndVariance(A*mb, A*A*vb);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductGaussianBetaVmpOp"]/message_doc[@name="AAverageLogarithm(Gaussian, Beta)"]/*'/>
        public static Gaussian AAverageLogarithm([SkipIfUniform] Gaussian Product, [Proper] Beta B)
        {
            if (B.IsPointMass) return GaussianProductVmpOp.AAverageLogarithm(Product, B.Point);
            if (Product.IsPointMass) return AAverageLogarithm(Product.Point, B);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            Gaussian result = new Gaussian();
            result.Precision = Product.Precision*(vb + mb*mb);
            result.MeanTimesPrecision = Product.MeanTimesPrecision*mb;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductGaussianBetaVmpOp"]/message_doc[@name="AAverageLogarithm(double, Beta)"]/*'/>
        [NotSupported(GaussianProductVmpOp.NotSupportedMessage)]
        public static Gaussian AAverageLogarithm(double Product, [Proper] Beta B)
        {
            // Throw an exception rather than return a meaningless point mass.
            throw new NotSupportedException(GaussianProductVmpOp.NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductGaussianBetaVmpOp"]/message_doc[@name="BAverageLogarithm(Gaussian, Gaussian, Beta, Beta)"]/*'/>
        public static Beta BAverageLogarithm([SkipIfUniform] Gaussian Product, [Proper, SkipIfUniform] Gaussian A, [Proper] Beta B, Beta to_B)
        {
            //double Brequired = Product.GetMean()/A.GetMean();
            //if (Brequired < 0 || Brequired > 1)
            //    Console.WriteLine("Warning: BAverageLogarithm E[B]={0}", Brequired); 
            if (B.IsPointMass) return Beta.Uniform();
            if (Product.IsPointMass) return BAverageLogarithm(Product.Point, A);
            double mp, vp;
            Product.GetMeanAndVariance(out mp, out vp);
            double trueCount = B.TrueCount;
            double falseCount = B.FalseCount;
            double totalCount = B.TotalCount;
            double ma, va;
            A.GetMeanAndVariance(out ma, out va);

            double c1 = ma*Product.MeanTimesPrecision;
            double c2 = -0.5*(ma*ma + va)*Product.Precision;
            double dmda = falseCount/(totalCount*totalCount);
            double dmdb = -trueCount/(totalCount*totalCount);
            double invTC12 = 1/((totalCount + 1)*(totalCount + 1));
            double q = (trueCount + 1)*(totalCount + 1 + totalCount);
            double dm2da = dmda*(q - totalCount)*invTC12;
            double dm2db = dmdb*q*invTC12;
            double dSda = c1*dmda + c2*dm2da;
            double dSdb = c1*dmdb + c2*dm2db;
            double triTrue = MMath.Trigamma(trueCount);
            double triFalse = MMath.Trigamma(falseCount);
            double triTotal = MMath.Trigamma(totalCount);
            double u = 1/(triTrue*triFalse - triTotal*(triTrue + triFalse));
            double msgTrue = 1 + (dSda*(triFalse - triTotal) + dSdb*triTotal)*u;
            double msgFalse = 1 + (dSdb*(triTrue - triTotal) + dSda*triTotal)*u;
            var approximateFactor = new Beta(msgTrue, msgFalse);
            //double damping = 0.0;
            if (damping == 0.0)
                return approximateFactor;
            else
                return (approximateFactor ^ (1 - damping))*(to_B ^ damping);
            //return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductGaussianBetaVmpOp"]/message_doc[@name="BAverageLogarithm(double, Gaussian)"]/*'/>
        [NotSupported(GaussianProductVmpOp.NotSupportedMessage)]
        public static Beta BAverageLogarithm(double Product, Gaussian A)
        {
            // Throw an exception rather than return a meaningless point mass.
            throw new NotSupportedException(GaussianProductVmpOp.NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductGaussianBetaVmpOp"]/message_doc[@name="BAverageLogarithm(Gaussian, double, Beta, Beta)"]/*'/>
        public static Beta BAverageLogarithm([SkipIfUniform] Gaussian Product, double A, [Proper] Beta B, Beta result)
        {
            return BAverageLogarithm(Product, Gaussian.PointMass(A), B, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ProductGaussianBetaVmpOp"]/message_doc[@name="BAverageLogarithm(double, double)"]/*'/>
        public static Beta BAverageLogarithm(double Product, double A)
        {
            Beta result = new Beta();
            if (A == 0)
            {
                if (Product != 0) throw new AllZeroException();
                result.SetToUniform();
            }
            else if ((Product > 0) != (A > 0)) throw new AllZeroException("Product and argument do not have the same sign");
            else if (Product > A) throw new AllZeroException("Product is greater than the argument");
            else result.Point = Product/A;
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GaussianBetaProductOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class GaussianBetaProductOp
    {
        /// <summary>
        /// Find a proposal distribution - we will use this to set the limits of
        /// integration
        /// </summary>
        /// <param name="y">Gaussian distribution for y</param>
        /// <param name="a">Gaussian distribution for a</param>
        /// <param name="b">Beta distribution for b</param>
        /// <returns>A proposal distribution</returns>
        public static Gaussian ProposalDistribution(Gaussian y, Gaussian a, Beta b)
        {
            double ma, my, va, vy;
            y.GetMeanAndVariance(out my, out vy);
            a.GetMeanAndVariance(out ma, out va);
            Gaussian g;
            // If ma is zero, just build the proposal distribution on the beta
            if (ma == 0.0)
                g = Gaussian.Uniform();
            else
                g = Gaussian.FromMeanAndVariance(my/ma, (vy + va)/ma);
            return LogisticProposalDistribution(b, g);
        }

        /// <summary>
        /// Find the Laplace approximation for Beta(Logistic(x)) * Gaussian(x))
        /// </summary>
        /// <param name="beta">Beta distribution</param>
        /// <param name="gauss">Gaussian distribution</param>
        /// <returns>A proposal distribution</returns>
        public static Gaussian LogisticProposalDistribution(Beta beta, Gaussian gauss)
        {
            if (beta.IsUniform())
                return new Gaussian(gauss);

            // if gauss is uniform, m,p = 0 below, and the following code will just ignore the Gaussian
            // and do a Laplace approximation for Beta(Logistic(x))

            double c = beta.TrueCount - 1;
            double d = beta.FalseCount - 1;
            double m = gauss.GetMean();
            double p = gauss.Precision;
            // We want to find the mode of
            // ln(g(x)) = c.ln(f(x)) + d.ln(1 - f(x)) - 0.5p((x - m)^2) + constant
            // First deriv:
            // h(x) = (ln(g(x))' = c.(1 - f(x)) - d.f(x) - p(x-m)
            // Second deriv:
            // h'(x) = (ln(g(x))' = -(c+d).f'(x) - p
            // Use Newton-Raphson to find unique root of h(x).
            // g(x) is log-concave so Newton-Raphson should converge quickly.
            // Set the initial point by projecting beta
            // to a Gaussian and taking the mean of the product:
            double bMean, bVar;
            beta.GetMeanAndVariance(out bMean, out bVar);
            Gaussian prod = new Gaussian();
            double invLogisticMean = Math.Log(bMean) - Math.Log(1.0 - bMean);
            prod.SetToProduct(Gaussian.FromMeanAndVariance(invLogisticMean, bVar), gauss);
            double xnew = prod.GetMean();
            double x = 0, fx, dfx, hx, dhx = 0;
            int maxIters = 100; // Should only need a handful of iters
            int cnt = 0;
            do
            {
                x = xnew;
                fx = MMath.Logistic(x);
                dfx = fx*(1.0 - fx);
                // Find the root of h(x)
                hx = c*(1.0 - fx) - d*fx - p*(x - m);
                dhx = -(c + d)*dfx - p;
                xnew = x - (hx/dhx); // The Newton step
                if (Math.Abs(x - xnew) < 0.00001)
                    break;
            } while (++cnt < maxIters);
            if (cnt >= maxIters)
                throw new InferRuntimeException("Unable to find proposal distribution mode");
            return Gaussian.FromMeanAndPrecision(x, -dhx);
        }

#if false
        /// <summary>
        /// EP message to 'product'.
        /// </summary>
        /// <param name="Product">Incoming message from 'product'.</param>
        /// <param name="A">Incoming message from 'a'.</param>
        /// <param name="B">Incoming message from 'b'.</param>
        /// <returns>The outgoing EP message to the 'product' argument.</returns>
        /// <remarks><para>
        /// The outgoing message is the integral of the factor times incoming messages, over all arguments except 'product'.
        /// The formula is <c>int f(product,x) q(x) dx</c> where <c>x = (a,b)</c>.
        /// </para></remarks>
        public static Gaussian ProductAverageConditional(Gaussian Product, Gaussian A, Beta B)
        {
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
            double mean = sumX / z;
            double var = sumX2 / z - mean * mean;
            Gaussian result = Gaussian.FromMeanAndVariance(mean, var);
            if (ForceProper) result.SetToRatioProper(result, Product);
            else result.SetToRatio(result, Product);
            return result;
        }
        /// <summary>
        /// EP message to 'a'.
        /// </summary>
        /// <param name="Product">Incoming message from 'product'.</param>
        /// <param name="A">Incoming message from 'a'.</param>
        /// <param name="B">Incoming message from 'b'.</param>
        /// <returns>The outgoing EP message to the 'a' argument.</returns>
        /// <remarks><para>
        /// The outgoing message is the integral of the factor times incoming messages, over all arguments except 'a'.
        /// The formula is <c>int f(a,x) q(x) dx</c> where <c>x = (product,b)</c>.
        /// </para></remarks>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, Gaussian B)
        {
            if (B.IsPointMass) return AAverageConditional(Product, B.Point);
            if (A.IsPointMass || Product.IsUniform()) return Gaussian.Uniform();
            Gaussian result = new Gaussian();
            // algorithm: quadrature on A from -1 to 1, plus quadrature on 1/A from -1 to 1.
            double mProduct, vProduct;
            Product.GetMeanAndVariance(out mProduct, out vProduct);
            double mA, vA;
            A.GetMeanAndVariance(out mA, out vA);
            double mB, vB;
            B.GetMeanAndVariance(out mB, out vB);
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
                double logfInvA = Gaussian.GetLogProb(mProduct * invA, mB, vProduct * invA * invA + vB) + Gaussian.GetLogProb(a, mA, vA) - Math.Log(Math.Abs(invA + Double.Epsilon));
                double fInvA = Math.Exp(logfInvA);
                z += fInvA;
                sumA += a * fInvA;
                sumA2 += a * a * fInvA;
            }
            double mean = sumA / z;
            double var = sumA2 / z - mean * mean;
            result.SetMeanAndVariance(mean, var);
            if (ForceProper) result.SetToRatioProper(result, A);
            else result.SetToRatio(result, A);
            return result;
        }
        /// <summary>
        /// EP message to 'a'.
        /// </summary>
        /// <param name="Product">Constant value for 'product'.</param>
        /// <param name="A">Incoming message from 'a'.</param>
        /// <param name="B">Incoming message from 'b'.</param>
        /// <returns>The outgoing EP message to the 'a' argument.</returns>
        /// <remarks><para>
        /// The outgoing message is the integral of the factor times incoming messages, over all arguments except 'a'.
        /// The formula is <c>int f(a,x) q(x) dx</c> where <c>x = (product,b)</c>.
        /// </para></remarks>
        public static Gaussian AAverageConditional(double Product, Gaussian A, Gaussian B)
        {
            return AAverageConditional(Gaussian.PointMass(Product), A, B);
        }
        /// <summary>
        /// EP message to 'b'.
        /// </summary>
        /// <param name="Product">Incoming message from 'product'.</param>
        /// <param name="A">Incoming message from 'a'.</param>
        /// <param name="B">Incoming message from 'b'.</param>
        /// <returns>The outgoing EP message to the 'b' argument.</returns>
        /// <remarks><para>
        /// The outgoing message is the integral of the factor times incoming messages, over all arguments except 'b'.
        /// The formula is <c>int f(b,x) q(x) dx</c> where <c>x = (product,a)</c>.
        /// </para></remarks>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, Beta B)
        {
            //return AAverageConditional(Product, B, A);
        }

        /// <summary>
        /// EP message to 'b'.
        /// </summary>
        /// <param name="Product">Incoming message from 'product'.</param>
        /// <param name="A">Incoming message from 'a'.</param>
        /// <param name="B">Incoming message from 'b'.</param>
        /// <returns>The outgoing EP message to the 'b' argument.</returns>
        /// <remarks><para>
        /// The outgoing message is the integral of the factor times incoming messages, over all arguments except 'b'.
        /// The formula is <c>int f(b,x) q(x) dx</c> where <c>x = (product,a)</c>.
        /// </para></remarks>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, Beta A, Gaussian B)
        {
            //return AAverageConditional(Product, B, A);
        }
        /// <summary>
        /// EP message to 'b'.
        /// </summary>
        /// <param name="Product">Constant value for 'product'.</param>
        /// <param name="A">Incoming message from 'a'.</param>
        /// <param name="B">Incoming message from 'b'.</param>
        /// <returns>The outgoing EP message to the 'b' argument.</returns>
        /// <remarks><para>
        /// The outgoing message is the integral of the factor times incoming messages, over all arguments except 'b'.
        /// The formula is <c>int f(b,x) q(x) dx</c> where <c>x = (product,a)</c>.
        /// </para></remarks>
        public static Beta BAverageConditional(double Product, Gaussian A, Beta B)
        {
            //return AAverageConditional(Product, B, A);
        }

        /// <summary>
        /// EP message to 'b'.
        /// </summary>
        /// <param name="Product">Constant value for 'product'.</param>
        /// <param name="A">Incoming message from 'a'.</param>
        /// <param name="B">Incoming message from 'b'.</param>
        /// <returns>The outgoing EP message to the 'b' argument.</returns>
        /// <remarks><para>
        /// The outgoing message is the integral of the factor times incoming messages, over all arguments except 'b'.
        /// The formula is <c>int f(b,x) q(x) dx</c> where <c>x = (product,a)</c>.
        /// </para></remarks>
        public static Gaussian BAverageConditional(double Product, Beta A, Gaussian B)
        {
            //return AAverageConditional(Product, B, A);
        }
#endif
    }
}
