// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RotateOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Rotate", typeof(double), typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class RotateOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RotateOp"]/message_doc[@name="AverageLogFactor(VectorGaussian)"]/*'/>
        [Skip]
        public static double AverageLogFactor(VectorGaussian rotate)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RotateOp"]/message_doc[@name="XAverageLogarithm(VectorGaussian, WrappedGaussian)"]/*'/>
        public static Gaussian XAverageLogarithm([SkipIfUniform] VectorGaussian rotate, [Proper] WrappedGaussian angle)
        {
            // for x ~ N(m,v):
            // E[cos(x)] = cos(m)*exp(-v/2)
            // E[sin(x)] = sin(m)*exp(-v/2)
            if (angle.Period != 2 * Math.PI)
                throw new ArgumentException("angle.Period (" + angle.Period + ") != 2*PI (" + 2 * Math.PI + ")");
            double angleMean, angleVar;
            angle.Gaussian.GetMeanAndVariance(out angleMean, out angleVar);
            double expVar = Math.Exp(-0.5 * angleVar);
            double mCos = Math.Cos(angleMean) * expVar;
            double mSin = Math.Sin(angleMean) * expVar;
            if (rotate.Dimension != 2)
                throw new ArgumentException("rotate.Dimension (" + rotate.Dimension + ") != 2");
            double prec = rotate.Precision[0, 0];
            if (rotate.Precision[0, 1] != 0)
                throw new ArgumentException("rotate.Precision is not diagonal");
            if (rotate.Precision[1, 1] != prec)
                throw new ArgumentException("rotate.Precision is not spherical");
#if false
            Vector rotateMean = rotate.GetMean();
            double mean = mCos*rotateMean[0] + mSin*rotateMean[1];
#else
            double rotateMean0 = rotate.MeanTimesPrecision[0] / rotate.Precision[0, 0];
            double rotateMean1 = rotate.MeanTimesPrecision[1] / rotate.Precision[1, 1];
            double mean = mCos * rotateMean0 + mSin * rotateMean1;
#endif
            if (double.IsNaN(mean))
                throw new InferRuntimeException("result is nan");
            return Gaussian.FromMeanAndPrecision(mean, prec);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RotateOp"]/message_doc[@name="YAverageLogarithm(VectorGaussian, WrappedGaussian)"]/*'/>
        public static Gaussian YAverageLogarithm([SkipIfUniform] VectorGaussian rotate, [Proper] WrappedGaussian angle)
        {
            // for x ~ N(m,v):
            // E[cos(x)] = cos(m)*exp(-v/2)
            // E[sin(x)] = sin(m)*exp(-v/2)
            if (angle.Period != 2 * Math.PI)
                throw new ArgumentException("angle.Period (" + angle.Period + ") != 2*PI (" + 2 * Math.PI + ")");
            double angleMean, angleVar;
            angle.Gaussian.GetMeanAndVariance(out angleMean, out angleVar);
            double expVar = Math.Exp(-0.5 * angleVar);
            double mCos = Math.Cos(angleMean) * expVar;
            double mSin = Math.Sin(angleMean) * expVar;
            if (rotate.Dimension != 2)
                throw new ArgumentException("rotate.Dimension (" + rotate.Dimension + ") != 2");
            double prec = rotate.Precision[0, 0];
            if (rotate.Precision[0, 1] != 0)
                throw new ArgumentException("rotate.Precision is not diagonal");
            if (rotate.Precision[1, 1] != prec)
                throw new ArgumentException("rotate.Precision is not spherical");
#if false
            Vector rotateMean = rotate.GetMean();
            double mean = -mSin*rotateMean[0] + mCos*rotateMean[1];
#else
            double rotateMean0 = rotate.MeanTimesPrecision[0] / rotate.Precision[0, 0];
            double rotateMean1 = rotate.MeanTimesPrecision[1] / rotate.Precision[1, 1];
            double mean = -mSin * rotateMean0 + mCos * rotateMean1;
#endif
            return Gaussian.FromMeanAndPrecision(mean, prec);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RotateOp"]/message_doc[@name="XAverageLogarithm(VectorGaussian, double)"]/*'/>
        public static Gaussian XAverageLogarithm([SkipIfUniform] VectorGaussian rotate, double angle)
        {
            return XAverageLogarithm(rotate, WrappedGaussian.PointMass(angle));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RotateOp"]/message_doc[@name="YAverageLogarithm(VectorGaussian, double)"]/*'/>
        public static Gaussian YAverageLogarithm([SkipIfUniform] VectorGaussian rotate, double angle)
        {
            return YAverageLogarithm(rotate, WrappedGaussian.PointMass(angle));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RotateOp"]/message_doc[@name="RotateAverageLogarithmInit()"]/*'/>
        [Skip]
        public static VectorGaussian RotateAverageLogarithmInit()
        {
            return new VectorGaussian(2);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RotateOp"]/message_doc[@name="RotateAverageLogarithm(Gaussian, Gaussian, WrappedGaussian, VectorGaussian)"]/*'/>
        public static VectorGaussian RotateAverageLogarithm(
            [SkipIfUniform] Gaussian x, [SkipIfUniform] Gaussian y, [Proper] WrappedGaussian angle, VectorGaussian result)
        {
            // for x ~ N(m,v):
            // E[cos(x)] = cos(m)*exp(-v/2)
            // E[sin(x)] = sin(m)*exp(-v/2)
            if (angle.Period != 2 * Math.PI)
                throw new ArgumentException("angle.Period (" + angle.Period + ") != 2*PI (" + 2 * Math.PI + ")");
            double angleMean, angleVar;
            angle.Gaussian.GetMeanAndVariance(out angleMean, out angleVar);
            double expHalfVar = Math.Exp(-0.5 * angleVar);
            double mCos = Math.Cos(angleMean) * expHalfVar;
            double mSin = Math.Sin(angleMean) * expHalfVar;
            double mCos2 = mCos * mCos;
            double mSin2 = mSin * mSin;
            //  E[cos(x)^2] = 0.5 E[1+cos(2x)] = 0.5 (1 + cos(2m) exp(-2v))
            //  E[sin(x)^2] = E[1 - cos(x)^2] = 0.5 (1 - cos(2m) exp(-2v))
            double expVar = expHalfVar * expHalfVar;
            // cos2m = cos(2m)*exp(-v)
            double cos2m = 2 * mCos2 - expVar;
            double mCosSqr = 0.5 * (1 + cos2m * expVar);
            double mSinSqr = 1 - mCosSqr;
            double mSinCos = mSin * mCos * expVar;
            if (result.Dimension != 2)
                throw new ArgumentException("result.Dimension (" + result.Dimension + ") != 2");
            double mx, vx, my, vy;
            x.GetMeanAndVariance(out mx, out vx);
            y.GetMeanAndVariance(out my, out vy);
            Vector mean = Vector.Zero(2);
            mean[0] = mCos * mx - mSin * my;
            mean[1] = mSin * mx + mCos * my;
            double mx2 = mx * mx + vx;
            double my2 = my * my + vy;
            double mxy = mx * my;
            PositiveDefiniteMatrix variance = new PositiveDefiniteMatrix(2, 2);
            variance[0, 0] = mx2 * mCosSqr - 2 * mxy * mSinCos + my2 * mSinSqr - mean[0] * mean[0];
            variance[0, 1] = (mx2 - my2) * mSinCos + mxy * (mCosSqr - mSinSqr) - mean[0] * mean[1];
            variance[1, 0] = variance[0, 1];
            variance[1, 1] = mx2 * mSinSqr + 2 * mxy * mSinCos + my2 * mCosSqr - mean[1] * mean[1];
            result.SetMeanAndVariance(mean, variance);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RotateOp"]/message_doc[@name="RotateAverageLogarithm(double, double, WrappedGaussian, VectorGaussian)"]/*'/>
        public static VectorGaussian RotateAverageLogarithm(double x, double y, [Proper] WrappedGaussian angle, VectorGaussian result)
        {
            return RotateAverageLogarithm(Gaussian.PointMass(x), Gaussian.PointMass(y), angle, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RotateOp"]/message_doc[@name="RotateAverageLogarithm(Gaussian, Gaussian, double, VectorGaussian)"]/*'/>
        public static VectorGaussian RotateAverageLogarithm([SkipIfUniform] Gaussian x, [SkipIfUniform] Gaussian y, double angle, VectorGaussian result)
        {
            return RotateAverageLogarithm(x, y, WrappedGaussian.PointMass(angle), result);
        }

#if false
        public static WrappedGaussian AngleAverageLogarithm([SkipIfUniform] VectorGaussian rotate, [Proper] Gaussian x, [Proper] Gaussian y, [Proper] WrappedGaussian angle)
        {
            return AngleAverageLogarithm(rotate, x.GetMean(), y.GetMean(), angle);
        }
        public static WrappedGaussian AngleAverageLogarithm([SkipIfUniform] VectorGaussian rotate, double x, double y, [Proper] WrappedGaussian angle)
        {
            if (rotate.Dimension != 2) throw new ArgumentException("rotate.Dimension ("+rotate.Dimension+") != 2");
            double rPrec = rotate.Precision[0, 0];
            if (rotate.Precision[0, 1] != 0) throw new ArgumentException("rotate.Precision is not diagonal");
            if (rotate.Precision[1, 1] != rPrec) throw new ArgumentException("rotate.Precision is not spherical");
            Vector rotateMean = rotate.GetMean();
            double a = x*rotateMean[0] + y*rotateMean[1];
            double b = x*rotateMean[1] - y*rotateMean[0];
            double c = Math.Sqrt(a*a + b*b)*rPrec;
            double angle0 = Math.Atan2(b, a);
            // the exact conditional is exp(c*cos(angle - angle0)) which is a von Mises distribution.
            // we will approximate this with a Gaussian lower bound that makes contact at the current angleMean.
            if (angle.Period != 2*Math.PI) throw new ArgumentException("angle.Period ("+angle.Period+") != 2*PI ("+2*Math.PI+")");
            double angleMean = angle.Gaussian.GetMean();
            double angleDiff = angleMean - angle0;
            double df = -c*Math.Sin(angleDiff);
            double precision = c*Math.Abs(Math.Cos(angleDiff*0.5)); // ensures a lower bound
            double meanTimesPrecision = angleMean*precision + df;
            if (double.IsNaN(meanTimesPrecision)) throw new InferRuntimeException("result is nan");
            WrappedGaussian result = WrappedGaussian.Uniform(angle.Period);
            result.Gaussian = Gaussian.FromNatural(meanTimesPrecision, precision);
            return result;
        }
#else
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RotateOp"]/message_doc[@name="AngleAverageLogarithm(VectorGaussian, Gaussian, Gaussian)"]/*'/>
        public static WrappedGaussian AngleAverageLogarithm([SkipIfUniform] VectorGaussian rotate, [Proper] Gaussian x, [Proper] Gaussian y)
        {
            return AngleAverageLogarithm(rotate, x.GetMean(), y.GetMean());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="RotateOp"]/message_doc[@name="AngleAverageLogarithm(VectorGaussian, double, double)"]/*'/>
        public static WrappedGaussian AngleAverageLogarithm([SkipIfUniform] VectorGaussian rotate, double x, double y)
        {
            if (rotate.Dimension != 2)
                throw new ArgumentException("rotate.Dimension (" + rotate.Dimension + ") != 2");
            double rPrec = rotate.Precision[0, 0];
            if (rotate.Precision[0, 1] != 0)
                throw new ArgumentException("rotate.Precision is not diagonal");
            if (rotate.Precision[1, 1] != rPrec)
                throw new ArgumentException("rotate.Precision is not spherical");
#if false
            Vector rotateMean = rotate.GetMean();
            double a = x*rotateMean[0] + y*rotateMean[1];
            double b = x*rotateMean[1] - y*rotateMean[0]; 
#else
            double rotateMean0 = rotate.MeanTimesPrecision[0] / rotate.Precision[0, 0];
            double rotateMean1 = rotate.MeanTimesPrecision[1] / rotate.Precision[1, 1];
            double a = x * rotateMean0 + y * rotateMean1;
            double b = x * rotateMean1 - y * rotateMean0;
#endif
            double c = Math.Sqrt(a * a + b * b) * rPrec;
            double angle0 = Math.Atan2(b, a);
            // the exact conditional is exp(c*cos(angle - angle0)) which is a von Mises distribution.
            // we will approximate this with a Gaussian lower bound that makes contact at the mode.
            WrappedGaussian result = WrappedGaussian.Uniform();
            result.Gaussian = Gaussian.FromMeanAndPrecision(angle0, c);
            return result;
        }
#endif
    }
}
