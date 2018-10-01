// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Linq;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndScaleOp"]/doc/*'/>
    [FactorMethod(typeof(Wishart), "SampleFromShapeAndScale")]
    [Quality(QualityBand.Stable)]
    public static class WishartFromShapeAndScaleOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndScaleOp"]/message_doc[@name="LogAverageFactor(PositiveDefiniteMatrix, double, PositiveDefiniteMatrix)"]/*'/>
        public static double LogAverageFactor(PositiveDefiniteMatrix sample, double shape, PositiveDefiniteMatrix scale)
        {
            Wishart to_sample = SampleAverageConditional(shape, scale);
            return to_sample.GetLogProb(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndScaleOp"]/message_doc[@name="LogEvidenceRatio(PositiveDefiniteMatrix, double, PositiveDefiniteMatrix)"]/*'/>
        public static double LogEvidenceRatio(PositiveDefiniteMatrix sample, double shape, PositiveDefiniteMatrix scale)
        {
            return LogAverageFactor(sample, shape, scale);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndScaleOp"]/message_doc[@name="LogEvidenceRatio(Wishart, double, PositiveDefiniteMatrix)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Wishart sample, double shape, PositiveDefiniteMatrix scale)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndScaleOp"]/message_doc[@name="AverageLogFactor(PositiveDefiniteMatrix, double, PositiveDefiniteMatrix)"]/*'/>
        public static double AverageLogFactor(PositiveDefiniteMatrix sample, double shape, PositiveDefiniteMatrix scale)
        {
            return LogAverageFactor(sample, shape, scale);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndScaleOp"]/message_doc[@name="AverageLogFactor(Wishart, Wishart)"]/*'/>
        public static double AverageLogFactor(Wishart sample, [Fresh] Wishart to_sample)
        {
            return LogAverageFactor(sample, to_sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndScaleOp"]/message_doc[@name="LogAverageFactor(Wishart, Wishart)"]/*'/>
        public static double LogAverageFactor(Wishart sample, [Fresh] Wishart to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndScaleOp"]/message_doc[@name="SampleAverageLogarithm(double, PositiveDefiniteMatrix)"]/*'/>
        public static Wishart SampleAverageLogarithm(double shape, PositiveDefiniteMatrix scale)
        {
            return Wishart.FromShapeAndScale(shape, scale);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndScaleOp"]/message_doc[@name="SampleAverageConditional(double, PositiveDefiniteMatrix)"]/*'/>
        public static Wishart SampleAverageConditional(double shape, PositiveDefiniteMatrix scale)
        {
            return Wishart.FromShapeAndScale(shape, scale);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp"]/doc/*'/>
    [FactorMethod(typeof(Wishart), "SampleFromShapeAndRate", typeof(double), typeof(PositiveDefiniteMatrix))]
    [Quality(QualityBand.Stable)]
    public static class WishartFromShapeAndRateOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp"]/message_doc[@name="LogAverageFactor(PositiveDefiniteMatrix, double, PositiveDefiniteMatrix)"]/*'/>
        public static double LogAverageFactor(PositiveDefiniteMatrix sample, double shape, PositiveDefiniteMatrix rate)
        {
            int dimension = sample.Rows;
            Wishart to_sample = SampleAverageConditional(shape, rate, new Wishart(dimension));
            return to_sample.GetLogProb(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp"]/message_doc[@name="LogEvidenceRatio(PositiveDefiniteMatrix, double, PositiveDefiniteMatrix)"]/*'/>
        public static double LogEvidenceRatio(PositiveDefiniteMatrix sample, double shape, PositiveDefiniteMatrix rate)
        {
            return LogAverageFactor(sample, shape, rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp"]/message_doc[@name="LogAverageFactor(PositiveDefiniteMatrix, double, Wishart)"]/*'/>
        public static double LogAverageFactor(PositiveDefiniteMatrix sample, double shape, Wishart rate)
        {
            // f(X,a,B) = |X|^(a-c) |B|^a/Gamma_d(a) exp(-tr(BX)) = |X|^(-2c) Gamma_d(a+c)/Gamma_d(a) W(B; a+c, X)
            // p(B) = |B|^(a'-c) |B'|^(a')/Gamma_d(a') exp(-tr(BB'))
            // int_B f p(B) dB = |X|^(a-c) Gamma_d(a+a')/Gamma_d(a)/Gamma_d(a') |B'|^(a') |B'+X|^(-(a+a'))
            int dimension = sample.Rows;
            double c = 0.5 * (dimension + 1);
            Wishart to_rate = new Wishart(dimension);
            to_rate = RateAverageConditional(sample, shape, to_rate);
            return rate.GetLogAverageOf(to_rate) - 2 * c * sample.LogDeterminant() + MMath.GammaLn(shape + c, dimension) - MMath.GammaLn(shape, dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp"]/message_doc[@name="LogEvidenceRatio(PositiveDefiniteMatrix, double, Wishart)"]/*'/>
        public static double LogEvidenceRatio(PositiveDefiniteMatrix sample, double shape, Wishart rate)
        {
            return LogAverageFactor(sample, shape, rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp"]/message_doc[@name="LogEvidenceRatio(Wishart, double, PositiveDefiniteMatrix)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Wishart sample, double shape, PositiveDefiniteMatrix rate)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp"]/message_doc[@name="RateAverageConditional(PositiveDefiniteMatrix, double, Wishart)"]/*'/>
        public static Wishart RateAverageConditional(PositiveDefiniteMatrix sample, double shape, Wishart result)
        {
            int dimension = result.Dimension;
            result.Shape = shape + 0.5 * (dimension + 1);
            result.Rate.SetTo(sample);
            return result;
        }

        // VMP //////////////////////////////////////////////////////////////////////////////////////////////////////////

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp"]/message_doc[@name="AverageLogFactor(PositiveDefiniteMatrix, double, PositiveDefiniteMatrix)"]/*'/>
        public static double AverageLogFactor(PositiveDefiniteMatrix sample, double shape, PositiveDefiniteMatrix rate)
        {
            return LogAverageFactor(sample, shape, rate);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp"]/message_doc[@name="LogAverageFactor(Wishart, Wishart)"]/*'/>
        public static double LogAverageFactor(Wishart sample, [Fresh] Wishart to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp"]/message_doc[@name="AverageLogFactor(Wishart, double, Wishart)"]/*'/>
        public static double AverageLogFactor([Proper] Wishart sample, double shape, [Proper] Wishart rate)
        {
            // factor = (a-(d+1)/2)*logdet(X) -tr(X*B) + a*logdet(B) - GammaLn(a,d)
            int dimension = sample.Dimension;
            return (shape - (dimension + 1) * 0.5) * sample.GetMeanLogDeterminant() - Matrix.TraceOfProduct(sample.GetMean(), rate.GetMean()) + shape * rate.GetMeanLogDeterminant() -
                         MMath.GammaLn(shape, dimension);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp"]/message_doc[@name="SampleAverageLogarithm(double, PositiveDefiniteMatrix, Wishart)"]/*'/>
        public static Wishart SampleAverageLogarithm(double shape, PositiveDefiniteMatrix rate, Wishart result)
        {
            result.Shape = shape;
            result.Rate.SetTo(rate);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp"]/message_doc[@name="SampleAverageLogarithm(double, Wishart, Wishart)"]/*'/>
        public static Wishart SampleAverageLogarithm(double shape, [SkipIfUniform] Wishart rate, Wishart result)
        {
            result.Shape = shape;
            rate.GetMean(result.Rate);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp"]/message_doc[@name="RateAverageLogarithm(Wishart, double, Wishart)"]/*'/>
        public static Wishart RateAverageLogarithm([SkipIfUniform] Wishart sample, double shape, Wishart result)
        {
            int dimension = result.Dimension;
            result.Shape = shape + 0.5 * (dimension + 1);
            sample.GetMean(result.Rate);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp"]/message_doc[@name="SampleAverageConditional(double, PositiveDefiniteMatrix, Wishart)"]/*'/>
        public static Wishart SampleAverageConditional(double shape, PositiveDefiniteMatrix rate, Wishart result)
        {
            return SampleAverageLogarithm(shape, rate, result);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp_Laplace2"]/doc/*'/>
    [FactorMethod(typeof(Wishart), "SampleFromShapeAndRate", typeof(double), typeof(PositiveDefiniteMatrix))]
    [Quality(QualityBand.Experimental)]
    public static class WishartFromShapeAndRateOp_Laplace2
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp_Laplace2"]/message_doc[@name="LogAverageFactor(Wishart, double, Wishart, Wishart)"]/*'/>
        public static double LogAverageFactor(Wishart sample, double shape, Wishart rate, [Fresh] Wishart to_sample)
        {
            // int_R f(Y,R) p(R) dR = |Y|^(a-c) |Y+B_r|^-(a+a_r) Gamma_d(a+a_r)/Gamma_d(a)/Gamma_d(a_r) |B_r|^a_r
            int dim = sample.Dimension;
            double c = 0.5 * (dim + 1);
            double shape2 = shape + rate.Shape;
            Wishart samplePost = sample * to_sample;
            PositiveDefiniteMatrix y = samplePost.GetMean();
            PositiveDefiniteMatrix yPlusBr = y + rate.Rate;
            double result = (shape - c) * y.LogDeterminant() - shape2 * yPlusBr.LogDeterminant() + sample.GetLogProb(y) - samplePost.GetLogProb(y);
            result += MMath.GammaLn(shape2, dim) - MMath.GammaLn(shape, dim) - MMath.GammaLn(rate.Shape, dim);
            result += rate.Shape * rate.Rate.LogDeterminant();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp_Laplace2"]/message_doc[@name="LogEvidenceRatio(Wishart, double, Wishart, Wishart)"]/*'/>
        public static double LogEvidenceRatio(Wishart sample, double shape, Wishart rate, [Fresh] Wishart to_sample)
        {
            return LogAverageFactor(sample, shape, rate, to_sample) - to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp_Laplace2"]/message_doc[@name="RateAverageConditional(Wishart, double, Wishart, Wishart, Wishart)"]/*'/>
        public static Wishart RateAverageConditional([SkipIfUniform] Wishart sample, double shape, Wishart rate, Wishart to_rate, Wishart result)
        {
            if (sample.IsPointMass)
                return WishartFromShapeAndRateOp.RateAverageConditional(sample.Point, shape, result);
            // f(Y,R) = |Y|^(a-c) |R|^a exp(-tr(YR))
            // p(Y) = |Y|^(a_y-c) exp(-tr(YB_y)
            // p(R) = |R|^(a_r-c) exp(-tr(RB_r)
            // int_Y f(Y,R) p(Y) dY = |R|^a |R+B_y|^-(a+a_y-c)
            int dim = sample.Dimension;
            double c = 0.5 * (dim + 1);
            double shape2 = shape - c + sample.Shape;
            Wishart ratePost = rate * to_rate;
            PositiveDefiniteMatrix r = ratePost.GetMean();
            PositiveDefiniteMatrix rby = r + sample.Rate;
            PositiveDefiniteMatrix invrby = rby.Inverse();
            PositiveDefiniteMatrix rInvrby = rby;
            rInvrby.SetToProduct(r, invrby);
            double xxddlogp = Matrix.TraceOfProduct(rInvrby, rInvrby) * shape2;
            double delta = -xxddlogp / dim;
            PositiveDefiniteMatrix invR = r.Inverse();
            PositiveDefiniteMatrix dlogp = invrby;
            dlogp.Scale(-shape2);
            LowerTriangularMatrix rChol = new LowerTriangularMatrix(dim, dim);
            rChol.SetToCholesky(r);
            result.SetDerivatives(rChol, invR, dlogp, xxddlogp, GammaFromShapeAndRateOp.ForceProper, shape);
            return result;
        }

        public static Wishart SampleAverageConditional2(double shape, [Proper] Wishart rate, Wishart to_rate, Wishart result)
        {
            Wishart ratePost = rate * to_rate;
            PositiveDefiniteMatrix r = ratePost.GetMean();
            return WishartFromShapeAndRateOp.SampleAverageConditional(shape, r, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="WishartFromShapeAndRateOp_Laplace2"]/message_doc[@name="SampleAverageConditional(Wishart, double, Wishart, Wishart, Wishart, Wishart)"]/*'/>
        public static Wishart SampleAverageConditional([NoInit] Wishart sample, double shape, [Proper] Wishart rate, [NoInit] Wishart to_rate, [NoInit] Wishart to_sample, Wishart result)
        {
            if (sample.IsUniform())
                return SampleAverageConditional2(shape, rate, to_rate, result);
            // f(Y,R) = |Y|^(a-c) |R|^a exp(-tr(YR))
            // p(Y) = |Y|^(a_y-c) exp(-tr(YB_y)
            // p(R) = |R|^(a_r-c) exp(-tr(RB_r)
            // int_R f(Y,R) p(R) dR = |Y|^(a-c) |Y+B_r|^-(a+a_r)
            int dim = sample.Dimension;
            double c = 0.5 * (dim + 1);
            double shape2 = shape + rate.Shape;
            Wishart samplePost = sample * to_sample;
            if(!samplePost.IsProper())
                return SampleAverageConditional2(shape, rate, to_rate, result);
            PositiveDefiniteMatrix y = samplePost.GetMean();
            PositiveDefiniteMatrix yPlusBr = y + rate.Rate;
            PositiveDefiniteMatrix invyPlusBr = yPlusBr.Inverse();
            PositiveDefiniteMatrix yInvyPlusBr = yPlusBr;
            yInvyPlusBr.SetToProduct(y, invyPlusBr);
            double xxddlogf = shape2 * Matrix.TraceOfProduct(yInvyPlusBr, yInvyPlusBr);
            PositiveDefiniteMatrix invY = y.Inverse();
            //double delta = -xxddlogf / dim;
            //result.Shape = delta + shape;
            //result.Rate.SetToSum(delta, invY, shape2, invyPlusBr);
            LowerTriangularMatrix yChol = new LowerTriangularMatrix(dim, dim);
            yChol.SetToCholesky(y);
            PositiveDefiniteMatrix dlogp = invyPlusBr;
            dlogp.Scale(-shape2);
            result.SetDerivatives(yChol, invY, dlogp, xxddlogf, GammaFromShapeAndRateOp.ForceProper, shape - c);
            if (result.Rate.Any(x => double.IsNaN(x)))
                throw new Exception("result.Rate is nan");
            return result;
        }
    }
}
