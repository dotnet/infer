// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "BetaFromMeanAndTotalCount")]
    [Quality(QualityBand.Experimental)]
    public static class BetaFromMeanAndTotalCountOp
    {
        /// <summary>
        /// How much damping to use to avoid improper messages. A higher value implies more damping. 
        /// </summary>
        public static double damping = 0.0;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="AverageLogFactor(double, double, double)"]/*'/>
        public static double AverageLogFactor(double prob, double mean, double totalCount)
        {
            return LogAverageFactor(prob, mean, totalCount);
        }

        // TODO: VMP evidence messages for stochastic inputs (see DirichletOp)

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="ProbAverageLogarithm(Beta, Gamma)"]/*'/>
        public static Beta ProbAverageLogarithm(Beta mean, [Proper] Gamma totalCount)
        {
            double meanMean = mean.GetMean();
            double totalCountMean = totalCount.GetMean();
            return new Beta(meanMean * totalCountMean, (1 - meanMean) * totalCountMean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="ProbAverageLogarithm(Beta, double)"]/*'/>
        public static Beta ProbAverageLogarithm(Beta mean, double totalCount)
        {
            double meanMean = mean.GetMean();
            return new Beta(meanMean * totalCount, (1 - meanMean) * totalCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="ProbAverageLogarithm(double, Gamma)"]/*'/>
        public static Beta ProbAverageLogarithm(double mean, [Proper] Gamma totalCount)
        {
            double totalCountMean = totalCount.GetMean();
            return new Beta(mean * totalCountMean, (1 - mean) * totalCountMean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="MeanAverageLogarithm(double, Beta, Gamma, Beta)"]/*'/>
        public static Beta MeanAverageLogarithm(double prob, Beta mean, [Proper] Gamma totalCount, Beta to_mean)
        {
            return MeanAverageLogarithm(Beta.PointMass(prob), mean, totalCount, to_mean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="MeanAverageLogarithm(double, Beta, double, Beta)"]/*'/>
        public static Beta MeanAverageLogarithm(double prob, Beta mean, double totalCount, Beta to_mean)
        {
            return MeanAverageLogarithm(Beta.PointMass(prob), mean, Gamma.PointMass(totalCount), to_mean);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="MeanAverageLogarithm(Beta, Beta, Gamma, Beta)"]/*'/>
        public static Beta MeanAverageLogarithm([Proper] Beta prob, Beta mean, [Proper] Gamma totalCount, Beta to_mean)
        {
            // Calculate gradient using method for DirichletOp
            prob.GetMeanLogs(out double ELogP, out double ELogOneMinusP);
            Vector gradS = DirichletOp.CalculateGradientForMean(
                Vector.FromArray(new double[] { mean.TrueCount, mean.FalseCount }),
                totalCount,
                Vector.FromArray(new double[] { ELogP, ELogOneMinusP }));
            // Project onto a Beta distribution 
            Matrix A = new Matrix(2, 2);
            double c = MMath.Trigamma(mean.TotalCount);
            A[0, 0] = MMath.Trigamma(mean.TrueCount) - c;
            A[1, 0] = A[0, 1] = -c;
            A[1, 1] = MMath.Trigamma(mean.FalseCount) - c;
            Vector theta = GammaFromShapeAndRateOp.twoByTwoInverse(A) * gradS;
            Beta approximateFactor = new Beta(theta[0] + 1, theta[1] + 1);
            if (damping == 0.0)
                return approximateFactor;
            else
                return (approximateFactor ^ (1 - damping)) * (to_mean ^ damping);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="TotalCountAverageLogarithm(Beta, Gamma, Beta, Gamma)"]/*'/>
        public static Gamma TotalCountAverageLogarithm([Proper] Beta mean, Gamma totalCount, [SkipIfUniform] Beta prob, Gamma to_totalCount)
        {
            prob.GetMeanLogs(out double ELogP, out double ELogOneMinusP);
            Gamma approximateFactor = DirichletOp.TotalCountAverageLogarithmHelper(
                Vector.FromArray(new double[] { mean.TrueCount, mean.FalseCount }),
                totalCount,
                Vector.FromArray(new double[] { ELogP, ELogOneMinusP }));
            if (damping == 0.0)
                return approximateFactor;
            else
                return (approximateFactor ^ (1 - damping)) * (to_totalCount ^ damping);
        }

        //---------------------------- EP -----------------------------

        private const string NotSupportedMessage = "Expectation Propagation does not currently support beta distributions with stochastic arguments.";

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="LogAverageFactor(double, double, double)"]/*'/>
        public static double LogAverageFactor(double prob, double mean, double totalCount)
        {
            var g = new Beta(mean * totalCount, (1 - mean) * totalCount);
            return g.GetLogProb(prob);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="LogAverageFactor(Beta, double, double)"]/*'/>
        public static double LogAverageFactor(Beta prob, double mean, double totalCount)
        {
            var g = new Beta(mean * totalCount, (1 - mean) * totalCount);
            return g.GetLogAverageOf(prob);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="ProbAverageConditional(double, double)"]/*'/>
        public static Beta ProbAverageConditional(double mean, double totalCount)
        {
            return new Beta(mean * totalCount, (1 - mean) * totalCount);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="ProbAverageConditional(Beta, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Beta ProbAverageConditional([SkipIfUniform] Beta mean, [SkipIfUniform] Gamma totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="ProbAverageConditional(Beta, double)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Beta ProbAverageConditional([SkipIfUniform] Beta mean, double totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="ProbAverageConditional(double, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Beta ProbAverageConditional(double mean, [SkipIfUniform] Gamma totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="MeanAverageConditional(Beta, Gamma, double, Beta)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Beta MeanAverageConditional([SkipIfUniform] Beta mean, [SkipIfUniform] Gamma totalCount, double prob, [SkipIfUniform] Beta result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="MeanAverageConditional(Beta, double, double, Beta)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Beta MeanAverageConditional([SkipIfUniform] Beta mean, double totalCount, double prob, [SkipIfUniform] Beta result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="MeanAverageConditional(Beta, double, Beta, Beta)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Beta MeanAverageConditional([SkipIfUniform] Beta mean, double totalCount, [SkipIfUniform] Beta prob, [SkipIfUniform] Beta result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="MeanAverageConditional(Beta, Gamma, Beta, Beta)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Beta MeanAverageConditional([SkipIfUniform] Beta mean, [SkipIfUniform] Gamma totalCount, [SkipIfUniform] Beta prob, [SkipIfUniform] Beta result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="TotalCountAverageConditional(Beta, Gamma, double, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gamma TotalCountAverageConditional([SkipIfUniform] Beta mean, [SkipIfUniform] Gamma totalCount, double prob, [SkipIfUniform] Gamma result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BetaFromMeanAndTotalCountOp"]/message_doc[@name="TotalCountAverageConditional(Beta, Gamma, Beta, Gamma)"]/*'/>
        [NotSupported(NotSupportedMessage)]
        public static Gamma TotalCountAverageConditional([SkipIfUniform] Beta mean, [SkipIfUniform] Gamma totalCount, [SkipIfUniform] Beta prob, [SkipIfUniform] Gamma result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }
    }
}
