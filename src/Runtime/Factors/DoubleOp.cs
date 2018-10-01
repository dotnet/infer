// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors.Attributes;

namespace Microsoft.ML.Probabilistic.Factors
{
    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Double", typeof(int))]
    [Quality(QualityBand.Preview)]
    public static class DoubleOp
    {
        public static bool ForceProper = true;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleOp"]/message_doc[@name="LogEvidenceRatio(double, int)"]/*'/>
        public static double LogEvidenceRatio(double Double, int Integer)
        {
            return (Double == Factor.Double(Integer)) ? 0.0 : double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleOp"]/message_doc[@name="LogEvidenceRatio(double, Discrete)"]/*'/>
        public static double LogEvidenceRatio(double Double, Discrete Integer)
        {
            int i = (int)Double;
            if (i != Double || i < 0 || i >= Integer.Dimension)
                return double.NegativeInfinity;
            return Integer.GetLogProb(i);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, int)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio(Gaussian Double, int Integer)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleOp"]/message_doc[@name="LogAverageFactor(Gaussian, Discrete)"]/*'/>
        public static double LogAverageFactor(Gaussian Double, Discrete Integer)
        {
            double logZ = double.NegativeInfinity;
            for (int i = 0; i < Integer.Dimension; i++)
            {
                double logp = Double.GetLogProb(i) + Integer.GetLogProb(i);
                logZ = MMath.LogSumExp(logZ, logp);
            }
            return logZ;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleOp"]/message_doc[@name="LogEvidenceRatio(Gaussian, Discrete, Gaussian)"]/*'/>
        public static double LogEvidenceRatio(Gaussian Double, Discrete Integer, Gaussian to_double)
        {
            return LogAverageFactor(Double, Integer) - to_double.GetLogAverageOf(Double);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleOp"]/message_doc[@name="DoubleAverageConditional(Gaussian, Discrete)"]/*'/>
        public static Gaussian DoubleAverageConditional(Gaussian Double, Discrete Integer)
        {
            if (Integer.IsPointMass)
                return Gaussian.PointMass(Factor.Double(Integer.Point));
            // Z = sum_i int_x q(x) delta(x - i) q(i) dx
            //   = sum_i q(x=i) q(i)
            double max = double.NegativeInfinity;
            for (int i = 0; i < Integer.Dimension; i++)
            {
                double logp = Double.GetLogProb(i);
                if (logp > max)
                    max = logp;
            }
            if (double.IsNegativeInfinity(max))
                throw new AllZeroException();
            GaussianEstimator est = new GaussianEstimator();
            for (int i = 0; i < Integer.Dimension; i++)
            {
                double logp = Double.GetLogProb(i);
                est.Add(i, Integer[i] * System.Math.Exp(logp - max));
            }
            Gaussian result = est.GetDistribution(new Gaussian());
            result.SetToRatio(result, Double, ForceProper);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DoubleOp"]/message_doc[@name="IntegerAverageConditional(Gaussian, Discrete)"]/*'/>
        public static Discrete IntegerAverageConditional(Gaussian Double, Discrete result)
        {
            if (Double.IsPointMass)
            {
                result.Point = (int)System.Math.Round(Double.Point);
                return result;
            }
            Vector probs = result.GetWorkspace();
            double max = double.NegativeInfinity;
            for (int i = 0; i < result.Dimension; i++)
            {
                double logp = Double.GetLogProb(i);
                probs[i] = logp;
                if (logp > max)
                    max = logp;
            }
            if (double.IsNegativeInfinity(max))
                throw new AllZeroException();
            probs.SetToFunction(probs, logp => System.Math.Exp(logp - max));
            result.SetProbs(probs);
            return result;
        }
    }
}
