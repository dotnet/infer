// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/doc/*'/>
    /// <remarks>Factor is f(sample, size) = 1(sample &lt; size)/size</remarks>
    [FactorMethod(typeof(Factor), "DiscreteUniform")]
    [Quality(QualityBand.Mature)]
    public static class DiscreteUniform
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="LogAverageFactor(int, int)"]/*'/>
        public static double LogAverageFactor(int sample, int size)
        {
            return (sample < size) ? -Math.Log(size) : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="LogEvidenceRatio(int, int)"]/*'/>
        public static double LogEvidenceRatio(int sample, int size)
        {
            return LogAverageFactor(sample, size);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="LogAverageFactor(int, Discrete)"]/*'/>
        public static double LogAverageFactor(int sample, Discrete size)
        {
            double z = 0.0;
            for (int i = sample + 1; i < size.Dimension; i++)
            {
                z += size[i] / i;
            }
            return Math.Log(z);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="LogEvidenceRatio(int, Discrete)"]/*'/>
        public static double LogEvidenceRatio(int sample, Discrete size)
        {
            return LogAverageFactor(sample, size);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="LogAverageFactor(Discrete, Discrete)"]/*'/>
        public static double LogAverageFactor(Discrete sample, [Fresh] Discrete to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="LogEvidenceRatio(Discrete, Discrete)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Discrete sample, Discrete size)
        {
            return Math.Log(1 - size[0]);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="LogEvidenceRatio(Discrete, int)"]/*'/>
        public static double LogEvidenceRatio([SkipIfUniform] Discrete sample, int size)
        {
            if (size == 0)
                return Double.NegativeInfinity;
            else
                return 0.0;
        }

        // This method cannot be marked [Skip] because size may be uncertain.  See DiscreteTests.BallCounting2.
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="SampleAverageConditional(int, Discrete)"]/*'/>
        public static Discrete SampleAverageConditional(int size, Discrete result)
        {
            if (size == 0)
                return result; //throw new AllZeroException();
            if (result.Dimension < size)
                throw new ArgumentException("result.Dimension (" + result.Dimension + ") < size (" + size + ")");
            Vector probs = result.GetWorkspace();
            double invSize = 1.0 / size;
            for (int i = 0; i < size; i++)
            {
                probs[i] = invSize;
            }
            for (int i = size; i < probs.Count; i++)
            {
                probs[i] = 0.0;
            }
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="SampleAverageConditional(Discrete, Discrete)"]/*'/>
        public static Discrete SampleAverageConditional(Discrete size, Discrete result)
        {
            if (size.IsPointMass)
                return SampleAverageConditional(size.Point, result);
            if (result.Dimension < size.Dimension - 1)
                throw new ArgumentException("result.Dimension (" + result.Dimension + ") < size.Dimension-1 (" + size.Dimension + "-1)");
            Vector probs = result.GetWorkspace();
            // if size.Dimension = 8, then size ranges 0..7, which means sample ranges 0..6
            for (int i = probs.Count - 1; i >= size.Dimension - 1; i--)
            {
                probs[i] = 0.0;
            }
            if (size.Dimension > 1)
            {
                // p(sample) = sum_size 1(sample < size)/size p(size)
                //           = sum_(size > sample) p(size)/size
                // e.g. if dimension == 4 then p(sample=0) = p(size=1)/1 + p(size=2)/2 + p(size=3)/3
                // note the value of size[0] is irrelevant
                probs[size.Dimension - 2] = size[size.Dimension - 1] / (size.Dimension - 1);
                for (int i = size.Dimension - 3; i >= 0; i--)
                {
                    probs[i] = probs[i + 1] + size[i + 1] / (i + 1);
                }
            }
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="SampleAverageConditionalInit(int)"]/*'/>
        [Skip]
        public static Discrete SampleAverageConditionalInit(int size)
        {
            return Discrete.Uniform(size);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="SampleAverageConditionalInit(Discrete)"]/*'/>
        [Skip]
        public static Discrete SampleAverageConditionalInit([IgnoreDependency] Discrete size)
        {
            // if size.Dimension = 8, then size ranges 0..7, which means sample ranges 0..6
            return Discrete.Uniform(size.Dimension - 1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="SizeAverageConditional(int, Discrete)"]/*'/>
        public static Discrete SizeAverageConditional(int sample, Discrete result)
        {
            Vector probs = result.GetWorkspace();
            for (int size = 0; size <= sample; size++)
            {
                probs[size] = 0.0;
            }
            for (int size = sample + 1; size < probs.Count; size++)
            {
                probs[size] = 1.0 / size;
            }
            result.SetProbs(probs);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="SizeAverageConditional(Discrete, Discrete)"]/*'/>
        public static Discrete SizeAverageConditional([SkipIfUniform] Discrete sample, Discrete result)
        {
            if (sample.IsPointMass)
                return SizeAverageConditional(sample.Point, result);
            // p(size) =propto sum_sample p(sample) 1(sample < size)/size
            //         =propto p(sample < size)/size
            // e.g. if dimension == 4 then p(size=3) = p(sample < 3)/3
            Vector probs = result.GetWorkspace();
            probs[0] = 0.0;
            for (int size = 1; size < probs.Count; size++)
            {
                probs[size] = probs[size - 1] + sample[size - 1];
            }
            for (int size = 1; size < probs.Count; size++)
            {
                probs[size] /= size;
            }
            result.SetProbs(probs);
            return result;
        }

        // VMP -----------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="AverageLogFactor(Discrete, int)"]/*'/>
        public static double AverageLogFactor(Discrete sample, int size)
        {
            double sum = 0.0;
            for (int i = 0; (i < sample.Dimension) && (i < size); i++)
            {
                sum += sample[i];
            }
            return -Math.Log(size) * sum;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="AverageLogFactor(int, Discrete)"]/*'/>
        public static double AverageLogFactor(int sample, Discrete size)
        {
            double logZ = Double.NegativeInfinity;
            for (int n = 1; n < size.Dimension; n++)
            {
                logZ = MMath.LogSumExp(logZ, AverageLogFactor(sample, n));
            }
            return logZ;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="AverageLogFactor(int, int)"]/*'/>
        public static double AverageLogFactor(int sample, int size)
        {
            return LogAverageFactor(sample, size);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="AverageLogFactor(Discrete, Discrete)"]/*'/>
        public static double AverageLogFactor(Discrete sample, Discrete size)
        {
            double logZ = Double.NegativeInfinity;
            for (int n = 1; n < size.Dimension; n++)
            {
                logZ = MMath.LogSumExp(logZ, AverageLogFactor(sample, n));
            }
            return logZ;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="SampleAverageLogarithmInit(int)"]/*'/>
        [Skip]
        public static Discrete SampleAverageLogarithmInit(int size)
        {
            return Discrete.Uniform(size);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="SampleAverageLogarithmInit(Discrete)"]/*'/>
        [Skip]
        public static Discrete SampleAverageLogarithmInit([IgnoreDependency] Discrete size)
        {
            // if size.Dimension = 8, then size ranges 0..7, which means sample ranges 0..6
            return Discrete.Uniform(size.Dimension - 1);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="SampleAverageLogarithm(int, Discrete)"]/*'/>
        public static Discrete SampleAverageLogarithm(int size, Discrete result)
        {
            return SampleAverageConditional(size, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="SampleAverageLogarithm(Discrete, Discrete)"]/*'/>
        public static Discrete SampleAverageLogarithm(Discrete size, Discrete result)
        {
            return SampleAverageConditional(size, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="SizeAverageLogarithm(int, Discrete)"]/*'/>
        public static Discrete SizeAverageLogarithm(int sample, Discrete result)
        {
            return SizeAverageConditional(sample, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteUniform"]/message_doc[@name="SizeAverageLogarithm(Discrete, Discrete)"]/*'/>
        public static Discrete SizeAverageLogarithm([SkipIfUniform] Discrete sample, Discrete result)
        {
            return SizeAverageConditional(sample, result);
        }
    }
}
