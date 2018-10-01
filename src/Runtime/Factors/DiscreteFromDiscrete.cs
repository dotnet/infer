// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDiscreteOp"]/doc/*'/>
    [FactorMethod(new String[] { "sample", "selector", "probs" }, typeof(Factor), "Discrete", typeof(int), typeof(Matrix))]
    [Quality(QualityBand.Experimental)]
    public static class DiscreteFromDiscreteOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDiscreteOp"]/message_doc[@name="LogAverageFactor(Discrete, Discrete, Matrix)"]/*'/>
        public static double LogAverageFactor(Discrete sample, Discrete selector, Matrix probs)
        {
            return Math.Log(probs.QuadraticForm(selector.GetProbs(), sample.GetProbs()));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDiscreteOp"]/message_doc[@name="LogEvidenceRatio(Discrete, Discrete, Matrix)"]/*'/>
        public static double LogEvidenceRatio(Discrete sample, Discrete selector, Matrix probs)
        {
            // use this if the rows are not normalized
            Discrete toSample = SampleAverageConditional(selector, probs, Discrete.Uniform(sample.Dimension, sample.Sparsity));
            return LogAverageFactor(sample, selector, probs)
                   - toSample.GetLogAverageOf(sample);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDiscreteOp"]/message_doc[@name="SampleAverageConditional(Discrete, Matrix, Discrete)"]/*'/>
        public static Discrete SampleAverageConditional(Discrete selector, Matrix probs, Discrete result)
        {
            Vector v = result.GetWorkspace();
            v.SetToProduct(selector.GetProbs(), probs);
            result.SetProbs(v);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DiscreteFromDiscreteOp"]/message_doc[@name="SelectorAverageConditional(Discrete, Matrix, Discrete)"]/*'/>
        public static Discrete SelectorAverageConditional(Discrete sample, Matrix probs, Discrete result)
        {
            Vector v = result.GetWorkspace();
            v.SetToProduct(probs, sample.GetProbs());
            result.SetProbs(v);
            return result;
        }
    }
}
