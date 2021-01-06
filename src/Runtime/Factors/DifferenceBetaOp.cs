// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    using System;
    using System.Collections.Generic;
    using System.Text;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DifferenceBetaOp"]/doc/*'/>
    [FactorMethod(typeof(Factor), "Difference", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class DifferenceBetaOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DifferenceBetaOp"]/message_doc[@name="DifferenceAverageConditional(Beta, Beta)"]/*'/>
        public static Gaussian DifferenceAverageConditional(Beta a, Beta b)
        {
            a.GetMeanAndVariance(out double ma, out double va);
            b.GetMeanAndVariance(out double mb, out double vb);
            return Gaussian.FromMeanAndVariance(ma - mb, va + vb);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DifferenceBetaOp"]/message_doc[@name="AAverageConditional(Gaussian, Beta)"]/*'/>
        public static Beta AAverageConditional([SkipIfUniform] Gaussian difference, Beta b)
        {
            difference.GetMeanAndVariance(out double md, out double vd);
            b.GetMeanAndVariance(out double mb, out double vb);
            return Beta.FromMeanAndVariance(md + mb, vd + vb);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="DifferenceBetaOp"]/message_doc[@name="BAverageConditional(Gaussian, Beta)"]/*'/>
        public static Beta BAverageConditional([SkipIfUniform] Gaussian difference, Beta a)
        {
            return AAverageConditional(difference, a);
        }
    }
}
