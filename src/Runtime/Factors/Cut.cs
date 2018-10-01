// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of the variable being copied.</typeparam>
    [FactorMethod(typeof(Factor), "Cut<>")]
    [Quality(QualityBand.Preview)]
    public static class CutOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutOp{T}"]/message_doc[@name="ValueAverageConditional{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        [Skip]
        public static TDist ValueAverageConditional<TDist>(TDist result)
            where TDist : IDistribution<T>
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutOp{T}"]/message_doc[@name="CutAverageConditional{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist CutAverageConditional<TDist>([IsReturned] TDist Value)
            where TDist : IDistribution<T>
        {
            return Value;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutOp{T}"]/message_doc[@name="CutAverageConditional(T)"]/*'/>
        public static T CutAverageConditional([IsReturned] T Value)
        {
            return Value;
        }

        // VMP /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutOp{T}"]/message_doc[@name="ValueAverageConditional{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        [Skip]
        public static TDist ValueAverageLogarithm<TDist>(TDist result)
            where TDist : IDistribution<T>
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutOp{T}"]/message_doc[@name="CutAverageConditional{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist CutAverageLogarithm<TDist>([IsReturned] TDist Value)
            where TDist : IDistribution<T>
        {
            return Value;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutOp{T}"]/message_doc[@name="CutAverageConditional(T)"]/*'/>
        public static T CutAverageLogarithm([IsReturned] T Value)
        {
            return Value;
        }
    }
}
