// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors.Attributes;

[assembly: HasMessageFunctions]

namespace Microsoft.ML.Probabilistic.Factors
{
    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UnaryOp{DomainType}"]/doc/*'/>
    /// <typeparam name="DomainType">The type of the sampled variable.</typeparam>
    [FactorMethod(typeof(Factor), "Random<>")]
    [Quality(QualityBand.Mature)]
    public static class UnaryOp<DomainType>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UnaryOp{DomainType}"]/message_doc[@name="LogAverageFactor{T}(T, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        public static double LogAverageFactor<T>(T random, T dist)
            where T : CanGetLogAverageOf<T>
        {
            return dist.GetLogAverageOf(random);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UnaryOp{DomainType}"]/message_doc[@name="LogAverageFactor{T}(DomainType, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        public static double LogAverageFactor<T>(DomainType random, T dist)
            where T : CanGetLogProb<DomainType>
        {
            return dist.GetLogProb(random);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UnaryOp{DomainType}"]/message_doc[@name="LogEvidenceRatio{T}(T, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<T>(T random, T dist)
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UnaryOp{DomainType}"]/message_doc[@name="LogEvidenceRatio{T}(DomainType, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        public static double LogEvidenceRatio<T>(DomainType random, T dist)
            where T : CanGetLogProb<DomainType>
        {
            return LogAverageFactor(random, dist);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UnaryOp{DomainType}"]/message_doc[@name="RandomAverageConditional{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        public static T RandomAverageConditional<T>([IsReturned] T dist)
        {
            return dist;
        }

        //-- VMP ---------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UnaryOp{DomainType}"]/message_doc[@name="AverageLogFactor{T}(T, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        public static double AverageLogFactor<T>(T random, T dist)
            where T : CanGetAverageLog<T>
        {
            return random.GetAverageLog(dist);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UnaryOp{DomainType}"]/message_doc[@name="AverageLogFactor{T}(DomainType, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        public static double AverageLogFactor<T>(DomainType random, T dist)
            where T : CanGetLogProb<DomainType>
        {
            return dist.GetLogProb(random);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UnaryOp{DomainType}"]/message_doc[@name="RandomAverageLogarithm{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        public static T RandomAverageLogarithm<T>([IsReturned] T dist)
        {
            return dist;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UnaryOp{DomainType}"]/message_doc[@name="DistAverageLogarithm{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the sampled variable.</typeparam>
        /// <typeparam name="TDist">The type of the distribution over the distribution of the sampled variable.</typeparam>
        public static TDist DistAverageLogarithm<T, TDist>(T random, TDist result)
            where TDist : HasPoint<T>
        {
            result.Point = random;
            return result;
        }

        //-- Max product ---------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UnaryOp{DomainType}"]/message_doc[@name="RandomMaxConditional(Bernoulli)"]/*'/>
        public static Bernoulli RandomMaxConditional([IsReturned] Bernoulli dist)
        {
            return dist;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UnaryOp{DomainType}"]/message_doc[@name="RandomMaxConditional(Discrete)"]/*'/>
        public static UnnormalizedDiscrete RandomMaxConditional([SkipIfUniform] Discrete dist)
        {
            return UnnormalizedDiscrete.FromDiscrete(dist);
        }
    }
}
