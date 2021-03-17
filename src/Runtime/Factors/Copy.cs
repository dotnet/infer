// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of the variable being copied.</typeparam>
    [FactorMethod(typeof(Clone), "Copy<>")]
    [Quality(QualityBand.Mature)]
    public static class CopyOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="LogAverageFactor(T, T)"]/*'/>
        public static double LogAverageFactor(T copy, T value)
        {
            IEqualityComparer<T> equalityComparer = Utilities.Util.GetEqualityComparer<T>();
            return equalityComparer.Equals(copy, value) ? 0.0 : Double.NegativeInfinity;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="LogAverageFactor{TDist}(TDist, T)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static double LogAverageFactor<TDist>(TDist copy, T value)
            where TDist : CanGetLogProb<T>
        {
            return copy.GetLogProb(value);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="LogAverageFactor{TDist}(TDist, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static double LogAverageFactor<TDist>(TDist copy, TDist value)
            where TDist : CanGetLogAverageOf<TDist>
        {
            return value.GetLogAverageOf(copy);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="LogAverageFactor{TDist}(T, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static double LogAverageFactor<TDist>(T copy, TDist value)
            where TDist : CanGetLogProb<T>
        {
            return value.GetLogProb(copy);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="LogEvidenceRatio(T, T)"]/*'/>
        public static double LogEvidenceRatio(T copy, T value)
        {
            return LogAverageFactor(copy, value);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="LogEvidenceRatio{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<TDist>(TDist copy)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="LogEvidenceRatio{TDist}(T, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static double LogEvidenceRatio<TDist>(T copy, [RequiredArgument] TDist value)
            where TDist : CanGetLogProb<T>
        {
            return value.GetLogProb(copy);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="ValueAverageConditional{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist ValueAverageConditional<TDist>([IsReturned] TDist copy)
            where TDist : IDistribution<T>
        {
            return copy;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="ValueAverageConditional(T)"]/*'/>
        public static T ValueAverageConditional([IsReturned] T copy)
        {
            return copy;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="ValueAverageConditional{TDist}(T, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist ValueAverageConditional<TDist>(T copy, TDist result)
            where TDist : IDistribution<T>
        {
            result.Point = copy;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="CopyAverageConditional{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist CopyAverageConditional<TDist>([IsReturned] TDist Value)
            where TDist : IDistribution<T>
        {
            return Value;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="CopyAverageConditional(T)"]/*'/>
        public static T CopyAverageConditional([IsReturned] T Value)
        {
            return Value;
        }

        //-- VMP ------------------------------------------------------------------------------------
        // NOTE: AverageLogFactor operators are explicit here so that the correct overload is used if
        // the copy or value is a truncated Gaussian

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="AverageLogFactor{TDist}(TDist, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        [Skip]
        public static double AverageLogFactor<TDist>(TDist copy, TDist Value)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="AverageLogFactor{TDist}(TDist, T)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        [Skip]
        public static double AverageLogFactor<TDist>(TDist copy, T Value)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="AverageLogFactor{TDist}(T, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        [Skip]
        public static double AverageLogFactor<TDist>(T copy, TDist Value)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="AverageLogFactor(T, T)"]/*'/>
        public static double AverageLogFactor(T copy, T Value)
        {
            return LogAverageFactor(copy, Value);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="ValueAverageLogarithm{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist ValueAverageLogarithm<TDist>([IsReturned] TDist copy) // must have upward Trigger to match the Trigger on UsesEqualDef.UsesAverageLogarithm
            where TDist : IDistribution<T>
        {
            return copy;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="ValueAverageLogarithm{TDist}(T, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist ValueAverageLogarithm<TDist>(T copy, TDist result)
            where TDist : IDistribution<T>
        {
            result.Point = copy;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="CopyAverageLogarithm{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist CopyAverageLogarithm<TDist>([IsReturned] TDist Value)
            where TDist : IDistribution<T>
        {
            return Value;
        }

        [Skip]
        public static TDist CopyDeriv<TDist>(TDist result)
            where TDist : IDistribution<T>, SettableToUniform
        {
            result.SetToUniform();
            return result;
        }

        /* Methods for using the Copy factor to convert between Gammas and Truncated Gammas */

        [Quality(QualityBand.Preview)]
        public static TruncatedGamma ValueAverageConditional(Gamma copy)
        {
            return new TruncatedGamma(copy);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="CopyAverageConditional(TruncatedGamma)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static Gamma CopyAverageConditional(TruncatedGamma value)
        {
            // TODO: use EP
            if (!value.IsPointMass)
                throw new ArgumentException("value is not a point mass");
            return value.ToGamma();
        }

        /* Methods for using the Copy factor to convert between Gaussians and Truncated Gaussians */

        [Quality(QualityBand.Preview)]
        public static TruncatedGaussian ValueAverageConditional(Gaussian copy)
        {
            return new TruncatedGaussian(copy);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="CopyAverageConditional(TruncatedGaussian)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static Gaussian CopyAverageConditional(TruncatedGaussian value)
        {
            // TODO: use EP
            if (!value.IsPointMass)
                throw new ArgumentException("value is not a point mass");
            return value.ToGaussian();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="ValueAverageLogarithm(Gaussian)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static TruncatedGaussian ValueAverageLogarithm(Gaussian copy)
        {
            return new TruncatedGaussian(copy);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="CopyAverageLogarithm(TruncatedGaussian)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static Gaussian CopyAverageLogarithm(TruncatedGaussian value)
        {
            return value.ToGaussian();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="ValueAverageLogarithm(TruncatedGaussian, Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static Gaussian ValueAverageLogarithm(TruncatedGaussian copy, [Proper] Gaussian value, Gaussian to_value)
        {
            var a = value / to_value;
            copy *= new TruncatedGaussian(a); // is this ok? 
            var result = copy.ToGaussian() / a;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="AverageLogFactor(TruncatedGaussian, Gaussian, Gaussian)"]/*'/>
        /// <remarks>
        /// <para>
        /// This factor is implicitly maintaining the truncated Gaussian variational posterior. Therefore
        /// we need to remove the entropy of the Gaussian representation, and add the entropy for the
        /// truncated Gaussian
        /// </para>
        /// </remarks>
        [Quality(QualityBand.Preview)]
        public static double AverageLogFactor(TruncatedGaussian copy, Gaussian value, Gaussian to_value)
        {
            var a = value / to_value;
            copy *= new TruncatedGaussian(a);
            return value.GetAverageLog(value) - copy.GetAverageLog(copy);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="CopyAverageLogarithm(Gaussian, Gaussian)"]/*'/>
        [Quality(QualityBand.Preview)]
        public static TruncatedGaussian CopyAverageLogarithm([SkipIfUniform, Stochastic] Gaussian value, Gaussian to_value)
        {
            return new TruncatedGaussian(value / to_value);
        }

        /*-------------- Nonconjugate Gaussian --------------------*/

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="ValueAverageLogarithm(NonconjugateGaussian, Gaussian, Gaussian)"]/*'/>
        /// <remarks><para>
        /// We reverse the direction of factor to get the behaviour we want here.
        /// </para></remarks>
        public static Gaussian ValueAverageLogarithm(NonconjugateGaussian copy, Gaussian value, Gaussian to_value)
        {
            var a = value / to_value;
            copy *= new NonconjugateGaussian(a);
            var result = copy.GetGaussian(true) / a;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CopyOp{T}"]/message_doc[@name="CopyAverageLogarithm(Gaussian, NonconjugateGaussian, Gaussian)"]/*'/>
        /// <remarks><para>
        /// This message should include the previous contribution.
        /// </para></remarks>
        public static NonconjugateGaussian CopyAverageLogarithm(Gaussian value, NonconjugateGaussian copy, Gaussian to_value)
        {
            return new NonconjugateGaussian(value / to_value);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxProductCopyOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of the variable being copied.</typeparam>
    [FactorMethod(new string[] { "copy", "Value" }, typeof(Clone), "Copy<>")]
    [Quality(QualityBand.Experimental)]
    public class MaxProductCopyOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxProductCopyOp{T}"]/message_doc[@name="ValueMaxConditional{TDist}(TDist, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist ValueMaxConditional<TDist>([IsReturned] TDist copy, TDist result)
            where TDist : IDistribution<T>
        {
            return copy;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="MaxProductCopyOp{T}"]/message_doc[@name="CopyMaxConditional{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist CopyMaxConditional<TDist>([IsReturned] TDist Value)
            where TDist : IDistribution<T>
        {
            return Value;
        }
    }
}
