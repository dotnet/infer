// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;

    // /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of the variable being copied.</typeparam>
    [FactorMethod(typeof(Cut), "Backward<>")]
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutOp{T}"]/message_doc[@name="BackwardAverageConditional{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist BackwardAverageConditional<TDist>([IsReturned] TDist Value)
            where TDist : IDistribution<T>
        {
            return Value;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutOp{T}"]/message_doc[@name="BackwardAverageConditional(T)"]/*'/>
        public static T BackwardAverageConditional([IsReturned] T Value)
        {
            return Value;
        }

        // VMP /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutOp{T}"]/message_doc[@name="ValueAverageLogarithm{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        [Skip]
        public static TDist ValueAverageLogarithm<TDist>(TDist result)
            where TDist : IDistribution<T>
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutOp{T}"]/message_doc[@name="BackwardAverageLogarithm{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist BackwardAverageLogarithm<TDist>([IsReturned] TDist Value)
            where TDist : IDistribution<T>
        {
            return Value;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutOp{T}"]/message_doc[@name="BackwardAverageLogarithm(T)"]/*'/>
        public static T BackwardAverageLogarithm([IsReturned] T Value)
        {
            return Value;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutForwardWhenOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of the variable being copied.</typeparam>
    [FactorMethod(typeof(Cut), "ForwardWhen<>")]
    [Quality(QualityBand.Preview)]
    public static class CutForwardWhenOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutForwardWhenOp{T}"]/message_doc[@name="ValueAverageConditional{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist ValueAverageConditional<TDist>([IsReturned] TDist forwardWhen)
            where TDist : IDistribution<T>
        {
            return forwardWhen;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutForwardWhenOp{T}"]/message_doc[@name="ForwardWhenAverageConditional{TDist}(TDist,bool,TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist ForwardWhenAverageConditional<TDist>(TDist Value, bool shouldCut, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            if (shouldCut) result.SetToUniform();
            else result.SetTo(Value);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutForwardWhenOp{T}"]/message_doc[@name="ForwardWhenAverageConditional(T)"]/*'/>
        public static T ForwardWhenAverageConditional([IsReturned] T Value)
        {
            return Value;
        }

        // VMP /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutForwardWhenOp{T}"]/message_doc[@name="ValueAverageLogarithm{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        [Skip]
        public static TDist ValueAverageLogarithm<TDist>(TDist result)
            where TDist : IDistribution<T>
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutForwardWhenOp{T}"]/message_doc[@name="ForwardWhenAverageLogarithm{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable being copied.</typeparam>
        public static TDist ForwardWhenAverageLogarithm<TDist>([IsReturned] TDist Value)
            where TDist : IDistribution<T>
        {
            return Value;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="CutForwardWhenOp{T}"]/message_doc[@name="ForwardWhenAverageLogarithm(T)"]/*'/>
        public static T ForwardWhenAverageLogarithm([IsReturned] T Value)
        {
            return Value;
        }
    }

    /// <summary>
    /// Cut factor methods
    /// </summary>
    [Hidden]
    public static class Cut
    {
        /// <summary>
        /// Copy a value and cut the backward message (it will always be uniform).
        /// </summary>
        /// <typeparam name="T">The type the input value.</typeparam>
        /// <param name="value">The value to return.</param>
        /// <returns>The supplied value.</returns>
        public static T Backward<T>([SkipIfUniform] T value)
        {
            return value;
        }

        /// <summary>
        /// Copy a value and cut the forward message when <paramref name="shouldCut"/> is true.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="shouldCut"></param>
        /// <returns></returns>
        public static T ForwardWhen<T>([IsReturned] T value, bool shouldCut)
        {
            return value;
        }
    }
}
