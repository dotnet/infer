// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#define UseRatioDir

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of the variable entering the gate.</typeparam>
    [FactorMethod(typeof(Gate), "EnterPartial<>", null, typeof(int), null, typeof(int[]))]
    [FactorMethod(typeof(Gate), "EnterPartial<>", null, typeof(bool), null, typeof(int[]))]
    [Quality(QualityBand.Mature)]
    public static class GateEnterPartialOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="LogEvidenceRatio{TDist}(IList{TDist})"]/*'/>
        [Skip]
        public static double LogEvidenceRatio<TDist>(IList<TDist> enterPartial)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="LogAverageFactor()"]/*'/>
        [Skip]
        public static double LogAverageFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="EnterPartialAverageConditional{TValue, TResultList}(TValue, TResultList)"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList EnterPartialAverageConditional<TValue, TResultList>([IsReturnedInEveryElement] TValue value, TResultList result)
            where TResultList : CanSetAllElementsTo<TValue>
        {
            result.SetAllElementsTo(value);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="EnterPartialAverageConditional{TValue}(TValue, TValue[])"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        public static TValue[] EnterPartialAverageConditional<TValue>([IsReturnedInEveryElement] TValue value, TValue[] result)
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="EnterPartialInit{TValue, TArray}(TValue, int[], IArrayFactory{TValue, TArray})"]/*'/>
        /// <typeparam name="TValue">The type of the incoming message from <c>value</c>.</typeparam>
        /// <typeparam name="TArray">The type of an array that can be produced by <paramref name="factory"/>.</typeparam>
        [Skip]
        public static TArray EnterPartialInit<TValue, TArray>([IgnoreDependency] TValue value, int[] indices, IArrayFactory<TValue, TArray> factory)
            where TValue : ICloneable
        {
            return factory.CreateArray(indices.Length, i => (TValue)value.Clone());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="SelectorAverageConditional(Discrete)"]/*'/>
        [Skip]
        public static Discrete SelectorAverageConditional(Discrete result)
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="SelectorAverageConditional(Bernoulli)"]/*'/>
        [Skip]
        public static Bernoulli SelectorAverageConditional(Bernoulli result)
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="ValueAverageConditional{TDist}(IList{TDist}, Discrete, TDist, int[], TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>(
            [SkipIfUniform] IList<TDist> enterPartial, [SkipIfUniform] Discrete selector, [NoInit,Proper] TDist value, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>,
                SettableToRatio<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>, SettableToPower<TDist>
        {
            if (value.IsPointMass)
                return ValueAverageLogarithm(enterPartial, selector, indices, result);
            if (indices.Length != enterPartial.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (selector.Dimension < enterPartial.Count)
                throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                // TODO: use pre-allocated buffers
                double logProbSum = selector.GetLogProb(indices[0]);
                if (!double.IsNegativeInfinity(logProbSum))
                {
                    try
                    {
                        result.SetToProduct(value, enterPartial[0]);
                    }
                    catch (AllZeroException)
                    {
                        logProbSum = double.NegativeInfinity;
                    }
                }
                if (indices.Length > 1)
                {
                    TDist product = (TDist)value.Clone();
                    for (int i = 1; i < indices.Length; i++)
                    {
                        double logProb = selector.GetLogProb(indices[i]);
                        double shift = Math.Max(logProbSum, logProb);
                        // avoid (-Infinity) - (-Infinity)
                        if (Double.IsNegativeInfinity(shift))
                        {
                            if (i == selector.Dimension - 1)
                            {
                                throw new AllZeroException();
                            }
                            // do nothing
                        }
                        else
                        {
                            double productWeight = Math.Exp(logProb - shift);
                            if (productWeight > 0)
                            {
                                try
                                {
                                    product.SetToProduct(value, enterPartial[i]);
                                }
                                catch (AllZeroException)
                                {
                                    productWeight = 0;
                                }
                                if (productWeight > 0)
                                {
                                    result.SetToSum(Math.Exp(logProbSum - shift), result, productWeight, product);
                                    logProbSum = MMath.LogSumExp(logProbSum, logProb);
                                }
                            }
                        }
                    }
                }
                if (indices.Length < selector.Dimension)
                {
                    double logProb = MMath.Log1MinusExp(logProbSum);
                    double shift = Math.Max(logProbSum, logProb);
                    if (Double.IsNegativeInfinity(shift))
                        throw new AllZeroException();
                    result.SetToSum(Math.Exp(logProbSum - shift), result, Math.Exp(logProb - shift), value);
                }
                result.SetToRatio(result, value, GateEnterOp<T>.ForceProper);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="ValueAverageConditional{TDist}(IList{TDist}, int, int[], TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartial, int selector, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            if (indices.Length != enterPartial.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                result.SetToUniform();
                for (int i = 0; i < indices.Length; i++)
                {
                    if (selector == indices[i])
                    {
                        result.SetTo(enterPartial[i]);
                        break;
                    }
                }
                return result;
            }
        }

#if false
        public static TDist ValueAverageConditional<TDist>(
            IList<T> enterPartial,
            int selector, int[] indices, TDist result)
            where TDist : IDistribution<T>
        {
            if (indices.Length != enterPartial.Count) throw new ArgumentException("indices.Length != enterPartial.Count");
            if (indices.Length == 0) throw new ArgumentException("indices.Length == 0");
            else {
                result.SetToUniform();
                for (int i = 0; i < indices.Length; i++) {
                    if (selector == indices[i]) {
                        result.Point = enterPartial[i];
                        break;
                    }
                }
                return result;
            }
        }
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="ValueAverageConditional{TDist}(IList{TDist}, Bernoulli, TDist, int[], TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>(
            [SkipIfUniform] IList<TDist> enterPartial, [SkipIfUniform] Bernoulli selector, [NoInit,Proper] TDist value, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToRatio<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>, SettableToPower<TDist>
        {
            // selector already includes the evidence messages from the gate body
            if (value.IsPointMass)
            {
                // f(x) = q(T) f1(x) + q(F) f2(x)
                // dlogf/dp = (q(T) df1(x) - q(F) df2(x))/f(x)
                // ddlogf/dp^2 = -(q(T) df1(x) - q(F) df2(x))^2/f(x)^2 + (q(T) ddf1(x) - q(F) ddf2(x))/f(x)
                // if f2(x)=1 then
                // dlogf/dp = q(T) df1(x)/f(x) = q(T) f1(x)/f(x) dlogf1(x)
                // ddlogf/dp^2 = -(q(T) df1(x))^2/f(x)^2 + q(T) ddf1(x)/f(x) 
                //             = q(T) f1(x)/f(x) (1 - q(T) f1(x)/f(x)) dlogf1(x)^2 + q(T) f1(x)/f(x) ddlogf1(x)
                // ddlogf1(x) = -dlogf1(x)^2 + ddf1(x)/f1(x)
                // we use VMP as an approximation here.
                // VMP matches the first derivative but only the 2nd term of the 2nd derivative.
                return ValueAverageLogarithm(enterPartial, selector, indices, result);
            }
            if (indices.Length != enterPartial.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (2 < enterPartial.Count)
                throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                // TODO: use pre-allocated buffers
                double logProbSum = (indices[0] == 0) ? selector.GetLogProbTrue() : selector.GetLogProbFalse();
                if (!double.IsNegativeInfinity(logProbSum))
                {
                    result.SetToProduct(value, enterPartial[0]);
                }
                if (indices.Length > 1)
                {
                    TDist product = (TDist)value.Clone();
                    for (int i = 1; i < indices.Length; i++)
                    {
                        double logProb = (indices[i] == 0) ? selector.GetLogProbTrue() : selector.GetLogProbFalse();
                        double shift = Math.Max(logProbSum, logProb);
                        // avoid (-Infinity) - (-Infinity)
                        if (Double.IsNegativeInfinity(shift))
                        {
                            if (i == 1)
                            {
                                throw new AllZeroException();
                            }
                            // do nothing
                        }
                        else
                        {
                            double productWeight = Math.Exp(logProb - shift);
                            if (productWeight > 0)
                            {
                                product.SetToProduct(value, enterPartial[i]);
                                result.SetToSum(Math.Exp(logProbSum - shift), result, productWeight, product);
                                logProbSum = MMath.LogSumExp(logProbSum, logProb);
                            }
                        }
                    }
                }
                if (indices.Length < 2)
                {
                    double logProb = MMath.Log1MinusExp(logProbSum);
                    double shift = Math.Max(logProbSum, logProb);
                    if (Double.IsNegativeInfinity(shift))
                        throw new AllZeroException();
                    result.SetToSum(Math.Exp(logProbSum - shift), result, Math.Exp(logProb - shift), value);
                }
                result.SetToRatio(result, value, GateEnterOp<T>.ForceProper);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="ValueAverageConditional{TDist}(IList{TDist}, bool, int[], TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartial, bool selector, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            if (indices.Length != enterPartial.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                int caseNumber = selector ? 0 : 1;
                result.SetToUniform();
                for (int i = 0; i < indices.Length; i++)
                {
                    if (caseNumber == indices[i])
                    {
                        result.SetTo(enterPartial[i]);
                        break;
                    }
                }
                return result;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="ValueAverageConditional{TDist}(IList{T}, bool, int[], TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>(
            IList<T> enterPartial, bool selector, int[] indices, TDist result)
            where TDist : IDistribution<T>
        {
            if (indices.Length != enterPartial.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                int caseNumber = selector ? 0 : 1;
                result.SetToUniform();
                for (int i = 0; i < indices.Length; i++)
                {
                    if (caseNumber == indices[i])
                    {
                        result.Point = enterPartial[i];
                        break;
                    }
                }
                return result;
            }
        }

#if false
    /// <summary>
    /// EP message to 'cases'
    /// </summary>
    /// <param name="result">Modified to contain the outgoing message</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the factor viewed as a function of 'cases' conditioned on the given values.
    /// </para></remarks>
        [Skip]
        public static BernoulliList CasesAverageConditional<BernoulliList>(BernoulliList result)
            where BernoulliList : SettableToUniform
        {
            result.SetToUniform();
            return result;
        }
        /// <summary>
        /// EP message to 'value'
        /// </summary>
        /// <param name="enterPartial">Incoming message from 'enterPartial'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
        /// <param name="cases">Incoming message from 'cases'. Must be a proper distribution.  If any element is uniform, the result will be uniform.</param>
        /// <param name="value">Incoming message from 'value'.</param>
        /// <param name="indices">Constant value for 'indices'.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'value' as the random arguments are varied.
        /// The formula is <c>proj[p(value) sum_(enterPartial,cases) p(enterPartial,cases) factor(enterPartial,cases,value,indices)]/p(value)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="enterPartial"/> is not a proper distribution</exception>
        /// <exception cref="ImproperMessageException"><paramref name="cases"/> is not a proper distribution</exception>
        public static TDist ValueAverageConditional<TDist>([SkipIfUniform] IList<TDist> enterPartial, [SkipIfUniform] IList<Bernoulli> cases, TDist value, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>,
                                SettableToRatio<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
        {
            if (indices.Length != enterPartial.Count) throw new ArgumentException("indices.Length != enterPartial.Count");
            if (cases.Count < enterPartial.Count) throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0) throw new ArgumentException("indices.Length == 0");
            else {
                // TODO: use pre-allocated buffers
                double logProbSum = cases[indices[0]].LogOdds;
                if (!double.IsNegativeInfinity(logProbSum)) {
                    result.SetToProduct(value, enterPartial[0]);
                }
                if (indices.Length > 1) {
                    TDist product = (TDist)value.Clone();
                    for (int i = 1; i < indices.Length; i++) {
                        double logProb = cases[indices[i]].LogOdds;
                        double shift = Math.Max(logProbSum, logProb);
                        // avoid (-Infinity) - (-Infinity)
                        if (Double.IsNegativeInfinity(shift)) {
                            if (i == cases.Count - 1) {
                                throw new AllZeroException();
                            }
                            // do nothing
                        } else {
                            double productWeight = Math.Exp(logProb - shift);
                            if (productWeight > 0) {
                                product.SetToProduct(value, enterPartial[i]);
                                result.SetToSum(Math.Exp(logProbSum - shift), result, productWeight, product);
                                logProbSum = MMath.LogSumExp(logProbSum, logProb);
                            }
                        }
                    }
                }
                if (indices.Length < cases.Count) {
                    double logProb = MMath.Log1MinusExp(logProbSum);
                    double shift = Math.Max(logProbSum, logProb);
                    if (Double.IsNegativeInfinity(shift)) throw new AllZeroException();
                    result.SetToSum(Math.Exp(logProbSum - shift), result, Math.Exp(logProb - shift), value);
                }
                if (GateEnterOp<T>.ForceProper && (result is Gaussian)) {
                    Gaussian r = (Gaussian)(object)result;
                    r.SetToRatioProper(r, (Gaussian)(object)value);
                    result = (TDist)(object)r;
                } else {
                    result.SetToRatio(result, value);
                }
            }
            return result;
        }
        /// <summary>
        /// EP message to 'value'
        /// </summary>
        /// <param name="enterPartial">Incoming message from 'enterPartial'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
        /// <param name="cases">Constant value for 'cases'.</param>
        /// <param name="indices">Constant value for 'indices'.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'value' as the random arguments are varied.
        /// The formula is <c>proj[p(value) sum_(enterPartial,cases) p(enterPartial,cases) factor(enterPartial,cases,value,indices)]/p(value)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="enterPartial"/> is not a proper distribution</exception>
        public static TDist ValueAverageConditional<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartial,
            IList<bool> cases, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            if (indices.Length != enterPartial.Count) throw new ArgumentException("indices.Length != enterPartial.Count");
            if (cases.Count < enterPartial.Count) throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0) throw new ArgumentException("indices.Length == 0");
            else {
                result.SetToUniform();
                for (int i = 0; i < indices.Length; i++) {
                    if (cases[indices[i]]) {
                        result.SetTo(enterPartial[i]);
                        break;
                    }
                }
                return result;
            }
        }
        public static TDist ValueAverageConditional<TDist>(
            IList<T> enterPartial,
            IList<bool> cases, int[] indices, TDist result)
            where TDist : IDistribution<T>
        {
            if (indices.Length != enterPartial.Count) throw new ArgumentException("indices.Length != enterPartial.Count");
            if (cases.Count < enterPartial.Count) throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0) throw new ArgumentException("indices.Length == 0");
            else {
                for (int i = 0; i < indices.Length; i++) {
                    if (cases[indices[i]]) {
                        result.Point = enterPartial[i];
                        break;
                    }
                }
                return result;
            }
        }
#endif

        //-- VMP ---------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="EnterPartialAverageLogarithm{TValue, TResultList}(TValue, TResultList)"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList EnterPartialAverageLogarithm<TValue, TResultList>([IsReturnedInEveryElement] TValue value, TResultList result)
            where TResultList : CanSetAllElementsTo<TValue>
        {
            return EnterPartialAverageConditional(value, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="EnterPartialAverageLogarithm{TValue}(TValue, TValue[])"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        public static TValue[] EnterPartialAverageLogarithm<TValue>([IsReturnedInEveryElement] TValue value, TValue[] result)
        {
            return EnterPartialAverageConditional(value, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="ValueAverageLogarithm{TDist}(IList{TDist}, int, int[], TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageLogarithm<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartial, int selector, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            return ValueAverageConditional(enterPartial, selector, indices, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="ValueAverageLogarithm{TDist}(IList{TDist}, Discrete, int[], TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam> 
        public static TDist ValueAverageLogarithm<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartial, Discrete selector, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToPower<TDist>
        {
            if (indices.Length != enterPartial.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (selector.Dimension < enterPartial.Count)
                throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                double scale = selector[indices[0]];
                result.SetToPower(enterPartial[0], scale);
                if (indices.Length > 1)
                {
                    // TODO: use pre-allocated buffer
                    TDist power = (TDist)result.Clone();
                    for (int i = 1; i < indices.Length; i++)
                    {
                        scale = selector[indices[i]];
                        power.SetToPower(enterPartial[i], scale);
                        result.SetToProduct(result, power);
                    }
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="ValueAverageLogarithm{TDist}(IList{TDist}, bool, int[], TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam> 
        public static TDist ValueAverageLogarithm<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartial, bool selector, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            return ValueAverageConditional(enterPartial, selector, indices, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="ValueAverageLogarithm{TDist}(IList{TDist}, Bernoulli, int[], TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam> 
        public static TDist ValueAverageLogarithm<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartial, Bernoulli selector, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToPower<TDist>
        {
            if (indices.Length != enterPartial.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (2 < enterPartial.Count)
                throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                double scale = (indices[0] == 0) ? selector.GetProbTrue() : selector.GetProbFalse();
                result.SetToPower(enterPartial[0], scale);
                if (indices.Length > 1)
                {
                    // TODO: use pre-allocated buffer
                    TDist power = (TDist)result.Clone();
                    for (int i = 1; i < indices.Length; i++)
                    {
                        scale = (indices[i] == 0) ? selector.GetProbTrue() : selector.GetProbFalse();
                        power.SetToPower(enterPartial[i], scale);
                        result.SetToProduct(result, power);
                    }
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="SelectorAverageLogarithm(Discrete)"]/*'/>
        [Skip]
        public static Discrete SelectorAverageLogarithm(Discrete result)
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialOp{T}"]/message_doc[@name="SelectorAverageLogarithm(Bernoulli)"]/*'/>
        [Skip]
        public static Bernoulli SelectorAverageLogarithm(Bernoulli result)
        {
            result.SetToUniform();
            return result;
        }

#if false
    /// <summary>
    /// VMP message to 'cases'
    /// </summary>
    /// <param name="result">Modified to contain the outgoing message</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except 'cases'.
    /// Because the factor is deterministic, 'enterPartial' is integrated out before taking the logarithm.
    /// The formula is <c>exp(sum_(value) p(value) log(sum_enterPartial p(enterPartial) factor(enterPartial,cases,value,indices)))</c>.
    /// </para></remarks>
        [Skip]
        public static BernoulliList CasesAverageLogarithm<BernoulliList>(BernoulliList result)
            where BernoulliList : SettableToUniform
        {
            result.SetToUniform();
            return result;
        }
        /// <summary>
        /// VMP message to 'value'
        /// </summary>
        /// <param name="enterPartial">Incoming message from 'enterPartial'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
        /// <param name="cases">Incoming message from 'cases'. Must be a proper distribution.  If any element is uniform, the result will be uniform.</param>
        /// <param name="indices">Constant value for 'indices'.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except 'value'.
        /// Because the factor is deterministic, 'enterPartial' is integrated out before taking the logarithm.
        /// The formula is <c>exp(sum_(cases) p(cases) log(sum_enterPartial p(enterPartial) factor(enterPartial,cases,value,indices)))</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="enterPartial"/> is not a proper distribution</exception>
        /// <exception cref="ImproperMessageException"><paramref name="cases"/> is not a proper distribution</exception>
        public static TDist ValueAverageLogarithm<TDist>([SkipIfAllUniform] IList<TDist> enterPartial, [SkipIfUniform] IList<Bernoulli> cases, int[] indices, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToPower<TDist>
        {
            if (indices.Length != enterPartial.Count) throw new ArgumentException("indices.Length != enterPartial.Count");
            if (cases.Count < enterPartial.Count) throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0) throw new ArgumentException("indices.Length == 0");
            else {
                double scale = Math.Exp(cases[indices[0]].LogOdds);
                result.SetToPower(enterPartial[0], scale);
                if (indices.Length > 1) {
                    // TODO: use pre-allocated buffer
                    TDist power = (TDist)result.Clone();
                    for (int i = 1; i < indices.Length; i++) {
                        scale = Math.Exp(cases[indices[i]].LogOdds);
                        power.SetToPower(enterPartial[i], scale);
                        result.SetToProduct(result, power);
                    }
                }
            }
            return result;
        }
#endif
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialTwoOp"]/doc/*'/>
    [FactorMethod(typeof(Gate), "EnterPartialTwo<>")]
    [Quality(QualityBand.Mature)]
    public static class GateEnterPartialTwoOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialTwoOp"]/message_doc[@name="LogEvidenceRatio()"]/*'/>
        [Skip]
        public static double LogEvidenceRatio()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialTwoOp"]/message_doc[@name="LogAverageFactor()"]/*'/>
        [Skip]
        public static double LogAverageFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialTwoOp"]/message_doc[@name="EnterPartialTwoAverageConditional{TValue, TResultList}(TValue, TResultList)"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList EnterPartialTwoAverageConditional<TValue, TResultList>([SkipIfUniform] TValue value, TResultList result)
            where TResultList : CanSetAllElementsTo<TValue>
        {
            result.SetAllElementsTo(value);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialTwoOp"]/message_doc[@name="Case0AverageConditional(Bernoulli)"]/*'/>
        [Skip]
        public static Bernoulli Case0AverageConditional(Bernoulli result)
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialTwoOp"]/message_doc[@name="Case1AverageConditional(Bernoulli)"]/*'/>
        [Skip]
        public static Bernoulli Case1AverageConditional(Bernoulli result)
        {
            return Case0AverageConditional(result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialTwoOp"]/message_doc[@name="ValueAverageConditional{TValue}(IList{TValue}, Bernoulli, Bernoulli, TValue, int[], TValue)"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        public static TValue ValueAverageConditional<TValue>(
            [SkipIfAllUniform] IList<TValue> enterPartialTwo, Bernoulli case0, Bernoulli case1, TValue value, int[] indices, TValue result)
            where TValue : ICloneable, SettableToUniform, SettableToProduct<TValue>, SettableToRatio<TValue>, SettableToWeightedSum<TValue>, CanGetLogAverageOf<TValue>
        {
            if (indices.Length != enterPartialTwo.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (2 < enterPartialTwo.Count)
                throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                // TODO: use pre-allocated buffers
                result.SetToProduct(value, enterPartialTwo[0]);
                double scale = Math.Exp((indices[0] == 0 ? case0 : case1).LogOdds);
                double sumCases = scale;
                double resultScale = scale;
                if (indices.Length > 1)
                {
                    TValue product = (TValue)value.Clone();
                    for (int i = 1; i < indices.Length; i++)
                    {
                        product.SetToProduct(value, enterPartialTwo[i]);
                        scale = Math.Exp((indices[i] == 0 ? case0 : case1).LogOdds);
                        result.SetToSum(resultScale, result, scale, product);
                        resultScale += scale;
                        sumCases += scale;
                    }
                }
                double totalCases = Math.Exp(case0.LogOdds) + Math.Exp(case1.LogOdds);
                result.SetToSum(resultScale, result, totalCases - sumCases, value);
                result.SetToRatio(result, value, GateEnterOp<TValue>.ForceProper);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialTwoOp"]/message_doc[@name="ValueAverageConditional{TValue, TDomain}(IList{TValue}, bool, bool, int[], TValue)"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        /// <typeparam name="TDomain">The type of the variable entering the gate.</typeparam>
        public static TValue ValueAverageConditional<TValue, TDomain>(
            [SkipIfAllUniform] IList<TValue> enterPartialTwo, bool case1, bool case2, int[] indices, TValue result)
            where TValue : IDistribution<TDomain>, SettableTo<TValue>
        {
            if (indices.Length != enterPartialTwo.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (2 < enterPartialTwo.Count)
                throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                result.SetToUniform();
                for (int i = 0; i < indices.Length; i++)
                {
                    if ((indices[i] == 0 && case1) || (indices[i] == 1 && case2))
                    {
                        result.SetTo(enterPartialTwo[indices[i]]);
                        break;
                    }
                }
                return result;
            }
        }

        //-- VMP ---------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialTwoOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialTwoOp"]/message_doc[@name="EnterPartialTwoAverageLogarithm{TValue, TResultList}(TValue, TResultList)"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList EnterPartialTwoAverageLogarithm<TValue, TResultList>([SkipIfUniform] TValue value, TResultList result)
            where TResultList : CanSetAllElementsTo<TValue>
        {
            result.SetAllElementsTo(value);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialTwoOp"]/message_doc[@name="Case0AverageLogarithm{TValue}(IList{TValue}, TValue, int[], Bernoulli)"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        [Skip]
        public static Bernoulli Case0AverageLogarithm<TValue>(IList<TValue> enterPartialTwo, TValue value, int[] indices, Bernoulli result)
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialTwoOp"]/message_doc[@name="Case1AverageLogarithm{TValue}(IList{TValue}, TValue, int[], Bernoulli)"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        [Skip]
        public static Bernoulli Case1AverageLogarithm<TValue>(IList<TValue> enterPartialTwo, TValue value, int[] indices, Bernoulli result)
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterPartialTwoOp"]/message_doc[@name="ValueAverageLogarithm{TValue}(IList{TValue}, Bernoulli, Bernoulli, int[], TValue)"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        public static TValue ValueAverageLogarithm<TValue>(
            [SkipIfAllUniform] IList<TValue> enterPartialTwo, Bernoulli case0, Bernoulli case1, int[] indices, TValue result)
            where TValue : ICloneable, SettableToProduct<TValue>, SettableToPower<TValue>
        {
            if (indices.Length != enterPartialTwo.Count)
                throw new ArgumentException("indices.Length != enterPartial.Count");
            if (2 < enterPartialTwo.Count)
                throw new ArgumentException("cases.Count < enterPartial.Count");
            if (indices.Length == 0)
                throw new ArgumentException("indices.Length == 0");
            else
            {
                double scale = Math.Exp((indices[0] == 0 ? case0 : case1).LogOdds);
                result.SetToPower(enterPartialTwo[0], scale);
                if (indices.Length > 1)
                {
                    // TODO: use pre-allocated buffer
                    TValue power = (TValue)result.Clone();
                    for (int i = 1; i < indices.Length; i++)
                    {
                        scale = Math.Exp((indices[i] == 0 ? case0 : case1).LogOdds);
                        power.SetToPower(enterPartialTwo[i], scale);
                        result.SetToProduct(result, power);
                    }
                }
            }
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOneOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of the variable entering the gate.</typeparam>
    [FactorMethod(typeof(Gate), "EnterOne<>", null, typeof(int), null, typeof(int))]
    [Quality(QualityBand.Mature)]
    public static class GateEnterOneOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOneOp{T}"]/message_doc[@name="LogEvidenceRatio{TDist}(TDist)"]/*'/>
        [Skip]
        public static double LogEvidenceRatio<TDist>(TDist enterOne)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOneOp{T}"]/message_doc[@name="LogAverageFactor()"]/*'/>
        [Skip]
        public static double LogAverageFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOneOp{T}"]/message_doc[@name="EnterOneAverageConditional{TValue}(TValue)"]/*'/>
        /// <typeparam name="TValue">The type of the incoming message from <c>value</c>.</typeparam>
        public static TValue EnterOneAverageConditional<TValue>([IsReturned] TValue value)
        {
            return value;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOneOp{T}"]/message_doc[@name="SelectorAverageConditional(Discrete)"]/*'/>
        [Skip]
        public static Discrete SelectorAverageConditional(Discrete result)
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOneOp{T}"]/message_doc[@name="ValueAverageConditional{TDist}(TDist, Discrete, TDist, int, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] TDist enterOne, Discrete selector, [NoInit,Proper] TDist value, int index, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToRatio<TDist>, SettableToWeightedSum<TDist>, SettableTo<TDist>, SettableToPower<TDist>
        {
            // TODO: improve this
            if (value.IsPointMass)
                return ValueAverageLogarithm(enterOne, selector, index, result);
            double logProb = selector.GetLogProb(index);
            if (logProb == 0.0)
            {
                result.SetTo(enterOne);
            }
            else if (double.IsNegativeInfinity(logProb))
            {
                result.SetToUniform();
            }
            else
            {
                result.SetToProduct(value, enterOne);
                double logOtherProb = MMath.Log1MinusExp(logProb);
                double shift = Math.Max(logProb, logOtherProb);
                // avoid (-Infinity) - (-Infinity)
                if (Double.IsNegativeInfinity(shift))
                    throw new AllZeroException();
                result.SetToSum(Math.Exp(logProb - shift), result, Math.Exp(logOtherProb - shift), value);
                result.SetToRatio(result, value, GateEnterOp<T>.ForceProper);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOneOp{T}"]/message_doc[@name="ValueAverageConditional{TDist}(TDist, int, int, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] TDist enterOne, int selector, int index, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            if (selector == index)
                result.SetTo(enterOne);
            else
                result.SetToUniform();
            return result;
        }

#if false
    /// <summary>
    /// EP message to 'b'
    /// </summary>
    /// <param name="result">Modified to contain the outgoing message</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the factor viewed as a function of 'b' conditioned on the given values.
    /// </para></remarks>
        [Skip]
        public static Bernoulli BAverageConditional(Bernoulli result)
        {
            result.SetToUniform();
            return result;
        }
        /// <summary>
        /// EP message to 'value'
        /// </summary>
        /// <param name="enterOne">Incoming message from 'enterOne'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="cases">Incoming message from 'cases'.</param>
        /// <param name="value">Incoming message from 'value'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="index">Constant value for 'index'.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'value' as the random arguments are varied.
        /// The formula is <c>proj[p(value) sum_(enterOne,cases) p(enterOne,cases) factor(enterOne,cases,value,index)]/p(value)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="enterOne"/> is not a proper distribution</exception>
        /// <exception cref="ImproperMessageException"><paramref name="value"/> is not a proper distribution</exception>
        public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] TDist enterOne, Bernoulli b, [Proper] TDist value, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToRatio<TDist>, SettableToWeightedSum<TDist>, SettableTo<TDist>
        {
            double logProb = b.LogOdds;
            if (logProb == 0.0) {
                result.SetTo(enterOne);
            } else if (double.IsNegativeInfinity(logProb)) {
                result.SetToUniform();
            } else {
                result.SetToProduct(value, enterOne);
                double logOtherProb = MMath.Log1MinusExp(logProb);
                double shift = Math.Max(logProb, logOtherProb);
                // avoid (-Infinity) - (-Infinity)
                if (Double.IsNegativeInfinity(shift)) throw new AllZeroException();
                result.SetToSum(Math.Exp(logProb - shift), result, Math.Exp(logOtherProb - shift), value);
                if (GateEnterOp<T>.ForceProper && (result is Gaussian)) {
                    Gaussian r = (Gaussian)(object)result;
                    r.SetToRatioProper(r, (Gaussian)(object)value);
                    result = (TDist)(object)r;
                } else {
                    result.SetToRatio(result, value);
                }
            }
            return result;
        }
        /// <summary>
        /// EP message to 'value'
        /// </summary>
        /// <param name="enterOne">Incoming message from 'enterOne'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="cases">Incoming message from 'cases'.</param>
        /// <param name="index">Constant value for 'index'.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'value' as the random arguments are varied.
        /// The formula is <c>proj[p(value) sum_(enterOne,cases) p(enterOne,cases) factor(enterOne,cases,value,index)]/p(value)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="enterOne"/> is not a proper distribution</exception>
        public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] TDist enterOne, bool b, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            if (b)
                result.SetTo(enterOne);
            else
                result.SetToUniform();
            return result;
        }
#endif

        //-- VMP ---------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOneOp{T}"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOneOp{T}"]/message_doc[@name="EnterOneAverageLogarithm{TValue}(TValue)"]/*'/>
        /// <typeparam name="TValue">The type of the incoming message from <c>value</c>.</typeparam>
        public static TValue EnterOneAverageLogarithm<TValue>([IsReturned] TValue value)
        {
            return value;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOneOp{T}"]/message_doc[@name="SelectorAverageLogarithm(Discrete)"]/*'/>
        [Skip]
        public static Discrete SelectorAverageLogarithm(Discrete result)
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOneOp{T}"]/message_doc[@name="ValueAverageLogarithm{TDist}(TDist, Discrete, int, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageLogarithm<TDist>([SkipIfUniform] TDist enterOne, Discrete selector, int index, TDist result)
            where TDist : SettableToPower<TDist>
        {
            double scale = selector[index];
            result.SetToPower(enterOne, scale);
            return result;
        }

#if false
    /// <summary>
    /// VMP message to 'b'
    /// </summary>
    /// <param name="result">Modified to contain the outgoing message</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the factor viewed as a function of 'b' conditioned on the given values.
    /// </para></remarks>
        [Skip]
        public static Bernoulli BAverageLogarithm(Bernoulli result)
        {
            result.SetToUniform();
            return result;
        }
        /// <summary>
        /// VMP message to 'value'
        /// </summary>
        /// <param name="enterOne">Incoming message from 'enterOne'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="b">Incoming message from 'b'.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        public static TDist ValueAverageLogarithm<TDist>([SkipIfUniform] TDist enterOne, Bernoulli b, TDist result)
            where TDist : SettableToPower<TDist>
        {
            double scale = Math.Exp(b.LogOdds);
            result.SetToPower(enterOne, scale);
            return result;
        }
#endif
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOp{T}"]/doc/*'/>
    [FactorMethod(typeof(Gate), "Enter<>", null, typeof(bool), null)]
    [FactorMethod(typeof(Gate), "Enter<>", null, typeof(int), null)]
    [Quality(QualityBand.Mature)]
    public static class GateEnterOp<T>
    {
        /// <summary>
        /// Force proper messages
        /// </summary>
        public static bool ForceProper = true;

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOp{T}"]/message_doc[@name="LogEvidenceRatio{TDist}(IList{TDist})"]/*'/>
        [Skip]
        public static double LogEvidenceRatio<TDist>(IList<TDist> enter)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOp{T}"]/message_doc[@name="LogAverageFactor()"]/*'/>
        [Skip]
        public static double LogAverageFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOp{T}"]/message_doc[@name="EnterAverageConditional{TValue}(TValue, TValue[])"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        public static TValue[] EnterAverageConditional<TValue>([IsReturnedInEveryElement] TValue value, TValue[] result)
        {
            return EnterAverageLogarithm(value, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOp{T}"]/message_doc[@name="EnterAverageConditional{TValue, TResultList}(TValue, TResultList)"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList EnterAverageConditional<TValue, TResultList>([IsReturnedInEveryElement] TValue value, TResultList result)
            where TResultList : CanSetAllElementsTo<TValue>
        {
            result.SetAllElementsTo(value);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOp{T}"]/message_doc[@name="EnterInit{TValue, TArray}(Discrete, TValue, IArrayFactory{TValue, TArray})"]/*'/>
        /// <typeparam name="TValue">The type of the incoming message from <c>value</c>.</typeparam>
        /// <typeparam name="TArray">The type of an array that can be produced by <paramref name="factory"/>.</typeparam>
        [Skip]
        public static TArray EnterInit<TValue, TArray>(
            Discrete selector, [IgnoreDependency] TValue value, IArrayFactory<TValue, TArray> factory)
            where TValue : ICloneable
        {
            return factory.CreateArray(selector.Dimension, i => (TValue)value.Clone());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOp{T}"]/message_doc[@name="SelectorAverageConditional(Discrete)"]/*'/>
        [Skip]
        public static Discrete SelectorAverageConditional(Discrete result)
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOp{T}"]/message_doc[@name="ValueAverageConditional{TDist}(IList{TDist}, Discrete, TDist, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] IList<TDist> enter, Discrete selector, TDist value, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>,
                SettableToRatio<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>, SettableToPower<TDist>
        {
            if (value.IsPointMass)
                return ValueAverageLogarithm(enter, selector, result);
            if (selector.Dimension != enter.Count)
                throw new ArgumentException("selector.Dimension != enter.Count");
            // TODO: use pre-allocated buffers
            double logProbSum = selector.GetLogProb(0);
            if (!double.IsNegativeInfinity(logProbSum))
            {
                result.SetToProduct(value, enter[0]);
            }
            if (selector.Dimension > 1)
            {
                TDist product = (TDist)value.Clone();
                for (int i = 1; i < selector.Dimension; i++)
                {
                    double logProb = selector.GetLogProb(i);
                    double shift = Math.Max(logProbSum, logProb);
                    // avoid (-Infinity) - (-Infinity)
                    if (Double.IsNegativeInfinity(shift))
                    {
                        if (i == selector.Dimension - 1)
                        {
                            throw new AllZeroException();
                        }
                        // do nothing
                    }
                    else
                    {
                        double productWeight = Math.Exp(logProb - shift);
                        if (productWeight > 0)
                        {
                            product.SetToProduct(value, enter[i]);
                            result.SetToSum(Math.Exp(logProbSum - shift), result, productWeight, product);
                            logProbSum = MMath.LogSumExp(logProbSum, logProb);
                        }
                    }
                }
            }
            result.SetToRatio(result, value, GateEnterOp<T>.ForceProper);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOp{T}"]/message_doc[@name="ValueAverageConditional{TDist}(IList{TDist}, int, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] IList<TDist> enter, int selector, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            result.SetTo(enter[selector]);
            return result;
        }

#if false
    /// <summary>
    /// EP message to 'cases'
    /// </summary>
    /// <param name="result">Modified to contain the outgoing message</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the factor viewed as a function of 'cases' conditioned on the given values.
    /// </para></remarks>
        [Skip]
        public static BernoulliList CasesAverageConditional<BernoulliList>(BernoulliList result)
            where BernoulliList : SettableToUniform
        {
            result.SetToUniform();
            return result;
        }
        /// <summary>
        /// EP message to 'value'
        /// </summary>
        /// <param name="enter">Incoming message from 'enter'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
        /// <param name="cases">Incoming message from 'cases'.</param>
        /// <param name="value">Incoming message from 'value'.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'value' as the random arguments are varied.
        /// The formula is <c>proj[p(value) sum_(enter,cases) p(enter,cases) factor(enter,cases,value)]/p(value)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="enter"/> is not a proper distribution</exception>
        public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] IList<TDist> enter, IList<Bernoulli> cases, TDist value, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>,
            SettableToRatio<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
        {
            if (cases.Count < enter.Count) throw new ArgumentException("cases.Count < enter.Count");
            // TODO: use pre-allocated buffers
            double logProbSum = cases[0].LogOdds;
            if (!double.IsNegativeInfinity(logProbSum)) {
                result.SetToProduct(value, enter[0]);
            }
            if (cases.Count > 1) {
                TDist product = (TDist)value.Clone();
                for (int i = 1; i < cases.Count; i++) {
                    double logProb = cases[i].LogOdds;
                    double shift = Math.Max(logProbSum, logProb);
                    // avoid (-Infinity) - (-Infinity)
                    if (Double.IsNegativeInfinity(shift)) {
                        if (i == cases.Count - 1) {
                            throw new AllZeroException();
                        }
                        // do nothing
                    } else {
                        double productWeight = Math.Exp(logProb - shift);
                        if (productWeight > 0) {
                            product.SetToProduct(value, enter[i]);
                            result.SetToSum(Math.Exp(logProbSum - shift), result, productWeight, product);
                            logProbSum = MMath.LogSumExp(logProbSum, logProb);
                        }
                    }
                }
            }
            if (ForceProper && (result is Gaussian)) {
                Gaussian r = (Gaussian)(object)result;
                r.SetToRatioProper(r, (Gaussian)(object)value);
                result = (TDist)(object)r;
            } else {
                result.SetToRatio(result, value);
            }
            return result;
        }
        /// <summary>
        /// EP message to 'value'
        /// </summary>
        /// <param name="enter">Incoming message from 'enter'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
        /// <param name="cases">Constant value for 'cases'.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is a distribution matching the moments of 'value' as the random arguments are varied.
        /// The formula is <c>proj[p(value) sum_(enter) p(enter) factor(enter,cases,value)]/p(value)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="enter"/> is not a proper distribution</exception>
        public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] IList<TDist> enter, bool[] cases, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>
        {
            if (cases.Length < enter.Count) throw new ArgumentException("cases.Count < enter.Count");
            result.SetToUniform();
            for (int i = 0; i < cases.Length; i++) {
                if (cases[i]) {
                    result.SetTo(enter[i]);
                    break;
                }
            }
            return result;
        }
#endif

        //-- VMP ---------------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOp{T}"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOp{T}"]/message_doc[@name="EnterAverageLogarithm{TValue}(TValue, TValue[])"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        public static TValue[] EnterAverageLogarithm<TValue>([IsReturnedInEveryElement] TValue value, TValue[] result)
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = value;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOp{T}"]/message_doc[@name="EnterAverageLogarithm{TValue, TResultList}(TValue, TResultList)"]/*'/>
        /// <typeparam name="TValue">The type of the message from <c>value</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList EnterAverageLogarithm<TValue, TResultList>([IsReturnedInEveryElement] TValue value, TResultList result)
            where TResultList : CanSetAllElementsTo<TValue>
        {
            result.SetAllElementsTo(value);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOp{T}"]/message_doc[@name="SelectorAverageLogarithm(Discrete)"]/*'/>
        [Skip]
        public static Discrete SelectorAverageLogarithm(Discrete result)
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateEnterOp{T}"]/message_doc[@name="ValueAverageLogarithm{TDist}(IList{TDist}, Discrete, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageLogarithm<TDist>([SkipIfAllUniform] IList<TDist> enter, Discrete selector, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToPower<TDist>
        {
            if (selector.Dimension != enter.Count)
                throw new ArgumentException("selector.Dimension != enterPartial.Count");
            double scale = selector[0];
            result.SetToPower(enter[0], scale);
            if (selector.Dimension > 1)
            {
                // TODO: use pre-allocated buffer
                TDist power = (TDist)result.Clone();
                for (int i = 1; i < selector.Dimension; i++)
                {
                    scale = selector[i];
                    power.SetToPower(enter[i], scale);
                    result.SetToProduct(result, power);
                }
            }
            return result;
        }

#if false
    /// <summary>
    /// VMP message to 'cases'
    /// </summary>
    /// <param name="result">Modified to contain the outgoing message</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the factor viewed as a function of 'cases' conditioned on the given values.
    /// </para></remarks>
        [Skip]
        public static BernoulliList CasesAverageLogarithm<BernoulliList>(BernoulliList result)
            where BernoulliList : SettableToUniform
        {
            result.SetToUniform();
            return result;
        }
        // result = prod_i enterPartial[i]^cases[indices[i]]
        /// <summary>
        /// VMP message to 'value'
        /// </summary>
        /// <param name="enter">Incoming message from 'enter'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
        /// <param name="cases">Incoming message from 'cases'.</param>
        /// <param name="result">Modified to contain the outgoing message</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except 'value'.
        /// Because the factor is deterministic, 'enter' is integrated out before taking the logarithm.
        /// The formula is <c>exp(sum_(cases) p(cases) log(sum_enter p(enter) factor(enter,cases,value)))</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="enter"/> is not a proper distribution</exception>
        public static TDist ValueAverageLogarithm<TDist>([SkipIfAllUniform] IList<TDist> enter, IList<Bernoulli> cases, TDist result)
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToPower<TDist>
        {
            if (cases.Count < enter.Count) throw new ArgumentException("cases.Count < enterPartial.Count");
            double scale = Math.Exp(cases[0].LogOdds);
            result.SetToPower(enter[0], scale);
            if (cases.Count > 1) {
                // TODO: use pre-allocated buffer
                TDist power = (TDist)result.Clone();
                for (int i = 1; i < cases.Count; i++) {
                    scale = Math.Exp(cases[i].LogOdds);
                    power.SetToPower(enter[i], scale);
                    result.SetToProduct(result, power);
                }
            }
            return result;
        }
#endif
    }
}
