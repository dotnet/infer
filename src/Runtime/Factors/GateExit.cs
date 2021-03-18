// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#define UseRatioDir

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

#if true
    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExitingVariableOp"]/doc/*'/>
    /// <remarks><para>
    /// This factor is like <see cref="Clone.ReplicateWithMarginal{T}"/> except <c>Uses[0]</c> plays the role of <c>Def</c>,
    /// and <c>Def</c> is considered a <c>Use</c>. Needed only when a variable exits a gate in VMP.
    /// </para></remarks>
    [FactorMethod(typeof(Gate), "ExitingVariable<>")]
    [Quality(QualityBand.Mature)]
    public static class ExitingVariableOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExitingVariableOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExitingVariableOp"]/message_doc[@name="MarginalAverageLogarithm{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T MarginalAverageLogarithm<T>([IsReturned] T Use)
        {
            return Use;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExitingVariableOp"]/message_doc[@name="MarginalAverageLogarithmInit{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [Skip]
        public static T MarginalAverageLogarithmInit<T>(T Def)
            where T : ICloneable
        {
            return (T)Def.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExitingVariableOp"]/message_doc[@name="UseAverageLogarithm{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T UseAverageLogarithm<T>([IsReturned] T Def)
        {
            return Def;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ExitingVariableOp"]/message_doc[@name="DefAverageLogarithm{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T DefAverageLogarithm<T>([IsReturned] T Use)
        {
            return Use;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateExitingOp"]/doc/*'/>
    /// <remarks><para>
    /// This factor is like <see cref="Clone.Replicate{T}"/> except <c>Uses[0]</c> plays the role of <c>Def</c>,
    /// and <c>Def</c> is considered a <c>Use</c>. Needed only when a variable exits a gate in VMP.
    /// </para></remarks>
    [FactorMethod(typeof(Gate), "ReplicateExiting<>")]
    [Quality(QualityBand.Mature)]
    public static class ReplicateExitingOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateExitingOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateExitingOp"]/message_doc[@name="UsesAverageLogarithm{T}(IList{T}, T, int, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        public static T UsesAverageLogarithm<T>([AllExceptIndex] IReadOnlyList<T> Uses, T Def, int resultIndex, T result)
            where T : SettableTo<T>, SettableToProduct<T>
        {
            if (resultIndex == 0)
            {
                result.SetTo(Def);
                result = Distribution.SetToProductWithAllExcept(result, Uses, 0);
            }
            else
            {
                result.SetTo(Uses[0]);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateExitingOp"]/message_doc[@name="UsesAverageLogarithmInit{T}(T, int)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [Skip]
        public static T UsesAverageLogarithmInit<T>(T Def, int resultIndex)
            where T : ICloneable
        {
            return (T)Def.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateExitingOp"]/message_doc[@name="DefAverageLogarithm{T}(IList{T}, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T DefAverageLogarithm<T>([SkipIfAllUniform] IList<T> Uses, T result)
            where T : SettableTo<T>
        {
            result.SetTo(Uses[0]);
            return result;
        }
    }
#else
    /// <summary>
    /// Provides outgoing messages for <see cref="Gate.ExitingVariable{T}"/>, given random arguments to the function.
    /// </summary>
    /// <remarks><para>
    /// This factor is like ReplicateWithMarginal except Uses[0] plays the role of Def, and Def is
    /// considered a Use.  Needed only when a variable exits a gate in VMP.
    /// </para></remarks>
    [FactorMethod(typeof(Gate), "ExitingVariable<>")]
    [Quality(QualityBand.Preview)]
    public static class ExitingVariableOp
    {
        /// <summary>
        /// Evidence message for VMP.
        /// </summary>
        /// <returns><c>sum_x marginal(x)*log(factor(x))</c></returns>
        /// <remarks><para>
        /// The formula for the result is <c>int log(f(x)) q(x) dx</c>
        /// where <c>x = (Uses,Def,Marginal)</c>.
        /// </para></remarks>
        [Skip]
        public static double AverageLogFactor() { return 0.0; }

        /// <summary>
        /// VMP message to 'Marginal'.
        /// </summary>
        /// <param name="Uses">Incoming message from 'Uses'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is the exponential of the integral of the log-factor times incoming messages, over all arguments except 'Marginal'.
        /// The formula is <c>int log(f(Marginal,x)) q(x) dx</c> where <c>x = (Uses,Def)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="Uses"/> is not a proper distribution</exception>
        public static T MarginalAverageLogarithm<T>([SkipIfAllUniform] IList<T> Uses, T result)
        where T : SettableTo<T>
        {
            result.SetTo(Uses[0]);
            return result;
        }

        /// <summary>
        /// VMP message to 'Uses'.
        /// </summary>
        /// <param name="Uses">Incoming message from 'Uses'.</param>
        /// <param name="Def">Incoming message from 'Def'.</param>
        /// <param name="resultIndex">Index of the 'Uses' array for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is the exponential of the integral of the log-factor times incoming messages, over all arguments except 'Uses'.
        /// The formula is <c>int log(f(Uses,x)) q(x) dx</c> where <c>x = (Def,Marginal)</c>.
        /// </para></remarks>
        [SkipIfAllUniform]
        public static T UsesAverageLogarithm<T>([AllExceptIndex] IList<T> Uses, T Def, int resultIndex, T result)
                where T : SettableTo<T>, SettableToProduct<T>
        {
            if (resultIndex == 0) {
                result.SetTo(Def);
                result = Distribution.SetToProductWithAllExcept(result, Uses, 0);
            } else {
                result.SetTo(Uses[0]);
            }
            return result;
        }

        /// <summary>
        /// VMP message to 'Def'.
        /// </summary>
        /// <param name="Uses">Incoming message from 'Uses'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns><paramref name="result"/></returns>
        /// <remarks><para>
        /// The outgoing message is the exponential of the integral of the log-factor times incoming messages, over all arguments except 'Def'.
        /// The formula is <c>int log(f(Def,x)) q(x) dx</c> where <c>x = (Uses,Marginal)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="Uses"/> is not a proper distribution</exception>
        public static T DefAverageLogarithm<T>([SkipIfAllUniform] IList<T> Uses, T result)
        where T : SettableTo<T>
        {
            return MarginalAverageLogarithm(Uses, result);
        }
    }
#endif

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of the variable exiting the gate.</typeparam>
    [FactorMethod(typeof(Gate), "Exit<>")]
    [Quality(QualityBand.Mature)]
    public static class GateExitOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="LogEvidenceRatio{TDist}(TDist, IList{bool})"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<TDist>(TDist exit, IList<bool> cases)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="LogEvidenceRatio{TDist}(TDist, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static double LogEvidenceRatio<TDist>([SkipIfUniform] TDist exit, [Fresh] TDist to_exit)
            where TDist : IDistribution<T>, CanGetLogAverageOf<TDist>
        {
            return -to_exit.GetLogAverageOf(exit);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="ValuesAverageConditional{TExit, TResultList}(TExit, TResultList)"]/*'/>
        /// <typeparam name="TExit">The type of the message from <c>exit</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList ValuesAverageConditional<TExit, TResultList>([IsReturnedInEveryElement] TExit exit, TResultList result)
            where TResultList : CanSetAllElementsTo<TExit>
        {
            result.SetAllElementsTo(exit);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="ValuesAverageConditional{TExit}(TExit, TExit[])"]/*'/>
        /// <typeparam name="TExit">The type of the message from <c>exit</c>.</typeparam>
        public static TExit[] ValuesAverageConditional<TExit>([IsReturnedInEveryElement] TExit exit, TExit[] result)
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = exit;
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="CasesAverageConditional{TDist}(TDist, TDist, int)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static Bernoulli CasesAverageConditional<TDist>(
            [SkipIfUniform] TDist exit, [Indexed] TDist values, [IgnoreDependency] int resultIndex)
            where TDist : IDistribution<T>, CanGetLogAverageOf<TDist>
        {
            return Bernoulli.FromLogOdds(exit.GetLogAverageOf(values));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="ExitAverageConditional{TDist}(TDist, IList{bool}, IList{TDist}, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TDist ExitAverageConditional<TDist>(TDist exit, IList<bool> cases, [SkipIfUniform] IList<TDist> values, TDist result)
            where TDist : SettableTo<TDist>
        {
            for (int i = 0; i < cases.Count; i++)
                if (cases[i])
                {
                    result.SetTo(values[i]);
                    return result;
                }

            throw new InferRuntimeException("no case is true");
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="ExitAverageConditional{TDist}(TDist, IList{Bernoulli}, IList{TDist}, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TDist ExitAverageConditional<TDist>(TDist exit, IList<Bernoulli> cases, [SkipIfUniform] IList<TDist> values, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>, SettableToProduct<TDist>,
                SettableToRatio<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
        {
            if (cases.Count != values.Count)
                throw new ArgumentException("cases.Count != values.Count");
            if (cases.Count == 0)
                throw new ArgumentException("cases.Count == 0");
            else if (cases.Count == 1)
            {
                result.SetTo(values[0]);
            }
            else
            {
                double resultScale = exit.GetLogAverageOf(values[0]) + cases[0].LogOdds;
                if (double.IsNaN(resultScale))
                    throw new AllZeroException();
                int resultIndex = 0;
                // TODO: use pre-allocated buffer
                TDist product = (TDist)exit.Clone();
                for (int i = 1; i < cases.Count; i++)
                {
                    double scale = exit.GetLogAverageOf(values[i]) + cases[i].LogOdds;
                    double shift = Math.Max(resultScale, scale);
                    // avoid (-Infinity) - (-Infinity)
                    if (Double.IsNegativeInfinity(shift))
                    {
                        if (i == cases.Count - 1)
                        {
                            throw new AllZeroException();
                        }
                        // do nothing
                    }
                    else
                    {
                        double weight1 = Math.Exp(resultScale - shift);
                        double weight2 = Math.Exp(scale - shift);
                        if (weight2 > 0)
                        {
                            if (weight1 == 0)
                            {
                                resultIndex = i;
                                resultScale = scale;
                            }
                            else
                            {
                                if (resultIndex >= 0)
                                {
                                    result.SetToProduct(exit, values[resultIndex]);
                                    resultIndex = -1;
                                }
                                product.SetToProduct(exit, values[i]);
                                result.SetToSum(weight1, result, weight2, product);
                                resultScale = MMath.LogSumExp(resultScale, scale);
                            }
                        }
                    }
                }
                if (resultIndex >= 0)
                {
                    // result is simply values[resultIndex]
                    return values[resultIndex];
                }
                result.SetToRatio(result, exit, GateEnterOp<T>.ForceProper);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="ExitAverageConditional{TDist}(TDist, IList{Bernoulli}, IList{TDist}, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TDist ExitAverageConditional1<TDist>(TDist exit, IList<Bernoulli> cases, [SkipIfUniform] IList<TDist> values, TDist result)
            where TDist : IDistribution<T>, SettableTo<TDist>, SettableToProduct<TDist>,
                SettableToRatio<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
        {
            if (cases.Count != values.Count)
                throw new ArgumentException("cases.Count != values.Count");
            if (cases.Count == 0)
                throw new ArgumentException("cases.Count == 0");
            else if (cases.Count == 1)
            {
                result.SetTo(values[0]);
            }
            else
            {
                double resultScale = Math.Exp(exit.GetLogAverageOf(values[0]) + cases[0].LogOdds);
                if (double.IsNaN(resultScale))
                    throw new AllZeroException();
                if (resultScale > 0)
                {
                    result.SetToProduct(exit, values[0]);
                }
                // TODO: use pre-allocated buffer
                TDist product = (TDist)exit.Clone();
                for (int i = 1; i < cases.Count; i++)
                {
                    double scale = Math.Exp(exit.GetLogAverageOf(values[i]) + cases[i].LogOdds);
                    if (scale > 0)
                    {
                        product.SetToProduct(exit, values[i]);
                        result.SetToSum(resultScale, result, scale, product);
                        resultScale += scale;
                    }
                }
                result.SetToRatio(result, exit, GateEnterOp<T>.ForceProper);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="ExitAverageConditional{TDist}(TDist, IList{Bernoulli}, IList{TDist}, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TDist ExitAverageConditional2<TDist>(
            TDist exit, IList<Bernoulli> cases, [SkipIfAllUniform] IList<TDist> values, TDist result)
            where TDist : SettableTo<TDist>, ICloneable, SettableToProduct<TDist>,
                SettableToRatio<TDist>, SettableToWeightedSum<TDist>
        {
            if (cases.Count != values.Count)
                throw new ArgumentException("cases.Count != values.Count");
            if (cases.Count == 0)
                throw new ArgumentException("cases.Count == 0");
            else if (cases.Count == 1)
            {
                result.SetTo(values[0]);
            }
            else
            {
                result.SetToProduct(exit, values[0]);
                double scale = cases[0].LogOdds;
                double resultScale = scale;
                // TODO: use pre-allocated buffer
                TDist product = (TDist)exit.Clone();
                for (int i = 1; i < cases.Count; i++)
                {
                    scale = cases[i].LogOdds;
                    double shift = Math.Max(resultScale, scale);
                    // avoid (-Infinity) - (-Infinity)
                    if (Double.IsNegativeInfinity(shift))
                    {
                        if (i == cases.Count - 1)
                        {
                            throw new AllZeroException();
                        }
                        // do nothing
                    }
                    else
                    {
                        double weight1 = Math.Exp(resultScale - shift);
                        double weight2 = Math.Exp(scale - shift);
                        if (weight2 > 0)
                        {
                            product.SetToProduct(exit, values[i]);
                            result.SetToSum(weight1, result, weight2, product);
                            resultScale = MMath.LogSumExp(resultScale, scale);
                        }
                    }
                }
                result.SetToRatio(result, exit, GateEnterOp<T>.ForceProper);
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="CasesAverageConditional{TDist}(TDist, IList{T}, int)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static Bernoulli CasesAverageConditional<TDist>(TDist exit, IList<T> values, int resultIndex)
            where TDist : CanGetLogProb<T>
        {
            return Bernoulli.FromLogOdds(exit.GetLogProb(values[resultIndex]));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="ExitAverageConditional{TDist}(TDist, IList{Bernoulli}, IList{T}, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TDist ExitAverageConditional<TDist>(
            TDist exit, IList<Bernoulli> cases, IList<T> values, TDist result)
            where TDist : ICloneable, HasPoint<T>, SettableToWeightedSum<TDist>
        {
            if (cases.Count != values.Count)
                throw new ArgumentException("cases.Count != values.Count");
            if (cases.Count == 0)
                throw new ArgumentException("cases.Count == 0");
            else if (cases.Count == 1)
            {
                result.Point = values[0];
            }
            else
            {
                result.Point = values[0];
                double scale = cases[0].LogOdds;
                double resultScale = scale;
                // TODO: overload SetToSum to accept constants.
                TDist product = (TDist)exit.Clone();
                for (int i = 1; i < cases.Count; i++)
                {
                    product.Point = values[i];
                    scale = cases[i].LogOdds;
                    double shift = Math.Max(resultScale, scale);
                    // avoid (-Infinity) - (-Infinity)
                    if (Double.IsNegativeInfinity(shift))
                    {
                        if (i == cases.Count - 1)
                        {
                            throw new AllZeroException();
                        }
                        // do nothing
                    }
                    else
                    {
                        result.SetToSum(Math.Exp(resultScale - shift), result, Math.Exp(scale - shift), product);
                        resultScale = MMath.LogSumExp(resultScale, scale);
                    }
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="ExitAverageConditionalInit{TDist}(IList{TDist})"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        [Skip]
        public static TDist ExitAverageConditionalInit<TDist>([IgnoreDependency] IList<TDist> values)
            where TDist : ICloneable
        {
            return (TDist)values[0].Clone();
        }

        //-- VMP ------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="AverageLogFactor{TDist}(TDist, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static double AverageLogFactor<TDist>([SkipIfUniform] TDist exit, [Fresh] TDist to_exit)
            where TDist : IDistribution<T>, CanGetAverageLog<TDist>
        {
            // cancel the evidence message from the child variable's child factors
            return -to_exit.GetAverageLog(exit);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="ValuesAverageLogarithm{TExit, TResultList}(TExit, TExit[])"]/*'/>
        /// <typeparam name="TExit">The type of the message from <c>exit</c>.</typeparam>
        public static TExit[] ValuesAverageLogarithm<TExit>([IsReturnedInEveryElement] TExit exit, TExit[] result)
        {
            return ValuesAverageConditional(exit, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="ValuesAverageLogarithm{TExit, TResultList}(TExit, TResultList)"]/*'/>
        /// <typeparam name="TExit">The type of the message from <c>exit</c>.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList ValuesAverageLogarithm<TExit, TResultList>([IsReturnedInEveryElement] TExit exit, TResultList result)
            where TResultList : CanSetAllElementsTo<TExit>
        {
            return ValuesAverageConditional(exit, result);
        }

#if true

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="CasesAverageLogarithm{TDist}(TDist, IList{TDist}, int)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        [NoTriggers] // see VmpTests.GateExitTriggerTest
        public static Bernoulli CasesAverageLogarithm<TDist>(
            [SkipIfUniform] TDist exit, [SkipIfAllUniform, Proper, Trigger] IList<TDist> values, int resultIndex)
            where TDist : CanGetAverageLog<TDist>
        {
            return Bernoulli.FromLogOdds(values[resultIndex].GetAverageLog(exit));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="ExitAverageLogarithm{TDist}(TDist, IList{bool}, IList{TDist}, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TDist ExitAverageLogarithm<TDist>(TDist exit, IList<bool> cases, [SkipIfUniform] IList<TDist> values, TDist result)
            where TDist : SettableTo<TDist>
        {
            return ExitAverageConditional(exit, cases, values, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="ExitAverageLogarithm{TDist}(IList{Bernoulli}, IList{TDist}, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TDist ExitAverageLogarithm<TDist>(IList<Bernoulli> cases, [SkipIfAllUniform, Proper] IList<TDist> values, TDist result)
            where TDist : ICloneable, SettableToProduct<TDist>,
                SettableToPower<TDist>, CanGetAverageLog<TDist>,
                SettableToUniform, SettableTo<TDist>, SettableToRatio<TDist>, SettableToWeightedSum<TDist>
        {
            // result = prod_i values[i]^cases[i]  (messages out of a gate are blurred)
#if DEBUG
            if (cases.Count != values.Count)
                throw new ArgumentException("cases.Count != values.Count");
#endif
            TDist uniform = (TDist)result.Clone();
            uniform.SetToUniform();
            return ExitAverageConditional2<TDist>(uniform, cases, values, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitOp{T}"]/message_doc[@name="ExitAverageLogarithmInit{TDist}(IList{TDist})"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        [Skip]
        public static TDist ExitAverageLogarithmInit<TDist>([IgnoreDependency] IList<TDist> values)
            where TDist : ICloneable
        {
            return (TDist)values[0].Clone();
        }
#else
        [Skip]
        public static DistributionArray<Bernoulli> CasesAverageLogarithm(DistributionArray<Bernoulli> result)
        {
            return result;
        }
        // result = prod_i values[i]^cases[i]  (messages out of a gate are blurred)
        public static T ExitAverageLogarithm<T>(DistributionArray<Bernoulli> cases, [SkipIfUniform] DistributionArray<T> values, T result)
            where T : Diffable, SettableTo<T>, ICloneable, SettableToUniform, SettableToProduct<T>,
            SettableToPower<T>, SettableToRatio<T>, SettableToWeightedSum<T>, LogInnerProductable<T>, CanGetAverageLog<T>
        {
            if (cases.Count != values.Count) throw new ArgumentException("cases.Count != values.Count");
            if (cases.Count == 0) throw new ArgumentException("cases.Count == 0");
            else {
                result.SetToPower(values[0], cases[0].LogOdds);
                if (cases.Count > 1) {
                    // TODO: use pre-allocated buffer
                    T power = (T)result.Clone();
                    for (int i = 1; i < cases.Count; i++) {
                        power.SetToPower(values[i], cases[i].LogOdds);
                        result.SetToProduct(result, power);
                    }
                }
            }
            return result;
        }
#endif
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitTwoOp"]/doc/*'/>
    [FactorMethod(typeof(Gate), "ExitTwo<>")]
    [Quality(QualityBand.Mature)]
    public static class GateExitTwoOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitTwoOp"]/message_doc[@name="ValuesAverageConditional{TExit, TResultList}(TExit, TResultList)"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList ValuesAverageConditional<TExit, TResultList>([SkipIfUniform] TExit exitTwo, TResultList result)
            where TResultList : CanSetAllElementsTo<TExit>
        {
            result.SetAllElementsTo(exitTwo);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitTwoOp"]/message_doc[@name="Case0AverageConditional{TExit}(IList{TExit})"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        [Skip]
        public static Bernoulli Case0AverageConditional<TExit>([SkipIfAllUniform] IList<TExit> values)
        {
            // must takes values as input to distinguish from the other overload.
            return Bernoulli.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitTwoOp"]/message_doc[@name="Case1AverageConditional{TExit}(IList{TExit})"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        [Skip]
        public static Bernoulli Case1AverageConditional<TExit>([SkipIfAllUniform] IList<TExit> values)
        {
            return Bernoulli.Uniform();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitTwoOp"]/message_doc[@name="ExitTwoAverageConditional{TExit}(TExit, Bernoulli, Bernoulli, IList{TExit}, TExit)"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TExit ExitTwoAverageConditional<TExit>(
            TExit exitTwo, Bernoulli case0, Bernoulli case1, [SkipIfAllUniform] IList<TExit> values, TExit result)
            where TExit : SettableTo<TExit>, ICloneable, SettableToProduct<TExit>, SettableToRatio<TExit>, SettableToWeightedSum<TExit>
        {
            result.SetToProduct(exitTwo, values[0]);
            double scale = case0.LogOdds;
            double resultScale = scale;
            // TODO: use pre-allocated buffer
            TExit product = (TExit)exitTwo.Clone();
            product.SetToProduct(exitTwo, values[1]);
            scale = case1.LogOdds;
            double shift = Math.Max(resultScale, scale);
            // avoid (-Infinity) - (-Infinity)
            if (Double.IsNegativeInfinity(shift))
            {
                throw new AllZeroException();
            }
            else
            {
                result.SetToSum(Math.Exp(resultScale - shift), result, Math.Exp(scale - shift), product);
                resultScale = MMath.LogSumExp(resultScale, scale);
            }
            result.SetToRatio(result, exitTwo);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitTwoOp"]/message_doc[@name="Case0AverageConditional{TExit, TExitDomain}(TExit, IList{TExitDomain})"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TExitDomain">The domain of the variable exiting the gate.</typeparam>
        public static Bernoulli Case0AverageConditional<TExit, TExitDomain>(TExit exitTwo, IList<TExitDomain> values)
            where TExit : CanGetLogProb<TExitDomain>
        {
            return Bernoulli.FromLogOdds(exitTwo.GetLogProb(values[0]));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitTwoOp"]/message_doc[@name="Case1AverageConditional{TExit, TExitDomain}(TExit, IList{TExitDomain})"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TExitDomain">The domain of the variable exiting the gate.</typeparam>
        public static Bernoulli Case1AverageConditional<TExit, TExitDomain>(TExit exitTwo, IList<TExitDomain> values)
            where TExit : CanGetLogProb<TExitDomain>
        {
            return Bernoulli.FromLogOdds(exitTwo.GetLogProb(values[1]));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitTwoOp"]/message_doc[@name="ExitTwoAverageConditional{TExit, TExitDomain}(TExit, Bernoulli, Bernoulli, IList{TExitDomain}, TExit)"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TExitDomain">The domain of the variable exiting the gate.</typeparam>
        public static TExit ExitTwoAverageConditional<TExit, TExitDomain>(
            TExit exitTwo, Bernoulli case0, Bernoulli case1, IList<TExitDomain> values, TExit result)
            where TExit : ICloneable, HasPoint<TExitDomain>, SettableToWeightedSum<TExit>
        {
            result.Point = values[0];
            double scale = case0.LogOdds;
            double resultScale = scale;
            // TODO: overload SetToSum to accept constants.
            TExit product = (TExit)exitTwo.Clone();
            product.Point = values[1];
            scale = case1.LogOdds;
            double shift = Math.Max(resultScale, scale);
            // avoid (-Infinity) - (-Infinity)
            if (Double.IsNegativeInfinity(shift))
            {
                throw new AllZeroException();
                // do nothing
            }
            else
            {
                result.SetToSum(Math.Exp(resultScale - shift), result, Math.Exp(scale - shift), product);
                resultScale = MMath.LogSumExp(resultScale, scale);
            }
            return result;
        }

        //-- VMP ------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitTwoOp"]/message_doc[@name="AverageLogFactor{TExit}(TExit, Bernoulli, Bernoulli, IList{TExit}, TExit)"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static double AverageLogFactor<TExit>(
            TExit exitTwo, Bernoulli case0, Bernoulli case1, IList<TExit> values, [Fresh] TExit to_exitTwo)
            where TExit : ICloneable, SettableToProduct<TExit>,
                SettableToPower<TExit>, CanGetAverageLog<TExit>,
                SettableToUniform, SettableTo<TExit>, SettableToRatio<TExit>, SettableToWeightedSum<TExit>
        {
            // cancel the evidence message from the child variable's child factors
            //T to_exit = (T)exitTwo.Clone();
            //to_exit = ExitTwoAverageLogarithm(case0, case1, values, to_exit);
            return -to_exitTwo.GetAverageLog(exitTwo);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitTwoOp"]/message_doc[@name="ValuesAverageLogarithm{TExit, TResultList}(TExit, TResultList)"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList ValuesAverageLogarithm<TExit, TResultList>([SkipIfUniform] TExit exitTwo, TResultList result)
            where TResultList : CanSetAllElementsTo<TExit>
        {
            // result is always exit (messages into a gate are unchanged)
            result.SetAllElementsTo(exitTwo);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitTwoOp"]/message_doc[@name="Case0AverageLogarithm{TExit}(TExit, IList{TExit})"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static Bernoulli Case0AverageLogarithm<TExit>(TExit exitTwo, [SkipIfAllUniform, Proper] IList<TExit> values)
            where TExit : CanGetAverageLog<TExit>
        {
            return Bernoulli.FromLogOdds(values[0].GetAverageLog(exitTwo));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitTwoOp"]/message_doc[@name="Case1AverageLogarithm{TExit}(TExit, IList{TExit})"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static Bernoulli Case1AverageLogarithm<TExit>(TExit exitTwo, [SkipIfAllUniform, Proper] IList<TExit> values)
            where TExit : CanGetAverageLog<TExit>
        {
            return Bernoulli.FromLogOdds(values[1].GetAverageLog(exitTwo));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitTwoOp"]/message_doc[@name="ExitTwoAverageLogarithm{TExit}(Bernoulli, Bernoulli, IList{TExit}, TExit)"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TExit ExitTwoAverageLogarithm<TExit>(
            Bernoulli case0, Bernoulli case1, [SkipIfAllUniform, Proper] IList<TExit> values, TExit result)
            where TExit : ICloneable, SettableToProduct<TExit>,
                SettableToPower<TExit>, CanGetAverageLog<TExit>,
                SettableToUniform, SettableTo<TExit>, SettableToRatio<TExit>, SettableToWeightedSum<TExit>
        {
            // result = prod_i values[i]^cases[i]  (messages out of a gate are blurred)
            TExit uniform = (TExit)result.Clone();
            uniform.SetToUniform();
            return ExitTwoAverageConditional<TExit>(uniform, case0, case1, values, result);
        }
    }


    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitRandomOp"]/doc/*'/>
    [FactorMethod(typeof(Gate), "ExitRandom<>")]
    [Quality(QualityBand.Mature)]
    public static class GateExitRandomOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitRandomOp"]/message_doc[@name="LogAverageFactor()"]/*'/>
        [Skip]
        public static double LogAverageFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitRandomOp"]/message_doc[@name="ValuesAverageConditional{TExit}(TExit, TExit[])"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TExit[] ValuesAverageConditional<TExit>([IsReturnedInEveryElement] TExit exit, TExit[] result)
        {
            for (int i = 0; i < result.Length; i++)
                result[i] = exit;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitRandomOp"]/message_doc[@name="ValuesAverageConditional{TExit, TResultList}(TExit, TResultList)"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList ValuesAverageConditional<TExit, TResultList>([IsReturnedInEveryElement] TExit exit, TResultList result)
            where TResultList : CanSetAllElementsTo<TExit>
        {
            // result is always exit (messages into a gate are unchanged)
            return ValuesAverageLogarithm(exit, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitRandomOp"]/message_doc[@name="CasesAverageConditional{TResultList}(TResultList)"]/*'/>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        [Skip]
        public static TResultList CasesAverageConditional<TResultList>(TResultList result)
            where TResultList : SettableToUniform
        {
            return CasesAverageLogarithm<TResultList>(result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitRandomOp"]/message_doc[@name="ExitAverageConditional{TExit}(bool[], IList{TExit})"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TExit ExitAverageConditional<TExit>(bool[] cases, [SkipIfAllUniform] IList<TExit> values)
        {
            for (int i = 0; i < cases.Length; i++)
            {
                if (cases[i])
                    return values[i];
            }
            // cases was entirely false
            throw new ArgumentException("cases is all false");
        }

#if false
        /// <summary>
        /// Gibbs message to 'Exit'
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="TDomain"></typeparam>
        /// <param name="cases"></param>
        /// <param name="values"></param>
        /// <param name="result"></param>
        /// <returns></returns>
        public static T ExitAverageConditional<T, TDomain>(IList<bool> cases, IList<TDomain> values, T result)
            where T : IDistribution<TDomain>
        {
            for (int i = 0; i < cases.Count; i++)
            {
                if (cases[i]) return Distribution.SetPoint<T, TDomain>(result, values[i]);
            }
            // cases was entirely false
            throw new ArgumentException("cases is all false");
        }
#endif

        //-- VMP ------------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitRandomOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitRandomOp"]/message_doc[@name="ValuesAverageLogarithm{TExit}(TExit, TExit[])"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TExit[] ValuesAverageLogarithm<TExit>([IsReturnedInEveryElement] TExit exit, TExit[] result)
        {
            return ValuesAverageConditional(exit, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitRandomOp"]/message_doc[@name="ValuesAverageLogarithm{TExit, TResultList}(TExit, TResultList)"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        public static TResultList ValuesAverageLogarithm<TExit, TResultList>([IsReturnedInEveryElement] TExit exit, TResultList result)
            where TResultList : CanSetAllElementsTo<TExit>
        {
            // result is always exit (messages into a gate are unchanged)
            result.SetAllElementsTo(exit);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitRandomOp"]/message_doc[@name="CasesAverageLogarithm{TResultList}(TResultList)"]/*'/>
        /// <typeparam name="TResultList">The type of the outgoing message.</typeparam>
        [Skip]
        public static TResultList CasesAverageLogarithm<TResultList>(TResultList result)
            where TResultList : SettableToUniform
        {
            result.SetToUniform();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitRandomOp"]/message_doc[@name="ExitAverageLogarithm{TExit}(IList{Bernoulli}, IList{TExit}, TExit)"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TExit ExitAverageLogarithm<TExit>(IList<Bernoulli> cases, [SkipIfAllUniform, Proper] IList<TExit> values, TExit result)
            where TExit : ICloneable, SettableToProduct<TExit>,
                SettableToPower<TExit>, CanGetAverageLog<TExit>,
                SettableToUniform, SettableTo<TExit>, SettableToRatio<TExit>, SettableToWeightedSum<TExit>
        {
            if (cases.Count != values.Count)
                throw new ArgumentException("cases.Count != values.Count");
            if (cases.Count == 0)
                throw new ArgumentException("cases.Count == 0");
            else
            {
                // result = prod_i values[i]^cases[i]  (messages out of a gate are blurred)
                result.SetToPower(values[0], Math.Exp(cases[0].LogOdds));
                if (cases.Count > 1)
                {
                    // TODO: use pre-allocated buffer
                    TExit power = (TExit)result.Clone();
                    for (int i = 1; i < cases.Count; i++)
                    {
                        power.SetToPower(values[i], Math.Exp(cases[i].LogOdds));
                        result.SetToProduct(result, power);
                    }
                }
            }
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="GateExitRandomOp"]/message_doc[@name="ExitAverageLogarithmInit{TExit}(IList{TExit})"]/*'/>
        /// <typeparam name="TExit">The type of the distribution over the variable exiting the gate.</typeparam>
        [Skip]
        public static TExit ExitAverageLogarithmInit<TExit>([IgnoreDependency] IList<TExit> values)
            where TExit : ICloneable
        {
            return (TExit)values[0].Clone();
        }
    }
}
