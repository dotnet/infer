// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BeliefPropagationGateEnterOneOp"]/doc/*'/>
    /// <remarks>
    /// The message operators contained in this class assume that the distribution
    /// of the variable entering the gate can represent mixtures exactly.
    /// </remarks>
    [FactorMethod(typeof(Gate), "EnterOne<>", null, typeof(int), null, typeof(int))]
    [Quality(QualityBand.Stable)]
    public static class BeliefPropagationGateEnterOneOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BeliefPropagationGateEnterOneOp"]/message_doc[@name="ValueAverageConditional{TDist}(TDist, Discrete, TDist, int, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] TDist enterOne, Discrete selector, TDist value, int index, TDist result)
            where TDist : ICloneable, SettableToUniform, SettableToWeightedSum<TDist>, SettableTo<TDist>, CanGetLogAverageOf<TDist>, CanGetLogNormalizer
        {
            if (selector == null)
            {
                throw new ArgumentNullException(nameof(selector));
            }

            double logProbSum = selector.GetLogProb(index);
            if (logProbSum == 0.0)
            {
                result.SetTo(enterOne);
            }
            else if (double.IsNegativeInfinity(logProbSum))
            {
                result.SetToUniform();
            }
            else
            {
                double logProb = MMath.Log1MinusExp(logProbSum);
                double shift = Math.Max(logProbSum, logProb);

                // Avoid (-Infinity) - (-Infinity)
                if (double.IsNegativeInfinity(shift))
                {
                    throw new AllZeroException();
                }

                TDist uniform = (TDist)result.Clone();
                uniform.SetToUniform();
                result.SetToSum(Math.Exp(logProbSum - shift - enterOne.GetLogAverageOf(value)), enterOne, Math.Exp(logProb - shift + uniform.GetLogNormalizer()), uniform);
            }

            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BeliefPropagationGateEnterOp"]/doc/*'/>
    /// <remarks>
    /// The message operators contained in this class assume that the distribution
    /// of the variable entering the gate can represent mixtures exactly.
    /// </remarks>
    [FactorMethod(typeof(Gate), "Enter<>", null, typeof(bool), null)]
    [FactorMethod(typeof(Gate), "Enter<>", null, typeof(int), null)]
    [Quality(QualityBand.Stable)]
    public static class BeliefPropagationGateEnterOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BeliefPropagationGateEnterOp"]/message_doc[@name="ValueAverageConditional{TDist}(IList{TDist}, Discrete, TDist, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>([SkipIfAllUniform] IList<TDist> enter, Discrete selector, TDist value, TDist result)
            where TDist : SettableTo<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
        {
            if (enter == null)
            {
                throw new ArgumentNullException(nameof(enter));
            }

            if (selector == null)
            {
                throw new ArgumentNullException(nameof(selector));
            }

            if (selector.Dimension != enter.Count)
            {
                throw new ArgumentException("selector.Dimension != enter.Count");
            }

            // TODO: use pre-allocated buffers
            double logWeightSum = selector.GetLogProb(0);
            if (!double.IsNegativeInfinity(logWeightSum))
            {
                logWeightSum -= enter[0].GetLogAverageOf(value);
                result.SetTo(enter[0]);
            }

            if (selector.Dimension > 1)
            {
                for (int i = 1; i < selector.Dimension; i++)
                {
                    double logWeight = selector.GetLogProb(i);
                    double shift = Math.Max(logWeightSum, logWeight);

                    // Avoid (-Infinity) - (-Infinity)
                    if (double.IsNegativeInfinity(shift))
                    {
                        if (i == selector.Dimension - 1)
                        {
                            throw new AllZeroException();
                        }

                        // Do nothing
                    }
                    else
                    {
                        double logWeightShifted = logWeight - shift;
                        if (!double.IsNegativeInfinity(logWeightShifted))
                        {
                            logWeightShifted -= enter[i].GetLogAverageOf(value);
                            result.SetToSum(Math.Exp(logWeightSum - shift), result, Math.Exp(logWeightShifted), enter[i]);
                            logWeightSum = MMath.LogSumExp(logWeightSum, logWeightShifted + shift);
                        }
                    }
                }
            }

            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BeliefPropagationGateEnterPartialOp"]/doc/*'/>
    /// <remarks>
    /// The message operators contained in this class assume that the distribution
    /// of the variable entering the gate can represent mixtures exactly.
    /// </remarks>
    [FactorMethod(typeof(Gate), "EnterPartial<>", null, typeof(int), null, typeof(int[]))]
    [FactorMethod(typeof(Gate), "EnterPartial<>", null, typeof(bool), null, typeof(int[]))]
    [Quality(QualityBand.Stable)]
    public static class BeliefPropagationGateEnterPartialOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BeliefPropagationGateEnterPartialOp"]/message_doc[@name="ValueAverageConditional{TDist}(IList{TDist}, Discrete, TDist, int[], TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>(
            [SkipIfUniform] IList<TDist> enterPartial, [SkipIfUniform] Discrete selector, TDist value, int[] indices, TDist result)
            where TDist : ICloneable, SettableToUniform, SettableTo<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>, CanGetLogNormalizer
        {
            if (enterPartial == null)
            {
                throw new ArgumentNullException(nameof(enterPartial));
            }

            if (selector == null)
            {
                throw new ArgumentNullException(nameof(selector));
            }

            if (indices == null)
            {
                throw new ArgumentNullException(nameof(indices));
            }

            if (indices.Length != enterPartial.Count)
            {
                throw new ArgumentException("indices.Length != enterPartial.Count");
            }

            if (selector.Dimension < enterPartial.Count)
            {
                throw new ArgumentException("selector.Dimension < enterPartial.Count");
            }

            if (indices.Length == 0)
            {
                throw new ArgumentException("indices.Length == 0");
            }

            // TODO: use pre-allocated buffers
            double logProbSum = selector.GetLogProb(indices[0]);
            double logWeightSum = logProbSum;
            if (!double.IsNegativeInfinity(logWeightSum))
            {
                logWeightSum -= enterPartial[0].GetLogAverageOf(value);
                result.SetTo(enterPartial[0]);
            }

            if (indices.Length > 1)
            {
                for (int i = 1; i < indices.Length; i++)
                {
                    double logProb = selector.GetLogProb(indices[i]);
                    logProbSum = MMath.LogSumExp(logProbSum, logProb);
                    double shift = Math.Max(logWeightSum, logProb);

                    // Avoid (-Infinity) - (-Infinity)
                    if (double.IsNegativeInfinity(shift))
                    {
                        if (i == selector.Dimension - 1)
                        {
                            throw new AllZeroException();
                        }

                        // Do nothing
                    }
                    else
                    {
                        double logWeightShifted = logProb - shift;
                        if (!double.IsNegativeInfinity(logWeightShifted))
                        {
                            logWeightShifted -= enterPartial[i].GetLogAverageOf(value);
                            result.SetToSum(Math.Exp(logWeightSum - shift), result, Math.Exp(logWeightShifted), enterPartial[i]);
                            logWeightSum = MMath.LogSumExp(logWeightSum, logWeightShifted + shift);
                        }
                    }
                }
            }

            if (indices.Length < selector.Dimension)
            {
                double logProb = MMath.Log1MinusExp(logProbSum);
                double shift = Math.Max(logWeightSum, logProb);
                if (double.IsNegativeInfinity(shift))
                {
                    throw new AllZeroException();
                }

                var uniform = (TDist)result.Clone();
                uniform.SetToUniform();
                double logWeight = logProb + uniform.GetLogNormalizer();
                result.SetToSum(Math.Exp(logWeightSum - shift), result, Math.Exp(logWeight - shift), uniform);
            }

            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BeliefPropagationGateEnterPartialOp"]/message_doc[@name="ValueAverageConditional{TDist}(IList{TDist}, Bernoulli, TDist, int[], TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>(
            [SkipIfUniform] IList<TDist> enterPartial, [SkipIfUniform] Bernoulli selector, TDist value, int[] indices, TDist result)
            where TDist : ICloneable, SettableToUniform, SettableTo<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>, CanGetLogNormalizer
        {
            if (enterPartial == null)
            {
                throw new ArgumentNullException(nameof(enterPartial));
            }

            if (indices == null)
            {
                throw new ArgumentNullException(nameof(indices));
            }

            if (indices.Length != enterPartial.Count)
            {
                throw new ArgumentException("indices.Length != enterPartial.Count");
            }

            if (2 < enterPartial.Count)
            {
                throw new ArgumentException("enterPartial.Count should be 2 or 1");
            }

            if (indices.Length == 0)
            {
                throw new ArgumentException("indices.Length == 0");
            }

            // TODO: use pre-allocated buffers
            double logProbSum = (indices[0] == 0) ? selector.GetLogProbTrue() : selector.GetLogProbFalse();
            double logWeightSum = logProbSum;
            if (!double.IsNegativeInfinity(logProbSum))
            {
                logWeightSum -= enterPartial[0].GetLogAverageOf(value);
                result.SetTo(enterPartial[0]);
            }

            if (indices.Length > 1)
            {
                for (int i = 1; i < indices.Length; i++)
                {
                    double logProb = (indices[i] == 0) ? selector.GetLogProbTrue() : selector.GetLogProbFalse();
                    logProbSum += logProb;
                    double shift = Math.Max(logWeightSum, logProb);

                    // Avoid (-Infinity) - (-Infinity)
                    if (double.IsNegativeInfinity(shift))
                    {
                        if (i == 1)
                        {
                            throw new AllZeroException();
                        }

                        // Do nothing
                    }
                    else
                    {
                        double logWeightShifted = logProb - shift;
                        if (!double.IsNegativeInfinity(logWeightShifted))
                        {
                            logWeightShifted -= enterPartial[i].GetLogAverageOf(value);
                            result.SetToSum(Math.Exp(logWeightSum - shift), result, Math.Exp(logWeightShifted), enterPartial[i]);
                            logWeightSum = MMath.LogSumExp(logWeightSum, logWeightShifted + shift);
                        }
                    }
                }
            }

            if (indices.Length < 2)
            {
                double logProb = MMath.Log1MinusExp(logProbSum);
                double shift = Math.Max(logWeightSum, logProb);
                if (double.IsNegativeInfinity(shift))
                {
                    throw new AllZeroException();
                }

                var uniform = (TDist)result.Clone();
                uniform.SetToUniform();
                double logWeight = logProb + uniform.GetLogNormalizer();
                result.SetToSum(Math.Exp(logWeightSum - shift), result, Math.Exp(logWeight - shift), uniform);
            }

            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BeliefPropagationGateEnterPartialTwoOp"]/doc/*'/>
    /// <remarks>
    /// The message operators contained in this class assume that the distribution
    /// of the variable entering the gate can represent mixtures exactly.
    /// </remarks>
    [FactorMethod(typeof(Gate), "EnterPartialTwo<>")]
    [Quality(QualityBand.Stable)]
    public static class BeliefPropagationGateEnterPartialTwoOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BeliefPropagationGateEnterPartialTwoOp"]/message_doc[@name="ValueAverageConditional{TDist}(IList{TDist}, Bernoulli, Bernoulli, TDist, int[], TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable entering the gate.</typeparam>
        public static TDist ValueAverageConditional<TDist>(
            [SkipIfAllUniform] IList<TDist> enterPartialTwo, Bernoulli case0, Bernoulli case1, TDist value, int[] indices, TDist result)
            where TDist : ICloneable, SettableToUniform, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>, CanGetLogNormalizer
        {
            if (enterPartialTwo == null)
            {
                throw new ArgumentNullException(nameof(enterPartialTwo));
            }

            if (indices == null)
            {
                throw new ArgumentNullException(nameof(indices));
            }

            if (indices.Length != enterPartialTwo.Count)
            {
                throw new ArgumentException("indices.Length != enterPartialTwo.Count");
            }

            if (2 < enterPartialTwo.Count)
            {
                throw new ArgumentException("enterPartialTwo.Count should be 2 or 1");
            }

            if (indices.Length == 0)
            {
                throw new ArgumentException("indices.Length == 0");
            }

            // TODO: use pre-allocated buffers
            double logProb0 = (indices[0] == 0 ? case0 : case1).LogOdds;
            double logProb1 = (indices[0] == 0 ? case1 : case0).LogOdds;
            double shift = Math.Max(logProb0, logProb1);

            // Avoid (-Infinity) - (-Infinity)
            if (double.IsNegativeInfinity(shift))
            {
                throw new AllZeroException();
            }

            if (indices.Length > 1)
            {
                result.SetToSum(
                    Math.Exp(logProb0 - shift - value.GetLogAverageOf(enterPartialTwo[0])),
                    enterPartialTwo[0],
                    Math.Exp(logProb1 - shift - value.GetLogAverageOf(enterPartialTwo[1])),
                    enterPartialTwo[1]);
            }
            else
            {
                TDist uniform = (TDist)result.Clone();
                uniform.SetToUniform();
                result.SetToSum(
                    Math.Exp(logProb0 - shift - value.GetLogAverageOf(enterPartialTwo[0])),
                    enterPartialTwo[0],
                    Math.Exp(logProb1 - shift + uniform.GetLogNormalizer()),
                    uniform);
            }

            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BeliefPropagationGateExitOp"]/doc/*'/>
    /// <remarks>
    /// The message operators contained in this class assume that the distribution
    /// of the variable exiting the gate can represent mixtures exactly.
    /// </remarks>
    [FactorMethod(typeof(Gate), "Exit<>")]
    [Quality(QualityBand.Stable)]
    public static class BeliefPropagationGateExitOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BeliefPropagationGateExitOp"]/message_doc[@name="ExitAverageConditional{TDist}(IList{Bernoulli}, IList{TDist}, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TDist ExitAverageConditional<TDist>(IList<Bernoulli> cases, [SkipIfUniform] IList<TDist> values, TDist result)
            where TDist : SettableTo<TDist>, SettableToWeightedSum<TDist>
        {
            if (cases == null)
            {
                throw new ArgumentNullException(nameof(cases));
            }

            if (values == null)
            {
                throw new ArgumentNullException(nameof(values));
            }

            if (cases.Count != values.Count)
            {
                throw new ArgumentException("cases.Count != values.Count");
            }

            if (cases.Count == 0)
            {
                throw new ArgumentException("cases.Count == 0");
            }

            if (cases.Count == 1)
            {
                result.SetTo(values[0]);
            }
            else
            {
                double logResultProb = cases[0].LogOdds;
                if (double.IsNaN(logResultProb))
                {
                    throw new AllZeroException();
                }

                int resultIndex = 0;
                for (int i = 1; i < cases.Count; i++)
                {
                    double logProb = cases[i].LogOdds;
                    double shift = Math.Max(logResultProb, logProb);

                    // Avoid (-Infinity) - (-Infinity)
                    if (double.IsNegativeInfinity(shift))
                    {
                        if (i == cases.Count - 1)
                        {
                            throw new AllZeroException();
                        }

                        // Do nothing
                    }
                    else
                    {
                        double weight1 = Math.Exp(logResultProb - shift);
                        double weight2 = Math.Exp(logProb - shift);
                        if (weight2 > 0)
                        {
                            if (weight1 == 0)
                            {
                                resultIndex = i;
                                logResultProb = logProb;
                            }
                            else
                            {
                                if (resultIndex >= 0)
                                {
                                    result.SetTo(values[resultIndex]);
                                    resultIndex = -1;
                                }

                                result.SetToSum(weight1, result, weight2, values[i]);
                                logResultProb = MMath.LogSumExp(logResultProb, logProb);
                            }
                        }
                    }
                }

                if (resultIndex >= 0)
                {
                    // result is simply values[resultIndex]
                    result.SetTo(values[resultIndex]);
                }
            }

            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BeliefPropagationGateExitTwoOp"]/doc/*'/>
    /// <remarks>
    /// The message operators contained in this class assume that the distribution
    /// of the variable exiting the gate can represent mixtures exactly.
    /// </remarks>
    [FactorMethod(typeof(Gate), "ExitTwo<>")]
    [Quality(QualityBand.Stable)]
    public static class BeliefPropagationGateExitTwoOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="BeliefPropagationGateExitTwoOp"]/message_doc[@name="ExitTwoAverageConditional{TDist}(TDist, Bernoulli, Bernoulli, IList{TDist}, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable exiting the gate.</typeparam>
        public static TDist ExitTwoAverageConditional<TDist>(
            TDist exitTwo, Bernoulli case0, Bernoulli case1, [SkipIfAllUniform] IList<TDist> values, TDist result)
            where TDist : SettableTo<TDist>, SettableToWeightedSum<TDist>, CanGetLogAverageOf<TDist>
        {
            if (values == null)
            {
                throw new ArgumentNullException(nameof(values));
            }

            if (values.Count != 2)
            {
                throw new ArgumentException("values.Count != 2");
            }

            double logProb0 = case0.LogOdds;
            double logProb1 = case1.LogOdds;
            double shift = Math.Max(logProb0, logProb1);

            // Avoid (-Infinity) - (-Infinity)
            if (double.IsNegativeInfinity(shift))
            {
                throw new AllZeroException();
            }

            result.SetToSum(
                Math.Exp(logProb0 - shift - exitTwo.GetLogAverageOf(values[0])),
                values[0],
                Math.Exp(logProb1 - shift - exitTwo.GetLogAverageOf(values[1])),
                values[1]);
            return result;
        }
    }
}
