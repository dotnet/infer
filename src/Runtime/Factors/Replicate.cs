// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#define SpecializeArrays
#define MinimalGenericTypeParameters

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_Divide"]/doc/*'/>
    [FactorMethod(typeof(Clone), "Replicate<>", Default = true)]
    [Buffers("marginal", "toDef")]
    [Quality(QualityBand.Mature)]
    public static class ReplicateOp_Divide
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_Divide"]/message_doc[@name="DefAverageConditional{T}(T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T DefAverageConditional<T>([IsReturned] T toDef, T result)
            where T : SettableTo<T>
        {
            result.SetTo(toDef);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_Divide"]/message_doc[@name="UsesAverageConditional{T}(T, T, int, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        // Uses is marked Cancels because the forward message does not really depend on the backward message
        public static T UsesAverageConditional<T>([Indexed, Cancels] T Uses, [SkipIfUniform] T marginal, [IgnoreDependency] int resultIndex, T result)
            where T : SettableToRatio<T>
        {
            result.SetToRatio(marginal, Uses);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_Divide"]/message_doc[@name="MarginalInit{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [Skip]  // must have Skip since marginal is Fresh
        public static T MarginalInit<T>([SkipIfUniform] T Def)
            where T : ICloneable
        {
            return (T)Def.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_Divide"]/message_doc[@name="Marginal{T}(T, T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        [Fresh]
        public static T Marginal<T>(T toDef, T Def, T result)
            where T : SettableToProduct<T>
        {
            result.SetToProduct(Def, toDef);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_Divide"]/message_doc[@name="MarginalIncrement{T}(T, T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        // SkipIfUniform on 'use' causes this line to be pruned when the backward message isn't changing
        [SkipIfAllUniform]
        [MultiplyAll]
        [Fresh]
        public static T MarginalIncrement<T>(T result, [InducedSource] T def, [SkipIfUniform, InducedTarget] T use)
            where T : SettableToProduct<T>
        {
            result.SetToProduct(use, def);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_Divide"]/message_doc[@name="ToDefInit{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [Skip] // this is needed to instruct the scheduler to treat the buffer as uninitialized
        public static T ToDefInit<T>(T Def)
            where T : ICloneable, SettableToUniform
        {
            // must construct from Def instead of Uses because Uses array may be empty
            return ArrayHelper.MakeUniform(Def);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_Divide"]/message_doc[@name="ToDef{T}(IList{T}, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        [Fresh]
        public static T ToDef<T>(IReadOnlyList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="Replicate2BufferOp"]/doc/*'/>
    [FactorMethod(typeof(Clone), "Replicate<>", Default = false)]
    [Buffers("marginal")]
    [Quality(QualityBand.Preview)]
    public static class Replicate2BufferOp
    {
#if false
    /// <summary>
    /// EP message to 'Uses'
    /// </summary>
    /// <param name="Uses">Incoming message from 'Uses'.</param>
    /// <param name="marginal">Buffer 'marginal'.</param>
    /// <param name="result">Modified to contain the outgoing message</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the factor viewed as a function of 'Uses' conditioned on the given values.
    /// </para></remarks>
    //[SkipIfAllUniform]
        public static TListRet UsesAverageConditional<T, TList, TListRet>(TList Uses, [Fresh, SkipIfUniform] T marginal, TListRet result)
            where TList : IList<T>
            where TListRet : IList<T>
            where T : SettableToRatio<T>
        {
            for (int i = 0; i < result.Count; i++) {
                T dist = result[i];
                dist.SetToRatio(marginal, Uses[i]);
                result[i] = dist;
            }
            return result;
        }
#endif

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="Replicate2BufferOp"]/message_doc[@name="UsesAverageConditional{T}(IList{T}, T, T, int, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        //[SkipIfAllUniform]
        public static T UsesAverageConditional<T>(
            [MatchingIndex, IgnoreDependency] IReadOnlyList<T> Uses, // Uses dependency must be ignored for Sequential schedule
            [IgnoreDependency, SkipIfUniform] T Def,
            [Fresh, SkipIfUniform] T marginal,
            int resultIndex,
            T result)
            where T : SettableToRatio<T>, SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Count)
                throw new ArgumentOutOfRangeException(nameof(resultIndex));
            if (Uses.Count == 1)
            {
                result.SetTo(Def);
                return result;
            }
            if (true)
            {
                try
                {
                    result.SetToRatio(marginal, Uses[resultIndex]);
                    return result;
                }
                catch (DivideByZeroException)
                {
                    return ReplicateOp_NoDivide.UsesAverageConditional(Uses, Def, resultIndex, result);
                }
            }
            else
            {
                // check that ratio is same as product
                result.SetToRatio(marginal, Uses[resultIndex]);
                T result2 = (T)((ICloneable)result).Clone();
                ReplicateOp_NoDivide.UsesAverageConditional(Uses, Def, resultIndex, result2);
                double err = ((Diffable)result).MaxDiff(result2);
                if (err > 1e-4)
                    Console.WriteLine(err);
                return result;
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

#if SpecializeArrays
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="Replicate2BufferOp"]/message_doc[@name="UsesAverageConditional{T}(T[], T, T, int, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        //[SkipIfAllUniform]
        public static T UsesAverageConditional<T>(
            [MatchingIndex, IgnoreDependency] T[] Uses, // Uses dependency must be ignored for Sequential schedule
            [IgnoreDependency, SkipIfUniform] T Def,
            [Fresh, SkipIfUniform] T marginal,
            int resultIndex,
            T result)
            where T : SettableToRatio<T>, SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Length)
                throw new ArgumentOutOfRangeException(nameof(resultIndex));
            if (Uses.Length == 1)
            {
                result.SetTo(Def);
                return result;
            }
            try
            {
                result.SetToRatio(marginal, Uses[resultIndex]);
                return result;
            }
            catch (DivideByZeroException)
            {
                return ReplicateOp_NoDivide.UsesAverageConditional(Uses, Def, resultIndex, result);
            }
        }
#endif
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="Replicate2BufferOp"]/message_doc[@name="UsesAverageConditionalInit{T, ArrayType}(T, int, IArrayFactory{T, ArrayType})"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        /// <typeparam name="ArrayType">The type of arrays produced by <paramref name="factory"/>.</typeparam>
        [Skip]
        public static ArrayType UsesAverageConditionalInit<T, ArrayType>(
            [IgnoreDependency] T Def, int count, IArrayFactory<T, ArrayType> factory)
            where T : ICloneable
        {
            return factory.CreateArray(count, i => (T)Def.Clone());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="Replicate2BufferOp"]/message_doc[@name="MarginalInit{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [Skip] // this is needed to instruct the scheduler to treat marginal as uninitialized
        public static T MarginalInit<T>([SkipIfUniform] T Def)
            where T : ICloneable
        {
            return (T)Def.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="Replicate2BufferOp"]/message_doc[@name="Marginal{T}(IList{T}, T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T Marginal<T>(IReadOnlyList<T> Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            return ReplicateOp_NoDivide.MarginalAverageConditional(Uses, Def, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="Replicate2BufferOp"]/message_doc[@name="MarginalIncrement{T}(T, T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T MarginalIncrement<T>(T result, [SkipIfUniform] T use, [SkipIfUniform] T def)
            where T : SettableToProduct<T>
        {
            result.SetToProduct(use, def);
            return result;
        }

#if MinimalGenericTypeParameters
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="Replicate2BufferOp"]/message_doc[@name="DefAverageConditional{T}(IList{T}, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [MultiplyAll]
        public static T DefAverageConditional<T>([SkipIfAllUniform] IReadOnlyList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }

#if SpecializeArrays
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="Replicate2BufferOp"]/message_doc[@name="DefAverageConditional{T}(T[], T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [MultiplyAll]
        public static T DefAverageConditional<T>([SkipIfAllUniform] T[] Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#endif
#else
            public static T DefAverageConditional<T,TUses>([SkipIfAllUniform] IList<TUses> Uses, T result)
            where T : SettableToProduct<TUses>, SettableTo<TUses>, TUses, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#if SpecializeArrays
        public static T DefAverageConditional<T,TUses>([SkipIfAllUniform] TUses[] Uses, T result)
            where T : SettableToProduct<TUses>, SettableTo<TUses>, TUses, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#endif
#endif
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateBufferOp"]/doc/*'/>
    [FactorMethod(typeof(Clone), "ReplicateWithMarginal<>", Default = true)]
    [Quality(QualityBand.Preview)]
    public static class ReplicateBufferOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateBufferOp"]/message_doc[@name="UsesAverageConditional{T}(IList{T}, T, T, int, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the replicated variable.</typeparam>
        //[SkipIfAllUniform]
        public static T UsesAverageConditional<T>([AllExceptIndex] IReadOnlyList<T> Uses, [SkipIfUniform] T Def, [SkipIfUniform, Fresh] T to_marginal, int resultIndex, T result)
            where T : SettableToRatio<T>, SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Count)
                throw new ArgumentOutOfRangeException(nameof(resultIndex));
            if (Uses.Count == 1)
            {
                result.SetTo(Def);
                return result;
            }
            try
            {
                result.SetToRatio(to_marginal, Uses[resultIndex]);
                return result;
            }
            catch (DivideByZeroException)
            {
                return ReplicateOp_NoDivide.UsesAverageConditional(Uses, Def, resultIndex, result);
            }
        }

#if SpecializeArrays
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateBufferOp"]/message_doc[@name="UsesAverageConditional{T}(T[], T, T, int, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the replicated variable.</typeparam>
        //[SkipIfAllUniform]
        public static T UsesAverageConditional<T>([AllExceptIndex] T[] Uses, [SkipIfUniform] T Def, [SkipIfUniform, Fresh] T to_marginal, int resultIndex, T result)
            where T : SettableToRatio<T>, SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Length)
                throw new ArgumentOutOfRangeException(nameof(resultIndex));
            if (Uses.Length == 1)
            {
                result.SetTo(Def);
                return result;
            }
            try
            {
                result.SetToRatio(to_marginal, Uses[resultIndex]);
                return result;
            }
            catch (DivideByZeroException)
            {
                return ReplicateOp_NoDivide.UsesAverageConditional(Uses, Def, resultIndex, result);
            }
        }
#endif
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_NoDivide"]/doc/*'/>
    [FactorMethod(typeof(Clone), "Replicate<>", Default = false)]
    [FactorMethod(typeof(Clone), "ReplicateWithMarginal<>", Default = false)]
    [Quality(QualityBand.Mature)]
    public static class ReplicateOp_NoDivide
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_NoDivide"]/message_doc[@name="MarginalAverageConditional{T}(IList{T}, T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T MarginalAverageConditional<T>(IReadOnlyList<T> Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            result.SetTo(Def);
            return Distribution.SetToProductWithAll(result, Uses);
        }

#if SpecializeArrays
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_NoDivide"]/message_doc[@name="MarginalAverageConditional{T}(T[], T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T MarginalAverageConditional<T>(T[] Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            result.SetTo(Def);
            return Distribution.SetToProductWithAll(result, Uses);
        }
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_NoDivide"]/message_doc[@name="UsesAverageConditional{T}(IList{T}, T, int, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        public static T UsesAverageConditional<T>([AllExceptIndex] IReadOnlyList<T> Uses, T Def, int resultIndex, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Count)
                throw new ArgumentOutOfRangeException(nameof(resultIndex));
            result.SetTo(Def);
            return Distribution.SetToProductWithAllExcept(result, Uses, resultIndex);
        }

#if SpecializeArrays
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_NoDivide"]/message_doc[@name="UsesAverageConditional{T}(T[], T, int, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        public static T UsesAverageConditional<T>([AllExceptIndex] T[] Uses, T Def, int resultIndex, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Length)
                throw new ArgumentOutOfRangeException(nameof(resultIndex));
            result.SetTo(Def);
            return Distribution.SetToProductWithAllExcept(result, Uses, resultIndex);
        }
#endif

#if MinimalGenericTypeParameters
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_NoDivide"]/message_doc[@name="DefAverageConditional{T}(IList{T}, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [MultiplyAll]
        public static T DefAverageConditional<T>([SkipIfAllUniform] IReadOnlyList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }

#if SpecializeArrays
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp_NoDivide"]/message_doc[@name="DefAverageConditional{T}(T[], T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [MultiplyAll]
        public static T DefAverageConditional<T>([SkipIfAllUniform] T[] Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#endif
#else
            public static T DefAverageConditional<T,TUses>([SkipIfAllUniform] IList<TUses> Uses, T result)
            where T : SettableToProduct<TUses>, SettableTo<TUses>, TUses, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#if SpecializeArrays
        public static T DefAverageConditional<T,TUses>([SkipIfAllUniform] TUses[] Uses, T result)
            where T : SettableToProduct<TUses>, SettableTo<TUses>, TUses, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#endif
#endif
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp"]/doc/*'/>
    [FactorMethod(typeof(Clone), "Replicate<>", Default = true)]
    [FactorMethod(typeof(Clone), "ReplicateWithMarginal<>", Default = true)]
    [Quality(QualityBand.Mature)]
    public static class ReplicateOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp"]/message_doc[@name="LogAverageFactor()"]/*'/>
        [Skip]
        public static double LogAverageFactor()
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp"]/message_doc[@name="LogEvidenceRatio{T}(IList{T}, T, IList{T})"]/*'/>
        /// <typeparam name="T">The type of the distribution over the replicated variable.</typeparam>
        public static double LogEvidenceRatio<T>([SkipIfAllUniform] IList<T> Uses, T Def, [Fresh] IList<T> to_Uses)
            where T : CanGetLogAverageOf<T>, SettableToProduct<T>, SettableTo<T>, ICloneable, SettableToUniform
        {
            return UsesEqualDefOp.LogEvidenceRatio(Uses, Def, to_Uses);
        }

        //-- VMP ----------------------------------------------------------------------------------------------

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp"]/message_doc[@name="AverageLogFactor()"]/*'/>
        [Skip]
        public static double AverageLogFactor()
        {
            // Deterministic variables send no evidence messages.
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp"]/message_doc[@name="MarginalAverageLogarithm{T, TDef}(TDef, T)"]/*'/>
        /// <typeparam name="T">The type of the outgoing message.</typeparam>
        /// <typeparam name="TDef">The type of the incoming message from <c>Def</c>.</typeparam>
        public static T MarginalAverageLogarithm<T, TDef>([SkipIfAllUniform] TDef Def, T result)
            where T : SettableTo<TDef>
        {
            return UsesAverageLogarithm<T, TDef>(Def, 0, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp"]/message_doc[@name="UsesAverageLogarithm{T, TDef}(TDef, int, T)"]/*'/>
        /// <typeparam name="T">The type of the outgoing message.</typeparam>
        /// <typeparam name="TDef">The type of the incoming message from <c>Def</c>.</typeparam>
        public static T UsesAverageLogarithm<T, TDef>([IsReturned] TDef Def, int resultIndex, T result)
            where T : SettableTo<TDef>
        {
            result.SetTo(Def);
            return result;
        }

        [Skip]
        public static T UsesDeriv<T>(T result)
            where T : SettableToUniform
        {
            result.SetToUniform();
            return result;
        }

        [Skip]
        public static T[] UsesDeriv<T>(T[] result)
            where T : SettableToUniform
        {
            for (int i = 0; i < result.Length; i++)
            {
                T item = result[i];
                item.SetToUniform();
                result[i] = item;
            }
            return result;
        }

        public static T UsesAverageLogarithm2<T, TDef>([IsReturnedInEveryElement] TDef Def, T result)
            where T : CanSetAllElementsTo<TDef>
        {
            result.SetAllElementsTo(Def);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp"]/message_doc[@name="UsesInit{T, ArrayType}(T, int, IArrayFactory{T, ArrayType})"]/*'/>
        /// <typeparam name="T">The type of the incoming message from <c>Def</c>.</typeparam>
        /// <typeparam name="ArrayType">The type of arrays produced by <paramref name="factory"/>.</typeparam>
        [Skip]
        public static ArrayType UsesInit<T, ArrayType>([IgnoreDependency] T Def, int count, IArrayFactory<T, ArrayType> factory)
            where T : ICloneable
        {
            return factory.CreateArray(count, i => (T)Def.Clone());
        }

#if MinimalGenericTypeParameters
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp"]/message_doc[@name="DefAverageLogarithm{T}(IList{T}, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [MultiplyAll]
        public static T DefAverageLogarithm<T>([SkipIfAllUniform] IReadOnlyList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return ReplicateOp_NoDivide.DefAverageConditional(Uses, result);
        }

#if SpecializeArrays
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateOp"]/message_doc[@name="DefAverageLogarithm{T}(T[], T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        // must have upward Trigger to match the Trigger on UsesEqualDef.UsesAverageLogarithm
        [MultiplyAll]
        public static T DefAverageLogarithm<T>([SkipIfAllUniform] T[] Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return ReplicateOp_NoDivide.DefAverageConditional(Uses, result);
        }
#endif
#else
    // must have upward Trigger to match the Trigger on UsesEqualDef.UsesAverageLogarithm
        public static T DefAverageLogarithm<T,TUses>([SkipIfAllUniform,Trigger] IList<TUses> Uses, T result)
            where T : SettableToProduct<TUses>, SettableTo<TUses>, TUses, SettableToUniform
        {
            return DefAverageConditional(Uses, result);
        }
#if SpecializeArrays
            // must have upward Trigger to match the Trigger on UsesEqualDef.UsesAverageLogarithm
        public static T DefAverageLogarithm<T,TUses>([SkipIfAllUniform,Trigger] TUses[] Uses, T result)
            where T : SettableToProduct<TUses>, SettableTo<TUses>, TUses, SettableToUniform
        {
            return DefAverageConditional(Uses, result);
        }
#endif
#endif
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateMaxOp"]/doc/*'/>
    [FactorMethod(typeof(Clone), "Replicate<>")]
    [Quality(QualityBand.Mature)]
    public static class ReplicateMaxOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateMaxOp"]/message_doc[@name="UsesMaxConditional{T}(IList{T}, T, int, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution o0ver the replicated variable.</typeparam>
        public static T UsesMaxConditional<T>([AllExceptIndex] IReadOnlyList<T> Uses, [SkipIfUniform] T Def, int resultIndex, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            T res = ReplicateOp_NoDivide.UsesAverageConditional<T>(Uses, Def, resultIndex, result);
            if (res is UnnormalizedDiscrete)
                ((UnnormalizedDiscrete)(object)res).SetMaxToZero();
            return res;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateMaxOp"]/message_doc[@name="UsesMaxConditionalInit{T}(T, int)"]/*'/>
        /// <typeparam name="T">The type of the distribution o0ver the replicated variable.</typeparam>
        [Skip]
        public static T UsesMaxConditionalInit<T>([IgnoreDependency] T Def, int resultIndex)
            where T : ICloneable
        {
            return (T)Def.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateMaxOp"]/message_doc[@name="DefMaxConditional{T}(IList{T}, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution o0ver the replicated variable.</typeparam>
        public static T DefMaxConditional<T>([SkipIfAllUniform] IReadOnlyList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return ReplicateOp_NoDivide.DefAverageConditional<T>(Uses, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateMaxOp"]/message_doc[@name="MarginalMaxConditional{T}(IList{T}, T, T)"]/*'/>
        /// <typeparam name="T">The type of the distribution over the replicated variable.</typeparam>
        public static T MarginalMaxConditional<T>(IReadOnlyList<T> Uses, [SkipIfUniform] T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            T res = ReplicateOp_NoDivide.MarginalAverageConditional<T>(Uses, Def, result);
            if (res is UnnormalizedDiscrete)
                ((UnnormalizedDiscrete)(object)res).SetMaxToZero();
            return res;
        }
    }
}
