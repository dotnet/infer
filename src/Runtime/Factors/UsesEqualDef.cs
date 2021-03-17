// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#define SpecializeArrays
#define MinimalGenericTypeParameters

namespace Microsoft.ML.Probabilistic.Factors
{
    using System;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Factors.Attributes;

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefOp"]/doc/*'/>
    [FactorMethod(typeof(Clone), "UsesEqualDef<>")]
    [Quality(QualityBand.Mature)]
    public static class UsesEqualDefOp
    {
        /// <summary>
        /// Evidence message for EP
        /// </summary>
        /// <param name="Uses">Incoming message from 'Uses'. Must be a proper distribution.  If all elements are uniform, the result will be uniform.</param>
        /// <param name="Def">Incoming message from 'Def'.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence</returns>
        /// <remarks><para>
        /// The formula for the result is <c>log(sum_(Uses,Def) p(Uses,Def) factor(Uses,Def,Marginal) / sum_Uses p(Uses) messageTo(Uses))</c>.
        /// Adding up these values across all factors and variables gives the log-evidence estimate for EP.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="Uses"/> is not a proper distribution</exception>
        public static double LogEvidenceRatio1<T>([SkipIfAllUniform] IList<T> Uses, T Def)
            where T : CanGetLogAverageOf<T>, SettableToProduct<T>, SettableTo<T>, ICloneable, SettableToUniform
        {
            if (Uses.Count <= 1)
                return 0.0;
            else
            {
                T toUse = (T)Def.Clone();
                T[] productBefore = new T[Uses.Count];
                T productAfter = (T)Def.Clone();
                productAfter.SetToUniform();
                double z = 0.0;
                for (int i = 0; i < Uses.Count; i++)
                {
                    productBefore[i] = (T)Def.Clone();
                    if (i > 0)
                        productBefore[i].SetToProduct(productBefore[i - 1], Uses[i - 1]);
                    z += productBefore[i].GetLogAverageOf(Uses[i]);
                }
                // z is now log(sum_x Def(x)*prod_i Uses[i](x))
                for (int i = Uses.Count - 1; i >= 0; i--)
                {
                    toUse.SetToProduct(productBefore[i], productAfter);
                    z -= toUse.GetLogAverageOf(Uses[i]);
                    productAfter.SetToProduct(productAfter, Uses[i]);
                }
                return z;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefOp"]/message_doc[@name="LogEvidenceRatio{T}(IList{T}, T, IList{T})"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static double LogEvidenceRatio<T>([SkipIfAllUniform] IList<T> Uses, T Def, [Fresh] IList<T> to_Uses)
            where T : CanGetLogAverageOf<T>, SettableToProduct<T>, SettableTo<T>, ICloneable, SettableToUniform
        {
            if (Uses.Count <= 1)
                return 0.0;
            else
            {
                T productBefore = (T)Def.Clone();
                double z = 0.0;
                T previous_use = Def;
                for (int i = 0; i < Uses.Count; i++)
                {
                    if (i > 0)
                        productBefore.SetToProduct(productBefore, previous_use);
                    T use = Uses[i];
                    z += productBefore.GetLogAverageOf(use);
                    // z is now log(sum_x Def(x)*prod_i Uses[i](x))
                    z -= to_Uses[i].GetLogAverageOf(use);
                    previous_use = use;
                }
                return z;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefOp"]/message_doc[@name="MarginalAverageConditional{T}(IList{T}, T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [MultiplyAll]
        public static T MarginalAverageConditional<T>([NoInit] IReadOnlyList<T> Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            result.SetTo(Def);
            return Distribution.SetToProductWithAll(result, Uses);
        }

#if SpecializeArrays
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefOp"]/message_doc[@name="MarginalAverageConditional{T}(T[], T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T MarginalAverageConditional<T>([NoInit] T[] Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            result.SetTo(Def);
            return Distribution.SetToProductWithAll(result, Uses);
        }
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefOp"]/message_doc[@name="UsesAverageConditional{T}(IList{T}, T, int, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        // TM: SkipIfUniform on Def added as a stronger constraint, to prevent improper messages in EP.
        //[SkipIfAllUniform]
        public static T UsesAverageConditional<T>([AllExceptIndex] IReadOnlyList<T> Uses, [SkipIfAllUniform] T Def, int resultIndex, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Count)
                throw new ArgumentOutOfRangeException(nameof(resultIndex));
            result.SetTo(Def);
            return Distribution.SetToProductWithAllExcept(result, Uses, resultIndex);
        }

#if SpecializeArrays
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefOp"]/message_doc[@name="UsesAverageConditional{T}(T[], T, int, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        // TM: SkipIfUniform on Def added as a stronger constraint, to prevent improper messages in EP.
        //[SkipIfAllUniform]
        public static T UsesAverageConditional<T>([AllExceptIndex] T[] Uses, [SkipIfAllUniform] T Def, int resultIndex, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Length)
                throw new ArgumentOutOfRangeException(nameof(resultIndex));
            result.SetTo(Def);
            return Distribution.SetToProductWithAllExcept(result, Uses, resultIndex);
        }
#endif

#if MinimalGenericTypeParameters
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefOp"]/message_doc[@name="DefAverageConditional{T}(IList{T}, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [MultiplyAll]
        public static T DefAverageConditional<T>([SkipIfAllUniform] IReadOnlyList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#else
            public static T DefAverageConditional<T,TUses>([SkipIfAllUniform] IList<TUses> Uses, T result)
                        where T : SettableToProduct<TUses>, SettableTo<TUses>, TUses, SettableToUniform
                {
                        return Distribution.SetToProductOfAll(result, Uses);
                }
#endif
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of the variable.</typeparam>
    [FactorMethod(typeof(Clone), "UsesEqualDef<>")]
    [Quality(QualityBand.Mature)]
    public static class UsesEqualDefGibbsOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp{T}"]/message_doc[@name="GibbsEvidence{TDist}(IList{TDist}, TDist, GibbsMarginal{TDist, T})"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        public static double GibbsEvidence<TDist>(IList<TDist> Uses, TDist Def, GibbsMarginal<TDist, T> to_marginal)
            where TDist : IDistribution<T>, Sampleable<T>, CanGetLogAverageOf<TDist>, SettableTo<TDist>, SettableToProduct<TDist>
        {
            if (Uses.Count == 1)
            {
                // the total evidence contribution of this variable should be Def.GetLogAverageOf(Uses[0]).
                // but since this variable is sending a sample to Def and Use, and those factors will send their own evidence contribution,
                // we need to cancel the contribution of those factors here.
                return Def.GetLogAverageOf(Uses[0]) - Def.GetLogProb(to_marginal.LastSample) - Uses[0].GetLogProb(to_marginal.LastSample);
            }
            else
            {
                //throw new InferRuntimeException("Gibbs Sampling does not support variables defined within a gate");
                double z = 0.0;
                TDist productBefore = (TDist)Def.Clone();
                TDist product = (TDist)Def.Clone();
                for (int i = 0; i < Uses.Count; i++)
                {
                    if (i > 0)
                        product.SetToProduct(productBefore, Uses[i - 1]);
                    z += product.GetLogAverageOf(Uses[i]);
                    productBefore.SetTo(product);
                }
                // z is now log(sum_x Def(x)*prod_i Uses[i](x)), which is the desired total evidence.
                // but we must also cancel the contribution of the parent and child factors that received a sample from us.
                z -= Def.GetLogProb(to_marginal.LastSample);
                for (int i = 0; i < Uses.Count; i++)
                {
                    z -= Uses[i].GetLogProb(to_marginal.LastSample);
                }
                return z;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp{T}"]/message_doc[@name="MarginalGibbs{TDist}(IList{TDist}, TDist, GibbsMarginal{TDist, T})"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        [Stochastic]
        //[SkipIfAllUniform("Uses","Def")]
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist>(
            IReadOnlyList<TDist> Uses,
            [Proper] TDist Def,
            GibbsMarginal<TDist, T> to_marginal) // must not be called 'result', because its value is used
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToRatio<TDist>, SettableTo<TDist>, Sampleable<T>
        {
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            marginal.SetTo(Def);
            marginal = Distribution.SetToProductWithAll(marginal, Uses);
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp{T}"]/message_doc[@name="MarginalGibbs{TDist}(IList{TDist}, T, GibbsMarginal{TDist, T})"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        [Stochastic]
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist>(
            IReadOnlyList<TDist> Uses,
            T Def,
            GibbsMarginal<TDist, T> to_marginal) // must not be called 'result', because its value is used
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToRatio<TDist>, SettableTo<TDist>, Sampleable<T>
        {
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            marginal.Point = Def;
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp{T}"]/message_doc[@name="MarginalGibbs{TDist}(IList{T}, TDist, GibbsMarginal{TDist, T})"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        [Stochastic]
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist>(
            IList<T> Uses,
            TDist Def,
            GibbsMarginal<TDist, T> to_marginal) // must not be called 'result', because its value is used
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToRatio<TDist>, SettableTo<TDist>, Sampleable<T>
        {
            if (Uses.Count > 1)
                throw new ArgumentException("Uses.Count > 1");
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            marginal.Point = Uses[0];
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp{T}"]/message_doc[@name="MarginalGibbsInit{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        [Skip]
        public static GibbsMarginal<TDist, T> MarginalGibbsInit<TDist>([IgnoreDependency] TDist Def)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return new GibbsMarginal<TDist, T>(Def, 100, 1, true, true, true);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp{T}"]/message_doc[@name="UsesGibbs{TDist}(GibbsMarginal{TDist, T}, int, T)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        public static T UsesGibbs<TDist>([SkipIfUniform] GibbsMarginal<TDist, T> to_marginal, int resultIndex, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return to_marginal.LastSample;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp{T}"]/message_doc[@name="UsesGibbs{TDist}(ICollection{TDist}, TDist, int, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        public static TDist UsesGibbs<TDist>(
            [IgnoreDependency] ICollection<TDist> Uses,
            [IsReturned] TDist Def,
            int resultIndex, TDist result)
            where TDist : IDistribution<T>, Sampleable<T>, SettableTo<TDist>, SettableToProduct<TDist>, SettableToRatio<TDist>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Count)
                throw new ArgumentOutOfRangeException(nameof(resultIndex));
            if (Uses.Count > 1)
                throw new ArgumentException("Uses.Count > 1");
            result.SetTo(Def);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp{T}"]/message_doc[@name="UsesGibbsInit{TArrayType, TDef}(TDef, int, IArrayFactory{TDef, TArrayType})"]/*'/>
        /// <typeparam name="TArrayType">The type of arrays produced by <paramref name="factory"/>.</typeparam>
        /// <typeparam name="TDef">The type of the incoming message from <c>Def</c>.</typeparam>
        [Skip]
        public static TArrayType UsesGibbsInit<TArrayType, TDef>(
            [IgnoreDependency] TDef Def, int count, IArrayFactory<TDef, TArrayType> factory)
            where TDef : ICloneable
        {
            return factory.CreateArray(count, i => (TDef)Def.Clone());
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp{T}"]/message_doc[@name="DefGibbs{TDist}(GibbsMarginal{TDist, T}, T)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        public static T DefGibbs<TDist>([SkipIfUniform] GibbsMarginal<TDist, T> to_marginal, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return to_marginal.LastSample;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp{T}"]/message_doc[@name="DefGibbs{TDist}(IList{TDist}, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        //[MultiplyAll]
        public static TDist DefGibbs<TDist>(
            [SkipIfAllUniform] IReadOnlyList<TDist> Uses,
            TDist result)
            where TDist : IDistribution<T>, Sampleable<T>, SettableTo<TDist>, SettableToProduct<TDist>, SettableToRatio<TDist>
        {
            result.SetToUniform();
            result = Distribution.SetToProductWithAll(result, Uses);
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/doc/*'/>
    /// <typeparam name="T">The type of the variable.</typeparam>
    [FactorMethod(typeof(Clone), "UsesEqualDefGibbs<>")]
    [Buffers("sample", "conditional", "marginalEstimator", "sampleAcc", "conditionalAcc")]
    [Quality(QualityBand.Mature)]
    public static class UsesEqualDefGibbsOp2<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="ConditionalInit{TDist}(TDist)"]/*'/>
        [Skip]
        public static TDist ConditionalInit<TDist>([IgnoreDependency] TDist def)
            where TDist : ICloneable
        {
            return (TDist)def.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="Conditional{TDist}(IList{TDist}, TDist, TDist)"]/*'/>
        public static TDist Conditional<TDist>(IReadOnlyList<TDist> Uses, [SkipIfAnyUniform] TDist Def, TDist result)
            where TDist : SettableTo<TDist>, SettableToProduct<TDist>
        {
            result.SetTo(Def);
            result = Distribution.SetToProductWithAll(result, Uses);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="Sample{TDist}(TDist, TDist)"]/*'/>
        [Stochastic]
        public static T Sample<TDist>([IgnoreDependency] TDist def, [Proper] TDist conditional)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return conditional.Sample();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="MarginalEstimatorInit{TDist}(TDist, int)"]/*'/>
        public static BurnInAccumulator<TDist> MarginalEstimatorInit<TDist>([IgnoreDependency] TDist to_marginal, int burnIn)
            where TDist : IDistribution<T>
        {
            Accumulator<TDist> est = (Accumulator<TDist>)ArrayEstimator.CreateEstimator<TDist, T>(to_marginal, true);
            return new BurnInAccumulator<TDist>(burnIn, 1, est);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="MarginalEstimator{TDist, TAcc}(TDist, TAcc)"]/*'/>
        public static TAcc MarginalEstimator<TDist, TAcc>([Proper] TDist conditional, TAcc marginalEstimator)
            where TAcc : Accumulator<TDist>
        {
            marginalEstimator.Add(conditional);
            return marginalEstimator;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="MarginalGibbs{TDist}(BurnInAccumulator{TDist}, TDist)"]/*'/>
        public static TDist MarginalGibbs<TDist>(BurnInAccumulator<TDist> marginalEstimator, TDist result)
        {
            return ((Estimator<TDist>)marginalEstimator.Accumulator).GetDistribution(result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="SampleAccInit(ICollection{T}, int, int)"]/*'/>
        public static Accumulator<T> SampleAccInit(ICollection<T> to_samples, int burnIn, int thin)
        {
            return new BurnInAccumulator<T>(burnIn, thin, new AccumulateIntoCollection<T>(to_samples));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="SampleAcc(T, Accumulator{T})"]/*'/>
        public static Accumulator<T> SampleAcc(T sample, Accumulator<T> sampleAcc)
        {
            sampleAcc.Add(sample);
            return sampleAcc;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="SamplesGibbs{TList}(Accumulator{T}, TList)"]/*'/>
        public static TList SamplesGibbs<TList>(Accumulator<T> sampleAcc, TList result)
            where TList : ICollection<T>
        {
            // do nothing since result was already modified by sampleAcc
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="ConditionalAccInit{TDist}(ICollection{TDist}, int, int)"]/*'/>
        public static Accumulator<TDist> ConditionalAccInit<TDist>(ICollection<TDist> to_conditionals, int burnIn, int thin)
        {
            return new BurnInAccumulator<TDist>(burnIn, thin, new AccumulateIntoCollection<TDist>(to_conditionals));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="ConditionalAcc{TDist}(TDist, Accumulator{TDist})"]/*'/>
        public static Accumulator<TDist> ConditionalAcc<TDist>(TDist conditional, Accumulator<TDist> conditionalAcc)
            where TDist : ICloneable
        {
            conditionalAcc.Add((TDist)conditional.Clone());
            return conditionalAcc;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="ConditionalsGibbs{TDist, TDistList}(Accumulator{TDist}, TDistList)"]/*'/>
        public static TDistList ConditionalsGibbs<TDist, TDistList>(Accumulator<TDist> conditionalAcc, TDistList result)
            where TDistList : ICollection<TDist>
        {
            // do nothing since result was already modified by Acc
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="GibbsEvidence{TDist}(IList{TDist}, TDist, T)"]/*'/>
        public static double GibbsEvidence<TDist>(IList<TDist> Uses, TDist Def, T sample)
            where TDist : IDistribution<T>, Sampleable<T>, CanGetLogAverageOf<TDist>, SettableTo<TDist>, SettableToProduct<TDist>
        {
            if (Uses.Count == 1)
            {
                // the total evidence contribution of this variable should be Def.GetLogAverageOf(Uses[0]).
                // but since this variable is sending a sample to Def and Use, and those factors will send their own evidence contribution,
                // we need to cancel the contribution of those factors here.
                return Def.GetLogAverageOf(Uses[0]) - Def.GetLogProb(sample) - Uses[0].GetLogProb(sample);
            }
            else
            {
                //throw new InferRuntimeException("Gibbs Sampling does not support variables defined within a gate");
                double z = 0.0;
                TDist productBefore = (TDist)Def.Clone();
                TDist product = (TDist)Def.Clone();
                for (int i = 0; i < Uses.Count; i++)
                {
                    if (i > 0)
                        product.SetToProduct(productBefore, Uses[i - 1]);
                    z += product.GetLogAverageOf(Uses[i]);
                    productBefore.SetTo(product);
                }
                // z is now log(sum_x Def(x)*prod_i Uses[i](x)), which is the desired total evidence.
                // but we must also cancel the contribution of the parent and child factors that received a sample from us.
                z -= Def.GetLogProb(sample);
                for (int i = 0; i < Uses.Count; i++)
                {
                    z -= Uses[i].GetLogProb(sample);
                }
                return z;
            }
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="UsesGibbs{TDist}(TDist, T, int, T)"]/*'/>
        public static T UsesGibbs<TDist>(TDist def, T sample, int resultIndex, T result)
            where TDist : IDistribution<T>
        {
            return sample;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="UsesGibbs{TDist}(ICollection{TDist}, TDist, int, TDist)"]/*'/>
        public static TDist UsesGibbs<TDist>(
            [IgnoreDependency] ICollection<TDist> Uses,
            [IsReturned] TDist Def,
            int resultIndex, TDist result)
            where TDist : IDistribution<T>, Sampleable<T>, SettableTo<TDist>, SettableToProduct<TDist>, SettableToRatio<TDist>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Count)
                throw new ArgumentOutOfRangeException(nameof(resultIndex));
            if (Uses.Count > 1)
                throw new ArgumentException("Uses.Count > 1");
            result.SetTo(Def);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="UsesGibbsInit{TDist}(TDist, int)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        [Skip]
        public static TDist UsesGibbsInit<TDist>([IgnoreDependency] TDist Def, int resultIndex)
            where TDist : ICloneable
        {
            return (TDist)Def.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="DefGibbs{TDist}(TDist, T)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        public static T DefGibbs<TDist>(TDist def, [IsReturned] T sample)
            where TDist : IDistribution<T>
        {
            return sample;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefGibbsOp2{T}"]/message_doc[@name="DefGibbs{TDist}(IList{TDist}, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the variable.</typeparam>
        //[MultiplyAll]
        public static TDist DefGibbs<TDist>(
            [SkipIfAllUniform] IReadOnlyList<TDist> Uses,
            TDist result)
            where TDist : IDistribution<T>, Sampleable<T>, SettableTo<TDist>, SettableToProduct<TDist>, SettableToRatio<TDist>
        {
            result.SetToUniform();
            result = Distribution.SetToProductWithAll(result, Uses);
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefMaxOp"]/doc/*'/>
    [FactorMethod(typeof(Clone), "UsesEqualDef<>")]
    [Quality(QualityBand.Mature)]
    public static class UsesEqualDefMaxOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefMaxOp"]/message_doc[@name="UsesMaxConditional{T}(IList{T}, T, int, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T UsesMaxConditional<T>([AllExceptIndex] IReadOnlyList<T> Uses, [SkipIfUniform] T Def, int resultIndex, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            T res = UsesEqualDefOp.UsesAverageConditional<T>(Uses, Def, resultIndex, result);
            if (res is UnnormalizedDiscrete)
                ((UnnormalizedDiscrete)(object)res).SetMaxToZero();
            return res;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefMaxOp"]/message_doc[@name="DefMaxConditional{T}(IList{T}, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T DefMaxConditional<T>([SkipIfAllUniform] IReadOnlyList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return UsesEqualDefOp.DefAverageConditional<T>(Uses, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefMaxOp"]/message_doc[@name="MarginalMaxConditional{T}(IList{T}, T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T MarginalMaxConditional<T>(IReadOnlyList<T> Uses, [SkipIfUniform] T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            T res = UsesEqualDefOp.MarginalAverageConditional<T>(Uses, Def, result);
            if (res is UnnormalizedDiscrete)
                ((UnnormalizedDiscrete)(object)res).SetMaxToZero();
            return res;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefVmpBufferOp"]/doc/*'/>
    [FactorMethod(typeof(Clone), "UsesEqualDef<>", Default = true)]
    [Quality(QualityBand.Mature)]
    public static class UsesEqualDefVmpBufferOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefVmpBufferOp"]/message_doc[@name="MarginalAverageLogarithm{T}(IList{T}, T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T MarginalAverageLogarithm<T>([NoInit] IReadOnlyList<T> Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            result.SetTo(Def);
            return Distribution.SetToProductWithAll(result, Uses);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefVmpBufferOp"]/message_doc[@name="UsesAverageLogarithm{T}(T, int, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T UsesAverageLogarithm<T>([IsReturned] T to_marginal, int resultIndex, T result)
            where T : SettableTo<T>
        {
            result.SetTo(to_marginal);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefVmpBufferOp"]/message_doc[@name="DefAverageLogarithm{T}(T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T DefAverageLogarithm<T>([IsReturned] T to_marginal, T result)
            where T : SettableTo<T>
        {
            result.SetTo(to_marginal);
            return result;
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefVmpOp"]/doc/*'/>
    [FactorMethod(typeof(Clone), "UsesEqualDef<>")]
    [Quality(QualityBand.Mature)]
    public static class UsesEqualDefVmpOp
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefVmpOp"]/message_doc[@name="AverageLogFactor{T}(T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static double AverageLogFactor<T>([Fresh] T to_marginal)
            where T : CanGetAverageLog<T>
        {
            return -to_marginal.GetAverageLog(to_marginal);
        }

#if MinimalGenericTypeParameters
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefVmpOp"]/message_doc[@name="MarginalAverageLogarithm{T}(IList{T}, T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        public static T MarginalAverageLogarithm<T>(IReadOnlyList<T> Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            return UsesAverageLogarithm(Uses, Def, 0, result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefVmpOp"]/message_doc[@name="UsesAverageLogarithm{T}(IList{T}, T, int, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T UsesAverageLogarithm<T>([NoInit] IReadOnlyList<T> Uses, T Def, int resultIndex, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            result.SetTo(Def);
            return Distribution.SetToProductWithAll(result, Uses);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="UsesEqualDefVmpOp"]/message_doc[@name="DefAverageLogarithm{T}(IList{T}, T, T)"]/*'/>
        /// <typeparam name="T">The type of the messages.</typeparam>
        // TM: Proper added on Def to avoid improper messages.
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T DefAverageLogarithm<T>([NoInit] IReadOnlyList<T> Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            return UsesAverageLogarithm(Uses, Def, 0, result);
        }
#else
        [SkipIfAllUniform]
        public static T MarginalAverageLogarithm<T, TUses, TDef>(IList<TUses> Uses, TDef Def, T result)
                where T : SettableToProduct<TUses>, SettableTo<TDef>, TUses
        {
            return UsesAverageLogarithm<T, TUses, TDef>(Uses, Def, 0, result);
        }

        [SkipIfAllUniform]
        public static T UsesAverageLogarithm<T, TUses, TDef>([MatchingIndexTrigger] IList<TUses> Uses, TDef Def, int resultIndex, T result)
                    where T : SettableToProduct<TUses>, SettableTo<TDef>, TUses
        {
            result.SetTo(Def);
            return Distribution.SetToProductWithAll(result, Uses);
        }

        [SkipIfAllUniform]
        public static T DefAverageLogarithm<T, TUses, TDef>(IList<TUses> Uses, [Trigger] TDef Def, T result)
                    where T : SettableToProduct<TUses>, SettableTo<TDef>, TUses
        {
            return UsesAverageLogarithm<T, TUses, TDef>(Uses, Def, 0, result);
        }
#endif
    }
}
