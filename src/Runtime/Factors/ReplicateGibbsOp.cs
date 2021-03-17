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

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp{T}"]/doc/*'/>
    /// <typeparam name="T">The type of the variable being replicated.</typeparam>
    [FactorMethod(typeof(Clone), "Replicate<>", Default = true)]
    [FactorMethod(typeof(Clone), "ReplicateWithMarginal<>", Default = true)]
    [Quality(QualityBand.Mature)]
    public static class ReplicateGibbsOp<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp{T}"]/message_doc[@name="LogEvidenceRatio{TDist}(IList{TDist}, T)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<TDist>([SkipIfAllUniform] IList<TDist> Uses, T Def)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp{T}"]/message_doc[@name="GibbsEvidence{TDist}(IList{TDist}, T)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Skip]
        public static double GibbsEvidence<TDist>(IList<TDist> Uses, T Def)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

#if false
        public static T UsesGibbs<TDist>([SkipIfUniform] GibbsMarginal<TDist, T> marginal, int resultIndex, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return marginal.LastSample;
        }
#elif false
        public static T UsesGibbs<TDist>([SkipIfUniform] GibbsMarginal<TDist, T> marginal, TDist def, int resultIndex, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return marginal.LastSample;
        }
        public static T UsesGibbs<TDist>([SkipIfUniform] GibbsMarginal<TDist, T> marginal, T def, int resultIndex, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            if (def is bool[]) {
                if (!Util.ValueEquals((bool[])(object)def, (bool[])(object)marginal.LastSample)) throw new Exception("gotcha");
            }
            else if (def is double[]) {
                if (!Util.ValueEquals((double[])(object)def, (double[])(object)marginal.LastSample)) throw new Exception("gotcha");
            } else if (!def.Equals(marginal.LastSample)) throw new Exception("gotcha");
            return marginal.LastSample;
        }
#else
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp{T}"]/message_doc[@name="UsesGibbs{TDist}(GibbsMarginal{TDist, T}, TDist, int, T)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static T UsesGibbs<TDist>([SkipIfUniform] GibbsMarginal<TDist, T> to_marginal, TDist def, int resultIndex, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            // This method must depend on Def, even though Def isn't used, in order to get the right triggers
            return to_marginal.LastSample;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp{T}"]/message_doc[@name="UsesGibbs(T, int, T)"]/*'/>
        public static T UsesGibbs([IsReturned] T def, int resultIndex, T result)
        {
            return def;
        }
#endif

#if true
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp{T}"]/message_doc[@name="UsesGibbs{TDist}(TDist, int, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        // until .NET 4
        public static TDist UsesGibbs<TDist>([IsReturned] TDist def, int resultIndex, TDist result)
            where TDist : IDistribution<T>
        {
            return def;
        }
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp{T}"]/message_doc[@name="DefGibbs{TDist}(IList{TDist}, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [MultiplyAll]
        public static TDist DefGibbs<TDist>(
            [SkipIfAllUniform] IReadOnlyList<TDist> Uses,
            TDist result)
            where TDist : IDistribution<T>, Sampleable<T>, SettableTo<TDist>, SettableToProduct<TDist>, SettableToRatio<TDist>
        {
            return ReplicateOp_NoDivide.DefAverageConditional(Uses, result);
        }

#if SpecializeArrays
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp{T}"]/message_doc[@name="DefGibbs{TDist}(TDist[], TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [MultiplyAll]
        public static TDist DefGibbs<TDist>(
            [SkipIfAllUniform] TDist[] Uses,
            TDist result)
            where TDist : IDistribution<T>, Sampleable<T>, SettableToProduct<TDist>, SettableTo<TDist>
        {
            return ReplicateOp_NoDivide.DefAverageConditional(Uses, result);
        }
#endif

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp{T}"]/message_doc[@name="DefGibbs{TDist}(GibbsMarginal{TDist, T}, T)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static T DefGibbs<TDist>([SkipIfUniform] GibbsMarginal<TDist, T> to_marginal, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return to_marginal.LastSample;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp{T}"]/message_doc[@name="MarginalGibbs{TDist}(IList{TDist}, TDist, GibbsMarginal{TDist, T})"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Stochastic]
        [SkipIfAllUniform]
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist>(
            IReadOnlyList<TDist> Uses,
            [SkipIfUniform] TDist Def,
            GibbsMarginal<TDist, T> to_marginal)
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp{T}"]/message_doc[@name="MarginalGibbs{TDist}(T, GibbsMarginal{TDist, T})"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        /// <typeparam name="TDomain">The domain type of <typeparamref name="TDist"/>.</typeparam>
        [Stochastic] // must be labelled Stochastic to get correct schedule, even though it isn't Stochastic
        public static GibbsMarginal<TDist, TDomain> MarginalGibbs<TDist, TDomain>(
            TDomain Def,
            GibbsMarginal<TDist, TDomain> to_marginal)
            where TDist : IDistribution<TDomain>, Sampleable<TDomain>
        {
            GibbsMarginal<TDist, TDomain> result = to_marginal;
            TDist marginal = result.LastConditional;
            marginal.Point = Def;
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp{T}"]/message_doc[@name="MarginalGibbs{TDist}(T[], GibbsMarginal{TDist, T})"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Stochastic] // must be labelled Stochastic to get correct schedule, even though it isn't Stochastic
        public static GibbsMarginal<TDist, T> MarginalGibbs2<TDist>(
            T[] Uses,
            GibbsMarginal<TDist, T> to_marginal)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            if (Uses.Length != 1)
                throw new ArgumentException("Uses.Length (" + Uses.Length + ") != 1");
            marginal.Point = Uses[0];
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp{T}"]/message_doc[@name="MarginalGibbsInit{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Skip]
        public static GibbsMarginal<TDist, T> MarginalGibbsInit<TDist>([IgnoreDependency] TDist def)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return new GibbsMarginal<TDist, T>(def, 100, 1, true, true, true);
        }
    }

    /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/doc/*'/>
    /// <typeparam name="T">The type of the replicated variable.</typeparam>
    [FactorMethod(typeof(Clone), "ReplicateWithMarginalGibbs<>")]
    [Buffers("sample", "conditional", "marginalEstimator", "sampleAcc", "conditionalAcc")]
    [Quality(QualityBand.Mature)]
    public static class ReplicateGibbsOp2<T>
    {
        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="ConditionalInit{TDist}(TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Skip]
        public static TDist ConditionalInit<TDist>([IgnoreDependency] TDist to_marginal)
            where TDist : ICloneable
        {
            return (TDist)to_marginal.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="Conditional{TDist}(T, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static TDist Conditional<TDist>(T Def, TDist result)
            where TDist : HasPoint<T>
        {
            result.Point = Def;
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="Conditional{TDist}(IList{TDist}, TDist, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static TDist Conditional<TDist>(IReadOnlyList<TDist> Uses, [SkipIfAnyUniform] TDist Def, TDist result)
            where TDist : SettableTo<TDist>, SettableToProduct<TDist>
        {
            result.SetTo(Def);
            result = Distribution.SetToProductWithAll(result, Uses);
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="Sample{TDist}(TDist, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Stochastic]
        public static T Sample<TDist>([IgnoreDependency] TDist to_marginal, [Proper] TDist conditional)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return conditional.Sample();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="MarginalEstimatorInit{TDist}(TDist, int)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static BurnInAccumulator<TDist> MarginalEstimatorInit<TDist>([IgnoreDependency] TDist to_marginal, int burnIn)
            where TDist : IDistribution<T>
        {
            Accumulator<TDist> est = (Accumulator<TDist>)ArrayEstimator.CreateEstimator<TDist, T>(to_marginal, true);
            return new BurnInAccumulator<TDist>(burnIn, 1, est);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="MarginalEstimator{TDist, TAcc}(TDist, TAcc)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        /// <typeparam name="TAcc">The type of the marginal estimator.</typeparam>
        public static TAcc MarginalEstimator<TDist, TAcc>([Proper] TDist conditional, TAcc marginalEstimator)
            where TAcc : Accumulator<TDist>
        {
            marginalEstimator.Add(conditional);
            return marginalEstimator;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="MarginalGibbs{TDist}(BurnInAccumulator{TDist}, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static TDist MarginalGibbs<TDist>(BurnInAccumulator<TDist> marginalEstimator, TDist result)
        {
            return ((Estimator<TDist>)marginalEstimator.Accumulator).GetDistribution(result);
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="SampleAccInit(ICollection{T}, int, int)"]/*'/>
        public static Accumulator<T> SampleAccInit(ICollection<T> to_samples, int burnIn, int thin)
        {
            return new BurnInAccumulator<T>(burnIn, thin, new AccumulateIntoCollection<T>(to_samples));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="SampleAcc(T, Accumulator{T})"]/*'/>
        public static Accumulator<T> SampleAcc(T sample, Accumulator<T> sampleAcc)
        {
            sampleAcc.Add(sample);
            return sampleAcc;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="SamplesGibbs{TList}(Accumulator{T}, TList)"]/*'/>
        /// <typeparam name="TList">The type of the outgoing message.</typeparam>
        public static TList SamplesGibbs<TList>(Accumulator<T> sampleAcc, TList result)
            where TList : ICollection<T>
        {
            // do nothing since result was already modified by sampleAcc
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="ConditionalAccInit{TDist}(ICollection{TDist}, int, int)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static Accumulator<TDist> ConditionalAccInit<TDist>(ICollection<TDist> to_conditionals, int burnIn, int thin)
        {
            return new BurnInAccumulator<TDist>(burnIn, thin, new AccumulateIntoCollection<TDist>(to_conditionals));
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="ConditionalAcc{TDist}(TDist, Accumulator{TDist})"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static Accumulator<TDist> ConditionalAcc<TDist>(TDist conditional, Accumulator<TDist> conditionalAcc)
            where TDist : ICloneable
        {
            conditionalAcc.Add((TDist)conditional.Clone());
            return conditionalAcc;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="ConditionalsGibbs{TDist, TDistList}(Accumulator{TDist}, TDistList)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        /// <typeparam name="TDistList">The type of the outgoing message.</typeparam>
        public static TDistList ConditionalsGibbs<TDist, TDistList>(Accumulator<TDist> conditionalAcc, TDistList result)
            where TDistList : ICollection<TDist>
        {
            // do nothing since result was already modified by Acc
            return result;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="GibbsEvidence{TDist}(IList{TDist}, T)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Skip]
        public static double GibbsEvidence<TDist>(IList<TDist> Uses, T Def)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="UsesGibbs(T, int, T)"]/*'/>
        public static T UsesGibbs([IsReturned] T def, int resultIndex, T result)
        {
            return def;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="UsesGibbs{TDist}(TDist, T, int, T)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static T UsesGibbs<TDist>(TDist def, T sample, int resultIndex, T result)
            where TDist : IDistribution<T>
        {
            // This method must depend on Def, even though Def isn't used, in order to get the right triggers
            return sample;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="UsesGibbs{TDist}(ICollection{TDist}, TDist, int, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
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

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="UsesGibbsInit{TDist}(TDist, int)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Skip]
        public static TDist UsesGibbsInit<TDist>([IgnoreDependency] TDist Def, int resultIndex)
            where TDist : ICloneable
        {
            return (TDist)Def.Clone();
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="DefGibbs{TDist}(TDist, T)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static T DefGibbs<TDist>(TDist def, [IsReturned] T sample)
            where TDist : IDistribution<T>
        {
            return sample;
        }

        /// <include file='FactorDocs.xml' path='factor_docs/message_op_class[@name="ReplicateGibbsOp2{T}"]/message_doc[@name="DefGibbs{TDist}(IList{TDist}, TDist)"]/*'/>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
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

}
