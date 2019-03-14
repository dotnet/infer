// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// A base class for weighted finite state transducers.
    /// </summary>
    /// <typeparam name="TSrcSequence">The type of the first sequence in a pair.</typeparam>
    /// <typeparam name="TSrcElement">The type of an element of <typeparamref name="TSrcSequence"/>.</typeparam>
    /// <typeparam name="TSrcElementDistribution">The type of a distribution over <typeparamref name="TSrcElement"/>.</typeparam>
    /// <typeparam name="TSrcSequenceManipulator">The type providing ways to manipulate sequences of type <typeparamref name="TSrcSequence"/>.</typeparam>
    /// <typeparam name="TSrcAutomaton">The type of an automaton defined on <typeparamref name="TSrcSequence"/>.</typeparam> 
    /// <typeparam name="TDestSequence">The type of the second sequence in a pair.</typeparam>
    /// <typeparam name="TDestElement">The type of an element of <typeparamref name="TDestSequence"/>.</typeparam>
    /// <typeparam name="TDestElementDistribution">The type of a distribution over <typeparamref name="TDestElement"/>.</typeparam>
    /// <typeparam name="TDestSequenceManipulator">The type providing ways to manipulate sequences of type <typeparamref name="TDestSequence"/>.</typeparam> 
    /// <typeparam name="TDestAutomaton">The type of an automaton defined on <typeparamref name="TDestSequence"/>.</typeparam> 
    /// <typeparam name="TPairDistribution">The type of a distribution over pairs of <typeparamref name="TSrcElement"/> and <typeparamref name="TDestElement"/>.</typeparam>
    /// <typeparam name="TThis">The type of a concrete transducer class.</typeparam>
    public abstract class TransducerBase<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis>
        where TSrcElementDistribution : IDistribution<TSrcElement>, CanGetLogAverageOf<TSrcElementDistribution>, SettableToProduct<TSrcElementDistribution>, SettableToWeightedSumExact<TSrcElementDistribution>, SettableToPartialUniform<TSrcElementDistribution>, Sampleable<TSrcElement>, new()
        where TDestElementDistribution : IDistribution<TDestElement>, CanGetLogAverageOf<TDestElementDistribution>, SettableToProduct<TDestElementDistribution>, SettableToWeightedSumExact<TDestElementDistribution>, SettableToPartialUniform<TDestElementDistribution>, Sampleable<TDestElement>, new()
        where TSrcSequence : class, IEnumerable<TSrcElement>
        where TDestSequence : class, IEnumerable<TDestElement>
        where TSrcSequenceManipulator : ISequenceManipulator<TSrcSequence, TSrcElement>, new()
        where TDestSequenceManipulator : ISequenceManipulator<TDestSequence, TDestElement>, new()
        where TSrcAutomaton : Automaton<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton>, new()
        where TDestAutomaton : Automaton<TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton>, new()
        where TPairDistribution : PairDistributionBase<TSrcElement, TSrcElementDistribution, TDestElement, TDestElementDistribution, TPairDistribution>, new()
        where TThis : TransducerBase<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TPairDistribution, TThis>, new()
    {
        /// <summary>
        /// An automaton defined on sequences of element pairs that represents the transducer.
        /// </summary>
        protected PairListAutomaton sequencePairToWeight = new PairListAutomaton();

        #region Properties

        /// <summary>
        /// Gets or sets the maximum number of states a transducer can have.
        /// </summary>
        public static int MaxStateCount
        {
            get { return PairListAutomaton.MaxStateCount; }
            set { PairListAutomaton.MaxStateCount = value; }
        }
        
        #endregion

        #region Factory methods

        /// <summary>
        /// Creates a transducer <c>T(a, b) = f(a) I[b = ""]</c>, where <c>f(a)</c> is a given automaton.
        /// </summary>
        /// <param name="srcAutomaton">The automaton defining weights for the first transducer argument.</param>
        /// <returns>The created transducer.</returns>
        public static TThis Consume(TSrcAutomaton srcAutomaton)
        {
            Argument.CheckIfNotNull(srcAutomaton, "srcAutomaton");

            var result = new TThis();
            result.sequencePairToWeight.SetToFunction(
                srcAutomaton,
                (dist, weight, group) => ValueTuple.Create(
                    dist.HasValue
                        ? Option.Some(PairDistributionBase<TSrcElement, TSrcElementDistribution, TDestElement, TDestElementDistribution, TPairDistribution>.FromFirst(dist))
                        : Option.None,
                    weight));

            return result;
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = I[a = A] I[b = ""]</c>, where <c>A</c> is a given sequence.
        /// </summary>
        /// <param name="srcSequence">The sequence to constrain the first transducer argument to.</param>
        /// <returns>The created transducer.</returns>
        public static TThis Consume(TSrcSequence srcSequence)
        {
            return Consume(Automaton<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton>.ConstantOn(1.0, srcSequence));
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = I[a[0] = c, |a| = 1] I[b = ""]</c>, where <c>c</c> is a given element.
        /// </summary>
        /// <param name="srcElement">The element to constrain the first transducer argument to.</param>
        /// <returns>The created transducer.</returns>
        public static TThis ConsumeElement(TSrcElement srcElement)
        {
            return Consume(Automaton<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton>.ConstantOnElement(1.0, srcElement));
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = g(b) I[a = ""]</c>, where <c>g(b)</c> is a given automaton.
        /// </summary>
        /// <param name="destAutomaton">The automaton defining weights for the second transducer argument.</param>
        /// <returns>The created transducer.</returns>
        public static TThis Produce(TDestAutomaton destAutomaton)
        {
            Argument.CheckIfNotNull(destAutomaton, "destAutomaton");

            var result = new TThis();
            result.sequencePairToWeight.SetToFunction(
                destAutomaton,
                (dist, weight, group) => ValueTuple.Create(
                    dist.HasValue
                        ? Option.Some(PairDistributionBase<TSrcElement, TSrcElementDistribution, TDestElement, TDestElementDistribution, TPairDistribution>.FromSecond(dist))
                        : Option.None,
                    weight));
            return result;
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = I[a = ""] I[b = B]</c>, where <c>B</c> is a given sequence.
        /// </summary>
        /// <param name="destSequence">The sequence to constrain the second transducer argument to.</param>
        /// <returns>The created transducer.</returns>
        public static TThis Produce(TDestSequence destSequence)
        {
            return Produce(Automaton<TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton>.ConstantOn(1.0, destSequence));
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = I[a = ""] I[b[0] = c, |b| = 1]</c>, where <c>c</c> is a given element.
        /// </summary>
        /// <param name="destElement">The element to constrain the second transducer argument to.</param>
        /// <returns>The created transducer.</returns>
        public static TThis ProduceElement(TDestElement destElement)
        {
            return Produce(Automaton<TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton>.ConstantOnElement(1.0, destElement));
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = I[a = ""] I[b[0] in c, |b| = 1]</c>, where <c>c</c> is a given element distribution.
        /// </summary>
        /// <param name="destElementDist">The element distribution to constrain the second transducer argument to.</param>
        /// <returns>The created transducer.</returns>
        public static TThis ProduceElement(TDestElementDistribution destElementDist)
        {
            return Produce(Automaton<TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton>.ConstantOnElement(1.0, destElementDist));
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = f(a) g(b)</c>, where <c>f(a)</c> and <c>g(b)</c> are given automata.
        /// </summary>
        /// <param name="srcAutomaton">The automaton defining weights for the first transducer argument.</param>
        /// <param name="destAutomaton">The automaton defining weights for the second transducer argument.</param>
        /// <returns>The created transducer.</returns>
        public static TThis Replace(TSrcAutomaton srcAutomaton, TDestAutomaton destAutomaton)
        {
            Argument.CheckIfNotNull(srcAutomaton, "srcAutomaton");
            Argument.CheckIfNotNull(destAutomaton, "destAutomaton");

            TThis result = Consume(srcAutomaton);
            result.AppendInPlace(Produce(destAutomaton));
            return result;
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = I[a = A] I[b = B]</c>, where <c>A</c> and <c>B</c> are given sequences.
        /// </summary>
        /// <param name="srcSequence">The sequence to constrain the first transducer argument to.</param>
        /// <param name="destSequence">The sequence to constrain the second transducer argument to.</param>
        /// <returns>The created transducer.</returns>
        public static TThis Replace(TSrcSequence srcSequence, TDestSequence destSequence)
        {
            Argument.CheckIfNotNull(srcSequence, "srcSequence");
            Argument.CheckIfNotNull(destSequence, "destSequence");

            return Replace(
                Automaton<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton>.ConstantOn(1.0, srcSequence),
                Automaton<TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton>.ConstantOn(1.0, destSequence));
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = I[a[i] is in X for all i] I[b[j] is in Y for all j]</c>, where <c>X</c> and <c>Y</c>
        /// are the supports of given element distributions.
        /// </summary>
        /// <param name="allowedSrcElements">Specifies the elements allowed in the first transducer argument.</param>
        /// <param name="allowedDestElements">Specifies the elements allowed in the second transducer argument.</param>
        /// <returns>The created transducer.</returns>
        public static TThis Replace(TSrcElementDistribution allowedSrcElements, TDestElementDistribution allowedDestElements)
        {
            Argument.CheckIfNotNull(allowedSrcElements, "allowedSrcElements");
            Argument.CheckIfNotNull(allowedDestElements, "allowedDestElements");

            return Replace(
                Automaton<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton>.Constant(1.0, allowedSrcElements),
                Automaton<TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton>.Constant(1.0, allowedDestElements));
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = sum_i Ti(a, b)</c>, where <c>Ti(a, b)</c> is an element of a given transducer collection.
        /// </summary>
        /// <param name="transducers">The transducers to sum.</param>
        /// <returns>The created transducer.</returns>
        public static TThis Sum(params TThis[] transducers)
        {
            return Sum((IEnumerable<TThis>)transducers);
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = sum_i Ti(a, b)</c>, where <c>Ti(a, b)</c> is an element of a given transducer collection.
        /// </summary>
        /// <param name="transducers">The transducers to sum.</param>
        /// <returns>The created transducer.</returns>
        public static TThis Sum(IEnumerable<TThis> transducers)
        {
            Argument.CheckIfNotNull(transducers, "transducers");

            return new TThis { sequencePairToWeight = PairListAutomaton.Sum(transducers.Select(t => t.sequencePairToWeight)) };
        }

        /// <summary>
        /// Creates a transducer <c>T'(a, b) = sum_{k=Kmin}^{Kmax} sum_{a1 a2 ... ak = a} sum_{b1 b2 ... bk = b} T(a1, b1)T(a2, b2)...T(ak, bk)</c>,
        /// where <c>T(a, b)</c> is a given transducer, and <c>Kmin</c> and <c>Kmax</c> are the minimum
        /// and the maximum number of factors in a sum term.
        /// </summary>
        /// <param name="transducer">The transducer.</param>
        /// <param name="minTimes">The minimum number of factors in a sum term. Defaults to 1.</param>
        /// <param name="maxTimes">An optional maximum number of factors in a sum term.</param>
        /// <returns>The created transducer.</returns>
        /// <remarks>
        /// The result is the sum of Cauchy products of the given transducer with itself,
        /// each product having a different number of factors.
        /// </remarks>
        public static TThis Repeat(TThis transducer, int minTimes = 1, int? maxTimes = null)
        {
            Argument.CheckIfNotNull(transducer, "transducer");

            return new TThis { sequencePairToWeight = PairListAutomaton.Repeat(transducer.sequencePairToWeight, minTimes, maxTimes) };
        }

        /// <summary>
        /// Creates a transducer <c>T'(a, b) = T(a, b) + I[a = "", b = ""]</c>, where <c>T(a, b)</c> is a given transducer.
        /// </summary>
        /// <param name="transducer">The transducer <c>U(a, b)</c>.</param>
        /// <returns>The created transducer.</returns>
        public static TThis Optional(TThis transducer)
        {
            Argument.CheckIfNotNull(transducer, "transducer");

            var emptySequence = new List<Pair<Option<TSrcElement>, Option<TDestElement>>>();
            return Sum(
                transducer,
                new TThis { sequencePairToWeight = PairListAutomaton.ConstantOnLog(0.0, emptySequence) });
        }

        #endregion

        #region Operations

        /// <summary>
        /// Gets the logarithm of the transducer value on a given pair of sequences.
        /// </summary>
        /// <param name="sequence1">The first sequence.</param>
        /// <param name="sequence2">The second sequence.</param>
        /// <returns>The logarithm of the transducer value on the given pair of sequences.</returns>
        public double GetLogValue(TSrcSequence sequence1, TDestSequence sequence2)
        {
            Argument.CheckIfNotNull(sequence1, "sequence1");
            Argument.CheckIfNotNull(sequence2, "sequence2");

            return this.ProjectSource(sequence1).GetLogValue(sequence2);
        }

        /// <summary>
        /// Gets the transducer value on a given pair of sequences.
        /// </summary>
        /// <param name="sequence1">The first sequence.</param>
        /// <param name="sequence2">The second sequence.</param>
        /// <returns>The transducer value on the given pair of sequences.</returns>
        public double GetValue(TSrcSequence sequence1, TDestSequence sequence2)
        {
            return Math.Exp(this.GetLogValue(sequence1, sequence2));
        }

        /// <summary>
        /// Creates a transducer <c>T'(a, b) = sum_{a1 a2 = a, b1 b2 = b} T(a1, b1) U(a2, b2)</c>,
        /// where <c>T(a1, b1)</c> is the current transducer and <c>U(a2, b2)</c> is a given transducer.
        /// The resulting transducer is also known as the Cauchy product of two transducers.
        /// </summary>
        /// <param name="transducer">The transducer to append.</param>
        /// <returns>The created transducer.</returns>
        public TThis Append(TThis transducer)
        {
            var result = this.Clone();
            result.AppendInPlace(transducer);
            return result;
        }

        /// <summary>
        /// Replaces the current transducer with a transducer <c>T'(a, b) = sum_{a1 a2 = a, b1 b2 = b} T(a1, b1) U(a2, b2)</c>,
        /// where <c>T(a1, b1)</c> is the current transducer and <c>U(a2, b2)</c> is a given transducer.
        /// The resulting transducer is also known as the Cauchy product of two transducers.
        /// </summary>
        /// <param name="transducer">The transducer to append.</param>
        /// <param name="group">The group.</param>
        public void AppendInPlace(TThis transducer, int group = 0)
        {
            Argument.CheckIfNotNull(transducer, "transducer");

            this.sequencePairToWeight.AppendInPlace(transducer.sequencePairToWeight, group);
        }

        /// <summary>
        /// Computes <c>g(b) = sum_a f(a) T(a, b)</c>, where <c>T(a, b)</c> is the current transducer and <c>f(a)</c> is a given automaton.
        /// </summary>
        /// <param name="srcAutomaton">The automaton to project.</param>
        /// <returns>The projection.</returns>
        /// <remarks>
        /// The code of this method has a lot in common with the code of Automaton.SetToProduct.
        /// Unfortunately, it's not clear how to avoid the duplication in the current design.
        /// </remarks>
        public TDestAutomaton ProjectSource(TSrcAutomaton srcAutomaton)
        {
            Argument.CheckIfNotNull(srcAutomaton, "srcAutomaton");

            var mappingAutomaton = this.sequencePairToWeight;
            if (srcAutomaton.IsCanonicZero() || mappingAutomaton.IsCanonicZero())
            {
                return Automaton<TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton>.Zero();
            }

            // The projected automaton must be epsilon-free
            srcAutomaton.MakeEpsilonFree();

            var result = new Automaton<TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton>.Builder();
            var destStateCache = new Dictionary<(int, int), int>();
            var stack = new Stack<(int state1, int state2, int destStateIndex)>();

            // Creates destination state and schedules projection computation for it.
            // If computation is already scheduled or done the state index is simply taken from cache
            int CreateDestState(
                PairListAutomaton.State mappingState,
                Automaton<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton>.State srcState)
            {
                var destPair = (mappingState.Index, srcState.Index);
                if (!destStateCache.TryGetValue(destPair, out var destStateIndex))
                {
                    var destState = result.AddState();
                    destState.SetEndWeight(mappingState.EndWeight * srcState.EndWeight);
                    stack.Push((mappingState.Index, srcState.Index, destState.Index));
                    destStateCache[destPair] = destState.Index;
                    destStateIndex = destState.Index;
                }

                return destStateIndex;
            }

            // Populate the stack with start destination state
            result.StartStateIndex = CreateDestState(mappingAutomaton.Start, srcAutomaton.Start);

            while (stack.Count > 0)
            {
                var (mappingStateIndex, srcStateIndex, destStateIndex) = stack.Pop();

                var mappingState = mappingAutomaton.States[mappingStateIndex];
                var srcState = srcAutomaton.States[srcStateIndex];
                var destState = result[destStateIndex];

                // Iterate over transitions from mappingState
                foreach (var mappingTransition in mappingState.Transitions)
                {
                    var childMappingState = mappingState.Owner.States[mappingTransition.DestinationStateIndex];

                    // Epsilon transition case
                    if (IsSrcEpsilon(mappingTransition))
                    {
                        var destElementDistribution =
                            mappingTransition.ElementDistribution.HasValue
                                ? mappingTransition.ElementDistribution.Value.Second
                                : Option.None;
                        var childDestStateIndex = CreateDestState(childMappingState, srcState);
                        destState.AddTransition(destElementDistribution, mappingTransition.Weight, childDestStateIndex, mappingTransition.Group);
                        continue;
                    }

                    // Iterate over states and transitions in the closure of srcState
                    foreach (var srcTransition in srcState.Transitions)
                    {
                        Debug.Assert(!srcTransition.IsEpsilon, "The automaton being projected must be epsilon-free.");
                        
                        var srcChildState = srcState.Owner.States[srcTransition.DestinationStateIndex];

                        var projectionLogScale = mappingTransition.ElementDistribution.Value.ProjectFirst(
                            srcTransition.ElementDistribution.Value, out var destElementDistribution);
                        if (double.IsNegativeInfinity(projectionLogScale))
                        {
                            continue;
                        }

                        var destWeight = Weight.Product(mappingTransition.Weight, srcTransition.Weight, Weight.FromLogValue(projectionLogScale));
                        var childDestStateIndex = CreateDestState(childMappingState, srcChildState);
                        destState.AddTransition(destElementDistribution, destWeight, childDestStateIndex, mappingTransition.Group);
                    }
                }
            }

            var simplification = new Automaton<TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton>.Simplification(result, null);
            simplification.RemoveDeadStates();
            simplification.SimplifyIfNeeded();

            return result.GetAutomaton();
        }

        /// <summary>
        /// Computes <c>g(b) = f(A) T(A, b)</c>, where <c>T(a, b)</c> is the current transducer and <c>A</c> is a given sequence.
        /// </summary>
        /// <param name="srcSequence">The sequence to project.</param>
        /// <returns>The projection.</returns>
        /// <remarks>
        /// Using this method is more efficient than applying <see cref="ProjectSource(TSrcAutomaton)"/>
        /// to the automaton representation of a projected sequence.
        /// </remarks>
        /// <remarks>
        /// The code of this method has a lot in common with the code of Automaton.SetToProduct.
        /// Unfortunately, it's not clear how to avoid the duplication in the current design.
        /// </remarks>
        public TDestAutomaton ProjectSource(TSrcSequence srcSequence)
        {
            Argument.CheckIfNotNull(srcSequence, "srcSequence");

            var mappingAutomaton = this.sequencePairToWeight;
            if (mappingAutomaton.IsCanonicZero())
            {
                return Automaton<TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton>.Zero();
            }

            var sourceSequenceManipulator =
                Automaton<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton>.SequenceManipulator;
            var srcSequenceLength = sourceSequenceManipulator.GetLength(srcSequence);
            
            var result = new Automaton<TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton>.Builder();
            var destStateCache = new Dictionary<(int, int), int>();
            var stack = new Stack<(int state1, int state2, int destStateIndex)>();

            // Creates destination state and schedules projection computation for it.
            // If computation is already scheduled or done the state index is simply taken from cache
            int CreateDestState(PairListAutomaton.State mappingState, int srcSequenceIndex)
            {
                var destPair = (mappingState.Index, srcSequenceIndex);
                if (!destStateCache.TryGetValue(destPair, out var destStateIndex))
                {
                    var destState = result.AddState();
                    destState.SetEndWeight(
                        srcSequenceIndex == srcSequenceLength
                            ? mappingState.EndWeight
                            : Weight.Zero);
                    stack.Push((mappingState.Index, srcSequenceIndex, destState.Index));
                    destStateCache[destPair] = destState.Index;
                    destStateIndex = destState.Index;
                }

                return destStateIndex;
            }

            // Populate the stack with start destination state
            result.StartStateIndex = CreateDestState(mappingAutomaton.Start, 0);

            while (stack.Count > 0)
            {
                var (mappingStateIndex, srcSequenceIndex, destStateIndex) = stack.Pop();

                var mappingState = mappingAutomaton.States[mappingStateIndex];
                var destState = result[destStateIndex];

                // Enumerate transitions from the current mapping state
                foreach (var mappingTransition in mappingState.Transitions)
                {
                    var destMappingState = mappingState.Owner.States[mappingTransition.DestinationStateIndex];

                    // Epsilon transition case
                    if (IsSrcEpsilon(mappingTransition))
                    {
                        var destElementWeights =
                            mappingTransition.ElementDistribution.HasValue
                                ? mappingTransition.ElementDistribution.Value.Second
                                : Option.None;
                        var childDestStateIndex = CreateDestState(destMappingState, srcSequenceIndex);
                        destState.AddTransition(destElementWeights, mappingTransition.Weight, childDestStateIndex, mappingTransition.Group);
                        continue;
                    }

                    // Normal transition case - Find epsilon-reachable states
                    if (srcSequenceIndex < srcSequenceLength)
                    {
                        var srcSequenceElement = sourceSequenceManipulator.GetElement(srcSequence, srcSequenceIndex);

                        var projectionLogScale = mappingTransition.ElementDistribution.Value.ProjectFirst(
                            srcSequenceElement, out var destElementDistribution);
                        if (double.IsNegativeInfinity(projectionLogScale))
                        {
                            continue;
                        }

                        var weight = mappingTransition.Weight * Weight.FromLogValue(projectionLogScale);
                        var childDestState = CreateDestState(destMappingState,srcSequenceIndex + 1);
                        destState.AddTransition(destElementDistribution, weight, childDestState, mappingTransition.Group);
                    }
                }
            }

            var simplification = new Automaton<TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton>.Simplification(result, null);
            simplification.RemoveDeadStates();
            simplification.SimplifyIfNeeded();

            return result.GetAutomaton();
        }

        /// <summary>
        /// Creates a copy of this transducer.
        /// </summary>
        /// <returns>The created copy.</returns>
        public TThis Clone()
        {
            var result = new TThis();
            result.SetTo((TThis)this);
            return result;
        }

        /// <summary>
        /// Replaces this transducer with a copy of a given transducer.
        /// </summary>
        /// <param name="otherTransducer">The transducer to replace this transducer with.</param>
        public void SetTo(TThis otherTransducer)
        {
            Argument.CheckIfNotNull(otherTransducer, "otherTransducer");

            this.sequencePairToWeight.SetTo(otherTransducer.sequencePairToWeight);
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Checks whether a given transducer transition is either epsilon of has epsilon source element.
        /// </summary>
        /// <param name="transition">The transition to check.</param>
        /// <returns>A value indicating whether a given transducer transition is either epsilon of has epsilon source element.</returns>
        private static bool IsSrcEpsilon(PairListAutomaton.Transition transition) =>
            !transition.ElementDistribution.HasValue || !transition.ElementDistribution.Value.First.HasValue;

        #endregion

        #region Nested classes

        /// <summary>
        /// Represents an automaton that maps lists of element pairs to real values. Such automata are used to represent transducers internally.
        /// </summary>
        protected class PairListAutomaton :
            ListAutomaton<List<Pair<Option<TSrcElement>, Option<TDestElement>>>, Pair<Option<TSrcElement>, Option<TDestElement>>, TPairDistribution, PairListAutomaton>
        {
            /// <summary>
            /// Computes a set of outgoing transitions from a given state of the determinization result.
            /// </summary>
            /// <param name="sourceState">The source state of the determinized automaton represented as 
            /// a set of (stateId, weight) pairs, where state ids correspond to states of the original automaton.</param>
            /// <returns>
            /// A collection of (element distribution, weight, weighted state set) triples corresponding to outgoing transitions from <paramref name="sourceState"/>.
            /// The first two elements of a tuple define the element distribution and the weight of a transition.
            /// The third element defines the outgoing state.
            /// </returns>
            protected override IEnumerable<(TPairDistribution, Weight, Determinization.WeightedStateSet)> GetOutgoingTransitionsForDeterminization(
                Determinization.WeightedStateSet sourceState)
            {
                throw new NotImplementedException("Determinization is not yet supported for this type of automata.");
            }
        }

        #endregion
    }
}