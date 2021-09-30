// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// A base class for weighted finite state transducers that are defined on sequences of different types.
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
    /// <typeparam name="TThis">The type of a concrete transducer class.</typeparam>
    public abstract class Transducer<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TThis> :
        TransducerBase<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, ImmutablePairDistribution<TSrcElement, TSrcElementDistribution, TDestElement, TDestElementDistribution>, TThis>
        where TSrcElementDistribution : IImmutableDistribution<TSrcElement, TSrcElementDistribution>, CanGetLogAverageOf<TSrcElementDistribution>, CanComputeProduct<TSrcElementDistribution>, CanCreatePartialUniform<TSrcElementDistribution>, SummableExactly<TSrcElementDistribution>, Sampleable<TSrcElement>, new()
        where TDestElementDistribution : IImmutableDistribution<TDestElement, TDestElementDistribution>, CanGetLogAverageOf<TDestElementDistribution>, CanComputeProduct<TDestElementDistribution>, CanCreatePartialUniform<TDestElementDistribution>, SummableExactly<TDestElementDistribution>, Sampleable<TDestElement>, new()
        where TSrcSequence : class, IEnumerable<TSrcElement>
        where TDestSequence : class, IEnumerable<TDestElement>
        where TSrcSequenceManipulator : ISequenceManipulator<TSrcSequence, TSrcElement>, new()
        where TDestSequenceManipulator : ISequenceManipulator<TDestSequence, TDestElement>, new()
        where TSrcAutomaton : Automaton<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton>, new()
        where TDestAutomaton : Automaton<TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton>, new()
        where TThis : Transducer<TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TThis>, new()
    {
        /// <summary>
        /// Creates a transducer <c>T'(b, a) = T(a, b)</c>, where <c>T(a, b)</c> is a given transducer.
        /// </summary>
        /// <typeparam name="TThat">The type of the resulting transducer.</typeparam>
        /// <param name="transducer">The transducer to transpose.</param>
        /// <returns>The created transducer.</returns>
        public static TThat Transpose<TThat>(TThis transducer)
            where TThat : Transducer<TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton, TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton, TThat>, new()
        {
            Argument.CheckIfNotNull(transducer, nameof(transducer));

            return new TThat
            {
                sequencePairToWeight = transducer.sequencePairToWeight.ApplyFunction<
                    List<(Option<TDestElement>, Option<TSrcElement>)>,
                    (Option<TDestElement>, Option<TSrcElement>),
                    ImmutablePairDistribution<TDestElement, TDestElementDistribution, TSrcElement, TSrcElementDistribution>,
                    ListManipulator<List<(Option<TDestElement>, Option<TSrcElement>)>, (Option<TDestElement>, Option<TSrcElement>)>,
                    TransducerBase<
                        TDestSequence, TDestElement, TDestElementDistribution, TDestSequenceManipulator, TDestAutomaton,
                        TSrcSequence, TSrcElement, TSrcElementDistribution, TSrcSequenceManipulator, TSrcAutomaton,
                        ImmutablePairDistribution<TDestElement, TDestElementDistribution, TSrcElement, TSrcElementDistribution>,
                        TThat>.PairListAutomaton>(
                (dist, weight, group) => ValueTuple.Create(dist.HasValue ? Option.Some(dist.Value.Transpose()) : Option.None, weight))
            };
        }
    }

    /// <summary>
    /// A base class for weighted finite state transducers that are defined on sequences of the same type.
    /// </summary>
    /// <typeparam name="TSequence">The type of a sequence in a pair.</typeparam>
    /// <typeparam name="TElement">The type of an element of <typeparamref name="TSequence"/>.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over <typeparamref name="TElement"/>.</typeparam>
    /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate sequences of type <typeparamref name="TSequence"/>.</typeparam>
    /// <typeparam name="TAutomaton">The type of an automaton defined on <typeparamref name="TSequence"/>.</typeparam> 
    /// <typeparam name="TThis">The type of a concrete transducer class.</typeparam>
    public abstract class Transducer<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis> :
        TransducerBase<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, ImmutablePairDistribution<TElement, TElementDistribution>, TThis>
        where TElementDistribution : IImmutableDistribution<TElement, TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, CanComputeProduct<TElementDistribution>, CanCreatePartialUniform<TElementDistribution>, SummableExactly<TElementDistribution>, Sampleable<TElement>, new()
        where TSequence : class, IEnumerable<TElement>
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
        where TThis : Transducer<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis>, new()
    {
        /// <summary>
        /// Creates a transducer <c>T(a, b) = I[a = b]</c>.
        /// </summary>
        /// <returns>The created transducer.</returns>
        public static TThis Copy()
        {
            return Copy(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.Constant(1.0));
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = I[A[i] is in C for all i] I[a = b]</c>,
        /// where <c>C</c> is the support of a given element distribution.
        /// </summary>
        /// <param name="allowedElements">Specifies the elements allowed in the transducer arguments.</param>
        /// <returns>The created transducer.</returns>
        public static TThis Copy(TElementDistribution allowedElements)
        {
            return Copy(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.Constant(1.0, allowedElements));
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = I[a = A] I[a = b]</c>, where <c>A</c> is a given sequence.
        /// </summary>
        /// <param name="sequence">The sequence to constrain the arguments.</param>
        /// <returns>The created transducer.</returns>
        public static TThis Copy(TSequence sequence)
        {
            return Copy(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.ConstantOn(1.0, sequence));
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = f(a) I[a = b]</c>, where <c>f(a)</c> is a given automaton.
        /// </summary>
        /// <param name="automaton">The automaton to weight the sequence.</param>
        /// <param name="group">The group.</param>
        /// <returns>The created transducer.</returns>
        public static TThis Copy(TAutomaton automaton, int group = 0)
        {
            Argument.CheckIfNotNull(automaton, nameof(automaton));

            var result = new TThis
            {
                sequencePairToWeight = automaton.ApplyFunction<
                    List<(Option<TElement>, Option<TElement>)>,
                    (Option<TElement>, Option<TElement>),
                    ImmutablePairDistribution<TElement, TElementDistribution>,
                    ListManipulator<List<(Option<TElement>, Option<TElement>)>, (Option<TElement>, Option<TElement>)>,
                    PairListAutomaton>(
                (transitionElementDistribution, transitionWeight, transitionGroup) =>
                    {
                        if (!transitionElementDistribution.HasValue)
                        {

                            return ValueTuple.Create<Option<ImmutablePairDistribution<TElement, TElementDistribution>>, Weight>(Option.None, transitionWeight);
                        }

                        if ((group == 0) || (transitionGroup == group))
                        {
                            // If a target group is specified: copy if this group is the target group
                            return ValueTuple.Create(
                                Option.Some(
                                    ImmutablePairDistribution<TElement, TElementDistribution>.Constrained(
                                        transitionElementDistribution.Value,
                                        transitionElementDistribution.Value.CreatePartialUniform())),
                                transitionWeight);
                        }

                        // Otherwise consume but don't copy
                        return ValueTuple.Create(
                            Option.Some(ImmutablePairDistribution<TElement, TElementDistribution>.FromFirst(transitionElementDistribution.Value)),
                            transitionWeight);
                    })
            };

            if (group != 0)
            {
                result.sequencePairToWeight = result.sequencePairToWeight.WithGroup(0);
            }

            return result;
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = I[a[0] = c, |a| = 1] I[a = b]</c>,
        /// where <c>c</c> is a given element.
        /// </summary>
        /// <param name="element">The element allowed in the transducer arguments.</param>
        /// <returns>The created transducer.</returns>
        public static TThis CopyElement(TElement element)
        {
            return Copy(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.ConstantOnElement(1.0, element));
        }

        /// <summary>
        /// Creates a transducer <c>T(a, b) = I[a[0] is in C, |a| = 1] I[a = b]</c>,
        /// where <c>C</c> is the support of a given element distribution.
        /// </summary>
        /// <param name="allowedElements">Specifies the elements allowed in the transducer arguments.</param>
        /// <returns>The created transducer.</returns>
        public static TThis CopyElement(TElementDistribution allowedElements)
        {
            return Copy(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.ConstantOnElement(1.0, allowedElements));
        }

        /// <summary>
        /// Creates a transducer <c>T'(b, a) = T(a, b)</c>, where <c>T(a, b)</c> is a given transducer.
        /// </summary>
        /// <param name="transducer">The transducer to transpose.</param>
        /// <returns>The created transducer.</returns>
        public static TThis Transpose(TThis transducer)
        {
            Argument.CheckIfNotNull(transducer, nameof(transducer));

            // Copy state parameters and transitions
            var builder = PairListAutomaton.Builder.FromAutomaton(transducer.sequencePairToWeight);

            // transpose element distributions in transitions
            for (var stateIndex = 0; stateIndex < builder.StatesCount; stateIndex++)
            {
                for (var iterator = builder[stateIndex].TransitionIterator; iterator.Ok; iterator.Next())
                {
                    var transition = iterator.Value;
                    if (transition.ElementDistribution.HasValue)
                    {
                        transition = transition.With(elementDistribution: transition.ElementDistribution.Value.Transpose());
                        iterator.Value = transition;
                    }
                }
            }

            return new TThis() { sequencePairToWeight = builder.GetAutomaton() };
        }

        /// <summary>
        /// Creates a transducer from a given automaton by applying a converter to its element distributions.
        /// </summary>
        /// <param name="automaton">The automaton.</param>
        /// <param name="transitionTransform">A function for converting automaton transitions into transducer transitions.</param>
        /// <returns>The created transducer.</returns>
        public static TThis FromAutomaton(
            TAutomaton automaton,
            Func<Option<TElementDistribution>, Weight, ValueTuple<Option<ImmutablePairDistribution<TElement, TElementDistribution>>, Weight>> transitionTransform)
        {
            Argument.CheckIfNotNull(transitionTransform, "transitionTransform");

            //var result = automaton.ApplyFunction<TSequence((elementDist, weight, group) => transitionTransform(elementDist, weight));
            var result = new TThis()
            {
                sequencePairToWeight = automaton.ApplyFunction<
                    List<(Option<TElement>, Option<TElement>)>,
                    (Option<TElement>, Option<TElement>),
                    ImmutablePairDistribution<TElement, TElementDistribution>,
                    ListManipulator<List<(Option<TElement>, Option<TElement>)>, (Option<TElement>, Option<TElement>)>,
                    PairListAutomaton>(
                    (elementDist, weight, group) => transitionTransform(elementDist, weight))
            };

            return result;
        }
    }
}