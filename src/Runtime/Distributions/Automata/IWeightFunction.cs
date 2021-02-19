// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;
    using System;
    using System.Collections.Generic;

    [Quality(QualityBand.Experimental)]
    public interface IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
    {
        /// <summary>
        /// Returns an automaton representation of the current weight function. If the weight function is normalized,
        /// the resulting automaton is stochastic, i.e. the sum of weights of all the outgoing transitions and the ending weight is 1 for every node.
        /// </summary>
        TAutomaton AsAutomaton();

        /// <summary>
        /// Gets or sets the only point to which the current function assigns a non-zero weight.
        /// </summary>
        /// <exception cref="System.InvalidOperationException">Thrown when an attempt is made to get the <see cref="Point"/>
        /// of a non-pointmass weight function.</exception>
        TSequence Point { get; set; }

        /// <summary>
        /// Gets a value indicating whether the current weight function assigns a non-zero weight to exactly one point.
        /// </summary>
        bool IsPointMass { get; }

        /// <summary>
        /// Gets a value indicating whether the current weight function is represented as an automaton internally.
        /// </summary>
        bool UsesAutomatonRepresentation { get; }

        /// <summary>
        /// Attempts to normalize the weight function so that sum of its values on all possible sequences equals to one (if it is possible).
        /// If successful, future calls to <see cref="AsAutomaton"/> produce a stochastic automaton,
        /// i.e. an automaton, in which the sum of weights of all the outgoing transitions and the ending weight is 1 for every node.
        /// </summary>
        /// <param name="logNormalizer">When the function returns, contains the logarithm of the normalizer.</param>
        /// <returns><see langword="true"/> if the automaton was successfully normalized, <see langword="false"/> otherwise.</returns>
        bool TryNormalizeValues(out double logNormalizer);

        /// <summary>
        /// Computes the logarithm of the normalizer (sum of values of the weight function on all sequences).
        /// </summary>
        /// <returns>The logarithm of the normalizer.</returns>
        /// <remarks>Returns <see cref="double.PositiveInfinity"/> if the sum diverges.</remarks>
        double GetLogNormalizer();

        /// <summary>
        /// Enumerates support of this weight function when possible.
        /// </summary>
        /// <param name="maxCount">The maximum support enumeration count.</param>
        /// <param name="tryDeterminize">Try to determinize if this is a string automaton</param>
        /// <exception cref="AutomatonEnumerationCountException">Thrown if enumeration is too large.</exception>
        /// <returns>The sequences in the support of this weight function</returns>
        IEnumerable<TSequence> EnumerateSupport(int maxCount = 1000000, bool tryDeterminize = true);

        /// <summary>
        /// Tries to enumerate support of this weight function.
        /// </summary>
        /// <param name="maxCount">The maximum support enumeration count.</param>
        /// <param name="result">The sequences in the support of this weight function</param>
        /// <param name="tryDeterminize">Try to determinize if this is a string automaton</param>
        /// <returns>True if successful, false otherwise</returns>
        bool TryEnumerateSupport(int maxCount, out IEnumerable<TSequence> result, bool tryDeterminize = true);

        /// <summary>
        /// Enumerates paths through this weight function.
        /// </summary>
        /// <returns>The paths through this weight function, with their log weights</returns>
        IEnumerable<Tuple<List<TElementDistribution>, double>> EnumeratePaths();

        /// <summary>
        /// Replaces the current weight function with a weight function that assigns provided weights
        /// to provided sequences and zero weights to all other sequnces.
        /// </summary>
        /// <param name="sequenceWeightPairs">Pairs of sequences and corresponding weights</param>
        void SetValues(IEnumerable<KeyValuePair<TSequence, double>> sequenceWeightPairs);

        /// <summary>
        /// Computes the logarithm of the value of the weight function on a given sequence.
        /// </summary>
        /// <param name="sequence">The sequence to compute the value on.</param>
        /// <returns>The logarithm of the value.</returns>
        double GetLogValue(TSequence sequence);

        /// <summary>
        /// Checks whether the current weight function is zero on all sequences.
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if the current weight function is zero on all sequences,
        /// <see langword="false"/> otherwise.
        /// </returns>
        bool IsZero();

        /// <summary>
        /// Replaces the current weight function with a weight function which is zero everywhere.
        /// </summary>
        void SetToZero();

        /// <summary>
        /// Replaces the current weight function with an weight function which maps every allowed sequence to
        /// a given value and maps all other sequences to zero.
        /// A sequence is allowed if all its elements have non-zero probability under a given distribution.
        /// </summary>
        /// <param name="logValue">The logarithm of the value to map every sequence to.</param>
        /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
        void SetToConstantLog(double logValue, TElementDistribution allowedElements);

        /// <summary>
        /// Checks if the weight function uses groups.
        /// </summary>
        /// <returns><see langword="true"/> if the weight function uses groups, <see langword="false"/> otherwise.</returns>
        bool UsesGroups { get; }

        /// <summary>
        /// Determines whether this weight function has the specified group (if applicable).
        /// </summary>
        /// <param name="group">The specified group.</param>
        /// <returns>True if it the weight function has this group, false otherwise.</returns>
        bool HasGroup(int group);

        /// <summary>
        /// Modifies the weight function to be in normalized form e.g. using special
        /// case structures for point masses and fucntions with small support.
        /// </summary>
        void NormalizeStructure();
    }

    [Quality(QualityBand.Experimental)]
    public interface IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis> :
        IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
        where TThis : IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis>, new()
    {

        /// <summary>
        /// Creates a weight function <c>f'(st) = f(s)</c>, where <c>f(s)</c> is the current weight function
        /// and <c>t</c> is the given sequence.
        /// </summary>
        /// <param name="sequence">The sequence.</param>
        /// <param name="group">The group.</param>
        /// <returns>The created weight function.</returns>
        TThis Append(TSequence sequence, int group = 0);

        /// <summary>
        /// Creates a weight function <c>f'(s) = sum_{tu=s} f(t)g(u)</c>, where <c>f(t)</c> is the current weight function
        /// and <c>g(u)</c> is the given weight function.
        /// </summary>
        /// <param name="weightFunction">The weight function to append.</param>
        /// <param name="group">The group.</param>
        /// <returns>The created weight function.</returns>
        TThis Append(TThis weightFunction, int group = 0);

        /// <summary>
        /// Computes the sum of the current weight function and a given weight function.
        /// </summary>
        /// <param name="weightFunction">The weight function to compute the sum with.</param>
        /// <returns>The computed sum.</returns>
        TThis Sum(TThis weightFunction);

        /// <summary>
        /// Computes the weighted sum of the current weight function and a given weight function.
        /// </summary>
        /// <param name="weight1">The weight of the current weight function.</param>
        /// <param name="weight2">The weight of the <paramref name="weightFunction"/>.</param>
        /// <param name="weightFunction">The weight function to compute the sum with.</param>
        TThis Sum(double weight1, double weight2, TThis weightFunction);

        /// <summary>
        /// Computes the weighted sum of the current weight function and a given weight function.
        /// </summary>
        /// <param name="logWeight1">The logarithm of the weight of the current weight function.</param>
        /// <param name="logWeight2">The logarithm of the weight of the <paramref name="weightFunction"/>.</param>
        /// <param name="weightFunction">The weight function to compute the sum with.</param>
        TThis SumLog(double logWeight1, double logWeight2, TThis weightFunction);

        /// <summary>
        /// Replaces the current weight function with a sum of given weight functions.
        /// </summary>
        /// <param name="weightFunctions">The weight functions to sum.</param>
        void SetToSum(IEnumerable<TThis> weightFunctions);

        /// <summary>
        /// Computes the product of the current weight function and a given one.
        /// </summary>
        /// <param name="weightFunction">The weight function to compute the product with.</param>
        /// <returns>The computed product.</returns>
        TThis Product(TThis weightFunction);

        /// <summary>
        /// Creates a weight function <c>g(s) = sum_{k=Kmin}^{Kmax} sum_{t1 t2 ... tk = s} f(t1)f(t2)...f(tk)</c>,
        /// where <c>f(t)</c> is the current weight function, and <c>Kmin</c> and <c>Kmax</c> are the minimum
        /// and the maximum number of factors in a sum term.
        /// </summary>
        /// <param name="minTimes">The minimum number of factors in a sum term. Defaults to 1.</param>
        /// <param name="maxTimes">An optional maximum number of factors in a sum term.</param>
        /// <returns>The created weight function.</returns>
        TThis Repeat(int minTimes = 1, int? maxTimes = null);

        /// <summary>
        /// Scales the weight function and returns the result.
        /// </summary>
        /// <param name="logScale">The logarithm of the scale.</param>
        /// <returns>The scaled weight function.</returns>
        TThis ScaleLog(double logScale);

        Dictionary<int, TThis> GetGroups();

        /// <summary>
        /// Gets a value indicating how close this weight function is to a given one
        /// in terms of weights they assign to sequences.
        /// </summary>
        /// <param name="that">The other weight function.</param>
        /// <returns>A non-negative value, which is close to zero if the two weight functions assign similar values to all sequences.</returns>
        double MaxDiff(TThis that);

        /// <summary>
        /// Sets the weight function to be constant on the support of a given weight function.
        /// </summary>
        /// <param name="logValue">The logarithm of the desired value on the support of <paramref name="weightFunction"/>.</param>
        /// <param name="weightFunction">The weight function.</param>
        ///<exception cref="NotImplementedException">Thrown if if the <paramref name="weightFunction"/> cannot be determinized.</exception>
        void SetToConstantOnSupportOfLog(double logValue, TThis weightFunction);
    }

    [Quality(QualityBand.Experimental)]
    public interface IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton, TThis> : IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TPointMass : PointMassWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>, new()
        where TDictionary : DictionaryWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TPointMass, TDictionary, TAutomaton>, new()
        where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
        where TThis : IWeightFunction<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton, TThis>, new()
    {
    }
}
