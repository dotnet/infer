// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Math;
    using System;
    using System.Collections.Generic;
    using System.Text;

    /// <typeparam name="TSequence">The type of a sequence.</typeparam>
    /// <typeparam name="TElement">The type of a sequence element.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over sequence elements.</typeparam>
    /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate sequences.</typeparam>
    /// <typeparam name="TAutomaton">The type of a weighted finite state automaton, that can be used to
    /// represent all weight functions.</typeparam>
    public partial class WeightFunctions<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>
        where TSequence : class, IEnumerable<TElement>
        where TElementDistribution : IImmutableDistribution<TElement, TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, CanComputeProduct<TElementDistribution>, CanCreatePartialUniform<TElementDistribution>, SummableExactly<TElementDistribution>, new()
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
    {

        /// <summary>
        /// The part of <see cref="IWeightFunction{TThis}"/> API,
        /// that is agnostic of the concrete weight function class. Implementations of closed constructed
        /// <see cref="IWeightFunction{TThis}"/> can all be upcasted to this type.
        /// </summary>
        /// <remarks>Types implementing this interface must be constant and thread-safe.</remarks>
        [Quality(QualityBand.Experimental)]
        public interface IWeightFunction
        {
            /// <summary>
            /// Returns an automaton representation of the current weight function. If the weight function is normalized,
            /// the resulting automaton is stochastic, i.e. the sum of weights of all the outgoing transitions and the ending weight is 1 for every node.
            /// </summary>
            TAutomaton AsAutomaton();

            /// <summary>
            /// Gets the only point to which the current function assigns a non-zero weight.
            /// </summary>
            /// <exception cref="System.InvalidOperationException">Thrown when an attempt is made to get the <see cref="Point"/>
            /// of a non-pointmass weight function.</exception>
            TSequence Point { get; }

            /// <summary>
            /// Gets a value indicating whether the current weight function assigns a non-zero weight to exactly one point.
            /// </summary>
            bool IsPointMass { get; }

            /// <summary>
            /// Gets a value indicating whether the current weight function is represented as an automaton internally.
            /// </summary>
            bool UsesAutomatonRepresentation { get; }

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
            /// <exception cref="AutomatonEnumerationCountException">Thrown if enumeration is too large.</exception>
            /// <returns>The sequences in the support of this weight function</returns>
            IEnumerable<TSequence> EnumerateSupport(int maxCount = 1000000);

            /// <summary>
            /// Tries to enumerate support of this weight function.
            /// </summary>
            /// <param name="maxCount">The maximum support enumeration count.</param>
            /// <param name="result">The sequences in the support of this weight function</param>
            /// <returns>True if successful, false otherwise</returns>
            bool TryEnumerateSupport(int maxCount, out IEnumerable<TSequence> result);

            /// <summary>
            /// Enumerates paths through this weight function.
            /// </summary>
            /// <returns>The paths through this weight function, with their log weights</returns>
            IEnumerable<Tuple<List<TElementDistribution>, double>> EnumeratePaths();

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
            /// Returns a string that represents the weight function.
            /// </summary>
            /// <param name="appendElement">Optional method for appending at the element distribution level.</param>
            /// <returns>A string that represents the weight function.</returns>
            string ToString(Action<TElementDistribution, StringBuilder> appendElement);
        }

        /// <summary>
        /// Interface for a function that maps arbitrary sequences of elements to real values,
        /// that can, but not necessarily has to be, represented as a weighted finite state automaton.
        /// </summary>
        /// <typeparam name="TThis">The type of a concrete weight function class.</typeparam>
        /// <remarks>Types implementing this interface must be constant and thread-safe.</remarks>
        [Quality(QualityBand.Experimental)]
        public interface IWeightFunction<TThis> : IWeightFunction, IEquatable<TThis>
            where TThis : IWeightFunction<TThis>, new()
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
            /// <param name="weightFunction">The weight function to compute the sum with.</param>
            /// <param name="weight2">The weight of the <paramref name="weightFunction"/>.</param>
            TThis Sum(double weight1, TThis weightFunction, double weight2);

            /// <summary>
            /// Computes the weighted sum of the current weight function and a given weight function.
            /// </summary>
            /// <param name="logWeight1">The logarithm of the weight of the current weight function.</param>
            /// <param name="weightFunction">The weight function to compute the sum with.</param>
            /// <param name="logWeight2">The logarithm of the weight of the <paramref name="weightFunction"/>.</param>
            TThis SumLog(double logWeight1, TThis weightFunction, double logWeight2);

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

            /// <summary>
            /// If the current weight function uses groups, returns a dictionary [group index] -> group.
            /// Otherwise, returns an empty dictionary.
            /// </summary>
            /// <returns>A possibly empty dictionary [group index] -> group.</returns>
            Dictionary<int, TThis> GetGroups();

            /// <summary>
            /// Gets a value indicating how close this weight function is to a given one
            /// in terms of weights they assign to sequences.
            /// </summary>
            /// <param name="that">The other weight function.</param>
            /// <returns>A non-negative value, which is close to zero if the two weight functions assign similar values to all sequences.</returns>
            double MaxDiff(TThis that);

            /// <summary>
            /// Attempts to normalize the weight function so that sum of its values on all possible sequences equals to one (if it is possible)
            /// and returns the result in an out parameter.
            /// If successful, calls to <see cref="IWeightFunction.AsAutomaton"/>
            /// methods of the resulting function produce a stochastic automaton,
            /// i.e. an automaton, in which the sum of weights of all the outgoing transitions and the ending weight is 1 for every node.
            /// </summary>
            /// <param name="normalizedFunction">Result of the normaliztion attempt.</param>
            /// <param name="logNormalizer">When the function returns, contains the logarithm of the normalizer.</param>
            /// <returns><see langword="true"/> if the weight function was successfully normalized, <see langword="false"/> otherwise.</returns>
            bool TryNormalizeValues(out TThis normalizedFunction, out double logNormalizer);

            /// <summary>
            /// Returns the weight function converted to the normalized form e.g. using special
            /// case structures for point masses and functions with small support.
            /// </summary>
            TThis NormalizeStructure();
        }

        /// <summary>
        /// Interface for a factory of weight functions, that is reponsible for performing operations that result
        /// in weight functions, but not necessarily accept them as arguments.
        /// </summary>
        /// <typeparam name="TWeightFunction">The type of a concrete weight function class.</typeparam>
        [Quality(QualityBand.Experimental)]
        public interface IWeightFunctionFactory<TWeightFunction>
            where TWeightFunction : IWeightFunction<TWeightFunction>, new()
        {
            /// <summary>
            /// Creates a weight function which is zero everywhere.
            /// </summary>
            /// <returns>The created weight function</returns>
            TWeightFunction Zero();

            /// <summary>
            /// Creates a point mass weight function.
            /// </summary>
            /// <param name="point">The point.</param>
            /// <returns>The created point mass weight function.</returns>
            TWeightFunction PointMass(TSequence point);

            /// <summary>
            /// Creates a weight function that assigns provided weights
            /// to provided sequences and zero weights to all other sequnces.
            /// </summary>
            /// <param name="sequenceWeightPairs">Pairs of sequences and corresponding weights</param>
            /// <returns>The created point mass weight function.</returns>
            TWeightFunction FromValues(IEnumerable<KeyValuePair<TSequence, double>> sequenceWeightPairs);

            /// <summary>
            /// Creates a weight function that is equal to the provided automaton on all sequences.
            /// </summary>
            /// <param name="automaton">An automaton defining the weight function.</param>
            /// <returns>The created point mass weight function.</returns>
            TWeightFunction FromAutomaton(TAutomaton automaton);

            /// <summary>
            /// Creates a weight function which maps every allowed sequence to
            /// a given value and maps all other sequences to zero.
            /// A sequence is allowed if all its elements have non-zero probability under a given distribution.
            /// </summary>
            /// <param name="logValue">The logarithm of the value to map every sequence to.</param>
            /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
            /// <returns>The created weight function</returns>
            TWeightFunction ConstantLog(double logValue, TElementDistribution allowedElements);

            /// <summary>
            /// Creates a weight function constant on the support of a given weight function.
            /// </summary>
            /// <param name="logValue">The logarithm of the desired value on the support of <paramref name="weightFunction"/>.</param>
            /// <param name="weightFunction">The weight function.</param>
            /// <exception cref="NotImplementedException">Thrown if if the <paramref name="weightFunction"/> cannot be determinized.</exception>
            /// <returns>The created weight function</returns>
            TWeightFunction ConstantOnSupportOfLog(double logValue, TWeightFunction weightFunction);

            /// <summary>
            /// Computes the sum of given weight functions.
            /// </summary>
            /// <param name="weightFunctions">The weight functions to sum.</param>
            /// <returns>The sum of the given weight functions</returns>
            TWeightFunction Sum(IEnumerable<TWeightFunction> weightFunctions);
        }
    }
}
