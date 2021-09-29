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


    /// <content>
    /// Contains classes and methods for automata simplification.
    /// </content>
    public abstract partial class Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis>
    {

        /// <summary>
        /// Enumerates support of this automaton when possible.
        /// Element distributions must be either point mass or implement <see cref="CanEnumerateSupport{T}"/>.
        /// </summary>
        /// <param name="maxCount">The maximum support enumeration count.</param>
        /// <exception cref="AutomatonEnumerationCountException">Thrown if enumeration is too large.</exception>
        /// <exception cref="InvalidOperationException">Thrown if distribution on some transition is not enumerable.</exception>
        /// <returns>The sequences in the support of this automaton</returns>
        public IEnumerable<TSequence> EnumerateSupport(int maxCount = 1000000)
        {
            int idx = 0;
            foreach (var seq in this.EnumerateSupportInternal())
            {
                if (seq == null)
                {
                    throw new NotSupportedException();
                }

                if (++idx > maxCount)
                {
                    throw new AutomatonEnumerationCountException(maxCount);
                }

                yield return seq;
            }
        }

        /// <summary>
        /// Tries to enumerate support of this automaton.
        /// Element distributions must be either point mass or implement <see cref="CanEnumerateSupport{T}"/>.
        /// </summary>
        /// <param name="maxCount">The maximum support enumeration count.</param>
        /// <param name="result">The sequences in the support of this automaton</param>
        /// <returns>True if successful, false otherwise</returns>
        public bool TryEnumerateSupport(int maxCount, out IEnumerable<TSequence> result)
            => TryEnumerateSupport(maxCount, out result, int.MaxValue, false);

        /// <summary>
        /// Tries to enumerate support of this automaton.
        /// Element distributions must be either point mass or implement <see cref="CanEnumerateSupport{T}"/>.
        /// </summary>
        /// <param name="maxCount">The maximum support enumeration count.</param>
        /// <param name="result">The sequences in the support of this automaton</param>\
        /// <param name="maxTraversedPaths">
        /// Maximum number of paths in the automaton this function
        /// is allowed to traverse before stopping.
        /// Can be used to limit the performance impact of this call in cases when
        /// the support is useful only if it can be obtained quickly.
        /// </param>
        /// <param name="stopOnNonPointMassElementDistribution">
        /// When set to true, the enumeration is canceled upon encountering a non-point mass
        /// element distribution on a transition.
        /// </param>
        /// <returns>True if successful, false otherwise</returns>
        public bool TryEnumerateSupport(
            int maxCount,
            out IEnumerable<TSequence> result,
            int maxTraversedPaths,
            bool stopOnNonPointMassElementDistribution)
        {
            var limitedResult = new List<TSequence>();
            try
            {
                foreach (var seq in this.EnumerateSupportInternal(maxTraversedPaths, stopOnNonPointMassElementDistribution))
                {
                    if (seq == null || limitedResult.Count >= maxCount)
                    {
                        result = null;
                        return false;
                    }

                    limitedResult.Add(seq);
                }

                result = limitedResult;
                return true;
            }
            catch (InvalidOperationException)
            {
                // This exception can be thrown only if some transitions contain non-enumerable
                // distributions on them.
                result = null;
                return false;
            }
        }


        /// <summary>
        /// Enumerate support of this automaton
        /// </summary>
        /// <param name="maxTraversedPaths">Maximum number of paths in the automaton this function
        /// is allowed to traverse before stopping. Defaults to <see cref="int.MaxValue"/>.
        /// Can be used to limit the performance impact of this call in cases when
        /// the support is useful only if it can be obtained quickly.</param>
        /// <param name="stopOnNonPointMassElementDistribution">
        /// When set to true, the enumeration is canceled upon encountering a non-point mass
        /// element distribution on a transition and a <see langword="null"/> value is yielded.
        /// </param>
        private IEnumerable<TSequence> EnumerateSupportInternal(
            int maxTraversedPaths = int.MaxValue,
            bool stopOnNonPointMassElementDistribution = false)
        {
            var isEnumerable = this.Data.IsEnumerable;
            if (isEnumerable != null && isEnumerable.Value == false)
            {
                // This automaton is definitely not enumerable
                return new TSequence[] { null };
            }

            var enumeration = this.EnumerateSupportInternalWithDuplicates(maxTraversedPaths, stopOnNonPointMassElementDistribution);
            if (Data.IsDeterminized != true)
            {
                // Enumerable.Distinct() uses some internal set implementation (different from HashSet) to store enumerated elements.
                // Somehow, it causes unacceptable slowdowns for large supports, hence the customized implementation.
                enumeration = HashSetBasedDistinct(enumeration);
            }

            return enumeration;

            IEnumerable<TSequence> HashSetBasedDistinct(IEnumerable<TSequence> source)
            {
                // Can not pre-allocate hashsets in netstandard2.0
                // TODO: start preallocatinng after switching to BCL that allows it, e.g. netstandard2.1
                var supportDistinctionSet = new HashSet<TSequence>(SequenceManipulator.SequenceEqualityComparer);

                foreach (var elem in source)
                {
                    if (supportDistinctionSet.Add(elem))
                        yield return elem;
                }
            }
        }

        /// <summary>
        /// Stores information needed for backtracking during support enumeration.
        /// </summary>
        private struct StateEnumerationState
        {
            public int StateIndex;

            // Used for loop and dead ends detection
            public int ProducedCount;
            public int PathLength;
            public int LongestOutputInPath;

            // Used for enumerating transitions reachable from current state
            public int TransitionIndex;
            public int RemainingTransitionsCount;
            public IEnumerator<TElement> ElementEnumerator;
        }

        [Flags]
        private enum StateEnumerationFlags : uint
        {
            IsDeadEnd = 0x80000000,
            DepthMask  = 0x7fffffff,
        }

        private enum TransitionAdvancementResult
        {
            Success,
            NoMoreTransitions,
            ShouldStopEnumeration
        }

        /// <summary>
        /// Enumerate support of this automaton without elimination of duplicate elements
        /// </summary>
        /// <param name="maxTraversedPaths">
        /// Maximum number of paths in the automaton this function
        /// is allowed to traverse before stopping. Defaults to <see cref="int.MaxValue"/>.
        /// Can be used to limit the performance impact of this call in cases when
        /// the support is useful only if it can be obtained quickly.
        /// </param>
        /// <param name="stopOnNonPointMassElementDistribution">
        /// When set to true, the enumeration is canceled upon encountering a non-point mass
        /// element distribution on a transition and a <see langword="null"/> value is yielded.
        /// </param>
        /// <returns>
        /// The sequences supporting this automaton. Sequences may be non-distinct if
        /// automaton is not determinized. A <see langword="null"/> value in enumeration means that
        /// an infinite loop was reached or that the enumeration was stopped because
        /// a condition set by one of this method's parameters was met.
        /// Public <see cref="EnumerateSupport(int)"/> /
        /// <see cref="TryEnumerateSupport(int, out IEnumerable{TSequence}, int, bool)"/>
        /// methods handle null value differently.
        /// </returns>
        /// <remarks>
        /// Conceptually enumerating support is just depth-first traversal of automaton from start
        /// state recording the elements met on path. Real implementation is a little hairy because
        /// of the following reasons:
        /// - Recursion can not be used, instead an explicit stack is used
        /// - Each transition can have a distribution with a lot of support, so it has to be
        ///   enumerated lazily, making state stored on stack large
        /// - Some highly-branching paths can have no end state in them. This fact has to be
        ///   tracked to void spending exponential time traversing states which produce no output
        /// - Loops have to be tracked: some loops make automaton non-enumerable, some don't
        /// - An fast-path for non-branchy automata is implemented. It makes traversing those 10x
        ///   faster by skipping some boilerplate for tracking traversal state in these cases.
        /// </remarks>
        private IEnumerable<TSequence> EnumerateSupportInternalWithDuplicates(
            int maxTraversedPaths = int.MaxValue,
            bool stopOnNonPointMassElementDistribution = false)
        {
            // Sequence of elements on path to current state in automaton
            var sequence = new List<TElement>();

            // Stack of states for backtracking during depth-first traversal
            var stack = new Stack<StateEnumerationState>();

            // Stores 2 bits of data about state:
            // - Is this state a dead end. I.e. end state is not reachable through any path starting
            //   from this state. This flag is computed incrementally most of the time. But if
            //   automaton contains loops, a special procedure `ComputeEndStatesReachability` is
            //   invoked which computes it efficiently for whole graph.
            // - Whether this state is being visited now and if yes - how long is the sequence from
            //   root to this state. This is used for (non-empty) loop detection. We store length + 1,
            //   because 0 value of flags has special meaning of "this state has not been visited yet".
            var flags = new StateEnumerationFlags[this.States.Count];

            // Number of sequences produced by this time. By comparing value of this counter
            // before entering the state and when leaving it is easy to detect whether the state
            // is a dead end.
            var producedCount = 0;

            // Number of paths traversed by this time. Used in early break condition.
            var traversedPathsCount = 0;

            // Enumeration state for current state. A top of traversal stack, materialized in local
            // variable for convenience.
            var current = InitEnumerationState(this.Start.Index, 0, this.Start.CanEnd ? 0 : -1);

            // If true, we can assume that `IsDeadEnd` flag is computed for all states
            var endStateReachabilityComputed = false;

            // Mark the first state as visited
            flags[this.Start.Index] = (StateEnumerationFlags)1;

            if (this.Start.CanEnd)
            {
                // We do not ever "enter" the start state via a transition, so need to handle
                // "start state is also an end state" case explicitly
                yield return SequenceManipulator.ToSequence(sequence);
            }

            while (true)
            {
                if (!TryBacktrackIfNeed())
                {
                    // Nowhere to backtrack, enumerated everything
                    yield break;
                }

                // In theory, this operation could be called only when `current.PathLength`
                // decreases. But that happens in multiple places, so it's easier to truncate path
                // on each iteration, even if it is a noop.
                sequence.RemoveRange(current.PathLength, sequence.Count - current.PathLength);

                var transitionAdvancementResult = AdvanceToNextTransition(out var nextStateIndex, out var nextElement);
                if (transitionAdvancementResult == TransitionAdvancementResult.NoMoreTransitions)
                {
                    // Failed to go to next transition: destination state is unreachable
                    continue;
                }
                if (transitionAdvancementResult == TransitionAdvancementResult.ShouldStopEnumeration)
                {
                    yield return null;
                    yield break;
                }

                if (nextElement.HasValue)
                {
                    sequence.Add(nextElement.Value);
                }

                nextStateIndex = TraverseTrivialStates(nextStateIndex);
                var nextState = this.Data.States[nextStateIndex];

                if (flags[nextStateIndex] != 0)
                {
                    ++traversedPathsCount;
                    if (traversedPathsCount > maxTraversedPaths)
                    {
                        // Hit the limit on the number of returned sequences.
                        yield return null;
                        yield break;
                    }

                    // This states is either dead end or a loop. If this is a dead end, then it
                    // will not be traversed at all.
                    if ((flags[nextStateIndex] & StateEnumerationFlags.IsDeadEnd) != 0)
                    {
                        continue;
                    }

                    // This is a loop, so let's check if it can produce anything.
                    // If it produces anything - then support is in-enumerable.
                    // If it doesn't, there's no point in entering this state at all
                    if (!endStateReachabilityComputed)
                    {
                        ComputeEndStateReachability(flags);
                        endStateReachabilityComputed = true;
                    }

                    if ((flags[nextStateIndex] & StateEnumerationFlags.IsDeadEnd) != 0 ||
                        sequence.Count + 1 == (int)(flags[nextStateIndex] & StateEnumerationFlags.DepthMask))
                    {
                        // Just skip this loop
                        // - Either no end state is reachable from this loop. So it can "produce"
                        //   as many elements as it wants, this does not matter because this
                        //   sequence would never be output.
                        // - Or this loop produces 0 elements (i.e. consists of epsilon transitions
                        //   only), because path was extended by 0 elements after traversing the loop.
                        continue;
                    }

                    // A non-empty loop that can produce something was found. Signal to outer call
                    // via returning null.
                    yield return null;
                    yield break;
                }

                var producedBeforeThisState = producedCount;
                if (nextState.CanEnd && sequence.Count > current.LongestOutputInPath)
                {
                    // Output sequence only if it is longer then any sequence produced on the path
                    // to this state. This is an important optimization in some real automata.
                    // Simplified example: graph `a -> b -> c -> d -> e`. Where both states `a` and
                    // `e` are marked as end states and every arrow corresponds to 2 epsilon
                    // transitions. There are 2^4 = 16 unique paths from `a` to `e`. We need to
                    // detect that `e` is practically a dead end even though it has `CanEnd` flag.
                    // By noticing that `e` and `a` states produce sequences of the same length,
                    // we can assume that `e` is effectively not an end state because it does not
                    // produce any new sequences.
                    ++producedCount;
                    ++traversedPathsCount;
                    if (traversedPathsCount > maxTraversedPaths)
                    {
                        // Hit the limit on the number of returned sequences.
                        yield return null;
                        yield break;
                    }
                    yield return SequenceManipulator.ToSequence(sequence);
                }

                if (nextState.TransitionsCount == 0)
                {
                    // Fast path: there is no need to enter this state at all because we would
                    // backtrack immediately to current state on next iteration.
                    if (!nextState.CanEnd)
                    {
                        flags[nextStateIndex] = StateEnumerationFlags.IsDeadEnd;
                    }
                }
                else
                {
                    // Slow path: store current state on stack and move to next state
                    flags[nextStateIndex] = (StateEnumerationFlags) (sequence.Count + 1);
                    stack.Push(current);
                    current = InitEnumerationState(
                        nextStateIndex,
                        producedBeforeThisState,
                        nextState.CanEnd ? sequence.Count : current.LongestOutputInPath);
                }
            }

            StateEnumerationState InitEnumerationState(
                int index, int producedBeforeThisState, int longestOutputInPath)
            {
                var state = this.Data.States[index];
                return new StateEnumerationState
                {
                    StateIndex = index,
                    ProducedCount = producedBeforeThisState,
                    PathLength = sequence.Count,
                    LongestOutputInPath = longestOutputInPath,
                    TransitionIndex = state.FirstTransitionIndex - 1,
                    RemainingTransitionsCount = state.TransitionsCount,
                    ElementEnumerator = null,
                };
            }

            // Backtracks according to traversal stack if that is needed.
            // Returns false if automaton has been fully enumerated.
            //
            // If backtracking is needed (i.e. no more non-processed transitions left in current
            // state), replaces `current` with the current top of the stack if that's possible.
            // Also updates the `flags` with information whether path through this state produces
            // any output.
            bool TryBacktrackIfNeed()
            {
                // While no more transitions are left in current state - backtrack
                while (current.ElementEnumerator == null && current.RemainingTransitionsCount == 0)
                {
                    if (stack.Count == 0)
                    {
                        // Nowhere to backtrack, this automaton has been fully enumerated
                        if (this.Data.IsEnumerable == null)
                        {
                            this.Data = this.Data.With(isEnumerable: true);
                        }

                        return false;
                    }

                    // If upon leaving current state, number of produced sequences is equal to
                    // the number upon entrance then this is a dead end.
                    flags[current.StateIndex] = current.ProducedCount == producedCount
                        ? StateEnumerationFlags.IsDeadEnd
                        : 0;

                    current = stack.Pop();
                }

                return true;
            }

            // Updates `current` enumeration state by moving to next element/state index reachable
            // from current state.
            TransitionAdvancementResult AdvanceToNextTransition(out int nextStateIndex, out Option<TElement> nextElement)
            {
                Debug.Assert(
                    current.ElementEnumerator != null || current.RemainingTransitionsCount != 0,
                    "TryBacktrack() must skip all states with no transitions left");

                if (current.ElementEnumerator != null)
                {
                    // Advance to next element in current transition
                    nextStateIndex = this.Data.Transitions[current.TransitionIndex].DestinationStateIndex;
                    if ((flags[nextStateIndex] & StateEnumerationFlags.IsDeadEnd) != 0)
                    {
                        // While enumerating current transition we learned that it leads to a
                        // dead end. Stop enumerating elements on this transition, because will
                        // waste time on enumerating dead end.
                        current.ElementEnumerator = null;
                    }
                    else
                    {
                        nextElement = current.ElementEnumerator.Current;
                        if (!current.ElementEnumerator.MoveNext())
                        {
                            // Element done, move to next transition on next iteration
                            current.ElementEnumerator = null;
                        }

                        return TransitionAdvancementResult.Success;
                    }
                }

                // Advance to next transition
                while (current.RemainingTransitionsCount != 0)
                {
                    ++current.TransitionIndex;
                    --current.RemainingTransitionsCount;

                    var transition = this.Data.Transitions[current.TransitionIndex];
                    nextStateIndex = transition.DestinationStateIndex;

                    if (transition.Weight.IsZero ||
                        (flags[nextStateIndex] & StateEnumerationFlags.IsDeadEnd) != 0)
                    {
                        // Do not follow paths which produce nothing and try next transition
                        continue;
                    }

                    if (transition.IsEpsilon)
                    {
                        nextElement = Option.None;
                    }
                    else
                    {
                        var elementDistribution = transition.ElementDistribution.Value;
                        if (elementDistribution.IsPointMass)
                        {
                            nextElement = elementDistribution.Point;
                        }
                        else
                        {
                            if (stopOnNonPointMassElementDistribution)
                            {
                                nextStateIndex = -1;
                                nextElement = Option.None;
                                return TransitionAdvancementResult.ShouldStopEnumeration;
                            }
                            if (!(elementDistribution is CanEnumerateSupport<TElement> supportEnumerator))
                            {
                                this.Data = this.Data.With(isEnumerable: false);
                                throw new InvalidOperationException();
                            }

                            var enumerator = supportEnumerator.EnumerateSupport().GetEnumerator();
                            if (!enumerator.MoveNext())
                            {
                                // This transition is not marked as epsilon (i.e. contains
                                // distribution on it). But this distribution contains no support.
                                // Go to next transition.
                                continue;
                            }

                            nextElement = enumerator.Current;
                            current.ElementEnumerator = enumerator.MoveNext() ? enumerator : null;
                        }
                    }

                    return TransitionAdvancementResult.Success;
                }

                nextStateIndex = -1;
                nextElement = Option.None;
                return TransitionAdvancementResult.NoMoreTransitions;
            }
            
            // Traverses all trivial states - non-terminal states with only one non-epsilon forward
            // transition with non-zero weight.
            //
            // It is safe (and a lot faster) to traverse these states without involving
            // backtracking logic or complicated code for enumerating transitions. The only tricky
            // part is loop detection - because these states are not backtracked, flags on the
            // won't be updated. But that is safe because loop involves at least 1 backward
            // transition, and that one won't be fast-tracked.
            //
            // This procedure is fast-path optimization for automata with very low uncertainty.
            // Most automata in practice belong to this category.
            int TraverseTrivialStates(int currentStateIndex)
            {
                while (true)
                {
                    var state = this.Data.States[currentStateIndex];

                    if (flags[currentStateIndex] != 0 ||
                        state.CanEnd ||
                        state.TransitionsCount != 1)
                    {
                        return currentStateIndex;
                    }

                    var transition = this.Data.Transitions[state.FirstTransitionIndex];

                    if (transition.Weight.IsZero ||
                        transition.IsEpsilon ||
                        transition.DestinationStateIndex <= currentStateIndex)
                    {
                        return currentStateIndex;
                    }

                    var dist = transition.ElementDistribution.Value;

                    if (!dist.IsPointMass)
                    {
                        return currentStateIndex;
                    }

                    sequence.Add(dist.Point);
                    currentStateIndex = transition.DestinationStateIndex;
                }
            }
        }

        /// <summary>
        /// Sets the `IsDeadEnd` flag for some states by traversing graph from end to start by
        /// inverted transitions. All states not reachable via this procedure are dead-ends -
        /// even if they are reachable from start, they will never reach end.
        /// </summary>
        /// <remarks>
        /// Recursive implementation would be simpler but prone to stack overflows with large automatons
        /// </remarks>
        private void ComputeEndStateReachability(StateEnumerationFlags[] flags)
        {
            //// First, build a reversed graph

            var edgePlacementIndices = new int[this.States.Count + 1];
            foreach (var transition in this.Data.Transitions)
            {
                if (!transition.Weight.IsZero)
                {
                    ++edgePlacementIndices[transition.DestinationStateIndex + 1];
                }
            }

            // The element of edgePlacementIndices at index i+1 contains a count of the number of edges 
            // going into the i'th state (the indegree of the state).
            // Convert this into a cumulative count (which will be used to give a unique index to each edge).
            for (var i = 1; i < edgePlacementIndices.Length; ++i)
            {
                edgePlacementIndices[i] += edgePlacementIndices[i - 1];
            }

            var edgeArrayStarts = (int[])edgePlacementIndices.Clone();
            var totalEdgeCount = edgePlacementIndices[this.States.Count];
            var edgeDestinationIndices = new int[totalEdgeCount];

            foreach (var state in this.States)
            {
                foreach (var transition in state.Transitions)
                {
                    if (!transition.Weight.IsZero)
                    {
                        // The unique index for this edge
                        var edgePlacementIndex = edgePlacementIndices[transition.DestinationStateIndex]++;

                        // The source index for the edge (which is the destination edge in the reversed graph)
                        edgeDestinationIndices[edgePlacementIndex] = state.Index;
                    }
                }
            }

            //// Now run a depth-first search to label all reachable nodes
            var stack = new Stack<int>();
            for (var i = 0; i < this.States.Count; ++i)
            {
                if (this.States[i].CanEnd)
                {
                    stack.Push(i);
                }
                else
                {
                    flags[i] |= StateEnumerationFlags.IsDeadEnd;
                }
            }

            while (stack.Count != 0)
            {
                var stateIndex = stack.Pop();
                for (var i = edgeArrayStarts[stateIndex]; i < edgeArrayStarts[stateIndex + 1]; ++i)
                {
                    var destinationIndex = edgeDestinationIndices[i];
                    if ((flags[destinationIndex] & StateEnumerationFlags.IsDeadEnd) != 0)
                    {
                        flags[destinationIndex] &= ~StateEnumerationFlags.IsDeadEnd;
                        stack.Push(destinationIndex);
                    }
                }
            }
        }
    }
}
