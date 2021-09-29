// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Core.Collections;

    /// <summary>
    /// A cache for preallocated temporary data to reduce GC pressure.
    /// </summary>
    /// <remarks>
    /// A lot of automata operations need large short-lived single-function scope temporary data
    /// structures. Allocating them using regular `new` keyword puts a lot op strain on GC.
    /// To avoid that a set of thread-local preallocated objects is kept around.
    ///
    /// These objects are put into a single static class for 3 reasons:
    /// 1. It is easier to see where these optimizations are used.
    /// 2. This class can be non-generic. Accessing non-generic static methods and fields is
    ///    slightly faster.
    /// 3. This also allows to reuse datastructures between methods of different instantiations
    ///    of automata and even between classes (Like `ProductState` which is used by `Automaton`
    ///    and `Transducer` classes).
    /// </remarks>
    internal static class PreallocatedAutomataObjects
    {
        [ThreadStatic]
        private static (GenerationalDictionary<(int, int), int>, Stack<(int, int, int)>) productState;

        [ThreadStatic]
        private static (Stack<int>, TarjanStateInfo[], Stack<(int, int)>) findStronglyConnectedComponentsState;

        [ThreadStatic]
        private static GenerationalDictionary<int, CondensationStateInfo> computeCondensationState;

        [ThreadStatic]
        private static (Stack<(int, int, Weight, int)>, Stack<Weight>, GenerationalDictionary<(int, int), Weight>) getLogValueState;

        [ThreadStatic]
        private static (int[], int[]) buildReversedGraphState;

        [ThreadStatic]
        private static (Stack<int>, bool[]) removeDeadStatesState;

        [ThreadStatic]
        private static int[] removeStatesState;

        /// <summary>
        /// Releases all pre-allocated objects in current thread.
        /// </summary>
        /// <remarks>
        /// <see cref="Automaton{TSequence,TElement,TElementDistribution,TSequenceManipulator,TThis}.UnlimitedStatesComputation"/>
        /// may create huge automata which will cause creation of large pre-allocated objects.
        /// These objects will be kept in memory even if no huge automata are created again and
        /// such large caches are not needed again. To avoid this we free all cached memory
        /// in `UnlimitedStatesComputation.Dispose()`.
        ///
        /// Freeing memory makes caching slightly less efficient. But UnlimitedStatesComputation is
        /// used very rarely so it is ok to just drop caches to avoid wasting too much memory.
        /// </remarks>
        internal static void FreeAllObjects()
        {
            productState = default;
            findStronglyConnectedComponentsState = default;
            computeCondensationState = default;
            getLogValueState = default;
            buildReversedGraphState = default;
            removeDeadStatesState = default;
            removeStatesState = default;
        }

        internal static (GenerationalDictionary<(int, int), int>, Stack<(int, int, int)>) LeaseProductState()
        {
            var result = productState;
            if (result.Item1 == null)
            {
                result = (new GenerationalDictionary<(int, int), int>(), new Stack<(int, int, int)>());
                productState = result;
            }

            result.Item1.Clear();
            result.Item2.Clear();

            return result;
        }

        internal static (Stack<int>, TarjanStateInfo[], Stack<(int, int)>) LeaseFindStronglyConnectedComponentsState(int stateCount)
        {
            var result = findStronglyConnectedComponentsState;
            if (result.Item1 == null)
            {
                result = ValueTuple.Create(
                    new Stack<int>(),
                    new TarjanStateInfo[stateCount],
                    new Stack<(int, int)>());
                findStronglyConnectedComponentsState = result;
            }

            if (result.Item2.Length < stateCount)
            {
                var newLength = Math.Max(stateCount, result.Item2.Length * 2);
                result = (result.Item1, new TarjanStateInfo[newLength], result.Item3);
            }

            result.Item1.Clear();
            Array.Clear(result.Item2, 0, stateCount);
            result.Item3.Clear();

            return result;
        }

        internal static GenerationalDictionary<int, CondensationStateInfo> LeaseComputeCondensationState()
        {
            var result = computeCondensationState;
            if (result == null)
            {
                result = new GenerationalDictionary<int, CondensationStateInfo>();
                computeCondensationState = result;
            }

            // In case some other Condensation object is created at the same time, it
            // should not reuse the same state. In practice (in single therad) 2 Condensation
            // objects never coexist. But there is no guarantee that it won't happen at some point.
            // So each time a state is used, let's reset cache to null. By the time condensation
            // is computed again, this value should be returned into cache
            computeCondensationState = null;
            return result;
        }

        internal static void ReleaseComputeCondensationState(GenerationalDictionary<int, CondensationStateInfo> state)
        {
            state.Clear();
            computeCondensationState = state;
        }

        internal static (Stack<(int, int, Weight, int)>, Stack<Weight>, GenerationalDictionary<(int, int), Weight>) LeaseGetLogValueState()
        {
            var result = getLogValueState;

            if (result.Item1 == null)
            {
                result = ValueTuple.Create(
                    new Stack<(int, int, Weight, int)>(),
                    new Stack<Weight>(),
                    new GenerationalDictionary<(int, int), Weight>());
                getLogValueState = result;
            }

            result.Item1.Clear();
            result.Item2.Clear();
            result.Item3.Clear();

            return result;
        }

        // Note: second array is returned without being cleared. This is intentional - it
        // would be fully overwritten by caller.
        internal static (int[], int[]) LeaseBuildReversedGraphState(int statesCount, int edgesCount)
        {
            var result = buildReversedGraphState;

            var currentStatesCount = result.Item1?.Length ?? -1;
            if (currentStatesCount < statesCount)
            {
                result = (new int[Math.Max(statesCount, currentStatesCount * 2)], result.Item2);
                buildReversedGraphState = result;
            }

            var currentEdgesCount = result.Item2?.Length ?? -1;
            if (currentEdgesCount < edgesCount)
            {
                result = (result.Item1, new int[Math.Max(edgesCount, currentEdgesCount * 2)]);
                buildReversedGraphState = result;
            }

            Array.Clear(result.Item1, 0, statesCount);

            return result;
        }

        internal static (Stack<int>, bool[]) LeaseRemoveDeadStatesState(int statesCount)
        {
            var result = removeDeadStatesState;

            if (result.Item1 == null)
            {
                result = (new Stack<int>(), result.Item2);
                removeDeadStatesState = result;
            }

            var resultLength = result.Item2?.Length ?? -1;
            if (resultLength < statesCount)
            {
                result = (result.Item1, new bool[Math.Max(statesCount, resultLength * 2)]);
                removeDeadStatesState = result;
            }

            result.Item1.Clear();
            Array.Clear(result.Item2, 0, statesCount);

            return result;
        }

        // Note: array is returned without being cleared. This is intentional - it
        // would be fully overwritten by caller.
        internal static int[] LeaseRemoveStatesState(int statesCount)
        {
            var result = removeStatesState;

            var resultLength = result?.Length ?? -1;
            if (resultLength < statesCount)
            {
                result = new int[Math.Max(statesCount, resultLength * 2)];
                removeStatesState = result;
            }

            return result;
        }
    }
}
