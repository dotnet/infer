// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Core.Collections;

    // TODO: document
    internal static class PreallocatedAutomataObjects
    {
        [ThreadStatic]
        private static (GenerationalDictionary<(int, int), int>, Stack<(int, int, int)>) productState;

        [ThreadStatic]
        private static (Stack<int>, GenerationalDictionary<int, TarjanStateInfo>, Stack<(TarjanStateInfo, int, int)>) findStronglyConnectedComponentsState;

        [ThreadStatic]
        private static GenerationalDictionary<int, CondensationStateInfo> computeCondensationState;

        [ThreadStatic]
        private static (int[], int[]) buildReversedGraphState;

        [ThreadStatic]
        private static (Stack<int>, bool[]) removeDeadStatesState;

        [ThreadStatic]
        private static int[] removeStatesState;

        internal static (GenerationalDictionary<(int, int), int>, Stack<(int, int, int)>) ProductState
        {
            get
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
        }

        internal static (Stack<int>, GenerationalDictionary<int, TarjanStateInfo>, Stack<(TarjanStateInfo, int, int)>) FindStronglyConnectedComponentsState
        {
            get
            {
                var result = findStronglyConnectedComponentsState;
                if (result.Item1 == null)
                {
                    result = ValueTuple.Create(
                        new Stack<int>(),
                        new GenerationalDictionary<int, TarjanStateInfo>(),
                        new Stack<(TarjanStateInfo, int, int)>());
                    findStronglyConnectedComponentsState = result;
                }

                result.Item1.Clear();
                result.Item2.Clear();
                result.Item3.Clear();

                return result;
            }
        }

        internal static GenerationalDictionary<int, CondensationStateInfo> ComputeCondensationState
        {
            set => computeCondensationState = value;
            get
            {
                var result = computeCondensationState;
                if (result == null)
                {
                    result = new GenerationalDictionary<int, CondensationStateInfo>();
                    computeCondensationState = result;
                }

                result.Clear();

                // In case some other Condensation object is created at the same time, it
                // should not reuse the same state. In practice (in single therad) 2 Condensation
                // objects never coexist. But there is no guarantee that it won't happen at some point.
                // So each time a state is used, let's reset cache to null. By the time condensation
                // is computed again, this value should be returned into cache
                computeCondensationState = null;
                return result;
            }
        }

        internal static (int[], int[]) GetBuildReversedGraphState(int statesCount, int edgesCount)
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

        internal static (Stack<int>, bool[]) GetRemoveDeadStatesState(int statesCount)
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

        internal static int[] GetRemoveStatesState(int statesCount)
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
