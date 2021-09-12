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
        internal static (GenerationalDictionary<(int, int), int>, Stack<(int, int, int)>) ProductState;

        [ThreadStatic]
        private static (Stack<int>, GenerationalDictionary<int, TarjanStateInfo>, Stack<(TarjanStateInfo, int, int)>) FindStronglyConnectedComponentsState;
    }
}
