// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    /// <summary>
    /// Represents node state used in Tarjan's strongly connected components algorithm.
    /// </summary>
    internal struct TarjanStateInfo
    {
        public TarjanStateInfo(int generation, int index)
        {
            this.Generation = generation;
            this.TraversalIndex = index;
            this.Lowlink = index;
            this.InStack = true;
        }

        /// <summary>
        /// Generation of the data.
        /// </summary>
        /// <remarks>
        /// An array of `TarjanStateInfo` is cached in
        /// <see cref="PreallocatedAutomataObjects.findStronglyConnectedComponentsState"/>.
        /// To avoid (costly) clear operations on this array, a generational scheme is used.
        /// Each time array must be logically cleared, it's generation is bumped.
        /// At the places where states array used, entries with non-matching generation
        /// must be treated as empty.
        /// </remarks>
        public int Generation { get; set;}

        /// <summary>
        /// Gets or sets the traversal index of the state. Zero value indicates that
        /// this state has not been visited yet.
        /// </summary>
        public int TraversalIndex { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the state is currently in stack.
        /// </summary>
        public bool InStack { get; set; }

        /// <summary>
        /// Gets or sets the lowlink of the state.
        /// </summary>
        public int Lowlink { get; set; }
    }
}