// <copyright file="TarjanStateInfo.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    internal struct TarjanStateInfo
    {
        public TarjanStateInfo(int index)
        {
            this.TraversalIndex = index;
            this.Lowlink = index;
            this.InStack = true;
        }

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