// <copyright file="TarjanStateInfo.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    // TODO: Make it a value type
    internal class TarjanStateInfo
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TarjanStateInfo"/> class.
        /// </summary>
        /// <param name="traversalIndex">The current traversal index.</param>
        public TarjanStateInfo(int traversalIndex)
        {
            this.TraversalIndex = traversalIndex;
            this.Lowlink = traversalIndex;
        }

        /// <summary>
        /// Gets or sets a value indicating whether the state is currently in stack.
        /// </summary>
        public bool InStack { get; set; }

        /// <summary>
        /// Gets the traversal index of the state.
        /// </summary>
        public int TraversalIndex { get; }

        /// <summary>
        /// Gets or sets the lowlink of the state.
        /// </summary>
        public int Lowlink { get; set; }
    }
}