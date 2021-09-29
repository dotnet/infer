// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    internal struct CondensationStateInfo
    {
        /// <summary>
        /// Gets the default state info with all the weights set to 0.
        /// </summary>
        public static CondensationStateInfo Default =>
            new CondensationStateInfo
            {
                WeightToEnd = Weight.Zero,
                WeightFromRoot = Weight.Zero,
                UpwardWeightFromRoot = Weight.Zero
            };

        /// <summary>
        /// Gets or sets the total weight of all paths starting at the state.
        /// Ending weights are taken into account.
        /// </summary>
        public Weight WeightToEnd { get; set; }

        /// <summary>
        /// Gets or sets the total weight of all paths from the root of the condensation
        /// to the state. Ending weights are not taken into account.
        /// </summary>
        public Weight WeightFromRoot { get; set; }

        /// <summary>
        /// Gets or sets the total weight of all paths from the root of the condensation
        /// to the state that don't go through any other states of the same strongly connected component.
        /// Ending weights are not taken into account.
        /// </summary>
        public Weight UpwardWeightFromRoot { get; set; }
    }
}