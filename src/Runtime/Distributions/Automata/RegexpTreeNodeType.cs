// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    /// <summary>
    /// Represents the type of a regular expression tree node.
    /// The value of an enumeration element corresponds to the priority of the operation it describes.
    /// </summary>
    public enum RegexpTreeNodeType
    {
        /// <summary>
        /// Represents the union operation.
        /// </summary>
        Union = 1,

        /// <summary>
        /// Represents the concatenation operation.
        /// </summary>
        Concat = 2,

        /// <summary>
        /// Represents the Kleene star operation.
        /// </summary>
        Star = 3,

        /// <summary>
        /// Represents an empty string.
        /// </summary>
        Empty = 4,

        /// <summary>
        /// Represents a set of elements.
        /// </summary>
        ElementSet = 5,

        /// <summary>
        /// Represents the empty language, i.e. a language that contains no strings.
        /// </summary>
        Nothing = 6
    }
}