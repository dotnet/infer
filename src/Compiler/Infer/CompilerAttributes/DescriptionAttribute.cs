// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler;

namespace Microsoft.ML.Probabilistic.Compiler.Attributes
{
    /// <summary>
    /// Specifies a description for whatever the attribute is attached to.
    /// </summary>
    internal class DescriptionAttribute : ICompilerAttribute
    {
        /// <summary>
        /// The description for the attribute.
        /// </summary>
        public string Description { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="DescriptionAttribute"/> class.
        /// </summary>
        /// <param name="description">The description for the attribute</param>
        public DescriptionAttribute(string description)
        {
            this.Description = description;
        }
    }
}