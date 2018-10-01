// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Array creation expression
    /// </summary>
    public interface IArrayCreateExpression : IExpression
    {
        /// <summary>
        /// Dimensions of array
        /// </summary>
        IList<IExpression> Dimensions { get; }

        /// <summary>
        /// Initializer expression
        /// </summary>
        IBlockExpression Initializer { get; set; }

        /// <summary>
        /// Element type
        /// </summary>
        IType Type { get; set; }
    }
}