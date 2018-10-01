// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Object creation expression
    /// </summary>
    public interface IObjectCreateExpression : IExpression
    {
        /// <summary>
        /// Arguments to the constructor
        /// </summary>
        IList<IExpression> Arguments { get; }

        /// <summary>
        /// Constructor
        /// </summary>
        IMethodReference Constructor { get; set; }

        /// <summary>
        /// Initializer expression
        /// </summary>
        IBlockExpression Initializer { get; set; }

        /// <summary>
        /// Type of object
        /// </summary>
        IType Type { get; set; }
    }
}