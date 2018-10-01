// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Custom attribute
    /// </summary>
    public interface ICustomAttribute
    {
        /// <summary>
        /// Attribute arguments
        /// </summary>
        IList<IExpression> Arguments { get; }

        /// <summary>
        /// Constructor
        /// </summary>
        IMethodReference Constructor { get; set; }
    }
}