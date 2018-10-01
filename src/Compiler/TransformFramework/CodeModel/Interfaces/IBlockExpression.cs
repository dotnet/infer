// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Block expression - i.e. an expression which is a collection of expressions
    /// </summary>
    public interface IBlockExpression : IExpression
    {
        /// <summary>
        /// The collection of expressions in the block
        /// </summary>
        IList<IExpression> Expressions { get; }
    }
}