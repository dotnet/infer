// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Block statement - i.e. a statement which is a collection of statements
    /// </summary>
    public interface IBlockStatement : IStatement
    {
        /// <summary>
        /// Collection of statements in the block
        /// </summary>
        IList<IStatement> Statements { get; }
    }
}