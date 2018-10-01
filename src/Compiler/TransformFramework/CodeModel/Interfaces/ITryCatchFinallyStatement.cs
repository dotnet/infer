// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Try/catch/finally statement
    /// </summary>
    public interface ITryCatchFinallyStatement : IStatement
    {
        /// <summary>
        /// The catch clauses
        /// </summary>
        IList<ICatchClause> CatchClauses { get; }

        /// <summary>
        /// The unconditional catch clause body
        /// </summary>
        IBlockStatement Fault { get; set; }

        /// <summary>
        /// The finally body
        /// </summary>
        IBlockStatement Finally { get; set; }

        /// <summary>
        /// The try body
        /// </summary>
        IBlockStatement Try { get; set; }
    }
}