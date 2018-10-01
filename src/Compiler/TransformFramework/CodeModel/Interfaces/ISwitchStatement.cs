// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Switch statement
    /// </summary>
    public interface ISwitchStatement : IStatement
    {
        /// <summary>
        /// The cases in the switch statement
        /// </summary>
        IList<ISwitchCase> Cases { get; }

        /// <summary>
        /// The expression
        /// </summary>
        IExpression Expression { get; set; }
    }
}