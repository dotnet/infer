// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Condition case
    /// </summary>
    public interface IConditionCase : ISwitchCase
    {
        /// <summary>
        /// Condition expression
        /// </summary>
        IExpression Condition { get; set; }
    }
}