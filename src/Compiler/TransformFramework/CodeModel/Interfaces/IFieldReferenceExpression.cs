// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Field reference expression
    /// </summary>
    public interface IFieldReferenceExpression : IExpression
    {
        /// <summary>
        /// The field reference
        /// </summary>
        IFieldReference Field { get; set; }

        /// <summary>
        /// The expression for the type instance
        /// </summary>
        IExpression Target { get; set; }
    }
}