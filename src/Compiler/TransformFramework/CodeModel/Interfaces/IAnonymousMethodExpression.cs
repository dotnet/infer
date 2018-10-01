// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Anonymous method expression
    /// </summary>
    public interface IAnonymousMethodExpression : IExpression
    {
        /// <summary>
        /// Body of statements
        /// </summary>
        IBlockStatement Body { get; set; }

        /// <summary>
        /// Delegate type
        /// </summary>
        IType DelegateType { get; set; }

        /// <summary>
        /// Parameter declarations for delegate
        /// </summary>
        IList<IParameterDeclaration> Parameters { get; }
    }
}