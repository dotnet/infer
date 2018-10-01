// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Method reference
    /// </summary>
    public interface IMethodReference : IMemberReference, IMethodSignature, IGenericArgumentProvider
    {
        /// <summary>
        /// Generic method reference
        /// </summary>
        IMethodReference GenericMethod { get; set; }

        /// <summary>
        /// Resolve to a method declaration
        /// </summary>
        /// <returns></returns>
        IMethodDeclaration Resolve();
    }
}