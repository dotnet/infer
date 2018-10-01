// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Generic argument
    /// </summary>
    public interface IGenericArgument : IType
    {
        /// <summary>
        /// The provider of types for the generic argument
        /// </summary>
        IGenericArgumentProvider Owner { get; set; }

        /// <summary>
        /// The position of the generic argument
        /// </summary>
        int Position { get; set; }

        /// <summary>
        /// Resolve the generic argument to a type
        /// </summary>
        /// <returns></returns>
        IType Resolve();
    }
}