// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler
{
    /// <summary>
    /// Code transform interface
    /// </summary>
    public interface ICodeTransform
    {
        /// <summary>
        /// Takes a type declaration and transforms it
        /// </summary>
        /// <param name="itd"></param>
        /// <returns></returns>
        ITypeDeclaration Transform(ITypeDeclaration itd);

        /// <summary>
        /// Context
        /// </summary>
        ICodeTransformContext Context { get; }

        /// <summary>
        /// Name of the transform
        /// </summary>
        string Name { get; }
    }
}