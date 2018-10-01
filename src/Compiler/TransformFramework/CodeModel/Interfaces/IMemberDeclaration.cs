// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// A member declaration
    /// </summary>
    public interface IMemberDeclaration : IMemberReference, ICustomAttributeProvider, IDocumentationProvider
    {
    }
}