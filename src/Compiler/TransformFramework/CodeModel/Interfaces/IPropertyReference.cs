// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Property reference
    /// </summary>
    public interface IPropertyReference : IMemberReference
    {
        /// <summary>
        /// Generic property reference
        /// </summary>
        IPropertyReference GenericProperty { get; set; }

        /// <summary>
        /// Parameter declarations
        /// </summary>
        IList<IParameterDeclaration> Parameters { get; }

        /// <summary>
        /// Property type
        /// </summary>
        IType PropertyType { get; set; }

        /// <summary>
        /// Resolve the reference to a property declaration
        /// </summary>
        /// <returns>Property declaration</returns>
        IPropertyDeclaration Resolve();
    }
}