// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Type visibility
    /// </summary>
    public enum TypeVisibility
    {
        /// <summary>
        /// Private
        /// </summary>
        Private = 0,

        /// <summary>
        /// Public
        /// </summary>
        Public = 1,

        /// <summary>
        /// 
        /// </summary>
        NestedPublic = 2,

        /// <summary>
        /// 
        /// </summary>
        NestedPrivate = 3,

        /// <summary>
        /// 
        /// </summary>
        NestedFamily = 4,

        /// <summary>
        /// 
        /// </summary>
        NestedAssembly = 5,

        /// <summary>
        /// 
        /// </summary>
        NestedFamilyAndAssembly = 6,

        /// <summary>
        /// 
        /// </summary>
        NestedFamilyOrAssembly = 7,
    }

    /// <summary>
    /// Type declaration
    /// </summary>
    public interface ITypeDeclaration : ITypeReference, ICustomAttributeProvider, IDocumentationProvider
    {
        /// <summary>
        /// Whether the type is abstract - i.e. cannot be instantiated
        /// </summary>
        bool Abstract { get; set; }

        /// <summary>
        /// Reference to base type
        /// </summary>
        ITypeReference BaseType { get; set; }

        /// <summary>
        /// Collection of fields
        /// </summary>
        List<IFieldDeclaration> Fields { get; }

        /// <summary>
        /// Whether the type is an interface
        /// </summary>
        bool Interface { get; set; }

        /// <summary>
        /// Interfaces this type is derived from
        /// </summary>
        List<ITypeReference> Interfaces { get; }

        /// <summary>
        /// Collection of methods
        /// </summary>
        List<IMethodDeclaration> Methods { get; }

        /// <summary>
        /// Collection of nested types
        /// </summary>
        List<ITypeDeclaration> NestedTypes { get; }

        /// <summary>
        /// Collection of properties
        /// </summary>
        List<IPropertyDeclaration> Properties { get; }

        /// <summary>
        /// Collection of events
        /// </summary>
        List<IEventDeclaration> Events { get; }

        /// <summary>
        /// True if the type is sealed (cannot be derived from)
        /// </summary>
        bool Sealed { get; set; }

        /// <summary>
        /// True if the declaration is partial
        /// </summary>
        bool Partial { get; set; }

        /// <summary>
        /// Visibility of the type
        /// </summary>
        TypeVisibility Visibility { get; set; }
    }
}