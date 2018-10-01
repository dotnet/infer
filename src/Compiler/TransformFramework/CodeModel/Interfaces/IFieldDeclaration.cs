// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel
{
    /// <summary>
    /// Field visibility
    /// </summary>
    public enum FieldVisibility
    {
        /// <summary>
        /// 
        /// </summary>
        PrivateScope = 0,

        /// <summary>
        /// Private scope
        /// </summary>
        Private = 1,

        /// <summary>
        /// 
        /// </summary>
        FamilyAndAssembly = 2,

        /// <summary>
        /// Assembly scope
        /// </summary>
        Assembly = 3,

        /// <summary>
        /// 
        /// </summary>
        Family = 4,

        /// <summary>
        /// 
        /// </summary>
        FamilyOrAssembly = 5,

        /// <summary>
        /// Public scope
        /// </summary>
        Public = 6,
    }

    /// <summary>
    /// Field declaration
    /// </summary>
    public interface IFieldDeclaration : IFieldReference, IMemberDeclaration
    {
        /// <summary>
        /// Initializer expression
        /// </summary>
        IExpression Initializer { get; set; }

        /// <summary>
        /// Literal flag
        /// </summary>
        bool Literal { get; set; }

        /// <summary>
        /// Read only flag
        /// </summary>
        bool ReadOnly { get; set; }

        /// <summary>
        /// Static flag
        /// </summary>
        bool Static { get; set; }

        /// <summary>
        /// Field visibility
        /// </summary>
        FieldVisibility Visibility { get; set; }
    }
}