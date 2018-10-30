// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Reflection;
using Microsoft.CodeAnalysis;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler
{
    public class TypeSymbolConverter
    {
        private static CodeBuilder Builder = CodeBuilder.Instance;

        internal static IType ConvertTypeReference(ITypeSymbol typeSymbol, IAssemblySymbol compilationAssembly, Assembly originalAssembly)
        {
            // currently type ref is defined using .Net type
            var t = ConvertTypeSymbolToType(typeSymbol, compilationAssembly, originalAssembly);
            return Builder.TypeRef(t);
        }

        internal static Type ConvertTypeSymbolToType(ITypeSymbol typeSymbol, IAssemblySymbol compilationAssembly, Assembly originalAssembly)
        {
            if (typeSymbol == null) throw new ArgumentNullException(nameof(typeSymbol));

            if (typeSymbol is IDynamicTypeSymbol)
            {
                // Hack to make TestTypeDeclarationGeneration pass
                // LanguageWriteTests.CompileTypeDeclaration has a dynamic return type which can't be represented in our current code model
                return typeof (object);
            }

            if (typeSymbol is IArrayTypeSymbol)
            {
                var iat = typeSymbol as IArrayTypeSymbol;
                var arrayType = ConvertTypeSymbolToType(iat.ElementType, compilationAssembly, originalAssembly);
                return iat.Rank == 1 ? arrayType.MakeArrayType() : arrayType.MakeArrayType(iat.Rank);
            }
            
            if (typeSymbol is INamedTypeSymbol)
            {
                var namedTypeSymbol = typeSymbol as INamedTypeSymbol;

                // Metadata name only gives the last component of the namespace
                var namespaceString = namedTypeSymbol.ContainingNamespace.ToDisplayString();

                var fullTypeName = namespaceString + "." + namedTypeSymbol.MetadataName;

                // Check for nested type
                if (namedTypeSymbol.ContainingType != null)
                {
                    var ownerType = ConvertTypeReference(namedTypeSymbol.ContainingType, compilationAssembly, originalAssembly).DotNetType;
                    if (ownerType != null)
                    {
                        try
                        {
                            BindingFlags bf = BindingFlags.Public |
                                BindingFlags.NonPublic |
                                BindingFlags.Instance |
                                BindingFlags.Static |
                                BindingFlags.FlattenHierarchy;
                            return ownerType.GetNestedType(fullTypeName, bf);
                        }
                        catch { }
                    }
                    return Type.GetType(fullTypeName, true);
                }
                // Check for instance of a generic type
                else if (namedTypeSymbol.IsGenericType && !namedTypeSymbol.IsUnboundGenericType)
                {
                    var typeArguments = namedTypeSymbol.TypeArguments.Select(arg => ConvertTypeSymbolToType(arg, compilationAssembly, originalAssembly)).ToArray();
                    var unboundGenericType = ConvertTypeSymbolToType(namedTypeSymbol.ConstructUnboundGenericType(), compilationAssembly, originalAssembly);
                    return unboundGenericType.MakeGenericType(typeArguments);
                }
                else
                {
                    // Types in the compilation will have a temporary assembly name so that can't be used. Instead, look them up in the original assembly
                    if (namedTypeSymbol.ContainingAssembly == compilationAssembly)
                    {
                        return originalAssembly.GetType(fullTypeName);
                    }
                    return Type.GetType(fullTypeName + ", " + namedTypeSymbol.ContainingAssembly.ToDisplayString(), true);
                }
            }

            throw new Exception("Could not translate type");
        }
    }
}
