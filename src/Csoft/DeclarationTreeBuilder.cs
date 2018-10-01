// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete;

namespace Microsoft.ML.Probabilistic.Compiler
{
    public class DeclarationTreeBuilder
    {
        private static CodeBuilder Builder = CodeBuilder.Instance;

        private Dictionary<string, ITypeDeclaration> declarations;
        private SemanticModel model;
        private Assembly originalAssembly;
        private Dictionary<IMethodDeclaration, MethodDeclarationSyntax> methodBodies = new Dictionary<IMethodDeclaration, MethodDeclarationSyntax>();

        public DeclarationTreeBuilder(SemanticModel model, Assembly originalAssembly, Dictionary<string, ITypeDeclaration> declarations)
        {
            this.model = model;
            this.originalAssembly = originalAssembly;
            this.declarations = declarations;
        }

        public Dictionary<IMethodDeclaration, MethodDeclarationSyntax> Build()
        {
            var compilationUnit = model.SyntaxTree.GetRoot() as CompilationUnitSyntax;
            if (compilationUnit == null)
            {
                throw new NotSupportedException("Unsupported root type");
            }

            foreach (var member in compilationUnit.Members)
            {
                ConvertMember(null, member);
            }

            return methodBodies;
        }

        private void ConvertMember(ITypeDeclaration declaringType, MemberDeclarationSyntax member)
        {
            if (member is NamespaceDeclarationSyntax)
            {
                ConvertNamespaceDeclaration(declaringType, (NamespaceDeclarationSyntax)member);
                return;
            }
            if (member is ClassDeclarationSyntax)
            {
                ConvertClassDeclaration((ClassDeclarationSyntax)member);
                return;
            }
            if (member is DelegateDeclarationSyntax)
            {
                // Code model doesn't currently support delegate declarations
                return;
            }
            throw new NotSupportedException();
        }

        private void ConvertNamespaceDeclaration(ITypeDeclaration declaringType, NamespaceDeclarationSyntax namespaceDeclaration)
        {
            foreach (var member in namespaceDeclaration.Members)
            {
                ConvertMember(declaringType, member);
            }
        }

        private void ConvertClassDeclaration(ClassDeclarationSyntax classDeclaration)
        {
            var classSymbol = model.GetDeclaredSymbol(classDeclaration);
            var declaringType = Builder.TypeDecl();
            declaringType.Name = classSymbol.Name;
            declaringType.Namespace = classSymbol.ContainingNamespace.ToDisplayString();
            declaringType.DotNetType = ConvertTypeSymbolToType(classSymbol);

            foreach (var member in classDeclaration.Members)
            {
                switch (member.Kind())
                {
                    case SyntaxKind.ClassDeclaration:
                        ConvertClassDeclaration((ClassDeclarationSyntax)member);
                        break;
                
                    case SyntaxKind.MethodDeclaration:
                        var methodDeclSyntax = (MethodDeclarationSyntax) member;
                        var methodDecl = ConvertMethodDecl(declaringType, methodDeclSyntax);
                        declaringType.Methods.Add(methodDecl);
                        // record that the method body needs converting
                        methodBodies[methodDecl] = methodDeclSyntax;
                        break;

                    case SyntaxKind.DelegateDeclaration:
                        // TODO Code model doesn't currently support delegate declarations
                        break;

                    case SyntaxKind.FieldDeclaration:
                        // TODO convert fields
                        break;

                    case SyntaxKind.ConstructorDeclaration:
                        // TODO convert constructors
                        break;

                    case SyntaxKind.PropertyDeclaration:
                        var propertyDecl = ConvertPropertyDecl(declaringType, (PropertyDeclarationSyntax)member);
                        declaringType.Properties.Add(propertyDecl);
                        break;

                    default:
                        throw new NotSupportedException();
                }
            }

            // TODO should we be using metadata name format?
            declarations[declaringType.Namespace + "." + declaringType.Name] = declaringType;
        }

        private IType ConvertTypeReference(ITypeSymbol typeSymbol)
        {
            return TypeSymbolConverter.ConvertTypeReference(typeSymbol, model.Compilation.Assembly, originalAssembly);
        }

        private Type ConvertTypeSymbolToType(ITypeSymbol typeSymbol)
        {
            return TypeSymbolConverter.ConvertTypeSymbolToType(typeSymbol, model.Compilation.Assembly, originalAssembly);
        }

        private IMethodDeclaration ConvertMethodDecl(ITypeDeclaration typeDeclaration, MethodDeclarationSyntax methodDeclarationSyntax)
        {
            var methodSymbol = model.GetDeclaredSymbol(methodDeclarationSyntax);

            var methodDecl = Builder.MethodDecl();
            methodDecl.Name = methodSymbol.Name;
            methodDecl.DeclaringType = typeDeclaration;
            methodDecl.Static = methodSymbol.IsStatic;
            methodDecl.Visibility = ConvertVisibility(methodSymbol.DeclaredAccessibility);
            methodDecl.Virtual = methodSymbol.IsVirtual;
            methodDecl.Abstract = methodSymbol.IsAbstract;
            methodDecl.Final = methodSymbol.IsSealed;

            var methodReturnType = ConvertTypeReference(methodSymbol.ReturnType);
            methodDecl.ReturnType = Builder.MethodReturnType(methodReturnType);
            
            ConvertAttributes(methodDeclarationSyntax.AttributeLists, methodDecl.Attributes);

            foreach (var paramSym in methodSymbol.Parameters)
            {
                var paramDecl = Builder.Param(paramSym.Name, ConvertTypeReference(paramSym.Type));
                methodDecl.Parameters.Add(paramDecl);
            }

            methodDecl.MethodInfo = Builder.ToMethodThrows(methodDecl);

            return methodDecl;
        }

        private IPropertyDeclaration ConvertPropertyDecl(ITypeDeclaration typeDeclaration, PropertyDeclarationSyntax propertyDeclarationSyntax)
        {
            var propertySymbol = model.GetDeclaredSymbol(propertyDeclarationSyntax);

            var propertyDecl = Builder.PropDecl();
            propertyDecl.Name = propertySymbol.Name;
            propertyDecl.DeclaringType = typeDeclaration;
            propertyDecl.PropertyType = ConvertTypeReference(propertySymbol.Type);
            ConvertAttributes(propertyDeclarationSyntax.AttributeLists, propertyDecl.Attributes);

            // TODO fix code model - propertyDecl.Static = propertySymbol.IsStatic;
            // TODO fix code model - propertyDecl.Visibility = ConvertVisibility(propertySymbol.DeclaredAccessibility);

            // TODO convert bodies of getters and setters

            return propertyDecl;
        }

        private void ConvertAttributes(IEnumerable<AttributeListSyntax> attributeLists, List<ICustomAttribute> outputList)
        {
            foreach (var attributeList in attributeLists)
            {
                foreach (var attribute in attributeList.Attributes)
                {
                    var customAttribute = new XCustomAttribute();
                    // TODO convert attribute
                }
            }
        }
    
        private static MethodVisibility ConvertVisibility(Accessibility accessibility)
        {
            switch (accessibility)
            {
                case Accessibility.Public:
                    return MethodVisibility.Public;
                case Accessibility.Private:
                    return MethodVisibility.Private;
                case Accessibility.Protected:
                    return MethodVisibility.Family;
                case Accessibility.Internal:
                    return MethodVisibility.Assembly;
                case Accessibility.ProtectedAndInternal:
                    return MethodVisibility.FamilyAndAssembly;
                case Accessibility.ProtectedOrInternal:
                    return MethodVisibility.FamilyOrAssembly;
                default:
                    throw new NotSupportedException("Could not convert accessibility");
            }
        }
    }
}
