// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler
{
    internal interface ILanguageWriter
    {
        string ExpressionSource(IExpression ie);
        string StatementSource(IStatement ist);
        string ParameterDeclarationSource(IParameterDeclaration ipd);
        string ParameterDeclarationCollectionSource(IList<IParameterDeclaration> ipdc);
        string TypeDeclarationSource(ITypeDeclaration itd);
        string TypeDeclarationCollectionSource(List<ITypeDeclaration> intdc);
        string MethodDeclarationSource(IMethodDeclaration imd);
        string MethodDeclarationCollectionSource(List<IMethodDeclaration> imdc);
        string FieldDeclarationSource(IFieldDeclaration ifd);
        string FieldDeclarationCollectionSource(List<IFieldDeclaration> ifdc);
        string PropertyDeclarationSource(IPropertyDeclaration ipd);
        string PropertyDeclarationCollectionSource(List<IPropertyDeclaration> ipdc);
        string EventDeclarationSource(IEventDeclaration ied);
        string EventDeclarationCollectionSource(List<IEventDeclaration> iedc);
        string TypeSource(IType it);
        string VariableDeclarationSource(IVariableDeclaration ivd);

        //void AppendExpression(StringBuilder sb, IExpression ie);
        //SourceNode AttachStatement(IStatement ist);
        //SourceNode AttachTypeDeclaration(ITypeDeclaration itd);
        //SourceNode AttachTypeDeclarationCollection(ITypeDeclarationCollection intdc);
        //SourceNode AttachMethodDeclaration(IMethodDeclaration imd);
        //SourceNode AttachMethodDeclarationCollection(IMethodDeclarationCollection imdc);
        //SourceNode AttachFieldDeclaration(IFieldDeclaration ifd);
        //SourceNode AttachFieldDeclarationCollection(IFieldDeclarationCollection ifdc);
        //SourceNode AttachPropertyDeclaration(IPropertyDeclaration ipd);
        //SourceNode AttachPropertyDeclarationCollection(IPropertyDeclarationCollection ipdc);

        void Initialise();
        SourceNode GenerateSource(ITypeDeclaration itd);
        ICollection<Assembly> ReferencedAssemblies { get; }
    }
}