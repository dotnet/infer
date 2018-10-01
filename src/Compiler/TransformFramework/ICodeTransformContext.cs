// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    public interface ICodeTransformContext
    {
        List<ITypeDeclaration> TypesToTransform { get; }
        AttributeRegistry<object, ICompilerAttribute> InputAttributes { get; set; }
        AttributeRegistry<object, ICompilerAttribute> OutputAttributes { get; }
        int Depth { get; }

        void OpenType(ITypeDeclaration itd);
        void CloseType(ITypeDeclaration itd);
        void OpenMember(IMemberDeclaration imd);
        void CloseMember(IMemberDeclaration imd);
        void OpenStatement(IStatement istmt);
        void CloseStatement(IStatement istmt);
        void OpenExpression(IExpression iexpr);
        void CloseExpression(IExpression iexpr);

        void Warning(string msg);
        void Error(string msg);
        TransformResults Results { get; }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}