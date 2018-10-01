// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Replace references to properties and fields with local variables
    /// </summary>
    internal class ExternalVariablesTransform : ShallowCopyTransform
    {
        public override string Name { get { return "ExternalVariablesTransform"; } }

        private Dictionary<IMemberDeclaration, IVariableDeclaration> conversions = new Dictionary<IMemberDeclaration, IVariableDeclaration>();
        private List<IParameterDeclaration> parameters = new List<IParameterDeclaration>();

        protected override IPropertyDeclaration ConvertProperty(ITypeDeclaration td, IPropertyDeclaration ipd, bool convertGetterAndSetter = true)
        {
            return null;
        }

        protected override IFieldDeclaration ConvertField(ITypeDeclaration td, IFieldDeclaration ifd)
        {
            return null;
        }

        protected override IMethodDeclaration ConvertMethod(IMethodDeclaration imd)
        {
            parameters = imd.Parameters.Cast<IParameterDeclaration>().ToList();
            return base.ConvertMethod(imd);
        }

        protected override IExpression ConvertPropertyRefExpr(IPropertyReferenceExpression ipre)
        {
            if (ipre.Target is IThisReferenceExpression)
            {
                return ConvertExternalVariableExpr(ipre.Property.Resolve(), ipre.Property.Name, ipre.Property.PropertyType);
            }
            return base.ConvertPropertyRefExpr(ipre);
        }

        protected override IExpression ConvertFieldRefExpr(IFieldReferenceExpression ifre)
        {
            if (ifre.Target is IThisReferenceExpression)
            {
                return ConvertExternalVariableExpr(ifre.Field.Resolve(), ifre.Field.Name, ifre.Field.FieldType);
            }
            return base.ConvertFieldRefExpr(ifre);
        }

        private IExpression ConvertExternalVariableExpr(IMemberDeclaration externalDecl, string name, IType type)
        {
            foreach (var paramDecl in parameters)
            {
                if (paramDecl.Name == name && paramDecl.ParameterType.Equals(type))
                {
                    return Builder.ParamRef(paramDecl);
                }
            }

            IVariableDeclaration varDecl;
            if (conversions.TryGetValue(externalDecl, out varDecl))
            {
                return Builder.VarRefExpr(Builder.VarRef(varDecl));
            }

            varDecl = Builder.VarDecl(name, type);
            conversions[externalDecl] = varDecl;
            context.AddStatementBeforeCurrent(Builder.ExprStatement(Builder.VarDeclExpr(varDecl)));
            return Builder.VarRefExpr(Builder.VarRef(varDecl));
        }

    }
}
