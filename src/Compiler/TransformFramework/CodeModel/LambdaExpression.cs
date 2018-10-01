// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    internal class XLambdaExpression : XExpression, ILambdaExpression
    {
        public IExpression Body { get; set; }
        private List<IVariableDeclaration> parameters = new List<IVariableDeclaration>();

        public IList<IVariableDeclaration> Parameters
        {
            get { return parameters; }
        }

        public override Type GetExpressionType()
        {
            var types = new List<Type>();
            foreach (var p in parameters) types.Add(CodeBuilder.Instance.ToType(p.VariableType));
            types.Add(Body.GetExpressionType());
            var funcType = System.Linq.Expressions.Expression.GetFuncType(types.ToArray());
            // temp
            var exprType = typeof (System.Linq.Expressions.Expression<>).MakeGenericType(funcType);
            return exprType;
        }
    }
}