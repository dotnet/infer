// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Support MSL from the API by mapping identical expressions to a new intermediate variable.
    /// If this isn't done, multiple intermediate get introduced and the gate transform complains that
    /// defintions don't match declaration as it can't tell that the variables are identical.
    /// </summary>
    internal class IntermediateVariableTransform : ShallowCopyTransform
    {
        public override string Name { get { return "IntermediateVariableTransform"; } }

        private int nextTempVarNum = 1;

        private Dictionary<IExpression, IVariableDeclaration> replacements = new Dictionary<IExpression, IVariableDeclaration>(); 

        // TODO should we also handle array accesses?

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            var copy = (IMethodInvokeExpression)base.ConvertMethodInvoke(imie);
            for (int i = 0; i < copy.Arguments.Count; i++)
            {
                var arg = copy.Arguments[i] as IMethodInvokeExpression;
                if (arg == null) continue;
                var n = (ITypeReferenceExpression) arg.Method.Target;

                // TODO we should be handling all methods but they cause compiler errors e.g.
                // if(expr) {
                //   Gaussian intermediate = Gaussian.Uniform();
                //   double rv = Factor.Random(intermediate);
                // }
                if (!Recognizer.IsTypeReferenceTo(n, typeof(Factor))) continue;

                copy.Arguments[i] = MakeIntermediateVariable(arg);
            }
            return copy;
        }

        protected override IExpression ConvertUnary(IUnaryExpression iue)
        {
            var copy = (IUnaryExpression)base.ConvertUnary(iue);
            if (NeedsIntermediate(copy.Expression))
            {
                // TODO this breaks things
                //copy.Expression = MakeIntermediateVariable(copy.Expression);
            }
            return copy;
        }

        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            var copy = (IConditionStatement)base.ConvertCondition(ics);
            if (NeedsIntermediate(copy.Condition) && !(copy.Condition is IUnaryExpression && ((IUnaryExpression)copy.Condition).Operator == UnaryOperator.BooleanNot))
            {
                // TODO this breaks things
                //copy.Condition = MakeIntermediateVariable(copy.Condition);
            }
            return copy;
        }

        private IExpression MakeIntermediateVariable(IExpression input)
        {
            IVariableDeclaration varDecl;
            if (!replacements.TryGetValue(input, out varDecl))
            {
                var varName = "intermediateVar" + nextTempVarNum++;
                var varType = input.GetExpressionType();
                varDecl = Builder.VarDecl(varName, varType);
                var varDeclExpr = Builder.VarDeclExpr(varDecl);
                var assignExpr = Builder.AssignExpr(varDeclExpr, input);
                context.AddStatementBeforeCurrent(Builder.ExprStatement(assignExpr));
                replacements[input] = varDecl;
            }
            
            return Builder.VarRefExpr(Builder.VarRef(varDecl));
        }

        private static bool NeedsIntermediate(IExpression expr)
        {
            return !(expr is IVariableReferenceExpression || expr is IFieldReferenceExpression || expr is IArgumentReferenceExpression);
        }
    }
}
