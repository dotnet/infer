// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Cuts if statements containing multiple child statements, so that each child statement is in its own if statement.
    /// </summary>
    internal class IfCuttingTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "IfCuttingTransform"; }
        }

        private IConditionStatement parent;

        /// <summary>
        /// Converts an if statement by deleting the outer statement and wrapping the inner statements.
        /// </summary>
        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            bool isStochastic = IsStochasticVariableReference(ics.Condition);
            if (!isStochastic) return base.ConvertCondition(ics);
            if ((ics.Else != null) && (ics.Else.Statements.Count > 0))
            {
                Error("Unexpected else clause");
                return ics;
            }
            IConditionStatement oldParent = parent;
            parent = ics;
            ics = (IConditionStatement) base.ConvertCondition(ics);
            parent = oldParent;
            foreach (IStatement ist in ics.Then.Statements)
                context.AddStatementBeforeCurrent(ist);
            return null;
        }

        protected override IStatement ConvertExpressionStatement(IExpressionStatement ies)
        {
            if (parent == null) return ies;
            bool keepIfStatement = false;
            // Only keep the surrounding if statement when a factor or constraint is being added.
            IExpression expr = ies.Expression;
            if (expr is IMethodInvokeExpression)
            {
                keepIfStatement = true;
                if (CodeRecognizer.IsInfer(expr)) keepIfStatement = false;
            }
            else if (expr is IAssignExpression)
            {
                keepIfStatement = false;
                IAssignExpression iae = (IAssignExpression) expr;
                IMethodInvokeExpression imie = iae.Expression as IMethodInvokeExpression;
                if (imie != null)
                {
                    keepIfStatement = true;
                    if (imie.Arguments.Count > 0)
                    {
                        // Statements that copy evidence variables should not send evidence messages.
                        IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
                        IVariableDeclaration ivdArg = Recognizer.GetVariableDeclaration(imie.Arguments[0]);
                        if (ivd != null && context.InputAttributes.Has<DoNotSendEvidence>(ivd) &&
                            ivdArg != null && context.InputAttributes.Has<DoNotSendEvidence>(ivdArg)) keepIfStatement = false;
                    }
                }
                else
                {
                    expr = iae.Target;
                }
            }
            if (expr is IVariableDeclarationExpression)
            {
                IVariableDeclarationExpression ivde = (IVariableDeclarationExpression) expr;
                IVariableDeclaration ivd = ivde.Variable;
                keepIfStatement = CodeRecognizer.IsStochastic(context, ivd) && !context.InputAttributes.Has<DoNotSendEvidence>(ivd);
            }
            if (!keepIfStatement) return ies;
            IConditionStatement cs = Builder.CondStmt(parent.Condition, Builder.BlockStmt());
            cs.Then.Statements.Add(ies);
            return cs;
        }

        // copied from ReplicationTransform
        /// <summary>
        /// Returns true if the supplied expression is stochastic.  
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        protected bool IsStochasticVariableReference(IExpression expr)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
            return (ivd != null) && CodeRecognizer.IsStochastic(context, ivd);
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}