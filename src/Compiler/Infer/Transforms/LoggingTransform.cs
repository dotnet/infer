// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    internal class LoggingTransform : ShallowCopyTransform
    {
        IExpression loggingAction;

        public override string Name
        {
            get
            {
                return "LoggingTransform";
            }
        }

        public LoggingTransform(ModelCompiler compiler)
        {
            if (compiler.Logging)
                loggingAction = Builder.VarRefExpr(Builder.VarDecl("loggingAction", typeof(Action<string>)));
        }

        public override ITypeDeclaration ConvertType(ITypeDeclaration itd)
        {
            var attr = context.InputAttributes.Get<LoggingAttribute>(itd);
            if (attr != null)
                this.loggingAction = attr.loggingAction;
            return base.ConvertType(itd);
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            if (loggingAction != null)
            {
                if (iae.Expression is IArrayCreateExpression || iae.Expression is IObjectCreateExpression)
                {
                    string name;
                    if (iae.Target is IVariableReferenceExpression ivre)
                        name = ivre.Variable.Resolve().Name;
                    else if (iae.Target is IVariableDeclarationExpression ivde)
                        name = ivde.Variable.Name;
                    else if (iae.Target is IFieldReferenceExpression ifre)
                        name = ifre.Field.Name;
                    else if (iae.Target is IPropertyReferenceExpression ipre)
                        name = ipre.Property.Name;
                    else
                        name = null;
                    if (name != null && !name.Contains("_local"))
                    {
                        var message = Builder.LiteralExpr("Allocating " + name);
                        var invoke = Builder.DelegateInvokeExpr();
                        invoke.Target = loggingAction;
                        invoke.Arguments.Add(message);
                        context.AddStatementAfterCurrent(Builder.ExprStatement(invoke));
                    }
                }
            }
            return base.ConvertAssign(iae);
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            string methodName = null;
            if (loggingAction != null)
            {
                var imd = context.FindAncestor<IMethodDeclaration>();
                if (imd.Name.Contains("Changed"))
                    methodName = imd.Name;
            }
            OpenOutputBlock(outputs);
            bool entered = false, mustEnter = false;
            foreach (IStatement ist in inputs)
            {
                IStatement st = ConvertStatement(ist);
                if (st != null)
                {
                    if (methodName != null && !entered)
                    {
                        // insert the logging statement after the first conditional of the method, so that logging only happens when the method actually executes.
                        // to avoid this hack, logging statements would have to be inserted by IterativeProcessTransform.
                        bool isCondition = st is IConditionStatement;
                        if (mustEnter || !isCondition)
                        {
                            var message = Builder.LiteralExpr("Enter " + methodName);
                            var invoke = Builder.DelegateInvokeExpr();
                            invoke.Target = loggingAction;
                            invoke.Arguments.Add(message);
                            outputs.Add(Builder.ExprStatement(invoke));
                            entered = true;
                        }
                        else
                            mustEnter = true;
                    }
                    outputs.Add(st);
                }
                FinishConvertStatement();
            }
            if (methodName != null)
            {
                var message = Builder.LiteralExpr("Leave " + methodName);
                var invoke = Builder.DelegateInvokeExpr();
                invoke.Target = loggingAction;
                invoke.Arguments.Add(message);
                outputs.Add(Builder.ExprStatement(invoke));
            }
            CloseOutputBlock();
        }
    }

    internal class LoggingAttribute : ICompilerAttribute
    {
        public IExpression loggingAction;

        public LoggingAttribute(IExpression loggingAction)
        {
            this.loggingAction = loggingAction;
        }
    }
}
