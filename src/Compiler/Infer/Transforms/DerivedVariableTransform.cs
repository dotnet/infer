// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// This transform clones variables that are derived in some contexts and non-derived in others.  This is needed for VMP to function correctly with deterministic gates.
    /// PREREQUISITE: 
    /// A variable with at least one derived definition must have DerivedVariable attribute set.
    /// </summary>
    internal class DerivedVariableTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "DerivedVariableTransform"; }
        }

        /// <summary>
        /// Counter used to generate variable names.
        /// </summary>
        protected int Count;

        /// <summary>
        /// Convert random assignments to derived variables
        /// </summary>
        /// <param name="iae"></param>
        /// <returns></returns>
        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            iae = (IAssignExpression) base.ConvertAssign(iae);
            IStatement ist = context.FindAncestor<IStatement>();
            if (!context.InputAttributes.Has<Models.Constraint>(ist) &&
               (iae.Expression is IMethodInvokeExpression))
            {
                IMethodInvokeExpression imie = (IMethodInvokeExpression) iae.Expression;
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
                if (ivd != null)
                {
                    bool isDerived = context.InputAttributes.Has<DerivedVariable>(ivd);
                    if (isDerived)
                    {
                        FactorManager.FactorInfo info = CodeRecognizer.GetFactorInfo(context, imie);
                        if (!info.IsDeterministicFactor)
                        {
                            // The variable is derived, but this definition is not derived.
                            // Thus we must convert 
                            //   y = sample()
                            // into
                            //   y_random = sample()
                            //   y = Copy(y_random)
                            // where y_random is not derived.
                            VariableInformation varInfo = VariableInformation.GetVariableInformation(context, ivd);
                            IList<IStatement> stmts = Builder.StmtCollection();
                            string name = ivd.Name + "_random" + (Count++);
                            List<IList<IExpression>> indices = Recognizer.GetIndices(iae.Target);
                            IVariableDeclaration cloneVar = varInfo.DeriveIndexedVariable(stmts, context, name, indices, copyInitializer: true);
                            context.OutputAttributes.Remove<DerivedVariable>(cloneVar);
                            stmts.Add(Builder.AssignStmt(Builder.VarRefExpr(cloneVar), iae.Expression));
                            int ancIndex = context.FindAncestorIndex<IStatement>();
                            context.AddStatementsBeforeAncestorIndex(ancIndex, stmts);
                            Type tp = iae.Target.GetExpressionType();
                            if (tp == null)
                            {
                                Error("Could not determine type of expression: " + iae.Target);
                                return iae;
                            }
                            IExpression copy = Builder.StaticGenericMethod(new Func<PlaceHolder, PlaceHolder>(Clone.Copy), new Type[] {tp},
                                                                           Builder.VarRefExpr(cloneVar));
                            iae = Builder.AssignExpr(iae.Target, copy);
                        }
                    }
                }
            }
            return iae;
        }
    }
}