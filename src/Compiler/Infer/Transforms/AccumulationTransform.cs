// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Optimises message passing by removing redundant messages and operations.
    /// </summary>
    internal class AccumulationTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "AccumulationTransform"; }
        }

        /// <summary>
        /// The number of temporary variables created.
        /// </summary>
        private int tempVarCount = 0;

        public AccumulationTransform()
        {
        }

        protected override IStatement DoConvertStatement(IStatement ist)
        {
            if (ist is IExpressionStatement)
            {
                bool isAccumulator = false;
                IExpressionStatement ies = (IExpressionStatement) ist;
                if (ies.Expression is IAssignExpression)
                {
                    IAssignExpression iae = (IAssignExpression) ies.Expression;
                    AccumulationInfo ac = context.InputAttributes.Get<AccumulationInfo>(ies);
                    if (ac != null)
                    {
                        // This statement defines an accumulator.
                        // Delete the statement and replace with any remaining accumulations.
                        if (!ac.isInitialized) context.AddStatementBeforeCurrent(ac.initializer);
                        ac.isInitialized = false;
                        context.AddStatementsBeforeCurrent(ac.statements);
                        isAccumulator = true;
                    }
                    object ivd = Recognizer.GetVariableDeclaration(iae.Target);
                    if (ivd == null) ivd = Recognizer.GetFieldReference(iae.Target);
                    if (ivd != null)
                    {
                        MessageArrayInformation mai = context.InputAttributes.Get<MessageArrayInformation>(ivd);
                        bool oneUse = (mai != null) && (mai.useCount == 1);
                        List<AccumulationInfo> ais = context.InputAttributes.GetAll<AccumulationInfo>(ivd);
                        foreach (AccumulationInfo ai in ais)
                        {
                            if (!context.InputAttributes.Has<OperatorStatement>(ist))
                            {
                                if (oneUse) return null; // remove the statement
                                else continue;
                            }
                            // assuming that we only accumulate one statement, we should always initialize
                            if (true || !ai.isInitialized)
                            {
                                // add the initializer in the same containers as the original accumulation statement
                                int ancIndex = ai.containers.GetMatchingAncestorIndex(context);
                                Containers missing = ai.containers.GetContainersNotInContext(context, ancIndex);
                                context.AddStatementBeforeAncestorIndex(ancIndex, Containers.WrapWithContainers(ai.initializer, missing.outputs));
                                ai.isInitialized = true;
                            }
                            // This statement defines a variable that contributes to an accumulator.
                            IList<IStatement> stmts = Builder.StmtCollection();
                            IExpression target = iae.Target;
                            Type exprType = target.GetExpressionType();
                            IExpression accumulator = ai.accumulator;
                            Type accumulatorType = ai.type;
                            int rank;
                            Type eltType = Util.GetElementType(exprType, out rank);
                            if (rank == 1 && eltType == accumulatorType)
                            {
                                // This statement defines an array of items to be accumulated.
                                VariableInformation varInfo = context.InputAttributes.Get<VariableInformation>(mai.ci.decl);
                                int indexingDepth = Recognizer.GetIndexingDepth(iae.Target);
                                if (mai.loopVarInfo != null) indexingDepth -= mai.loopVarInfo.indexVarRefs.Length;
                                IExpression size = varInfo.sizes[indexingDepth][0];
                                IVariableDeclaration indexVar = varInfo.indexVars[indexingDepth][0];
                                IForStatement ifs = Builder.ForStmt(indexVar, size);
                                target = Builder.ArrayIndex(target, Builder.VarRefExpr(indexVar));
                                ifs.Body.Statements.Add(ai.GetAccumulateStatement(context, accumulator, target));
                                stmts.Add(ifs);
                            }
                            else
                            {
                                Stack<IList<IExpression>> indexStack = new Stack<IList<IExpression>>();
                                while (exprType != accumulatorType && target is IArrayIndexerExpression)
                                {
                                    // The accumulator is an array type, and this statement defines an element of the accumulator type.
                                    // Peel off the indices of the target.
                                    IArrayIndexerExpression iaie = (IArrayIndexerExpression) target;
                                    indexStack.Push(iaie.Indices);
                                    target = iaie.Target;
                                    accumulatorType = Util.GetElementType(accumulatorType);
                                }
                                while (indexStack.Count > 0)
                                {
                                    // Attach the indices of the target to the accumulator, in reverse order that they were peeled.
                                    accumulator = Builder.ArrayIndex(accumulator, indexStack.Pop());
                                }
                                if (exprType == accumulatorType)
                                {
                                    if (oneUse && !isAccumulator)
                                    {
                                        // convert
                                        //   msg = expr;
                                        // into
                                        //   T temp = expr;
                                        //   acc = SetToProduct(acc, temp);
                                        // and move into ai.statement.
                                        tempVarCount++;
                                        string name = "_temp" + tempVarCount;
                                        IVariableDeclaration tempVar = Builder.VarDecl(name, exprType);
                                        IStatement newStatement = Builder.AssignStmt(Builder.VarDeclExpr(tempVar), iae.Expression);
                                        context.AddStatementBeforeCurrent(newStatement);
                                        context.AddStatementBeforeCurrent(ai.GetAccumulateStatement(context, accumulator, Builder.VarRefExpr(tempVar)));
                                        return null;
                                    }
                                    else
                                    {
                                        // must keep the original statement.
                                        // add 
                                        //   acc = SetToProduct(acc, msg);
                                        // to ai.statements.
                                        context.AddStatementAfterCurrent(ai.GetAccumulateStatement(context, accumulator, iae.Target));
                                    }
                                }
                                else
                                {
                                    Error("Unrecognized variable type: " + StringUtil.TypeToString(exprType));
                                }
                            }
                            Containers containers = new Containers(context);
                            containers = containers.Remove(ai.containers);
                            IList<IStatement> output2 = Builder.StmtCollection();
                            Containers.AddStatementWithContainers(output2, stmts, containers.inputs);
                            foreach (IStatement ist2 in output2)
                                context.AddStatementAfterCurrent(ist2);
                        }
                    }
                }
                if (!isAccumulator)
                {
                    return ist;
                }
                else
                {
                    return null;
                }
            }
            else
            {
                return base.DoConvertStatement(ist);
            }
        }
    }

    /// <summary>
    /// Attached to methods.
    /// </summary>
    internal class AccumulationInfo : ICompilerAttribute
    {
        public static CodeBuilder Builder = CodeBuilder.Instance;

        public IExpression accumulator;
        public Type type;
        internal bool isInitialized;
        internal IStatement initializer;
        public IList<IStatement> statements;

        /// <summary>
        /// Containers of the original statement.
        /// </summary>
        internal Containers containers;

        internal Delegate accumulateMethod;

        internal AccumulationInfo(IExpression accumulator)
        {
            this.accumulator = accumulator;
            this.type = accumulator.GetExpressionType();
            this.statements = new List<IStatement>();
        }

        internal IStatement GetAccumulateStatement(BasicTransformContext context, IExpression accumulator, IExpression value)
        {
            IExpression imie = Builder.StaticGenericMethod(accumulateMethod, new Type[] {accumulator.GetExpressionType()}, accumulator, value);
            return Builder.AssignStmt(accumulator, imie);
        }

        public override string ToString()
        {
            return "AccumulationInfo(" + accumulator + ")";
        }
    }
}