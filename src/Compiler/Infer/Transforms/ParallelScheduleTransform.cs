// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Converts 'for' loops with ParallelSchedule attributes into parallel for loops.
    /// </summary>
    internal class ParallelScheduleTransform : ShallowCopyTransform
    {
        List<Tuple<IExpression, DistributedCommunicationExpression>> distributedCommunicationExpressions = new List<Tuple<IExpression, DistributedCommunicationExpression>>();

        public override string Name
        {
            get { return "ParallelScheduleTransform"; }
        }

        /// <summary>
        /// Converts 'for' loops with ParallelSchedule attributes into parallel for loops.
        /// </summary>
        /// <param name="ifs">The for loop to convert</param>
        /// <returns>The converted statement</returns>
        protected override IStatement ConvertFor(IForStatement ifs)
        {
            if (context.InputAttributes.Has<ConvergenceLoop>(ifs) || (ifs is IBrokenForStatement))
                return base.ConvertFor(ifs);
            IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
            ParallelScheduleExpression pse = context.InputAttributes.Get<ParallelScheduleExpression>(loopVar);
            if (pse != null)
                return ConvertForWithParallelSchedule(ifs, loopVar, pse);
            DistributedScheduleExpression dse = context.InputAttributes.Get<DistributedScheduleExpression>(loopVar);
            if (dse != null)
                return ConvertForWithDistributedSchedule(ifs, loopVar, dse);
            return base.ConvertFor(ifs);
        }

        private IForStatement ConvertForWithParallelSchedule(IForStatement ifs, IVariableDeclaration loopVar, ParallelScheduleExpression pse)
        {
            // Convert this loop into for(block) Parallel.For(thread) for(indexInBlock)
            IVariableDeclaration loopVarBlock = VariableInformation.GenerateLoopVar(context, loopVar.Name + "_Block");
            Sequential sequentialAttr = new Sequential();
            context.OutputAttributes.Set(loopVarBlock, sequentialAttr);
            IVariableDeclaration loopVarInBlock = VariableInformation.GenerateLoopVar(context, loopVar.Name + "_inBlock");
            context.OutputAttributes.Set(loopVarInBlock, sequentialAttr);
            string paramName = VariableInformation.GenerateName(context, loopVar.Name + "_thread");
            var threadParam = Builder.Param(paramName, typeof(int));
            if (!pse.scheduleExpression.GetExpressionType().Equals(typeof(int[][][])))
                Error("argument to ParallelSchedule attribute is not of type int[][][]");
            IExpression itemsInThread = Builder.ArrayIndex(pse.scheduleExpression, Builder.ParamRef(threadParam));
            IExpression itemsInBlock = Builder.ArrayIndex(itemsInThread, Builder.VarRefExpr(loopVarBlock));
            IExpression itemCountInBlock = Builder.PropRefExpr(itemsInBlock, typeof(int[]), "Length");
            IExpression threadCount = Builder.PropRefExpr(pse.scheduleExpression, typeof(int[][][]), "Length");
            IExpression zero = Builder.LiteralExpr(0);
            IExpression blockCount = Builder.PropRefExpr(Builder.ArrayIndex(pse.scheduleExpression, zero), typeof(int[][]), "Length");
            IForStatement loopInBlock = Builder.ForStmt(loopVarInBlock, itemCountInBlock);
            bool isBackwardLoop = !Recognizer.IsForwardLoop(ifs);
            if (isBackwardLoop)
                Recognizer.ReverseLoopDirection(loopInBlock);
            var assignLoopVar = Builder.AssignStmt(Builder.VarDeclExpr(loopVar), Builder.ArrayIndex(itemsInBlock, Builder.VarRefExpr(loopVarInBlock)));
            loopInBlock.Body.Statements.Add(assignLoopVar);
            ConvertStatements(loopInBlock.Body.Statements, ifs.Body.Statements);
            //loopInBlock.Body.Statements.AddRange(ifs.Body.Statements);
            IAnonymousMethodExpression bodyDelegate = Builder.AnonMethodExpr(typeof(Action<int>));
            bodyDelegate.Body = Builder.BlockStmt();
            bodyDelegate.Body.Statements.Add(loopInBlock);
            bodyDelegate.Parameters.Add(threadParam);
            Delegate d = new Func<int, int, Action<int>, ParallelLoopResult>(Parallel.For);
            IMethodInvokeExpression parallelFor = Builder.StaticMethod(d, zero, threadCount, bodyDelegate);
            IStatement loopThread = Builder.ExprStatement(parallelFor);
            IForStatement loopBlock = Builder.ForStmt(loopVarBlock, blockCount);
            loopBlock.Body.Statements.Add(loopThread);
            if (isBackwardLoop)
                Recognizer.ReverseLoopDirection(loopBlock);
            return loopBlock;
        }

        private IForStatement ConvertForWithDistributedSchedule(IForStatement ifs, IVariableDeclaration loopVar, DistributedScheduleExpression dse)
        {
            if(dse.scheduleExpression == null && dse.schedulePerThreadExpression == null)
                return (IForStatement)base.ConvertFor(ifs);
            Sequential attr = context.InputAttributes.Get<Sequential>(loopVar);
            if (attr == null || !attr.BackwardPass)
            {
                Error($"Range '{loopVar.Name}' has a DistributedSchedule attribute but does not have a Sequential attribute with BackwardPass = true");
            }
            // Convert this loop into for(stage) for(indexInBlock) { ... }
            IVariableDeclaration loopVarBlock = VariableInformation.GenerateLoopVar(context, loopVar.Name + "_Block");
            Sequential sequentialAttr = new Sequential();
            context.OutputAttributes.Set(loopVarBlock, sequentialAttr);
            bool isBackwardLoop = !Recognizer.IsForwardLoop(ifs);
            this.distributedCommunicationExpressions.Clear();
            IForStatement loopInBlock;
            IExpression zero = Builder.LiteralExpr(0);
            IExpression distributedStageCount;
            if (dse.schedulePerThreadExpression != null)
            {
                var scheduleForBlock = Builder.ArrayIndex(dse.schedulePerThreadExpression, Builder.VarRefExpr(loopVarBlock));
                distributedStageCount = Builder.PropRefExpr(dse.schedulePerThreadExpression, typeof(int[][][][]), "Length");
                loopInBlock = ConvertForWithParallelSchedule(ifs, loopVar, new ParallelScheduleExpression(scheduleForBlock));
            }
            else
            {
                IVariableDeclaration loopVarInBlock = VariableInformation.GenerateLoopVar(context, loopVar.Name + "_inBlock");
                context.OutputAttributes.Set(loopVarInBlock, sequentialAttr);
                if (!dse.scheduleExpression.GetExpressionType().Equals(typeof(int[][])))
                    Error("argument to DistributedSchedule attribute is not of type int[][]");
                IExpression itemsInBlock = Builder.ArrayIndex(dse.scheduleExpression, Builder.VarRefExpr(loopVarBlock));
                IExpression itemCountInBlock = Builder.PropRefExpr(itemsInBlock, typeof(int[]), "Length");
                distributedStageCount = Builder.PropRefExpr(dse.scheduleExpression, typeof(int[][]), "Length");
                loopInBlock = Builder.ForStmt(loopVarInBlock, itemCountInBlock);
                if (isBackwardLoop)
                    Recognizer.ReverseLoopDirection(loopInBlock);
                var assignLoopVar = Builder.AssignStmt(Builder.VarDeclExpr(loopVar), Builder.ArrayIndex(itemsInBlock, Builder.VarRefExpr(loopVarInBlock)));
                loopInBlock.Body.Statements.Add(assignLoopVar);
                ConvertStatements(loopInBlock.Body.Statements, ifs.Body.Statements);
            }
            IForStatement loopBlock = Builder.ForStmt(loopVarBlock, distributedStageCount);
            IExpression commExpr = dse.commExpression;
            if (isBackwardLoop)
            {
                loopBlock.Body.Statements.Add(loopInBlock);
                foreach (var tuple in this.distributedCommunicationExpressions)
                {
                    var arrayExpr = tuple.Item1;
                    var dce = tuple.Item2;
                    IExpression sendExpr = Builder.ArrayIndex(dce.arrayIndicesToSendExpression, Builder.VarRefExpr(loopVarBlock));
                    IExpression receiveExpr = Builder.ArrayIndex(dce.arrayIndicesToReceiveExpression, Builder.VarRefExpr(loopVarBlock));
                    var sendReceiveMethod = Builder.StaticGenericMethod(new Action<ICommunicator, IList<PlaceHolder>, int[][], int[][]>(Communicator.AlltoallSubarrays),
                        new Type[] { Utilities.Util.GetElementType(arrayExpr.GetExpressionType()) },
                        commExpr, arrayExpr,
                        receiveExpr,
                        sendExpr);
                    loopBlock.Body.Statements.Add(Builder.ExprStatement(sendReceiveMethod));
                }
                Recognizer.ReverseLoopDirection(loopBlock);
            }
            else
            {
                foreach (var tuple in this.distributedCommunicationExpressions)
                {
                    var arrayExpr = tuple.Item1;
                    var dce = tuple.Item2;
                    IExpression sendExpr = Builder.ArrayIndex(dce.arrayIndicesToSendExpression, Builder.VarRefExpr(loopVarBlock));
                    IExpression receiveExpr = Builder.ArrayIndex(dce.arrayIndicesToReceiveExpression, Builder.VarRefExpr(loopVarBlock));
                    var sendReceiveMethod = Builder.StaticGenericMethod(new Action<ICommunicator, IList<PlaceHolder>, int[][], int[][]>(Communicator.AlltoallSubarrays),
                        new Type[] { Utilities.Util.GetElementType(arrayExpr.GetExpressionType()) },
                        commExpr, arrayExpr,
                        sendExpr,
                        receiveExpr);
                    loopBlock.Body.Statements.Add(Builder.ExprStatement(sendReceiveMethod));
                }
                loopBlock.Body.Statements.Add(loopInBlock);
            }
            return loopBlock;
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            var dce = context.GetAttribute<DistributedCommunicationExpression>(imie);
            if (dce != null)
            {
                if (Recognizer.IsStaticGenericMethod(imie, typeof(JaggedSubarrayWithMarginalOp<>), "ItemsAverageConditional"))
                {
                    IExpression backwardExpr = imie.Arguments[0];
                    IExpression marginalExpr = imie.Arguments[2];
                    IExpression indicesExpr = imie.Arguments[3];
                    IExpression indexExpr = imie.Arguments[4];
                    IExpression forwardExpr = imie.Arguments[5];
                    this.distributedCommunicationExpressions.Add(Tuple.Create(marginalExpr, dce));
                }
            }
            // check for statements that multiply across a distributed range
            if (Recognizer.IsStaticGenericMethod(imie, typeof(ReplicatePointOp), "ToDef") ||
                Recognizer.IsStaticGenericMethod(imie, typeof(ReplicateOp_Divide), "ToDef") ||
                Recognizer.IsStaticGenericMethod(imie, typeof(ReplicateOp_NoDivide), "ToDef") ||
                Recognizer.IsStaticGenericMethod(imie, typeof(GetItemsOp<>), "ArrayAverageConditional") ||
                Recognizer.IsStaticGenericMethod(imie, typeof(GetItemsFromJaggedOp<>), "ArrayAverageConditional") ||
                Recognizer.IsStaticGenericMethod(imie, typeof(GetItemsFromDeepJaggedOp<>), "ArrayAverageConditional") ||
                Recognizer.IsStaticGenericMethod(imie, typeof(GetJaggedItemsOp<>), "ArrayAverageConditional") ||
                Recognizer.IsStaticGenericMethod(imie, typeof(GetJaggedItemsFromJaggedOp<>), "ArrayAverageConditional") ||
                Recognizer.IsStaticGenericMethod(imie, typeof(GetDeepJaggedItemsOp<>), "ArrayAverageConditional")
                )
            {
                IExpression arg = imie.Arguments[0];
                object argDecl = Recognizer.GetDeclaration(arg);
                if (argDecl != null)
                {
                    VariableInformation varInfo = context.InputAttributes.Get<VariableInformation>(argDecl);
                    int indexingDepth = Recognizer.GetIndexingDepth(arg);
                    if (varInfo != null && varInfo.indexVars.Count > indexingDepth)
                    {
                        IVariableDeclaration loopVar = varInfo.indexVars[indexingDepth][0];
                        if (loopVar != null)
                        {
                            var dse = context.InputAttributes.Get<DistributedScheduleExpression>(loopVar);
                            bool isDistributed = (dse != null);
                            if (isDistributed)
                            {
                                // A statement that multiplies across a distributed range will only multiply the items in the local process.
                                // To combine the results from each process, add the statement:
                                // shift_rep_B_toDef = Communicator.MultiplyAll(comm, shift_rep_B_toDef);
                                IExpression targetExpr = imie.Arguments[imie.Arguments.Count-1];
                                IExpression commExpr = dse.commExpression;
                                var multiplyMethod = Builder.StaticGenericMethod(new Func<ICommunicator, PlaceHolder, PlaceHolder>(Communicator.MultiplyAll),
                                    new Type[] { targetExpr.GetExpressionType() },
                                    commExpr,
                                    targetExpr);
                                IStatement stmt = Builder.AssignStmt(targetExpr, multiplyMethod);
                                context.AddStatementAfterCurrent(stmt);
                            }
                        }
                    }
                }
            }
            return base.ConvertMethodInvoke(imie);
        }
    }
}