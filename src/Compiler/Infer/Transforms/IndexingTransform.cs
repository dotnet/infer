// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Handles any replication made necessary by constant indexing.
    /// PREREQUISITE: 
    /// Variable declarations must have 'Containers' attributes indicating the containers they belong to.
    /// </summary>
    internal class IndexingTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "IndexingTransform"; }
        }

        /// <summary>
        /// If true, convert Subarray calls into JaggedSubarray whenever possible
        /// </summary>
        internal static bool UseJaggedSubarray = true;

        /// <summary>
        /// If true, convert array indexing into GetItems whenever possible.
        /// </summary>
        internal static bool UseGetItems = true;

        private IndexAnalysisTransform analysis;

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            analysis = new IndexAnalysisTransform();
            analysis.Context.InputAttributes = context.InputAttributes;
            analysis.Transform(itd);
            context.Results = analysis.Context.Results;
            if (!context.Results.IsSuccess)
            {
                Error("analysis failed");
                return itd;
            }
            return base.Transform(itd);
        }

        // Record the containers that a variable is created in
        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            context.InputAttributes.Remove<Containers>(ivde.Variable);
            context.InputAttributes.Set(ivde.Variable, new Containers(context));
            return ivde;
        }

        /// <summary>
        /// This method does all the work of converting literal indexing expressions.
        /// </summary>
        /// <param name="iaie"></param>
        /// <returns></returns>
        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            IndexAnalysisTransform.IndexInfo info;
            if (!analysis.indexInfoOf.TryGetValue(iaie, out info))
            {
                return base.ConvertArrayIndexer(iaie);
            }
            // Determine if this is a definition i.e. the variable is on the left hand side of an assignment
            // This must be done before base.ConvertArrayIndexer changes the expression!
            bool isDef = Recognizer.IsBeingMutated(context, iaie);
            if (info.clone != null)
            {
                if (isDef)
                {
                    // check that extra literal indices in the target are zero.
                    // for example, if iae is x[i][0] = (...) then it is safe to add x_uses[i] = Rep(x[i])
                    // if iae is x[i][1] = (...) then it is safe to add x_uses[i][1] = Rep(x[i][1]) 
                    // but not x_uses[i] = Rep(x[i]) since this will be a duplicate.
                    bool extraLiteralsAreZero = true;
                    int parentIndex = context.InputStack.Count - 2;
                    object parent = context.GetAncestor(parentIndex);
                    while (parent is IArrayIndexerExpression parent_iaie)
                    {
                        foreach (IExpression index in parent_iaie.Indices)
                        {
                            if (index is ILiteralExpression ile)
                            {
                                int value = (int)ile.Value;
                                if (value != 0)
                                {
                                    extraLiteralsAreZero = false;
                                    break;
                                }
                            }
                        }
                        parentIndex--;
                        parent = context.GetAncestor(parentIndex);
                    }
                    if (false && extraLiteralsAreZero)
                    {
                        // change:
                        //   array[0] = f()
                        // into:
                        //   array_item0 = f()
                        //   array[0] = Copy(array_item0)
                        IExpression copy = Builder.StaticGenericMethod(new Func<PlaceHolder, PlaceHolder>(Clone.Copy<PlaceHolder>), new Type[] { iaie.GetExpressionType() },
                                                                       info.clone);
                        IStatement copySt = Builder.AssignStmt(iaie, copy);
                        context.AddStatementAfterCurrent(copySt);
                    }
                }
                return info.clone;
            }

            if (isDef)
            {
                // do not clone the lhs of an array create assignment.
                IAssignExpression assignExpr = context.FindAncestor<IAssignExpression>();
                if (assignExpr.Expression is IArrayCreateExpression)
                    return iaie;
            }

            IVariableDeclaration originalBaseVar = Recognizer.GetVariableDeclaration(iaie);
            // If the variable is not stochastic, return
            if (!CodeRecognizer.IsStochastic(context, originalBaseVar))
                return iaie;

            IExpression newExpr = null;
            IVariableDeclaration baseVar = originalBaseVar;
            IVariableDeclaration newvd = null;
            IExpression rhsExpr = null;
            Containers containers = info.containers;
            Type tp = iaie.GetExpressionType();
            if (tp == null)
            {
                Error("Could not determine type of expression: " + iaie);
                return iaie;
            }
            var stmts = Builder.StmtCollection();
            var stmtsAfter = Builder.StmtCollection();

            // does the expression have the form array[indices[k]][indices2[k]][indices3[k]]?
            if (newvd == null && UseGetItems && iaie.Indices.Count == 1)
            {
                if (iaie.Target is IArrayIndexerExpression iaie2 &&
                    iaie.Indices[0] is IArrayIndexerExpression index3 &&
                    index3.Indices.Count == 1 &&
                    index3.Indices[0] is IVariableReferenceExpression innerIndex3 &&
                    iaie2.Target is IArrayIndexerExpression iaie3 &&
                    iaie2.Indices.Count == 1 &&
                    iaie2.Indices[0] is IArrayIndexerExpression index2 &&
                    index2.Indices.Count == 1 &&
                    index2.Indices[0] is IVariableReferenceExpression innerIndex2 &&
                    innerIndex2.Equals(innerIndex3) &&
                    iaie3.Indices.Count == 1 &&
                    iaie3.Indices[0] is IArrayIndexerExpression index &&
                    index.Indices.Count == 1 &&
                    index.Indices[0] is IVariableReferenceExpression innerIndex &&
                    innerIndex.Equals(innerIndex2))
                {
                    IForStatement innerLoop = Recognizer.GetLoopForVariable(context, innerIndex);
                    if (innerLoop != null && 
                        AreLoopsDisjoint(innerLoop, iaie3.Target, index.Target))
                    {
                        // expression has the form array[indices[k]][indices2[k]][indices3[k]]
                        if (isDef)
                        {
                            Error("fancy indexing not allowed on left hand side");
                            return iaie;
                        }
                        WarnIfLocal(index.Target, iaie3.Target, iaie);
                        WarnIfLocal(index2.Target, iaie3.Target, iaie);
                        WarnIfLocal(index3.Target, iaie3.Target, iaie);
                        containers = RemoveReferencesTo(containers, innerIndex);
                        IExpression loopSize = Recognizer.LoopSizeExpression(innerLoop);
                        var indices = Recognizer.GetIndices(iaie);
                        // Build name of replacement variable from index values
                        StringBuilder sb = new StringBuilder("_item");
                        AppendIndexString(sb, iaie3);
                        AppendIndexString(sb, iaie2);
                        AppendIndexString(sb, iaie);
                        string name = ToString(iaie3.Target) + sb.ToString();
                        VariableInformation varInfo = VariableInformation.GetVariableInformation(context, baseVar);
                        newvd = varInfo.DeriveArrayVariable(stmts, context, name, loopSize, Recognizer.GetVariableDeclaration(innerIndex), indices);
                        if (!context.InputAttributes.Has<DerivedVariable>(newvd))
                            context.InputAttributes.Set(newvd, new DerivedVariable());
                        IExpression getItems = Builder.StaticGenericMethod(new Func<IReadOnlyList<IReadOnlyList<IReadOnlyList<PlaceHolder>>>, IReadOnlyList<int>, IReadOnlyList<int>, IReadOnlyList<int>, PlaceHolder[]>(Collection.GetItemsFromDeepJagged),
                           new Type[] { tp }, iaie3.Target, index.Target, index2.Target, index3.Target);
                        context.InputAttributes.CopyObjectAttributesTo<Algorithm>(baseVar, context.OutputAttributes, getItems);
                        stmts.Add(Builder.AssignStmt(Builder.VarRefExpr(newvd), getItems));
                        newExpr = Builder.ArrayIndex(Builder.VarRefExpr(newvd), innerIndex);
                        rhsExpr = getItems;
                    }
                }
            }
            // does the expression have the form array[indices[k]][indices2[k]]?
            if (newvd == null && UseGetItems && iaie.Indices.Count == 1)
            {
                if (iaie.Target is IArrayIndexerExpression target &&
                    iaie.Indices[0] is IArrayIndexerExpression index2 &&
                    index2.Indices.Count == 1 && 
                    index2.Indices[0] is IVariableReferenceExpression innerIndex &&
                    target.Indices.Count == 1 && 
                    target.Indices[0] is IArrayIndexerExpression index)
                {
                    IForStatement innerLoop = Recognizer.GetLoopForVariable(context, innerIndex);
                    if (index.Indices.Count == 1 &&
                        index.Indices[0].Equals(innerIndex) && 
                        innerLoop != null && 
                        AreLoopsDisjoint(innerLoop, target.Target, index.Target))
                    {
                        // expression has the form array[indices[k]][indices2[k]]
                        if (isDef)
                        {
                            Error("fancy indexing not allowed on left hand side");
                            return iaie;
                        }
                        var innerLoops = new List<IForStatement>();
                        innerLoops.Add(innerLoop);
                        var indexTarget = index.Target;
                        var index2Target = index2.Target;
                        // check if the index array is jagged, i.e. array[indices[k][j]]
                        while (indexTarget is IArrayIndexerExpression indexTargetExpr && 
                            index2Target is IArrayIndexerExpression index2TargetExpr)
                        {
                            if (indexTargetExpr.Indices.Count == 1 && 
                                indexTargetExpr.Indices[0] is IVariableReferenceExpression innerIndexTarget &&
                                index2TargetExpr.Indices.Count == 1 && 
                                index2TargetExpr.Indices[0] is IVariableReferenceExpression innerIndex2Target)
                            {
                                IForStatement indexTargetLoop = Recognizer.GetLoopForVariable(context, innerIndexTarget);
                                if (indexTargetLoop != null && 
                                    AreLoopsDisjoint(indexTargetLoop, target.Target, indexTargetExpr.Target) &&
                                    innerIndexTarget.Equals(innerIndex2Target))
                                {
                                    innerLoops.Add(indexTargetLoop);
                                    indexTarget = indexTargetExpr.Target;
                                    index2Target = index2TargetExpr.Target;
                                }
                                else
                                    break;
                            }
                            else
                                break;
                        }
                        WarnIfLocal(indexTarget, target.Target, iaie);
                        WarnIfLocal(index2Target, target.Target, iaie);
                        innerLoops.Reverse();
                        var loopSizes = innerLoops.ListSelect(ifs => new[] { Recognizer.LoopSizeExpression(ifs) });
                        var newIndexVars = innerLoops.ListSelect(ifs => new[] { Recognizer.LoopVariable(ifs) });
                        // Build name of replacement variable from index values
                        StringBuilder sb = new StringBuilder("_item");
                        AppendIndexString(sb, target);
                        AppendIndexString(sb, iaie);
                        string name = ToString(target.Target) + sb.ToString();
                        VariableInformation varInfo = VariableInformation.GetVariableInformation(context, baseVar);
                        var indices = Recognizer.GetIndices(iaie);
                        newvd = varInfo.DeriveArrayVariable(stmts, context, name, loopSizes, newIndexVars, indices);
                        if (!context.InputAttributes.Has<DerivedVariable>(newvd))
                            context.InputAttributes.Set(newvd, new DerivedVariable());
                        IExpression getItems;
                        if (innerLoops.Count == 1)
                        {
                            getItems = Builder.StaticGenericMethod(new Func<IReadOnlyList<IReadOnlyList<PlaceHolder>>, IReadOnlyList<int>, IReadOnlyList<int>, PlaceHolder[]>(Collection.GetItemsFromJagged),
                                                                               new Type[] { tp }, target.Target, indexTarget, index2Target);
                        }
                        else if (innerLoops.Count == 2)
                        {
                            getItems = Builder.StaticGenericMethod(new Func<IReadOnlyList<IReadOnlyList<PlaceHolder>>, IReadOnlyList<IReadOnlyList<int>>, IReadOnlyList<IReadOnlyList<int>>, PlaceHolder[][]>(Collection.GetJaggedItemsFromJagged),
                                                                               new Type[] { tp }, target.Target, indexTarget, index2Target);
                        }
                        else
                            throw new NotImplementedException($"innerLoops.Count = {innerLoops.Count}");
                        context.InputAttributes.CopyObjectAttributesTo<Algorithm>(baseVar, context.OutputAttributes, getItems);
                        stmts.Add(Builder.AssignStmt(Builder.VarRefExpr(newvd), getItems));
                        var newIndices = newIndexVars.ListSelect(ivds => Util.ArrayInit(ivds.Length, i => Builder.VarRefExpr(ivds[i])));
                        newExpr = Builder.JaggedArrayIndex(Builder.VarRefExpr(newvd), newIndices);
                        rhsExpr = getItems;
                    }
                    else if(HasAnyCommonLoops(index, index2))
                    {
                        Warning($"This model will consume excess memory due to the indexing expression {iaie} since {index} and {index2} have larger depth than the compiler can handle.");
                    }
                }
            }
            if (newvd == null)
            {
                IArrayIndexerExpression originalExpr = iaie;
                if (UseGetItems)
                    iaie = (IArrayIndexerExpression)base.ConvertArrayIndexer(iaie);
                if (!object.ReferenceEquals(iaie.Target, originalExpr.Target) && false)
                {
                    // TODO: determine if this warning is useful or not
                    string warningText = "This model may consume excess memory due to the jagged indexing expression {0}";
                    Warning(string.Format(warningText, originalExpr));
                }

                // get the baseVar of the new expression.
                baseVar = Recognizer.GetVariableDeclaration(iaie);
                VariableInformation varInfo = VariableInformation.GetVariableInformation(context, baseVar);

                var indices = Recognizer.GetIndices(iaie);
                // Build name of replacement variable from index values
                StringBuilder sb = new StringBuilder("_item");
                AppendIndexString(sb, iaie);
                string name = ToString(iaie.Target) + sb.ToString();

                // does the expression have the form array[indices[k]]?
                if (UseGetItems &&
                    iaie.Indices.Count == 1 &&
                    iaie.Indices[0] is IArrayIndexerExpression index &&
                    index.Indices.Count == 1 &&
                    index.Indices[0] is IVariableReferenceExpression innerIndex)
                {
                    // expression has the form array[indices[k]]
                    IForStatement innerLoop = Recognizer.GetLoopForVariable(context, innerIndex);
                    if (innerLoop != null &&
                        AreLoopsDisjoint(innerLoop, iaie.Target, index.Target))
                    {
                        if (isDef)
                        {
                            Error("fancy indexing not allowed on left hand side");
                            return iaie;
                        }
                        var innerLoops = new List<IForStatement>();
                        innerLoops.Add(innerLoop);
                        var indexTarget = index.Target;
                        // check if the index array is jagged, i.e. array[indices[k][j]]
                        while (indexTarget is IArrayIndexerExpression index2)
                        {
                            if (index2.Indices.Count == 1 &&
                                index2.Indices[0] is IVariableReferenceExpression innerIndex2)
                            {
                                IForStatement innerLoop2 = Recognizer.GetLoopForVariable(context, innerIndex2);
                                if (innerLoop2 != null &&
                                    AreLoopsDisjoint(innerLoop2, iaie.Target, index2.Target))
                                {
                                    innerLoops.Add(innerLoop2);
                                    indexTarget = index2.Target;
                                    // This limit must match the number of handled cases below.
                                    if (innerLoops.Count == 3) break;
                                }
                                else
                                    break;
                            }
                            else
                                break;
                        }
                        WarnIfLocal(indexTarget, iaie.Target, originalExpr);
                        innerLoops.Reverse();
                        var loopSizes = innerLoops.ListSelect(ifs => new[] { Recognizer.LoopSizeExpression(ifs) });
                        var newIndexVars = innerLoops.ListSelect(ifs => new[] { Recognizer.LoopVariable(ifs) });
                        newvd = varInfo.DeriveArrayVariable(stmts, context, name, loopSizes, newIndexVars, indices);
                        if (!context.InputAttributes.Has<DerivedVariable>(newvd))
                            context.InputAttributes.Set(newvd, new DerivedVariable());
                        IExpression getItems;
                        if (innerLoops.Count == 1)
                        {
                            getItems = Builder.StaticGenericMethod(new Func<IReadOnlyList<PlaceHolder>, IReadOnlyList<int>, PlaceHolder[]>(Collection.GetItems),
                                                                           new Type[] { tp }, iaie.Target, indexTarget);
                        }
                        else if (innerLoops.Count == 2)
                        {
                            getItems = Builder.StaticGenericMethod(new Func<IReadOnlyList<PlaceHolder>, IReadOnlyList<IReadOnlyList<int>>, PlaceHolder[][]>(Collection.GetJaggedItems),
                                                                           new Type[] { tp }, iaie.Target, indexTarget);
                        }
                        else if (innerLoops.Count == 3)
                        {
                            getItems = Builder.StaticGenericMethod(new Func<IReadOnlyList<PlaceHolder>, IReadOnlyList<IReadOnlyList<IReadOnlyList<int>>>, PlaceHolder[][][]>(Collection.GetDeepJaggedItems),
                                                                           new Type[] { tp }, iaie.Target, indexTarget);
                        }
                        else
                            throw new NotImplementedException($"innerLoops.Count = {innerLoops.Count}");
                        context.InputAttributes.CopyObjectAttributesTo<Algorithm>(baseVar, context.OutputAttributes, getItems);
                        stmts.Add(Builder.AssignStmt(Builder.VarRefExpr(newvd), getItems));
                        var newIndices = newIndexVars.ListSelect(ivds => Util.ArrayInit(ivds.Length, i => Builder.VarRefExpr(ivds[i])));
                        newExpr = Builder.JaggedArrayIndex(Builder.VarRefExpr(newvd), newIndices);
                        rhsExpr = getItems;
                    }
                }
                if (newvd == null)
                {
                    if (UseGetItems && info.count < 2)
                        return iaie;
                    try
                    {
                        newvd = varInfo.DeriveIndexedVariable(stmts, context, name, indices, copyInitializer: isDef);
                    }
                    catch (Exception ex)
                    {
                        Error(ex.Message, ex);
                        return iaie;
                    }
                    context.OutputAttributes.Remove<DerivedVariable>(newvd);
                    newExpr = Builder.VarRefExpr(newvd);
                    rhsExpr = iaie;
                    if (isDef)
                    {
                        // change:
                        //   array[0] = f()
                        // into:
                        //   array_item0 = f()
                        //   array[0] = Copy(array_item0)
                        IExpression copy = Builder.StaticGenericMethod(new Func<PlaceHolder, PlaceHolder>(Clone.Copy), new Type[] { tp }, newExpr);
                        IStatement copySt = Builder.AssignStmt(iaie, copy);
                        stmtsAfter.Add(copySt);
                        if (!context.InputAttributes.Has<DerivedVariable>(baseVar))
                            context.InputAttributes.Set(baseVar, new DerivedVariable());
                    }
                    else if (!info.IsAssignedTo)
                    {
                        // change:
                        //   x = f(array[0])
                        // into:
                        //   array_item0 = Copy(array[0])
                        //   x = f(array_item0)
                        IExpression copy = Builder.StaticGenericMethod(new Func<PlaceHolder, PlaceHolder>(Clone.Copy), new Type[] { tp }, iaie);
                        IStatement copySt = Builder.AssignStmt(Builder.VarRefExpr(newvd), copy);
                        //if (attr != null) context.OutputAttributes.Set(copySt, attr);
                        stmts.Add(copySt);
                        context.InputAttributes.Set(newvd, new DerivedVariable());
                    }
                }
            }

            // Reduce memory consumption by declaring the clone outside of unnecessary loops.
            // This way, the item is cloned outside the loop and then replicated, instead of replicating the entire array and cloning the item.
            containers = Containers.RemoveUnusedLoops(containers, context, rhsExpr);
            if (context.InputAttributes.Has<DoNotSendEvidence>(originalBaseVar))
                containers = Containers.RemoveStochasticConditionals(containers, context);
            if (true)
            {
                IStatement st = GetBindingSetContainer(FilterBindingSet(info.bindings, 
                    binding => Containers.ContainsExpression(containers.inputs, context, binding.GetExpression())));
                if (st != null)
                    containers.Add(st);
            }
            // To put the declaration in the desired containers, we find an ancestor which includes as many of the containers as possible,
            // then wrap the declaration with the remaining containers.
            int ancIndex = containers.GetMatchingAncestorIndex(context);
            Containers missing = containers.GetContainersNotInContext(context, ancIndex);
            stmts = Containers.WrapWithContainers(stmts, missing.outputs);
            context.AddStatementsBeforeAncestorIndex(ancIndex, stmts);
            stmtsAfter = Containers.WrapWithContainers(stmtsAfter, missing.outputs);
            context.AddStatementsAfterAncestorIndex(ancIndex, stmtsAfter);
            context.InputAttributes.Set(newvd, containers);
            info.clone = newExpr;
            return newExpr;
        }

        private void WarnIfLocal(IExpression indexTarget, IExpression target, IExpression originalExpr)
        {
            if (indexTarget is IArrayIndexerExpression && HasAnyNewLoops(target, indexTarget))
            {
                Warning($"This model will consume excess memory due to the indexing expression {originalExpr} since {indexTarget} has larger depth than the compiler can handle.");
            }
            // is the indexTarget local to a loop?
            IVariableDeclaration indexVar = Recognizer.GetVariableDeclaration(indexTarget);
            if (indexVar != null)
            {
                var indexContainers = context.InputAttributes.Get<Containers>(indexVar);
                if (indexContainers.inputs.Exists(container => container is IForStatement))
                {
                    string warningText = "This model will consume excess memory due to the indexing expression {0} since {1} is local to a loop.  To use less memory, make {1} an array at the top level.";
                    Warning(string.Format(warningText, originalExpr, indexTarget));
                }
            }
        }

        private bool AreLoopsDisjoint(IForStatement loop, IExpression expr, IExpression expr2)
        {
            List<IStatement> targetLoops = Containers.GetLoopsNeededForExpression(context, expr, -1, false);
            List<IStatement> indexLoops = Containers.GetLoopsNeededForExpression(context, expr2, -1, false);
            Set<IStatement> parentLoops = new Set<IStatement>();
            parentLoops.AddRange(targetLoops);
            parentLoops.AddRange(indexLoops);
            if (Containers.ListContains(parentLoops, loop))
            {
                // expression has the form array[k][indices[k]] or array[indices[k][k]]
                return false;
            }
            return true;
        }

        private bool HasAnyNewLoops(IExpression expr, IExpression expr2)
        {
            List<IStatement> targetLoops = Containers.GetLoopsNeededForExpression(context, expr, -1, false);
            List<IStatement> indexLoops = Containers.GetLoopsNeededForExpression(context, expr2, -1, false);
            return indexLoops.Any(loop => !Containers.ListContains(targetLoops, loop));
        }

        private bool HasAnyCommonLoops(IExpression expr, IExpression expr2)
        {
            List<IStatement> targetLoops = Containers.GetLoopsNeededForExpression(context, expr, -1, false);
            List<IStatement> indexLoops = Containers.GetLoopsNeededForExpression(context, expr2, -1, false);
            return indexLoops.Any(loop => Containers.ListContains(targetLoops, loop));
        }

        private void AppendIndexString(StringBuilder sb, IArrayIndexerExpression iaie)
        {
            for (int i = 0; i < iaie.Indices.Count; i++)
            {
                if (i > 0)
                    sb.Append("_");
                IExpression indExpr = iaie.Indices[i];
                sb.Append(indExpr.ToString());
            }
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            IExpression result = base.ConvertMethodInvoke(imie);
            if (result is IMethodInvokeExpression)
                imie = (IMethodInvokeExpression)result;
            else
                return result;
            if (UseJaggedSubarray && Recognizer.IsStaticGenericMethod(imie, new Func<IReadOnlyList<PlaceHolder>, IReadOnlyList<int>, IReadOnlyList<PlaceHolder>>(Collection.Subarray)))
            {
                // check for the form Subarray(arrayExpr, indices[i]) where arrayExpr does not depend on i
                IExpression arrayExpr = imie.Arguments[0];
                IExpression arg1 = imie.Arguments[1];
                if (arg1 is IArrayIndexerExpression)
                {
                    IArrayIndexerExpression index = (IArrayIndexerExpression)arg1;
                    if (index.Indices.Count == 1 && index.Indices[0] is IVariableReferenceExpression)
                    {
                        // index has the form indices[i]
                        List<IStatement> targetLoops = Containers.GetLoopsNeededForExpression(context, arrayExpr, -1, false);
                        List<IStatement> indexLoops = Containers.GetLoopsNeededForExpression(context, index.Target, -1, false);
                        Set<IStatement> parentLoops = new Set<IStatement>();
                        parentLoops.AddRange(targetLoops);
                        parentLoops.AddRange(indexLoops);
                        IVariableReferenceExpression innerIndex = (IVariableReferenceExpression)index.Indices[0];
                        IForStatement innerLoop = Recognizer.GetLoopForVariable(context, innerIndex);
                        foreach (IStatement loop in parentLoops)
                        {
                            if (Containers.ContainersAreEqual(loop, innerLoop))
                            {
                                // arrayExpr depends on i
                                return imie;
                            }
                        }
                        IVariableDeclaration arrayVar = Recognizer.GetVariableDeclaration(arrayExpr);
                        // If the variable is not stochastic, return
                        if (arrayVar == null)
                            return imie;
                        VariableInformation arrayInfo = VariableInformation.GetVariableInformation(context, arrayVar);
                        if (!arrayInfo.IsStochastic)
                            return imie;
                        object indexVar = Recognizer.GetDeclaration(index);
                        VariableInformation indexInfo = VariableInformation.GetVariableInformation(context, indexVar);
                        int depth = Recognizer.GetIndexingDepth(index);
                        IExpression resultSize = indexInfo.sizes[depth][0];
                        var indices = Recognizer.GetIndices(index);
                        int replaceCount = 0;
                        resultSize = indexInfo.ReplaceIndexVars(context, resultSize, indices, null, ref replaceCount);
                        indexInfo.DefineIndexVarsUpToDepth(context, depth + 1);
                        IVariableDeclaration resultIndex = indexInfo.indexVars[depth][0];
                        Type arrayType = arrayExpr.GetExpressionType();
                        Type elementType = Util.GetElementType(arrayType);

                        // create a new variable arrayExpr_indices = JaggedSubarray(arrayExpr, indices)
                        string name = ToString(arrayExpr) + "_" + ToString(index.Target);

                        var stmts = Builder.StmtCollection();
                        var arrayIndices = Recognizer.GetIndices(arrayExpr);
                        var bracket = Builder.ExprCollection();
                        bracket.Add(Builder.ArrayIndex(index, Builder.VarRefExpr(resultIndex)));
                        arrayIndices.Add(bracket);
                        IExpression loopSize = Recognizer.LoopSizeExpression(innerLoop);
                        IVariableDeclaration temp = arrayInfo.DeriveArrayVariable(stmts, context, name, resultSize, resultIndex, arrayIndices);
                        VariableInformation tempInfo = VariableInformation.GetVariableInformation(context, temp);
                        stmts.Clear();
                        IVariableDeclaration newvd = tempInfo.DeriveArrayVariable(stmts, context, name, loopSize, Recognizer.GetVariableDeclaration(innerIndex));
                        if (!context.InputAttributes.Has<DerivedVariable>(newvd))
                            context.InputAttributes.Set(newvd, new DerivedVariable());
                        IExpression rhs = Builder.StaticGenericMethod(new Func<IReadOnlyList<PlaceHolder>, int[][], PlaceHolder[][]>(Collection.JaggedSubarray),
                                                                      new Type[] { elementType }, arrayExpr, index.Target);
                        context.InputAttributes.CopyObjectAttributesTo<Algorithm>(newvd, context.OutputAttributes, rhs);
                        stmts.Add(Builder.AssignStmt(Builder.VarRefExpr(newvd), rhs));

                        // Reduce memory consumption by declaring the clone outside of unnecessary loops.
                        // This way, the item is cloned outside the loop and then replicated, instead of replicating the entire array and cloning the item.
                        Containers containers = new Containers(context);
                        containers = RemoveReferencesTo(containers, innerIndex);
                        containers = Containers.RemoveUnusedLoops(containers, context, rhs);
                        if (context.InputAttributes.Has<DoNotSendEvidence>(arrayVar))
                            containers = Containers.RemoveStochasticConditionals(containers, context);
                        // To put the declaration in the desired containers, we find an ancestor which includes as many of the containers as possible,
                        // then wrap the declaration with the remaining containers.
                        int ancIndex = containers.GetMatchingAncestorIndex(context);
                        Containers missing = containers.GetContainersNotInContext(context, ancIndex);
                        stmts = Containers.WrapWithContainers(stmts, missing.outputs);
                        context.AddStatementsBeforeAncestorIndex(ancIndex, stmts);
                        context.InputAttributes.Set(newvd, containers);

                        // convert into arrayExpr_indices[i]
                        IExpression newExpr = Builder.ArrayIndex(Builder.VarRefExpr(newvd), innerIndex);
                        newExpr = Builder.StaticGenericMethod(new Func<PlaceHolder, PlaceHolder>(Clone.Copy<PlaceHolder>), new Type[] { newExpr.GetExpressionType() }, newExpr);
                        return newExpr;
                    }
                }
            }
            return imie;
        }

        /// <summary>
        /// Attach DerivedVariable attributes to newly created variables
        /// </summary>
        /// <param name="iae"></param>
        /// <returns></returns>
        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            iae = (IAssignExpression)base.ConvertAssign(iae);
            if (iae.Expression is IMethodInvokeExpression imie)
            {
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
                if (ivd != null)
                {
                    bool isDerived = context.InputAttributes.Has<DerivedVariable>(ivd);
                    if (!isDerived)
                    {
                        FactorManager.FactorInfo info = CodeRecognizer.GetFactorInfo(context, imie);
                        if (info.IsDeterministicFactor)
                        {
                            context.InputAttributes.Set(ivd, new DerivedVariable());
                        }
                    }
                }
            }
            return iae;
        }

        /// <summary>
        /// Make a condition statement whose condition is the disjunction of all bindings that satisfy a predicate.
        /// </summary>
        /// <param name="bindingSet">Disjunction of conjunctions of bindings.</param>
        /// <returns></returns>
        internal static IStatement GetBindingSetContainer(Set<IReadOnlyCollection<ConditionBinding>> bindingSet)
        {
            IExpression condition = null;
            foreach (var bindings in bindingSet)
            {
                IExpression branch = null;
                foreach (ConditionBinding binding in bindings)
                {
                    IExpression term = binding.GetExpression();
                    if (branch == null)
                        branch = term;
                    else
                        branch = Builder.BinaryExpr(branch, BinaryOperator.BooleanAnd, term);
                }
                if (branch == null) return null;
                if (condition == null)
                    condition = branch;
                else
                    condition = Builder.BinaryExpr(condition, BinaryOperator.BooleanOr, branch);
            }
            if (condition == null)
                return null;
            return Builder.CondStmt(condition, Builder.BlockStmt());
        }

        internal static Set<IReadOnlyCollection<ConditionBinding>> FilterBindingSet(Set<IReadOnlyCollection<ConditionBinding>> bindingSet, Func<ConditionBinding, bool> predicate)
        {
            return Set<IReadOnlyCollection<ConditionBinding>>.FromEnumerable(bindingSet.Select(bindings =>
                    bindings.Where(predicate).ToList()
                ));
        }

        private Containers RemoveReferencesTo(Containers containers, IExpression expr)
        {
            Containers result = new Containers();
            for (int i = 0; i < containers.inputs.Count; i++)
            {
                IStatement container = containers.inputs[i];
                if (container is IForStatement)
                {
                    IForStatement ifs = container as IForStatement;
                    if (Builder.ContainsExpression(ifs.Condition, expr))
                        continue;
                }
                else if (container is IConditionStatement)
                {
                    IConditionStatement ics = (IConditionStatement)container;
                    if (Builder.ContainsExpression(ics.Condition, expr))
                        continue;
                }
                result.inputs.Add(container);
                result.outputs.Add(containers.outputs[i]);
            }
            return result;
        }

        private string ToString(IExpression expr)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
            if (ivd != null)
                return ivd.Name;
            return expr.ToString();
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}