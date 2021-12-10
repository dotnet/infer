// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Handles any replication made necessary by loops.
    /// </summary>
    internal class ReplicationTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "ReplicationTransform"; }
        }

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            var loopSize = Recognizer.LoopSizeExpression(ifs);
            if (!((loopSize is IVariableReferenceExpression) || (loopSize is IArrayIndexerExpression) ||
                  (loopSize is IArgumentReferenceExpression) || (loopSize is ILiteralExpression)
                  || (loopSize is IPropertyReferenceExpression)))
            {
                Error("Invalid expression type for the size of a loop (" + loopSize.GetType().Name + "): " + loopSize);
                return ifs;
            }
            if (loopSize is ILiteralExpression ile)
            {
                // remove loops that execute for 0 iterations
                int loopSizeAsInt = (int)ile.Value;
                if (loopSizeAsInt == 0)
                    return null;
            }
            Containers c = Containers.GetContainersNeededForExpression(context, loopSize);
            c.Add(ifs);
            IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
            context.InputAttributes.Remove<Containers>(loopVar);
            context.InputAttributes.Set(loopVar, c);
            return base.ConvertFor(ifs);
        }

#if false
    /// <summary>
    /// Converts the body of a for loop, leaving the initializer,
    /// condition and increment statements unchanged.
    /// </summary>
    protected override void ConvertFor(IList<IStatement> bs, IForStatement ifs)
    {
      IForStatement fs = Builder.ForStmt();
      context.SetPrimaryOutput(fs);
      fs.Condition = ifs.Condition;
      fs.Increment = ifs.Increment;
      fs.Initializer = ifs.Initializer;
      fs.Body = ConvertBlock(Builder.BlockStmt(), ifs.Body, null);
      bs.Add(fs);
      context.InputAttributes.CopyObjectAttributesTo(ifs, context.OutputAttributes, fs);
    }
#endif

        // Record the loop context that a variable is created in
        protected override IVariableDeclaration ConvertVariableDecl(IVariableDeclaration ivd)
        {
            context.InputAttributes.Remove<LoopContext>(ivd);
            context.InputAttributes.Set(ivd, new LoopContext(context));
            // do not set containers for loop variables (these are already set by ConvertFor)
            int ancIndex = Recognizer.GetAncestorIndexOfLoopBeingInitialized(context);
            if (ancIndex == -1)
            {
                context.InputAttributes.Remove<Containers>(ivd);
                context.InputAttributes.Set(ivd, new Containers(context));
            }
            return ivd;
        }

        // Shallow copy variable references for efficiency
        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            return ConvertWithReplication(ivre);
        }

        // Shallow copy array creations for efficiency
        protected override IExpression ConvertArrayCreate(IArrayCreateExpression iace)
        {
            return iace;
        }

        /// <summary>
        /// Converts an array index expression to replicate variables referenced in a loop.
        /// </summary>
        /// <param name="iaie">The array indexer expression to convert</param>
        /// <returns>The new expression</returns>
        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            return ConvertWithReplication(iaie);
        }

        /// <summary>
        /// Returns true if expr must be evaluated inside the given container.
        /// </summary>
        /// <param name="container"></param>
        /// <param name="expr"></param>
        /// <returns></returns>
        protected bool NeedsContainer(IStatement container, IExpression expr)
        {
            return Recognizer.GetVariables(expr).Any(ivd =>
            {
                Containers c = context.InputAttributes.Get<Containers>(ivd);
                return c.Contains(container);
            });
        }

        protected bool IsPartitionedExpr(IExpression expr)
        {
            return Recognizer.GetVariables(expr).Any(ivd =>
            {
                Containers c = context.InputAttributes.Get<Containers>(ivd);
                foreach (IStatement container in c.inputs)
                {
                    if (container is IForStatement loop)
                    {
                        IVariableDeclaration loopVar = Recognizer.LoopVariable(loop);
                        bool isPartitionedLoop = context.InputAttributes.Has<Partitioned>(loopVar);
                        if (isPartitionedLoop && loopVar.Equals(ivd)) return true;
                    }
                }
                return false;
            });
        }

        protected bool AnyPartitionedExpr(IList<IExpression> exprs)
        {
            foreach (IExpression expr in exprs)
            {
                if (IsPartitionedExpr(expr)) return true;
            }
            return false;
        }

        /// <summary>
        /// Split expr into a target and extra indices, where target will be replicated and extra indices will be added later.
        /// </summary>
        /// <param name="loop">The loop we are replicating over</param>
        /// <param name="expr">Expression being replicated</param>
        /// <param name="indices">Modified to contain the extra indices</param>
        /// <param name="target">Prefix of expr to replicate</param>
        protected void AddUnreplicatedIndices(IForStatement loop, IExpression expr, List<IEnumerable<IExpression>> indices, out IExpression target)
        {
            if (expr is IArrayIndexerExpression iaie)
            {
                AddUnreplicatedIndices(loop, iaie.Target, indices, out target);
                if (indices.Count == 0)
                {
                    // determine if iaie.Indices can be replicated
                    bool canReplicate = true;
                    foreach (IExpression index in iaie.Indices)
                    {
                        if (NeedsContainer(loop, index))
                        {
                            canReplicate = false;
                            break;
                        }
                    }
                    IVariableDeclaration loopVar = Recognizer.LoopVariable(loop);
                    bool isPartitioned = context.InputAttributes.Has<Partitioned>(loopVar);
                    if (isPartitioned && !AnyPartitionedExpr(iaie.Indices)) canReplicate = false;
                    if (canReplicate)
                    {
                        target = expr;
                        return;
                    }
                    // fall through
                }
                indices.Add(iaie.Indices);
            }
            else target = expr;
        }

        /// <summary>
        /// Returns true if ivd is constant for a given value of the loop variables in lc
        /// </summary>
        /// <param name="ivd"></param>
        /// <param name="constantLoopVars"></param>
        /// <returns></returns>
        private bool IsConstantWrtLoops(IVariableDeclaration ivd, Set<IVariableDeclaration> constantLoopVars)
        {
            if (ivd == null) return true;
            if (CodeRecognizer.IsStochastic(context, ivd)) return false;
            if (constantLoopVars.Contains(ivd)) return true;
            LoopContext lc2 = context.InputAttributes.Get<LoopContext>(ivd);
            // return false if lc2 has any loops that are not in constantLoopVars
            return constantLoopVars.ContainsAll(lc2.loopVariables);
        }

        protected IExpression ConvertWithReplication(IExpression expr)
        {
            IVariableDeclaration baseVar = Recognizer.GetVariableDeclaration(expr);
            // Check if this is an index local variable
            if (baseVar == null) return expr;
            // Check if the variable is stochastic
            if (!CodeRecognizer.IsStochastic(context, baseVar)) return expr;

            // Get the loop context for this variable
            LoopContext lc = context.InputAttributes.Get<LoopContext>(baseVar);
            if (lc == null)
            {
                Error("Loop context not found for '" + baseVar.Name + "'.");
                return expr;
            }

            // Get the reference loop context for this expression
            RefLoopContext rlc = lc.GetReferenceLoopContext(context);
            // If the reference is in the same loop context as the declaration, do nothing.
            if (rlc.loops.Count == 0) return expr;

            // the set of loop variables that are constant wrt the expr
            Set<IVariableDeclaration> constantLoopVars = new Set<IVariableDeclaration>();
            constantLoopVars.AddRange(lc.loopVariables);

            // collect set of all loop variable indices in the expression
            Set<int> embeddedLoopIndices = new Set<int>();
            List<IList<IExpression>> brackets = Recognizer.GetIndices(expr);
            foreach (IList<IExpression> bracket in brackets)
            {
                foreach (IExpression index in bracket)
                {
                    IExpression indExpr = index;
                    if (indExpr is IBinaryExpression ibe)
                    {
                        indExpr = ibe.Left;
                    }
                    IVariableDeclaration indVar = Recognizer.GetVariableDeclaration(indExpr);
                    if (indVar != null)
                    {
                        if (!constantLoopVars.Contains(indVar))
                        {
                            int loopIndex = rlc.loopVariables.IndexOf(indVar);
                            if (loopIndex != -1)
                            {
                                // indVar is a loop variable
                                constantLoopVars.Add(rlc.loopVariables[loopIndex]);
                            }
                            else 
                            {
                                // indVar is not a loop variable
                                LoopContext lc2 = context.InputAttributes.Get<LoopContext>(indVar);
                                foreach (var ivd in lc2.loopVariables)
                                {
                                    if (!constantLoopVars.Contains(ivd))
                                    {
                                        int loopIndex2 = rlc.loopVariables.IndexOf(ivd);
                                        if (loopIndex2 != -1)
                                            embeddedLoopIndices.Add(loopIndex2);
                                        else
                                            Error($"Index {ivd} is not in {rlc} for expression {expr}");
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        foreach(var ivd in Recognizer.GetVariables(indExpr))
                        {
                            if (!constantLoopVars.Contains(ivd))
                            {
                                // copied from above
                                LoopContext lc2 = context.InputAttributes.Get<LoopContext>(ivd);
                                foreach (var ivd2 in lc2.loopVariables)
                                {
                                    if (!constantLoopVars.Contains(ivd2))
                                    {
                                        int loopIndex2 = rlc.loopVariables.IndexOf(ivd2);
                                        if (loopIndex2 != -1)
                                            embeddedLoopIndices.Add(loopIndex2);
                                        else
                                            Error($"Index {ivd2} is not in {rlc} for expression {expr}");
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Find loop variables that must be constant due to condition statements.
            List<IStatement> ancestors = context.FindAncestors<IStatement>();
            foreach (IStatement ancestor in ancestors)
            {
                if (!(ancestor is IConditionStatement ics))
                    continue;
                ConditionBinding binding = new ConditionBinding(ics.Condition);
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(binding.lhs);
                IVariableDeclaration ivd2 = Recognizer.GetVariableDeclaration(binding.rhs);
                int index = rlc.loopVariables.IndexOf(ivd);
                if (index >= 0 && IsConstantWrtLoops(ivd2, constantLoopVars))
                {
                    constantLoopVars.Add(ivd);
                    continue;
                }
                int index2 = rlc.loopVariables.IndexOf(ivd2);
                if (index2 >= 0 && IsConstantWrtLoops(ivd, constantLoopVars))
                {
                    constantLoopVars.Add(ivd2);
                    continue;
                }
            }

            // Determine if this expression is being defined (is on the LHS of an assignment)
            bool isDef = Recognizer.IsBeingMutated(context, expr);

            Containers containers = context.InputAttributes.Get<Containers>(baseVar);

            IExpression originalExpr = expr;

            for (int currentLoop = 0; currentLoop < rlc.loopVariables.Count; currentLoop++)
            {
                IVariableDeclaration loopVar = rlc.loopVariables[currentLoop];
                if (constantLoopVars.Contains(loopVar))
                    continue;
                IForStatement loop = rlc.loops[currentLoop];
                // must replicate across this loop.
                if (isDef)
                {
                    Error("Cannot re-define a variable in a loop.  Variables on the left hand side of an assignment must be indexed by all containing loops.");
                    continue;
                }
                if (embeddedLoopIndices.Contains(currentLoop))
                {
                    string warningText = "This model will consume excess memory due to the indexing expression {0} inside of a loop over {1}. Try simplifying this expression in your model, perhaps by creating auxiliary index arrays.";
                    Warning(string.Format(warningText, originalExpr, loopVar.Name));
                }
                // split expr into a target and extra indices, where target will be replicated and extra indices will be added later
                var extraIndices = new List<IEnumerable<IExpression>>();
                AddUnreplicatedIndices(rlc.loops[currentLoop], expr, extraIndices, out IExpression exprToReplicate);

                VariableInformation varInfo = VariableInformation.GetVariableInformation(context, baseVar);
                IExpression loopSize = Recognizer.LoopSizeExpression(loop);
                IList<IStatement> stmts = Builder.StmtCollection();
                List<IList<IExpression>> inds = Recognizer.GetIndices(exprToReplicate);
                IVariableDeclaration newIndexVar = loopVar;
                // if loopVar is already an indexVar of varInfo, create a new variable
                if (varInfo.HasIndexVar(loopVar))
                {
                    newIndexVar = VariableInformation.GenerateLoopVar(context, "_a");
                    context.InputAttributes.CopyObjectAttributesTo(loopVar, context.OutputAttributes, newIndexVar);
                }
                IVariableDeclaration repVar = varInfo.DeriveArrayVariable(stmts, context, VariableInformation.GenerateName(context, varInfo.Name + "_rep"),
                                                                          loopSize, newIndexVar, inds, useArrays: true);
                if (!context.InputAttributes.Has<DerivedVariable>(repVar))
                    context.OutputAttributes.Set(repVar, new DerivedVariable());
                if (context.InputAttributes.Has<ChannelInfo>(baseVar))
                {
                    VariableInformation repVarInfo = VariableInformation.GetVariableInformation(context, repVar);
                    ChannelInfo ci = ChannelInfo.UseChannel(repVarInfo);
                    ci.decl = repVar;
                    context.OutputAttributes.Set(repVar, ci);
                }

                // Create replicate factor
                Type returnType = Builder.ToType(repVar.VariableType);
                IMethodInvokeExpression repMethod = Builder.StaticGenericMethod(
                    new Func<PlaceHolder, int, PlaceHolder[]>(Clone.Replicate),
                    new Type[] {returnType.GetElementType()}, exprToReplicate, loopSize);

                IExpression assignExpression = Builder.AssignExpr(Builder.VarRefExpr(repVar), repMethod);
                // Copy attributes across from variable to replication expression
                context.InputAttributes.CopyObjectAttributesTo<Algorithm>(baseVar, context.OutputAttributes, repMethod);
                context.InputAttributes.CopyObjectAttributesTo<DivideMessages>(baseVar, context.OutputAttributes, repMethod);
                context.InputAttributes.CopyObjectAttributesTo<GivePriorityTo>(baseVar, context.OutputAttributes, repMethod);
                stmts.Add(Builder.ExprStatement(assignExpression));

                // add any containers missing from context.
                containers = new Containers(context);
                // RemoveUnusedLoops will also remove conditionals involving those loop variables.
                // TODO: investigate whether removing these conditionals could cause a problem, e.g. when the condition is a conjunction of many terms.
                containers = Containers.RemoveUnusedLoops(containers, context, repMethod);
                if (context.InputAttributes.Has<DoNotSendEvidence>(baseVar)) containers = Containers.RemoveStochasticConditionals(containers, context);
                //Containers shouldBeEmpty = containers.GetContainersNotInContext(context, context.InputStack.Count);
                //if (shouldBeEmpty.inputs.Count > 0) { Error("Internal: Variable is out of scope"); return expr; }
                if (containers.Contains(loop))
                {
                    Error("Internal: invalid containers for replicating " + baseVar);
                    break;
                }
                int ancIndex = containers.GetMatchingAncestorIndex(context);
                Containers missing = containers.GetContainersNotInContext(context, ancIndex);
                stmts = Containers.WrapWithContainers(stmts, missing.inputs);
                context.OutputAttributes.Set(repVar, containers);
                List<IForStatement> loops = context.FindAncestors<IForStatement>(ancIndex);
                foreach (IStatement container in missing.inputs)
                {
                    if (container is IForStatement ifs) loops.Add(ifs);
                }
                context.OutputAttributes.Set(repVar, new LoopContext(loops));
                // must convert the output since it may contain 'if' conditions
                context.AddStatementsBeforeAncestorIndex(ancIndex, stmts, true);
                baseVar = repVar;
                expr = Builder.ArrayIndex(Builder.VarRefExpr(repVar), Builder.VarRefExpr(loopVar));
                expr = Builder.JaggedArrayIndex(expr, extraIndices);
            }

            return expr;
        }
    }
}