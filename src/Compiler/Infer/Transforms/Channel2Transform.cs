// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Transforms variable references into channels, by duplicating variables into uses arrays.  
    /// A channel is a variable which is assigned once and referenced only once.  
    /// It corresponds to an edge in a factor graph.
    /// </summary>
    internal class Channel2Transform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "Channel2Transform"; }
        }

        internal bool debug;

        private ChannelAnalysisTransform analysis;

        private readonly Dictionary<IVariableDeclaration, VariableToChannelInformation> usesOfVariable =
            new Dictionary<IVariableDeclaration, VariableToChannelInformation>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            analysis = new ChannelAnalysisTransform();
            analysis.Context.InputAttributes = context.InputAttributes;
            analysis.Transform(itd);
            context.Results = analysis.Context.Results;
            var itdOut = base.Transform(itd);
            if (context.trackTransform && debug)
            {
                IBlockStatement block = Builder.BlockStmt();
                foreach (var entry in analysis.usageInfo)
                {
                    IVariableDeclaration ivd = entry.Key;
                    var info = entry.Value;
                    block.Statements.Add(Builder.CommentStmt(info.ToString()));
                }
                context.OutputAttributes.Add(itdOut, new DebugInfo()
                {
                    Transform = this,
                    Name = "analysis",
                    Value = block
                });
            }
            return itdOut;
        }

        private readonly bool SwapIndices;
        private readonly bool replicateEvidenceVars;

        public Channel2Transform(bool swapIndices, bool replicateEvidenceVars)
        {
            this.SwapIndices = swapIndices;
            this.replicateEvidenceVars = replicateEvidenceVars;
        }

        /// <summary>
        /// Only converts the contained statements in a for loop, leaving the initializer,
        /// condition and increment statements unchanged.
        /// </summary>
        /// <remarks>This method includes a number of checks to ensure the loop is valid e.g. that the initializer, condition and increment 
        /// are all of the appropriate form.</remarks>
        protected override IStatement ConvertFor(IForStatement ifs)
        {
            IForStatement fs = Builder.ForStmt();
            context.SetPrimaryOutput(fs);
            // Check condition is valid
            fs.Condition = ifs.Condition;
            if (!(fs.Condition is IBinaryExpression expression) || (expression.Operator != BinaryOperator.LessThan))
                Error("For statement condition must be of the form 'indexVar<loopSize', was " + fs.Condition);

            // Check increment is valid
            fs.Increment = ifs.Increment;
            bool validIncrement = false;
            if (fs.Increment is IExpressionStatement ies)
            {
                if (ies.Expression is IAssignExpression iae)
                {
                    validIncrement = (iae.Expression is IBinaryExpression ibe) && (ibe.Operator == BinaryOperator.Add);
                }
                else if (ies.Expression is IUnaryExpression iue)
                {
                    validIncrement = (iue.Operator == UnaryOperator.PostIncrement);
                }
            }
            if (!validIncrement)
            {
                Error("For statement increment must be of the form 'varname++' or 'varname=varname+1', was " + fs.Increment + ".");
            }


            // Check initializer is valid
            fs.Initializer = ifs.Initializer;
            if (fs.Initializer is IExpressionStatement ies2)
            {
                if (ies2.Expression is IAssignExpression iae2)
                {
                    if (!(iae2.Target is IVariableDeclarationExpression)) Error("For statement initializer must be a variable declaration, was " + iae2.Target.GetType().Name);
                    if (!Recognizer.IsLiteral(iae2.Expression, 0)) Error("Loop index must start at 0, was " + iae2.Expression);
                }
                else
                {
                    Error("For statement initializer must be an assignment, was " + fs.Initializer.GetType());
                }
            }
            else
            {
                Error("For statement initializer must be an expression statement, was " + fs.Initializer.GetType());
            }

            fs.Body = ConvertBlock(ifs.Body);
            if (ReferenceEquals(fs.Body, ifs.Body)) return ifs;
            return fs;
        }

        protected override IStatement DoConvertStatement(IStatement ist)
        {
            if ((ist is IForStatement) || (ist is IExpressionStatement) || (ist is IBlockStatement) || (ist is IConditionStatement))
            {
                return base.DoConvertStatement(ist);
            }
            Error("Unsupported statement type: " + ist.GetType().Name);
            return ist;
        }

        private VariableToChannelInformation DeclareUsesArray(IList<IStatement> stmts, IVariableDeclaration ivd, VariableInformation vi, int useCount, int usageDepth)
        {
            // Create AnyIndex expressions up to usageDepth
            List<IList<IExpression>> prefixSizes = new List<IList<IExpression>>();
            List<IList<IExpression>> prefixVars = new List<IList<IExpression>>();
            vi.DefineAllIndexVars(context);
            for (int d = 0; d < usageDepth; d++)
            {
                IList<IExpression> sizeBracket = Builder.ExprCollection();
                IList<IExpression> varBracket = Builder.ExprCollection();
                for (int i = 0; i < vi.sizes[d].Length; i++)
                {
                    sizeBracket.Add(Builder.StaticMethod(new Func<int>(GateAnalysisTransform.AnyIndex)));
                    varBracket.Add(Builder.VarRefExpr(vi.indexVars[d][i]));
                }
                prefixSizes.Add(sizeBracket);
                prefixVars.Add(varBracket);
            }
            string prefix = vi.Name;
            if (prefix.EndsWith("_use")) prefix = prefix.Substring(0, prefix.Length - 4);
            string arrayName = VariableInformation.GenerateName(context, prefix + "_uses");
            IVariableDeclaration usesDecl = vi.DeriveArrayVariable(stmts, context, arrayName, Builder.LiteralExpr(useCount), Builder.VarDecl("_ind", typeof(int)),
                                                                   prefixSizes, prefixVars, useLiteralIndices: true);
            context.OutputAttributes.Remove<ChannelInfo>(usesDecl);
            context.OutputAttributes.Add(usesDecl, new DescriptionAttribute($"uses of '{vi.Name}'"));
            ChannelInfo ci = ChannelInfo.UseChannel(vi);
            ci.decl = usesDecl;
            context.OutputAttributes.Set(usesDecl, ci);
            VariableToChannelInformation vtci = new VariableToChannelInformation();
            vtci.usesDecl = usesDecl;
            vtci.usageDepth = usageDepth;
            usesOfVariable[ivd] = vtci;
            return vtci;
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            iae = (IAssignExpression)base.ConvertAssign(iae);
            // array allocation should not be treated as defining the variable
            bool shouldDelete = false;
            if (!(iae.Expression is IArrayCreateExpression))
                AddReplicateStatement(iae.Target, iae.Expression, ref shouldDelete);
            return shouldDelete ? null : iae;
        }

        protected override IExpression ConvertAddressOut(IAddressOutExpression iaoe)
        {
            iaoe = (IAddressOutExpression)base.ConvertAddressOut(iaoe);
            bool shouldDelete = false;
            AddReplicateStatement(iaoe.Expression, null, ref shouldDelete);
            return iaoe;
        }

        /// <summary>
        /// Adds a Replicate statement to the output code, if needed.
        /// </summary>
        /// <param name="target">LHS of assignment</param>
        /// <param name="rhs">RHS of assignment</param>
        /// <param name="shouldDelete">True if the original assignment statement should be deleted, i.e. it can be optimized away.</param>
        protected void AddReplicateStatement(IExpression target, IExpression rhs, ref bool shouldDelete)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(target);
            if (ivd == null) return;
            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
            if (!vi.IsStochastic) return;
            bool isEvidenceVar = context.InputAttributes.Has<DoNotSendEvidence>(ivd);
            if (SwapIndices && replicateEvidenceVars != isEvidenceVar) return;
            // iae defines a stochastic variable
            ChannelAnalysisTransform.UsageInfo usageInfo;
            if (!analysis.usageInfo.TryGetValue(ivd, out usageInfo)) return;
            int useCount = usageInfo.NumberOfUses;
            if (useCount <= 1) return;

            bool firstTime = !usesOfVariable.TryGetValue(ivd, out VariableToChannelInformation vtci);
            int targetDepth = Recognizer.GetIndexingDepth(target);
            int minDepth = usageInfo.indexingDepths[0];
            int usageDepth = minDepth;
            if (!SwapIndices) usageDepth = 0;
            Containers defContainers = context.InputAttributes.Get<Containers>(ivd);
            int ancIndex = defContainers.GetMatchingAncestorIndex(context);
            Containers missing = defContainers.GetContainersNotInContext(context, ancIndex);
            if (firstTime)
            {
                // declaration of uses array
                IList<IStatement> stmts = Builder.StmtCollection();
                vtci = DeclareUsesArray(stmts, ivd, vi, useCount, usageDepth);
                stmts = Containers.WrapWithContainers(stmts, missing.outputs);
                context.AddStatementsBeforeAncestorIndex(ancIndex, stmts);
            }
            // check that extra literal indices in the target are zero.
            // for example, if iae is x[i][0] = (...) then it is safe to add x_uses[i] = Rep(x[i])
            // if iae is x[i][1] = (...) then it is safe to add x_uses[i][1] = Rep(x[i][1]) 
            // but not x_uses[i] = Rep(x[i]) since this will be a duplicate.
            bool extraLiteralsAreZero = CheckExtraLiteralsAreZero(target, targetDepth, usageDepth);
            if (extraLiteralsAreZero)
            {
                // definition of uses array
                IExpression defExpr = Builder.VarRefExpr(ivd);
                IExpression usesExpr = Builder.VarRefExpr(vtci.usesDecl);
                List<IStatement> loops = new List<IStatement>();
                if (usageDepth == targetDepth)
                {
                    defExpr = target;
                    if (defExpr is IVariableDeclarationExpression) defExpr = Builder.VarRefExpr(ivd);
                    usesExpr = Builder.ReplaceVariable(defExpr, ivd, vtci.usesDecl);
                }
                else
                {
                    // loops over the last indexing bracket
                    for (int d = 0; d < usageDepth; d++)
                    {
                        List<IExpression> indices = new List<IExpression>();
                        for (int i = 0; i < vi.sizes[d].Length; i++)
                        {
                            IVariableDeclaration v = vi.indexVars[d][i];
                            IStatement loop = Builder.ForStmt(v, vi.sizes[d][i]);
                            loops.Add(loop);
                            indices.Add(Builder.VarRefExpr(v));
                        }
                        defExpr = Builder.ArrayIndex(defExpr, indices);
                        usesExpr = Builder.ArrayIndex(usesExpr, indices);
                    }
                }

                if (rhs != null && rhs is IMethodInvokeExpression imie)
                {
                    bool copyPropagation = false;
                    if (Recognizer.IsStaticGenericMethod(imie, new Func<PlaceHolder, PlaceHolder>(Clone.Copy)) && copyPropagation)
                    {
                        // if a variable is a copy, use the original expression since it will give more precise dependencies.
                        defExpr = imie.Arguments[0];
                        shouldDelete = true;
                    }
                }

                // Add the statement:
                //   x_uses = Replicate(x, useCount)
                var genArgs = new Type[] { ivd.VariableType.DotNetType };
                IMethodInvokeExpression repMethod = Builder.StaticGenericMethod(
                    new Func<PlaceHolder, int, PlaceHolder[]>(Clone.Replicate),
                    genArgs, defExpr, Builder.LiteralExpr(useCount));
                bool isGateExitRandom = context.InputAttributes.Has<Algorithms.VariationalMessagePassing.GateExitRandomVariable>(ivd);
                if (isGateExitRandom)
                {
                    repMethod = Builder.StaticGenericMethod(
                        new Func<PlaceHolder, int, PlaceHolder[]>(Gate.ReplicateExiting),
                        genArgs, defExpr, Builder.LiteralExpr(useCount));
                }
                context.InputAttributes.CopyObjectAttributesTo<Algorithm>(ivd, context.OutputAttributes, repMethod);
                if (context.InputAttributes.Has<DivideMessages>(ivd))
                {
                    context.InputAttributes.CopyObjectAttributesTo<DivideMessages>(ivd, context.OutputAttributes, repMethod);
                }
                else if (useCount == 2)
                {
                    // division has no benefit for 2 uses, and degrades the schedule
                    context.OutputAttributes.Set(repMethod, new DivideMessages(false));
                }
                context.InputAttributes.CopyObjectAttributesTo<GivePriorityTo>(ivd, context.OutputAttributes, repMethod);
                IStatement repSt = Builder.AssignStmt(usesExpr, repMethod);
                if (usageDepth == targetDepth)
                {
                    if (isEvidenceVar)
                    {
                        // place the Replicate after the current statement, but outside of any evidence conditionals
                        ancIndex = context.FindAncestorIndex<IStatement>();
                        for (int i = ancIndex - 2; i > 0; i--)
                        {
                            object ancestor = context.GetAncestor(i);
                            if (ancestor is IConditionStatement ics)
                            {
                                if (CodeRecognizer.IsStochastic(context, ics.Condition))
                                {
                                    ancIndex = i;
                                    break;
                                }
                            }
                        }
                        context.AddStatementAfterAncestorIndex(ancIndex, repSt);
                    }
                    else
                    {
                        context.AddStatementAfterCurrent(repSt);
                    }
                }
                else if (firstTime)
                {
                    // place Replicate after assignment but outside of definition loops (can't have any uses there)
                    repSt = Containers.WrapWithContainers(repSt, loops);
                    repSt = Containers.WrapWithContainers(repSt, missing.outputs);
                    context.AddStatementAfterAncestorIndex(ancIndex, repSt);
                }
            }
        }

        /// <summary>
        /// Determine if all literal indices beyond the usageDepth are zero
        /// </summary>
        /// <param name="target"></param>
        /// <param name="targetDepth"></param>
        /// <param name="usageDepth"></param>
        /// <returns></returns>
        private static bool CheckExtraLiteralsAreZero(IExpression target, int targetDepth, int usageDepth)
        {
            bool extraLiteralsAreZero = true;
            int depth = targetDepth;
            while (depth > usageDepth)
            {
                IArrayIndexerExpression iaie = (IArrayIndexerExpression)target;
                foreach (IExpression index in iaie.Indices)
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
                target = iaie.Target;
                depth--;
            }
            return extraLiteralsAreZero;
        }

        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            context.InputAttributes.Remove<Containers>(ivde.Variable);
            context.InputAttributes.Set(ivde.Variable, new Containers(context));
            return ivde;
        }

        /// <summary>
        /// Converts a variable reference.
        /// </summary>
        /// <param name="ivre"></param>
        /// <returns></returns>
        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            if (Recognizer.IsBeingMutated(context, ivre)) return ivre;
            else if (Recognizer.IsBeingIndexed(context)) return ivre;
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(ivre);
            if (ivd == null) return ivre;
            if (usesOfVariable.TryGetValue(ivd, out VariableToChannelInformation vtci))
            {
                if (vtci.usageDepth != 0) Error("wrong usageDepth (" + vtci.usageDepth + " instead of 0)");
                return Builder.ArrayIndex(Builder.VarRefExpr(vtci.usesDecl), Builder.LiteralExpr(GetUseNumber(ivd, vtci)));
            }
            return ivre;
        }

        private int GetUseNumber(IVariableDeclaration ivd, VariableToChannelInformation vtci)
        {
            var info = analysis.usageInfo[ivd];
            IStatement st = context.FindAncestorNotSelf<IStatement>();
            int useNumber;
            if (info.useNumberOfStatement.TryGetValue(st, out Queue<int> queue))
            {
                useNumber = queue.Dequeue();
            }
            else useNumber = vtci.useCount++;
            return useNumber;
        }

        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            // convert the indices first
            IExpression expr = base.ConvertArrayIndexer(iaie);
            // Check if this is the top level indexer
            if (Recognizer.IsBeingIndexed(context)) return expr;
            if (Recognizer.IsBeingMutated(context, iaie)) return expr;
            IExpression target;
            List<IList<IExpression>> indices = Recognizer.GetIndices(expr, out target);
            if (!(target is IVariableReferenceExpression)) return expr;
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(target);
            if (ivd == null) return expr;
            if (usesOfVariable.TryGetValue(ivd, out VariableToChannelInformation vtci))
            {
                IExpression usageIndex = Builder.LiteralExpr(GetUseNumber(ivd, vtci));
                IExpression newExpr = Builder.VarRefExpr(vtci.usesDecl);
                if (vtci.usageDepth > indices.Count) Error("usageDepth (" + vtci.usageDepth + ") > indices.Count (" + indices.Count + ")");
                // append the indices up to the usageDepth
                for (int i = 0; i < vtci.usageDepth; i++)
                {
                    if (indices.Count <= i) return newExpr;
                    newExpr = Builder.ArrayIndex(newExpr, indices[i]);
                }
                newExpr = Builder.ArrayIndex(newExpr, usageIndex);
                // append the remaining indices
                for (int i = vtci.usageDepth; i < indices.Count; i++)
                {
                    newExpr = Builder.ArrayIndex(newExpr, indices[i]);
                }
                return newExpr;
            }
            return expr;
        }

        /// <summary>
        /// Records information about the variable to channel transformation.
        /// </summary>
        private class VariableToChannelInformation
        {
            public IVariableDeclaration usesDecl;
            public int usageDepth;

            /// <summary>
            /// A running total of the number of uses transformed so far.
            /// </summary>
            public int useCount;
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            if (CodeRecognizer.IsInfer(imie)) return imie;
            CheckForDuplicateArguments(imie);
            return base.ConvertMethodInvoke(imie);
        }

        private void CheckForDuplicateArguments(IMethodInvokeExpression mie)
        {
            Set<IExpression> args = new Set<IExpression>();
            for (int i = 0; i < mie.Arguments.Count; i++)
            {
                IExpression arg = mie.Arguments[i];
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(arg);
                if (ivd != null)
                {
                    VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
                    if (vi.IsStochastic)
                    {
                        if (args.Contains(arg))
                        {
                            Error("Variable '" + vi.Name + "' appears twice in the same function call.  This will lead to inaccurate inference.  Try reformulating the model.");
                        }
                        args.Add(arg);
                    }
                }
            }
        }
    }
}