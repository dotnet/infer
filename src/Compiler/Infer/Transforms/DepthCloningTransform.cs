// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    using IndexInfo = DepthAnalysisTransform.IndexInfo;
    using DepthInfo = DepthAnalysisTransform.DepthInfo;

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Clones variables to ensure that each is used at a fixed depth
    /// </summary>
    /// <remarks>
    /// More documentation can be found on the sharepoint.
    /// </remarks>
    internal class DepthCloningTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "DepthCloningTransform"; }
        }

        private readonly bool cloneEvidenceVars;

        public DepthCloningTransform(bool cloneEvidenceVars)
        {
            this.cloneEvidenceVars = cloneEvidenceVars;
        }

        private DepthAnalysisTransform analysis;

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            analysis = new DepthAnalysisTransform();
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

        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            bool isDef = Recognizer.IsBeingMutated(context, ivre);
            if (isDef) return ivre;
            return GetClone(ivre);
        }

        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            bool isDef = Recognizer.IsBeingMutated(context, iaie);
            iaie = (IArrayIndexerExpression) base.ConvertArrayIndexer(iaie);
            if (isDef) return iaie;
            return GetClone(iaie);
        }

        private IExpression GetClone(IExpression expr)
        {
            // Determine if this is a definition i.e. the variable is on the left hand side of an assignment
            // This must be done before base.ConvertArrayIndexer changes the expression!
            if (Recognizer.IsBeingIndexed(context)) return expr;
            IVariableDeclaration baseVar = Recognizer.GetVariableDeclaration(expr);
            // If not an indexed variable reference, skip it (e.g. an indexed argument reference)
            if (baseVar == null) return expr;
            DepthInfo depthInfo;
            if (!analysis.depthInfos.TryGetValue(baseVar, out depthInfo)) return expr;
            if (depthInfo.useCount <= 1 || depthInfo.indexInfoOfDepth.Count == 1) return expr;
            bool isEvidenceVar = context.InputAttributes.Has<DoNotSendEvidence>(baseVar);
            if (isEvidenceVar && !cloneEvidenceVars) return expr;
            int depth = Recognizer.GetIndexingDepth(expr);
            IndexInfo info = depthInfo.indexInfoOfDepth[depth];
            if (info.clone == null)
            {
                if (depth == depthInfo.definitionDepth) return expr;
                VariableInformation varInfo = VariableInformation.GetVariableInformation(context, baseVar);
                string name = baseVar.Name + "_depth" + depth;
                IList<IStatement> stmts = Builder.StmtCollection();

                // declare the clone
                varInfo.DefineAllIndexVars(context);
                IVariableDeclaration newvd = varInfo.DeriveIndexedVariable(stmts, context, name);
                VariableInformation newVariableInformation = VariableInformation.GetVariableInformation(context, newvd);
                newVariableInformation.LiteralIndexingDepth = info.literalIndexingDepth;
                if (!context.InputAttributes.Has<DerivedVariable>(newvd))
                    context.InputAttributes.Set(newvd, new DerivedVariable());
                if (context.InputAttributes.Has<ChannelInfo>(baseVar))
                {
                    ChannelInfo ci = ChannelInfo.UseChannel(varInfo);
                    ci.decl = newvd;
                    context.OutputAttributes.Set(newvd, ci);
                }

                // define the clone
                // if depth < definitionDepth, index by the definitionDepth
                // else index by depth
                // e.g.
                // x[i] = definition
                // x_depth0[i] = Copy(x[i])
                // x_depth2[i][j] = Copy(x[i][j])
                int indexingDepth = System.Math.Max(depth, depthInfo.definitionDepth);
                IExpression lhs = Builder.VarRefExpr(newvd);
                // TODO: clone the next lower depth, not always the baseVar
                IExpression rhs = Builder.VarRefExpr(baseVar);
                AddCopyStatements(stmts, newVariableInformation, indexingDepth, lhs, rhs);

                // Reduce memory consumption by declaring the clone outside of unnecessary loops.
                // This way, the item is cloned outside the loop and then replicated, instead of replicating the entire array and cloning the item.
                Containers containers = info.containers;
                containers = Containers.RemoveUnusedLoops(containers, context, Builder.VarRefExpr(baseVar));
                if (context.InputAttributes.Has<DoNotSendEvidence>(baseVar)) containers = Containers.RemoveStochasticConditionals(containers, context);
                // To put the declaration in the desired containers, we find an ancestor which includes as many of the containers as possible,
                // then wrap the declaration with the remaining containers.
                int ancIndex = containers.GetMatchingAncestorIndex(context);
                Containers missing = containers.GetContainersNotInContext(context, ancIndex);
                stmts = Containers.WrapWithContainers(stmts, missing.outputs);
                context.AddStatementsBeforeAncestorIndex(ancIndex, stmts);
                context.InputAttributes.Set(newvd, containers);
                info.clone = Builder.VarRefExpr(newvd);
            }
            List<IList<IExpression>> indices = Recognizer.GetIndices(expr);
            return Builder.JaggedArrayIndex(info.clone, indices);
        }

        private static void AddCopyStatements(ICollection<IStatement> stmts, VariableInformation varInfo, int indexingDepth, IExpression lhs, IExpression rhs, 
            int bracket = 0, Dictionary<IExpression,IExpression> replacements = null)
        {
            if (indexingDepth == bracket)
            {
                Type tp = rhs.GetExpressionType();
                IExpression copy = Builder.StaticGenericMethod(new Func<PlaceHolder, PlaceHolder>(Clone.Copy<PlaceHolder>), new Type[] { tp }, rhs);
                IStatement copySt = Builder.AssignStmt(lhs, copy);
                stmts.Add(copySt);
            }
            else if (varInfo.LiteralIndexingDepth == bracket+1 && varInfo.sizes[bracket].All(size => size is ILiteralExpression))
            {
                if (replacements == null)
                    replacements = new Dictionary<IExpression, IExpression>();
                int[] sizes = Util.ArrayInit(varInfo.sizes[bracket].Length, i => (int)((ILiteralExpression)varInfo.sizes[bracket][i]).Value);
                ForEachLiteralIndex(sizes, index =>
                {
                    IExpression[] bracketIndices = Util.ArrayInit(index.Length, i => Builder.LiteralExpr(index[i]));
                    var newLhs = Builder.ArrayIndex(lhs, bracketIndices);
                    var newRhs = Builder.ArrayIndex(rhs, bracketIndices);
                    for (int dim = 0; dim < index.Length; dim++)
                    {
                        var indexVarRef = Builder.VarRefExpr(varInfo.indexVars[bracket][dim]);
                        replacements[indexVarRef] = bracketIndices[dim];
                    }
                    AddCopyStatements(stmts, varInfo, indexingDepth, newLhs, newRhs, bracket + 1, replacements);
                });
            }
            else
            {
                IReadOnlyList<IExpression> replacedSizes = varInfo.sizes[bracket];
                if(replacements != null)
                {
                    replacedSizes = Util.ArrayInit(replacedSizes.Count, i => Replace(replacedSizes[i], replacements));
                }
                IForStatement innerForStatement;
                var fs = Builder.NestedForStmt(varInfo.indexVars[bracket], replacedSizes, out innerForStatement);
                IExpression[] bracketIndices = Util.ArrayInit(varInfo.indexVars[bracket].Length, i =>
                    Builder.VarRefExpr(varInfo.indexVars[bracket][i])
                );
                var newLhs = Builder.ArrayIndex(lhs, bracketIndices);
                var newRhs = Builder.ArrayIndex(rhs, bracketIndices);
                AddCopyStatements(innerForStatement.Body.Statements, varInfo, indexingDepth, newLhs, newRhs, bracket + 1, replacements);
                stmts.Add(fs);
            }
        }

        private static IExpression Replace(IExpression expr, IReadOnlyDictionary<IExpression,IExpression> replacements)
        {
            foreach(var entry in replacements)
            {
                expr = Builder.ReplaceExpression(expr, entry.Key, entry.Value);
            }
            return expr;
        }

        private static void ForEachLiteralIndex(int[] sizes, Action<int[]> action)
        {
            int[] strides = StringUtil.ArrayStrides(sizes);
            int arrayLength = strides[0] * sizes[0];
            int[] mIndex = new int[sizes.Length];
            for (int i = 0; i < arrayLength; i++)
            {
                StringUtil.LinearIndexToMultidimensionalIndex(i, strides, mIndex);
                action(mIndex);
            }
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            // do not count argument of Infer as a use
            if (CodeRecognizer.IsInfer(imie)) return imie;
            return base.ConvertMethodInvoke(imie);
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}