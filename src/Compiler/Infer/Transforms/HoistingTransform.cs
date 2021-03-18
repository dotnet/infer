// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Factors;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    using HoistingInfo = HoistingAnalysisTransform.HoistingInfo;

    /// <summary>
    /// Creates copies of arrays with irrelevant dimensions removed.
    /// Requires LoopCuttingTransform to have been applied to the input.
    /// The transform has 4 parts:
    /// 1. Analysis to find irrelevant dimensions.
    /// 2. Scaffolding.  For each array with irrelevant dimensions, make a new array by copying all statements that assign to the old array, keeping all loops for now.
    /// 3. Replacement.  Replace reads of the old array, and replace writes of the old array with assignments to the new array.  
    ///    These writes will be deleted as dead code later.
    /// 4. Remove loops over the irrelevant dimensions.  Done by LoopRemovalTransform.
    /// This transformation is always beneficial therefore it doesn't need a profitability heuristic.
    /// It can only remove dependence edges.
    /// Furthermore, this transformation completely summarizes its analysis, so there is no need to retain the analysis afterward.
    /// </summary>
    internal class HoistingTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "HoistingTransform"; }
        }

        readonly ModelCompiler compiler;
        HoistingAnalysisTransform analysis;
        public static bool debug;

        internal HoistingTransform(ModelCompiler compiler)
        {
            this.compiler = compiler;
        }

        private void CreateReducedVariableInformation(IVariableDeclaration ivd, HoistingInfo info)
        {
            int arrayDepth = info.maxDepthWhereDimensionCouldMatter.Length;
            var varInfo = VariableInformation.GetVariableInformation(context, ivd);
            var reducedVarInfo = VariableInformation.GetVariableInformation(context, info.newVariable);
            for (int bracket = 0; bracket < varInfo.sizes.Count; bracket++)
            {
                List<IExpression> newSizes = new List<IExpression>();
                List<IVariableDeclaration> newIndexVars = new List<IVariableDeclaration>();
                for (int dim = 0; dim < varInfo.sizes[bracket].Length; dim++)
                {
                    if (info.maxDepthWhereDimensionCouldMatter[bracket][dim] == arrayDepth)
                    {
                        newSizes.Add(varInfo.sizes[bracket][dim]);
                        if (varInfo.indexVars.Count > bracket && varInfo.indexVars[bracket].Length > dim)
                            newIndexVars.Add(varInfo.indexVars[bracket][dim]);
                    }
                }
                if (newSizes.Count > 0)
                {
                    reducedVarInfo.sizes.Add(newSizes.ToArray());
                    reducedVarInfo.indexVars.Add(newIndexVars.ToArray());
                }
            }
        }

        private void PromoteDimensions(HoistingInfo info)
        {
            int arrayDepth = info.maxDepthWhereDimensionCouldMatter.Length;
            bool previousBracketMattersAtArrayDepth = false;
            for (int bracket = arrayDepth - 1; bracket >= 0; bracket--)
            {
                bool bracketMattersAtArrayDepth = false;
                for (int dim = 0; dim < info.maxDepthWhereDimensionCouldMatter[bracket].Length; dim++)
                {
                    int depth = info.maxDepthWhereDimensionCouldMatter[bracket][dim];
                    if (depth == arrayDepth) bracketMattersAtArrayDepth = true;
                    else if (depth > 0 && previousBracketMattersAtArrayDepth)
                    {
                        info.maxDepthWhereDimensionCouldMatter[bracket][dim] = arrayDepth;
                        bracketMattersAtArrayDepth = true;
                    }
                }
                previousBracketMattersAtArrayDepth = bracketMattersAtArrayDepth;
            }
        }

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            analysis = new HoistingAnalysisTransform();
            Stopwatch watch = null;
            if (compiler.ShowProgress)
            {
                Console.Write($"({analysis.Name} ");
                watch = Stopwatch.StartNew();
            }
            analysis.Context.InputAttributes = context.InputAttributes;
            analysis.Transform(itd);
            if (compiler.ShowProgress)
            {
                watch.Stop();
                Console.Write("{0}ms) ", watch.ElapsedMilliseconds);
            }
            context.Results = analysis.Context.Results;
            if (!context.Results.IsSuccess)
            {
                Error("analysis failed");
                return itd;
            }
            // create new variable names
            foreach (var entry in analysis.infoOfVariable)
            {
                var ivd = entry.Key;
                var info = entry.Value;
                PromoteDimensions(info);
                int arrayDepth = info.maxDepthWhereDimensionCouldMatter.Length;
                bool hasUnusedIndex = info.maxDepthWhereDimensionCouldMatter.Any(bracket => bracket.Any(depth => depth < arrayDepth));
                if (hasUnusedIndex)
                {
                    info.newVariable = Builder.VarDecl(ivd.Name + "_reduced", GetReducedType(ivd.VariableType.DotNetType, info));
                    CreateReducedVariableInformation(ivd, info);
                }
            }
            var itdOut = base.Transform(itd);
            if (context.trackTransform && debug)
            {
                IBlockStatement block = Builder.BlockStmt();
                foreach (var entry in analysis.infoOfVariable)
                {
                    var ivd = entry.Key;
                    var info = entry.Value;
                    int arrayDepth = info.maxDepthWhereDimensionCouldMatter.Length;
                    for (int bracket = 0; bracket < info.maxDepthWhereDimensionCouldMatter.Length; bracket++)
                    {
                        for (int dim = 0; dim < info.maxDepthWhereDimensionCouldMatter[bracket].Length; dim++)
                        {
                            int depth = info.maxDepthWhereDimensionCouldMatter[bracket][dim];
                            if (depth != arrayDepth)
                            {
                                var varInfo = VariableInformation.GetVariableInformation(context, ivd);
                                block.Statements.Add(Builder.CommentStmt($"{ivd.Name} index {varInfo.indexVars[bracket][dim].Name} only matters at depth {depth}"));
                            }
                        }
                    }
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

        private Type GetReducedType(Type type, HoistingInfo info)
        {
            int arrayDepth = info.maxDepthWhereDimensionCouldMatter.Length;
            Stack<Type> stack = new Stack<Type>();
            for (int bracket = 0; bracket < arrayDepth; bracket++)
            {
                stack.Push(type);
                type = Util.GetElementType(type);
            }
            Type reducedType = type;
            for (int bracket = arrayDepth - 1; bracket >= 0; bracket--)
            {
                Type originalType = stack.Pop();
                int reducedRank = info.maxDepthWhereDimensionCouldMatter[bracket].Count(d => d == arrayDepth);
                if (reducedRank > 0)
                {
                    reducedType = Util.ChangeElementTypeAndRank(originalType, reducedType, reducedRank);
                }
            }
            return reducedType;
        }

        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            HoistingInfo info;
            if (analysis.infoOfVariable.TryGetValue(ivde.Variable, out info) && info.newVariable != null)
            {
                IStatement newDeclStmt = Builder.ExprStatement(ivde);
                IStatement declStmt = context.FindAncestor<IStatement>();
                context.InputAttributes.CopyObjectAttributesTo(declStmt, context.OutputAttributes, newDeclStmt);
                context.AddStatementBeforeCurrent(newDeclStmt);
                return Builder.VarDeclExpr(info.newVariable);
            }
            else return base.ConvertVariableDeclExpr(ivde);
        }

        private bool IsReducibleAtDepth(HoistingInfo info, int depth)
        {
            int arrayDepth = info.maxDepthWhereDimensionCouldMatter.Length;
            for (int bracket = depth; bracket < info.maxDepthWhereDimensionCouldMatter.Length; bracket++)
            {
                for (int dim = 0; dim < info.maxDepthWhereDimensionCouldMatter[bracket].Length; dim++)
                {
                    if (info.maxDepthWhereDimensionCouldMatter[bracket][dim] != arrayDepth)
                        return false;
                }
            }
            return true;
        }

        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            IExpression target;
            var indices = Recognizer.GetIndices(iaie, out target);
            if (target is IVariableReferenceExpression)
            {
                int indexingDepth = indices.Count;
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(target);
                HoistingInfo info;
                if (analysis.infoOfVariable.TryGetValue(ivd, out info) && info.newVariable != null
                    && IsReducibleAtDepth(info, indexingDepth))
                {
                    int arrayDepth = info.maxDepthWhereDimensionCouldMatter.Length;
                    IExpression newExpr = Builder.VarRefExpr(info.newVariable);
                    for (int bracket = 0; bracket < indexingDepth; bracket++)
                    {
                        if (info.maxDepthWhereDimensionCouldMatter[bracket].Any(d => d == arrayDepth))
                        {
                            List<IExpression> newIndices = new List<IExpression>();
                            for (int dim = 0; dim < indices[bracket].Count; dim++)
                            {
                                if (info.maxDepthWhereDimensionCouldMatter[bracket][dim] == arrayDepth)
                                {
                                    newIndices.Add(indices[bracket][dim]);
                                }
                            }
                            newExpr = Builder.ArrayIndex(newExpr, newIndices);
                        }
                    }
                    return newExpr;
                }
                // fall through
            }
            return base.ConvertArrayIndexer(iaie);
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
            HoistingInfo info;
            if (ivd != null && analysis.infoOfVariable.TryGetValue(ivd, out info) && info.newVariable != null)
            {
                int indexingDepth = Recognizer.GetIndexingDepth(iae.Target);
                bool mustConvertRhs = false;
                int newRank = 0;
                int arrayDepth = info.maxDepthWhereDimensionCouldMatter.Length;
                for (int bracket = indexingDepth; bracket < info.maxDepthWhereDimensionCouldMatter.Length; bracket++)
                {
                    for (int dim = 0; dim < info.maxDepthWhereDimensionCouldMatter[bracket].Length; dim++)
                    {
                        if (info.maxDepthWhereDimensionCouldMatter[bracket][dim] != arrayDepth)
                        {
                            mustConvertRhs = true;
                        }
                        else if (bracket == indexingDepth) newRank++;
                    }
                }
                IExpression newTarget;
                if (iae.Target is IVariableReferenceExpression)
                {
                    newTarget = Builder.VarRefExpr(info.newVariable);
                }
                else if (iae.Target is IVariableDeclarationExpression)
                {
                    newTarget = Builder.VarDeclExpr(info.newVariable);
                }
                else if (iae.Target is IArrayIndexerExpression)
                {
                    IExpression target;
                    var indices = Recognizer.GetIndices(iae.Target, out target);
                    newTarget = Builder.VarRefExpr(info.newVariable);
                    for (int bracket = 0; bracket < indexingDepth; bracket++)
                    {
                        if (info.maxDepthWhereDimensionCouldMatter[bracket].Any(d => d == arrayDepth))
                        {
                            List<IExpression> newIndices = new List<IExpression>();
                            for (int dim = 0; dim < indices[bracket].Count; dim++)
                            {
                                if (info.maxDepthWhereDimensionCouldMatter[bracket][dim] == arrayDepth)
                                {
                                    newIndices.Add(indices[bracket][dim]);
                                }
                            }
                            newTarget = Builder.ArrayIndex(newTarget, newIndices);
                        }
                    }
                }
                else throw new NotSupportedException();
                if (mustConvertRhs)
                {
                    if (HoistingAnalysisTransform.IsReducibleArrayCreateExpression(iae.Expression))
                    {
                        IStatement assignStmt = context.FindAncestor<IStatement>();
                        context.AddStatementAfterCurrent(assignStmt);
                        Type type = info.newVariable.VariableType.DotNetType;
                        int newDepth = Recognizer.GetIndexingDepth(newTarget);
                        for (int bracket = 0; bracket < newDepth; bracket++)
                        {
                            type = Util.GetElementType(type);
                        }
                        IExpression newRhs;
                        if (newRank == 0)
                        {
                            newRhs = Builder.DefaultExpr(type);
                            if (newDepth == 0)
                            {
                                // If we have already initialized newTarget at the top level, there is no need to initialize again.
                                if (info.isInitialized) return null;
                                else if (Containers.FindContainers(context).Count == 0) info.isInitialized = true;
                            }
                        }
                        else if (iae.Expression is IArrayCreateExpression iace)
                        {
                            Type elementType = Util.GetElementType(type);
                            newRhs = Builder.ArrayCreateExpr(elementType, GetReducedDimensions(iace.Dimensions, info.maxDepthWhereDimensionCouldMatter[indexingDepth], arrayDepth));
                        }
                        else if (iae.Expression is IObjectCreateExpression ioce)
                        {
                            newRhs = Builder.NewObject(type, GetReducedDimensions(ioce.Arguments, info.maxDepthWhereDimensionCouldMatter[indexingDepth], arrayDepth));
                        }
                        else throw new InferCompilerException($"Unexpected expression type: {iae.Expression.GetType()}");
                        return Builder.AssignExpr(newTarget, newRhs);
                    }
                    else throw new NotSupportedException();
                }
                else
                {
                    IStatement copyStmt = Builder.AssignStmt(iae.Target, newTarget);
                    IStatement assignStmt = context.FindAncestor<IStatement>();
                    context.InputAttributes.CopyObjectAttributesTo(assignStmt, context.OutputAttributes, copyStmt);
                    context.AddStatementAfterCurrent(copyStmt);
                }
            }
            return base.ConvertAssign(iae);
        }

        private List<IExpression> GetReducedDimensions(IList<IExpression> dimensions, int[] maxDepthWhereDimensionCouldMatter, int arrayDepth)
        {
            List<IExpression> reducedDimensions = new List<IExpression>();
            for (int dim = 0; dim < dimensions.Count; dim++)
            {
                if (maxDepthWhereDimensionCouldMatter[dim] == arrayDepth)
                    reducedDimensions.Add(dimensions[dim]);
            }
            return reducedDimensions;
        }
    }

    /// <summary>
    /// Performs a dataflow analysis to determine which array dimensions matter.
    /// </summary>
    internal class HoistingAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "HoistingAnalysisTransform"; }
        }

        public class HoistingInfo
        {
            // The size of an inner dimension can depend on an outer dimension, which means it's possible for [i][j] to not depend on (i,j) but [i] depends on i.
            // However, if [i] [j] depends on i then [i] must depend on i.
            // Therefore, for each dimension, we store the largest depth where it matters, since all smaller depths are implied.
            public int[][] maxDepthWhereDimensionCouldMatter;
            public IVariableDeclaration newVariable;
            public bool isInitialized;

            public HoistingInfo(VariableInformation variableInformation)
            {
                maxDepthWhereDimensionCouldMatter = Util.ArrayInit(variableInformation.sizes.Count, bracket =>
                    Util.ArrayInit(variableInformation.sizes[bracket].Length, dim => 0));
            }

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder();
                if (newVariable != null)
                    sb.Append(newVariable.Name);
                for (int bracket = 0; bracket < maxDepthWhereDimensionCouldMatter.Length; bracket++)
                {
                    sb.Append("[");
                    for (int dim = 0; dim < maxDepthWhereDimensionCouldMatter[bracket].Length; dim++)
                    {
                        if (dim > 0) sb.Append(",");
                        sb.Append(maxDepthWhereDimensionCouldMatter[bracket][dim]);
                    }
                    sb.Append("]");
                }
                return sb.ToString();
            }
        }

        public Dictionary<IVariableDeclaration, HoistingInfo> infoOfVariable = new Dictionary<IVariableDeclaration, HoistingInfo>();
        Set<IVariableDeclaration> variablesInExpression = new Set<IVariableDeclaration>();
        Set<IVariableDeclaration> unbrokenLoopVars = new Set<IVariableDeclaration>();
        bool anyChanged;

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            // We must loop to convergence because there may be cyclic dependencies.
            // In the future, MessageTransform should put OperatorStatements into a while(true) loop instead of marking them.
            // Then we iterate around while loops as normal.
            // We could detect convergence faster by only checking for changes to back edge sources.
            // To reduce the number of iterations, we could reduce the number of back edge sources, by performing this analysis after scheduling.  Perhaps before LAT.
            // Hoisting has no effect on scheduling anyway.  But it does affect FBT.
            anyChanged = true;
            while (anyChanged)
            {
                anyChanged = false;
                base.DoConvertMethodBody(outputs, inputs);
            }
        }

        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            IExpression target;
            var indices = Recognizer.GetIndices(iaie, out target);
            if (target is IVariableReferenceExpression)
            {
                // Only convert indices that could matter.
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(target);
                HoistingInfo info = GetOrCreateHoistingInfo(ivd);
                int indexingDepth = indices.Count;
                for (int bracket = 0; bracket < indexingDepth; bracket++)
                {
                    for (int dim = 0; dim < indices[bracket].Count; dim++)
                    {
                        if (info.maxDepthWhereDimensionCouldMatter.Length <= bracket ||
                            info.maxDepthWhereDimensionCouldMatter[bracket][dim] >= indexingDepth)
                        {
                            ConvertExpression(indices[bracket][dim]);
                        }
                    }
                }
                return iaie;
            }
            else
            {
                return base.ConvertArrayIndexer(iaie);
            }
        }

        private HoistingInfo GetOrCreateHoistingInfo(IVariableDeclaration ivd)
        {
            HoistingInfo info;
            if (!infoOfVariable.TryGetValue(ivd, out info))
            {
                info = new HoistingInfo(VariableInformation.GetVariableInformation(context, ivd));
                infoOfVariable.Add(ivd, info);
            }
            return info;
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            if (imie.Arguments.Any(arg => arg is IAddressOutExpression))
            {
                Set<IVariableDeclaration> oldVariablesInExpression = (Set<IVariableDeclaration>)variablesInExpression.Clone();
                variablesInExpression.Clear();
                ConvertExpression(imie.Method);
                foreach (var arg in imie.Arguments)
                {
                    if (!(arg is IAddressOutExpression))
                        ConvertExpression(arg);
                }
                foreach (var arg in imie.Arguments)
                {
                    if (arg is IAddressOutExpression)
                    {
                        IAddressOutExpression iaoe = (IAddressOutExpression)arg;
                        IExpression target;
                        var indices = Recognizer.GetIndices(iaoe.Expression, out target);
                        if (target is IVariableReferenceExpression || target is IVariableDeclarationExpression)
                        {
                            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(target);
                            HoistingInfo info = GetOrCreateHoistingInfo(ivd);
                            int arrayDepth = info.maxDepthWhereDimensionCouldMatter.Length;
                            for (int bracket = 0; bracket < indices.Count; bracket++)
                            {
                                for (int dim = 0; dim < indices[bracket].Count; dim++)
                                {
                                    var index = indices[bracket][dim];
                                    if (index is IVariableReferenceExpression)
                                    {
                                        IVariableDeclaration indexVar = Recognizer.GetVariableDeclaration(index);
                                        if (variablesInExpression.Contains(indexVar) || !unbrokenLoopVars.Contains(indexVar))
                                        {
                                            // index is used on rhs or the loop is broken, so the dimension matters.
                                            SetDimensionMattersAtDepth(info, bracket, dim, arrayDepth);
                                        }
                                    }
                                    else
                                    {
                                        // index is not a loop counter, so the dimension matters.
                                        SetDimensionMattersAtDepth(info, bracket, dim, arrayDepth);
                                    }
                                }
                            }
                            // all deeper dimensions matter
                            for (int bracket = indices.Count; bracket < arrayDepth; bracket++)
                            {
                                for (int dim = 0; dim < info.maxDepthWhereDimensionCouldMatter[bracket].Length; dim++)
                                {
                                    SetDimensionMattersAtDepth(info, bracket, dim, arrayDepth);
                                }
                            }
                        }
                    }
                }
                variablesInExpression.AddRange(oldVariablesInExpression);
                return imie;
            }
            else
                return base.ConvertMethodInvoke(imie);
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            IExpression target;
            var indices = Recognizer.GetIndices(iae.Target, out target);
            if (target is IVariableReferenceExpression || target is IVariableDeclarationExpression)
            {
                variablesInExpression.Clear();
                ConvertExpression(iae.Expression);
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(target);
                HoistingInfo info = GetOrCreateHoistingInfo(ivd);
                int depth = indices.Count;
                bool rhsIsReducible = IsReducibleArrayCreateExpression(iae.Expression);
                if (!rhsIsReducible)
                {
                    bool rhsIsCopy = Recognizer.IsStaticGenericMethod(iae.Expression, new Func<PlaceHolder, PlaceHolder, PlaceHolder>(ArrayHelper.SetTo));
                    if (rhsIsCopy)
                    {
                        IMethodInvokeExpression imie = (IMethodInvokeExpression)iae.Expression;
                        var sourceExpr = imie.Arguments[1];
                        IExpression source;
                        var sourceIndices = Recognizer.GetIndices(sourceExpr, out source);
                        if (source is IVariableDeclaration)
                        {
                            IVariableDeclaration sourceVar = Recognizer.GetVariableDeclaration(source);
                            HoistingInfo sourceInfo = GetOrCreateHoistingInfo(sourceVar);
                            for (int bracket = indices.Count; bracket < info.maxDepthWhereDimensionCouldMatter.Length; bracket++)
                            {
                                int sourceBracket = sourceIndices.Count + bracket;
                                for (int dim = 0; dim < info.maxDepthWhereDimensionCouldMatter[bracket].Length; dim++)
                                {
                                    // the dimension matters if it matters in the source array.
                                    int sourceDepth = sourceInfo.maxDepthWhereDimensionCouldMatter[sourceBracket][dim];
                                    SetDimensionMattersAtDepth(info, bracket, dim, sourceDepth);
                                    depth = System.Math.Max(depth, sourceDepth);
                                }
                            }
                        }
                        else
                        {
                            rhsIsCopy = false;
                        }
                    }
                    if (!rhsIsCopy)
                    {
                        // all deeper dimensions matter
                        for (int bracket = indices.Count; bracket < info.maxDepthWhereDimensionCouldMatter.Length; bracket++)
                        {
                            for (int dim = 0; dim < info.maxDepthWhereDimensionCouldMatter[bracket].Length; dim++)
                            {
                                int arrayDepth = info.maxDepthWhereDimensionCouldMatter.Length;
                                SetDimensionMattersAtDepth(info, bracket, dim, arrayDepth);
                                depth = System.Math.Max(depth, arrayDepth);
                            }
                        }
                    }
                }
                // expand variablesInExpression based on correlations in the index expressions.
                // For example:
                //   array[i][index[i][j]] = j
                // depends on i and j.
                bool changed;
                do
                {
                    changed = false;
                    for (int bracket = 0; bracket < indices.Count; bracket++)
                    {
                        for (int dim = 0; dim < indices[bracket].Count; dim++)
                        {
                            var index = indices[bracket][dim];
                            var indexVars = Recognizer.GetVariables(index).ToList();
                            if (variablesInExpression.ContainsAny(indexVars) && indexVars.Any(v => !variablesInExpression.Contains(v)))
                            {
                                variablesInExpression.AddRange(indexVars);
                                changed = true;
                            }
                        }
                    }
                } while (changed);
                for (int bracket = 0; bracket < indices.Count; bracket++)
                {
                    for (int dim = 0; dim < indices[bracket].Count; dim++)
                    {
                        var index = indices[bracket][dim];
                        if (index is IVariableReferenceExpression)
                        {
                            IVariableDeclaration indexVar = Recognizer.GetVariableDeclaration(index);
                            if (variablesInExpression.Contains(indexVar) || !unbrokenLoopVars.Contains(indexVar))
                            {
                                // index is used on rhs or the loop is broken, so the dimension matters.
                                SetDimensionMattersAtDepth(info, bracket, dim, depth);
                            }
                        }
                        else
                        {
                            // index is not a loop counter, so the dimension matters.
                            SetDimensionMattersAtDepth(info, bracket, dim, depth);
                        }
                    }
                }
                return iae;
            }
            else
            {
                return base.ConvertAssign(iae);
            }
        }

        public static bool IsReducibleArrayCreateExpression(IExpression expr)
        {
            if (expr is IArrayCreateExpression)
            {
                IArrayCreateExpression iace = (IArrayCreateExpression)expr;
                return iace.Initializer == null;
            }
            else if (expr is IObjectCreateExpression)
            {
                IObjectCreateExpression ioce = (IObjectCreateExpression)expr;
                Type type = ioce.Type.DotNetType;
                Type gtd = type.IsGenericType ? type.GetGenericTypeDefinition() : type;
                bool reducible =
                    (gtd == typeof(Distributions.DistributionRefArray<,>)
                    && ioce.Arguments.Count == 1) ||
                    (gtd == typeof(Distributions.DistributionStructArray<,>)
                    && ioce.Arguments.Count == 1) ||
                    (gtd == typeof(Distributions.DistributionRefArray2D<,>)
                    && ioce.Arguments.Count == 2) ||
                    (gtd == typeof(Distributions.DistributionStructArray2D<,>)
                    && ioce.Arguments.Count == 2);
                return reducible && (ioce.Initializer == null);
            }
            else return false;
        }

        private void SetDimensionMattersAtDepth(HoistingInfo info, int bracket, int dim, int depth)
        {
            var oldDepth = info.maxDepthWhereDimensionCouldMatter[bracket][dim];
            if (depth > oldDepth)
            {
                info.maxDepthWhereDimensionCouldMatter[bracket][dim] = depth;
                anyChanged = true;
            }
        }

        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            variablesInExpression.Add(ivre.Variable.Variable);
            return base.ConvertVariableRefExpr(ivre);
        }

        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            variablesInExpression.Clear();
            ConvertExpression(ics.Condition);
            var variablesAdded = (Set<IVariableDeclaration>)variablesInExpression.Clone();
            unbrokenLoopVars.Remove(variablesInExpression);
            ConvertBlock(ics.Then);
            if (ics.Else != null) ConvertBlock(ics.Else);
            return ics;
        }

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            var loopVar = Recognizer.LoopVariable(ifs);
            bool isBrokenLoop = ifs is IBrokenForStatement;
            if (!isBrokenLoop) unbrokenLoopVars.Add(loopVar);
            base.ConvertFor(ifs);
            unbrokenLoopVars.Remove(loopVar);
            return ifs;
        }
    }
}