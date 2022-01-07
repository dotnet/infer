// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Convert array index expressions into local variables where needed
    /// </summary>
    /// <remarks>
    /// The general pattern is that
    /// for(i) { ...a[i]... }
    /// is converted into
    /// for(i) {
    ///   T local = a[i];
    ///   ...local...
    ///   a[i] = local;
    /// }
    /// This transformation is only valid when a[i] is the only element of the array accessed in the loop body.
    /// </remarks>
    internal class LocalTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "LocalTransform";
            }
        }

        internal static bool debug;
        private Dictionary<IStatement, Dictionary<IExpression, LocalAnalysisTransform.LocalInfo>> localInfoOfStmt;
        private readonly Stack<IStatement> openContainers = new Stack<IStatement>();
        private readonly ModelCompiler compiler;
        private bool isTransformingContainer;

        internal LocalTransform(ModelCompiler compiler)
        {
            this.compiler = compiler;
        }

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            var analysis = new LocalAnalysisTransform(this.compiler);
            analysis.Context.InputAttributes = context.InputAttributes;
            analysis.Transform(itd);
            context.Results = analysis.Context.Results;
            if (!context.Results.IsSuccess)
            {
                Error("analysis failed");
                return itd;
            }
            this.localInfoOfStmt = analysis.localInfoOfStmt;
            var itdOut = base.Transform(itd);
            if (context.trackTransform && debug)
            {
                IBlockStatement block = Builder.BlockStmt();
                foreach (var entry in analysis.localInfoOfStmt.Values.SelectMany(dict => dict))
                {
                    var expr = entry.Key;
                    var info = entry.Value;
                    block.Statements.Add(Builder.CommentStmt(expr.ToString() + " " + StringUtil.ToString(info)));
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

        protected override IStatement DoConvertStatement(IStatement ist)
        {
            bool hasLocals = localInfoOfStmt.ContainsKey(ist);
            if (hasLocals)
                openContainers.Push(ist);
            IStatement st = base.DoConvertStatement(ist);
            if (hasLocals)
                openContainers.Pop();
            return st;
        }

        internal class DoNotUseLocal : ICompilerAttribute
        {
        }

        private bool ShouldUseLocal(LocalAnalysisTransform.LocalInfo info, int depth)
        {
            // has the local already been used?
            if (info.localVar != null || info.convertedCount > 0)
            {
                // when transforming a container, it is possible that the ancestor index of the container 
                // is earlier than the local in the container expression.  In such a case, we can't use the local.
                if(!isTransformingContainer)
                    return true;
            }
            if (isTransformingContainer) return false;
            // first time using this local
            bool readsFromArray = info.hasReadBeforeWrite || info.minWriteDepth > depth;
            int redundantCount;
            if (readsFromArray)
            {
                if (info.hasWrite)
                    redundantCount = 2;
                else
                    redundantCount = 1;
            }
            else
            {
                if (info.hasWrite)
                    redundantCount = 1;
                else
                    redundantCount = 0;
            }
            bool redundant = (info.containers.Count == 0) ||
                (info.count == redundantCount && !info.appearsInNestedLoop);
            // A FileArray must be read before written at a higher depth.
            if (info.minWriteDepth > depth && HasPartitionedLoop(info.containers))
                redundant = false;
            return !redundant;
        }

        private bool HasPartitionedLoop(Containers containers)
        {
            foreach (var container in containers.inputs)
            {
                if (container is IForStatement ifs)
                {
                    IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
                    if (context.InputAttributes.Has<Partitioned>(loopVar)) return true;
                }
            }
            return false;
        }

        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            IExpression expr = base.ConvertArrayIndexer(iaie);
            foreach (IStatement container in openContainers)
            {
                Dictionary<IExpression, LocalAnalysisTransform.LocalInfo> localInfoOfExpr = localInfoOfStmt[container];
                LocalAnalysisTransform.LocalInfo info;
                if (localInfoOfExpr.TryGetValue(iaie, out info))
                {
                    int depth = Recognizer.GetIndexingDepth(iaie);
                    if (ShouldUseLocal(info, depth))
                    {
                        object decl = Recognizer.GetDeclaration(expr);
                        if (info.localVar == null)
                        {
                            // name must end with a fixed suffix, to avoid clashes
                            string name = VariableInformation.GenerateName(context, MessageTransform.GetName(iaie) + "_local");
                            info.localVar = Builder.VarDecl(name, iaie.GetExpressionType());
                        }
                        if (!isTransformingContainer)
                        {
                            IExpression localExpr = Builder.VarRefExpr(info.localVar);
                            if (info.convertedCount == 0)
                            {
                                // this is the first conversion, so output a declaration and initializer
                                var declExpr = Builder.VarDeclExpr(info.localVar);
                                IStatement declSt = Builder.ExprStatement(declExpr);
                                // an initializer is needed to suppress compiler errors about 'Use of unassigned local variable'
                                declSt = Builder.AssignStmt(declExpr, Builder.DefaultExpr(info.localVar.VariableType));
                                int ancIndex = GetMatchingAncestorIndex(context, info.containingStatements);
                                context.AddStatementBeforeAncestorIndex(ancIndex, declSt);
                                if (info.hasReadBeforeWrite || info.minWriteDepth > depth)
                                {
                                    // initialize the local from the array element
                                    IStatement init = Builder.AssignStmt(localExpr, expr);
                                    int ancIndex2 = info.containers.GetMatchingAncestorIndex(context);
                                    Containers missing = info.containers.GetContainersNotInContext(context, ancIndex2);
                                    init = Containers.WrapWithContainers(init, TransformContainers(missing.outputs));
                                    context.AddStatementBeforeAncestorIndex(ancIndex2, init);
                                }
                            }
                            info.convertedCount++;
                            if (info.convertedCount == info.count)
                            {
                                if (info.hasWrite)
                                {
                                    // this is the last appearance of the local, so update the array element from the local
                                    IStatement st = Builder.AssignStmt(expr, localExpr);
                                    int ancIndex2 = info.containers.GetMatchingAncestorIndex(context);
                                    Containers missing = info.containers.GetContainersNotInContext(context, ancIndex2);
                                    // containers may contain locals, so they must be transformed.
                                    st = Containers.WrapWithContainers(st, TransformContainers(missing.outputs));
                                    context.AddStatementAfterAncestorIndex(ancIndex2, st);
                                }
                                info.convertedCount = 0; // reset in case this variable appears in another block
                            }
                        }
                        return Builder.VarRefExpr(info.localVar);
                    }
                }
            }
            return expr;
        }

        private ICollection<IStatement> TransformContainers(IEnumerable<IStatement> containers)
        {
            this.isTransformingContainer = true;
            // must put a dummy statement inside each container so that conversion doesn't prune them.
            var dummySt = Builder.CommentStmt();
            var result = containers.Select(c => ConvertStatement(Containers.WrapWithContainers(dummySt, new[] { Containers.CreateContainer(c) }))).ToList();
            this.isTransformingContainer = false;
            return result;
        }

        /// <summary>
        /// Returns the outermost container in the context that is not in this.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="containers"></param>
        /// <returns></returns>
        private int GetMatchingAncestorIndex(BasicTransformContext context, ICollection<IStatement> containers)
        {
            int ancIndex = context.FindAncestorIndex<IStatement>();
            foreach (IStatement container in Containers.FindContainers(context))
            {
                if (context.InputAttributes.Has<ConvergenceLoop>(container)) continue;
                if (!containers.Contains(container))
                {
                    // found a container unique to the current context.
                    // statements must be added here.
                    ancIndex = context.GetAncestorIndex(container);
                    break;
                }
            }
            return ancIndex;
        }
    }

    /// <summary>
    /// Determines which array index expressions can be local variables
    /// </summary>
    internal class LocalAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "LocalAnalysisTransform";
            }
        }

        internal class UsageInfo
        {
            public Dictionary<IExpression, LocalInfo> localInfos = new Dictionary<IExpression, LocalInfo>();
        }

        /// <summary>
        /// Stores usage information for an array indexer expression
        /// </summary>
        internal class LocalInfo
        {
            /// <summary>
            /// A fresh variable to hold the value of the expression
            /// </summary>
            public IVariableDeclaration localVar;
            /// <summary>
            /// True if the expression is read before being modified (or not modified at all) in this loop
            /// </summary>
            public bool hasReadBeforeWrite;
            /// <summary>
            /// The minimum indexing depth at which the expression is read before being modified (or not modified at all) in this loop
            /// </summary>
            public int minReadBeforeWriteDepth;
            /// <summary>
            /// True if the expression is modified in this loop
            /// </summary>
            public bool hasWrite;
            /// <summary>
            /// The minimum indexing depth of a write
            /// </summary>
            public int minWriteDepth;
            /// <summary>
            /// True if the expression appears in a nested loop
            /// </summary>
            public bool appearsInNestedLoop;
            /// <summary>
            /// The number of times the expression appears in this loop
            /// </summary>
            public int count;
            /// <summary>
            /// Used by LocalTransform to keep track of the number of conversions
            /// </summary>
            public int convertedCount;

            /// <summary>
            /// The intersection of all containers the indexing expression has appeared in, where each statement is unique.
            /// </summary>
            public HashSet<IStatement> containingStatements;

            /// <summary>
            /// The intersection of all containers the indexing expression has appeared in, merging equivalent containers.
            /// </summary>
            public Containers containers;

            public LocalInfo Clone()
            {
                var result = new LocalInfo();
                result.hasReadBeforeWrite = this.hasReadBeforeWrite;
                result.minReadBeforeWriteDepth = this.minReadBeforeWriteDepth;
                result.hasWrite = this.hasWrite;
                result.minWriteDepth = this.minWriteDepth;
                result.appearsInNestedLoop = this.appearsInNestedLoop;
                result.count = this.count;
                // don't need to clone containers since they are never mutated
                result.containers = this.containers;
                result.containingStatements = new HashSet<IStatement>(this.containingStatements, this.containingStatements.Comparer);
                return result;
            }

            public void Add(LocalInfo localInfo)
            {
                this.hasWrite = this.hasWrite || localInfo.hasWrite;
                this.minWriteDepth = System.Math.Min(this.minWriteDepth, localInfo.minWriteDepth);
                this.hasReadBeforeWrite = this.hasReadBeforeWrite || localInfo.hasReadBeforeWrite;
                this.minReadBeforeWriteDepth = System.Math.Min(this.minReadBeforeWriteDepth, localInfo.minReadBeforeWriteDepth);
                this.appearsInNestedLoop = this.appearsInNestedLoop || localInfo.appearsInNestedLoop;
                this.count += localInfo.count;
                this.containingStatements.IntersectWith(localInfo.containingStatements);
                this.containers = Containers.Intersect(this.containers, localInfo.containers, allowBrokenLoops: true, ignoreLoopDirection: true);
            }
        }

        internal class ContainerInfo
        {
            public Dictionary<object, UsageInfo> usageInfoOfVariable = new Dictionary<object, UsageInfo>();
        }
        /// <summary>
        /// Holds the ContainerInfo for each 'for' loop on the inputStack as they are being converted.
        /// </summary>
        private readonly Stack<ContainerInfo> containerInfos = new Stack<ContainerInfo>();
        /// <summary>
        /// Maps an IForStatement to its LocalInfos
        /// </summary>
        internal Dictionary<IStatement, Dictionary<IExpression, LocalInfo>> localInfoOfStmt = new Dictionary<IStatement, Dictionary<IExpression, LocalInfo>>(ReferenceEqualityComparer<IStatement>.Instance);
        private readonly Stack<IStatement> openContainers = new Stack<IStatement>();
        private bool InPartitionedLoop;
        readonly ModelCompiler compiler;

        internal LocalAnalysisTransform(ModelCompiler compiler)
        {
            this.compiler = compiler;
        }

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            // convert the initializer, condition, and increment outside of the container.
            context.SetPrimaryOutput(ifs);
            ConvertStatement(ifs.Initializer);
            ConvertExpression(ifs.Condition);
            ConvertStatement(ifs.Increment);

            // convert the body inside of a new ContainerInfo.
            bool wasPartitioned = this.InPartitionedLoop;
            IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
            bool isPartitioned = InPartitionedLoop || context.InputAttributes.Has<Partitioned>(loopVar) || (this.compiler.UseLocals && this.compiler.OptimiseInferenceCode);
            if (isPartitioned)
            {
                this.InPartitionedLoop = true;
                containerInfos.Push(new ContainerInfo());
            }
            openContainers.Push(ifs);
            ConvertBlock(ifs.Body);
            openContainers.Pop();
            if (isPartitioned)
            {
                ContainerInfo containerInfo = containerInfos.Pop();
                var localInfos = new Dictionary<IExpression, LocalInfo>();
                localInfoOfStmt[ifs] = localInfos;
                // find the longest common prefix of each variable's expressions
                foreach (var entry in containerInfo.usageInfoOfVariable)
                {
                    object decl = entry.Key;
                    UsageInfo usageInfo = entry.Value;
                    CombineLocalInfos(usageInfo.localInfos);
                    foreach (var pair in usageInfo.localInfos)
                    {
                        IExpression expr = pair.Key;
                        LocalInfo localInfo = pair.Value;
                        IExpression prefix = GetPrefixInParent(expr, loopVar);
                        if (!ReferenceEquals(prefix, expr))
                        {
                            // expr refers to this loop variable
                            IArrayIndexerExpression iaie = (IArrayIndexerExpression)expr;
                            if (usageInfo.localInfos.Count == 1)
                            {
                                localInfos.Add(expr, localInfo);
                            }
                        }
                        // add the prefix to the parent containerInfo
                        var info = AddLocalInfo(decl, prefix, localInfo, ifs);
                    }
                }
                this.InPartitionedLoop = wasPartitioned;
            }
            return ifs;
        }

        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            context.SetPrimaryOutput(ics);
            ConvertExpression(ics.Condition);
            openContainers.Push(ics);
            ConvertBlock(ics.Then);
            openContainers.Pop();
            return ics;
        }

        protected static void CombineLocalInfos(Dictionary<IExpression, LocalInfo> localInfos)
        {
            if (localInfos.Count <= 1)
                return;
            IExpression minExpr = null;
            foreach (var expr in localInfos.Keys)
            {
                if (minExpr == null)
                    minExpr = expr;
                else
                    minExpr = GetCommonPrefix(minExpr, expr);
            }
            localInfos.TryGetValue(minExpr, out LocalInfo minInfo);
            foreach (var entry in localInfos)
            {
                var expr = entry.Key;
                if (!expr.Equals(minExpr))
                {
                    if (minInfo == null)
                    {
                        minInfo = entry.Value;
                    }
                    else
                    {
                        minInfo.Add(entry.Value);
                    }
                }
            }
            localInfos.Clear();
            localInfos.Add(minExpr, minInfo);
        }

        protected static IExpression GetCommonPrefix(IExpression expr, IExpression expr2)
        {
            List<IExpression> prefixes1 = Recognizer.GetAllPrefixes(expr);
            List<IExpression> prefixes2 = Recognizer.GetAllPrefixes(expr2);
            if (!prefixes1[0].Equals(prefixes2[0]))
                throw new Exception($"Expressions have no overlap: {expr}, {expr2}");
            int count = System.Math.Min(prefixes1.Count, prefixes2.Count);
            for (int i = 1; i < count; i++)
            {
                IExpression prefix1 = prefixes1[i];
                IExpression prefix2 = prefixes2[i];
                if (!prefix1.Equals(prefix2))
                {
                    return prefixes1[i - 1];
                }
            }
            return prefixes1[count - 1];
        }

        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            context.OutputAttributes.Remove<Containers>(ivde.Variable);
            context.OutputAttributes.Set(ivde.Variable, new Containers(context));
            return base.ConvertVariableDeclExpr(ivde);
        }

        /// <summary>
        /// Add localInfo for expr to the current containerInfo.
        /// </summary>
        /// <param name="decl"></param>
        /// <param name="expr"></param>
        /// <param name="localInfo"></param>
        /// <param name="closedContainer"></param>
        protected LocalInfo AddLocalInfo(object decl, IExpression expr, LocalInfo localInfo, IForStatement closedContainer)
        {
            if (containerInfos.Count == 0)
                return localInfo;
            ContainerInfo containerInfo = containerInfos.Peek();
            UsageInfo usageInfo;
            if (!containerInfo.usageInfoOfVariable.TryGetValue(decl, out usageInfo))
            {
                usageInfo = new UsageInfo();
                containerInfo.usageInfoOfVariable[decl] = usageInfo;
            }
            LocalInfo info;
            if (!usageInfo.localInfos.TryGetValue(expr, out info))
            {
                info = localInfo.Clone();
                info.containingStatements.IntersectWith(openContainers);
                IVariableDeclaration loopVar = Recognizer.LoopVariable(closedContainer);
                int index = info.containers.inputs.FindIndex(container => Containers.ContainersAreEqual(container, closedContainer, allowBrokenLoops: true, ignoreLoopDirection: true));
                bool conditionsContainLoopVar = info.containers.inputs.Skip(index + 1).Any(container => Recognizer.GetVariables(GetContainerExpression(container)).Contains(loopVar));
                IStatement replacement = null;
                if (conditionsContainLoopVar)
                {
                    replacement = Builder.BrokenForStatement(closedContainer);
                    info.containingStatements.Add(replacement);
                }
                else
                {
                    var loopSize = Recognizer.LoopSizeExpression(closedContainer);
                    bool loopMustExecute = false;
                    if (loopSize is ILiteralExpression ile)
                    {
                        int loopSizeAsInt = (int)ile.Value;
                        if (loopSizeAsInt > 0)
                        {
                            loopMustExecute = true;
                        }
                    }
                    if (!loopMustExecute)
                    {
                        var condition = Builder.BinaryExpr(loopSize, BinaryOperator.GreaterThan, Builder.LiteralExpr(0));
                        replacement = Builder.CondStmt(condition, Builder.BlockStmt());
                    }
                }
                if (info.containers.inputs.Contains(replacement)) replacement = null;
                if (replacement == null)
                {
                    info.containers = info.containers.Where(container => !Containers.ContainersAreEqual(container, closedContainer, true, true));
                }
                else
                {
                    // this only replaces containers.inputs
                    info.containers = info.containers.Replace(container => !Containers.ContainersAreEqual(container, closedContainer, true, true)
                        ? container
                        : replacement);
                }
                int previousCount = info.containers.Count;
                info.containers = Containers.RemoveInvalidConditions(info.containers, context);
                if (info.containers.Count != previousCount && expr is IArrayIndexerExpression)
                {
                    // when dropping conditionals, we need to show that if the indices were valid inside the conditionals, they remain valid outside the conditionals.
                    // This is automatically true if the indices match the indexVars.
                    expr = GetPrefixIndexedByIndexVars(expr);
                }
                usageInfo.localInfos[expr] = info;
            }
            else
            {
                info.Add(localInfo);
            }
            info.appearsInNestedLoop = true;
            return info;
        }

        internal static void WriteDebugString(IExpression expr, IEnumerable<IStatement> containers)
        {
            string containerString = StringUtil.ToString(containers.Select(c =>
            {
                if (c is IForStatement ifs)
                {
                    if (c is IBrokenForStatement)
                        return ifs.Initializer.ToString() + " // broken";
                    else
                        return ifs.Initializer.ToString();
                }
                else
                {
                    return ((IConditionStatement)c).Condition.ToString();
                }
            }));
            Trace.WriteLine(StringUtil.JoinColumns(expr, " ", containerString));
        }

        private static IExpression GetContainerExpression(IStatement container)
        {
            if (container is IForStatement ifs) return Recognizer.LoopSizeExpression(ifs);
            else if (container is IConditionStatement ics) return ics.Condition;
            else if (container is IRepeatStatement irs) return irs.Count;
            else throw new ArgumentException($"unrecognized container type: {container.GetType()}");
        }

        protected void AddUsage(IExpression expr, bool isWrite)
        {
            bool anyIndexIsPartitioned = (expr is IArrayIndexerExpression) && 
                Recognizer.GetIndices(expr).Any(bracket => bracket.Any(index => Recognizer.GetVariables(index).Any(context.InputAttributes.Has<Partitioned>)));
            object decl = Recognizer.GetDeclaration(expr);
            if (decl == null || (!anyIndexIsPartitioned && context.InputAttributes.Has<LocalTransform.DoNotUseLocal>(decl)))
                return;
            if (containerInfos.Count == 0)
            {
                return;
            }
            ContainerInfo containerInfo = containerInfos.Peek();
            UsageInfo usageInfo;
            if (!containerInfo.usageInfoOfVariable.TryGetValue(decl, out usageInfo))
            {
                usageInfo = new UsageInfo();
                containerInfo.usageInfoOfVariable[decl] = usageInfo;
            }
            LocalInfo info;
            if (!usageInfo.localInfos.TryGetValue(expr, out info))
            {
                info = new LocalInfo();
                info.minWriteDepth = int.MaxValue;
                info.minReadBeforeWriteDepth = int.MaxValue;
                // this assignment is not needed since we must have depth < info.minWriteDepth
                //info.hasReadBeforeWrite = !isWrite;
                usageInfo.localInfos[expr] = info;
            }
            int depth = Recognizer.GetIndexingDepth(expr);
            if (isWrite)
            {
                info.hasWrite = true;
                info.minWriteDepth = System.Math.Min(info.minWriteDepth, depth);
            }
            else if (depth < info.minWriteDepth)
            {
                info.hasReadBeforeWrite = true;
                info.minReadBeforeWriteDepth = System.Math.Min(info.minReadBeforeWriteDepth, depth);
            }
            info.count++;
            if (info.containingStatements == null)
            {
                info.containingStatements = new HashSet<IStatement>(openContainers, ReferenceEqualityComparer<IStatement>.Instance);
                info.containers = GetContainers();
            }
            else
            {
                info.containingStatements.IntersectWith(openContainers);
                info.containers = Containers.Intersect(info.containers, GetContainers(), allowBrokenLoops: true, ignoreLoopDirection: true);
            }
        }

        private IExpression GetPrefixIndexedByIndexVars(IExpression expr)
        {
            var decl = Recognizer.GetDeclaration(expr);
            if (decl != null)
            {
                var varInfo = VariableInformation.GetVariableInformation(context, decl);
                var indices = Recognizer.GetIndices(expr);
                for (int bracket = 0; bracket < indices.Count; bracket++)
                {
                    if (!IndicesMatchIndexVars(indices[bracket], varInfo.indexVars[bracket]))
                    {
                        return GetExpressionUpToDepth(expr, bracket, indices.Count);
                    }
                }
            }
            return expr;
        }

        private IExpression GetExpressionUpToDepth(IExpression expr, int desiredDepth, int expressionDepth)
        {
            if (desiredDepth >= expressionDepth) return expr;
            else return GetExpressionUpToDepth(((IArrayIndexerExpression)expr).Target, desiredDepth, expressionDepth - 1);
        }

        private bool IndicesMatchIndexVars(IList<IExpression> indices, IReadOnlyList<IVariableDeclaration> indexVars)
        {
            for (int dim = 0; dim < indices.Count; dim++)
            {
                IExpression index = indices[dim];
                if (!GetPrefixIndexedByIndexVars(index).Equals(index)) return false;
                IVariableDeclaration indexVar = Recognizer.GetVariableDeclaration(index);
                if (indexVar != null && !indexVar.Equals(indexVars[dim]))
                    return false;
            }
            return true;
        }

        private Containers GetContainers()
        {
            var containers = new Containers();
            foreach (var container in openContainers.Reverse())
                containers.Add(Containers.CreateContainer(container));
            return containers;
        }

        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            bool isPartOfLhs = Recognizer.IsBeingMutated(context, ivre);
            AddUsage(ivre, isPartOfLhs);
            return base.ConvertVariableRefExpr(ivre);
        }

        protected override IExpression ConvertFieldRefExpr(IFieldReferenceExpression ifre)
        {
            if (ifre.Target is IThisReferenceExpression)
            {
                bool isPartOfLhs = Recognizer.IsBeingMutated(context, ifre);
                AddUsage(ifre, isPartOfLhs);
                return ifre;
            }
            else
            {
                return base.ConvertFieldRefExpr(ifre);
            }
        }

        protected override IExpression ConvertPropertyRefExpr(IPropertyReferenceExpression ipre)
        {
            if (ipre.Target is IThisReferenceExpression)
            {
                bool isPartOfLhs = Recognizer.IsBeingMutated(context, ipre);
                AddUsage(ipre, isPartOfLhs);
                return ipre;
            }
            else
            {
                return base.ConvertPropertyRefExpr(ipre);
            }
        }

        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            bool isPartOfLhs = Recognizer.IsBeingMutated(context, iaie);
            AddUsage(iaie, isPartOfLhs);
            // loop over prefixes of the expression
            IArrayIndexerExpression prefix = iaie;
            bool isPrefix = false;
            while (true)
            {
                foreach (IExpression index in prefix.Indices)
                    ConvertExpression(index);
                if (isPrefix && false)
                {
                    bool anyIndexIsPartitioned = prefix.Indices.Any(index => Recognizer.GetVariables(index).Any(ivd => context.InputAttributes.Has<Partitioned>(ivd)));
                    if (anyIndexIsPartitioned)
                    {
                        AddUsage(prefix, false);
                    }
                }
                if (prefix.Target is IArrayIndexerExpression target)
                {
                    prefix = target;
                    isPrefix = true;
                }
                else
                    break;
            }
            return iaie;
        }

        /// <summary>
        /// Find the longest prefix that does not have any index expression containing the current loop variable.
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="excludedLoopVar"></param>
        /// <returns></returns>
        private IExpression GetPrefixInParent(IExpression expr, IVariableDeclaration excludedLoopVar)
        {
            if (expr is IArrayIndexerExpression)
            {
                IExpression expr2 = expr;
                while (expr2 is IArrayIndexerExpression iaie)
                {
                    bool hasExcludedLoopVar = iaie.Indices.Any(index => Recognizer.GetVariables(index).Any(excludedLoopVar.Equals));
                    if (hasExcludedLoopVar)
                        return GetPrefixInParent(iaie.Target, excludedLoopVar);
                    expr2 = iaie.Target;
                }
            }
            return expr;
        }
    }
}