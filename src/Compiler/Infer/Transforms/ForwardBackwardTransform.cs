// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;
using Microsoft.ML.Probabilistic.Utilities;
using System.Diagnostics;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Creates statement groups corresponding to forward/backward iterations when there are offset indices
    /// </summary>
    /// <remarks><para>
    /// Purpose of the ForwardBackwardTransform is to put statements into groups.  
    /// Groups are either "forward", "backward", or "neutral" wrt a loop index.
    /// Groups are denoted by "while(true)" statements, except for the neutral group which has no container.  
    /// If a statement appears in multiple groups, it is cloned. 
    /// The statements in a backward group have their loop bounds reversed, so that they iterate backwards.
    /// </para><para>
    /// The transform assumes that loop cutting has been performed so that each statement has its own set of containers.  
    /// Each statement must be annotated with DependencyInformation, where each dependency is annotated with a set of offsets.  
    /// An offset is described by an integer and a loop index, e.g. "i-1".  
    /// A single dependency edge can have multiple offsets on the same loop index, e.g. "i-1" and "i+1".
    /// </para><para>
    /// Chain case:  There is only one loop index with offsets, so there will be 3 groups: forward, backward, and neutral.
    /// A node is in the forward group iff it lies on a dependency cycle where:
    /// 1. at least one edge has a forward offset, and
    /// 2. all edges have either a forward or neutral offset.
    /// The backward group is defined the same way, but with "forward" replaced by "backward".  
    /// The neutral group consists of all other nodes.
    /// Note that a node may end up in both the forward and backward group.  But it cannot be in both the forward and neutral group.
    /// </para><para>
    /// Grid case: There are multiple loop indices with offsets.  This case is handled recursively.  
    /// The loop indices are put into an arbitrary order.  For the first loop index, we create 3 groups as above.  
    /// Then we recurse on each group, splitting along the second loop index, and so on.  This produces a hierarchy of groups.
    /// </para><para>
    /// Variable declarations must have 'Containers' attributes indicating the containers they belong to.  (Should be attached by DependencyAnalysisTransform)
    /// On input, "while(true)" statements must not be nested.
    /// </para></remarks>
    internal class ForwardBackwardTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "ForwardBackwardTransform";
            }
        }

        internal static bool debug;
        private readonly ModelCompiler compiler;
        private Set<IVariableDeclaration> loopVarsToReverse = new Set<IVariableDeclaration>();
        /// <summary>
        /// For each context, store a map from original statement to transformed statement
        /// </summary>
        private Dictionary<ICollection<IStatement>, Dictionary<IStatement, IStatement>> transformedStmtsInContext = new Dictionary<ICollection<IStatement>, Dictionary<IStatement, IStatement>>(new IdentityComparer<object>());
        private Dictionary<IStatement, Set<IVariableDeclaration>> reversedLoopVarsInStmt = new Dictionary<IStatement, Set<IVariableDeclaration>>(new IdentityComparer<IStatement>());
        private LoopMergingInfo loopMergingInfo;
        private Stack<StackFrame> stackFrames = new Stack<StackFrame>();
        private IVariableDeclaration currentLoopVar;
        private bool convertedStmtHasLoopVar;
        private Containers containersOfLoopVar;
        /// <summary>
        /// If true, external statements will be added to sequential loops
        /// </summary>
        private bool expandSequentialLoops = false;
        private Dictionary<IVariableDeclaration, Set<IStatement>> specialStmts = new Dictionary<IVariableDeclaration, Set<IStatement>>();
        /// <summary>
        /// For each original special statement, gives its transformed statement in each loop context
        /// </summary>
        private Dictionary<IStatement, Dictionary<Set<IVariableDeclaration>, IStatement>> specialTransformedStmts =
            new Dictionary<IStatement, Dictionary<Set<IVariableDeclaration>, IStatement>>(new IdentityComparer<IStatement>());

        protected class StackFrame
        {
            public Containers containers;
            public Dictionary<IExpression, IExpression> replacements = new Dictionary<IExpression, IExpression>();
        }

        internal ForwardBackwardTransform(ModelCompiler compiler)
        {
            this.compiler = compiler;
        }

        protected override IMethodDeclaration DoConvertMethod(IMethodDeclaration md, IMethodDeclaration imd)
        {
            if (!context.InputAttributes.Has<OperatorMethod>(imd))
                return imd;
            loopMergingInfo = context.InputAttributes.Get<LoopMergingInfo>(imd);
            if (debug && loopMergingInfo != null)
            {
                var itdOut = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
                context.OutputAttributes.Add(itdOut, loopMergingInfo.GetDebugInfo(this));
            }
            return base.DoConvertMethod(md, imd);
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            base.DoConvertMethodBody(outputs, inputs);
            PostProcessDependencies(outputs);
            PostProcessSpecialStmts();
        }

        /// <summary>
        /// Each special statement whose loopVar is reversed gets a dependency on the non-reversed copy of that statement.
        /// </summary>
        private void PostProcessSpecialStmts()
        {
            foreach (var entry in specialStmts)
            {
                var loopVar = entry.Key;
                var set = entry.Value;
                foreach (var ist in set)
                {
                    var dict = specialTransformedStmts[ist];
                    foreach (var entry2 in dict)
                    {
                        var reversed = entry2.Key;
                        var transformedStmt = entry2.Value;
                        if (!reversed.Contains(loopVar))
                            continue;
                        // find the non-reversed stmt
                        var complement = (Set<IVariableDeclaration>)reversed.Clone();
                        complement.Remove(loopVar);
                        var transformedStmt2 = dict[complement];
                        DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(transformedStmt);
                        if (di != null)
                        {
                            di.Add(DependencyType.Dependency, transformedStmt2);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Fill in the replacements dictionary for this block and collect replacements from descendant blocks
        /// </summary>
        /// <param name="outputBlock"></param>
        /// <param name="replacementsInContext"></param>
        private void CollectTransformedStmts(ICollection<IStatement> outputBlock, Dictionary<ICollection<IStatement>, Dictionary<IStatement, IStatement>> replacementsInContext)
        {
            var replacements = new Dictionary<IStatement, IStatement>(new IdentityComparer<IStatement>());
            replacementsInContext[outputBlock] = replacements;
            Dictionary<IStatement, IStatement> transformedStmts;
            if (transformedStmtsInContext.TryGetValue(outputBlock, out transformedStmts))
            {
                foreach (var entry in transformedStmts)
                {
                    var originalStmt = entry.Key;
                    var transformedStmt = entry.Value;
                    replacements[originalStmt] = transformedStmt;
                }
            }
            foreach (var stmt in outputBlock)
            {
                if (stmt is IWhileStatement)
                {
                    // recursively collect the child block
                    IWhileStatement iws = (IWhileStatement)stmt;
                    CollectTransformedStmts(iws.Body.Statements, replacementsInContext);
                    // merge the child replacements into this block's replacements
                    var childReplacements = replacementsInContext[iws.Body.Statements];
                    foreach (var entry in childReplacements)
                    {
                        AddReplacement(replacements, entry.Key, entry.Value);
                    }
                }
            }
        }

        private void AddReplacement(Dictionary<IStatement, IStatement> replacements, IStatement original, IStatement transformed)
        {
            IStatement existing;
            if (replacements.TryGetValue(original, out existing))
            {
                // construct Any statement
                AnyStatement anySt = new AnyStatement();
                anySt.Statements.Add(existing);
                anySt.Statements.Add(transformed);
                replacements[original] = anySt;
            }
            else
            {
                replacements[original] = transformed;
            }
        }

        /// <summary>
        /// Distribute the replacements dictionary for this block to its descendant blocks that lack the replacements
        /// </summary>
        /// <param name="outputBlock"></param>
        /// <param name="replacementsInContext"></param>
        private void DistributeTransformedStmts(ICollection<IStatement> outputBlock, Dictionary<ICollection<IStatement>, Dictionary<IStatement, IStatement>> replacementsInContext)
        {
            var replacements = replacementsInContext[outputBlock];
            foreach (var stmt in outputBlock)
            {
                if (stmt is IWhileStatement)
                {
                    IWhileStatement iws = (IWhileStatement)stmt;
                    // merge this block's replacements into the child's replacements
                    var childReplacements = replacementsInContext[iws.Body.Statements];
                    foreach (var entry in replacements)
                    {
                        if (!childReplacements.ContainsKey(entry.Key))
                        {
                            childReplacements[entry.Key] = entry.Value;
                        }
                    }
                    // recursively distribute the child block
                    DistributeTransformedStmts(iws.Body.Statements, replacementsInContext);
                }
            }
        }

        private void PostProcessDependencies(ICollection<IStatement> outputs)
        {
            // construct a replacement dictionary for every context
            // the replacement will be an AnyStatement containing all transformed versions that are compatible with this context
            var replacementsInContext = new Dictionary<ICollection<IStatement>, Dictionary<IStatement, IStatement>>(new IdentityComparer<object>());
            ICollection<IStatement> rootContext = outputs;
            transformedStmtsInContext[rootContext] = new Dictionary<IStatement, IStatement>(new IdentityComparer<IStatement>());
            CollectTransformedStmts(rootContext, replacementsInContext);
            DistributeTransformedStmts(rootContext, replacementsInContext);

            // apply the replacements, using the dictionary for the parent context
            if (replacementsInContext.Count > 0)
            {
                // contexts are bodies of innermost while statements
                var stack = new Stack<Dictionary<IStatement, IStatement>>();
                Dictionary<IStatement, IStatement> replacements = replacementsInContext[rootContext];
                SerialSchedulingInfo ssinfo = null;
                stack.Push(replacements);
                DeadCodeTransform.ForEachStatement(outputs,
                  delegate (IWhileStatement iws)
                  {
                      var ssinfo2 = context.OutputAttributes.Get<SerialSchedulingInfo>(iws);
                      if (ssinfo2 != null)
                          ssinfo = ssinfo2;
                      var loopBody = iws.Body.Statements;
                      replacementsInContext.TryGetValue(loopBody, out replacements);
                      stack.Push(replacements);
                  },
                  delegate (IWhileStatement iws)
                  {
                      var ssinfo2 = context.OutputAttributes.Get<SerialSchedulingInfo>(iws);
                      if (ssinfo2 != null)
                          ssinfo = null;
                      stack.Pop();
                      replacements = stack.Peek();
                  },
                  _ =>
                  {
                  },
                  _ =>
                  {
                  },
                  delegate (IStatement ist)
                  {
                      if (replacements != null)
                      {
                          DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                          if (di != null)
                          {
                              // must make a clone since this statement may appear in multiple contexts
                              DependencyInformation di2 = (DependencyInformation)di.Clone();
                              di2.Replace(replacements);
                              FilterOffsetDependencies(ssinfo, ist, di2);
                              context.OutputAttributes.Remove<DependencyInformation>(ist);
                              context.OutputAttributes.Set(ist, di2);
                          }
                      }
                  });
            }
        }

        private void FilterOffsetDependencies(SerialSchedulingInfo ssinfo, IStatement ist, DependencyInformation di)
        {
            // to preserve an offset, both stmts must have the same sign for that loopVar, as well as all parent loopVars
            if (ssinfo == null)
            {
                return;
            }
            Set<IVariableDeclaration> reversedLoopVars;
            if (!reversedLoopVarsInStmt.TryGetValue(ist, out reversedLoopVars))
            {
                reversedLoopVars = new Set<IVariableDeclaration>();
            }
            List<IStatement> toRemove = new List<IStatement>();
            List<KeyValuePair<IStatement, OffsetInfo>> toAdd = new List<KeyValuePair<IStatement, OffsetInfo>>();
            foreach (var entry in di.offsetIndexOf)
            {
                IStatement dependency = entry.Key;
                Set<IVariableDeclaration> reversedLoopVarsOther;
                if (!reversedLoopVarsInStmt.TryGetValue(dependency, out reversedLoopVarsOther))
                {
                    reversedLoopVarsOther = new Set<IVariableDeclaration>();
                }
                var offsetInfo = entry.Value;
                bool changed = false;
                OffsetInfo newOffsetInfo = new OffsetInfo();
                foreach (var offset in offsetInfo)
                {
                    if (CanKeepOffsetDependency(ssinfo, reversedLoopVars, reversedLoopVarsOther, offset))
                        newOffsetInfo.Add(offset);
                    else
                        changed = true;
                }
                if (changed)
                {
                    toRemove.Add(dependency);
                    if (newOffsetInfo.Count > 0)
                        toAdd.Add(new KeyValuePair<IStatement, OffsetInfo>(dependency, newOffsetInfo));
                }
            }
            foreach (var dependency in toRemove)
            {
                di.offsetIndexOf.Remove(dependency);
            }
            foreach (var pair in toAdd)
            {
                di.offsetIndexOf.Add(pair.Key, pair.Value);
            }
        }

        private bool CanKeepOffsetDependency(SerialSchedulingInfo ssinfo, Set<IVariableDeclaration> reversedLoopVars, Set<IVariableDeclaration> reversedLoopVarsOther, Offset offset)
        {
            foreach (var loopVar in ssinfo.loopInfos.Select(info => info.loopVar))
            {
                bool compatible = (reversedLoopVars.Contains(loopVar) == reversedLoopVarsOther.Contains(loopVar));
                if (!compatible)
                    return false;
                if (offset.loopVar == loopVar)
                    break;
            }
            return true;
        }

        /// <summary>
        /// Reverse the direction of 'for' loops according to loopVarsToReverse
        /// </summary>
        /// <param name="ifs"></param>
        /// <returns></returns>
        protected override IStatement ConvertFor(IForStatement ifs)
        {
            // convert body first
            IBlockStatement convertedBody = ConvertBlock(ifs.Body);
            // is the loop variable being replaced?
            IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
            if (loopVar.Equals(this.currentLoopVar) && !(ifs is IBrokenForStatement))
            {
                this.convertedStmtHasLoopVar = true;
                if (this.containersOfLoopVar == null)
                {
                    this.containersOfLoopVar = Containers.GetContainersNeededForExpression(context, ifs.Condition);
                }
            }
            bool changed = !ReferenceEquals(convertedBody, ifs.Body);
            bool mustReverse = loopVarsToReverse.Contains(loopVar);
            IForStatement fs = ifs;
            if (changed || mustReverse)
            {
                fs = Builder.ForStmt(ifs);
                fs.Initializer = ConvertStatement(ifs.Initializer);
                fs.Condition = ConvertExpression(ifs.Condition);
                fs.Increment = ConvertStatement(ifs.Increment);
                fs.Body = convertedBody;
                if (mustReverse && !(ifs is IBrokenForStatement))
                {
                    Recognizer.ReverseLoopDirection(fs);
                }
                context.InputAttributes.CopyObjectAttributesTo(ifs, context.OutputAttributes, fs);
            }
            return fs;
        }

        public override IExpression ConvertExpression(IExpression expr)
        {
            expr = base.ConvertExpression(expr);
            // apply replacements on the stack
            foreach (var frame in stackFrames)
            {
                IExpression value;
                if (frame.replacements.TryGetValue(expr, out value))
                    expr = value;
            }
            return expr;
        }

        /// <summary>
        /// Create forward/backward composites in the body of while(true) loops
        /// </summary>
        /// <param name="iws"></param>
        protected override IStatement ConvertWhile(IWhileStatement iws)
        {
            bool isWhileTrue = (iws.Condition is ILiteralExpression) && (((ILiteralExpression)iws.Condition).Value.Equals(true));
            if (!isWhileTrue)
                return base.ConvertWhile(iws);
            IWhileStatement ws = Builder.WhileStmt(iws);
            context.SetPrimaryOutput(ws);
            ws.Condition = ConvertExpression(iws.Condition);
            ws.Body = Builder.BlockStmt();
            Context.OpenStatement(iws.Body);
            context.SetPrimaryOutput(ws.Body);
            Schedule(ws.Body.Statements, (IReadOnlyList<IStatement>)iws.Body.Statements, context.InputAttributes.Has<DoNotSchedule>(iws));
            Context.CloseStatement(iws.Body);
            context.InputAttributes.CopyObjectAttributesTo(iws, context.OutputAttributes, ws);
            return ws;
        }

        protected IStatement ConvertTopLevelStatement(IStatement ist, IList<IStatement> containerPrefix)
        {
            IStatement permuted = PermuteContainers(ist, containerPrefix);
            IStatement newSt = base.ConvertStatement(permuted);
            if (newSt == null)
                return newSt;
            if (stackFrames.Count > 0)
            {
                List<IStatement> containersOfSt = new List<IStatement>();
                IList<IStatement> core = LoopMergingTransform.UnwrapStatement(newSt, containersOfSt);
                bool containersChanged = false;
                foreach (var frame in stackFrames)
                {
                    // if no containers are replaced in this frame, add new containers to the end
                    // if a container is replaced in this frame, add new containers at that position
                    bool frameContainersAdded = false;
                    List<IStatement> newContainers = new List<IStatement>();
                    foreach (IStatement container in containersOfSt)
                    {
                        if (container is IForStatement)
                        {
                            IExpression loopVarExpr = Builder.VarRefExpr(Recognizer.LoopVariable((IForStatement)container));
                            if (frame.replacements.ContainsKey(loopVarExpr))
                            {
                                // add all missing containers
                                foreach (IStatement newContainer in frame.containers.inputs)
                                {
                                    if (!Containers.ListContains(newContainers, newContainer))
                                        newContainers.Add(newContainer);
                                }
                                frameContainersAdded = true;
                                containersChanged = true;
                                // do not add the current container
                                continue;
                            }
                        }
                        if (!Containers.ListContains(newContainers, container))
                            newContainers.Add(container);
                    }
                    if (!frameContainersAdded)
                    {
                        // add all missing containers
                        foreach (IStatement newContainer in frame.containers.inputs)
                        {
                            if (!Containers.ListContains(newContainers, newContainer))
                            {
                                newContainers.Add(newContainer);
                                containersChanged = true;
                            }
                        }
                    }
                    containersOfSt = newContainers;
                }
                if (containersChanged)
                    newSt = Containers.WrapWithContainers(core, containersOfSt)[0];
            }
            if (ReferenceEquals(newSt, ist))
                return newSt;
            // this copies across the DependencyInformation unchanged
            // PostProcessDependencies later updates the DependencyInformation
            context.InputAttributes.CopyObjectAttributesTo(ist, context.OutputAttributes, newSt);
            if (loopMergingInfo != null)
            {
                // update loopMergingInfo with the new statement
                loopMergingInfo.AddEquivalentStatement(newSt, loopMergingInfo.GetIndexOf(ist));
            }
            return newSt;
        }

        internal static IStatement ExtractFirstIterationPostProcessingBlock(BasicTransformContext context, ref IReadOnlyList<IStatement> stmts)
        {
            IStatement firstIterPostBlock = null;
            foreach (var ist in stmts)
            {
                if (context.InputAttributes.Has<FirstIterationPostProcessingBlock>(ist))
                {
                    firstIterPostBlock = ist;
                    break;
                }
            }
            if (firstIterPostBlock != null)
                stmts = stmts.Where(ist => !context.InputAttributes.Has<FirstIterationPostProcessingBlock>(ist)).ToList();
            return firstIterPostBlock;
        }

        /// <summary>
        /// Convert inputStmts into forward/backward composites, placing result in outputStmts
        /// </summary>
        /// <param name="outputStmts"></param>
        /// <param name="inputStmts"></param>
        /// <param name="doNotSchedule"></param>
        protected void Schedule(IList<IStatement> outputStmts, IReadOnlyList<IStatement> inputStmts, bool doNotSchedule)
        {
            if (doNotSchedule)
            {
                foreach (IStatement ist in inputStmts)
                {
                    IStatement st = ConvertStatement(ist);
                    outputStmts.Add(st);
                }
                return;
            }

            IStatement firstIterPostBlock = ExtractFirstIterationPostProcessingBlock(context, ref inputStmts);
            DependencyGraph g = new DependencyGraph(context, inputStmts, ignoreMissingNodes: true, ignoreRequirements: false, deleteCancels: true);
            if (debug && inputStmts.Count < 100)
                SchedulingTransform.DrawOffsetEdges(g);
            // collect the set of forward and backward offset edges
            Dictionary<IVariableDeclaration, SerialLoopInfo> loopInfoOfVariable = new Dictionary<IVariableDeclaration, SerialLoopInfo>();
            List<SerialLoopInfo> infos = new List<SerialLoopInfo>();
            Action<IVariableDeclaration, EdgeIndex> foundSpecialEdge = delegate (IVariableDeclaration loopVar, EdgeIndex edge)
             {
                 NodeIndex source = g.dependencyGraph.SourceOf(edge);
                 IStatement stmt = inputStmts[source];
                 Set<IStatement> stmts;
                 if (!specialStmts.TryGetValue(loopVar, out stmts))
                 {
                     stmts = new Set<IStatement>();
                     specialStmts.Add(loopVar, stmts);
                 }
                 stmts.Add(stmt);
                 if (!specialTransformedStmts.ContainsKey(stmt))
                 {
                     specialTransformedStmts.Add(stmt, new Dictionary<Set<IVariableDeclaration>, IStatement>());
                 }
             };
            foreach (KeyValuePair<EdgeIndex, IOffsetInfo> entry in g.OffsetIndices)
            {
                EdgeIndex edge = entry.Key;
                IOffsetInfo offsetIndices = entry.Value;
                foreach (var entry2 in offsetIndices)
                {
                    IVariableDeclaration loopVar = entry2.loopVar;
                    int offset = entry2.offset;
                    SerialLoopInfo info;
                    if (!loopInfoOfVariable.TryGetValue(loopVar, out info))
                    {
                        info = new SerialLoopInfo();
                        info.loopVar = loopVar;
                        infos.Add(info);
                        loopInfoOfVariable[loopVar] = info;
                    }
                    if (offset > 0)
                    {
                        info.backwardEdges.Add(edge);
                        if (info.forwardEdges.Contains(edge))
                            foundSpecialEdge(loopVar, edge);
                    }
                    else if (offset < 0)
                    {
                        info.forwardEdges.Add(edge);
                        if (info.backwardEdges.Contains(edge))
                            foundSpecialEdge(loopVar, edge);
                    }
                }
            }
            // TODO: use LoopAnalysisTransform
            // determine the set of nodes in each loop
            for (int i = 0; i < infos.Count; i++)
            {
                SerialLoopInfo info = infos[i];
                // find the statements looping over this variable
                this.currentLoopVar = info.loopVar;
                this.containersOfLoopVar = null;
                foreach (NodeIndex node in g.dependencyGraph.Nodes)
                {
                    // analyze each statement for the loopVar
                    this.convertedStmtHasLoopVar = false;
                    // will set containersOfLoopVar
                    ConvertStatement(inputStmts[node]);
                    if (this.convertedStmtHasLoopVar)
                        info.nodesInLoop.Add(node);
                }
            }
            // sort infos by loop priority
            SchedulingTransform.SortByLoopPriority(context, infos);
            // attach group information for SchedulingTransform
            SerialSchedulingInfo ssinfo = new SerialSchedulingInfo();
            ssinfo.loopInfos.AddRange(infos);
            IWhileStatement parent = context.FindAncestor<IWhileStatement>();
            context.OutputAttributes.Set(parent, ssinfo);
            SplitForwardBackward(inputStmts, outputStmts, g, g.dependencyGraph.Nodes, infos, 0, new List<IStatement>());
            if (firstIterPostBlock != null)
                outputStmts.Add(firstIterPostBlock);
        }

        /// <summary>
        /// Create forward/backward composites for all loop variables >= loopIndex
        /// </summary>
        /// <param name="inputStmts"></param>
        /// <param name="outputStmts"></param>
        /// <param name="g"></param>
        /// <param name="nodes"></param>
        /// <param name="infos"></param>
        /// <param name="loopIndex"></param>
        /// <param name="containerPrefix">The containers that have already been grouped.</param>
        private void SplitForwardBackward(IReadOnlyList<IStatement> inputStmts, ICollection<IStatement> outputStmts, DependencyGraph g, IEnumerable<NodeIndex> nodes,
                                          List<SerialLoopInfo> infos, int loopIndex, List<IStatement> containerPrefix)
        {
            if (loopIndex >= infos.Count)
            {
                // start a new context
                Dictionary<IStatement, IStatement> transformedStmts = new Dictionary<IStatement, IStatement>(new IdentityComparer<IStatement>());
                if (transformedStmtsInContext.ContainsKey(outputStmts))
                    throw new Exception("Internal: context transformed twice");
                transformedStmtsInContext[outputStmts] = transformedStmts;
                var key = (Set<IVariableDeclaration>)loopVarsToReverse.Clone();
                foreach (NodeIndex node in nodes)
                {
                    IStatement ist = inputStmts[node];
                    IStatement st = ConvertTopLevelStatement(ist, containerPrefix);
                    outputStmts.Add(st);
                    transformedStmts.Add(ist, st);
                    reversedLoopVarsInStmt.Add(st, key);
                    Dictionary<Set<IVariableDeclaration>, IStatement> dict;
                    if (specialTransformedStmts.TryGetValue(ist, out dict))
                    {
                        dict.Add(key, st);
                    }
                }
                return;
            }
            // algorithm for two offset loop vars:
            // delete forward i, forward j, backward j edges
            // remaining SCC is backward i nodes
            // delete backward i nodes and toposort, giving forward i SCC
            // in forward i SCC, delete forward i edges and toposort, giving j SCC
            // - in j SCC, delete forward j edges - remaining SCC is backward j nodes
            // - delete backward j nodes and toposort, giving forward j SCC
            // create a composite statement for each forward SCC and backward SCC
            SerialLoopInfo info = infos[loopIndex];
            IVariableDeclaration loopVar = info.loopVar;
            if (debug)
                Debug.WriteLine("ForwardBackwardTransform: splitting on " + loopVar.Name);
            // find the statements looping over this variable
            Set<NodeIndex> nodesInLoop = Set<NodeIndex>.Intersection(info.nodesInLoop, nodes);
            if (nodesInLoop.Count == 0)
            {
                SplitForwardBackward(inputStmts, outputStmts, g, nodes, infos, loopIndex + 1, containerPrefix);
                return;
            }
            // graph2 is a dynamic subset of g according to isDeleted
            IndexedProperty<EdgeIndex, bool> isDeleted = g.edgeData.CreateEdgeData<bool>(true);
            DirectedGraphFilter<NodeIndex, EdgeIndex> graph2 = new DirectedGraphFilter<NodeIndex, EdgeIndex>(g.dependencyGraph, edge => !isDeleted[edge] && !g.isDeleted[edge]);
            StrongComponents2<NodeIndex> sccBack = new StrongComponents2<NodeIndex>(graph2.SourcesOf, graph2);
            // sccNodes contains all nodes that have been put into any composite
            Set<NodeIndex> sccNodes = new Set<EdgeIndex>();
            Set<NodeIndex> currentBlock = new Set<NodeIndex>();
            sccBack.AddNode += delegate (NodeIndex node)
            {
                currentBlock.Add(node);
            };
            sccBack.BeginComponent += delegate ()
            {
                currentBlock = new Set<NodeIndex>();
            };
            sccBack.EndComponent += delegate ()
            {
                // check that the block has some offset edges on loopVar
                bool isForwardLoop = !loopVarsToReverse.Contains(loopVar);
                bool hasOffsetEdge = BlockHasOffsetEdge(g, info, currentBlock, isForwardLoop);
                if (!hasOffsetEdge)
                    return;
                if (debug) // for debugging
                {
                    Debug.WriteLine("found block:");
                    foreach (NodeIndex node in currentBlock)
                    {
                        Debug.WriteLine("{0} {1}", node, g.NodeToShortString(node));
                    }
                }
                // put the nodes into a Set for fast membership checking
                Set<IStatement> originalStmts = new Set<IStatement>(new IdentityComparer<IStatement>());
                foreach (NodeIndex node in currentBlock)
                {
                    originalStmts.Add(inputStmts[node]);
                }
                CheckSequentialUpdates(inputStmts, g, loopVar, currentBlock, originalStmts);
                CheckLoopMerging(inputStmts, loopVar, currentBlock, isForwardLoop);
                if (!this.compiler.AllowSerialInitialisers)
                {
                    AddRequirementsToBlock(g, currentBlock, sccNodes);
                }
                sccNodes.AddRange(currentBlock);
                if (expandSequentialLoops)
                {
                    StackFrame frame = BuildStackFrame(inputStmts, g, nodesInLoop, currentBlock);
                    stackFrames.Push(frame);
                }
                var extraPrefix = GetContainerPrefix(originalStmts, containerPrefix, loopVar);
                if (extraPrefix == null)
                {
                    Error($"Variable {loopVar.Name} has inconsistent loop nesting.  Cannot construct a sequential loop.");
                    return;
                }
                List<IStatement> newContainerPrefix = new List<IStatement>(containerPrefix);
                newContainerPrefix.AddRange(extraPrefix);
                IWhileStatement outerWhile = Builder.FusedBlockStatement(GetContainerDescription(extraPrefix[0]));
                IWhileStatement innerWhile = outerWhile;
                for (int i = 1; i < extraPrefix.Count; i++)
                {
                    // there are containers in between the previous sequential loop and this one.
                    // we need to create nested "while" loops to represent these containers.
                    IWhileStatement newWhile = Builder.FusedBlockStatement(GetContainerDescription(extraPrefix[i]));
                    innerWhile.Body.Statements.Add(newWhile);
                    innerWhile = newWhile;
                }
                SplitForwardBackward(inputStmts, innerWhile.Body.Statements, g, currentBlock, infos, loopIndex + 1, newContainerPrefix);
                if (expandSequentialLoops)
                    stackFrames.Pop();
                DependencyInformation blockDeps = GetBlockDependencies(innerWhile.Body.Statements, originalStmts);
                context.OutputAttributes.Set(outerWhile, blockDeps);
                IWhileStatement iws = outerWhile;
                for (int i = 1; i < extraPrefix.Count; i++)
                {
                    iws = (IWhileStatement)iws.Body.Statements[0];
                    context.OutputAttributes.Set(iws, blockDeps);
                }
                // in JaggedChainsTest, this attaches info to the wrong block, but it doesn't seem to cause problems.
                context.OutputAttributes.Set(outerWhile, info);
                outputStmts.Add(outerWhile);
            };
            if (info.backwardEdges.Count > 0)
            {
                if (expandSequentialLoops)
                {
                    foreach (EdgeIndex edge in g.dependencyGraph.Edges)
                    {
                        isDeleted[edge] = false;
                    }
                }
                else
                {
                    // delete all edges except between statements in this loop
                    foreach (NodeIndex node in nodesInLoop)
                    {
                        foreach (EdgeIndex edge in g.dependencyGraph.EdgesOutOf(node))
                        {
                            if (g.isDeleted[edge] || g.diode[edge])
                                continue;
                            NodeIndex target = g.dependencyGraph.TargetOf(edge);
                            if (nodesInLoop.Contains(target))
                            {
                                isDeleted[edge] = false;
                            }
                        }
                    }
                }
                // delete offset edges of loop vars that have already been processed
                for (int i = 0; i < loopIndex; i++)
                {
                    SerialLoopInfo info2 = infos[i];
                    foreach (EdgeIndex edge in info2.forwardEdges)
                        isDeleted[edge] = true;
                    foreach (EdgeIndex edge in info2.backwardEdges)
                        isDeleted[edge] = true;
                }
                // delete forward edges and find an SCC for the backward edges
                foreach (EdgeIndex edge in info.forwardEdges)
                {
                    if (!info.backwardEdges.Contains(edge))
                        isDeleted[edge] = true;
                }
                loopVarsToReverse.Add(loopVar);
                if (debug) // for debugging
                {
                    Debug.WriteLine($"backward loop for {loopVar.Name}");
                    foreach (EdgeIndex edge in g.dependencyGraph.Edges)
                    {
                        if (isDeleted[edge])
                            Debug.WriteLine("deleting " + EdgeToString(g, edge));
                    }
                    Debug.WriteLine(g.dependencyGraph);
                    Debug.WriteLine("searching:");
                    foreach (NodeIndex node in nodesInLoop)
                    {
                        Debug.WriteLine("{0} {1}", node, g.NodeToShortString(node));
                    }
                }
                sccBack.SearchFrom(nodesInLoop);
                loopVarsToReverse.Remove(loopVar);
                isDeleted.Clear();
                sccBack.Clear();
            }
            if (info.forwardEdges.Count > 0)
            {
                if (expandSequentialLoops)
                {
                    foreach (EdgeIndex edge in g.dependencyGraph.Edges)
                    {
                        isDeleted[edge] = false;
                    }
                }
                else
                {
                    // delete all edges except between statements in this loop
                    foreach (NodeIndex node in nodesInLoop)
                    {
                        foreach (EdgeIndex edge in g.dependencyGraph.EdgesOutOf(node))
                        {
                            NodeIndex target = g.dependencyGraph.TargetOf(edge);
                            if (nodesInLoop.Contains(target))
                            {
                                isDeleted[edge] = false;
                            }
                        }
                    }
                }
                // delete offset edges of loop vars that have already been processed
                for (int i = 0; i < loopIndex; i++)
                {
                    SerialLoopInfo info2 = infos[i];
                    foreach (EdgeIndex edge in info2.forwardEdges)
                        isDeleted[edge] = true;
                    foreach (EdgeIndex edge in info2.backwardEdges)
                        isDeleted[edge] = true;
                }
                // delete backward edges and find an SCC for the forward edges
                foreach (EdgeIndex edge in info.backwardEdges)
                {
                    if (!info.forwardEdges.Contains(edge))
                        isDeleted[edge] = true;
                }
                if (debug) // for debugging
                {
                    Debug.WriteLine($"forward loop for {info.loopVar.Name}");
                    foreach (EdgeIndex edge in g.dependencyGraph.Edges)
                    {
                        if (isDeleted[edge])
                            Debug.WriteLine("deleting " + EdgeToString(g, edge));
                    }
                    Debug.WriteLine(g.dependencyGraph);
                    Debug.WriteLine("searching:");
                    foreach (NodeIndex node in nodesInLoop)
                    {
                        Debug.WriteLine("{0} {1}", node, g.NodeToShortString(node));
                    }
                }
                sccBack.SearchFrom(nodesInLoop);
            }

            List<NodeIndex> remainingNodes = new List<NodeIndex>();
            foreach (NodeIndex node in nodes)
            {
                if (!sccNodes.Contains(node))
                {
                    remainingNodes.Add(node);
                }
            }
            SplitForwardBackward(inputStmts, outputStmts, g, remainingNodes, infos, loopIndex + 1, containerPrefix);
        }

        private IExpression GetContainerDescription(IStatement container)
        {
            if (container is IForStatement)
            {
                IForStatement ifs = (IForStatement)container;
                return Builder.VarRefExpr(Recognizer.LoopVariable(ifs));
            }
            else if (container is IConditionStatement)
            {
                IConditionStatement ics = (IConditionStatement)container;
                return ics.Condition;
            }
            else throw new NotSupportedException(container.ToString());
        }

        private StackFrame BuildStackFrame(IReadOnlyList<IStatement> inputStmts, DependencyGraph g, Set<NodeIndex> nodesInLoop, Set<NodeIndex> currentBlock)
        {
            // construct a new StackFrame
            StackFrame frame = new StackFrame();
            frame.containers = this.containersOfLoopVar;
            foreach (NodeIndex node in currentBlock)
            {
                foreach (EdgeIndex edge in g.dependencyGraph.EdgesOutOf(node))
                {
                    if (g.isDeleted[edge])
                        continue;
                    NodeIndex target = g.dependencyGraph.TargetOf(edge);
                    if (!currentBlock.Contains(target))
                        continue;
                    if (nodesInLoop.Contains(node) && !nodesInLoop.Contains(target))
                    {
                        IStatement sourceSt = inputStmts[node];
                        // mine replacements from the lhs of sourceSt
                        foreach(IExpression expr in Recognizer.GetTargets(sourceSt))
                        {
                            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
                            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
                            var indices = Recognizer.GetIndices(expr);
                            for (int depth = 0; depth < indices.Count; depth++)
                            {
                                if (vi.indexVars.Count > depth)
                                {
                                    for (int i = 0; i < indices[depth].Count; i++)
                                    {
                                        IVariableDeclaration indexVar = vi.indexVars[depth][i];
                                        if (indexVar != null)
                                        {
                                            IExpression key = Builder.VarRefExpr(indexVar);
                                            IExpression value = indices[depth][i];
                                            if (frame.replacements.ContainsKey(key))
                                            {
                                                if (!frame.replacements[key].Equals(value))
                                                    Error(string.Format("conflicting replacements for {0}: {1} and {2}", key, frame.replacements[key], value));
                                            }
                                            else if (!key.Equals(value))
                                                frame.replacements.Add(key, value);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (debug)
            {
                Debug.WriteLine("frame:");
                Debug.WriteLine(StringUtil.ToString(frame.replacements));
            }
            return frame;
        }

        /// <summary>
        /// Get the index of the container (where outermost is 0) that loops over loopVar.
        /// </summary>
        /// <param name="containers"></param>
        /// <param name="loopVar"></param>
        /// <returns></returns>
        private int GetContainerIndex(IList<IStatement> containers, IVariableDeclaration loopVar)
        {
            for (int i = 0; i < containers.Count; i++)
            {
                var container = containers[i];
                if (container is IForStatement)
                {
                    IForStatement ifs = (IForStatement)container;
                    if (Recognizer.LoopVariable(ifs) == loopVar)
                        return i;
                }
            }
            return -1;
        }

        /// <summary>
        /// Get the containers that must precede the loop over loopVar, followed by the loop itself.
        /// </summary>
        /// <param name="stmts"></param>
        /// <param name="excludedContainers"></param>
        /// <param name="loopVar"></param>
        /// <returns>null if there is no common prefix</returns>
        private List<IStatement> GetContainerPrefix(ICollection<IStatement> stmts, List<IStatement> excludedContainers, IVariableDeclaration loopVar)
        {
            List<IStatement> commonPrefix = null;
            foreach (var stmt in stmts)
            {
                List<IStatement> containers = new List<IStatement>();
                var innerStmts = LoopMergingTransform.UnwrapStatement(stmt, containers);
                int loopIndex = GetContainerIndex(containers, loopVar);
                if (loopIndex < 0)
                    throw new Exception($"Statement does not loop over {loopVar}: {stmt}");
                // find all containers that must precede loopVar
                List<IStatement> mustPrecede = new List<IStatement>();
                IForStatement ifs = (IForStatement)containers[loopIndex];
                // initialize affectingVariables to the variables in the loop bound expressions.
                Set<IVariableDeclaration> affectingVariables = Set<IVariableDeclaration>.FromEnumerable(
                    Recognizer.GetVariables(Recognizer.LoopStartExpression(ifs)).Concat(
                    Recognizer.GetVariables(ifs.Condition)
                    ));
                if (affectingVariables.Count > 0)
                {
                    // we only need to consider outer containers.
                    for (int i = loopIndex - 1; i >= 0; i--)
                    {
                        IStatement container = containers[i];
                        // modifies affectingVariables
                        if (LoopMergingTransform.ContainerAffectsVariables(container, affectingVariables) &&
                            !Containers.ListContains(excludedContainers, container))
                        {
                            mustPrecede.Add(container);
                        }
                    }
                }
                mustPrecede.Add(ifs);
                if (commonPrefix == null)
                {
                    commonPrefix = mustPrecede;
                }
                else
                {
                    if (mustPrecede.Count != commonPrefix.Count)
                        return null;
                    foreach (var container in mustPrecede)
                    {
                        if (!Containers.ListContains(commonPrefix, container))
                            return null;
                    }
                }
            }
            return commonPrefix;
        }

        private IStatement PermuteContainers(IStatement stmt, IList<IStatement> containerPrefix)
        {
            List<IStatement> containers = new List<IStatement>();
            var innerStmts = LoopMergingTransform.UnwrapStatement(stmt, containers);
            List<IStatement> newContainers = new List<IStatement>();
            newContainers.AddRange(containerPrefix);
            foreach (var container in containers)
            {
                if (!Containers.ListContains(containerPrefix, container))
                    newContainers.Add(container);
            }
            return Containers.WrapWithContainers(innerStmts, newContainers)[0];
        }

        // check that LoopMergingInfo allows the statements in currentBlock to be merged
        private void CheckLoopMerging(IReadOnlyList<IStatement> inputStmts, IVariableDeclaration loopVar, Set<NodeIndex> currentBlock, bool isForwardLoop)
        {
            Set<int> checkedStmts = new Set<int>();
            foreach (NodeIndex node in currentBlock)
            {
                int stmtIndex = loopMergingInfo.GetIndexOf(inputStmts[node]);
                int conflict = loopMergingInfo.GetConflictingStmt(checkedStmts, stmtIndex, loopVar, isForwardLoop);
                if (conflict != -1)
                {
                    // TEMPORARY
                    if (!expandSequentialLoops)
                    {
                        IStatement conflictStmt = loopMergingInfo.GetStatement(conflict);
                        Error("Cannot construct serial schedule: cannot merge statement with loop over " + loopVar.Name + ": " +
                            inputStmts[node] + " because of " + conflictStmt);
                    }
                }
                checkedStmts.Add(stmtIndex);
            }
        }

        // emit warnings for increment statements that are not sequential
        private void CheckSequentialUpdates(IReadOnlyList<IStatement> inputStmts, DependencyGraph g, IVariableDeclaration loopVar, Set<NodeIndex> currentBlock, Set<IStatement> originalStmts)
        {
            // collect all increment statements in the block
            Set<NodeIndex> incrementStmts = new Set<NodeIndex>();
            foreach (NodeIndex node in currentBlock)
            {
                IStatement stmt = inputStmts[node];
                var attr = context.InputAttributes.Get<IncrementStatement>(stmt);
                if (attr != null && attr.loopVar == loopVar)
                    incrementStmts.Add(node);
            }
            // check that the dependencies of each increment statement are also in the block
            if (incrementStmts.Count > 0)
            {
                foreach (NodeIndex node in incrementStmts)
                {
                    IStatement stmt = inputStmts[node];
                    DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(stmt);
                    if (di == null)
                    {
                        Error("Dependency information not found for statement: " + stmt);
                        continue;
                    }
                    foreach (IStatement dep in di.Dependencies.Where(dep => 
                        !di.ContainerDependencies.Contains(dep) && 
                        !context.InputAttributes.Has<Initializer>(dep) && 
                        !originalStmts.Contains(dep)))
                    {
                        // dependency is not in the block
                        Warning($"Update for {g.NodeToShortString(node)} is not sequential.  Statement outside of the loop: {dep}");
                        break;
                    }
                }
            }
        }

        private void AddRequirementsToBlock(DependencyGraph g, Set<NodeIndex> currentBlock, Set<NodeIndex> previousNodes)
        {
            // look for a node in the block with a requirement in this block and an earlier block.
            bool foundSplitRequirement = false;
            foreach (var node in currentBlock)
            {
                if (!previousNodes.Contains(node)) continue;
                NodeIndex requiredNodeInBlock = -1;
                NodeIndex requiredNodeInEarlierBlock = -1;
                foreach (var edge in g.dependencyGraph.EdgesInto(node))
                {
                    if (g.isRequired[edge])
                    {
                        NodeIndex source = g.dependencyGraph.SourceOf(edge);
                        if (currentBlock.Contains(source)) requiredNodeInBlock = source;
                        else if (previousNodes.Contains(source)) requiredNodeInEarlierBlock = source;
                    }
                }
                if (requiredNodeInBlock >= 0 && requiredNodeInEarlierBlock >= 0)
                {
                    if (debug)
                    {
                        Trace.WriteLine($"node {node} requires {requiredNodeInBlock} in block and {requiredNodeInEarlierBlock} in earlier block");
                        //Trace.WriteLine(g.NodeToString(node));
                        //Trace.WriteLine(g.NodeToString(requiredNodeInBlock));
                        //Trace.WriteLine(g.NodeToString(requiredNodeInEarlierBlock));
                    }
                    foundSplitRequirement = true;
                }
            }
            if (foundSplitRequirement)
            {
                // must search from all nodes since there may be other required sources in previous blocks.
                DepthFirstSearch<NodeIndex> dfsRequired = new DepthFirstSearch<NodeIndex>(node => g.dependencyGraph.EdgesInto(node)
                    .Where(edge => g.isRequired[edge])
                    .Select(edge => g.dependencyGraph.SourceOf(edge))
                    .Where(source => previousNodes.Contains(source)), g.dependencyGraph);
                Set<NodeIndex> nodesToAdd = new Set<NodeIndex>();
                dfsRequired.DiscoverNode += nodesToAdd.Add;
                dfsRequired.SearchFrom(currentBlock);
                if(debug)
                    Trace.WriteLine($"adding {nodesToAdd - currentBlock}");
                currentBlock.AddRange(nodesToAdd);
            }
        }

        private static bool BlockHasOffsetEdge(DependencyGraph g, SerialLoopInfo info, Set<NodeIndex> currentBlock, bool isForwardLoop)
        {
            bool hasOffsetEdge = false;
            foreach (NodeIndex node in currentBlock)
            {
                foreach (EdgeIndex edge in g.dependencyGraph.EdgesOutOf(node))
                {
                    if (g.isDeleted[edge])
                        continue;
                    NodeIndex target = g.dependencyGraph.TargetOf(edge);
                    if (!currentBlock.Contains(target))
                        continue;
                    if (isForwardLoop)
                    {
                        if (info.forwardEdges.Contains(edge))
                        {
                            hasOffsetEdge = true;
                            break;
                        }
                    }
                    else
                    {
                        if (info.backwardEdges.Contains(edge))
                        {
                            hasOffsetEdge = true;
                            break;
                        }
                    }
                }
                if (hasOffsetEdge)
                    break;
            }
            return hasOffsetEdge;
        }

        private string EdgeToString(DependencyGraph g, EdgeIndex edge)
        {
            return "(" + g.dependencyGraph.SourceOf(edge) + "," + g.dependencyGraph.TargetOf(edge) + ")";
        }

        /// <summary>
        /// Construct DependencyInformation for a block of transformed statements, excluding dependencies on statements in the same block
        /// </summary>
        /// <param name="newStmts">Transformed statements</param>
        /// <param name="originalStmts">The same statements, before transformation</param>
        /// <returns>All external dependencies of the transformed statements</returns>
        private DependencyInformation GetBlockDependencies(IEnumerable<IStatement> newStmts, Set<IStatement> originalStmts)
        {
            DependencyInformation blockDeps = new DependencyInformation();
            foreach (IStatement ist in newStmts)
            {
                DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                foreach (KeyValuePair<IStatement, DependencyType> entry in di.dependencyTypeOf)
                {
                    IStatement source = entry.Key;
                    if (originalStmts.Contains(source))
                        continue;
                    DependencyType type = entry.Value;
                    // exclude initializers
                    type &= ~DependencyType.Overwrite;
                    // exclude SkipIfUniform
                    type &= ~DependencyType.SkipIfUniform;
                    type &= ~DependencyType.Fresh;
                    type &= ~DependencyType.Trigger;
                    blockDeps.Add(type, source);
                }
            }
            blockDeps.RemoveAll(originalStmts.Contains);
            return blockDeps;
        }

        private void ForEachEdgeNotInSet(IDirectedGraph<NodeIndex, EdgeIndex> graph, ICollection<NodeIndex> nodes, Action<EdgeIndex> action)
        {
            foreach (NodeIndex node in nodes)
            {
                foreach (EdgeIndex edge in graph.EdgesInto(node))
                {
                    NodeIndex source = graph.SourceOf(edge);
                    if (!nodes.Contains(source))
                        action(edge);
                }
                foreach (EdgeIndex edge in graph.EdgesOutOf(node))
                {
                    NodeIndex target = graph.TargetOf(edge);
                    if (!nodes.Contains(target))
                        action(edge);
                }
            }
        }
    }

    internal class SerialLoopInfo : ICompilerAttribute
    {
        public IVariableDeclaration loopVar;
        public Set<EdgeIndex> forwardEdges = new Set<EdgeIndex>();
        public Set<EdgeIndex> backwardEdges = new Set<EdgeIndex>();
        public Set<NodeIndex> nodesInLoop = new Set<EdgeIndex>();

        public override string ToString()
        {
            return "SerialLoopInfo(" + loopVar + ")";
        }
    }

    internal class SerialSchedulingInfo : ICompilerAttribute
    {
        public List<SerialLoopInfo> loopInfos = new List<SerialLoopInfo>();

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder("SerialSchedulingInfo:");
            foreach (SerialLoopInfo info in loopInfos)
            {
                sb.Append(" ");
                sb.Append(info);
            }
            return sb.ToString();
        }
    }
}