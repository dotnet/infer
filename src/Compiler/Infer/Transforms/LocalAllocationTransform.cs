// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Diagnostics;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Moves allocations inside while loops when possible.
    /// Input code can have while(true) wrappers to indicate statements that should be grouped together.
    /// Nested wrappers are removed on output, leaving only top-level wrappers.
    /// </summary>
    internal class LocalAllocationTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "LocalAllocationTransform";
            }
        }

        internal static bool debug;
        private readonly ModelCompiler compiler;
        private LoopMergingInfo loopMergingInfo;
        private DependencyGraph2 g;
        private IndexedGraph dependencyGraph;
        private List<IStatement> nodes;
        private Dictionary<IStatement, int> indexOfNode;
        List<IStatement[]> containersOfNode = new List<IStatement[]>();
        IndexedProperty<EdgeIndex, bool> isWriteAfterRead;
        private GroupGraph groupGraph;
        private Set<IVariableDeclaration> loopVarsToReverse = new Set<IVariableDeclaration>();
        Replacements globalReplacements = new Replacements();

        internal LocalAllocationTransform(ModelCompiler compiler)
        {
            this.compiler = compiler;
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            IBlockStatement debugBlock;
            if (context.trackTransform)
            {
                debugBlock = Builder.BlockStmt();
                ITypeDeclaration itdOut = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
                context.OutputAttributes.Add(itdOut, new DebugInfo()
                {
                    Transform = this,
                    Name = "debug",
                    Value = debugBlock
                });
            }
            else
                debugBlock = null;
            IMethodDeclaration imd = context.FindAncestor<IMethodDeclaration>();
            loopMergingInfo = context.InputAttributes.Get<LoopMergingInfo>(imd);
            List<IWhileStatement> whileStatements = new List<IWhileStatement>();
            Set<IStatement> initializerStatements = new Set<IStatement>(ReferenceEqualityComparer<IStatement>.Instance);
            List<int> whileNumberOfNode = new List<int>();
            // the number of containers that must be shared with the previous statement.
            List<int> sharedContainerCountOfNode = new List<NodeIndex>();
            List<IStatement> containerSet = new List<IStatement>();
            if (true)
            {
                // the code may have multiple while(true) loops, however these must be disjoint.
                // therefore we treat 'while' as one container, but give each loop a different 'while number'.
                int outerWhileCount = 0;
                int currentOuterWhileNumber = 0;
                // the current number of open containers that must be shared.
                int sharedContainerCount = 0;
                // the number of currently open shared containers that were also open on the previous statement.
                int previousSharedContainerCount = 0;
                IConditionStatement currentFirstIterPostBlock = null;

                // build the dependency graph
                g = new DependencyGraph2(context, inputs, DependencyGraph2.BackEdgeHandling.Reverse,
                                             delegate (IWhileStatement iws)
                                             {
                                                 ForEachInitializerStatement(iws, initializerStatements.Add);
                                                 if (sharedContainerCount == 0)
                                                 {
                                                     outerWhileCount++;
                                                     whileStatements.Add(iws);
                                                     currentOuterWhileNumber = outerWhileCount;
                                                 }
                                                 sharedContainerCount++;
                                             },
                                             delegate (IWhileStatement iws)
                                             {
                                                 sharedContainerCount--;
                                                 previousSharedContainerCount = System.Math.Min(previousSharedContainerCount, sharedContainerCount);
                                                 if (sharedContainerCount == 0)
                                                 {
                                                     currentOuterWhileNumber = 0;
                                                 }
                                             },
                                             delegate (IConditionStatement ics)
                                             {
                                                 currentFirstIterPostBlock = ics;
                                             },
                                             delegate (IConditionStatement ics)
                                             {
                                                 currentFirstIterPostBlock = null;
                                             },
                                             delegate (IStatement ist, int targetIndex)
                                             {
                                                 if (currentFirstIterPostBlock != null)
                                                 {
                                                     // this statement is in a first iter post block.
                                                     // its initializers must not be placed into the while loop.
                                                     ForEachInitializerStatement(ist, initializerStatements.Add);
                                                 }
                                                 ConvertStatement(ist);
                                                 int whileNumber = currentOuterWhileNumber;
                                                 whileNumberOfNode.Add(whileNumber);
                                                 IWhileStatement iws;
                                                 if (whileNumber > 0)
                                                     iws = whileStatements[whileNumber - 1];
                                                 else
                                                     iws = null;
                                                 containersOfNode.Add(GetContainersOfStatement(ist, iws, currentFirstIterPostBlock, containerSet));
                                                 sharedContainerCountOfNode.Add(previousSharedContainerCount);
                                                 previousSharedContainerCount = sharedContainerCount;
                                             });
            }
            nodes = g.nodes;
            dependencyGraph = g.dependencyGraph;
            // statements are not necessarily unique.
            // indexOfNode gives the index of the last copy of a statement.
            indexOfNode = g.indexOfNode;
            isWriteAfterRead = g.isWriteAfterRead;

            // put each node in its own group hierarchy by prefix length
            List<NodeIndex> groupOf = new List<NodeIndex>(dependencyGraph.Nodes.Count);
            int firstGroup = dependencyGraph.Nodes.Count;
            // allocate space in the list
            for (int node = 0; node < dependencyGraph.Nodes.Count; node++)
            {
                groupOf.Add(-1);
            }
            Stack<int> groupStack = new Stack<int>();
            groupStack.Push(-1);
            // create the remaining groups
            List<int> groupOfWhileStatement = new List<NodeIndex>();
            for (int node = 0; node < dependencyGraph.Nodes.Count; node++)
            {
                int sharedContainerCount = sharedContainerCountOfNode[node];
                int containerCount = containersOfNode[node].Length;
                while (groupStack.Count - 1 > sharedContainerCount)
                    groupStack.Pop();
                NodeIndex parentGroup = groupStack.Peek();
                // create the outermost groups first
                for (int i = sharedContainerCount; i < containerCount; i++)
                {
                    NodeIndex group = groupOf.Count;
                    groupOf.Add(parentGroup);
                    containersOfNode.Add(containersOfNode[node].Take(i + 1).ToArray());
                    parentGroup = group;
                    groupStack.Push(group);
                    if (i == 0 && containersOfNode[node][i] is IWhileStatement)
                    {
                        IWhileStatement iws = (IWhileStatement)containersOfNode[node][i];
                        int whileNumber = whileStatements.IndexOf(iws);
                        while (groupOfWhileStatement.Count < whileNumber)
                        {
                            groupOfWhileStatement.Add(-1);
                        }
                        groupOfWhileStatement.Add(group);
                    }
                }
                groupOf[node] = parentGroup;
            }

            Set<NodeIndex> movedNodes = new Set<NodeIndex>();
            bool moveIntoWhileLoops = compiler.UseLocals && compiler.OptimiseInferenceCode; 
            if (moveIntoWhileLoops)
            {
                // If a statement is only used in a while loop, move it inside the loop (if it is safe to do so).
                // InitializerStatements could be instead filled in by g.BackEdges.Keys.
                // After initializerStatements is filled in, determine the initializerTargets.
                Set<int> cannotMove = new Set<NodeIndex>();
                for (int node = 0; node < dependencyGraph.Nodes.Count; node++)
                {
                    IStatement ist = nodes[node];
                    DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                    if (di == null)
                    {
                        context.Error("statement is missing dependency information", null, ist);
                        return;
                    }
                    bool dependsOnInitializer = di.Dependencies.Any(initializerStatements.Contains);
                    if (dependsOnInitializer)
                        cannotMove.Add(node);
                    // Container dependencies must not be moved into the while loop, since this causes problems in IterativeProcessTransform.
                    // (Every statement that depends on it will inherit the parameterDeps of the while loop.) 
                    cannotMove.AddRange(di.ContainerDependencies.Select(dependency => indexOfNode[dependency]));
                }
                Set<int> usedAtDepthZero = new Set<NodeIndex>();

                // fill in whileNumberOfTargets
                IndexedProperty<NodeIndex, int> whileNumberOfTargets = dependencyGraph.CreateNodeData<int>();
                IndexedProperty<NodeIndex, bool> initializesStmtWithOutsideTarget = dependencyGraph.CreateNodeData<bool>();
                DepthFirstSearch<NodeIndex> dfsForward = new DepthFirstSearch<EdgeIndex>(dependencyGraph);
                dfsForward.FinishNode += delegate (NodeIndex node)
                {
                    // at this point, we know that all targets have been processed.
                    if (whileNumberOfNode[node] != 0)
                    {
                        whileNumberOfTargets[node] = whileNumberOfNode[node];
                        bool hasOutsideTarget = dependencyGraph.TargetsOf(node).Any(target =>
                            whileNumberOfNode[target] != whileNumberOfNode[node]) ||
                            initializesStmtWithOutsideTarget[node];
                        if (hasOutsideTarget || usedAtDepthZero.Contains(node))
                        {
                            // if a node has a target outside the while loop, then its initializers must remain outside the while loop.
                            ForEachInitializerStatement(nodes[node], ist =>
                            {
                                NodeIndex init = indexOfNode[ist];
                                initializesStmtWithOutsideTarget[init] = true;
                            });
                        }
                        bool hasForLoop = containersOfNode[node].Any(container => container is IForStatement);
                        if (!hasForLoop)
                        {
                            // If an array is ever used at depth zero (i.e. not indexed), then there is no benefit to moving it
                            // since it cannot be contracted.
                            var readAfterWriteEdges = dependencyGraph.EdgesInto(node).Where(edge => !g.isWriteAfterRead[edge]);
                            usedAtDepthZero.AddRange(readAfterWriteEdges.Select(dependencyGraph.SourceOf));
                        }
                    }
                    else if (!initializerStatements.Contains(nodes[node]) &&
                        !initializesStmtWithOutsideTarget[node] && !usedAtDepthZero.Contains(node) &&
                        !cannotMove.Contains(node))
                    {
                        int commonWhileNumber = 0;
                        bool firstTime = true;
                        foreach (var target in dependencyGraph.TargetsOf(node))
                        {
                            int whileNumber = whileNumberOfTargets[target];
                            if (firstTime)
                            {
                                commonWhileNumber = whileNumber;
                                firstTime = false;
                            }
                            else if (whileNumber != commonWhileNumber)
                            {
                                commonWhileNumber = 0;
                                if(debugBlock != null)
                                    debugBlock.Statements.Add(Builder.CommentStmt(NodeToShortString(node) + " used outside of while loop"));
                            }
                        }
                        whileNumberOfTargets[node] = commonWhileNumber;
                    }
                    else if (debugBlock != null)
                    {
                        StringBuilder sb = new StringBuilder();
                        if (usedAtDepthZero.Contains(node))
                            sb.Append(" usedAtDepthZero");
                        if (cannotMove.Contains(node))
                            sb.Append(" dependsOnInitializer or is a container dependency");
                        if (initializerStatements.Contains(nodes[node]))
                            sb.Append(" initializerStatement");
                        if (initializesStmtWithOutsideTarget[node])
                            sb.Append(" initializesStmtWithOutsideTarget");
                        debugBlock.Statements.Add(Builder.CommentStmt(NodeToShortString(node) + sb.ToString()));
                    }
                };
                dfsForward.SearchFrom(dependencyGraph.Nodes);

                // compute the common desired while number of each group's children.
                var whileNumberOfChildren = new Dictionary<NodeIndex, int>();
                int lastGroup = groupOf.Count;
                ForEachNodeBeforeItsGroup(firstGroup, lastGroup, node =>
                {
                    NodeIndex group = groupOf[node];
                    if (group != -1)
                    {
                        bool nodeIsGroup = (node >= firstGroup);
                        int desiredWhileNumber;
                        if (nodeIsGroup)
                            desiredWhileNumber = whileNumberOfChildren[node];
                        else if (whileNumberOfNode[node] == 0)
                            desiredWhileNumber = whileNumberOfTargets[node];
                        else
                            desiredWhileNumber = 0; // already in a while loop
                        int commonWhileNumber;
                        if (whileNumberOfChildren.TryGetValue(group, out commonWhileNumber))
                        {
                            if (desiredWhileNumber != commonWhileNumber)
                                whileNumberOfChildren[group] = 0;
                        }
                        else
                        {
                            whileNumberOfChildren[group] = desiredWhileNumber;
                        }
                    }
                });
                // change the containers of groups with a common while number.
                ForEachNodeAfterItsGroup(firstGroup, lastGroup, node =>
                {
                    bool nodeIsGroup = (node >= firstGroup);
                    NodeIndex group = groupOf[node];
                    int commonWhileNumber;
                    if (group != -1)
                    {
                        // inherit commonWhileNumber from the parent group
                        commonWhileNumber = whileNumberOfChildren[group];
                        if (nodeIsGroup)
                            whileNumberOfChildren[node] = commonWhileNumber;
                        else
                            whileNumberOfTargets[node] = commonWhileNumber;
                    }
                    else
                    {
                        commonWhileNumber = nodeIsGroup ? whileNumberOfChildren[node] : whileNumberOfTargets[node];
                    }
                    if (commonWhileNumber > 0)
                    {
                        IWhileStatement iws = whileStatements[commonWhileNumber - 1];
                        if (!context.InputAttributes.Has<HasOffsetIndices>(iws))
                        {
                            // add the while loop as a new container
                            containersOfNode[node] = AddFirst(containersOfNode[node], iws);
                            if (group == -1)
                                groupOf[node] = groupOfWhileStatement[commonWhileNumber - 1];
                            movedNodes.Add(node);
                        }
                    }
                });
            }

            bool makeReversedClones = compiler.UseLocals && compiler.OptimiseInferenceCode;
            if (makeReversedClones)
            {
                // If a variable is used in multiple independent loop contexts that don't interact, make them refer to different variables by cloning.
                // This increases the opportunities for optimization, particularly array contraction.
                // Example:
                //   for(i) x[i] = defn
                //   for(i) y[i] = f(x[i])
                //   forr(i) z[i] = f(x[i])
                // (forr denotes a reversed loop)
                // transforms into:
                //   for(i) x[i] = defn
                //   for(i) y[i] = f(x[i])
                //   forr(i) x2[i] = defn
                //   forr(i) z[i] = f(x2[i])
                // We must make these replacements in reverse topological order, since "defn" may refer to variables which can themselves be cloned.
                // We avoid cycles by prohibiting clones of back edge sources (and their allocations).
                // Example where cloning is prohibited:
                //   for(i) y[i] = init
                //   while {
                //     forr(i) x[i] = f(y[i])
                //     for(i) y[i] = g(x[i])
                //   }
                // Here y is a back edge source in the while loop, so it cannot be cloned.
                // Example where cloning is prohibited:
                //   for(i) {
                //     x[i] = f(y[i-1])
                //     y[i] = g(x[i])
                //   }
                // Here y->x is an offset dependency.  y cannot be cloned.  
                // x can be cloned as long as the clone is in the same loop or a later loop.
                // For simplicity, this case is excluded, i.e. x will not be cloned.
                // We also impose the restriction that a node can only be cloned if its allocations can be cloned.

                // If a statement is duplicated, then we can't reliably replace it in the DependencyInformation of its targets.
                // So we require all nodes to be unique.
                if (!CheckNodesAreUnique()) return;
                Set<NodeIndex> cannotBeCloned = new Set<NodeIndex>();
                cannotBeCloned.AddRange(initializerStatements.Select(ist => indexOfNode[ist]));
                if (debug && debugBlock != null)
                {
                    foreach (IStatement ist in initializerStatements)
                    {
                        StringBuilder sb = new StringBuilder();
                        var node = indexOfNode[ist];
                        sb.Append(NodeToShortString(node));
                        sb.Append(" cannot be cloned because initializer");
                        debugBlock.Statements.Add(Builder.CommentStmt(sb.ToString()));
                    }
                }
                Dictionary<NodeIndex, Set<IVariableDeclaration>> irreversibleLoopsOfNode = new Dictionary<NodeIndex, Set<IVariableDeclaration>>();
                foreach (var entry in g.backEdges)
                {
                    IStatement sourceSt = entry.Key;
                    // source of a back edge
                    NodeIndex source = indexOfNode[sourceSt];
                    bool hasTargetDifferentFromSource = false;
                    foreach (var target in entry.Value)
                    {
                        if (target == source) continue;
                        hasTargetDifferentFromSource = true;
                        // check if target has an offset dependency on sourceSt
                        DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(nodes[target]);
                        IOffsetInfo offsetInfo;
                        if (di.offsetIndexOf.TryGetValue(sourceSt, out offsetInfo))
                        {
                            Set<IVariableDeclaration> irreversibleLoops;
                            if (!irreversibleLoopsOfNode.TryGetValue(target, out irreversibleLoops))
                            {
                                irreversibleLoops = new Set<IVariableDeclaration>();
                                irreversibleLoopsOfNode.Add(target, irreversibleLoops);
                            }
                            foreach (var offset in offsetInfo)
                            {
                                irreversibleLoops.Add(offset.loopVar);
                            }
                        }
                    }
                    if (hasTargetDifferentFromSource)
                    {
                        cannotBeCloned.Add(source);
                        if (debug && debugBlock != null)
                        {
                            StringBuilder sb = new StringBuilder();
                            sb.Append(NodeToShortString(source));
                            sb.Append(" cannot be cloned because source of back edge");
                            debugBlock.Statements.Add(Builder.CommentStmt(sb.ToString()));
                        }
                    }
                }
                if (debug && debugBlock != null)
                {
                    foreach (var entry in irreversibleLoopsOfNode)
                    {
                        NodeIndex node = entry.Key;
                        StringBuilder sb = new StringBuilder();
                        sb.Append(NodeToShortString(node));
                        sb.Append(" cannot reverse loop ");
                        foreach (var loopVar in entry.Value)
                        {
                            sb.Append(loopVar.Name);
                            sb.Append(' ');
                        }
                        debugBlock.Statements.Add(Builder.CommentStmt(sb.ToString()));
                    }
                }
                // If a node cannot merge with a target, then there is no point in cloning it.
                foreach (var node in dependencyGraph.Nodes)
                {
                    var readAfterWriteEdges = dependencyGraph.EdgesOutOf(node).Where(edge => !g.isWriteAfterRead[edge]);
                    foreach (var target in readAfterWriteEdges.Select(dependencyGraph.TargetOf))
                    {
                        if (!AllWhileLoopsMatch(containersOfNode[node], containersOfNode[target]))
                        {
                            cannotBeCloned.Add(node);
                            if (debug && debugBlock != null)
                            {
                                StringBuilder sb = new StringBuilder();
                                sb.Append(NodeToShortString(node));
                                sb.Append(" cannot be cloned because cannot merge with while loop of target ");
                                sb.Append(NodeToShortString(target));
                                debugBlock.Statements.Add(Builder.CommentStmt(sb.ToString()));
                            }
                        }
                    }
                }
                // Propagate cannotBeCloned to all initializer sources and targets
                // TODO: This should only propagate to children by default.
                // It propagates to parents only if the node wants to be cloned, since this indicates that it is
                // not profitable to clone the parents.
                // The same applies to irreversibleLoopsOfNode.
                List<NodeIndex> toAdd = new List<NodeIndex>();
                DepthFirstSearch<NodeIndex> dfsOverwrite = new DepthFirstSearch<NodeIndex>(node =>
                    dependencyGraph.SourcesOf(node)
                    .Where(source => IsOverwrite(source, node))
                    .Concat(
                        dependencyGraph.TargetsOf(node)
                        .Where(target => IsOverwrite(node, target))
                        ), dependencyGraph);
                dfsOverwrite.DiscoverNode += toAdd.Add;
                dfsOverwrite.SearchFrom(cannotBeCloned);
                cannotBeCloned.AddRange(toAdd);
                if (debug && debugBlock != null)
                {
                    foreach (NodeIndex node in toAdd)
                    {
                        StringBuilder sb = new StringBuilder();
                        sb.Append(NodeToShortString(node));
                        sb.Append(" cannot be cloned because of overwrite");
                        debugBlock.Statements.Add(Builder.CommentStmt(sb.ToString()));
                    }
                }

                // for each node with targets, groups the targets by their loop directions.
                Dictionary<NodeIndex, CloneInfos> cloneInfosOfNode = new Dictionary<NodeIndex, CloneInfos>();
                // maps from (moved node, loop vars to reverse) to clone.
                Set<NodeIndex> clones = new Set<NodeIndex>();
                DepthFirstSearch<NodeIndex> dfsForward = new DepthFirstSearch<EdgeIndex>(dependencyGraph);
                dfsForward.FinishNode += delegate (NodeIndex node)
                {
                    // at this point, we know that all targets have been processed.
                    var containers = containersOfNode[node];
                    CloneInfos infos = new CloneInfos();
                    var readAfterWriteEdges = dependencyGraph.EdgesOutOf(node).Where(edge => !g.isWriteAfterRead[edge]);
                    if (readAfterWriteEdges.Count() == 0)
                    {
                        infos.targetsByReversedLoops.Add(new Set<IVariableDeclaration>(), new CloneInfo() { clone = node });
                    }
                    Set<IVariableDeclaration> irreversibleLoops;
                    irreversibleLoopsOfNode.TryGetValue(node, out irreversibleLoops);
                    // check for a target with a reversed loop
                    // the node may have some targets that were moved into the while loop and some that were originally in the loop.
                    // targets that were moved may have clones.  
                    foreach (var target in readAfterWriteEdges.Select(dependencyGraph.TargetOf))
                    {
                        ForEachClone(cloneInfosOfNode, target, (targetClone, reversedLoopsInClone) =>
                        {
                            bool targetOverwritesNode = IsOverwrite(node, target);
                            // Example 1:
                            //   for(i) x[i] = y
                            //   forr(i) x[i] = z
                            // This is an overwrite and reversedLoops = i
                            // Example 2:
                            //   x = y
                            //   for(i) x[i] = z
                            // This is an overwrite so we inherit reversedLoopsInClone
                            Set<IVariableDeclaration> commonReversedLoops = GetReversedLoops(containers, containersOfNode[target], reversedLoopsInClone, targetOverwritesNode);
                            if (cannotBeCloned.Contains(node)) commonReversedLoops.Clear();
                            if (irreversibleLoops != null) commonReversedLoops.Remove(irreversibleLoops);
                            CloneInfo info;
                            if (!infos.targetsByReversedLoops.TryGetValue(commonReversedLoops, out info))
                            {
                                info = new CloneInfo();
                                if (commonReversedLoops.Count == 0)
                                {
                                    info.clone = node;
                                }
                                else
                                {
                                    info.clone = AllocateNode(groupOf);
                                    clones.Add(info.clone);
                                }
                                infos.targetsByReversedLoops.Add(commonReversedLoops, info);
                            }
                            info.targets.Add(targetClone);
                        });
                    }
                    if (debug && debugBlock != null)
                    {
                        foreach (var entry in infos.targetsByReversedLoops)
                        {
                            StringBuilder sb = new StringBuilder();
                            sb.Append(NodeToShortString(node));
                            //sb.Append(" ");
                            //sb.Append(g.nodes[node]);
                            if (entry.Value.clone != node)
                            {
                                sb.Append(" cloned as ");
                                sb.Append(entry.Value.clone);
                            }
                            sb.Append(" has targets ");
                            foreach (var targetClone in entry.Value.targets)
                            {
                                sb.Append(targetClone);
                                sb.Append(" ");
                            }
                            debugBlock.Statements.Add(Builder.CommentStmt(sb.ToString()));
                        }
                    }
                    cloneInfosOfNode.Add(node, infos);
                };
                dfsForward.SearchFrom(dependencyGraph.Nodes);
                if (clones.Count > 0)
                {
                    // rebuild the graph
                    Dictionary<NodeIndex, Replacements> replacementsForStatement = new Dictionary<NodeIndex, Replacements>();
                    IndexedGraph dependencyGraph2 = new IndexedGraph(dependencyGraph.Nodes.Count);
                    dependencyGraph2.NodeCountIsConstant = false;
                    isWriteAfterRead = dependencyGraph2.CreateEdgeData(false);
                    bool copyWriteAfterReadEdges = true;
                    if (copyWriteAfterReadEdges)
                    {
                        Set<NodeIndex> excludedTargets = new Set<NodeIndex>();
                        foreach (NodeIndex node in dependencyGraph.Nodes)
                        {
                            CloneInfos info;
                            if (cloneInfosOfNode.TryGetValue(node, out info))
                            {
                                foreach (var entry in info.targetsByReversedLoops)
                                {
                                    if (entry.Key.Count > 0)
                                    {
                                        excludedTargets.AddRange(entry.Value.targets);
                                    }
                                }
                            }
                            foreach (EdgeIndex edge in dependencyGraph.EdgesOutOf(node))
                            {
                                NodeIndex target = dependencyGraph.TargetOf(edge);
                                if (!excludedTargets.Contains(target) && g.isWriteAfterRead[edge])
                                {
                                    EdgeIndex edge2 = dependencyGraph2.AddEdge(node, target);
                                    // targetsByReversedLoops (and therefore excludedTargets) never contains a WAR edge target, therefore all WAR edges are preserved here.
                                    isWriteAfterRead[edge2] = true;
                                }
                            }
                            excludedTargets.Clear();
                        }
                    }
                    // create clones
                    HashSet<string> names = new HashSet<string>();
                    Replacements pendingReplacements = new Replacements();
                    DepthFirstSearch<NodeIndex> dfsClones = new DepthFirstSearch<NodeIndex>(dependencyGraph.SourcesOf, dependencyGraph);
                    dfsClones.FinishNode += delegate (NodeIndex node)
                    {
                        // all sources have already been processed.
                        CloneInfos info = cloneInfosOfNode[node];
                        IStatement initSt = nodes[node];
                        Set<IVariableDeclaration> mutatedVariables = GetMutatedVariables(initSt);
                        Set<IVariableDeclaration> newMutatedVariables = new Set<IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
                        foreach (var entry in info.targetsByReversedLoops)
                        {
                            var reversedLoops = entry.Key;
                            var clone = entry.Value.clone;
                            var cloneTargets = entry.Value.targets;
                            if ((clone != node) != (reversedLoops.Count > 0))
                                throw new Exception();
                            loopVarsToReverse.AddRange(reversedLoops);
                            if (loopVarsToReverse.Count > 0 && initSt is IExpressionStatement)
                            {
                                IExpressionStatement ies = (IExpressionStatement)initSt;
                                IVariableDeclaration ivd = GetVariableDeclaration(ies);
                                if (ivd != null)
                                {
                                    // create a new variable
                                    StringBuilder sb = new StringBuilder();
                                    sb.Append(ivd.Name);
                                    foreach (var loopVar in loopVarsToReverse)
                                    {
                                        sb.Append("_R");
                                        sb.Append(loopVar.Name);
                                    }
                                    string name = sb.ToString();
                                    if (names.Contains(name))
                                        Error($"names.Contains({name})");
                                    names.Add(name);
                                    IVariableDeclaration newvd = Builder.VarDecl(name, ivd.VariableType);
                                    context.InputAttributes.CopyObjectAttributesTo(ivd, context.OutputAttributes, newvd);
                                    globalReplacements.variableReplacements.Add(ivd, newvd);
                                }
                            }
                            Replacements existingReplacements;
                            if (replacementsForStatement.TryGetValue(clone, out existingReplacements))
                            {
                                globalReplacements.Add(existingReplacements);
                            }
                            // This conversion can do up to 3 things:
                            // 1. Reverse the direction of loops.  
                            // 2. Replace variables written with dead variables.
                            // 3. Replace variables read with equivalent clones.
                            // The 1st is always valid.  The 2nd and 3rd are valid as long as all ancestor statements that assign to the old variables
                            // have been converted to assign to the new variables, and there are no other assignments to the new variables.
                            // The 2nd must be done for at least the number of clones.
                            bool anyMutated = false;
                            bool allMutated = true;
                            foreach (var mutatedVariable in mutatedVariables)
                            {
                                IVariableDeclaration newMutatedVariable;
                                if (globalReplacements.variableReplacements.TryGetValue(mutatedVariable, out newMutatedVariable))
                                {
                                    anyMutated = true;
                                }
                                else
                                {
                                    allMutated = false;
                                    newMutatedVariable = mutatedVariable;
                                }
                                if (newMutatedVariables.Contains(newMutatedVariable))
                                    Error($"created multiple assignments to the same variable: {newMutatedVariable}");
                                newMutatedVariables.Add(newMutatedVariable);
                            }
                            if (anyMutated && !allMutated) Error("anyMutated && !allMutated");
                            IStatement convertedSt = ConvertStatement(initSt);
                            globalReplacements.ClearExcept(mutatedVariables);
                            if (!convertedSt.Equals(initSt))
                            {
                                loopMergingInfo.AddEquivalentStatement(convertedSt, loopMergingInfo.GetIndexOf(initSt));
                                globalReplacements.statementReplacements.Add(initSt, convertedSt);
                                if (clone == node)
                                {
                                    nodes[node] = convertedSt;
                                    // save this replacement to be applied again on the next pass, to handle the case when a back edge source has changed.
                                    IStatement existingReplacement;
                                    if (pendingReplacements.statementReplacements.TryGetValue(initSt, out existingReplacement))
                                    {
                                        if (!existingReplacement.Equals(convertedSt))
                                        {
                                            // this can only happen if two nodes point to the same statement.
                                            Error("mismatched replacement");
                                        }
                                    }
                                    else
                                        pendingReplacements.statementReplacements.Add(initSt, convertedSt);
                                }
                            }
                            if (clone != node)
                            {
                                IWhileStatement iws = null;
                                if (containersOfNode[node].Length > 0 && containersOfNode[node][0] is IWhileStatement)
                                    iws = (IWhileStatement)containersOfNode[node][0];
                                var containers = GetContainersOfStatement(convertedSt, iws, null, containerSet);
                                AddNode(clone, convertedSt, containers, groupOf, whileStatements, groupOfWhileStatement);
                                if (!CheckNodesAreUnique()) return;
                                // some of these new nodes will correspond to groups.
                                while (dependencyGraph2.Nodes.Count <= clone)
                                    dependencyGraph2.AddNode();
                            }
                            // apply replacements to the non-moved targets
                            Stack<NodeIndex> writes = new Stack<NodeIndex>();
                            Set<NodeIndex> targetsToConvert = new Set<NodeIndex>();
                            foreach (var target in cloneTargets)
                            {
                                while (dependencyGraph2.Nodes.Count <= target)
                                    dependencyGraph2.AddNode();
                                if (!dependencyGraph2.ContainsEdge(clone, target))
                                    dependencyGraph2.AddEdge(clone, target);
                                if (globalReplacements.Count > 0)
                                {
                                    Replacements replacements;
                                    if (!replacementsForStatement.TryGetValue(target, out replacements))
                                    {
                                        replacements = new Replacements();
                                        replacementsForStatement.Add(target, replacements);
                                    }
                                    replacements.Add(globalReplacements);
                                }
                            }
                            // done with this clone
                            loopVarsToReverse.Clear();
                            globalReplacements.Clear();
                        }
                    };
                    dfsClones.SearchFrom(dependencyGraph.Nodes);
                    dependencyGraph2.NodeCountIsConstant = true;
                    dependencyGraph2.IsReadOnly = true;
                    dependencyGraph = dependencyGraph2;
                    // at this point, nodes[node] == null implies a group.
                    globalReplacements = pendingReplacements;
                }
            }

            groupGraph = new GroupGraph(dependencyGraph, groupOf, firstGroup);
            groupGraph.BuildGroupEdges();
            // clear group edges for false groups
            for (NodeIndex node = firstGroup; node < nodes.Count; node++)
            {
                if (nodes[node] != null)
                {
                    int groupIndex = node - firstGroup;
                    groupGraph.edgesIntoGroup[groupIndex] = null;
                    groupGraph.edgesOutOfGroup[groupIndex] = null;
                }
            }

            if (debug && debugBlock != null)
            {
                for (NodeIndex node = 0; node < nodes.Count; node++)
                {
                    StringBuilder sb = new StringBuilder();
                    sb.Append(NodeToString(node));
                    //Trace.WriteLine(sb.ToString());
                    debugBlock.Statements.Add(Builder.CommentStmt(sb.ToString()));
                }
                //Trace.WriteLine(dependencyGraph);
            }

            if (compiler.OptimiseInferenceCode)
            {
                // sort nodes by array size (approximated by number of containers)
                int[] negContainerCounts = Util.ArrayInit(dependencyGraph.Nodes.Count, i => -containersOfNode[i].Length);
                NodeIndex[] nodeIndices = Util.ArrayInit(dependencyGraph.Nodes.Count, i => i);
                Array.Sort(negContainerCounts, nodeIndices);
                foreach (NodeIndex node in nodeIndices)
                {
                    if (nodes[node] == null)
                        continue;
                    // try to merge node with all of its children
                    bool merged = TryMerge(node);                    
                    if(!merged && containersOfNode[node].Any(c => c is IForStatement))
                    {
                        foreach(var mutatedVariable in GetMutatedVariables(nodes[node]))
                        {
                            if(!context.OutputAttributes.Has<LocalTransform.DoNotUseLocal>(mutatedVariable))
                                context.OutputAttributes.Set(mutatedVariable, new LocalTransform.DoNotUseLocal());
                        }
                    }
                }
            }

            // compute a group schedule
            var schedule = groupGraph.GetScheduleWithGroups(groupGraph.SourcesOf);

            IWhileStatement previousWhileStatement = null;
            IWhileStatement whileLoop = null;
            IConditionStatement previousFirstIterPostStatement = null;
            IConditionStatement firstIterPostStatement = null;
            foreach (var node in schedule)
            {
                IStatement ist = nodes[node];
                if (ist == null)
                    continue;
                // there is no actual conversion here, only a change in DependencyInformation.
                IStatement convertedSt = ConvertStatement(ist);
                if (containersOfNode[node].Length > 0 && containersOfNode[node][0] is IWhileStatement)
                {
                    if (containersOfNode[node][0] != previousWhileStatement)
                    {
                        whileLoop = null;
                        previousWhileStatement = (IWhileStatement)containersOfNode[node][0];
                    }
                    if (whileLoop == null)
                    {
                        whileLoop = Builder.WhileStmt(previousWhileStatement);
                        context.InputAttributes.CopyObjectAttributesTo(previousWhileStatement, context.OutputAttributes, whileLoop);
                        outputs.Add(whileLoop);
                    }
                    if (containersOfNode[node].Length > 1 && context.InputAttributes.Has<FirstIterationPostProcessingBlock>(containersOfNode[node][1]))
                    {
                        if (containersOfNode[node][1] != previousFirstIterPostStatement)
                        {
                            firstIterPostStatement = null;
                            previousFirstIterPostStatement = (IConditionStatement)containersOfNode[node][1];
                        }
                        if (firstIterPostStatement == null)
                        {
                            firstIterPostStatement = Builder.CondStmt(previousFirstIterPostStatement.Condition, Builder.BlockStmt());
                            context.InputAttributes.CopyObjectAttributesTo(previousFirstIterPostStatement, context.OutputAttributes, firstIterPostStatement);
                            whileLoop.Body.Statements.Add(firstIterPostStatement);
                        }
                        firstIterPostStatement.Then.Statements.Add(convertedSt);
                    }
                    else
                        whileLoop.Body.Statements.Add(convertedSt);
                }
                else
                {
                    whileLoop = null;
                    previousWhileStatement = null;
                    outputs.Add(convertedSt);
                }
            }
        }

        private bool CannotMergeWithTargets(int node)
        {
            var readAfterWriteEdges = dependencyGraph.EdgesOutOf(node).Where(edge => !g.isWriteAfterRead[edge]);
            int stmtIndex = loopMergingInfo.GetIndexOf(nodes[node]);
            Set<int> targets = new Set<NodeIndex>();
            foreach (var target in readAfterWriteEdges.Select(dependencyGraph.TargetOf))
            {
                int targetIndex = loopMergingInfo.GetIndexOf(nodes[target]);
                targets.Add(targetIndex);
            }
            foreach (var container in containersOfNode[node])
            {
                if (container is IForStatement)
                {
                    IForStatement ifs = (IForStatement)container;
                    var loopVar = Recognizer.LoopVariable(ifs);
                    int conflict = loopMergingInfo.GetConflictingStmt(targets, stmtIndex, loopVar, isForwardLoop: true);
                    return (conflict != -1);
                }
            }
            return true;
        }

        private IVariableDeclaration GetVariableDeclaration(IExpressionStatement ies)
        {
            IExpression target = ies.Expression;
            if (ies.Expression is IAssignExpression)
            {
                IAssignExpression iae = (IAssignExpression)ies.Expression;
                target = iae.Target;
            }
            if (target is IVariableDeclarationExpression)
            {
                IVariableDeclarationExpression ivde = (IVariableDeclarationExpression)target;
                IVariableDeclaration ivd = ivde.Variable;
                return ivd;
            }
            return null;
        }

        private bool CheckNodesAreUnique()
        {
            // check all non-null nodes are unique
            Set<IStatement> stmts = new Set<IStatement>(ReferenceEqualityComparer<IStatement>.Instance);
            foreach (var ist in nodes)
            {
                if (ist == null)
                    continue;
                if (stmts.Contains(ist))
                {
                    context.Error("duplicate node", null, ist);
                    return false;
                }
                stmts.Add(ist);
            }
            return true;
        }

        private void ForEachNodeBeforeItsGroup(int firstGroup, int lastGroup, Action<NodeIndex> action)
        {
            for (int node = 0; node < firstGroup; node++)
            {
                action(node);
            }
            for (int group = lastGroup - 1; group >= firstGroup; group--)
            {
                action(group);
            }
        }

        private void ForEachNodeAfterItsGroup(int firstGroup, int lastGroup, Action<NodeIndex> action)
        {
            for (int group = firstGroup; group < lastGroup; group++)
            {
                action(group);
            }
            for (int node = firstGroup - 1; node >= 0; node--)
            {
                action(node);
            }
        }

        private Dictionary<NodeIndex, List<NodeIndex>> GetChildrenOfGroup(List<NodeIndex> groupOf)
        {
            var childrenOfGroup = new Dictionary<NodeIndex, List<NodeIndex>>();
            for (int node = 0; node < groupOf.Count; node++)
            {
                NodeIndex group = groupOf[node];
                List<NodeIndex> list;
                if (!childrenOfGroup.TryGetValue(group, out list))
                {
                    list = new List<NodeIndex>();
                }
                list.Add(node);
            }
            return childrenOfGroup;
        }

        private class Replacements
        {
            public Dictionary<IStatement, IStatement> statementReplacements = new Dictionary<IStatement, IStatement>(ReferenceEqualityComparer<IStatement>.Instance);
            public Dictionary<IVariableDeclaration, IVariableDeclaration> variableReplacements = new Dictionary<IVariableDeclaration, IVariableDeclaration>();

            public int Count
            {
                get { return statementReplacements.Count + variableReplacements.Count; }
            }

            public void Clear()
            {
                statementReplacements.Clear();
                variableReplacements.Clear();
            }

            public void ClearExcept(ICollection<IVariableDeclaration> variablesToKeep)
            {
                statementReplacements.Clear();
                var newVariableReplacements = new Dictionary<IVariableDeclaration, IVariableDeclaration>();
                foreach (var entry in variableReplacements)
                {
                    if (variablesToKeep.Contains(entry.Key)) newVariableReplacements.Add(entry.Key, entry.Value);
                }
                variableReplacements = newVariableReplacements;
            }

            public Replacements Clone()
            {
                Replacements result = new Replacements();
                result.statementReplacements.AddRange(this.statementReplacements);
                result.variableReplacements.AddRange(this.variableReplacements);
                return result;
            }

            public void Add(Replacements that)
            {
                AddWithoutConflicts(statementReplacements, that.statementReplacements);
                AddWithoutConflicts(variableReplacements, that.variableReplacements);
            }

            public void AddWithoutConflicts<TKey, TValue>(Dictionary<TKey, TValue> dict, Dictionary<TKey, TValue> that)
            {
                foreach (var entry in that)
                {
                    TValue existingReplacement;
                    if (dict.TryGetValue(entry.Key, out existingReplacement))
                    {
                        if (!existingReplacement.Equals(entry.Value))
                            throw new ArgumentException("Dictionary has conflicting value for the same key");
                    }
                    else
                    {
                        dict.Add(entry.Key, entry.Value);
                    }
                }
            }
        }

        private NodeIndex AllocateNode(List<NodeIndex> groupOf)
        {
            NodeIndex node = groupOf.Count;
            groupOf.Add(-1);
            return node;
        }

        private void AddNode(NodeIndex node, IStatement ist, IStatement[] containers, List<NodeIndex> groupOf, List<IWhileStatement> whileStatements, List<int> groupOfWhileStatement)
        {
            while (nodes.Count <= node)
                nodes.Add(null);
            nodes[node] = ist;
            indexOfNode.Add(ist, node);
            while (containersOfNode.Count <= node)
                containersOfNode.Add(null);
            containersOfNode[node] = containers;
            while (groupOf.Count <= node)
                groupOf.Add(-1);
            NodeIndex parentGroup = -1;
            for (int i = 0; i < containers.Length; i++)
            {
                if (i == 0 && containersOfNode[node][i] is IWhileStatement)
                {
                    IWhileStatement iws = (IWhileStatement)containersOfNode[node][i];
                    int whileNumber = whileStatements.IndexOf(iws);
                    parentGroup = groupOfWhileStatement[whileNumber];
                }
                else
                {
                    // Create a new group for the container.
                    // This can create a dependency cycle among groups, if merging is not done.
                    NodeIndex group = groupOf.Count;
                    groupOf.Add(parentGroup);
                    while (containersOfNode.Count <= group)
                        containersOfNode.Add(null);
                    containersOfNode[group] = containersOfNode[node].Take(i + 1).ToArray();
                    parentGroup = group;
                }
            }
            groupOf[node] = parentGroup;
        }

        private void ForEachClone(Dictionary<NodeIndex, CloneInfos> cloneInfos, NodeIndex node, Action<NodeIndex, Set<IVariableDeclaration>> action)
        {
            CloneInfos info;
            if (cloneInfos.TryGetValue(node, out info))
            {
                // node has clones
                foreach (var entry in info.targetsByReversedLoops)
                {
                    var reversedLoopsInClone = entry.Key;
                    var clone = entry.Value.clone;
                    action(clone, reversedLoopsInClone);
                }
            }
            else
            {
                // node does not have clones
                action(node, null);
            }
        }

        private class CloneInfos
        {
            /// <summary>
            /// Groups the targets of a node by their set of reversed loops.
            /// </summary>
            public Dictionary<Set<IVariableDeclaration>, CloneInfo> targetsByReversedLoops = new Dictionary<Set<IVariableDeclaration>, CloneInfo>();

            public override string ToString()
            {
                return $"CloneInfos";
            }
        }

        private class CloneInfo
        {
            public NodeIndex clone;
            public readonly List<NodeIndex> targets = new List<NodeIndex>();

            public override string ToString()
            {
                return $"CloneInfo({clone}, {StringUtil.CollectionToString(targets, " ")})";
            }
        }

        private IEnumerable<NodeIndex> GetAllocations(NodeIndex node)
        {
            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(nodes[node]);
            foreach (var stmt in di.DeclDependencies)
            {
                if (di.HasDependency(DependencyType.Container, stmt))
                    continue;
                yield return indexOfNode[stmt];
            }
        }

        private Set<IVariableDeclaration> GetMutatedVariables(IStatement stmt)
        {
            Set<IVariableDeclaration> mutatedVariables = new Set<IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
            ForEachInitializerStatement(stmt, init =>
            {
                if (init is IExpressionStatement)
                {
                    IExpressionStatement ies = (IExpressionStatement)init;
                    IVariableDeclaration ivd = GetVariableDeclaration(ies);
                    if (ivd != null)
                        mutatedVariables.Add(ivd);
                }
            });
            return mutatedVariables;
        }

        private bool IsOverwrite(NodeIndex node, NodeIndex target)
        {
            IStatement source = nodes[node];
            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(nodes[target]);
            return di.Overwrites.Contains(source);
        }

        private NodeIndex GetLargestGroup(NodeIndex node, IList<int> groupOf)
        {
            while (groupOf[node] != -1)
            {
                node = groupOf[node];
            }
            return node;
        }

        private bool TryMerge(NodeIndex node)
        {
            // in order to contract an array, it must be able to merge with all of its readers.
            // write-after-read dependencies do not count.
            var readAfterWriteEdges = dependencyGraph.EdgesOutOf(node).Where(edge => !isWriteAfterRead[edge]).ToList();
            if (readAfterWriteEdges.Count == 0)
                return true;
            // check LoopMergingInfo for prohibited merges
            if (CannotMergeWithTargets(node))
                return false;
            var containers = containersOfNode[node];
            int minPrefixLength = readAfterWriteEdges.Min(edge => GetLongestMatchingPrefixLength(containersOfNode[dependencyGraph.TargetOf(edge)], containers));
            // try to merge at increasing prefix lengths until we fail to do so.
            bool merged = false;
            for (int mergePrefix = 1; mergePrefix <= minPrefixLength; mergePrefix++)
            {
                HashSet<NodeIndex> targets = new HashSet<NodeIndex>(readAfterWriteEdges.Select(groupGraph.TargetOf));
                IndexedProperty<NodeIndex, bool> isAncestorOfTarget = groupGraph.CreateNodeData(false);
                // must be unique by construction.
                List<NodeIndex> nodesToMerge = new List<NodeIndex>();
                bool mergeProhibited = false;
                DepthFirstSearch<NodeIndex> dfsDescendants = new DepthFirstSearch<int>(groupGraph.TargetsOf, groupGraph);
                dfsDescendants.FinishNode += descendant =>
                {
                    // all targets have already been processed.
                    bool ancestorOfTarget = targets.Contains(descendant) ||
                        groupGraph.TargetsOf(descendant).Any(target => isAncestorOfTarget[target]);
                    isAncestorOfTarget[descendant] = ancestorOfTarget;
                    if (ancestorOfTarget)
                    {
                        nodesToMerge.Add(descendant);
                        int prefixLength = GetLongestMatchingPrefixLength(containersOfNode[descendant], containers);
                        if (prefixLength < mergePrefix)
                        {
                            mergeProhibited = true;
                        }
                    }
                };
                NodeIndex group = GetGroupForPrefix(node, mergePrefix, containers.Length);
                if (group < 0)
                {
                    Error("containers of statement do not match the block grouping");
                    break;
                }
                dfsDescendants.SearchFrom(group);
                if (mergeProhibited)
                    break;
                foreach (var descendant in nodesToMerge)
                {
                    if (descendant == group)
                        continue;
                    groupGraph.MergeGroups(group, descendant);
                }
                if (containers[mergePrefix - 1] is IForStatement) merged = true;
            }
            return merged;
        }

        private NodeIndex GetGroupForPrefix(NodeIndex node, int prefixLength, int containerCount)
        {
            NodeIndex group = groupGraph.groupOf[node];
            for (int i = prefixLength; i < containerCount; i++)
            {
                group = groupGraph.groupOf[group];
            }
            return group;
        }

        private string NodeToShortString(NodeIndex node)
        {
            return NodeToString(node, true);
        }

        private string NodeToString(NodeIndex node, bool shortString = false)
        {
            string groupString = "";
            if (groupGraph != null)
            {
                Set<NodeIndex> groups = groupGraph.GetGroupSet(node);
                if (groups.Count > 0)
                    groupString = "[" + groups.ToString() + "] ";
            }
            string nodeString;
            if (node >= dependencyGraph.Nodes.Count || nodes[node] == null)
                nodeString = "group";
            else if (shortString)
            {
                nodeString = DependencyGraph.StatementToShortString(nodes[node]);
            }
            else
                nodeString = nodes[node].ToString();
            return string.Format("{0} {1}{2}", node, groupString, nodeString);
        }

        protected override IStatement DoConvertStatement(IStatement ist)
        {
            IStatement st = base.DoConvertStatement(ist);
            if (globalReplacements.statementReplacements.Count > 0)
            {
                DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                if (di != null)
                {
                    di = (DependencyInformation)di.Clone();
                    di.Replace(globalReplacements.statementReplacements);
                    context.OutputAttributes.Remove<DependencyInformation>(st);
                    context.OutputAttributes.Set(st, di);
                }
            }
            return st;
        }

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            var loopVar = Recognizer.LoopVariable(ifs);
            if (loopVarsToReverse.Contains(loopVar) && !(ifs is IBrokenForStatement))
            {
                // copied from CopyTransform.ConvertFor
                IForStatement fs = Builder.ForStmt();
                context.SetPrimaryOutput(fs);
                fs.Initializer = ConvertStatement(ifs.Initializer);
                fs.Condition = ConvertExpression(ifs.Condition);
                fs.Increment = ConvertStatement(ifs.Increment);
                fs.Body = ConvertBlock(ifs.Body);
                context.InputAttributes.CopyObjectAttributesTo(ifs, context.OutputAttributes, fs);
                Recognizer.ReverseLoopDirection(fs);
                return fs;
            }
            else
            {
                return base.ConvertFor(ifs);
            }
        }

        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            IVariableDeclaration newvd;
            if (globalReplacements.variableReplacements.TryGetValue(ivde.Variable, out newvd))
                return Builder.VarDeclExpr(newvd);
            else
                return base.ConvertVariableDeclExpr(ivde);
        }

        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            IVariableDeclaration newvd;
            if (globalReplacements.variableReplacements.TryGetValue(ivre.Variable.Variable, out newvd))
                return Builder.VarRefExpr(newvd);
            else
                return base.ConvertVariableRefExpr(ivre);
        }

        /// <summary>
        /// Get the set of loops that have a different direction in B and A.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="reversedLoopsInB"></param>
        /// <param name="treatMissingAsForward">If true, loops missing from A or B are assumed to be in the forward direction.</param>
        /// <returns></returns>
        protected Set<IVariableDeclaration> GetReversedLoops(IStatement[] a, IStatement[] b, Set<IVariableDeclaration> reversedLoopsInB = null, bool treatMissingAsForward = false)
        {
            Set<IVariableDeclaration> reversedLoops = new Set<IVariableDeclaration>();
            Set<IVariableDeclaration> bLoopVars = new Set<IVariableDeclaration>();
            int i = 0;
            for (; i < System.Math.Min(a.Length, b.Length); i++)
            {
                if (a[i] is IForStatement && b[i] is IForStatement)
                {
                    IForStatement afs = (IForStatement)a[i];
                    IForStatement bfs = (IForStatement)b[i];
                    bLoopVars.Add(Recognizer.LoopVariable(bfs));
                    var aIsBrokenLoop = afs is IBrokenForStatement;
                    var bIsBrokenLoop = bfs is IBrokenForStatement;
                    if (!aIsBrokenLoop && !bIsBrokenLoop)
                    {
                        var loopVar = Recognizer.LoopVariable(afs);
                        bool reversedInB = (reversedLoopsInB != null) && reversedLoopsInB.Contains(loopVar);
                        if (reversedInB)
                        {
                            if (ReferenceEquals(a[i], b[i]))
                                reversedLoops.Add(loopVar);
                        }
                        else if (AreEqualAfterReversal(afs, bfs))
                        {
                            reversedLoops.Add(loopVar);
                        }
                    }
                }
                else if (!ReferenceEquals(a[i], b[i]))
                    break;
            }
            if (treatMissingAsForward)
            {
                bool bHasExtraWhile = b.Skip(i).Any(ist => ist is IWhileStatement);
                if (!bHasExtraWhile)
                {
                    for (; i < b.Length; i++)
                    {
                        if (b[i] is IForStatement)
                        {
                            IForStatement bfs = (IForStatement)b[i];
                            var loopVar = Recognizer.LoopVariable(bfs);
                            bLoopVars.Add(loopVar);
                            bool reversedInB = (reversedLoopsInB != null) && reversedLoopsInB.Contains(loopVar);
                            // bfs is backward if the loop direction is backward and not reversedInB, 
                            // or the loop direction is foward and reversedInB.
                            if (Recognizer.IsForwardLoop(bfs) == reversedInB)
                            {
                                reversedLoops.Add(loopVar);
                            }
                        }
                    }
                    if (reversedLoopsInB != null)
                    {
                        // loop variables in reversedLoopsInB that do not appear in B are added to the result.
                        foreach (var loopVar in reversedLoopsInB)
                        {
                            if (!bLoopVars.Contains(loopVar))
                                reversedLoops.Add(loopVar);
                        }
                    }
                }
            }
            return reversedLoops;
        }

        internal static bool AreEqualAfterReversal(IForStatement fs, IForStatement fs2)
        {
            return (Recognizer.LoopVariable(fs) == Recognizer.LoopVariable(fs2)) &&
                (Recognizer.IsForwardLoop(fs) != Recognizer.IsForwardLoop(fs2));
        }

        protected int GetLongestMatchingPrefixLength(IStatement[] a, IStatement[] b)
        {
            int i = 0;
            for (; i < System.Math.Min(a.Length, b.Length); i++)
            {
                if (!ReferenceEquals(a[i], b[i]))
                    break;
            }
            return i;
        }

        protected bool AllWhileLoopsMatch(IStatement[] a, IStatement[] b)
        {
            int i = 0;
            for (; i < System.Math.Min(a.Length, b.Length); i++)
            {
                if (!(a[i] is IWhileStatement))
                    break;
                if (!ReferenceEquals(a[i], b[i]))
                    return false;
            }
            if (a.Length > i && a[i] is IWhileStatement) return false;
            if (b.Length > i && b[i] is IWhileStatement) return false;
            return true;
        }

        private static T[] AddFirst<T>(T[] array, T item)
        {
            T[] result = new T[array.Length + 1];
            result[0] = item;
            array.CopyTo(result, 1);
            return result;
        }

        protected void ForEachInitializerStatement(IWhileStatement iws, Action<IStatement> action)
        {
            if (context.InputAttributes.Has<HasOffsetIndices>(iws))
            {
                return;
            }
            InitializerSet initSet = context.InputAttributes.Get<InitializerSet>(iws);
            if (initSet == null)
                return;
            foreach (var init in initSet.initializers)
            {
                ForEachInitializerStatement(init, action);
            }
        }

        protected void ForEachInitializerStatement(IStatement init, Action<IStatement> action)
        {
            action(init);
            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(init);
            foreach (IStatement ow in di.Overwrites)
            {
                ForEachInitializerStatement(ow, action);
            }
        }

        protected IStatement[] GetContainersOfStatement(IStatement ist, IStatement whileStatement, IStatement firstIterPostStatement, List<IStatement> containerSet)
        {
            List<IStatement> containers = new List<IStatement>();
            if (whileStatement != null)
                containers.Add(whileStatement);
            if (firstIterPostStatement != null)
                containers.Add(firstIterPostStatement);
            LoopMergingTransform.UnwrapStatement(ist, containers);
            return containers.Select(container => FindContainer(containerSet, container)).ToArray();
        }

        protected IStatement FindContainer(List<IStatement> containerSet, IStatement container)
        {
            foreach (var c in containerSet)
            {
                if (Containers.ContainersAreEqual(c, container))
                    return c;
            }
            containerSet.Add(container);
            return container;
        }
    }
}