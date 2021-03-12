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
using Microsoft.ML.Probabilistic.Compiler;
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
    internal class LocalAllocationTransform2 : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "LocalAllocationTransform";
            }
        }

        public static bool debug;
        public static int maxFuel = int.MaxValue;
        private int fuel = maxFuel;
        IBlockStatement debugBlock;
        private ModelCompiler compiler;
        private LoopMergingInfo loopMergingInfo;
        private IndexedGraph dependencyGraph;
        /// <summary>
        /// Maps from a dependencyGraph node index to a statement, or null if the node is a group.
        /// </summary>
        private List<IStatement> nodes;
        /// <summary>
        /// Maps from a statement to a dependencyGraph node index.
        /// </summary>
        private Dictionary<IStatement, int> indexOfNode;
        /// <summary>
        /// Maps from a dependencyGraph node index to an array of containers.
        /// </summary>
        List<IStatement[]> containersOfNode = new List<IStatement[]>();
        IndexedProperty<EdgeIndex, bool> isWriteAfterRead, isWriteAfterWrite;
        /// <summary>
        /// Shares the same node indexing as dependencyGraph.
        /// </summary>
        private GroupGraph groupGraph;
        private Set<IVariableDeclaration> loopVarsToReverse = new Set<IVariableDeclaration>();
        Replacements globalReplacements = new Replacements();
        Replacements pendingReplacements = new Replacements();
        List<IStatement> containerSet = new List<IStatement>();

        internal LocalAllocationTransform2(ModelCompiler compiler)
        {
            this.compiler = compiler;
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
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
            DependencyGraph2 g;
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
            isWriteAfterWrite = g.isWriteAfterWrite;

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

            IndexedProperty<NodeIndex, int> whileNumberOfTargets = GetWhileNumberOfTargets(debugBlock, initializerStatements, whileNumberOfNode, groupOf, firstGroup);

            bool makeReversedClones = compiler.UseLocals && compiler.OptimiseInferenceCode;

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

            BuildGroupGraph(firstGroup, groupOf);

            if (makeReversedClones)
            {
                // find the set of variables
                List<NodeIndex> variableDeclarations = new List<NodeIndex>();
                List<int> arraySizes = new List<NodeIndex>();
                foreach (var node in dependencyGraph.Nodes)
                {
                    IStatement ist = nodes[node];
                    if (ist == null)
                        continue;
                    if (whileNumberOfTargets[node] == 0)
                        continue;
                    bool hasAnyInitializers = dependencyGraph.EdgesInto(node).Any(edge => isWriteAfterWrite[edge]);
                    if (!hasAnyInitializers)
                    {
                        IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(ist);
                        if (ivd != null)
                        {
                            variableDeclarations.Add(node);
                            var varInfo = VariableInformation.GetVariableInformation(context, ivd);
                            int arraySize = varInfo.ArrayDepth;
                            arraySizes.Add(arraySize);
                        }
                    }
                }
                // sort variables by decreasing array size
                NodeIndex[] variableDeclarationsSorted = variableDeclarations.ToArray();
                int[] negativeArraySizes = arraySizes.Select(s => -s).ToArray();
                Array.Sort(negativeArraySizes, variableDeclarationsSorted);

                foreach (NodeIndex node in variableDeclarationsSorted)
                {
                    // try to merge node with all of its children
                    if (debug && debugBlock != null)
                        debugBlock.Statements.Add(nodes[node]);
                    TryCloneAndMerge(node, groupOf, whileStatements, groupOfWhileStatement, cannotBeCloned);
                }
            }

            if (debug && debugBlock != null)
            {
                foreach (var stmt in debugBlock.Statements)
                    Trace.WriteLine(stmt);
                for (NodeIndex node = 0; node < nodes.Count; node++)
                {
                    StringBuilder sb = new StringBuilder();
                    sb.Append(NodeToString(node));
                    Trace.WriteLine(sb.ToString());
                    //debugBlock.Statements.Add(Builder.CommentStmt(sb.ToString()));
                }
                Trace.WriteLine(dependencyGraph);
            }

            // compute a group schedule
            var schedule = groupGraph.GetScheduleWithGroups(groupGraph.SourcesOf);

            IWhileStatement previousWhileStatement = null;
            IWhileStatement whileLoop = null;
            IConditionStatement previousFirstIterPostStatement = null;
            IConditionStatement firstIterPostStatement = null;
            globalReplacements.Add(pendingReplacements);
            MakeTransitive(globalReplacements.statementReplacements);
            foreach (var node in schedule)
            {
                IStatement ist = nodes[node];
                if (ist == null)
                    continue;
                // there is no actual conversion here, only a change in DependencyInformation.
                IStatement convertedSt = ConvertStatement(ist);
                var containers = containersOfNode[node];
                if (containers.Length > 0 && containers[0] is IWhileStatement)
                {
                    if (containers[0] != previousWhileStatement)
                    {
                        whileLoop = null;
                        previousWhileStatement = (IWhileStatement)containers[0];
                    }
                    if (whileLoop == null)
                    {
                        whileLoop = Builder.WhileStmt(previousWhileStatement);
                        context.InputAttributes.CopyObjectAttributesTo(previousWhileStatement, context.OutputAttributes, whileLoop);
                        outputs.Add(whileLoop);
                    }
                    if (containers.Length > 1 && context.InputAttributes.Has<FirstIterationPostProcessingBlock>(containers[1]))
                    {
                        if (containers[1] != previousFirstIterPostStatement)
                        {
                            firstIterPostStatement = null;
                            previousFirstIterPostStatement = (IConditionStatement)containers[1];
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

        private IndexedProperty<NodeIndex, int> GetWhileNumberOfTargets(IBlockStatement debugBlock, Set<IStatement> initializerStatements, List<int> whileNumberOfNode, List<int> groupOf, int firstGroup)
        {
            // InitializerStatements could be instead filled in by g.BackEdges.Keys.
            // After initializerStatements is filled in, determine the initializerTargets.
            Set<int> cannotMove = new Set<NodeIndex>();
            for (int node = 0; node < dependencyGraph.Nodes.Count; node++)
            {
                IStatement ist = nodes[node];
                DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                bool dependsOnInitializer = di.Dependencies.Any(dependency => initializerStatements.Contains(dependency));
                if (dependsOnInitializer)
                    cannotMove.Add(node);
                // Container dependencies must not be moved into the while loop, since this causes problems in IterativeProcessTransform.
                // (Every statement that depends on it will inherit the parameterDeps of the while loop.) 
                cannotMove.AddRange(di.ContainerDependencies.Select(dependency => indexOfNode[dependency]));
            }

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
                    if (hasOutsideTarget)
                    {
                        // if a node has a target outside the while loop, then its initializers must remain outside the while loop.
                        ForEachInitializerStatement(nodes[node], ist =>
                        {
                            NodeIndex init = indexOfNode[ist];
                            initializesStmtWithOutsideTarget[init] = true;
                        });
                    }
                }
                else if (!initializerStatements.Contains(nodes[node]) &&
                    !initializesStmtWithOutsideTarget[node] &&
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
                            commonWhileNumber = 0;
                    }
                    whileNumberOfTargets[node] = commonWhileNumber;
                }
                else if (debug && debugBlock != null)
                {
                    StringBuilder sb = new StringBuilder();
                    if (cannotMove.Contains(node))
                        sb.Append(" cannotMove");
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
            });
            return whileNumberOfTargets;
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

        private void MakeTransitive<T>(Dictionary<T,T> dict)
        {
            foreach(var entry in new List<KeyValuePair<T,T>>(dict))
            {
                T value;
                if (dict.TryGetValue(entry.Value, out value))
                {
                    T value2;
                    while(dict.TryGetValue(value, out value2))
                    {
                        value = value2;
                    }
                    dict[entry.Key] = value;
                }
            }
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

        private NodeIndex AddNode(IStatement ist, IStatement[] containers, List<NodeIndex> groupOf, IReadOnlyList<IWhileStatement> whileStatements, IReadOnlyList<int> groupOfWhileStatement)
        {
            NodeIndex node = AllocateNode(groupOf);
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
            return node;
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

        private NodeIndex GetLargestGroup(NodeIndex node, IList<int> groupOf)
        {
            while (groupOf[node] != -1)
            {
                node = groupOf[node];
            }
            return node;
        }

        private IEnumerable<NodeIndex> WriteAfterWriteSourcesOf(NodeIndex node)
        {
            foreach (var edge in dependencyGraph.EdgesInto(node))
            {
                if (isWriteAfterWrite[edge])
                {
                    yield return dependencyGraph.SourceOf(edge);
                }
            }
        }

        private IEnumerable<NodeIndex> WriteAfterWriteTargetsOf(NodeIndex node)
        {
            foreach (var edge in dependencyGraph.EdgesOutOf(node))
            {
                if (isWriteAfterWrite[edge])
                {
                    yield return dependencyGraph.TargetOf(edge);
                }
            }
        }

        Set<NodeIndex> writers = new Set<NodeIndex>();
        DepthFirstSearch<NodeIndex> dfsWawForward;

        private class CloneInfo
        {
            public readonly Dictionary<NodeIndex,NodeIndex> cloneOfNode = new Dictionary<NodeIndex,NodeIndex>();
            public readonly Replacements replacements = new Replacements();

            public override string ToString()
            {
                return $"CloneInfo";
            }
        }

        private void TryCloneAndMerge(NodeIndex variableNode, List<int> groupOf, IReadOnlyList<IWhileStatement> whileStatements, IReadOnlyList<int> groupOfWhileStatement, Set<NodeIndex> cannotBeCloned)
        {
            // find all writes to this variable
            if (dfsWawForward == null)
            {
                dfsWawForward = new DepthFirstSearch<NodeIndex>(WriteAfterWriteTargetsOf, dependencyGraph);
                dfsWawForward.DiscoverNode += writers.Add;
            }
            writers.Clear();
            dfsWawForward.SearchFrom(variableNode);

            if (writers.Any(cannotBeCloned.Contains))
            {
                return;
            }

            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(nodes[variableNode]);

            if (debug && debugBlock != null)
            {
                debugBlock.Statements.Add(Builder.CommentStmt("writes:"));
                foreach (var write in writers)
                    debugBlock.Statements.Add(nodes[write]);
            }

            // find all reads of this variable
            Set<NodeIndex> readers = new Set<NodeIndex>();
            foreach (var writer in writers)
            {
                foreach (var edge in dependencyGraph.EdgesOutOf(writer))
                {
                    NodeIndex target = dependencyGraph.TargetOf(edge);
                    if (isWriteAfterWrite[edge])
                        continue;
                    if (!isWriteAfterRead[edge])
                    {
                         readers.Add(target);
                    }
                    else
                    {
                        if(debug && debugBlock != null)
                        {
                            debugBlock.Statements.Add(Builder.CommentStmt("writer:"));
                            debugBlock.Statements.Add(nodes[writer]);
                            debugBlock.Statements.Add(Builder.CommentStmt("target:"));
                            debugBlock.Statements.Add(nodes[target]);
                            debugBlock.Statements.Add(Builder.CommentStmt("writeAfterRead edges prevent loop merging."));
                        }
                        //return;
                    }
                }
            }

            // check whether this variable can be contracted if cloned a sufficient number of times.
            // for each reader, check that it can merge with all writers it depends on.
            foreach (var reader in readers)
            {
                Set<NodeIndex> sources = new Set<NodeIndex>();
                foreach (var edge in dependencyGraph.EdgesInto(reader))
                {
                    if (!isWriteAfterRead[edge] && !isWriteAfterWrite[edge])
                    {
                        NodeIndex source = dependencyGraph.SourceOf(edge);
                        if (writers.Contains(source))
                        {
                            sources.Add(loopMergingInfo.GetIndexOf(nodes[source]));
                        }
                    }
                }
                int stmtIndex = loopMergingInfo.GetIndexOf(nodes[reader]);
                int mergableContainerCount = 0;
                foreach (var container in containersOfNode[reader])
                {
                    if (container is IForStatement)
                    {
                        IForStatement ifs = (IForStatement)container;
                        var loopVar = Recognizer.LoopVariable(ifs);
                        int conflict = loopMergingInfo.GetConflictingStmt(sources, stmtIndex, loopVar, Recognizer.IsForwardLoop(ifs));
                        if (conflict != -1)
                        {
                            if (debug && debugBlock != null)
                            {
                                debugBlock.Statements.Add(Builder.CommentStmt("reader:"));
                                debugBlock.Statements.Add(nodes[reader]);
                                debugBlock.Statements.Add(Builder.CommentStmt($"cannot merge with writer on loop {loopVar.Name}"));
                            }
                        }
                        else mergableContainerCount++;
                    }
                }
                if (mergableContainerCount == 0) return;
            }

            // remember previous clones
            // Maps from loopVarsToReverse to a list of lists of writers
            List<CloneInfo> clones = new List<CloneInfo>();

            // for each reader, find all ancestors and make a clone.  transform the read to refer to the clone.
            foreach (var reader in readers)
            {
                //if (fuel == 0) break;
                fuel--;
                // find all ancestors of this reader.
                // Every ancestor is a writer of the variable.
                // Ancestors must include the variable declaration.
                // Ancestor list has sources before targets.
                List<NodeIndex> ancestors = new List<NodeIndex>();
                foreach (var edge in dependencyGraph.EdgesInto(reader))
                {
                    if (!isWriteAfterRead[edge])
                    {
                        NodeIndex source = dependencyGraph.SourceOf(edge);
                        if (writers.Contains(source))
                        {
                            ForEachOverwriteAncestor(source, node =>
                            {
                                if (!ancestors.Contains(node))
                                    ancestors.Add(node);
                            });
                        }
                    }
                }
                if (ancestors.Count == 0) throw new Exception("ancestors.Count == 0");
                if (debug && debugBlock != null)
                {
                    debugBlock.Statements.Add(Builder.CommentStmt($"fuel: {fuel}"));
                    debugBlock.Statements.Add(Builder.CommentStmt("reader:"));
                    debugBlock.Statements.Add(nodes[reader]);
                    debugBlock.Statements.Add(Builder.CommentStmt("ancestors:"));
                    foreach (var write in ancestors)
                        debugBlock.Statements.Add(nodes[write]);
                }

                var readerContainers = containersOfNode[reader];
                IWhileStatement iws = null;
                if (readerContainers.Length > 0 && readerContainers[0] is IWhileStatement)
                    iws = (IWhileStatement)readerContainers[0];

                // find all descendants of the reader.
                var readerGroups = groupGraph.GetGroupSet(reader);
                Set<NodeIndex> descendants = new Set<NodeIndex>();
                DepthFirstSearch<NodeIndex> dfsDescendants = new DepthFirstSearch<NodeIndex>(groupGraph.TargetsOf, groupGraph);
                dfsDescendants.DiscoverNode += descendants.Add;
                dfsDescendants.SearchFrom(reader);
                foreach(var group in readerGroups)
                {
                    dfsDescendants.SearchFrom(group);
                }
                descendants.Remove(readerGroups);

                // check for an existing compatible clone
                CloneInfo cloneInfo = null;
                bool allAncestorsPreviouslyCloned = false;
                foreach (var previousCloneInfo in clones)
                {
                    // TODO: check that new clones would not create a cycle
                    bool anyNodeIsDescendant = previousCloneInfo.cloneOfNode.Values.Any(writer =>
                        descendants.ContainsAny(groupGraph.GetGroupSet(writer)));
                    if (anyNodeIsDescendant) continue;
                    bool anyReversedLoops = previousCloneInfo.cloneOfNode.Values.Any(writer =>
                    {
                        // determine the loop variables to reverse
                        var writerContainers = containersOfNode[writer];
                        if (iws != null && !(writerContainers.Length > 0 && writerContainers[0] is IWhileStatement))
                            writerContainers = AddFirst(writerContainers, iws);
                        return GetReversedLoops(writerContainers, readerContainers).Count > 0;
                    });
                    if (!anyReversedLoops)
                    {
                        // TEMPORARY
                        //cloneInfo = previousCloneInfo;
                        allAncestorsPreviouslyCloned = previousCloneInfo.cloneOfNode.Keys.ContainsAll(ancestors);
                        if (allAncestorsPreviouslyCloned)
                            break;
                    }
                }
                if (cloneInfo == null)
                {
                    // make a clone
                    cloneInfo = new CloneInfo();
                    clones.Add(cloneInfo);
                    // create a new variable
                    StringBuilder sb = new StringBuilder();
                    sb.Append(ivd.Name);
                    foreach (var loopVar in loopVarsToReverse)
                    {
                        sb.Append("_R");
                        sb.Append(loopVar.Name);
                    }
                    sb.Append("_");
                    string name = sb.ToString();
                    name = VariableInformation.GenerateName(context, name);
                    if (debug && debugBlock != null)
                    {
                        debugBlock.Statements.Add(Builder.CommentStmt($"writes to clone {name}:"));
                    }
                    IVariableDeclaration newvd = Builder.VarDecl(name, ivd.VariableType);
                    context.InputAttributes.CopyObjectAttributesTo(ivd, context.OutputAttributes, newvd);
                    cloneInfo.replacements.variableReplacements.Add(ivd, newvd);
                }
                else if(debug && debugBlock != null)
                {
                    debugBlock.Statements.Add(Builder.CommentStmt($"previous clone is compatible: {cloneInfo.replacements.variableReplacements.First()}"));
                }

                // build a new dependency graph including clones
                IndexedGraph dependencyGraph2 = new IndexedGraph(dependencyGraph.Nodes.Count);
                dependencyGraph2.NodeCountIsConstant = false;
                IndexedProperty<EdgeIndex, bool> isWriteAfterRead2 = dependencyGraph2.CreateEdgeData(false);
                IndexedProperty<EdgeIndex, bool> isWriteAfterWrite2 = dependencyGraph2.CreateEdgeData(false);
                // copy everything except edges into reader
                foreach (var node in dependencyGraph.Nodes)
                {
                    if (node == reader) continue;
                    foreach (var edge in dependencyGraph.EdgesInto(node))
                    {
                        NodeIndex source = dependencyGraph.SourceOf(edge);
                        EdgeIndex edge2 = dependencyGraph2.AddEdge(source, node);
                        isWriteAfterRead2[edge2] = isWriteAfterRead[edge];
                        isWriteAfterWrite2[edge2] = isWriteAfterWrite[edge];
                    }
                }

                foreach (var writer in ancestors)
                {
                    // skip if already cloned
                    if (cloneInfo.cloneOfNode.ContainsKey(writer)) continue;
                    IStatement ist = nodes[writer];
                    globalReplacements.Clear();
                    globalReplacements.Add(pendingReplacements);
                    globalReplacements.Add(cloneInfo.replacements);
                    MakeTransitive(globalReplacements.statementReplacements);
                    // This conversion can do up to 3 things:
                    // 1. Reverse the direction of loops.  
                    // 2. Replace variables written with dead variables.
                    // 3. Replace variables read with equivalent clones.
                    // The 1st is always valid.  The 2nd and 3rd are valid as long as all ancestor statements that assign to the old variables
                    // have been converted to assign to the new variables, and there are no other assignments to the new variables.
                    // The 2nd must be done for at least the number of clones.
                    bool anyMutated = false;
                    bool allMutated = true;
                    Set<IVariableDeclaration> mutatedVariables = GetMutatedVariables(ist);
                    Set<IVariableDeclaration> newMutatedVariables = new Set<IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
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
                    // determine the loop variables to reverse
                    var writerContainers = containersOfNode[writer];
                    if (iws != null)
                        writerContainers = AddFirst(writerContainers, iws);
                    // this may perform an unnecessary reversal, if a child is missing the container.
                    loopVarsToReverse.AddRange(GetReversedLoops(writerContainers, readerContainers));
                    // this conversion must create a new statement.
                    this.ShallowCopy = true;
                    IStatement convertedSt = ConvertStatement(ist);
                    this.ShallowCopy = false;
                    loopVarsToReverse.Clear();
                    globalReplacements.Clear();
                    //globalReplacements.ClearExcept(mutatedVariables);
                    loopMergingInfo.AddEquivalentStatement(convertedSt, loopMergingInfo.GetIndexOf(ist));
                    cloneInfo.replacements.statementReplacements.Add(ist, convertedSt);
                    var containers = GetContainersOfStatement(convertedSt, iws, null, containerSet);
                    NodeIndex clone = AddNode(convertedSt, containers, groupOf, whileStatements, groupOfWhileStatement);
                    if (!CheckNodesAreUnique()) return;
                    // synchronize node indices between dependencyGraph2 and groupOf.
                    // some of these new nodes will correspond to groups.
                    while (dependencyGraph2.Nodes.Count <= clone)
                        dependencyGraph2.AddNode();
                    cloneInfo.cloneOfNode.Add(writer, clone);
                    // add edges into the clone
                    foreach (var edge in dependencyGraph.EdgesInto(writer))
                    {
                        NodeIndex source = dependencyGraph.SourceOf(edge);
                        NodeIndex replaced;
                        if (!cloneInfo.cloneOfNode.TryGetValue(source, out replaced))
                        {
                            replaced = source;
                        }
                        EdgeIndex edge2 = dependencyGraph2.AddEdge(replaced, clone);
                        isWriteAfterRead2[edge2] = isWriteAfterRead[edge];
                        isWriteAfterWrite2[edge2] = isWriteAfterWrite[edge];
                    }
                    // add edges out of the clone
                    foreach(var edge in dependencyGraph.EdgesOutOf(writer))
                    {
                        if (!isWriteAfterRead[edge]) continue;
                        NodeIndex target = dependencyGraph.TargetOf(edge);
                        NodeIndex replaced;
                        if (!cloneInfo.cloneOfNode.TryGetValue(target, out replaced))
                        {
                            replaced = target;
                        }
                        EdgeIndex edge2 = dependencyGraph2.AddEdge(clone, replaced);
                        isWriteAfterRead2[edge2] = true;
                    }
                    if (debug && debugBlock != null)
                    {
                        debugBlock.Statements.Add(convertedSt);
                    }
                }

                // convert the reader
                IStatement readerStmt = nodes[reader];
                globalReplacements.Clear();
                globalReplacements.Add(pendingReplacements);
                globalReplacements.Add(cloneInfo.replacements);
                MakeTransitive(globalReplacements.statementReplacements);
                // this conversion need not create a new statement.
                IStatement convertedReader = ConvertStatement(readerStmt);
                nodes[reader] = convertedReader;
                indexOfNode[convertedReader] = reader;
                loopMergingInfo.AddEquivalentStatement(convertedReader, loopMergingInfo.GetIndexOf(readerStmt));
                pendingReplacements.statementReplacements.Add(readerStmt, convertedReader);
                globalReplacements.Clear();

                // add edges into reader
                foreach (var edge in dependencyGraph.EdgesInto(reader))
                {
                    NodeIndex source = dependencyGraph.SourceOf(edge);
                    NodeIndex replaced;
                    if (!cloneInfo.cloneOfNode.TryGetValue(source, out replaced))
                    {
                        replaced = source;
                    }
                    EdgeIndex edge2 = dependencyGraph2.AddEdge(replaced, reader);
                    isWriteAfterRead2[edge2] = isWriteAfterRead[edge];
                    isWriteAfterWrite2[edge2] = isWriteAfterWrite[edge];
                }

                dependencyGraph2.NodeCountIsConstant = true;
                dependencyGraph2.IsReadOnly = true;
                dependencyGraph = dependencyGraph2;
                isWriteAfterRead = isWriteAfterRead2;
                isWriteAfterWrite = isWriteAfterWrite2;
                BuildGroupGraph(groupGraph.firstGroup, groupOf);

                // merge the reader with all clones
                ancestors.Reverse();
                Dictionary<NodeIndex, int> minPrefixOfNode = new Dictionary<NodeIndex, NodeIndex>();
                foreach (var writer in ancestors)
                {
                    TryMerge(reader, cloneInfo.cloneOfNode[writer], minPrefixOfNode);
                }
            }
        }

        private void BuildGroupGraph(int firstGroup, IList<int> groupOf)
        {
            groupGraph = new GroupGraph(dependencyGraph, groupOf, firstGroup);
            groupGraph.BuildGroupEdges();
            // clear group edges for false groups
            for (NodeIndex node = groupGraph.firstGroup; node < nodes.Count; node++)
            {
                if (nodes[node] != null)
                {
                    int groupIndex = node - groupGraph.firstGroup;
                    groupGraph.edgesIntoGroup[groupIndex] = null;
                    groupGraph.edgesOutOfGroup[groupIndex] = null;
                }
            }
        }

        private void ForEachOverwriteAncestor(NodeIndex node, Action<NodeIndex> action)
        {
            foreach (var source in WriteAfterWriteSourcesOf(node))
            {
                ForEachOverwriteAncestor(source, action);
            }
            action(node);
        }

        private void TryMerge(NodeIndex target, NodeIndex source, Dictionary<NodeIndex, int> minPrefixOfNode)
        {
            var targetContainers = containersOfNode[target];
            var sourceContainers = containersOfNode[source];
            int minPrefixLength = GetLongestMatchingPrefixLength(sourceContainers, targetContainers);
            // take minimum of child prefix lengths.
            minPrefixLength = System.Math.Min(minPrefixLength, dependencyGraph.TargetsOf(source).Min(node =>
            {
                int value;
                if (minPrefixOfNode.TryGetValue(node, out value)) return value;
                else return int.MaxValue;
            }));
            minPrefixOfNode.Add(source, minPrefixLength);
            // try to merge at increasing prefix lengths until we fail to do so.
            for (int mergePrefix = 1; mergePrefix <= minPrefixLength; mergePrefix++)
            {
                NodeIndex targetGroup = GetGroupForPrefix(target, mergePrefix, targetContainers.Length);
                if (targetGroup < 0)
                {
                    Error("containers of statement do not match the block grouping");
                    break;
                }
                NodeIndex sourceGroup = GetGroupForPrefix(source, mergePrefix, sourceContainers.Length);
                if (sourceGroup < 0)
                {
                    Error("containers of statement do not match the block grouping");
                    break;
                }
                IndexedProperty<NodeIndex, bool> isAncestorOfTarget = groupGraph.CreateNodeData(false);
                // must be unique by construction.
                List<NodeIndex> nodesToMerge = new List<NodeIndex>();
                bool mergeProhibited = false;
                DepthFirstSearch<NodeIndex> dfsDescendants = new DepthFirstSearch<int>(groupGraph.TargetsOf, groupGraph);
                dfsDescendants.FinishNode += descendant =>
                {
                    // all targets have already been processed.
                    bool ancestorOfTarget = (targetGroup == descendant) ||
                        groupGraph.TargetsOf(descendant).Any(target2 => isAncestorOfTarget[target2]);
                    isAncestorOfTarget[descendant] = ancestorOfTarget;
                    if (ancestorOfTarget)
                    {
                        nodesToMerge.Add(descendant);
                        int prefixLength = GetLongestMatchingPrefixLength(containersOfNode[descendant], targetContainers);
                        if (prefixLength < mergePrefix)
                        {
                            mergeProhibited = true;
                        }
                    }
                };
                dfsDescendants.SearchFrom(sourceGroup);
                if (mergeProhibited)
                    break;
                foreach (var descendant in nodesToMerge)
                {
                    if (descendant == sourceGroup)
                        continue;
                    groupGraph.MergeGroups(sourceGroup, descendant);
                }
            }
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