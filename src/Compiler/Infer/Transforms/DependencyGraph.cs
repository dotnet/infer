// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif
    /// <summary>
    /// For when the input statements have no ordering i.e. before scheduling
    /// </summary>
    internal class DependencyGraph
    {
        public static bool debug;
        internal static bool initsCanBeStale = false;
        internal IndexedGraph dependencyGraph;
        internal CanCreateNodeData<NodeIndex> nodeData;
        internal CanCreateEdgeData<EdgeIndex> edgeData;
        /// <summary>
        /// Nodes initialized by the user
        /// </summary>
        public Set<NodeIndex> initializedNodes = new Set<NodeIndex>();
        /// <summary>
        /// Nodes that have a non-uniform initializer statement.  Currently equivalent to initializedNodes but could be different in the future.
        /// </summary>
        public Set<NodeIndex> hasNonUniformInitializer = new Set<EdgeIndex>();
        /// <summary>
        /// Edges initialized by the user
        /// </summary>
        public Set<EdgeIndex> initializedEdges = new Set<EdgeIndex>();
        /// <summary>
        /// Nodes that cannot be initialized (they must execute before their children)
        /// </summary>
        public Set<NodeIndex> mustNotInit = new Set<EdgeIndex>();
        public Set<NodeIndex> outputNodes = new Set<NodeIndex>();
        public Set<NodeIndex> initiallyStaleNodes = new Set<EdgeIndex>();
        public Set<NodeIndex> initiallyInvalidNodes = new Set<EdgeIndex>();
        public Set<NodeIndex> isIncrement = new Set<EdgeIndex>();

        /// <summary>
        /// Get an index for the lhs expression of a node.  Used for de-duplication.
        /// </summary>
        public Converter<NodeIndex, TargetIndex> getTargetIndex;

        /// <summary>
        /// Has a 1 in each bit position that is required.  bitsProvided defines what bit is provided by each statement.
        /// </summary>
        internal IndexedProperty<NodeIndex, byte> requiredBits;

        /// <summary>
        /// isEssential[node] == true if a node must run even if isUniform[node] == true.
        /// </summary>
        internal IndexedProperty<NodeIndex, bool> isEssential;

        /// <summary>
        /// isUniform[node] == true if a node is known to always be uniform.
        /// </summary>
        internal IndexedProperty<NodeIndex, bool> isUniform;

        // Edge properties
        internal IndexedProperty<EdgeIndex, bool> isRequired;
        internal IndexedProperty<EdgeIndex, byte> bitsProvided;
        internal IndexedProperty<EdgeIndex, bool> isTrigger, isFreshEdge;
        // used by PruningTransform
        internal IndexedProperty<EdgeIndex, bool> isInitializer;
        internal IndexedProperty<EdgeIndex, bool> isDeleted;
        internal IndexedProperty<EdgeIndex, bool> noInit, diode;
        internal IndexedProperty<EdgeIndex, bool> isCancels;
        internal Dictionary<EdgeIndex, IOffsetInfo> OffsetIndices = new Dictionary<EdgeIndex, IOffsetInfo>();

        // Debugging
        public Converter<NodeIndex, string> NodeToString;
        public Converter<NodeIndex, string> NodeToShortString;
        internal IReadOnlyList<IStatement> Nodes;
        internal AttributeRegistry<object, ICompilerAttribute> attributes;
        internal IndexedProperty<NodeIndex, List<string>> EventHistory;
        public int EventCounter;

        /// <summary>
        /// Mark offset edges (on the given loop variables) as deleted
        /// </summary>
        /// <param name="offsetVarsToDelete">If null, all offset edges are deleted.</param>
        /// <param name="isCyclic">True if the offset edges form a cycle</param>
        /// <returns>True if any edge was deleted</returns>
        internal bool DeleteAllOffsetIndexEdges(ICollection<IVariableDeclaration> offsetVarsToDelete, out bool isCyclic)
        {
            isCyclic = false;
            bool anyDeleted = false;
            foreach (KeyValuePair<EdgeIndex, IOffsetInfo> entry in OffsetIndices)
            {
                EdgeIndex edge = entry.Key;
                IOffsetInfo offsetIndices = entry.Value;
                bool isBackward = false;
                bool isForward = false;
                foreach (var entry2 in offsetIndices)
                {
                    if (offsetVarsToDelete != null && !offsetVarsToDelete.Contains(entry2.loopVar))
                        continue;
                    int offset = entry2.offset;
                    if (offset > 0)
                    {
                        isBackward = true;
                    }
                    else if (offset < 0)
                    {
                        isForward = true;
                        if (offset == DependencyAnalysisTransform.sequentialOffset)
                            isCyclic = true;
                    }
                }
                if (isForward || isBackward)
                {
                    isDeleted[edge] = true;
                    anyDeleted = true;
                    // This test is conservative in that it will catch all cases where there is a cycle,
                    // but it may think there is a cycle in cases where there is not.
                    if (isForward && isBackward)
                        isCyclic = true;
                }
            }
            return anyDeleted;
        }
        internal bool DeleteAllOffsetIndexEdges()
        {
            bool ignore;
            return DeleteAllOffsetIndexEdges(null, out ignore);
        }

        internal static string StatementToShortString(IStatement ist)
        {
            IExpression target = TargetExpression(ist);
            string s;
            if (target == null)
            {
                s = ist.ToString();
                if (s.Length > 80)
                    s = s.Substring(0, 80);
            }
            else
                s = target.ToString();
            return s;
        }

        internal static IExpression TargetExpression(IStatement ist)
        {
            if (ist is IForStatement ifs)
                return TargetExpression(ifs.Body.Statements[0]);
            else if (ist is IConditionStatement ics)
                return TargetExpression(ics.Then.Statements[0]);
            else if (ist is IBlockStatement ibs)
                return TargetExpression(ibs.Statements[0]);
            else if (ist is ICommentStatement icms)
            {
                return CodeBuilder.Instance.LiteralExpr(icms.Comment.Text);
            }
            else if (ist is IExpressionStatement ies)
            {
                IExpression expr = ies.Expression;
                if (expr is IAssignExpression iae)
                {
                    expr = iae.Target;
                }
                if (expr is IVariableDeclarationExpression ivde)
                    return CodeBuilder.Instance.LiteralExpr(ivde.Variable.Name);
                else
                    return expr;
            }
            else
                return null;
        }

        internal DependencyGraph(BasicTransformContext context,
            IReadOnlyList<IStatement> nodes,
            IDictionary<IStatement, IStatement> replacements = null,
            bool ignoreMissingNodes = false,
            bool ignoreRequirements = false,
            bool deleteCancels = false,
            bool readAfterWriteOnly = false)
        {
            Dictionary<IStatement, int> indexOfNode = new Dictionary<IStatement, int>(ReferenceEqualityComparer<IStatement>.Instance);
            for (int i = 0; i < nodes.Count; i++)
            {
                indexOfNode[nodes[i]] = i;
            }
            this.Nodes = nodes;
            this.attributes = context.InputAttributes;
            IndexedGraph dependencyGraph = new IndexedGraph(nodes.Count);
            this.dependencyGraph = dependencyGraph;
            this.nodeData = dependencyGraph;
            this.edgeData = dependencyGraph;
            this.getTargetIndex = (i => new TargetIndex(i));
            NodeToString = delegate (int i) { return nodes[i].ToString(); };
            NodeToShortString = delegate (int i)
            {
                string s = StatementToShortString(nodes[i]);
                if (debug && context.InputAttributes.Has<IncrementStatement>(nodes[i]))
                    s += " (Increment)";
                return s;
            };
            List<KeyValuePair<NodeIndex, string>> pendingEvents = new List<KeyValuePair<NodeIndex, string>>();
            requiredBits = dependencyGraph.CreateNodeData<byte>();
            isEssential = dependencyGraph.CreateNodeData<bool>();
            isUniform = dependencyGraph.CreateNodeData<bool>();
            isRequired = dependencyGraph.CreateEdgeData<bool>(false);
            NodeIndex iterationNode = -1;
            if (nodes.Count > 0 && context.InputAttributes.Has<IterationStatement>(nodes[nodes.Count - 1]))
                iterationNode = nodes.Count - 1;
            // there are 3 types of dependencies: http://en.wikipedia.org/wiki/Data_dependency
            // read-after-write, write-after-read, write-after-write
            // start by adding edges corresponding to read-after-write dependencies
            foreach (int targetIndex in dependencyGraph.Nodes)
            {
                if (targetIndex == iterationNode)
                    continue;
                IStatement ist = nodes[targetIndex];
                bool targetIsIncrement = context.InputAttributes.Has<IncrementStatement>(ist);
                if (targetIsIncrement)
                    isIncrement.Add(targetIndex);
                DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                if (di == null)
                {
                    context.Error("Dependency information not found for statement: " + ist);
                    di = new DependencyInformation();
                }
                if (replacements != null && replacements.Count > 0)
                {
                    di = (DependencyInformation)di.Clone();
                    di.Replace(replacements);
                }
                if (di.IsOutput)
                    outputNodes.Add(targetIndex);
                if (di.IsUniform)
                    isUniform[targetIndex] = true;
                if (!context.InputAttributes.Has<OperatorStatement>(ist) || context.InputAttributes.Has<Initializer>(ist))
                {
                    isEssential[targetIndex] = true;
                    if (debug)
                        pendingEvents.Add(new KeyValuePair<NodeIndex, string>(targetIndex, "essential due to !OperatorStatement or Initializer"));
                }
                else
                {
                    IExpression targetExpr = TargetExpression(ist);
                    if (targetExpr != null)
                    {
                        IVariableDeclaration ivd = CodeRecognizer.Instance.GetVariableDeclaration(targetExpr);
                        if (ivd != null)
                        {
                            if (context.InputAttributes.Has<DoesNotHaveInitializer>(ivd))
                            {
                                isEssential[targetIndex] = true;
                                if (debug)
                                    pendingEvents.Add(new KeyValuePair<NodeIndex, string>(targetIndex, "essential due to DoesNotHaveInitializer"));
                            }
                        }
                    }
                }
                if (iterationNode != -1 && context.InputAttributes.Has<DependsOnIteration>(ist))
                {
                    // add edges to and from iterationNode
                    dependencyGraph.AddEdge(iterationNode, targetIndex);
                    dependencyGraph.AddEdge(targetIndex, iterationNode);
                }

                // This condition is not satisfied when an inferred variable has no other uses.
                // In that case, x_B has RequiredBits=1 but no dependencies, which is correct.
                //Assert.IsTrue(di.Dependencies.Count >= di.RequiredBits);

                Set<NodeIndex> sources = new Set<NodeIndex>(); // for fast checking of duplicate sources
                List<NodeIndex> deps = new List<EdgeIndex>();
                foreach (IStatement source in di.Dependencies)
                {
                    int sourceIndex;
                    if (indexOfNode.TryGetValue(source, out sourceIndex))
                    {
                        if (!sources.Contains(sourceIndex))
                        {
                            sources.Add(sourceIndex);
                            deps.Add(sourceIndex);
                        }
                    }
                    else
                    {
                        if (!ignoreMissingNodes)
                            context.Error("Dependency statement not found: " + source);
                    }
                }
                deps.Sort();
                foreach (NodeIndex sourceIndex in deps)
                {
                    int edge = dependencyGraph.AddEdge(sourceIndex, targetIndex);
                    IStatement source = nodes[sourceIndex];
                    if (context.InputAttributes.Has<InitialiseBackward>(source) && targetIsIncrement)
                        initializedEdges.Add(edge);
                    if (di.offsetIndexOf.TryGetValue(source, out IOffsetInfo offsetIndices))
                    {
                        OffsetIndices[edge] = offsetIndices;
                    }
                }
                deps.Clear();
                foreach (IStatement source in di.DeclDependencies)
                {
                    if (indexOfNode.TryGetValue(source, out int sourceIndex))
                    {
                        if (!sources.Contains(sourceIndex))
                        {
                            sources.Add(sourceIndex);
                            deps.Add(sourceIndex);
                        }
                    }
                    else
                    {
                        if (!ignoreMissingNodes)
                            context.Error("DeclDependency statement not found: " + source);
                    }
                }
                deps.Sort();
                foreach (NodeIndex sourceIndex in deps)
                {
                    int edge = dependencyGraph.AddEdge(sourceIndex, targetIndex);
                    isEssential[sourceIndex] = true;
                    if (debug)
                        pendingEvents.Add(new KeyValuePair<NodeIndex, string>(sourceIndex, $"essential due to {targetIndex} {NodeToShortString(targetIndex)}"));
                    if (!ignoreRequirements)
                        isRequired[edge] = true;
                }
                deps.Clear();
            }
            if (!readAfterWriteOnly)
            {
                // now add edges corresponding to write-after-read and write-after-write dependencies
                // must add them at the end so that the graph only contains read-after-write dependencies
                List<Edge<NodeIndex>> newEdges = new List<Edge<NodeIndex>>();
                foreach (int targetIndex in dependencyGraph.Nodes)
                {
                    if (targetIndex == iterationNode)
                        continue;
                    IStatement ist = nodes[targetIndex];
                    DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                    if (di == null)
                    {
                        context.Error("Dependency information not found for statement: " + ist);
                        di = new DependencyInformation();
                    }
                    if (replacements != null && replacements.Count > 0)
                    {
                        di = (DependencyInformation)di.Clone();
                        di.Replace(replacements);
                    }
                    bool targetIsInitializer = context.InputAttributes.Has<Initializer>(ist);
                    foreach (IStatement source in di.Overwrites)
                    {
                        if (indexOfNode.TryGetValue(source, out int sourceIndex))
                        {
                            // write-after-write dependency
                            newEdges.Add(new Edge<NodeIndex>(sourceIndex, targetIndex));
                            // since source initializes the variable updated by ist, any statement that reads the value of source must precede ist
                            // TODO check that the writes overlap and that readerIndex is actually a reader
                            foreach (NodeIndex readerIndex in dependencyGraph.TargetsOf(sourceIndex))
                            {
                                if (readerIndex == targetIndex)
                                    continue;
                                bool readerIsInitializer = context.InputAttributes.Has<Initializer>(nodes[readerIndex]);
                                if (readerIsInitializer && !targetIsInitializer)
                                {
                                    // write-after-read dependency
                                    newEdges.Add(new Edge<NodeIndex>(readerIndex, targetIndex));
                                }
                            }
                        }
                        else
                        {
                            if (!ignoreMissingNodes)
                                context.Error("Initializer statement not found: " + source);
                        }
                    }
                }
                foreach (Edge<NodeIndex> edge in newEdges)
                {
                    if (!dependencyGraph.ContainsEdge(edge.Source, edge.Target))
                        dependencyGraph.AddEdge(edge.Source, edge.Target);
                }
            }
            // the graph is now complete.
            dependencyGraph.IsReadOnly = true;

            if (debug)
                LoggingStart("Construction");
            foreach (var kvp in pendingEvents)
            {
                AddEvent(kvp.Key, kvp.Value);
            }
            noInit = dependencyGraph.CreateEdgeData<bool>(false);
            diode = dependencyGraph.CreateEdgeData<bool>(false);
            isCancels = dependencyGraph.CreateEdgeData<bool>(false);
            bitsProvided = dependencyGraph.CreateEdgeData<byte>();
            isTrigger = dependencyGraph.CreateEdgeData<bool>(false);
            isFreshEdge = dependencyGraph.CreateEdgeData<bool>(false);
            isInitializer = dependencyGraph.CreateEdgeData<bool>(false);
            isDeleted = dependencyGraph.CreateEdgeData<bool>(false);
            foreach (int targetIndex in dependencyGraph.Nodes)
            {
                IStatement ist = nodes[targetIndex];
                DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                if (di == null)
                    di = new DependencyInformation();
                if (replacements != null && replacements.Count > 0)
                {
                    di = (DependencyInformation)di.Clone();
                    di.Replace(replacements);
                }
                // label 'required' edges
                if (!ignoreRequirements)
                    LabelRequiredEdges(indexOfNode, di, ist, targetIndex, di.Requirements, 0, context, ignoreMissingNodes);
                // if requirements are not ignored, then we want to exclude NoInit edges here, otherwise the scheduling constraints will be unsatisfiable.
                LabelRequiredEdges(indexOfNode, di, ist, targetIndex, di.SkipIfUniform.Where(dep => ignoreRequirements || !di.HasDependency(DependencyType.NoInit, dep)), 0, context, ignoreMissingNodes);
                // label 'trigger' edges
                foreach (IStatement source in di.Triggers.Where(dep => ignoreRequirements || !di.HasDependency(DependencyType.NoInit, dep)))
                {
                    if (indexOfNode.TryGetValue(source, out int sourceIndex))
                    {
                        try
                        {
                            int edge = dependencyGraph.GetEdge(sourceIndex, targetIndex);
                            isTrigger[edge] = true;
                        }
                        catch (AmbiguousEdgeException)
                        {
                            context.Error(ist + "has duplicate dependency on: " + source);
                        }
                    }
                    else
                    {
                        if (!IsUniform(context, source) && !di.IsUniform)
                        {
                            if (debug)
                                AddEvent(targetIndex, "initially invalid due to " + source);
                            initiallyInvalidNodes.Add(targetIndex);
                        }
                        if (!ignoreMissingNodes)
                            context.Error("Dependency statement not found: " + source);
                    }
                }
                // label 'fresh' edges
                foreach (IStatement source in di.FreshDependencies)
                {
                    // TEMPORARY: ignore fresh edges with offsets
                    // this is necessary until RepairSchedule is aware of offsets
                    if (di.offsetIndexOf.ContainsKey(source))
                        continue;
                    if (indexOfNode.TryGetValue(source, out int sourceIndex))
                    {
                        try
                        {
                            int edge = dependencyGraph.GetEdge(sourceIndex, targetIndex);
                            isFreshEdge[edge] = true;
                        }
                        catch (AmbiguousEdgeException)
                        {
                            context.Error(ist + "has duplicate dependency on: " + source);
                        }
                    }
                    else
                    {
                        if (!ignoreMissingNodes)
                            context.Error("Dependency statement not found: " + source);
                    }
                }
                // label 'noInit' edges
                foreach (IStatement source in di.GetDependenciesOfType(DependencyType.NoInit))
                {
                    if (indexOfNode.TryGetValue(source, out int sourceIndex))
                    {
                        try
                        {
                            int edge = dependencyGraph.GetEdge(sourceIndex, targetIndex);
                            noInit[edge] = true;
                            if (context.InputAttributes.Has<InitialiseBackward>(source))
                                initializedEdges.Add(edge);
                        }
                        catch (AmbiguousEdgeException)
                        {
                            context.Error(ist + "has duplicate dependency on: " + source);
                        }
                    }
                    else
                    {
                        if (!ignoreMissingNodes)
                            context.Error("Dependency statement not found: " + source);
                    }
                }
                // label 'Diode' edges
                foreach (IStatement source in di.GetDependenciesOfType(DependencyType.Diode))
                {
                    int sourceIndex;
                    if (indexOfNode.TryGetValue(source, out sourceIndex))
                    {
                        try
                        {
                            int edge = dependencyGraph.GetEdge(sourceIndex, targetIndex);
                            diode[edge] = true;
                        }
                        catch (AmbiguousEdgeException)
                        {
                            context.Error(ist + "has duplicate dependency on: " + source);
                        }
                    }
                    else
                    {
                        if (!ignoreMissingNodes)
                            context.Error("Dependency statement not found: " + source);
                    }
                }
                // label 'Cancels' edges
                foreach (IStatement source in di.GetDependenciesOfType(DependencyType.Cancels))
                {
                    if (indexOfNode.TryGetValue(source, out int sourceIndex))
                    {
                        try
                        {
                            int edge = dependencyGraph.GetEdge(sourceIndex, targetIndex);
                            isCancels[edge] = true;
                            // TODO: this deletion should be done outside of the constructor.
                            if (deleteCancels)
                            {
                                isDeleted[edge] = true;
                                if (debug)
                                    Trace.WriteLine($"Deleted cancels edge ({sourceIndex},{targetIndex})");
                            }
                        }
                        catch (AmbiguousEdgeException)
                        {
                            context.Error(ist + "has duplicate dependency on: " + source);
                        }
                    }
                    else
                    {
                        if (!ignoreMissingNodes)
                            context.Error("Dependency statement not found: " + source);
                    }
                }
                // label 'initializer' edges
                foreach (IStatement source in di.Overwrites)
                {
                    if (true)
                    {
                        // If source is non-uniform, then the target is initialized.
                        // Must get DependencyInformation for source since it may not be in the DependencyGraph.
                        Initializer attr = context.GetAttribute<Initializer>(source);
                        if (attr != null && attr.UserInitialized)
                        {
                            if (debug)
                                AddEvent(targetIndex, "initialized by user");
                            initializedNodes.Add(targetIndex);
                            hasNonUniformInitializer.Add(targetIndex);
                        }
                    }
                    int sourceIndex;
                    if (indexOfNode.TryGetValue(source, out sourceIndex))
                    {
                        try
                        {
                            int edge = dependencyGraph.GetEdge(sourceIndex, targetIndex);
                            isInitializer[edge] = true;
                        }
                        catch (AmbiguousEdgeException)
                        {
                            context.Error(ist + "has duplicate dependency on: " + source);
                        }
                    }
                    else
                    {
                        if (!ignoreMissingNodes)
                            context.Error("Dependency statement not found: " + source);
                    }
                }
                // label initiallyStaleNodes
                // a node is initially stale if:
                // 1. it is not always uniform, and
                // 2. it has an external source which is not uniform, and
                // 3. it has no SkipIfUniform dependencies remaining (no requiredBits or isRequired edges)
                if (!isUniform[targetIndex])
                {
                    bool hasExternalSource = false;
                    IStatement externalSource = null;
                    foreach (IStatement source in di.Dependencies)
                    {
                        if (!indexOfNode.ContainsKey(source) && !IsUniform(context, source))
                        {
                            hasExternalSource = true;
                            externalSource = source;
                            break;
                        }
                    }
                    if (hasExternalSource && !IsUniform(targetIndex, node => false))
                    {
                        initiallyStaleNodes.Add(targetIndex);
                        if (debug)
                            AddEvent(targetIndex, "initially stale because of external source " + externalSource);
                    }
                }
            }
            ExtendInitiallyInvalidNodes();
            ExtendInitiallyStaleNodes();
            LoggingStop();
        }

        public void ExtendInitiallyInvalidNodes()
        {
            Set<NodeIndex> nodesToAdd = new Set<EdgeIndex>();
            // extend the set of initiallyInvalidNodes to include triggee descendants
            foreach (NodeIndex node in initiallyInvalidNodes)
            {
                ForEachTriggeeDescendant(node, nodesToAdd.Add);
            }
            initiallyInvalidNodes.AddRange(nodesToAdd);
        }

        public void ExtendInitiallyStaleNodes()
        {
            List<NodeIndex> nodesToAdd = new List<NodeIndex>();
            // extend the set of initiallyStaleNodes to include fresh descendants
            foreach (NodeIndex node in initiallyStaleNodes)
            {
                if (debug)
                    ForEachFreshDescendant(node, node2 =>
                    {
                        if (node2 != node) AddEvent(node2, string.Format("initially stale because of ancestor {0} {1}", node, NodeToShortString(node)));
                        nodesToAdd.Add(node2);
                    });
                else
                    ForEachFreshDescendant(node, nodesToAdd.Add);
            }
            initiallyStaleNodes.AddRange(nodesToAdd);
        }

        // Requirements = (stmt1, Any(stmt2,stmt3), Any(stmt4, stmt5)) goes to 
        //     isRequired[edge1] = true
        //     bitsProvided[edge2] = 1
        //     bitsProvided[edge3] = 1
        //     bitsProvided[edge4] = 2
        //     bitsProvided[edge5] = 2
        //     requiredBits = 3
        internal bool LabelRequiredEdges(Dictionary<IStatement, int> indexOfNode, DependencyInformation di,
            IStatement target, int targetIndex, IEnumerable<IStatement> sources, int anyCounter,
            BasicTransformContext context, bool ignoreMissingNodes)
        {
            bool topLevel = (anyCounter == 0);
            foreach (IStatement source in sources)
            {
                if (source is AnyStatement anyBlock)
                {
                    List<IStatement> children = new List<IStatement>();
                    children.AddRange(anyBlock.Statements);
                    if (children.Count == 1)
                    {
                        bool requirementsMet = LabelRequiredEdges(indexOfNode, di, target, targetIndex, children, anyCounter, context, ignoreMissingNodes);
                        if (requirementsMet && !topLevel)
                            return requirementsMet;
                    }
                    else
                    {
                        if (topLevel)
                            anyCounter++;
                        bool requirementsMet = LabelRequiredEdges(indexOfNode, di, target, targetIndex, children, anyCounter, context, ignoreMissingNodes);
                        if (requirementsMet && !topLevel)
                            return requirementsMet;
                    }
                }
                else
                {
                    if (indexOfNode.TryGetValue(source, out int sourceIndex))
                    {
                        try
                        {
                            int edge = dependencyGraph.GetEdge(sourceIndex, targetIndex);
                            if (topLevel)
                            {
                                isRequired[edge] = true;
                            }
                            else
                            {
                                // inside an AnyStatement
                                if (anyCounter < 1)
                                    throw new InferCompilerException("anyCounter < 1");
                                if (anyCounter > 8)
                                    throw new NotSupportedException("Method has > 8 SkipIfAllUniform annotations");
                                byte bit = (byte)(1 << (anyCounter - 1));
                                bitsProvided[edge] |= bit;
                                requiredBits[targetIndex] |= bit;
                            }
                        }
                        catch (AmbiguousEdgeException)
                        {
                            context.Error(target + "has duplicate dependency on: " + source);
                        }
                        catch (EdgeNotFoundException)
                        {
                            context.Error("Cannot find edge from " + source + " to " + target);
                        }
                    }
                    else if (ignoreMissingNodes)
                    {
                        if (!topLevel && !IsUniform(context, source))
                        {
                            if (anyCounter > 8)
                                throw new NotSupportedException("Method has > 8 SkipIfAllUniform annotations");
                            byte bit = (byte)(1 << (anyCounter - 1));
                            // when a required node is missing and not labelled uniform, assume that it is available.  thus this bit is no longer required.
                            // unchecked is required because the "~" operator returns an int, which then overflows when casting to a byte.
                            requiredBits[targetIndex] &= unchecked((byte)~bit);
                            // no need to consider rest of the AnyStatement
                            return true;
                        }
                    }
                    else
                        context.Error("Dependency statement not found: " + source);
                }
            }
            return false;
        }

        public static bool IsUniform(BasicTransformContext context, IStatement ist)
        {
            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
            return di.IsUniform;
        }

        /// <summary>
        /// Returns true if all instances of node (for all settings of loop variables), must be uniform
        /// </summary>
        /// <param name="node"></param>
        /// <param name="isNotUniform"></param>
        /// <returns></returns>
        public bool IsUniform(NodeIndex node, Predicate<NodeIndex> isNotUniform)
        {
            if (isUniform[node])
                return true;
            // if a message has a non-uniform initializer, then we cannot prune it since it must execute in order to override the initializer
            if (hasNonUniformInitializer.Contains(node) && !isIncrement.Contains(node))
            {
                if (debug)
                    AddEvent(node, "cannot prune due to non-uniform initializer");
                return false;
            }
            byte availableBits = 0;
            string missingString = "";
            foreach (EdgeIndex edge in dependencyGraph.EdgesInto(node))
            {
                NodeIndex source = dependencyGraph.SourceOf(edge);
                if (isNotUniform(source))
                {
                    availableBits |= bitsProvided[edge];
                }
                else
                {
                    // uniform
                    if (isRequired[edge])
                    {
                        if (debug)
                            AddEvent(node, $"{source} {NodeToShortString(source)} is uniform");
                        return true; // requirements not met
                    }
                    if (debug)
                    {
                        if (bitsProvided[edge] > 0)
                            missingString += $" {source} {NodeToShortString(source)} is uniform";
                    }
                }
            }
            if (availableBits < requiredBits[node])
            {
                if (debug)
                    AddEvent(node, missingString);
                return true; // requirements not met
            }
            if (debug)
                AddEvent(node, "requirements met");
            return false;
        }

        /// <summary>
        /// Set the isUniform property by determining which nodes will always be uniform.
        /// </summary>
        public void PropagateUniformNodes()
        {
            if (debug)
                LoggingStart("PropagateUniformNodes");
            var sorter = new CyclicDependencySort<NodeIndex, bool>(dependencyGraph,
                                                                   delegate (NodeIndex node, IndexedProperty<NodeIndex, bool> isScheduled, bool cost)
                                                                   { return IsUniform(node, source => isScheduled[source]); });
            // only nodes whose cost=false will be scheduled
            sorter.Threshold = true;
            //sorter.ScheduleTargetsOnly = true;
            sorter.AddRange(outputNodes);
            // any node not on the schedule must be uniform
            foreach (NodeIndex node in dependencyGraph.Nodes)
            {
                if (!sorter.IsScheduled[node])
                {
                    if (debug)
                        AddEvent(node, "inferred to be unneeded or uniform");
                    isUniform[node] = true;
                }
            }
            if (debug)
                LoggingStop();
        }

        public NodeIndex FindInitializer(NodeIndex node)
        {
            foreach (EdgeIndex edge in dependencyGraph.EdgesInto(node))
            {
                if (isInitializer[edge])
                {
                    NodeIndex source = dependencyGraph.SourceOf(edge);
                    if (isUniform[source] && !isEssential[source])
                        return FindInitializer(source);
                    else
                        return source;
                }
            }
            return node;
        }

        public IEnumerable<NodeIndex> SourcesNeededForOutput(NodeIndex node)
        {
            foreach (NodeIndex source in dependencyGraph.SourcesOf(node))
            {
                if (isUniform[source] && !isEssential[source])
                {
                    yield return FindInitializer(source);
                }
                else
                    yield return source;
            }
        }

        private IEnumerable<NodeIndex> InvalidOrStaleSources(NodeIndex node, ICollection<NodeIndex> invalidNodes, ICollection<TargetIndex> staleNodes,
            ICollection<NodeIndex> scheduled, ICollection<NodeIndex> initialized, Action<NodeIndex> action)
        {
            // could use 'SourcesOf' here
            foreach (EdgeIndex edge in dependencyGraph.EdgesInto(node))
            {
                if (isDeleted[edge])
                    continue;
                NodeIndex source = dependencyGraph.SourceOf(edge);
                if (isFreshEdge[edge] && staleNodes.Contains(getTargetIndex(source)))
                    yield return source;
                if (invalidNodes.Contains(source) && (initialized == null || scheduled.Contains(source)))
                    yield return source;
            }
            action(node);
        }

        public List<NodeIndex> RepairScheduleCyclic(ICollection<NodeIndex> schedule)
        {
            Set<NodeIndex> invalid = new Set<EdgeIndex>();
            Set<TargetIndex> stale = new Set<TargetIndex>();
            while (true)
            {
                // go through the schedule once to build up the set of invalid/stale nodes
                CollectInvalidNodes(schedule, invalid, stale);
                int oldCount = schedule.Count;
                List<NodeIndex> newSchedule = RepairSchedule(schedule, invalid, stale);
                schedule = newSchedule;
                int newCount = schedule.Count;
                if (newCount == oldCount)
                    return newSchedule;
                invalid.Clear();
                stale.Clear();
            }
        }

        /// <summary>
        /// Used to index unique lhs expressions
        /// </summary>
        public struct TargetIndex
        {
            public int value;

            public TargetIndex(int value)
            {
                this.value = value;
            }

            public override bool Equals(object obj)
            {
                if (!(obj is TargetIndex))
                    return false;
                TargetIndex that = (TargetIndex)obj;
                return value.Equals(that.value);
            }

            public override EdgeIndex GetHashCode()
            {
                return value.GetHashCode();
            }

            public override string ToString()
            {
                return value.ToString();
            }
        }

        public IEnumerable<NodeIndex> SourcesOf(NodeIndex node, Predicate<NodeIndex> predicate)
        {
            foreach (EdgeIndex edge in dependencyGraph.EdgesInto(node))
            {
                if (isDeleted[edge])
                    continue;
                NodeIndex source = dependencyGraph.SourceOf(edge);
                if (predicate == null || predicate(source))
                    yield return source;
            }
        }

        public List<NodeIndex> RepairSchedule2Cyclic(ICollection<NodeIndex> schedule)
        {
            if (schedule.Count == 1)
                return new List<NodeIndex>(schedule);  // exit early to handle self-loops
            Set<TargetIndex> freshCopy = new Set<TargetIndex>();
            Set<TargetIndex> fresh = new Set<TargetIndex>();
            // use the schedule to prime the initial set of fresh nodes
            CollectFreshNodes(schedule, fresh);
            int oldCount = schedule.Count;
            while (true)
            {
                freshCopy.Clear();
                freshCopy.AddRange(fresh);
                List<NodeIndex> newSchedule = RepairSchedule2(schedule, freshCopy);
                newSchedule = PruneDeadNodesCyclic(newSchedule);
                CollectFreshNodes(newSchedule, fresh);
                int newCount = newSchedule.Count;
                if (newCount == oldCount)
                {
                    // converged
                    return newSchedule;
                }
                oldCount = newCount;
            }
        }

        public List<NodeIndex> PruneDeadNodesCyclic(IList<NodeIndex> schedule)
        {
            if (schedule.Count == 1)
                return new List<NodeIndex>(schedule);  // exit early to handle self-loops
            Set<NodeIndex> usedNodes = new Set<EdgeIndex>();
            usedNodes.AddRange(schedule);
            List<NodeIndex> lastSchedule = PruneDeadNodes(schedule, usedNodes);
            schedule = lastSchedule;
            // repeat until convergence
            while (true)
            {
                List<NodeIndex> newSchedule = PruneDeadNodes(schedule, usedNodes);
                if (newSchedule.Count == schedule.Count)
                {
                    // converged
                    return newSchedule;
                }
                schedule = newSchedule;
            }
        }

        /// <summary>
        /// Remove nodes from the schedule whose result does not reach usedNodes.
        /// </summary>
        /// <param name="schedule">An ordered list of nodes.</param>
        /// <param name="usedNodes">On entry, the set of nodes whose value is needed at the end of the schedule.
        /// On exit, the set of nodes whose value is needed at the beginning of the schedule.</param>
        /// <returns>A subset of the schedule, in the same order.</returns>
        public List<NodeIndex> PruneDeadNodes(IList<NodeIndex> schedule, Set<NodeIndex> usedNodes)
        {
            List<NodeIndex> newSchedule = new List<NodeIndex>();
            // loop the schedule in reverse order
            for (int i = schedule.Count - 1; i >= 0; i--)
            {
                NodeIndex node = schedule[i];
                bool used = usedNodes.Contains(node);
                usedNodes.Remove(node);
                if (used)
                {
                    newSchedule.Add(node);
                    foreach (NodeIndex source in dependencyGraph.SourcesOf(node))
                    {
                        // if the node is used only by itself (due to cyclic dependency) then consider the node as dead.
                        if (source != node)
                        {
                            usedNodes.Add(source);
                        }
                    }
                }
            }
            newSchedule.Reverse();
            return newSchedule;
        }

        /// <summary>
        /// Add copies of nodes to satisfy fresh/trigger constraints
        /// </summary>
        /// <param name="schedule"></param>
        /// <param name="fresh">On entry, the set of fresh targets at the start of the schedule (including initialized nodes).  On exit, the set of fresh targets at the end of the schedule.</param>
        /// <param name="initialized"></param>
        /// <returns></returns>
        public List<NodeIndex> RepairSchedule2(IEnumerable<NodeIndex> schedule, Set<TargetIndex> fresh, ICollection<NodeIndex> initialized = null)
        {
            if (debug)
                LoggingStart("RepairSchedule2" + (initialized == null ? "" : " Init"));
            List<NodeIndex> newSchedule = new List<NodeIndex>();
            Set<NodeIndex> scheduled = new Set<EdgeIndex>();
            List<NodeIndex> changed = new List<EdgeIndex>();
            DepthFirstSearch<NodeIndex> dfsSources = new DepthFirstSearch<EdgeIndex>(node => FreshSourcesOf(node, source => !fresh.Contains(getTargetIndex(source))), dependencyGraph);
            Stack<NodeIndex> todo = new Stack<EdgeIndex>();
            dfsSources.DiscoverNode += todo.Push;
            dfsSources.BackEdge += delegate (Edge<NodeIndex> edge)
            {
                if (edge.Source != edge.Target)
                    throw new Exception("Cycle of fresh edges");
            };
            DepthFirstSearch<NodeIndex> dfsTargets = new DepthFirstSearch<EdgeIndex>(TriggeesOf, dependencyGraph);
            dfsTargets.FinishNode += delegate (NodeIndex triggee)
            {
                todo.Push(triggee);
                changed.Add(triggee);
            };
            dfsTargets.BackEdge += delegate (Edge<NodeIndex> edge)
            {
                if (edge.Source != edge.Target)
                    throw new Exception("Cycle of trigger edges");
            };
            var dfsStale = new DepthFirstSearch<NodeIndex>(FreshTargetsOf, dependencyGraph);
            dfsStale.DiscoverNode += delegate (NodeIndex target)
            {
                fresh.Remove(getTargetIndex(target));
            };
            foreach (NodeIndex node2 in schedule)
            {
                todo.Push(node2);
                while (todo.Count > 0)
                {
                    int oldCount = todo.Count;
                    NodeIndex node = todo.Pop();
                    // push all stale ancestors on the stack (always includes node)
                    dfsSources.Clear();
                    dfsSources.SearchFrom(node);
                    if (todo.Count != oldCount)
                        continue; // must schedule the new nodes first
                    todo.Pop();
                    // node has no stale ancestors
                    TargetIndex lhs = getTargetIndex(node);
                    bool skip = fresh.Contains(lhs);
                    if (initialized != null && ResultWouldBeUniform(node, scheduled, initialized))
                    {
                        skip = true;
                        fresh.Add(lhs);
                    }
                    if (skip)
                        continue;
                    if (debug)
                    {
                        if (node == node2)
                        {
                            AddEvent(node, "scheduled");
                        }
                        else
                        {
                            // this is useful for debugging an infinite repair loop
                            string s = "repairing " + node;
                            s += " for " + node2 + " " + NodeToShortString(node2);
                            AddEvent(node, s);
                        }
                    }
                    newSchedule.Add(node);
                    scheduled.Add(node);
                    changed.Clear();
                    ForEachTriggeeDescendant(node, changed.Add);
                    // changed is currently in topological order.  
                    // we want to push on the stack in reverse topological order.
                    changed.Reverse();
                    foreach (NodeIndex change in changed)
                    {
                        todo.Push(change);
                    }
                    if (debug)
                    {
                        foreach (NodeIndex change in changed)
                            AddEvent(change, "invalidated by " + node + " " + NodeToShortString(node));
                    }
                    changed.Add(node);
                    // mark children of changed nodes as stale
                    dfsStale.Clear();
                    foreach (NodeIndex source in changed)
                    {
                        // deleted edges must be included here, but not self-loops
                        foreach (NodeIndex target in dependencyGraph.TargetsOf(source))
                        {
                            if (target == source)
                                continue;
                            dfsStale.SearchFrom(target);
                        }
                    }
                    fresh.Add(lhs);
                }
            }
            if (debug)
                LoggingStop();
            return newSchedule;
        }

        /// <summary>
        /// Collect the set of fresh nodes at the end of the schedule
        /// </summary>
        /// <param name="schedule"></param>
        /// <param name="fresh">On entry, the set of fresh targets at the start of the schedule (including initialized nodes).  On exit, the set of fresh targets at the end of the schedule.</param>
        public void CollectFreshNodes(IEnumerable<NodeIndex> schedule, Set<TargetIndex> fresh)
        {
            var dfsStale = new DepthFirstSearch<NodeIndex>(FreshTargetsOf, dependencyGraph);
            dfsStale.DiscoverNode += delegate (NodeIndex target)
            {
                fresh.Remove(getTargetIndex(target));
            };
            List<NodeIndex> changed = new List<EdgeIndex>();
            foreach (NodeIndex node in schedule)
            {
                TargetIndex lhs = getTargetIndex(node);
                changed.Clear();
                ForEachTriggeeDescendant(node, changed.Add);
                changed.Add(node);
                // mark children of changed nodes as stale
                dfsStale.Clear();
                foreach (NodeIndex source in changed)
                {
                    // deleted edges must be included here, but not self-loops
                    foreach (NodeIndex target in dependencyGraph.TargetsOf(source))
                    {
                        if (target == source)
                            continue;
                        dfsStale.SearchFrom(target);
                    }
                }
                fresh.Add(lhs);
            }
        }

        // removes all nodes from fresh
        public List<NodeIndex> RotateSchedule(IEnumerable<NodeIndex> schedule, Set<TargetIndex> fresh)
        {
            // this code is based on CollectFreshNodes
            var dfsStale = new DepthFirstSearch<NodeIndex>(FreshTargetsOf, dependencyGraph);
            dfsStale.DiscoverNode += delegate (NodeIndex target)
            {
                fresh.Remove(getTargetIndex(target));
            };
            List<NodeIndex> changed = new List<EdgeIndex>();
            List<NodeIndex> suffix = new List<EdgeIndex>();
            List<NodeIndex> newSchedule = new List<EdgeIndex>();
            foreach (NodeIndex node in schedule)
            {
                TargetIndex lhs = getTargetIndex(node);
                if (fresh.Contains(lhs))
                {
                    suffix.Add(node);
                    continue;
                }
                newSchedule.Add(node);
                changed.Clear();
                ForEachTriggeeDescendant(node, changed.Add);
                changed.Add(node);
                // mark children of changed nodes as stale
                dfsStale.Clear();
                foreach (NodeIndex source in changed)
                {
                    // deleted edges must be included here, but not self-loops
                    foreach (NodeIndex target in dependencyGraph.TargetsOf(source))
                    {
                        if (target == source)
                            continue;
                        dfsStale.SearchFrom(target);
                    }
                }
            }
            newSchedule.AddRange(suffix);
            return newSchedule;
        }

        // Note this routine does not consider requirements.  That is because InitSchedule should have already handled them.
        public List<NodeIndex> RepairSchedule(IEnumerable<NodeIndex> schedule, Set<NodeIndex> invalid, Set<TargetIndex> stale, ICollection<NodeIndex> initialized = null)
        {
            if (debug)
                LoggingStart("RepairSchedule" + (initialized == null ? "" : " Init"));
            // when initializing, a node cannot become invalid until it is scheduled, and an initialized node cannot become stale until it is scheduled
            // if initialized=null, we assume all nodes have been updated at least once already
            List<NodeIndex> newSchedule = new List<NodeIndex>();
            Set<NodeIndex> scheduled = new Set<EdgeIndex>();
            Set<NodeIndex> changed = new Set<EdgeIndex>();
            List<NodeIndex> sources = new List<NodeIndex>();
            //DepthFirstSearch<NodeIndex> dfs = new DepthFirstSearch<EdgeIndex>(source => InvalidOrStaleSources(source, invalid, stale, scheduled, initialized), dependencyGraph);
            //dfs.FinishNode += sources.Add;
            DepthFirstSearch<NodeIndex> dfs =
                new DepthFirstSearch<EdgeIndex>(source => InvalidOrStaleSources(source, invalid, stale, scheduled, initialized, sources.Add), dependencyGraph);
            List<NodeIndex> emptyList = new List<EdgeIndex>();
            // when initializing, propagating invalid/stale must stop at unscheduled nodes
            DepthFirstSearch<NodeIndex> dfsF = null, dfsT = null;
            if (initialized != null)
            {
                dfsF = new DepthFirstSearch<EdgeIndex>(node =>
                    (initsCanBeStale || scheduled.Contains(node) ||
                      (!initialized.Contains(node) && !ResultWouldBeUniform(node, scheduled, initialized)))
                    ? FreshTargetsOf(node) : emptyList, dependencyGraph);
                dfsT = new DepthFirstSearch<EdgeIndex>(node => scheduled.Contains(node) ? TriggeesOf(node) : emptyList, dependencyGraph);
            }
            foreach (NodeIndex node in schedule)
            {
                while (true)
                {
                    sources.Clear();
                    dfs.Clear();
                    invalid.Add(node); // for InvalidOrStaleSources
                    dfs.SearchFrom(node);
                    NodeIndex source = sources[0];
                    newSchedule.Add(source);
                    if (debug)
                    {
                        if (sources.Count == 1)
                        {
                            AddEvent(source, "scheduled");
                        }
                        else
                        {
                            // this is useful for debugging an infinite repair loop
                            string s = "repairing " + source;
                            if (invalid.Contains(source))
                                s += " was invalid";
                            if (stale.Contains(getTargetIndex(source)))
                                s += " was stale";
                            s += " for " + node + " " + NodeToShortString(node);
                            AddEvent(source, s);
                        }
                    }
                    // This exception occurs when the dependency annotations cannot be satisfied.
                    // One way to proceed is to change some compiler settings (such as OptimiseInferenceCode).
                    // To find the incorrect annotations, collect the text written to console below to get the nodes on the offending cycle.
                    // Then you check all operators on that cycle.  It may help to view the dependency graph with engine.ShowSchedule.
                    // To be able to display the graph, set SchedulingTransform.doRepair = false.
                    if (newSchedule.Count > 100 * dependencyGraph.Nodes.Count)
                    {
                        // find a repeating subsequence of the schedule
                        int posEnd = newSchedule.Count - 1;
                        int posBegin;
                        for (posBegin = newSchedule.Count - 2; posBegin >= 0; posBegin--)
                        {
                            // check for repeated subsequence
                            int count = posEnd - posBegin;
                            if (count - 1 <= posBegin)
                            {
                                bool allMatch = true;
                                for (int i = 0; i < count; i++)
                                {
                                    if (newSchedule[posBegin - i] != newSchedule[posEnd - i])
                                    {
                                        allMatch = false;
                                        break;
                                    }
                                }
                                if (allMatch)
                                    break;
                            }
                        }
                        if (posBegin >= 0)
                        {
                            for (int i = posBegin; i < posEnd; i++)
                            {
                                source = newSchedule[i];
                                // this is useful for debugging an infinite repair loop
                                string s = "repairing " + source;
                                if (invalid.Contains(source))
                                    s += " was invalid";
                                if (stale.Contains(getTargetIndex(source)))
                                    s += " was stale";
                                s += " " + NodeToString(source);
                                Console.WriteLine(s);
                            }
                        }
                        throw new Exception("Internal: RepairSchedule is not converging");
                    }
                    changed.Clear();
                    if (initialized == null)
                        ForEachTriggeeDescendant(source, changed.Add);
                    else
                    {
                        scheduled.Add(source);
                        ForEachTriggeeDescendant(dfsT, source, changed.Add);
                    }
                    invalid.AddRange(changed);
                    if (debug)
                    {
                        foreach (NodeIndex change in changed)
                            AddEvent(change, "invalidated by " + source + " " + NodeToShortString(source));
                    }
                    changed.Add(source);
                    // mark children of changed nodes as stale
                    foreach (NodeIndex node2 in changed)
                    {
                        foreach (EdgeIndex edge in dependencyGraph.EdgesOutOf(node2))
                        {
                            if (isDeleted[edge])
                                continue;
                            NodeIndex target = dependencyGraph.TargetOf(edge);
                            if (target == node2)
                                continue;
                            if (initialized == null)
                                ForEachFreshDescendant(target, i => stale.Add(getTargetIndex(i)));
                            else
                                ForEachFreshDescendant(dfsF, target, i => stale.Add(getTargetIndex(i)));
                        }
                    }
                    invalid.Remove(source);
                    stale.Remove(getTargetIndex(source));
                    if (sources.Count == 1)
                        break;
                }
            }
            if (debug)
                LoggingStop();
            return newSchedule;
        }

        public void CollectInvalidNodes(ICollection<NodeIndex> schedule, Set<NodeIndex> invalid, Set<TargetIndex> stale)
        {
            Set<NodeIndex> changed = new Set<EdgeIndex>();
            int timestamp = -schedule.Count;
            foreach (NodeIndex node in schedule)
            {
                changed.Clear();
                ForEachTriggeeDescendant(node, changed.Add);
                invalid.AddRange(changed);
                changed.Add(node);
                // mark children of changed nodes as stale
                foreach (NodeIndex node2 in changed)
                {
                    foreach (EdgeIndex edge in dependencyGraph.EdgesOutOf(node2))
                    {
                        if (isDeleted[edge])
                            continue;
                        NodeIndex target = dependencyGraph.TargetOf(edge);
                        if (target == node2)
                            continue;
                        ForEachFreshDescendant(target, i => stale.Add(getTargetIndex(i)));
                    }
                }
                invalid.Remove(node);
                stale.Remove(getTargetIndex(node));
            }
        }

        /// <summary>
        /// Check that the schedule meets the constraints given by trigger/fresh annotations, throwing an exception if not
        /// </summary>
        /// <param name="schedule"></param>
        /// <param name="initialized"></param>
        public void CheckSchedule(IEnumerable<NodeIndex> schedule, Set<NodeIndex> initialized)
        {
            // TODO: this code does not work correctly for offset dependencies
            // code is similar to RepairSchedule
            // initialized nodes invalidate their triggees (even if not scheduled)
            Set<NodeIndex> scheduled = new Set<EdgeIndex>();
            Set<NodeIndex> invalid = new Set<NodeIndex>();
            invalid.AddRange(initiallyInvalidNodes);
            Set<TargetIndex> stale = new Set<TargetIndex>();
            foreach (NodeIndex node in initiallyStaleNodes)
                stale.Add(getTargetIndex(node));
            Set<NodeIndex> changed = new Set<EdgeIndex>();
            Set<NodeIndex> sources = new Set<EdgeIndex>();
            List<NodeIndex> emptyList = new List<EdgeIndex>();
            DepthFirstSearch<NodeIndex> dfs =
                new DepthFirstSearch<EdgeIndex>(source => InvalidOrStaleSources(source, invalid, stale, scheduled, initialized, sources.Add), dependencyGraph);
            // when initializing, propagating invalid/stale must stop at unscheduled nodes
            var dfsF = new DepthFirstSearch<EdgeIndex>(node => (scheduled.Contains(node) || (!initialized.Contains(node) && !ResultWouldBeUniform(node, scheduled, initialized))) ? FreshTargetsOf(node) : emptyList, dependencyGraph);
            var dfsT = new DepthFirstSearch<EdgeIndex>(node => scheduled.Contains(node) ? TriggeesOf(node) : emptyList, dependencyGraph);
            foreach (NodeIndex node in schedule)
            {
                sources.Clear();
                dfs.Clear();
                invalid.Add(node);
                dfs.SearchFrom(node);
                if (sources.Count > 1)
                {
                    foreach (NodeIndex source in sources)
                    {
                        if (invalid.Contains(source) && scheduled.Contains(source))
                        {
                            throw new Exception(NodeToShortString(source) + " is invalid for " + NodeToShortString(node));
                        }
                        else
                        {
                            throw new Exception(NodeToShortString(source) + " is stale for " + NodeToShortString(node));
                        }
                    }
                }
                changed.Clear();
                // invalidate triggered nodes
                ForEachTriggeeDescendant(dfsT, node, changed.Add);
                invalid.AddRange(changed);
                changed.Add(node);
                // mark children of changed nodes as stale
                foreach (NodeIndex node2 in changed)
                {
                    foreach (EdgeIndex edge in dependencyGraph.EdgesOutOf(node2))
                    {
                        if (isDeleted[edge])
                            continue;
                        NodeIndex target = dependencyGraph.TargetOf(edge);
                        if (target == node2)
                            continue;
                        ForEachFreshDescendant(dfsF, target, i => stale.Add(getTargetIndex(i)));
                    }
                }
                scheduled.Add(node);
                stale.Remove(getTargetIndex(node));
                invalid.Remove(node);
            }
        }

        private bool ResultWouldBeUniform(NodeIndex node, ICollection<NodeIndex> scheduled, ICollection<NodeIndex> initialized)
        {
            byte availableBits = 0;
            foreach (EdgeIndex edge in dependencyGraph.EdgesInto(node))
            {
                NodeIndex source = dependencyGraph.SourceOf(edge);
                if (scheduled.Contains(source) || initialized.Contains(source) || isDeleted[edge])
                {
                    availableBits |= bitsProvided[edge];
                }
                else if (isRequired[edge])
                {
                    return true;
                }
            }
            if (availableBits < requiredBits[node])
            {
                return true;
            }
            return false;
        }

        /// <summary>
        /// Returns error messages for all nodes which cannot execute in the given schedule.
        /// </summary>
        /// <param name="schedule"></param>
        /// <param name="available">On entry, the set of nodes whose value is available at the start of the schedule.
        /// On exit, the set of nodes whose value is available at the end of the schedule.</param>
        /// <param name="initialize"></param>
        /// <returns></returns>
        public List<string> CollectRequirements(IEnumerable<NodeIndex> schedule, Set<NodeIndex> available, bool initialize)
        {
            List<string> messages = new List<string>();
            foreach (NodeIndex node in schedule)
            {
                if (!initialize || !hasNonUniformInitializer.Contains(node))
                {
                    byte availableBits = 0;
                    foreach (EdgeIndex edge in dependencyGraph.EdgesInto(node))
                    {
                        NodeIndex source = dependencyGraph.SourceOf(edge);
                        if (available.Contains(source) || isDeleted[edge] || isCancels[edge] || initializedEdges.Contains(edge))
                        {
                            availableBits |= bitsProvided[edge];
                        }
                        else if (isRequired[edge])
                        {
                            messages.Add(string.Format("{0} is missing required input {1}.  Try initializing one of these variables.", NodeToString(node), NodeToShortString(source)));
                            availableBits |= bitsProvided[edge];
                        }
                    }
                    if (availableBits < requiredBits[node])
                    {
                        messages.Add(string.Format("{0} is missing required inputs.  Try initializing this variable or one of its inputs.", NodeToString(node)));
                    }
                }
                available.Add(node);
            }
            return messages;
        }

        /// <summary>
        /// Returns a new schedule where nodes without their requirements satisfied are pruned.
        /// </summary>
        /// <param name="schedule"></param>
        /// <param name="available">On entry, the set of nodes whose value is available at the start of the schedule.
        /// On exit, the set of nodes whose value is available at the end of the schedule.</param>
        /// <param name="initialize">If true, initialized nodes are assumed to have all requirements met.</param>
        /// <returns></returns>
        public List<NodeIndex> PruneNodesMissingRequirements(IEnumerable<NodeIndex> schedule, ICollection<NodeIndex> available, bool initialize)
        {
            List<NodeIndex> newSchedule = new List<NodeIndex>();
            foreach (NodeIndex i in schedule)
            {
                byte availableBits = 0;
                bool hasRequirements = true;
                if (initialize && hasNonUniformInitializer.Contains(i))
                {
                    // requirements are met
                }
                else
                {
                    foreach (EdgeIndex edge in dependencyGraph.EdgesInto(i))
                    {
                        NodeIndex source = dependencyGraph.SourceOf(edge);
                        if (available.Contains(source))
                        {
                            availableBits |= bitsProvided[edge];
                        }
                        else if (isRequired[edge])
                        {
                            hasRequirements = false;
                            break;
                        }
                    }
                    if (availableBits < requiredBits[i])
                    {
                        hasRequirements = false;
                    }
                }
                if (hasRequirements)
                {
                    //Console.WriteLine("{0} has requirements: {1}",i,NodeToString(i));
                    newSchedule.Add(i);
                    available.Add(i);
                    // invalidate triggered nodes
                    //foreach (Node triggee in TriggeesOf(i)) {
                    ForEachTriggeeDescendant(i, delegate (NodeIndex triggee) { available.Remove(triggee); });
                }
                else
                {
                    if (debug)
                        Debug.WriteLine("[SchedulingTransform] Missing requirements: " + NodeToString(i));
                }
            }
            return newSchedule;
        }

        // caching this dfs significantly improves memory performance.
        private DepthFirstSearch<NodeIndex> dfsFreshDescendant;
        private bool dfsFreshDescendantIsBusy;

        // action is also applied to node itself
        public void ForEachFreshDescendant(NodeIndex node, Action<NodeIndex> action)
        {
            if (dfsFreshDescendantIsBusy)
                throw new Exception("Nested calls not allowed");
            dfsFreshDescendantIsBusy = true;
            if (dfsFreshDescendant == null)
                dfsFreshDescendant = new DepthFirstSearch<NodeIndex>(FreshTargetsOf, nodeData);
            ForEachFreshDescendant(dfsFreshDescendant, node, action);
            dfsFreshDescendantIsBusy = false;
        }

        internal void ForEachFreshDescendant(DepthFirstSearch<NodeIndex> dfs, NodeIndex node, Action<NodeIndex> action)
        {
            dfs.Clear();
            dfs.ClearActions();
            dfs.DiscoverNode += action;
            bool findloops = false;
            if (findloops)
            {
                dfs.BackEdge += delegate (Edge<NodeIndex> edge)
                    {
                        FindLoop(node, FreshTargetsOf, nodeData, "needs a fresh");
                        //Console.WriteLine("{0} triggers {1}", edge.Source, edge.Target);
                        //Console.WriteLine(StringUtil.JoinColumns("[" + edge.Source + "] ", NodeToString(edge.Source)));
                        //Console.WriteLine(StringUtil.JoinColumns("[" + edge.Target + "] ", NodeToString(edge.Target)));
                        throw new Exception("The model has a loop of fresh edges");
                    };
            }
            dfs.SearchFrom(node);
        }

        // caching this dfs significantly improves memory performance.
        private DepthFirstSearch<NodeIndex> dfsTriggees;
        private bool dfsTriggeesIsBusy;

        // triggees are processed in topological order
        // trigger is not included
        public void ForEachTriggeeDescendant(NodeIndex trigger, Action<NodeIndex> action)
        {
            if (dfsTriggeesIsBusy)
                throw new Exception("Nested calls not allowed");
            dfsTriggeesIsBusy = true;
            if (dfsTriggees == null)
                dfsTriggees = new DepthFirstSearch<NodeIndex>(TriggeesOf, nodeData);
            ForEachTriggeeDescendant(dfsTriggees, trigger, action);
            dfsTriggeesIsBusy = false;
        }

        internal void ForEachTriggeeDescendant(DepthFirstSearch<NodeIndex> dfs, NodeIndex trigger, Action<NodeIndex> action)
        {
            dfs.Clear();
            dfs.ClearActions();
            dfs.TreeEdge += delegate (Edge<NodeIndex> edge) { action(edge.Target); };
            dfs.BackEdge += delegate (Edge<NodeIndex> edge)
                {
                    FindLoop(trigger, TriggeesOf, nodeData, "is triggered by");
                    throw new Exception("The model has a loop of deterministic edges, which is not allowed by Variational Message Passing.");
                };
            dfs.SearchFrom(trigger);
        }

        /// <summary>
        /// Return the set of nodes that this node triggers.
        /// </summary>
        /// <param name="node"></param>
        /// <returns></returns>
        public IEnumerable<NodeIndex> TriggeesOf(NodeIndex node)
        {
            foreach (EdgeIndex edge in dependencyGraph.EdgesOutOf(node))
            {
                if (isTrigger[edge] && !isDeleted[edge])
                {
                    int target = dependencyGraph.TargetOf(edge);
                    if (!isUniform[target])
                        yield return target;
                }
            }
        }

        /// <summary>
        /// Return the set of nodes that would become stale when this node executes.
        /// </summary>
        /// <param name="node"></param>
        /// <returns></returns>
        public IEnumerable<NodeIndex> FreshTargetsOf(NodeIndex node)
        {
            foreach (EdgeIndex edge in dependencyGraph.EdgesOutOf(node))
            {
                if (isFreshEdge[edge] && !isDeleted[edge])
                {
                    int target = dependencyGraph.TargetOf(edge);
                    if (!isUniform[target])
                        yield return target;
                }
            }
        }

        public IEnumerable<NodeIndex> FreshSourcesOf(NodeIndex node, Predicate<NodeIndex> predicate)
        {
            foreach (EdgeIndex edge in dependencyGraph.EdgesInto(node))
            {
                if (isFreshEdge[edge] && !isDeleted[edge])
                {
                    int source = dependencyGraph.SourceOf(edge);
                    if (!isUniform[source] && (predicate == null || predicate(source)))
                        yield return source;
                }
            }
        }

        internal void FindLoop(NodeIndex start, Converter<NodeIndex, IEnumerable<NodeIndex>> successors, CanCreateNodeData<NodeIndex> data, string message)
        {
            bool found = false;
            IndexedProperty<NodeIndex, NodeIndex> parent = data.CreateNodeData<NodeIndex>(0);
            DepthFirstSearch<NodeIndex> dfs = new DepthFirstSearch<NodeIndex>(successors, data);
            dfs.TreeEdge += delegate (Edge<NodeIndex> edge) { parent[edge.Target] = edge.Source; };
            dfs.BackEdge += delegate (Edge<NodeIndex> edge)
                {
                    if (!found)
                    {
                        found = true;
                        NodeIndex target = edge.Target;
                        Console.WriteLine("{0} {2} {1}", edge.Target, edge.Source, message);
                        Console.WriteLine(StringUtil.JoinColumns("[" + edge.Target + "] ", NodeToString(edge.Target)));
                        NodeIndex node = edge.Source;
                        while (!node.Equals(target))
                        {
                            Console.WriteLine("{0} {2} {1}", node, parent[node], message);
                            Console.WriteLine(StringUtil.JoinColumns("[" + node + "] ", NodeToString(node)));
                            node = parent[node];
                        }
                    }
                };
            dfs.SearchFrom(start);
        }

        internal void LoggingStart(string description)
        {
            EventHistory = dependencyGraph.CreateNodeData<List<string>>();
            foreach (NodeIndex node in dependencyGraph.Nodes)
            {
                EventHistory[node] = new List<string>();
                IStatement ist = Nodes[node];
                attributes.Add(ist, new ScheduleHistoryDebug(description, node, EventHistory[node]));
            }
        }

        internal void LoggingStop()
        {
            EventHistory = default(IndexedProperty<EdgeIndex, List<String>>);
        }

        public void AddEvent(NodeIndex node, string message)
        {
            if (EventHistory != null)
            {
                EventHistory[node].Add(String.Format("[{0}] {1}", EventCounter++, message));
            }
        }

        public string NodesToString(IEnumerable<NodeIndex> list)
        {
            StringBuilder s = new StringBuilder();
            bool first = true;
            foreach (int node in list)
            {
                if (!first)
                    s.AppendLine();
                s.Append(NodeToString(node));
                first = false;
            }
            return s.ToString();
        }

        private class ScheduleHistoryDebug : ICompilerAttribute
        {
            public string description;
            public int nodeIndex;
            public List<string> history;

            public ScheduleHistoryDebug(string description, int nodeIndex, List<string> history)
            {
                this.description = description;
                this.nodeIndex = nodeIndex;
                this.history = history;
            }

            public override string ToString()
            {
                StringBuilder s = new StringBuilder();
                s.AppendFormat("ScheduleHistoryDebug({0})(node {1}):", description, nodeIndex);
                foreach (string line in history)
                {
                    s.AppendLine();
                    s.Append(line);
                }
                return s.ToString();
            }
        }
    }
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}