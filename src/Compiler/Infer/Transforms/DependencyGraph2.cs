// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Stores dependencies between statements based on their order.
    /// Includes read-after-write and write-after-read dependencies.
    /// Does not include write-after-write dependencies, but does include write-after-alloc dependencies.
    /// </summary>
    internal class DependencyGraph2
    {
        public List<IStatement> nodes = new List<IStatement>();
        public Set<NodeIndex> outputNodes = new Set<EdgeIndex>();
        public IndexedGraph dependencyGraph = new IndexedGraph();

        /// <summary>
        /// True if the target writes to a location that the source reads.  This is known as an anti-dependency.
        /// </summary>
        public IndexedProperty<EdgeIndex, bool> isWriteAfterRead;

        /// <summary>
        /// True if the target writes to a location that the source writes.  This is known as an anti-dependency.
        /// </summary>
        public IndexedProperty<EdgeIndex, bool> isWriteAfterWrite;

        /// <summary>
        /// Maps from a statement to the index of every position it occurs in the schedule (but only if it occurs more than once)
        /// </summary>
        public Dictionary<IStatement, Set<NodeIndex>> duplicates = new Dictionary<IStatement, Set<NodeIndex>>(new IdentityComparer<IStatement>());

        /// <summary>
        /// Maps from a source statement (which has not yet received an index) to the index of its targets
        /// </summary>
        public Dictionary<IStatement, Set<NodeIndex>> backEdges = new Dictionary<IStatement, Set<EdgeIndex>>(new IdentityComparer<IStatement>());

        /// <summary>
        /// Maps from a statement to the last position it occurs in the schedule
        /// </summary>
        public Dictionary<IStatement, NodeIndex> indexOfNode = new Dictionary<IStatement, int>(new IdentityComparer<IStatement>());

        public enum BackEdgeHandling
        {
            Ignore,
            Include,
            Reverse
        }

        public DependencyGraph2(BasicTransformContext context, 
            IEnumerable<IStatement> inputs, 
            BackEdgeHandling backEdgeHandling,
            Action<IWhileStatement> beginWhile, 
            Action<IWhileStatement> endWhile,
            Action<IConditionStatement> beginFirstIterPost,
            Action<IConditionStatement> endFirstIterPost,
            Action<IStatement, NodeIndex> action)
        {
            Set<NodeIndex> nodesInCurrentWhile = new Set<EdgeIndex>();
            int whileDepth = 0;

            // create a dependency graph where while loops are flattened (the while loops themselves are not nodes)
            // the graph will only contain read-after-write and write-after-alloc dependencies for now
            // add write-after-read dependencies
            isWriteAfterRead = dependencyGraph.CreateEdgeData(false);
            DeadCodeTransform.ForEachStatement(inputs,
                                               delegate (IWhileStatement iws)
                                               {
                                                   beginWhile(iws);
                                                   nodesInCurrentWhile.Clear();
                                                   whileDepth++;
                                               },
                                               delegate (IWhileStatement iws)
                                               {
                                                   // all duplicates in a while loop should share all targets
                                                   foreach (KeyValuePair<IStatement, Set<NodeIndex>> entry in duplicates)
                                                   {
                                                       IStatement ist = entry.Key;
                                                       Set<NodeIndex> set = entry.Value;
                                                       Set<NodeIndex> targets = new Set<EdgeIndex>();
                                                       // collect all targets in the while loop
                                                       foreach (NodeIndex node in set)
                                                       {
                                                           foreach (NodeIndex target in dependencyGraph.TargetsOf(node))
                                                           {
                                                               if (nodesInCurrentWhile.Contains(target))
                                                                   targets.Add(target);
                                                           }
                                                       }
                                                       Set<NodeIndex> backEdgeTargets = null;
                                                       if (!backEdges.TryGetValue(ist, out backEdgeTargets))
                                                       {
                                                           backEdgeTargets = new Set<EdgeIndex>();
                                                           backEdges[ist] = backEdgeTargets;
                                                       }
                                                       // add all targets to all duplicates in the loop
                                                       foreach (NodeIndex node in set)
                                                       {
                                                           if (!nodesInCurrentWhile.Contains(node))
                                                               continue;
                                                           foreach (NodeIndex target in targets)
                                                           {
                                                               if (!dependencyGraph.ContainsEdge(node, target))
                                                               {
                                                                   if (backEdgeHandling == BackEdgeHandling.Include)
                                                                       dependencyGraph.AddEdge(node, target);
                                                                   else if (backEdgeHandling == BackEdgeHandling.Reverse)
                                                                   {
                                                                       if (!dependencyGraph.ContainsEdge(target, node))
                                                                           dependencyGraph.AddEdge(target, node);
                                                                   }
                                                                   if (target < node)
                                                                   {
                                                                       if (backEdgeTargets == null && !backEdges.TryGetValue(ist, out backEdgeTargets))
                                                                       {
                                                                           backEdgeTargets = new Set<EdgeIndex>();
                                                                           backEdges[ist] = backEdgeTargets;
                                                                       }
                                                                       backEdgeTargets.Add(target);
                                                                   }
                                                               }
                                                           }
                                                       }
                                                   }
                                                   endWhile(iws);
                                                   whileDepth--;
                                               },
                                               beginFirstIterPost,
                                               endFirstIterPost,
                                               delegate (IStatement ist)
                                               {
                                                   DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                                                   if (di == null)
                                                   {
                                                       context.Error("Dependency information not found for statement: " + ist);
                                                       di = new DependencyInformation();
                                                   }
                                                   NodeIndex targetIndex = dependencyGraph.AddNode();
                                                   Set<NodeIndex> backEdgeTargets;
                                                   Set<NodeIndex> sources = new Set<NodeIndex>(); // for fast checking of duplicate sources
                                                   foreach (IStatement source in
                                                           di.GetDependenciesOfType(DependencyType.Dependency | DependencyType.Declaration))
                                                   {
                                                       int sourceIndex;
                                                       // we assume that the statements are already ordered properly to respect dependencies.
                                                       // if the source is not in indexOfNode, then it must be a cyclic dependency in this while loop.
                                                       if (indexOfNode.TryGetValue(source, out sourceIndex))
                                                       {
                                                           if (!sources.Contains(sourceIndex))
                                                           {
                                                               sources.Add(sourceIndex);
                                                               EdgeIndex edge = dependencyGraph.AddEdge(sourceIndex, targetIndex);
                                                           }
                                                       }
                                                       else
                                                           sourceIndex = -1;
                                                       if (sourceIndex == -1)
                                                       {
                                                           // add a back edge
                                                           if (!backEdges.TryGetValue(source, out backEdgeTargets))
                                                           {
                                                               backEdgeTargets = new Set<EdgeIndex>();
                                                               backEdges[source] = backEdgeTargets;
                                                           }
                                                           backEdgeTargets.Add(targetIndex);
                                                           // add a dependency on the initializers of source
                                                           Stack<IStatement> todo = new Stack<IStatement>();
                                                           todo.Push(source);
                                                           while (todo.Count > 0)
                                                           {
                                                               IStatement source2 = todo.Pop();
                                                               DependencyInformation di2 = context.InputAttributes.Get<DependencyInformation>(source2);
                                                               if (di2 == null)
                                                               {
                                                                   context.Error("Dependency information not found for statement: " + source2);
                                                                   continue;
                                                               }
                                                               foreach (IStatement init in di2.Overwrites)
                                                               {
                                                                   int initIndex;
                                                                   if (indexOfNode.TryGetValue(init, out initIndex))
                                                                   {
                                                                       if (!sources.Contains(initIndex))
                                                                       {
                                                                           sources.Add(initIndex);
                                                                           EdgeIndex edge = dependencyGraph.AddEdge(initIndex, targetIndex);
                                                                       }
                                                                   }
                                                                   else
                                                                       todo.Push(init);
                                                               }
                                                           }
                                                       }
                                                   }
                                                   if (indexOfNode.ContainsKey(ist))
                                                   {
                                                       Set<int> set;
                                                       if (!duplicates.TryGetValue(ist, out set))
                                                       {
                                                           set = new Set<int>();
                                                           duplicates[ist] = set;
                                                           set.Add(indexOfNode[ist]);
                                                       }
                                                       set.Add(targetIndex);
                                                   }
                                                   // the same statement may appear multiple times.  when looking up indexOfNode, we want to use the last occurrence of the statement.
                                                   indexOfNode[ist] = targetIndex; // must do this at the end, in case the stmt depends on a previous occurrence of itself
                                                   nodesInCurrentWhile.Add(targetIndex);
                                                   nodes.Add(ist);
                                                   if (backEdgeHandling != BackEdgeHandling.Ignore && backEdges.TryGetValue(ist, out backEdgeTargets))
                                                   {
                                                       // now that ist has an index, we can fill in the back edges
                                                       foreach (NodeIndex node in backEdgeTargets)
                                                       {
                                                           if (backEdgeHandling == BackEdgeHandling.Include)
                                                           {
                                                               if (dependencyGraph.ContainsEdge(targetIndex, node))
                                                                   throw new Exception("Internal: back edge already present");
                                                               dependencyGraph.AddEdge(targetIndex, node);
                                                           }
                                                           else if (backEdgeHandling == BackEdgeHandling.Reverse)
                                                           {
                                                               // make a new edge, even if one exists.
                                                               EdgeIndex edge = dependencyGraph.AddEdge(node, targetIndex);
                                                               isWriteAfterRead[edge] = true;
                                                           }
                                                           else throw new NotSupportedException($"backEdgeHandling == {backEdgeHandling}");
                                                       }
                                                   }
                                                   action(ist, targetIndex);
                                               });
            bool includeWriteAfterRead = false;
            if (includeWriteAfterRead)
            {
                // loop statements in their original order
                foreach (NodeIndex target in dependencyGraph.Nodes)
                {
                    IStatement ist = nodes[target];
                    if (ist is IWhileStatement)
                        continue;
                    foreach (NodeIndex source in GetPreviousReaders(context, dependencyGraph, target, nodes, indexOfNode).ToReadOnlyList())
                    {
                        if (source > target)
                            throw new Exception("Internal: source statement follows target");
                        // make a new edge, even if one exists.
                        EdgeIndex edge = dependencyGraph.AddEdge(source, target);
                        isWriteAfterRead[edge] = true;
                    }
                }
            }
            isWriteAfterWrite = dependencyGraph.CreateEdgeData(false);
            // loop statements in their original order
            foreach (NodeIndex target in dependencyGraph.Nodes)
            {
                IStatement ist = nodes[target];
                if (ist is IWhileStatement)
                    continue;

                foreach (NodeIndex source in GetOverwrites(context, dependencyGraph, target, nodes, indexOfNode).ToReadOnlyList())
                {
                    if (source > target)
                        throw new Exception("Internal: source statement follows target");
                    if (dependencyGraph.ContainsEdge(source, target))
                    {
                        foreach (EdgeIndex edge in dependencyGraph.EdgesLinking(source, target))
                            isWriteAfterWrite[edge] = true;
                    }
                    else
                    { 
                        EdgeIndex edge = dependencyGraph.AddEdge(source, target);
                        isWriteAfterWrite[edge] = true;
                    }
                }
            }

            dependencyGraph.NodeCountIsConstant = true;
            dependencyGraph.IsReadOnly = true;
            for (int targetIndex = 0; targetIndex < dependencyGraph.Nodes.Count; targetIndex++)
            {
                IStatement ist = nodes[targetIndex];
                DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                if (di == null) continue;
                if (di.IsOutput) outputNodes.Add(targetIndex);
            }
        }

        internal static IEnumerable<NodeIndex> GetPreviousReaders(BasicTransformContext context, IndexedGraph dependencyGraph, NodeIndex writer,
                                                   List<IStatement> nodes, Dictionary<IStatement, int> indexOfNode)
        {
            IStatement ist = nodes[writer];
            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
            if (di == null)
            {
                context.Error("Dependency information not found for statement: " + ist);
                di = new DependencyInformation();
            }
            foreach (IStatement previousWriteStmt in di.Overwrites)
            {
                NodeIndex previousWriter = indexOfNode[previousWriteStmt];
                foreach (NodeIndex reader in dependencyGraph.TargetsOf(previousWriter))
                {
                    if (reader >= writer)
                        continue;
                    yield return reader;
                }
            }
        }

        internal static IEnumerable<NodeIndex> GetOverwrites(BasicTransformContext context, IndexedGraph dependencyGraph, NodeIndex writer,
                                                   List<IStatement> nodes, Dictionary<IStatement, int> indexOfNode)
        {
            IStatement ist = nodes[writer];
            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
            if (di == null)
            {
                context.Error("Dependency information not found for statement: " + ist);
                di = new DependencyInformation();
            }
            foreach (IStatement previousWriteStmt in di.Overwrites)
            {
                NodeIndex previousWriter = indexOfNode[previousWriteStmt];
                if(previousWriter < writer)
                    yield return previousWriter;
            }
        }
    }
}