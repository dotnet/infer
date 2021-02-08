// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;
using Microsoft.ML.Probabilistic.Compiler.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Remove statements whose result is never used.  Attach InitializerSet attribute to while loops.
    /// </summary>
    internal class DeadCodeTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "DeadCodeTransform"; }
        }

        public ModelCompiler compiler;
        private bool pruneDeadCode;
        private LoopMergingInfo loopMergingInfo;

        public DeadCodeTransform(ModelCompiler compiler, bool pruneDeadCode)
        {
            this.compiler = compiler;
            this.pruneDeadCode = pruneDeadCode;
        }

        protected override IMethodDeclaration DoConvertMethod(IMethodDeclaration md, IMethodDeclaration imd)
        {
            if (!context.InputAttributes.Has<OperatorMethod>(imd)) return imd;
            return base.DoConvertMethod(md, imd);
        }

        /// <summary>
        /// Removes dead code in the method body.
        /// </summary>
        /// <param name="outputs"></param>
        /// <param name="inputs"></param>
        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            IMethodDeclaration imd = context.FindAncestor<IMethodDeclaration>();
            loopMergingInfo = context.InputAttributes.Get<LoopMergingInfo>(imd);
            // imd must have the OperatorMethod attribute
            IList<IStatement> isc = Schedule(inputs);
            outputs.AddRange(isc);
        }

        /// <summary>
        /// Adds the original statement to the output, without transforming.  All attributes of the statement are preserved.
        /// </summary>
        /// <param name="ist"></param>
        /// <returns></returns>
        protected override IStatement DoConvertStatement(IStatement ist)
        {
            return ist;
        }

        public static void ForEachStatement(IEnumerable<IStatement> isc, 
            Action<IWhileStatement> beginWhile, 
            Action<IWhileStatement> endWhile, 
            Action<IConditionStatement> beginFirstIterPost,
            Action<IConditionStatement> endFirstIterPost,
            Action<IStatement> action)
        {
            foreach (IStatement stmt in isc)
            {
                if (stmt is IWhileStatement)
                {
                    IWhileStatement iws = (IWhileStatement)stmt;
                    beginWhile(iws);
                    ForEachStatement(iws.Body.Statements, beginWhile, endWhile, beginFirstIterPost, endFirstIterPost, action);
                    endWhile(iws);
                    continue;
                }
                else if (stmt is IConditionStatement)
                {
                    IConditionStatement ics = (IConditionStatement)stmt;
                    bool testsIteration = Recognizer.GetVariables(ics.Condition).Any(ivd => (ivd.Name == "iteration"));
                    if (testsIteration)
                    {
                        beginFirstIterPost(ics);
                        ForEachStatement(ics.Then.Statements, beginWhile, endWhile, beginFirstIterPost, endFirstIterPost, action);
                        endFirstIterPost(ics);
                        continue;
                    }
                    // fall through
                }
                action(stmt);
            }
        }

        protected class StatementBlock
        {
            public List<NodeIndex> indices = new List<int>();
        }

        protected class Loop : StatementBlock
        {
            public readonly IWhileStatement loopStatement;

            /// <summary>
            /// Nodes in the loop whose outputs are used by the loop before they are updated.
            /// </summary>
            public Set<IStatement> initializers = new Set<IStatement>(new IdentityComparer<IStatement>());

            public List<NodeIndex> tail;
            public List<NodeIndex> firstIterPostBlock = new List<NodeIndex>();

            public Loop(IWhileStatement loopStatement)
            {
                this.loopStatement = loopStatement;
            }
        }

        protected class StraightLine : StatementBlock
        {
        }

        protected IList<IStatement> Schedule(IList<IStatement> isc)
        {
            List<StatementBlock> blocks = new List<StatementBlock>();
            StatementBlock currentBlock = new StraightLine();
            Dictionary<NodeIndex, StatementBlock> blockOfNode = new Dictionary<NodeIndex, StatementBlock>();
            int firstIterPostBlockCount = 0;
            IConditionStatement firstIterPostStatement = null;
            // must include back edges for computing InitializerSets
            DependencyGraph2 g = new DependencyGraph2(context, isc, DependencyGraph2.BackEdgeHandling.Include,
                                                      delegate (IWhileStatement iws)
                                                      {
                                                          blocks.Add(currentBlock);
                                                          currentBlock = new Loop(iws);
                                                      },
                                                      delegate (IWhileStatement iws)
                                                      {
                                                          blocks.Add(currentBlock);
                                                          currentBlock = new StraightLine();
                                                      },
                                                      delegate (IConditionStatement ics)
                                                      {
                                                          firstIterPostBlockCount++;
                                                          firstIterPostStatement = ics;
                                                      },
                                                      delegate (IConditionStatement ics)
                                                      {
                                                          firstIterPostBlockCount--;
                                                      },
                                                      delegate (IStatement ist, int index)
                                                      {
                                                          if (firstIterPostBlockCount > 0)
                                                              ((Loop)currentBlock).firstIterPostBlock.Add(index);
                                                          currentBlock.indices.Add(index);
                                                          blockOfNode[index] = currentBlock;
                                                      });
            var dependencyGraph = g.dependencyGraph;
            blocks.Add(currentBlock);

            Set<NodeIndex> usedNodes = Set<NodeIndex>.FromEnumerable(g.outputNodes);
            Set<NodeIndex> usedBySelf = new Set<NodeIndex>();
            // loop blocks in reverse order
            for (int i = blocks.Count - 1; i >= 0; i--)
            {
                StatementBlock block = blocks[i];
                if (block is Loop)
                {
                    Loop loop = (Loop) block;
                    if (!pruneDeadCode)
                    {
                        usedNodes = CollectUses(dependencyGraph, block.indices);
                    }
                    else
                    {
                        usedBySelf.Clear();
                        List<NodeIndex> tailStmts;
                        block.indices = PruneDeadNodesCyclic(g, block.indices, usedNodes, usedBySelf, out tailStmts); // modifies usedNodes
                        loop.tail = tailStmts;
                    }
                    RemoveSuffix(block.indices, loop.firstIterPostBlock);
                }
                else
                {
                    // StraightLine
                    if (pruneDeadCode)
                    {
                        block.indices = PruneDeadNodes(g, block.indices, usedNodes, usedBySelf); // modifies usedNodes
                    }
                }
                AddLoopInitializers(block, usedNodes, blockOfNode, g);
            }

            IList<IStatement> sc = Builder.StmtCollection();
            foreach (StatementBlock block in blocks)
            {
                if (block is Loop)
                {
                    Loop loop = (Loop) block;
                    context.OpenStatement(loop.loopStatement);
                    IWhileStatement ws = Builder.WhileStmt(loop.loopStatement);
                    context.SetPrimaryOutput(ws);
                    IList<IStatement> sc2 = ws.Body.Statements;
                    foreach (NodeIndex i in loop.indices)
                    {
                        IStatement st = ConvertStatement(g.nodes[i]);
                        sc2.Add(st);
                    }
                    context.CloseStatement(loop.loopStatement);
                    context.InputAttributes.CopyObjectAttributesTo(loop.loopStatement, context.OutputAttributes, ws);
                    sc.Add(ws);
                    List<IStatement> initStmts = new List<IStatement>();
                    initStmts.AddRange(loop.initializers);
                    if (loop.firstIterPostBlock.Count > 0)
                    {
                        var firstIterPostStatements = loop.firstIterPostBlock.Select(i => g.nodes[i]);
                        var thenBlock = Builder.BlockStmt();
                        ConvertStatements(thenBlock.Statements, firstIterPostStatements);
                        var firstIterPostStmt = Builder.CondStmt(firstIterPostStatement.Condition, thenBlock);
                        context.OutputAttributes.Set(firstIterPostStmt, new FirstIterationPostProcessingBlock());
                        sc2.Add(firstIterPostStmt);
                        loopMergingInfo.AddNode(firstIterPostStmt);
                    }
                    context.OutputAttributes.Remove<InitializerSet>(ws);
                    context.OutputAttributes.Set(ws, new InitializerSet(initStmts));
                    if (loop.tail != null)
                    {
                        foreach (NodeIndex i in loop.tail)
                        {
                            IStatement st = g.nodes[i];
                            sc.Add(st);
                        }
                    }
                }
                else
                {
                    foreach (NodeIndex i in block.indices)
                    {
                        IStatement st = ConvertStatement(g.nodes[i]);
                        sc.Add(st);
                    }
                }
            }
            return sc;
        }

        // for each back edge whose source is in a while loop, add the appropriate initializer of source to the InitializerSet
        private void AddLoopInitializers(StatementBlock block, ICollection<NodeIndex> usedNodes, Dictionary<NodeIndex, StatementBlock> blockOfNode, DependencyGraph2 g)
        {
            if (block is Loop)
            {
                Loop loop = (Loop)block;
                AddLoopInitializers(loop.tail, usedNodes, blockOfNode, g);
                AddLoopInitializers(loop.firstIterPostBlock, usedNodes, blockOfNode, g);
            }
            AddLoopInitializers(block.indices, usedNodes, blockOfNode, g);
        }

        private void AddLoopInitializers(IEnumerable<NodeIndex> nodes, ICollection<NodeIndex> usedNodes, Dictionary<NodeIndex, StatementBlock> blockOfNode, DependencyGraph2 g)
        {
            foreach (NodeIndex node in nodes)
            {
                Loop nodeLoop = blockOfNode[node] as Loop;
                // when the target is also in a while loop, you need the initializer to precede target's entire loop
                NodeIndex target = (nodeLoop == null || nodeLoop.indices.Count == 0) ? node : nodeLoop.indices[0];
                foreach (NodeIndex source in g.dependencyGraph.SourcesOf(node))
                {
                    if (source >= node && usedNodes.Contains(source) && blockOfNode[source] is Loop)
                    {
                        Loop loop = (Loop)blockOfNode[source];
                        ForEachInitializer(g.nodes[source], target, g, loop.initializers.Add);
                    }
                }
            }
        }

        private void ForEachInitializer(IStatement source, NodeIndex target, DependencyGraph2 g, Action<IStatement> action)
        {
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
                    int initIndex = g.indexOfNode[init];
                    if (initIndex < target)
                    {
                        // found a valid initializer
                        action(init);
                    }
                    else
                    {
                        // keep looking backward for a valid initializer
                        todo.Push(init);
                    }
                }
            }
        }

        private static void RemoveSuffix(List<NodeIndex> list, List<NodeIndex> suffix)
        {
            for (int i = suffix.Count - 1; i >= 0; i--)
            {
                int pos = list.Count - 1;
                if (list[pos] == suffix[i])
                {
                    list.RemoveAt(pos);
                }
            }
        }

        /// <summary>
        /// Returns all nodes in the schedule whose target appears prior to the node.
        /// </summary>
        /// <param name="g"></param>
        /// <param name="schedule"></param>
        /// <param name="isDeleted"></param>
        /// <returns></returns>
        internal static Set<NodeIndex> CollectUses(IndexedGraph g, IEnumerable<NodeIndex> schedule, Func<EdgeIndex, bool> isDeleted = null)
        {
            Set<NodeIndex> uses = new Set<NodeIndex>();
            Set<NodeIndex> available = new Set<NodeIndex>();
            foreach (NodeIndex node in schedule)
            {
                foreach (EdgeIndex edge in g.EdgesInto(node).Where(e => isDeleted == null || !isDeleted(e)))
                {
                    NodeIndex source = g.SourceOf(edge);
                    if (!available.Contains(source)) uses.Add(source);
                }
                available.Add(node);
            }
            return uses;
        }

        /// <summary>
        /// Remove nodes from the cyclic schedule whose result does not reach usedNodes.
        /// </summary>
        /// <param name="g"></param>
        /// <param name="schedule"></param>
        /// <param name="usedNodes">On entry, the set of nodes whose value is needed at the end of the schedule.
        /// On exit, the set of nodes whose value is needed at the beginning of the schedule.</param>
        /// <param name="usedBySelf">On entry, the set of nodes whose value is used only by itself (due to a cyclic dependency).
        /// On exit, the set of nodes whose value is used only by itself.</param>
        /// <param name="tailSchedule">On exit, a special schedule to use for the final iteration.  Empty if the final schedule is the same as the regular schedule.</param>
        /// <returns>A new schedule.</returns>
        /// <remarks>
        /// We cannot simply do a graph search because we want to compute reachability with respect to the nodes 
        /// that are actually on the schedule.
        /// </remarks>
        private static List<NodeIndex> PruneDeadNodesCyclic(DependencyGraph2 g, IList<NodeIndex> schedule, ICollection<NodeIndex> usedNodes, ICollection<NodeIndex> usedBySelf,
                                                            out List<NodeIndex> tailSchedule)
        {
            tailSchedule = new List<NodeIndex>();
            List<NodeIndex> lastSchedule = PruneDeadNodes(g, schedule, usedNodes, usedBySelf);
            Set<NodeIndex> everUsed = new Set<EdgeIndex>();
            everUsed.AddRange(usedNodes);
            // repeat until convergence
            while (true)
            {
                // this prevents usedNodes from getting smaller 
                usedNodes.AddRange(everUsed);
                List<NodeIndex> newSchedule = PruneDeadNodes(g, schedule, usedNodes, usedBySelf);
                int usedNodeCount = everUsed.Count;
                everUsed.AddRange(usedNodes);
                if (everUsed.Count == usedNodeCount)
                {
                    // converged
                    // does lastSchedule have a statement not in newSchedule?
                    foreach (NodeIndex node in lastSchedule)
                    {
                        if (!newSchedule.Contains(node))
                        {
                            tailSchedule.Add(node);
                        }
                    }
                    return newSchedule;
                }
            }
        }

        /// <summary>
        /// Remove nodes from the schedule whose result does not reach usedNodes.
        /// </summary>
        /// <param name="g"></param>
        /// <param name="schedule">An ordered list of nodes.</param>
        /// <param name="usedNodes">On entry, the set of nodes whose value is needed at the end of the schedule.
        /// On exit, the set of nodes whose value is needed at the beginning of the schedule.</param>
        /// <param name="usedBySelf">On entry, the set of nodes whose value is used only by itself (due to a cyclic dependency).
        /// On exit, the set of nodes whose value is used only by itself.</param>
        /// <returns>A subset of the schedule, in the same order.</returns>
        private static List<NodeIndex> PruneDeadNodes(DependencyGraph2 g, IList<NodeIndex> schedule, ICollection<NodeIndex> usedNodes, ICollection<NodeIndex> usedBySelf)
        {
            List<NodeIndex> newSchedule = new List<NodeIndex>();
            // loop the schedule in reverse order
            for (int i = schedule.Count - 1; i >= 0; i--)
            {
                NodeIndex node = schedule[i];
                bool used = usedNodes.Contains(node);
                if (usedBySelf.Contains(node))
                {
                    // if the node is used only by itself (due to cyclic dependency) then consider the node as dead.
                    used = false;
                    // initializers must still be considered used
                    foreach (EdgeIndex edge in g.dependencyGraph.EdgesInto(node))
                    {
                        NodeIndex source = g.dependencyGraph.SourceOf(edge);
                        if (source == node) continue;
                        usedNodes.Add(source);
                        if (source != node && g.nodes[source] == g.nodes[node]) usedBySelf.Add(source);
                    }
                }
                usedNodes.Remove(node);
                if (used)
                {
                    newSchedule.Add(node);
                    foreach (NodeIndex source in g.dependencyGraph.SourcesOf(node))
                    {
                        if (!usedNodes.Contains(source)) usedNodes.Add(source);
                        // a node is added to usedBySelf only if it is used by a copy of itself.
                        // we don't need to have multiple copies of a self-loop, but we do need one instance of it.
                        if (source != node && g.nodes[source] == g.nodes[node]) usedBySelf.Add(source);
                        else usedBySelf.Remove(source);
                    }
                }
            }
            newSchedule.Reverse();
            return newSchedule;
        }
    }

    /// <summary>
    /// An attribute attached to while loops.  Holds the set of message updates whose outputs are used before being updated by the loop.
    /// These messages represent the state of the loop from one iteration to the next.
    /// </summary>
    internal class InitializerSet : ICompilerAttribute
    {
        /// <summary>
        /// Message updates whose outputs are used before being updated by the loop.
        /// </summary>
        public ICollection<IStatement> initializers;

        /// <summary>
        /// Replace initializers
        /// </summary>
        /// <param name="replacements"></param>
        public void Replace(Dictionary<IStatement, IStatement> replacements)
        {
            // this code works even if values are not distinct from keys.
            List<IStatement> toAdd = new List<IStatement>();
            foreach (var init in initializers)
            {
                IStatement newStatement;
                if (replacements.TryGetValue(init, out newStatement))
                    toAdd.Add(newStatement);
            }
            foreach (var oldStatement in replacements.Keys)
                initializers.Remove(oldStatement);
            initializers.AddRange(toAdd);
        }

        public InitializerSet(ICollection<IStatement> initializers)
        {
            this.initializers = initializers;
        }

        public override string ToString()
        {
            return "InitializerSet(" + StringUtil.VerboseToString(initializers) + ")";
        }
    }

    /// <summary>
    /// Attached to an 'if' statement to indicate that it is the 'if(iteration==0)' block.
    /// </summary>
    internal class FirstIterationPostProcessingBlock : ICompilerAttribute
    {
    }
}