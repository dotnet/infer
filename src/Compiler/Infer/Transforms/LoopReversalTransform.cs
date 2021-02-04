// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.Diagnostics;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Reverses loop directions to enable statements to fuse with their dependencies.
    /// This reduces the number of reversed clones that need to be created by LocalAllocationTransform.
    /// </summary>
    internal class LoopReversalTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "LoopReversalTransform";
            }
        }

        internal static bool debug;
        private LoopMergingInfo loopMergingInfo;
        private Set<IVariableDeclaration> loopVarsToReverse = new Set<IVariableDeclaration>();
        private Dictionary<IStatement, IStatement> replacements = new Dictionary<IStatement, IStatement>(new IdentityComparer<IStatement>());
        private Dictionary<IStatement, Set<IVariableDeclaration>> loopVarsToReverseInStatement;
        IBlockStatement debugBlock;

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            var analysis = new LoopReversalAnalysisTransform(debug);
            analysis.Context.InputAttributes = context.InputAttributes;
            analysis.Transform(itd);
            context.Results = analysis.Context.Results;
            if (!context.Results.IsSuccess)
            {
                Error("analysis failed");
                return itd;
            }
            this.loopVarsToReverseInStatement = analysis.loopVarsToReverseInStatement;
            var itdOut = base.Transform(itd);
            if (context.trackTransform && debug)
            {
                IBlockStatement block = Builder.BlockStmt();
                foreach (var logString in analysis.log)
                {
                    block.Statements.Add(Builder.CommentStmt(logString));
                }
                foreach (var entry in analysis.loopVarsToReverseInStatement)
                {
                    var stmt = entry.Key;
                    var info = entry.Value;
                    block.Statements.Add(Builder.CommentStmt(StringUtil.ToString(info) + " " + stmt.ToString()));
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

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            if (context.trackTransform && debug)
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
            IMethodDeclaration imd = context.FindAncestor<IMethodDeclaration>();
            loopMergingInfo = context.InputAttributes.Get<LoopMergingInfo>(imd);
            base.DoConvertMethodBody(outputs, inputs);
            PostProcessDependencies(outputs);
        }

        private void PostProcessDependencies(ICollection<IStatement> outputs)
        {
            if (replacements.Count > 0)
            {
                // contexts are bodies of innermost while statements
                DeadCodeTransform.ForEachStatement(outputs,
                  delegate (IWhileStatement iws)
                  {
                  },
                  delegate (IWhileStatement iws)
                  {
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
                              context.OutputAttributes.Remove<DependencyInformation>(ist);
                              context.OutputAttributes.Set(ist, di2);
                          }
                      }
                  });
            }
        }

        protected override IStatement DoConvertStatement(IStatement ist)
        {
            bool isTopLevel = context.InputAttributes.Has<DependencyInformation>(ist);
            if (isTopLevel)
            {
                Set<IVariableDeclaration> loopVars;
                if (loopVarsToReverseInStatement.TryGetValue(ist, out loopVars))
                    loopVarsToReverse.AddRange(loopVars);
            }
            IStatement st = base.DoConvertStatement(ist);
            if (isTopLevel && loopVarsToReverse.Count > 0)
            {
                replacements.Add(ist, st);
                loopMergingInfo.AddEquivalentStatement(st, loopMergingInfo.GetIndexOf(ist));
                loopVarsToReverse.Clear();
            }
            return st;
        }

        protected override IStatement ConvertWhile(IWhileStatement iws)
        {
            IStatement st = base.ConvertWhile(iws);
            InitializerSet initSet = context.InputAttributes.Get<InitializerSet>(iws);
            if (initSet != null)
            {
                // initializers occur in the loop, so all replacements should already be known.
                initSet.Replace(replacements);
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
    }

    /// <summary>
    /// Determines the optimal loop directions to minimize the number of reversed clones that need to be created by LocalAllocationTransform.
    /// </summary>
    internal class LoopReversalAnalysisTransform : ShallowCopyTransform
    {
        private bool debug;

        public LoopReversalAnalysisTransform(bool debug)
        {
            this.debug = debug;
        }

        protected enum Direction { None, Forward, Backward };

        protected class DesiredDirections
        {
            public Dictionary<IVariableDeclaration, Direction> desiredDirectionOfLoopVar = new Dictionary<IVariableDeclaration, Direction>();

            public override string ToString()
            {
                return StringUtil.DictionaryToString<IVariableDeclaration, Direction>(desiredDirectionOfLoopVar, " ");
            }
        }

        public Dictionary<IStatement, Set<IVariableDeclaration>> loopVarsToReverseInStatement = new Dictionary<IStatement, Set<IVariableDeclaration>>(new IdentityComparer<IStatement>());

        public List<string> log = new List<string>();

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            List<int> whileNumberOfNode = new List<int>();
            List<int> fusedCountOfNode = new List<int>();
            List<List<IStatement>> containersOfNode = new List<List<IStatement>>();
            // the code may have multiple while(true) loops, however these must be disjoint.
            // therefore we treat 'while' as one container, but give each loop a different 'while number'.
            int outerWhileCount = 0;
            int currentOuterWhileNumber = 0;
            int currentFusedCount = 0;
            List<Set<IVariableDeclaration>> loopVarsOfWhileNumber = new List<Set<IVariableDeclaration>>();
            // build the dependency graph
            var g = new DependencyGraph2(context, inputs, DependencyGraph2.BackEdgeHandling.Ignore,
                                         delegate (IWhileStatement iws)
                                         {
                                             if (iws is IFusedBlockStatement)
                                             {
                                                 if (iws.Condition is IVariableReferenceExpression)
                                                    currentFusedCount++;
                                             }
                                             else
                                             {
                                                 outerWhileCount++;
                                                 currentOuterWhileNumber = outerWhileCount;
                                             }
                                         },
                                         delegate (IWhileStatement iws)
                                         {
                                             if (iws is IFusedBlockStatement)
                                             {
                                                 if(iws.Condition is IVariableReferenceExpression)
                                                    currentFusedCount--;
                                             }
                                             else
                                             {
                                                 currentOuterWhileNumber = 0;
                                             }
                                         },
                                         delegate (IConditionStatement ics)
                                         {
                                         },
                                         delegate (IConditionStatement ics)
                                         {
                                         },
                                         delegate (IStatement ist, int targetIndex)
                                         {
                                             int whileNumber = currentOuterWhileNumber;
                                             whileNumberOfNode.Add(whileNumber);
                                             fusedCountOfNode.Add(currentFusedCount);
                                             List<IStatement> containers = new List<IStatement>();
                                             LoopMergingTransform.UnwrapStatement(ist, containers);
                                             containersOfNode.Add(containers);
                                             for (int i = 0; i < currentFusedCount; i++)
                                             {
                                                 if (containers[i] is IForStatement ifs)
                                                 {
                                                     var loopVar = Recognizer.LoopVariable(ifs);
                                                     if (loopVarsOfWhileNumber.Count <= whileNumber)
                                                     {
                                                         while (loopVarsOfWhileNumber.Count <= whileNumber)
                                                             loopVarsOfWhileNumber.Add(new Set<IVariableDeclaration>());
                                                     }
                                                     Set<IVariableDeclaration> loopVars = loopVarsOfWhileNumber[whileNumber];
                                                     loopVars.Add(loopVar);
                                                 }
                                             }
                                         });
            var nodes = g.nodes;
            var dependencyGraph = g.dependencyGraph;

            for (int whileNumber = 1; whileNumber < loopVarsOfWhileNumber.Count; whileNumber++)
            {
                foreach (var loopVar in loopVarsOfWhileNumber[whileNumber])
                {
                    // Any statement (in the while loop) that has a forward descendant and a backward descendant will be cloned, so we want to minimize the number of such nodes.
                    // The free variables in this problem are the loop directions at the leaf statements, since all other loop directions are forced by these.
                    // We find the optimal labeling of the free variables by solving a min cut problem on a special network.
                    // The network is constructed so that the cost of a cut is equal to the number of statements that will be cloned.
                    // The network has 2 nodes for every statement: an in-node and an out-node.  
                    // For a non-leaf statement, there is a capacity 1 edge from the in-node to out-node.  This edge is cut when the statement is cloned.
                    // For a leaf statement, there is an infinite capacity edge in both directions, or equivalently a single node.
                    // If statement A depends on statement B, then there is an infinite capacity edge from in-A to in-B, and from out-B to out-A, 
                    // representing the fact that cloning A requires cloning B, but not the reverse.
                    // If a statement must appear with a forward loop, it is connected to the source.
                    // If a statement must appear with a backward loop, it is connected to the sink.

                    // construct a capacitated graph
                    int inNodeStart = 0;
                    int outNodeStart = inNodeStart + dependencyGraph.Nodes.Count;
                    int sourceNode = outNodeStart + dependencyGraph.Nodes.Count;
                    int sinkNode = sourceNode + 1;
                    int cutNodeCount = sinkNode + 1;
                    Func<NodeIndex, int> getInNode = node => node + inNodeStart;
                    Func<NodeIndex, int> getOutNode = node => node + outNodeStart;
                    IndexedGraph network = new IndexedGraph(cutNodeCount);
                    const float infinity = 1000000f;
                    List<float> capacity = new List<float>();
                    List<NodeIndex> nodesOfInterest = new List<NodeIndex>();
                    foreach (var node in dependencyGraph.Nodes)
                    {
                        if (whileNumberOfNode[node] != whileNumber)
                            continue;
                        NodeIndex source = node;
                        List<IStatement> containersOfSource = containersOfNode[source];
                        bool hasLoopVar = containersOfSource.Any(container => container is IForStatement && Recognizer.LoopVariable((IForStatement)container) == loopVar);
                        if (!hasLoopVar) continue;
                        nodesOfInterest.Add(node);
                        IStatement sourceSt = nodes[source];
                        var readAfterWriteEdges = dependencyGraph.EdgesOutOf(source).Where(edge => !g.isWriteAfterRead[edge]);
                        bool isLeaf = true;
                        int inNode = getInNode(node);
                        int outNode = getOutNode(node);
                        foreach (var target in readAfterWriteEdges.Select(dependencyGraph.TargetOf))
                        {
                            List<IStatement> containersOfTarget = containersOfNode[target];
                            IStatement targetSt = nodes[target];
                            ForEachMatchingLoopVariable(containersOfSource, containersOfTarget, (loopVar2, afs, bfs) =>
                            {
                                if (loopVar2 == loopVar)
                                {
                                    int inTarget = getInNode(target);
                                    int outTarget = getOutNode(target);
                                    network.AddEdge(inTarget, inNode);
                                    capacity.Add(infinity);
                                    network.AddEdge(outNode, outTarget);
                                    capacity.Add(infinity);
                                    isLeaf = false;
                                }
                            });
                        }
                        if (isLeaf)
                        {
                            if(debug)
                                log.Add($"loopVar={loopVar.Name} leaf {sourceSt}");
                            network.AddEdge(inNode, outNode);
                            capacity.Add(infinity);
                            network.AddEdge(outNode, inNode);
                            capacity.Add(infinity);
                        }
                        else
                        {
                            network.AddEdge(inNode, outNode);
                            capacity.Add(1f);
                        }
                        int fusedCount = fusedCountOfNode[node];
                        Direction desiredDirectionOfSource = GetDesiredDirection(loopVar, containersOfSource, fusedCount);
                        if (desiredDirectionOfSource == Direction.Forward)
                        {
                            if(debug)
                                log.Add($"loopVar={loopVar.Name} forward {sourceSt}");
                            network.AddEdge(sourceNode, inNode);
                            capacity.Add(infinity);
                        }
                        else if (desiredDirectionOfSource == Direction.Backward)
                        {
                            if(debug)
                                log.Add($"loopVar={loopVar.Name} backward {sourceSt}");
                            network.AddEdge(outNode, sinkNode);
                            capacity.Add(infinity);
                        }
                    }
                    network.IsReadOnly = true;

                    // compute the min cut
                    MinCut<NodeIndex, EdgeIndex> mc = new MinCut<EdgeIndex, EdgeIndex>(network, e => capacity[e]);
                    mc.Sources.Add(sourceNode);
                    mc.Sinks.Add(sinkNode);
                    Set<NodeIndex> sourceGroup = mc.GetSourceGroup();
                    foreach (NodeIndex node in nodesOfInterest)
                    {
                        IStatement sourceSt = nodes[node];
                        bool forwardIn = sourceGroup.Contains(getInNode(node));
                        bool forwardOut = sourceGroup.Contains(getOutNode(node));
                        if (forwardIn != forwardOut)
                        {
                            if (debug)
                                log.Add($"loopVar={loopVar.Name} will clone {sourceSt}");
                        }
                        else if (forwardIn)
                        {
                            if(debug)
                                log.Add($"loopVar={loopVar.Name} wants forward {sourceSt}");
                        }
                        else
                        {
                            if(debug)
                                log.Add($"loopVar={loopVar.Name} wants backward {sourceSt}");
                            var containers = containersOfNode[node];
                            bool isForwardLoop = true;
                            foreach (var container in containers)
                            {
                                if (container is IForStatement)
                                {
                                    IForStatement ifs = (IForStatement)container;
                                    if (Recognizer.LoopVariable(ifs) == loopVar)
                                    {
                                        isForwardLoop = Recognizer.IsForwardLoop(ifs);
                                    }
                                }
                            }
                            if (isForwardLoop)
                            {
                                Set<IVariableDeclaration> loopVarsToReverse;
                                if (!loopVarsToReverseInStatement.TryGetValue(sourceSt, out loopVarsToReverse))
                                {
                                    // TODO: re-use equivalent sets
                                    loopVarsToReverse = new Set<IVariableDeclaration>();
                                    loopVarsToReverseInStatement.Add(sourceSt, loopVarsToReverse);
                                }
                                loopVarsToReverse.Add(loopVar);
                            }
                        }
                    }
                }
            }

            base.DoConvertMethodBody(outputs, inputs);
        }

        private Direction GetDesiredDirection(IVariableDeclaration loopVar, List<IStatement> containers, int fusedCount)
        {
            for (int i = 0; i < fusedCount; i++)
            {
                IForStatement ifs = (IForStatement)containers[i];
                if (ifs != null && loopVar == Recognizer.LoopVariable(ifs))
                {
                    return Recognizer.IsForwardLoop(ifs) ? Direction.Forward : Direction.Backward;
                }
            }
            return Direction.None;
        }

        private static void ForEachMatchingLoopVariable(
            List<IStatement> containersOfSource,
            List<IStatement> containersOfTarget,
            Action<IVariableDeclaration, IForStatement, IForStatement> action)
        {
            for (int i = 0; i < System.Math.Min(containersOfTarget.Count, containersOfSource.Count); i++)
            {
                if (containersOfTarget[i] is IForStatement)
                {
                    if (!(containersOfSource[i] is IForStatement))
                        break;
                    IForStatement afs = (IForStatement)containersOfTarget[i];
                    IForStatement bfs = (IForStatement)containersOfSource[i];
                    IVariableDeclaration loopVar = Recognizer.LoopVariable(afs);
                    if (Recognizer.LoopVariable(bfs) != loopVar)
                        break;
                    // both loops use the same variable.
                    action(loopVar, afs, bfs);
                }
                else if (!Containers.ContainersAreEqual(containersOfTarget[i], containersOfSource[i]))
                    break;
            }
        }
    }
}
