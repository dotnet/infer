// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Remove statements whose result is never used by an output.  Also remove updates whose result is uniform.
    /// </summary>
    /// <remarks>
    /// The transform assumes that statements have been annotated with DependencyInformation attributes.  
    /// The transform works by constructing a dependency graph and performing a depth-first search backward from the output statements.  Any statement not reached by this search is pruned.
    /// To infer which statements are uniform, the transform puts all statements onto a priority queue, and repeatedly removes statements whose result is NOT uniform.  
    /// A statement on the queue is determined to be non-uniform if it meets the following conditions:
    /// <list type="bullet">
    /// <item>It is not annotated as uniform by its DependencyInformation</item>
    /// <item>It has no Requirement dependencies waiting on the queue.</item>
    /// </list>
    /// After convergence, any statements left on the queue are assumed to be uniform, and pruned.  
    /// </remarks>
    internal class PruningTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "PruningTransform"; }
        }

        public static bool PruneUniformStmts = true;
        public static bool debug;

        protected override IMethodDeclaration DoConvertMethod(IMethodDeclaration md, IMethodDeclaration imd)
        {
            if (!context.InputAttributes.Has<OperatorMethod>(imd)) return imd;
            return base.DoConvertMethod(md, imd);
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            IList<IStatement> isc = Schedule((IReadOnlyList<IStatement>)inputs);
            RegisterUnchangedStatements(isc);
            outputs.AddRange(isc);
        }

        protected IList<IStatement> Schedule(IReadOnlyList<IStatement> isc)
        {
            DependencyGraph g = new DependencyGraph(context, isc, ignoreMissingNodes: false, ignoreRequirements: true, readAfterWriteOnly: true);
            // The uniform statements must be removed first, before doing the search backward from outputs.  
            // This is important because it allows dependencies of uniform statements to be pruned.  
            // For example, suppose A requires B and C, B is uniform.  This means that A is uniform and A will be pruned.  
            // As a result, C doesn't need to be computed (even if it might be non-uniform).
            if (PruneUniformStmts)
                g.PropagateUniformNodes();

            // Propagate DependsOnIteration
            DependsOnIteration attr = null;
            DepthFirstSearch<NodeIndex> dfsIter = new DepthFirstSearch<EdgeIndex>(g.dependencyGraph.TargetsOf, g.nodeData);
            dfsIter.DiscoverNode += delegate(NodeIndex node) {
                if(!context.OutputAttributes.Has<DependsOnIteration>(isc[node]))
                    context.OutputAttributes.Set(isc[node], attr);
            };
            foreach (NodeIndex node in g.dependencyGraph.Nodes)
            {
                attr = context.GetAttribute<DependsOnIteration>(isc[node]);
                if (attr != null) dfsIter.SearchFrom(node);
            }

            // search backward from the outputs to find statements that are relevant.
            DepthFirstSearch<NodeIndex> dfs = new DepthFirstSearch<NodeIndex>(
                PruneUniformStmts ? (Converter<int, IEnumerable<int>>) g.SourcesNeededForOutput : g.dependencyGraph.SourcesOf,
                g.nodeData);
            dfs.SearchFrom(g.outputNodes);
            List<NodeIndex> schedule = new List<NodeIndex>();
            foreach (NodeIndex node in g.dependencyGraph.Nodes)
            {
                // any statement found in the search is relevant.  other statements are pruned.
                if (dfs.IsVisited[node] == VisitState.Finished) schedule.Add(node);
            }

            if (debug)
            {
                var itdOut = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
                IBlockStatement block = Builder.BlockStmt();
                foreach (var line in StringUtil.Lines(g.dependencyGraph.ToString()))
                {
                    block.Statements.Add(Builder.CommentStmt(line));
                }
                foreach (NodeIndex node in g.dependencyGraph.Nodes)
                {
                    block.Statements.Add(Builder.CommentStmt($"{node} {isc[node]}"));
                }
                context.OutputAttributes.Add(itdOut, new DebugInfo()
                {
                    Transform = this,
                    Name = "Graph",
                    Value = block
                });
            }

            if (PruneUniformStmts)
            {
                // When statements are pruned from the code, they must also be pruned from the DependencyInformation attributes of all other statements that might refer to them.
                Dictionary<IStatement, IStatement> replacements = new Dictionary<IStatement, IStatement>(new IdentityComparer<IStatement>());
                foreach (NodeIndex targetIndex in g.dependencyGraph.Nodes)
                {
                    if (g.isUniform[targetIndex])
                    {
                        IStatement ist = isc[targetIndex];
                        DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
                        // this is needed for detection of non-uniform initializers
                        di.IsUniform = true;
                    }
                    if (g.isUniform[targetIndex] && !g.isEssential[targetIndex])
                    {
                        NodeIndex source = g.FindInitializer(targetIndex);
                        if (source != targetIndex)
                        {
                            IStatement target = isc[targetIndex];
                            IStatement replacement = isc[source];
                            replacements[target] = replacement;
                        }
                    }
                }
                foreach (NodeIndex targetIndex in schedule)
                {
                    IStatement target = isc[targetIndex];
                    DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(target);
                    di.Replace(replacements);
                }
            }

            IList<IStatement> sc = Builder.StmtCollection();
            foreach (NodeIndex i in schedule)
            {
                IStatement st = isc[i];
                sc.Add(st);
            }
            return sc;
        }
    }
}