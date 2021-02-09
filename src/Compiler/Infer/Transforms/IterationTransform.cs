// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;
using System.Linq;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Find statements with cyclic dependencies and place them in while loops.  Statements not in loops, and the loops themselves, are
    /// topologically sorted.  DependencyInformation.IsUniform is updated based on this ordering.  Statements within while loops have arbitrary order.
    /// When messages are initialized, while loops may be given "first iteration postprocessing blocks".  Statements within these blocks are topologically sorted.
    /// </summary>
    internal class IterationTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "IterationTransform"; }
        }

        private ModelCompiler compiler;
        LoopCountAnalysisTransform analysis;
        Dictionary<IStatement, IEnumerable<IStatement>> clonesOfStatement = new Dictionary<IStatement, IEnumerable<IStatement>>(new IdentityComparer<IStatement>());
        private LoopMergingInfo loopMergingInfo;

        public IterationTransform(ModelCompiler compiler)
        {
            this.compiler = compiler;
        }

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            analysis = new LoopCountAnalysisTransform();
            analysis.Context.InputAttributes = context.InputAttributes;
            analysis.Transform(itd);
            context.Results = analysis.Context.Results;
            if (!context.Results.IsSuccess)
            {
                Error("analysis failed");
                return itd;
            }
            var td = base.Transform(itd);
            return td;
        }

        protected override IMethodDeclaration DoConvertMethod(IMethodDeclaration md, IMethodDeclaration imd)
        {
            if (!context.InputAttributes.Has<OperatorMethod>(imd))
                return imd;
            loopMergingInfo = context.InputAttributes.Get<LoopMergingInfo>(imd);
            return base.DoConvertMethod(md, imd);
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            List<IStatement> isc = Schedule((IReadOnlyList<IStatement>)inputs);
            RegisterUnchangedStatements(isc);
            outputs.AddRange(isc);
        }

        protected List<IStatement> Schedule(IReadOnlyList<IStatement> isc)
        {
            bool dependsOnIteration = false;
            foreach (IStatement ist in isc)
            {
                if (context.InputAttributes.Has<DependsOnIteration>(ist))
                {
                    dependsOnIteration = true;
                    break;
                }
            }
            if (dependsOnIteration)
            {
                // all statements that depend on iteration should be part of the same convergence loop.
                // to ensure this, we create a dummy statement that links to all of them.
                List<IStatement> stmts2 = new List<IStatement>();
                stmts2.AddRange(isc);
                IStatement iterationStmt = Builder.CommentStmt("iteration");
                context.InputAttributes.Set(iterationStmt, new IterationStatement());
                stmts2.Add(iterationStmt);
                isc = stmts2;
            }
            DependencyGraph g = new DependencyGraph(context, isc, ignoreMissingNodes: false, ignoreRequirements: true);
            foreach (var edge in g.dependencyGraph.Edges)
            {
                // Ignore Cancel dependencies of increment statements.
                // Otherwise, spurious cycles will be created where a message update is falsely believed to produce a new value for the increment.
                // We don't want increment statements to cause cycles to be created.
                // Needed for JaggedSubarrayTest, SparseFactorizedBayesPointEvidence, ThreeStateImportanceModel.
                if (CancelsIntoIncrement(g, isc, edge))
                    g.isDeleted[edge] = true;
            }
            SetLoopPriorities(g);
            return Schedule(g, isc, true);
        }

        private void SetLoopPriorities(DependencyGraph g)
        {
            var ssinfo = SchedulingTransform.GetSerialSchedulingInfo(g);
            var infos = ssinfo.loopInfos;
            var loopVarCount = analysis.loopVarCount;
            // sort infos by decreasing number of nodes
            infos.Sort((info1, info2) => Comparer<int>.Default.Compare(loopVarCount[info2.loopVar], loopVarCount[info1.loopVar]));
            // attach loop ordering information for LoopMergingTransform
            List<IVariableDeclaration> loopVars = new List<IVariableDeclaration>(loopVarCount.Keys);
            if (loopVars.Count > 0)
            {
                int maxCount = loopVarCount.Values.Max() + 1;
                // ensure Sequential loops have highest priority
                foreach (var info in infos) loopVarCount[info.loopVar] += maxCount;
                // sort loopVars by decreasing number of nodes
                loopVars.Sort((loopVar1, loopVar2) => Comparer<int>.Default.Compare(loopVarCount[loopVar2], loopVarCount[loopVar1]));
                for (int i = 0; i < loopVars.Count; i++)
                {
                    IVariableDeclaration loopVar = loopVars[i];
                    context.OutputAttributes.Set(loopVar, new LoopPriority()
                    {
                        Priority = 1 + loopVars.Count - i
                    });
                }
            }
        }

        private bool CancelsIntoIncrement(DependencyGraph g, IReadOnlyList<IStatement> stmts, EdgeIndex edge)
        {
            if (g.isCancels[edge])
            {
                NodeIndex target = g.dependencyGraph.TargetOf(edge);
                bool isIncrement = context.InputAttributes.Has<IncrementStatement>(stmts[target]);
                return isIncrement;
            }
            return false;
        }

        protected class StatementBlock
        {
            public List<NodeIndex> indices;
        }

        protected class Loop : StatementBlock
        {
        }

        protected class StraightLine : StatementBlock
        {
        }

        private List<IStatement> Schedule(DependencyGraph g, IReadOnlyList<IStatement> stmts, bool createFirstIterPostBlocks)
        {
            List<IStatement> output = new List<IStatement>();
            List<StatementBlock> blocks = new List<StatementBlock>();
            List<NodeIndex> currentBlock = null;
            DirectedGraphFilter<NodeIndex, EdgeIndex> graph2 = new DirectedGraphFilter<NodeIndex, EdgeIndex>(g.dependencyGraph, edge => !g.isDeleted[edge]);
            StrongComponents2<NodeIndex> scc = new StrongComponents2<NodeIndex>(graph2.SourcesOf, graph2);
            scc.AddNode += delegate (NodeIndex node) { currentBlock.Add(node); };
            scc.BeginComponent += delegate () { currentBlock = new List<int>(); };
            scc.EndComponent += delegate () {
                bool isCyclic = false;
                if (currentBlock.Count == 1)
                {
                    NodeIndex node = currentBlock[0];
                    foreach (NodeIndex source in graph2.SourcesOf(node))
                    {
                        if (source == node)
                        {
                            isCyclic = true;
                            break;
                        }
                    }
                }
                else
                    isCyclic = true;
                if (isCyclic)
                    blocks.Add(new Loop() { indices = currentBlock });
                else
                    blocks.Add(new StraightLine() { indices = currentBlock });
            };
            scc.SearchFrom(graph2.Nodes);
            //scc.SearchFrom(g.outputNodes);

            bool check = false;
            if (check)
            {
                // check that there are no edges from a later component to an earlier component
                Set<NodeIndex> earlierNodes = new Set<int>();
                foreach (StatementBlock block in blocks)
                {
                    earlierNodes.AddRange(block.indices);
                    foreach (NodeIndex node in block.indices)
                    {
                        foreach (NodeIndex source in graph2.SourcesOf(node))
                        {
                            if (!earlierNodes.Contains(source))
                            {
                                Console.WriteLine(g.NodeToString(node) + Environment.NewLine + "   depends on later node " + g.NodeToString(source));
                                Error("Internal error: Strong components are not ordered properly");
                            }
                        }
                    }
                }
            }

            Set<NodeIndex> nodesToMove = new Set<NodeIndex>();
            Dictionary<Loop, IBlockStatement> firstIterPostprocessing = null;
            if(createFirstIterPostBlocks)
                firstIterPostprocessing = GetFirstIterPostprocessing(blocks, graph2, stmts, nodesToMove);
            IVariableDeclaration iteration = Builder.VarDecl("iteration", typeof(int));

            IndexedProperty<NodeIndex, bool> isUniform = graph2.CreateNodeData<bool>(true);
            foreach (StatementBlock block in blocks)
            {
                if (block is Loop)
                {
                    foreach (NodeIndex i in block.indices)
                    {
                        isUniform[i] = false;
                    }
                    IWhileStatement ws = Builder.WhileStmt(Builder.LiteralExpr(true));
                    IList<IStatement> whileBody = ws.Body.Statements;

                    if (ContainsIterationStatement(stmts, block.indices))
                    {
                        List<IStatement> nodes = new List<IStatement>();
                        foreach (NodeIndex i in block.indices)
                        {
                            IStatement ist = stmts[i];
                            if (!context.InputAttributes.Has<IterationStatement>(ist))
                                nodes.Add(ist);
                        }
                        // build a new dependency graph with the dummy iteration statement removed
                        DependencyGraph g2 = new DependencyGraph(context, nodes, ignoreMissingNodes: true, ignoreRequirements: true);
                        List<IStatement> sc3 = Schedule(g2, nodes, false);
                        if (sc3.Count == 1 && sc3[0] is IWhileStatement)
                            ws = (IWhileStatement)sc3[0];
                        else
                        {
                            // The statements in the outer loop are not strongly connected.  
                            // Since we want the next transform to only process strong components, 
                            // we mark the outer while loop as DoNotSchedule, leaving only the
                            // inner while loops to be scheduled.
                            // add all statements in sc3 to whileBody, but remove while loops around a single statement.
                            foreach (IStatement ist in sc3)
                            {
                                if (ist is IWhileStatement)
                                {
                                    IWhileStatement iws2 = (IWhileStatement)ist;
                                    if (iws2.Body.Statements.Count == 1)
                                    {
                                        whileBody.AddRange(iws2.Body.Statements);
                                        continue;
                                    }
                                }
                                whileBody.Add(ist);
                            }
                            context.OutputAttributes.Set(ws, new DoNotSchedule());
                        }
                    }
                    else // !ContainsIterationStatement
                    {
                        foreach (NodeIndex i in block.indices)
                        {
                            IStatement st = stmts[i];
                            whileBody.Add(st);
                            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(st);
                            di.AddClones(clonesOfStatement);
                        }
                        RegisterUnchangedStatements(whileBody);
                    }
                    Loop loop = (Loop)block;
                    if (firstIterPostprocessing != null && firstIterPostprocessing.ContainsKey(loop))
                    {
                        var thenBlock = firstIterPostprocessing[loop];
                        var iterIsZero = Builder.BinaryExpr(BinaryOperator.ValueEquality, Builder.VarRefExpr(iteration), Builder.LiteralExpr(0));
                        var firstIterPostStmt = Builder.CondStmt(iterIsZero, thenBlock);
                        context.OutputAttributes.Set(firstIterPostStmt, new FirstIterationPostProcessingBlock());
                        whileBody.Add(firstIterPostStmt);
                    }
                    output.Add(ws);
                }
                else
                {
                    // not cyclic
                    foreach (NodeIndex i in block.indices)
                    {
                        IStatement st = stmts[i];
                        if (!nodesToMove.Contains(i))
                        {
                            output.Add(st);
                            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(st);
                            di.AddClones(clonesOfStatement);
                        }
                        isUniform[i] = g.IsUniform(i, source => !isUniform[source]);
                        if (isUniform[i] != g.isUniform[i])
                        {
                            Assert.IsTrue(isUniform[i]);
                            g.isUniform[i] = isUniform[i];
                            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(st);
                            di.IsUniform = isUniform[i];
                        }
                        // mark sources of output statements (for SchedulingTransform)
                        if (g.outputNodes.Contains(i))
                        {
                            foreach (NodeIndex source in graph2.SourcesOf(i))
                            {
                                IStatement sourceSt = stmts[source];
                                if (!context.InputAttributes.Has<OutputSource>(sourceSt))
                                    context.OutputAttributes.Set(sourceSt, new OutputSource());
                            }
                        }
                    }
                }
            }
            return output;
        }

        /// <summary>
        /// Collect statements that need to re-execute due to initialization.
        /// </summary>
        /// <param name="blocks"></param>
        /// <param name="dependencyGraph"></param>
        /// <param name="stmts"></param>
        /// <param name="nodesToMove">Modified on exit</param>
        private Dictionary<Loop, IBlockStatement> GetFirstIterPostprocessing(List<StatementBlock> blocks, DirectedGraphFilter<NodeIndex, EdgeIndex> dependencyGraph, IReadOnlyList<IStatement> stmts, ICollection<NodeIndex> nodesToMove)
        {
            var hasUserInitializedAncestor = dependencyGraph.CreateNodeData(false);
            DepthFirstSearch<NodeIndex> dfsInitBlock = new DepthFirstSearch<int>(dependencyGraph.SourcesOf, dependencyGraph);
            List<NodeIndex> nodesToRerun = new List<NodeIndex>();
            dfsInitBlock.FinishNode += delegate (NodeIndex node)
            {
                var stmt = stmts[node];
                if (SchedulingTransform.IsUserInitialized(context, stmt))
                {
                    // do nothing
                }
                else if (SchedulingTransform.HasUserInitializedInitializer(context, stmt))
                {
                    hasUserInitializedAncestor[node] = true;
                    nodesToMove.Add(node);
                    nodesToRerun.Add(node);
                }
                else
                {
                    var inherit = dependencyGraph.SourcesOf(node).Any(source => hasUserInitializedAncestor[source]);
                    if (inherit)
                    {
                        hasUserInitializedAncestor[node] = true;
                        nodesToRerun.Add(node);
                    }
                }
            };
            Dictionary<Loop, IBlockStatement> firstIterPostprocessing = new Dictionary<Loop, IBlockStatement>();
            foreach (StatementBlock block in blocks)
            {
                if (block is Loop)
                {
                    Loop loop = (Loop)block;
                    // find the set of initialized nodes that are ancestors of the loop (and not ancestors of an ancestor loop)
                    // find all nodes that are ancestors of the loop and descendants of (and including) the init nodes - cannot contain any loops
                    // because we do not clear dfs, all previous stmts in loops are excluded.
                    dfsInitBlock.SearchFrom(loop.indices);
                    foreach (var i in loop.indices)
                    {
                        nodesToMove.Remove(i);
                        nodesToRerun.Remove(i);
                    }
                    if (nodesToRerun.Count > 0)
                    {
                        var firstIterPost = Builder.BlockStmt();
                        foreach (var node in nodesToRerun)
                        {
                            IStatement stmt = stmts[node];
                            if (nodesToMove.Contains(node))
                            {
                                firstIterPost.Statements.Add(stmt);
                            }
                            else
                            {
                                // clone the statement
                                this.ShallowCopy = true;
                                var convertedStmt = ConvertStatement(stmt);
                                this.ShallowCopy = false;
                                firstIterPost.Statements.Add(convertedStmt);
                                loopMergingInfo.AddEquivalentStatement(convertedStmt, loopMergingInfo.GetIndexOf(stmt));
                                clonesOfStatement.Add(stmt, new List<IStatement>() { convertedStmt });
                            }
                        }
                        firstIterPostprocessing.Add(loop, firstIterPost);
                        nodesToRerun.Clear();
                    }
                }
            }
            return firstIterPostprocessing;
        }

        private bool ContainsIterationStatement(IReadOnlyList<IStatement> stmts, IEnumerable<NodeIndex> block)
        {
            foreach (NodeIndex i in block)
            {
                IStatement ist = stmts[i];
                if (context.InputAttributes.Has<IterationStatement>(ist))
                    return true;
            }
            return false;
        }
    }

    /// <summary>
    /// Attached to a dummy statement used to put all DependsOnIteration statements in the same loop
    /// </summary>
    internal class IterationStatement : ICompilerAttribute
    {
    }

    /// <summary>
    /// Attached to a while loop to tell the scheduler to preserve the order of statements (and preserve the loop itself)
    /// </summary>
    internal class DoNotSchedule : ICompilerAttribute
    {
    }

    /// <summary>
    /// Attached to source statements of output statements
    /// </summary>
    internal class OutputSource : ICompilerAttribute
    {
    }

    internal class LoopCountAnalysisTransform : ShallowCopyTransform
    {
        public Dictionary<IVariableDeclaration, int> loopVarCount = new Dictionary<IVariableDeclaration, int>(new IdentityComparer<IVariableDeclaration>());

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
            int count;
            loopVarCount.TryGetValue(loopVar, out count);
            loopVarCount[loopVar] = count + 1;
            return base.ConvertFor(ifs);
        }
    }
}