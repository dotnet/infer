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
using System.Linq;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Creates clones of specific variables, to break up cycles of user-initialized statements.  
    /// On exit, there should be no cycles of user-initialized statements.
    /// </summary>
    internal class InitializerTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "InitializerTransform";
            }
        }

        List<IStatement> containers = new List<IStatement>();
        List<IStatement> cloneDecls = new List<IStatement>();
        List<IStatement> cloneUpdates = new List<IStatement>();
        private LoopMergingInfo loopMergingInfo;
        internal static bool debug = false;
        private ModelCompiler compiler;

        internal InitializerTransform(ModelCompiler compiler)
        {
            this.compiler = compiler;
        }

        protected override IMethodDeclaration DoConvertMethod(IMethodDeclaration md, IMethodDeclaration imd)
        {
            if (!context.InputAttributes.Has<OperatorMethod>(imd) || !compiler.UseSpecialFirstIteration)
                return imd;
            loopMergingInfo = context.InputAttributes.Get<LoopMergingInfo>(imd);
            if (debug && loopMergingInfo != null)
            {
                var itdOut = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
                context.OutputAttributes.Add(itdOut, loopMergingInfo.GetDebugInfo(this));
            }
            return base.DoConvertMethod(md, imd);
        }

        void ProcessStatements(IList<IStatement> outputs, IList<IStatement> outputDecls, IList<IStatement> inputs, Dictionary<IStatement, IStatement> replacements)
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                IStatement ist = inputs[i];
                if (ist is IWhileStatement)
                {
                    IWhileStatement iws = (IWhileStatement)ist;
                    IWhileStatement ws = Builder.WhileStmt(iws);
                    bool doNotSchedule = context.InputAttributes.Has<DoNotSchedule>(iws);
                    if (doNotSchedule)
                    {
                        // iws may contain nested while loops
                        // TODO: make sure the new decls go in the right place
                        ProcessStatements(ws.Body.Statements, outputDecls, iws.Body.Statements, replacements);
                    }
                    else
                    {
                        IReadOnlyList<IStatement> inputStmts = (IReadOnlyList<IStatement>)iws.Body.Statements;
                        IStatement firstIterPostBlock = ForwardBackwardTransform.ExtractFirstIterationPostProcessingBlock(context, ref inputStmts);
                        DependencyGraph g = new DependencyGraph(context, inputStmts, ignoreMissingNodes: true, ignoreRequirements: true);
                        // look for cycles of initialized nodes and insert clones as needed
                        Set<NodeIndex> nodesToClone = GetNodesToClone(g, g.dependencyGraph.Nodes);
                        for (int node = 0; node < inputStmts.Count; node++)
                        {
                            IStatement st = inputStmts[node];
                            if (nodesToClone.Contains(node))
                            {
                                cloneDecls.Clear();
                                cloneUpdates.Clear();
                                containers.Clear();
                                IStatement newStmt = ConvertStatement(st);
                                IStatement declStmt = cloneDecls[0];
                                IStatement setToStmt = cloneUpdates[0];
                                outputDecls.AddRange(cloneDecls);
                                DependencyInformation diNew = (DependencyInformation)context.InputAttributes.Get<DependencyInformation>(newStmt).Clone();
                                context.OutputAttributes.Remove<DependencyInformation>(newStmt);
                                context.OutputAttributes.Set(newStmt, diNew);
                                DependencyInformation diSet = new DependencyInformation();
                                diSet.Add(DependencyType.Dependency | DependencyType.Requirement, newStmt);
                                context.OutputAttributes.Set(setToStmt, diSet);
                                DependencyInformation diDecl = new DependencyInformation();
                                context.OutputAttributes.Set(declStmt, diDecl);
                                foreach (IStatement writer in diNew.Overwrites)
                                {
                                    diDecl.Add(DependencyType.Dependency, writer);
                                    diSet.Add(DependencyType.Overwrite, writer);
                                }
                                diNew.Remove(DependencyType.Overwrite);
                                diNew.Add(DependencyType.Declaration | DependencyType.Dependency | DependencyType.Overwrite, declStmt);
                                if (loopMergingInfo != null)
                                {
                                    // update loopMergingInfo with the new statement
                                    int oldNode = loopMergingInfo.GetIndexOf(st);
                                    int newNode = loopMergingInfo.AddNode(newStmt);
                                    loopMergingInfo.InheritSourceConflicts(newNode, oldNode);
                                    int setToNode = loopMergingInfo.AddNode(setToStmt);
                                    loopMergingInfo.InheritTargetConflicts(setToNode, oldNode);
                                    int declNode = loopMergingInfo.AddNode(declStmt);
                                }
                                replacements[st] = setToStmt;
                                context.InputAttributes.CopyObjectAttributesTo<InitialiseBackward>(st, context.OutputAttributes, setToStmt);
                                st = newStmt;
                                ws.Body.Statements.AddRange(cloneUpdates);
                            }
                            else
                            {
                                RegisterUnchangedStatement(st);
                            }
                            ws.Body.Statements.Add(st);
                        }
                        if (firstIterPostBlock != null)
                            ws.Body.Statements.Add(firstIterPostBlock);
                    }
                    context.InputAttributes.CopyObjectAttributesTo(iws, context.OutputAttributes, ws);
                    ist = ws;
                }
                else
                {
                    RegisterUnchangedStatement(ist);
                }
                outputs.Add(ist);
            }
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            Dictionary<IStatement, IStatement> replacements = new Dictionary<IStatement, IStatement>(new IdentityComparer<IStatement>());
            ProcessStatements(outputs, outputs, inputs, replacements);
            // update all dependencies
            foreach (IStatement ist in outputs)
            {
                if (ist is IWhileStatement)
                {
                    IWhileStatement iws = (IWhileStatement)ist;
                    foreach (IStatement st in iws.Body.Statements)
                    {
                        DependencyInformation di2 = context.OutputAttributes.Get<DependencyInformation>(st);
                        if(di2 != null)
                            di2.Replace(replacements);
                    }
                }
                else
                {
                    DependencyInformation di = context.OutputAttributes.Get<DependencyInformation>(ist);
                    if (di != null)
                        di.Replace(replacements);
                }
            }
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            // for(i) array[i] = rhs
            // becomes:
            // temp = CopyStorage(array)
            // for(i) temp[i] = rhs
            // for(i) array[i] = SetTo(temp[i])
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
            string name = VariableInformation.GenerateName(context, ivd.Name + "_new");
            IVariableDeclaration clone = Builder.VarDecl(name, ivd.VariableType);
            var cloneDeclExpr = Builder.VarDeclExpr(clone);
            var newTarget = Builder.ReplaceExpression(iae.Target, Builder.VarRefExpr(ivd), Builder.VarRefExpr(clone));
            IExpression copyStorage = Builder.StaticGenericMethod(
                new Func<PlaceHolder, PlaceHolder>(ArrayHelper.CopyStorage),
                new[] { ivd.VariableType }, Builder.VarRefExpr(ivd));
            var cloneDeclStmt = Builder.AssignStmt(cloneDeclExpr, copyStorage);
            context.OutputAttributes.Set(cloneDeclStmt, new Initializer());
            cloneDecls.Add(cloneDeclStmt);
            IExpression setTo = Builder.StaticGenericMethod(
                new Func<PlaceHolder, PlaceHolder, PlaceHolder>(ArrayHelper.SetTo),
                new[] { iae.GetExpressionType() }, iae.Target, newTarget);
            IStatement setToStmt = Builder.AssignStmt(iae.Target, setTo);
            setToStmt = Containers.WrapWithContainers(setToStmt, containers);
            cloneUpdates.Add(setToStmt);
            return Builder.AssignExpr(newTarget, iae.Expression);
        }

        /// <summary>
        /// Convert only the body, leaving the initializer, condition and increment statements unchanged.
        /// </summary>
        protected override IStatement ConvertFor(IForStatement ifs)
        {
            if (ifs is IBrokenForStatement)
                throw new NotImplementedException("broken loop");
            containers.Add(ifs);
            IForStatement fs = Builder.ForStmt();
            context.SetPrimaryOutput(fs);
            fs.Condition = ifs.Condition;
            fs.Increment = ifs.Increment;
            fs.Initializer = ifs.Initializer;
            fs.Body = ConvertBlock(ifs.Body);
            context.InputAttributes.CopyObjectAttributesTo(ifs, context.OutputAttributes, fs);
            return fs;
        }

        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            containers.Add(ics);
            return base.ConvertCondition(ics);
        }

        private Set<NodeIndex> GetNodesToClone(DependencyGraph g, IEnumerable<NodeIndex> nodes)
        {
            Set<NodeIndex> nodesToClone = new Set<EdgeIndex>();
            Set<NodeIndex> nodeSet = new Set<NodeIndex>();
            nodeSet.AddRange(nodes);
            DepthFirstSearch<NodeIndex> dfs = new DepthFirstSearch<EdgeIndex>(node => Successors(g, nodeSet, nodesToClone, node), g.dependencyGraph);
            dfs.BackEdge += delegate(Edge<NodeIndex> edge)
            {
                nodesToClone.Add(System.Math.Min(edge.Source, edge.Target));
            };
            dfs.SearchFrom(nodes);
            return nodesToClone;
        }

        private IEnumerable<NodeIndex> Successors(DependencyGraph g, ICollection<NodeIndex> nodes, ICollection<NodeIndex> nodesToClone, NodeIndex node)
        {
            if (nodesToClone.Contains(node))
                yield break;
            if (g.initializedNodes.Contains(node))
            {
                foreach (NodeIndex target in g.dependencyGraph.TargetsOf(node))
                {
                    if (nodes.Contains(target))
                        yield return target;
                }
            }
            else
            {
                // node is not initialized, but could have initialized outgoing edges
                foreach (EdgeIndex edge in g.dependencyGraph.EdgesOutOf(node))
                {
                    if (g.initializedEdges.Contains(edge))
                    {
                        NodeIndex target = g.dependencyGraph.TargetOf(edge);
                        if (nodes.Contains(target))
                            yield return target;
                    }
                }
            }
        }
    }
}
