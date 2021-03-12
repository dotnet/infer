// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers.GraphViews
{
    /// <summary>
    /// A view of a task graph, which is a schedule when the tasks are message computations.
    /// </summary>
    internal class TaskGraphView
    {
        private GraphWriter graph;

        protected enum Stage
        {
            Initialisation,
            Update
        };

        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static CodeBuilder Builder = CodeBuilder.Instance;

        internal IList<IStatement> pretasks;
        internal IList<IStatement> looptasks;
        internal BasicTransformContext context;

        internal TaskGraphView(ITypeDeclaration itd, BasicTransformContext context)
        {
            pretasks = Builder.StmtCollection();
            looptasks = Builder.StmtCollection();
            foreach (IMethodDeclaration imd in itd.Methods)
            {
                if (!context.InputAttributes.Has<OperatorMethod>(imd)) continue;
                foreach (IStatement ist in imd.Body.Statements)
                {
                    if (ist is IWhileStatement)
                    {
                        looptasks.AddRange(((IWhileStatement) ist).Body.Statements);
                        continue;
                    }
                    if (context.InputAttributes.Has<OperatorStatement>(ist)) pretasks.Add(ist);
                }
                //                if (imd.Name == "Initialise") pretasks.AddRange(((IBlockStatement)imd.Body).Statements);
                //if (imd.Name == "Update") looptasks.AddRange(((IBlockStatement)imd.Body).Statements);
            }
            this.context = context;
            OnTasksChanged();
        }

        protected void OnTasksChanged()
        {
            graph = ToGraph(pretasks, looptasks);
        }

        public static bool ShowNonIterativeTaskNodes = false;

        private int Count = 0;

        internal GraphWriter ToGraph(IList<IStatement> pretasks, IList<IStatement> looptasks)
        {
            GraphWriter g = new DotGraphWriter("Model");
            Count = 1;
            if (ShowNonIterativeTaskNodes)
            {
                foreach (IStatement ist in pretasks) AddNode(g, ist, Stage.Initialisation);
            }
            foreach (IStatement ist in looptasks) AddNode(g, ist, Stage.Update);
            foreach (IStatement ist in pretasks) AddEdges(g, ist, Stage.Initialisation);
            Set<IStatement> edgesDone = new Set<IStatement>(ReferenceEqualityComparer<IStatement>.Instance);
            foreach (IStatement ist in looptasks)
            {
                if (edgesDone.Contains(ist)) continue;
                edgesDone.Add(ist);
                AddEdges(g, ist, Stage.Update);
            }
            return g;
        }

        protected void AddEdges(GraphWriter g, IStatement ist, Stage stage)
        {
            Node nd = GetNodeForStatement(ist, stage);
            if (nd == null) return;
            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
            if (di == null) return;
            AddDependencyEdges(g, nd, di, Stage.Initialisation, stage);
            if (stage != Stage.Initialisation) AddDependencyEdges(g, nd, di, Stage.Update, stage);
        }

        protected void AddDependencyEdges(GraphWriter g, Node nd, DependencyInformation di, Stage stage, Stage parentStage)
        {
            foreach (IStatement source in di.GetDependenciesOfType(DependencyType.Dependency | DependencyType.Declaration))
            {
                Node nd2 = GetNodeForStatement(source, stage);
                if (nd2 == null) continue;
                bool backwards = ((int) nd.UserData) < ((int) nd2.UserData);
                //Console.WriteLine("stage=" + stage + " " + nd.UserData + " " + nd2.UserData);
                Edge e;
                if (backwards)
                {
                    e = g.AddEdge(nd.ID, nd2.ID);
                    e.Color = Color.Red;
                    e.Reverse = true;
                }
                else
                {
                    e = g.AddEdge(nd2.ID, nd.ID);
                    if (parentStage != stage) e.Color = Color.LightGray;
                }
                if (di.HasDependency(DependencyType.Trigger, source)) e.Width = 2;
                if (di.HasDependency(DependencyType.Fresh, source)) e.Style = EdgeStyle.Dashed;
                if (di.HasDependency(DependencyType.Cancels, source))
                    e.Color = Color.LightGray;
            }
            //e.Label = StatementLabel(ist);            e.Label.FontSize = 8;
        }

        //CodeTransformer ct = new CodeTransformer();
        protected Node AddNode(GraphWriter g, IStatement ist, Stage stage)
        {
            if (ist is ICommentStatement) return null;
            Node nd = GetNodeForStatement(ist, stage);
            if (nd != null)
            {
                nd.Label = Count + "," + nd.Label;
                Count++;
                return nd;
            }
            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
            if ((di != null) && (di.IsOutput)) return null;
            string s = Count + ". " + StatementLabel(ist);
            nd = g.AddNode("node" + Count);
            nd.UserData = Count++;
            SetNodeForStatement(ist, stage, nd);
            //if (di.IsOutput) nd.Fillcolor = Color.LightBlue;
            nd.Label = s;
            if (stage == Stage.Initialisation) nd.FillColor = Color.LightGray;
            if (ist is IExpressionStatement)
            {
                IExpressionStatement ies = (IExpressionStatement) ist;
                IAssignExpression iae = ies.Expression as IAssignExpression;
                if ((iae != null) && (iae.Target is IVariableDeclarationExpression)) nd.BorderWidth = 2;
            }
            nd.Shape = ShapeStyle.Box;
            nd.FontSize = 9;
            return nd;
        }

        private Dictionary<Stage, Dictionary<IStatement, Node>> stmtNodeMap = new Dictionary<Stage, Dictionary<IStatement, Node>>();

        protected Node GetNodeForStatement(IStatement ist, Stage stg)
        {
            if (!stmtNodeMap.ContainsKey(stg)) return null; // stmtNodeMap[stg] = new Dictionary<IStatement, Node>(new IdentityComparer<IStatement>());
            if (!stmtNodeMap[stg].ContainsKey(ist)) return null;
            return stmtNodeMap[stg][ist];
        }

        protected void SetNodeForStatement(IStatement ist, Stage stg, Node nd)
        {
            if (!stmtNodeMap.ContainsKey(stg)) stmtNodeMap[stg] = new Dictionary<IStatement, Node>(ReferenceEqualityComparer<IStatement>.Instance);
            stmtNodeMap[stg][ist] = nd;
        }


        protected bool ReferenceContains(IList<IStatement> list, object value)
        {
            foreach (object obj in list) if (object.ReferenceEquals(obj, value)) return true;
            return false;
        }

        protected string StatementLabel(IStatement ist)
        {
            if (ist is IExpressionStatement)
            {
                IExpressionStatement ies = (IExpressionStatement) ist;
                string s;
                if (ies.Expression is IAssignExpression)
                {
                    s = ExpressionToString(((IAssignExpression) ies.Expression).Target);
                }
                else s = ExpressionToString(ies.Expression);
                if (s.StartsWith("this.")) s = s.Substring(5);
                //if (s.EndsWith("[0]")) s = s.Substring(0, s.Length - 3);
                return s;
            }
            if (ist is IForStatement)
            {
                return StatementLabel(((IForStatement) ist).Body.Statements[0]);
            }
            if (ist is IConditionStatement)
            {
                return String.Format("if ({0}) {1}", ((IConditionStatement) ist).Condition.ToString(),
                                     StatementLabel(((IConditionStatement) ist).Then));
            }
            if (ist is IBlockStatement)
            {
                int blockSize = ((IBlockStatement) ist).Statements.Count;
                string s;
                if (blockSize > 0)
                {
                    s = StatementLabel(((IBlockStatement) ist).Statements[0]);
                }
                else
                {
                    s = "EmptyBlock";
                }
                if (blockSize > 1)
                {
                    s += " ...";
                }
                return s;
            }
            return ist.GetType().Name;
        }

        private string ExpressionToString(IExpression expr)
        {
            return CodeCompiler.ExpressionToString(expr);
        }

        public void Show(string title = "Infer.NET Schedule")
        {
            Graphviz.ShowGraph(graph, title);
        }
    }
}