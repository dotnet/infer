// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.CompilerServices;
using Microsoft.Msagl.GraphViewerGdi;
using Microsoft.Msagl.Drawing;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Node = Microsoft.Msagl.Drawing.Node;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    /// <summary>
    /// A view of a task graph, which is a schedule when the tasks are message computations.
    /// </summary>
    internal class TaskGraphView : IDisposable
    {
        private readonly GViewer gviewer = new GViewer();

        protected enum Stage
        {
            Initialisation,
            Update
        };

        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static readonly CodeBuilder Builder = CodeBuilder.Instance;

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
                    if (ist is IWhileStatement iws)
                    {
                        looptasks.AddRange(iws.Body.Statements);
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
            gviewer.Graph = ToGraph(pretasks, looptasks);
        }

        public static bool ShowNonIterativeTaskNodes = false;

        private int Count = 0;

        internal Graph ToGraph(IList<IStatement> pretasks, IList<IStatement> looptasks)
        {
            Graph g = new Graph("Model");
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

        protected void AddEdges(Graph g, IStatement ist, Stage stage)
        {
            Node nd = GetNodeForStatement(ist, stage);
            if (nd == null) return;
            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
            if (di == null) return;
            AddDependencyEdges(g, nd, di, Stage.Initialisation, stage);
            if (stage != Stage.Initialisation) AddDependencyEdges(g, nd, di, Stage.Update, stage);
        }

        protected void AddDependencyEdges(Graph g, Node nd, DependencyInformation di, Stage stage, Stage parentStage)
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
                    e = g.AddEdge(nd.Attr.Id, nd2.Attr.Id);
                    e.Attr.Color = Color.Red;
                    e.Attr.ArrowheadAtSource = ArrowStyle.Normal;
                    e.Attr.ArrowheadAtTarget = ArrowStyle.None;
                }
                else
                {
                    e = g.AddEdge(nd2.Attr.Id, nd.Attr.Id);
                    if (parentStage != stage) e.Attr.Color = Color.LightGray;
                }
                if (di.HasDependency(DependencyType.Trigger, source)) e.Attr.LineWidth = 2;
                if (di.HasDependency(DependencyType.Fresh, source)) e.Attr.AddStyle(Style.Dashed);
                if (di.HasDependency(DependencyType.Cancels, source))
                    e.Attr.Color = Color.LightGray;
            }
            //e.LabelText = StatementLabel(ist);            e.Label.FontSize = 8;
        }

        //CodeTransformer ct = new CodeTransformer();
        protected Node AddNode(Graph g, IStatement ist, Stage stage)
        {
            if (ist is ICommentStatement) return null;
            Node nd = GetNodeForStatement(ist, stage);
            if (nd != null)
            {
                nd.LabelText = Count + "," + nd.LabelText;
                Count++;
                return nd;
            }
            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
            if ((di != null) && (di.IsOutput)) return null;
            string s = Count + ". " + StatementLabel(ist);
            nd = g.AddNode("node" + Count);
            nd.UserData = Count++;
            SetNodeForStatement(ist, stage, nd);
            //if (di.IsOutput) nd.Attr.Fillcolor = Color.LightBlue;
            nd.LabelText = s;
            if (stage == Stage.Initialisation) nd.Attr.FillColor = Color.LightGray;
            if (ist is IExpressionStatement ies)
            {
                if ((ies.Expression is IAssignExpression iae) && (iae.Target is IVariableDeclarationExpression)) nd.Attr.LineWidth = 2;
            }
            nd.Attr.Shape = Shape.Box;
            nd.Label.FontSize = 9;
            return nd;
        }

        private readonly Dictionary<Stage, Dictionary<IStatement, Node>> stmtNodeMap = new Dictionary<Stage, Dictionary<IStatement, Node>>();

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

        protected string StatementLabel(IStatement ist)
        {
            if (ist is IExpressionStatement ies)
            {
                string s;
                if (ies.Expression is IAssignExpression iae)
                {
                    s = ExpressionToString(iae.Target);
                }
                else s = ExpressionToString(ies.Expression);
                if (s.StartsWith("this.")) s = s.Substring(5);
                //if (s.EndsWith("[0]")) s = s.Substring(0, s.Length - 3);
                return s;
            }
            if (ist is IForStatement ifs)
            {
                return StatementLabel(ifs.Body.Statements[0]);
            }
            if (ist is IConditionStatement ics)
            {
                return String.Format("if ({0}) {1}", ics.Condition.ToString(),
                                     StatementLabel(ics.Then));
            }
            if (ist is IBlockStatement ibs)
            {
                int blockSize = ibs.Statements.Count;
                string s;
                if (blockSize > 0)
                {
                    s = StatementLabel(ibs.Statements[0]);
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

        // This must not be inlined to avoid dependence on MSAGL when the method is never called.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void RunInForm()
        {
            WindowsVisualizer.FormHelper.RunInForm(gviewer, "Infer.NET Schedule Viewer", false);
        }

        public void Dispose()
        {
            gviewer.Dispose();
        }
    }
}