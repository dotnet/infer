// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.Msagl.GraphViewerGdi;
using Microsoft.Msagl.Drawing;
using Microsoft.ML.Probabilistic.Compiler;
using System.Windows.Forms;
using System.IO;
using Microsoft.ML.Probabilistic.Collections;
using System.Xml;
using Microsoft.ML.Probabilistic.Utilities;
using System.Linq;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    internal class ModelView
    {
        private GViewer gviewer = new GViewer();

        internal ModelBuilder modelBuilder;

        internal ModelView()
        {
        }

        internal ModelView(ModelBuilder model)
        {
            Model = model;
        }

        /// <summary>
        /// The model being shown in the view
        /// </summary>
        internal ModelBuilder Model
        {
            get { return modelBuilder; }
            set
            {
                modelBuilder = value;
                OnModelChange();
            }
        }

        protected void OnModelChange()
        {
            var builder = new GraphBuilder();
            gviewer.Graph = builder.ToGraph(modelBuilder);
        }

        // This must not be inlined to avoid dependence on MSAGL when the method is never called.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ShowInForm(string title, bool maximise)
        {
            Form f = WindowsVisualizer.FormHelper.ShowInForm(gviewer, title, maximise);
            Application.Run(f);
        }

        /// <summary>
        /// Write the graph to a file in DGML format
        /// </summary>
        /// <param name="path"></param>
        public void WriteDgml(string path)
        {
            // References:
            // http://en.wikipedia.org/wiki/DGML
            // http://msdn.microsoft.com/library/ee842619.aspx
            // http://ceyhunciper.wordpress.com/category/dgml/
            // http://www.lovettsoftware.com/blogengine.net/post/2010/07/16/DGML-with-Style.aspx
            var builder = new GraphBuilder();
            builder.UseContainers = true;
            Graph g = builder.ToGraph(modelBuilder);
            var settings = new XmlWriterSettings();
            settings.Indent = true;
            Func<Color,string> colorToString = c => c.ToString().Trim('"');
            using (var writer = XmlWriter.Create(path, settings))
            {
                writer.WriteStartElement("DirectedGraph", "http://schemas.microsoft.com/vs/2009/dgml");
                writer.WriteStartElement("Nodes");
                foreach (string key in g.NodeMap.Keys)
                {
                    Node node = (Node)g.NodeMap[key];
                    writer.WriteStartElement("Node");
                    writer.WriteAttributeString("Id", key);
                    writer.WriteAttributeString("Label", node.LabelText);
                    writer.WriteAttributeString("FontSize", node.Label.FontSize.ToString());
                    writer.WriteAttributeString("Foreground", colorToString(node.Label.FontColor));
                    writer.WriteAttributeString("Background", colorToString(node.Attr.FillColor));
                    if (node.Attr.Color == Color.White)
                        writer.WriteAttributeString("Shape", "None");
                    if (node.UserData is GraphBuilder.Group)
                        writer.WriteAttributeString("Group", "Expanded");
                    else {
                        // The Shape attribute in DGML can only be used to remove the outline.
                        // To actually change the shape, use the NodeRadius attribute.
                        if (node.Attr.Shape == Shape.Box)
                            writer.WriteAttributeString("NodeRadius", "0");
                        else if (node.Attr.Shape == Shape.Ellipse)
                            writer.WriteAttributeString("NodeRadius", "100");
                    }
                    // The Outline attribute seems to be ignored.
                    //writer.WriteAttributeString("Outline", colorToString(node.Attr.Color));
                    writer.WriteEndElement();
                }
                writer.WriteEndElement();
                writer.WriteStartElement("Links");
                foreach (Edge edge in g.Edges)
                {
                    writer.WriteStartElement("Link");
                    writer.WriteAttributeString("Source", edge.Source);
                    writer.WriteAttributeString("Target", edge.Target);
                    writer.WriteAttributeString("Label", edge.LabelText);
                    writer.WriteAttributeString("FontSize", edge.Label?.FontSize.ToString() ?? Microsoft.Msagl.Drawing.Label.DefaultFontSize.ToString());
                    // Sadly there is no way to change the arrow head style in DGML.
                    if (edge.UserData is GraphBuilder.Group)
                        writer.WriteAttributeString("Category", "Contains");
                    writer.WriteEndElement();
                }
                writer.WriteEndElement();
                writer.WriteEndElement();
            }
        }
    }

    internal class GraphBuilder
    {
        public bool UseContainers;
        private Group GroupInstance = new Group();
        private int Count = 0;
        private Set<Node> childNodes = new Set<Node>();
        private Dictionary<IModelExpression, Node> nodeOfExpr = new Dictionary<IModelExpression, Node>(new IdentityComparer<IModelExpression>());
        // condition variables that have already been linked to an expression
        private Dictionary<IModelExpression, List<Variable>> conditionVariables = new Dictionary<IModelExpression, List<Variable>>();
        private Dictionary<ConditionContext, Node> nodeOfContext = new Dictionary<ConditionContext, Node>();

        public class Group
        {
        }

        protected class ConditionContext
        {
            List<ConditionBlock> blocks = new List<ConditionBlock>();

            public static ConditionContext GetContext(IEnumerable<IStatementBlock> containers)
            {
                ConditionContext context = new ConditionContext();
                foreach (var block in containers)
                {
                    if (block is ConditionBlock)
                        context.blocks.Add((ConditionBlock)block);
                }
                if (context.blocks.Count == 0)
                    return null;
                else
                    return context;
            }

            public override bool Equals(object obj)
            {
                ConditionContext that = obj as ConditionContext;
                if(ReferenceEquals(that, null)) return false;
                return EnumerableExtensions.AreEqual(this.blocks, that.blocks);
            }

            public override int GetHashCode()
            {
                return Hash.GetHashCodeAsSequence(blocks);
            }

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder();
                foreach (ConditionBlock block in blocks)
                {
                    sb.Append(block.ToString());
                }
                return sb.ToString();
            }

            public string GetLabel()
            {
                return blocks[blocks.Count-1].GetConditionExpression().ToString();
            }

            public ConditionContext GetParentContext()
            {
                if (blocks.Count == 1)
                    return null;
                ConditionContext parent = new ConditionContext();
                parent.blocks.AddRange(blocks.Take(blocks.Count - 1));
                return parent;
            }

            public IModelExpression GetConditionVariable()
            {
                return blocks[blocks.Count - 1].ConditionVariableUntyped;
            }
        }

        internal Graph ToGraph(ModelBuilder mb)
        {
            Graph g = new Graph("Model");
            Count = 0;
            if (mb == null)
                return g;
            foreach (IModelExpression me in mb.ModelExpressions)
            {
                if (me is MethodInvoke)
                    AddFactorEdges(g, (MethodInvoke)me);
            }
            // connect nodes that represent the same variable with undirected edges
            Dictionary<Variable, List<Node>> nodesOfVariable = new Dictionary<Variable, List<Node>>();
            foreach (KeyValuePair<IModelExpression, Node> entry in nodeOfExpr)
            {
                if (!(entry.Key is Variable))
                    continue;
                Variable v = (Variable)entry.Key;
                v = GetBaseVariable(v);
                List<Node> nodes;
                if (!nodesOfVariable.TryGetValue(v, out nodes))
                {
                    nodes = new List<Node>();
                    nodesOfVariable[v] = nodes;
                }
                nodes.Add(entry.Value);
            }
            foreach (List<Node> nodes in nodesOfVariable.Values)
            {
                for (int i = 0; i < nodes.Count; i++)
                {
                    for (int j = 0; j < nodes.Count; j++)
                    {
                        if (i == j)
                            continue;
                        // Glee uses the edge direction as a layout hint, so only use child nodes as sources
                        if (!childNodes.Contains(nodes[i]))
                            continue;
                        Edge edge = g.AddEdge(nodes[i].Attr.Id, nodes[j].Attr.Id);
                        edge.Attr.ArrowheadAtTarget = ArrowStyle.None;
                    }
                }
            }
            return g;
        }

        protected Variable GetBaseVariable(Variable v)
        {
            while (v.ArrayVariable != null)
                v = (Variable)v.ArrayVariable;
            return v;
        }

        protected Node GetNode(Graph g, IModelExpression expr)
        {
            if (nodeOfExpr.ContainsKey(expr))
                return nodeOfExpr[expr];
            Node nd = g.AddNode("node" + (Count++));
            nodeOfExpr[expr] = nd;
            nd.LabelText = expr.ToString();
            nd.Label.FontSize = 9;
            if (expr is Variable)
            {
                Variable ve = (Variable)expr;
                if (ve.IsObserved)
                {
                    nd.Attr.Shape = Shape.Box;
                    nd.Attr.Color = Color.White;
                    if (ve.IsBase)
                    {
                        // if the observed value is a ValueType, display it directly rather than the variable name
                        object value = ((HasObservedValue)ve).ObservedValue;
                        if (ReferenceEquals(value, null))
                            nd.LabelText = "null";
                        else if (value.GetType().IsValueType)
                            nd.LabelText = value.ToString();
                    }
                }
                if (!ve.IsReadOnly)
                {
                    nd.Label.FontSize = 10;
                    nd.Label.FontColor = Color.Blue;
                }
                if (UseContainers && ve.Containers.Count > 0)
                {
                    var context = ConditionContext.GetContext(ve.Containers);
                    if (context != null)
                    {
                        var contextNode = GetNode(g, context);
                        AddGroupEdge(g, contextNode, nd);
                    }
                }
            }
            else if (expr is MethodInvoke)
            {
                MethodInvoke mi = (MethodInvoke)expr;
                nd.Attr.FillColor = Color.Black;
                nd.Label.FontColor = Color.White;
                nd.Attr.Shape = Shape.Box;
                nd.Label.FontSize = 8;
                string methodName = mi.method.Name;
                if (mi.op != null)
                    methodName = mi.op.ToString();
                nd.LabelText = methodName;
                if (UseContainers && mi.Containers.Count > 0)
                {
                    var context = ConditionContext.GetContext(mi.Containers);
                    if (context != null)
                    {
                        var contextNode = GetNode(g, context);
                        AddGroupEdge(g, contextNode, nd);
                    }
                }
            }
            return nd;
        }

        protected Node GetNode(Graph g, ConditionContext context)
        {
            Node node;
            if (!nodeOfContext.TryGetValue(context, out node))
            {
                node = g.AddNode("node" + (Count++));
                nodeOfContext[context] = node;
                node.LabelText = context.GetLabel();
                node.UserData = this.GroupInstance;
                ConditionContext parent = context.GetParentContext();
                if (parent != null)
                {
                    var parentNode = GetNode(g, parent);
                    AddGroupEdge(g, parentNode, node);
                }
                var variable = context.GetConditionVariable();
                Node conditionNode = GetNode(g, variable);
                g.AddEdge(conditionNode.Attr.Id, node.Attr.Id);
            }
            return node;
        }

        protected void AddFactorEdges(Graph g, MethodInvoke mi)
        {
            var parameters = mi.method.GetParameters();
            for (int i = 0; i < mi.args.Count; i++)
            {
                var parameter = parameters[i];
                if (parameter.IsOut)
                    AddEdge(g, mi, mi.args[i], parameter.Name);
                else
                    AddEdge(g, mi.args[i], mi, parameter.Name);
            }
            if (mi.returnValue != null)
            {
                AddEdge(g, mi, mi.returnValue, "");
            }
            if (!UseContainers)
            {
                // add edges from condition variables to target (if there are no such edges already)
                IModelExpression target = (mi.returnValue != null) ? mi.returnValue : mi;
                Set<IStatementBlock> excluded = new Set<IStatementBlock>();
                if (target is Variable)
                {
                    // if target is in the ConditionBlock, then don't connect with the condition variable
                    Variable targetVar = (Variable)target;
                    excluded.AddRange(targetVar.Containers);
                }
                foreach (IStatementBlock block in mi.Containers)
                {
                    if (excluded.Contains(block))
                        continue;
                    if (block is ConditionBlock)
                    {
                        ConditionBlock cb = (ConditionBlock)block;
                        Variable c = cb.ConditionVariableUntyped;
                        List<Variable> condVars;
                        if (!conditionVariables.TryGetValue(target, out condVars))
                        {
                            condVars = new List<Variable>();
                            conditionVariables[target] = condVars;
                        }
                        if (!condVars.Contains(c))
                        {
                            AddEdge(g, c, target, "condition");
                            condVars.Add(c);
                        }
                    }
                }
            }
        }

        protected Edge AddEdge(Graph g, IModelExpression from, IModelExpression to, string name)
        {
            Node sourceNode = GetNode(g, from);
            Node targetNode = GetNode(g, to);
            childNodes.Add(targetNode);
            return AddEdge(g, sourceNode, targetNode, name);
        }

        protected Edge AddEdge(Graph g, Node sourceNode, Node targetNode, string name)
        {
            string source = sourceNode.Attr.Id;
            string target = targetNode.Attr.Id;
            Edge edge = g.AddEdge(source, name, target);
            edge.LabelText = name;
            edge.Label.FontSize = 8;
            edge.Label.FontColor = Color.LightGray;
            return edge;
        }

        protected void AddGroupEdge(Graph g, Node parent, Node child)
        {
            Edge edge = g.AddEdge(parent.Attr.Id, child.Attr.Id);
            edge.UserData = this.GroupInstance;
        }
    }

}