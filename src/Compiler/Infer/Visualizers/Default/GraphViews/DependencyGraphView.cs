// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;
using DOTEdgeStyle = Microsoft.ML.Probabilistic.Compiler.Visualizers.GraphViews.EdgeStyle;
using GVEdgeStyle = Microsoft.ML.Probabilistic.Compiler.Visualizers.EdgeStyle;
using System.Linq;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers.GraphViews
{
    internal class DependencyGraphView
    {
        private DotGraphWriter graph;
        private IndexedGraph dg;
        private IndexedProperty<NodeIndex, Node> nodeOf;
        private Func<NodeIndex, string> nodeName;
        private Func<EdgeIndex, string> edgeName;
        private IEnumerable<EdgeStylePredicate> edgeStyles;

        internal DependencyGraphView(IndexedGraph dg, IEnumerable<EdgeStylePredicate> edgeStyles = null, 
                                    Func<NodeIndex, string> nodeName = null,
                                    Func<EdgeIndex, string> edgeName = null)
        {
            this.dg = dg;
            this.edgeStyles = edgeStyles;
            this.nodeName = nodeName;
            this.edgeName = edgeName;
            nodeOf = dg.CreateNodeData<Node>(null);
            OnGraphChanged();

            //make the graph.Legend
            if (edgeStyles != null)
            {
                graph.Legend = new Legend(edgeStyles.Count());
                graph.Legend.BackColor = Color.LightGray;

                foreach(EdgeStylePredicate esp in edgeStyles)
                {
                    LegendItem item = new LegendItem();
                    item.Label = esp.Name;

                    item.LineColor = Color.Black;
                    if ((esp.Style & GVEdgeStyle.Dimmed) > 0)
                    {
                        item.LineColor = Color.LightGray;
                    }
                    else if ((esp.Style & GVEdgeStyle.Back) > 0)
                    {
                        item.LineColor = Color.Red;
                    }
                    else if ((esp.Style & GVEdgeStyle.Blue) > 0)
                    {
                        item.LineColor = Color.Blue;
                    }

                    item.LineWidth = 1;
                    if ((esp.Style & GVEdgeStyle.Bold) > 0)
                    {
                        item.LineWidth = 2;
                    }

                    if ((esp.Style & GVEdgeStyle.Dashed) > 0)
                    {
                        item.Style = EdgeStyle.Dashed;
                    }

                    graph.Legend.Items.Add(item);
                }
            }
        }

        protected void OnGraphChanged()
        {
            graph = ToGraph(dg);
        }

        private int Count = 0;

        internal DotGraphWriter ToGraph(IndexedGraph dg)
        {
            DotGraphWriter g = new DotGraphWriter("Model");
            Count = 1;
            foreach (NodeIndex index in dg.Nodes) AddNode(g, index);
            foreach (NodeIndex index in dg.Nodes) AddEdges(g, index);
            return g;
        }

        protected void AddEdges(DotGraphWriter g, NodeIndex index)
        {
            Node nd = nodeOf[index];
            if (nd == null) return;
            foreach (EdgeIndex edge in dg.EdgesInto(index))
            {
                NodeIndex sourceIndex = dg.SourceOf(edge);
                Node nd2 = nodeOf[sourceIndex];
                if (nd2 == null) continue;
                GVEdgeStyle style = GetEdgeStyle(edge);
                Edge e;
                if ((style & GVEdgeStyle.Back) > 0)
                {
                    e = g.AddEdge(nd.ID, nd2.ID);
                    e.Color = Color.Red;
                    e.ArrowheadAtSource = ArrowheadStyle.Normal;
                    e.ArrowheadAtTarget = ArrowheadStyle.None;
                    if ((style & GVEdgeStyle.Blue) > 0) e.Color = Color.Purple;
                }
                else
                {
                    e = g.AddEdge(nd2.ID, nd.ID);
                    if ((style & GVEdgeStyle.Blue) > 0) e.Color = Color.Blue;
                    if ((style & GVEdgeStyle.Dimmed) > 0)
                        e.Color = Color.LightGray;
                }
                if (edgeName != null) e.Label = edgeName(edge);
                if ((style & GVEdgeStyle.Bold) > 0) e.Width = 2;
                if ((style & GVEdgeStyle.Dashed) > 0) e.Style = DOTEdgeStyle.Dashed;
            }
        }

        protected GVEdgeStyle GetEdgeStyle(EdgeIndex index)
        {
            GVEdgeStyle style = GVEdgeStyle.Black;
            if (edgeStyles != null)
            {
                foreach (var esp in edgeStyles)
                {
                    if (esp.Predicate(index))
                        style |= esp.Style;
                }
            }
            return style;
        }

        protected Node AddNode(DotGraphWriter g, NodeIndex index)
        {
            Node nd = nodeOf[index];
            if (nd != null)
            {
                return nd;
            }
            string s = (nodeName != null) ? nodeName(index) : index.ToString(CultureInfo.InvariantCulture);
            nd = g.AddNode("node" + Count);
            nd.UserData = Count++;
            nodeOf[index] = nd;
            nd.Label = s;
            nd.Shape = ShapeStyle.Box;
            nd.FontSize = 9;
            return nd;
        }

        public void Show(string title = "Infer.NET Dependency Graph")
        {
            Graphviz.ShowGraph(graph, title);
        }
    }
}