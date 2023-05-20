// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Microsoft.Msagl.GraphViewerGdi;
using Microsoft.Msagl.Drawing;
using Node = Microsoft.Msagl.Drawing.Node;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;
using ToolLabel = System.Windows.Forms.Label;
using System.Windows.Forms;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    internal class DependencyGraphView : IDisposable
    {
        private readonly Panel panel = new Panel();
        private readonly GViewer gviewer = new GViewer();
        private readonly IndexedGraph dg;
        private readonly IndexedProperty<NodeIndex, Node> nodeOf;
        private readonly Func<NodeIndex, string> nodeName;
        private readonly Func<EdgeIndex, string> edgeName;
        private readonly IEnumerable<EdgeStylePredicate> edgeStyles;

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
            panel.Dock = DockStyle.Fill;
            gviewer.Dock = DockStyle.Fill;
            panel.Controls.Add(gviewer);
            if (edgeStyles != null)
            {
                // draw the legend
                TableLayoutPanel legend = new TableLayoutPanel();
                legend.AutoSize = true;
                legend.ColumnCount = 2 * edgeStyles.Count();
                foreach (EdgeStylePredicate esp in edgeStyles)
                {
                    PictureBox pic = new PictureBox();
                    pic.SizeMode = PictureBoxSizeMode.AutoSize;
                    legend.Controls.Add(pic);
                    int size = 8;
                    var bitmap = new System.Drawing.Bitmap(size, size);
                    var graphics = System.Drawing.Graphics.FromImage(bitmap);
                    var color = System.Drawing.Color.Black;
                    if ((esp.Style & EdgeStyle.Dimmed) > 0)
                        color = System.Drawing.Color.LightGray;
                    else if ((esp.Style & EdgeStyle.Back) > 0)
                        color = System.Drawing.Color.Red;
                    else if ((esp.Style & EdgeStyle.Blue) > 0)
                        color = System.Drawing.Color.Blue;
                    int width = 1;
                    if ((esp.Style & EdgeStyle.Bold) > 0)
                        width = 2;
                    var pen = new System.Drawing.Pen(color, width);
                    if ((esp.Style & EdgeStyle.Dashed) > 0)
                        pen.DashStyle = System.Drawing.Drawing2D.DashStyle.Dash;
                    int y = bitmap.Height / 2;
                    graphics.DrawLine(pen, 0, y, bitmap.Width - 1, y);
                    pic.Image = bitmap;
                    ToolLabel label = new ToolLabel();
                    label.Text = esp.Name;
                    label.AutoSize = true;
                    legend.Controls.Add(label);
                }
                legend.Anchor = AnchorStyles.None; // centers the legend in its parent
                TableLayoutPanel legendPanel = new TableLayoutPanel();
                legendPanel.AutoSize = true;
                legendPanel.BackColor = System.Drawing.Color.LightGray;
                legendPanel.Controls.Add(legend);
                legendPanel.Dock = DockStyle.Bottom;
                panel.Controls.Add(legendPanel);
            }
        }

        protected void OnGraphChanged()
        {
            gviewer.Graph = ToGraph(dg);
        }

        private int Count = 0;

        internal Graph ToGraph(IndexedGraph dg)
        {
            Graph g = new Graph("Model");
            Count = 1;
            foreach (NodeIndex index in dg.Nodes) AddNode(g, index);
            foreach (NodeIndex index in dg.Nodes) AddEdges(g, index);
            return g;
        }

        protected void AddEdges(Graph g, NodeIndex index)
        {
            Node nd = nodeOf[index];
            if (nd == null) return;
            foreach (EdgeIndex edge in dg.EdgesInto(index))
            {
                NodeIndex sourceIndex = dg.SourceOf(edge);
                Node nd2 = nodeOf[sourceIndex];
                if (nd2 == null) continue;
                EdgeStyle style = GetEdgeStyle(edge);
                Edge e;
                if ((style & EdgeStyle.Back) > 0)
                {
                    e = g.AddEdge(nd.Attr.Id, nd2.Attr.Id);
                    e.Attr.Color = Color.Red;
                    e.Attr.ArrowheadAtSource = ArrowStyle.Normal;
                    e.Attr.ArrowheadAtTarget = ArrowStyle.None;
                    if ((style & EdgeStyle.Blue) > 0) e.Attr.Color = Color.Purple;
                }
                else
                {
                    e = g.AddEdge(nd2.Attr.Id, nd.Attr.Id);
                    if ((style & EdgeStyle.Blue) > 0) e.Attr.Color = Color.Blue;
                    if ((style & EdgeStyle.Dimmed) > 0)
                        e.Attr.Color = Color.LightGray;
                }
                if (edgeName != null) e.LabelText = edgeName(edge);
                if ((style & EdgeStyle.Bold) > 0) e.Attr.LineWidth = 2;
                if ((style & EdgeStyle.Dashed) > 0) e.Attr.AddStyle(Style.Dashed);
            }
        }

        protected EdgeStyle GetEdgeStyle(EdgeIndex index)
        {
            EdgeStyle style = EdgeStyle.Black;
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

        protected Node AddNode(Graph g, NodeIndex index)
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
            nd.LabelText = s;
            nd.Attr.Shape = Shape.Box;
            nd.Label.FontSize = 9;
            return nd;
        }

        public void RunInForm(string title = "Infer.NET Dependency Graph Viewer")
        {
            WindowsVisualizer.FormHelper.RunInForm(panel, title, false);
        }

        public void Dispose()
        {
            panel.Dispose();
            gviewer.Dispose();
        }
    }
}