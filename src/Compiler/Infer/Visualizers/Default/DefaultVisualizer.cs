// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Xml;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler.Visualizers.GraphViews;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    public class DefaultVisualizer : Visualizer
    {
        internal override IFactorManager FactorManager { get; } = new HtmlFactorManager();

        internal override IDependencyGraphVisualizer DependencyGraphVisualizer { get; } = new DotDependencyGraphVisualizer();

        internal override ITransformerChainVisualizer TransformerChainVisualizer { get; } = new HtmlTransformerChainVisualizer();

        internal override ITaskGraphVisualizer TaskGraphVisualizer { get; } = new DotTaskGraphVisualizer();

        internal override IFactorGraphVisualizer FactorGraphVisualizer { get; } = new DotFactorGraphVisualizer();

        internal override IGraphWriter GraphWriter { get; } = new MultipleGraphWriter(new DotGraphWriter(), new DgmlGraphWriter());

        internal class HtmlTransformerChainVisualizer : ITransformerChainVisualizer
        {
            public void VisualizeTransformerChain(TransformerChain target, string folder, string modelName)
            {
                HtmlTransformChainView tcv = new HtmlTransformChainView();

                foreach (var transformer in target.transformers)
                {
                    if (!transformer.OutputEqualsInput)
                    {
                        tcv.AddTransformer(transformer);
                        if (tcv.SelectedTransformer == null || transformer.Transform.Context.Results.ErrorCount > 0)
                        {
                            tcv.SelectedTransformer = transformer;
                        }
                    }
                }
                // Avoid spaces in folder names since they can confuse some apps
                tcv.Visualize(Path.Combine(folder, modelName + "_Transforms"));
            }
        }

        internal class HtmlFactorManager : IFactorManager
        {
            public void ShowFactorManager(bool showMissingEvidences, params IAlgorithm[] algorithms)
            {
                DefaultFactorManager fmv = new DefaultFactorManager(algorithms);
                fmv.ShowMissingEvidences = showMissingEvidences;
                fmv.Show();
            }
        }

        internal class DotDependencyGraphVisualizer : IDependencyGraphVisualizer
        {
            public void VisualizeDependencyGraph(IndexedGraph dg, IEnumerable<EdgeStylePredicate> edgeStyles = null, Func<int, string> nodeName = null, Func<int, string> edgeName = null, string visualizationTitle = "Dependency_Graph")
            {
                DependencyGraphView view = new DependencyGraphView(dg, edgeStyles, nodeName, edgeName);
                view.Show(visualizationTitle);
            }
        }

        internal class DotTaskGraphVisualizer : ITaskGraphVisualizer
        {
            public void VisualizeTaskGraph(ITypeDeclaration itd, BasicTransformContext context)
            {
                TaskGraphView view = new TaskGraphView(itd, context);
                view.Show("Task_Graph");
            }
        }

        internal class DotFactorGraphVisualizer : IFactorGraphVisualizer
        {
            public void VisualizeFactorGraph(ModelBuilder model)
            {
                ModelView view = new ModelView(model);
                view.ShowGraph(model.modelType.Name);
            }
        }

        internal class MultipleGraphWriter : IGraphWriter
        {
            readonly IGraphWriter[] GraphWriters;

            public MultipleGraphWriter(params IGraphWriter[] graphWriters)
            {
                this.GraphWriters = graphWriters;
            }

            public void WriteGraph(ModelBuilder model, string path)
            {
                foreach(var graphWriter in GraphWriters)
                {
                    graphWriter.WriteGraph(model, path);
                }
            }
        }

        internal class DotGraphWriter : IGraphWriter
        {
            public void WriteGraph(ModelBuilder model, string path)
            {
                ModelView view = new ModelView(model);
                view.WriteDot(path + ".gv");
            }
        }

        internal class DgmlGraphWriter : IGraphWriter
        {
            public void WriteGraph(ModelBuilder model, string path)
            {
                var g = new DgmlWriter("Model");
                GraphBuilder.Add(g, model, true);
                g.Write(path + ".dgml");
            }
        }

        internal class DgmlWriter : GraphWriter
        {
            public DgmlWriter(string name) : base(name) { }

            public override void Write(string path)
            {
                // References:
                // http://en.wikipedia.org/wiki/DGML
                // http://msdn.microsoft.com/library/ee842619.aspx
                // http://ceyhunciper.wordpress.com/category/dgml/
                // http://www.lovettsoftware.com/blogengine.net/post/2010/07/16/DGML-with-Style.aspx
                var settings = new XmlWriterSettings();
                settings.Indent = true;
                ////string colorToString(Color c) => c.ToString().Trim('"');
                using (var writer = XmlWriter.Create(path, settings))
                {
                    writer.WriteStartElement("DirectedGraph", "http://schemas.microsoft.com/vs/2009/dgml");
                    writer.WriteStartElement("Nodes");
                    foreach (string key in nodes.Keys)
                    {
                        GraphViews.Node node = (GraphViews.Node)nodes[key];
                        writer.WriteStartElement("Node");
                        writer.WriteAttributeString("Id", key);
                        writer.WriteAttributeString("Label", node.Label);
                        writer.WriteAttributeString("FontSize", node.FontSize.ToString());
                        writer.WriteAttributeString("Foreground", node.FontColor.ToString());
                        writer.WriteAttributeString("Background", node.FillColor.ToString());
                        if (node.Shape == GraphViews.ShapeStyle.None)
                            writer.WriteAttributeString("Shape", "None");
                        if (node.UserData is GraphViews.GraphBuilder.Group)
                            writer.WriteAttributeString("Group", "Expanded");
                        else
                        {
                            // The Shape attribute in DGML can only be used to remove the outline.
                            // To actually change the shape, use the NodeRadius attribute.
                            if (node.Shape == GraphViews.ShapeStyle.Box)
                                writer.WriteAttributeString("NodeRadius", "0");
                            else if (node.Shape == GraphViews.ShapeStyle.Ellipse)
                                writer.WriteAttributeString("NodeRadius", "100");
                        }
                        // The Outline attribute seems to be ignored.
                        ////writer.WriteAttributeString("Outline", colorToString(node.Attr.Color));
                        writer.WriteEndElement();
                    }
                    writer.WriteEndElement();
                    writer.WriteStartElement("Links");
                    foreach (GraphViews.Edge edge in edges)
                    {
                        writer.WriteStartElement("Link");
                        writer.WriteAttributeString("Source", edge.Source.ID);
                        writer.WriteAttributeString("Target", edge.Target.ID);
                        writer.WriteAttributeString("Label", edge.Label);
                        writer.WriteAttributeString("FontSize", edge.FontSize.ToString());
                        // Sadly there is no way to change the arrow head style in DGML.
                        if (edge.UserData is GraphViews.GraphBuilder.Group)
                            writer.WriteAttributeString("Category", "Contains");
                        writer.WriteEndElement();
                    }
                    writer.WriteEndElement();
                    writer.WriteEndElement();
                }
            }
        }
    }
}
