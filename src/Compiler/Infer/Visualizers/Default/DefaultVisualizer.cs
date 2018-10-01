// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
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

        internal override IGraphWriter GraphWriter { get; } = new DotGraphWriter();

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

                tcv.Visualize(Path.Combine(folder, modelName + " Transforms"));
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

        internal class DotGraphWriter : IGraphWriter
        {
            public void WriteGraph(ModelBuilder model, string path)
            {
                ModelView view = new ModelView(model);
                view.WriteDot(path + ".gv");
            }
        }
    }
}
