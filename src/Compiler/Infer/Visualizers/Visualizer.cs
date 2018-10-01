// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    internal interface ITransformerChainVisualizer
    {
        void VisualizeTransformerChain(TransformerChain target, string folder, string modelName);
    }

    internal interface IDependencyGraphVisualizer
    {
        void VisualizeDependencyGraph(IndexedGraph dg, IEnumerable<EdgeStylePredicate> edgeStyles = null,
                                    Func<NodeIndex, string> nodeName = null,
                                    Func<EdgeIndex, string> edgeName = null,
                                    string visualizationTitle = "Infer.NET Dependency Graph Viewer");
    }

    internal interface ITaskGraphVisualizer
    {
        void VisualizeTaskGraph(ITypeDeclaration itd, BasicTransformContext context);
    }

    internal interface IFactorGraphVisualizer
    {
        void VisualizeFactorGraph(ModelBuilder model);
    }

    internal interface IGraphWriter
    {
        void WriteGraph(ModelBuilder model, string path);
    }

    internal interface IFactorManager
    {
        void ShowFactorManager(bool showMissingEvidences, params IAlgorithm[] algorithms);
    }

    public abstract class Visualizer
    {
        internal abstract IFactorManager FactorManager { get; }
        internal abstract IDependencyGraphVisualizer DependencyGraphVisualizer { get; }
        internal abstract ITransformerChainVisualizer TransformerChainVisualizer { get; }
        internal abstract ITaskGraphVisualizer TaskGraphVisualizer { get; }
        internal abstract IFactorGraphVisualizer FactorGraphVisualizer { get; }
        internal abstract IGraphWriter GraphWriter { get; }
    }

    [Flags]
    public enum EdgeStyle
    {
        Black = 0,
        Back = 1,
        Dashed = 2,
        Bold = 4,
        Blue = 8,
        Dimmed = 16
    }

    public class EdgeStylePredicate
    {
        public string Name;
        public Predicate<EdgeIndex> Predicate;
        public EdgeStyle Style;

        public EdgeStylePredicate(string name, Predicate<EdgeIndex> predicate, EdgeStyle style)
        {
            this.Name = name;
            this.Predicate = predicate;
            this.Style = style;
        }
    }
}
