// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers.GraphViews
{
    internal class ModelView
    {
        private GraphWriter graph;
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
            graph = new DotGraphWriter("Model");
            GraphBuilder.Add(graph, modelBuilder, false);
        }

        /// <summary>
        /// Write the graph to a file in DOT format
        /// </summary>
        /// <param name="path"></param>
        public void WriteDot(string path)
        {
            // References:
            // https://en.wikipedia.org/wiki/DOT_(graph_description_language)

            GraphWriter g = new DotGraphWriter("Model");
            GraphBuilder.Add(g, modelBuilder, true);
            g.Write(path);
        }

        /// <summary>
        /// Show graph on a screen.
        /// </summary>
        /// <param name="title"></param>
        public void ShowGraph(string title)
        {
            Graphviz.ShowGraph(graph, title);
        }
    }

    internal class GraphBuilder
    {
        public bool UseContainers;
        private Group GroupInstance = new Group();
        private int Count = 0;
        private Set<Node> childNodes = new Set<Node>();
        private Dictionary<IModelExpression, Node> nodeOfExpr = new Dictionary<IModelExpression, Node>(ReferenceEqualityComparer<IModelExpression>.Instance);
        // condition variables that have already been linked to an expression
        private Dictionary<IModelExpression, List<Variable>> conditionVariables = new Dictionary<IModelExpression, List<Variable>>();
        private Dictionary<ConditionContext, Node> nodeOfContext = new Dictionary<ConditionContext, Node>();

        public static void Add(GraphWriter g, ModelBuilder modelBuilder, bool useContainers)
        {
            GraphBuilder graphBuilder = new GraphBuilder();
            graphBuilder.UseContainers = useContainers;
            graphBuilder.Add(g, modelBuilder);
        }

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

        private void Add(GraphWriter g, ModelBuilder mb)
        {
            Count = 0;
            if (mb == null)
                return;
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
                        if (i != j && childNodes.Contains(nodes[i]))
                        {
                            // Graph layout tools use the edge direction as a layout hint, so only use child nodes as sources
                            Edge edge = g.AddEdge(nodes[i].ID, nodes[j].ID);
                            edge.ArrowheadAtTarget = ArrowheadStyle.None;
                        }
                    }
                }
            }
        }

        protected Variable GetBaseVariable(Variable v)
        {
            while (v.ArrayVariable != null)
                v = (Variable)v.ArrayVariable;
            return v;
        }

        private string GetForEachSuffix(Variable ve)
        {
            StringBuilder sb = new StringBuilder();
            foreach (var container in ve.Containers)
            {
                if (container is ForEachBlock forEachBlock)
                {
                    sb.Append($"[{forEachBlock.Index}]");
                }
            }
            return sb.ToString();
        }

        protected Node GetNode(GraphWriter g, IModelExpression expr)
        {
            if (nodeOfExpr.ContainsKey(expr))
                return nodeOfExpr[expr];
            Node nd = g.AddNode("node" + (Count++));
            nodeOfExpr[expr] = nd;
            nd.Label = expr.ToString();
            nd.FontSize = 9;
            if (expr is Variable)
            {
                Variable ve = (Variable)expr;
                if (ve.IsObserved)
                {
                    nd.Shape = ShapeStyle.None;

                    if (ve.IsBase)
                    {
                        // if the observed value is a ValueType, display it directly rather than the variable name
                        object value = ((HasObservedValue)ve).ObservedValue;
                        if (ReferenceEquals(value, null))
                            nd.Label = "null";
                        else if (value.GetType().IsValueType)
                            nd.Label = value.ToString();
                    }
                }
                else if (ve.Containers.Count > 0)
                {
                    nd.Label += GetForEachSuffix(ve);
                }
                if (!ve.IsReadOnly)
                {
                    nd.FontSize = 10;
                    nd.FontColor = Color.Blue;
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
                nd.FillColor = Color.Black;
                nd.FontColor = Color.White;
                nd.Shape = ShapeStyle.Box;
                nd.FontSize = 8;
                string methodName = mi.method.Name;
                if (mi.op != null)
                    methodName = mi.op.ToString();
                nd.Label = methodName;
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

        protected Node GetNode(GraphWriter g, ConditionContext context)
        {
            Node node;
            if (!nodeOfContext.TryGetValue(context, out node))
            {
                node = g.AddNode("node" + (Count++));
                nodeOfContext[context] = node;
                node.Label = context.GetLabel();
                node.UserData = this.GroupInstance;
                ConditionContext parent = context.GetParentContext();
                if (parent != null)
                {
                    var parentNode = GetNode(g, parent);
                    AddGroupEdge(g, parentNode, node);
                }
                var variable = context.GetConditionVariable();
                Node conditionNode = GetNode(g, variable);
                g.AddEdge(conditionNode.ID, node.ID);
            }
            return node;
        }

        protected void AddFactorEdges(GraphWriter g, MethodInvoke mi)
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

        protected Edge AddEdge(GraphWriter g, IModelExpression from, IModelExpression to, string name)
        {
            Node sourceNode = GetNode(g, from);
            Node targetNode = GetNode(g, to);
            childNodes.Add(targetNode);
            return AddEdge(g, sourceNode, targetNode, name);
        }

        protected Edge AddEdge(GraphWriter g, Node sourceNode, Node targetNode, string name)
        {
            string source = sourceNode.ID;
            string target = targetNode.ID;
            Edge edge = g.AddEdge(source, target);
            edge.Label = name;
            edge.FontSize = 8;
            edge.FontColor = Color.LightGray;
            return edge;
        }

        protected void AddGroupEdge(GraphWriter g, Node parent, Node child)
        {
            Edge edge = g.AddEdge(parent.ID, child.ID);
            edge.UserData = this.GroupInstance;
        }
    }
}