// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Models;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Collapses variables which are statically known to be equal.
    /// </summary>
    internal class EqualityPropagationTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "EqualityPropagationTransform"; }
        }

        private EqualityAnalysisTransform analysis;

        /// <summary>
        /// The original lhs expression that we are currently converting, if any
        /// </summary>
        private IExpression lhs;

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            analysis = new EqualityAnalysisTransform();
            analysis.Context.InputAttributes = context.InputAttributes;
            analysis.Transform(itd);
            context.Results = analysis.Context.Results;
            analysis.PropagateDeterministicValues();
            return base.Transform(itd);
        }

        protected override IStatement DoConvertStatement(IStatement ist)
        {
            IStatement st = base.DoConvertStatement(ist);
            // if the statement has been changed, then its lhs must have been replaced with a constant, 
            // in which case this is no longer an assignment but a constraint.
            if (!ReferenceEquals(st, ist)) context.OutputAttributes.Set(st, new Microsoft.ML.Probabilistic.Models.Constraint());
            return st;
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            IExpression oldLhs = lhs;
            lhs = null;
            IExpression expr = ConvertExpression(iae.Expression);
            // do not replace the lhs if this is just an array create statement
            if (!(expr is IArrayCreateExpression))
                lhs = iae.Target;
            IExpression target = ConvertExpression(iae.Target);
            lhs = oldLhs;
            if (ReferenceEquals(expr, iae.Expression) && ReferenceEquals(target, iae.Target)) return iae;
            IAssignExpression ae = Builder.AssignExpr();
            ae.Expression = expr;
            ae.Target = target;
            context.InputAttributes.CopyObjectAttributesTo(iae, context.OutputAttributes, ae);
            return ae;
        }

        protected override IExpression DoConvertExpression(IExpression expr)
        {
            if (lhs != null)
            {
                // expr must be a sub-expression of lhs
                Containers containers = new Containers(context);
                Containers exprContainers;
                IExpression newExpr = analysis.GetNewExpression(expr, containers, out exprContainers);
                if (newExpr != null)
                {
                    // replace the assignment
                    //   expr[indices] = rhs;
                    // with
                    //   newExpr[indices] = rhs;
                    //   expr[indices] = newExpr[indices];
                    IExpression replacedLhs = Builder.ReplaceExpression(lhs, expr, newExpr);
                    IExpression ae = Builder.AssignExpr(lhs, replacedLhs);
                    IStatement st = Builder.ExprStatement(ae);
                    containers = Containers.RemoveUnusedLoops(containers, context, lhs);
                    int ancIndex = containers.GetMatchingAncestorIndex(context);
                    Containers missing = containers.GetContainersNotInContext(context, ancIndex);
                    st = Containers.WrapWithContainers(st, missing.inputs);
                    context.AddStatementAfterAncestorIndex(ancIndex, st);
                    // attach the original rhs as an attribute on the new assignment
                    IAssignExpression iae = context.FindAncestor<IAssignExpression>();
                    MarginalPrototype mp = new MarginalPrototype(null);
                    mp.prototypeExpression = iae.Expression;
                    context.OutputAttributes.Set(ae, mp);
                    return newExpr;
                }
            }
            return base.DoConvertExpression(expr);
        }
    }

    /// <summary>
    /// Analyses equality constraints.
    /// </summary>
    internal class EqualityAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "EqualityAnalysisTransform"; }
        }

        /// <summary>
        /// Edges go from general scopes to more specific scopes
        /// </summary>
        private IndexedGraph graph = new IndexedGraph();
        private IndexedProperty<int, IExpression> newExpression;
        private Set<KeyValuePair<int, IExpression>> observedNodes = new Set<KeyValuePair<int, IExpression>>();
        private ScopedDictionary<IExpression, int, Containers> nodeOf = new ScopedDictionary<IExpression, int, Containers>();

        internal EqualityAnalysisTransform()
        {
            newExpression = graph.CreateNodeData<IExpression>();
        }

        /// <summary>
        /// Given an expression appearing in a set of containers, return an equivalent expression and the containers that it is valid in.
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="containers"></param>
        /// <param name="exprContainers"></param>
        /// <returns></returns>
        internal IExpression GetNewExpression(IExpression expr, Containers containers, out Containers exprContainers)
        {
            // find all nodes whose containers are more general than the current containers
            var nodesAndScopes = nodeOf.GetAll(expr).Where(vis => containers.Contains(vis.Scope));
            foreach (var nodeInScope in nodesAndScopes)
            {
                int node = nodeInScope.Value;
                IExpression newExpr = newExpression[node];
                if (newExpr != null)
                {
                    exprContainers = nodeInScope.Scope;
                    return newExpr;
                }
            }
            exprContainers = null;
            return null;
        }

        /// <summary>
        /// Propagate the expressions attached to observedNodes to all nodes they can reach.
        /// </summary>
        internal void PropagateDeterministicValues()
        {
            DepthFirstSearch<int> dfs = new DepthFirstSearch<int>(graph);
            int observedNode = -1;
            IExpression observedExpr = null;
            dfs.DiscoverNode += delegate(int node) { if (node != observedNode) newExpression[node] = observedExpr; };
            foreach (KeyValuePair<int, IExpression> entry in observedNodes)
            {
                observedNode = entry.Key;
                observedExpr = entry.Value;
                dfs.SearchFrom(observedNode);
            }
        }

        /// <summary>
        /// Analyse equality constraints, which appear as static method calls to Constrain.Equal
        /// </summary>
        /// <param name="imie"></param>
        /// <returns></returns>
        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            // If this is a constrain equal, then we will add the equality to the equality map.
            if (Recognizer.IsStaticGenericMethod(imie, new Action<PlaceHolder, PlaceHolder>(Constrain.Equal<PlaceHolder>)))
            {
                int node0 = CreateNodeAndEdges(imie.Arguments[0]);
                int node1 = CreateNodeAndEdges(imie.Arguments[1]);
                graph.AddEdge(node0, node1);
                graph.AddEdge(node1, node0);
            }
            return base.ConvertMethodInvoke(imie);
        }

        /// <summary>
        /// Creates the node (if needed) and adds directed edges to the graph.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        private int CreateNodeAndEdges(IExpression expr)
        {
            Containers scope = new Containers(context);
            int node;
            if (!nodeOf.TryGetExact(expr, scope, out node))
            {
                node = graph.AddNode();
                nodeOf.Add(expr, node, scope);
            }
            if (Recognizer.GetParameterDeclaration(expr) != null || expr is ILiteralExpression)
            {
                // expr is an observed value
                observedNodes.Add(new KeyValuePair<int, IExpression>(node, expr));
            }
            foreach (var vis in nodeOf.GetAll(expr))
            {
                // If scopes are equal (same node), do nothing
                if (vis.Value == node) continue;

                if (scope.Contains(vis.Scope))
                {
                    // This scope contains the other scope, therefore is more specific
                    graph.AddEdge(vis.Value, node);
                }
                else if (vis.Scope.Contains(scope))
                {
                    // This scope is contained in the other scope, therefore is more general
                    graph.AddEdge(node, vis.Value);
                }
            }
            return node;
        }

        /// <summary>
        /// Show the expression graph for debugging
        /// </summary>
        internal void ShowGraph()
        {
            if (InferenceEngine.Visualizer?.DependencyGraphVisualizer != null)
            {
                InferenceEngine.Visualizer.DependencyGraphVisualizer.VisualizeDependencyGraph(graph, null,
                    node =>
                    {
                        foreach (var entry in nodeOf)
                        {
                            if (entry.Value.Value == node)
                                return entry.Key + " " + entry.Value.Scope;
                        }
                        return "?";
                    },
                    null, "Equality Graph");
            }
        }
    }

        /// <summary>
        /// Dictionary where the key-value pairs are contained in a particular scope.
        /// </summary>
        internal class ScopedDictionary<TKey, TValue, TScope> : IEnumerable<KeyValuePair<TKey,ScopedDictionary<TKey,TValue,TScope>.ValueInScope>>
    {
        private Dictionary<TKey, List<ValueInScope>> dict =
            new Dictionary<TKey, List<ValueInScope>>();

        public IEnumerator<KeyValuePair<TKey, ValueInScope>> GetEnumerator()
        {
            foreach (var entry in dict)
            {
                foreach (var valueInScope in entry.Value)
                {
                    yield return new KeyValuePair<TKey, ValueInScope>(entry.Key, valueInScope);
                }
            }
        }
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        /// <summary>
        /// Tries to get a value in exactly the specifed scope, without inheritance.
        /// </summary>
        /// <param name="key">The key for the required value</param>
        /// <param name="scope">The specified scope</param>
        /// <param name="value">The value, if any is found or default(TValue)</param>
        /// <returns>True if a value was found, false otherwise</returns>
        internal bool TryGetExact(TKey key, TScope scope, out TValue value)
        {
            List<ValueInScope> l;
            if (dict.TryGetValue(key, out l))
            {
                foreach (var vic in l)
                {
                    if (scope.Equals(vic.Scope))
                    {
                        value = vic.Value;
                        return true;
                    }
                }
            }
            value = default(TValue);
            return false;
        }

        /// <summary>
        /// Adds a key-value pair in the specified scope.
        /// </summary>
        /// <param name="key"></param>
        /// <param name="value"></param>
        /// <param name="scope"></param>
        internal void Add(TKey key, TValue value, TScope scope)
        {
            List<ValueInScope> l;
            if (!dict.TryGetValue(key, out l))
            {
                l = new List<ValueInScope>();
                dict[key] = l;
            }
            l.Add(new ValueInScope {Value = value, Scope = scope});
        }

        /// <summary>
        /// Gets all values associated with the key, in any scope.
        /// In combination with a Linq Where() clause can be used to filter values by scope.
        /// </summary>
        /// <param name="key">The key to retrieve values for</param>
        /// <returns></returns>
        internal IEnumerable<ValueInScope> GetAll(TKey key)
        {
            List<ValueInScope> l;
            if (dict.TryGetValue(key, out l))
            {
                foreach (var vis in l)
                {
                    yield return vis;
                }
            }
        }

        public override string ToString()
        {
            return StringUtil.DictionaryToString<TKey, List<ValueInScope>>(dict, ",");
        }

        internal class ValueInScope
        {
            internal TValue Value { get; set; }
            internal TScope Scope { get; set; }

            public override string ToString()
            {
                return Value.ToString() + " " + Scope.ToString();
            }
        }
    }
}