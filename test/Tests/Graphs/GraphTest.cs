// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Xunit;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Collections;
//using BasicNode = Microsoft.ML.Probabilistic.Compiler.Graphs.BasicNode<object>;
using BasicEdge = Microsoft.ML.Probabilistic.Compiler.Graphs.Edge<Microsoft.ML.Probabilistic.Compiler.Graphs.BasicEdgeNode>;

namespace Microsoft.ML.Probabilistic.Tests
{
    using System.Linq;
    using Assert = Xunit.Assert;

    
    public class GraphTests
    {
        //public static void Main()
        //{
        //  Console.WriteLine("---------- Test1 ------------");
        //  Test1();
        //  Console.WriteLine("\n---------- Test2 ------------");
        //  Test2();
        //}

        [Fact]
        public void GraphTest()
        {
            Graph<BasicNode> g = new Graph<BasicNode>();
            BasicNode a = new BasicNode("a");
            BasicNode b = new BasicNode("b");
            BasicNode c = new BasicNode("c");
            g.Nodes.Add(a);
            g.Nodes.Add(a); // duplicate is ignored
            g.Nodes.Add(b);
            g.Nodes.Add(c);
            g.AddEdge(a, b);
            g.AddEdge(b, b); // self loop
            g.AddEdge(b, c);
            g.AddEdge(b, c); // double edge
            Assert.True(g.Nodes.Contains(a));
            Assert.True(g.ContainsEdge(a, b));
            Assert.True(g.ContainsEdge(b, b));
            Assert.Equal(4, g.EdgeCount());
            Assert.Equal(1, g.NeighborCount(a));
            Assert.Equal(2, g.NeighborCount(c));
            Assert.Equal(1, g.TargetCount(a));
            Assert.Equal(0, g.SourceCount(a));

            Console.WriteLine("g Edges:");
            foreach (BasicNode source in g.Nodes)
            {
                foreach (BasicNode target in g.TargetsOf(source))
                {
                    Console.WriteLine(" {0} -> {1}", source, target);
                }
            }
            Console.Write("SourcesOf(b):");
            foreach (BasicNode source in g.SourcesOf(b))
            {
                Console.Write(" {0}", source);
            }
            Console.WriteLine();
            Console.Write("NeighborsOf(b):");
            foreach (BasicNode node in g.NeighborsOf(b))
            {
                Console.Write(" {0}", node);
            }
            Console.WriteLine();

            g.RemoveNodeAndEdges(a);
            Console.WriteLine("shape removed");
            Console.WriteLine("g Edges:");
            foreach (BasicNode source in g.Nodes)
            {
                foreach (BasicNode target in g.TargetsOf(source))
                {
                    Console.WriteLine(" {0} -> {1}", source, target);
                }
            }
            Console.WriteLine("b.Sources:");
            foreach (BasicNode source in g.SourcesOf(b))
            {
                Console.WriteLine(" {0}", source);
            }

            g.RemoveEdge(b, c);
            Console.WriteLine("(b,c) removed");
            Console.WriteLine("g Edges:");
            foreach (BasicNode source in g.Nodes)
            {
                foreach (BasicNode target in g.TargetsOf(source))
                {
                    Console.WriteLine(" {0} -> {1}", source, target);
                }
            }
            g.ClearEdgesOf(c);
            Console.WriteLine("c edges cleared");
            Assert.Equal(0, g.NeighborCount(c));
            Console.WriteLine("g Edges:");
            foreach (BasicNode source in g.Nodes)
            {
                foreach (BasicNode target in g.TargetsOf(source))
                {
                    Console.WriteLine(" {0} -> {1}", source, target);
                }
            }
            g.ClearEdges();
            Assert.Equal(0, g.EdgeCount());
            g.Clear();
        }

        [Fact]
        public void GraphEdgeTest()
        {
            Graph<BasicEdgeNode, BasicEdge> g = new Graph<BasicEdgeNode, BasicEdge>(BasicEdge.New);
            BasicEdgeNode a = new BasicEdgeNode("a");
            BasicEdgeNode b = new BasicEdgeNode("b");
            BasicEdgeNode c = new BasicEdgeNode("c");
            g.Nodes.Add(a);
            g.Nodes.Add(a); // duplicate should be ignored
            g.Nodes.Add(b);
            g.Nodes.Add(c);
            BasicEdge ab = g.AddEdge(a, b);
            BasicEdge bb = g.AddEdge(b, b); // self loop
            BasicEdge bc = g.AddEdge(b, c);
            BasicEdge bc2 = g.AddEdge(b, c); // double edge
            Assert.Equal(c, bc.Target);
            Assert.Equal(c, g.TargetOf(bc));
            Assert.Equal(bb, g.GetEdge(b, b));
            Assert.Equal(0, g.EdgeCount(a, c));
            Assert.Equal(2, g.EdgeCount(b, c));
            Assert.Equal(4, g.EdgeCount());
            Assert.Equal(1, g.NeighborCount(a));
            Assert.Equal(2, g.NeighborCount(c));
            Assert.Equal(1, g.TargetCount(a));
            Assert.Equal(0, g.SourceCount(a));

            Console.Write("EdgesOf(b):");
            foreach (BasicEdge edge in g.EdgesOf(b))
            {
                Console.Write(" {0}", edge);
            }
            Console.WriteLine();
            Console.Write("EdgesInto(b):");
            foreach (BasicEdge edge in g.EdgesInto(b))
            {
                Console.Write(" {0}", edge);
            }
            Console.WriteLine();
            Console.Write("EdgesOutOf(b):");
            foreach (BasicEdge edge in g.EdgesOutOf(b))
            {
                Console.Write(" {0}", edge);
            }
            Console.WriteLine();
            Console.Write("EdgesLinking(b,c):");
            foreach (BasicEdge edge in g.EdgesLinking(b, c))
            {
                Console.Write(" {0}", edge);
            }
            Console.WriteLine();

            // clone the graph
            Graph<BasicEdgeNode, BasicEdge> g2 = new Graph<BasicEdgeNode, BasicEdge>(g);

            g.RemoveEdge(bc2);
            Assert.Equal(1, g.NeighborCount(c));
            g.RemoveEdge(b, c);
            Assert.Equal(0, g.NeighborCount(c));
            g.ClearEdgesOf(a);
            Assert.Equal(0, g.NeighborCount(a));
            g.ClearEdges();
            Assert.Equal(0, g.NeighborCount(b));
            g.Clear();

            Assert.Equal(4, g2.EdgeCount());
        }

        [Fact]
        public void IndexedGraphTest()
        {
            IndexedGraph g = new IndexedGraph();
            int a = g.AddNode();
            int b = g.AddNode();
            int c = g.AddNode();
            int ab = g.AddEdge(a, b);
            int bb = g.AddEdge(b, b); // self loop
            int bc = g.AddEdge(b, c);
            int bc2 = g.AddEdge(b, c); // double edge
            Assert.Equal(c, g.TargetOf(bc));
            Assert.Equal(bb, g.GetEdge(b, b));
            Assert.Equal(0, g.EdgeCount(a, c));
            Assert.Equal(2, g.EdgeCount(b, c));
            Assert.Equal(4, g.EdgeCount());
            Assert.Equal(1, g.NeighborCount(a));
            Assert.Equal(2, g.NeighborCount(c));
            Assert.Equal(1, g.TargetCount(a));
            Assert.Equal(0, g.SourceCount(a));

            Console.Write("EdgesOf(b):");
            foreach (int edge in g.EdgesOf(b))
            {
                Console.Write(" {0}", edge);
            }
            Console.WriteLine();
            Console.Write("EdgesInto(b):");
            foreach (int edge in g.EdgesInto(b))
            {
                Console.Write(" {0}", edge);
            }
            Console.WriteLine();
            Console.Write("EdgesOutOf(b):");
            foreach (int edge in g.EdgesOutOf(b))
            {
                Console.Write(" {0}", edge);
            }
            Console.WriteLine();
            Console.Write("EdgesLinking(b,c):");
            foreach (int edge in g.EdgesLinking(b, c))
            {
                Console.Write(" {0}", edge);
            }
            Console.WriteLine();
        }

#if false
    // A graph with edge labels and node labels.
    [Fact]
    public void LabelTest()
    {
        LabeledEdgeGraph<BasicLabeledDirectedNode,object> g = new LabeledEdgeGraph<BasicLabeledDirectedNode,object>();
        BasicLabeledDirectedNode v1 = new BasicLabeledDirectedNode("v1");
        BasicLabeledDirectedNode v2 = new BasicLabeledDirectedNode("v2");
        BasicLabeledDirectedNode f12 = new BasicLabeledDirectedNode("f12");
        g.Nodes.WithLabel("variables").Add(v1);
        g.Nodes.WithLabel("variables").Add(v2);
        g.Nodes.WithLabel("factors").Add(f12);
        g.AddEdge(v1,f12,"arg1");
        g.AddEdge(v2,f12,"arg2");
        g.Nodes.WithLabel("variables").Add(v1); // duplicate node (should be ignored)
        g.CheckValid();

        LabeledEdgeGraph<BasicLabeledDirectedNode,object> g2 = (LabeledEdgeGraph<BasicLabeledDirectedNode,object>)g.Clone();

        Console.WriteLine("{0} is in {1}", v2, g.LabelOf(v2));
        g.ClearNodesAndEdges();
        Console.WriteLine(g2);
    }
#endif

        [Fact]
        public void GraphSearchTest()
        {
            // this is the example graph at http://www.codeproject.com/cs/miscctrl/quickgraph.asp
            Graph<BasicNode> g = new Graph<BasicNode>();
            BasicNode u = new BasicNode("u");
            BasicNode v = new BasicNode("v");
            BasicNode w = new BasicNode("w");
            BasicNode x = new BasicNode("x");
            BasicNode y = new BasicNode("y");
            BasicNode z = new BasicNode("z");
            g.Nodes.Add(u);
            g.Nodes.Add(v);
            g.Nodes.Add(w);
            g.Nodes.Add(x);
            g.Nodes.Add(y);
            g.Nodes.Add(z);
            g.AddEdge(u, v);
            g.AddEdge(u, x);
            g.AddEdge(v, y);
            g.AddEdge(y, x);
            g.AddEdge(x, v);
            g.AddEdge(w, u);
            g.AddEdge(w, y);
            g.AddEdge(w, z);

            DepthFirstSearch<BasicNode> dfs = new DepthFirstSearch<BasicNode>(g);
            dfs.DiscoverNode += delegate(BasicNode node) { Console.WriteLine("discover " + node); };
            dfs.FinishNode += delegate(BasicNode node) { Console.WriteLine("finish " + node); };
            dfs.DiscoverEdge += delegate(Edge<BasicNode> edge) { Console.WriteLine("discover " + edge); };
            dfs.TreeEdge += delegate(Edge<BasicNode> edge) { Console.WriteLine("tree edge " + edge); };
            dfs.FinishTreeEdge += delegate(Edge<BasicNode> edge) { Console.WriteLine("finish tree edge " + edge); };
            dfs.CrossEdge += delegate(Edge<BasicNode> edge) { Console.WriteLine("cross edge " + edge); };
            dfs.BackEdge += delegate(Edge<BasicNode> edge) { Console.WriteLine("back edge " + edge); };
            Console.WriteLine("dfs from u:");
            dfs.SearchFrom(u);
            Console.WriteLine();
            Console.WriteLine("dfs from w:");
            dfs.SearchFrom(w);
            Console.WriteLine();
            dfs.Clear();
            Console.WriteLine("cleared dfs from w:");
            dfs.SearchFrom(w);
            Console.WriteLine();

            Console.WriteLine("bfs:");
            BreadthFirstSearch<BasicNode> bfs = new BreadthFirstSearch<BasicNode>(g);
            IndexedProperty<BasicNode, double> distance = g.CreateNodeData<double>(Double.PositiveInfinity);
            bfs.DiscoverNode += delegate(BasicNode node) { Console.WriteLine("discover " + node); };
            bfs.FinishNode += delegate(BasicNode node) { Console.WriteLine("finish " + node); };
            bfs.TreeEdge += delegate(Edge<BasicNode> edge)
                {
                    Console.WriteLine("tree edge " + edge);
                    distance[edge.Target] = distance[edge.Source] + 1;
                };
            // compute distances from w
            distance[w] = 0;
            bfs.SearchFrom(w);
            Console.WriteLine("distances from w:");
            foreach (BasicNode node in g.Nodes)
            {
                Console.WriteLine("[" + node + "] " + distance[node]);
            }
            Assert.Equal(2.0, distance[x]);
            Console.WriteLine();

            Console.WriteLine("distances from w:");
            DistanceSearch<BasicNode> dists = new DistanceSearch<BasicNode>(g);
            dists.SetDistance += delegate(BasicNode node, int dist) { Console.WriteLine("[" + node + "] " + dist); };
            dists.SearchFrom(w);
            Console.WriteLine();

            BasicNode start = z, end;
            (new PseudoPeripheralSearch<BasicNode>(g)).SearchFrom(ref start, out end);
            Console.WriteLine("pseudo-peripheral nodes: " + start + "," + end);

            StrongComponents<BasicNode> scc = new StrongComponents<BasicNode>(g);
            int count = 0;
            scc.AddNode += delegate(BasicNode node) { Console.Write(" " + node); };
            scc.BeginComponent += delegate()
            {
                Console.Write("[");
                count++;
            };
            scc.EndComponent += delegate() { Console.Write("]"); };
            Console.Write("strong components reachable from w (topological order): ");
            scc.SearchFrom(w);
            Assert.Equal(4, count);
            Console.WriteLine();

            count = 0;
            StrongComponents2<BasicNode> scc2 = new StrongComponents2<BasicNode>(g);
            scc2.AddNode += delegate(BasicNode node) { Console.Write(" " + node); };
            scc2.BeginComponent += delegate()
            {
                Console.Write("[");
                count++;
            };
            scc2.EndComponent += delegate() { Console.Write("]"); };
            Console.Write("strong components reachable from w (rev topological order): ");
            scc2.SearchFrom(w);
            Assert.Equal(4, count);
            Console.WriteLine();

#if false
        Console.WriteLine("CyclicDependencySort:");
        List<BasicNode> schedule = CyclicDependencySort<BasicNode>.Schedule(g,
            delegate(BasicNode node, ICollection<BasicNode> available) {
                if(node == x) return available.Contains(u);
                else return false;
            }, 
            g.Nodes);
        foreach(BasicNode node in schedule) {
            Console.Write(node+" ");
        }
        Console.WriteLine();
        Assert.Equal<int>(6,schedule.Count);
        // order should be: w u x v y z (z can be re-ordered)
        Assert.Equal<BasicNode>(w,schedule[0]);
        Assert.Equal<BasicNode>(u,schedule[1]);
        //Assert.Equal<BasicNode>(z,schedule[5]);
#endif
        }

        internal static Graph<BasicNode> WestGraph()
        {
            // this is the example graph in West sec 7.3.21
            Graph<BasicNode> g = new Graph<BasicNode>();
            BasicNode u = new BasicNode("u");
            BasicNode v = new BasicNode("v");
            BasicNode w = new BasicNode("w");
            BasicNode x = new BasicNode("x");
            BasicNode y = new BasicNode("y");
            BasicNode z = new BasicNode("z");
            g.Nodes.Add(u);
            g.Nodes.Add(v);
            g.Nodes.Add(w);
            g.Nodes.Add(x);
            g.Nodes.Add(y);
            g.Nodes.Add(z);
            g.AddEdge(u, v);
            g.AddEdge(v, w);
            g.AddEdge(w, x);
            g.AddEdge(w, y);
            g.AddEdge(x, u);
            g.AddEdge(x, y);
            g.AddEdge(y, z);
            g.AddEdge(z, u);
            g.AddEdge(z, v);
            return g;
        }

        internal static Graph<BasicEdgeNode, BasicEdge> WestGraph2()
        {
            // this is the example graph in West sec 7.3.21
            var g = new Graph<BasicEdgeNode, BasicEdge>(BasicEdge.New);
            BasicEdgeNode u = new BasicEdgeNode("u");
            BasicEdgeNode v = new BasicEdgeNode("v");
            BasicEdgeNode w = new BasicEdgeNode("w");
            BasicEdgeNode x = new BasicEdgeNode("x");
            BasicEdgeNode y = new BasicEdgeNode("y");
            BasicEdgeNode z = new BasicEdgeNode("z");
            g.Nodes.Add(u);
            g.Nodes.Add(v);
            g.Nodes.Add(w);
            g.Nodes.Add(x);
            g.Nodes.Add(y);
            g.Nodes.Add(z);
            g.AddEdge(u, v);
            g.AddEdge(v, w);
            g.AddEdge(w, x);
            g.AddEdge(w, y);
            g.AddEdge(x, u);
            g.AddEdge(x, y);
            g.AddEdge(y, z);
            g.AddEdge(z, u);
            g.AddEdge(z, v);
            return g;
        }

        internal void CycleFinderTest()
        {
            var g = WestGraph();
            CycleFinder<BasicNode> cf = new CycleFinder<BasicNode>(g);
            cf.AddNode += delegate(BasicNode node) { Console.Write(" " + node); };
            cf.BeginCycle += delegate() { Console.Write("["); };
            cf.EndCycle += delegate() { Console.Write("]"); };
            cf.Search();
            Console.WriteLine();
        }

        internal void CycleFinderTest2()
        {
            var g = WestGraph2();
            CycleFinder<BasicEdgeNode, BasicEdge> cf = new CycleFinder<BasicEdgeNode, BasicEdge>(g);
            cf.AddEdge += delegate(BasicEdge edge) { Console.Write(" " + edge); };
            cf.BeginCycle += delegate() { Console.Write("["); };
            cf.EndCycle += delegate() { Console.Write("]"); };
            cf.Search();
            Console.WriteLine();
        }

        internal void PathFinderTest()
        {
            var g = WestGraph();
            BasicNode u = g.Nodes.Where(node => (string)node.Data == "u").First();
            PathFinder<BasicNode> pf = new PathFinder<BasicNode>(g);
            pf.AddNode += delegate(BasicNode node) { Console.Write(" " + node); };
            pf.BeginPath += delegate() { Console.Write("["); };
            pf.EndPath += delegate() { Console.Write("]"); };
            pf.SearchFrom(u);
            Console.WriteLine();
        }

        internal void PathFinderTest2()
        {
            var g = WestGraph2();
            BasicEdgeNode u = g.Nodes.Where(node => (string)node.Data == "u").First();
            PathFinder<BasicEdgeNode, BasicEdge> pf = new PathFinder<BasicEdgeNode, BasicEdge>(g);
            pf.AddEdge += delegate(BasicEdge edge) { Console.Write(edge); };
            pf.BeginPath += delegate() { Console.Write("["); };
            pf.EndPath += delegate() { Console.Write("]"); };
            pf.SearchFrom(u);
            Console.WriteLine();
        }

        internal void NodeOnPathFinderTest()
        {
            Graph<BasicNode> g = new Graph<BasicNode>();
            BasicNode u = new BasicNode("u");
            BasicNode v = new BasicNode("v");
            BasicNode w = new BasicNode("w");
            BasicNode x = new BasicNode("x");
            BasicNode y = new BasicNode("y");
            BasicNode z = new BasicNode("z");
            g.Nodes.Add(u);
            g.Nodes.Add(v);
            g.Nodes.Add(w);
            g.Nodes.Add(x);
            g.Nodes.Add(y);
            g.Nodes.Add(z);
            g.AddEdge(u, v);
            g.AddEdge(v, w);
            g.AddEdge(w, x);
            g.AddEdge(x, v);
            g.AddEdge(v, y);
            g.AddEdge(v, z);
            g.AddEdge(z, u);

            var nodesOnPath = new HashSet<BasicNode>();
            var pf = new NodeOnPathFinder<BasicNode>(g.TargetsOf, g, MakeIndexedProperty.FromSet(nodesOnPath), node => node.Data.Equals("z"));
            pf.SearchFrom(u);
            Console.Write("nodes on path: ");
            Console.WriteLine(StringUtil.CollectionToString(nodesOnPath, " "));
        }

        internal void EdgeOnPathFinderTest()
        {
            var g = new Graph<BasicEdgeNode, BasicEdge>(BasicEdge.New);
            var u = new BasicEdgeNode("u");
            var v = new BasicEdgeNode("v");
            var w = new BasicEdgeNode("w");
            var x = new BasicEdgeNode("x");
            var y = new BasicEdgeNode("y");
            var z = new BasicEdgeNode("z");
            g.Nodes.Add(u);
            g.Nodes.Add(v);
            g.Nodes.Add(w);
            g.Nodes.Add(x);
            g.Nodes.Add(y);
            g.Nodes.Add(z);
            g.AddEdge(u, v);
            g.AddEdge(v, w);
            g.AddEdge(w, x);
            g.AddEdge(x, v);
            g.AddEdge(v, y);
            g.AddEdge(v, z);
            g.AddEdge(z, u);

            var edgesOnPath = new HashSet<BasicEdge>();
            var pf = new EdgeOnPathFinder<BasicEdgeNode, BasicEdge>(g.EdgesOutOf, g.TargetOf, g, MakeIndexedProperty.FromSet(edgesOnPath), node => node.Data.Equals("z"));
            pf.SearchFrom(u);
            Console.Write("edges on path: ");
            Console.WriteLine(StringUtil.CollectionToString(edgesOnPath, " "));
        }

        internal void CliqueFinderTest()
        {
            var g = WestGraph();
            int count = 0;
            CliqueFinder<BasicNode> cliqueFinder = new CliqueFinder<BasicNode>(g.NeighborsOf);
            cliqueFinder.ForEachClique(g.Nodes, clique =>
            {
                Console.WriteLine("clique: {0}", StringUtil.CollectionToString(clique, ","));
                count++;
            });
            Assert.Equal(5, count);
        }

        [Fact]
        public void MinCutTest()
        {
            Graph<BasicEdgeNode, BasicEdge> g = new Graph<BasicEdgeNode, BasicEdge>(BasicEdge.New);
            // this graph is from Cormen et al, fig 27.1
            BasicEdgeNode s = new BasicEdgeNode("s");
            BasicEdgeNode v1 = new BasicEdgeNode("v1");
            BasicEdgeNode v2 = new BasicEdgeNode("v2");
            BasicEdgeNode v3 = new BasicEdgeNode("v3");
            BasicEdgeNode v4 = new BasicEdgeNode("v4");
            BasicEdgeNode t = new BasicEdgeNode("t");
            g.Nodes.Add(s);
            g.Nodes.Add(v1);
            g.Nodes.Add(v2);
            g.Nodes.Add(v3);
            g.Nodes.Add(v4);
            g.Nodes.Add(t);
            Dictionary<BasicEdge, float> capacity = new Dictionary<BasicEdge, float>();
            BasicEdge e;
            e = g.AddEdge(s, v1);
            capacity[e] = 16;
            e = g.AddEdge(s, v2);
            capacity[e] = 13;
            e = g.AddEdge(v1, v2);
            capacity[e] = 10;
            e = g.AddEdge(v1, v3);
            capacity[e] = 12;
            e = g.AddEdge(v2, v1);
            capacity[e] = 4;
            e = g.AddEdge(v2, v4);
            capacity[e] = 14;
            e = g.AddEdge(v3, v2);
            capacity[e] = 9;
            e = g.AddEdge(v3, t);
            capacity[e] = 20;
            e = g.AddEdge(v4, v3);
            capacity[e] = 7;
            e = g.AddEdge(v4, t);
            capacity[e] = 4;
            var mc = new MinCut<BasicEdgeNode, BasicEdge>(g, e2 => capacity[e2]);
            mc.Sources.Add(s);
            mc.Sinks.Add(t);
            // sourceGroup should be s,v1,v2,v4
            Set<BasicEdgeNode> sourceGroup = mc.GetSourceGroup();
            Console.WriteLine("sourceGroup = {0}", sourceGroup);
            Assert.True(sourceGroup.Count == 4 && sourceGroup.ContainsAll(new BasicEdgeNode[] {s, v1, v2, v4}));
            mc.Sources.Add(v1);
            mc.Sinks.Add(v3);
            sourceGroup = mc.GetSourceGroup();
            Console.WriteLine("sourceGroup = {0}", sourceGroup);
            Assert.True(sourceGroup.Count == 4 && sourceGroup.ContainsAll(new BasicEdgeNode[] {s, v1, v2, v4}));
            capacity[g.GetEdge(v2, v4)] = 1;
            sourceGroup = mc.GetSourceGroup();
            Console.WriteLine("sourceGroup = {0}", sourceGroup);
            Assert.True(sourceGroup.Count == 3 && sourceGroup.ContainsAll(new BasicEdgeNode[] {s, v1, v2}));
        }

        [Fact]
        public void MinCutTest2()
        {
            Graph<BasicEdgeNode, BasicEdge> g = new Graph<BasicEdgeNode, BasicEdge>(BasicEdge.New);
            // this graph is from Cormen et al, fig 27.1
            BasicEdgeNode s = new BasicEdgeNode("s");
            BasicEdgeNode v1 = new BasicEdgeNode("v1");
            BasicEdgeNode v2 = new BasicEdgeNode("v2");
            BasicEdgeNode v3 = new BasicEdgeNode("v3");
            BasicEdgeNode v4 = new BasicEdgeNode("v4");
            BasicEdgeNode t = new BasicEdgeNode("t");
            g.Nodes.Add(s);
            g.Nodes.Add(v1);
            g.Nodes.Add(v2);
            g.Nodes.Add(v3);
            g.Nodes.Add(v4);
            g.Nodes.Add(t);
            Dictionary<BasicEdge, float> capacity = new Dictionary<BasicEdge, float>();
            Dictionary<BasicEdge, float> reverseCapacity = new Dictionary<BasicEdge, float>();
            BasicEdge e;
            e = g.AddEdge(s, v1);
            capacity[e] = 16;
            reverseCapacity[e] = 0;
            e = g.AddEdge(s, v2);
            capacity[e] = 13;
            reverseCapacity[e] = 0;
            e = g.AddEdge(v1, v2);
            capacity[e] = 10;
            reverseCapacity[e] = 4;
            e = g.AddEdge(v1, v3);
            capacity[e] = 12;
            reverseCapacity[e] = 0;
            e = g.AddEdge(v2, v4);
            capacity[e] = 14;
            reverseCapacity[e] = 0;
            e = g.AddEdge(v2, v3);
            capacity[e] = 0;
            reverseCapacity[e] = 9;
            e = g.AddEdge(v3, t);
            capacity[e] = 20;
            reverseCapacity[e] = 0;
            e = g.AddEdge(v4, v3);
            capacity[e] = 7;
            reverseCapacity[e] = 0;
            e = g.AddEdge(v4, t);
            capacity[e] = 4;
            reverseCapacity[e] = 0;
            var mc = new MinCut<BasicEdgeNode, BasicEdge>(g, e2 => capacity[e2]);
            mc.reverseCapacity = e2 => reverseCapacity[e2];
            mc.Sources.Add(s);
            mc.Sinks.Add(t);
            // sourceGroup should be s,v1,v2,v4
            Set<BasicEdgeNode> sourceGroup = mc.GetSourceGroup();
            Console.WriteLine("sourceGroup = {0}", sourceGroup);
            Assert.True(sourceGroup.Count == 4 && sourceGroup.ContainsAll(new BasicEdgeNode[] {s, v1, v2, v4}));
            mc.Sources.Add(v1);
            mc.Sinks.Add(v3);
            sourceGroup = mc.GetSourceGroup();
            Console.WriteLine("sourceGroup = {0}", sourceGroup);
            Assert.True(sourceGroup.Count == 4 && sourceGroup.ContainsAll(new BasicEdgeNode[] {s, v1, v2, v4}));
            capacity[g.GetEdge(v2, v4)] = 1;
            sourceGroup = mc.GetSourceGroup();
            Console.WriteLine("sourceGroup = {0}", sourceGroup);
            Assert.True(sourceGroup.Count == 3 && sourceGroup.ContainsAll(new BasicEdgeNode[] {s, v1, v2}));
        }

        /// <summary>
        /// Test of solving an initializer scheduling problem as a min cut problem.
        /// </summary>
        //[Fact]
        internal void MinCutTest3()
        {
            Graph<BasicEdgeNode, BasicEdge> g = new Graph<BasicEdgeNode, BasicEdge>(BasicEdge.New);
            BasicEdgeNode s = new BasicEdgeNode("s");
            BasicEdgeNode v1 = new BasicEdgeNode("v1");
            BasicEdgeNode v2 = new BasicEdgeNode("v2");
            BasicEdgeNode v3 = new BasicEdgeNode("v3");
            BasicEdgeNode b1 = new BasicEdgeNode("b1");
            BasicEdgeNode b2 = new BasicEdgeNode("b2");
            BasicEdgeNode t = new BasicEdgeNode("t");
            g.Nodes.Add(s);
            g.Nodes.Add(v1);
            g.Nodes.Add(v2);
            g.Nodes.Add(v3);
            g.Nodes.Add(b1);
            g.Nodes.Add(b2);
            g.Nodes.Add(t);
            Dictionary<BasicEdge, float> capacity = new Dictionary<BasicEdge, float>();
            foreach (var node in new[] { v1, v2, v3 })
            {
                BasicEdge e2 = g.AddEdge(s, node);
                capacity[e2] = 1;
            }
            BasicEdgeNode[,] pairs = {
                { v1, b1 },
                { v2, b1 },
                { v2, b2 },
                { v3, b2 }
            };
            for (int i = 0; i < pairs.GetLength(0); i++)
            {
                var source = pairs[i, 0];
                var target = pairs[i, 1];
                BasicEdge e2 = g.AddEdge(source, target);
                capacity[e2] = 1;
            }
            BasicEdge e;
            e = g.AddEdge(b1, t);
            capacity[e] = 1.1f;
            e = g.AddEdge(b2, t);
            capacity[e] = 1.1f;
            var mc = new MinCut<BasicEdgeNode, BasicEdge>(g, e2 => capacity[e2]);
            mc.Sources.Add(s);
            mc.Sinks.Add(t);
            Set<BasicEdgeNode> sourceGroup = mc.GetSourceGroup();
            Console.WriteLine("sourceGroup = {0}", sourceGroup);
        }

        [Fact]
        public void MinCutCycleTest()
        {
            Graph<BasicEdgeNode, BasicEdge> g = new Graph<BasicEdgeNode, BasicEdge>(BasicEdge.New);
            // cycle graph
            BasicEdgeNode s = new BasicEdgeNode("s");
            BasicEdgeNode v1 = new BasicEdgeNode("v1");
            BasicEdgeNode v2 = new BasicEdgeNode("v2");
            BasicEdgeNode t = new BasicEdgeNode("t");
            g.Nodes.Add(s);
            g.Nodes.Add(v1);
            g.Nodes.Add(v2);
            g.Nodes.Add(t);
            Dictionary<BasicEdge, float> capacity = new Dictionary<BasicEdge, float>();
            BasicEdge e;
            e = g.AddEdge(s, v1);
            capacity[e] = 16;
            e = g.AddEdge(v1, v2);
            capacity[e] = 10;
            e = g.AddEdge(v2, s);
            capacity[e] = 14;
            var mc = new MinCut<BasicEdgeNode, BasicEdge>(g, e2 => capacity[e2]);
            mc.Sources.Add(s);
            mc.Sinks.Add(s);
            // sourceGroup should be s,v1
            Set<BasicEdgeNode> sourceGroup = mc.GetSourceGroup();
            Console.WriteLine("sourceGroup = {0}", sourceGroup);
            Assert.True(sourceGroup.Count == 2 && sourceGroup.ContainsAll(new BasicEdgeNode[] {s, v1}));
            mc.Sinks.Remove(s);
            // test the case where t is not reachable from s
            mc.Sinks.Add(t);
            // sourceGroup should be s,v1,v2
            sourceGroup = mc.GetSourceGroup();
            Console.WriteLine("sourceGroup = {0}", sourceGroup);
            Assert.True(sourceGroup.Count == 3 && sourceGroup.ContainsAll(new BasicEdgeNode[] {s, v1, v2}));
            // test IsSinkEdge
            mc.Sinks.Remove(t);
            e = g.GetEdge(v2, s);
            mc.IsSinkEdge = edge => edge.Equals(e);
            // sourceGroup should be s,v1
            sourceGroup = mc.GetSourceGroup();
            Console.WriteLine("sourceGroup = {0}", sourceGroup);
            Assert.True(sourceGroup.Count == 2 && sourceGroup.ContainsAll(new BasicEdgeNode[] {s, v1}));
        }

        [Fact]
        public void IndexedPropertyTest()
        {
            bool[] array = new bool[10];
            IndexedProperty<int, bool> prop = MakeIndexedProperty.FromArray<bool>(array);
            prop[0] = true;
            Assert.True(prop[0]);
            prop.Clear();
            Assert.False(prop[0]);
        }
    }
}