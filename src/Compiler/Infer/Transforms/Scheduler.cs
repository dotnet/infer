// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Diagnostics;
using System.Text;
using System.Threading;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Compiler.Visualizers;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Serialization;
using Microsoft.ML.Probabilistic.Utilities;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    using Cycle = List<EdgeIndex>;

    internal class Scheduler
    {
        internal static bool verbose, showAncestors, showGraphs, showUnwind, showMinCut, showTimings, showCapacityBreakdown, showOffsetEdges;
        internal bool debug, doRepair = true, useRepair2 = false;
        internal bool useExperimentalSerialSchedules;
        internal Action<string, IEnumerable<string>> RecordText;
        internal static bool useFakeGraph;
        internal static bool initsCanBeStale = false;
        internal static bool UseOffsetEdgesDuringInit = true;
        /// <summary>
        /// Toggle this to debug scheduling problems.  Setting it to true breaks TutorialTests.MatchboxRecommender.
        /// </summary>
        /// <remarks>
        /// To isolate the source of the scheduling problem, set SchedulingTransform.debug = true to find the cycle with the most back edges.
        /// Check each back edge on this cycle.  The debug output explains why the edge is backward, including the edge costs.
        /// To get a more detailed look at edge costs, set verbose = true or showGraphs = true.
        /// </remarks>
        internal static bool NoInitEdgesAreInfinite = false;
        /// <summary>
        /// Ignore requirements due to non-sequential offset edges. Only meaningful if UseOffsetEdgesDuringInit = true
        /// </summary>
        internal static bool IgnoreOffsetRequirements = false;
        /// <summary>
        /// Used for debugging
        /// </summary>
        private List<Cycle> cycles;
        /// <summary>
        /// Maps a node/group in g to a node/group in dg
        /// </summary>
        private List<NodeIndex> originalNode;
        /// <summary>
        /// Maps a node/group in dg to nodes/groups in g
        /// </summary>
        private List<List<NodeIndex>> newNodes;
        /// <summary>
        /// Maps an edge in g to an edge in dg.  Used to look up edge attributes.
        /// </summary>
        private List<EdgeIndex> originalEdge;

        private DependencyGraph dg;
        private IndexedGraph g;

        private enum Direction
        {
            Unknown,
            Forward,
            Backward
        };

        private List<Direction> direction;
        // maps a node or group index into a group index (or -1 if no group)
        private NodeIndex[] groupOf;
        private GroupGraph groupGraph;
        private bool graphHasVirtualEdges;
        private Set<EdgeIndex> deletedEdges = new Set<EdgeIndex>();
        internal IndexedProperty<NodeIndex, HashSet<IVariableDeclaration>> loopVarsOfNode;
        internal List<IVariableDeclaration> loopVarsWithOffset;

        // variables used by initialization
        private Set<EdgeIndex> backwardInSchedule;
        private bool mustInitBackwardEdges;

        // variables used by UpdateCosts
        /// <summary>
        /// precomputed all-pairs distances to reduce time at expense of memory. 
        /// distance[source][target] is the number of directed edges on the shortest path from source to target.
        /// </summary>
        private int[][] distance;
        // used for updating edge costs
        private Stack<EdgeIndex> newForwardEdges;
        // used for updating edge costs
        private Stack<EdgeIndex> newBackEdges = new Stack<EdgeIndex>();

        // variables used by PropagateConstraints
        // cached dfs objects to reduce gc
        private DepthFirstSearch<NodeIndex> dfsAncestorsWithoutGroups, dfsAncestorsWithGroups;
        private DepthFirstSearch<NodeIndex> dfsDescendantsWithoutGroups, dfsDescendantsWithGroups;
        private Set<NodeIndex> ancestors = new Set<EdgeIndex>();
        private Stack<EdgeIndex> todo = new Stack<EdgeIndex>();
        private Dictionary<NodeIndex, OffsetBoundCollection> ancestorOffset = new Dictionary<NodeIndex, OffsetBoundCollection>();
        private Dictionary<NodeIndex, OffsetBoundCollection> descendantOffset = new Dictionary<NodeIndex, OffsetBoundCollection>();

        // variables used by GetSchedule
        private DepthFirstSearch<NodeIndex> dfsSchedule;
        private List<NodeIndex> schedule;

        internal List<NodeIndex> IterationSchedule(DependencyGraph dg, NodeIndex[] groupOf = null,
            bool forceInitializedNodes = false, IEnumerable<EdgeIndex> edgesToReverse = null)
        {
            this.dg = dg;
            Set<EdgeIndex> forcedForwardEdges = new Set<EdgeIndex>();
            Set<EdgeIndex> forcedBackEdges = new Set<EdgeIndex>();
            if (useFakeGraph)
            {
                //CreateCompleteGraph(3);
                CreateGridGraph();
                CreateNodeData();
            }
            else
            {
                this.groupOf = groupOf;
                CreateGraph(forcedForwardEdges, forcedBackEdges, forceInitializedNodes);
            }
            CreateEdgeData();
            foreach (EdgeIndex edge in forcedForwardEdges)
            {
                if (direction[edge] == Direction.Backward)
                    throw new Exception($"Internal: {EdgeToString(edge)} was not forced forward");
                direction[edge] = Direction.Forward;
            }
            // set to false for debugging
            bool canRotateSchedule = true;
            if (useFakeGraph)
            {
                cycles = FindCycles(g);
                if (verbose)
                    Debug.WriteLine("found {0} cycles", cycles.Count);
                // sort cycles by length, shortest first
                cycles.Sort((c1, c2) => c1.Count.CompareTo(c2.Count));
                if (true)
                {
                    Rand.Restart(DateTime.Now.Millisecond);
                    g = ShuffleGraph(g);
                    g = ShuffleGraph(g);
                }
                canRotateSchedule = false;
            }
            // when there are groups, the leaf graph need not be strongly connected
            if (debug && groupOf == null)
                CheckStronglyConnected(g);
            ComputeDistances();
            float[] edgeCost = GetEdgeCosts3();
            if (debug) RecordNodes();
            if (verbose)
                WriteEdgeCosts(edgeCost);
            if (debug && g.Nodes.Count < 100 && verbose && showOffsetEdges)
                SchedulingTransform.DrawOffsetEdges(dg);
            PropagateMustNotInit(forcedForwardEdges, edgeCost);
            bool fixDotaTest = false;
            if (fixDotaTest)
            {
                // for DotaTest
                EdgeIndex edge = g.GetEdge(7, 13);
                forcedForwardEdges.Add(edge);
            }
            if (forceInitializedNodes)
            {
                foreach (EdgeIndex edge in forcedBackEdges)
                {
                    if (direction[edge] == Direction.Forward)
                    {
                        throw new Exception($"Internal: {EdgeToString(edge)} was not forced backward");
                    }
                    // no need to Debug.WriteLine the edge here since it was already printed when added to forcedBackEdges.
                    direction[edge] = Direction.Backward;
                    newBackEdges.Push(edge);
                    todo.Push(edge);
                }
                PropagateConstraints(true);
                UpdateCosts3(edgeCost);
            }
            UpdateCostsFromOffsetEdges(edgeCost);
            if (debug && verbose)
            {
                if (g.Nodes.Count < 300 && showGraphs)
                    DrawLabeledGraph("initial edge costs", false, edgeCost, showNoInit: true);
                else
                    WriteEdgeCosts(edgeCost);
            }
            while (true)
            {
                EdgeIndex newBackEdge = FindNewBackEdge(edgeCost, forcedForwardEdges);
                if (newBackEdge == -1)
                    break;
                if (debug)
                {
                    Debug.WriteLine("adding back edge " + edgeCost[newBackEdge] + " " + EdgeToString(newBackEdge));
                    //DrawLabeledGraph("adding back edge", false, edgeCost);
                }
                direction[newBackEdge] = Direction.Backward;
                newBackEdges.Push(newBackEdge);
                todo.Push(newBackEdge);
                while (todo.Count > 0)
                {
                    PropagateConstraints(true);
                    PropagateMustNotInit2(forcedForwardEdges);
                }
                //UpdateCosts2(edgeCost);
                UpdateCosts3(edgeCost);
            }
            //if (debug) CheckSchedule(schedule, cycles, cycleGraph);
            List<NodeIndex> schedule;
            if (debug)
            {
                if (g.Nodes.Count > 1 && false)
                    DrawLabeledGraph("before rotate");
                schedule = GetSchedule(true);
                schedule = ConvertToOriginalNodes(schedule);
                RecordSchedule("pre rotate", schedule);
            }
            if (canRotateSchedule)
            {
                if (edgesToReverse != null)
                {
                    // used by SpecialFirstiteration to influence the iter schedule
                    ForEachEdgeInG(edgesToReverse, edge =>
                    {
                        NodeIndex source = g.SourceOf(edge);
                        NodeIndex target = g.TargetOf(edge);
                        if (groupGraph.GetGroupSet(source).ContainsAny(groupGraph.GetGroups(target)))
                            return;
                        forcedBackEdges.Add(edge);
                    });
                }
                Stopwatch watch = new Stopwatch();
                watch.Start();
                AssignLabelsByMinCutWithGroups(forcedForwardEdges, forcedBackEdges);
                watch.Stop();
                if (showTimings)
                    Console.WriteLine("({0}ms for MinCut)", watch.ElapsedMilliseconds);
                CheckMustNotInit();
                schedule = GetSchedule(true);
            }
            else
            {
                schedule = GetSchedule(true);
            }
            schedule = ConvertToOriginalNodes(schedule);
            if (useFakeGraph)
            {
                // two-way grid
                //schedule = new List<NodeIndex>() { 12, 15, 21, 18, 0, 1, 2, 13, 16, 22, 19, 3, 4, 5, 14, 17, 23, 20, 11, 10, 9, 8, 7, 6 };
                // corner2corner grid
                //schedule = new List<NodeIndex>() { 12, 15, 0, 1, 2, 13, 16, 3, 4, 5, 14, 17, 23, 20, 11, 10, 9, 22, 19, 8, 7, 6, 21, 18 };
                // write cycle lengths and number of back edges
                int[] counts = GetBackEdgeCounts(schedule);
                using (MatlabWriter writer = new MatlabWriter("grid.mat"))
                {
                    writer.Write("counts", Array.ConvertAll(counts, i => (double)i));
                    writer.Write("lengths", Array.ConvertAll(cycles.ToArray(), c => (double)c.Count));
                    writer.Write("schedule", Array.ConvertAll(schedule.ToArray(), i => (double)i));
                    object[,] cyclesMatrix = new object[cycles.Count, 1];
                    for (int i = 0; i < cycles.Count; i++)
                    {
                        cyclesMatrix[i, 0] = Array.ConvertAll(new List<NodeIndex>(NodesOfCycle(cycles[i])).ToArray(), node => (double)node);
                    }
                    writer.Write("cycles", cyclesMatrix);
                }
                //Console.ReadKey();
            }
            int count = schedule.Count;
            if (debug)
            {
                Debug.WriteLine("schedule has {0} nodes before repair", schedule.Count);
                RecordSchedule("pre repair", schedule);
                WriteSchedule(schedule);
                if (count > 1 && count < 300 && showGraphs)
                {
                    DrawLabeledGraph("iteration schedule", inThread: false);
                }
            }
            if (doRepair)
            {
                if (useRepair2)
                    schedule = dg.RepairSchedule2Cyclic(schedule);
                else
                    schedule = dg.RepairScheduleCyclic(schedule);
            }
            if (!useFakeGraph)
                AssignLabelsFromSchedule(schedule);
            if (doRepair && debug && schedule.Count != count)
            {
                Debug.WriteLine("schedule has {0} nodes after repair", schedule.Count);
                RecordSchedule("post repair", schedule);
                if (count > 1 && count < 3000)
                {
                    WriteSchedule(schedule);
                }
            }
            if (groupOf != null)
                CheckGroups(schedule);
            return schedule;
        }

        /// <summary>
        /// Check that the schedule satisfies the group constraints
        /// </summary>
        /// <param name="schedule">Contains nodes in dg</param>
        private void CheckGroups(IEnumerable<NodeIndex> schedule)
        {
            Set<NodeIndex> openGroups = new Set<EdgeIndex>();
            Set<NodeIndex> closedGroups = new Set<EdgeIndex>();
            List<NodeIndex> newlyClosedGroups = new List<EdgeIndex>();
            foreach (NodeIndex node in schedule)
            {
                Set<NodeIndex> groups = GetGroupsInDg(node);
                bool hasFreshOutEdge = g.EdgesOutOf(node).Any(IsFreshEdge);
                if (!hasFreshOutEdge)
                {
                    // it is okay for repair to pull a statement out of a group.
                    // the problem is inserting a statement into a group.
                    foreach (NodeIndex group in groups)
                    {
                        if (closedGroups.Contains(group))
                            throw new Exception($"Internal: schedule splits group {group} at node {NodeToString(node)}");
                        if (!openGroups.Contains(group))
                            openGroups.Add(group);
                    }
                }
                newlyClosedGroups.Clear();
                foreach (NodeIndex group in openGroups)
                {
                    if (!groups.Contains(group))
                        newlyClosedGroups.Add(group);
                }
                closedGroups.AddRange(newlyClosedGroups);
                openGroups.Remove(newlyClosedGroups);
            }
        }

        /// <summary>
        /// Check that the edge directions satisfy the mustNotInit constraints
        /// </summary>
        private void CheckMustNotInit()
        {
            if (dg.mustNotInit == null || dg.mustNotInit.Count == 0)
                return;
            // for efficiency, this could be restricted to nodes in mustNotInit
            float[] initCosts = GetEdgeCostsInit(inIteration: true);
            foreach (EdgeIndex edge in g.Edges)
            {
                if (direction[edge] == Direction.Backward && float.IsPositiveInfinity(initCosts[edge]))
                {
                    NodeIndex source = g.SourceOf(edge);
                    NodeIndex originalSource = originalNode[source];
                    NodeIndex target = g.TargetOf(edge);
                    NodeIndex originalTarget = originalNode[target];
                    if (dg.mustNotInit.Contains(originalSource))
                    {
                        List<NodeIndex> targets = new List<NodeIndex>();
                        foreach (EdgeIndex edge2 in g.EdgesOutOf(source))
                        {
                            if (float.IsPositiveInfinity(initCosts[edge2]))
                                targets.Add(g.TargetOf(edge2));
                        }
                        // To workaround this error, change Any requirements to more specific requirements (on a single operator method argument)
                        // A complete fix would require allowing the scheduler to backtrack to find the right set of edges to force forward during IterationSchedule
                        Trace.WriteLine(string.Format("Scheduler failed: {0} must not init but required by {1}",
                            NodeToString(source),
                            StringUtil.CollectionToString(targets.Select(t => NodeToString(t)), " ")));
                    }
                }
            }
        }

        private void PropagateMustNotInit(Set<EdgeIndex> forcedForwardEdges, float[] edgeCost)
        {
            if (dg.mustNotInit == null || dg.mustNotInit.Count == 0)
                return;
            // propagate mustNotInit to Required descendants that are not initialized
            Set<NodeIndex> toAdd = new Set<EdgeIndex>();
            Converter<NodeIndex, IEnumerable<NodeIndex>> successors;
            if (dg.initializedNodes.Count == 0)
                successors = RequiredTargets;
            else
            {
                successors = node => RequiredTargets(node).Where(target => !dg.initializedNodes.Contains(target));
            }
            DepthFirstSearch<NodeIndex> dfsMust = new DepthFirstSearch<EdgeIndex>(successors, dg.dependencyGraph);
            dfsMust.FinishNode += toAdd.Add;
            dfsMust.SearchFrom(dg.mustNotInit);
            dg.mustNotInit.AddRange(toAdd);
            // force all required edges out of mustNotInit nodes to be forward edges
            if (debug)
            {
                Debug.Write("must not init: ");
                var array = dg.mustNotInit.ToArray();
                Array.Sort(array);
                Debug.WriteLine(StringUtil.CollectionToString(array, " "));
            }
            foreach (NodeIndex node in dg.mustNotInit)
            {
                if (dg.initializedNodes.Contains(node))
                    continue;
                foreach (EdgeIndex edgeOrig in dg.dependencyGraph.EdgesOutOf(node))
                {
                    NodeIndex targetOrig = dg.dependencyGraph.TargetOf(edgeOrig);
                    if (targetOrig == node) continue;
                    if (dg.initializedEdges.Contains(edgeOrig)) continue;
                    if (dg.isRequired[edgeOrig])
                    {
                        foreach (NodeIndex source in newNodes[node])
                        {
                            foreach (NodeIndex target in newNodes[targetOrig])
                            {
                                EdgeIndex edge;
                                if (!g.TryGetEdge(source, target, out edge))
                                    continue;
                                if (direction[edge] == Direction.Backward)
                                    throw new Exception($"Internal: {EdgeToString(edge)} was not forced forward");
                                if (debug)
                                    Debug.WriteLine($"mustNotInit forced edge {EdgeToString(edge)} Forward");
                                direction[edge] = Direction.Forward;
                                forcedForwardEdges.Add(edge);
                                todo.Push(edge);
                            }
                        }
                    }
                    else if (!dg.noInit[edgeOrig])
                    {
                        foreach (NodeIndex source in newNodes[node])
                        {
                            foreach (NodeIndex target in newNodes[targetOrig])
                            {
                                EdgeIndex edge;
                                if (!g.TryGetEdge(source, target, out edge))
                                    continue;
                                if (verbose)
                                {
                                    Debug.WriteLine($"mustNotInit incremented edgeCost[{EdgeToString(edge)}] by 10 because {source} must not init");
                                }
                                edgeCost[edge] += 10f;
                            }
                        }
                    }
                }
            }
            while (todo.Count > 0)
            {
                PropagateConstraints(true);
                PropagateMustNotInit2(forcedForwardEdges);
            }
            UpdateCosts3(edgeCost);
        }

        private void PropagateMustNotInit2(Set<EdgeIndex> forcedForwardEdges)
        {
            if (dg.mustNotInit == null || dg.mustNotInit.Count == 0)
                return;
            float[] edgeCostInit = GetEdgeCostsInit(inIteration: true);
            foreach (NodeIndex node in dg.mustNotInit)
            {
                if (dg.initializedNodes.Contains(node))
                    continue;
                foreach (EdgeIndex edgeOrig in dg.dependencyGraph.EdgesOutOf(node))
                {
                    foreach (NodeIndex source in newNodes[node])
                    {
                        foreach (NodeIndex target in newNodes[dg.dependencyGraph.TargetOf(edgeOrig)])
                        {
                            EdgeIndex edge;
                            if (!g.TryGetEdge(source, target, out edge))
                                continue;
                            if (float.IsPositiveInfinity(edgeCostInit[edge]) && (direction[edge] == Direction.Unknown))
                            {
                                if (debug)
                                    Debug.WriteLine($"mustNotInit forced edge {EdgeToString(edge)} Forward due to Any dependency");
                                direction[edge] = Direction.Forward;
                                forcedForwardEdges.Add(edge);
                                todo.Push(edge);
                            }
                        }
                    }
                }
            }
        }

        private void LabelInitializedEdges(Set<EdgeIndex> forcedForwardEdges, Set<EdgeIndex> forcedBackEdges, bool forceInitializedNodes)
        {
            // force all edges out of initialised nodes to be back edges (except to other initialised nodes or increments)
            if (debug)
            {
                Debug.WriteLine("initialized nodes: " + StringUtil.CollectionToString(dg.initializedNodes.OrderBy(node => node), " "));
                Debug.WriteLine("initialized edges: " + StringUtil.CollectionToString(dg.initializedEdges.Select(edge => EdgeToString(dg.dependencyGraph, edge)), ""));
            }

            // collect the set of initialized edges
            Set<EdgeIndex> initializedEdges = new Set<NodeIndex>();
            ForEachEdgeInG(dg.initializedEdges, initializedEdges.Add);
            foreach (NodeIndex nodeInDg in dg.initializedNodes)
            {
                foreach (NodeIndex source in newNodes[nodeInDg])
                {
                    // all edges out of an initialized node are considered initialized.
                    foreach (EdgeIndex edge in g.EdgesOutOf(source))
                    {
                        initializedEdges.Add(edge);
                    }
                }
            }

            List<Edge<NodeIndex>> edgesToAdd = new List<Edge<EdgeIndex>>();
            foreach (EdgeIndex edge in initializedEdges)
            {
                NodeIndex source = g.SourceOf(edge);
                Set<NodeIndex> sourceGroups = groupGraph.GetGroupSet(source);
                NodeIndex target = g.TargetOf(edge);
                NodeIndex origTarget = originalNode[target];
                if (dg.initializedNodes.Contains(origTarget) && !forceInitializedNodes)
                    continue;
                // don't force edges within a group
                if (!forceInitializedNodes && sourceGroups.ContainsAny(groupGraph.GetGroups(target)))
                    continue;
                // we want to rotate the schedule based on initialised nodes, but we don't want to influence the convergence rate.
                // thus we add the edge to forcedBackEdges but we don't label it.
                if (debug)
                    Debug.WriteLine($"forcing edge {EdgeToString(edge)} backward due to initializer");
                forcedBackEdges.Add(edge);
                if (forceInitializedNodes)
                {
                    // whenever an edge is forced backward, we have to enforce the following constraints:
                    // 1. If the edge is fresh or the source is triggered, force a forward path from target to source.  Otherwise a parent of source could precede target and force source to run early.
                    // 2. If the edge is a trigger or the target has a fresh outgoing edge, force a forward edge from the child of target to source.  Otherwise the child of target could follow source and force target to run again.
                    // TODO: a similar constraint is needed for triggers
                    if (IsFreshEdge(edge))
                    {
                        //throw new Exception(string.Format("{0} is initialized but {1} is fresh", NodeToString(g.SourceOf(edge)), EdgeToString(edge)));
                        // 1. all edges into the Fresh node must be forward
                        // 2. parents of the Fresh node must follow its children
                        foreach (EdgeIndex edge2 in g.EdgesInto(source))
                        {
                            forcedForwardEdges.Add(edge2);
                            NodeIndex parent = g.SourceOf(edge2);
                            var edgeToAdd = new Edge<NodeIndex>(target, parent);
                            edgesToAdd.Add(edgeToAdd);
                            if (debug)
                                Debug.WriteLine($"forcing {edgeToAdd} forward due to fresh initialized edge {EdgeToString(edge)}");
                        }
                    }
                    else
                    {
                        foreach (EdgeIndex edge2 in g.EdgesOutOf(target))
                        {
                            if (IsFreshEdge(edge2))
                            {
                                //forcedForwardEdges.Add(edge2);
                                NodeIndex child = g.TargetOf(edge2);
                                var edgeToAdd = new Edge<NodeIndex>(child, source);
                                edgesToAdd.Add(edgeToAdd);
                                if (debug)
                                    Debug.WriteLine($"forcing {edgeToAdd} forward due to fresh edge {EdgeToString(edge2)} out of target {target} of initialized edge {EdgeToString(edge)}");
                            }
                        }
                    }
                }
            }
            // actually add the edges
            foreach (var edge in edgesToAdd)
            {
                NodeIndex source = edge.Source;
                NodeIndex target = edge.Target;
                EdgeIndex forcedEdge;
                if (!g.TryGetEdge(source, target, out forcedEdge))
                {
                    forcedEdge = g.AddEdge(source, target);
                    originalEdge.Add(-1);
                    graphHasVirtualEdges = true;
                }
                forcedForwardEdges.Add(forcedEdge);
            }
        }

        private NodeIndex GetNewGroup(NodeIndex oldGroup)
        {
            if (oldGroup == -1)
                return oldGroup;
            foreach (var group in this.newNodes[oldGroup])
                return group;
            throw new Exception("dg group " + oldGroup + " has no g group");
        }

        /// <summary>
        /// Invokes action on the statements that need to be executed at the end of the convergence loop.
        /// </summary>
        /// <param name="dg"></param>
        /// <param name="outputs"></param>
        /// <param name="action"></param>
        public void ForEachFinalizer(DependencyGraph dg, IEnumerable<NodeIndex> outputs, Action<NodeIndex> action)
        {
            foreach (NodeIndex node in outputs)
            {
                foreach (NodeIndex target in newNodes[node])
                {
                    // does target have any back edges into it?
                    foreach (EdgeIndex edge in g.EdgesInto(target))
                    {
                        if (direction[edge] == Direction.Backward)
                        {
                            action(node);
                            break;
                        }
                    }
                }
            }
        }

        private void ForEachEdgeInG(IEnumerable<EdgeIndex> edgesInDg, Action<EdgeIndex> action)
        {
            foreach (var edgeInDg in edgesInDg)
            {
                foreach (var source in newNodes[dg.dependencyGraph.SourceOf(edgeInDg)])
                {
                    foreach (var target in newNodes[dg.dependencyGraph.TargetOf(edgeInDg)])
                    {
                        if (g.TryGetEdge(source, target, out int edgeInG))
                        {
                            action(edgeInG);
                        }
                    }
                }
            }
        }

        public class StackFrame
        {
            public EdgeIndex edge;
            public List<EdgeIndex> forcedEdges = new List<EdgeIndex>();
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>
        /// Select a subset of nodes from the dependency graph to execute prior to the iterSchedule.  IterationSchedule must have been called previously to set up the graph.
        /// </summary>
        /// <param name="iterSchedule">The iteration schedule</param>
        /// <returns></returns>
        public List<NodeIndex> UserInitSchedule(List<NodeIndex> iterSchedule)
        {
            // TODO
            if (dg.initializedNodes.Count == 0)
                return new List<NodeIndex>();
            newForwardEdges = new Stack<EdgeIndex>();
            if (graphHasVirtualEdges)
            {
                // rebuild the graph without virtual edges
                CreateGraph(null, null, false);
                CreateEdgeData();
                AssignLabelsFromSchedule(iterSchedule);
            }
            // un-delete the Cancels edges and label them according to the iterSchedule
            if (AddCancelsEdges())
                AssignLabelsFromSchedule(iterSchedule);
            if (UseOffsetEdgesDuringInit && AddOffsetEdges())
                AssignLabelsFromSchedule(iterSchedule);

            // backwardInSchedule includes all backward edges in iterSchedule except for:
            // - edges with initialized sources
            // - edges with mustNotInit sources
            // - self-loops
            // - offset edges
            backwardInSchedule = new Set<EdgeIndex>();
            // collect set of existing back edges
            foreach (EdgeIndex edge in g.Edges)
            {
                if (direction[edge] == Direction.Backward && !deletedEdges.Contains(edge))
                {
                    NodeIndex source = g.SourceOf(edge);
                    NodeIndex target = g.TargetOf(edge);
                    if (source == target)
                        continue; // self-loop
                    bool notInitialized = (useFakeGraph || !dg.initializedNodes.Contains(originalNode[source]));
                    //if (!notInitialized && !initsCanBeStale)
                    if (!notInitialized)
                        continue;
                    backwardInSchedule.Add(edge);
                }
            }
            // save the old directions
            List<Direction> directionIter = new List<Direction>(direction);
            // clear the labels
            foreach (EdgeIndex edge in g.Edges)
            {
                // self-loops must be labeled backward here otherwise they will be labeled forward below
                if (g.SourceOf(edge) == g.TargetOf(edge))
                    direction[edge] = Direction.Backward;
                else
                    direction[edge] = Direction.Unknown;
            }
            // initializerChildren holds all children of user-initialized nodes
            Set<NodeIndex> initializerChildren = new Set<EdgeIndex>();
            if (!useFakeGraph && dg.initializedNodes.Count > 0)
            {
                // force all edges out of initialised nodes to be back edges (except to other initialised nodes)
                // a node whose outgoing edges are all backward can never be part of the initialization schedule
                foreach (NodeIndex node in dg.initializedNodes)
                {
                    foreach (NodeIndex source in newNodes[node])
                    {
                        Set<NodeIndex> sourceGroups = groupGraph.GetGroupSet(source);
                        foreach (EdgeIndex edge in g.EdgesOutOf(source))
                        {
                            NodeIndex target = g.TargetOf(edge);
                            if (dg.initializedNodes.Contains(originalNode[target]))
                                continue;
                            // if this edge is back in the iterSchedule, then we shouldn't search from the target
                            if (directionIter[edge] == Direction.Forward)
                                initializerChildren.Add(target);
                            // TEMPORARY
                            continue;
                            var targetGroups = groupGraph.GetGroups(target);
                            if (sourceGroups.ContainsAny(targetGroups))
                                continue;  // don't force edges within a group
                            direction[edge] = Direction.Backward;
                            if (debug)
                                Debug.WriteLine("{0} forced backward since {1}{2} is initialized", EdgeToString(edge), source, (node == source) ? "" : String.Format(" (originally {0})", node));
                            newBackEdges.Push(edge);
                            todo.Push(edge);
                        }
                    }
                }
                PropagateConstraints();
            }
            return new List<NodeIndex>();
        }

        /// <summary>
        /// Select a subset of nodes from the dependency graph to execute prior to the iterSchedule.  IterationSchedule must have been called previously to set up the graph.
        /// </summary>
        /// <param name="iterSchedule">The iteration schedule</param>
        /// <param name="mustInitBackwardEdges">If true, will reconstruct all backward messages (experimental option for UseSpecialFirst)</param>
        /// <returns></returns>
        public List<NodeIndex> InitSchedule(List<NodeIndex> iterSchedule, bool mustInitBackwardEdges = false)
        {
            this.mustInitBackwardEdges = mustInitBackwardEdges;
            if (debug && mustInitBackwardEdges)
                Debug.WriteLine("mustInitBackwardEdges: {0}", mustInitBackwardEdges);
            newForwardEdges = new Stack<EdgeIndex>();
            if (graphHasVirtualEdges)
            {
                // rebuild the graph without virtual edges
                CreateGraph(null, null, false);
                CreateEdgeData();
                AssignLabelsFromSchedule(iterSchedule);
            }
            if (mustInitBackwardEdges)
            {
                AddCancelsEdges();
            }
            else
            {
                // un-delete the Cancels edges and label them according to the iterSchedule
                if (AddCancelsEdges())
                    AssignLabelsFromSchedule(iterSchedule);
            }
            if (UseOffsetEdgesDuringInit && AddOffsetEdges())
                AssignLabelsFromSchedule(iterSchedule);

            // backwardInSchedule includes all backward edges in iterSchedule except for:
            // - edges with initialized sources
            // - edges with mustNotInit sources
            // - self-loops
            // - offset edges
            backwardInSchedule = new Set<EdgeIndex>();
            // collect set of existing back edges
            foreach (EdgeIndex edge in g.Edges.Where(edge => direction[edge] == Direction.Backward && !deletedEdges.Contains(edge)))
            {
                NodeIndex source = g.SourceOf(edge);
                NodeIndex target = g.TargetOf(edge);
                if (source == target && !mustInitBackwardEdges)
                    continue; // self-loop
                bool notInitialized = (useFakeGraph || !dg.initializedNodes.Contains(originalNode[source]));
                //if (!notInitialized && !initsCanBeStale)
                if (!notInitialized)
                    continue;
                if (dg.initializedEdges.Contains(originalEdge[edge]))
                    continue;
                if (IsOffsetEdge(originalEdge[edge]))
                    continue;
                if (dg.mustNotInit != null && dg.mustNotInit.Contains(originalNode[source]))
                {
                    if (IsRequired(edge, includeAny: false))
                        throw new Exception($"Internal: node {NodeToString(source)} is required by iteration schedule but must not init");
                    continue;
                }
                backwardInSchedule.Add(edge);
            }
            // compute the cost of existing back edges
            // uses backwardInSchedule
            float[] edgeCostIter = GetEdgeCostsInit(inIteration: true);
            if (debug && verbose) WriteEdgeCosts(edgeCostIter);
            foreach (var iterBackEdge in g.Edges.Where(edge => direction[edge] == Direction.Backward && !deletedEdges.Contains(edge)))
            {
                var source = g.SourceOf(iterBackEdge);
                var originalSource = originalNode[source];
                var sourceStmt = dg.Nodes[originalSource];
                bool sourceIsIncrement = dg.attributes.Has<IncrementStatement>(sourceStmt);
                string incrementString;
                if (sourceIsIncrement) incrementString = "increment to ";
                else incrementString = "";
                var message = dg.NodeToShortString(originalSource);
                var target = g.TargetOf(iterBackEdge);
                var originalTarget = originalNode[target];
                var targetStmt = dg.Nodes[originalTarget];
                var cost = edgeCostIter[iterBackEdge];
                dg.attributes.Add(targetStmt, new BackEdgeAttribute($"{incrementString}{message}, cost = {cost}"));
            }
            // save the old directions
            List<Direction> directionIter = new List<Direction>(direction);
            // clear the labels
            foreach (EdgeIndex edge in g.Edges)
            {
                // self-loops must be labeled backward here otherwise they will be labeled forward below
                if (g.SourceOf(edge) == g.TargetOf(edge))
                    direction[edge] = Direction.Backward;
                else
                    direction[edge] = Direction.Unknown;
            }
            float[] edgeCost = GetEdgeCostsInit(inIteration: false);
            if (debug && verbose) WriteEdgeCosts(edgeCost);
            // initializerChildren holds all children of user-initialized nodes
            Set<NodeIndex> initializerChildren = new Set<EdgeIndex>();
            if (!useFakeGraph && dg.initializedNodes.Count > 0)
            {
                // force all edges out of initialised nodes to be back edges (except to other initialised nodes)
                // a node whose outgoing edges are all backward can never be part of the initialization schedule
                foreach (NodeIndex node in dg.initializedNodes)
                {
                    foreach (NodeIndex source in newNodes[node])
                    {
                        Set<NodeIndex> sourceGroups = groupGraph.GetGroupSet(source);
                        foreach (EdgeIndex edge in g.EdgesOutOf(source))
                        {
                            NodeIndex target = g.TargetOf(edge);
                            if (dg.initializedNodes.Contains(originalNode[target]))
                                continue;
                            // if this edge is back in the iterSchedule, then we shouldn't search from the target
                            if (directionIter[edge] == Direction.Forward)
                                initializerChildren.Add(target);
                            // TEMPORARY
                            continue;
                            var targetGroups = groupGraph.GetGroups(target);
                            if (sourceGroups.ContainsAny(targetGroups))
                                continue;  // don't force edges within a group
                            direction[edge] = Direction.Backward;
                            if (debug)
                                Debug.WriteLine("{0} forced backward since {1}{2} is initialized", EdgeToString(edge), source, (node == source) ? "" : String.Format(" (originally {0})", node));
                            newBackEdges.Push(edge);
                            todo.Push(edge);
                        }
                    }
                }
                PropagateConstraints();
                UpdateCostsInit(edgeCost);
            }
            if (dg.mustNotInit != null)
            {
                // force all edges out of mustNotInit nodes to be back edges (except to other mustNotInit nodes)
                // a node whose outgoing edges are all backward can never be part of the initialization schedule
                // note: this only works because UpdateCostsInit ignores these edges
                if (debug)
                {
                    Debug.Write("must not init: ");
                    var array = dg.mustNotInit.ToArray();
                    Array.Sort(array);
                    Debug.WriteLine(Util.CollectionToString(array));
                }
                foreach (NodeIndex node in dg.mustNotInit)
                {
                    foreach (EdgeIndex edge in g.EdgesOutOf(node))
                    {
                        NodeIndex target = g.TargetOf(edge);
                        if (dg.mustNotInit.Contains(originalNode[target]))
                            continue;
                        if (direction[edge] == Direction.Unknown)
                        {
                            direction[edge] = Direction.Backward;
                            newBackEdges.Push(edge);
                            todo.Push(edge);
                        }
                    }
                }
                PropagateConstraints();
                UpdateCostsInit(edgeCost);
            }

            // Phase 1: label edges to satisfy Required/SkipIfUniform constraints
            Stack<StackFrame> frames = new Stack<StackFrame>();
            bool cannotInitialise = false;
            bool backtracking = false;
            while (true)
            {
                EdgeIndex currentEdge = -1;
                if (!backtracking)
                {
                    // find the unlabeled edge with highest back edge cost and make it a forward edge
                    // can speed this up by using a heap
                    float maxCost = float.NegativeInfinity;
                    maxCost = 0; // this causes alg to stop when remaining edges have cost 0
                    maxCost = 2 * defaultInitCost; // this causes alg to only label requirement edges
                    EdgeIndex newForwardEdge = -1;
                    for (int edge = 0; edge < edgeCost.Length; edge++)
                    {
                        if (direction[edge] == Direction.Unknown && edgeCost[edge] > maxCost)
                        {
                            maxCost = edgeCost[edge];
                            newForwardEdge = edge;
                        }
                    }
                    if (newForwardEdge == -1)
                        break;
                    if (debug)
                    {
                        Debug.WriteLine("phase 1: new forward edge = {0} (cost {1})", EdgeToString(newForwardEdge), maxCost);
                        //if (verbose && maxCost < float.PositiveInfinity) DrawLabeledGraph("new forward edge", false, edgeCost);
                    }
                    direction[newForwardEdge] = Direction.Forward;
                    newForwardEdges.Push(newForwardEdge);
                    todo.Push(newForwardEdge);
                    currentEdge = newForwardEdge;
                }
                else
                {
                    if (frames.Count == 0)
                    {
                        // to debug this problem:
                        // set verbose=true, showGraphs=true
                        if (debug && g.Nodes.Count > 1)
                            DrawLabeledGraph("init failure", false, edgeCost);
                        cannotInitialise = true;
                        break;
                    }
                    StackFrame oldFrame = frames.Pop();
                    EdgeIndex newBackEdge = oldFrame.edge;
                    Direction oldDirection = direction[newBackEdge];
                    if (debug && verbose)
                    {
                        Debug.WriteLine("unwinding " + EdgeToString(oldFrame.edge));
                        if (showUnwind)
                            DrawLabeledGraph("unwind", false, edgeCost);
                    }
                    // unwind labels
                    foreach (EdgeIndex edge in oldFrame.forcedEdges)
                    {
                        direction[edge] = Direction.Unknown;
                    }
                    foreach (EdgeIndex edge in oldFrame.forcedEdges)
                    {
                        NodeIndex target = g.TargetOf(edge);
                        UpdateCostsInit(edgeCost, target, false);
                    }
                    if (oldDirection == Direction.Forward)
                    {
                        // labeling the edge forward led to failure, so try backward
                        //if (debug && verbose) Debug.WriteLine("changing {0} to backward", EdgeToString(newBackEdge), false);
                        direction[newBackEdge] = Direction.Backward;
                        newBackEdges.Push(newBackEdge);
                        todo.Push(newBackEdge);
                        currentEdge = newBackEdge;
                    }
                    else
                    {
                        // labeling the edge backward led to failure, so remove the label and backtrack
                        //if (debug && verbose) Debug.WriteLine("erasing " + EdgeToString(newBackEdge));
                        direction[newBackEdge] = Direction.Unknown;
                        continue;
                    }
                }
                PropagateConstraints();
                StackFrame frame = new StackFrame();
                frame.edge = currentEdge;
                // record all edges labeled by PropagateConstraints (this includes currentEdge)
                frame.forcedEdges.AddRange(newForwardEdges);
                frame.forcedEdges.AddRange(newBackEdges);
                frames.Push(frame);
                backtracking = !UpdateCostsInit(edgeCost);
            }

            // Phase 2: label edges to make use of user-provided initializations
            // initializerDescendants will be modified to hold all backwardInSchedule sources that descend from user-initialized nodes in the init graph
            Set<NodeIndex> initializerDescendants = new Set<EdgeIndex>();
            if (initializerChildren.Count > 0)
            {
                if (debug)
                    Debug.WriteLine("initializerChildren: {0}", initializerChildren);
                // Unknown edges between a user-provided initialization and iter back edge are made forward
                // For an example, see (new NonconjugateVMP2Tests()).MixtureOfMultivariateGaussiansWithHyperLearning();
                bool labelDescendants = false;
                if (labelDescendants)
                    LabelEdgesDescendingFromInits(dg.initializedNodes, initializerChildren, initializerDescendants);
            }

            // Phase 3: Label the remaining edges with cost > 0
            // this is needed for later cost calculations to be accurate
            // if want to avoid initialization, should prefer existing forward edges
            while (true)
            {
                float maxCost = float.NegativeInfinity;
                maxCost = 0; // this causes alg to stop when remaining edges have cost 0
                EdgeIndex newForwardEdge = -1;
                for (int edge = 0; edge < edgeCost.Length; edge++)
                {
                    if (direction[edge] == Direction.Unknown && edgeCost[edge] > maxCost)
                    {
                        maxCost = edgeCost[edge];
                        newForwardEdge = edge;
                    }
                }
                if (newForwardEdge == -1)
                    break;
                if (debug)
                {
                    Debug.WriteLine("phase 3: new forward edge = {0} (cost {1})", EdgeToString(newForwardEdge), maxCost);
                    //if (maxCost < float.PositiveInfinity) DrawLabeledGraph(false, edgeCost);
                }
                direction[newForwardEdge] = Direction.Forward;
                newForwardEdges.Push(newForwardEdge);
                todo.Push(newForwardEdge);
                PropagateConstraints();
            }

            // Edge labeling is complete.  Now collect nodes needed by the iterSchedule.
            // Starting from a seed node, collect a set of nodes bracketed by init back edges at top and iter/init back edges at bottom.
            // Call these edge sets A and B,
            // if the total cost of A is less than the total cost of B, then keep these nodes in the init schedule, otherwise delete them.
            Set<NodeIndex> nodesToInit = new Set<EdgeIndex>();
            bool costIsInfinite;
            bool useGroups = false;
            bool newMethod = true;
            if (newMethod)
            {
                costIsInfinite = GetNodesToInit(edgeCostIter, directionIter, edgeCost, initializerDescendants, nodesToInit, useGroups);
            }
            else
            {
                costIsInfinite = GetNodesToInit1(edgeCostIter, directionIter, edgeCost, initializerDescendants, nodesToInit, useGroups);
            }
            if (nodesToInit.Count > 0 && cannotInitialise)
                throw new Exception("Scheduling constraints cannot be satisfied.  This is usually due to a missing or overspecified method attribute");
            // construct a schedule for nodesToInit
            // label any remaining unlabeled edges
            foreach (NodeIndex source in nodesToInit)
            {
                foreach (EdgeIndex edge in (groupGraph == null) ? g.EdgesOutOf(source) : groupGraph.EdgesOutOf(source))
                {
                    if (direction[edge] == Direction.Unknown)
                    {
                        NodeIndex target = g.TargetOf(edge);
                        if (nodesToInit.Contains(target))
                        {
                            direction[edge] = Direction.Forward;
                            todo.Push(edge);
                            PropagateConstraints();
                        }
                    }
                }
            }
            List<NodeIndex> schedule = GetSchedule(useGroups);
            if (useGroups)
                schedule.RemoveAll(node => !nodesToInit.Contains(node) && !nodesToInit.ContainsAny(groupGraph.GetGroups(node)));
            else
                schedule.RemoveAll(node => !nodesToInit.Contains(node));
            schedule = ConvertToOriginalNodes(schedule);
            if (!useGroups && !initsCanBeStale)
            {
                // delete initialized nodes from the schedule
                schedule.RemoveAll(node => dg.initializedNodes.Contains(node));
            }
            if (debug)
            {
                Debug.WriteLine("init schedule before repair:");
                WriteSchedule(schedule);
                if (g.Nodes.Count > 1 && g.Nodes.Count < 200 && showGraphs)
                    DrawLabeledGraph("init schedule", true, edgeCost);
            }
            if (costIsInfinite)
                throw new Exception("Scheduler failed to produce a valid initialization. Try initializing some variables with InitialiseTo().");
            // if we're initializing a prefix of the iterSchedule, then something is wrong (a common bug)
            if (schedule.Count > 0)
            {
                bool isPrefix = true;
                for (int i = 0; i < schedule.Count; i++)
                {
                    if (iterSchedule[i] != schedule[i])
                    {
                        isPrefix = false;
                    }
                }
                if (isPrefix)
                    throw new Exception("Internal: init schedule is redundant");
            }
            if (doRepair)
            {
                schedule = RepairCombinedSchedule(iterSchedule, schedule);
                if (EnumerableExtensions.AreEqual(schedule, iterSchedule))
                {
                    // this can happen if RepairSchedule creates new back edges
                    if (debug)
                        Debug.WriteLine("init schedule matches iter schedule");
                    schedule.Clear();
                }
            }
            return schedule;
        }

        void writeEdges(IEnumerable<EdgeIndex> edges, float[] edgeCosts)
        {
            foreach (EdgeIndex edge in edges)
            {
                string suffix = DoubleToString(edgeCosts[edge]);
                Debug.Write(EdgeToString(edge) + suffix + " ");
            }
            Debug.WriteLine("");
        }

        // old method
        private bool GetNodesToInit1(float[] edgeCostIter, List<Direction> directionIter, float[] edgeCost, Set<int> initializerDescendants, Set<int> nodesToInit, bool useGroups)
        {
            bool costIsInfinite = false;
            Set<EdgeIndex> initBackEdges = new Set<EdgeIndex>();
            Set<EdgeIndex> iterBackEdges = new Set<EdgeIndex>();
            Set<EdgeIndex> newInitBackEdges = new Set<EdgeIndex>();
            Set<EdgeIndex> newIterBackEdges = new Set<EdgeIndex>();
            DepthFirstSearch<NodeIndex> dfsInitBackEdges, dfsIterBackEdges;
            if (useGroups)
            {
                throw new Exception();
                dfsIterBackEdges = new DepthFirstSearch<NodeIndex>(ForwardSourcesAndForwardTargetsWithGroups, groupGraph);
                dfsIterBackEdges.FinishNode += delegate (NodeIndex node)
                {
                    // if we initialize this node, we remove iter back edges out but we add init back edges in.
                    foreach (EdgeIndex edge in groupGraph.EdgesInto(node))
                    {
                        if (direction[edge] == Direction.Backward && g.SourceOf(edge) != g.TargetOf(edge))
                        {
                            initBackEdges.Add(edge);
                        }
                    }
                    foreach (EdgeIndex edge in groupGraph.EdgesOutOf(node))
                    {
                        if (backwardInSchedule.Contains(edge))
                        {
                            iterBackEdges.Add(edge);
                        }
                    }
                };
            }
            else
            {
                dfsInitBackEdges = new DepthFirstSearch<NodeIndex>(ForwardSources, g);
                dfsInitBackEdges.FinishNode += delegate (NodeIndex node)
                {
                    // if we initialize this node, we remove iter back edges out but we add init back edges in.
                    foreach (EdgeIndex edge in g.EdgesInto(node))
                    {
                        if (direction[edge] == Direction.Backward && g.SourceOf(edge) != g.TargetOf(edge))
                        {
                            newInitBackEdges.Add(edge);
                        }
                    }
                };
                dfsIterBackEdges = new DepthFirstSearch<NodeIndex>(ForwardTargets, g);
                dfsIterBackEdges.FinishNode += delegate (NodeIndex node)
                {
                    // if we initialize this node, we remove iter back edges out but we add init back edges in.
                    foreach (EdgeIndex edge in g.EdgesOutOf(node))
                    {
                        if (backwardInSchedule.Contains(edge))
                        {
                            newIterBackEdges.Add(edge);
                        }
                    }
                };
            }
            DepthFirstSearch<NodeIndex> dfsBack;
            if (useGroups)
            {
                dfsBack = new DepthFirstSearch<EdgeIndex>(ForwardSourcesWithGroups, groupGraph);
            }
            else
            {
                dfsBack = new DepthFirstSearch<EdgeIndex>(ForwardSources, g);
            }
            dfsBack.FinishNode += delegate (NodeIndex node)
            {
                if (debug)
                {
                    string groupString = "";
                    if (useGroups && (node >= groupGraph.firstGroup))
                        groupString = "group ";
                    Debug.WriteLine("found {1}{0}", node, groupString);
                }
                nodesToInit.Add(node);
            };
            Set<EdgeIndex> edgesTodo = new Set<EdgeIndex>();
            edgesTodo.AddRange(backwardInSchedule);
            while (edgesTodo.Count > 0)
            {
                initBackEdges.Clear();
                iterBackEdges.Clear();
                EdgeIndex seedEdge = -1;
                // give priority to iter back edges that are not init back edges
                foreach (EdgeIndex edge in edgesTodo)
                {
                    if (direction[edge] != Direction.Backward)
                    {
                        seedEdge = edge;
                        break;
                    }
                }
                if (seedEdge == -1)
                {
                    // pick any iter back edge
                    seedEdge = edgesTodo.First();
                }
                newIterBackEdges.Add(seedEdge);
                // enlarge the seed set to have more iterBackEdges
                while (newIterBackEdges.Count > 0)
                {
                    if (debug)
                    {
                        Debug.Write("newIterBackEdges: ");
                        writeEdges(newIterBackEdges, edgeCostIter);
                    }
                    // find all initBackEdges through forward paths
                    foreach (var edge in newIterBackEdges)
                    {
                        NodeIndex seedNode = g.SourceOf(edge);
                        dfsInitBackEdges.SearchFrom(seedNode);
                    }
                    iterBackEdges.AddRange(newIterBackEdges);
                    newIterBackEdges.Clear();
                    if (debug)
                    {
                        Debug.Write("newInitBackEdges: ");
                        writeEdges(newInitBackEdges, edgeCost);
                    }
                    initBackEdges.AddRange(newInitBackEdges);
                    if (dg.mustNotInit != null && dg.mustNotInit.Count > 0)
                        break;
                    // find all iterBackEdges through forward paths
                    foreach (var edge in newInitBackEdges)
                    {
                        dfsIterBackEdges.SearchFrom(g.TargetOf(edge));
                    }
                    newInitBackEdges.Clear();
                }
                // compute cost of the two edge sets
                float initCost = SumSubarray(edgeCost, initBackEdges);
                float iterCost = SumSubarray(edgeCostIter, iterBackEdges);
                if (debug)
                {
                    Debug.Write("initBackEdges: ");
                    writeEdges(initBackEdges, edgeCost);
                    Debug.Write("iterBackEdges: ");
                    writeEdges(iterBackEdges, edgeCostIter);
                }
                bool hasInitializerDescendant = false;
                foreach (EdgeIndex edge in iterBackEdges)
                {
                    NodeIndex source = g.SourceOf(edge);
                    if (initializerDescendants.Contains(source))
                    {
                        hasInitializerDescendant = true;
                        break;
                    }
                }
                if (initCost < iterCost || hasInitializerDescendant)
                {
                    if (debug && initCost < iterCost)
                        Debug.WriteLine("initializer reduced cost from {0} to {1}", iterCost, initCost);
                    // initialize these nodes, but only the ones that are ancestors of iter back edges with non-zero cost, or user-provided initializer descendants
                    // check edges that are forward in the init schedule first
                    foreach (EdgeIndex edge in iterBackEdges)
                    {
                        List<Direction> directionOld = direction;
                        direction = directionIter;
                        NodeIndex target = g.TargetOf(edge);
                        UpdateCostsInit(edgeCostIter, target, inIteration: true);
                        direction = directionOld;
                        NodeIndex source = g.SourceOf(edge);
                        if (edgeCostIter[edge] > 0 || initializerDescendants.Contains(source))
                        {
                            if (debug)
                            {
                                string txt = DoubleToString(edgeCostIter[edge]);
                                if (initializerDescendants.Contains(source))
                                    txt += " init";
                                Debug.WriteLine("searching from {0} {1}", EdgeToString(edge), txt);
                            }
                            if (useGroups)
                            {
                                Set<NodeIndex> targetGroups = groupGraph.GetGroupSet(target);
                                dfsBack.SearchFrom(groupGraph.GetLargestGroupExcluding(source, targetGroups));
                            }
                            else
                            {
                                dfsBack.SearchFrom(source);
                            }
                            directionIter[edge] = Direction.Forward;
                        }
                        else if (debug)
                            Debug.WriteLine("cost[{0}] = {1}", EdgeToString(edge), edgeCostIter[edge]);
                    }
                }
                else
                {
                    if (debug)
                        Debug.WriteLine("did not initialize since cost did not reduce (from {0} to {1})", iterCost, initCost);
                    if (float.IsPositiveInfinity(iterCost))
                        costIsInfinite = true;
                }
                edgesTodo.Remove(iterBackEdges);
            }

            return costIsInfinite;
        }

        /// <summary>
        /// Determine which nodes are worth initializing based on the relative costs of back edges in the iter schedule and init schedule.
        /// </summary>
        /// <param name="edgeCostIter"></param>
        /// <param name="directionIter"></param>
        /// <param name="edgeCost"></param>
        /// <param name="initializerDescendants"></param>
        /// <param name="nodesToInit">Modified on exit</param>
        /// <param name="useGroups"></param>
        /// <returns>True if we failed to initialize an edge with infinite cost</returns>
        private bool GetNodesToInit(float[] edgeCostIter, List<Direction> directionIter, float[] edgeCost, Set<int> initializerDescendants, Set<int> nodesToInit, bool useGroups)
        {
            bool costIsInfinite = false;
            if (debug && false)
            {
                Debug.WriteLine("edgeCostIter");
                writeEdges(g.Edges, edgeCostIter);
                Debug.WriteLine("edgeCost");
                writeEdges(g.Edges, edgeCost);
            }
            // when a source of an iter back edge is initialized, we eliminate the cost of that edge, but we gain the cost of any init back edges.
            // finding the optimal set of iter back edges to initialize is solved optimally as a min cut problem in a flow graph.
            DepthFirstSearch<NodeIndex> dfsInitBackEdges = new DepthFirstSearch<NodeIndex>(ForwardSources, g);
            Set<EdgeIndex> newInitBackEdges = new Set<EdgeIndex>();
            dfsInitBackEdges.FinishNode += delegate (NodeIndex node)
            {
                // if we initialize this node, we remove iter back edges out but we add init back edges in.
                foreach (EdgeIndex edge in g.EdgesInto(node))
                {
                    if (direction[edge] == Direction.Backward && g.SourceOf(edge) != g.TargetOf(edge))
                    {
                        newInitBackEdges.Add(edge);
                    }
                }
            };
            // construct the flow graph
            IndexedGraph flowGraph = new IndexedGraph();
            NodeIndex sourceNode = flowGraph.AddNode();
            NodeIndex sinkNode = flowGraph.AddNode();
            IndexedProperty<EdgeIndex, float> capacity = flowGraph.CreateEdgeData<float>();
            Dictionary<EdgeIndex, NodeIndex> initNodeInFlowGraph = new Dictionary<NodeIndex, NodeIndex>();
            Dictionary<EdgeIndex, NodeIndex> iterNodeInFlowGraph = new Dictionary<EdgeIndex, NodeIndex>();
            const float costOfInit = 1e-4f;
            foreach (var iterBackEdge in backwardInSchedule)
            {
                newInitBackEdges.Clear();
                dfsInitBackEdges.Clear();
                dfsInitBackEdges.SearchFrom(g.SourceOf(iterBackEdge));
                if (debug && verbose)
                {
                    Debug.Write($"iterBackEdge {EdgeToString(iterBackEdge)}{DoubleToString(edgeCostIter[iterBackEdge])} leads to: ");
                    writeEdges(newInitBackEdges, edgeCost);
                }
                bool leadsToSelf = newInitBackEdges.Contains(iterBackEdge);
                var modifiedCost = leadsToSelf ? (edgeCostIter[iterBackEdge] - edgeCost[iterBackEdge]) : edgeCostIter[iterBackEdge];
                // an edge with zero capacity should not be added to the graph.  it will not necessarily end up in the source group.
                if (modifiedCost <= 0f)
                    continue;
                NodeIndex flowGraphNodeOfIterBackEdge = flowGraph.AddNode();
                iterNodeInFlowGraph[iterBackEdge] = flowGraphNodeOfIterBackEdge;
                EdgeIndex edgeToSink = flowGraph.AddEdge(flowGraphNodeOfIterBackEdge, sinkNode);
                capacity[edgeToSink] = modifiedCost;
                // connect in the flow graph
                foreach (var initBackEdge in newInitBackEdges)
                {
                    NodeIndex flowGraphNodeOfInitBackEdge;
                    if (!initNodeInFlowGraph.TryGetValue(initBackEdge, out flowGraphNodeOfInitBackEdge))
                    {
                        flowGraphNodeOfInitBackEdge = flowGraph.AddNode();
                        initNodeInFlowGraph[initBackEdge] = flowGraphNodeOfInitBackEdge;
                        EdgeIndex edgeFromSource = flowGraph.AddEdge(sourceNode, flowGraphNodeOfInitBackEdge);
                        // increase cost slightly so that there is no initialization when cost is unchanged.
                        capacity[edgeFromSource] = edgeCost[initBackEdge] + costOfInit;
                    }
                    EdgeIndex innerEdge = flowGraph.AddEdge(flowGraphNodeOfInitBackEdge, flowGraphNodeOfIterBackEdge);
                    capacity[innerEdge] = float.PositiveInfinity;
                }
            }
            MinCut<NodeIndex, EdgeIndex> mc = new MinCut<EdgeIndex, EdgeIndex>(flowGraph, e => capacity[e]);
            mc.Sources.Add(sourceNode);
            mc.Sinks.Add(sinkNode);
            Set<NodeIndex> sourceGroup = mc.GetSourceGroup();
            // iterBackEdges in the source group do not get initialized.
            if (debug)
            {
                // display info about the selected edges
                Set<EdgeIndex> initBackEdges = new Set<int>();
                Set<EdgeIndex> iterBackEdges = new Set<int>();
                foreach (var iterBackEdge in backwardInSchedule)
                {
                    if (iterNodeInFlowGraph.ContainsKey(iterBackEdge) && !sourceGroup.Contains(iterNodeInFlowGraph[iterBackEdge]))
                        iterBackEdges.Add(iterBackEdge);
                }
                foreach (var entry in initNodeInFlowGraph)
                {
                    if (!sourceGroup.Contains(entry.Value))
                        initBackEdges.Add(entry.Key);
                }
                Debug.Write("initBackEdges: ");
                writeEdges(initBackEdges, edgeCost);
                Debug.Write("iterBackEdges: ");
                writeEdges(iterBackEdges, edgeCostIter);
                float initCost = SumSubarray(edgeCost, initBackEdges);
                float iterCost = SumSubarray(edgeCostIter, iterBackEdges);
                if (initCost < iterCost)
                {
                    Debug.WriteLine($"initializer reduced cost from {iterCost} to {initCost}");
                }
                else
                {
                    Debug.WriteLine($"did not initialize since cost did not reduce (from {iterCost} to {initCost})");
                }
            }
            DepthFirstSearch<NodeIndex> dfsBack;
            if (useGroups)
            {
                dfsBack = new DepthFirstSearch<EdgeIndex>(ForwardSourcesWithGroups, groupGraph);
            }
            else
            {
                dfsBack = new DepthFirstSearch<EdgeIndex>(ForwardSources, g);
            }
            dfsBack.FinishNode += delegate (NodeIndex node)
            {
                if (debug)
                {
                    string groupString = "";
                    if (useGroups && (node >= groupGraph.firstGroup))
                        groupString = "group ";
                    Debug.WriteLine($"found {groupString}{node}");
                }
                nodesToInit.Add(node);
            };
            foreach (var iterBackEdge in backwardInSchedule)
            {
                if (!iterNodeInFlowGraph.ContainsKey(iterBackEdge) || sourceGroup.Contains(iterNodeInFlowGraph[iterBackEdge]))
                {
                    if (float.IsPositiveInfinity(edgeCostIter[iterBackEdge]))
                        costIsInfinite = true;
                    continue;
                }
                List<Direction> directionOld = direction;
                direction = directionIter;
                NodeIndex target = g.TargetOf(iterBackEdge);
                UpdateCostsInit(edgeCostIter, target, inIteration: true);
                direction = directionOld;
                NodeIndex source = g.SourceOf(iterBackEdge);
                if (edgeCostIter[iterBackEdge] > 0 || initializerDescendants.Contains(source))
                {
                    if (debug)
                    {
                        string txt = DoubleToString(edgeCostIter[iterBackEdge]);
                        if (initializerDescendants.Contains(source))
                            txt += " init";
                        Debug.WriteLine($"searching from {EdgeToString(iterBackEdge)} {txt}");
                    }
                    if (useGroups)
                    {
                        Set<NodeIndex> targetGroups = groupGraph.GetGroupSet(target);
                        dfsBack.SearchFrom(groupGraph.GetLargestGroupExcluding(source, targetGroups));
                    }
                    else
                    {
                        dfsBack.SearchFrom(source);
                    }
                    directionIter[iterBackEdge] = Direction.Forward;
                }
                else if (debug)
                    Debug.WriteLine($"cost[{EdgeToString(iterBackEdge)}] = {edgeCostIter[iterBackEdge]}");
            }

            return costIsInfinite;
        }

        public List<NodeIndex> RepairCombinedSchedule(List<NodeIndex> iterSchedule, List<NodeIndex> schedule)
        {
            if (useRepair2)
            {
                var fresh = new Set<DependencyGraph.TargetIndex>();
                foreach (NodeIndex node in dg.initializedNodes)
                    fresh.Add(dg.getTargetIndex(node));
                var freshCopy = new Set<DependencyGraph.TargetIndex>();
                freshCopy.AddRange(fresh);
                schedule = dg.RepairSchedule2(schedule, freshCopy, dg.initializedNodes);
                // repair may have introduced dead code, so we remove it before rotation
                Set<NodeIndex> usedNodes = new Set<EdgeIndex>();
                usedNodes.AddRange(schedule);
                dg.PruneDeadNodes(iterSchedule, usedNodes);
                schedule = dg.PruneDeadNodes(schedule, usedNodes);
                if (true)
                {
                    // rotate the iter schedule to match the init schedule
                    // FIXME: this may violate grouping
                    dg.CollectFreshNodes(schedule, fresh);
                    freshCopy.Clear();
                    freshCopy.AddRange(fresh);
                    List<NodeIndex> newIterSchedule = dg.RotateSchedule(iterSchedule, freshCopy);
                    newIterSchedule = dg.RepairSchedule2(newIterSchedule, fresh, dg.initializedNodes);
                    newIterSchedule = dg.PruneDeadNodesCyclic(newIterSchedule);
                    if (debug)
                    {
                        Console.WriteLine("new iter schedule:");
                        WriteSchedule(newIterSchedule);
                    }
                    // modify the argument in place
                    iterSchedule.Clear();
                    iterSchedule.AddRange(newIterSchedule);
                }
            }
            else
            {
                // Insert statements to satisfy trigger and fresh constraints.
                // This needs to be done within the init schedule as well as between the init and iter schedule.
                Set<NodeIndex> invalid = new Set<EdgeIndex>();
                Set<DependencyGraph.TargetIndex> stale = new Set<DependencyGraph.TargetIndex>();
                List<int> combinedSchedule = new List<int>();
                while (true)
                {
                    int count = schedule.Count;
                    combinedSchedule.Clear();
                    combinedSchedule.AddRange(schedule);
                    combinedSchedule.AddRange(iterSchedule);
                    // repair the combined schedule, then strip out statements from the iter schedule
                    // must iterate since this may change the position of statements
                    invalid.Clear();
                    invalid.AddRange(dg.initiallyInvalidNodes);
                    stale.Clear();
                    foreach (NodeIndex node in dg.initiallyStaleNodes)
                        stale.Add(dg.getTargetIndex(node));
                    combinedSchedule = dg.RepairSchedule(combinedSchedule, invalid, stale, dg.initializedNodes);
                    schedule.Clear();
                    int pos = iterSchedule.Count - 1;
                    for (int i = combinedSchedule.Count - 1; i >= 0; i--)
                    {
                        if (pos >= 0 && combinedSchedule[i] == iterSchedule[pos])
                            pos--;
                        else
                            schedule.Add(combinedSchedule[i]);
                    }
                    schedule.Reverse();
                    if (schedule.Count == count)
                        break;
                }
            }
            if (debug)
            {
                Debug.WriteLine("init schedule after repair:");
                WriteSchedule(schedule);
            }
            return schedule;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <summary>
        /// Label unknown edges to make the most use of user-provided initializations
        /// </summary>
        /// <param name="initializedNodes"></param>
        /// <param name="initializerChildren">Children of user-initialized nodes</param>
        /// <param name="initializerDescendants">Modified to contain sink nodes reachable from user-inits via forward edges</param>
        /// <remarks>
        /// Algorithm: Find a path from any user-initialized node to any backwardInSchedule edge, where the path
        /// contains no backward edges and at least one unlabeled edge.  Label these edges forward, and repeat until no more paths
        /// can be found.
        /// </remarks>
        protected void LabelEdgesDescendingFromInits(IEnumerable<NodeIndex> initializedNodes, IEnumerable<NodeIndex> initializerChildren, Set<NodeIndex> initializerDescendants)
        {
            while (true)
            {
                var edges = FindUnlabeledEdges(initializedNodes, initializerChildren, initializerDescendants);
                if (edges == null)
                    return;
                foreach (EdgeIndex newForwardEdge in edges)
                {
                    direction[newForwardEdge] = Direction.Forward;
                    if (debug)
                        Debug.WriteLine("user-init forced new forward edge " + EdgeToString(newForwardEdge));
                    newForwardEdges.Push(newForwardEdge);
                    todo.Push(newForwardEdge);
                }
                PropagateConstraints();
            }
        }

        public class StackFrame2
        {
            public EdgeIndex edge;
            public bool visited;
        }

        /// <summary>
        /// Label unknown edges on a path from a user-initialized node to a backwardInSchedule edge
        /// </summary>
        /// <param name="initializedNodes"></param>
        /// <param name="initializerChildren">Children of user-initialized nodes</param>
        /// <param name="initializerDescendants">Modified to contain sink nodes reachable from user-inits via forward edges</param>
        protected ICollection<EdgeIndex> FindUnlabeledEdges(IEnumerable<NodeIndex> initializedNodes, IEnumerable<NodeIndex> initializerChildren, Set<NodeIndex> initializerDescendants)
        {
            // Algorithm:
            // Search through all paths from a user-initialized node to a backwardInSchedule edge,
            // excluding paths with back edges and paths that would create forward cycles.
            // We only care about paths with a distinct set of unlabeled edges, so the backtracking stack
            // only stores unlabeled edges.  From each unlabeled edge on the stack, we search for descendant unlabeled edges
            // to add to the stack.  To speed up the search, nodes are blocked when they are known to not lead to a sink edge.
            Stack<StackFrame2> stack = new Stack<StackFrame2>();
            // a node can be blocked in two ways: 
            // 1. an ancestor of a visited unlabeled edge (tagged with min depth)
            // 2. all children are blocked (tagged with max depth)
            // blockSet[node] is the search depth at which the node is blocked.
            // the node will be unblocked when backtracking above this depth.
            // a depth of -1 means this node will never be unblocked.
            Dictionary<NodeIndex, int> blockedSet = new Dictionary<EdgeIndex, EdgeIndex>();
            foreach (NodeIndex node in initializedNodes)
                blockedSet.Add(node, -1);
            List<Set<NodeIndex>> blockedSetAtDepth = new List<Set<EdgeIndex>>();
            Converter<NodeIndex, IEnumerable<NodeIndex>> forwardUnblockedTargets = node => ForwardTargets(node).Where(node2 => !blockedSet.ContainsKey(node2));
            bool foundBackwardInSchedule = false;
            DepthFirstSearch<NodeIndex> dfsDesc = new DepthFirstSearch<NodeIndex>(forwardUnblockedTargets, g);
            dfsDesc.FinishNode += delegate (NodeIndex node)
                {
                    if (blockedSet.ContainsKey(node))
                        return;
                    //throw new Exception("Internal: descendant is blocked");
                    int maxDepth = -1;
                    foreach (EdgeIndex edge in g.EdgesOutOf(node))
                    {
                        if (backwardInSchedule.Contains(edge))
                        {
                            foundBackwardInSchedule = true;
                            initializerDescendants.Add(node);
                        }
                        if (direction[edge] == Direction.Backward)
                            continue;
                        NodeIndex target = g.TargetOf(edge);
                        int targetDepth;
                        if (blockedSet.TryGetValue(target, out targetDepth))
                        {
                            maxDepth = System.Math.Max(maxDepth, targetDepth);
                        }
                        else
                        {
                            maxDepth = int.MaxValue; // not blocked
                            if (direction[edge] == Direction.Unknown)
                            {
                                stack.Push(new StackFrame2()
                                {
                                    edge = edge
                                });
                            }
                        }
                    }
                    if (maxDepth < int.MaxValue)
                    {
                        // if all children are blocked, then this node is blocked, at their maximum depth
                        if (debug && verbose)
                            Debug.WriteLine("blocking descendant {0} at depth {1}", node, maxDepth);
                        blockedSet[node] = maxDepth;
                        if (maxDepth >= 0)
                            blockedSetAtDepth[maxDepth].Add(node);
                    }
                };
            // from each source node, find all unknown descendants through forward edges
            foreach (NodeIndex source in initializerChildren)
            {
                if (blockedSet.ContainsKey(source))
                    continue;
                // this will push edges onto the search stack and add nodes to the blockedSet
                dfsDesc.SearchFrom(source);
            }
            // ignore any sink edges found by the above search
            foundBackwardInSchedule = false;
            int depth = -1;
            // when searching through ancestors of an unblocked node, the only blocked nodes we can encounter are ancestors of a previous unlabeled edge.
            // in this case, we don't need to search any further.
            Converter<NodeIndex, IEnumerable<NodeIndex>> forwardUnblockedSources = node => ForwardSources(node).Where(node2 => !blockedSet.ContainsKey(node2));
            DepthFirstSearch<NodeIndex> dfsAnc = new DepthFirstSearch<EdgeIndex>(forwardUnblockedSources, g);
            dfsAnc.FinishNode += delegate (NodeIndex node)
                {
                    if (debug && verbose)
                        Debug.WriteLine("blocking ancestor {0} at depth {1}", node, depth);
                    if (blockedSet.ContainsKey(node))
                        throw new Exception("Internal: ancestor is already blocked");
                    blockedSet[node] = depth;
                    blockedSetAtDepth[depth].Add(node);
                };
            while (stack.Count > 0)
            {
                StackFrame2 frame = stack.Peek();
                if (!frame.visited)
                {
                    NodeIndex target = g.TargetOf(frame.edge);
                    if (blockedSet.ContainsKey(target))
                    {
                        stack.Pop();
                    }
                    else
                    {
                        depth++;
                        if (debug && verbose)
                            Debug.WriteLine("depth {0} visiting {1}", depth, EdgeToString(frame.edge));
                        blockedSetAtDepth.Add(new Set<EdgeIndex>());
                        // find all ancestors through forward edges and block them (this prevents creating cycles)
                        dfsAnc.Clear();
                        dfsAnc.SearchFrom(g.SourceOf(frame.edge));
                        // find all descendant unblocked unknown edges and push them on the stack
                        dfsDesc.Clear();
                        dfsDesc.SearchFrom(target);
                        frame.visited = true;
                        if (foundBackwardInSchedule)
                        {
                            // return all unlabeled edges on this path
                            List<EdgeIndex> list = new List<EdgeIndex>();
                            while (stack.Count > 0)
                            {
                                var frame2 = stack.Pop();
                                if (frame2.visited)
                                    list.Add(frame2.edge);
                            }
                            return list;
                        }
                    }
                }
                else
                {
                    // backtracking step
                    // unblock all nodes blocked at this depth
                    foreach (NodeIndex node in blockedSetAtDepth[depth])
                    {
                        blockedSet.Remove(node);
                    }
                    blockedSetAtDepth.RemoveAt(depth);
                    depth--;
                    stack.Pop();
                }
            }
            return null;
        }

        private static T Pop<T>(ICollection<T> list)
        {
            T item = list.First();
            list.Remove(item);
            return item;
        }

        private static float SumSubarray(float[] values, IEnumerable<int> indices)
        {
            float sum = 0f;
            foreach (int index in indices)
            {
                sum += values[index];
            }
            return sum;
        }

        public void AssignLabelsByMinCutWithGroups(Set<EdgeIndex> forcedForwardEdges, Set<EdgeIndex> forcedBackEdges)
        {
            if (debug)
            {
                if (forcedBackEdges.Count > 0)
                {
                    Debug.Write("forcedBackEdges: ");
                    foreach (EdgeIndex edge in forcedBackEdges)
                    {
                        Debug.Write(EdgeToString(edge));
                    }
                    Debug.WriteLine("");
                }
                if (forcedForwardEdges.Count > 0)
                {
                    Debug.Write("forcedForwardEdges: ");
                    foreach (EdgeIndex edge in forcedForwardEdges)
                    {
                        Debug.Write(EdgeToString(edge));
                    }
                    Debug.WriteLine("");
                }
            }
            if (groupGraph == null)
            {
                AssignLabelsByMinCut(forcedForwardEdges, forcedBackEdges);
                return;
            }
            // Algorithm:  partition edges into groups (smallest group shared by endpoints), call MinCut on each edge set, holding other edges fixed
            Set<EdgeIndex> preservedEdges = new Set<EdgeIndex>();
            List<Edge<NodeIndex>> extraEdges = new List<Edge<EdgeIndex>>();
            Set<NodeIndex> allGroups = new Set<EdgeIndex>();
            allGroups.Add(-1);
            for (int group = groupGraph.firstGroup; group < groupGraph.lastGroup; group++)
            {
                allGroups.Add(group);
            }
            foreach (int group in allGroups)
            {
                preservedEdges.Clear();
                Set<NodeIndex> nodesInGroup = new Set<EdgeIndex>();
                // loop all edges
                foreach (NodeIndex source in g.Nodes)
                {
                    var sourceGroups = groupGraph.GetGroupSet(source);
                    if (group == -1 || sourceGroups.Contains(group))
                        nodesInGroup.Add(source);
                    foreach (EdgeIndex edge in g.EdgesOutOf(source))
                    {
                        NodeIndex target = g.TargetOf(edge);
                        NodeIndex sharedGroup = groupGraph.GetSmallestGroup(target, sourceGroups.Contains);
                        if (sharedGroup != group)
                        {
                            preservedEdges.Add(edge);
                        }
                    }
                }
                // groups must be connected in the capacitated graph.
                // therefore if a subgroup of this group has multiple connected components, 
                // we collect a set of extra edges that would make the subgroup connected.
                extraEdges.Clear();
                // maps from subgroups to nodes
                Dictionary<NodeIndex, NodeIndex> nodeInSubgroup = new Dictionary<EdgeIndex, EdgeIndex>();
                var preservedGraph = new DirectedGraphFilter<NodeIndex, EdgeIndex>(g, preservedEdges.Contains);
                DepthFirstSearch<NodeIndex> dfs = new DepthFirstSearch<NodeIndex>(preservedGraph.NeighborsOf, preservedGraph);
                foreach (NodeIndex node in nodesInGroup)
                {
                    if (dfs.IsVisited[node] != VisitState.Unvisited)
                        continue;
                    NodeIndex subgroup = groupGraph.GetLargestGroupExcluding(node, group.Equals);
                    if (subgroup == node)
                        continue;
                    // node is in a subgroup.
                    // skip all other nodes in this connected component
                    dfs.SearchFrom(node);
                    NodeIndex node2;
                    if (nodeInSubgroup.TryGetValue(subgroup, out node2))
                    {
                        // conect to previous connected component
                        extraEdges.Add(new Edge<EdgeIndex>(node2, node));
                    }
                    else
                    {
                        // this is the first connected component
                        nodeInSubgroup.Add(subgroup, node);
                    }
                }
                if (debug)
                    Debug.WriteLine("rotating group {0}", group);
                // in order for this to work, forced edges must truly be forced
                // note this changes the directions of non-forced edges
                AssignLabelsByMinCut(forcedForwardEdges, forcedBackEdges, preservedEdges, extraEdges);
            }
        }

        /// <summary>
        /// Rotate the back edges around each cycle to minimize the number of back edges with Required annotations
        /// </summary>
        /// <remarks>
        /// Algorithm: Construct a flow graph representing all possible rotations and find a min cut.
        /// </remarks>
        public void AssignLabelsByMinCut(ICollection<EdgeIndex> forcedForwardEdges, ICollection<EdgeIndex> forcedBackEdges,
            ICollection<EdgeIndex> preservedEdges = null, IEnumerable<Edge<NodeIndex>> extraEdges = null)
        {
            if (debug && false)
            {
                if (forcedBackEdges.Count > 0)
                {
                    Debug.Write("forcedBackEdges: ");
                    foreach (EdgeIndex edge in forcedBackEdges)
                    {
                        Debug.Write(EdgeToString(edge));
                    }
                    Debug.WriteLine("");
                }
                if (forcedForwardEdges.Count > 0)
                {
                    Debug.Write("forcedForwardEdges: ");
                    foreach (EdgeIndex edge in forcedForwardEdges)
                    {
                        Debug.Write(EdgeToString(edge));
                    }
                    Debug.WriteLine("");
                }
            }
            if (debug && extraEdges != null && extraEdges.Count() > 0)
            {
                Debug.Write("extraEdges: ");
                foreach (var edge in extraEdges)
                {
                    Debug.Write(edge);
                }
                Debug.WriteLine("");
            }
            // repeat until the total capacity does not increase
            while (true)
            {
                // create a flow graph representing all possible rotations of equivalent schedules
                Dictionary<EdgeIndex, EdgeIndex> edgeInG = new Dictionary<EdgeIndex, EdgeIndex>();
                Dictionary<EdgeIndex, EdgeIndex> edgeInGforCost = new Dictionary<EdgeIndex, EdgeIndex>();
                int nodeCount = g.Nodes.Count;
                NodeIndex sourceNode = nodeCount++;
                NodeIndex sinkNode = nodeCount++;
                // if a node is the source of multiple required edges, we create clones to represent whether it will be in the init schedule or not
                // we want the cost to increase by 100 when the node has any backward outgoing edge (not per backward edge)
                Dictionary<NodeIndex, NodeIndex> forwardCloneOfNode = new Dictionary<EdgeIndex, EdgeIndex>();
                Dictionary<NodeIndex, NodeIndex> backwardCloneOfNode = new Dictionary<EdgeIndex, EdgeIndex>();
                foreach (NodeIndex source in g.Nodes)
                {
                    NodeIndex originalSource = originalNode[source];
                    if (dg.initializedNodes.Contains(originalSource))
                        continue;
                    int requiredCount = 0;
                    foreach (EdgeIndex edge in g.EdgesOutOf(source))
                    {
                        if (preservedEdges != null && preservedEdges.Contains(edge))
                            continue;
                        NodeIndex target = g.TargetOf(edge);
                        NodeIndex originalTarget = originalNode[target];
                        EdgeIndex edgeOrig = originalEdge[edge];
                        // TODO: exclude initializedEdges
                        if (dg.isRequired[edgeOrig])
                            requiredCount++;
                    }
                    if (requiredCount > 1 && false)
                    {
                        forwardCloneOfNode[source] = nodeCount++;
                        backwardCloneOfNode[source] = nodeCount++;
                    }
                }
                IndexedGraph network = new IndexedGraph(nodeCount);
                const float infinity = 1000000f;
                const float userInitCost = -5000f;
                const float requiredCost = 100f;
                Func<EdgeIndex, float> edgeCost = delegate (EdgeIndex edge)
                    {
                        if (preservedEdges != null && preservedEdges.Contains(edge))
                            return (direction[edge] == Direction.Forward) ? infinity : -infinity;
                        if (forcedBackEdges.Contains(edge))
                            return userInitCost;
                        float cost = GetEdgeCost(edge);
                        if (forcedForwardEdges.Contains(edge))
                            cost += infinity;
                        return cost;
                    };
                IndexedProperty<EdgeIndex, float> capacity = network.CreateEdgeData<float>();
                foreach (var entry in forwardCloneOfNode)
                {
                    EdgeIndex edge2 = network.AddEdge(entry.Key, entry.Value);
                    capacity[edge2] = requiredCost;
                }
                Set<EdgeIndex> cloneEdges = new Set<EdgeIndex>();
                foreach (EdgeIndex edge in g.Edges)
                {
                    NodeIndex source = g.SourceOf(edge);
                    NodeIndex target = g.TargetOf(edge);
                    float cost = edgeCost(edge);
                    if (direction[edge] == Direction.Forward)
                    {
                        EdgeIndex edge2 = network.AddEdge(source, target);
                        edgeInG[edge2] = edge;
                        if (cost >= 0f)
                        {
                            if (cost == requiredCost && forwardCloneOfNode.ContainsKey(source))
                            {
                                // the forwardClone has infinite capacity edges to the children, and no reverse capacity
                                source = forwardCloneOfNode[source];
                                EdgeIndex edgeClone = network.AddEdge(source, target);
                                capacity[edgeClone] = infinity;
                                cloneEdges.Add(edgeClone);
                                cost = 0f;  // edge must still exist since it has reverse capacity
                            }
                            capacity[edge2] = cost;
                        }
                        else
                        {
                            capacity[edge2] = 0f; // edge must still exist since it has reverse capacity
                            EdgeIndex edgeS = network.AddEdge(sourceNode, source);
                            capacity[edgeS] = -cost;
                            EdgeIndex edgeT = network.AddEdge(target, sinkNode);
                            capacity[edgeT] = -cost;
                            edgeInGforCost[edgeS] = edge;
                            edgeInGforCost[edgeT] = edge;
                        }
                    }
                    else if (direction[edge] == Direction.Backward)
                    {
                        EdgeIndex edge2 = network.AddEdge(target, source);
                        edgeInG[edge2] = edge;
                        if (cost <= 0f)
                        {
                            capacity[edge2] = -cost;
                        }
                        else
                        {
                            capacity[edge2] = 0f; // edge must still exist since it has reverse capacity
                            if (cost == requiredCost && backwardCloneOfNode.ContainsKey(source))
                            {
                                // the backwardClone has infinite capacity edges to the targets, and no reverse capacity
                                NodeIndex clone = backwardCloneOfNode[source];
                                EdgeIndex edgeClone = network.AddEdge(clone, target);
                                capacity[edgeClone] = infinity;
                                cloneEdges.Add(edgeClone);
                                if (!network.ContainsEdge(sourceNode, clone))
                                {
                                    // these edges must only be created once, and only if the cloned node has backward outgoing edges
                                    EdgeIndex edgeS = network.AddEdge(sourceNode, clone);
                                    capacity[edgeS] = requiredCost;
                                    EdgeIndex edgeT = network.AddEdge(source, sinkNode);
                                    capacity[edgeT] = requiredCost;
                                    edgeInGforCost[edgeS] = edge;
                                    edgeInGforCost[edgeT] = edge;
                                }
                            }
                            else
                            {
                                EdgeIndex edgeS = network.AddEdge(sourceNode, target);
                                capacity[edgeS] = cost;
                                EdgeIndex edgeT = network.AddEdge(source, sinkNode);
                                capacity[edgeT] = cost;
                                edgeInGforCost[edgeS] = edge;
                                edgeInGforCost[edgeT] = edge;
                            }
                        }
                    }
                    else
                        throw new Exception("unknown direction for " + EdgeToString(edge));
                }
                if (extraEdges != null)
                {
                    foreach (var edge in extraEdges)
                    {
                        EdgeIndex edge2 = network.AddEdge(edge.Source, edge.Target);
                        capacity[edge2] = infinity;
                        edgeInGforCost[edge2] = 0;
                    }
                }
                float sourceTotal = 0f;
                foreach (EdgeIndex edge2 in network.EdgesOutOf(sourceNode))
                {
                    sourceTotal += capacity[edge2];
                    if (showCapacityBreakdown && capacity[edge2] > 0f)
                    {
                        EdgeIndex edge;
                        if (!edgeInG.TryGetValue(edge2, out edge))
                        {
                            edge = edgeInGforCost[edge2];
                        }
                        Debug.WriteLine($"capacity={capacity[edge2]} edge={EdgeToString(edge)} edgeIndex={edge}");
                    }
                }
                if (debug)
                    Debug.WriteLine("capacity of edges out of source = {0}", sourceTotal);

                // compute the min cut
                MinCut<NodeIndex, EdgeIndex> mc = new MinCut<EdgeIndex, EdgeIndex>(network, e => capacity[e]);
                // infinite reverse capacity ensures that there are no edges from the sink set to the source set.
                mc.reverseCapacity = edge => cloneEdges.Contains(edge) ? 0f : float.PositiveInfinity;
                mc.Sources.Add(sourceNode);
                mc.Sinks.Add(sinkNode);
                Set<NodeIndex> sourceGroup = mc.GetSourceGroup();

                var originalDirection = CloneDirections();

                // flip the direction of cut edges
                float finalCapacity = 0f;
                StringBuilder capacityBreakdown = new StringBuilder();
                foreach (EdgeIndex edge2 in network.Edges)
                {
                    NodeIndex source = network.SourceOf(edge2);
                    NodeIndex target = network.TargetOf(edge2);
                    bool isCut = sourceGroup.Contains(source) && !sourceGroup.Contains(target);
                    if (isCut)
                    {
                        EdgeIndex edge;
                        if (edgeInG.TryGetValue(edge2, out edge))
                        {
                            // cutting an edge flips its direction
                            direction[edge] = (originalDirection[edge] == Direction.Forward) ? Direction.Backward : Direction.Forward;
                        }
                        else
                        {
                            edge = edgeInGforCost[edge2];
                        }
                        finalCapacity += capacity[edge2];
                        if (showCapacityBreakdown && capacity[edge2] > 0f)
                        {
                            Func<NodeIndex, string> capacityNodeToString = delegate (NodeIndex node)
                            {
                                if (node == sourceNode)
                                    return "source";
                                else if (node == sinkNode)
                                    return "sink";
                                else
                                    return node.ToString();
                            };
                            capacityBreakdown.AppendLine($"{capacity[edge2]} {EdgeToString(edge)}");
                        }
                    }
                }
                if (finalCapacity >= infinity)
                {
                    string s = "Cannot satisfy scheduling constraints";
                    if (debug)
                        Debug.WriteLine(s);
                    else
                        throw new Exception(s);
                }
                if (debug)
                {
                    // check that forcedBackEdges actually became back edges
                    foreach (EdgeIndex edge in forcedBackEdges)
                    {
                        if (preservedEdges != null && preservedEdges.Contains(edge))
                            continue;
                        if (direction[edge] != Direction.Backward)
                        {
                            Debug.WriteLine($"{EdgeToString(edge)} was not forced backward");
                        }
                    }
                }
                foreach (EdgeIndex edge in forcedForwardEdges)
                {
                    if (preservedEdges != null && preservedEdges.Contains(edge))
                        continue;
                    if (direction[edge] != Direction.Forward)
                    {
                        string s = $"{EdgeToString(edge)} was not forced forward";
                        if (debug)
                            Debug.WriteLine(s);
                        else
                            throw new Exception(s);
                    }
                }
                if (debug)
                {
                    Debug.WriteLine("final capacity = {0}", finalCapacity);
                    if (showCapacityBreakdown)
                    {
                        Debug.WriteLine(capacityBreakdown.ToString());
                    }
                }
                if (debug && showMinCut && g.Nodes.Count > 1)
                {
                    DrawMinCut(edgeInG, sourceNode, sinkNode, forwardCloneOfNode, backwardCloneOfNode, network, capacity, sourceGroup, originalDirection);
                }
                if (finalCapacity >= sourceTotal)
                    break;
            }
        }

        // This must be a separate method to avoid dependence on GLEE in the normal case.
        private void DrawMinCut(Dictionary<int, int> edgeInG, int sourceNode, int sinkNode, Dictionary<int, int> forwardCloneOfNode, Dictionary<int, int> backwardCloneOfNode, IndexedGraph g2, IndexedProperty<int, float> capacity, Set<int> sourceGroup, IndexedProperty<int, Direction> originalDirection)
        {
            if (InferenceEngine.Visualizer?.DependencyGraphVisualizer != null)
            {
                // display the graph and the minimum cut
                Func<NodeIndex, string> nodeName = delegate (NodeIndex node)
                {
                    if (node == sourceNode)
                        return "s";
                    else if (node == sinkNode)
                        return "t";
                    else if (node < g.Nodes.Count)
                        return NodeName(node);
                    else if (forwardCloneOfNode.ContainsValue(node))
                        return forwardCloneOfNode.First(entry => entry.Value == node).Key + "f";
                    else if (backwardCloneOfNode.ContainsValue(node))
                        return backwardCloneOfNode.First(entry => entry.Value == node).Key + "b";
                    else
                        return "?";
                };
                Predicate<EdgeIndex> isCut = delegate (EdgeIndex edge)
                {
                    NodeIndex source = g2.SourceOf(edge);
                    NodeIndex target = g2.TargetOf(edge);
                    return sourceGroup.Contains(source) && !sourceGroup.Contains(target);
                };
                Predicate<EdgeIndex> isBackInG0 = edge => direction[edgeInG[edge]] == Direction.Backward;
                Predicate<EdgeIndex> isBackInG = edge => edgeInG.ContainsKey(edge) ? isBackInG0(edge) : false;
                Predicate<EdgeIndex> isOrigBackInG0 = edge => originalDirection[edgeInG[edge]] == Direction.Backward;
                Predicate<EdgeIndex> isOrigBackInG = edge => edgeInG.ContainsKey(edge) ? isOrigBackInG0(edge) : false;
                var edgeStyles = new EdgeStylePredicate[] {
                            new EdgeStylePredicate("cut", isCut, EdgeStyle.Bold),
                            new EdgeStylePredicate("origBack", isOrigBackInG, EdgeStyle.Dashed),
                            new EdgeStylePredicate("back", isBackInG, EdgeStyle.Back)
                        };
                InferenceEngine.Visualizer.DependencyGraphVisualizer.VisualizeDependencyGraph(g2, edgeStyles,
                    node => nodeName(node) + (sourceGroup.Contains(node) ? "" : "^"),
                    edge => DoubleToString(capacity[edge]));
            }
        }

        /// <summary>
        /// Get the heuristic cost of making an edge backward in the iteration schedule
        /// </summary>
        /// <param name="edge"></param>
        /// <returns></returns>
        private float GetEdgeCost(EdgeIndex edge)
        {
            float scale = 1;
            NodeIndex source = originalNode[g.SourceOf(edge)];
            NodeIndex target = originalNode[g.TargetOf(edge)];
            if (source == -1 || target == -1)
                return 0.1f * scale; // edge doesn't exist in the original graph
            EdgeIndex edgeOrig = originalEdge[edge];
            if (edgeOrig == -1)
                return 0.1f * scale;
            if (dg.initializedNodes.Contains(source) || dg.initializedEdges.Contains(edgeOrig))
                return 0.001f;
            if (!dg.hasNonUniformInitializer.Contains(source))
            {
                if (dg.isRequired[edgeOrig])
                    return 100f * scale;
                int smallestEdgeSetSize = int.MaxValue;
                foreach (var edgeSet in GetRequiredEdgeSets(g.TargetOf(edge), true, false))
                {
                    if (edgeSet.Contains(edge))
                        smallestEdgeSetSize = System.Math.Min(smallestEdgeSetSize, edgeSet.Count);
                }
                if (smallestEdgeSetSize != int.MaxValue)
                    return 100f / smallestEdgeSetSize * scale;
            }
            if (!dg.noInit[edgeOrig])
                return 1f * scale;
            return 0.0001f;
        }

        private float[] GetEdgeCosts()
        {
            float[] edgeCost = new float[g.EdgeCount()];
            NodeIndex source = -1, target;
            EdgeIndex edge = -1;
            DistanceSearch<NodeIndex> ds = new DistanceSearch<NodeIndex>(g);
            ds.SetDistance += delegate (NodeIndex node, int distance)
            {
                if (node == source)
                    edgeCost[edge] = 1f / (distance + 1);
            };
            for (edge = 0; edge < edgeCost.Length; edge++)
            {
                source = g.SourceOf(edge);
                target = g.TargetOf(edge);
                if (source == target)
                    continue;
                ds.SearchFrom(target);
            }
            return edgeCost;
        }

        private float[] GetEdgeCosts2()
        {
            float[] edgeCost = new float[g.EdgeCount()];
            NodeIndex source = -1, target;
            EdgeIndex edge = -1;
            DistanceSearch<NodeIndex> dsForward = new DistanceSearch<EdgeIndex>(g);
            IndexedProperty<NodeIndex, int> distanceForward = g.CreateNodeData<int>();
            dsForward.SetDistance += delegate (NodeIndex node, int dist)
            {
                distanceForward[node] = dist;
            };
            DistanceSearch<NodeIndex> dsBackward = new DistanceSearch<EdgeIndex>(g.SourcesOf, g);
            IndexedProperty<NodeIndex, int> distanceBackward = g.CreateNodeData<int>();
            dsBackward.SetDistance += delegate (NodeIndex node, int dist)
            {
                distanceBackward[node] = dist;
            };
            for (edge = 0; edge < edgeCost.Length; edge++)
            {
                source = g.SourceOf(edge);
                target = g.TargetOf(edge);
                if (source == target)
                    continue;
                dsForward.SearchFrom(target);
                dsBackward.SearchFrom(source);
                foreach (NodeIndex node in g.Nodes)
                {
                    int length = 1 + distanceForward[node] + distanceBackward[node];
                    edgeCost[edge] += 1f / (length * length);
                }
            }
            return edgeCost;
        }

        /// <summary>
        /// Compute the cost of reversing each edge, before any edges have been labeled.  For each cycle that the edge is on, we add 1/(length of cycle).
        /// </summary>
        /// <returns></returns>
        private float[] GetEdgeCosts3()
        {
            float[] edgeCost = new float[g.EdgeCount()];
            for (EdgeIndex edge = 0; edge < edgeCost.Length; edge++)
            {
                NodeIndex source = g.SourceOf(edge);
                NodeIndex target = g.TargetOf(edge);
                if (source == target)
                    continue;
                NodeIndex sharedGroup = -1;
                if (groupGraph != null)
                {
                    Set<NodeIndex> sourceGroups = groupGraph.GetGroupSet(source);
                    sharedGroup = groupGraph.GetSmallestGroup(target, sourceGroups.Contains);
                    // lift source and target to the level of sharedGroup (even if sharedGroup == -1)
                    source = groupGraph.GetLargestGroupInsideGroup(source, sharedGroup);
                    target = groupGraph.GetLargestGroupInsideGroup(target, sharedGroup);
                }
                Assert.IsTrue(source != target);
                var edgeLength = GetEdgeLength(edge);
                foreach (NodeIndex node in g.Nodes)
                {
                    NodeIndex node2 = node;
                    if (groupGraph != null)
                        node2 = groupGraph.GetLargestGroupInsideGroup(node, sharedGroup);
                    if (node2 == -1)
                        continue;
                    if (distance[target][node2] == infiniteDistance ||
                        distance[node2][source] == infiniteDistance)
                        continue;
                    // compute the length of the shortest cycle containing edge and node
                    int length = edgeLength + distance[target][node2] + distance[node2][source];
                    // we want to add 1/length per cycle.  
                    // since we are adding per node and the cycle has 'length' nodes, we divide by an extra factor of 'length'.
                    // divide by 100 so that it can only be used to break ties (when the number of back edges is the same)
                    edgeCost[edge] += 1f / (length * length) / 100;

                    if (verbose && edgeCost.Length < 100)
                        Debug.WriteLine($"{EdgeToString(edge)} length={length} distance[{target} to {node2}]={distance[target][node2]} distance[{node2} to {source}]={distance[node2][source]}");
                }
                if (edgeCost[edge] == 0f && NoInitEdgesAreInfinite && !IsNoInit(edge) && false) // TODO
                    edgeCost[edge] = 100f;
            }
            return edgeCost;
        }

        /// <summary>
        /// Unused
        /// </summary>
        /// <returns></returns>
        private float[] GetEdgeCosts4()
        {
            float[] edgeCost = new float[g.EdgeCount()];
            for (EdgeIndex edge = 0; edge < edgeCost.Length; edge++)
            {
                NodeIndex source = g.SourceOf(edge);
                NodeIndex target = g.TargetOf(edge);
                foreach (EdgeIndex edge2 in g.Edges)
                {
                    NodeIndex source2 = g.SourceOf(edge2);
                    NodeIndex target2 = g.TargetOf(edge2);
                    if (distance[target][source2] == infiniteDistance ||
                        distance[target2][source] == infiniteDistance)
                        continue;
                    // compute the length of the shortest cycle containing edge and edge2
                    int length = 2 + distance[target][source2] + distance[target2][source];
                    edgeCost[edge2] += 1f / (length * length);
                }
            }
            return edgeCost;
        }

        private const int infiniteDistance = int.MaxValue;

        private void ComputeDistances()
        {
            if (groupGraph != null)
            {
                ComputeDistancesWithGroups3();
                return;
            }
            distance = new int[g.Nodes.Count][];
            int[] distanceFromSource = null;
            DistanceSearch<NodeIndex> dsForward = new DistanceSearch<EdgeIndex>(g);
            dsForward.SetDistance += delegate (NodeIndex target, int dist)
            {
                distanceFromSource[target] = dist;
            };
            foreach (NodeIndex source in g.Nodes)
            {
                distanceFromSource = Util.ArrayInit(g.Nodes.Count, i => infiniteDistance);
                distance[source] = distanceFromSource;
                dsForward.SearchFrom(source);
                Assert.IsTrue(distanceFromSource[source] == 0);
            }
        }

        private void ComputeDistancesWithGroups()
        {
            distance = new int[groupGraph.lastGroup][];
            int[] distanceFromSource = null;
            DistanceSearch<NodeIndex> dsForward = new DistanceSearch<EdgeIndex>(TargetsInSameGroup, groupGraph);
            dsForward.SetDistance += delegate (NodeIndex target, int dist)
            {
                distanceFromSource[target] = dist;
            };
            for (NodeIndex source = 0; source < groupGraph.lastGroup; source++)
            {
                distanceFromSource = Util.ArrayInit(groupGraph.lastGroup, i => infiniteDistance);
                distance[source] = distanceFromSource;
                dsForward.SearchFrom(source);
                Assert.IsTrue(distanceFromSource[source] == 0);
            }
        }

        private int GetEdgeLength(EdgeIndex edge)
        {
            bool useEdgeLengths = true;
            if (!useEdgeLengths)
                return 1;
            if (IsRequired(edge))
            {
                return 1;
            }
            else if (IsNoInit(edge))
            {
                return 1000;
            }
            else if (false && IsRequired(edge, includeAny: true)) // TODO: enable this
            {
                // This case is needed for SumForwardBackwardTest2
                // It must follow the IsNoInit case for TrueSkillChainTest3
                return 2;
            }
            else
            {
                return 2;
            }
        }

        private void ComputeDistancesWithGroups2()
        {
            distance = new int[groupGraph.lastGroup][];
            int[] distanceFromSource = null;
            // because we're using BFS, this does not work with edge lengths
            BreadthFirstSearch<NodeIndex> bfs = new BreadthFirstSearch<NodeIndex>(TargetsInSameGroup, groupGraph);
            bfs.TreeEdge += delegate (Edge<NodeIndex> edge)
            {
                distanceFromSource[edge.Target] = distanceFromSource[edge.Source] + 1;
            };
            for (NodeIndex source = 0; source < groupGraph.lastGroup; source++)
            {
                distanceFromSource = Util.ArrayInit(groupGraph.lastGroup, i => infiniteDistance);
                distanceFromSource[source] = 0;
                distance[source] = distanceFromSource;
                bfs.Clear();
                bfs.SearchFrom(source);
                Assert.IsTrue(distanceFromSource[source] == 0);
                //Trace.WriteLine(source.ToString() + ": " + StringUtil.CollectionToString(distanceFromSource, " "));
            }
        }

        private void ComputeDistancesWithGroups3()
        {
            distance = new int[groupGraph.lastGroup][];
            int[] distanceFromSource = null;
            for (NodeIndex source = 0; source < groupGraph.lastGroup; source++)
            {
                distanceFromSource = Util.ArrayInit(groupGraph.lastGroup, i => infiniteDistance);
                distanceFromSource[source] = 0;
                distance[source] = distanceFromSource;
                var isVisited = groupGraph.CreateNodeData<bool>(false);
                while (true)
                {
                    // Find the unvisited node with the smallest distance from source
                    int minDist = infiniteDistance;
                    NodeIndex minNode = -1;
                    for (NodeIndex node = 0; node < groupGraph.lastGroup; node++)
                    {
                        if (!isVisited[node] && distanceFromSource[node] < minDist)
                        {
                            minDist = distanceFromSource[node];
                            minNode = node;
                        }
                    }
                    if (minNode == -1)
                        break;
                    isVisited[minNode] = true;
                    NodeIndex group = groupGraph.groupOf[minNode];
                    foreach (var edge in groupGraph.EdgesOutOf(minNode))
                    {
                        NodeIndex target = groupGraph.TargetOf(edge);
                        if (isVisited[target])
                            continue;
                        target = groupGraph.GetLargestGroupInsideGroup(target, group);
                        if (target == -1)
                            continue;
                        int dist = minDist + GetEdgeLength(edge);
                        if (dist < distanceFromSource[target])
                            distanceFromSource[target] = dist;
                    }
                }
                //Trace.WriteLine(source.ToString() + ": " + StringUtil.CollectionToString(distanceFromSource, " "));
            }
        }

        // as we traverse the graph, we will always stay at the highest level of abstraction (the largest group) that we can.
        private IEnumerable<NodeIndex> TargetsInSameGroup(NodeIndex node)
        {
            NodeIndex group = groupGraph.groupOf[node];
            foreach (EdgeIndex edge in groupGraph.EdgesOutOf(node))
            {
                // do not traverse NoInit edges
                if (NoInitEdgesAreInfinite && IsNoInit(edge))
                    continue;
                NodeIndex target = g.TargetOf(edge);
                if (target != node)
                {
                    target = groupGraph.GetLargestGroupInsideGroup(target, group);
                    if (target != -1)
                        yield return target;
                }
            }
        }

        /// <summary>
        /// Find the cycle whose cheapest unlabeled edge has the highest cost among all cycles, and return that edge.
        /// </summary>
        /// <param name="edgeCost"></param>
        /// <param name="forcedForwardEdges"></param>
        /// <returns></returns>
        public int FindNewBackEdge(float[] edgeCost, IReadOnlyCollection<EdgeIndex> forcedForwardEdges)
        {
            // Algorithm: loop unlabeled edges in order of decreasing reversal cost.  If the edge can be labeled as forward without creating a cycle, do so.
            // Otherwise, this must be the cheapest unlabeled edge on the cycle (since it was discovered last), so return the edge.
            EdgeIndex[] sortedEdges = new EdgeIndex[g.EdgeCount()];
            for (int edge = 0; edge < sortedEdges.Length; edge++)
            {
                sortedEdges[edge] = edge;
            }
            //sortedEdges = Shuffle(sortedEdges);
            // sort by decreasing cost
            Array.Sort(sortedEdges, (a, b) => edgeCost[b].CompareTo(edgeCost[a]));
            NodeIndex source = -1;
            bool reachable = false;
            BreadthFirstSearch<NodeIndex> bfs;
            Set<NodeIndex> groups = new Set<EdgeIndex>();
            if (groupGraph == null)
            {
                bfs = new BreadthFirstSearch<EdgeIndex>(BackwardSourcesAndForwardTargets, g);
                bfs.DiscoverNode += delegate (NodeIndex node)
                    {
                        if (node == source)
                        {
                            reachable = true;
                            bfs.Stop();
                        }
                    };
            }
            else
            {
                bfs = new BreadthFirstSearch<EdgeIndex>(BackwardSourcesAndForwardTargetsWithGroups, groupGraph);
                bfs.DiscoverNode += delegate (NodeIndex node)
                {
                    // note that we are testing whether node is equal to one of the groups, not membership in a group
                    if (groups.Contains(node))
                    {
                        reachable = true;
                        bfs.Stop();
                    }
                };
            }
            DepthFirstSearch<NodeIndex> dfs = null;
            List<NodeIndex> descendantList = new List<NodeIndex>();
            if (useExperimentalSerialSchedules && dfs == null)
            {
                dfs = new DepthFirstSearch<NodeIndex>(BackwardSourcesAndForwardTargets, g);
                dfs.FinishNode += delegate (NodeIndex node)
                {
                    // children will be added before parents
                    descendantList.Add(node);
                };
            }

            // edges whose cost is higher than the most expensive unlabeled edge will never satisfy the condition,
            // so (for efficiency) we remove them from the list of edges to check.
            List<EdgeIndex> sortedEdges2 = new List<EdgeIndex>();
            List<EdgeIndex> unlabeledEdges = null;
            if (debug)
                unlabeledEdges = new List<EdgeIndex>();
            bool foundUnlabeled = false;
            foreach (EdgeIndex edge in sortedEdges)
            {
                if (direction[edge] == Direction.Backward)
                    continue;
                if (foundUnlabeled && !forcedForwardEdges.Contains(edge))
                {
                    if (debug)
                        unlabeledEdges.Add(edge);
                    direction[edge] = Direction.Unknown;
                    sortedEdges2.Add(edge);
                }
                else if (direction[edge] == Direction.Unknown)
                {
                    foundUnlabeled = true;
                    sortedEdges2.Add(edge);
                }
            }
            if (debug)
                Debug.WriteLine($"unlabeling {StringUtil.CollectionToString(unlabeledEdges.Select(EdgeToString), "")}");
            foreach (EdgeIndex edge in sortedEdges2)
            {
                source = g.SourceOf(edge);
                NodeIndex target = g.TargetOf(edge);
                // is there a forward path from target to source?
                if (useExperimentalSerialSchedules)
                {
                    descendantList.Clear();
                    dfs.Clear();
                    dfs.SearchFrom(target);
                    // reversing the list ensures that a node is processed before its children.
                    descendantList.Reverse();
                    descendantOffset.Clear();
                    EdgeIndex edgeOrig = originalEdge[edge];
                    IOffsetInfo offsetInfo;
                    if (dg.OffsetIndices.TryGetValue(edgeOrig, out offsetInfo))
                    {
                        descendantOffset.Add(target, new OffsetBoundCollection(offsetInfo, this.loopVarsOfNode[target]));
                    }
                    else
                    {
                        descendantOffset.Add(target, new OffsetBoundCollection(this.loopVarsOfNode[target]));
                    }
                    foreach (var descendant in descendantList)
                    {
                        if (descendant == target)
                            continue;
                        PropagateOffsetBounds(descendant, false, descendantOffset.ContainsKey, descendantOffset);
                    }
                    if (verbose && showAncestors)
                    {
                        Trace.Write($"edge {EdgeToString(edge)} descendants: ");
                        foreach (var descendant in descendantList)
                        {
                            Trace.Write(descendant);
                            Trace.Write(" ");
                            OffsetBoundCollection obc;
                            if (descendantOffset.TryGetValue(descendant, out obc))
                            {
                                Trace.Write(obc);
                                Trace.Write(" ");
                            }
                        }
                        Trace.WriteLine("");
                    }
                    OffsetBoundCollection sourceBounds;
                    if (!descendantOffset.TryGetValue(source, out sourceBounds))
                    {
                        reachable = false;
                    }
                    else
                    {
                        reachable = !sourceBounds.ContainsNonZero();
                    }
                }
                else
                {
                    reachable = false;
                    bfs.Clear();
                    if (groupGraph != null)
                    {
                        groups = groupGraph.GetGroupSet(source);
                        groups.Add(source);
                        NodeIndex targetGroup = groupGraph.GetLargestGroupExcluding(target, groups);
                        // bfs will never add a group containing targetGroup, so this works even if source and target have a common group.
                        bfs.SearchFrom(targetGroup);
                    }
                    else
                    {
                        bfs.SearchFrom(target);
                    }
                }
                if (reachable)
                {
                    if (debug && groupGraph != null)
                    {
                        NodeIndex targetGroup = groupGraph.GetLargestGroupExcluding(target, groups);
                        var path = GetAnyPath(targetGroup, groups);
                        Debug.WriteLine("cycle:");
                        foreach (var node in path)
                        {
                            Debug.WriteLine(NodeToString(node));
                        }
                    }
                    return edge;
                }
                else
                {
                    if (debug)
                        Debug.WriteLine($"adding forward edge {edgeCost[edge]} {EdgeToString(edge)}");
                    direction[edge] = Direction.Forward;
                }
            }
            // all edges will be labeled
            return -1;
        }

        private List<NodeIndex> GetAnyPath(NodeIndex source, ICollection<NodeIndex> groups)
        {
            List<NodeIndex> path = new List<NodeIndex>();
            var dfs = new DepthFirstSearch<EdgeIndex>(BackwardSourcesAndForwardTargetsWithGroups, groupGraph);
            dfs.DiscoverNode += delegate (NodeIndex node)
            {
                // note that we are testing whether node is equal to one of the groups, not membership in a group
                if (groups.Contains(node))
                {
                    dfs.ForEachStackNode(delegate (NodeIndex node2)
                    {
                        path.Add(node2);
                    });
                    path.Reverse();
                }
            };
            dfs.SearchFrom(source);
            return path;
        }

        private List<NodeIndex> GetAnyPath(NodeIndex source, NodeIndex target)
        {
            List<NodeIndex> path = new List<NodeIndex>();
            var dfs = new DepthFirstSearch<EdgeIndex>(BackwardSourcesAndForwardTargets, g);
            dfs.DiscoverNode += delegate (NodeIndex node)
            {
                if (node == target)
                {
                    dfs.ForEachStackNode(delegate (NodeIndex node2)
                    {
                        path.Add(node2);
                    });
                    path.Reverse();
                }
            };
            dfs.SearchFrom(source);
            return path;
        }

        private void WriteEdgeCosts(float[] edgeCost)
        {
            Trace.WriteLine("edgeCosts:");
            foreach (EdgeIndex edge in g.Edges)
            {
                string noInitText = IsNoInit(edge) ? "NoInit" : "";
                Trace.WriteLine($"{EdgeToString(edge)} {edgeCost[edge]} {direction[edge]} {noInitText}");
            }
        }

        private void UpdateCosts(float[] edgeCost)
        {
            NodeIndex source = -1, target = -1;
            DistanceSearch<NodeIndex> dsForward = new DistanceSearch<EdgeIndex>(g);
            IndexedProperty<NodeIndex, int> distanceForward = g.CreateNodeData<int>();
            dsForward.SetDistance += delegate (NodeIndex node, int dist)
            {
                distanceForward[node] = dist;
            };
            DistanceSearch<NodeIndex> dsBackward = new DistanceSearch<EdgeIndex>(g.SourcesOf, g);
            IndexedProperty<NodeIndex, int> distanceBackward = g.CreateNodeData<int>();
            dsBackward.SetDistance += delegate (NodeIndex node, int dist)
            {
                distanceBackward[node] = dist;
            };
            while (newBackEdges.Count > 0)
            {
                EdgeIndex edge = newBackEdges.Pop();
                source = g.SourceOf(edge);
                target = g.TargetOf(edge);
                if (source == target)
                    continue;
                distanceForward.Clear();
                distanceBackward.Clear();
                dsForward.SearchFrom(target);
                dsBackward.SearchFrom(source);
                foreach (EdgeIndex edge2 in g.Edges)
                {
                    // compute the length of the shortest cycle containing edge and edge2
                    int length = 2 + distanceForward[g.SourceOf(edge2)] + distanceBackward[g.TargetOf(edge2)];
                    // assume one back edge (though there may be more than one)
                    int numBack = 1;
                    float cost = (float)(numBack + 1) / length;
                    if (cost > edgeCost[edge2])
                        edgeCost[edge2] = cost;
                }
            }
        }

        private void UpdateCosts2(float[] edgeCost)
        {
            NodeIndex source = -1, target = -1;
            DistanceSearch<NodeIndex> dsForward = new DistanceSearch<EdgeIndex>(g);
            IndexedProperty<NodeIndex, int> distanceForward = g.CreateNodeData<int>();
            dsForward.SetDistance += delegate (NodeIndex node, int dist)
            {
                distanceForward[node] = dist;
            };
            DistanceSearch<NodeIndex> dsBackward = new DistanceSearch<EdgeIndex>(g.SourcesOf, g);
            IndexedProperty<NodeIndex, int> distanceBackward = g.CreateNodeData<int>();
            dsBackward.SetDistance += delegate (NodeIndex node, int dist)
            {
                distanceBackward[node] = dist;
            };
            while (newBackEdges.Count > 0)
            {
                EdgeIndex edge = newBackEdges.Pop();
                source = g.SourceOf(edge);
                target = g.TargetOf(edge);
                if (source == target)
                    continue;
                distanceForward.Clear();
                distanceBackward.Clear();
                dsForward.SearchFrom(target);
                dsBackward.SearchFrom(source);
                foreach (EdgeIndex edge2 in g.Edges)
                {
                    // compute the length of the shortest cycle containing edge and edge2
                    int length = 2 + distanceForward[g.SourceOf(edge2)] + distanceBackward[g.TargetOf(edge2)];
                    edgeCost[edge2] += 1f / length;
                }
            }
        }

        private void UpdateCosts3(float[] edgeCost)
        {
            while (newBackEdges.Count > 0)
            {
                EdgeIndex edge = newBackEdges.Pop();
                NodeIndex source = g.SourceOf(edge);
                NodeIndex target = g.TargetOf(edge);
                if (source == target)
                    continue;
                NodeIndex sharedGroup = -1;
                if (groupGraph != null)
                {
                    Set<NodeIndex> sourceGroups = groupGraph.GetGroupSet(source);
                    sharedGroup = groupGraph.GetSmallestGroup(target, sourceGroups.Contains);
                    // lift source and target to the level of sharedGroup (even if sharedGroup == -1)
                    source = groupGraph.GetLargestGroupInsideGroup(source, sharedGroup);
                    target = groupGraph.GetLargestGroupInsideGroup(target, sharedGroup);
                    Assert.IsTrue(source != target);
                }
                var edgeLength = GetEdgeLength(edge);
                foreach (EdgeIndex edge2 in g.Edges)
                {
                    NodeIndex source2 = g.SourceOf(edge2);
                    NodeIndex target2 = g.TargetOf(edge2);
                    if (groupGraph != null)
                    {
                        // lift source2 and target2 to the level of sharedGroup (even if sharedGroup == -1)
                        source2 = groupGraph.GetLargestGroupInsideGroup(source2, sharedGroup);
                        target2 = groupGraph.GetLargestGroupInsideGroup(target2, sharedGroup);
                    }
                    if (source2 == target2 || source2 == -1 || target2 == -1)
                        continue;
                    if (distance[target][source2] == infiniteDistance ||
                        distance[target2][source] == infiniteDistance)
                        continue;
                    // compute the length of the shortest cycle containing edge and edge2
                    var edgeLength2 = GetEdgeLength(edge2);
                    int length = edgeLength + edgeLength2 + distance[target][source2] + distance[target2][source];
                    edgeCost[edge2] += 1f / length;

                    if (verbose)
                        Debug.WriteLine($"{EdgeToString(edge)} increased cost of {EdgeToString(edge2)} to {edgeCost[edge2]} (shortest cycle length={length})");
                }
            }
        }

        // increase the cost of reversing edges on offset cycles, since these will correspond to many more reversed edges in the unrolled graph.
        private void UpdateCostsFromOffsetEdges(float[] edgeCost)
        {
            Dictionary<int, IndexedGraph> offsetEdgesByGroup = GetOffsetEdgesByGroup();
            // for each group, find all edges on a path from offset target to source
            NodeIndex group = default(NodeIndex); // modified prior to each search
            IndexedGraph deletedGraph = null; // modified prior to each search
            Set<EdgeIndex> edgesOnCycle = new Set<EdgeIndex>();
            Set<NodeIndex> nodesOnCycle = new Set<EdgeIndex>();
            DepthFirstSearch<NodeIndex> dfs = new DepthFirstSearch<NodeIndex>(node => TargetsInGroup(node, group, deletedGraph), g);
            void finishEdge(Edge<NodeIndex> edge)
            {
                if (nodesOnCycle.Contains(edge.Target))
                {
                    nodesOnCycle.Add(edge.Source);
                    if (g.TryGetEdge(edge.Source, edge.Target, out int edge2))
                    {
                        edgesOnCycle.Add(edge2);
                    }
                }
            }
            dfs.FinishTreeEdge += finishEdge;
            dfs.CrossEdge += finishEdge;
            if (debug)
            {
                void backEdge(Edge<NodeIndex> edge)
                {
                    if (edge.Source != edge.Target && dg.mustNotInit.Contains(originalNode[edge.Source]))
                    {
                        List<NodeIndex> cycle = new Cycle();
                        cycle.Add(edge.Target);
                        bool found = false;
                        dfs.ForEachStackNode(delegate (NodeIndex node)
                        {
                            if (node == edge.Target)
                                found = true;
                            if (!found)
                                cycle.Add(node);
                        });
                        cycle.Reverse();
                        Debug.WriteLine($"UpdateCostsFromOffsetEdges dfs back edge ({edge.Source},{edge.Target}) due to cycle:");
                        foreach (var node in cycle)
                        {
                            Debug.WriteLine(NodeToString(node));
                        }
                    }
                }
                dfs.BackEdge += backEdge;
            }
            var onPath = MakeIndexedProperty.FromSet(edgesOnCycle);
            NodeIndex sinkNode = default(NodeIndex); // modified prior to each search
            var finder = new EdgeOnPathFinder<NodeIndex, EdgeIndex>(node => EdgesInGroup(node, group, deletedGraph),
                g.TargetOf, g, onPath, node => (node == sinkNode));
            foreach (var entry in offsetEdgesByGroup)
            {
                deletedGraph = entry.Value;
                group = entry.Key;
                edgesOnCycle.Clear();
                foreach (var deletedEdge in deletedGraph.Edges)
                {
                    sinkNode = deletedGraph.SourceOf(deletedEdge);
                    NodeIndex sourceNode = deletedGraph.TargetOf(deletedEdge);
                    if (verbose)
                        Debug.WriteLine("UpdateCostsFromOffsetEdges searching from {0} to {1}", sourceNode, sinkNode);
                    // newMethod can be very slow for some graphs, so we leave the option of the old fast method.
                    bool newMethod = false;
                    if (newMethod)
                    {
                        // This modifies onPath and therefore edgesOnCycle
                        finder.SearchFrom(sourceNode);
                    }
                    else
                    {
                        nodesOnCycle.Clear();
                        nodesOnCycle.Add(sinkNode);
                        dfs.Clear();
                        dfs.SearchFrom(sourceNode);
                    }
                }
                foreach (var edge in edgesOnCycle)
                {
                    edgeCost[edge] += 100f;
                    if (verbose)
                        Debug.WriteLine($"UpdateCostsFromOffsetEdges incrementing {EdgeToString(edge)}");
                }
            }

            Dictionary<int, IndexedGraph> GetOffsetEdgesByGroup()
            {
                Dictionary<NodeIndex, IndexedGraph> offsetEdgesByGroup2 = new Dictionary<EdgeIndex, IndexedGraph>();
                foreach (NodeIndex source in dg.dependencyGraph.Nodes)
                {
                    Set<NodeIndex> sourceGroups = null;
                    if (groupGraph != null)
                        sourceGroups = groupGraph.GetGroupSet(source);
                    foreach (EdgeIndex edge in dg.dependencyGraph.EdgesOutOf(source))
                    {
                        //if (IsAvailableOffsetEdge(edge)) // TODO: make this work
                        if (IsOffsetEdge(edge))
                        {
                            NodeIndex target = dg.dependencyGraph.TargetOf(edge);
                            NodeIndex sharedGroup = -1;
                            if (sourceGroups != null)
                                sharedGroup = groupGraph.GetSmallestGroup(target, sourceGroups.Contains);
                            IndexedGraph offsetGraph;
                            if (!offsetEdgesByGroup2.TryGetValue(sharedGroup, out offsetGraph))
                            {
                                offsetGraph = new IndexedGraph(g.Nodes.Count);
                                offsetEdgesByGroup2[sharedGroup] = offsetGraph;
                            }
                            offsetGraph.AddEdge(source, target);
                        }
                    }
                }

                return offsetEdgesByGroup2;
            }
        }
        // only used by UpdateCostsFromOffsetEdges
        private IEnumerable<NodeIndex> TargetsInGroup(NodeIndex node, NodeIndex group, IDirectedGraph<NodeIndex> deletedGraph)
        {
            foreach (EdgeIndex edge in g.EdgesOutOf(node))
            {
                // do not traverse NoInit edges
                if (IsNoInit(edge) || !(IsRequired(edge, includeAny: true) || IsFreshEdge(edge)))
                {
                    if (verbose)
                        Debug.WriteLine($"Blocked at {EdgeToString(edge)}");
                    continue;
                }
                NodeIndex target = g.TargetOf(edge);
                if (groupGraph != null)
                {
                    var targetGroups = groupGraph.GetGroupSet(target);
                    if (targetGroups.Contains(group))
                        yield return target;
                }
            }
            foreach (NodeIndex target in deletedGraph.TargetsOf(node))
            {
                yield return target;
            }
        }
        // only used by UpdateCostsFromOffsetEdges
        private IEnumerable<EdgeIndex> EdgesInGroup(NodeIndex node, NodeIndex group, IDirectedGraph<NodeIndex, EdgeIndex> deletedGraph)
        {
            foreach (EdgeIndex edge in g.EdgesOutOf(node))
            {
                // do not traverse NoInit edges
                if (IsNoInit(edge) || !(IsRequired(edge, includeAny: true) || IsFreshEdge(edge)))
                {
                    if (verbose)
                        Debug.WriteLine($"Blocked at {EdgeToString(edge)}");
                    continue;
                }
                NodeIndex target = g.TargetOf(edge);
                var targetGroups = groupGraph.GetGroupSet(target);
                if (targetGroups.Contains(group))
                    yield return edge;
            }
            // cannot use edge indices in deletedGraph
            foreach (NodeIndex target2 in deletedGraph.TargetsOf(node))
            {
                // the edge (node,target) is an offset edge that doesn't exist in g
                foreach (EdgeIndex edge in g.EdgesOutOf(target2))
                {
                    // do not traverse NoInit edges
                    if (IsNoInit(edge) || !(IsRequired(edge, includeAny: true) || IsFreshEdge(edge)))
                    {
                        if (verbose)
                            Debug.WriteLine($"Blocked at {EdgeToString(edge)}");
                        continue;
                    }
                    NodeIndex target = g.TargetOf(edge);
                    var targetGroups = groupGraph.GetGroupSet(target);
                    if (targetGroups.Contains(group))
                        yield return edge;
                }
            }
        }

        private bool IsOffsetEdge(EdgeIndex edgeOrig)
        {
            return dg.OffsetIndices.ContainsKey(edgeOrig);
        }

        private bool IsAvailableOffsetEdge(EdgeIndex edgeOrig)
        {
            IOffsetInfo offsetInfo;
            if (!dg.OffsetIndices.TryGetValue(edgeOrig, out offsetInfo))
            {
                return false;
            }
            return offsetInfo.All(info => info.isAvailable);
        }

        private IEnumerable<IReadOnlyList<EdgeIndex>> GetRequiredEdgeSets(NodeIndex target, bool inIteration, bool useDirections)
        {
            NodeIndex originalTarget = originalNode[target];
            byte bits = dg.requiredBits[originalTarget];
            byte bit = 1;
            List<EdgeIndex> edges = new List<EdgeIndex>();
            while ((bits & bit) != 0)
            {
                // find all edges providing this bit
                edges.Clear();
                bool hasForward = false;
                foreach (EdgeIndex edge in g.EdgesInto(target))
                {
                    NodeIndex source = g.SourceOf(edge);
                    NodeIndex originalSource = originalNode[source];
                    EdgeIndex edgeOrig = originalEdge[edge];
                    if (edgeOrig == -1)
                        continue;
                    if ((inIteration || IgnoreOffsetRequirements) && IsAvailableOffsetEdge(edgeOrig))
                    {
                        hasForward = true;
                        break;
                    }
                    byte bits2 = dg.bitsProvided[edgeOrig];
                    if ((bits2 & bit) != 0)
                    {
                        bool isInitialized = dg.hasNonUniformInitializer.Contains(originalSource);
                        if ((useDirections && direction[edge] == Direction.Forward) || isInitialized)
                            hasForward = true;
                        else
                            edges.Add(edge);
                    }
                }
                if (!hasForward)
                    yield return edges;
                bit <<= 1;
            }
        }

        private const float defaultInitCost = 0.001f;

        /// <summary>
        /// Compute the reversal cost, i.e. to be labeled backward, for every edge into target
        /// </summary>
        /// <param name="edgeCost"></param>
        /// <param name="target"></param>
        /// <param name="inIteration">true if we are scoring edges in the iteration schedule</param>
        /// <remarks>
        /// A required edge has infinite cost.
        /// An edge in an Any group whose alternatives are all backward has infinite cost.
        /// If an alternative is unlabeled, the cost is 1/(group size).
        /// </remarks>
        private void UpdateCostsInit(float[] edgeCost, NodeIndex target, bool inIteration)
        {
            NodeIndex originalTarget = originalNode[target];
            foreach (EdgeIndex edge in g.EdgesInto(target))
            {
                NodeIndex source = g.SourceOf(edge);
                NodeIndex originalSource = originalNode[source];
                if (dg.initializedNodes.Contains(originalSource) &&
                    (!dg.initializedNodes.Contains(originalTarget) || inIteration || !initsCanBeStale))
                    continue;
                EdgeIndex edgeOrig = originalEdge[edge];
                if (dg.initializedEdges.Contains(edgeOrig))
                    continue;
                bool isOffsetEdge = (edgeOrig >= 0) && IsOffsetEdge(edgeOrig);
                if (inIteration && isOffsetEdge)
                    continue;
                float cost = 0;
                bool isRequired = (edgeOrig >= 0) && dg.isRequired[edgeOrig];
                if ((isRequired && (!IgnoreOffsetRequirements || !isOffsetEdge)) || (mustInitBackwardEdges && !inIteration))
                    cost = float.PositiveInfinity;
                bool mustInit = (edgeOrig < 0) || !dg.noInit[edgeOrig];
                if (mustInitBackwardEdges && backwardInSchedule != null && backwardInSchedule.Contains(edge))
                    mustInit = true;
                if (mustInit)
                {
                    cost += defaultInitCost;
                    // among several equivalent edges, we want the existing forward edges to be picked, so we increase their cost
                    if (backwardInSchedule != null && !backwardInSchedule.Contains(edge))
                        cost += defaultInitCost;
                }
                edgeCost[edge] = cost;
            }
            foreach (var edgeSet in GetRequiredEdgeSets(target, inIteration, true))
            {
                int unknownCount = edgeSet.Count(edge => direction[edge] == Direction.Unknown);
                float cost = (unknownCount <= 1) ? float.PositiveInfinity : 1f / edgeSet.Count;
                foreach (EdgeIndex edge in edgeSet)
                {
                    if (unknownCount > 0 && direction[edge] != Direction.Unknown)
                        continue;
                    edgeCost[edge] += cost;
                }
            }
        }

        /// <summary>
        /// Get the cost of reversing (i.e. labeling backward) each edge in the dependency graph.
        /// </summary>
        /// <param name="inIteration">true if we are scoring edges in the iteration schedule</param>
        /// <returns></returns>
        private float[] GetEdgeCostsInit(bool inIteration)
        {
            float[] edgeCost = new float[g.EdgeCount()];
            foreach (NodeIndex target in g.Nodes)
            {
                UpdateCostsInit(edgeCost, target, inIteration);
            }
            return edgeCost;
        }

        /// <summary>
        /// Update all costs affected by the edges in newBackEdges and newForwardEdges
        /// </summary>
        /// <param name="edgeCost"></param>
        /// <returns></returns>
        private bool UpdateCostsInit(float[] edgeCost)
        {
            bool satisfiable = true;
            while (newBackEdges.Count > 0)
            {
                EdgeIndex edge = newBackEdges.Pop();
                if (float.IsPositiveInfinity(edgeCost[edge]))
                {
                    if (debug && verbose)
                        ShowConflict(edge);
                    satisfiable = false;
                }
                NodeIndex target = g.TargetOf(edge);
                UpdateCostsInit(edgeCost, target, false);
                if (satisfiable && float.IsPositiveInfinity(edgeCost[edge]))
                {
                    if (debug && verbose)
                        ShowConflict(edge);
                    satisfiable = false;
                }
            }
            while (newForwardEdges.Count > 0)
            {
                EdgeIndex edge = newForwardEdges.Pop();
                NodeIndex target = g.TargetOf(edge);
                UpdateCostsInit(edgeCost, target, false);
            }
            return satisfiable;
        }

        // Called when edge has infinite reversal cost and yet was chosen to be reversed
        private void ShowConflict(EdgeIndex edge)
        {
            Debug.WriteLine("conflict at " + EdgeToString(edge));
            if (direction[edge] == Direction.Backward)
            {
                // find a cycle of forward edges, not involving this edge
                direction[edge] = Direction.Unknown;
                NodeIndex source = g.SourceOf(edge);
                NodeIndex target = g.TargetOf(edge);
                var path = GetAnyPath(target, source);
                Debug.WriteLine("cycle:");
                foreach (var node in path)
                {
                    Debug.WriteLine(NodeToString(node));
                }
                direction[edge] = Direction.Backward;
            }
        }

        private string EndpointsToString(EdgeIndex edge)
        {
            return dg.NodeToString(g.SourceOf(edge)) + Environment.NewLine + dg.NodeToString(g.TargetOf(edge));
        }

        private int GetClosestUpdate(List<NodeIndex> schedule, NodeIndex source, int targetPos)
        {
            int bestPos = -1;
            int minDistance = int.MaxValue;
            for (int i = 0; i < schedule.Count; i++)
            {
                if (schedule[i] == source)
                {
                    int distance = targetPos - i;
                    if (distance < 0)
                        distance += schedule.Count;
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        bestPos = i;
                    }
                }
            }
            return bestPos;
        }

        private List<NodeIndex> ConvertToOriginalNodes(IEnumerable<NodeIndex> schedule)
        {
            List<NodeIndex> newSchedule = new List<NodeIndex>();
            foreach (NodeIndex node in schedule)
            {
                newSchedule.Add(originalNode[node]);
            }
            return newSchedule;
        }

        /// <summary>
        /// Set the direction of edges whose label is forced by existing labels.  On input, todo contains all newly labeled edges.
        /// On output, todo is empty and (newForwardEdges,newBackEdges) contains all newly labeled edges.
        /// </summary>
        private void PropagateConstraints(bool useGroups = false)
        {
            if (useExperimentalSerialSchedules)
            {
                PropagateConstraintsWithOffsets();
                return;
            }
            if (groupGraph == null)
                useGroups = false;
            while (todo.Count > 0)
            {
                EdgeIndex edge = todo.Pop();
                if (deletedEdges.Contains(edge))
                    continue;
                NodeIndex source, target;
                if (direction[edge] == Direction.Forward)
                {
                    source = g.SourceOf(edge);
                    target = g.TargetOf(edge);
                }
                else
                {
                    source = g.TargetOf(edge);
                    target = g.SourceOf(edge);
                }
                if (verbose)
                {
                    Debug.WriteLine("propagating edge {0} {1}", EdgeToString(edge), direction[edge]);
                }

                DepthFirstSearch<NodeIndex> dfsAncestors;
                if (!useGroups)
                {
                    if (dfsAncestorsWithoutGroups == null)
                    {
                        dfsAncestorsWithoutGroups = new DepthFirstSearch<NodeIndex>(ForwardSourcesAndBackwardTargets, g);
                        // nodes are only discovered once by dfs, so a node is never added twice
                        dfsAncestorsWithoutGroups.DiscoverNode += ancestors.Add;
                    }
                    else
                    {
                        dfsAncestorsWithoutGroups.Clear();
                    }
                    dfsAncestors = dfsAncestorsWithoutGroups;
                }
                else
                {
                    if (dfsAncestorsWithGroups == null)
                    {
                        dfsAncestorsWithGroups = new DepthFirstSearch<NodeIndex>(ForwardSourcesAndBackwardTargetsWithGroups, groupGraph);
                        // nodes are only discovered once by dfs, so a node is never added twice
                        dfsAncestorsWithGroups.DiscoverNode += ancestors.Add;
                    }
                    else
                    {
                        dfsAncestorsWithGroups.Clear();
                    }
                    dfsAncestors = dfsAncestorsWithGroups;
                }
                ancestors.Clear();
                dfsAncestors.SearchFrom(source);
                if (useGroups)
                {
                    NodeIndex origSource = originalNode[source];
                    ancestors.AddRange(groupGraph.GetGroups(origSource));
                }
                bool showAncestors = false;
                if (verbose && showAncestors)
                    Debug.WriteLine("ancestors: " + ancestors);
                DepthFirstSearch<NodeIndex> dfsDescendants;
                if (!useGroups)
                {
                    if (dfsDescendantsWithoutGroups == null)
                    {
                        dfsDescendantsWithoutGroups = new DepthFirstSearch<NodeIndex>(BackwardSourcesAndForwardTargets, g);
                        // nodes are only discovered once by dfs, so a node is never added twice
                        dfsDescendantsWithoutGroups.DiscoverNode += LabelEdges;
                    }
                    dfsDescendants = dfsDescendantsWithoutGroups;
                }
                else
                {
                    if (dfsDescendantsWithGroups == null)
                    {
                        dfsDescendantsWithGroups = new DepthFirstSearch<NodeIndex>(BackwardSourcesAndForwardTargetsWithGroups, groupGraph);
                        // nodes are only discovered once by dfs, so a node is never added twice
                        dfsDescendantsWithGroups.DiscoverNode += LabelEdgesWithGroups;
                    }
                    dfsDescendants = dfsDescendantsWithGroups;
                }
                dfsDescendants.Clear();
                NodeIndex targetGroup = !useGroups ? target : groupGraph.GetLargestGroupExcluding(target, ancestors);
                dfsDescendants.SearchFrom(targetGroup);
            }
        }

        private void LabelEdges(NodeIndex node)
        {
            foreach (EdgeIndex edge in g.EdgesInto(node))
            {
                if (direction[edge] == Direction.Unknown && !deletedEdges.Contains(edge))
                {
                    NodeIndex source = g.SourceOf(edge);
                    bool mustPrecede = ancestors.Contains(source);
                    if (mustPrecede)
                    {
                        direction[edge] = Direction.Forward;
                        if (newForwardEdges != null)
                            newForwardEdges.Push(edge);
                        todo.Push(edge);
                        if (debug)
                            Debug.WriteLine($"propagated forward edge {EdgeToString(edge)}");
                    }
                }
            }
            foreach (EdgeIndex edge in g.EdgesOutOf(node))
            {
                if (direction[edge] == Direction.Unknown && !deletedEdges.Contains(edge))
                {
                    NodeIndex target = g.TargetOf(edge);
                    if (ancestors.Contains(target))
                    {
                        direction[edge] = Direction.Backward;
                        newBackEdges.Push(edge);
                        todo.Push(edge);
                        if (debug)
                            Debug.WriteLine($"propagated backward edge {EdgeToString(edge)}");
                    }
                }
            }
        }

        /// <summary>
        /// Label edges into and out of node, based on the contents of ancestors.
        /// </summary>
        /// <param name="node">Can be a group</param>
        private void LabelEdgesWithGroups(NodeIndex node)
        {
            NodeIndex origNode = originalNode[node];
            Set<NodeIndex> groups = groupGraph.GetGroupSet(origNode);
            foreach (EdgeIndex edge in groupGraph.EdgesInto(node))
            {
                if (direction[edge] == Direction.Unknown)
                {
                    NodeIndex source = g.SourceOf(edge);
                    NodeIndex sourceGroup = groupGraph.GetLargestGroupExcluding(source, groups);
                    if (ancestors.Contains(sourceGroup))
                    {
                        direction[edge] = Direction.Forward;
                        if (newForwardEdges != null)
                            newForwardEdges.Push(edge);
                        todo.Push(edge); // is this needed?
                        if (debug)
                            Debug.WriteLine($"propagated forward edge {EdgeToString(edge)}");
                    }
                }
            }
            foreach (EdgeIndex edge in groupGraph.EdgesOutOf(node))
            {
                if (direction[edge] == Direction.Unknown)
                {
                    NodeIndex target = g.TargetOf(edge);
                    NodeIndex targetGroup = groupGraph.GetLargestGroupExcluding(target, groups);
                    if (ancestors.Contains(targetGroup))
                    {
                        direction[edge] = Direction.Backward;
                        newBackEdges.Push(edge);
                        todo.Push(edge);
                        if (debug)
                            Debug.WriteLine($"propagated backward edge {EdgeToString(edge)}");
                    }
                }
            }
        }

        private class OffsetBounds
        {
            public int LowerBound, UpperBound;

            public OffsetBounds(int lowerBound, int upperBound)
            {
                this.LowerBound = lowerBound;
                this.UpperBound = upperBound;
            }

            public OffsetBounds(OffsetBounds that) : this(that.LowerBound, that.UpperBound)
            {
            }

            public OffsetBounds Union(OffsetBounds a, OffsetBounds b)
            {
                return new OffsetBounds(System.Math.Min(a.LowerBound, b.LowerBound),
                    System.Math.Max(a.UpperBound, b.UpperBound));
            }

            public void UnionWithZero()
            {
                if (0 < this.LowerBound)
                    this.LowerBound = 0;
                if (0 > this.UpperBound)
                    this.UpperBound = 0;
            }

            public void Union(OffsetBounds bounds)
            {
                if (bounds.LowerBound < this.LowerBound)
                    this.LowerBound = bounds.LowerBound;
                if (bounds.UpperBound > this.UpperBound)
                    this.UpperBound = bounds.UpperBound;
            }

            public void Add(int offset)
            {
                this.LowerBound += offset;
                this.UpperBound += offset;
            }

            public void Add(OffsetBounds bounds)
            {
                this.LowerBound += bounds.LowerBound;
                this.UpperBound += bounds.UpperBound;
            }

            public bool Contains(int value)
            {
                return (value >= LowerBound) && (value <= UpperBound);
            }

            public override string ToString()
            {
                if (LowerBound == UpperBound)
                    return $"{LowerBound}";
                else
                    return $"[{LowerBound}, {UpperBound}]";
            }
        }

        private class OffsetBoundCollection
        {
            // TODO: what comparer should be used for keys?
            /// <summary>
            /// Missing entries imply the variable is not eligible for offset
            /// </summary>
            public Dictionary<IVariableDeclaration, OffsetBounds> dict = new Dictionary<IVariableDeclaration, OffsetBounds>();

            /// <summary>
            /// Copy constructor.  Makes a deep copy.
            /// </summary>
            /// <param name="that"></param>
            /// <param name="keys">The set of keys that will be copied.</param>
            public OffsetBoundCollection(OffsetBoundCollection that, IEnumerable<IVariableDeclaration> keys)
            {
                foreach (var key in keys)
                {
                    OffsetBounds newBounds;
                    if (that.dict.TryGetValue(key, out newBounds))
                    {
                        dict.Add(key, new OffsetBounds(newBounds));
                    }
                }
            }

            public OffsetBoundCollection(ICollection<IVariableDeclaration> keys)
            {
                foreach (var key in keys)
                {
                    dict.Add(key, new OffsetBounds(0, 0));
                }
            }

            public OffsetBoundCollection(IOffsetInfo info, ICollection<IVariableDeclaration> keys, int scale = 1)
            {
                // the same loopVar may occur multiple times
                foreach (var offset in info)
                {
                    OffsetBounds newBounds = new OffsetBounds(offset.offset * scale, offset.offset * scale);
                    OffsetBounds bounds;
                    if (dict.TryGetValue(offset.loopVar, out bounds))
                    {
                        bounds.Union(newBounds);
                    }
                    else if (keys.Contains(offset.loopVar))
                    {
                        dict.Add(offset.loopVar, newBounds);
                    }
                }
                foreach (var key in keys)
                {
                    if (!dict.ContainsKey(key))
                    {
                        dict.Add(key, new OffsetBounds(0, 0));
                    }
                }
            }

            public void Add(IOffsetInfo info, int scale = 1)
            {
                var obc = new OffsetBoundCollection(info, dict.Keys, scale);
                foreach (var entry in obc.dict)
                {
                    OffsetBounds newBounds = entry.Value;
                    dict[entry.Key].Add(newBounds);
                }
            }

            public bool ContainsNonZero()
            {
                foreach (var bounds in dict.Values)
                {
                    if (!bounds.Contains(0))
                        return true;
                }
                return false;
            }

            /// <summary>
            /// Sum of bounds over the intersection of keys.
            /// </summary>
            /// <param name="that"></param>
            public void Add(OffsetBoundCollection that)
            {
                List<IVariableDeclaration> keysToRemove = new List<IVariableDeclaration>();
                foreach (var entry in dict)
                {
                    OffsetBounds bounds = entry.Value;
                    OffsetBounds thatBounds;
                    if (that.dict.TryGetValue(entry.Key, out thatBounds))
                    {
                        bounds.Add(thatBounds);
                    }
                    else
                    {
                        keysToRemove.Add(entry.Key);
                    }
                }
                foreach (var key in keysToRemove)
                    dict.Remove(key);
            }

            /// <summary>
            /// Union of bounds over the intersection of keys.
            /// </summary>
            /// <param name="that"></param>
            public void Union(OffsetBoundCollection that)
            {
                List<IVariableDeclaration> keysToRemove = new List<IVariableDeclaration>();
                foreach (var entry in dict)
                {
                    OffsetBounds bounds = entry.Value;
                    OffsetBounds thatBounds;
                    if (that.dict.TryGetValue(entry.Key, out thatBounds))
                    {
                        bounds.Union(thatBounds);
                    }
                    else
                    {
                        keysToRemove.Add(entry.Key);
                    }
                }
                foreach (var key in keysToRemove)
                    dict.Remove(key);
            }

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder();
                sb.Append("OffsetBoundCollection(");
                foreach (var entry in dict)
                {
                    sb.Append(entry.Key.Name);
                    sb.Append("=");
                    sb.Append(entry.Value);
                }
                sb.Append(")");
                return sb.ToString();
            }
        }

        /// <summary>
        /// Set the direction of edges whose label is forced by existing labels.  On input, todo contains all newly labeled edges.
        /// On output, todo is empty and (newForwardEdges,newBackEdges) contains all newly labeled edges.
        /// </summary>
        private void PropagateConstraintsWithOffsets()
        {
            while (todo.Count > 0)
            {
                EdgeIndex edge = todo.Pop();
                if (deletedEdges.Contains(edge))
                    continue;
                NodeIndex source, target;
                if (direction[edge] == Direction.Forward)
                {
                    source = g.SourceOf(edge);
                    target = g.TargetOf(edge);
                }
                else
                {
                    source = g.TargetOf(edge);
                    target = g.SourceOf(edge);
                }
                if (verbose)
                {
                    Debug.WriteLine("propagating edge {0} {1}", EdgeToString(edge), direction[edge]);
                }

                var dfsAncestors = new DepthFirstSearch<NodeIndex>(ForwardSourcesAndBackwardTargets, g);
                List<NodeIndex> ancestorList = new List<NodeIndex>();
                dfsAncestors.FinishNode += delegate (NodeIndex node)
                {
                    ancestors.Add(node);
                    // parents will be added before children
                    ancestorList.Add(node);
                };
                ancestors.Clear();
                dfsAncestors.SearchFrom(source);
                // reversing the list ensures that a node is processed after its children.
                ancestorList.Reverse();
                ancestorOffset.Clear();
                ancestorOffset.Add(source, new OffsetBoundCollection(this.loopVarsOfNode[source]));
                foreach (var ancestor in ancestorList)
                {
                    if (ancestor == source)
                        continue;
                    PropagateOffsetBounds(ancestor, true, ancestorOffset.ContainsKey, ancestorOffset);
                }
                if (verbose && showAncestors)
                {
                    Trace.Write("ancestors: ");
                    foreach (var ancestor in ancestorList)
                    {
                        Trace.Write(ancestor);
                        Trace.Write(" ");
                        OffsetBoundCollection obc;
                        if (ancestorOffset.TryGetValue(ancestor, out obc))
                        {
                            Trace.Write(obc);
                            Trace.Write(" ");
                        }
                    }
                    Trace.WriteLine("");
                }
                if (ancestors.Contains(target))
                    continue;  // there are no new constraints to propagate
                DepthFirstSearch<NodeIndex> dfsDescendants = null;
                List<NodeIndex> descendantList = new List<NodeIndex>();
                if (dfsDescendants == null)
                {
                    var g2 = new DirectedGraphFilter<NodeIndex, EdgeIndex>(g, edge2 => !ancestors.Contains(g.SourceOf(edge2)) && !ancestors.Contains(g.TargetOf(edge2)));
                    dfsDescendants = new DepthFirstSearch<NodeIndex>(BackwardSourcesAndForwardTargets, g2);
                    dfsDescendants.FinishNode += delegate (NodeIndex node)
                    {
                        if (ancestors.Contains(node))
                            throw new Exception($"node {node} is both ancestor and descendant");
                        // children will be added before parents
                        descendantList.Add(node);
                    };
                }
                descendantList.Clear();
                dfsDescendants.Clear();
                dfsDescendants.SearchFrom(target);
                // reversing the list ensures that a node is processed before its children.
                descendantList.Reverse();
                descendantOffset.Clear();
                EdgeIndex edgeOrig = originalEdge[edge];
                IOffsetInfo offsetInfo;
                if (dg.OffsetIndices.TryGetValue(edgeOrig, out offsetInfo))
                {
                    // a negative offset means the source precedes the target
                    int scale = (direction[edge] == Direction.Forward) ? 1 : -1;
                    descendantOffset.Add(target, new OffsetBoundCollection(offsetInfo, this.loopVarsOfNode[target], scale));
                }
                else
                {
                    descendantOffset.Add(target, new OffsetBoundCollection(this.loopVarsOfNode[target]));
                }
                foreach (var descendant in descendantList)
                {
                    if (descendant == target)
                        continue;
                    PropagateOffsetBounds(descendant, false, descendantOffset.ContainsKey, descendantOffset);
                    LabelEdgesWithOffsets(descendant);
                }
                if (verbose && showAncestors)
                {
                    Trace.Write("descendants: ");
                    foreach (var descendant in descendantList)
                    {
                        Trace.Write(descendant);
                        Trace.Write(" ");
                        OffsetBoundCollection obc;
                        if (descendantOffset.TryGetValue(descendant, out obc))
                        {
                            Trace.Write(obc);
                            Trace.Write(" ");
                        }
                    }
                    Trace.WriteLine("");
                }
            }
        }

        private void PropagateOffsetBounds(NodeIndex node, bool isAncestor, Predicate<NodeIndex> isVisited, Dictionary<NodeIndex, OffsetBoundCollection> offsetBoundsOfNode)
        {
            Assert.IsTrue(!offsetBoundsOfNode.ContainsKey(node));
            bool firstTarget = true;
            var keys = this.loopVarsOfNode[node];
            OffsetBoundCollection obc = null;
            Action<EdgeIndex, bool> processEdge = (edge, isForward) =>
            {
                // inherit offset bounds from target
                NodeIndex target = (isForward == isAncestor) ? g.TargetOf(edge) : g.SourceOf(edge);
                if (target == node)
                    return;
                if (!isVisited(target))
                    return;
                OffsetBoundCollection obc2 = offsetBoundsOfNode[target];
                EdgeIndex edgeOrig = originalEdge[edge];
                IOffsetInfo offsetInfo;
                if (!dg.OffsetIndices.TryGetValue(edgeOrig, out offsetInfo))
                {
                    // edge has zero offset on all loopVars
                    if (firstTarget)
                        obc = new OffsetBoundCollection(obc2, keys);
                    else
                    {
                        obc.Union(obc2);
                    }
                }
                else
                {
                    // edge has non-zero offset on some loopVars
                    var obc3 = new OffsetBoundCollection(offsetInfo, keys);
                    obc3.Add(obc2);
                    if (firstTarget)
                    {
                        obc = obc3;
                    }
                    else
                    {
                        obc.Union(obc3);
                    }
                }
                firstTarget = false;
            };
            foreach (EdgeIndex edge in g.EdgesInto(node))
            {
                if (deletedEdges.Contains(edge) || direction[edge] == Direction.Unknown)
                    continue;
                bool isForward = (direction[edge] == Direction.Forward);
                if (isAncestor != isForward)
                {
                    processEdge(edge, isForward);
                }
            }
            foreach (EdgeIndex edge in g.EdgesOutOf(node))
            {
                if (deletedEdges.Contains(edge) || direction[edge] == Direction.Unknown)
                    continue;
                bool isForward = (direction[edge] == Direction.Forward);
                if (isAncestor == isForward)
                {
                    processEdge(edge, isForward);
                }
            }
            if (obc == null)
                throw new Exception("node has no predecessor");
            offsetBoundsOfNode[node] = obc;
        }

        private bool DirectionIsForced(EdgeIndex edge, Direction direction)
        {
            NodeIndex ancestor, descendant;
            if (direction == Direction.Forward)
            {
                ancestor = g.SourceOf(edge);
                descendant = g.TargetOf(edge);
            }
            else
            {
                ancestor = g.TargetOf(edge);
                descendant = g.SourceOf(edge);
            }
            var keys = this.loopVarsOfNode[descendant];
            EdgeIndex edgeOrig = originalEdge[edge];
            IOffsetInfo offsetInfo;
            if (!dg.OffsetIndices.TryGetValue(edgeOrig, out offsetInfo))
            {
                OffsetBoundCollection obc2 = new OffsetBoundCollection(ancestorOffset[ancestor], keys);
                obc2.Add(descendantOffset[descendant]);
                return !obc2.ContainsNonZero();
            }
            else
            {
                int scale = (direction == Direction.Backward) ? 1 : -1;
                OffsetBoundCollection obc2 = new OffsetBoundCollection(offsetInfo, keys);
                obc2.Add(descendantOffset[descendant]);
                obc2.Add(ancestorOffset[ancestor]);
                return !obc2.ContainsNonZero();
            }
        }

        private void LabelEdgesWithOffsets(NodeIndex node)
        {
            foreach (EdgeIndex edge in g.EdgesInto(node))
            {
                if (direction[edge] == Direction.Unknown && !deletedEdges.Contains(edge))
                {
                    NodeIndex source = g.SourceOf(edge);
                    if (ancestors.Contains(source) && DirectionIsForced(edge, Direction.Forward))
                    {
                        // source must precede node
                        direction[edge] = Direction.Forward;
                        if (newForwardEdges != null)
                            newForwardEdges.Push(edge);
                        todo.Push(edge);
                        if (debug)
                            Debug.WriteLine($"propagated forward edge {EdgeToString(edge)}");
                    }
                }
            }
            foreach (EdgeIndex edge in g.EdgesOutOf(node))
            {
                if (direction[edge] == Direction.Unknown && !deletedEdges.Contains(edge))
                {
                    NodeIndex target = g.TargetOf(edge);
                    if (ancestors.Contains(target) && DirectionIsForced(edge, Direction.Backward))
                    {
                        // target must precede node
                        direction[edge] = Direction.Backward;
                        newBackEdges.Push(edge);
                        todo.Push(edge);
                        if (debug)
                            Debug.WriteLine($"propagated backward edge {EdgeToString(edge)}");
                    }
                }
            }
        }

        private int ScheduleSearch(List<NodeIndex> schedule, int desiredSize, int[] actualCount, int[] desiredCount, int bestScore)
        {
            if (schedule.Count == desiredSize)
            {
                double mean;
                int score = MaxBackEdgeCount(schedule, out mean);
                if (score < bestScore)
                {
                    bestScore = score;
                    if (bestScore == 1)
                        Console.WriteLine(StringUtil.CollectionToString(schedule, " "));
                }
                return bestScore;
            }
            int numValues = actualCount.Length;
            int lastNode = schedule[schedule.Count - 1];
            for (int i = 0; i < numValues; i++)
            {
                if (actualCount[i] == desiredCount[i])
                    continue;
                if (i == lastNode)
                    continue;
                schedule.Add(i);
                actualCount[i]++;
                bestScore = ScheduleSearch(schedule, desiredSize, actualCount, desiredCount, bestScore);
                schedule.RemoveAt(schedule.Count - 1);
                actualCount[i]--;
                if (bestScore == 1)
                    break;
            }
            return bestScore;
        }

        private IEnumerable<string> ScheduleToText(IEnumerable<NodeIndex> schedule)
        {
            List<string> text = new List<string>();
            int i = 0;
            foreach (NodeIndex node in schedule)
            {
                text.Add(NodeToString(node));
                i++;
            }
            return text;
        }

        private string NodeToString(NodeIndex node)
        {
            string groupString = "";
            if (groupOf != null)
            {
                Set<NodeIndex> groups = GetGroupsInDg(node);
                if (groups.Count > 0)
                    groupString = "[" + groups.ToString() + "] ";
            }
            string nodeString = (node >= dg.dependencyGraph.Nodes.Count) ? "group" : dg.NodeToShortString(node);
            return $"{node} {groupString}{nodeString}";
        }

        private void RecordNodes()
        {
            foreach (var node in dg.dependencyGraph.Nodes)
                Debug.WriteLine(NodeToString(node));
            RecordText("Nodes", dg.dependencyGraph.Nodes.Select(NodeToString));
        }

        private void RecordSchedule(string name, IEnumerable<NodeIndex> schedule)
        {
            RecordText(name, ScheduleToText(schedule));
        }

        private void WriteSchedule(IEnumerable<NodeIndex> schedule)
        {
            foreach (string line in ScheduleToText(schedule))
                Debug.WriteLine(line);
        }

        private Set<NodeIndex> GetInvalidNodes(IEnumerable<NodeIndex> schedule)
        {
            Set<NodeIndex> invalid = new Set<EdgeIndex>();
            foreach (NodeIndex node in schedule)
            {
                invalid.Remove(node);
                foreach (EdgeIndex edge in g.EdgesOutOf(node))
                {
                    if (IsTrigger(edge))
                    {
                        NodeIndex target = g.TargetOf(edge);
                        invalid.Add(target);
                    }
                }
            }
            return invalid;
        }

        private string EdgeToString(IndexedGraph g, EdgeIndex edge)
        {
            return "(" + g.SourceOf(edge) + "," + g.TargetOf(edge) + ")";
        }

        private string EdgeToString(EdgeIndex edge)
        {
            return EdgeToString(this.g, edge);
        }

        private IndexedProperty<EdgeIndex, Direction> CloneDirections()
        {
            IndexedProperty<EdgeIndex, Direction> clone = g.CreateEdgeData<Direction>();
            foreach (EdgeIndex edge in g.Edges)
            {
                clone[edge] = direction[edge];
            }
            return clone;
        }

        private int[] GetBackEdgeCounts(IList<NodeIndex> schedule)
        {
            if (cycles == null)
                cycles = FindCycles(g);
            int[] counts = new int[cycles.Count];
            int i = 0;
            foreach (Cycle c in cycles)
            {
                counts[i++] = CountBackEdges(NodesOfCycle(c, g), schedule);
            }
            return counts;
        }

        internal int MaxBackEdgeCount(IList<NodeIndex> schedule, out double maxBackOverLength)
        {
            Trace.WriteLine("");
            if (cycles == null)
                cycles = FindCycles(g);
            int maxBack = 0;
            maxBackOverLength = 0;
            foreach (Cycle c in cycles)
            {
                if (c.Count == 1)
                    continue;
                int count = CountBackEdges(NodesOfCycle(c, g), schedule);
                if (count > maxBack)
                {
                    maxBack = count;
                    Trace.Write($"{count} ");
                    WriteCycle(c);
                }
                double ratio = (double)count / c.Count;
                if (ratio > maxBackOverLength)
                    maxBackOverLength = ratio;
            }
            return maxBack;
        }

        private IEnumerable<NodeIndex> NodesOfCycle(Cycle c)
        {
            return NodesOfCycle(c, g);
        }

        private IEnumerable<NodeIndex> NodesOfCycle(Cycle c, IndexedGraph g)
        {
            foreach (EdgeIndex edge in c)
            {
                NodeIndex source = g.SourceOf(edge);
                yield return source;
            }
        }

        private static int CountBackEdges(IEnumerable<NodeIndex> c, IList<NodeIndex> nodes)
        {
            int minCount = int.MaxValue;
            foreach (NodeIndex node in c)
            {
                //NodeIndex orig = useOriginalNodes ? originalNode[node] : node;
                for (int i = 0; i < nodes.Count; i++)
                {
                    if (node == nodes[i])
                    {
                        int count = CountBackEdges(c, nodes, i);
                        if (count < minCount)
                        {
                            minCount = count;
                            if (minCount == 1)
                                return minCount;
                        }
                    }
                }
                break;
            }
            return minCount;
        }

        private static int CountBackEdges(IEnumerable<NodeIndex> c, IList<NodeIndex> nodes, int i)
        {
            int firstMatchPos = -1;
            int count = 1;
            foreach (NodeIndex node in c)
            {
                //NodeIndex orig = useOriginalNodes ? originalNode[node] : node;
                if (firstMatchPos == -1)
                {
                    for (; i < nodes.Count; i++)
                    {
                        if (node == nodes[i])
                        {
                            firstMatchPos = i;
                            break;
                        }
                    }
                    if (firstMatchPos == -1)
                        return -1;
                }
                else
                {
                    int start = i;
                    while (true)
                    {
                        i++;
                        i = i % nodes.Count;
                        if (i == firstMatchPos)
                            count++;
                        if (node == nodes[i])
                            break;
                        if (i == start)
                            return -1;
                    }
                }
            }
            return count;
        }

        /// <summary>
        /// Create g from dg
        /// </summary>
        /// <param name="forcedForwardEdges">Modified on exit</param>
        /// <param name="forcedBackEdges">Modified on exit</param>
        /// <param name="forceInitializedNodes"></param>
        private void CreateGraph(Set<EdgeIndex> forcedForwardEdges, Set<EdgeIndex> forcedBackEdges, bool forceInitializedNodes)
        {
            bool splitNoInits = false;
            if (splitNoInits)
            {
                g = new IndexedGraph();
                for (int i = 0; i < dg.dependencyGraph.Nodes.Count; i++)
                {
                    g.AddNode();
                }
            }
            else
            {
                g = new IndexedGraph(dg.dependencyGraph.Nodes.Count);
            }
            CreateNodeData();
            this.newNodes = new List<List<NodeIndex>>();
            foreach (NodeIndex node in dg.dependencyGraph.Nodes)
                this.newNodes.Add(new List<NodeIndex>() { node });
            deletedEdges.Clear();
            originalEdge = new List<EdgeIndex>();
            foreach (NodeIndex source in dg.dependencyGraph.Nodes)
            {
                bool hasNoInitEdges = false;
                bool hasSomeInitEdges = false;
                if (splitNoInits)
                {
                    // determine if source has a mixture of NoInit and regular outgoing edges
                    foreach (EdgeIndex edge in dg.dependencyGraph.EdgesOutOf(source))
                    {
                        if (dg.isDeleted[edge])
                            continue;
                        NodeIndex target = dg.dependencyGraph.TargetOf(edge);
                        if (source != target)
                        {
                            if (dg.noInit[edge])
                                hasNoInitEdges = true;
                            else
                                hasSomeInitEdges = true;
                        }
                    }
                }
                NodeIndex sourceNoInit = source;
                if (hasNoInitEdges && hasSomeInitEdges)
                {
                    if (debug)
                        Debug.WriteLine("{0} has mixed out edges", source);
                    // split source into 2 nodes, having the same parents
                    // but one having only NoInit outgoing edges and one having only regular outgoing edges.
                    // sourceNoInit has only NoInit outgoing edges.
                    sourceNoInit = g.AddNode();
                    this.newNodes[source].Add(sourceNoInit);
                    this.originalNode.Add(source);
                    // new node inherits all parents in g
                    foreach (EdgeIndex edge in g.EdgesInto(source))
                    {
                        NodeIndex parent = g.SourceOf(edge);
                        EdgeIndex newEdge = g.AddEdge(parent, sourceNoInit);
                        originalEdge.Add(originalEdge[edge]);
                    }
                }
                foreach (EdgeIndex edge in dg.dependencyGraph.EdgesOutOf(source))
                {
                    if (dg.isDeleted[edge])
                        continue;
                    foreach (NodeIndex target in this.newNodes[dg.dependencyGraph.TargetOf(edge)])
                    {
                        if (dg.noInit[edge])
                        {
                            EdgeIndex newEdge = g.AddEdge(sourceNoInit, target);
                        }
                        else
                        {
                            EdgeIndex newEdge = g.AddEdge(source, target);
                        }
                        originalEdge.Add(edge);
                    }
                }
            }
            int firstGroup = originalNode.Count;
            if (groupOf != null)
            {
                int firstGroupInDg = dg.dependencyGraph.Nodes.Count;
                int lastGroupInDg = groupOf.Length;
                // construct the mapping from old and new group indices
                for (int i = firstGroupInDg; i < lastGroupInDg; i++)
                {
                    newNodes.Add(new List<NodeIndex>() { originalNode.Count });
                    originalNode.Add(i);
                }
                groupGraph = new GroupGraph(g, groupOf, firstGroup);
                if (forcedForwardEdges != null)
                {
                    AddFreshGroupConstraintEdges(forcedForwardEdges);
                    AddTriggerGroupConstraintEdges(forcedForwardEdges);
                }
            }
            if (forcedForwardEdges != null)
                LabelInitializedEdges(forcedForwardEdges, forcedBackEdges, forceInitializedNodes);
            g.IsReadOnly = true;
            // build group edges only after all other edges have been added
            if (groupGraph != null)
                groupGraph.BuildGroupEdges();
        }

        // not used yet
        private void CreateGraph2(Set<EdgeIndex> forcedForwardEdges)
        {
            g = new IndexedGraph(dg.dependencyGraph.Nodes.Count);
            CreateNodeData();
            this.newNodes = new List<List<NodeIndex>>();
            foreach (NodeIndex node in dg.dependencyGraph.Nodes)
                this.newNodes.Add(new List<NodeIndex>() { node });
            deletedEdges.Clear();
            originalEdge = new List<EdgeIndex>();
            Predicate<NodeIndex> hasTrigger = delegate (NodeIndex node)
            {
                foreach (EdgeIndex edge in dg.dependencyGraph.EdgesInto(node))
                {
                    if (!dg.isDeleted[edge] && dg.isTrigger[edge])
                        return true;
                }
                return false;
            };

            Converter<NodeIndex, IEnumerable<NodeIndex>> sourcesIfTriggered = delegate (NodeIndex node)
            {
                if (hasTrigger(node))
                    return dg.dependencyGraph.SourcesOf(node).Where(edge => !dg.isDeleted[edge]);
                else
                    return new NodeIndex[0];
            };
            var dfsSources = new DepthFirstSearch<NodeIndex>(sourcesIfTriggered, dg.dependencyGraph);
            Set<NodeIndex> sources = new Set<EdgeIndex>();
            Set<NodeIndex> nodesToSkip = new Set<EdgeIndex>();
            dfsSources.FinishNode += delegate (NodeIndex node)
            {
                if (hasTrigger(node))
                    nodesToSkip.Add(node);
                else
                    sources.Add(node);
            };

            Converter<NodeIndex, IEnumerable<NodeIndex>> targetsIfTriggered = delegate (NodeIndex node)
            {
                if (hasTrigger(node))
                    return dg.dependencyGraph.TargetsOf(node).Where(edge => !dg.isDeleted[edge]);
                else
                    return new NodeIndex[0];
            };
            var dfsTargets = new DepthFirstSearch<NodeIndex>(targetsIfTriggered, dg.dependencyGraph);
            Set<NodeIndex> targets = new Set<EdgeIndex>();
            dfsTargets.FinishNode += delegate (NodeIndex node)
            {
                if (hasTrigger(node))
                    nodesToSkip.Add(node);
                else
                    targets.Add(node);
            };

            foreach (NodeIndex node in dg.dependencyGraph.Nodes)
            {
                if (nodesToSkip.Contains(node))
                    continue;
                dfsSources.SearchFrom(node);
                foreach (EdgeIndex edge in dg.dependencyGraph.EdgesOutOf(node))
                {
                    if (dg.isDeleted[edge])
                        continue;
                    dfsTargets.SearchFrom(dg.dependencyGraph.TargetOf(edge));
                }
                // connect every source to every target
                foreach (NodeIndex source in sources)
                {
                    foreach (NodeIndex target in targets)
                    {
                        foreach (NodeIndex newSource in this.newNodes[source])
                        {
                            foreach (NodeIndex newTarget in this.newNodes[target])
                            {
                                EdgeIndex newEdge = g.AddEdge(newSource, newTarget);
                                originalEdge.Add(-1);
                            }
                        }
                    }
                }
            }
            int firstGroup = originalNode.Count;
            if (groupGraph != null)
            {
                int firstGroupInDg = dg.dependencyGraph.Nodes.Count;
                int lastGroupInDg = groupOf.Length;
                // construct the mapping from old and new group indices
                for (int i = firstGroupInDg; i < lastGroupInDg; i++)
                {
                    newNodes.Add(new List<NodeIndex>() { originalNode.Count });
                    originalNode.Add(i);
                }
                groupGraph = new GroupGraph(g, groupOf, firstGroup);
                if (forcedForwardEdges != null)
                {
                    AddFreshGroupConstraintEdges(forcedForwardEdges);
                    AddTriggerGroupConstraintEdges(forcedForwardEdges);
                }
            }
            g.IsReadOnly = true;
            // build group edges only after all other edges have been added
            if (groupGraph != null)
                groupGraph.BuildGroupEdges();
        }

        /// <summary>
        /// Add edges to the graph to enforce fresh constraints between groups
        /// </summary>
        /// <param name="forcedForwardEdges">Modified to have the forced edges</param>
        private void AddFreshGroupConstraintEdges(Set<EdgeIndex> forcedForwardEdges)
        {
            // Any fresh path from a group to itself must have a backward edge.
            // This implies that ancestors of this path must follow descendants of the path.
            List<Edge<NodeIndex>> edgesToAdd = new List<Edge<EdgeIndex>>();
            foreach (NodeIndex source in g.Nodes)
            {
                foreach (EdgeIndex edge in g.EdgesOutOf(source))
                {
                    NodeIndex target = g.TargetOf(edge);
                    Set<NodeIndex> targetGroups = groupGraph.GetGroupSet(target);
                    NodeIndex group = groupGraph.GetLargestGroupExcluding(source, targetGroups);
                    if (group == source)
                        continue;
                    // source is in group, but target is not.
                    // look for a path of fresh edges from target back to source.
                    List<NodeIndex> targets = new List<NodeIndex>();
                    List<NodeIndex> sources = new List<NodeIndex>();
                    DepthFirstSearch<NodeIndex> dfsTargets = new DepthFirstSearch<EdgeIndex>(FreshTargetsOf, g);
                    dfsTargets.FinishNode += delegate (NodeIndex node)
                    {
                        if (groupGraph.InGroup(node, group))
                            targets.Add(node);
                    };
                    dfsTargets.SearchFrom(target);
                    if (targets.Count > 0)
                    {
                        // found a fresh path
                        sources.Add(source);
                        if (IsFreshEdge(edge))
                        {
                            // this edge is part of a longer fresh path.  find all sources of this path.
                            DepthFirstSearch<NodeIndex> dfsSources = new DepthFirstSearch<EdgeIndex>(FreshSourcesOf, g);
                            dfsSources.FinishNode += delegate (NodeIndex node)
                            {
                                foreach (NodeIndex parent in g.SourcesOf(node))
                                {
                                    if (groupGraph.InGroup(parent, group))
                                        sources.Add(parent);
                                }
                            };
                            dfsSources.SearchFrom(source);
                        }
                        foreach (NodeIndex target2 in targets)
                        {
                            foreach (NodeIndex source2 in sources)
                            {
                                foreach (NodeIndex source3 in this.newNodes[originalNode[source2]])
                                {
                                    if (source3 != target2)
                                        edgesToAdd.Add(new Edge<EdgeIndex>(target2, source3));
                                }
                            }
                        }
                    }
                }
            }
            // actually add the edges
            foreach (var edge in edgesToAdd)
            {
                NodeIndex source = edge.Source;
                NodeIndex target = edge.Target;
                if (g.ContainsEdge(source, target))
                    continue;
                if (debug)
                    Debug.WriteLine("adding group constraint edge " + edge);
                EdgeIndex newEdge = g.AddEdge(source, target);
                originalEdge.Add(-1);
                forcedForwardEdges.Add(newEdge);
                graphHasVirtualEdges = true;
            }
        }

        /// <summary>
        /// Add edges to the graph to enforce trigger constraints between groups
        /// </summary>
        /// <param name="forcedForwardEdges">Modified to have the forced edges</param>
        private void AddTriggerGroupConstraintEdges(Set<EdgeIndex> forcedForwardEdges)
        {
            // Any trigger path from a group to itself must have a backward edge.
            // This implies that ancestors of this path must follow descendants of the path.
            List<Edge<NodeIndex>> edgesToAdd = new List<Edge<EdgeIndex>>();
            foreach (NodeIndex target in g.Nodes)
            {
                foreach (EdgeIndex edge in g.EdgesInto(target))
                {
                    NodeIndex source = g.SourceOf(edge);
                    Set<NodeIndex> sourceGroups = groupGraph.GetGroupSet(source);
                    NodeIndex group = groupGraph.GetLargestGroupExcluding(target, sourceGroups);
                    if (group == target)
                        continue;
                    // target is in group, but source is not.
                    // look for a path of trigger edges from source back to group.
                    List<NodeIndex> sources = new List<NodeIndex>();
                    List<NodeIndex> targets = new List<NodeIndex>();
                    DepthFirstSearch<NodeIndex> dfsSources = new DepthFirstSearch<EdgeIndex>(TriggersOf, g);
                    dfsSources.FinishNode += delegate (NodeIndex node)
                    {
                        if (groupGraph.InGroup(node, group))
                            sources.Add(node);
                    };
                    dfsSources.SearchFrom(source);
                    if (sources.Count > 0)
                    {
                        // found a fresh path
                        targets.Add(target);
                        if (IsTrigger(edge))
                        {
                            // this edge is part of a longer trigger path.  find all targets of this path.
                            DepthFirstSearch<NodeIndex> dfsTargets = new DepthFirstSearch<EdgeIndex>(TriggeesOf, g);
                            dfsTargets.FinishNode += delegate (NodeIndex node)
                            {
                                foreach (NodeIndex child in g.TargetsOf(node))
                                {
                                    if (groupGraph.InGroup(child, group))
                                        targets.Add(child);
                                }
                            };
                            dfsTargets.SearchFrom(target);
                        }
                        foreach (NodeIndex source2 in sources)
                        {
                            foreach (NodeIndex target2 in targets)
                            {
                                // any equivalent source must also be constrained
                                foreach (NodeIndex source3 in this.newNodes[originalNode[source2]])
                                {
                                    if (source3 != target2)
                                        edgesToAdd.Add(new Edge<EdgeIndex>(target2, source3));
                                }
                            }
                        }
                    }
                }
            }
            // actually add the edges
            foreach (var edge in edgesToAdd)
            {
                NodeIndex source = edge.Source;
                NodeIndex target = edge.Target;
                if (g.ContainsEdge(source, target))
                    continue;
                if (debug)
                    Debug.WriteLine("adding group constraint edge " + edge);
                EdgeIndex newEdge = g.AddEdge(source, target);
                originalEdge.Add(-1);
                forcedForwardEdges.Add(newEdge);
                graphHasVirtualEdges = true;
            }
        }

        private bool AddNoInitEdges()
        {
            if (debug)
                Debug.WriteLine("restoring NoInit edges");
            bool edgesAdded = false;
            g.IsReadOnly = false;
            foreach (EdgeIndex edge in dg.dependencyGraph.Edges)
            {
                if (dg.noInit[edge] && !dg.isDeleted[edge])
                {
                    foreach (NodeIndex source in this.newNodes[dg.dependencyGraph.SourceOf(edge)])
                    {
                        foreach (NodeIndex target in this.newNodes[dg.dependencyGraph.TargetOf(edge)])
                        {
                            if (g.ContainsEdge(source, target))
                                continue;
                            EdgeIndex newEdge = g.AddEdge(source, target);
                            //deletedEdges.Add(newEdge);
                            originalEdge.Add(edge);
                            //direction.Add(Direction.Unknown);
                            direction.Add(Direction.Backward);
                            edgesAdded = true;
                            if (debug)
                                Debug.WriteLine($"labeling NoInit edge {EdgeToString(newEdge)} backward");
                        }
                    }
                }
            }
            g.IsReadOnly = true;
            if (edgesAdded)
                ClearCaches();
            return edgesAdded;
        }

        private bool AddCancelsEdges()
        {
            if (debug)
                Debug.WriteLine("restoring Cancels edges");
            bool edgesAdded = false;
            g.IsReadOnly = false;
            foreach (EdgeIndex edge in dg.dependencyGraph.Edges)
            {
                if (dg.isDeleted[edge] && dg.isCancels[edge])
                {
                    foreach (NodeIndex source in this.newNodes[dg.dependencyGraph.SourceOf(edge)])
                    {
                        foreach (NodeIndex target in this.newNodes[dg.dependencyGraph.TargetOf(edge)])
                        {
                            if (g.ContainsEdge(source, target))
                                continue;
                            EdgeIndex newEdge = g.AddEdge(source, target);
                            deletedEdges.Add(newEdge);
                            originalEdge.Add(edge);
                            //direction.Add(Direction.Unknown);
                            direction.Add(Direction.Backward);
                            edgesAdded = true;
                            if (debug)
                                Debug.WriteLine($"labeling cancels edge {EdgeToString(newEdge)} backward");
                        }
                    }
                }
            }
            g.IsReadOnly = true;
            if (edgesAdded)
                ClearCaches();
            return edgesAdded;
        }

        private bool AddOffsetEdges()
        {
            if (debug)
                Debug.WriteLine("restoring offset edges");
            bool edgesAdded = false;
            g.IsReadOnly = false;
            foreach (EdgeIndex edge in dg.dependencyGraph.Edges)
            {
                if (dg.isDeleted[edge] && IsOffsetEdge(edge))
                {
                    foreach (NodeIndex source in this.newNodes[dg.dependencyGraph.SourceOf(edge)])
                    {
                        foreach (NodeIndex target in this.newNodes[dg.dependencyGraph.TargetOf(edge)])
                        {
                            if (g.ContainsEdge(source, target))
                                continue;
                            EdgeIndex newEdge = g.AddEdge(source, target);
                            originalEdge.Add(edge);
                            direction.Add(Direction.Unknown);
                            //deletedEdges.Add(newEdge);
                            edgesAdded = true;
                            if (debug)
                                Debug.WriteLine("Offset edge: " + EdgeToString(newEdge));
                        }
                    }
                }
            }
            g.IsReadOnly = true;
            if (edgesAdded)
                ClearCaches();
            return edgesAdded;
        }

        private void ClearCaches()
        {
            // clear all caches
            dfsAncestorsWithGroups = null;
            dfsAncestorsWithoutGroups = null;
            dfsDescendantsWithGroups = null;
            dfsDescendantsWithoutGroups = null;
            dfsSchedule = null;
            distance = null;
        }

        private void CreateCompleteGraph(int n)
        {
            g = new IndexedGraph(n);
            foreach (NodeIndex node in g.Nodes)
            {
                foreach (NodeIndex target in g.Nodes)
                {
                    if (node == target)
                        continue;
                    g.AddEdge(node, target);
                }
            }
            g.NodeCountIsConstant = true;
            g.IsReadOnly = true;
        }

        private void CreateGridGraph()
        {
            // 3x3 grid
            g = new IndexedGraph(24);
            g.AddEdge(18, 0);
            g.AddEdge(12, 1);
            g.AddEdge(21, 1);
            g.AddEdge(15, 2);
            g.AddEdge(0, 3);
            g.AddEdge(19, 3);
            g.AddEdge(1, 4);
            g.AddEdge(13, 4);
            g.AddEdge(22, 4);
            g.AddEdge(2, 5);
            g.AddEdge(16, 5);
            g.AddEdge(9, 6);
            g.AddEdge(19, 6);
            g.AddEdge(10, 7);
            g.AddEdge(13, 7);
            g.AddEdge(22, 7);
            g.AddEdge(11, 8);
            g.AddEdge(16, 8);
            g.AddEdge(20, 9);
            g.AddEdge(14, 10);
            g.AddEdge(23, 10);
            g.AddEdge(17, 11);
            g.AddEdge(6, 12);
            g.AddEdge(0, 13);
            g.AddEdge(9, 13);
            g.AddEdge(3, 14);
            g.AddEdge(7, 15);
            g.AddEdge(12, 15);
            g.AddEdge(1, 16);
            g.AddEdge(10, 16);
            g.AddEdge(13, 16);
            g.AddEdge(4, 17);
            g.AddEdge(14, 17);
            g.AddEdge(7, 18);
            g.AddEdge(21, 18);
            g.AddEdge(1, 19);
            g.AddEdge(10, 19);
            g.AddEdge(22, 19);
            g.AddEdge(4, 20);
            g.AddEdge(23, 20);
            g.AddEdge(8, 21);
            g.AddEdge(2, 22);
            g.AddEdge(11, 22);
            g.AddEdge(5, 23);
            g.NodeCountIsConstant = true;
            g.IsReadOnly = true;
        }

        private IndexedGraph ShuffleGraph(IndexedGraph g)
        {
            int n = g.Nodes.Count;
            int[] newNode = Rand.Perm(n);
            List<int> oldOriginalNode = new List<int>(originalNode);
            IndexedGraph g2 = new IndexedGraph(n);
            foreach (NodeIndex source in g.Nodes)
            {
                NodeIndex source2 = newNode[source];
                originalNode[source2] = oldOriginalNode[source];
                foreach (NodeIndex target in g.TargetsOf(source))
                {
                    NodeIndex target2 = newNode[target];
                    g2.AddEdge(source2, target2);
                }
            }
            g2.IsReadOnly = true;
            return g2;
        }

        private void CreateNodeData()
        {
            originalNode = new List<NodeIndex>();
            foreach (NodeIndex node in g.Nodes)
            {
                originalNode.Add(node);
            }
        }

        private void CreateEdgeData()
        {
            direction = new List<Direction>();
            int edgeCount = g.EdgeCount();
            for (int i = 0; i < edgeCount; i++)
            {
                direction.Add(Direction.Unknown);
            }
        }

        private void DrawReducedGraph(IndexedGraph g)
        {
            if (InferenceEngine.Visualizer?.DependencyGraphVisualizer != null)
            {
                Set<NodeIndex> group = new Set<EdgeIndex>();
                Dictionary<NodeIndex, string> nodeNames = new Dictionary<EdgeIndex, string>();
                foreach (NodeIndex node in g.Nodes)
                    nodeNames[node] = node.ToString(CultureInfo.InvariantCulture);
                while (true)
                {
                    group.Clear();
                    // find a node with one source or one target
                    foreach (NodeIndex node in g.Nodes)
                    {
                        if (g.SourceCount(node) == 1)
                        {
                            group.Add(node);
                            group.AddRange(g.SourcesOf(node));
                            break;
                        }
                        else if (g.TargetCount(node) == 1)
                        {
                            group.Add(node);
                            group.AddRange(g.TargetsOf(node));
                            break;
                        }
                    }
                    if (group.Count == 0)
                        break;
                    // merge the nodes
                    Dictionary<NodeIndex, NodeIndex> mergedIndex;
                    g = MergeNodes(g, group, out mergedIndex);
                    // compute the new node names
                    Dictionary<NodeIndex, string> newNames = new Dictionary<EdgeIndex, string>();
                    foreach (KeyValuePair<NodeIndex, NodeIndex> entry in mergedIndex)
                    {
                        string newName;
                        if (newNames.TryGetValue(entry.Value, out newName))
                        {
                            newNames[entry.Value] = newName + "," + nodeNames[entry.Key];
                        }
                        else
                        {
                            newNames[entry.Value] = nodeNames[entry.Key];
                        }
                    }
                    nodeNames = newNames;
                }
                InferenceEngine.Visualizer.DependencyGraphVisualizer.VisualizeDependencyGraph(g, null, node => nodeNames[node]);
            }
        }

        private IndexedGraph MergeNodes(IndexedGraph g, Set<NodeIndex> group, out Dictionary<NodeIndex, NodeIndex> mergedIndex)
        {
            IndexedGraph g2 = new IndexedGraph(g.Nodes.Count - group.Count + 1);
            mergedIndex = new Dictionary<EdgeIndex, EdgeIndex>();
            int count = 1;
            int groupIndex = 0;
            foreach (NodeIndex node in g.Nodes)
            {
                if (group.Contains(node))
                {
                    mergedIndex[node] = groupIndex;
                }
                else
                    mergedIndex[node] = count++;
            }
            foreach (EdgeIndex edge in g.Edges)
            {
                NodeIndex source = g.SourceOf(edge);
                NodeIndex target = g.TargetOf(edge);
                NodeIndex source2 = mergedIndex[source];
                NodeIndex target2 = mergedIndex[target];
                if (source2 != target2 && !g2.ContainsEdge(source2, target2))
                    g2.AddEdge(source2, target2);
            }
            return g2;
        }

        /// <summary>
        /// Check if an edge in g is required in dg.dependencyGraph
        /// </summary>
        /// <param name="edge"></param>
        /// <param name="includeAny"></param>
        /// <returns></returns>
        private bool IsRequired(EdgeIndex edge, bool includeAny = false)
        {
            if (useFakeGraph)
                return false;
            NodeIndex target = originalNode[g.TargetOf(edge)];
            EdgeIndex edgeOrig = originalEdge[edge];
            if (target == -1 || edgeOrig == -1)
                return false;
            if (includeAny && ((dg.bitsProvided[edgeOrig] & dg.requiredBits[target]) != 0))
                return true;
            return dg.isRequired[edgeOrig];
        }

        /// <summary>
        /// Check if an edge in g is labeled NoInit in dg.dependencyGraph
        /// </summary>
        /// <param name="edge"></param>
        /// <returns></returns>
        private bool IsNoInit(EdgeIndex edge)
        {
            if (useFakeGraph)
                return false;
            EdgeIndex edgeOrig = originalEdge[edge];
            if (edgeOrig == -1)
                return false;
            return dg.noInit[edgeOrig];
        }

        private bool IsFreshEdge(EdgeIndex edge)
        {
            if (useFakeGraph)
                return false;
            EdgeIndex edgeOrig = originalEdge[edge];
            if (edgeOrig == -1)
                return false;
            return dg.isFreshEdge[edgeOrig];
        }

        private IEnumerable<NodeIndex> FreshSourcesOf(NodeIndex node)
        {
            foreach (EdgeIndex edge in g.EdgesInto(node))
            {
                if (!IsFreshEdge(edge))
                    continue;
                yield return g.SourceOf(edge);
            }
        }

        private IEnumerable<NodeIndex> FreshTargetsOf(NodeIndex node)
        {
            foreach (EdgeIndex edge in g.EdgesOutOf(node))
            {
                if (!IsFreshEdge(edge))
                    continue;
                yield return g.TargetOf(edge);
            }
        }

        private bool IsTrigger(EdgeIndex edge)
        {
            if (useFakeGraph)
                return false;
            EdgeIndex edgeOrig = originalEdge[edge];
            if (edgeOrig == -1)
                return false;
            return dg.isTrigger[edgeOrig];
        }

        private IEnumerable<NodeIndex> TriggeesOf(NodeIndex node)
        {
            foreach (EdgeIndex edge in g.EdgesOutOf(node))
            {
                if (!IsTrigger(edge))
                    continue;
                yield return g.TargetOf(edge);
            }
        }

        private IEnumerable<NodeIndex> TriggersOf(NodeIndex node)
        {
            foreach (EdgeIndex edge in g.EdgesInto(node))
            {
                if (!IsTrigger(edge))
                    continue;
                yield return g.SourceOf(edge);
            }
        }

        private void WriteCycle(Cycle c)
        {
            WriteCycle(c, g);
        }

        private void WriteCycle(Cycle c, IndexedGraph g)
        {
            Trace.Write("cycle:");
            foreach (EdgeIndex edge in c)
            {
                NodeIndex source = g.SourceOf(edge);
                Trace.Write(" ");
                Trace.Write(source);
            }
            Trace.WriteLine("");
        }

        private Cycle GetCycle(IList<NodeIndex> nodes)
        {
            Cycle cycle = new Cycle();
            for (int i = 0; i < nodes.Count; i++)
            {
                NodeIndex prevNode = nodes[(i == 0) ? (nodes.Count - 1) : (i - 1)];
                EdgeIndex edge = g.GetEdge(prevNode, nodes[i]);
                cycle.Add(edge);
            }
            return cycle;
        }

        private IEnumerable<NodeIndex> RequiredTargets(NodeIndex node)
        {
            foreach (EdgeIndex edge in dg.dependencyGraph.EdgesOutOf(node))
            {
                if (dg.isRequired[edge])
                {
                    NodeIndex target = dg.dependencyGraph.TargetOf(edge);
                    yield return target;
                }
            }
        }

        private IEnumerable<NodeIndex> ForwardSourcesAndForwardTargets(NodeIndex node)
        {
            foreach (EdgeIndex edge in g.EdgesInto(node))
            {
                if (direction[edge] == Direction.Forward)
                {
                    NodeIndex source = g.SourceOf(edge);
                    if (source != node)
                        yield return source;
                }
            }
            foreach (EdgeIndex edge in g.EdgesOutOf(node))
            {
                if (direction[edge] == Direction.Forward)
                {
                    NodeIndex target = g.TargetOf(edge);
                    if (target != node)
                        yield return target;
                }
            }
        }

        private IEnumerable<NodeIndex> ForwardSources(NodeIndex node)
        {
            foreach (EdgeIndex edge in g.EdgesInto(node))
            {
                if (direction[edge] == Direction.Forward)
                {
                    NodeIndex source = g.SourceOf(edge);
                    if (source != node)
                        yield return source;
                }
            }
        }

        private IEnumerable<NodeIndex> ForwardSourcesAndBackwardTargets(NodeIndex node)
        {
            foreach (EdgeIndex edge in g.EdgesInto(node))
            {
                if (direction[edge] == Direction.Forward && !deletedEdges.Contains(edge))
                {
                    NodeIndex source = g.SourceOf(edge);
                    if (source != node)
                        yield return source;
                }
            }
            foreach (EdgeIndex edge in g.EdgesOutOf(node))
            {
                if (direction[edge] == Direction.Backward && !deletedEdges.Contains(edge))
                {
                    NodeIndex target = g.TargetOf(edge);
                    if (target != node)
                        yield return target;
                }
            }
        }

        /// <summary>
        /// result does not include node
        /// </summary>
        /// <param name="node">Node in dg</param>
        /// <returns></returns>
        private Set<NodeIndex> GetGroupsInDg(NodeIndex node)
        {
            Set<NodeIndex> groups = new Set<EdgeIndex>();
            ForEachGroupOfDg(node, groups.Add);
            return groups;
        }

        /// <summary>
        /// Invokes action on each group of node (excluding node itself)
        /// </summary>
        /// <param name="node">Node in g</param>
        /// <param name="action">Accepts group in g</param>
        private void ForEachGroupOf(NodeIndex node, Action<NodeIndex> action)
        {
            ForEachGroupOfDg(this.originalNode[node], g => action(GetNewGroup(g)));
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="node">Node in dg</param>
        /// <param name="action">Accepts group in dg</param>
        private void ForEachGroupOfDg(NodeIndex node, Action<NodeIndex> action)
        {
            if (this.groupOf != null)
            {
                NodeIndex group = node;
                while (true)
                {
                    group = groupOf[group];
                    if (group == -1)
                        break;
                    action(group);
                }
            }
        }

        // as we traverse the graph, we will always stay at the highest level of abstraction (the largest group) that we can.
        private IEnumerable<NodeIndex> ForwardSourcesAndBackwardTargetsWithGroups(NodeIndex node)
        {
            Set<NodeIndex> groups = groupGraph.GetGroupSet(node);
            foreach (EdgeIndex edge in groupGraph.EdgesInto(node))
            {
                if (direction[edge] == Direction.Forward && !deletedEdges.Contains(edge))
                {
                    NodeIndex source = g.SourceOf(edge);
                    if (source != node)
                        yield return groupGraph.GetLargestGroupExcluding(source, groups);
                }
            }
            foreach (EdgeIndex edge in groupGraph.EdgesOutOf(node))
            {
                if (direction[edge] == Direction.Backward && !deletedEdges.Contains(edge))
                {
                    NodeIndex target = g.TargetOf(edge);
                    if (target != node)
                        yield return groupGraph.GetLargestGroupExcluding(target, groups);
                }
            }
        }

        // as we traverse the graph, we will always stay at the highest level of abstraction (the largest group) that we can.
        private IEnumerable<NodeIndex> ForwardSourcesAndForwardTargetsWithGroups(NodeIndex node)
        {
            Set<NodeIndex> groups = groupGraph.GetGroupSet(node);
            foreach (EdgeIndex edge in groupGraph.EdgesInto(node))
            {
                if (direction[edge] == Direction.Forward)
                {
                    NodeIndex source = g.SourceOf(edge);
                    if (source != node)
                        yield return groupGraph.GetLargestGroupExcluding(source, groups);
                }
            }
            foreach (EdgeIndex edge in groupGraph.EdgesOutOf(node))
            {
                if (direction[edge] == Direction.Forward)
                {
                    NodeIndex target = g.TargetOf(edge);
                    if (target != node)
                        yield return groupGraph.GetLargestGroupExcluding(target, groups);
                }
            }
        }

        // as we traverse the graph, we will always stay at the highest level of abstraction (the largest group) that we can.
        private IEnumerable<NodeIndex> ForwardSourcesWithGroups(NodeIndex node)
        {
            Set<NodeIndex> groups = groupGraph.GetGroupSet(node);
            foreach (EdgeIndex edge in groupGraph.EdgesInto(node))
            {
                if (direction[edge] == Direction.Forward)
                {
                    NodeIndex source = g.SourceOf(edge);
                    if (source != node)
                        yield return groupGraph.GetLargestGroupExcluding(source, groups);
                }
            }
        }

        // as we traverse the graph, we will always stay at the highest level of abstraction (the largest group) that we can.
        private IEnumerable<NodeIndex> BackwardSourcesAndForwardTargetsWithGroups(NodeIndex node)
        {
            Set<NodeIndex> groups = groupGraph.GetGroupSet(node);
            foreach (EdgeIndex edge in groupGraph.EdgesInto(node))
            {
                if (direction[edge] == Direction.Backward)
                {
                    NodeIndex source = g.SourceOf(edge);
                    if (source != node)
                        yield return groupGraph.GetLargestGroupExcluding(source, groups);
                }
            }
            foreach (EdgeIndex edge in groupGraph.EdgesOutOf(node))
            {
                if (direction[edge] == Direction.Forward)
                {
                    NodeIndex target = g.TargetOf(edge);
                    if (target != node)
                        yield return groupGraph.GetLargestGroupExcluding(target, groups);
                }
            }
        }

        private IEnumerable<NodeIndex> BackwardSourcesAndForwardTargets(NodeIndex node)
        {
            foreach (EdgeIndex edge in g.EdgesInto(node))
            {
                if (direction[edge] == Direction.Backward)
                {
                    NodeIndex source = g.SourceOf(edge);
                    if (source != node)
                        yield return source;
                }
            }
            foreach (EdgeIndex edge in g.EdgesOutOf(node))
            {
                if (direction[edge] == Direction.Forward)
                {
                    NodeIndex target = g.TargetOf(edge);
                    if (target != node)
                        yield return target;
                }
            }
        }

        private IEnumerable<NodeIndex> ForwardTargets(NodeIndex node)
        {
            foreach (EdgeIndex edge in g.EdgesOutOf(node))
            {
                if (direction[edge] == Direction.Forward)
                {
                    NodeIndex target = g.TargetOf(edge);
                    if (target != node)
                        yield return target;
                }
            }
        }

        private string NodeName(NodeIndex node)
        {
            if (dg.initializedNodes.Contains(node))
                return node.ToString(CultureInfo.InvariantCulture) + "*";
            NodeIndex orig = originalNode[node];
            if (orig == node)
                return node.ToString(CultureInfo.InvariantCulture);
            else
                return node.ToString(CultureInfo.InvariantCulture) + " (" + orig + ")";
        }

        private static void CheckStronglyConnected(IndexedGraph g)
        {
            List<Set<NodeIndex>> blocks = new List<Set<NodeIndex>>();
            Set<NodeIndex> currentBlock = null;
            StrongComponents2<NodeIndex> scc = new StrongComponents2<NodeIndex>(g.SourcesOf, g);
            scc.AddNode += delegate (NodeIndex node)
            {
                currentBlock.Add(node);
            };
            scc.BeginComponent += delegate ()
            {
                currentBlock = new Set<NodeIndex>();
            };
            scc.EndComponent += delegate ()
            {
                blocks.Add(currentBlock);
            };
            scc.SearchFrom(g.Nodes);
            if (blocks.Count > 1)
                throw new Exception("graph is not strongly connected");
        }

        private List<T> Shuffle<T>(IList<T> list)
        {
            List<T> result = new List<T>();
            int[] perm = Microsoft.ML.Probabilistic.Math.Rand.Perm(list.Count);
            for (int i = 0; i < perm.Length; i++)
            {
                result.Add(list[perm[i]]);
            }
            return result;
        }

        private T[] Shuffle<T>(T[] array)
        {
            T[] result = new T[array.Length];
            int[] perm = Microsoft.ML.Probabilistic.Math.Rand.Perm(array.Length);
            for (int i = 0; i < perm.Length; i++)
            {
                result[i] = array[perm[i]];
            }
            return result;
        }

        private List<NodeIndex> GetSchedule(bool useGroups)
        {
            if (groupGraph == null)
                useGroups = false;
            if (useGroups)
                return groupGraph.GetScheduleWithGroups(ForwardSourcesAndBackwardTargetsWithGroups);
            // toposort the forward edges to get a schedule
            this.schedule = new List<NodeIndex>();
            if (dfsSchedule == null)
            {
                dfsSchedule = new DepthFirstSearch<NodeIndex>(ForwardSourcesAndBackwardTargets, g);
                dfsSchedule.BackEdge += delegate (Edge<NodeIndex> edge)
                    {
                        List<NodeIndex> cycle = new Cycle();
                        cycle.Add(edge.Target);
                        bool found = false;
                        dfsSchedule.ForEachStackNode(delegate (NodeIndex node)
                            {
                                if (node == edge.Target)
                                    found = true;
                                if (!found)
                                    cycle.Add(node);
                            });
                        //cycle.Reverse();
                        Console.WriteLine("cycle: " + StringUtil.CollectionToString(cycle, " "));
                        if (!debug)
                            throw new InferCompilerException("Cycle of forward edges");
                    };
                dfsSchedule.FinishNode += node => this.schedule.Add(node);
            }
            else
            {
                dfsSchedule.Clear();
            }
            dfsSchedule.SearchFrom(g.Nodes);
            return schedule;
        }

        private void AssignLabelsFromSchedule(IEnumerable<NodeIndex> schedule)
        {
            // relabel the labeled edges according to the schedule
            bool[] isScheduled = new bool[g.Nodes.Count];
            foreach (NodeIndex nodeOrig in schedule)
            {
                foreach (NodeIndex node in this.newNodes[nodeOrig])
                {
                    if (node == -1 || isScheduled[node])
                        continue;
                    foreach (EdgeIndex edge in g.EdgesInto(node))
                    {
                        NodeIndex source = g.SourceOf(edge);
                        if (isScheduled[source])
                        {
                            direction[edge] = Direction.Forward;
                        }
                        else
                        {
                            direction[edge] = Direction.Backward;
                        }
                    }
                    isScheduled[node] = true;
                }
            }
        }

        private void DrawLabeledGraph(string title, bool inThread = false, float[] edgeCosts = null, bool showNoInit = false)
        {
            if (InferenceEngine.Visualizer?.DependencyGraphVisualizer != null)
            {
                var edgeStyles = new List<EdgeStylePredicate>() {
                //new DependencyGraphView.EdgeStylePredicate("Required", edge => IsRequired(edge, true), DependencyGraphView.EdgeStyle.Bold),
                new EdgeStylePredicate("Trigger", IsTrigger, EdgeStyle.Bold),
                new EdgeStylePredicate("Fresh", IsFreshEdge, EdgeStyle.Dashed),
                new EdgeStylePredicate("Backward", edge => direction[edge] == Direction.Backward, EdgeStyle.Back),
                new EdgeStylePredicate("Deleted", edge => deletedEdges.Contains(edge), EdgeStyle.Dimmed),
            };
                if (showNoInit)
                {
                    edgeStyles.Add(
                        new EdgeStylePredicate("NoInit", IsNoInit, EdgeStyle.Blue)
                    );
                }
                else
                {
                    edgeStyles.Add(
                        new EdgeStylePredicate("Unlabeled", edge => direction[edge] == Direction.Unknown, EdgeStyle.Blue)
                    );
                }
                Func<EdgeIndex, string> edgeName = null;
                if (edgeCosts != null)
                    edgeName = edge => (edge < edgeCosts.Length) ? DoubleToString(edgeCosts[edge]) : "";
                if (inThread)
                {
                    Thread viewThread = new Thread(delegate ()
                    {
                        InferenceEngine.Visualizer.DependencyGraphVisualizer.VisualizeDependencyGraph(g, edgeStyles, NodeName, edgeName, title);
                    });
                    viewThread.Start();
                }
                else
                {
                    InferenceEngine.Visualizer.DependencyGraphVisualizer.VisualizeDependencyGraph(g, edgeStyles, NodeName, edgeName, title);
                }
            }
        }

        /// <summary>
        /// Represents a double to (at least) one digit of precision, using the minimum number of characters
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private static string DoubleToString(double x)
        {
            if (x == 0)
                return "0";
            else if (double.IsPositiveInfinity(x))
                return "∞";
            else if (System.Math.Abs(x) < 1e-2)
            {
                return x.ToString("0e0");
            }
            else if (System.Math.Abs(x) < 1)
            {
                return x.ToString("g1");
            }
            else
            {
                x = System.Math.Round(x);
                if (System.Math.Abs(x) >= 1000)
                    return x.ToString("0e0");
                else
                    return x.ToString("g");
            }
        }

        private static void DoubleToStringTest()
        {
            Console.WriteLine(DoubleToString(-123456));
            Console.WriteLine(DoubleToString(-12345));
            Console.WriteLine(DoubleToString(-1234));
            Console.WriteLine(DoubleToString(-123));
            Console.WriteLine(DoubleToString(0.01234));
            Console.WriteLine(DoubleToString(0.001234));
            Console.WriteLine(DoubleToString(0.0001234));
            Console.WriteLine(DoubleToString(0.00001234));
        }

        private void DrawOriginalGraph(bool inThread = false)
        {
            if (InferenceEngine.Visualizer?.DependencyGraphVisualizer != null)
            {
                var edgeStyles = new EdgeStylePredicate[] {
                    new EdgeStylePredicate("Trigger", edge => dg.isTrigger[edge], EdgeStyle.Bold),
                    new EdgeStylePredicate("Fresh", edge => dg.isFreshEdge[edge], EdgeStyle.Dashed)
                };
                if (inThread)
                {
                    Thread viewThread = new Thread(delegate ()
                    {
                        InferenceEngine.Visualizer.DependencyGraphVisualizer.VisualizeDependencyGraph(dg.dependencyGraph, edgeStyles);
                    });
                    viewThread.Start();
                }
                else
                {
                    InferenceEngine.Visualizer.DependencyGraphVisualizer.VisualizeDependencyGraph(dg.dependencyGraph, edgeStyles);
                }
            }
        }

        private List<Cycle> FindCycles(IndexedGraph g)
        {
            List<Cycle> cycles = new List<Cycle>();
            var cf = new CycleFinder<NodeIndex, EdgeIndex>(g);
            Cycle currentCycle = null;
            cf.AddEdge += delegate (EdgeIndex edge)
            {
                currentCycle.Add(edge);
            };
            cf.BeginCycle += delegate ()
            {
                currentCycle = new Cycle();
            };
            cf.EndCycle += delegate ()
                {
                    cycles.Add(currentCycle);
                    if (verbose && cycles.Count % 10000 == 0)
                        Debug.WriteLine(cycles.Count);
                };
            cf.Search();
            return cycles;
        }

        private List<Cycle> FindCycles2(IndexedGraph g)
        {
            List<Cycle> cycles = new List<Cycle>();
            var cf = new CycleFinder2<NodeIndex, EdgeIndex>(g);
            Cycle currentCycle = null;
            cf.AddEdge += delegate (EdgeIndex edge)
            {
                currentCycle.Add(edge);
            };
            cf.BeginCycle += delegate ()
            {
                currentCycle = new Cycle();
            };
            cf.EndCycle += delegate ()
                {
                    cycles.Add(currentCycle);
                    if (verbose && cycles.Count % 10000 == 0)
                        Debug.WriteLine(cycles.Count);
                    //if (cycles.Count > 10000) throw new Exception("too many cycles");
                };
            cf.Search();
            return cycles;
        }
    }

    // find a set of cycles that covers every edge of a graph
    internal class CycleFinder2<NodeType, EdgeType>
    {
        private IDirectedGraph<NodeType, EdgeType> graph;
        private IndexedProperty<NodeType, bool> isBlocked;
        private IndexedProperty<NodeType, Set<NodeType>> blockedSources;
        private IndexedProperty<EdgeType, bool> excluded;
        private Stack<EdgeType> stack = new Stack<EdgeType>();
        public event Action<EdgeType> AddEdge;
        public event Action BeginCycle, EndCycle;

        public CycleFinder2(IDirectedGraph<NodeType, EdgeType> graph)
        {
            this.graph = graph;
            CanCreateNodeData<NodeType> data = (CanCreateNodeData<NodeType>)graph;
            isBlocked = data.CreateNodeData<bool>(false);
            blockedSources = data.CreateNodeData<Set<NodeType>>(null);
            excluded = ((CanCreateEdgeData<EdgeType>)graph).CreateEdgeData<bool>(false);
        }

        public void Search()
        {
            foreach (NodeType node in graph.Nodes)
            {
                SearchFrom(node);
            }
        }

        public void SearchFrom(NodeType node)
        {
            SearchFrom(node, node);
            foreach (NodeType node2 in graph.Nodes)
            {
                isBlocked[node2] = false;
                blockedSources[node2] = null;
            }
            // check that every edge was covered
            foreach (EdgeType edge in graph.EdgesOutOf(node))
            {
                if (!excluded[edge])
                    throw new Exception("edge is not on a cycle");
            }
        }

        /// <summary>
        /// Find cycles containing root 
        /// </summary>
        /// <param name="node"></param>
        /// <param name="root"></param>
        /// <returns></returns>
        private bool SearchFrom(NodeType node, NodeType root)
        {
            bool nodeIsRoot = node.Equals(root);
            bool foundCycle = false;
            isBlocked[node] = true;
            List<EdgeType> edgesOut = new List<EdgeType>();
            // on the first try, we only consider non-excluded edges
            // on the second try, we only consider excluded edges
            for (int trial = 0; trial < 2; trial++)
            {
                foreach (EdgeType edge in graph.EdgesOutOf(node))
                {
                    if (trial == 0)
                    {
                        if (excluded[edge])
                            continue;
                    }
                    else
                    {
                        if (!excluded[edge])
                            continue;
                    }
                    edgesOut.Add(edge);
                }
                // do not consider excluded edges if node = root
                if (nodeIsRoot)
                    break;
            }
            foreach (EdgeType edge in edgesOut)
            {
                NodeType target = graph.TargetOf(edge);
                // ignore self-loops
                if (!nodeIsRoot && target.Equals(node))
                    continue;
                if (target.Equals(root))
                {
                    foundCycle = true;
                    OnBeginCycle();
                    Stack<EdgeType> temp = new Stack<EdgeType>();
                    foreach (EdgeType edgeOnStack in stack)
                    {
                        temp.Push(edgeOnStack);
                    }
                    foreach (EdgeType edgeOnStack in temp)
                    {
                        OnAddEdge(edgeOnStack);
                    }
                    OnAddEdge(edge);
                    OnEndCycle();
                }
                else if (!isBlocked[target])
                {
                    // recursive call
                    stack.Push(edge);
                    if (SearchFrom(target, root))
                        foundCycle = true;
                    stack.Pop();
                }
                // once a cycle back to the root is found, we don't need to look for more
                if (!nodeIsRoot && foundCycle)
                {
                    break;
                }
            }
            // at this point, we could always set isBlocked[node]=false,
            // but as an optimization we leave it set if no cycle was discovered,
            // to prevent repeated searching of the same paths.
            if (foundCycle)
                Unblock(node);
            else
            {
                // at this point, all targets are blocked
                foreach (NodeType target in graph.TargetsOf(node))
                {
                    Set<NodeType> blockedSourcesOfTarget = blockedSources[target];
                    if (blockedSourcesOfTarget == null)
                    {
                        blockedSourcesOfTarget = new Set<NodeType>();
                        blockedSources[target] = blockedSourcesOfTarget;
                    }
                    blockedSourcesOfTarget.Add(node);
                }
            }
            return foundCycle;
        }

        private void Unblock(NodeType node)
        {
            isBlocked[node] = false;
            Set<NodeType> blockedSourcesOfNode = blockedSources[node];
            if (blockedSourcesOfNode != null)
            {
                blockedSources[node] = null;
                foreach (NodeType source in blockedSourcesOfNode)
                {
                    if (isBlocked[source])
                        Unblock(source);
                }
            }
        }

        public void OnAddEdge(EdgeType edge)
        {
            excluded[edge] = true;
            AddEdge?.Invoke(edge);
        }

        public void OnBeginCycle()
        {
            BeginCycle?.Invoke();
        }

        public void OnEndCycle()
        {
            EndCycle?.Invoke();
        }
    }
}