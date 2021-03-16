// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using NodeIndex = System.Int32;
using EdgeIndex = System.Int32;
using Microsoft.ML.Probabilistic.Compiler;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// Represents a graph in which each node may belong to a group, and each group may belong to another group.
    /// Group membership must be acyclic.
    /// Nodes and groups are identified by integers.
    /// An integer n identifies a group if two conditions hold:
    /// 1. n is at least firstGroup and less than lastGroup.
    /// 2. n is at least g.Nodes.Count, or g.NeighborCount(n) == 0.
    /// </summary>
    internal class GroupGraph : //IMutableDirectedGraph<int, int>, IMultigraph<int, int>,
                                  CanCreateNodeData<int>, CanCreateEdgeData<int>
    {
        internal readonly IndexedGraph g;
        // maps a node or group index into a group index (or -1 if no group)
        internal readonly IList<NodeIndex> groupOf;
        /// <summary>
        /// firstGroup is at most g.Nodes.Count.
        /// </summary>
        internal readonly int firstGroup, lastGroup;
        /// <summary>
        /// The set of edges whose target is in the group and source is not.  Indexed by (group - firstGroup).  A null indicates a non-group.
        /// </summary>
        internal readonly IList<ICollection<EdgeIndex>> edgesIntoGroup, edgesOutOfGroup;
        private DepthFirstSearch<NodeIndex> dfsScheduleWithGroups;
        private List<NodeIndex> groupSchedule;

        /// <summary>
        /// Caller must subsequently call BuildGroupEdges
        /// </summary>
        /// <param name="g"></param>
        /// <param name="groupOf"></param>
        /// <param name="firstGroup"></param>
        internal GroupGraph(IndexedGraph g, IList<NodeIndex> groupOf, int firstGroup)
        {
            this.g = g;
            this.groupOf = groupOf;
            if (firstGroup > g.Nodes.Count) throw new ArgumentException("firstGroup > g.Nodes.Count");
            this.firstGroup = firstGroup;
            this.lastGroup = groupOf.Count;
            this.edgesIntoGroup = new List<ICollection<EdgeIndex>>();
            this.edgesOutOfGroup = new List<ICollection<EdgeIndex>>();
        }

        public IndexedProperty<int, T> CreateEdgeData<T>(T defaultValue)
        {
            throw new NotImplementedException();
        }

        public IndexedProperty<EdgeIndex, T> CreateNodeData<T>(T defaultValue)
        {
            return MakeIndexedProperty.FromArray(new T[lastGroup], defaultValue);
        }

        public bool IsGroup(NodeIndex node)
        {
            return (node >= firstGroup) && (node >= g.Nodes.Count || g.NeighborCount(node) == 0);
        }

        public IEnumerable<NodeIndex> EdgesInto(NodeIndex node)
        {
            if (IsGroup(node))
            {
                // node is actually a group
                int groupIndex = node - firstGroup;
                var edges = edgesIntoGroup[groupIndex];
                if (edges != null)
                    return edges;
                else
                    return new NodeIndex[0];
            }
            else 
            {
                return g.EdgesInto(node);
            }
        }

        public IEnumerable<NodeIndex> EdgesOutOf(NodeIndex node)
        {
            if (IsGroup(node))
            {
                // node is actually a group
                int groupIndex = node - firstGroup;
                var edges = edgesOutOfGroup[groupIndex];
                if (edges != null)
                    return edges;
                else 
                    return new NodeIndex[0];
            }
            else 
            {
                return g.EdgesOutOf(node);
            }
        }

        public NodeIndex TargetOf(EdgeIndex edge)
        {
            Set<NodeIndex> groups = GetGroupSet(g.SourceOf(edge));
            return GetLargestGroupExcluding(g.TargetOf(edge), groups);
        }

        // as we traverse the graph, we will always stay at the highest level of abstraction (the largest group) that we can.
        public IEnumerable<NodeIndex> SourcesOf(NodeIndex node)
        {
            Set<NodeIndex> groups = GetGroupSet(node);
            foreach (EdgeIndex edge in EdgesInto(node))
            {
                NodeIndex source = g.SourceOf(edge);
                if (source != node)
                    yield return GetLargestGroupExcluding(source, groups);
            }
        }

        // as we traverse the graph, we will always stay at the highest level of abstraction (the largest group) that we can.
        public IEnumerable<NodeIndex> TargetsOf(NodeIndex node)
        {
            Set<NodeIndex> groups = GetGroupSet(node);
            foreach (EdgeIndex edge in EdgesOutOf(node))
            {
                NodeIndex target = g.TargetOf(edge);
                if (target != node)
                    yield return GetLargestGroupExcluding(target, groups);
            }
        }

        public void BuildGroupEdges()
        {
            int numGroups = lastGroup - firstGroup;
            while (edgesIntoGroup.Count < numGroups)
            {
                edgesIntoGroup.Add(new Set<EdgeIndex>());
                edgesOutOfGroup.Add(new Set<EdgeIndex>());
            }
            foreach (NodeIndex target in g.Nodes)
            {
                Set<NodeIndex> targetGroups = GetGroupSet(target);
                foreach (EdgeIndex edge in g.EdgesInto(target))
                {
                    NodeIndex source = g.SourceOf(edge);
                    Set<NodeIndex> sourceGroups = GetGroupSet(source);
                    foreach (NodeIndex sourceGroup in sourceGroups)
                    {
                        if (!targetGroups.Contains(sourceGroup))
                            edgesOutOfGroup[sourceGroup - firstGroup].Add(edge);
                    }
                    foreach (NodeIndex targetGroup in targetGroups)
                    {
                        if (!sourceGroups.Contains(targetGroup))
                            edgesIntoGroup[targetGroup - firstGroup].Add(edge);
                    }
                }
            }
        }

        private IEnumerable<EdgeIndex> GetAllEdges(NodeIndex source, NodeIndex target)
        {
            if (IsGroup(source))
            {
                foreach (var edge in edgesOutOfGroup[source - firstGroup])
                {
                    NodeIndex target2 = g.TargetOf(edge);
                    if (target2 == target ||
                        GetGroups(target2).Any(group => group == target))
                        yield return edge;
                }
            }
            else if (IsGroup(target))
            {
                foreach (var edge in edgesIntoGroup[target - firstGroup])
                {
                    NodeIndex source2 = g.SourceOf(edge);
                    if (source2 == source ||
                        GetGroups(source2).Any(group => group == source))
                        yield return edge;
                }
            }
            else yield return g.GetAnyEdge(source, target);
        }

        private EdgeIndex GetAnyEdge(NodeIndex source, NodeIndex target)
        {
            foreach(var edge in GetAllEdges(source, target))
            {
                return edge;
            }
            throw new EdgeNotFoundException(source, target);
        }

        private void CheckGroupEdges()
        {
            for (int node = 0; node < lastGroup; node++)
            {
                if (groupOf[node] != -1 && !IsGroup(groupOf[node])) throw new Exception("!IsGroup(groupOf[node])");
            }
            int numGroups = lastGroup - firstGroup;
            for (int groupIndex = 0; groupIndex < numGroups; groupIndex++)
            {
                NodeIndex group = firstGroup + groupIndex;
                if (edgesIntoGroup[groupIndex] == null) continue;
                foreach(EdgeIndex edge in edgesIntoGroup[groupIndex])
                {
                    NodeIndex source = g.SourceOf(edge);
                    Set<NodeIndex> sourceGroups = GetGroupSet(source);
                    NodeIndex target = g.TargetOf(edge);
                    Set<NodeIndex> targetGroups = GetGroupSet(target);
                    if (!targetGroups.Contains(group)) throw new Exception("!targetGroups.Contains(group)");
                    if (sourceGroups.Contains(group)) throw new Exception("sourceGroups.Contains(group)");
                }
            }
        }

        /// <summary>
        /// Merge two groups that have the same parent group.
        /// </summary>
        /// <param name="group">Group that will receive all nodes in group2.</param>
        /// <param name="group2">Group that will be empty on exit.</param>
        public void MergeGroups(NodeIndex group, NodeIndex group2)
        {
            if (!IsGroup(group)) throw new ArgumentException($"!IsGroup(group)");
            if (!IsGroup(group2)) throw new ArgumentException($"!IsGroup(group2)");
            if (groupOf[group] != groupOf[group2])
                throw new ArgumentException("groups do not have the same parent group");
            if (group == group2)
                return;
            for (int node = 0; node < groupOf.Count; node++)
            {
                if (groupOf[node] == group2)
                {
                    groupOf[node] = group;
                }
            }
            NodeIndex groupIndex = group - firstGroup;
            NodeIndex group2Index = group2 - firstGroup;
            edgesIntoGroup[groupIndex].AddRange(edgesIntoGroup[group2Index]);
            edgesIntoGroup[group2Index] = null;
            edgesOutOfGroup[groupIndex].AddRange(edgesOutOfGroup[group2Index]);
            edgesOutOfGroup[group2Index] = null;
            // remove edges between the two groups
            var edgesToRemove = edgesIntoGroup[groupIndex].Where(edgesOutOfGroup[groupIndex].Contains).ToList();
            foreach (var edge in edgesToRemove)
            {
                edgesIntoGroup[groupIndex].Remove(edge);
                edgesOutOfGroup[groupIndex].Remove(edge);
            }
            //CheckGroupEdges();
        }

        /// <summary>
        /// result does not include node
        /// </summary>
        /// <param name="node">Node in dg</param>
        /// <returns></returns>
        public Set<NodeIndex> GetGroupSet(NodeIndex node)
        {
            return Set<NodeIndex>.FromEnumerable(GetGroups(node));
        }

        /// <summary>
        /// Returns true if node is in group.
        /// </summary>
        /// <param name="node">Node in g</param>
        /// <param name="group">Group in g</param>
        /// <returns></returns>
        public bool InGroup(NodeIndex node, NodeIndex group)
        {
            return GetGroups(node).Any(g => g == group);
        }

        public IEnumerable<NodeIndex> GetGroups(NodeIndex node)
        {
            NodeIndex group = node;
            while (true)
            {
                group = groupOf[group];
                if (group == -1)
                    break;
                yield return group;
            }
        }

        /// <summary>
        /// Get the largest group of node (including node itself) that is not in the set.
        /// </summary>
        /// <param name="node">Node in g</param>
        /// <param name="set">Groups in g</param>
        /// <param name="mustBeInGroup">If true and set is not empty, result must be contained in some group in the set</param>
        /// <returns></returns>
        public NodeIndex GetLargestGroupExcluding(NodeIndex node, Set<NodeIndex> set, bool mustBeInGroup = false)
        {
            if (mustBeInGroup && set.Count == 0)
                return node;
            else
                return GetLargestGroupExcluding(node, set.Contains, mustBeInGroup);
        }

        /// <summary>
        /// Get the largest group of node (including node itself) that belongs to group.
        /// </summary>
        /// <param name="node">Node in g</param>
        /// <param name="group">Group in g, or -1 (to get the largest group of node)</param>
        /// <returns></returns>
        public NodeIndex GetLargestGroupInsideGroup(NodeIndex node, NodeIndex group)
        {
            return GetLargestGroupExcluding(node, group.Equals, mustBeInGroup: (group != -1));
        }

        /// <summary>
        /// Get the largest group of node (including node itself) that does not satisfy a predicate.
        /// </summary>
        /// <param name="node">Node in g</param>
        /// <param name="predicate">Accepts groups in g</param>
        /// <param name="mustBeInGroup">If true, result be in a group (i.e. groupOf[result] != -1)</param>
        /// <returns></returns>
        public NodeIndex GetLargestGroupExcluding(NodeIndex node, Predicate<NodeIndex> predicate, bool mustBeInGroup = false)
        {
            if (groupOf == null)
                return node;
            // group is in g
            NodeIndex group = node;
            while (true)
            {
                NodeIndex nextGroup = groupOf[group];
                if (nextGroup == -1)
                    return mustBeInGroup ? -1 : group;
                if (predicate(nextGroup))
                    return group;
                group = nextGroup;
            }
        }

        /// <summary>
        /// Get the smallest group of node (including node itself) that satisfies the predicate, or -1 if none 
        /// </summary>
        /// <param name="node">Node in g</param>
        /// <param name="predicate">Accepts group in g</param>
        /// <returns></returns>
        public NodeIndex GetSmallestGroup(NodeIndex node, Predicate<NodeIndex> predicate)
        {
            NodeIndex group = node;
            while (true)
            {
                if (predicate(group))
                    return group;
                if (groupOf == null)
                    return -1;
                NodeIndex nextGroup = groupOf[group];
                if (nextGroup == -1)
                    return nextGroup;
                group = nextGroup;
            }
        }

        public List<NodeIndex> GetScheduleWithGroups(Converter<NodeIndex, IEnumerable<NodeIndex>> predecessors)
        {
            // toposort the forward edges to get a schedule
            // algorithm: create a schedule for each group and then stitch them together
            List<NodeIndex> schedule = new List<NodeIndex>();
            if (dfsScheduleWithGroups == null)
            {
                // used by SearchFrom
                dfsScheduleWithGroups = new DepthFirstSearch<NodeIndex>(predecessors, this);
                dfsScheduleWithGroups.BackEdge += delegate (Edge<NodeIndex> edge)
                {
                    List<NodeIndex> cycle = new List<NodeIndex>
                    {
                        edge.Target
                    };
                    bool found = false;
                    dfsScheduleWithGroups.ForEachStackNode(delegate (NodeIndex node)
                    {
                        if (node == edge.Target)
                            found = true;
                        if (!found)
                            cycle.Add(node);
                    });
                    //cycle.Reverse();
                    NodeIndex source = cycle[cycle.Count - 1];
                    Debug.Write("cycle: ");
                    Debug.Write(IsGroup(source) ? $"[{source}] " : $"{source}");
                    foreach (var target in cycle)
                    {
                        foreach (EdgeIndex edge2 in GetAllEdges(source, target))
                        {
                            if (IsGroup(source))
                            {
                                Debug.Write($"{g.SourceOf(edge2)}");
                            }
                            Debug.Write($"->{g.TargetOf(edge2)} ");
                        }
                        if(IsGroup(target))
                        {
                            Debug.Write($"[{target}] ");
                        }
                        source = target;
                    }
                    Debug.WriteLine("");
                    throw new InferCompilerException("Cycle of forward edges");
                };
                dfsScheduleWithGroups.FinishNode += node => groupSchedule.Add(node);
            }
            else
            {
                dfsScheduleWithGroups.Clear();
            }
            // the top-level schedule.  will only contain nodes/groups that are not in groups.
            List<NodeIndex> topSchedule = new List<NodeIndex>();
            Dictionary<NodeIndex, List<NodeIndex>> scheduleOfGroup = new Dictionary<EdgeIndex, List<NodeIndex>>();
            scheduleOfGroup[-1] = topSchedule;
            schedule.Clear();
            // build a schedule by visiting each node and placing all predecessors on the schedule.
            // predecessors are added by DFS FinishNode.
            foreach (NodeIndex node in g.Nodes)
            {
                if(!IsGroup(node))
                    SearchFrom(node, scheduleOfGroup);
            }
            // The top-level schedule may contain references to groups, whose schedule is contained in scheduleOfGroup.
            // Insert the group schedules into the top-level schedule to get one combined schedule.
            ForEachLeafNode(topSchedule, scheduleOfGroup, schedule.Add);
            return schedule;
        }

        /// <summary>
        /// Invoke DFS on a graph node or group.  
        /// If the node/group is in a group, then search from that group first (to ensure all predecessors of the group are scheduled), 
        /// then search from the node, placing the node's results (which must all be within the group) on that group's schedule.
        /// </summary>
        /// <param name="node">Node in g</param>
        /// <param name="scheduleOfGroup">Holds schedule of each group (modified on exit)</param>
        private void SearchFrom(NodeIndex node, Dictionary<NodeIndex, List<NodeIndex>> scheduleOfGroup)
        {
            // first search from all groups of this node
            NodeIndex group = groupOf[node];
            if (group != -1)
                SearchFrom(group, scheduleOfGroup);
            if (!scheduleOfGroup.TryGetValue(group, out groupSchedule))
            {
                groupSchedule = new List<NodeIndex>();
                scheduleOfGroup[group] = groupSchedule;
            }
            dfsScheduleWithGroups.SearchFrom(node);
        }

        /// <summary>
        /// Invoke action on all nodes in order, except for groups which are expanded recursively using scheduleOfGroup
        /// </summary>
        /// <param name="nodes">Nodes in dg</param>
        /// <param name="scheduleOfGroup">Maps groups in dg to nodes in dg</param>
        /// <param name="action">Accepts a node in dg</param>
        private void ForEachLeafNode(List<NodeIndex> nodes, Dictionary<NodeIndex, List<NodeIndex>> scheduleOfGroup, Action<NodeIndex> action)
        {
            foreach (NodeIndex node in nodes)
            {
                if(IsGroup(node))
                {
                    List<NodeIndex> groupSchedule = scheduleOfGroup[node];
                    ForEachLeafNode(groupSchedule, scheduleOfGroup, action);
                }
                else
                    action(node);
            }
        }
    }
}
