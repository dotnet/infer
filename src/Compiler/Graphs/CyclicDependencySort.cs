// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// Order the nodes to best satisfy cyclic dependencies.
    /// </summary>
    /// <remarks><p>
    /// The algorithm is essentially a topological sort, modified to deal with cycles.
    /// In case of a directed cycle, we search for a node which can execute before some of its parents.
    /// This judgement is made by the canExecute predicate.
    /// </p><p>
    /// Algorithm:
    /// For each target node, we run bfs backward to collect a list of ancestors and their finishing times.
    /// These ancestor nodes are placed in a priority queue according to their finishing time.
    /// We then examine each node on the queue to determine if its input requirements are satisfied.
    /// If so, the node is scheduled.  If not, we put it aside to wait until one of its unscheduled parents is scheduled.
    /// </p><p>
    /// SourcesOf and CreateNodeData are the only graph methods used.  The Nodes property is not used.
    /// </p></remarks>
    internal class CyclicDependencySort<Node, Cost>
        where Cost : IComparable<Cost>
    {
        private Converter<Node, IEnumerable<Node>> successors;
        private BreadthFirstSearch<Node> bfs;

        /// <summary>
        /// Indicates if the node has been placed on the schedule.
        /// </summary>
        /// <remarks>
        /// Unlike WasScheduledLastIteration, this information changes throughout the scheduling process.
        /// </remarks>
        public IndexedProperty<Node, bool> IsScheduled;

#if false
    /// <summary>
    /// Indicates if the node was scheduled on the previous iteration, i.e. its results are available for the current iteration.
    /// </summary>
    /// <remarks>
    /// This information is fixed on entry to the sorter and is not modified.
    /// </remarks>
        public IndexedProperty<Node, bool> WasScheduledLastIteration;
#endif
        private int visitCount;

        /// <summary>
        /// Indicates if scheduling should stop.  Takes the latest node to be scheduled.
        /// </summary>
        public Converter<Node, bool> StopScheduling;

        private bool done;
        private Cost threshold;

        /// <summary>
        /// Cost at which nodes will not be scheduled.
        /// </summary>
        public Cost Threshold
        {
            get { return threshold; }
            set
            {
                threshold = value;
                ApplyThreshold = true;
            }
        }

        /// <summary>
        /// The maximum cost of a scheduled node.
        /// </summary>
        public Cost MaxScheduledCost;

        /// <summary>
        /// Indicates that only nodes whose cost is less than Threshold will be scheduled.
        /// </summary>
        public bool ApplyThreshold;

        public CostUpdater updateCost;

        /// <summary>
        /// Called to update the cost of scheduling a node.
        /// </summary>
        /// <param name="node"></param>
        /// <param name="isScheduled"></param>
        /// <param name="cost">The previous cost, which may be modified in place for efficiency.  May be null.</param>
        /// <returns></returns>
        public delegate Cost CostUpdater(Node node, IndexedProperty<Node, bool> isScheduled, Cost cost);

        /// <summary>
        /// Called just before IsScheduled[node] is set to true.
        /// </summary>
        public Func<Node, bool> addToSchedule;

        /// <summary>
        /// Queue of nodes waiting to be scheduled.
        /// </summary>
        private PriorityQueue<QueueEntry> queue;

        public IndexedProperty<Node, QueueEntry> EntryOfNode;

#if false
    /// <summary>
    /// Represents the undesirability of scheduling a node.
    /// </summary>
        public struct Badness : IComparable<Badness>
        {
            public int ClosenessToTarget;
            public bool OutOfOrderTrigger;
            public bool OutOfOrderTriggee;
            public bool ChildOfStaleParent;

            public int CompareTo(Badness that)
            {
                if(this < that) return -1;
                else if(that < this) return 1;
                else return 0;
            }

            public static bool operator<(Badness a, Badness b)
            {
                return (a.Badness < b.Badness) || (a.ClosenessToTarget < b.ClosenessToTarget);
            }
        }
#endif

        public class QueueEntry : IComparable<QueueEntry>
        {
            public Node Node;
            public Cost Cost;

            /// <summary>
            /// Used to break ties between nodes of the same scheduling cost.
            /// Tries to schedule close nodes last.
            /// </summary>
            public int ClosenessToTarget;

            public int QueuePosition;

            public int CompareTo(QueueEntry that)
            {
                int costCompare = this.Cost.CompareTo(that.Cost);
                if (costCompare != 0) return costCompare;
                else return Comparer<int>.Default.Compare(this.ClosenessToTarget, that.ClosenessToTarget);
            }

            public static readonly EntryComparer Comparer = new EntryComparer();

            public class EntryComparer : IComparer<QueueEntry>
            {
                public int Compare(QueueEntry x, QueueEntry y)
                {
                    return x.CompareTo(y);
                }
            }

            public override string ToString()
            {
                return $"Pos={QueuePosition},Cost={Cost},Node={Node}";
            }
        }

#if false
        public  class KeyValueComparer<KeyType, ValueType> : IComparer<KeyValuePair<KeyType, ValueType>>
        {
            public IComparer<KeyType> KeyComparer;
            public int Compare(KeyValuePair<KeyType, ValueType> x, KeyValuePair<KeyType, ValueType> y)
            {
                return KeyComparer.Compare(x.Key, y.Key);
            }
            public KeyValueComparer(IComparer<KeyType> keyComparer)
            {
                this.KeyComparer = keyComparer;
            }
        }
#endif

        public CyclicDependencySort(IDirectedGraph<Node> dependencyGraph, CostUpdater updateCost)
            : this(dependencyGraph.SourcesOf, dependencyGraph.TargetsOf,
                   (CanCreateNodeData<Node>) dependencyGraph, updateCost)
        {
        }

        public CyclicDependencySort(
            Converter<Node, IEnumerable<Node>> predecessors,
            Converter<Node, IEnumerable<Node>> successors,
            CanCreateNodeData<Node> data,
            CostUpdater updateCost)
        {
            this.successors = successors;
            bfs = new BreadthFirstSearch<Node>(predecessors, data);
            this.updateCost = updateCost;
            queue = new PriorityQueue<QueueEntry>(QueueEntry.Comparer);
            queue.Moved += delegate(QueueEntry entry, int pos) { entry.QueuePosition = pos; };
            EntryOfNode = data.CreateNodeData<QueueEntry>(null);
            visitCount = 0;
            bfs.FinishNode += delegate(Node node)
                {
                    QueueEntry entry = new QueueEntry();
                    entry.ClosenessToTarget = ++visitCount;
                    entry.Node = node;
                    entry.Cost = Threshold;
                    EntryOfNode[node] = entry;
                    queue.Add(entry);
                    UpdateEntry(entry);
                };
            IsScheduled = data.CreateNodeData<bool>(false);
        }

        public void DrainQueue(Action<QueueEntry> action)
        {
            while (queue.Count > 0)
            {
                QueueEntry entry = queue.ExtractMinimum();
                action(entry);
            }
        }

        public void Reschedule(Node node)
        {
            //Console.WriteLine("rescheduling "+node);
            IsScheduled[node] = false;
            QueueEntry entry = EntryOfNode[node];
            if (entry == null)
            {
                bfs.IsVisited[node] = VisitState.Unvisited;
                bfs.SearchFrom(node);
            }
            else if (entry.QueuePosition < 0)
            {
                queue.Add(entry);
                UpdateEntry(entry);
            }
        }

        public void Clear()
        {
            queue.Clear();
            bfs.Clear();
            visitCount = 0;
            IsScheduled.Clear();
            MaxScheduledCost = default(Cost);
        }

        public void MarkScheduled(IEnumerable<Node> targets)
        {
            foreach (Node node in targets)
            {
                bfs.IsVisited[node] = VisitState.Finished;
                IsScheduled[node] = true;
            }
        }

        public void AddRange(IEnumerable<Node> targets)
        {
            // bfs will add all ancestors to the queue.
            bfs.SearchFrom(targets);
            done = false;
            while (queue.Count > 0)
            {
                QueueEntry entry = queue[0];
                // make sure the cost of this entry is up-to-date (inefficient, but safe)
                UpdateEntry(entry);
                if (entry.QueuePosition == 0)
                {
                    if (ApplyThreshold && (entry.Cost.CompareTo(Threshold) >= 0))
                    {
                        return;
                    }
                    queue.ExtractMinimum();
                    Node node = entry.Node;
                    //Console.WriteLine("scheduling " + node);
                    if (addToSchedule != null)
                    {
                        if (!addToSchedule(node))
                        {
                            // put back on the queue
                            queue.Add(entry);
                            continue;
                        }
                    }
                    if (entry.Cost.CompareTo(MaxScheduledCost) > 0) MaxScheduledCost = entry.Cost;
                    IsScheduled[node] = true;
                    if (StopScheduling != null && StopScheduling(node))
                    {
                        done = true;
                        return;
                    }
                    //Console.WriteLine("updating targets:");
                    foreach (Node target in successors(node))
                    {
                        if (!IsScheduled[target])
                        {
                            UpdateCost(target);
                        }
                    }
                    //Console.WriteLine("done with targets");
                }
            }
            done = true;
        }

        public void UpdateCost(Node node)
        {
            // the node may not have an entry if it was never visited.  In that case, ignore it.
            QueueEntry entry = EntryOfNode[node];
            if (entry != null && entry.QueuePosition >= 0)
            {
                UpdateEntry(entry);
            }
        }

        public void UpdateEntry(QueueEntry entry)
        {
            Node node = entry.Node;
            if (IsScheduled[node]) throw new Exception("node " + node + " was already scheduled");
            entry.Cost = updateCost(node, IsScheduled, entry.Cost);
            queue.Changed(entry.QueuePosition);
        }

        public bool IncompleteSchedule
        {
            get { return !done; }
        }
    }
}