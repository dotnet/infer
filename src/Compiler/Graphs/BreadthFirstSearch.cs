// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    internal class BreadthFirstSearch<NodeType> : GraphSearcher<NodeType>
    {
        public BreadthFirstSearch(Converter<NodeType, IEnumerable<NodeType>> successors,
                                  CanCreateNodeData<NodeType> data)
            : base(successors, data)
        {
            Initialize();
        }

        public BreadthFirstSearch(IGraph<NodeType> graph)
            : base(graph)
        {
            Initialize();
        }

        public BreadthFirstSearch(IDirectedGraph<NodeType> graph)
            : base(graph)
        {
            Initialize();
        }

        /// <summary>
        /// A queue of nodes for breadth-first search.
        /// </summary>
        protected IList<NodeType> SearchQueue;

        protected void Initialize()
        {
            SearchQueue = new QueueAsList<NodeType>();
        }

        public override void Clear()
        {
            base.Clear();
            SearchQueue.Clear();
        }

        public override void SearchFrom(NodeType start)
        {
            SearchQueue.Add(start);
            DoSearch();
        }

        public override void SearchFrom(IEnumerable<NodeType> startNodes)
        {
            foreach (NodeType node in startNodes)
            {
                SearchQueue.Add(node);
            }
            DoSearch();
        }

        protected void DoSearch()
        {
            while (SearchQueue.Count > 0)
            {
                NodeType node = SearchQueue[0];
                SearchQueue.RemoveAt(0);
                switch (IsVisited[node])
                {
                    case VisitState.Unvisited:
                        OnDiscoverNode(node);
                        break;
                    case VisitState.Discovered:
                        break;
                    case VisitState.Visiting:
                        throw new Exception("BUG: start is Visiting and on the SearchQueue.");
                    case VisitState.Finished:
                        // this happens if we SearchFrom a Finished node.
                        continue;
                }
                // node was previously Unvisited or Discovered
                IsVisited[node] = VisitState.Visiting;
                foreach (NodeType target in Successors(node))
                {
                    Edge<NodeType> edge = new Edge<NodeType>(node, target);
                    OnDiscoverEdge(edge);
                    VisitState targetIsVisited = IsVisited[target];
                    switch (targetIsVisited)
                    {
                        case VisitState.Unvisited:
                            IsVisited[target] = VisitState.Discovered;
                            SearchQueue.Add(target);
                            OnDiscoverNode(target);
                            // tree edge
                            OnTreeEdge(edge);
                            break;
                        case VisitState.Visiting:
                            // back edge
                            OnBackEdge(edge);
                            break;
                        case VisitState.Discovered:
                            // cross edge
                            OnCrossEdge(edge);
                            break;
                        case VisitState.Finished:
                            // cross edge
                            OnCrossEdge(edge);
                            break;
                    }
                }
                IsVisited[node] = VisitState.Finished;
                OnFinishNode(node);
                if (stopped)
                {
                    SearchQueue.Clear();
                    stopped = false;
                    break;
                }
            }
        }
    }
}