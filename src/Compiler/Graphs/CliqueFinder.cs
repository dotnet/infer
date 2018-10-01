// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// Find all maximal cliques of an undirected graph
    /// </summary>
    /// <typeparam name="NodeType">The node type</typeparam>
    /// <remarks>
    /// Cliques are found using the Bron-Kerbosch algorithm.
    /// </remarks>
    internal class CliqueFinder<NodeType>
    {
        Stack<NodeType> clique;
        HashSet<NodeType> exclude;
        Func<NodeType, IEnumerable<NodeType>> neighbors;
        Action<Stack<NodeType>> action;

        /// <summary>
        /// Create a new CliqueFinder
        /// </summary>        
        /// <param name="neighbors">A dictionary giving the set of neighbors of any node.</param>
        public CliqueFinder(Func<NodeType, IEnumerable<NodeType>> neighbors)
        {
            this.clique = new Stack<NodeType>();
            this.exclude = new HashSet<NodeType>();
            this.neighbors = neighbors;
        }

        /// <summary>
        /// Find all maximal cliques in an undirected graph using the Bron-Kerbosch algorithm
        /// </summary>
        /// <param name="candidates">The set of nodes.</param>
        /// <param name="action">Called with each clique as it is found.</param>
        /// <remarks>
        /// The graph must not have self-loops.
        /// </remarks>
        public void ForEachClique(ICollection<NodeType> candidates, Action<Stack<NodeType>> action)
        {
            this.action = action;
            ForEachClique(candidates, exclude);
        }

        /// <summary>
        /// Find all maximal cliques in an undirected graph using the Bron-Kerbosch algorithm
        /// </summary>
        /// <param name="candidates">The set of nodes.</param>
        /// <param name="exclude">An empty workspace used by recursive calls.</param>
        /// <remarks>
        /// The graph must not have self-loops.
        /// </remarks>
        private void ForEachClique(ICollection<NodeType> candidates, HashSet<NodeType> exclude)
        {
            HashSet<NodeType> visitedCandidates = new HashSet<NodeType>();
            NodeType pivot = default(NodeType);
            pivot = candidates.FirstOrDefault();
            //int maxNeighbors = 0;
            //foreach (NodeType i in candidates)
            //{
            //    int numNeighbors = neighbors(i).Count;
            //    if (numNeighbors > maxNeighbors)
            //    {
            //        maxNeighbors = numNeighbors;
            //        pivot = i;
            //    }
            //}
            //foreach (NodeType i in exclude)
            //{
            //    int numNeighbors = neighbors(i).Count;
            //    if (numNeighbors > maxNeighbors)
            //    {
            //        maxNeighbors = numNeighbors;
            //        pivot = i;
            //    }
            //}
            foreach (NodeType i in candidates)
            {
                IEnumerable<NodeType> nbrs = neighbors(i);
                // skip neighbors of pivot
                if (nbrs.Contains(pivot))
                    continue;  
                HashSet<NodeType> matches = new HashSet<NodeType>();
                HashSet<NodeType> excludeMatches = new HashSet<NodeType>();
                foreach (NodeType neighbor in nbrs)
                {
                    bool isVisited = visitedCandidates.Contains(neighbor);
                    if (candidates.Contains(neighbor) && !isVisited)
                        matches.Add(neighbor);
                    if (exclude.Contains(neighbor) || isVisited)
                        excludeMatches.Add(neighbor);
                }
                clique.Push(i);
                if (matches.Count > 0)
                    ForEachClique(matches, excludeMatches);
                else if (excludeMatches.Count == 0)
                    action(clique);
                clique.Pop();
                visitedCandidates.Add(i);
            }
        }
    }
}
