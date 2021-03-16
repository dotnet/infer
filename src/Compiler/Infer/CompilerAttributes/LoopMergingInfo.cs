// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler.Attributes
{
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using Compiler.Graphs;
    using Microsoft.ML.Probabilistic.Compiler.CodeModel;
    using Microsoft.ML.Probabilistic.Compiler;
    using NodeIndex = System.Int32;
    using EdgeIndex = System.Int32;
    using Collections;
    using Utilities;

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    internal class LoopMergingInfo : ICompilerAttribute
    {
        /// <summary>
        /// Maps statements into graph node numbers
        /// </summary>
        private Dictionary<IStatement, NodeIndex> indexOf = new Dictionary<IStatement, NodeIndex>(ReferenceEqualityComparer<IStatement>.Instance);

        /// <summary>
        /// A graph where nodes are statements and edges indicate that loop merging is prohibited between them.
        /// </summary>
        public IndexedGraph graph;

        /// <summary>
        /// For each graph edge, stores the loop variables on which merging is prohibited.
        /// </summary>
        private IndexedProperty<EdgeIndex, ICollection<IVariableDeclaration>> prohibitedLoopVars;

        /// <summary>
        /// For each graph edge, stores a pairing of loop variables and their offsets.
        /// </summary>
        public IndexedProperty<EdgeIndex, IOffsetInfo> offsetInfos;

        public LoopMergingInfo(IList<IStatement> stmts)
        {
            graph = new IndexedGraph();
            foreach (var stmt in stmts)
            {
                indexOf[stmt] = graph.AddNode();
            }
            prohibitedLoopVars = graph.CreateEdgeData<ICollection<IVariableDeclaration>>();
            offsetInfos = graph.CreateEdgeData<IOffsetInfo>();
        }

        /// <summary>
        /// Get the index of a top-level statement.
        /// </summary>
        /// <param name="statement">A top-level statement.</param>
        /// <returns></returns>
        public NodeIndex GetIndexOf(IStatement statement)
        {
            return indexOf[statement];
        }

        /// <summary>
        /// Add a statement that shares all conflicts with an existing node.
        /// </summary>
        /// <param name="statement"></param>
        /// <param name="node"></param>
        public void AddEquivalentStatement(IStatement statement, NodeIndex node)
        {
            indexOf[statement] = node;
        }

        /// <summary>
        /// Add a new statement with no conflicts.
        /// </summary>
        /// <param name="statement"></param>
        /// <returns></returns>
        public NodeIndex AddNode(IStatement statement)
        {
            NodeIndex node = graph.AddNode();
            indexOf[statement] = node;
            return node;
        }

        public void InheritSourceConflicts(NodeIndex newNode, NodeIndex oldNode)
        {
            foreach (EdgeIndex edge in graph.EdgesInto(oldNode).ToArray())
            {
                int source = graph.SourceOf(edge);
                EdgeIndex edge2 = graph.AddEdge(source, newNode);
                prohibitedLoopVars[edge2] = prohibitedLoopVars[edge];
                offsetInfos[edge2] = offsetInfos[edge];
            }
        }

        public void InheritTargetConflicts(NodeIndex newNode, NodeIndex oldNode)
        {
            foreach (EdgeIndex edge in graph.EdgesOutOf(oldNode).ToArray())
            {
                int source = graph.SourceOf(edge);
                EdgeIndex edge2 = graph.AddEdge(source, newNode);
                prohibitedLoopVars[edge2] = prohibitedLoopVars[edge];
                offsetInfos[edge2] = offsetInfos[edge];
            }
        }

        /// <summary>
        /// Get the index of the statement that prevents loop merging, or -1 if none
        /// </summary>
        /// <param name="stmts"></param>
        /// <param name="stmtIndex"></param>
        /// <param name="loopVar"></param>
        /// <param name="isForwardLoop"></param>
        /// <returns></returns>
        public int GetConflictingStmt(Set<int> stmts, int stmtIndex, IVariableDeclaration loopVar, bool isForwardLoop)
        {
            foreach (EdgeIndex edge in graph.EdgesInto(stmtIndex))
            {
                int source = graph.SourceOf(edge);
                if (stmts.Contains(source) && IsProhibited(edge, loopVar, isForwardLoop, true))
                    return source;
            }
            foreach (EdgeIndex edge in graph.EdgesOutOf(stmtIndex))
            {
                int target = graph.TargetOf(edge);
                if (stmts.Contains(target) && IsProhibited(edge, loopVar, isForwardLoop, false))
                    return target;
            }
            return -1;
        }

        private bool IsProhibited(int edge, IVariableDeclaration loopVar, bool isForwardLoop, bool isForwardEdge)
        {
            ICollection<IVariableDeclaration> prohibited = prohibitedLoopVars[edge];
            if (prohibited != null && prohibited.Contains(loopVar))
                return true;
            IOffsetInfo offsetInfo = offsetInfos[edge];
            if (offsetInfo != null)
            {
                foreach (var entry in offsetInfo)
                {
                    if (entry.loopVar == loopVar)
                    {
                        int offset = entry.offset;
                        if ((offset > 0) && isForwardLoop && isForwardEdge)
                            return true;
                        if ((offset < 0) && !isForwardLoop && isForwardEdge)
                            return true;
                    }
                }
            }
            return false;
        }

        public void PreventLoopMerging(IStatement mutated, IStatement affected, ICollection<IVariableDeclaration> loopVars)
        {
            int edge;
            int source = indexOf[mutated];
            int target = indexOf[affected];
            if (!graph.TryGetEdge(source, target, out edge))
            {
                edge = graph.AddEdge(source, target);
                prohibitedLoopVars[edge] = loopVars;
            }
            else
            {
                ICollection<IVariableDeclaration> list = prohibitedLoopVars[edge];
                if (list == null)
                {
                    list = new Set<IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
                    prohibitedLoopVars[edge] = list;
                }
                list.AddRange(loopVars);
            }
        }

        public void SetOffsetInfo(IStatement mutated, IStatement affected, IOffsetInfo offsetInfo)
        {
            int edge;
            int source = indexOf[mutated];
            int target = indexOf[affected];
            if (!graph.TryGetEdge(source, target, out edge))
            {
                edge = graph.AddEdge(source, target);
                prohibitedLoopVars[edge] = null;
                offsetInfos[edge] = offsetInfo;
            }
            else
            {
                offsetInfos[edge] = offsetInfo;
            }
        }

        public IStatement GetStatement(int index)
        {
            foreach (KeyValuePair<IStatement, int> entry in indexOf)
            {
                if (entry.Value == index)
                    return entry.Key;
            }
            return null;
        }

        public string VerboseToString()
        {
            StringBuilder sb = new StringBuilder();
            for (int edge = 0; edge < graph.EdgeCount(); edge++)
            {
                ICollection<IVariableDeclaration> list = prohibitedLoopVars[edge];
                string indexString = (list == null) ? "" : list.Aggregate("", (s, ivd) => (s + ivd.Name + " "));
                string stmtString = GetStatement(graph.SourceOf(edge)).ToString() + GetStatement(graph.TargetOf(edge)).ToString();
                sb.AppendLine(StringUtil.JoinColumns(indexString, stmtString));
            }
            return sb.ToString();
        }

        public DebugInfo GetDebugInfo(ICodeTransform transform)
        {
            CodeBuilder Builder = CodeBuilder.Instance;
            IBlockStatement block = Builder.BlockStmt();
            bool includeStatementNumbers = false;
            if (includeStatementNumbers)
            {
                List<List<IStatement>> stmts = new List<List<IStatement>>();
                foreach (var entry in indexOf)
                {
                    IStatement ist = entry.Key;
                    int index = entry.Value;
                    while (stmts.Count <= index)
                        stmts.Add(new List<IStatement>());
                    stmts[index].Add(ist);
                }
                for (int i = 0; i < stmts.Count; i++)
                {
                    block.Statements.Add(Builder.CommentStmt(i.ToString()));
                    foreach (var ist in stmts[i])
                    {
                        block.Statements.Add(ist);
                    }
                }
            }
            for (int edge = 0; edge < graph.EdgeCount(); edge++)
            {
                ICollection<IVariableDeclaration> list = prohibitedLoopVars[edge];
                IBlockStatement body = Builder.BlockStmt();
                body.Statements.Add(GetStatement(graph.SourceOf(edge)));
                body.Statements.Add(GetStatement(graph.TargetOf(edge)));
                if (list != null)
                {
                    foreach (IVariableDeclaration ivd in list)
                    {
                        IForEachStatement ifes = Builder.ForEachStmt();
                        ifes.Variable = ivd;
                        ifes.Expression = null;
                        ifes.Body = body;
                        body = Builder.BlockStmt();
                        body.Statements.Add(ifes);
                    }
                }
                if (body.Statements.Count == 1)
                    block.Statements.Add(body.Statements[0]);
                else
                    block.Statements.Add(body);
            }
            DebugInfo info = new DebugInfo();
            info.Transform = transform;
            info.Name = "LoopMergingInfo";
            info.Value = block;
            return info;
        }

        public override string ToString()
        {
            return "LoopMergingInfo";
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}
