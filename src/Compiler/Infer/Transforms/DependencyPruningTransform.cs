// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Prunes dependencies on equivalent statements.
    /// </summary>
    internal class DependencyPruningTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "DependencyPruningTransform";
            }
        }

        internal static bool debug;
        IBlockStatement debugBlock;
        private LoopMergingInfo loopMergingInfo;
        private Dictionary<IStatement, int> indexOfStatement = new Dictionary<IStatement, int>(ReferenceEqualityComparer<IStatement>.Instance);

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            if (context.trackTransform && debug)
            {
                debugBlock = Builder.BlockStmt();
                ITypeDeclaration itdOut = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
                context.OutputAttributes.Add(itdOut, new DebugInfo()
                {
                    Transform = this,
                    Name = "debug",
                    Value = debugBlock
                });
            }
            IMethodDeclaration imd = context.FindAncestor<IMethodDeclaration>();
            loopMergingInfo = context.InputAttributes.Get<LoopMergingInfo>(imd);
            base.DoConvertMethodBody(outputs, inputs);
        }

        protected override IStatement DoConvertStatement(IStatement ist)
        {
            DependencyInformation di = context.InputAttributes.Get<DependencyInformation>(ist);
            if (di != null && !indexOfStatement.ContainsKey(ist))
            {
                RemoveRedundantDependencies(ist, di);
                int index = indexOfStatement.Count;
                indexOfStatement.Add(ist, index);
            }
            IStatement st = base.DoConvertStatement(ist);
            return st;
        }

        private void RemoveRedundantDependencies(IStatement ist, DependencyInformation di)
        {
            // collapse dependencies that have the same unique index.
            Dictionary<int, KeyValuePair<int, IStatement>> nodeOfUniqueIndex = new Dictionary<int, KeyValuePair<int, IStatement>>();
            Set<IStatement> sourcesToRemove = new Set<IStatement>(ReferenceEqualityComparer<IStatement>.Instance);
            foreach (IStatement source in di.GetDependenciesOfType(DependencyType.Dependency))
            {
                int sourceIndex;
                if (indexOfStatement.TryGetValue(source, out sourceIndex))
                {
                    int uniqueIndex = loopMergingInfo.GetIndexOf(source);
                    KeyValuePair<int, IStatement> previousEntry;
                    if (nodeOfUniqueIndex.TryGetValue(uniqueIndex, out previousEntry))
                    {
                        int previousNode = previousEntry.Key;
                        // ist depends on two equivalent statements.
                        // only keep a dependency on the later statement.
                        if (sourceIndex > previousNode)
                        {
                            // source is the later statement.
                            nodeOfUniqueIndex[uniqueIndex] = new KeyValuePair<int, IStatement>(sourceIndex, source);
                            sourcesToRemove.Add(previousEntry.Value);
                        }
                        else
                        {
                            // source is the earlier statement.
                            sourcesToRemove.Add(source);
                        }
                    }
                    else
                    {
                        nodeOfUniqueIndex.Add(uniqueIndex, new KeyValuePair<int, IStatement>(sourceIndex, source));
                    }
                }
            }
            if (sourcesToRemove.Count > 0)
            {
                if (debugBlock != null)
                {
                    debugBlock.Statements.Add(Builder.CommentStmt($"dropping dependency of {ist} on {sourcesToRemove}"));
                }
                di.RemoveAll(sourcesToRemove.Contains);
            }
        }
    }
}
