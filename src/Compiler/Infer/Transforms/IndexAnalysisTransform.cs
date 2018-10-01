// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    internal class IndexAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "IndexAnalysisTransform"; }
        }

        public class IndexInfo
        {
            /// <summary>
            /// The intersection of all containers that the expression has appeared in
            /// </summary>
            internal Containers containers;

            /// <summary>
            /// The set of all binding contexts that the expression has appeared in 
            /// </summary>
            public Set<IReadOnlyCollection<ConditionBinding>> bindings = new Set<IReadOnlyCollection<ConditionBinding>>();

            /// <summary>
            /// The number of times the expression has appeared
            /// </summary>
            public int count;

            /// <summary>
            /// True if the expression is ever assigned to
            /// </summary>
            public bool IsAssignedTo;

            /// <summary>
            /// The variable reference that the expression should be replaced with.
            /// </summary>
            public IExpression clone;

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder("IndexInfo");
                sb.Append("(");
                sb.Append(count);
                sb.AppendLine(")");
                sb.Append("  containers = ");
                sb.AppendLine(containers.ToString());
                return sb.ToString();
            }
        }

        /// <summary>
        /// Stores information about every stochastic array indexer expression where the indices are not all loop variables
        /// </summary>
        public Dictionary<IExpression, IndexInfo> indexInfoOf = new Dictionary<IExpression, IndexInfo>();

        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            bool isDef = Recognizer.IsBeingMutated(context, iaie);
            if (isDef)
            {
                // do not clone the lhs of an array create assignment.
                IAssignExpression assignExpr = context.FindAncestor<IAssignExpression>();
                if (assignExpr.Expression is IArrayCreateExpression) return iaie;
            }
            base.ConvertArrayIndexer(iaie);
            IndexInfo info;
            // TODO: Instead of storing an IndexInfo for each distinct expression, we should try to unify expressions, as in GateAnalysisTransform.
            // For example, we could unify a[0,i] and a[0,0] and use the same clone array for both.
            if (indexInfoOf.TryGetValue(iaie, out info))
            {
                Containers containers = new Containers(context);
                if (info.bindings.Count > 0)
                {
                    List<ConditionBinding> bindings = GetBindings(context, containers.inputs);
                    if (bindings.Count == 0) info.bindings.Clear();
                    else info.bindings.Add(bindings);
                }
                info.containers = Containers.Intersect(info.containers, containers);
                info.count++;
                if (isDef) info.IsAssignedTo = true;
                return iaie;
            }
            CheckIndicesAreNotStochastic(iaie.Indices);
            IVariableDeclaration baseVar = Recognizer.GetVariableDeclaration(iaie);
            // If not an indexed variable reference, skip it (e.g. an indexed argument reference)
            if (baseVar == null) return iaie;
            // If the variable is not stochastic, skip it
            if (!CodeRecognizer.IsStochastic(context, baseVar)) return iaie;
            // If the indices are all loop variables, skip it
            var indices = Recognizer.GetIndices(iaie);
            bool allLoopIndices = indices.All(bracket => bracket.All(indexExpr =>
            {
                if (indexExpr is IVariableReferenceExpression)
                {
                    IVariableReferenceExpression ivre = (IVariableReferenceExpression)indexExpr;
                    return (Recognizer.GetLoopForVariable(context, ivre) != null);
                }
                else
                {
                    return false;
                }
            }));
            if (allLoopIndices) return iaie;

            info = new IndexInfo();
            info.containers = new Containers(context);
            List<ConditionBinding> bindings2 = GetBindings(context, info.containers.inputs);
            if (bindings2.Count > 0) info.bindings.Add(bindings2);
            info.count = 1;
            info.IsAssignedTo = isDef;
            indexInfoOf[iaie] = info;
            return iaie;
        }

        internal static List<ConditionBinding> GetBindings(BasicTransformContext context, IEnumerable<IStatement> containers)
        {
            List<ConditionBinding> bindings = new List<ConditionBinding>();
            foreach (IStatement st in containers)
            {
                if (st is IConditionStatement)
                {
                    IConditionStatement ics = (IConditionStatement) st;
                    if (!CodeRecognizer.IsStochastic(context, ics.Condition))
                    {
                        ConditionBinding binding = new ConditionBinding(ics.Condition);
                        bindings.Add(binding);
                    }
                }
            }
            return bindings;
        }

        /// <summary>
        /// Raise an error if any expression is stochastic.
        /// </summary>
        /// <param name="exprs"></param>
        private void CheckIndicesAreNotStochastic(IList<IExpression> exprs)
        {
            foreach (IExpression index in exprs)
            {
                foreach (var ivd in Recognizer.GetVariables(index))
                {
                    if (CodeRecognizer.IsStochastic(context, ivd))
                    {
                        string msg = "Indexing by a random variable '" + ivd.Name + "'.  You must wrap this statement with Variable.Switch(" + index + ")";
                        Error(msg);
                    }
                }
            }
        }
    }
}