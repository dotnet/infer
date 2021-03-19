// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    using GroupKey = Set<ChannelAnalysisTransform.ExpressionWithBinding>;

    /// <summary>
    /// Collects information about the uses of each variable, assigning the smallest possible number to each use.
    /// Assumes variables are declared before they are used.
    /// </summary>
    internal class ChannelAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "ChannelAnalysisTransform";
            }
        }

        public readonly Dictionary<IVariableDeclaration, UsageInfo> usageInfo = new Dictionary<IVariableDeclaration, UsageInfo>();
        /// <summary>
        /// The set of all loop variables discovered so far
        /// </summary>
        private readonly Set<IVariableDeclaration> loopVars = new Set<IVariableDeclaration>();

        /// <summary>
        /// The list of bindings made by all conditional statements in the input stack
        /// </summary>
        private readonly List<ConditionBinding> conditionContext = new List<ConditionBinding>();

        /// <summary>
        /// Analyse the condition body using an augmented conditionContext
        /// </summary>
        /// <param name="ics"></param>
        /// <returns></returns>
        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            if (CodeRecognizer.IsStochastic(context, ics.Condition))
                return base.ConvertCondition(ics);
            // ics.Condition is not stochastic
            context.SetPrimaryOutput(ics);
            ConvertExpression(ics.Condition);
            ConditionBinding binding = new ConditionBinding(ics.Condition);
            int startIndex = conditionContext.Count;
            conditionContext.Add(binding);
            ConvertBlock(ics.Then);
            if (ics.Else != null)
            {
                conditionContext.RemoveRange(startIndex, conditionContext.Count - startIndex);
                binding = binding.FlipCondition();
                conditionContext.Add(binding);
                ConvertBlock(ics.Else);
            }
            conditionContext.RemoveRange(startIndex, conditionContext.Count - startIndex);
            return ics;
        }

        /// <summary>
        /// Assignments to non-stochastic non-loop integer variables are added to the conditionContext
        /// </summary>
        /// <param name="iae"></param>
        /// <returns></returns>
        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            IExpression expr = ConvertExpression(iae.Expression);
            IExpression target = ConvertExpression(iae.Target);
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(target);
            if (ivd != null)
            {
                var varInfo = VariableInformation.GetVariableInformation(context, ivd);
                if (!varInfo.IsStochastic && varInfo.varType.Equals(typeof(int)) && Recognizer.GetLoopForVariable(context, ivd) == null)
                {
                    // add the assignment as a binding
                    if (target is IVariableDeclarationExpression ivde)
                        target = Builder.VarRefExpr(ivde.Variable);
                    ConditionBinding binding = new ConditionBinding(target, expr);
                    conditionContext.Add(binding);
                    // when current lexical scope ends, remove this binding?
                    // no, because locals aren't correctly scoped yet
                }
            }
            return iae;
        }

        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(ivde);
            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
            if (vi.IsStochastic)
            {
                usageInfo[ivd] = new UsageInfo
                {
                    containers = new Containers(context)
                };
            }
            if (Recognizer.GetAncestorIndexOfLoopBeingInitialized(context) != -1)
                loopVars.Add(ivd);
            return base.ConvertVariableDeclExpr(ivde);
        }

        /// <summary>
        /// Register a use of a variable without indices
        /// </summary>
        /// <param name="ivre"></param>
        /// <returns></returns>
        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            IExpression expr = base.ConvertVariableRefExpr(ivre);
            if (Recognizer.IsBeingIndexed(context) || Recognizer.IsBeingMutated(context, expr))
                return expr;
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(expr);
            if (usageInfo.TryGetValue(ivd, out UsageInfo info))
            {
                info.indexingDepths.Add(0);
                RegisterUse(info, expr);
            }
            return expr;
        }

        /// <summary>
        /// Register a use of a variable at a specific indexing depth
        /// </summary>
        /// <param name="iaie"></param>
        /// <returns></returns>
        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            IExpression expr = (IArrayIndexerExpression)base.ConvertArrayIndexer(iaie);
            // Only register the full indexer expression, not its sub-expressions
            if (Recognizer.IsBeingIndexed(context) || Recognizer.IsBeingMutated(context, expr))
                return expr;
            List<IList<IExpression>> indices = Recognizer.GetIndices(expr, out IExpression target);
            if (target is IVariableReferenceExpression)
            {
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(target);
                if (ivd != null)
                {
                    if (usageInfo.TryGetValue(ivd, out UsageInfo info))
                    {
                        info.indexingDepths.Add(indices.Count);
                        RegisterUse(info, expr);
                    }
                }
            }
            return expr;
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            if (CodeRecognizer.IsInfer(imie))
            {
                // ignore the variable use
                return imie;
            }
            return base.ConvertMethodInvoke(imie);
        }

        protected int GetUseNumber(UsageInfo info, IExpression expr)
        {
            // package the expr with its conditionContext, 
            // so that uses in disjoint deterministic conditions can get the same number.
            Set<ConditionBinding> bindings = new Set<ConditionBinding>();
            bindings.AddRange(conditionContext);
            ExpressionWithBinding eb = new ExpressionWithBinding
            {
                Expression = expr,
                Binding = bindings
            };
            return info.GetUseNumber(eb, loopVars);
        }

        /// <summary>
        /// Compute the use number and attach it to the current statement
        /// </summary>
        /// <param name="info"></param>
        /// <param name="expr"></param>
        protected void RegisterUse(UsageInfo info, IExpression expr)
        {
            int useNumber = GetUseNumber(info, expr);
            IStatement st = context.FindAncestorNotSelf<IStatement>();
            Queue<int> queue;
            if (!info.useNumberOfStatement.TryGetValue(st, out queue))
            {
                queue = new Queue<int>();
                info.useNumberOfStatement[st] = queue;
            }
            queue.Enqueue(useNumber);
        }

        internal class UsageInfo
        {
            /// <summary>
            /// Containers of the variable declaration
            /// </summary>
            public Containers containers;
            public Collections.SortedSet<int> indexingDepths = new Microsoft.ML.Probabilistic.Collections.SortedSet<int>();
            /// <summary>
            /// The set of all uses, organized into groups such that the members of each group are disjoint, i.e.
            /// they access different parts of the variable (such as different elements of an array) or they
            /// have disjoint condition contexts.  Copies of the same group are stored in a Stack.
            /// </summary>
            protected Dictionary<GroupKey, Stack<Group>> groups = new Dictionary<GroupKey, Stack<Group>>();
            /// <summary>
            /// Used to speed up searching through groups
            /// </summary>
            protected Dictionary<IExpression, Dictionary<object, List<GroupKey>>> invertedIndex;
            protected static IVariableDeclaration tempDecl = Builder.VarDecl("_EmptyBinding", typeof(int));
            public Dictionary<IStatement, Queue<int>> useNumberOfStatement = new Dictionary<IStatement, Queue<int>>(ReferenceEqualityComparer<IStatement>.Instance);
            public int NumberOfUses;

            /// <summary>
            /// Only for backward compatibility with GibbsSampling
            /// </summary>
            public int NumberOfUsesOld
            {
                get
                {
                    if (NumberOfUses == 1)
                        return 1;
                    int count = 0;
                    foreach (var bucket in groups.Values)
                    {
                        Group group = bucket.Peek();
                        count += bucket.Count * group.Key.Count;
                    }
                    return count;
                }
            }

            /// <summary>
            /// Compute the smallest possible use number that can be assigned to the expression
            /// </summary>
            /// <param name="eb"></param>
            /// <param name="loopVars">The set of all known loop variables</param>
            /// <returns></returns>
            public int GetUseNumber(ExpressionWithBinding eb, Set<IVariableDeclaration> loopVars)
            {
                if (containers.LoopCount > 0)
                {
                    // exclude loop variables in the containers
                    loopVars = (Set<IVariableDeclaration>)loopVars.Clone();
                    foreach (IStatement container in containers.inputs)
                    {
                        if (container is IForStatement ifs)
                        {
                            var loopVar = Recognizer.LoopVariable(ifs);
                            loopVars.Remove(loopVar);
                        }
                    }
                }
                if (eb.Binding.Count > 0)
                {
                    // collect the set of loop variables in the expression
                    Set<IVariableDeclaration> loopVarsNotInExpression = Set<IVariableDeclaration>.FromEnumerable(loopVars);
                    loopVarsNotInExpression.Remove(Recognizer.GetVariables(eb.Expression));

                    // eliminate any loop variables in the binding that are not in the expression
                    eb.Binding = Recognizer.RemoveLoopVars(eb.Binding, loopVarsNotInExpression.Contains);
                    if (eb.Binding == null)
                        return 0;  // the conditions are contradictory so this use will never happen
                }

                // if eb exactly matches an existing group key, then it cannot join any existing group
                Group newGroup = new Group();
                newGroup.Key.Add(eb);
                bool mustMakeNewGroup = groups.ContainsKey(newGroup.Key);

                // this threshold was tuned on SpeedTests.MarkovChainUnrolledTest3
                if (invertedIndex == null && !mustMakeNewGroup && groups.Count > 4)
                    BuildInvertedIndex(loopVars);

                // collection of all GroupKeys that could be disjoint from eb
                ICollection<GroupKey> keysToSearch;
                Set<object> varsInBinding = null;
                if (invertedIndex == null)
                {
                    if (mustMakeNewGroup)
                        keysToSearch = new Set<GroupKey>();
                    else
                    {
                        // search all GroupKeys
                        keysToSearch = groups.Keys;
                    }
                }
                else
                {
                    // construct a minimal set of GroupKeys to search
                    keysToSearch = new Set<GroupKey>();
                    varsInBinding = GetVariablesAndParameters(eb.Binding, loopVars);
                    if (!mustMakeNewGroup)
                    {
                        foreach (var entry in invertedIndex)
                        {
                            var dict = entry.Value;
                            if (entry.Key.Equals(eb.Expression))
                            {
                                // If the expression matches a previous expression, then the only way it could be disjoint 
                                // is if the condition context has some overlap with previous contexts.
                                // Therefore we search only GroupKeys whose binding variables overlap with eb.
                                foreach (var ivd in varsInBinding)
                                {
                                    if (dict.TryGetValue(ivd, out List<GroupKey> keys))
                                        keysToSearch.AddRange(keys);
                                }
                            }
                            else
                            {
                                // if expression doesn't match, we conservatively search all GroupKeys
                                foreach (var keys in dict.Values)
                                    keysToSearch.AddRange(keys);
                            }
                        }
                    }
                }

                // find the first group in which eb is disjoint from all members of the group, and add eb to that group.
                // the use number becomes the index of that group.
                foreach (var key in keysToSearch)
                {
                    var bucket = groups[key];
                    Group group = bucket.Peek();
                    bool disjointWithAll = true;
                    foreach (ExpressionWithBinding eb2 in group.Key)
                    {
                        if (!Recognizer.AreDisjoint(eb.Expression, eb.Binding, eb2.Expression, eb2.Binding, loopVars.Contains))
                        {
                            disjointWithAll = false;
                            break;
                        }
                    }
                    if (disjointWithAll)
                    {
                        group = bucket.Pop();
                        if (bucket.Count == 0)
                            groups.Remove(group.Key);
                        // This modifies the key of an existing group.
                        // We must ensure that this is not a key in the groups dictionary.
                        // This is ensured by using a Stack and setting the dictionary key to the key of the bottom group.
                        group.Key.Add(eb);
                        AddGroup(group, eb.Expression, varsInBinding);
                        return group.UseNumber;
                    }
                }
                // no matching group found.  create a new one.
                newGroup.UseNumber = NumberOfUses++;
                AddGroup(newGroup, eb.Expression, varsInBinding);
                return newGroup.UseNumber;
            }

            /// <summary>
            /// Add a new group to the groups dictionary and the inverted index
            /// </summary>
            /// <param name="group"></param>
            /// <param name="expr"></param>
            /// <param name="varsInBinding"></param>
            protected void AddGroup(Group group, IExpression expr, ICollection<object> varsInBinding)
            {
                Stack<Group> bucket;
                if (!groups.TryGetValue(group.Key, out bucket))
                {
                    bucket = new Stack<Group>();
                    // the dictionary key will be the key of the bottom group in the Stack.
                    groups[group.Key] = bucket;
                }
                bucket.Push(group);

                if (invertedIndex != null)
                {
                    AddToInvertedIndex(group.Key, expr, varsInBinding);
                }
            }

            protected void AddToInvertedIndex(GroupKey key, IExpression expr, ICollection<object> declsInBinding)
            {
                Dictionary<object, List<GroupKey>> dict;
                if (!invertedIndex.TryGetValue(expr, out dict))
                {
                    dict = new Dictionary<object, List<GroupKey>>();
                    invertedIndex[expr] = dict;
                }
                // the GroupKey must appear somewhere in the inverted index, so if there are no
                // variables in the binding we put the GroupKey under a dummy variable
                if (declsInBinding.Count == 0)
                    declsInBinding = new List<object>() { tempDecl };
                foreach (var ivd in declsInBinding)
                {
                    List<GroupKey> keys;
                    if (!dict.TryGetValue(ivd, out keys))
                    {
                        keys = new List<GroupKey>();
                        dict[ivd] = keys;
                        keys.Add(key);
                    }
                    else
                    {
                        // simplify the existing keys
                        // The GroupKeys in the inverted index must be the same objects as the keys in the groups dictionary.
                        // We ensure this by adding the existing keys first, then the new key if it is not a duplicate.
                        Set<GroupKey> set = new Set<GroupKey>();
                        set.AddRange(keys);
                        set.Add(key);
                        keys.Clear();
                        keys.AddRange(set);
                    }
                }
            }

            /// <summary>
            /// Build an inverted index to the groups dictionary
            /// </summary>
            /// <param name="loopVars"></param>
            void BuildInvertedIndex(ICollection<IVariableDeclaration> loopVars)
            {
                invertedIndex = new Dictionary<IExpression, Dictionary<object, List<GroupKey>>>();
                foreach (var key in groups.Keys)
                {
                    foreach (var eb in key)
                    {
                        AddToInvertedIndex(key, eb.Expression, GetVariablesAndParameters(eb.Binding, loopVars));
                    }
                }
            }

            /// <summary>
            /// Get all non-loop variables or parameters in bindings
            /// </summary>
            /// <param name="bindings"></param>
            /// <param name="loopVars"></param>
            /// <returns></returns>
            Set<object> GetVariablesAndParameters(IEnumerable<ConditionBinding> bindings, ICollection<IVariableDeclaration> loopVars)
            {
                return Set<object>.FromEnumerable(bindings.Select(binding =>
                    Recognizer.GetVariablesAndParameters(binding.GetExpression())
                    .Where(decl => !(decl is IVariableDeclaration declaration) || !loopVars.Contains(declaration))
                    ));
            }

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder("UsageInfo(");
                string indexingDepthsString = indexingDepths.ToString();
                if (indexingDepthsString.Length > 0)
                {
                    sb.Append("indexingDepths=");
                    sb.Append(indexingDepthsString);
                }
                sb.Append(")");
                int count = 0;
                foreach (var entry in groups)
                {
                    GroupKey groupKey = entry.Key;                    
                    sb.AppendLine();
                    sb.Append("[");
                    sb.Append(count++);
                    sb.Append("] ");
                    sb.Append(entry.Value.Count);
                    sb.Append(" ");
                    foreach (ExpressionWithBinding eb in groupKey)
                    {
                        sb.Append("(");
                        sb.Append(eb);
                        sb.Append(")");
                    }
                }
                return sb.ToString();
            }
        }

        public class ExpressionWithBinding
        {
            public IExpression Expression;
            public IReadOnlyCollection<ConditionBinding> Binding;

            public override bool Equals(object obj)
            {
                return (obj is ExpressionWithBinding that) &&
                    Expression.Equals(that.Expression) &&
                    Binding.Equals(that.Binding);
            }

            public override int GetHashCode()
            {
                return Hash.Combine(Expression.GetHashCode(), Binding.GetHashCode());
            }

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder();
                sb.Append(Expression);
                if (Binding.Count > 0)
                {
                    sb.Append(" ");
                    sb.Append(StringUtil.CollectionToString(Binding, ","));
                }
                return sb.ToString();
            }
        }

        public class Group
        {
            public GroupKey Key = new GroupKey();
            public int UseNumber;

            public override string ToString()
            {
                return Key.ToString();
            }
        }
    }
}