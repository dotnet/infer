// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Attributes
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Stores information about how a statement depends on other statements.
    /// </summary>
    internal class DependencyInformation : ICompilerAttribute, ICloneable
    {
        /// <summary>
        /// Stores the dependency type of all dependent statements.
        /// </summary>
        public Dictionary<IStatement, DependencyType> dependencyTypeOf = new Dictionary<IStatement, DependencyType>(new IdentityComparer<IStatement>());

        //public SortedDictionary<IStatement,DependencyType> dependencyTypeOf = new SortedDictionary<IStatement, DependencyType>(new StatementComparer());
        /// <summary>
        /// Stores the offsets of all dependent statements that have an offset.
        /// </summary>
        public Dictionary<IStatement, IOffsetInfo> offsetIndexOf = new Dictionary<IStatement, IOffsetInfo>(new IdentityComparer<IStatement>());

        /// <summary>
        /// True if this statement assigns to an output variable of the inference.
        /// </summary>
        public bool IsOutput;

        /// <summary>
        /// True if this statement always returns a uniform distribution or zero evidence value.
        /// </summary>
        public bool IsUniform;

        /// <summary>
        /// True if this statement must be updated whenever any dependency changes.
        /// </summary>
        public bool IsFresh;

        /// <summary>
        /// List of method arguments that the statement depends on.
        /// </summary>
        public Set<IParameterDeclaration> ParameterDependencies = new Set<IParameterDeclaration>(new IdentityComparer<IParameterDeclaration>());

        public void AddOffsetIndices(IOffsetInfo offsetIndex, IStatement ist)
        {
            AddOffsetIndices(offsetIndexOf, offsetIndex, ist);
        }

        private static void AddOffsetIndices(Dictionary<IStatement, IOffsetInfo> offsetIndexOf, IOffsetInfo offsetIndex, IStatement ist)
        {
            IOffsetInfo offsetIndices;
            if (!offsetIndexOf.TryGetValue(ist, out offsetIndices))
            {
                offsetIndexOf[ist] = offsetIndex;
            }
            else
            {
                OffsetInfo newOffsetInfo = new OffsetInfo();
                newOffsetInfo.AddRange(offsetIndices);
                newOffsetInfo.AddRange(offsetIndex);
                offsetIndexOf[ist] = newOffsetInfo;
            }
        }

        public bool HasAnyDependencyOfType(DependencyType type)
        {
            foreach (IStatement ist in GetDependenciesOfType(type))
                return true;
            return false;
        }

        public int Count(DependencyType type)
        {
            int count = 0;
            foreach (IStatement ist in GetDependenciesOfType(type))
                count++;
            return count;
        }

        public bool HasDependency(DependencyType type, IStatement ist)
        {
            if (type == DependencyType.SkipIfUniform)
            {
                foreach (var dependencySt in SkipIfUniform)
                {
                    if (ContainsStatement(dependencySt, ist))
                    {
                        return true;
                    }
                }
                return false;
            }
            else
            {
                DependencyType depType;
                if (!dependencyTypeOf.TryGetValue(ist, out depType))
                    return false;
                return (depType & type) > 0;
            }
        }

        private bool ContainsStatement(IStatement dependencySt, IStatement ist)
        {
            if (ReferenceEquals(dependencySt, ist))
                return true;
            if (dependencySt is AnyStatement)
            {
                AnyStatement anySt = (AnyStatement)dependencySt;
                bool found = false;
                ForEachStatement(anySt, st =>
                {
                    if (ContainsStatement(st, ist))
                        found = true;
                });
                return found;
            }
            if (dependencySt is IExpressionStatement && ist is IExpressionStatement)
            {
                IExpressionStatement es = (IExpressionStatement)dependencySt;
                IExpressionStatement ies = (IExpressionStatement)ist;
                return CodeBuilder.Instance.ContainsExpression(es.Expression, ies.Expression);
            }
            return false;
        }

        public IEnumerable<IStatement> GetDependenciesOfType(DependencyType type)
        {
            foreach (KeyValuePair<IStatement, DependencyType> entry in dependencyTypeOf)
            {
                if ((entry.Value & type) > 0)
                    yield return entry.Key;
            }
        }

        public void Add(DependencyType type, IStatement ist)
        {
            Add(dependencyTypeOf, type, ist);
        }

        private static void Add(IDictionary<IStatement, DependencyType> dependencyTypeOf, DependencyType type, IStatement ist)
        {
            DependencyType depType;
            dependencyTypeOf.TryGetValue(ist, out depType);
            dependencyTypeOf[ist] = type | depType;
        }

        public void AddRange(DependencyType type, IEnumerable<IStatement> stmts)
        {
            foreach (IStatement ist in stmts)
                Add(type, ist);
        }

        public void AddRange(IEnumerable<KeyValuePair<IStatement, DependencyType>> pairs)
        {
            foreach (KeyValuePair<IStatement, DependencyType> pair in pairs)
            {
                Add(pair.Value, pair.Key);
            }
        }

        public void Remove(DependencyType type, IStatement stmt)
        {
            DependencyType depType;
            if (dependencyTypeOf.TryGetValue(stmt, out depType))
            {
                depType = depType & (~type);
                if (depType == 0)
                    dependencyTypeOf.Remove(stmt);
                else
                    dependencyTypeOf[stmt] = depType;
            }
        }

        public void Remove(DependencyType type)
        {
            Remove(type, _ => true);
        }

        public void Remove(DependencyType type, Predicate<IStatement> predicate)
        {
            List<IStatement> deps = new List<IStatement>();
            deps.AddRange(GetDependenciesOfType(type));
            foreach (IStatement ist in deps)
            {
                if (predicate(ist))
                {
                    DependencyType depType = dependencyTypeOf[ist] & (~type);
                    if (depType == 0)
                        dependencyTypeOf.Remove(ist);
                    else
                        dependencyTypeOf[ist] = depType;
                }
            }
        }

        public void Remove(IStatement stmt)
        {
            dependencyTypeOf.Remove(stmt);
        }

        public void RemoveAll(IStatement stmt)
        {
            RemoveAll(ist => ReferenceEquals(ist, stmt));
        }

        public void RemoveAll(Predicate<IStatement> predicate)
        {
            Replace(ist => predicate(ist) ? null : ist);
        }

        public void Replace(IDictionary<IStatement, IStatement> replacements)
        {
            Replace(delegate (IStatement ist)
                {
                    IStatement newStmt;
                    if (replacements.TryGetValue(ist, out newStmt))
                        return newStmt;
                    else
                        return ist;
                });
        }

        public void Replace(Converter<IStatement, IStatement> converter)
        {
            List<IStatement> toRemove = new List<IStatement>();
            Dictionary<IStatement, DependencyType> toAdd = new Dictionary<IStatement, DependencyType>(new IdentityComparer<IStatement>());
            Dictionary<IStatement, IOffsetInfo> toAddOffset = new Dictionary<IStatement, IOffsetInfo>(new IdentityComparer<IStatement>());
            foreach (KeyValuePair<IStatement, DependencyType> entry in dependencyTypeOf)
            {
                IStatement stmt = entry.Key;
                IStatement newStmt = Replace(stmt, converter);
                if (!ReferenceEquals(newStmt, stmt))
                {
                    toRemove.Add(stmt);
                    if (newStmt != null)
                    {
                        DependencyType type = entry.Value;
                        IOffsetInfo offsetIndices;
                        offsetIndexOf.TryGetValue(stmt, out offsetIndices);
                        if (newStmt is AnyStatement)
                        {
                            AnyStatement anySt = (AnyStatement)newStmt;
                            DependencyType anyTypes = DependencyType.Requirement | DependencyType.SkipIfUniform;
                            DependencyType otherType = type & ~anyTypes;
                            if (otherType > 0)
                            {
                                // must split Any for these types
                                ForEachStatement(anySt, ist =>
                                {
                                    Add(toAdd, otherType, ist);
                                    if (offsetIndices != default(OffsetInfo))
                                        AddOffsetIndices(toAddOffset, offsetIndices, ist);
                                });
                                type &= anyTypes;
                            }
                        }
                        if (type > 0)
                        {
                            Add(toAdd, type, newStmt);
                            if (offsetIndices != default(OffsetInfo))
                                AddOffsetIndices(toAddOffset, offsetIndices, newStmt);
                        }
                    }
                }
            }
            foreach (IStatement ist in toRemove)
            {
                dependencyTypeOf.Remove(ist);
                offsetIndexOf.Remove(ist);
            }
            foreach (KeyValuePair<IStatement, DependencyType> entry in toAdd)
            {
                Add(entry.Value, entry.Key);
            }
            foreach (KeyValuePair<IStatement, IOffsetInfo> entry in toAddOffset)
            {
                AddOffsetIndices(entry.Value, entry.Key);
            }
        }

        private static IStatement Replace(IStatement stmt, Converter<IStatement, IStatement> converter)
        {
            IStatement newStmt = converter(stmt);
            if (!ReferenceEquals(newStmt, stmt))
                return newStmt;
            else if (stmt is AnyStatement)
            {
                AnyStatement anySt = (AnyStatement)stmt;
                AnyStatement newAnySt = new AnyStatement();
                bool replaced = false;
                foreach (IStatement ist in anySt.Statements)
                {
                    newStmt = Replace(ist, converter);
                    if (!ReferenceEquals(newStmt, ist))
                        replaced = true;
                    if (newStmt != null)
                    {
                        // flatten nested Any statements
                        if (newStmt is AnyStatement)
                            newAnySt.Statements.AddRange(((AnyStatement)newStmt).Statements);
                        else
                            newAnySt.Statements.Add(newStmt);
                    }
                }
                if (replaced)
                {
                    if (newAnySt.Statements.Count == 0)
                        return null;
                    else
                        return newAnySt;
                }
                else
                    return stmt;
            }
            else
                return stmt;
        }

        /// <summary>
        /// Change a dependency on a statement to also depend on its clones (in the same way).
        /// </summary>
        /// <param name="clonesOfStatement">Provides the clones of each statement that has clones.  May be empty.</param>
        public void AddClones(IDictionary<IStatement, IEnumerable<IStatement>> clonesOfStatement)
        {
            AddClones(delegate (IStatement ist)
            {
                IEnumerable<IStatement> clones;
                if (clonesOfStatement.TryGetValue(ist, out clones))
                    return clones;
                else
                    return null;
            });
        }

        public void AddClones(Converter<IStatement, IEnumerable<IStatement>> getClones)
        {
            List<IStatement> toRemove = new List<IStatement>();
            Dictionary<IStatement, DependencyType> toAdd = new Dictionary<IStatement, DependencyType>(new IdentityComparer<IStatement>());
            Dictionary<IStatement, IOffsetInfo> toAddOffset = new Dictionary<IStatement, IOffsetInfo>(new IdentityComparer<IStatement>());
            foreach (KeyValuePair<IStatement, DependencyType> entry in dependencyTypeOf)
            {
                IStatement stmt = entry.Key;
                if (stmt is AnyStatement)
                {
                    AnyStatement anySt = (AnyStatement)stmt;
                    AnyStatement newAnySt = new AnyStatement();
                    bool changed = false;
                    foreach (IStatement ist in anySt.Statements)
                    {
                        newAnySt.Statements.Add(ist);
                        var clones = getClones(ist);
                        if (clones != null)
                        {
                            changed = true;
                            // flatten nested Any statements
                            newAnySt.Statements.AddRange(clones);
                        }
                    }
                    if (changed)
                    {
                        toRemove.Add(stmt);
                        toAdd.Add(newAnySt, entry.Value);
                    }
                }
                else
                {
                    var clones = getClones(stmt);
                    if (clones != null)
                    {
                        DependencyType type = entry.Value;
                        IOffsetInfo offsetIndices;
                        offsetIndexOf.TryGetValue(stmt, out offsetIndices);
                        if (type > 0)
                        {
                            foreach (var clone in clones)
                            {
                                Add(toAdd, type, clone);
                                if (offsetIndices != default(OffsetInfo))
                                    AddOffsetIndices(toAddOffset, offsetIndices, clone);
                            }
                        }
                    }
                }
            }
            foreach (IStatement ist in toRemove)
            {
                dependencyTypeOf.Remove(ist);
                offsetIndexOf.Remove(ist);
            }
            foreach (KeyValuePair<IStatement, DependencyType> entry in toAdd)
            {
                Add(entry.Value, entry.Key);
            }
            foreach (KeyValuePair<IStatement, IOffsetInfo> entry in toAddOffset)
            {
                AddOffsetIndices(entry.Value, entry.Key);
            }
        }

        private void ForEachStatement(AnyStatement anySt, Action<IStatement> action)
        {
            foreach (IStatement ist in anySt.Statements)
            {
                if (ist is AnyStatement)
                    ForEachStatement((AnyStatement)ist, action);
                else
                    action(ist);
            }
        }

        /// <summary>
        /// Statements that modify variables used in this statement.  Excludes initializers and allocations.
        /// </summary>
        public IEnumerable<IStatement> Dependencies
        {
            get
            {
                return GetDependenciesOfType(DependencyType.Dependency);
            }
        }

        /// <summary>
        /// Statements that allocate (or in some cases initialize) variables used in this statement.
        /// </summary>
        /// <remarks>
        /// DeclDependencies and Dependencies must be disjoint.
        /// </remarks>
        public IEnumerable<IStatement> DeclDependencies
        {
            get
            {
                return GetDependenciesOfType(DependencyType.Declaration);
            }
        }

        /// <summary>
        /// Statements which determine whether or not this statement executes, or what its target is.
        /// </summary>
        public IEnumerable<IStatement> ContainerDependencies
        {
            get
            {
                return GetDependenciesOfType(DependencyType.Container);
            }
        }

        /// <summary>
        /// Statements which must be up-to-date before executing this statement.
        /// </summary>
        public IEnumerable<IStatement> FreshDependencies
        {
            get
            {
                return GetDependenciesOfType(DependencyType.Fresh);
            }
        }

        /// <summary>
        /// Statements that must be executed before this statement.
        /// </summary>
        /// <remarks>
        /// AnyStatements can be used to create disjunctive requirements, e.g. "either A or B must execute before this statement".
        /// </remarks>
        public IEnumerable<IStatement> Requirements
        {
            get
            {
                return GetDependenciesOfType(DependencyType.Requirement);
            }
        }

        /// <summary>
        /// Statements that must be executed before this statement, and must return a non-uniform result.
        /// </summary>
        /// <remarks>
        /// AnyStatements can be used to create disjunctive requirements, e.g. "either A or B must execute before this statement".
        /// </remarks>
        public IEnumerable<IStatement> SkipIfUniform
        {
            get
            {
                return GetDependenciesOfType(DependencyType.SkipIfUniform);
            }
        }

        /// <summary>
        /// Gets statements that modify or allocate variables that this statement mutates.
        /// </summary>
        public IEnumerable<IStatement> Overwrites
        {
            get
            {
                return GetDependenciesOfType(DependencyType.Overwrite);
            }
        }

        /// <summary>
        /// Statements whose execution invalidates the result of this statement.
        /// </summary>
        public IEnumerable<IStatement> Triggers
        {
            get
            {
                return GetDependenciesOfType(DependencyType.Trigger);
            }
        }

        public object Clone()
        {
            DependencyInformation that = new DependencyInformation();
            that.dependencyTypeOf = Clone(dependencyTypeOf);
            // the OffsetInfo values inside the offsetIndexOf dictionary are not cloned.
            that.offsetIndexOf = Clone(offsetIndexOf);
            that.IsOutput = IsOutput;
            that.IsUniform = IsUniform;
            that.IsFresh = IsFresh;
            that.ParameterDependencies = Clone(ParameterDependencies);
            return that;
        }

        private List<T> Clone<T>(List<T> list)
        {
            List<T> result = new List<T>();
            result.AddRange(list);
            return result;
        }

        private T Clone<T>(T set)
            where T : ICloneable
        {
            return (T)set.Clone();
        }

        private Dictionary<TKey, TValue> Clone<TKey, TValue>(Dictionary<TKey, TValue> that)
        {
            Dictionary<TKey, TValue> result = new Dictionary<TKey, TValue>(that.Comparer);
            foreach (KeyValuePair<TKey, TValue> entry in that)
            {
                result[entry.Key] = entry.Value;
            }
            return result;
        }

        private SortedDictionary<TKey, TValue> Clone<TKey, TValue>(SortedDictionary<TKey, TValue> that)
        {
            SortedDictionary<TKey, TValue> result = new SortedDictionary<TKey, TValue>(that.Comparer);
            foreach (KeyValuePair<TKey, TValue> entry in that)
            {
                result[entry.Key] = entry.Value;
            }
            return result;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("Dependency information:");
            sb.AppendLine("  IsOutput=" + IsOutput + "  DependsOnParameters=" + StringUtil.CollectionToString(ParameterDependencies, ","));
            sb.AppendLine("  IsUniform=" + IsUniform + "  IsFresh=" + IsFresh);
            sb.AppendLine(StringUtil.JoinColumns("  ContainerDependencies=", StringUtil.VerboseToString(ContainerDependencies)));
            sb.AppendLine(StringUtil.JoinColumns("  DeclDependencies=", StringUtil.VerboseToString(DeclDependencies)));
            sb.AppendLine(StringUtil.JoinColumns("  Dependencies=", StringUtil.VerboseToString(Dependencies)));
            sb.AppendLine(StringUtil.JoinColumns("  Requirements=", StringUtil.VerboseToString(Requirements)));
            sb.AppendLine(StringUtil.JoinColumns("  SkipIfUniform=", StringUtil.VerboseToString(SkipIfUniform)));
            sb.AppendLine(StringUtil.JoinColumns("  Triggers=", StringUtil.VerboseToString(Triggers)));
            sb.AppendLine(StringUtil.JoinColumns("  FreshDependencies=", StringUtil.VerboseToString(FreshDependencies)));
            sb.AppendLine(StringUtil.JoinColumns("  Overwrites=", StringUtil.ToString(Overwrites)));
            if (HasAnyDependencyOfType(DependencyType.Cancels))
                sb.AppendLine(StringUtil.JoinColumns("  Cancels=", StringUtil.ToString(GetDependenciesOfType(DependencyType.Cancels))));
            if (HasAnyDependencyOfType(DependencyType.NoInit))
                sb.AppendLine(StringUtil.JoinColumns("  NoInit=", StringUtil.ToString(GetDependenciesOfType(DependencyType.NoInit))));
            if (HasAnyDependencyOfType(DependencyType.Diode))
                sb.AppendLine(StringUtil.JoinColumns("  Diode=", StringUtil.ToString(GetDependenciesOfType(DependencyType.Diode))));
            if (offsetIndexOf.Count > 0)
            {
                StringBuilder sb2 = new StringBuilder();
                int count = 0;
                foreach (KeyValuePair<IStatement, IOffsetInfo> entry in offsetIndexOf)
                {
                    if (count > 0)
                        sb2.AppendLine();
                    sb2.Append("[");
                    sb2.Append(count++);
                    sb2.Append("] ");
                    IStatement ist = entry.Key;
                    sb2.Append(entry.Value);
                    sb2.Append(" ");
                    sb2.Append(ist);
                }
                sb.AppendLine(StringUtil.JoinColumns("  OffsetIndices=", sb2.ToString()));
            }
            return sb.ToString();
        }

        protected string ToString(IList<IStatement> ls)
        {
            StringBuilder sb = new StringBuilder("[");
            foreach (IStatement st in ls)
                sb.AppendLine(st.ToString());
            sb.AppendLine("]");
            return sb.ToString();
        }

#if false
        public override bool Equals(object obj)
        {
            DependencyInformation di = obj as DependencyInformation;
            if (di == null) return false;
            if (di.IsUniform != IsUniform) return false;
            if (ParameterDependencies != di.ParameterDependencies) return false;
            if (di.IsOutput != IsOutput) return false;
            if (!SetsAreEqual(Dependencies, di.Dependencies)) return false;
            //  if (!SetEquals(DeclDependencies,di.DeclDependencies)) return false;
            if (!SetsAreEqual(Requirements, di.Requirements)) return false;
            //if (!SetEquals(RequiredNumberSet,di.RequiredNumberSet)) return false;
            if (!SetsAreEqual(Triggers, di.Triggers)) return false;
            if (!SetsAreEqual(FreshDependencies, di.FreshDependencies)) return false;
            if (!SetsAreEqual(Initializers, di.Initializers)) return false;
            return true;
        }
        public override int GetHashCode()
        {
            int hash = Hash.Start;
            hash = Hash.Combine(hash, IsUniform.GetHashCode());
            hash = Hash.Combine(hash, ParameterDependencies.GetHashCode());
            hash = Hash.Combine(hash, IsOutput.GetHashCode());
            hash = Hash.Combine(hash, Enumerable.GetHashCodeAsSet(Dependencies));
            hash = Hash.Combine(hash, Enumerable.GetHashCodeAsSet(Requirements));
            hash = Hash.Combine(hash, Enumerable.GetHashCodeAsSet(Triggers));
            hash = Hash.Combine(hash, Enumerable.GetHashCodeAsSet(FreshDependencies));
            hash = Hash.Combine(hash, Enumerable.GetHashCodeAsSet(Initializers));
            return hash;
        }

        protected static bool SetsAreEqual(List<IStatement> l1, List<IStatement> l2)
        {
            if (l1.Count != l2.Count) return false;
            foreach (IStatement ist in l1) {
                if (!l2.Contains(ist)) return false;
            }
            return true;
        }
#endif

        // Compares statements by lexicographic order
        internal class StatementComparer : IComparer<IStatement>
        {
            public int Compare(IStatement x, IStatement y)
            {
                return String.Compare(x.ToString(), y.ToString(), StringComparison.InvariantCulture);
            }
        }
    }

    [Flags]
    internal enum DependencyType
    {
        /// <summary>
        /// Statements that modify variables read by this statement, i.e. read-after-write dependencies.  Excludes allocations.
        /// </summary>
        Dependency = 1,

        /// <summary>
        /// Statements that must be executed before this statement.
        /// </summary>
        /// <remarks>
        /// AnyStatements can be used to create disjunctive requirements, e.g. "either A or B must execute before this statement".
        /// </remarks>
        Requirement = 2,

        /// <summary>
        /// Statements that must be executed before this statement, and must return a non-uniform result.
        /// </summary>
        /// <remarks>
        /// AnyStatements can be used to create disjunctive requirements, e.g. "either A or B must execute before this statement".
        /// </remarks>
        SkipIfUniform = 4,

        /// <summary>
        /// Statements whose execution invalidates the result of this statement.
        /// </summary>
        Trigger = 8,

        /// <summary>
        /// Statements which must be up-to-date before executing this statement.
        /// </summary>
        Fresh = 16,

        /// <summary>
        /// Statements that allocate variables read by this statement.
        /// </summary>
        /// <remarks>
        /// DeclDependencies and Dependencies must be disjoint.
        /// </remarks>
        Declaration = 32,

        /// <summary>
        /// Statements that modify variables used in the containers of this statement.
        /// </summary>
        Container = 64,

        /// <summary>
        /// Statements that modify or allocate variables that this statement modifies.
        /// </summary>
        Overwrite = 128,
        Cancels = 256,
        NoInit = 512,
        Diode = 1024,
        All = 2047
    };

    internal class AllTriggersAttribute : ICompilerAttribute
    {
    }

    /// <summary>
    /// Represents an offset dependency between a read and a write
    /// </summary>
    internal class Offset
    {
        /// <summary>
        /// Loop counter variable
        /// </summary>
        public readonly IVariableDeclaration loopVar;
        /// <summary>
        /// (loop counter of write) - (loop counter of read) 
        /// The special value <see cref="DependencyAnalysisTransform.sequentialOffset"/> means that there is no fixed offset.
        /// </summary>
        public readonly int offset;
        /// <summary>
        /// True if the first affected element will be mutated at an earlier loop iteration, based on the loop direction.
        /// </summary>
        public readonly bool isAvailable;

        /// <summary>
        /// Create a new offset dependency
        /// </summary>
        /// <param name="loopVar">Loop counter</param>
        /// <param name="offset">(loop counter of write) - (loop counter of read)</param>
        /// <param name="isAvailable">True if the first affected element will be mutated at an earlier loop iteration, based on the loop direction</param>
        public Offset(IVariableDeclaration loopVar, int offset, bool isAvailable)
        {
            this.loopVar = loopVar;
            this.offset = offset;
            this.isAvailable = isAvailable;
        }
    }

    /// <summary>
    /// Read-only interface to OffsetInfo.
    /// </summary>
    internal interface IOffsetInfo : IEnumerable<Offset>
    {
        bool ContainsKey(IVariableDeclaration ivd);
    }

    internal class OffsetInfo : ICollection<Offset>, IOffsetInfo
    {
        HashSet<Offset> offsetOfVar;

        public IEnumerator<Offset> GetEnumerator()
        {
            if (offsetOfVar == null)
                return new HashSet<Offset>().GetEnumerator();
            return offsetOfVar.GetEnumerator();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public bool ContainsKey(IVariableDeclaration ivd)
        {
            if (offsetOfVar == null)
                return false;
            foreach (var entry in this)
            {
                if (entry.loopVar.Equals(ivd))
                    return true;
            }
            return false;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("OffsetInfo");
            if (offsetOfVar != null)
            {
                foreach (var entry in offsetOfVar)
                {
                    sb.Append("(");
                    sb.Append(entry.loopVar.Name);
                    sb.Append(",");
                    sb.Append(entry.offset);
                    sb.Append(",");
                    sb.Append(entry.isAvailable);
                    sb.Append(")");
                }
            }
            return sb.ToString();
        }

        /// <summary>
        /// Add an offset dependency
        /// </summary>
        /// <param name="ivd">Loop counter</param>
        /// <param name="offset">(loop counter of write) - (loop counter of read)</param>
        /// <param name="isAvailable">True if the first affected element will be mutated at an earlier loop iteration, based on the loop direction</param>
        public void Add(IVariableDeclaration ivd, int offset, bool isAvailable)
        {
            Add(new Offset(ivd, offset, isAvailable));
        }

        public void Add(Offset item)
        {
            if (offsetOfVar == null)
                offsetOfVar = new HashSet<Offset>();
            offsetOfVar.Add(item);
        }

        public void Clear()
        {
            if (offsetOfVar != null)
                offsetOfVar.Clear();
        }

        public bool Contains(Offset item)
        {
            throw new NotImplementedException();
        }

        public void CopyTo(Offset[] array, int arrayIndex)
        {
            throw new NotImplementedException();
        }

        public int Count
        {
            get
            {
                return (offsetOfVar == null) ? 0 : offsetOfVar.Count;
            }
        }

        public bool IsReadOnly
        {
            get
            {
                return false;
            }
        }

        public bool Remove(Offset item)
        {
            throw new NotImplementedException();
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}