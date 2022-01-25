// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Following are deliberate
#pragma warning disable 661  // 'class' defines operator == or operator != but does not override Object.GetHashCode()
#pragma warning disable 660  // 'class' defines operator == or operator != but does not override Object.Equals(object o

namespace Microsoft.ML.Probabilistic.Models
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Reflection;

    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Compiler.Transforms;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Compiler;
    using Microsoft.ML.Probabilistic.Compiler.CodeModel;
    using Microsoft.ML.Probabilistic.Compiler.Attributes;
    using Microsoft.ML.Probabilistic.Models.Attributes;

    /// <summary>
    /// A variable in a model
    /// </summary>
    /// <remarks>
    /// Variables can be base or derived.  A base Variable is explicitly declared as a variable in MSL.
    /// A derived Variable is simply an expression built from variables.
    /// For example, the expression <c>a[i]</c> is a derived Variable called an item variable.
    /// Every method that manipulates Variable objects must be aware of this distinction.
    /// </remarks>
    public abstract class Variable : IModelExpression, IVariable, CanGetContainers, HasObservedValue
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static readonly CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// Automatically generate names for variables based on their definition.
        /// </summary>
        internal static bool AutoNaming = false;

        internal int timestamp;
        private string name;
        private string nameInGeneratedCode;

        protected Variable()
        {
            timestamp = MethodInvoke.GetTimestamp();
        }

        /// <summary>
        /// Name used in generated code
        /// </summary>
        public string NameInGeneratedCode
        {
            get
            {
                if (nameInGeneratedCode == null) nameInGeneratedCode = CodeBuilder.MakeValid(Name);
                return nameInGeneratedCode;
            }
        }

        string IModelExpression.Name
        {
            get { return NameInGeneratedCode; }
        }

        /// <summary>
        /// Name
        /// </summary>
        public virtual string Name
        {
            get { return name; }
            set
            {
                if (IsBase) name = value;
                else if (IsArrayElement) ((Variable)ArrayVariable).Name = value;
                else throw new NotImplementedException();
                nameInGeneratedCode = null;
            }
        }

        /// <summary>
        /// Overridden ToString method
        /// </summary>
        /// <returns></returns>
        /// <exclude/>
        public override string ToString()
        {
            if (IsBase) return Name;
            else if (IsArrayElement)
            {
                return array.ToString() + "[" +
                       StringUtil.CollectionToString(indices, ",")
                       + "]";
            }
            else if (IsLoopIndex)
            {
                return loopRange.Name;
            }
            else return base.ToString();
        }

        internal Variable StripIndexers()
        {
            if (IsArrayElement) return ((Variable)array).StripIndexers();
            else return this;
        }

        internal List<MethodInvoke> constraints = new List<MethodInvoke>();

        /// <summary>
        /// Stores the parent factor for the variable, and the factor arguments.
        /// </summary>
        internal MethodInvoke definition;

        /// <summary>
        /// Gets the definition of this variable in the current context.  Will return
        /// null if the variable is undefined or if it is only defined in a subcontext (such as an If or Switch).
        /// </summary>
        public IModelExpression Definition
        {
            get { return GetDefinition(); }
        }

        /// <summary>
        /// True if the variable is defined in the current condition context.
        /// </summary>
        /// <remarks>
        /// Suppose we have x.SetTo(def) inside [c==0][d==0]
        /// Then x.IsDefined is true inside [c==0][d==0], [c==0][d==0][e==0], [c==0], and nothing.
        /// x.IsDefined is false inside [c==0][d==1] and [c==1].
        /// In other words, x.IsDefined is true when the current condition context is a prefix of the definition context,
        /// or when the definition context is a prefix of the current condition context.
        /// </remarks>
        public bool IsDefined
        {
            get
            {
                List<ConditionBlock> currentConditions = ConditionBlock.GetOpenBlocks<ConditionBlock>();
                return IsDefinedInContext(currentConditions);
            }
        }

        /// <summary>
        /// True if the variable is defined in the given condition context.
        /// </summary>
        /// <remarks>
        /// Suppose we have x.SetTo(def) inside [c==0][d==0]
        /// Then x.IsDefinedInContext is true inside [c==0][d==0], [c==0][d==0][e==0], [c==0], and nothing.
        /// x.IsDefinedInContext is false inside [c==0][d==1] and [c==1].
        /// In other words, IsDefinedInContext is true when the context is a prefix of the definition context,
        /// or when the definition context is a prefix of the context.
        /// </remarks>
        public bool IsDefinedInContext(List<ConditionBlock> context)
        {
            // find the base variable
            Variable parent = this;
            List<List<IModelExpression>> indices = new List<List<IModelExpression>>();
            while (parent.ArrayVariable != null)
            {
                indices.Add(parent.indices);
                parent = (Variable)parent.ArrayVariable;
            }
            indices.Reverse();
            return parent.HasDefinedItem(context, indices, 0);
        }

        /// <summary>
        /// True if prefix is a prefix of list.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="prefix"></param>
        /// <param name="list"></param>
        /// <returns></returns>
        protected static bool IsPrefixOf<T>(IReadOnlyList<T> prefix, IReadOnlyList<T> list)
        {
            if (prefix.Count > list.Count) return false;
            for (int i = 0; i < prefix.Count; i++)
            {
                if (!prefix[i].Equals(list[i])) return false;
            }
            return true;
        }

        /// <summary>
        /// True if the shorter list is a prefix of the longer list.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        protected static bool ShorterIsPrefixOfLonger<T>(IReadOnlyList<T> a, IReadOnlyList<T> b)
        {
            int count = System.Math.Min(a.Count, b.Count);
            for (int i = 0; i < count; i++)
            {
                if (!a[i].Equals(b[i])) return false;
            }
            return true;
        }

        /// <summary>
        /// True if the variable (or any item of it) is defined in the given context.
        /// </summary>
        internal bool HasAnyItemDefined()
        {
            List<ConditionBlock> currentConditions = ConditionBlock.GetOpenBlocks<ConditionBlock>();
            return HasAnyItemDefined(currentConditions);
        }

        /// <summary>
        /// True if the variable (or any item of it) is defined in the given context.
        /// </summary>
        /// <param name="currentConditions"></param>
        /// <returns></returns>
        protected bool HasAnyItemDefined(IReadOnlyList<ConditionBlock> currentConditions)
        {
            if (definition != null) return true;
            foreach (var defConditions in conditionalDefinitions.Keys)
            {
                if (ShorterIsPrefixOfLonger(defConditions, currentConditions)) return true;
            }
            if (this is HasItemVariables hiv)
            {
                ICollection<IVariable> ie = hiv.GetItemsUntyped().Values;
                foreach (IVariable irv in ie)
                {
                    Variable v = (Variable)irv;
                    if (v.HasAnyItemDefined(currentConditions)) return true;
                }
            }
            return false;
        }

        protected bool HasDefinedItem(IReadOnlyList<ConditionBlock> currentConditions, List<List<IModelExpression>> indices, int depth)
        {
            if (depth >= indices.Count) return HasAnyItemDefined(currentConditions);
            if (definition != null) return true;
            foreach (var defConditions in conditionalDefinitions.Keys)
            {
                if (ShorterIsPrefixOfLonger(defConditions, currentConditions)) return true;
            }
            if (this is HasItemVariables hiv)
            {
                foreach (KeyValuePair<IReadOnlyList<IModelExpression>, IVariable> entry in hiv.GetItemsUntyped())
                {
                    if (MayOverlap(entry.Key, indices[depth]))
                    {
                        Variable v = (Variable)entry.Value;
                        if (v.HasDefinedItem(currentConditions, indices, depth + 1)) return true;
                    }
                }
            }
            return false;
        }

        protected bool MayOverlap(IReadOnlyList<IModelExpression> list1, IReadOnlyList<IModelExpression> list2)
        {
            Assert.IsTrue(list1.Count == list2.Count);
            for (int i = 0; i < list1.Count; i++)
            {
                IModelExpression index1 = list1[i];
                IModelExpression index2 = list2[i];
                if (index1 is Range || index2 is Range) continue;
                if (index1 == index2) continue;
                if (!IsConstantScalar(index1)) continue;
                if (!IsConstantScalar(index2)) continue;
                // both are constant scalars
                HasObservedValue obs1 = (HasObservedValue)index1;
                HasObservedValue obs2 = (HasObservedValue)index2;
                if (obs1.ObservedValue.Equals(obs2.ObservedValue)) continue;
                // index1 can never equal index2
                return false;
            }
            return true;
        }

        protected bool IsConstantScalar(IModelExpression expr)
        {
            if (!(expr is Variable)) return false;
            Variable v = (Variable)expr;
            return v.IsReadOnly && v.IsObserved && (v.indices.Count == 0);
        }

        /// <summary>
        /// Enumerates all definitions made within the given context.
        /// </summary>
        /// <param name="context"></param>
        /// <returns></returns>
        /// <remarks>
        /// If context is [c==0][d==0] then a definition made inside [c==0][d==0] or [c==0][d==0][e==0] will be returned
        /// but a definition made inside [c==0] or [c==0][d==1] will not be returned.
        /// </remarks>
        internal IEnumerable<MethodInvoke> GetDefinitionsMadeWithin(List<ConditionBlock> context)
        {
            if (definition != null)
            {
                if (context.Count == 0)
                {
                    yield return definition;
                }
            }
            else
            {
                foreach (var defConditions in conditionalDefinitions.Keys)
                {
                    if (IsPrefixOf(context, defConditions))
                    {
                        yield return conditionalDefinitions[defConditions];
                    }
                }
            }
        }

        /// <summary>
        /// Stores definitions of this variable which are given in condition blocks.
        /// </summary>
        internal Dictionary<IReadOnlyList<ConditionBlock>, MethodInvoke> conditionalDefinitions =
            new Dictionary<IReadOnlyList<ConditionBlock>, MethodInvoke>(new ReadOnlyListComparer<ConditionBlock>());

        /// <summary>
        /// Field backing the IsObserved property for base variables.  Unused for non-base variables.
        /// </summary>
        protected bool isObserved;

        /// <summary>
        /// Is Observed property
        /// </summary>
        public bool IsObserved
        {
            get
            {
                // find the base variable
                Variable parent = this;
                while (parent.ArrayVariable != null) parent = (Variable)parent.ArrayVariable;
                return parent.isObserved;
            }
        }

        /// <summary>
        /// Observed value property
        /// </summary>
        object HasObservedValue.ObservedValue { get; set; }

        /// <summary>
        /// Read only property
        /// </summary>
        public abstract bool IsReadOnly { get; set; }

        internal MethodInvoke GetDefinition()
        {
            List<ConditionBlock> cb = ConditionBlock.GetOpenBlocks<ConditionBlock>();
            if (cb.Count == 0) return definition;
            if (!conditionalDefinitions.ContainsKey(cb)) return null;
            return conditionalDefinitions[cb];
        }

        internal void RemoveDefinitions()
        {
            definition = null;
            conditionalDefinitions.Clear();
        }

        internal bool HasConditionedIndices()
        {
            if (ArrayVariable != null && ((Variable)ArrayVariable).HasConditionedIndices()) return true;
            foreach (IModelExpression ind in indices)
            {
                if (ind is Variable<int>)
                {
                    Variable indexVar = (Variable)ind;
                    if (!indexVar.IsLoopIndex)
                    {
                        ConditionBlock cb = ConditionBlock.GetConditionBlock(indexVar);
                        if (cb != null) return true;
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Sets the definition for the variable, in a given condition block context
        /// </summary>
        /// <param name="methodInvoke"></param>
        internal void SetDefinition(MethodInvoke methodInvoke)
        {
            List<ConditionBlock> context = new List<ConditionBlock>(methodInvoke.Containers.OfType<ConditionBlock>());
            if (IsDefinedInContext(context))
            {
                if (context.Count == 0) throw new InvalidOperationException("Cannot define a variable more than once");
                else
                    throw new InvalidOperationException("Cannot define a variable more than once in the same condition block context: " +
                                                        Util.CollectionToString(context));
            }
            if (HasConditionedIndices()) throw new InvalidOperationException("Cannot assign to a random element of an array");
            if (context.Count == 0)
            {
                Assert.IsTrue(definition == null);
                definition = methodInvoke;
            }
            else
            {
                Assert.IsTrue(!conditionalDefinitions.ContainsKey(context));
                conditionalDefinitions[context] = methodInvoke;
            }
        }

        internal static void CreateVariableArrayFromItem(Variable item, IList<Range> ranges)
        {
            Type domainType = item.GetDomainType();
            Type variableType = typeof(Variable<>).MakeGenericType(domainType);
            MethodInfo method = variableType.GetMethod("CreateVariableArrayFromItem", BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.InvokeMethod);
            Util.Invoke(method, null, item, ranges);
        }

        /// <summary>
        /// Gets the domain type of a Variable&lt;T&gt;
        /// </summary>
        /// <returns>The type parameter T of Variable&lt;T&gt;</returns>
        /// <remarks>
        /// <c>this</c> must be a Variable&lt;T&gt;
        /// </remarks>
        public Type GetDomainType()
        {
            Type baseClass = GetType();
            do
            {
                if (baseClass.IsGenericType)
                {
                    Type gtd = baseClass.GetGenericTypeDefinition();
                    if (gtd.Equals(typeof(Variable<>))) return baseClass.GetGenericArguments()[0];
                }
                baseClass = baseClass.BaseType;
            } while (baseClass != null);
            throw new ArgumentException("Variable '" + this + "' is not a Variable<T>");
        }

        internal abstract object GetDeclaration();

        /// <summary>
        /// Gets a syntax tree which refers to this variable in MSL.
        /// </summary>
        /// <returns></returns>
        public IExpression GetExpression()
        {
            if (IsBase)
            {
                if (!IsObserved)
                {
                    // random variable
                    if (Inline)
                    {
                        if (definition != null) return definition.GetMethodInvokeExpression(inline: true);
                        else
                        {
                            if (conditionalDefinitions.Count > 1) throw new Exception("Variable " + this + " has multiple definitions so cannot be inlined");
                            foreach (KeyValuePair<IReadOnlyList<ConditionBlock>, MethodInvoke> entry in conditionalDefinitions)
                            {
                                return entry.Value.GetMethodInvokeExpression(inline: true);
                            }
                            throw new Exception("Variable " + this + " has no definition");
                        }
                    }
                    else
                    {
                        return Builder.VarRefExpr((IVariableDeclaration)GetDeclaration());
                    }
                }
                else if (!IsReadOnly)
                {
                    // given
                    return Builder.ParamRef((IParameterDeclaration)GetDeclaration());
                }
                else
                {
                    // constant
                    if (Inline)
                    {
                        HasObservedValue hov = this as HasObservedValue;
                        return Quoter.Quote(hov.ObservedValue);
                    }
                    return Builder.VarRefExpr((IVariableDeclaration)GetDeclaration());
                }
            }
            else if (IsArrayElement) return GetItemExpression();
            else if (IsLoopIndex) return loopRange.GetExpression();
            else
            {
                throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Temporary property to allow variables to be used as expressions.  Setting
        /// this true, means that wherever the variable is used, its definition will be substituted instead.
        /// </summary>
        internal bool Inline { get; set; }


        /// <summary>
        /// An expression for initialising the forward messages for this variable.
        /// </summary>
        internal IModelExpression initialiseTo;

        /// <summary>
        /// An expression for initialising the backward messages for this variable.
        /// </summary>
        internal IModelExpression initialiseBackwardTo;

        /// <summary>
        /// Factors that this variable is an argument of.
        /// </summary>
        internal List<MethodInvoke> childFactors = new List<MethodInvoke>();

        /// <summary>
        /// The indices this variable is indexed by, if any.
        /// </summary>
        internal List<IModelExpression> indices = new List<IModelExpression>();

        internal static void ForEachBaseVariable(IModelExpression expr, Action<Variable> action)
        {
            if (expr is Variable var)
            {
                if (var.IsBase)
                {
                    action(var);
                }
                else if (var.IsArrayElement)
                {
                    ForEachBaseVariable(var.ArrayVariable, action);
                    foreach (IModelExpression index in var.indices)
                    {
                        ForEachBaseVariable(index, action);
                    }
                }
                else if (var.IsLoopIndex)
                {
                    // do nothing
                }
                else
                {
                    throw new NotImplementedException();
                }
            }
        }

        #region Attributes

        /// <summary>
        /// The attributes associated with this variable.
        /// </summary>
        protected List<ICompilerAttribute> attributes = new List<ICompilerAttribute>();

        /// <summary>
        /// Inline method for adding an attribute to a variable.  This method
        /// returns the variable object, so that is can be used in an inline expression.
        /// e.g. Variable.GaussianFromMeanAndVariance(0,1).Attrib(new MyAttribute());
        /// </summary>
        /// <param name="attr">The attribute to add</param>
        /// <returns>The variable object</returns>
        public Variable Attrib(ICompilerAttribute attr)
        {
            AddAttribute(attr);
            return this;
        }

        /// <summary>
        /// Adds an attribute to this variable.  Attributes can be used
        /// to modify how inference is performed on this variable.
        /// </summary>
        /// <param name="attr">The attribute to add</param>
        public void AddAttribute(ICompilerAttribute attr)
        {
            if (IsBase)
            {
                InferenceEngine.InvalidateAllEngines(this);
                attributes.Add(attr);
            }
            else if (IsArrayElement)
            {
                ((Variable)ArrayVariable).AddAttribute(attr);
            }
            else throw new NotImplementedException();
        }

        /// <summary>
        /// Helper to add a query type attribute to this variable.
        /// </summary>
        /// <param name="queryType">The query type to use to create the attribute</param>
        public void AddAttribute(QueryType queryType)
        {
            AddAttribute(new QueryTypeCompilerAttribute(queryType));
        }

        /// <summary>
        /// Adds multiple attributes to this variable.
        /// </summary>
        /// <param name="attrs">The attributes to add</param>
        public void AddAttributes(params ICompilerAttribute[] attrs)
        {
            foreach (ICompilerAttribute obj in attrs) AddAttribute(obj);
        }

        /// <summary>
        /// Adds multiple attributes to this variable.
        /// </summary>
        /// <param name="attrs">The attributes to add</param>
        public void AddAttributes(IEnumerable<ICompilerAttribute> attrs)
        {
            foreach (ICompilerAttribute obj in attrs) AddAttribute(obj);
        }

        /// <summary>
        /// Add an attribute to the factor defining this variable.
        /// </summary>
        /// <param name="attribute"></param>
        public void AddDefinitionAttribute(ICompilerAttribute attribute)
        {
            if (definition == null) throw new NullReferenceException("No definition has been given for this variable.");
            definition.AddAttribute(attribute);
        }

        /// <summary>
        /// Get the ValueRange attribute of this variable, if any has been set, otherwise throws an exception.
        /// </summary>
        /// <returns></returns>
        /// <exception cref="ArgumentException">If the variable has no ValueRange attribute.</exception>
        public Range GetValueRange()
        {
            return GetValueRange(true);
        }

        /// <summary>
        /// Get the ValueRange attribute of this variable, if any has been set.
        /// </summary>
        /// <param name="throwIfMissing">Indicates if a missing attribute should throw an exception.</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException">If the variable has no ValueRange attribute and <paramref name="throwIfMissing"/> is true.</exception>
        public Range GetValueRange(bool throwIfMissing)
        {
            List<ValueRange> ranges = new List<ValueRange>(GetAttributes<ValueRange>());
            if (ranges.Count == 0)
            {
                if (throwIfMissing)
                    throw new ArgumentException(this +
                                                " has no ValueRange attribute, which is needed for indexing arrays.  Try adding the range as an argument to the constructor, or attaching it via .SetValueRange(myRange)");
                else
                    return null;
            }
            else
            {
                Range valueRange = ranges[0].Range;
                // the ValueRange may refer to Ranges.
                // replace all indices of this variable in the ValueRange.
                Dictionary<Range, Range> rangeReplacements = new Dictionary<Range, Range>();
                Dictionary<IModelExpression, IModelExpression> expressionReplacements = new Dictionary<IModelExpression, IModelExpression>();
                Variable v = (Variable)this;
                IVariableArray parent = v.ArrayVariable;
                while (parent != null)
                {
                    for (int i = 0; i < v.indices.Count; i++)
                    {
                        expressionReplacements.Add(parent.Ranges[i], v.indices[i]);
                    }
                    v = (Variable)parent;
                    parent = v.ArrayVariable;
                }
                return valueRange.Replace(rangeReplacements, expressionReplacements);
            }
        }

        /// <summary>
        /// Sets the ValueRange attribute of this variable, replacing any previously set.
        /// </summary>
        /// <param name="valueRange">A range defining the set of values this variable can take on.  Only meaningful for Variable&lt;int&gt; and Variable&lt;Vector&gt;</param>
        public void SetValueRange(Range valueRange)
        {
            List<ValueRange> ranges = new List<ValueRange>(GetAttributes<ValueRange>());
            if (ranges.Count == 0) AddAttribute(new ValueRange(valueRange));
            else
            {
                ranges[0].Range = valueRange;
            }
        }

        /// <summary>
        /// Sets the Sparsity attribute of this variable, replacing any previously set.
        /// </summary>
        /// <param name="sparsity">A sparsity specification for vector messages</param>
        public void SetSparsity(Sparsity sparsity)
        {
            List<SparsityAttribute> sparsities = new List<SparsityAttribute>(GetAttributes<SparsityAttribute>());
            if (sparsities.Count > 0) RemoveAllAttributes<SparsityAttribute>();
            AddAttribute(new SparsityAttribute(sparsity));
        }

        /// <summary>
        /// Get all attributes of this variable having type AttributeType.
        /// </summary>
        /// <typeparam name="AttributeType"></typeparam>
        /// <returns></returns>
        public IEnumerable<AttributeType> GetAttributes<AttributeType>() where AttributeType : ICompilerAttribute
        {
            // find the base variable
            Variable parent = this;
            while (parent.array != null) parent = (Variable)parent.array;
            foreach (ICompilerAttribute attr in parent.attributes)
            {
                if (attr is AttributeType type) yield return type;
            }
        }

        private static bool IsType<AttributeType>(object elt)
        {
            return elt is AttributeType;
        }

        /// <summary>
        /// Remove all attributes of the specified type
        /// </summary>
        /// <typeparam name="AttributeType">The attribute type to remove</typeparam>
        public void RemoveAllAttributes<AttributeType>()
        {
            // find the base variable
            Variable parent = this;
            while (parent.array != null) parent = (Variable)parent.array;
            parent.attributes.RemoveAll(IsType<AttributeType>);
        }

        /// <summary>
        /// Determines if this variable has at least one attribute of type AttributeType.
        /// </summary>
        /// <typeparam name="AttributeType">The type of attribute to look for</typeparam>
        /// <returns>True if the variable has one or more attribute of that type, false otherwise</returns>
        public bool HasAttribute<AttributeType>() where AttributeType : ICompilerAttribute
        {
            return GetAttributes<AttributeType>().GetEnumerator().MoveNext();
        }

        /// <summary>
        /// Gets the first attribute of the specified type
        /// </summary>
        /// <typeparam name="AttributeType">The type of attribute to look for</typeparam>
        /// <returns>The first attribute of the specified type</returns>
        public AttributeType GetFirstAttribute<AttributeType>() where AttributeType : ICompilerAttribute
        {
            var en = GetAttributes<AttributeType>().GetEnumerator();
            en.MoveNext();
            return en.Current;
        }

        #endregion Attributes

        #region Containers

        // The loops and conditionals this variable is contained in
        internal List<IStatementBlock> containers;

        /// <summary>
        /// The loops and conditionals this variable is contained in
        /// </summary>
        internal List<IStatementBlock> Containers
        {
            get { return containers; }
        }

        /// <summary>
        /// List of containers for this variable (ForEachBlock, IfBlock, etc.)
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public List<T> GetContainers<T>()
        {
            List<T> result = new List<T>();
            if (containers != null)
            {
                foreach (IStatementBlock sb in containers)
                {
                    if (sb is T) result.Add((T)(object)sb);
                }
            }
            return result;
        }

        internal bool HasAllContainersOpen()
        {
            List<IStatementBlock> openBlocks = StatementBlock.GetOpenBlocks();
            foreach (IStatementBlock block in Containers)
            {
                if (!openBlocks.Contains(block)) return false;
            }
            return true;
        }

        internal static List<IStatementBlock> MergeContainers(List<IStatementBlock> list1, List<IStatementBlock> list2)
        {
            List<IStatementBlock> result = new List<IStatementBlock>();
            int index1 = 0, index2 = 0;
            while (index1 < list1.Count)
            {
                if (index2 < list2.Count)
                {
                    IStatementBlock item1 = list1[index1];
                    IStatementBlock item2 = list2[index2];
                    if (item1 == item2)
                    {
                        result.Add(item1);
                        index1++;
                        index2++;
                    }
                    else if (!list2.Contains(item1))
                    {
                        result.Add(item1);
                        index1++;
                    }
                    else if (!list1.Contains(item2))
                    {
                        result.Add(item2);
                        index2++;
                    }
                    else
                    {
                        throw new ArgumentException("Container orderings are inconsistent");
                    }
                }
                else
                {
                    result.Add(list1[index1++]);
                }
            }
            while (index2 < list2.Count)
            {
                result.Add(list2[index2++]);
            }
            return result;
        }

        /// <summary>
        /// True if the variable was created inside a ForEachBlock.
        /// </summary>
        internal bool IsReplicated
        {
            get
            {
                foreach (ForEachBlock fb in GetContainers<ForEachBlock>())
                {
                    return true;
                }
                return false;
            }
        }

        /// <summary>
        /// The array this variable is an element of, if it is an array element.
        /// </summary>
        /// <remarks>null if the variable is not an array element.</remarks>
        protected IVariableArray array;

        /// <summary>
        /// The array that this variable is an element of (otherwise null).
        /// </summary>
        /// <remarks>
        /// This array may be created implicitly by applying ForEach to a variable.  In that case, the variable
        /// becomes an element of a fresh array.
        /// </remarks>
        public IVariableArray ArrayVariable
        {
            get { return array; }
        }

        /// <summary>
        /// Whether this variable is an element of an array
        /// </summary>
        public bool IsArrayElement
        {
            get { return (array != null); }
        }

        /// <summary>
        /// Helper function for implementing GetExpression.
        /// </summary>
        /// <returns></returns>
        protected IExpression GetItemExpression()
        {
            if (!IsArrayElement) throw new ArgumentException(this + " is not an item variable");
            IArrayIndexerExpression aie = Builder.ArrayIndxrExpr();
            aie.Target = array.GetExpression();
            foreach (IModelExpression ind in indices)
            {
                aie.Indices.Add(ind.GetExpression());
            }
            return aie;
        }

        public Range loopRange;

        public bool IsLoopIndex
        {
            get { return (loopRange != null); }
        }

        /// <summary>
        /// Return true if a base variable, false if derived
        /// </summary>
        internal bool IsBase
        {
            get { return !IsArrayElement && !IsLoopIndex; }
        }

        /// <summary>
        /// Modify the variable to be an element of an array, keeping the same definition.
        /// </summary>
        /// <param name="varArray"></param>
        /// <param name="inds"></param>
        internal abstract void MakeItem(IVariableArray varArray, params IModelExpression[] inds);

        /// <summary>
        /// Throws an exception if this variable and value are defined over a different set of ranges.
        /// </summary>
        /// <param name="value"></param>
        internal void CheckCompatibleIndexing(Variable value)
        {
            Set<IModelExpression> set1 = new Set<IModelExpression>();
            Models.MethodInvoke.ForEachRange(this, set1.Add);
            Set<IModelExpression> set2 = new Set<IModelExpression>();
            //set2.AddRange(value.indices);
            Models.MethodInvoke.ForEachRange(value, set2.Add);
            foreach (ForEachBlock fb in this.GetContainers<ForEachBlock>())
            {
                set1.Add(fb.Range);
            }
            foreach (ForEachBlock fb in value.GetContainers<ForEachBlock>())
            {
                set2.Add(fb.Range);
            }
            foreach (IStatementBlock sb in StatementBlock.GetOpenBlocks())
            {
                if (sb is ConditionBlock cb)
                {
                    // Remove condition variables from lhs and rhs
                    Variable condVar = cb.ConditionVariableUntyped;
                    set1.Remove(condVar);
                    set2.Remove(condVar);
                }
                else if (sb is ForEachBlock)
                {
                    // Add open ranges to rhs
                    //set2.Add(((ForEachBlock)sb).Range);
                }
            }
            Range.CheckCompatible(set1, set2);
        }

        #endregion Containers

#if false
    /// <summary>
    /// A variable item used to represent all items in the array, if this is an array.
    /// </summary>
    /// <remarks>null if the variable is not an array.</remarks>
        internal Variable item;
        public bool IsArray { get { return (item != null); } }
        public Range Range { get { return indices[0]; } }
        public Range Ranges { get { return indices; } }

        /// <summary>
        /// Get or set the elements of an array.
        /// </summary>
        /// <param name="range"></param>
        /// <returns>A derived variable that indexes <c>this</c> by <paramref name="range"/>.</returns>
        /// <remarks>
        /// When setting the elements of an array, the right hand side must be a fresh variable with no other uses.
        /// The right hand side must be an item variable with exactly the same indices as the array.
        /// </remarks>
        public virtual Variable this[Range range]
        {
            get { throw new NotSupportedException(this + " is not an array"); }
            set { throw new NotSupportedException(this + " is not an array"); }
        }
        public virtual Variable this[Range range1, Range range2]
        {
            get { throw new NotSupportedException(this + " is not an array"); }
            set { throw new NotSupportedException(this + " is not an array"); }
        }
#endif

        /// <summary>
        /// Defines a constant
        /// </summary>
        /// <typeparam name="T">The type of the constant</typeparam>
        /// <param name="value">The value of the constant</param>
        /// <returns>The constant object</returns>
        public static Variable<T> Constant<T>(T value)
        {
            Variable<T> var = new Variable<T>();
            var.ObservedValue = value;
            var.IsReadOnly = true;
            // Inherit sparsity from vector - this may be explicitly over-ridden
            if (value is Vector)
            {
                Vector vect = value as Vector;
                var.SetSparsity(vect.Sparsity);
            }
            return var;
        }

        /// <summary>
        /// Defines a constant which is a 1D array.
        /// </summary>
        /// <typeparam name="T">The element type</typeparam>
        /// <param name="value">The constant array</param>
        /// <returns>A new constant variable.</returns>
        public static VariableArray<T> Constant<T>(T[] value)
        {
            return Constant(value, new Range(value.Length));
        }

        /// <summary>
        /// Defines a constant which is a 1D array.
        /// </summary>
        /// <typeparam name="T">The element type</typeparam>
        /// <param name="value">The constant array</param>
        /// <param name="r">The range associated with this constant array</param>
        /// <returns>A new constant variable.</returns>
        //public static ConstantArray<T> Constant<T>(T[] value, Range r) { return new ConstantArray<T>(value, r); }
        public static VariableArray<T> Constant<T>(T[] value, Range r)
        {
            VariableArray<T> var = new VariableArray<T>(Constant(default(T)), r);
            var.ObservedValue = value;
            var.IsReadOnly = true;
            // Inherit sparsity from vector - this may be explicitly over-ridden
            if (value != null && value.Length > 0 && value[0] is Vector)
            {
                Vector vect = value[0] as Vector;
                var.SetSparsity(vect.Sparsity);
            }
            return var;
        }

        /// <summary>
        /// Defines a constant which is a 2D array.
        /// </summary>
        /// <typeparam name="T">The element type</typeparam>
        /// <param name="value">The constant array</param>
        /// <returns>A new constant variable.</returns>
        public static VariableArray2D<T> Constant<T>(T[,] value)
        {
            return Constant(value, new Range(value.GetLength(0)), new Range(value.GetLength(1)));
        }

        /// <summary>
        /// Defines a constant which is a 2D array.
        /// </summary>
        /// <typeparam name="T">The element type</typeparam>
        /// <param name="value">The constant array</param>
        /// <param name="r1">The range associated with the first index</param>
        /// <param name="r2">The range associated with the second index</param>
        /// <returns>A new constant variable.</returns>
        public static VariableArray2D<T> Constant<T>(T[,] value, Range r1, Range r2)
        {
            var var = new VariableArray2D<T>(Constant(default(T)), r1, r2);
            var.ObservedValue = value;
            var.IsReadOnly = true;
            // Inherit sparsity from vector - this may be explicitly over-ridden
            if (value != null && value.GetLength(0) > 0 && value.GetLength(1) > 0 && value[0, 0] is Vector)
            {
                Vector vect = value[0, 0] as Vector;
                var.SetSparsity(vect.Sparsity);
            }

            return var;
        }

        /// <summary>
        /// Defines a constant which is a 2-D jagged array
        /// </summary>
        /// <typeparam name="T">The element type</typeparam>
        /// <param name="value">The constant array</param>
        /// <param name="r1">The range associated with the first index</param>
        /// <param name="r2">The range associated with the second index</param>
        /// <returns>A new constant jagged array variable.</returns>
        public static VariableArray<VariableArray<T>, T[][]> Constant<T>(T[][] value, Range r1, Range r2)
        {
            var var = new VariableArray<VariableArray<T>, T[][]>(Constant(default(T[]), r2), r1);
            var.ObservedValue = value;
            var.IsReadOnly = true;
            // Inherit sparsity from vector - this may be explicitly over-ridden
            if (value != null && value.Length > 0 && value[0] != null &&
                value[0].Length > 0 && value[0][0] is Vector)
            {
                Vector vect = value[0][0] as Vector;
                var.SetSparsity(vect.Sparsity);
            }

            return var;
        }

        /// <summary>
        /// Defines a constant array of 2-D arrays
        /// </summary>
        /// <typeparam name="T">The element type</typeparam>
        /// <param name="value">The constant array</param>
        /// <param name="r1">The range associated with the first index</param>
        /// <param name="r2">The range associated with the second index</param>
        /// <param name="r3">The range associated with the third index</param>
        /// <returns>A new constant jagged array variable.</returns>
        /// <returns></returns>
        public static VariableArray<VariableArray2D<T>, T[][,]> Constant<T>(T[][,] value, Range r1, Range r2, Range r3)
        {
            var var = new VariableArray<VariableArray2D<T>, T[][,]>(Constant(default(T[,]), r2, r3), r1);
            var.ObservedValue = value;
            var.IsReadOnly = true;
            // Inherit sparsity from vector - this may be explicitly over-ridden
            if (value != null && value.Length > 0 && value[0] != null &&
                value[0].GetLength(0) > 0 && value[0].GetLength(1) > 0 && value[0][0, 0] is Vector)
            {
                Vector vect = value[0][0, 0] as Vector;
                var.SetSparsity(vect.Sparsity);
            }

            return var;
        }


        /// <summary>
        /// Defines a constant 3-D jagged array
        /// </summary>
        /// <typeparam name="T">The element type</typeparam>
        /// <param name="value">The constant array</param>
        /// <param name="r1">The range associated with the first index</param>
        /// <param name="r2">The range associated with the second index</param>
        /// <param name="r3">The range associated with the third index</param>
        /// <returns>A new constant jagged array variable.</returns>
        /// <returns></returns>
        public static VariableArray<VariableArray<VariableArray<T>, T[][]>, T[][][]> Constant<T>(T[][][] value, Range r1, Range r2, Range r3)
        {
            var var = new VariableArray<VariableArray<VariableArray<T>, T[][]>, T[][][]>(Constant(default(T[][]), r2, r3), r1);
            var.ObservedValue = value;
            var.IsReadOnly = true;
            // Inherit sparsity from vector - this may be explicitly over-ridden
            if (value != null && value.Length > 0 && value[0] != null &&
                value[0].Length > 0 && value[0][0] != null &&
                value[0][0].Length > 0 && value[0][0][0] is Vector)
            {
                Vector vect = value[0][0][0] as Vector;
                var.SetSparsity(vect.Sparsity);
            }
            return var;
        }

        /// <summary>
        /// Defines a constant which is a 1D array.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="value">The constant list.</param>
        /// <returns>A new constant variable.</returns>
        public static VariableArray<Variable<T>, IList<T>> Constant<T>(IList<T> value)
        {
            return Constant(value, new Range(value.Count));
        }

        /// <summary>
        /// Defines a constant which is a 1D array.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="value">The constant list.</param>
        /// <param name="r">The range associated with this constant list.</param>
        /// <returns>A new constant variable.</returns>
        public static VariableArray<Variable<T>, IList<T>> Constant<T>(IList<T> value, Range r)
        {
            var result = new VariableArray<Variable<T>, IList<T>>(Constant(default(T)), r)
            {
                ObservedValue = value,
                IsReadOnly = true
            };

            // Inherit sparsity from vector - this may be explicitly over-ridden
            if (value != null && value.Count > 0 && value[0] is Vector)
            {
                var vector = value[0] as Vector;
                result.SetSparsity(vector.Sparsity);
            }

            return result;
        }

        /// <summary>
        /// Defines a constant which is a 2-D jagged array.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="value">The constant list of lists.</param>
        /// <param name="r1">The range associated with the first index.</param>
        /// <param name="r2">The range associated with the second index.</param>
        /// <returns>A new constant jagged array variable.</returns>
        public static VariableArray<VariableArray<Variable<T>, IList<T>>, IList<IList<T>>> Constant<T>(IList<IList<T>> value, Range r1, Range r2)
        {
            var result = new VariableArray<VariableArray<Variable<T>, IList<T>>, IList<IList<T>>>(Constant(default(IList<T>), r2), r1)
            {
                ObservedValue = value,
                IsReadOnly = true
            };

            // Inherit sparsity from vector - this may be explicitly over-ridden
            if (value != null && value.Count > 0 && value[0] != null &&
                value[0].Count > 0 && value[0][0] is Vector)
            {
                Vector vector = value[0][0] as Vector;
                result.SetSparsity(vector.Sparsity);
            }

            return result;
        }

        /// <summary>
        /// Defines a constant which is a 1D array.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="value">The constant list.</param>
        /// <returns>A new constant variable.</returns>
        public static VariableArray<Variable<T>, IReadOnlyList<T>> Constant<T>(IReadOnlyList<T> value)
        {
            return Constant(value, new Range(value.Count));
        }

        /// <summary>
        /// Defines a constant which is a 1D array.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="value">The constant list.</param>
        /// <param name="r">The range associated with this constant list.</param>
        /// <returns>A new constant variable.</returns>
        public static VariableArray<Variable<T>, IReadOnlyList<T>> Constant<T>(IReadOnlyList<T> value, Range r)
        {
            var result = new VariableArray<Variable<T>, IReadOnlyList<T>>(Constant(default(T)), r)
            {
                ObservedValue = value,
                IsReadOnly = true
            };

            // Inherit sparsity from vector - this may be explicitly over-ridden
            if (value != null && value.Count > 0 && value[0] is Vector)
            {
                var vector = value[0] as Vector;
                result.SetSparsity(vector.Sparsity);
            }

            return result;
        }

        /// <summary>
        /// Defines a constant which is a 2-D jagged array.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="value">The constant list of lists.</param>
        /// <param name="r1">The range associated with the first index.</param>
        /// <param name="r2">The range associated with the second index.</param>
        /// <returns>A new constant jagged array variable.</returns>
        public static VariableArray<VariableArray<Variable<T>, IReadOnlyList<T>>, IReadOnlyList<IReadOnlyList<T>>> Constant<T>(IReadOnlyList<IReadOnlyList<T>> value, Range r1, Range r2)
        {
            var result = new VariableArray<VariableArray<Variable<T>, IReadOnlyList<T>>, IReadOnlyList<IReadOnlyList<T>>>(Constant(default(IReadOnlyList<T>), r2), r1)
            {
                ObservedValue = value,
                IsReadOnly = true
            };

            // Inherit sparsity from vector - this may be explicitly over-ridden
            if (value != null && value.Count > 0 && value[0] != null &&
                value[0].Count > 0 && value[0][0] is Vector)
            {
                Vector vector = value[0][0] as Vector;
                result.SetSparsity(vector.Sparsity);
            }

            return result;
        }

        /// <summary>
        /// Creates a variable and observes it.
        /// </summary>
        /// <typeparam name="T">The type of the observed value</typeparam>
        /// <param name="observedValue">The observed value</param>
        /// <returns>A new variable</returns>
        /// <remarks>The variable is not constant; its ObservedValue can be changed.</remarks>
        public static Variable<T> Observed<T>(T observedValue)
        {
            Variable<T> g = new Variable<T>();
            g.ObservedValue = observedValue;
            return g;
        }

        /// <summary>
        /// Creates a variable array and observes it.
        /// </summary>
        /// <typeparam name="T">The type of the observed array elements.</typeparam>
        /// <param name="observedValue">The observed value</param>
        /// <returns>A new variable</returns>
        /// <remarks>The variable is not constant; its ObservedValue can be changed.</remarks>
        public static VariableArray<T> Observed<T>(T[] observedValue)
        {
            return Observed(observedValue, new Range(observedValue.Length));
        }

        /// <summary>
        /// Creates a 2D variable array and observes it.
        /// </summary>
        /// <typeparam name="T">The type of the observed array elements.</typeparam>
        /// <param name="observedValue">The observed value</param>
        /// <returns>A new variable</returns>
        /// <remarks>The variable is not constant; its ObservedValue can be changed.</remarks>
        public static VariableArray2D<T> Observed<T>(T[,] observedValue)
        {
            return Observed(observedValue, new Range(observedValue.GetLength(0)), new Range(observedValue.GetLength(1)));
        }

        /// <summary>
        /// Creates a variable array and observes it.
        /// </summary>
        /// <typeparam name="T">The type of the observed array elements.</typeparam>
        /// <param name="observedValue">The observed value</param>
        /// <param name="r">The range used to index the array</param>
        /// <returns>A new variable</returns>
        /// <remarks>The variable is not constant; its ObservedValue can be changed.</remarks>
        public static VariableArray<T> Observed<T>(T[] observedValue, Range r)
        {
            VariableArray<T> g = new VariableArray<T>(r);
            g.ObservedValue = observedValue;
            return g;
        }

        /// <summary>
        /// Creates a jagged variable array and observes it.
        /// </summary>
        /// <typeparam name="T">The type of the observed array elements.</typeparam>
        /// <param name="observedValue">The observed value</param>
        /// <param name="r1">The range used for the first index</param>
        /// <param name="r2">The range used for the second index</param>
        /// <returns>A new variable</returns>
        /// <remarks>The variable is not constant; its ObservedValue can be changed.</remarks>
        public static VariableArray<VariableArray<T>, T[][]> Observed<T>(T[][] observedValue, Range r1, Range r2)
        {
            VariableArray<VariableArray<T>, T[][]> g = new VariableArray<VariableArray<T>, T[][]>(Constant(default(T[]), r2), r1);
            g.ObservedValue = observedValue;
            return g;
        }

        /// <summary>
        /// Creates a jagged variable array and observes it.
        /// </summary>
        /// <typeparam name="T">The type of the observed array elements.</typeparam>
        /// <param name="observedValue">The observed value</param>
        /// <param name="r1">The range used for the first index</param>
        /// <param name="r2">The range used for the second index</param>
        /// <param name="r3">The range used for the third index</param>
        /// <returns>A new variable</returns>
        /// <remarks>The variable is not constant; its ObservedValue can be changed.</remarks>
        public static VariableArray<VariableArray<VariableArray<T>, T[][]>, T[][][]> Observed<T>(T[][][] observedValue, Range r1, Range r2, Range r3)
        {
            VariableArray<VariableArray<VariableArray<T>, T[][]>, T[][][]> g =
                new VariableArray<VariableArray<VariableArray<T>, T[][]>, T[][][]>(Constant(default(T[][]), r2, r3), r1);
            g.ObservedValue = observedValue;
            return g;
        }

        /// <summary>
        /// Creates a jagged variable array and observes it.
        /// </summary>
        /// <typeparam name="T">The type of the observed array elements.</typeparam>
        /// <param name="observedValue">The observed value</param>
        /// <param name="r1">The range used for the first index</param>
        /// <param name="r2">The range used for the second index</param>
        /// <param name="r3">The range used for the third index</param>
        /// <param name="r4">The range used for the fourth index</param>
        /// <returns>A new variable</returns>
        /// <remarks>The variable is not constant; its ObservedValue can be changed.</remarks>
        public static VariableArray<VariableArray<VariableArray<VariableArray<T>, T[][]>, T[][][]>, T[][][][]> Observed<T>(T[][][][] observedValue, Range r1, Range r2, Range r3, Range r4)
        {
            var g = new VariableArray<VariableArray<VariableArray<VariableArray<T>, T[][]>, T[][][]>, T[][][][]>(Constant(default(T[][][]), r2, r3, r4), r1);
            g.ObservedValue = observedValue;
            return g;
        }

        /// <summary>
        /// Creates a 2D variable array and observes it.
        /// </summary>
        /// <typeparam name="T">The type of the observed array elements.</typeparam>
        /// <param name="observedValue">The observed value</param>
        /// <param name="r0">The range used for the first index</param>
        /// <param name="r1">The range used for the second index</param>
        /// <returns>A new variable</returns>
        /// <remarks>The variable is not constant; its ObservedValue can be changed.</remarks>
        public static VariableArray2D<T> Observed<T>(T[,] observedValue, Range r0, Range r1)
        {
            VariableArray2D<T> g = new VariableArray2D<T>(r0, r1);
            g.ObservedValue = observedValue;
            return g;
        }

#if false
        /// <summary>
        /// Creates a variable array and observes it.
        /// </summary>
        /// <typeparam name="T">The type of the observed array elements.</typeparam>
        /// <param name="observedValue">The observed value.</param>
        /// <returns>A new variable.</returns>
        /// <remarks>The variable is not constant; its ObservedValue can be changed.</remarks>
        public static VariableArray<Variable<T>, IList<T>> Observed<T>(IList<T> observedValue)
        {
            return Observed(observedValue, new Range(observedValue.Count));
        }

        /// <summary>
        /// Creates a variable array and observes it.
        /// </summary>
        /// <typeparam name="T">The type of the observed array elements.</typeparam>
        /// <param name="observedValue">The observed value.</param>
        /// <param name="r">The range used to index the array.</param>
        /// <returns>A new variable.</returns>
        /// <remarks>The variable is not constant; its ObservedValue can be changed.</remarks>
        public static VariableArray<Variable<T>, IList<T>> Observed<T>(IList<T> observedValue, Range r)
        {
            var result = new VariableArray<Variable<T>, IList<T>>(Constant(default(T)), r)
            {
                ObservedValue = observedValue
            };
            return result;
        }

        /// <summary>
        /// Creates a jagged variable array and observes it.
        /// </summary>
        /// <typeparam name="T">The type of the observed array elements.</typeparam>
        /// <param name="observedValue">The observed value.</param>
        /// <param name="r1">The range used for the first index.</param>
        /// <param name="r2">The range used for the second index.</param>
        /// <returns>A new variable.</returns>
        /// <remarks>The variable is not constant; its ObservedValue can be changed.</remarks>
        public static VariableArray<VariableArray<Variable<T>, IList<T>>, IList<IList<T>>> Observed<T>(IList<IList<T>> observedValue, Range r1, Range r2)
        {
            var result = new VariableArray<VariableArray<Variable<T>, IList<T>>, IList<IList<T>>>(Constant(default(IList<T>), r2), r1);
            result.ObservedValue = observedValue;
            return result;
        }
#endif

        /// <summary>
        /// Creates a variable array and observes it.
        /// </summary>
        /// <typeparam name="T">The type of the observed array elements.</typeparam>
        /// <param name="observedValue">The observed value.</param>
        /// <returns>A new variable.</returns>
        /// <remarks>The variable is not constant; its ObservedValue can be changed.</remarks>
        public static VariableArray<Variable<T>, IReadOnlyList<T>> Observed<T>(IReadOnlyList<T> observedValue)
        {
            return Observed(observedValue, new Range(observedValue.Count));
        }

        /// <summary>
        /// Creates a variable array and observes it.
        /// </summary>
        /// <typeparam name="T">The type of the observed array elements.</typeparam>
        /// <param name="observedValue">The observed value.</param>
        /// <param name="r">The range used to index the array.</param>
        /// <returns>A new variable.</returns>
        /// <remarks>The variable is not constant; its ObservedValue can be changed.</remarks>
        public static VariableArray<Variable<T>, IReadOnlyList<T>> Observed<T>(IReadOnlyList<T> observedValue, Range r)
        {
            return new VariableArray<Variable<T>, IReadOnlyList<T>>(Constant(default(T)), r)
            {
                ObservedValue = observedValue
            };
        }

        /// <summary>
        /// Creates a jagged variable array and observes it.
        /// </summary>
        /// <typeparam name="T">The type of the observed array elements.</typeparam>
        /// <param name="observedValue">The observed value.</param>
        /// <param name="r1">The range used for the first index.</param>
        /// <param name="r2">The range used for the second index.</param>
        /// <returns>A new variable.</returns>
        /// <remarks>The variable is not constant; its ObservedValue can be changed.</remarks>
        public static VariableArray<VariableArray<Variable<T>, IReadOnlyList<T>>, IReadOnlyList<IReadOnlyList<T>>> Observed<T>(IReadOnlyList<IReadOnlyList<T>> observedValue, Range r1, Range r2)
        {
            return new VariableArray<VariableArray<Variable<T>, IReadOnlyList<T>>, IReadOnlyList<IReadOnlyList<T>>>(Constant(default(IReadOnlyList<T>), r2), r1)
            {
                ObservedValue = observedValue
            };
        }

        /// <summary>
        /// Creates a random variable with a specified prior distribution.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="dist">The prior distribution.</param>
        /// <returns>Returns a random variable that is statistically defined by the specified prior.</returns>
        public static Variable<T> Random<T>(IDistribution<T> dist)
        {
            if (dist is HasPoint<T> hasPoint)
            {
                if (hasPoint.IsPointMass)
                {
                    // Variable.Random(PointMass(x)) -> Variable.Constant(x)
                    return Variable.Constant(hasPoint.Point).Attrib(new MarginalPrototype(dist));
                }
            }
            object distConst = Microsoft.ML.Probabilistic.Compiler.Reflection.Invoker.InvokeStatic(typeof(Variable), "Constant", dist);
            return Variable<T>.FactorUntyped(new Func<Sampleable<T>, T>(Factor.Random<T>).Method, (Variable)distConst);
        }

        /// <summary>
        /// Creates a random variable with a specified prior distribution.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <typeparam name="TDist">The distribution type.</typeparam>
        /// <param name="dist">The prior distribution.</param>
        /// <returns>Returns a random variable that is statistically defined by the specified prior.</returns>
        /// <remarks>
        /// Consider using <see cref="Variable&lt;T&gt;.Random"/> instead, as then the second type can be automatically inferred.
        /// </remarks>
        public static Variable<T> Random<T, TDist>(Variable<TDist> dist) where TDist : IDistribution<T> //, Sampleable<T>
        {
            return Variable<T>.Random(dist);
        }

        /// <summary>
        /// Creates a variable with no statistical definition.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <returns>A random variable without a statistical definition.</returns>
        /// <remarks>
        /// This method is intended for constructing variables whose statistical definition is conditional,
        /// and will be provided later using If/Case/Switch blocks.
        /// </remarks>
        public static Variable<T> New<T>()
        {
            return new Variable<T>();
        }

        /// <summary>
        /// Creates a new 'for each' block
        /// </summary>
        /// 
        /// <param name="range"></param>
        /// <returns></returns>
        public static ForEachBlock ForEach(Range range)
        {
            return new ForEachBlock(range);
        }

        /// <summary>
        /// Creates a new 'repeat' block
        /// </summary>
        /// <param name="count">The count of times to repeat the contained block</param>
        /// <returns>The repeat block</returns>
        public static RepeatBlock Repeat(Variable<double> count)
        {
            return new RepeatBlock(count);
        }

        /// <summary>
        /// Creates a 1D random variable array with a specified size.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="r">
        /// A <c>Range</c> object that is initialized with the array's length. 
        /// </param>
        /// <returns>
        /// Returns a <c>VariableArray</c> object whose size is specified by <paramref name="r"/>.
        /// </returns>
        public static VariableArray<T> Array<T>(Range r)
        {
            return new VariableArray<T>(r);
        }

        /// <summary>
        /// Creates a 1D random variable IList with a specified size.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="r">
        /// A <c>Range</c> object that is initialized with the array's length. 
        /// </param>
        /// <returns>
        /// Returns a <c>VariableArray</c> object whose size is specified by <paramref name="r"/>.
        /// </returns>
        public static VariableArray<Variable<T>, IList<T>> IList<T>(Range r)
        {
            return new VariableArray<Variable<T>, IList<T>>(new Variable<T>(), r);
        }

        /// <summary>
        /// Creates a 1D random variable ISparseList with a specified size.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="r">
        /// A <c>Range</c> object that is initialized with the array's length. 
        /// </param>
        /// <returns>
        /// Returns a <c>VariableArray</c> object whose size is specified by <paramref name="r"/>.
        /// </returns>
        public static VariableArray<Variable<T>, ISparseList<T>> ISparseList<T>(Range r)
        {
            return new VariableArray<Variable<T>, ISparseList<T>>(new Variable<T>(), r);
        }

        /// <summary>
        /// Creates a 1D random variable IArray with a specified size.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="r">
        /// A <c>Range</c> object that is initialized with the array's length. 
        /// </param>
        /// <returns>
        /// Returns a <c>VariableArray</c> object whose size is specified by <paramref name="r"/>.
        /// </returns>
        public static VariableArray<Variable<T>, IArray<T>> IArray<T>(Range r)
        {
            return new VariableArray<Variable<T>, IArray<T>>(new Variable<T>(), r);
        }

        /// <summary>
        ///  Creates a 2D random variable array with specified dimensions.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="r1">
        /// A <c>Range</c> object that is initialized with the size of the array's first dimension. 
        /// </param>
        /// <param name="r2">
        /// A <c>Range</c> object that is initialized with the size of the array's second dimension. 
        /// </param>
        /// <returns>
        /// Returns a <c>VariableArray2D</c> object whose dimensions are pecified by <paramref name="r1"/> and <paramref name="r2"/>.
        /// </returns>
        public static VariableArray2D<T> Array<T>(Range r1, Range r2)
        {
            return new VariableArray2D<T>(r1, r2);
        }

        /// <summary>
        ///  Creates a 2D random variable array with specified dimensions.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="r1">
        /// A <c>Range</c> object that is initialized with the size of the array's first dimension. 
        /// </param>
        /// <param name="r2">
        /// A <c>Range</c> object that is initialized with the size of the array's second dimension. 
        /// </param>
        /// <returns>
        /// Returns a <c>VariableArray2D</c> object whose dimensions are pecified by <paramref name="r1"/> and <paramref name="r2"/>.
        /// </returns>
        public static VariableArray2D<Variable<T>, IArray2D<T>> IArray<T>(Range r1, Range r2)
        {
            return new VariableArray2D<Variable<T>, IArray2D<T>>(new Variable<T>(), r1, r2);
        }

        /// <summary>
        ///  Creates a 3D random variable array with specified dimensions.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="r1">
        /// A <c>Range</c> object that is initialized with the size of the array's first dimension. 
        /// </param>
        /// <param name="r2">
        /// A <c>Range</c> object that is initialized with the size of the array's second dimension. 
        /// </param>
        /// <param name="r3">
        /// A <c>Range</c> object that is initialized with the size of the array's third dimension. 
        /// </param>
        /// <returns>
        /// Returns a <c>VariableArray3D</c> object whose dimensions are pecified by <paramref name="r1"/>, 
        /// <paramref name="r2"/>, and <paramref name="r3"/>.
        /// </returns>
        public static VariableArray3D<T> Array<T>(Range r1, Range r2, Range r3)
        {
            return new VariableArray3D<T>(r1, r2, r3);
        }


        /// <summary>
        /// Creates a 1D or 2D random variable array whose dimensions are specified by a list of <c>Range</c> objects.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="ranges">A list of <c>Range</c> objects, each object initialized to 
        /// the size of one of the array's dimensions. The list can contain no more than two <c>Range objects</c></param>
        /// <returns>
        /// Returns a <c>VariableArray</c> or <c>VariableArray2D</c> object whose dimensions are specified
        /// by <paramref name="ranges"/>.
        /// </returns>
        /// <exception cref="NotSupportedException">Throws <c>NotSupportedException</c> if <paramref name="ranges"/>
        /// contains more than two <c>Range</c> objects.</exception>
        public static IVariableArray Array<T>(IList<Range> ranges)
        {
            if (ranges.Count == 0) throw new ArgumentException("Range list is empty.", nameof(ranges));
            else if (ranges.Count == 1) return Variable.Array<T>(ranges[0]);
            else if (ranges.Count == 2) return Variable.Array<T>(ranges[0], ranges[1]);
            else throw new NotSupportedException("More than two ranges were specified, high rank arrays are not yet supported.");
        }

        /// <summary>
        /// Creates a 1D random variable array that contains a jagged array of 1D random variables.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="array">A <c>VariableArray</c> object that serves as the item prototype.</param>
        /// <param name="r">A <c>Range</c> object that is initialized with the array's length.</param>
        /// <returns>Returns a <c>VariableArray</c> object whose length is defined by <paramref name="r"/>. Each element of the array is
        /// a <c>VariableArray</c>object whose prototype is defined by <paramref name="array"/>.</returns>
        public static VariableArray<VariableArray<T>, T[][]> Array<T>(VariableArray<T> array, Range r)
        {
            return new VariableArray<VariableArray<T>, T[][]>(array, r);
        }

        /// <summary>
        /// Creates a 1D random variable IList that contains a jagged array of 1D random variables.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="array">A <c>VariableArray</c> object that serves as the item prototype.</param>
        /// <param name="r">A <c>Range</c> object that is initialized with the array's length.</param>
        /// <returns>Returns a <c>VariableArray</c> object whose length is defined by <paramref name="r"/>. Each element of the array is
        /// a <c>VariableArray</c>object whose prototype is defined by <paramref name="array"/>.</returns>
        public static VariableArray<VariableArray<T>, IList<T[]>> IList<T>(VariableArray<T> array, Range r)
        {
            return new VariableArray<VariableArray<T>, IList<T[]>>(array, r);
        }

        /// <summary>
        /// Creates a 1D random variable IArray that contains a jagged array of 1D random variables.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="array">A <c>VariableArray</c> object that serves as the item prototype.</param>
        /// <param name="r">A <c>Range</c> object that is initialized with the array's length.</param>
        /// <returns>Returns a <c>VariableArray</c> object whose length is defined by <paramref name="r"/>. Each element of the array is
        /// a <c>VariableArray</c>object whose prototype is defined by <paramref name="array"/>.</returns>
        public static VariableArray<VariableArray<T>, IArray<T[]>> IArray<T>(VariableArray<T> array, Range r)
        {
            return new VariableArray<VariableArray<T>, IArray<T[]>>(array, r);
        }

        /// <summary>
        /// Creates a 1D random variable array that contains a jagged array of 2D random variables.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="array">A <c>VariableArray2D</c> object that serves as the item prototype.</param>
        /// <param name="r">A <c>Range</c> object that is initialized with the array's length.</param>
        /// <returns>
        /// Returns a <c>VariableArray</c> object whose length is defined by <paramref name="r"/>. Each element of the array is
        /// a <c>VariableArray2D</c>object whose prototype is defined by <paramref name="array"/>.
        /// </returns>
        public static VariableArray<VariableArray2D<T>, T[][,]> Array<T>(VariableArray2D<T> array, Range r)
        {
            return new VariableArray<VariableArray2D<T>, T[][,]>(array, r);
        }

        /// <summary>
        /// Creates a 1D random variable IArray that contains a jagged array of 2D random variables.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="array">A <c>VariableArray2D</c> object that serves as the item prototype.</param>
        /// <param name="r">A <c>Range</c> object that is initialized with the array's length.</param>
        /// <returns>
        /// Returns a <c>VariableArray</c> object whose length is defined by <paramref name="r"/>. Each element of the array is
        /// a <c>VariableArray2D</c>object whose prototype is defined by <paramref name="array"/>.
        /// </returns>
        public static VariableArray<VariableArray2D<T>, IArray<T[,]>> IArray<T>(VariableArray2D<T> array, Range r)
        {
            return new VariableArray<VariableArray2D<T>, IArray<T[,]>>(array, r);
        }

        /// <summary>
        /// Creates a 2-D random variable array that contains a jagged array of 1D random variables.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="array">A <c>VariableArray</c> object that serves as the item prototype.</param>
        /// <param name="r1">A <c>Range</c> object that is initialized with the size of the array's first dimension. </param>
        /// <param name="r2">A <c>Range</c> object that is initialized with the size of the array's second dimension. </param>
        /// <returns>
        /// Returns a <c>VariableArray2D</c> object whose dimensions are defined by <paramref name="r1"/> and <paramref name="r2"/>.
        /// Each element of the array is a <c>VariableArray</c>object whose prototype is defined by <paramref name="array"/>.
        /// </returns>
        public static VariableArray2D<VariableArray<T>, T[,][]> Array<T>(VariableArray<T> array, Range r1, Range r2)
        {
            return new VariableArray2D<VariableArray<T>, T[,][]>(array, r1, r2);
        }

        /// <summary>
        /// Create a 2-D random variable IArray2D that contains a jagged array of 1D random variables.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="array">A <c>VariableArray</c> object that serves as the item prototype.</param>
        /// <param name="r1">A <c>Range</c> object that is initialized with the size of the array's first dimension. </param>
        /// <param name="r2">A <c>Range</c> object that is initialized with the size of the array's second dimension. </param>
        /// <returns>
        /// Returns a <c>VariableArray2D</c> object whose dimensions are defined by <paramref name="r1"/> and <paramref name="r2"/>.
        /// Each element of the array is a <c>VariableArray</c>object whose prototype is defined by <paramref name="array"/>.
        /// </returns>
        public static VariableArray2D<VariableArray<T>, IArray2D<T[]>> IArray<T>(VariableArray<T> array, Range r1, Range r2)
        {
            return new VariableArray2D<VariableArray<T>, IArray2D<T[]>>(array, r1, r2);
        }

        /// <summary>
        ///  Creates a 3-D <c>VariableArray</c> object that contains a jagged array of 1D random variables.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="array">A<c>VariableArray</c> object that serves as the item prototype.</param>
        /// <param name="r1">A <c>Range</c> object that is initialized with the size of the array's first dimension. </param>
        /// <param name="r2">A <c>Range</c> object that is initialized with the size of the array's second dimension. </param>
        /// <param name="r3">A <c>Range</c> object that is initialized with the size of the array's third dimension. </param>
        /// <returns>
        /// Returns a <c>VariableArray3D</c> object whose dimensions are defined by <paramref name="r1"/>, <paramref name="r2"/>,
        /// and <paramref name="r3"/>.
        /// Each element of the array is a 1D <c>VariableArray</c>object whose prototype is defined by <paramref name="array"/>.
        /// </returns>
        public static VariableArray3D<VariableArray<T>, T[,,][]> Array<T>(VariableArray<T> array, Range r1, Range r2, Range r3)
        {
            return new VariableArray3D<VariableArray<T>, T[,,][]>(array, r1, r2, r3);
        }

        /// <summary>
        /// Create a 1D array of 1D random variable arrays
        /// </summary>
        /// <typeparam name="TItem">The variable type of an item after two levels of indexing.</typeparam>
        /// <typeparam name="TArray">The domain type of an item.</typeparam>
        /// <param name="array">A variable object that serves as a prototype for the array elements.</param>
        /// <param name="r">A <c>Range</c> object that specifies the array length.</param>
        /// <returns>Returns a <c>VariableArray</c> object whose length is defined by <paramref name="r"/>. Each element of this
        /// array is a object of type <c>VariableArray&lt;TItem,TArray&gt;</c></returns>
        public static VariableArray<VariableArray<TItem, TArray>, TArray[]> Array<TItem, TArray>(VariableArray<TItem, TArray> array, Range r)
            where TItem : Variable, SettableTo<TItem>, ICloneable
        {
            return new VariableArray<VariableArray<TItem, TArray>, TArray[]>(array, r);
        }

        /// <summary>
        /// Create a 1D IList of 1D random variable arrays
        /// </summary>
        /// <typeparam name="TItem">The variable type of an item after two levels of indexing.</typeparam>
        /// <typeparam name="TArray">The domain type of an item.</typeparam>
        /// <param name="array">A variable object that serves as a prototype for the array elements.</param>
        /// <param name="r">A <c>Range</c> object that specifies the array length.</param>
        /// <returns>Returns a <c>VariableArray</c> object whose length is defined by <paramref name="r"/>. Each element of this
        /// array is a object of type <c>VariableArray&lt;TItem,TArray&gt;</c></returns>
        public static VariableArray<VariableArray<TItem, TArray>, IList<TArray>> IList<TItem, TArray>(VariableArray<TItem, TArray> array, Range r)
            where TItem : Variable, SettableTo<TItem>, ICloneable
        {
            return new VariableArray<VariableArray<TItem, TArray>, IList<TArray>>(array, r);
        }

        /// <summary>
        /// Create a 1D IArray of 1D random variable arrays
        /// </summary>
        /// <typeparam name="TItem">The variable type of an item after two levels of indexing.</typeparam>
        /// <typeparam name="TArray">The domain type of an item.</typeparam>
        /// <param name="array">A variable object that serves as a prototype for the array elements.</param>
        /// <param name="r">A <c>Range</c> object that specifies the array length.</param>
        /// <returns>Returns a <c>VariableArray</c> object whose length is defined by <paramref name="r"/>. Each element of this
        /// array is a object of type <c>VariableArray&lt;TItem,TArray&gt;</c></returns>
        public static VariableArray<VariableArray<TItem, TArray>, IArray<TArray>> IArray<TItem, TArray>(VariableArray<TItem, TArray> array, Range r)
            where TItem : Variable, SettableTo<TItem>, ICloneable
        {
            return new VariableArray<VariableArray<TItem, TArray>, IArray<TArray>>(array, r);
        }

        /// <summary>
        /// Create a 1D array of random variables
        /// </summary>
        /// <typeparam name="TItem">The variable type of an item.</typeparam>
        /// <typeparam name="TArray">The domain type of the variable.</typeparam>
        /// <param name="itemPrototype">A variable object that serves as a prototype for the array elements.</param>
        /// <param name="r">A <c>Range</c> object that is initialized with the array's length.</param>
        /// <returns>Returns a <c>VariableArray</c> object whose length is also defined by <paramref name="r"/>. Each element of this
        /// array is a object of type <c>TItem</c> whose prototype is defined by <paramref name="itemPrototype"/>.</returns>
        public static VariableArray<TItem, TArray> Array<TItem, TArray>(TItem itemPrototype, Range r)
            where TItem : Variable, SettableTo<TItem>, ICloneable
        {
            return new VariableArray<TItem, TArray>(itemPrototype, r);
        }

        /// <summary>
        /// Applies a constraint using a constraint method with one argument.
        /// </summary>
        /// <param name="constraint">The method that represents the constraint</param>
        /// <param name="arg1">The argument for the constraint</param>
        /// <returns></returns>
        public static void Constrain<T>(Action<T> constraint, Variable<T> arg1)
        {
            ConstrainInternal(constraint.Method, arg1);
        }

        /// <summary>
        /// Applies a constraint using a constraint method with two arguments.
        /// </summary>
        /// <param name="constraint">The method that represents the constraint</param>
        /// <param name="arg1">First argument for the constraint</param>
        /// <param name="arg2">Second argument for the constraint</param>
        /// <returns></returns>
        public static void Constrain<T1, T2>(Action<T1, T2> constraint, Variable<T1> arg1, Variable<T2> arg2)
        {
            ConstrainInternal(constraint.Method, arg1, arg2);
        }

        /// <summary>
        /// Applies a constraint using a constraint method with three arguments.
        /// </summary>
        /// <param name="constraint">The method that represents the constraint</param>
        /// <param name="arg1">First argument for the constraint</param>
        /// <param name="arg2">Second argument for the constraint</param>
        /// <param name="arg3">Third argument for the constraint</param>
        /// <returns></returns>
        public static void Constrain<T1, T2, T3>(Action<T1, T2, T3> constraint, Variable<T1> arg1, Variable<T2> arg2, Variable<T3> arg3)
        {
            ConstrainInternal(constraint.Method, arg1, arg2, arg3);
        }

        /// <summary>
        /// Applies a constraint using a constraint method with four arguments.
        /// </summary>
        /// <param name="constraint">The method that represents the constraint</param>
        /// <param name="arg1">First argument for the constraint</param>
        /// <param name="arg2">Second argument for the constraint</param>
        /// <param name="arg3">Third argument for the constraint</param>
        /// <param name="arg4">Fourth argument for the constraint</param>
        /// <returns></returns>
        public static void Constrain<T1, T2, T3, T4>(Action<T1, T2, T3, T4> constraint, Variable<T1> arg1, Variable<T2> arg2, Variable<T3> arg3, Variable<T4> arg4)
        {
            ConstrainInternal(constraint.Method, arg1, arg2, arg3, arg4);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="methodInfo"></param>
        /// <param name="args"></param>
        protected static void ConstrainInternal(MethodInfo methodInfo, params Variable[] args)
        {
            MethodInvoke mi = new MethodInvoke(methodInfo, args);
            foreach (Variable variable in args)
            {
                ForEachBaseVariable(variable, delegate (Variable v)
                {
                    InferenceEngine.InvalidateAllEngines(v);
                    v.constraints.Add(mi);
                });
            }
        }

        //****************************** FACTOR METHODS ********************************

#region Factor convenience methods

        /****************** Convenience methods for various commonly used distributions and factors*********************/

        /// <summary>
        /// Creates a Gaussian-distributed random variable with mean and precision represented
        /// by random variables.
        /// </summary>
        /// <param name="mean">A <c>double</c> random variable that represents the mean value.</param>
        /// <param name="precision">A <c>double</c> random variable that represents the precision value.</param>
        /// <returns>
        /// Returns a Gaussian-distributed random variable with the specified mean and precision.
        /// </returns>
        /// <remarks>The variance is 1/precision.</remarks>
        public static Variable<double> GaussianFromMeanAndPrecision(Variable<double> mean, Variable<double> precision)
        {
            return Variable<double>.Factor(Factor.Gaussian, mean, precision);
        }

        /// <summary>
        /// Creates a Gaussian-distributed random variable with specified mean and precision.
        /// </summary>
        /// <param name="mean">The mean.</param>
        /// <param name="precision">The precision.</param>
        /// <returns>
        /// Returns a Gaussian-distributed random variable with the specified mean and precision.
        /// </returns>
        /// <remarks>The variance is 1/precision.</remarks>
        public static Variable<double> GaussianFromMeanAndPrecision(double mean, double precision)
        {
            return GaussianFromMeanAndPrecision(Constant(mean), Constant(precision));
            //return Variable<double>.Random(Constant(new Gaussian(mean, 1.0 / precision)));
        }


        /// <summary>
        /// Creates a Gaussian-distributed random variable with the mean and variance represented by random variables.
        /// </summary>
        /// <param name="mean">A <c>double</c> random variable that represents the mean.</param>
        /// <param name="variance">A <c>double</c> random variable that represents the variance.</param>
        /// <returns>Returns a Gaussian-distributed random variable with the specified mean and variance.</returns>
        public static Variable<double> GaussianFromMeanAndVariance(Variable<double> mean, Variable<double> variance)
        {
            return Variable<double>.Factor(Factor.GaussianFromMeanAndVariance, mean, variance);
        }

        /// <summary>
        /// Creates a Gaussian-distributed random variable with a specified variance, and the mean
        /// represented by a random variable.
        /// </summary>
        /// <param name="mean">An <c>int</c> random variable that represents the mean.</param>
        /// <param name="variance">The variance.</param>
        /// <returns>Returns a Gaussian-distributed random variable with the specified mean and variance.</returns>
        public static Variable<double> GaussianFromMeanAndVariance(Variable<double> mean, double variance)
        {
            return GaussianFromMeanAndVariance(mean, Constant(variance));
            //return Variable<double>.Factor(Factor.Gaussian, mean, 1.0 / variance);
        }

        /// <summary>
        /// Creates a Gaussian-distributed random variable with specified mean and variance.
        /// </summary>
        /// <param name="mean">The mean.</param>
        /// <param name="variance">The variance.</param>
        /// <returns>Returns a Gaussian-distributed random variable with the specified mean and variance.</returns>
        public static Variable<double> GaussianFromMeanAndVariance(double mean, double variance)
        {
            return GaussianFromMeanAndVariance(Constant(mean), Constant(variance));
            //return Variable<double>.Random(Constant(new Gaussian(mean, variance)));
        }

        /// <summary>
        ///  Returns a random variable that is statistically defined by a truncated Gaussian distribution
        ///  with specified mean, variance and bounds. 
        /// </summary>
        /// <param name="mean">The mean.</param>
        /// <param name="variance">The variance.</param>
        /// <param name="lowerBound">The distribution's upper bound.</param>
        /// <param name="upperBound">The distribution's lower bound.</param>
        /// <returns>Returns a truncated Gaussian-distributed random variable with the specified mean, variance,
        /// and bounds.</returns>
        public static Variable<double> TruncatedGaussian(double mean, double variance, double lowerBound, double upperBound)
        {
            return Variable<double>.Random(new TruncatedGaussian(mean, variance, lowerBound, upperBound));
        }

        /// <summary>
        ///  Returns a random variable that is statistically defined by a truncated Gamma distribution
        ///  with specified shape, rate and bounds. 
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <param name="rate">The rate.</param>
        /// <param name="lowerBound">The distribution's upper bound.</param>
        /// <param name="upperBound">The distribution's lower bound.</param>
        /// <returns>Returns a truncated Gamma-distributed random variable with the specified shape, rate,
        /// and bounds.</returns>
        public static Variable<double> TruncatedGammaFromShapeAndRate(Variable<double> shape, Variable<double> rate, Variable<double> lowerBound, Variable<double> upperBound)
        {
            return Variable<double>.Factor(Factor.TruncatedGammaFromShapeAndRate, shape, rate, lowerBound, upperBound);
        }

        /// <summary>
        /// Returns a random variable over a sparse list domain where each element is statistically
        /// defined in terms of the corresponding mean and precision elements in sparse vectors
        /// of means and precisions
        /// </summary>
        /// <param name="mean">The sparse list of means variable</param>
        /// <param name="precision">The sparse list of precisions variable</param>
        /// <returns>A <see cref="SparseGaussianList"/> distributed random variable</returns>
        public static Variable<ISparseList<double>> GaussianListFromMeanAndPrecision(Variable<ISparseList<double>> mean, Variable<ISparseList<double>> precision)
        {
            return Variable<ISparseList<double>>.Factor(SparseGaussianList.Sample, mean, precision);
        }

        /// <summary>
        /// Returns a random variable over a sparse list domain where each element is statistically
        /// defined in terms of the corresponding mean and precision elements in sparse vectors
        /// of means and precisions
        /// </summary>
        /// <param name="mean">The sparse list of means</param>
        /// <param name="precision">The sparse list of precisions variable</param>
        /// <returns>A <see cref="SparseGaussianList"/> distributed random variable</returns>
        public static Variable<ISparseList<double>> GaussianListFromMeanAndPrecision(ISparseList<double> mean, Variable<ISparseList<double>> precision)
        {
            return Variable<ISparseList<double>>.Factor(SparseGaussianList.Sample, mean, precision);
        }

        /// <summary>
        /// Returns a random variable over a sparse vector domain where each element is statistically
        /// defined in terms of the corresponding mean and precision elements in sparse vectors
        /// of means and precisions
        /// </summary>
        /// <param name="mean">The sparse list of means variable</param>
        /// <param name="precision">The sparse list of precisions</param>
        /// <returns>A <see cref="SparseGaussianList"/> distributed random variable</returns>
        public static Variable<ISparseList<double>> GaussianListFromMeanAndPrecision(Variable<ISparseList<double>> mean, ISparseList<double> precision)
        {
            return Variable<ISparseList<double>>.Factor(SparseGaussianList.Sample, mean, precision);
        }

        /// <summary>
        /// Returns a random variable over a sparse vector domain where each element is statistically
        /// defined in terms of the corresponding mean and precision elements in sparse vectors
        /// of means and precisions
        /// </summary>
        /// <param name="mean">The sparse list of means</param>
        /// <param name="precision">The sparse list of precisions</param>
        /// <returns>A <see cref="SparseGaussianList"/> distributed random variable</returns>
        public static Variable<ISparseList<double>> GaussianListFromMeanAndPrecision(ISparseList<double> mean, ISparseList<double> precision)
        {
            return Variable<ISparseList<double>>.Random(SparseGaussianList.FromMeanAndPrecision(mean, precision));
        }

        /// <summary>
        /// Creates a vector Gaussian-distributed random vector with the mean and precision matrix represented
        /// by random variables.
        /// </summary>
        /// <param name="mean">A <c>Vector</c> random variable that represents the mean.</param>
        /// <param name="precision">A <c>PositiveDefiniteMatrix</c> random variable that represents the precision matrix.</param>
        /// <returns>Returns a vector Gaussian-distributed random variable with the specified mean and precision matrix.</returns>
        public static Variable<Vector> VectorGaussianFromMeanAndPrecision(Variable<Vector> mean, Variable<PositiveDefiniteMatrix> precision)
        {
            return Variable<Vector>.Factor(Factor.VectorGaussian, mean, precision);
        }

        /// <summary>
        /// Creates a Gaussian-distributed random vector with a specified mean and precision matrix.
        /// </summary>
        /// <param name="mean">A <c>Vector</c> object that specifies the mean values.</param>
        /// <param name="precision">A <c>PositiveDefiniteMatrix</c> object that specified the precision matrix.</param>
        /// <returns>Returns a vector Gaussian-distributed random variable with the specified mean and precision matrix.</returns>
        public static Variable<Vector> VectorGaussianFromMeanAndPrecision(Vector mean, PositiveDefiniteMatrix precision)
        {
            return VectorGaussianFromMeanAndPrecision(Constant(mean), Constant(precision));
            //return Random<Vector>(VectorGaussian.FromMeanAndPrecision(mean, precision));
        }

        /// <summary>
        /// Creates a Gaussian-distributed random vector from a mean vector and variance positive definite matrix.
        /// </summary>
        /// <param name="mean">The mean vector of the Gaussian</param>
        /// <param name="variance">The variance matrix of the Gaussian</param>
        /// <returns>Gaussian-distributed random vector variable</returns>
        public static Variable<Vector> VectorGaussianFromMeanAndVariance(Vector mean, PositiveDefiniteMatrix variance)
        {
            return VectorGaussianFromMeanAndVariance(Constant(mean), Constant(variance));
            //return Random<Vector>(VectorGaussian.FromMeanAndVariance(mean, variance));
        }

        /// <summary>
        /// Creates a Gaussian-distributed random vector from a mean vector and variance positive definite matrix.
        /// </summary>
        /// <param name="mean">A variable containing mean vector of the Gaussian</param>
        /// <param name="variance">The variance matrix of the Gaussian</param>
        /// <returns>Gaussian-distributed random vector variable</returns>
        public static Variable<Vector> VectorGaussianFromMeanAndVariance(Variable<Vector> mean, PositiveDefiniteMatrix variance)
        {
            return VectorGaussianFromMeanAndVariance(mean, Constant(variance));
            //return VectorGaussianFromMeanAndPrecision(mean, variance.Inverse());
        }

        /// <summary>
        /// Creates a Gaussian-distributed random vector from a mean vector and variance positive definite matrix.
        /// </summary>
        /// <param name="mean">A variable containing mean vector of the Gaussian</param>
        /// <param name="variance">The variance matrix of the Gaussian</param>
        /// <returns>Gaussian-distributed random vector variable</returns>
        public static Variable<Vector> VectorGaussianFromMeanAndVariance(Variable<Vector> mean, Variable<PositiveDefiniteMatrix> variance)
        {
            return Variable<Vector>.Factor(VectorGaussian.SampleFromMeanAndVariance, mean, variance);
        }

        /// <summary>
        /// Creates a Gamma-distributed random variable with specified shape and scale parameters.
        /// </summary>
        /// <param name="shape">The shape parameter value.</param>
        /// <param name="scale">The scale parameter value.</param>
        /// <returns>Returns a Gamma-distributed random variable with the specified shape and scale parameters.</returns>
        public static Variable<double> GammaFromShapeAndScale(double shape, double scale)
        {
            return GammaFromShapeAndScale(Constant(shape), Constant(scale));
            //return Variable<double>.Random<double>(new Gamma(shape, scale));
        }

        /// <summary>
        /// Creates a Gamma-distributed random variable with its shape and scale parameters represented by random variables.
        /// </summary>
        /// <param name="shape">A <c>double</c> random variable that represents the shape parameter.</param>
        /// <param name="scale">A <c>double</c> random variable that represents the scale parameter.</param>
        /// <returns>Returns a Gamma-distributed random variable with the specified shape and scale parameters.</returns>
        public static Variable<double> GammaFromShapeAndScale(Variable<double> shape, Variable<double> scale)
        {
            return Variable<double>.Factor(Gamma.Sample, shape, scale);
        }

        /// <summary>
        /// Creates a Gamma-distributed random variable with specified shape and rate parameters.
        /// </summary>
        /// <param name="shape">The shape parametervalue.</param>
        /// <param name="rate">The rate parameter value.</param>
        /// <returns>Returns a Gamma-distributed random variable with the specified shape and rate parameters.</returns>
        public static Variable<double> GammaFromShapeAndRate(double shape, double rate)
        {
            return GammaFromShapeAndRate(Constant(shape), Constant(rate));
            //return Variable<double>.Random<double>(Gamma.FromShapeAndRate(shape, rate));
        }

        /// <summary>
        /// Creates a Gamma-distributed random variable with its shape and rate parameters represented by random variables.
        /// </summary>
        /// <param name="shape">A <c>double</c> random variable that represents the shape parameter.</param>
        /// <param name="rate">A <c>double</c> random variable that represents the rate parameter.</param>
        /// <returns>Returns a Gamma-distributed random variable with the specified shape and rate parameters.</returns>
        public static Variable<double> GammaFromShapeAndRate(Variable<double> shape, Variable<double> rate)
        {
            return Variable<double>.Factor(Factor.GammaFromShapeAndRate, shape, rate);
        }

        /// <summary>
        /// Creates a Gamma-distributed random variable with specified mean and variance parameters.
        /// </summary>
        /// <param name="mean">The mean parameter value.</param>
        /// <param name="variance">The variance parameter value.</param>
        /// <returns>Returns a Gamma-distributed random variable with the specified mean and variance parameters.</returns>
        public static Variable<double> GammaFromMeanAndVariance(double mean, double variance)
        {
            return Variable<double>.Random<double>(Gamma.FromMeanAndVariance(mean, variance));
        }

        /// <summary>
        /// Creates a Gamma-distributed random variable with its mean and variance parameters represented by random variables.
        /// </summary>
        /// <param name="mean">A <c>double</c> random variable that represents the mean parameter.</param>
        /// <param name="variance">A <c>double</c> random variable that represents the variance parameter.</param>
        /// <returns>Returns a Gamma-distributed random variable with the specified mean and variance parameters.</returns>
        public static Variable<double> GammaFromMeanAndVariance(Variable<double> mean, Variable<double> variance)
        {
            return Variable<double>.Factor(Gamma.SampleFromMeanAndVariance, mean, variance);
        }

        /// <summary>
        /// Creates a Wishart-distributed random variable with specified shape and scale.
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <param name="scale">The scale matrix.</param>
        /// <returns>Returns a Wishart-distributed random variable with the specified shape and scale matrix.</returns>
        public static Variable<PositiveDefiniteMatrix> WishartFromShapeAndScale(double shape,
                                                                                PositiveDefiniteMatrix scale)
        {
            return WishartFromShapeAndScale(shape, Constant(scale));
            //return Variable<PositiveDefiniteMatrix>.Random(Wishart.FromShapeAndScale(shape, scale));
        }

        /// <summary>
        /// Creates a Wishart-distributed random matrix with the shape and scale represented by random variables.
        /// </summary>
        /// <param name="shape">A <c>double</c> random variable that represents the shape.</param>
        /// <param name="scale">A <c>PositiveDefiniteMatrix</c> random variable that represents the scale matrix.</param>
        /// <returns>Returns a Wishart-distributed random variable with the specified shape and scale matrix.</returns>
        public static Variable<PositiveDefiniteMatrix> WishartFromShapeAndScale(Variable<double> shape,
                                                                                Variable<PositiveDefiniteMatrix> scale)
        {
            return Variable<PositiveDefiniteMatrix>.Factor(Wishart.SampleFromShapeAndScale, shape, scale);
        }

        /// <summary>
        /// Creates a Wishart-distributed random matrix with the shape and rate represented by random variables.
        /// </summary>
        /// <param name="shape">A <c>double</c> random variable that represents the shape.</param>
        /// <param name="rate">A <c>PositiveDefiniteMatrix</c> random variable that represents the rate matrix.</param>
        /// <returns>Returns a Wishart-distributed random variable with the specified shape and rate matrix.</returns>
        public static Variable<PositiveDefiniteMatrix> WishartFromShapeAndRate(Variable<double> shape,
                                                                               Variable<PositiveDefiniteMatrix> rate)
        {
            return Variable<PositiveDefiniteMatrix>.Factor(Wishart.SampleFromShapeAndRate, shape, rate);
        }

        /// <summary>
        /// Creates a Boolean random variable with a specified probability of being true.
        /// </summary>
        /// <param name="probTrue">A <c>double</c>value from [0, 1] that specifies the probability that
        /// the variable is true.</param>
        /// <returns>Returns a Boolean random variable.</returns>
        public static Variable<bool> Bernoulli(double probTrue)
        {
            return Variable<bool>.Random(Constant(new Bernoulli(probTrue)));
        }

        /// <summary>
        /// Creates a Boolean random variable with the probability of being true specified by a random variable.
        /// </summary>
        /// <param name="probTrue">A <c>double</c> random variable over [0,1], typically statistically defined by a
        /// <c>Beta</c> distribution, that specifies the probability that the output variable is true.</param>
        /// <returns>Returns a Boolean random variable.</returns>
        public static Variable<bool> Bernoulli(Variable<double> probTrue)
        {
            return Variable<bool>.Factor(Factor.Bernoulli, probTrue);
        }

        /// <summary>
        /// Creates a random variable whose domain is a sparse list of bools.
        /// </summary>
        /// <param name="probTrue">The sparse list of probTrue elements</param>
        /// <returns>A <see cref="SparseBernoulliList"/> distributed random variable</returns>
        public static Variable<ISparseList<bool>> BernoulliList(Variable<ISparseList<double>> probTrue)
        {
            return Variable<ISparseList<bool>>.Factor(SparseBernoulliList.Sample, probTrue);
        }

        /// <summary>
        /// Creates a random variable whose domain is a sparse list of bools.
        /// </summary>
        /// <param name="probTrue">The sparse list of probTrue elements</param>
        /// <returns>A <see cref="SparseBernoulliList"/> distributed random variable</returns>
        public static VariableArray<Variable<bool>, ISparseList<bool>> BernoulliList(VariableArray<Variable<bool>, ISparseList<double>> probTrue)
        {
            var list = ISparseList<bool>(probTrue.Range);
            list.SetTo(Variable<ISparseList<bool>>.Factor(SparseBernoulliList.Sample, probTrue));
            return list;
        }

        /// <summary>
        /// Creates a random variable whose domain is a sparse list of bools.
        /// </summary>
        /// <param name="probTrue">The sparse list of probTrue elements</param>
        /// <returns>A <see cref="SparseBernoulliList"/> distributed random variable</returns>
        public static VariableArray<Variable<bool>, ISparseList<bool>> BernoulliList(ISparseList<double> probTrue)
        {
            var range = new Range(probTrue.Count);
            var list = ISparseList<bool>(range);
            list.SetTo(Variable<ISparseList<bool>>.Random(SparseBernoulliList.FromProbTrue(probTrue)));
            return list;
        }

        /// <summary>
        /// Creates a random variable whose domain is a variable-length list of type integer.
        /// </summary>
        /// <param name="probInSubset">The probability of a given integer being in the random list. This is given as random variable over a sparse list - i.e. most of the probabilities are the same.</param>
        /// <returns>A <see cref="Distributions.BernoulliIntegerSubset"/> distributed random variable</returns>
        public static Variable<IList<int>> BernoulliIntegerSubset(Variable<ISparseList<double>> probInSubset)
        {
            // TODO: if the input is an array, its range becomes the output's valueRange
            return Variable<IList<int>>.Factor(Distributions.BernoulliIntegerSubset.Sample, probInSubset);
        }

        /// <summary>
        /// Creates a random variable whose domain is a variable-length list of type integer.
        /// </summary>
        /// <param name="probInSubset">The probability of a given integer being in the random list. This is given as random variable over a sparse list - i.e. most of the probabilities are the same.</param>
        /// <returns>A <see cref="Distributions.BernoulliIntegerSubset"/> distributed random variable</returns>
        public static Variable<IList<int>> BernoulliIntegerSubset(VariableArray<Variable<double>, ISparseList<double>> probInSubset)
        {
            var subset = Variable<IList<int>>.Factor(Distributions.BernoulliIntegerSubset.Sample, probInSubset);
            var valueRange = probInSubset.Range;
            subset.SetValueRange(valueRange);
            return subset;
        }

        /// <summary>
        /// Creates a random variable whose domain is a variable-length list of type integer.
        /// </summary>
        /// <param name="probInSubset">The probability of a given integer being in the random list. This is given as a sparse list - i.e. most of the probabilities are the same.</param>
        /// <returns>A <see cref="Distributions.BernoulliIntegerSubset"/> distributed random variable</returns>
        public static Variable<IList<int>> BernoulliIntegerSubset(ISparseList<double> probInSubset)
        {
            return Variable<IList<int>>.Random(Distributions.BernoulliIntegerSubset.FromProbTrue(probInSubset));
        }

#if false
        /// <summary>
        /// Creates a random variable whose domain is a fixed-length list of type integer.
        /// </summary>
        /// <param name="range">A Range defining the length of the list</param>
        /// <param name="probInSubset">The probability of a given integer being in the random list. This is given as random variable over a sparse list - i.e. most of the probabilities are the same.</param>
        /// <returns>A <see cref="Distributions.BernoulliIntegerSubset"/> distributed random variable</returns>
        public static VariableArray<Variable<int>, IList<int>> BernoulliIntegerSubset(Range range, Variable<ISparseList<double>> probInSubset)
        {
            var subset = IList<int>(range);
            subset.SetTo(Distributions.BernoulliIntegerSubset.Sample, probInSubset);
            return subset;
        }

        /// <summary>
        /// Creates a random variable whose domain is a fixed-length list of type integer.
        /// </summary>
        /// <param name="range">A Range defining the length of the list</param>
        /// <param name="probInSubset">The probability of a given integer being in the random list. This is given as a sparse list - i.e. most of the probabilities are the same.</param>
        /// <returns>A <see cref="Distributions.BernoulliIntegerSubset"/> distributed random variable</returns>
        public static VariableArray<Variable<int>, IList<int>> BernoulliIntegerSubset(Range range, ISparseList<double> probInSubset)
        {
            var subset = IList<int>(range);
            subset.SetTo(Variable<IList<int>>.Random(Distributions.BernoulliIntegerSubset.FromProbTrue(probInSubset)));
            return subset;
        }
#endif

        /// <summary>
        /// Creates a Boolean random variable with the probability of being true specified by the input's
        /// logistic function.
        /// </summary>
        /// <param name="logOdds">The input's logistic function, which specifies the probability that the
        /// variable is true as probTrue = 1/(1 + exp(-logOdds)).</param>
        /// <returns>Returns a Boolean random variable</returns>
        public static Variable<bool> BernoulliFromLogOdds(double logOdds)
        {
            return Variable<bool>.Random<bool>(Distributions.Bernoulli.FromLogOdds(logOdds));
        }

        /// <summary>
        /// Creates a Boolean random variable with the probability of being true specified by the input's
        /// logistic function, which is represented by a random variable.
        /// </summary>
        /// <param name="logOdds">A <c>double</c> random variable that represents the logistic function,
        /// which specifies the probability that the variable is true as probTrue = 1/(1 + exp(-logOdds))</param>
        /// <returns>Returns a Boolean random variable.</returns>
        public static Variable<bool> BernoulliFromLogOdds(Variable<double> logOdds)
        {
            Variable<double> probTrue = Logistic(logOdds).Attrib(new DoNotInfer());
            return Bernoulli(probTrue);
        }

        /// <summary>
        /// Creates a random variable that is statistically defined by a Discrete distribution with a specified
        /// set of probabilities.
        /// </summary>
        /// <param name="probs">An array that specifies the probability of each possible value, from [0, probs.Length-1].
        /// The array must have more than one element. The probabilities should sum to 1.0.
        /// If not, the probabilities will be normalized.</param>
        /// <returns>Returns a random variable that is statistically defined by the specified Discrete distribution.</returns>
        /// <exception cref="ArgumentException"><paramref name="probs"/> contains only one element.
        /// To specify a uniform Discrete distribution, use <c>Variable.DiscreteUniform.</c></exception>
        public static Variable<int> Discrete(params double[] probs)
        {
            if (probs.Length == 1 && probs[0] != 1)
                throw new ArgumentException("Only one probability was provided to Variable.Discrete.  Perhaps you meant Variable.DiscreteUniform(" + probs[0] + ")?");
            return Variable<int>.Random(Constant(new Discrete(probs)));
        }

        /// <summary>
        /// Creates a random variable that is statistically defined by a Discrete distribution with a specified
        /// number of possible values, and a corresponding set of probabilities.
        /// </summary>
        /// <param name="valueRange">A <c>Range</c> object that is initialized to the number of possible values.</param>
        /// <param name="probs">An array that specifies the probability of each possible value, from [0, probs.Length-1].
        /// The array must have more than one element. The probabilities should sum to 1.0.
        /// If not, the probabilities will be normalized.</param>
        /// <returns>Returns a random variable that is statistically defined by the specified Discrete distribution.</returns>
        /// <exception cref="ArgumentException">
        /// <paramref name="probs"/> contains only one element.
        /// To specify a uniform Discrete distribution, use <c>Variable.DiscreteUniform.</c>
        /// </exception>
        public static Variable<int> Discrete(Range valueRange, params double[] probs)
        {
            if (probs.Length == 0) return DiscreteUniform(valueRange);
            else
            {
                if (valueRange.SizeAsInt != probs.Length) throw new ArgumentException("probs.Length (" + probs.Length + ") != range.Size (" + valueRange.SizeAsInt + ")");
                return Variable<int>.Random(Constant(new Discrete(probs)))
                                    .Attrib(new ValueRange(valueRange));
            }
        }

        /// <summary>
        /// Creates a random variable that is statistically defined by a Discrete distribution with the set of possible
        /// values specified by a <c>Vector</c> object.
        /// </summary>
        /// <param name="v">A <c>Vector</c> object that specifies the probability of each possible value, from [0, probs.Length-1].
        /// The probabilities should sum to 1.0. If not, the probabilities will be normalized.</param>
        /// <returns>Returns a random variable that is statistically defined by the specified Discrete distribution.</returns>
        public static Variable<int> Discrete(Vector v)
        {
            var rv = Discrete(v.ToArray());
            if (!v.Sparsity.IsDense)
                rv.SetSparsity(v.Sparsity);
            return rv;
        }

        /// <summary>
        /// Creates a random variable that is statistically defined by a Discrete distribution with the number of possible
        /// values specified by a <c>Range</c> object and the probabilities by a <c>Vector</c> object.
        /// </summary>
        /// <param name="valueRange">A <c>Range</c> object that is initialized to the number of possible values.</param>
        /// <param name="v">A <c>Vector</c> object that specifies the probability of each possible value, from [0, probs.Length-1].
        /// The probabilities should sum to 1.0. If not, the probabilities will be normalized.</param>
        /// <returns>Returns a random variable that is statistically defined by the specified Discrete distribution.</returns>
        public static Variable<int> Discrete(Range valueRange, Vector v)
        {
            var rv = Discrete(v.ToArray()).Attrib(new ValueRange(valueRange));
            if (!v.Sparsity.IsDense)
                rv.SetSparsity(v.Sparsity);
            return rv;
        }

        /// <summary>
        /// Create a random integer by drawing uniformly from the range 0..(size-1)
        /// </summary>
        /// <param name="size">The number of possible values.</param>
        /// <returns>A random variable with an equal probability of taking any value in the range 0..(size-1)</returns>
        public static Variable<int> DiscreteUniform(int size)
        {
            //return Variable<int>.Random(Constant(Distributions.Discrete.Uniform(size)));
            return DiscreteUniform(Constant(size));
        }

        /// <summary>
        /// Create a random integer by drawing uniformly from a range.
        /// </summary>
        /// <param name="range">A <c>Range</c> object specifying the number of possible values.</param>
        /// <returns>A random integer with an equal probability of taking any value in the range.</returns>
        public static Variable<int> DiscreteUniform(Range range)
        {
            Variable<int> sample = Variable.New<int>();
            sample.SetTo(Factor.DiscreteUniform, range.Size);
            return sample.Attrib(new ValueRange(range));
        }

        /// <summary>
        /// Creates a random variable that is statistically defined by a Discrete distribution with the number
        /// of possible values specified by an <c>int</c> random variable.
        /// </summary>
        /// <param name="size">An <c>int</c> random variable that represents the number of possible values.</param>
        /// <returns>Returns a random variable that is statistically defined by a Discrete distribution with equal
        /// probabilities for each possible value.</returns>
        public static Variable<int> DiscreteUniform(Variable<int> size)
        {
            Variable<int> sample = Variable<int>.Factor(Factors.Factor.DiscreteUniform, size);
            Range valueRange = size.GetValueRange(false);
            if (valueRange != null) sample.AddAttribute(new ValueRange(valueRange));
            return sample;
        }

        /// <summary>
        /// Creates a random variable that is statistically defined by a uniform Discrete distribution with the number
        /// of possible values specified by <c>Range</c> object, and the upper bound specified by a random variable.
        /// </summary>
        /// <param name="valueRange">A <c>Range</c> object for <paramref name="size"/> that represents the value range.</param>
        /// <param name="size">A random variable that represents the dimension of the discrete distribution.</param>
        /// <returns>Returns a random variable that is statistically defined by a Discrete distribution with equal
        /// probabilities for each possible value.</returns>
        public static Variable<int> DiscreteUniform(Range valueRange, Variable<int> size)
        {
            Variable<int> sample = Variable<int>.Factor(Factors.Factor.DiscreteUniform, size);
            sample.AddAttribute(new ValueRange(valueRange));
            return sample;
        }

        /// <summary>
        /// Creates a random variable that is statistically defined by a Discrete distribution with the probabilities of
        /// the possible values specified by an <c>Vector</c> random variable.
        /// </summary>
        /// <param name="probs">A <c>Vector</c> random variable that represents the probabilities of the
        /// possible values.  The probabilities should sum to 1.0. If not, the probabilities will be normalized.</param>
        /// <returns>Returns a random variable that is statistically defined by the specified Discrete distribution.</returns>
        public static Variable<int> Discrete(Variable<Vector> probs)
        {
            Variable<int> v = Variable<int>.Factor(Factor.Discrete, probs);
            Range range = probs.GetValueRange(false);
            if (range != null) v.AddAttribute(new ValueRange(range));
            SparsityAttribute sparsityAttr = probs.GetFirstAttribute<SparsityAttribute>();
            if (sparsityAttr != null && sparsityAttr.Sparsity != null)
                v.SetSparsity(sparsityAttr.Sparsity);
            return v;
        }

        /// <summary>
        /// Creates a random variable that is statistically defined by a Discrete distribution with the number of
        /// possible values specified by a <c>Range</c> object and the probabilities of
        /// the possible values specified by an <c>Vector</c> random variable.
        /// </summary>
        /// <param name="valueRange">A range defining the possible values for the variable.</param>
        /// <param name="probs">A variable holding the set of probabilities of having each value.  Must add up to one.</param>
        /// <returns>Returns a random variable that is statistically defined by the specified Discrete distribution.</returns>
        public static Variable<int> Discrete(Range valueRange, Variable<Vector> probs)
        {
            Variable<int> v = Variable<int>.Factor(Factor.Discrete, probs);
            Range range2 = probs.GetValueRange(false);
            if (range2 != null && !valueRange.IsCompatibleWith(range2))
                throw new ArgumentException(probs + " has ValueRange '" + range2 + "', which is inconsistent with the provided range '" + valueRange + "'.  Try omitting '" +
                                            valueRange + "'");
            v.AddAttribute(new ValueRange(valueRange));
            SparsityAttribute sparsityAttr = probs.GetFirstAttribute<SparsityAttribute>();
            if (sparsityAttr != null && sparsityAttr.Sparsity != null)
                v.SetSparsity(sparsityAttr.Sparsity);
            return v;
        }

        /// <summary>
        /// Creates a random variable with a Discrete distribution using probabilities specified by a <c>Vector</c>
        /// object for each of the values of the specified enum type.
        /// </summary>
        /// <typeparam name="TEnum">The enum type.</typeparam>
        /// <param name="probs">A <c>Vector</c> random variable that represents of probabilities for each of the enum's
        /// values. The probabilities must sum to one. If not, they will be normalized.</param>
        /// <returns>Returns a random variable that is statistically defined by the specified Discrete distribution.</returns>
        public static Variable<TEnum> EnumDiscrete<TEnum>(Variable<Vector> probs)
        {
            var v = Variable<TEnum>.Factor(EnumSupport.DiscreteEnum<TEnum>, probs);
            Range range = probs.GetValueRange(false);
            if (range != null) v.AddAttribute(new ValueRange(range));
            return v;
        }

        /// <summary>
        /// Creates a random variable with a Discrete distribution with specified probabilities for each of the
        /// values of the specified enum type.
        /// </summary>
        /// <typeparam name="TEnum">The enum type.</typeparam>
        /// <param name="probs">A <c>double</c> array that contains the probabilities for each of the enum's
        /// values. The probabilities must sum to one. If not, they will be normalized.</param>
        /// <returns>Returns a random variable that is statistically defined by the specified Discrete distribution.</returns>
        public static Variable<TEnum> EnumDiscrete<TEnum>(params double[] probs)
        {
            int valueCount = Enum.GetValues(typeof(TEnum)).Length;
            return EnumDiscrete<TEnum>(new Range(valueCount), probs);
        }

        /// <summary>
        /// Creates a random variable with a Discrete distribution with the dimension specified by a <c>Range</c>
        /// object and specified probabilities for each of the values of the enum type.
        /// </summary>
        /// <typeparam name="TEnum">The enum type.</typeparam>
        /// <param name="valueRange">A <c>Range</c> object initialized to the number of enum elements.</param>
        /// <param name="probs">A <c>double</c> array that contains the probabilities for each of the enum's
        /// values. The probabilities must sum to one. If not, they will be normalized.</param>
        /// <returns>Returns a random variable that is statistically defined by the specified Discrete distribution.
        /// </returns>
        public static Variable<TEnum> EnumDiscrete<TEnum>(Range valueRange, params double[] probs)
        {
            if (probs.Length == 0) return EnumUniform<TEnum>(valueRange);
            int valueCount = Enum.GetValues(typeof(TEnum)).Length;
            return Variable<TEnum>.Random(Variable.Constant(new DiscreteEnum<TEnum>(probs)))
                                  .Attrib(new ValueRange(valueRange));
        }

        /// <summary>
        /// Creates a random variable with a Discrete distribution with the probabilities for each of
        /// the values of the enum type specified by a <c>Vector</c> object.
        /// </summary>
        /// <typeparam name="TEnum">The enum type.</typeparam>
        /// <param name="probs">A <c>Vector</c> object that contains the probabilities for each of the enum's
        /// values. The probabilities must sum to one. If not, they will be normalized.</param>
        /// <returns>Returns a random variable that is statistically defined by the specified Discrete distribution.
        /// </returns>
        public static Variable<TEnum> EnumDiscrete<TEnum>(Vector probs)
        {
            return EnumDiscrete<TEnum>(probs.ToArray());
        }

        /// <summary>
        /// Creates a random variable with a Discrete distribution with the number of possible values
        /// specified by a <c>Range</c> object and the probabilities for each of
        /// the values of the enum type specified by a <c>Vector</c> object.
        /// </summary>
        /// <typeparam name="TEnum">The type of the enum</typeparam>
        /// <param name="valueRange">A <c>Range</c> object initialized to the number of enum values.</param>
        /// <param name="probs">The vector of probabilities of having each value.  Must add up to one.</param>
        /// <returns>Returns a random variable that is statistically defined by the specified Discrete distribution.
        /// </returns>
        public static Variable<TEnum> EnumDiscrete<TEnum>(Range valueRange, Vector probs)
        {
            return EnumDiscrete<TEnum>(valueRange, probs.ToArray());
        }

        /// <summary>
        /// Creates a random variable with a Discrete distribution with a uniform distribution.
        /// </summary>
        /// <typeparam name="TEnum">The enum type.</typeparam>
        /// <returns>Returns a random variable that is statistically defined by the a Discrete distribution with equal
        /// probabilities for all possible values.
        /// </returns>
        public static Variable<TEnum> EnumUniform<TEnum>()
        {
            int valueCount = Enum.GetValues(typeof(TEnum)).Length;
            return EnumUniform<TEnum>(new Range(valueCount));
        }

        /// <summary>
        /// Creates a random variable with a Discrete distribution with the dimension specified by a <c>Range</c>
        /// object and a uniform distribution.
        /// </summary>
        /// <typeparam name="TEnum">The type of the enum</typeparam>
        /// <param name="valueRange">A range over the enum values.</param>
        /// <returns>Enum random variable</returns>
        public static Variable<TEnum> EnumUniform<TEnum>(Range valueRange)
        {
            var prior = Variable<DiscreteEnum<TEnum>>.Factor(DiscreteEnum<TEnum>.Uniform).Attrib(new DoNotInfer());
            return Variable<TEnum>.Random(prior)
                                  .Attrib(new ValueRange(valueRange));
        }

        /// <summary>
        /// Creates a random int variable corresponding to a random enum variable.  The returned
        /// variable can be used as the condition for a Switch or Case block.
        /// </summary>
        /// <typeparam name="TEnum">The enum type</typeparam>
        /// <param name="enumVar">The enum variable</param>
        /// <returns>An integer random variable</returns>
        public static Variable<int> EnumToInt<TEnum>(Variable<TEnum> enumVar)
        {
            Variable<int> intVar = Variable<int>.Factor(EnumSupport.EnumToInt<TEnum>, enumVar);
            intVar.SetValueRange(enumVar.GetValueRange());
            return intVar;
        }

        /// <summary>
        /// Creates a random int variable x where p(x=k) is proportional to exp(logProbs[k]), i.e. the softmax function of the logProbs.
        /// </summary>
        /// <param name="logProbs">Arguments to the softmax.</param>
        /// <returns>An integer random variable</returns>
        public static Variable<int> DiscreteFromLogProbs(VariableArray<double> logProbs)
        {
            // this is an equivalent model, using different factors
            Variable<Vector> probs = Variable.Softmax(logProbs).Attrib(new DoNotInfer());
            return Variable.Discrete(probs);
        }

        /// <summary>
        ///  Creates a Beta-distributed random variable with specified mean and variance
        /// </summary>
        /// <param name="mean">The distribution's mean.</param>
        /// <param name="variance">The distribution's variance.</param>
        /// <returns>Returns a Beta-distributed random variable whose mean and variance are specified by <paramref name="mean"/>
        /// and <paramref name="variance"/>.</returns>
        public static Variable<double> BetaFromMeanAndVariance(double mean, double variance)
        {
            return Variable<double>.Random<double>(Distributions.Beta.FromMeanAndVariance(mean, variance));
        }

        /// <summary>
        ///  Creates a Beta-distributed random variable with the mean and variance specified by random variables.
        /// </summary>
        /// <param name="mean"> A <c>double</c> random variable that represents the distribution's mean.</param>
        /// <param name="variance"> A <c>double</c> random variable that represents the distribution's variance.</param>
        /// <returns> Returns a Beta-distributed random variable whose mean and variance are specified by random variables, <paramref name="mean"/>
        /// and <paramref name="variance"/>.</returns>
        public static Variable<double> BetaFromMeanAndVariance(Variable<double> mean, Variable<double> variance)
        {
            return Variable<double>.Factor(Distributions.Beta.SampleFromMeanAndVariance, mean, variance);
        }

        /// <summary>
        /// Creates a Beta-distributed random variable from initial success/failure counts.
        /// </summary>
        /// <param name="trueCount">The initial success count.</param>
        /// <param name="falseCount">The initial failure count.</param>
        /// <returns>Returns a Beta-distributed random variable that is statistically defined by
        /// Beta(trueCount,falseCount).</returns>
        /// <remarks>
        /// The distribution's formula is
        /// <c>prob(x) = (Gamma(trueCount+falseCount)/Gamma(trueCount)/Gamma(falseCount)) x^{trueCount-1} (1-x)^(falseCount-1)</c>
        /// where x is between [0, 1].
        /// </remarks>
        public static Variable<double> Beta(double trueCount, double falseCount)
        {
            return Variable<double>.Random(Constant(new Beta(trueCount, falseCount)));
        }

        /// <summary>
        /// Creates a Beta-distributed random variable from initial success/failure counts that are
        /// represented by random variables.
        /// </summary>
        /// <param name="trueCount">A <c>double</c> random variable that represents the initial success count.</param>
        /// <param name="falseCount">A <c>double</c> random variable that represents the initial failure count.</param>
        /// <returns>Returns a Beta-distributed random variable that is statistically defined by
        /// Beta(trueCount,falseCount).</returns>
        /// <remarks>
        /// The distribution's formula is
        /// <c>prob(x) = (Gamma(trueCount+falseCount)/Gamma(trueCount)/Gamma(falseCount)) x^{trueCount-1} (1-x)^(falseCount-1)</c>
        /// where x is between [0, 1].
        /// </remarks>
        public static Variable<double> Beta(Variable<double> trueCount, Variable<double> falseCount)
        {
            return Variable<double>.Factor(Distributions.Beta.Sample, trueCount, falseCount);
        }

        /// <summary>
        /// Create a uniform Dirichlet-distributed random variable with a specified dimension.
        /// </summary>
        /// <param name="dimension">The the Dirichlet distribution's dimensionality.</param>
        /// <returns>Returns a uniform Dirichlet-distributed random variable. The distribution's pseudo-counts are
        /// all 1.</returns>
        public static Variable<Vector> DirichletUniform(int dimension)
        {
            return DirichletUniform(Constant(dimension));
        }

        /// <summary>
        /// Create a uniform Dirichlet-distributed random variable with a specified dimension.
        /// </summary>
        /// <param name="dimension">The the Dirichlet distribution's dimensionality.</param>
        /// <returns>Returns a uniform Dirichlet-distributed random variable. The distribution's pseudo-counts are
        /// all 1.</returns>
        public static Variable<Vector> DirichletUniform(Variable<int> dimension)
        {
            Variable<Dirichlet> prior = Variable<Dirichlet>.Factor(Distributions.Dirichlet.Uniform, dimension).Attrib(new DoNotInfer());
            return Variable<Vector>.Random(prior);
        }

        /// <summary>
        /// Creates a uniform Dirichlet-distributed random variable with dimension specified by a
        /// <c>Range</c> object.
        /// </summary>
        /// <param name="valueRange">A <c>Range</c> object that is initialized to the  Dirichlet
        /// distribution's dimensionality.</param>
        /// <returns>Returns a uniform Dirichlet-distributed random variable. The distribution's pseudo-counts are
        /// all set to 1.</returns>
        public static Variable<Vector> DirichletUniform(Range valueRange)
        {
            Variable<Dirichlet> prior = Variable.New<Dirichlet>().Attrib(new DoNotInfer());
            prior.SetTo(Distributions.Dirichlet.Uniform, valueRange.Size);
            return Variable<Vector>.Random(prior).Attrib(new ValueRange(valueRange));
        }

        /// <summary>
        /// Creates a symmetric Dirichlet-distributed random variable with a specified dimension
        /// and a common pseudo-count.
        /// </summary>
        /// <param name="dimension">The the Dirichlet distribution's dimensionality.</param>
        /// <param name="pseudocount">A pseudo-count, that is applied to all dimensions.</param>
        /// <returns>Returns a symmetric Dirichlet-distributed random variable with the pseudo-counts
        /// set to <paramref name="pseudocount"/>.</returns>
        public static Variable<Vector> DirichletSymmetric(int dimension, double pseudocount)
        {
            return DirichletSymmetric(dimension, Constant(pseudocount));
            //var prior = Variable<Dirichlet>.Factor(Distributions.Dirichlet.Symmetric, dimension, pseudocount).Attrib(new DoNotInfer());
            //return Variable<Vector>.Random(prior);
        }

        /// <summary>
        /// Creates a symmetric Dirichlet-distributed random variable with a specified dimension
        /// and a common pseudo-count, which is represented by a random variable.
        /// </summary>
        /// <param name="dimension">The the Dirichlet distribution's dimensionality.</param>
        /// <param name="pseudocount">A pseudo-count, represented by a random variable, which
        /// is applied to all dimensions.</param>
        /// <returns>Returns a symmetric Dirichlet-distributed random variable with the pseudo-counts
        /// set to <paramref name="pseudocount"/>.</returns>
        public static Variable<Vector> DirichletSymmetric(int dimension, Variable<double> pseudocount)
        {
            return Variable<Vector>.Factor(Factor.DirichletSymmetric, dimension, pseudocount);
        }

        /// <summary>
        /// Creates a symmetric Dirichlet-distributed random variable with the dimension specified by
        /// a <c>Range</c> object and a common pseudo-count, which is represented by a random variable.
        /// </summary>
        /// <param name="valueRange">A <c>Range</c> object that is initialized to the  Dirichlet
        /// distribution's dimensionality.</param>
        /// <param name="pseudocount">A pseudo-count, represented by a random variable, which
        /// is applied to all dimensions.</param>
        /// <returns>Returns a symmetric Dirichlet-distributed random variable with the pseudo-counts
        /// set to <paramref name="pseudocount"/>.</returns>
        public static Variable<Vector> DirichletSymmetric(Range valueRange, Variable<double> pseudocount)
        {
            Variable<Vector> v = Variable.New<Vector>();
            v.SetTo(Factor.DirichletSymmetric, valueRange.Size, pseudocount);
            v.SetValueRange(valueRange);
            return v;
        }


        /// <summary>
        /// Creates a Dirichlet-distributed random variable with a specified set of pseudo-counts.
        /// </summary>
        /// <param name="u">An array containing the pseudo-counts.</param>
        /// <returns>Returns a Dirichlet-distributed random variable with the pseudo-counts specified
        /// by <paramref name="u"/>.</returns>
        public static Variable<Vector> Dirichlet(double[] u)
        {
            return Variable<Vector>.Random(Constant(new Dirichlet(u)));
        }

        /// <summary>
        /// Creates a Dirichlet-distributed random variable with the dimensionality specified by
        /// a <c>Range</c> object and a specified set of pseudo-counts.
        /// </summary>
        /// <param name="valueRange">A <c>Range</c> object that is initialized to the  Dirichlet
        /// distribution's dimensionality.</param>
        /// <param name="u">An array containing the pseudo-counts.</param>
        /// <returns>Returns a Dirichlet-distributed random variable of dimension <paramref name="u"/></returns>
        public static Variable<Vector> Dirichlet(Range valueRange, double[] u)
        {
            return Dirichlet(u).Attrib(new ValueRange(valueRange));
        }

        /// <summary>
        /// Creates a Dirichlet-distributed random variable with a set of pseudo-counts specified
        /// by a <c>Vector</c> object.
        /// </summary>
        /// <param name="v">A <c>Vector</c> object containing the pseudo-counts.</param>
        /// <returns>Returns a Dirichlet-distributed random variable with the pseudo-counts specified
        /// by <paramref name="v"/>.</returns>
        public static Variable<Vector> Dirichlet(Vector v)
        {
            var rv = Variable<Vector>.Random(Constant(new Dirichlet(v)));
            if (!v.Sparsity.IsDense)
                rv.SetSparsity(v.Sparsity);
            return rv;
        }

        /// <summary>
        /// Creates a Dirichlet-distributed random variable with pseudo-counts specified by a random variable.
        /// </summary>
        /// <param name="pseudoCount">A <c>Vector</c> random variable that represents the pseudo-counts.</param>
        /// <returns>Returns a Dirichlet-distributed random variable with the pseudo-counts specified
        /// by <paramref name="pseudoCount"/>.</returns>
        public static Variable<Vector> Dirichlet(Variable<Vector> pseudoCount)
        {
            Variable<Vector> v = Variable<Vector>.Factor(Distributions.Dirichlet.SampleFromPseudoCounts, pseudoCount);

            SparsityAttribute sparsityAttr = pseudoCount.GetFirstAttribute<SparsityAttribute>();
            if (sparsityAttr != null && sparsityAttr.Sparsity != null)
                v.SetSparsity(sparsityAttr.Sparsity);
            return v;
        }

        /// <summary>
        /// Creates a Dirichlet-distributed random variable with the dimensionality specified by
        /// a <c>Range</c> object and the pseudo-counts specified by a <c>Vector</c> object.
        /// </summary>
        /// <param name="valueRange">A <c>Range</c> object that is initialized to the Dirichlet
        /// distribution's dimensionality.</param>
        /// <param name="v">A <c>Vector</c> object containing the pseudo-counts.</param>
        /// <returns>Returns a Dirichlet-distributed random variable with the dimensionality specified by
        /// <paramref name="valueRange"/> and the pseudo-counts specified
        /// by <paramref name="v"/>.</returns>
        public static Variable<Vector> Dirichlet(Range valueRange, Vector v)
        {
            return Dirichlet(v).Attrib(new ValueRange(valueRange));
        }

        /// <summary>
        /// Creates a Dirichlet-distributed random variable with the dimensionality specified by
        /// a <c>Range</c> object and the pseudo-counts represented by a random variable.
        /// </summary>
        /// <param name="valueRange">A <c>Range</c> object that is initialized to the  Dirichlet
        /// distribution's dimensionality.</param>
        /// <param name="v">A <c>Vector</c> random variable that represents the pseudo-counts.</param>
        /// <returns>Returns a Dirichlet-distributed random variable with the dimensionality specified by
        /// <paramref name="valueRange"/> and the pseudo-counts specified
        /// by <paramref name="v"/>.</returns>
        public static Variable<Vector> Dirichlet(Range valueRange, Variable<Vector> v)
        {
            return Dirichlet(v).Attrib(new ValueRange(valueRange));
        }

        /// <summary>
        /// Create a double-precision random variable that is constrained to equal the given integer variable.
        /// </summary>
        /// <param name="integer">An integer variable</param>
        /// <returns>A new variable</returns>
        public static Variable<double> Double(Variable<int> integer)
        {
            return Variable<double>.Factor(Factor.Double, integer);
        }

        /// <summary>
        /// Create a double-precision random variable that is constrained to equal the given integer expression.
        /// </summary>
        /// <param name="integer">An integer expression</param>
        /// <returns>A new variable</returns>
        public static Variable<double> Double(IModelExpression<int> integer)
        {
            var v = Variable.New<double>();
            v.SetTo(Factor.Double, integer);
            return v;
        }

        /// <summary>
        /// Creates a double-precision random variable that is constrained to equal 1.0 when the given boolean is true, otherwise 0.0.
        /// </summary>
        /// <param name="boolean">A boolean variable</param>
        /// <returns>A new variable</returns>
        public static Variable<double> Double(Variable<bool> boolean)
        {
            Variable<double> result = Variable.New<double>();
            CreateVariableArray(result, boolean);
            var blocks = OpenRangesInExpression(boolean);
            using (Variable.If(boolean))
            {
                result.SetTo(1.0);
            }
            using (Variable.IfNot(boolean))
            {
                result.SetTo(0.0);
            }
            ReverseAndCloseBlocks(blocks);
            return result;
        }

        /// <summary>
        /// Creates a Poisson-distributed random variable with a specified mean.
        /// </summary>
        /// <param name="mean">The mean.</param>
        /// <returns></returns>
        public static Variable<int> Poisson(double mean)
        {
            return Variable<int>.Random<int>(new Poisson(mean));
        }

        /// <summary>
        /// Creates a Poisson-distributed random variable with the mean represented by a random variable.
        /// </summary>
        /// <param name="mean">A random variable that represents the mean.</param>
        /// <returns></returns>
        public static Variable<int> Poisson(Variable<double> mean)
        {
            return Variable<int>.Factor(Factor.Poisson, mean);
        }

        /// <summary>
        /// Creates a Binomially-distributed random variable with the specified probability of success per trial and number of trials.
        /// </summary>
        /// <param name="trialCount">The number of trials</param>
        /// <param name="probSuccess">A variable containing the probability of success per trial</param>
        /// <returns></returns>
        public static Variable<int> Binomial(int trialCount, Variable<double> probSuccess)
        {
            var sample = Variable<int>.Factor(Rand.Binomial, Constant(trialCount), probSuccess);
            // sample ranges from 0 to trialCount
            Range valueRange = new Range(trialCount + 1);
            sample.SetValueRange(valueRange);
            return sample;
        }

        /// <summary>
        /// Creates a Binomially-distributed random variable with the specified probability of success per trial and number of trials.
        /// </summary>
        /// <param name="trialCount">A variable containing the number of trials</param>
        /// <param name="probSuccess">A variable containing the probability of success per trial</param>
        /// <returns></returns>
        public static Variable<int> Binomial(Variable<int> trialCount, Variable<double> probSuccess)
        {
            Variable<int> sample = Variable<int>.Factor(Rand.Binomial, trialCount, probSuccess);
            // if trialCount ranges from 0 to valueRange.Size-1
            // then sample also ranges from 0 to valueRange.Size-1
            Range valueRange = trialCount.GetValueRange(false);
            if (valueRange != null)
                sample.SetValueRange(valueRange);
            return sample;
        }

        /// <summary>
        /// Creates a list x of random integers where x[i] is the number of times that value i is drawn in the given number of trials.
        /// </summary>
        /// <param name="trialCount">The number of trials</param>
        /// <param name="probs">A variable containing the probability distribution for drawing each value per trial.</param>
        /// <returns>A variable containing a list of counts</returns>
        public static VariableArray<Variable<int>, IList<int>> MultinomialList(Variable<int> trialCount, Variable<Vector> probs)
        {
            var valueRange = probs.GetValueRange();
            var list = IList<int>(valueRange);
            list.SetTo(Factor.MultinomialList, trialCount, probs);
            return list;
        }

        /// <summary>
        /// Creates an array x of random integers where x[i] is the number of times that value i is drawn in the given number of trials.
        /// </summary>
        /// <param name="trialCount">The number of trials</param>
        /// <param name="probs">A variable containing the probability distribution for drawing each value per trial.  Must have a ValueRange attribute specifying the number of possible values.</param>
        /// <returns></returns>
        public static VariableArray<int> Multinomial(Variable<int> trialCount, Variable<Vector> probs)
        {
            Range valueRange = probs.GetValueRange();
            VariableArray<int> samples = Variable.Array<int>(valueRange);
            // SetTo(Variable.Factor(...)) will not work here
            samples.SetTo(Rand.Multinomial, trialCount, probs);
            return samples;
        }

        /// <summary>
        /// Creates an array x of random integers where x[i] is the number of times that value i is drawn in the given number of trials.
        /// </summary>
        /// <param name="trialCount">The number of trials</param>
        /// <param name="probs">A variable containing the probability distribution for drawing each value per trial.  Must have a ValueRange attribute specifying the number of possible values.</param>
        /// <returns></returns>
        public static VariableArray<int> Multinomial(Variable<int> trialCount, Vector probs)
        {
            Variable<Vector> probsVar = Constant(probs);
            Range valueRange = new Range(probs.Count);
            probsVar.SetValueRange(valueRange);
            return Multinomial(trialCount, probsVar);
        }

        /// <summary>
        /// Creates an array x of random integers where x[i] is the number of times that value i is drawn in the given number of trials.
        /// </summary>
        /// <param name="trials">A range whose length is the number of trials.  This is becomes the ValueRange for each x[i].</param>
        /// <param name="probs">A variable containing the probability distribution for drawing each value per trial.  Must have a ValueRange attribute specifying the number of possible values.</param>
        /// <returns></returns>
        public static VariableArray<int> Multinomial(Range trials, Variable<Vector> probs)
        {
            Range valueRange = probs.GetValueRange();
            VariableArray<int> samples = Variable.Array<int>(valueRange);
            samples.SetValueRange(trials);
            samples.SetTo(Rand.Multinomial, trials.Size, probs);
            return samples;
        }

        /// <summary>
        /// Evaluate a random function at a point
        /// </summary>
        /// <param name="func">A random function</param>
        /// <param name="x">The location to evaluate the function</param>
        /// <returns>A new variable equal to <c>func(x)</c></returns>
        public static Variable<double> FunctionEvaluate(Variable<IFunction> func, Variable<Vector> x)
        {
            return Variable<double>.Factor(Factor.FunctionEvaluate, func, x);
        }

        /// <summary>
        /// Returns a boolean random variable indicating if the supplied double random variable is positive.
        /// </summary>
        /// <param name="x">The random variable to test for positivity</param>
        /// <returns>True if (x > 0)</returns>
        public static Variable<bool> IsPositive(Variable<double> x)
        {
            return Variable<bool>.Factor(Factor.IsPositive, x);
        }

        /// <summary>
        /// Returns a boolean random variable indicating if the supplied double random variable is between
        /// the specified limits.
        /// </summary>
        /// <param name="x">The double variable to test</param>
        /// <param name="lowerBound">The lower limit</param>
        /// <param name="upperBound">The upper limit</param>
        /// <returns>True if (lowerBound &lt;= x) and (x &lt; upperBound)</returns>
        public static Variable<bool> IsBetween(Variable<double> x, Variable<double> lowerBound, Variable<double> upperBound)
        {
            return Variable<bool>.Factor(Factor.IsBetween, x, lowerBound, upperBound);
        }


        /// <summary>
        /// Returns a double random variable which is the inner product of two vector variables.
        /// </summary>
        /// <param name="a">The first vector variable</param>
        /// <param name="b">The second vector variable</param>
        /// <returns><c>sum_i a[i]*b[i]</c></returns>
        public static Variable<double> InnerProduct(Variable<Vector> a, Variable<Vector> b)
        {
            return Variable<double>.Factor(Probabilistic.Math.Vector.InnerProduct, a, b);
        }

        /// <summary>
        /// Returns a double random variable which is the inner product of two vector variables.
        /// </summary>
        /// <param name="a">The first vector variable</param>
        /// <param name="b">The second vector variable</param>
        /// <returns><c>sum_i a[i]*b[i]</c></returns>
        public static Variable<double> InnerProduct(Variable<Vector> a, Variable<DenseVector> b)
        {
            return Variable<double>.Factor(Probabilistic.Math.Vector.InnerProduct, a, b);
        }

        /// <summary>
        /// Returns a double random variable which is the inner product of two vector variables.
        /// </summary>
        /// <param name="a">The first vector variable</param>
        /// <param name="b">The second vector variable</param>
        /// <returns><c>sum_i a[i]*b[i]</c></returns>
        public static Variable<double> InnerProduct(Variable<DenseVector> a, Variable<Vector> b)
        {
            return Variable<double>.Factor(Probabilistic.Math.Vector.InnerProduct, a, b);
        }

        /// <summary>
        /// Returns a double random variable which is the inner product of an array and a vector variable.
        /// </summary>
        /// <param name="a">The array variable</param>
        /// <param name="b">The vector variable</param>
        /// <returns><c>sum_i a[i]*b[i]</c></returns>
        public static Variable<double> InnerProduct(VariableArray<double> a, Variable<Vector> b)
        {
            return Variable<double>.Factor(Factor.InnerProduct, a, b);
        }

        /// <summary>
        /// Returns a double random variable which is the inner product of two array variables.
        /// </summary>
        /// <param name="a">The first array variable</param>
        /// <param name="b">The second array variable</param>
        /// <returns><c>sum_i a[i]*b[i]</c></returns>
        public static Variable<double> InnerProduct(VariableArray<double> a, VariableArray<double> b)
        {
            return Variable<double>.Factor(Factor.InnerProduct, a, b);
        }

        /// <summary>
        /// Returns a double random variable which is the inner product of a array of binary variables and a vector variable.
        /// </summary>
        /// <param name="a">The first, binary vector variable</param>
        /// <param name="b">The second vector variable</param>
        /// <returns><c>sum_i a[i]*b[i]</c></returns>
        public static Variable<double> SumWhere(VariableArray<bool> a, Variable<Vector> b)
        {
            Range range = a.Range;
            var aDouble = Variable.Double(a[range]);
            return Variable.InnerProduct((VariableArray<double>)aDouble.ArrayVariable, b);
        }

        /// <summary>
        /// Returns a double random variable which is the inner product of a array of binary variables and an array of double variables.
        /// </summary>
        /// <param name="a">A binary array variable</param>
        /// <param name="b">A double array variable</param>
        /// <returns><c>sum_i a[i]*b[i]</c></returns>
        public static Variable<double> SumWhere(VariableArray<bool> a, VariableArray<double> b)
        {
            var blocks = OpenRangesInExpression(a);
            Range range = a.Range;
            VariableArray<double> products = Variable.Array<double>(range);
            using (Variable.ForEach(range))
            {
                using (Variable.If(a[range]))
                {
                    products[range] = b[range];
                }
                using (Variable.IfNot(a[range]))
                {
                    products[range] = 0.0;
                }
            }
            var result = Variable.Sum(products);
            ReverseAndCloseBlocks(blocks);
            return result;
        }

        /// <summary>
        /// Create a random Matrix from values in an array.
        /// </summary>
        /// <param name="array">A random array</param>
        /// <returns>A new matrix variable whose value is equal to <c>Matrix.FromArray(array)</c></returns>
        public static Variable<Matrix> Matrix(Variable<double[,]> array)
        {
            return Variable<Matrix>.Factor(Probabilistic.Math.Matrix.FromArray, array);
        }

        /// <summary>
        /// Returns a Vector variable which is the product of a Matrix variable with a Vector variable
        /// </summary>
        /// <param name="matrix">The matrix</param>
        /// <param name="vector">The vector</param>
        /// <returns></returns>
        public static Variable<Vector> MatrixTimesVector(Variable<Matrix> matrix, Variable<Vector> vector)
        {
            return Variable<Vector>.Factor(Factor.Product, matrix, vector);
        }

        /// <summary>
        /// Returns a Vector variable which is the product of a two-dimensional array variable (treated as a Matrix) with a Vector variable
        /// </summary>
        /// <param name="array">The matrix</param>
        /// <param name="vector">The vector</param>
        /// <returns></returns>
        public static Variable<Vector> MatrixTimesVector(Variable<double[,]> array, Variable<Vector> vector)
        {
            return Variable<Vector>.Factor(Factor.Product, array, vector);
        }

        /// <summary>
        /// Create a matrix variable whose [i,j] entry equals a[i,j]*b
        /// </summary>
        /// <param name="a">The matrix</param>
        /// <param name="b">The scalar</param>
        /// <returns></returns>
        public static Variable<PositiveDefiniteMatrix> MatrixTimesScalar(Variable<PositiveDefiniteMatrix> a, Variable<double> b)
        {
            return Variable<PositiveDefiniteMatrix>.Factor(Factor.Product, a, b);
        }

        /// <summary>
        /// Creates a vector variable whose [i] entry equals a[i]*b
        /// </summary>
        /// <param name="a">The vector</param>
        /// <param name="b">The scalar</param>
        /// <returns></returns>
        public static Variable<Vector> VectorTimesScalar(Variable<Vector> a, Variable<double> b)
        {
            return Variable<Vector>.Factor(Factor.Product, a, b);
        }

        /// <summary>
        /// Returns a 2-D array of variables which is the matrix product of two other 2-D arrays of variables
        /// </summary>
        /// <param name="a">The first 2-D array</param>
        /// <param name="b">The second 2-D array</param>
        /// <returns></returns>
        public static VariableArray2D<double> MatrixMultiply(VariableArray2D<double> a, VariableArray2D<double> b)
        {
            VariableArray2D<double> result = Variable.Array<double>(a.Range0, b.Range1);
            result.SetTo(Factor.MatrixMultiply, a, b);
            return result;
        }

        /// <summary>
        /// Returns a double variable which is the sum of the elements of an array variable.
        /// For sum of two variables, use the + operator.
        /// </summary>
        /// <param name="array">The array variable</param>
        /// <returns><c>sum_i array[i]</c></returns>
        public static Variable<int> Sum(Variable<int[]> array)
        {
            return Variable<int>.Factor(Factor.Sum, array);
        }

        /// <summary>
        /// Returns a double variable which is the sum of the elements of an array variable.
        /// For sum of two variables, use the + operator.
        /// </summary>
        /// <param name="array">The array variable</param>
        /// <returns><c>sum_i array[i]</c></returns>
        public static Variable<double> Sum(Variable<IList<double>> array)
        {
            return Variable<double>.Factor(Factor.Sum, array);
        }

        /// <summary>
        /// Returns a double variable which is the sum of the elements of an array variable.
        /// For sum of two variables, use the + operator.
        /// </summary>
        /// <param name="array">The array variable</param>
        /// <returns><c>sum_i array[i]</c></returns>
        public static Variable<double> Sum(Variable<double[]> array)
        {
            return Variable<double>.Factor(Factor.Sum, array);
        }

        /// <summary>
        /// Returns a double variable which is the sum of the elements of an array variable.
        /// For sum of two variables, use the + operator.
        /// </summary>
        /// <param name="array">The array variable</param>
        /// <returns><c>sum_i array[i]</c></returns>
        public static Variable<double> Sum_Expanded(VariableArray<double> array)
        {
            var blocks = OpenRangesInExpression(array);
            Range n = array.Range;
            var sumUpTo = Variable.Array<double>(n);
            sumUpTo.Name = array.ToString() + "_sumUpTo";
            sumUpTo.AddAttribute(new DoNotInfer());
            using (var fb = Variable.ForEach(n))
            {
                var i = fb.Index;
                using (Variable.Case(i, 0))
                {
                    sumUpTo[i] = Variable.Copy(array[i]);
                }
                using (Variable.If(i > 0))
                {
                    sumUpTo[i] = sumUpTo[i - 1] + array[i];
                }
            }
            var size = (Variable<int>)n.Size;
            var sizeIsZero = (size == 0);
            var sum = Variable.New<double>();
            using (Variable.If(sizeIsZero))
            {
                sum.SetTo(Variable.Constant(0.0));
            }
            using (Variable.IfNot(sizeIsZero))
            {
                sum.SetTo(Variable.Copy(sumUpTo[size - 1]));
            }
            ReverseAndCloseBlocks(blocks);
            return sum;
        }

        /// <summary>
        /// Create an array that includes all ranges in the prototype (that are not open in ForEach blocks)
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="item">Modified to be an array</param>
        /// <param name="modelExpressionWithRanges">An expression that refers to ranges</param>
        /// <returns></returns>
        private static IVariable CreateVariableArray<T>(Variable<T> item, IModelExpression modelExpressionWithRanges)
        {
            var ranges = GetRangesNotOpen(modelExpressionWithRanges);
            if (ranges.Count > 0)
                return Variable<T>.CreateVariableArrayFromItem(item, ranges);
            else
                return item;
        }

        /// <summary>
        /// Creates a list of all ranges in the expression that are not open in ForEach blocks.
        /// </summary>
        /// <param name="modelExpressionWithRanges"></param>
        /// <returns></returns>
        private static List<Range> GetRangesNotOpen(IModelExpression modelExpressionWithRanges)
        {
            List<Range> ranges = new List<Range>();
            MethodInvoke.ForEachRange(modelExpressionWithRanges, delegate (Range r)
            {
                if (!ranges.Contains(r))
                    ranges.Add(r);
            });
            foreach (IStatementBlock b in StatementBlock.GetOpenBlocks())
            {
                if (b is HasRange br)
                {
                    ranges.Remove(br.Range);
                }
            }
            return ranges;
        }

        private static List<ForEachBlock> OpenRangesInExpression(IModelExpression modelExpressionWithRanges)
        {
            var ranges = GetRangesNotOpen(modelExpressionWithRanges);
            List<ForEachBlock> blocks = new List<ForEachBlock>();
            foreach (var range in ranges)
            {
                blocks.Add(Variable.ForEach(range));
            }
            return blocks;
        }

        private static void ReverseAndCloseBlocks(List<ForEachBlock> blocks)
        {
            blocks.Reverse();
            foreach (var block in blocks)
            {
                block.CloseBlock();
            }
        }

        /// <summary>
        /// Returns a <see cref="Vector"/> variable which is the sum of the elements of an array variable.
        /// </summary>
        /// <param name="array">The array variable.</param>
        /// <returns><c>sum_i array[i]</c>.</returns>
        public static Variable<Vector> Sum(Variable<IList<Vector>> array)
        {
            return Variable<Vector>.Factor(Factor.Sum, array);
        }

        /// <summary>
        /// Returns a <see cref="Vector"/> variable which is the sum of the elements of an array variable.
        /// </summary>
        /// <param name="array">The array variable.</param>
        /// <returns><c>sum_i array[i]</c>.</returns>
        public static Variable<Vector> Sum(Variable<Vector[]> array)
        {
            return Variable<Vector>.Factor(Factor.Sum, array);
        }

        /// <summary>
        /// Returns a double variable which is the max of two double variables
        /// </summary>
        /// <param name="a">The first variable</param>
        /// <param name="b">The second variable</param>
        /// <returns>A new variable</returns>
        public static Variable<double> Max(Variable<double> a, Variable<double> b)
        {
            return Variable<double>.Factor<double, double>(System.Math.Max, a, b);
        }

        /// <summary>
        /// Returns an int variable which is the maximum of two int variables
        /// </summary>
        /// <param name="a">The first variable</param>
        /// <param name="b">The second variable</param>
        /// <returns>A new variable</returns>
        public static Variable<int> Max(Variable<int> a, Variable<int> b)
        {
            return Variable<int>.Factor(System.Math.Max, a, b);
        }

        /// <summary>
        /// Returns a double variable which is the minimum of two double variables
        /// </summary>
        /// <param name="a">The first variable</param>
        /// <param name="b">The second variable</param>
        /// <returns>A new variable</returns>
        public static Variable<double> Min(Variable<double> a, Variable<double> b)
        {
            return -Max(-a, -b);
        }

        /// <summary>
        /// Returns an int variable which is the minimum of two int variables
        /// </summary>
        /// <param name="a">The first variable</param>
        /// <param name="b">The second variable</param>
        /// <returns>A new variable</returns>
        public static Variable<int> Min(Variable<int> a, Variable<int> b)
        {
            return Variable<int>.Factor(System.Math.Min, a, b);
        }

        /// <summary>
        /// Returns a variable which takes e to the power of another random variable
        /// </summary>
        /// <param name="exponent">The specified exponent</param>
        /// <returns>A new variable</returns>
        public static Variable<double> Exp(Variable<double> exponent)
        {
            return Variable<double>.Factor(System.Math.Exp, exponent);
        }

        /// <summary>
        /// Returns a variable equal to the natural logarithm of d
        /// </summary>
        /// <param name="d">The variable argument</param>
        /// <returns>A new variable</returns>
        public static Variable<double> Log(Variable<double> d)
        {
            return Variable<double>.Factor(Math.Log, d);
        }

        /// <summary>
        /// Creates a variable equal to 1/(1+exp(-x))
        /// </summary>
        /// <param name="x">The variable argument</param>
        /// <returns>A new variable</returns>
        public static Variable<double> Logistic(Variable<double> x)
        {
            return Variable<double>.Factor(MMath.Logistic, x);
        }

        /// <summary>
        /// Creates a vector variable y where y[i] = exp(x[i])/(sum_j exp(x[j])).  y has the same length as x.
        /// </summary>
        /// <param name="x">The variable array argument</param>
        /// <returns>A new variable</returns>
        public static Variable<Vector> Softmax(VariableArray<double> x)
        {
            Variable<Vector> result = Variable<Vector>.Factor(MMath.Softmax, x);
            result.SetValueRange(x.Range);
            return result;
        }

        /// <summary>
        /// Creates a vector variable y where y[i] = exp(x[i])/(sum_j exp(x[j])).  y has the same length as x.
        /// </summary>
        /// <param name="x">The vector variable argument</param>
        /// <returns>A new variable</returns>
        public static Variable<Vector> Softmax(Variable<IList<double>> x)
        {
            return Variable<Vector>.Factor(MMath.Softmax, x);
        }

        /// <summary>
        /// Creates a vector variable y where y[i] = exp(x[i])/(sum_j exp(x[j])).  y has the same length as x.
        /// </summary>
        /// <param name="x">The vector variable argument</param>
        /// <returns>A new variable</returns>
        public static Variable<Vector> Softmax(Variable<Vector> x)
        {
            return Variable<Vector>.Factor(MMath.Softmax, x);
        }

        /// <summary>
        /// Creates a vector variable y where y[i] = exp(x[i])/(sum_j exp(x[j])).  y has the same length as x.
        /// </summary>
        /// <param name="x">The sparse list variable argument</param>
        /// <returns>A new variable</returns>
        public static Variable<Vector> Softmax(Variable<ISparseList<double>> x)
        {
            return Variable<Vector>.Factor(MMath.Softmax, x);
        }

        /// <summary>
        /// Returns a Gaussian variable which is the product of A times the exponential of B. 
        /// </summary>
        public static Variable<double> ProductExp(Variable<double> A, Variable<double> B)
        {
            return Variable<double>.Factor(Microsoft.ML.Probabilistic.Factors.Factor.ProductExp, A, B);
        }

        /// <summary>
        /// A random vector equal to the vector (x,y) rotated by an angle about the origin.
        /// </summary>
        /// <param name="x">First coordinate of the vector to rotate</param>
        /// <param name="y">Second coordinate of the vector to rotate</param>
        /// <param name="angle">Counter-clockwise rotation angle in radians</param>
        /// <returns>A new variable</returns>
        public static Variable<Vector> Rotate(Variable<double> x, Variable<double> y, Variable<double> angle)
        {
            Variable<Vector> result = Variable<Vector>.Factor(Factor.Rotate, x, y, angle);
            result.AddAttribute(new MarginalPrototype(new VectorGaussian(2)));
            return result;
        }

        /// <summary>
        /// Returns a boolean variable which is true if all array elements are true.
        /// For AND of two variables, use the &amp; operator.
        /// </summary>
        /// <param name="array">The array variable</param>
        /// <returns><c>AND_i array[i]</c></returns>
        public static Variable<bool> AllTrue(Variable<IList<bool>> array)
        {
            return Variable<bool>.Factor(Factor.AllTrue, array);
        }

        /// <summary>
        /// Returns a boolean variable which is true if all array elements are true.
        /// For AND of two variables, use the &amp; operator.
        /// </summary>
        /// <param name="array">The array variable</param>
        /// <returns><c>AND_i array[i]</c></returns>
        public static Variable<bool> AllTrue(Variable<bool[]> array)
        {
            return Variable<bool>.Factor(Factor.AllTrue, array);
        }

        /// <summary>
        /// Returns an integer variable equal to the number of array elements that are true.
        /// </summary>
        /// <param name="array">The array variable</param>
        /// <returns><c>sum_i array[i]</c></returns>
        public static Variable<int> CountTrue(Variable<bool[]> array)
        {
            return Variable<int>.Factor(Factor.CountTrue, array);
        }

        /// <summary>
        /// Creates a random Vector by concatenating two random Vectors.
        /// </summary>
        /// <param name="first">First vector</param>
        /// <param name="second">Second vector</param>
        /// <returns>A new vector variable whose value is equal to <c>Vector.Concat(first,second)</c></returns>
        public static Variable<Vector> Concat(Variable<Vector> first, Variable<Vector> second)
        {
            return Variable<Vector>.Factor(Probabilistic.Math.Vector.Concat, first, second);
        }

        /// <summary>
        /// Create a random Vector from values in an array.
        /// </summary>
        /// <param name="array">A random array</param>
        /// <returns>A new vector variable whose value is equal to <c>Vector.FromArray(array)</c></returns>
        public static Variable<Vector> Vector(Variable<double[]> array)
        {
            var result = Variable<Vector>.Factor(Probabilistic.Math.Vector.FromArray, array);
            if (array is IVariableArray iva)
                result.SetValueRange(iva.Ranges[0]);
            return result;
        }

        /// <summary>
        /// Create a random array from values in a Vector.
        /// </summary>
        /// <param name="vector">A random vector</param>
        /// <param name="range">The range to use for indexing the array.  Must match the length of the vector.</param>
        /// <returns>A new array whose elements are equal to the elements of the vector</returns>
        public static VariableArray<double> ArrayFromVector(Variable<Vector> vector, Range range)
        {
            VariableArray<double> result = Variable.Array<double>(range);
            // SetTo(Variable.Factor(...)) will not work here
            result.SetTo(Factor.ArrayFromVector, vector);
            return result;
        }

        /// <summary>
        /// Creates two arrays by splitting a random array into disjoint pieces.
        /// </summary>
        /// <typeparam name="T">The element type</typeparam>
        /// <param name="array">The source array</param>
        /// <param name="headRange">Specifies the number of elements to put into the returned array</param>
        /// <param name="tailRange">The range associated with the tail array</param>
        /// <param name="tail">On return, holds the remaining elements</param>
        /// <returns>The initial elements of <paramref name="array"/></returns>
        public static VariableArray<T> Split<T>(VariableArray<T> array, Range headRange, Range tailRange, out VariableArray<T> tail)
        {
            VariableArray<T> head = Variable.Array<T>(headRange);
            tail = Variable.Array<T>(tailRange);
            head.SetTo(Collection.Split, array, headRange.Size, tail);
            return head;
        }

        /// <summary>
        /// Creates two arrays by splitting a random array into disjoint pieces.
        /// </summary>
        /// <typeparam name="TItem">The inner variable type</typeparam>
        /// <typeparam name="T">The domain type of array elements</typeparam>
        /// <param name="array">The source array</param>
        /// <param name="headRange">Specifies the number of elements to put into the returned array</param>
        /// <param name="tailRange">The range associated with the tail array</param>
        /// <param name="tail">On return, holds the remaining elements</param>
        /// <returns>The initial elements of <paramref name="array"/></returns>
        public static VariableArray<TItem, T[]> Split<TItem, T>(VariableArray<TItem, T[]> array, Range headRange, Range tailRange, out VariableArray<TItem, T[]> tail)
            where TItem : Variable, ICloneable, SettableTo<TItem>
        {
            VariableArray<TItem, T[]> head = ReplaceRanges(array, headRange);
            tail = ReplaceRanges(array, tailRange);
            head.SetTo(Collection.Split, array, headRange.Size, tail);
            return head;
        }

        /// <summary>
        /// Copy a contiguous subvector of a random vector.
        /// </summary>
        /// <param name="source">Random vector</param>
        /// <param name="startIndex">Index of the first element to copy</param>
        /// <param name="count">Number of elements to copy</param>
        /// <returns>A new vector variable whose value is equal to <c>Vector.Subvector(source, startIndex, count)</c></returns>
        public static Variable<Vector> Subvector(Variable<Vector> source, Variable<int> startIndex, Variable<int> count)
        {
            return Variable<Vector>.Factor(Probabilistic.Math.Vector.Subvector, source, startIndex, count);
        }

        /// <summary>
        /// Copy an element of a vector.
        /// </summary>
        /// <param name="source">Random vector.</param>
        /// <param name="index">Index of the element to copy.</param>
        /// <returns>A new double variable equal to <c>source[index]</c></returns>
        public static Variable<double> GetItem(Variable<Vector> source, Variable<int> index)
        {
            return Variable<double>.Factor(Collection.GetItem<double>, source, index);
        }


        /// <summary>
        /// Returns a copy of the argument
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="x">variable to copy.</param>
        /// <returns>A new variable that is constrained to equal the argument.</returns>
        public static Variable<T> Copy<T>(Variable<T> x)
        {
            Variable<T> result = Variable<T>.Factor(Clone.Copy<T>, x);
            Range valueRange = x.GetValueRange(false);
            if (valueRange != null)
                result.AddAttribute(new ValueRange(valueRange));
            return result;
        }

        /// <summary>
        /// Copy an array
        /// </summary>
        /// <typeparam name="T">The domain type of an array element.</typeparam>
        /// <param name="array">array to copy.</param>
        /// <returns>A new array that is constrained to equal the argument.</returns>
        public static VariableArray<T> Copy<T>(VariableArray<T> array)
        {
            var result = Array<T>(array.Range);
            CreateVariableArray(result, array);
            result[array.Range] = Copy(array[array.Range]);
            return result;
        }

        /// <summary>
        /// Copy an array
        /// </summary>
        /// <typeparam name="T">The domain type of an array element.</typeparam>
        /// <param name="array">array to copy.</param>
        /// <returns>A new array that is constrained to equal the argument.</returns>
        public static VariableArray<VariableArray<T>, T[][]> Copy<T>(VariableArray<VariableArray<T>, T[][]> array)
        {
            var result = Array<T>(array[array.Range], array.Range);
            CreateVariableArray(result, array);
            result[array.Range] = Copy(array[array.Range]);
            return result;
        }

        /// <summary>
        /// Copy an array
        /// </summary>
        /// <typeparam name="TItem">The item prototype</typeparam>
        /// <typeparam name="T">The domain type of an array element.</typeparam>
        /// <param name="array">array to copy.</param>
        /// <returns>A new array that is constrained to equal the argument.</returns>
        public static VariableArray<VariableArray<TItem, T>, T[]> Copy<TItem, T>(VariableArray<VariableArray<TItem, T>, T[]> array)
            where TItem : Variable, ICloneable, SettableTo<TItem>
        {
            var result = Array(array[array.Range], array.Range);
            CreateVariableArray(result, array);
            // value = Copy(array[array.Range]);
            var value = Microsoft.ML.Probabilistic.Compiler.Reflection.Invoker.InvokeStatic(typeof(Variable), "Copy", array[array.Range]);
            // result[array.Range] = value;
            Microsoft.ML.Probabilistic.Compiler.Reflection.Invoker.InvokeMember(result.GetType(), "set_Item", BindingFlags.Instance | BindingFlags.InvokeMethod | BindingFlags.Public, result, array.Range, value);
            return result;
        }

        /// <summary>
        /// Creates two copies of the argument, that will be updated in order during inference.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="v">Variable to copy.</param>
        /// <param name="second">The second copy.</param>
        /// <returns>The first copy.</returns>
        public static Variable<T> SequentialCopy<T>(Variable<T> v, out Variable<T> second)
        {
            second = Variable.New<T>();
            return Variable<T>.Factor(LowPriority.SequentialCopy<T>, v, second);
        }

        /// <summary>
        /// Creates two copies of the argument, that will be updated in order during inference.
        /// </summary>
        /// <typeparam name="T">The domain type of an array element.</typeparam>
        /// <param name="array">Array to copy.</param>
        /// <param name="second">The second copy.</param>
        /// <returns>The first copy.</returns>
        public static VariableArray<T> SequentialCopy<T>(VariableArray<T> array, out VariableArray<T> second)
        {
            var range = array.Range;
            var first = Variable.Array<T>(range);
            CreateVariableArray(first, array);
            second = Variable.Array<T>(range);
            CreateVariableArray(second, array);
            first[range] = Variable<T>.Factor(LowPriority.SequentialCopy<T>, array[range], second[range]);
            return first;
        }

        /// <summary>
        /// Creates two copies of the argument, that will be updated in order during inference.
        /// </summary>
        /// <typeparam name="T">The domain type of an array element.</typeparam>
        /// <param name="array">Array to copy.</param>
        /// <param name="second">The second copy.</param>
        /// <returns>The first copy.</returns>
        public static VariableArray<VariableArray<T>, T[][]> SequentialCopy<T>(VariableArray<VariableArray<T>, T[][]> array, out VariableArray<VariableArray<T>, T[][]> second)
        {
            var range1 = array.Range;
            var range2 = array[array.Range].Range;
            var first = Array<T>(array[array.Range], array.Range);
            CreateVariableArray(first, array);
            second = Array<T>(array[array.Range], array.Range);
            CreateVariableArray(second, array);
            first[range1][range2] = Variable<T>.Factor(LowPriority.SequentialCopy<T>, array[range1][range2], second[range1][range2]);
            return first;
        }

        /// <summary>
        /// Creates a copy of the argument where the forward message is uniform when <paramref name="shouldCut"/> is true.  Used to control inference.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="x"></param>
        /// <param name="shouldCut"></param>
        /// <returns></returns>
        public static Variable<T> CutForwardWhen<T>(Variable<T> x, Variable<bool> shouldCut)
        {
            Variable<T> result = Variable<T>.Factor(Factors.Cut.ForwardWhen, x, shouldCut);
            Range valueRange = x.GetValueRange(false);
            if (valueRange != null)
                result.AddAttribute(new ValueRange(valueRange));
            return result;
        }

        /// <summary>
        /// Returns a cut of the argument. Cut is equivalent to random(infer()).
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="x">variable to copy.</param>
        /// <returns>A new variable that is constrained to equal the argument.</returns>
        /// <remarks>Cut allows forward messages to pass through unchanged, whereas backward messages are cut off.</remarks>
        public static Variable<T> Cut<T>(Variable<T> x)
        {
            Variable<T> result = Variable<T>.Factor(Factors.Cut.Backward, x);
            Range valueRange = x.GetValueRange(false);
            if (valueRange != null)
                result.AddAttribute(new ValueRange(valueRange));
            return result;
        }

        /// <summary>
        /// Returns a cut of the argument. Cut is equivalent to random(infer()).
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="array">variable to copy.</param>
        /// <returns>A new variable that is constrained to equal the argument.</returns>
        /// <remarks>Cut allows forward messages to pass through unchanged, whereas backward messages are cut off.</remarks>
        public static VariableArray<T> Cut<T>(VariableArray<T> array)
        {
            var result = Array<T>(array.Range);
            CreateVariableArray(result, array);
            result[result.Range] = Cut(array[array.Range]);
            return result;
        }

        /// <summary>
        /// Returns a cut of the argument. Cut is equivalent to random(infer()).
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="array">variable to copy.</param>
        /// <returns>A new variable that is constrained to equal the argument.</returns>
        /// <remarks>Cut allows forward messages to pass through unchanged, whereas backward messages are cut off.</remarks>
        public static VariableArray<VariableArray<T>, T[][]> Cut<T>(VariableArray<VariableArray<T>, T[][]> array)
        {
            var result = Array<T>(array[array.Range], array.Range);
            CreateVariableArray(result, array);
            result[result.Range] = Cut(array[array.Range]);
            return result;
        }

        /// <summary>
        /// Replicates a value multiple times.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="value">The value to replicate.</param>
        /// <param name="range">The range used to index the output array.</param>
        /// <returns>The array of replicated values.</returns>
        public static VariableArray<T> Replicate<T>(Variable<T> value, Range range)
        {
            var result = Array<T>(range);
            CreateVariableArray(result, value);
            result[range] = Variable.Copy(value).ForEach(range);
            return result;
        }

        /// <summary>
        /// Replicates an array multiple times.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="array">The array to replicate.</param>
        /// <param name="range">The range used to index the output array.</param>
        /// <returns>The array of replicated arrays.</returns>
        public static VariableArray<VariableArray<T>, T[][]> Replicate<T>(VariableArray<T> array, Range range)
        {
            var result = Array<T>(array, range);
            CreateVariableArray(result, array);
            result[range][array.Range] = Variable.Copy(array[array.Range]).ForEach(range);
            return result;
        }

        /// <summary>
        /// Replicates an array multiple times.
        /// </summary>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="array">The array to replicate.</param>
        /// <param name="range">The range used to index the output array.</param>
        /// <returns>The array of replicated arrays.</returns>
        public static VariableArray<VariableArray<VariableArray<T>, T[][]>, T[][][]> Replicate<T>(VariableArray<VariableArray<T>, T[][]> array, Range range)
        {
            var result = Array(array, range);
            CreateVariableArray(result, array);
            var array2 = array[array.Range];
            result[range][array.Range][array2.Range] = Variable.Copy(array2[array2.Range]).ForEach(range);
            return result;
        }

        /// <summary>
        /// Replicates an array multiple times.
        /// </summary>
        /// <typeparam name="TItem">The element variable type.</typeparam>
        /// <typeparam name="T">The domain type.</typeparam>
        /// <param name="array">The array to replicate.</param>
        /// <param name="range">The range used to index the output array.</param>
        /// <returns>The array of replicated arrays.</returns>
        public static VariableArray<VariableArray<VariableArray<TItem, T>, T[]>, T[][]> Replicate<TItem, T>(VariableArray<VariableArray<TItem, T>, T[]> array, Range range)
            where TItem : Variable, ICloneable, SettableTo<TItem>
        {
            var result = Array(array, range);
            CreateVariableArray(result, array);
            result[range] = (VariableArray<VariableArray<TItem, T>, T[]>)Variable.Copy<TItem, T>(array).ForEach(range);
            return result;
        }

#if SUPPRESS_AMBIGUOUS_REFERENCE_WARNINGS
#pragma warning disable 419
#endif

        /// <summary>
        /// Gets a variable array containing different items of a source array.
        /// </summary>
        /// <typeparam name="T">The domain type of array elements.</typeparam>
        /// <param name="array">The source array</param>
        /// <param name="indices">Variable array containing the indices of the elements to get.  The indices must all be different.</param>
        /// <returns>variable array with the specified items</returns>
        /// <remarks>
        /// To allow duplicate indices, use <see cref="GetItems"/>.
        /// </remarks>
        public static VariableArray<T> Subarray<T>(Variable<T[]> array, VariableArray<int> indices)
        {
            VariableArray<T> result = new VariableArray<T>(indices.Range);
            result.SetTo(Collection.Subarray, array, indices);
            return result;
        }

        /// <summary>
        /// Gets a variable array containing different items of a source list
        /// </summary>
        /// <typeparam name="T">The domain type of array elements.</typeparam>
        /// <param name="array">The source array</param>
        /// <param name="indices">Variable array containing the indices of the elements to get.  The indices must all be different.</param>
        /// <returns>variable array with the specified items</returns>
        /// <remarks>
        /// To allow duplicate indices, use <see cref="GetItems"/>.
        /// </remarks>
        public static VariableArray<T> Subarray<T>(Variable<IReadOnlyList<T>> array, VariableArray<int> indices)
        {
            VariableArray<T> result = new VariableArray<T>(indices.Range);
            result.SetTo(Collection.Subarray, array, indices);
            return result;
        }

        /// <summary>
        /// Gets a variable array containing different items of a source list
        /// </summary>
        /// <typeparam name="T">The domain type of array elements.</typeparam>
        /// <param name="array">The source array</param>
        /// <param name="indices">Variable array containing the indices of the elements to get.  The indices must all be different.</param>
        /// <returns>variable array with the specified items</returns>
        /// <remarks>
        /// To allow duplicate indices, use <see cref="GetItems"/>.
        /// </remarks>
        public static VariableArray<T> Subarray<T>(VariableArray<T> array, VariableArray<int> indices)
        {
            VariableArray<T> result = new VariableArray<T>(indices.Range);
            result.SetTo(Collection.Subarray, array, indices);
            return result;
        }

        /// <summary>
        /// Gets a variable array containing different items of two source lists
        /// </summary>
        /// <typeparam name="T">The domain type of array elements.</typeparam>
        /// <param name="array">The source array</param>
        /// <param name="indices">Variable array containing the indices of the elements to get.  The indices must all be different.  Any index less than zero corresponds to fetching from the second source array at the same position.</param>
        /// <param name="array2">The second source array</param>
        /// <returns>variable array with the specified items</returns>        
        /// <remarks>
        /// To allow duplicate indices, use <see cref="GetItems"/>.
        /// </remarks>
        public static VariableArray<T> Subarray2<T>(VariableArray<T> array, VariableArray<int> indices, VariableArray<T> array2)
        {
            VariableArray<T> result = new VariableArray<T>(indices.Range);
            result.SetTo(Collection.Subarray2, array, indices, array2);
            return result;
        }

        /// <summary>
        /// Gets a variable array containing different items of a source list.
        /// </summary>
        /// <typeparam name="T">The domain type of array elements.</typeparam>
        /// <param name="array">The source array.</param>
        /// <param name="indices">Variable array containing the indices of the elements to get.  The indices must all be different.</param>
        /// <returns>variable array with the specified items.</returns>
        /// <remarks>
        /// To allow duplicate indices, use <see cref="GetItems"/>.
        /// </remarks>
        public static VariableArray<T> Subarray<T>(VariableArray<T> array, VariableArray<Variable<int>, IReadOnlyList<int>> indices)
        {
            var result = new VariableArray<T>(indices.Range);
            result.SetTo(Collection.Subarray, array, indices);
            return result;
        }

        /// <summary>
        /// Gets a jagged variable array containing different items of a source list.
        /// </summary>
        /// <typeparam name="T">The domain type of array elements.</typeparam>
        /// <param name="array">The source array.</param>
        /// <param name="indices">Jagged array containing the indices of the elements to get.  The indices must all be different.  The shape of this array determines the shape of the result.</param>
        /// <returns>variable array with the specified items.</returns>
        /// <remarks>
        /// To allow duplicate indices, use <see cref="JaggedSubarray"/> or <see cref="GetItems"/>.
        /// </remarks>
        public static VariableArray<VariableArray<T>, T[][]> SplitSubarray<T>(VariableArray<T> array, VariableArray<VariableArray<int>, int[][]> indices)
        {
            VariableArray<VariableArray<T>, T[][]> result = Variable.Array(Variable<T>.Array(indices[indices.Range].Range), indices.Range);
            result.SetTo(Collection.SplitSubarray, array, indices);
            return result;
        }

        /// <summary>
        /// Gets a jagged variable array containing different items of a source list.
        /// </summary>
        /// <typeparam name="TItem">The inner variable type</typeparam>
        /// <typeparam name="T">The domain type of array elements.</typeparam>
        /// <param name="array">The source array.</param>
        /// <param name="indices">Jagged array containing the indices of the elements to get.  The indices must all be different.  The shape of this array determines the shape of the result.</param>
        /// <returns>variable array with the specified items.</returns>
        /// <remarks>
        /// To allow duplicate indices, use <see cref="JaggedSubarray"/> or <see cref="GetItems"/>.
        /// </remarks>
        public static VariableArray<VariableArray<TItem, T[]>, T[][]> SplitSubarray<TItem, T>(VariableArray<TItem, T[]> array, VariableArray<VariableArray<int>, int[][]> indices)
            where TItem : Variable, ICloneable, SettableTo<TItem>
        {
            var result = ReplaceRanges(array, indices);
            result.SetTo(Collection.SplitSubarray, array, indices);
            return result;
        }

        /// <summary>
        /// Gets a jagged variable array containing items of a source list.
        /// </summary>
        /// <typeparam name="T">The domain type of array elements.</typeparam>
        /// <param name="array">The source array.</param>
        /// <param name="indices">Variable array containing the indices of the elements to get.  indices[i][j] must be different for different j and same i, but can match for different i. The shape of this array determines the shape of the result.</param>
        /// <returns>A jagged variable array with the specified items.</returns>
        /// <remarks>
        /// If all indices are different, use <see cref="SplitSubarray"/>.
        /// To allow duplicate indices, use <see cref="GetItems"/>.
        /// </remarks>
        public static VariableArray<VariableArray<T>, T[][]> JaggedSubarray<T>(VariableArray<T> array, VariableArray<VariableArray<int>, int[][]> indices)
        {
            VariableArray<VariableArray<T>, T[][]> result = Variable.Array(Variable<T>.Array(indices[indices.Range].Range), indices.Range);
            result.SetTo(Collection.JaggedSubarray, array, indices);
            return result;
        }

        /// <summary>
        /// Gets a variable array containing different items of a source list
        /// </summary>
        /// <typeparam name="TItem">The inner variable type</typeparam>
        /// <typeparam name="T">The domain type of array elements</typeparam>
        /// <param name="array">The source array</param>
        /// <param name="indices">Variable array containing the indices of the elements to get.  The indices must all be different.</param>
        /// <returns>variable array with the specified items</returns>
        /// <remarks>
        /// To allow duplicate indices, use <see cref="GetItems"/>.
        /// </remarks>
        public static VariableArray<TItem, T[]> Subarray<TItem, T>(VariableArray<TItem, T[]> array, VariableArray<int> indices)
            where TItem : Variable, ICloneable, SettableTo<TItem>
        {
            VariableArray<TItem, T[]> result = ReplaceRanges(array, indices);
            result.SetTo(Collection.Subarray, array, indices);
            return result;
        }

        /// <summary>
        /// Create a new array like 'array' but where the first range is changed to indices.Range
        /// </summary>
        /// <typeparam name="TItem"></typeparam>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        private static VariableArray<VariableArray<TItem, T[]>, T[][]> ReplaceRanges<TItem, T>(VariableArray<TItem, T[]> array, VariableArray<VariableArray<int>, int[][]> indices)
            where TItem : Variable, ICloneable, SettableTo<TItem>
        {
            Dictionary<Range, Range> replacements = new Dictionary<Range, Range>();
            Dictionary<IModelExpression, IModelExpression> expressionReplacements = new Dictionary<IModelExpression, IModelExpression>();
            replacements.Add(array.Range, indices.Range);
            expressionReplacements.Add(array.Range, indices[indices.Range]);
            Variable v = (Variable)array;
            IVariableArray parent = v.ArrayVariable;
            while (parent != null)
            {
                for (int i = 0; i < v.indices.Count; i++)
                {
                    expressionReplacements.Add(parent.Ranges[i], v.indices[i]);
                }
                v = (Variable)parent;
                parent = v.ArrayVariable;
            }
            var result = (VariableArray<VariableArray<TItem, T[]>, T[][]>)((IVariableArray)array).ReplaceRanges(replacements, expressionReplacements, deepCopy: true);
            return result;
        }

        /// <summary>
        /// Create a new array like 'array' but where the first range is changed to indices.Range
        /// </summary>
        /// <typeparam name="TItem"></typeparam>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        private static VariableArray<TItem, T[]> ReplaceRanges<TItem, T>(VariableArray<TItem, T[]> array, VariableArray<int> indices)
            where TItem : Variable, ICloneable, SettableTo<TItem>
        {
            Dictionary<Range, Range> replacements = new Dictionary<Range, Range>();
            Dictionary<IModelExpression, IModelExpression> expressionReplacements = new Dictionary<IModelExpression, IModelExpression>();
            replacements.Add(array.Range, indices.Range);
            expressionReplacements.Add(array.Range, indices[indices.Range]);
            Variable v = (Variable)array;
            IVariableArray parent = v.ArrayVariable;
            while (parent != null)
            {
                for (int i = 0; i < v.indices.Count; i++)
                {
                    expressionReplacements.Add(parent.Ranges[i], v.indices[i]);
                }
                v = (Variable)parent;
                parent = v.ArrayVariable;
            }
            VariableArray<TItem, T[]> result = (VariableArray<TItem, T[]>)((IVariableArray)array).ReplaceRanges(replacements, expressionReplacements, deepCopy: true);
            return result;
        }

        /// <summary>
        /// Create a new array like 'array' but where the first range is changed
        /// </summary>
        /// <typeparam name="TItem"></typeparam>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <param name="newRange"></param>
        /// <returns></returns>
        private static VariableArray<TItem, T[]> ReplaceRanges<TItem, T>(VariableArray<TItem, T[]> array, Range newRange)
            where TItem : Variable, ICloneable, SettableTo<TItem>
        {
            Dictionary<Range, Range> replacements = new Dictionary<Range, Range>();
            Dictionary<IModelExpression, IModelExpression> expressionReplacements = new Dictionary<IModelExpression, IModelExpression>();
            replacements.Add(array.Range, newRange);
            Variable v = (Variable)array;
            IVariableArray parent = v.ArrayVariable;
            while (parent != null)
            {
                for (int i = 0; i < v.indices.Count; i++)
                {
                    expressionReplacements.Add(parent.Ranges[i], v.indices[i]);
                }
                v = (Variable)parent;
                parent = v.ArrayVariable;
            }
            VariableArray<TItem, T[]> result = (VariableArray<TItem, T[]>)((IVariableArray)array).ReplaceRanges(replacements, expressionReplacements, deepCopy: true);
            return result;
        }

        /// <summary>
        /// Gets a variable array containing (possibly duplicated) items of a source array
        /// </summary>
        /// <typeparam name="T">The domain type of array elements</typeparam>
        /// <param name="array">The source array</param>
        /// <param name="indices">Variable array containing the indices of the elements to get.  Indices may be duplicated.</param>
        /// <returns>variable array with the specified items</returns>
        /// <remarks>
        /// If the indices are known to be all different, use <see cref="Subarray"/> for greater efficiency.
        /// </remarks>
        public static VariableArray<T> GetItems<T>(Variable<T[]> array, VariableArray<int> indices)
        {
            VariableArray<T> result = new VariableArray<T>(indices.Range);
            result.SetTo(Collection.GetItems, array, indices);
            return result;
        }

        /// <summary>
        /// Gets a variable array containing (possibly duplicated) items of a source array
        /// </summary>
        /// <typeparam name="T">The domain type of array elements</typeparam>
        /// <param name="array">The source array</param>
        /// <param name="indices">Variable array containing the indices of the elements to get.  Indices may be duplicated.</param>
        /// <returns>variable array with the specified items</returns>
        /// <remarks>
        /// If the indices are known to be all different, use <see cref="Subarray"/> for greater efficiency.
        /// </remarks>
        public static VariableArray<T> GetItems<T>(Variable<IReadOnlyList<T>> array, VariableArray<int> indices)
        {
            VariableArray<T> result = new VariableArray<T>(indices.Range);
            result.SetTo(Collection.GetItems, array, indices);
            return result;
        }

        /// <summary>
        /// Gets a variable array containing (possibly duplicated) items of a source array
        /// </summary>
        /// <typeparam name="T">The domain type of array elements</typeparam>
        /// <param name="array">The source array</param>
        /// <param name="indices">Variable array containing the indices of the elements to get.  Indices may be duplicated.</param>
        /// <returns>variable array with the specified items</returns>
        /// <remarks>
        /// If the indices are known to be all different, use <see cref="Subarray"/> for greater efficiency.
        /// </remarks>
        public static VariableArray<T> GetItems<T>(VariableArray<T> array, VariableArray<int> indices)
        {
            VariableArray<T> result = new VariableArray<T>(indices.Range);
            result.SetTo(Collection.GetItems, array, indices);
            return result;
        }

        /// <summary>
        /// Gets a variable array containing (possibly duplicated) items of a source array
        /// </summary>
        /// <typeparam name="TItem">The inner variable type</typeparam>
        /// <typeparam name="T">The domain type of array elements</typeparam>
        /// <param name="array">The source array</param>
        /// <param name="indices">Variable array containing the indices of the elements to get.  Indices may be duplicated.</param>
        /// <returns>variable array with the specified items</returns>
        /// <remarks>
        /// If the indices are known to be all different, use <see cref="Subarray"/> for greater efficiency.
        /// </remarks>
        public static VariableArray<TItem, T[]> GetItems<TItem, T>(VariableArray<TItem, T[]> array, VariableArray<int> indices)
            where TItem : Variable, ICloneable, SettableTo<TItem>
        {
            VariableArray<TItem, T[]> result = ReplaceRanges(array, indices);
            result.SetTo(Collection.GetItems, array, indices);
            return result;
        }

#region Variable<char> factories

        /// <summary>
        /// Creates a character random variable defined by a discrete distribution induced by a given probability vector.
        /// </summary>
        /// <param name="probabilities">The probability vector.</param>
        /// <returns>The created random variable.</returns>
        public static Variable<char> Char(Variable<Vector> probabilities)
        {
            return Variable<char>.Factor(Factor.Char, probabilities);
        }

        /// <summary>
        /// Creates a character random variable from a uniform distribution over all possible characters.
        /// </summary>
        /// <returns>The created random variable.</returns>
        public static Variable<char> CharUniform()
        {
            return Variable.Random(DiscreteChar.Uniform());
        }

        /// <summary>
        /// Creates a character random variable from a uniform distribution over lowercase letters 'a'..'z'.
        /// </summary>
        /// <returns>The created random variable.</returns>
        public static Variable<char> CharLower()
        {
            return Variable.Random(DiscreteChar.Lower());
        }

        /// <summary>
        /// Creates a character random variable from a uniform distribution over uppercase letters 'A'..'Z'.
        /// </summary>
        /// <returns>The created random variable.</returns>
        public static Variable<char> CharUpper()
        {
            return Variable.Random(DiscreteChar.Upper());
        }

        /// <summary>
        /// Creates a character random variable from a uniform distribution over letters in 'a'..'z' and 'A'..'Z'.
        /// </summary>
        /// <returns>The created random variable.</returns>
        public static Variable<char> CharLetter()
        {
            return Variable.Random(DiscreteChar.Letter());
        }

        /// <summary>
        /// Creates a character random variable from a uniform distribution over digits '0'..'9'.
        /// </summary>
        /// <returns>The created random variable.</returns>
        public static Variable<char> CharDigit()
        {
            return Variable.Random(DiscreteChar.Digit());
        }

        /// <summary>
        /// Creates a character random variable from a uniform distribution over 'a'..'z', 'A'..'Z' and '0'..'9'.
        /// </summary>
        /// <returns>The created random variable.</returns>
        public static Variable<char> CharLetterOrDigit()
        {
            return Variable.Random(DiscreteChar.LetterOrDigit());
        }

        /// <summary>
        /// Creates a character random variable from a uniform distribution over word characters
        /// ('a'..'z', 'A'..'Z', '0'..'9', '_' and '\'').
        /// </summary>
        /// <returns>The created random variable.</returns>
        public static Variable<char> CharWord()
        {
            return Variable.Random(DiscreteChar.WordChar());
        }

        /// <summary>
        /// Creates a character random variable from a uniform distribution over all characters except
        /// ('a'..'z', 'A'..'Z', '0'..'9', '_' and '\'').
        /// </summary>
        /// <returns>The created random variable.</returns>
        public static Variable<char> CharNonWord()
        {
            return Variable.Random(DiscreteChar.NonWordChar());
        }

        /// <summary>
        /// Creates a character random variable from a uniform distribution over whitespace characters
        /// ('\t'..'\r', ' ').
        /// </summary>
        /// <returns>The created random variable.</returns>
        public static Variable<char> CharWhitespace()
        {
            return Variable.Random(DiscreteChar.Whitespace());
        }

#endregion

#region Variable<string> factories

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all possible strings.
        /// </summary>
        /// <returns>The created random variable.</returns>
        /// <remarks>The resulting random variable has an improper distribution.</remarks>
        public static Variable<string> StringUniform()
        {
            return Variable.Random(StringDistribution.Any());
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all non-empty strings
        /// of lowercase letters (see <see cref="Variable.CharLower"/>).
        /// </summary>
        /// <returns>The created random variable.</returns>
        /// <remarks>The resulting random variable has an improper distribution.</remarks>
        public static Variable<string> StringLower()
        {
            return Variable.StringLower(minLength: 1);
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all strings
        /// of lowercase letters (see <see cref="Variable.CharLower"/>) with length in given bounds.
        /// </summary>
        /// <param name="minLength">The minimum possible string length.</param>
        /// <param name="maxLength">
        /// The maximum possible string length. If <see langword="null"/>, there will be no upper limit on length.
        /// </param>
        /// <returns>The created random variable.</returns>
        /// <remarks>
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// the resulting random variable has an improper distribution.</remarks>
        public static Variable<string> StringLower(Variable<int> minLength, Variable<int> maxLength = null)
        {
            return Variable.String(minLength, maxLength, DiscreteChar.Lower());
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all non-empty strings
        /// of uppercase letters (see <see cref="Variable.CharUpper"/>).
        /// </summary>
        /// <returns>The created random variable.</returns>
        /// <remarks>The resulting random variable has an improper distribution.</remarks>
        public static Variable<string> StringUpper()
        {
            return Variable.StringUpper(minLength: 1);
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all strings
        /// of uppercase letters (see <see cref="Variable.CharUpper"/>) with length in given bounds.
        /// </summary>
        /// <param name="minLength">The minimum possible string length.</param>
        /// <param name="maxLength">
        /// The maximum possible string length. If <see langword="null"/>, there will be no upper limit on length.
        /// </param>
        /// <returns>The created random variable.</returns>
        /// <remarks>
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// the resulting random variable has an improper distribution.
        /// </remarks>
        public static Variable<string> StringUpper(Variable<int> minLength, Variable<int> maxLength = null)
        {
            return Variable.String(minLength, maxLength, DiscreteChar.Upper());
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all non-empty strings
        /// of letters (see <see cref="Variable.CharLetter"/>).
        /// </summary>
        /// <returns>The created random variable.</returns>
        /// <remarks>The resulting random variable has an improper distribution.</remarks>
        public static Variable<string> StringLetters()
        {
            return Variable.StringLetters(minLength: 1);
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all strings
        /// of letters (see <see cref="Variable.CharLetter"/>) with length in given bounds.
        /// </summary>
        /// <param name="minLength">The minimum possible string length.</param>
        /// <param name="maxLength">
        /// The maximum possible string length. If <see langword="null"/>, there will be no upper limit on length.
        /// </param>
        /// <returns>The created random variable.</returns>
        /// <remarks>
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// the resulting random variable has an improper distribution.
        /// </remarks>
        public static Variable<string> StringLetters(Variable<int> minLength, Variable<int> maxLength = null)
        {
            return Variable.String(minLength, maxLength, DiscreteChar.Letter());
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all non-empty strings
        /// of digits (see <see cref="Variable.CharDigit"/>).
        /// </summary>
        /// <returns>The created random variable.</returns>
        /// <remarks>The resulting random variable has an improper distribution.</remarks>
        public static Variable<string> StringDigits()
        {
            return Variable.StringDigits(minLength: 1);
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all strings
        /// of digits (see <see cref="Variable.CharDigit"/>) with length in given bounds.
        /// </summary>
        /// <param name="minLength">The minimum possible string length.</param>
        /// <param name="maxLength">
        /// The maximum possible string length. If <see langword="null"/>, there will be no upper limit on length.
        /// </param>
        /// <returns>The created random variable.</returns>
        /// <remarks>
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// the resulting random variable has an improper distribution.
        /// </remarks>
        public static Variable<string> StringDigits(Variable<int> minLength, Variable<int> maxLength = null)
        {
            return Variable.String(minLength, maxLength, DiscreteChar.Digit());
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all non-empty strings
        /// of letters or digits (see <see cref="Variable.CharLetterOrDigit"/>).
        /// </summary>
        /// <returns>The created random variable.</returns>
        /// <remarks>The resulting random variable has an improper distribution.</remarks>
        public static Variable<string> StringLettersOrDigits()
        {
            return Variable.StringLettersOrDigits(minLength: 1);
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all strings
        /// of letters or digits (see <see cref="Variable.CharLetterOrDigit"/>) with length in given bounds.
        /// </summary>
        /// <param name="minLength">The minimum possible string length.</param>
        /// <param name="maxLength">
        /// The maximum possible string length. If <see langword="null"/>, there will be no upper limit on length.
        /// </param>
        /// <returns>The created random variable.</returns>
        /// <remarks>
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// the resulting random variable has an improper distribution.
        /// </remarks>
        public static Variable<string> StringLettersOrDigits(Variable<int> minLength, Variable<int> maxLength = null)
        {
            return Variable.String(minLength, maxLength, DiscreteChar.LetterOrDigit());
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all non-empty strings
        /// of whitespace characters (see <see cref="Variable.CharWhitespace"/>).
        /// </summary>
        /// <returns>The created random variable.</returns>
        /// <remarks>The resulting random variable has an improper distribution.</remarks>
        public static Variable<string> StringWhitespace()
        {
            return Variable.StringWhitespace(minLength: 1);
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all strings
        /// of whitespace characters (see <see cref="Variable.CharWhitespace"/>) with length in given bounds.
        /// </summary>
        /// <param name="minLength">The minimum possible string length.</param>
        /// <param name="maxLength">
        /// The maximum possible string length. If <see langword="null"/>, there will be no upper limit on length.
        /// </param>
        /// <returns>The created random variable.</returns>
        /// <remarks>
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// the resulting random variable has an improper distribution.
        /// </remarks>
        public static Variable<string> StringWhitespace(Variable<int> minLength, Variable<int> maxLength = null)
        {
            return Variable.String(minLength, maxLength, DiscreteChar.Whitespace());
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all non-empty strings
        /// of word characters (see <see cref="Variable.CharWord"/>).
        /// </summary>
        /// <returns>The created random variable.</returns>
        /// <remarks>The resulting random variable has an improper distribution.</remarks>
        public static Variable<string> StringWord()
        {
            return Variable.StringWord(minLength: 1);
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all strings
        /// of word characters (see <see cref="Variable.CharWord"/>) with length in given bounds.
        /// </summary>
        /// <param name="minLength">The minimum possible string length.</param>
        /// <param name="maxLength">
        /// The maximum possible string length. If <see langword="null"/>, there will be no upper limit on length.
        /// </param>
        /// <returns>The created random variable.</returns>
        /// <remarks>
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// the resulting random variable has an improper distribution.
        /// </remarks>
        public static Variable<string> StringWord(Variable<int> minLength, Variable<int> maxLength = null)
        {
            return Variable.String(minLength, maxLength, DiscreteChar.WordChar());
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all non-empty strings
        /// of non-word characters (see <see cref="Variable.CharNonWord"/>).
        /// </summary>
        /// <returns>The created random variable.</returns>
        /// <remarks>The resulting random variable has an improper distribution.</remarks>
        public static Variable<string> StringNonWord()
        {
            return Variable.StringNonWord(minLength: 1);
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all strings
        /// of non-word characters (see <see cref="Variable.CharNonWord"/>) with length in given bounds.
        /// </summary>
        /// <param name="minLength">The minimum possible string length.</param>
        /// <param name="maxLength">
        /// The maximum possible string length. If <see langword="null"/>, there will be no upper limit on length
        /// .</param>
        /// <returns>The created random variable.</returns>
        /// <remarks>
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// the resulting random variable has an improper distribution.
        /// </remarks>
        public static Variable<string> StringNonWord(Variable<int> minLength, Variable<int> maxLength = null)
        {
            return Variable.String(minLength, maxLength, DiscreteChar.NonWordChar());
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all strings
        /// consisting of an uppercase letter followed by one or more lowercase letters.
        /// </summary>
        /// <returns>The created random variable.</returns>
        /// <remarks>The resulting random variable has an improper distribution.</remarks>
        public static Variable<string> StringCapitalized()
        {
            return StringCapitalized(minLength: 2);
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all strings
        /// consisting of an uppercase letter followed by one or more lowercase letters, with length in specified bounds.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Should be 2 or more.</param>
        /// <param name="maxLength">The maximum possible string length.
        /// If <see langword="null"/>, there will be no upper limit on length.
        /// </param>
        /// <returns>The created random variable.</returns>
        /// <remarks>
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// the resulting random variable has an improper distribution.
        /// </remarks>
        public static Variable<string> StringCapitalized(Variable<int> minLength, Variable<int> maxLength = null)
        {
            return maxLength is null
                ? Variable<string>.Factor(Factor.StringCapitalized, minLength)
                : Variable<string>.Factor(Factor.StringCapitalized, minLength, maxLength);
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all strings of given length.
        /// </summary>
        /// <param name="length">The desired string length.</param>
        /// <returns>The created random variable.</returns>
        /// <remarks>The resulting random variable has an improper distribution.</remarks>
        public static Variable<string> StringOfLength(Variable<int> length)
        {
            return StringOfLength(length, DiscreteChar.Uniform());
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all strings of given length.
        /// String characters are restricted to be non zero probability characters under the given character distribution.
        /// </summary>
        /// <param name="length">The desired string length.</param>
        /// <param name="allowedCharacters">The distribution specifying the allowed string characters.</param>
        /// <returns>The created random variable.</returns>
        /// <remarks>The resulting random variable has an improper distribution.</remarks>
        public static Variable<string> StringOfLength(Variable<int> length, Variable<DiscreteChar> allowedCharacters)
        {
            return Variable<string>.Factor(Factor.StringOfLength, length, allowedCharacters);
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all strings with length
        /// in given bounds.
        /// </summary>
        /// <param name="minLength">The minimum possible string length.</param>
        /// <param name="maxLength">
        /// The maximum possible string length. If <see langword="null"/>, there will be no upper limit on length.
        /// </param>
        /// <returns>The created random variable.</returns>
        /// <remarks>
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// the resulting random variable has an improper distribution.
        /// </remarks>
        public static Variable<string> String(Variable<int> minLength, Variable<int> maxLength = null)
        {
            return String(minLength, maxLength, DiscreteChar.Uniform());
        }

        /// <summary>
        /// Creates a string random variable from a uniform distribution over all strings with length
        /// in given bounds. String characters are restricted to be non zero probability characters under the given character distribution.
        /// </summary>
        /// <param name="minLength">The minimum possible string length.</param>
        /// <param name="maxLength">
        /// The maximum possible string length. If <see langword="null"/>, there will be no upper limit on length.
        /// </param>
        /// <param name="allowedCharacters">The distribution specifying the allowed string characters.</param>
        /// <returns>The created random variable.</returns>
        /// <remarks>
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// the resulting random variable has an improper distribution.
        /// </remarks>
        public static Variable<string> String(Variable<int> minLength, Variable<int> maxLength, Variable<DiscreteChar> allowedCharacters)
        {
            return maxLength is null
                ? Variable<string>.Factor(Factor.String, minLength, allowedCharacters)
                : Variable<string>.Factor(Factor.String, minLength, maxLength, allowedCharacters);
        }

        /// <summary>
        /// Creates a string random variable from an array of characters.
        /// </summary>
        /// <param name="chars">The array of characters.</param>
        /// <returns>The created random variable.</returns>
        public static Variable<string> StringFromArray(VariableArray<char> chars)
        {
            return Variable<string>.Factor(Factor.StringFromArray, chars);
        }

#endregion

#region Operations on characters and strings

        /// <summary>
        /// Creates a string random variable which is a substring of a given string.
        /// </summary>
        /// <param name="str">The string.</param>
        /// <param name="start">The substring start index.</param>
        /// <param name="length">The substring length.</param>
        /// <returns>The created random variable for the substring of <paramref name="str"/>.</returns>
        public static Variable<string> Substring(Variable<string> str, Variable<int> start, Variable<int> length)
        {
            return Variable<string>.Factor(Factor.Substring, str, start, length);
        }

        /// <summary>
        /// Replaces argument placeholders such as {0}, {1} etc with arguments having the corresponding index,
        /// similar to what <see cref="string.Format(string, object[])"/> does.
        /// </summary>
        /// <param name="format">The string with argument placeholders.</param>
        /// <param name="args">The array of arguments.</param>
        /// <returns><paramref name="format"/> with argument placeholders replaced by arguments.</returns>
        /// <remarks>
        /// This method has the following notable differences from <see cref="string.Format(string, object[])"/>:
        /// <list type="bullet"></list>
        /// <item><description>Placeholder for each of the arguments must be present in the format string exactly once.</description></item>
        /// <item><description>No braces are allowed except for those used to specify placeholders.</description></item>
        /// </remarks>
        public static Variable<string> StringFormat(Variable<string> format, params Variable<string>[] args)
        {
            var argsRange = new Range(args.Length);
            VariableArray<string> argsArray = Variable.Array<string>(argsRange);
            argsArray.AddAttribute(new DoNotInfer());
            for (int i = 0; i < args.Length; ++i)
            {
                argsArray[i] = Variable.Copy(args[i]);
            }

            return StringFormat(format, argsArray);
        }

        /// <summary>
        /// Replaces argument placeholders such as {0}, {1} etc with arguments having the corresponding index,
        /// similar to what <see cref="string.Format(string, object[])"/> does.
        /// </summary>
        /// <param name="format">The string with argument placeholders.</param>
        /// <param name="args">The array of arguments.</param>
        /// <returns><paramref name="format"/> with argument placeholders replaced by arguments.</returns>
        /// <remarks>
        /// This method has the following notable differences from <see cref="string.Format(string, object[])"/>:
        /// <list type="bullet"></list>
        /// <item><description>Placeholder for each of the arguments must be present in the format string exactly once.</description></item>
        /// <item><description>No braces are allowed except for those used to specify placeholders.</description></item>
        /// </remarks>
        public static Variable<string> StringFormat(Variable<string> format, VariableArray<string> args)
        {
            return Variable<string>.Factor(Factor.StringFormat, format, args);
        }

        /// <summary>
        /// Creates a character random variable representing the character on a given position inside a given string.
        /// </summary>
        /// <param name="str">The string.</param>
        /// <param name="position">The character position.</param>
        /// <returns>The created random variable.</returns>
        public static Variable<char> GetItem(Variable<string> str, Variable<int> position)
        {
            var substr = Substring(str, position, 1);
            substr.AddAttribute(new DoNotInfer());
            return Variable<char>.Factor(Factor.Single, substr);
        }

#endregion

#if SUPPRESS_AMBIGUOUS_REFERENCE_WARNINGS
#pragma warning restore 419
#endif

#endregion Factor convenience methods

#region Constraint convenience methods

        /****************** Convenience methods for various commonly used constraints*********************/

        /// <summary>
        /// Constrains a boolean variable to be true.
        /// </summary>
        /// <param name="v">The variable to constrain to be true</param>
        public static void ConstrainTrue(Variable<bool> v)
        {
            ConstrainEqual(true, v);
        }

        /// <summary>
        /// Constrains a boolean variable to be false.
        /// </summary>
        /// <param name="v">The variable to constrain to be false</param>
        public static void ConstrainFalse(Variable<bool> v)
        {
            ConstrainEqual(false, v);
        }


        /// <summary>
        /// Constrains a double variable to be positive.
        /// </summary>
        /// <param name="v">The variable to constrain to be positive</param>
        public static void ConstrainPositive(Variable<double> v)
        {
            ConstrainTrue(IsPositive(v).Attrib(new DoNotInfer()));
        }

        /// <summary>
        /// Constrains a double variable to be between two limits.
        /// </summary>
        /// <param name="x">The variable to constrain</param>
        /// <param name="lowerBound">The lower limit</param>
        /// <param name="upperBound">The upper limit</param>
        public static void ConstrainBetween(Variable<double> x, Variable<double> lowerBound, Variable<double> upperBound)
        {
            ConstrainTrue(IsBetween(x, lowerBound, upperBound));
        }

        /// <summary>
        /// Constrains two variables to be equal.
        /// </summary>
        /// <typeparam name="T">The type of the variables</typeparam>
        /// <param name="a">The first variable</param>
        /// <param name="b">The second variable</param>
        public static void ConstrainEqual<T>(Variable<T> a, Variable<T> b)
        {
            Constrain(Factors.Constrain.Equal<T>, a, b);
        }

        /// <summary>
        /// Constrains a variable to equal a constant value.
        /// </summary>
        /// <typeparam name="T">The type of the variable</typeparam>
        /// <param name="a">The constant value</param>
        /// <param name="b">The variable</param>
        public static void ConstrainEqual<T>(T a, Variable<T> b)
        {
            ConstrainEqual(Constant(a), b);
        }

        /// <summary>
        /// Constrains a variable to equal a constant value.
        /// </summary>
        /// <typeparam name="T">The type of the variable</typeparam>
        /// <param name="a">The variable</param>
        /// <param name="b">The constant value</param>
        public static void ConstrainEqual<T>(Variable<T> a, T b)
        {
            ConstrainEqual(a, Constant(b));
        }

        /// <summary>
        /// Constrains a variable to be equal to a random sample from a distribution.
        /// </summary>
        /// <typeparam name="T">The variable type</typeparam>
        /// <typeparam name="TDist">The distribution type</typeparam>
        /// <param name="a">The variable to constrain</param>
        /// <param name="b">The distribution</param>
        public static void ConstrainEqualRandom<T, TDist>(Variable<T> a, Variable<TDist> b) where TDist : Sampleable<T>
        {
            Constrain(Factors.Constrain.EqualRandom<T, TDist>, a, b);
        }

        /// <summary>
        /// Constrains a value to be equal to a random sample from a distribution.
        /// </summary>
        /// <typeparam name="T">The variable type</typeparam>
        /// <typeparam name="TDist">The distribution type</typeparam>
        /// <param name="a">The variable to constrain</param>
        /// <param name="b">The distribution</param>
        public static void ConstrainEqualRandom<T, TDist>(T a, Variable<TDist> b) where TDist : Sampleable<T>
        {
            ConstrainEqualRandom<T, TDist>(Constant(a), b);
        }

        /// <summary>
        /// Constrains a variable to be equal to a random sample from a known distribution.
        /// </summary>
        /// <typeparam name="T">The variable type</typeparam>
        /// <typeparam name="TDist">The distribution type</typeparam>
        /// <param name="a">The variable to constrain</param>
        /// <param name="b">The distribution</param>
        public static void ConstrainEqualRandom<T, TDist>(Variable<T> a, TDist b) where TDist : Sampleable<T>
        {
            ConstrainEqualRandom<T, TDist>(a, Constant(b));
        }

#endregion Constraint convenience methods

#region Undirected factor convenience methods

        /// <summary>
        /// Adds a Potts factor between two boolean variables (max product only!).
        /// </summary>
        /// <remarks>
        /// This factor has the value of 1 if a==b and exp(-logCost) otherwise.</remarks>
        /// <param name="a">The first bool variable</param>
        /// <param name="b">The second bool variable</param>
        /// <param name="logCost">The negative log cost if the variables are not equal</param>
        public static void Potts(Variable<bool> a, Variable<bool> b, Variable<double> logCost)
        {
            Variable.Constrain(Undirected.Potts, a, b, logCost);
        }

        /// <summary>
        /// Adds a Potts factor between two int variables (max product only!).
        /// </summary>
        /// <remarks>
        /// This factor has the value of 1 if a==b and exp(-logCost) otherwise.</remarks>
        /// <param name="a">The first int variable</param>
        /// <param name="b">The second int variable</param>
        /// <param name="logCost">The negative log cost if the variables are not equal</param>
        public static void Potts(Variable<int> a, Variable<int> b, Variable<double> logCost)
        {
            Variable.Constrain(Undirected.Potts, a, b, logCost);
        }

        /// <summary>
        /// Adds a linear factor between two int variables (max product only!).
        /// </summary>
        /// <remarks>
        /// This factor has the value of exp( -|a-b|* logUnitCost ).</remarks>
        /// <param name="a">The first int variable</param>
        /// <param name="b">The second int variable</param>
        /// <param name="logUnitCost">The negative log cost per unit absolute different between the variables</param>
        public static void Linear(Variable<int> a, Variable<int> b, Variable<double> logUnitCost)
        {
            Variable.Constrain(Undirected.Linear, a, b, logUnitCost);
        }

        /// <summary>
        /// Adds a truncated linear factor between two int variables (max product only!).
        /// </summary>
        /// <remarks>
        /// This factor has the value of exp( - min( |a-b|* logUnitCost, maxCost) )).</remarks>
        /// <param name="a">The first int variable</param>
        /// <param name="b">The second int variable</param>
        /// <param name="logUnitCost">The negative log cost per unit absolute different between the variables</param>
        /// <param name="maxCost">The maximum negative log cost</param>
        public static void LinearTrunc(Variable<int> a, Variable<int> b, Variable<double> logUnitCost,
                                       Variable<double> maxCost)
        {
            Variable.Constrain(Undirected.LinearTrunc, a, b, logUnitCost, maxCost);
        }

#endregion

        //****************************** OPERATOR METHODS ********************************
        /// <summary>
        /// Operator to factor registry
        /// </summary>
        protected static readonly Dictionary<Operator, List<Delegate>> operatorFactorRegistry = new Dictionary<Operator, List<Delegate>>();

        static Variable()
        {
            RegisterOperatorFactor(Operator.Plus, new Func<double, double, double>(Factor.Plus));
            RegisterOperatorFactor(Operator.Minus, new Func<double, double, double>(Factors.Factor.Difference));
            RegisterOperatorFactor(Operator.Multiply, new Func<double, double, double>(Factors.Factor.Product));
            RegisterOperatorFactor(Operator.Divide, new Func<double, double, double>(Factors.Factor.Ratio));
            RegisterOperatorFactor(Operator.And, new Func<bool, bool, bool>(Factors.Factor.And));
            RegisterOperatorFactor(Operator.Or, new Func<bool, bool, bool>(Factors.Factor.Or));
            RegisterOperatorFactor(Operator.Not, new Func<bool, bool>(Factors.Factor.Not));
            RegisterOperatorFactor(Operator.Complement, new Func<bool, bool>(Factors.Factor.Not));
            RegisterOperatorFactor(Operator.Equal, new Func<bool, bool, bool>(Factors.Factor.AreEqual));
            RegisterOperatorFactor(Operator.Equal, new Func<int, int, bool>(Factors.Factor.AreEqual));
            RegisterOperatorFactor(Operator.Equal, new Func<Enum, Enum, bool>(EnumSupport.AreEqual));
            RegisterOperatorFactor(Operator.Equal, new Func<double, double, bool>(Factors.Factor.AreEqual));
            RegisterOperatorFactor(Operator.Plus, new Func<int, int, int>(Factors.Factor.Plus));
            RegisterOperatorFactor(Operator.Minus, new Func<int, int, int>(Factors.Factor.Difference));
            RegisterOperatorFactor(Operator.Multiply, new Func<int, int, int>(Factors.Factor.Product));
            RegisterOperatorFactor(Operator.GreaterThan, new Func<int, int, bool>(Factors.Factor.IsGreaterThan));
            RegisterOperatorFactor(Operator.GreaterThan, new Func<uint, uint, bool>(Factors.Factor.IsGreaterThan));
            //RegisterOperatorFactor(Operator.GreaterThan, new FactorMethod<bool, double, double>(Factors.Factor.IsGreaterThan));
            RegisterOperatorFactor(Operator.Plus, new Func<string, string, string>(Factors.Factor.Concat));
            RegisterOperatorFactor(Operator.Plus, new Func<string, char, string>(Factors.Factor.Concat));
            RegisterOperatorFactor(Operator.Plus, new Func<string, char, string>(Factors.Factor.Concat));
            RegisterOperatorFactor(Operator.Equal, new Func<string, string, bool>(Factors.Factor.AreEqual));
            RegisterOperatorFactor(Operator.Xor, new Func<double, double, double>(Math.Pow));
            RegisterOperatorFactor(Operator.Plus, new Func<Vector, Vector, Vector>(Factors.Factor.Plus));
        }

        /// <summary>
        /// Gets the number of parameters of a given operator.
        /// </summary>
        /// <param name="op">The operator.</param>
        /// <returns>The number of parameters of the operator.</returns>
        private static int GetOperatorArity(Operator op)
        {
            switch (op)
            {
                case Operator.Negative:
                    return 1;
                case Operator.Not:
                    return 1;
                case Operator.Complement:
                    return 1;
                default:
                    return 2; // Most of the operators are binary
            }
        }

        /// <summary>
        /// Registers a factor method against a particular operator.
        /// </summary>
        /// <param name="op">The operator to register against</param>
        /// <param name="factorMethod">The factor method to register</param>
        public static void RegisterOperatorFactor(Operator op, Delegate factorMethod)
        {
            Argument.CheckIfNotNull(factorMethod, "factorMethod");
            Argument.CheckIfValid(
                factorMethod.Method.GetParameters().Length == GetOperatorArity(op),
                "The number of parameters of the given method doesn't match the arity of the operator.");

            if (!operatorFactorRegistry.ContainsKey(op))
            {
                operatorFactorRegistry[op] = new List<Delegate>();
            }

            operatorFactorRegistry[op].Add(factorMethod);
        }

        /// <summary>
        /// Retrieves the factor method for a given operator and parameter types.
        /// </summary>
        /// <param name="op">The operator.</param>
        /// <param name="parameterTypes">The types of the parameters.</param>
        /// <returns>The retrieved factor method, or <see langword="null"/> if no suitable method found.</returns>
        public static Delegate LookupOperatorFactor(Operator op, params Type[] parameterTypes)
        {
            Argument.CheckIfNotNull(parameterTypes, "argumentTypes");
            Argument.CheckIfValid(
                parameterTypes.Length == GetOperatorArity(op),
                "The number of parameters doesn't match the arity of the operator.");

            if (operatorFactorRegistry.ContainsKey(op))
            {
                foreach (Delegate del in operatorFactorRegistry[op])
                {
                    var parameters = del.Method.GetParameters();
                    Debug.Assert(
                        parameters.Length == parameterTypes.Length,
                        "It should be impossible to register a method with wrong number of parameters.");

                    bool match = true;
                    for (int i = 0; i < parameterTypes.Length; ++i)
                    {
                        if (!parameters[i].ParameterType.IsAssignableFrom(parameterTypes[i]))
                        {
                            match = false;
                            break;
                        }
                    }

                    if (match)
                    {
                        if (del.Method.IsGenericMethod)
                        {
                            var genericDel = del.Method.GetGenericMethodDefinition();
                            var genericArguments = genericDel.GetGenericArguments();
                            var genericDelParameters = genericDel.GetParameters();

                            Type factorMethodType;
                            switch (parameterTypes.Length)
                            {
                                case 1:
                                    factorMethodType = typeof(Func<,>);
                                    break;
                                case 2:
                                    factorMethodType = typeof(Func<,,>);
                                    break;
                                case 3:
                                    factorMethodType = typeof(Func<,,,>);
                                    break;
                                default:
                                    throw new NotSupportedException("Operators with " + parameterTypes.Length + " parameters are not supported.");
                            }

                            var genericArgumentTypes = new Type[genericArguments.Length];
                            for (int i = 0; i < genericDelParameters.Length; ++i)
                            {
                                if (genericDelParameters[i].ParameterType.IsGenericParameter)
                                {
                                    int genericArgumentIndex = genericArguments.IndexOf(genericDelParameters[i].ParameterType);
                                    Debug.Assert(genericArgumentIndex != -1, "Must be found.");

                                    if (genericArgumentTypes[genericArgumentIndex] == null)
                                    {
                                        genericArgumentTypes[genericArgumentIndex] = parameterTypes[i];
                                    }
                                    else if (genericArgumentTypes[genericArgumentIndex] != parameterTypes[i])
                                    {
                                        // TODO: what should we do in this case?
                                        throw new NotImplementedException("Unsupported factor method signature.");
                                    }
                                }
                            }

                            var factorMethodTypeGenericArguments = new List<Type>();
                            factorMethodTypeGenericArguments.AddRange(parameterTypes);
                            factorMethodTypeGenericArguments.Add(del.Method.ReturnType);

                            Type delType = factorMethodType.MakeGenericType(factorMethodTypeGenericArguments.ToArray());
                            return Delegate.CreateDelegate(delType, genericDel.MakeGenericMethod(genericArgumentTypes));
                        }

                        return del;
                    }
                }
            }

            return null;
        }

        //****************************** CONDITION METHODS ********************************

        /// <summary>
        /// Opens a stochastic if statement, active when the argument is true.  
        /// </summary>
        /// <remarks>
        /// This method should be used as the argument to a using() statement,
        /// so that the if statement is automatically closed.  If this is not possible, 
        /// the returned IfBlock must be closed manually by calling CloseBlock().
        /// </remarks>
        /// <param name="b">The condition of the if block</param>
        /// <returns>An IfBlock object which must be closed before inference is performed.</returns>
        public static IfBlock If(Variable<bool> b)
        {
            return new IfBlock(b, true);
        }

        /// <summary>
        /// Opens a stochastic if statement, active when the argument is false.  
        /// </summary>
        /// <remarks>
        /// This method should be used as the argument to a using() statement,
        /// so that the if statement is automatically closed.  If this is not possible, 
        /// the returned IfBlock must be closed manually by calling CloseBlock().
        /// </remarks>
        /// <param name="b">The condition of the if block</param>
        /// <returns>An IfBlock object which must be closed before inference is performed.</returns>
        public static IfBlock IfNot(Variable<bool> b)
        {
            return new IfBlock(b, false);
        }

        /// <summary>
        /// Opens a stochastic case statement, active when the integer argument has the specified value.  
        /// </summary>
        /// <remarks>
        /// This method should be used as the argument to a using() statement,
        /// so that the if statement is automatically closed.  If this is not possible, 
        /// the returned CaseBlock must be closed manually by calling CloseBlock().
        /// </remarks>
        /// <param name="i">The condition of the case block</param>
        /// <param name="value">The value of the condition for which the block is active</param>
        /// <returns>A CaseBlock object which must be closed before inference is performed.</returns>
        public static CaseBlock Case(Variable<int> i, int value)
        {
            return new CaseBlock(i, value);
        }

        /// <summary>
        /// Close blocks in order to recover from exceptions
        /// </summary>
        public static void CloseAllBlocks()
        {
            StatementBlock.CloseAllBlocks();
        }

        /// <summary>
        /// Opens a stochastic switch statement using the specified condition variable.  This is equivalent
        /// to creating a set of identical Variable.Case() statements for each value of i.  Within a switch block,
        /// you can use the variable i as an array index.
        /// </summary>
        /// <remarks>
        /// This method should be used as the argument to a using() statement,
        /// so that the if statement is automatically closed.  If this is not possible, 
        /// the returned SwitchBlock must be closed manually by calling CloseBlock().
        /// </remarks>
        /// <param name="i">The condition of the switch block</param>
        /// <returns>A SwitchBlock object which must be closed before inference is performed.</returns>
        public static SwitchBlock Switch(Variable<int> i)
        {
            Range range = i.GetValueRange();
            return new SwitchBlock(i, range);
        }

        /// <summary>
        /// Enumeration over supported operators.
        /// </summary>
        public enum Operator
        {
            Plus,
            Minus,
            Multiply,
            Divide,
            Modulus,
            Negative,
            Not,
            And,
            Or,
            Xor,
            Complement,
            Equal,
            NotEqual,
            LeftShift,
            RightShift,
            GreaterThan,
            LessThan,
            GreaterThanOrEqual,
            LessThanOrEqual
        };
    }

    /// <summary>
    /// A typed variable in a model.
    /// </summary>
    /// <typeparam name="T">The domain of the variable.</typeparam>
    public class Variable<T> : Variable, IModelExpression<T>, SettableTo<Variable<T>>, ICloneable, HasObservedValue
    {
        /// <summary>
        /// </summary>
        protected bool isReadOnly;

        /// <summary>
        /// Read only property
        /// </summary>
        public override bool IsReadOnly
        {
            get { return isReadOnly || (ArrayVariable != null && ((Variable)ArrayVariable).IsReadOnly); }
            set
            {
                if (IsBase)
                {
                    if (isReadOnly != value)
                    {
                        isReadOnly = value;
                        declaration = null;
                        InferenceEngine.InvalidateAllEngines(this);
                    }
                }
                else if (IsArrayElement)
                {
                    throw new InvalidOperationException(this + " is an array element.  To set IsReadOnly of the array, use " + this.array + ".IsReadOnly");
                }
                else
                {
                    throw new NotImplementedException();
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        protected T observedValue;

        /// <summary>
        /// Observed value property
        /// </summary>
        object HasObservedValue.ObservedValue
        {
            get { return ObservedValue; }
            set { ObservedValue = (T)value; }
        }

        /// <summary>
        /// Observed value property
        /// </summary>
        public T ObservedValue
        {
            get
            {
                if (IsBase)
                {
                    if (isObserved) return observedValue;
                    else throw new InvalidOperationException("No ObservedValue has been set on " + this);
                }
                else if (IsArrayElement)
                {
                    throw new InvalidOperationException(this + " is an array element.  To get the ObservedValue of the array, use " + this.array + ".ObservedValue");
                }
                else
                {
                    throw new NotImplementedException();
                }
            }
            set
            {
                if (IsReadOnly) throw new InvalidOperationException(this + " is marked read-only.  Cannot change ObservedValue.");
                if (IsBase)
                {
                    bool wasObserved = isObserved;
                    observedValue = value;
                    isObserved = true;
                    if (!wasObserved)
                    {
                        declaration = null;
                        InferenceEngine.InvalidateAllEngines(this);
                    }
                    else InferenceEngine.ObservedValueChanged(this);
                }
                else if (IsArrayElement)
                {
                    throw new InvalidOperationException(this + " is an array element.  To set the value of the array, use " + this.array + ".ObservedValue");
                }
                else
                {
                    throw new NotImplementedException();
                }
            }
        }

        /// <summary>
        /// Provides implicit conversion from .NET instances to constant Infer.NET variables.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static implicit operator Variable<T>(T value)
        {
            return Variable.Constant<T>(value);
        }

        /// <summary>
        /// Clear the observed value.
        /// </summary>
        /// <remarks>Calling this method sets the <see cref="IsReadOnly"/> and <see cref="Variable.IsObserved"/> flags to false.
        /// This cannot be called for array items. </remarks>
        public void ClearObservedValue()
        {
            if (IsBase)
            {
                if (IsObserved)
                {
                    isReadOnly = false;
                    isObserved = false;
                    observedValue = default(T);
                    declaration = null;
                    InferenceEngine.InvalidateAllEngines(this);
                }
            }
            else if (IsArrayElement)
            {
                throw new InvalidOperationException(this + " is an array element.  To clear the value of the array, use " + this.array + ".ClearObservedValue()");
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        internal Variable()
            : this(StatementBlock.GetOpenBlocks())
        {
        }

        /// <summary>
        /// Global counter used to generate variable names.
        /// </summary>
        private static readonly GlobalCounter globalCounter = new GlobalCounter();

        internal Variable(IEnumerable<IStatementBlock> containers)
        {
            base.Name = $"v{StringUtil.TypeToString(typeof(T))}{globalCounter.GetNext()}";
            this.containers = new List<IStatementBlock>(containers);
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        protected Variable(Variable<T> that)
            : this(that.containers)
        {
            this.attributes = that.attributes;
            // when cloning, we must maintain the type (random/given/constant) of the variable.
            isObserved = that.isObserved;
            isReadOnly = that.IsReadOnly;
        }

        internal Variable(Range range)
            : this()
        {
            loopRange = range;
        }

        /// <summary>
        /// Clone this variable
        /// </summary>
        /// <returns></returns>
        public virtual object Clone()
        {
            return new Variable<T>(this);
        }

        /// <summary>
        /// Sets the value of a random variable.  Should only be invoked on variables created using New() which 
        /// do not yet have a value.
        /// </summary>
        /// <param name="variable">A variable whose definition will be consumed by <c>this</c> and no longer available for use</param>
        /// <remarks>
        /// <paramref name="variable"/> must have exactly the same set of ranges as <c>this</c>.
        /// </remarks>
        public void SetTo(Variable<T> variable)
        {
            AddAttributes(variable.GetAttributes<ICompilerAttribute>());
            if (variable.initialiseTo != null) initialiseTo = variable.initialiseTo;
            if (variable.initialiseBackwardTo != null)
                initialiseBackwardTo = variable.initialiseBackwardTo;
            // copy all conditional definitions from the given variable to this.
            List<ConditionBlock> currentConditions = ConditionBlock.GetOpenBlocks<ConditionBlock>();
            bool foundDef = false;
            foreach (MethodInvoke mi in variable.GetDefinitionsMadeWithin(currentConditions))
            {
                foundDef = true;
                SetTo(mi);
            }
            // copying of item definitions is handled by VariableArrayBase.SetTo()
            if (!foundDef)
            {
                // The rhs variable does not have a definition made within this context.  This can happen if the variable was created
                // with Variable.New, if the variable is an array element, or is being used inside a condition block.  In these cases, the lhs should be
                // defined as a copy of the rhs.
                //throw new InvalidOperationException(variable+" was not defined in this condition block.");
                MethodInvoke mi = new MethodInvoke(new Func<T, T>(Factors.Clone.Copy<T>).Method, variable);
                SetTo(mi);
                return;
            }
            // variable must not have any constraints
            if (variable.constraints.Count > 0) throw new InvalidOperationException(variable + " has constraints. The argument of SetTo must not have constraints.");
            if (variable.childFactors.Count > 0) throw new InvalidOperationException(variable + " has child factors. The argument of SetTo must not have child factors.");
            // if we did not use Factor.Copy, the variable must have exactly the same set of ranges as this.
            CheckCompatibleIndexing(variable);
            // remove the definitions of the given variable to ensure it is not used again.
            variable.RemoveDefinitions();
            // give the variable a special definition which throws an error (in ModelBuilder) if the variable is used.
            // use an empty container set to ensure this variable is not built accidentally.
            variable.SetTo(new MethodInvoke(new List<IStatementBlock>(), new Func<T>(RemovedBySetTo).Method));
            // The following is valid but adds unnecessary code to the MSL.
            //variable.SetTo(Factors.Factor.Copy, this);
        }

        /// <summary>
        /// A special factor attached to variables whose definition was consumed by SetTo().
        /// </summary>
        /// <returns></returns>
        internal static T RemovedBySetTo()
        {
            throw new NotSupportedException("This variable's definition was consumed by variable.SetTo()");
        }

        internal void SetTo(Func<T> factor)
        {
            SetTo(factor.Method);
        }

        internal void SetTo<T1>(Func<T1, T> factor, IModelExpression<T1> arg1)
        {
            SetTo(factor.Method, arg1);
        }

        internal void SetTo<T1, T2>(Func<T1, T2, T> factor,
                                    IModelExpression<T1> arg1, IModelExpression<T2> arg2)
        {
            SetTo(factor.Method, arg1, arg2);
        }

        internal void SetTo<T1, T2, T3>(Func<T1, T2, T3, T> factor,
                                        IModelExpression<T1> arg1, IModelExpression<T2> arg2, IModelExpression<T3> arg3)
        {
            SetTo(factor.Method, arg1, arg2, arg3);
        }

        internal void SetTo<T1, T2, T3, T4>(Func<T1, T2, T3, T4, T> factor,
                                        IModelExpression<T1> arg1, IModelExpression<T2> arg2, IModelExpression<T3> arg3, IModelExpression<T4> arg4)
        {
            SetTo(factor.Method, arg1, arg2, arg3, arg4);
        }

        internal void SetTo<T1, T2>(FuncOut<T1, T2, T> factor,
                                    IModelExpression<T1> arg1, Variable<T2> arg2)
        {
            var methodInvoke = new MethodInvoke(factor.Method, new IModelExpression[] { arg1, arg2 });
            arg2.SetTo(methodInvoke);
            SetTo(methodInvoke);
        }

        internal void SetTo<T1, T2, T3>(FuncOut<T1, T2, T3, T> factor,
                                        IModelExpression<T1> arg1, IModelExpression<T2> arg2, Variable<T3> arg3)
        {
            var methodInvoke = new MethodInvoke(factor.Method, new IModelExpression[] { arg1, arg2, arg3 });
            arg3.SetTo(methodInvoke);
            SetTo(methodInvoke);
        }

        /// <summary>
        /// Set the parent factor of the variable.
        /// </summary>
        internal void SetTo(MethodInfo methodInfo, params IModelExpression[] args)
        {
            if (Variable.AutoNaming)
            {
                if (methodInfo.Name == "Random")
                {
                    Name = ((Variable)args[0]).StripIndexers().ToString();
                }
                else
                {
                    IEnumerable<Variable> strippedArgs = args.Select(v => ((Variable)v).StripIndexers());
                    Name = methodInfo.Name + "(" + StringUtil.CollectionToString(strippedArgs, ",") + ")";
                }
            }
            if (!methodInfo.IsStatic) throw new ArgumentException("factor method is not static");
            SetTo(new MethodInvoke(methodInfo, args));
        }

        /// <summary>
        /// Set the parent factor of the variable.
        /// </summary>
        /// <param name="methodInvoke">The parent factor and its arguments.</param>
        internal void SetTo(MethodInvoke methodInvoke)
        {
            methodInvoke.returnValue = this;
            //if (!ListComparer<IStatementBlock>.EqualLists(methodInvoke.Containers, Containers)) {
            //  throw new InvalidOperationException("A variable created in one block context cannot be assigned to a variable in another block context.");
            //}
            SetDefinition(methodInvoke);
            InferenceEngine.InvalidateAllEngines(this);
            // if any of the arguments are indexed, the result is also indexed with those indices,
            // excluding ranges open in ForEach blocks.
            Set<Range> indices = methodInvoke.GetLocalRangeSet();
            if (indices.Count > 0)
            {
                if (!MethodInvoke.IsIndexedByAll(this, indices))
                {
                    // This can only happen when called by Variable.Factor, in which case "this" is
                    // a fresh variable.
                    if (!IsBase) throw new InvalidOperationException("Array elements cannot have different definition types");
                    CreateVariableArrayFromItem(this, new List<Range>(indices));
                    //throw new InferCompilerException("The indices on the left-hand side (" + methodInvoke.returnValue + ") do not include all ranges on the right-hand side: " + Range.ToString(ranges));
                }
            }
            foreach (IModelExpression arg in methodInvoke.args)
            {
                ForEachBaseVariable(arg, delegate (Variable v)
                {
                    InferenceEngine.InvalidateAllEngines(v);
                    v.childFactors.Add(methodInvoke);
                });
            }
        }

        /// <summary>
        /// Creates a variable array with the given ranges and modify item to be an item of that array, keeping the same definition.
        /// </summary>
        /// <param name="item"></param>
        /// <param name="ranges"></param>
        /// <returns></returns>
        /// <remarks>
        /// Only the definition and containers of item are used.  item.array and item.indices are ignored.
        /// </remarks>
        internal static IVariableArray CreateVariableArrayFromItem(Variable<T> item, IList<Range> ranges)
        {
            if (ranges.Count == 0) throw new ArgumentException("range list is empty");
            if (item.IsObserved || item.IsReadOnly) return null; // throw new InvalidOperationException(item + " is already observed.  Cannot apply ForEach to it.");
            if (ranges.Count > 1)
            {
                List<Range> headRanges;
                List<Range> tailRanges = new List<Range>(ranges.Skip(1, out headRanges));
                Range headRange = headRanges[0];
                ForEachBlock block = Variable.ForEach(headRange);
                IVariableArray array = CreateVariableArrayFromItem(item, tailRanges);
                block.CloseBlock();
                // return Variable<T2>.CreateVariableArrayFromItem(array, headRanges);
                // where T2 is the domain type of array
                Type type = array.GetType();
                while (!type.IsGenericType || !type.GetGenericTypeDefinition().Equals(typeof(Variable<>))) type = type.BaseType;
                MethodInfo method = type.GetMethod("CreateVariableArrayFromItem", BindingFlags.NonPublic | BindingFlags.Static);
                return (IVariableArray)Util.Invoke(method, null, array, headRanges);
            }
            // check for jagged dependencies
            Set<Range> previousRanges = new Set<Range>();
            bool isJagged = false;
            foreach (Range range in ranges)
            {
                Models.MethodInvoke.ForEachRange(range.Size, delegate (Range r) { if (previousRanges.Contains(r)) isJagged = true; });
                if (isJagged) break;
                previousRanges.Add(range);
            }
            if (isJagged)
            {
                List<Range> headRanges;
                List<Range> tailRanges = new List<Range>(ranges.Skip(previousRanges.Count, out headRanges));
                // open all the headRanges
                List<ForEachBlock> blocks = new List<ForEachBlock>();
                foreach (Range r in headRanges)
                {
                    blocks.Add(Variable.ForEach(r));
                }
                IVariableArray array = CreateVariableArrayFromItem(item, tailRanges);
                // close all the headRanges
                blocks.Reverse();
                foreach (ForEachBlock block in blocks)
                {
                    block.CloseBlock();
                }
                // return Variable<T2>.CreateVariableArrayFromItem(array, headRanges);
                // where T2 is the domain type of array
                Type type = array.GetType();
                while (!type.IsGenericType || !type.GetGenericTypeDefinition().Equals(typeof(Variable<>))) type = type.BaseType;
                MethodInfo method = type.GetMethod("CreateVariableArrayFromItem", BindingFlags.NonPublic | BindingFlags.Static);
                return (IVariableArray)Util.Invoke(method, null, array, headRanges);
            }

            Variable<T> itemPrototype = (Variable<T>)item.Clone();

            VariableArray<T> variableArray = new VariableArray<T>(itemPrototype, ranges[0]);
            variableArray.timestamp = item.timestamp;
            if (Variable.AutoNaming)
            {
                variableArray.Name = item.StripIndexers().ToString();
            }
            item.MakeItem(variableArray, ranges[0]);

            return variableArray;
        }

        /// <summary>
        /// Modify the variable to be an element of an array, keeping the same definition.
        /// </summary>
        /// <param name="varArray"></param>
        /// <param name="inds"></param>
        internal override void MakeItem(IVariableArray varArray, params IModelExpression[] inds)
        {
            ((Variable)varArray).AddAttributes(GetAttributes<ICompilerAttribute>());
            Set<Range> openRanges = new Set<Range>();
            foreach (HasRange fb in StatementBlock.EnumerateOpenBlocks<HasRange>())
            {
                openRanges.Add(fb.Range);
            }
            Models.MethodInvoke.ForEachRange(varArray, openRanges.Add);
            containers = ((Variable)varArray).Containers;
            foreach (IModelExpression ind in inds)
            {
                if (ind is Range range)
                {
                    Models.MethodInvoke.ForEachRange(range.Size,
                                                     delegate (Range r)
                                                         {
                                                             if (!openRanges.Contains(r))
                                                                 throw new InvalidOperationException("Range '" + range + "' depends on range '" + r + "', but range '" + r +
                                                                                                     "' is not open in a ForEach block.  Insert 'Variable.ForEach(" + r +
                                                                                                     ")' around the statement with '.ForEach(" + range + ")'.");
                                                         });
                    continue;
                }
                else if (ind is Variable<int> variable)
                {
                    containers = MergeContainers(containers, variable.Containers);
                    continue;
                }
                else
                {
                    throw new ArgumentException("Can only index by ranges or integer variables, not " + ind);
                }
            }
            if (varArray is HasItemVariables irva)
            {
                if (irva.GetItemsUntyped().ContainsKey(inds)) throw new InvalidOperationException("Duplicate indexed variable created.");
                irva.GetItemsUntyped()[inds] = this;
            }
            this.indices = new List<IModelExpression>(inds);
            // set array last since this finally changes the variable to IsItem
            this.array = varArray;
        }

        /// <summary>
        /// Helps build class declarations
        /// </summary>
        internal static CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// Cache of GetDeclaration.  Stores the name and type of the variable.
        /// </summary>
        /// <remarks>
        /// Always null for derived variables.
        /// The declaration must be cached because we later use reference equality between declarations.
        /// </remarks>
        protected object declaration;

        internal override object GetDeclaration()
        {
            if (declaration == null)
            {
                if (!IsBase) throw new NotSupportedException("A derived variable does not have a declaration");
                if (!IsObserved)
                {
                    declaration = Builder.VarDecl(NameInGeneratedCode, typeof(T));
                }
                else if (!IsReadOnly)
                {
                    declaration = Builder.Param(NameInGeneratedCode, typeof(T));
                }
                else
                {
                    // Find the actual type of the constant
                    var tp = ReferenceEquals(ObservedValue, null) ? typeof(T) : ObservedValue.GetType();
                    // Get the public version of the type, if necessary
                    tp = Quoter.GetPublicType(tp);
                    declaration = Builder.VarDecl(NameInGeneratedCode, tp);
                }
            }
            return declaration;
        }

        /// <summary>
        /// Change the declaration of the variable if none has been set yet.
        /// </summary>
        /// <param name="declaration"></param>
        /// <returns>True if the declaration was changed, false otherwise.</returns>
        internal bool SetDeclaration(object declaration)
        {
            if (this.declaration != null && this.declaration != declaration) return false;
            this.declaration = declaration;
            return true;
        }

        /// <summary>
        /// Name
        /// </summary>
        public override string Name
        {
            get
            {
                if (IsArrayElement)
                {
                    return ToString();
                }
                return base.Name;
            }
            set
            {
                //if (IsItem) throw new InvalidOperationException("Cannot set the name of array element "+Name+" , instead set the name of the array "+array.Name+".");
                base.Name = value;
                if (declaration != null)
                {
                    declaration = null;
                    InferenceEngine.InvalidateAllEngines(this);
                }
            }
        }

        string IModelExpression.Name
        {
            get { return NameInGeneratedCode; }
        }

        /// <summary>
        /// Set the name of the variable.
        /// </summary>
        /// <param name="name"></param>
        /// <returns><c>this</c></returns>
        public Variable<T> Named(string name)
        {
            this.Name = name;
            return this;
        }

        /// <summary>
        /// Inline method for adding an attribute to a variable.  This method
        /// returns the variable object, so that is can be used in an inline expression.
        /// e.g. Variable.GaussianFromMeanAndVariance(0,1).Attrib(new MyAttribute());
        /// </summary>
        /// <param name="attr">The attribute to add</param>
        /// <returns>The variable object</returns>
        public new Variable<T> Attrib(ICompilerAttribute attr)
        {
            base.Attrib(attr);
            return this;
        }

        /// <summary>
        /// Helper to add a query type attribute to this variable.
        /// </summary>
        /// <param name="queryType">The query type to use to create the attribute</param>
        public Variable<T> Attrib(QueryType queryType)
        {
            base.Attrib(new QueryTypeCompilerAttribute(queryType));
            return this;
        }

        /// <summary>
        /// Provide a marginal distribution to initialize inference.
        /// </summary>
        /// <typeparam name="TDist"></typeparam>
        /// <param name="initialMarginal"></param>
        /// <returns><c>this</c></returns>
        /// <remarks>Only relevant for iterative algorithms.  May be ignored by some inference algorithms.</remarks>
        public Variable<T> InitialiseTo<TDist>(TDist initialMarginal)
            where TDist : IDistribution<T>
        {
            return InitialiseTo<TDist>(Constant(initialMarginal));
        }

        /// <summary>
        /// Provide a marginal distribution to initialize inference.
        /// </summary>
        /// <typeparam name="TDist"></typeparam>
        /// <param name="initialMarginal"></param>
        /// <returns><c>this</c></returns>
        /// <remarks>Only relevant for iterative algorithms.  May be ignored by some inference algorithms.</remarks>
        public Variable<T> InitialiseTo<TDist>(Variable<TDist> initialMarginal)
            where TDist : IDistribution<T>
        {
            return InitialiseTo((IModelExpression<TDist>)initialMarginal);
        }

        /// <summary>
        /// Set the initialiseTo field if it is valid to do so
        /// </summary>
        /// <typeparam name="TDist"></typeparam>
        /// <param name="initialMarginal"></param>
        /// <returns></returns>
        protected Variable<T> InitialiseTo<TDist>(IModelExpression<TDist> initialMarginal)
            where TDist : IDistribution<T>
        {
            if (!AllIndicesAreRanges())
                throw new InvalidOperationException(this + ".InitialiseTo is not allowed since the indices are not Ranges");
            initialiseTo = initialMarginal;
            InferenceEngine.InvalidateAllEngines(this);
            return this;
        }

        /// <summary>
        /// Provide a backward distribution to initialize inference.
        /// </summary>
        /// <typeparam name="TDist"></typeparam>
        /// <param name="initialBackward"></param>
        /// <returns><c>this</c></returns>
        /// <remarks>Only relevant for iterative algorithms.  May be ignored by some inference algorithms.</remarks>
        public Variable<T> InitialiseBackwardTo<TDist>(TDist initialBackward)
            where TDist : IDistribution<T>
        {
            return InitialiseBackwardTo<TDist>(Constant(initialBackward));
        }

        /// <summary>
        /// Provide a backward distribution to initialize inference.
        /// </summary>
        /// <typeparam name="TDist"></typeparam>
        /// <param name="initialBackward"></param>
        /// <returns><c>this</c></returns>
        /// <remarks>Only relevant for iterative algorithms.  May be ignored by some inference algorithms.</remarks>
        public Variable<T> InitialiseBackwardTo<TDist>(Variable<TDist> initialBackward)
            where TDist : IDistribution<T>
        {
            return InitialiseBackwardTo((IModelExpression<TDist>)initialBackward);
        }

        /// <summary>
        /// Set the initialiseBackwardTo field if it is valid to do so
        /// </summary>
        /// <typeparam name="TDist"></typeparam>
        /// <param name="initialBackward"></param>
        /// <returns></returns>
        protected Variable<T> InitialiseBackwardTo<TDist>(IModelExpression<TDist> initialBackward)
            where TDist : IDistribution<T>
        {
            if (!AllIndicesAreRanges())
                throw new InvalidOperationException(this + ".InitialiseBackwardTo is not allowed since the indices are not Ranges");
            initialiseBackwardTo = initialBackward;
            InferenceEngine.InvalidateAllEngines(this);
            return this;
        }

        /// <summary>
        /// Add a MarginalPrototype attribute
        /// </summary>
        /// <typeparam name="TDist">A distribution type</typeparam>
        /// <param name="prototype">An expression for the marginal prototype</param>
        public void SetMarginalPrototype<TDist>(IModelExpression<TDist> prototype)
            where TDist : IDistribution<T>
        {
            AddAttribute(new MarginalPrototype(null) { prototypeExpression = prototype.GetExpression() });
        }

        /// <summary>
        /// Returns true if this is not an array element or all indices are ranges
        /// </summary>
        /// <returns></returns>
        private bool AllIndicesAreRanges()
        {
            if (IsArrayElement)
            {
                // check that all indices are Ranges
                Variable parent = this;
                while (parent.IsArrayElement)
                {
                    foreach (IModelExpression expr in parent.indices)
                    {
                        if (!(expr is Range))
                            return false;
                    }
                    parent = (Variable)parent.ArrayVariable;
                }
            }
            return true;
        }

        /// <summary>
        /// Create multiple variables with the same definition.
        /// </summary>
        /// <param name="range">The desired range.</param>
        /// <returns><c>this</c>, modified to range over the newly created items.</returns>
        public new Variable<T> ForEach(Range range)
        {
            return ForEach(new Range[] { range });
        }

        /// <summary>
        /// Create multiple variables with the same definition.
        /// </summary>
        /// <param name="ranges">The desired ranges.</param>
        /// <returns><c>this</c>, modified to range over the newly created items.</returns>
        /// <remarks>
        /// <list type="bullet"><item>
        /// <c>Variable.Bernoulli(0.3).ForEach(r)</c> returns a VariableArray&lt;bool&gt;
        /// </item><item>
        /// <c>Variable.Bernoulli(a[r1]).ForEach(r2)</c> returns a VariableArray2D&lt;bool&gt;
        /// or VariableArray&lt;VariableArray&lt;bool&gt;,bool[][]&gt; depending on whether the
        /// size of r2 depends on r1.
        /// </item><item>
        /// <c>Variable.Bernoulli(a[r1][r2]).ForEach(r3)</c> returns a 
        /// VariableArray&lt;VariableArray2D&lt;bool&gt;,bool[][,]&gt; 
        /// or 3-deep jagged VariableArray depending on whether the size of r3 depends on r2.
        /// </item></list>
        /// </remarks>
        public Variable<T> ForEach(params Range[] ranges)
        {
            // mutate the variable to become an item variable of a new array with no definition.
            if (ranges.Length == 0) throw new ArgumentException("range list is empty");
            List<Range> fullRanges = new List<Range>();
            MethodInvoke.ForEachRange(this, fullRanges.Add);
            foreach (Range r in ranges)
            {
                if (fullRanges.Contains(r)) throw new ArgumentException("ForEach is not needed on range '" + r + "' since the expression is already indexed by this range");
                fullRanges.Add(r);
            }
            foreach (ForEachBlock fb in GetContainers<ForEachBlock>())
            {
                if (fullRanges.Contains(fb.Range)) throw new InvalidOperationException("Range " + fb.Range + " is already open in a ForEach block");
            }
            CreateVariableArrayFromItem(this, fullRanges);
            return this;
        }

        /// <summary>
        /// Create a 1D random variable array with a specified size.
        /// </summary>
        /// <param name="r">
        /// A <c>Range</c> object that specifies the array length. 
        /// </param>
        /// <returns>
        /// Returns a <c>VariableArray</c> object whose size is specified by <paramref name="r"/>.
        /// </returns>
        public static VariableArray<T> Array(Range r)
        {
            return new VariableArray<T>(r);
        }

        /// <summary>
        /// Creates a random variable with the specified prior distribution
        /// </summary>
        /// <typeparam name="TDist">The type of the distribution</typeparam>
        /// <param name="dist">The distribution to use</param>
        /// <returns></returns>
        public static Variable<T> Random<TDist>(Variable<TDist> dist) where TDist : IDistribution<T> //, Sampleable<T>
        {
            return FactorUntyped(new Func<Sampleable<T>, T>(Factors.Factor.Random<T>).Method, (Variable)dist);
        }

        /// <summary>
        /// Creates a random variable from a factor method with no arguments.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <returns></returns>
        public static Variable<T> Factor(Func<T> factorDelegate)
        {
            return FactorUntyped(factorDelegate.Method);
        }

        /// <summary>
        /// Creates a random variable from a factor method with one argument.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1"></param>
        /// <returns></returns>
        public static Variable<T> Factor<T1>(Func<T1, T> factorDelegate, Variable<T1> arg1)
        {
            return FactorUntyped(factorDelegate.Method, arg1);
        }

        /// <summary>
        /// Creates a random variable from a factor method with one argument.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1"></param>
        /// <returns></returns>
        public static Variable<T> Factor<T1>(Func<T1, T> factorDelegate, T1 arg1)
        {
            return FactorUntyped(factorDelegate.Method, Constant(arg1));
        }


        /// <summary>
        /// Creates a random variable from a factor method with two arguments.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1"></param>
        /// <param name="arg2"></param>
        /// <returns></returns>
        public static Variable<T> Factor<T1, T2>(Func<T1, T2, T> factorDelegate,
                                                 Variable<T1> arg1, Variable<T2> arg2)
        {
            return FactorUntyped(factorDelegate.Method, arg1, arg2);
        }

        /// <summary>
        /// Creates a random variable from a factor method with two arguments.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1"></param>
        /// <param name="arg2"></param>
        /// <returns></returns>
        public static Variable<T> Factor<T1, T2>(FuncOut<T1, T2, T> factorDelegate,
                                                 Variable<T1> arg1, Variable<T2> arg2)
        {
            Variable<T> var = new Variable<T>();
            var.SetTo(factorDelegate, arg1, arg2);
            return var;
        }

        /// <summary>
        /// Creates a random variable from a factor method with two arguments.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1">Fixed first argument</param>
        /// <param name="arg2">Variable second argument</param>
        /// <returns></returns>
        public static Variable<T> Factor<T1, T2>(Func<T1, T2, T> factorDelegate,
                                                 T1 arg1, Variable<T2> arg2)
        {
            return FactorUntyped(factorDelegate.Method, Constant(arg1), arg2);
        }

        /// <summary>
        /// Creates a random variable from a factor method with two arguments.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1">Variable first argument</param>
        /// <param name="arg2">Fixed second argument</param>
        /// <returns></returns>
        public static Variable<T> Factor<T1, T2>(Func<T1, T2, T> factorDelegate,
                                                 Variable<T1> arg1, T2 arg2)
        {
            return FactorUntyped(factorDelegate.Method, arg1, Constant(arg2));
        }

        /// <summary>
        /// Creates a random variable from a factor method with two arguments.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1">Variable first argument</param>
        /// <param name="arg2">Variable second argument</param>
        /// <returns></returns>
        public static Variable<T> Factor<T1, T2>(Func<T1, T2, T> factorDelegate,
                                                 T1 arg1, T2 arg2)
        {
            return FactorUntyped(factorDelegate.Method, Constant(arg1), Constant(arg2));
        }

        /// <summary>
        /// Creates a random variable from a factor method with three arguments.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1">Variable first argument</param>
        /// <param name="arg2">Variable second argument</param>
        /// <param name="arg3">Variable third argument</param>
        /// <returns></returns>
        public static Variable<T> Factor<T1, T2, T3>(Func<T1, T2, T3, T> factorDelegate,
                                                     Variable<T1> arg1, Variable<T2> arg2, Variable<T3> arg3)
        {
            return FactorUntyped(factorDelegate.Method, arg1, arg2, arg3);
        }

        /// <summary>
        /// Creates a random variable from a factor method with three arguments.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1">Fixed first argument</param>
        /// <param name="arg2">Variable second argument</param>
        /// <param name="arg3">Variable third argument</param>
        /// <returns></returns>
        public static Variable<T> Factor<T1, T2, T3>(Func<T1, T2, T3, T> factorDelegate,
                                                     T1 arg1, Variable<T2> arg2, Variable<T3> arg3)
        {
            return FactorUntyped(factorDelegate.Method, Constant(arg1), arg2, arg3);
        }

        /// <summary>
        /// Creates a random variable from a factor method with three arguments.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1">Fixed first argument</param>
        /// <param name="arg2">Fixed second argument</param>
        /// <param name="arg3">Variable third argument</param>
        /// <returns></returns>
        public static Variable<T> Factor<T1, T2, T3>(Func<T1, T2, T3, T> factorDelegate,
                                                     T1 arg1, T2 arg2, Variable<T3> arg3)
        {
            return FactorUntyped(factorDelegate.Method, Constant(arg1), Constant(arg2), arg3);
        }

        /// <summary>
        /// Creates a random variable from a factor method with three arguments.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1">Fixed first argument</param>
        /// <param name="arg2">Variable second argument</param>
        /// <param name="arg3">Fixed third argument</param>
        /// <returns></returns>
        public static Variable<T> Factor<T1, T2, T3>(Func<T1, T2, T3, T> factorDelegate,
                                                     T1 arg1, Variable<T2> arg2, T3 arg3)
        {
            return FactorUntyped(factorDelegate.Method, Constant(arg1), arg2, Constant(arg3));
        }

        /// <summary>
        /// Creates a random variable from a factor method with three arguments.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1">Variable first argument</param>
        /// <param name="arg2">Fixed second argument</param>
        /// <param name="arg3">Variable third argument</param>
        /// <returns></returns>
        public static Variable<T> Factor<T1, T2, T3>(Func<T1, T2, T3, T> factorDelegate,
                                                     Variable<T1> arg1, T2 arg2, Variable<T3> arg3)
        {
            return FactorUntyped(factorDelegate.Method, arg1, Constant(arg2), arg3);
        }

        /// <summary>
        /// Creates a random variable from a factor method with three arguments.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1">Variable first argument</param>
        /// <param name="arg2">Variable second argument</param>
        /// <param name="arg3">Fixed third argument</param>
        /// <returns></returns>
        public static Variable<T> Factor<T1, T2, T3>(Func<T1, T2, T3, T> factorDelegate,
                                                     Variable<T1> arg1, Variable<T2> arg2, T3 arg3)
        {
            return FactorUntyped(factorDelegate.Method, arg1, arg2, Constant(arg3));
        }

        /// <summary>
        /// Creates a random variable from a factor method with three arguments.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1">Variable first argument</param>
        /// <param name="arg2">Fixed second argument</param>
        /// <param name="arg3">Fixed third argument</param>
        /// <returns></returns>
        public static Variable<T> Factor<T1, T2, T3>(Func<T1, T2, T3, T> factorDelegate,
                                                     Variable<T1> arg1, T2 arg2, T3 arg3)
        {
            return FactorUntyped(factorDelegate.Method, arg1, Constant(arg2), Constant(arg3));
        }

        /// <summary>
        /// Creates a random variable from a factor method with three arguments.
        /// </summary>
        /// <param name="factorDelegate">The method that represents the factor</param>
        /// <param name="arg1">Variable first argument</param>
        /// <param name="arg2">Fixed second argument</param>
        /// <param name="arg3">Fixed third argument</param>
        /// <returns></returns>
        public static Variable<T> Factor<T1, T2, T3>(Func<T1, T2, T3, T> factorDelegate,
                                                     T1 arg1, T2 arg2, T3 arg3)
        {
            return FactorUntyped(factorDelegate.Method, Constant(arg1), Constant(arg2), Constant(arg3));
        }

        /// <summary>
        /// Creates a random variable from a factor method with four arguments.
        /// </summary>
        /// <param name="factorDelegate">Factor delegate</param>
        /// <param name="arg1">Variable first argument</param>
        /// <param name="arg2">Variable second argument</param>
        /// <param name="arg3">Variable third argument</param>
        /// <param name="arg4">Variable fourth argument</param>
        /// <returns></returns>
        public static Variable<T> Factor<T1, T2, T3, T4>(Func<T1, T2, T3, T4, T> factorDelegate,
                                                         Variable<T1> arg1, Variable<T2> arg2, Variable<T3> arg3, Variable<T4> arg4)
        {
            return FactorUntyped(factorDelegate.Method, arg1, arg2, arg3, arg4);
        }

        /// <summary>
        /// Creates a random variable from a factor
        /// </summary>
        /// <param name="methodInfo">The method</param>
        /// <param name="args">The arguments</param>
        /// <returns></returns>
        public static Variable<T> FactorUntyped(MethodInfo methodInfo, params Variable[] args)
        {
            Variable<T> var = new Variable<T>();
            var.SetTo(methodInfo, args);
            return var;
        }

        /// <summary>
        /// Creates a random variable from an operator
        /// </summary>
        /// <param name="op">The operator</param>
        /// <param name="methodInfo">The factor method corresponding to the operator</param>
        /// <param name="args">The method arguments</param>
        /// <returns></returns>
        public static Variable<T> OperatorUntyped(Operator op, MethodInfo methodInfo, params Variable[] args)
        {
            Variable<T> var = FactorUntyped(methodInfo, args);
            var.GetDefinition().op = op;
            return var;
        }


        /*****************  Convenience methods for operator overloading ******************************/

        internal static Variable<TRet> OperatorFactor<TRet>(Operator op, Variable<T> arg1)
        {
            Delegate del = LookupOperatorFactor(op, typeof(T));
            if (del == null) return null;
            return Variable<TRet>.OperatorUntyped(op, del.Method, arg1);
        }

        internal static Variable<TRet> OperatorFactor<TRet, TLeft, TRight>(Operator op, Variable<TLeft> arg1, Variable<TRight> arg2)
        {
            Delegate del = LookupOperatorFactor(op, typeof(TLeft), typeof(TRight));
            if (del == null) return null;
            return Variable<TRet>.OperatorUntyped(op, del.Method, arg1, arg2);
        }

        internal static Variable<TRet> OperatorFactor<TRet>(Operator op, Variable<T> arg1, Variable<T> arg2)
        {
            return OperatorFactor<TRet, T, T>(op, arg1, arg2);
        }

        internal static Variable<TRet> OperatorFactorThrows<TRet>(Operator op, Variable<T> arg1)
        {
            Delegate del = LookupOperatorFactor(op, typeof(T));
            if (del == null) throw new InvalidOperationException("No operator factor registered for : " + op + " with argument type " + typeof(T) + ".");
            return Variable<TRet>.OperatorUntyped(op, del.Method, arg1);
        }

        internal static Variable<TRet> OperatorFactorThrows<TRet, TLeft, TRight>(Operator op, Variable<TLeft> arg1, Variable<TRight> arg2)
        {
            Delegate del = LookupOperatorFactor(op, typeof(TLeft), typeof(TRight));
            if (del == null)
                throw new InvalidOperationException("No operator factor registered for '" + op + "' with argument types " + typeof(TLeft) + " and " + typeof(TRight) + ".");
            return Variable<TRet>.OperatorUntyped(op, del.Method, arg1, arg2);
        }

        internal static Variable<TRet> OperatorFactorThrows<TRet>(Operator op, Variable<T> arg1, Variable<T> arg2)
        {
            return OperatorFactorThrows<TRet, T, T>(op, arg1, arg2);
        }

        /// <summary>
        /// Operator overload for addition
        /// </summary>
        public static Variable<T> operator +(Variable<T> a, Variable<T> b)
        {
            return OperatorFactorThrows<T>(Operator.Plus, a, b);
        }

        public static Variable<T> operator +(Variable<T> a, Variable<char> b)
        {
            return OperatorFactorThrows<T, T, char>(Operator.Plus, a, b);
        }

        public static Variable<T> operator +(Variable<char> a, Variable<T> b)
        {
            return OperatorFactorThrows<T, char, T>(Operator.Plus, a, b);
        }

        /// <summary>
        /// Operator overload for addition
        /// </summary>
        public static Variable<T> operator +(Variable<T> a, T b)
        {
            return (a + Constant(b));
        }

        public static Variable<T> operator +(Variable<T> a, char b)
        {
            return (a + Constant(b));
        }

        public static Variable<T> operator +(char a, Variable<T> b)
        {
            return (Constant(a) + b);
        }

        /// <summary>
        /// Operator overload for subtraction
        /// </summary>
        public static Variable<T> operator -(Variable<T> a, Variable<T> b)
        {
            return OperatorFactorThrows<T>(Operator.Minus, a, b);
        }

        /// <summary>
        /// Operator overload for subtraction
        /// </summary>
        public static Variable<T> operator -(Variable<T> a, T b)
        {
            return (a - Constant(b));
        }

        /// <summary>
        /// Operator overload for subtraction
        /// </summary>
        public static Variable<T> operator -(T a, Variable<T> b)
        {
            return (Constant(a) - b);
        }

        /// <summary>
        /// Operator overload for multiplication
        /// </summary>
        public static Variable<T> operator *(Variable<T> a, Variable<T> b)
        {
            return OperatorFactorThrows<T>(Operator.Multiply, a, b);
        }

        /// <summary>
        /// Operator overload for multiplication
        /// </summary>
        public static Variable<T> operator *(Variable<T> a, T b)
        {
            return (a * Constant(b));
        }

        /// <summary>
        /// Operator overload for multiplication
        /// </summary>
        public static Variable<T> operator *(T a, Variable<T> b)
        {
            return (Constant(a) * b);
        }

        /// <summary>
        /// Operator overload for division
        /// </summary>
        public static Variable<T> operator /(Variable<T> a, Variable<T> b)
        {
            return OperatorFactorThrows<T>(Operator.Divide, a, b);
        }

        /// <summary>
        /// Operator overload for division
        /// </summary>
        public static Variable<T> operator /(Variable<T> a, T b)
        {
            return (a / Constant(b));
        }

        /// <summary>
        /// Operator overload for division
        /// </summary>
        public static Variable<T> operator /(T a, Variable<T> b)
        {
            return (Constant(a) / b);
        }

        /// <summary>
        /// Operator overload for modulus
        /// </summary>
        public static Variable<T> operator %(Variable<T> a, Variable<T> b)
        {
            return OperatorFactorThrows<T>(Operator.Modulus, a, b);
        }

        /// <summary>
        /// Operator overload for modulus
        /// </summary>
        public static Variable<T> operator %(Variable<T> a, T b)
        {
            return (a % Constant(b));
        }

        /// <summary>
        /// Operator overload for greater than
        /// </summary>
        public static Variable<bool> operator >(Variable<T> a, Variable<T> b)
        {
            return GreaterThan(a, b);
        }

        /// <summary>
        /// Operator overload for greater than
        /// </summary>
        public static Variable<bool> operator >(Variable<T> a, T b)
        {
            return (a > Constant(b));
        }

        /// <summary>
        /// 
        /// </summary>
        protected static Variable<bool> GreaterThan(Variable<T> a, Variable<T> b)
        {
            return OperatorFactor<bool>(Operator.GreaterThan, a, b)
                ?? OperatorFactor<bool>(Operator.LessThan, b, a)
                ?? NotOrNull(OperatorFactor<bool>(Operator.LessThanOrEqual, a, b))
                ?? NotOrNull(OperatorFactor<bool>(Operator.GreaterThanOrEqual, b, a))
                ?? GreaterThanFromMinus(a, b)
                ?? throw new InvalidOperationException("Neither of the operators (<,<=,>,>=) has a registered factor for argument type " + typeof(T) + ".");
        }

        private static Variable<bool> GreaterThanFromMinus(Variable<T> a, Variable<T> b)
        {
            if (typeof(double).IsAssignableFrom(typeof(T)))
            {
                Variable<T> diff;
                if (b.IsObserved && b.IsReadOnly && b.IsBase && b.ObservedValue.Equals(0.0))
                {
                    diff = a;
                }
                else if (a.IsObserved && a.IsReadOnly && a.IsBase && a.ObservedValue.Equals(0.0))
                {
                    diff = OperatorFactor<T>(Operator.Negative, b);
                }
                else
                {
                    diff = null;
                }
                if (diff is null)
                {
                    diff = OperatorFactor<T>(Operator.Minus, a, b);
                }
                if (diff is object)
                {
                    return IsPositive((Variable<double>)(Variable)diff);
                }
                // Fall through
            }
            return null;
        }

        private static Variable<bool> NotOrNull(Variable<bool> Variable)
        {
            return (Variable is null) ? null : !Variable;
        }

        /// <summary>
        /// 
        /// </summary>
        protected static Variable<bool> GreaterThanOrEqual(Variable<T> a, Variable<T> b)
        {
            return OperatorFactor<bool>(Operator.GreaterThanOrEqual, a, b)
                ?? OperatorFactor<bool>(Operator.LessThanOrEqual, b, a)
                ?? NotOrNull(OperatorFactor<bool>(Operator.LessThan, a, b))
                ?? NotOrNull(OperatorFactor<bool>(Operator.GreaterThan, b, a))
                ?? NotOrNull(GreaterThanFromMinus(b, a))
                ?? throw new InvalidOperationException("Neither of the operators (<,<=,>,>=) has a registered factor for argument type " + typeof(T) + ".");
        }

        /// <summary>
        /// Operator overload for less than or equal
        /// </summary>
        public static Variable<bool> operator <=(Variable<T> a, Variable<T> b)
        {
            Variable<bool> f = OperatorFactor<bool>(Operator.LessThanOrEqual, a, b);
            return f ?? GreaterThanOrEqual(b, a);
        }

        /// <summary>
        /// Operator overload for less than or equal
        /// </summary>
        public static Variable<bool> operator <=(Variable<T> a, T b)
        {
            return (a <= Constant(b));
        }

        /// <summary>
        /// Operator overload for less than
        /// </summary>
        public static Variable<bool> operator <(Variable<T> a, Variable<T> b)
        {
            Variable<bool> f = OperatorFactor<bool>(Operator.LessThan, a, b);
            return f ?? GreaterThan(b, a);
        }

        /// <summary>
        /// Operator overload for less than
        /// </summary>
        public static Variable<bool> operator <(Variable<T> a, T b)
        {
            return (a < Constant(b));
        }

        /// <summary>
        /// Operator overload for greater than or equal
        /// </summary>
        public static Variable<bool> operator >=(Variable<T> a, Variable<T> b)
        {
            return GreaterThanOrEqual(a, b);
        }

        /// <summary>
        /// Operator overload for greater than or equal
        /// </summary>
        public static Variable<bool> operator >=(Variable<T> a, T b)
        {
            return (a >= Constant(b));
        }

        /// <summary>
        /// Returns a new variable that is true when two variables are equal.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>A new variable that is true when a and b are equal.</returns>
        /// <remarks>
        /// If you want to test if a variable object points to null, or if two variable objects are the same object, 
        /// then cast to object first and use reference equality, like so:
        /// <list type="bullet">
        /// <item><c>if((object)var == null) ...</c></item>
        /// <item><c>if((object)var1 == (object)var2) ...</c></item>
        /// <item><c>if(object.ReferenceEquals(var1,var2)) ...</c></item>
        /// </list>
        /// </remarks>
        public static Variable<bool> operator ==(Variable<T> a, Variable<T> b)
        {
            Variable<bool> f = OperatorFactor<bool>(Operator.Equal, a, b);
            return f ?? !OperatorFactorThrows<bool>(Operator.NotEqual, a, b);
        }

        /// <summary>
        /// Returns a new variable that is true when a variable equals a given value.
        /// </summary>
        /// <param name="a">A variable</param>
        /// <param name="b">A value</param>
        /// <returns>A new variable that is true when a equals b.</returns>
        /// <remarks>
        /// If you want to test if a variable object points to null,
        /// then cast to object first and use reference equality, like so:
        /// <c>if((object)var == null) ...</c>
        /// </remarks>
        public static Variable<bool> operator ==(Variable<T> a, T b)
        {
            return (a == Constant(b));
        }

        /// <summary>
        /// Returns a new variable that is true when two variables are not equal.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>A new variable that is true when a and b are not equal.</returns>
        /// <remarks>
        /// If you want to test if a variable object does not point to null, or if two variable objects are not the same object, 
        /// then cast to object first and use reference equality, like so:
        /// <list type="bullet">
        /// <item><c>if((object)var != null) ...</c></item>
        /// <item><c>if((object)var1 != (object)var2) ...</c></item>
        /// <item><c>if(!object.ReferenceEquals(var1,var2)) ...</c></item>
        /// </list>
        /// </remarks>
        public static Variable<bool> operator !=(Variable<T> a, Variable<T> b)
        {
            Variable<bool> f = OperatorFactor<bool>(Operator.NotEqual, a, b);
            return f ?? !OperatorFactorThrows<bool>(Operator.Equal, a, b);
        }

        /// <summary>
        /// Returns a new variable that is true when a variable does not equal a given value.
        /// </summary>
        /// <param name="a">A variable</param>
        /// <param name="b">A value</param>
        /// <returns>A new variable that is true when a does not equal b.</returns>
        /// <remarks>
        /// If you want to test if a variable object does not point to null,
        /// then cast to object first and use reference equality, like so:
        /// <c>if((object)var != null) ...</c>
        /// </remarks>
        public static Variable<bool> operator !=(Variable<T> a, T b)
        {
            return (a != Constant(b));
        }

        /// <summary>
        /// Operator overload for NOT
        /// </summary>
        public static Variable<T> operator !(Variable<T> a)
        {
            return OperatorFactorThrows<T>(Operator.Not, a);
        }

        /// <summary>
        /// Operator overload for OR
        /// </summary>
        public static Variable<T> operator |(Variable<T> a, Variable<T> b)
        {
            return OperatorFactorThrows<T>(Operator.Or, a, b);
        }

        /// <summary>
        /// Operator overload for AND
        /// </summary>
        public static Variable<T> operator &(Variable<T> a, Variable<T> b)
        {
            return OperatorFactorThrows<T>(Operator.And, a, b);
        }

        /// <summary>
        /// Operator overload for XOR
        /// </summary>
        public static Variable<T> operator ^(Variable<T> a, Variable<T> b)
        {
            return OperatorFactorThrows<T>(Operator.Xor, a, b);
        }

        /// <summary>
        /// Operator overload for NOT
        /// </summary>
        public static Variable<T> operator ~(Variable<T> a)
        {
            return OperatorFactorThrows<T>(Operator.Complement, a);
        }

        /// <summary>
        /// Operator overload for unary negation
        /// </summary>
        public static Variable<T> operator -(Variable<T> a)
        {
            Variable<T> f = OperatorFactor<T>(Operator.Negative, a);
            if (f is object) return f;
            else if (a is Variable<double>)
            {
                Variable<T> zero = (Variable<T>)(object)Constant(0.0);
                return zero - a;
            }
            else
            {
                throw new InvalidOperationException("Operator (-) does not have a registered factor for argument type " + typeof(T) + ".");
            }
        }

#if false
        public static Variable<T> operator <<(Variable<T> a, int b)
        {
            return OperatorFactorThrows<T>(Operator.LeftShift, a, b);
        }
        public static Variable<T> operator >>(Variable<T> a, int b)
        {
            return OperatorFactorThrows<T>(Operator.RightShift, a, b);
        }
#endif
    }
}