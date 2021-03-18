// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Models
{
    /// <summary>
    /// Interface to a multidimensional array of variables.
    /// </summary>
    public interface IVariableArray : IVariable
    {
        /// <summary>
        /// List of ranges associated with the array
        /// </summary>
        IList<Range> Ranges { get; }

        /// <summary>
        /// Create a clone of this array with ranges and size expressions replaced
        /// </summary>
        /// <param name="rangeReplacements">Ranges to replace</param>
        /// <param name="expressionReplacements">Size expressions to replace</param>
        /// <param name="deepCopy">If true, clones all item prototypes, otherwise they are not cloned</param>
        /// <returns>A new array</returns>
        IVariableArray ReplaceRanges(Dictionary<Range, Range> rangeReplacements, Dictionary<IModelExpression, IModelExpression> expressionReplacements, bool deepCopy);
    }

    /// <summary>
    /// Interface to a jagged array of variables
    /// </summary>
    public interface IVariableJaggedArray
    {
        /// <summary>
        /// Item prototype
        /// </summary>
        IVariable ItemPrototype { get; }
    }

    /// <summary>
    /// Interface for an object having item variables
    /// </summary>
    public interface HasItemVariables
    {
        /// <summary>
        /// Gets the items
        /// </summary>
        /// <returns></returns>
        Dictionary<IReadOnlyList<IModelExpression>, IVariable> GetItemsUntyped();
    }

    /// <summary>
    /// Represents a jagged variable array of arbitrary rank
    /// </summary>
    /// <typeparam name="TItem">The item variable type.</typeparam>
    /// <typeparam name="TArray">The domain type of the array.</typeparam>
    /// <remarks>
    /// TItem is either a VariableArray&lt;T&gt; or another VariableArray&lt;,&gt;
    /// </remarks>
    /// <exclude/>
    public abstract class VariableArrayBase<TItem, TArray> : Variable<TArray>, HasItemVariables,
                                                             IVariableArray, IVariableJaggedArray
        where TItem : Variable, ICloneable, SettableTo<TItem>
    {
        private Range[] ranges;

        /// <summary>
        /// List of ranges associated with the array
        /// </summary>
        public IList<Range> Ranges
        {
            get { return ranges; }
        }

        protected TItem itemPrototype;

        IVariable IVariableJaggedArray.ItemPrototype
        {
            get { return itemPrototype; }
        }

        internal VariableArrayBase(TItem itemPrototype, params Range[] ranges)
            : base(itemPrototype.Containers)
        {
            this.ranges = ranges;
            this.itemPrototype = itemPrototype;
            // check that none of the ranges match an open container
            foreach (Range r in ranges)
            {
                if (r == null) throw new ArgumentNullException("range");
                foreach (IStatementBlock stBlock in containers)
                {
                    if (stBlock is ForEachBlock)
                    {
                        ForEachBlock fb = (ForEachBlock) stBlock;
                        if (r.Equals(fb.Range)) throw new InvalidOperationException("Range '" + r + "' is already open in a ForEach block.  Use a cloned range instead.");
                    }
                }
            }
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="array"></param>
        protected VariableArrayBase(VariableArrayBase<TItem, TArray> array)
            : base(array)
        {
            this.ranges = array.ranges;
            this.itemPrototype = array.itemPrototype;
        }

        // TM: this is hidden since users will not know what IModelExpression is.
        // Also, IModelExpression is too weak of a type.
        /// <summary>
        /// </summary>
        protected TItem this[params IModelExpression[] index]
        {
            get
            {
                // returns a random variable whose indices match the array size
                return GetItem(this, itemPrototype, index);
            }
            set { SetItem(this, itemPrototype, value, index); }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        protected TItem this[params Range[] index]
        {
            get { return this[(IModelExpression[]) index]; }
            set
            {
                if (IsDefined) throw new InvalidOperationException("Cannot assign to array more than once.");
                SetItem(this, itemPrototype, value, index);
            }
        }

        /// <summary>
        /// Get a random variable representing an item of an array.
        /// </summary>
        /// <param name="array"></param>
        /// <param name="itemPrototype"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        internal static TItem GetItem(VariableArrayBase<TItem, TArray> array, TItem itemPrototype, params IModelExpression[] index)
        {
            Set<Range> switchRanges = new Set<Range>();
            IList<Range> ranges = array.Ranges;
            if (index.Length != ranges.Count) throw new ArgumentException("Provided " + index.Length + " indices to an array of rank " + ranges.Count);
            for (int i = 0; i < ranges.Count; i++)
            {
                ranges[i].CheckCompatible(index[i], array);
                foreach (SwitchBlock block in StatementBlock.EnumerateOpenBlocks<SwitchBlock>())
                {
                    if (block.Range.Equals(index[i]))
                        throw new ArgumentException("Cannot index by '" + index[i] + "' in a switch block over '" + block.ConditionVariable + "'");
                }
            }
            IVariable item;
            Dictionary<IReadOnlyList<IModelExpression>, IVariable> itemVariables = ((HasItemVariables) array).GetItemsUntyped();
            if (itemVariables.TryGetValue(index, out item)) return (TItem) item;
            // the item must be in the same containers as the array (not the currently open containers)
            if (itemPrototype is IVariableArray)
            {
                Dictionary<Range, Range> replacements = new Dictionary<Range, Range>();
                Dictionary<IModelExpression, IModelExpression> expressionReplacements = new Dictionary<IModelExpression, IModelExpression>();
                for (int i = 0; i < ranges.Count; i++)
                {
                    expressionReplacements.Add(ranges[i], index[i]);
                }
                IVariable result = ((IVariableArray) itemPrototype).ReplaceRanges(replacements, expressionReplacements, deepCopy: false);
                TItem v = (TItem) result;
                v.MakeItem(array, index);
                return v;
            }
            else
            {
                TItem v = (TItem) itemPrototype.Clone();
                v.MakeItem(array, index);
                return v;
            }
        }

        internal static void SetItem(VariableArrayBase<TItem, TArray> array, TItem itemPrototype, TItem value, params IModelExpression[] index)
        {
            TItem item = GetItem(array, itemPrototype, index);
            //  SetTo already checks for compatible indexing.
            //item.CheckCompatibleIndexing(value);
            item.SetTo(value);
            if (Variable.AutoNaming)
            {
                //this.Name = value.ArrayVar.ToString();// +"," + range.Name;
            }
        }

        /// <summary>
        /// Set the variable array to the given value
        /// </summary>
        /// <param name="that"></param>
        public void SetTo(VariableArrayBase<TItem, TArray> that)
        {
            bool foundDef = false;
            foreach (TItem item in that.items.Values)
            {
                if (item.IsDefined)
                {
                    SetItem(this, itemPrototype, item, item.indices.ToArray());
                    foundDef = true;
                }
            }
            if (!foundDef) base.SetTo(that);
        }

        IVariableArray IVariableArray.ReplaceRanges(Dictionary<Range, Range> rangeReplacements, Dictionary<IModelExpression, IModelExpression> expressionReplacements, bool deepCopy)
        {
            return this;
        }

        /// <summary>
        /// All item variables referring to this array.
        /// </summary>
        internal Dictionary<IReadOnlyList<IModelExpression>, IVariable> items = new Dictionary<IReadOnlyList<IModelExpression>, IVariable>(new ReadOnlyListComparer<IModelExpression>());

        Dictionary<IReadOnlyList<IModelExpression>, IVariable> HasItemVariables.GetItemsUntyped()
        {
            return items;
        }
    }
}