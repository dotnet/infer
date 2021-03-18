// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Compiler;

namespace Microsoft.ML.Probabilistic.Models
{
    /// <summary>
    /// Interface to a jagged array of variables.
    /// </summary>
    public interface IJaggedVariableArray<TItem> : IVariableArray
    {
        /// <summary>
        /// Range for variable array
        /// </summary>
        Range Range { get; }

        /// <summary>
        /// Sets/Gets element in array given by index expression
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        TItem this[IModelExpression index] { get; set; }
    }

    /// <summary>
    /// Interface to an array of variables.
    /// </summary>
    public interface IVariableArray<T> : IJaggedVariableArray<Variable<T>>
    {
    }

    /// <summary>
    /// One-dimensional jagged variable array.
    /// </summary>
    /// <typeparam name="TItem">Item type</typeparam>
    /// <typeparam name="TArray">Array type</typeparam>
    public class VariableArray<TItem, TArray> : VariableArrayBase<TItem, TArray>, SettableTo<VariableArray<TItem, TArray>>, IJaggedVariableArray<TItem>
        where TItem : Variable, ICloneable, SettableTo<TItem>
    {
        /// <summary>
        /// Range for the array
        /// </summary>
        public Range Range
        {
            get { return Ranges[0]; }
        }

        internal VariableArray(TItem itemPrototype, Range range)
            : base(itemPrototype, range)
        {
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        protected VariableArray(VariableArray<TItem, TArray> that)
            : base(that)
        {
        }

        /// <summary>
        /// Get or set elements of the array.
        /// </summary>
        /// <param name="range">The range used to create the array.</param>
        /// <returns>A derived variable that indexes <c>this</c> by <paramref name="range"/>.</returns>
        /// <remarks>
        /// When setting the elements of an array, the right hand side must be a fresh variable with no other uses.
        /// The right-hand side must be an item variable indexed by exactly the ranges of the array, but possibly in a different order.
        /// </remarks>
        public TItem this[Range range]
        {
            get { return this[(IModelExpression) range]; }
            set { base[range] = value; }
        }

        /// <summary>
        /// Get a variable element of the array.
        /// </summary>
        /// <param name="index">A variable which selects the element.</param>
        /// <returns>A derived variable that indexes <c>this</c> by <paramref name="index"/>.</returns>
        public TItem this[Variable<int> index]
        {
            get { return this[(IModelExpression) index]; }
            set { this[(IModelExpression) index] = value; }
        }

        /// <summary>
        /// Get an element of the array.
        /// </summary>
        /// <param name="index">An integer in [0,array.Length-1]</param>
        /// <returns>A derived variable that indexes <c>this</c> by <paramref name="index"/>.</returns>
        public TItem this[int index]
        {
            get { return this[Variable.Constant(index)]; }
            set { this[Variable.Constant(index)] = value; }
        }

        /// <summary>
        /// Sets/Gets element in array given by index expression
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        TItem IJaggedVariableArray<TItem>.this[IModelExpression index]
        {
            get { return this[index]; }
            set { this[index] = value; }
        }

        /// <summary>
        /// Set the name of the array.
        /// </summary>
        /// <param name="name"></param>
        /// <returns><c>this</c></returns>
        public new VariableArray<TItem, TArray> Named(string name)
        {
            base.Named(name);
            return this;
        }

        /// <summary>
        /// Inline method for adding an attribute to a random variable.  This method
        /// returns the random variable object, so that is can be used in an inline expression.
        /// e.g. Variable.GaussianFromMeanAndVariance(0,1).Attrib(new MyAttribute());
        /// </summary>
        /// <param name="attr">The attribute to add</param>
        /// <returns>The random variable object</returns>
        public new VariableArray<TItem, TArray> Attrib(ICompilerAttribute attr)
        {
            base.Attrib(attr);
            return this;
        }

        /// <summary>
        /// Helper to add a query type attribute to this variable.
        /// </summary>
        /// <param name="queryType">The query type to use to create the attribute</param>
        public new VariableArray<TItem, TArray> Attrib(QueryType queryType)
        {
            return Attrib(new QueryTypeCompilerAttribute(queryType));
        }

        /// <summary>
        /// Set the variable array to the given value.  Should only be invoked on arrays created using Variable.Array() 
        /// where the elements have not yet been filled in.
        /// </summary>
        /// <param name="that">A variable array whose definition will be consumed by <c>this</c> and no longer available for use</param>
        /// <remarks>
        /// <paramref name="that"/> must have exactly the same set of ranges as <c>this</c>.
        /// </remarks>
        public void SetTo(VariableArray<TItem, TArray> that)
        {
            base.SetTo(that);
        }

        /// <summary>
        /// Clone the variable array
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            return new VariableArray<TItem, TArray>(this);
        }

        IVariableArray IVariableArray.ReplaceRanges(Dictionary<Range, Range> rangeReplacements, Dictionary<IModelExpression, IModelExpression> expressionReplacements, bool deepCopy)
        {
            // must do this replacement first, since it will influence how we replace the itemPrototype
            Range newRange = Range.Replace(rangeReplacements, expressionReplacements);
            TItem itemPrototype = (TItem) ((IVariableJaggedArray) this).ItemPrototype;
            if (itemPrototype is IVariableArray)
            {
                IVariable result = ((IVariableArray)itemPrototype).ReplaceRanges(rangeReplacements, expressionReplacements, deepCopy);
                itemPrototype = (TItem)result;
            }
            else
            {
                // make a clone in the current containers
                itemPrototype = (TItem)itemPrototype.Clone();
                itemPrototype.containers = StatementBlock.GetOpenBlocks();
            }
            return new VariableArray<TItem, TArray>(itemPrototype, newRange);
        }
    }

    /// <summary>
    /// One-dimensional flat variable array.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class VariableArray<T> : VariableArray<Variable<T>, T[]>, IVariableArray<T>, SettableTo<VariableArray<T>>, IVariableArray
    {
        internal VariableArray(Range range)
            : base(new Variable<T>(), range)
        {
        }

        internal VariableArray(Variable<T> itemPrototype, Range range)
            : base(itemPrototype, range)
        {
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        protected VariableArray(VariableArray<T> that)
            : base(that)
        {
        }

        /// <summary>
        /// Set the name of the array.
        /// </summary>
        /// <param name="name"></param>
        /// <returns><c>this</c></returns>
        public new VariableArray<T> Named(string name)
        {
            base.Named(name);
            return this;
        }

        /// <summary>
        /// Inline method for adding an attribute to a random variable.  This method
        /// returns the random variable object, so that is can be used in an inline expression.
        /// e.g. Variable.GaussianFromMeanAndVariance(0,1).Attrib(new MyAttribute());
        /// </summary>
        /// <param name="attr">The attribute to add</param>
        /// <returns>The random variable object</returns>
        public new VariableArray<T> Attrib(ICompilerAttribute attr)
        {
            base.Attrib(attr);
            return this;
        }

        /// <summary>
        /// Helper to add a query type attribute to this variable.
        /// </summary>
        /// <param name="queryType">The query type to use to create the attribute</param>
        public new VariableArray<T> Attrib(QueryType queryType)
        {
            return Attrib(new QueryTypeCompilerAttribute(queryType));
        }

        /// <summary>
        /// Set the 1-D array to the given value
        /// </summary>
        /// <param name="that"></param>
        public void SetTo(VariableArray<T> that)
        {
            base.SetTo(that);
        }

        /// <summary>
        /// Clone the array
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            return new VariableArray<T>(this);
        }

        IVariableArray IVariableArray.ReplaceRanges(Dictionary<Range, Range> rangeReplacements, Dictionary<IModelExpression, IModelExpression> expressionReplacements, bool deepCopy)
        {
            Range newRange = Range.Replace(rangeReplacements, expressionReplacements);
            Variable<T> itemPrototype2 = itemPrototype;
            if (deepCopy)
            {
                itemPrototype2 = (Variable<T>)itemPrototype.Clone();
                itemPrototype2.containers = StatementBlock.GetOpenBlocks();
            }
            return new VariableArray<T>(itemPrototype2, newRange);
        }
    }
}