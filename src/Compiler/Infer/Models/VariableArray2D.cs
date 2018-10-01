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
    /// Interface to a 2D array of variables.
    /// </summary>
    public interface IVariableArray2D<T> : IVariable, IVariableArray
    {
        /// <summary>
        /// Range for first index
        /// </summary>
        Range Range0 { get; }

        /// <summary>
        /// Range for second index
        /// </summary>
        Range Range1 { get; }

        /// <summary>
        /// Sets/Gets element in array given by the index expressions
        /// </summary>
        /// <param name="index0">First index expression</param>
        /// <param name="index1">Second index expression</param>
        /// <returns></returns>
        Variable<T> this[IModelExpression index0, IModelExpression index1] { get; set; }
    }

    /// <summary>
    /// Two-dimensional jagged variable array.
    /// </summary>
    /// <typeparam name="TItem"></typeparam>
    /// <typeparam name="TArray"></typeparam>
    public class VariableArray2D<TItem, TArray> : VariableArrayBase<TItem, TArray>, SettableTo<VariableArray2D<TItem, TArray>>, IVariableArray
        where TItem : Variable, ICloneable, SettableTo<TItem>
    {
        /// <summary>
        /// Range for first index
        /// </summary>
        public Range Range0
        {
            get { return Ranges[0]; }
        }

        /// <summary>
        /// Range for second index
        /// </summary>
        public Range Range1
        {
            get { return Ranges[1]; }
        }

        internal VariableArray2D(TItem itemPrototype, Range range0, Range range1)
            : base(itemPrototype, range0, range1)
        {
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        protected VariableArray2D(VariableArray2D<TItem, TArray> that)
            : base(that)
        {
        }

        /// <summary>
        /// Get or set elements of the array.
        /// </summary>
        /// <param name="range0">The first range used to create the array.</param>
        /// <param name="range1">The second range used to create the array.</param>
        /// <returns>A derived variable that indexes <c>this</c> by <paramref name="range0"/>,<paramref name="range1"/>.</returns>
        /// <remarks>
        /// When setting the elements of an array, the right hand side must be a fresh variable with no other uses.
        /// The right-hand side must be an item variable indexed by exactly the ranges of the array, but possibly in a different order.
        /// </remarks>
        public TItem this[Range range0, Range range1]
        {
            get { return base[range0, range1]; }
            set { base[range0, range1] = value; }
        }

        /// <summary>
        /// Get or set elements of the array.
        /// </summary>
        /// <param name="index0">The first range used to create the array.</param>
        /// <param name="index1">The second index.</param>
        /// <returns>A derived variable that indexes <c>this</c> by <paramref name="index0"/>,<paramref name="index1"/>.</returns>
        /// <remarks>
        /// When setting the elements of an array, the right hand side must be a fresh variable with no other uses.
        /// The right-hand side must be an item variable indexed by exactly the ranges of the array, but possibly in a different order.
        /// </remarks>
        public TItem this[Range index0, Variable<int> index1]
        {
            get { return base[index0, index1]; }
            set { base[index0, index1] = value; }
        }

        /// <summary>
        /// Get or set elements of the array.
        /// </summary>
        /// <param name="index0">The first index.</param>
        /// <param name="index1">The second range used to create the array.</param>
        /// <returns>A derived variable that indexes <c>this</c> by <paramref name="index0"/>,<paramref name="index1"/>.</returns>
        /// <remarks>
        /// When setting the elements of an array, the right hand side must be a fresh variable with no other uses.
        /// The right-hand side must be an item variable indexed by exactly the ranges of the array, but possibly in a different order.
        /// </remarks>
        public TItem this[Variable<int> index0, Range index1]
        {
            get { return base[index0, index1]; }
            set { base[index0, index1] = value; }
        }

        /// <summary>
        /// Get or set elements of the array.
        /// </summary>
        /// <param name="index0">The first index.</param>
        /// <param name="index1">The second index.</param>
        /// <returns>A derived variable that indexes <c>this</c> by <paramref name="index0"/>,<paramref name="index1"/>.</returns>
        /// <remarks>
        /// When setting the elements of an array, the right hand side must be a fresh variable with no other uses.
        /// The right-hand side must be an item variable indexed by exactly the ranges of the array, but possibly in a different order.
        /// </remarks>
        public TItem this[Variable<int> index0, Variable<int> index1]
        {
            get { return base[index0, index1]; }
            set { base[index0, index1] = value; }
        }

        /// <summary>
        /// Inline method to name the array.
        /// </summary>
        /// <param name="name"></param>
        /// <returns><c>this</c></returns>
        public new VariableArray2D<TItem, TArray> Named(string name)
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
        public new VariableArray2D<TItem, TArray> Attrib(ICompilerAttribute attr)
        {
            base.Attrib(attr);
            return this;
        }

        /// <summary>
        /// Helper to add a query type attribute to this variable.
        /// </summary>
        /// <param name="queryType">The query type to use to create the attribute</param>
        public new VariableArray2D<TItem, TArray> Attrib(QueryType queryType)
        {
            return Attrib(new QueryTypeCompilerAttribute(queryType));
        }

        /// <summary>
        /// Set the 2-D jagged array to a specified value
        /// </summary>
        /// <param name="that"></param>
        public void SetTo(VariableArray2D<TItem, TArray> that)
        {
            base.SetTo(that);
        }

        /// <summary>
        /// Clone the 2-D jagged array
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            return new VariableArray2D<TItem, TArray>(this);
        }

        IVariableArray IVariableArray.ReplaceRanges(Dictionary<Range, Range> rangeReplacements, Dictionary<IModelExpression, IModelExpression> expressionReplacements, bool deepCopy)
        {
            // must do this replacement first, since it will influence how we replace the itemPrototype
            Range newRange0 = Range0.Replace(rangeReplacements, expressionReplacements);
            Range newRange1 = Range1.Replace(rangeReplacements, expressionReplacements);
            TItem itemPrototype = (TItem) ((IVariableJaggedArray) this).ItemPrototype;
            if (itemPrototype is IVariableArray)
            {
                IVariable result = ((IVariableArray) itemPrototype).ReplaceRanges(rangeReplacements, expressionReplacements, deepCopy);
                itemPrototype = (TItem) result;
            }
            else if (deepCopy)
            {
                itemPrototype = (TItem)itemPrototype.Clone();
                itemPrototype.containers = StatementBlock.GetOpenBlocks();
            }
            return new VariableArray2D<TItem, TArray>(itemPrototype, newRange0, newRange1);
        }
    }

    /// <summary>
    /// Two-dimensional flat variable array.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class VariableArray2D<T> : VariableArray2D<Variable<T>, T[,]>, IVariableArray2D<T>, SettableTo<VariableArray2D<T>>, IVariableArray
    {
        internal VariableArray2D(Range range0, Range range1)
            : base(new Variable<T>(), range0, range1)
        {
        }

        internal VariableArray2D(Variable<T> itemPrototype, Range range0, Range range1)
            : base(itemPrototype, range0, range1)
        {
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        protected VariableArray2D(VariableArray2D<T> that)
            : base(that)
        {
        }

        /// <summary>
        /// Set the name of the array.
        /// </summary>
        /// <param name="name"></param>
        /// <returns><c>this</c></returns>
        public new VariableArray2D<T> Named(string name)
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
        public new VariableArray2D<T> Attrib(ICompilerAttribute attr)
        {
            base.Attrib(attr);
            return this;
        }

        /// <summary>
        /// Helper to add a query type attribute to this variable.
        /// </summary>
        /// <param name="queryType">The query type to use to create the attribute</param>
        public new VariableArray2D<T> Attrib(QueryType queryType)
        {
            return Attrib(new QueryTypeCompilerAttribute(queryType));
        }

        /// <summary>
        /// Set this 2-D array to a specified value
        /// </summary>
        /// <param name="that"></param>
        public void SetTo(VariableArray2D<T> that)
        {
            base.SetTo(that);
        }

        /// <summary>
        /// Clone this 2-D array
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            return new VariableArray2D<T>(this);
        }

        IVariableArray IVariableArray.ReplaceRanges(Dictionary<Range, Range> rangeReplacements, Dictionary<IModelExpression, IModelExpression> expressionReplacements, bool deepCopy)
        {
            Range newRange0 = Range0.Replace(rangeReplacements, expressionReplacements);
            Range newRange1 = Range1.Replace(rangeReplacements, expressionReplacements);
            Variable<T> itemPrototype2 = itemPrototype;
            if (deepCopy)
            {
                itemPrototype2 = (Variable<T>)itemPrototype.Clone();
                itemPrototype2.containers = StatementBlock.GetOpenBlocks();
            }
            return new VariableArray2D<T>(itemPrototype2, newRange0, newRange1);
        }

        /// <summary>
        /// Sets/Gets element in array given by the index expressions
        /// </summary>
        /// <param name="index0">First index expression</param>
        /// <param name="index1">Second index expression</param>
        /// <returns></returns>
        Variable<T> IVariableArray2D<T>.this[IModelExpression index0, IModelExpression index1]
        {
            get { return this[index0, index1]; }
            set { this[(Range) index0, (Range) index1] = value; }
        }
    }
}