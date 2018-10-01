// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Compiler;

namespace Microsoft.ML.Probabilistic.Models
{
    /// <summary>
    /// Interface to a 3D array of variables.
    /// </summary>
    public interface IVariableArray3D<T> : IVariable, IVariableArray
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
        /// Range for third index
        /// </summary>
        Range Range2 { get; }

        /// <summary>
        /// Sets/Gets element in array given by the index expressions
        /// </summary>
        /// <param name="index0">First index expression</param>
        /// <param name="index1">Second index expression</param>
        /// <param name="index2">Second index expression</param>
        /// <returns></returns>
        Variable<T> this[IModelExpression index0, IModelExpression index1, IModelExpression index2] { get; set; }
    }

    /// <summary>
    /// Three-dimensional jagged variable array.
    /// </summary>
    /// <typeparam name="TItem"></typeparam>
    /// <typeparam name="TArray"></typeparam>
    public class VariableArray3D<TItem, TArray> : VariableArrayBase<TItem, TArray>, SettableTo<VariableArray3D<TItem, TArray>>, IVariableArray
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

        /// <summary>
        /// Range for third index
        /// </summary>
        public Range Range2
        {
            get { return Ranges[2]; }
        }

        internal VariableArray3D(TItem itemPrototype, Range range0, Range range1, Range range2)
            : base(itemPrototype, range0, range1, range2)
        {
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        protected VariableArray3D(VariableArray3D<TItem, TArray> that)
            : base(that)
        {
        }

        /// <summary>
        /// Get or set elements of the array.
        /// </summary>
        /// <param name="range0">The first range used to create the array.</param>
        /// <param name="range1">The second range used to create the array.</param>
        /// <param name="range2">The third range used to create the array.</param>
        /// <returns>A derived variable that indexes <c>this</c> by the given ranges.</returns>
        /// <remarks>
        /// When setting the elements of an array, the right hand side must be a fresh variable with no other uses.
        /// The right-hand side must be an item variable indexed by exactly the ranges of the array, but possibly in a different order.
        /// </remarks>
        public TItem this[Range range0, Range range1, Range range2]
        {
            get { return base[range0, range1, range2]; }
            set { base[range0, range1, range2] = value; }
        }

        /// <summary>
        /// Get elements of the array.
        /// </summary>
        /// <param name="range0">The first range used to create the array.</param>
        /// <param name="range1">The second range used to create the array.</param>
        /// <param name="range2">The third range used to create the array.</param>
        /// <returns>A derived variable that indexes <c>this</c> by the given ranges.</returns>
        /// <remarks>
        /// When setting the elements of an array, the right hand side must be a fresh variable with no other uses.
        /// The right-hand side must be an item variable indexed by exactly the ranges of the array, but possibly in a different order.
        /// </remarks>
        public TItem this[Variable<int> range0, Range range1, Range range2]
        {
            get { return base[range0, range1, range2]; }
        }

        /// <summary>
        /// Get elements of the array.
        /// </summary>
        /// <param name="range0">The first range used to create the array.</param>
        /// <param name="range1">The second range used to create the array.</param>
        /// <param name="range2">The third range used to create the array.</param>
        /// <returns>A derived variable that indexes <c>this</c> by the given ranges.</returns>
        /// <remarks>
        /// When setting the elements of an array, the right hand side must be a fresh variable with no other uses.
        /// The right-hand side must be an item variable indexed by exactly the ranges of the array, but possibly in a different order.
        /// </remarks>
        public TItem this[Variable<int> range0, Variable<int> range1, Range range2]
        {
            get { return base[range0, range1, range2]; }
        }

        /// <summary>
        /// Get elements of the array.
        /// </summary>
        /// <param name="range0">The first range used to create the array.</param>
        /// <param name="range1">The second range used to create the array.</param>
        /// <param name="range2">The third range used to create the array.</param>
        /// <returns>A derived variable that indexes <c>this</c> by the given ranges.</returns>
        /// <remarks>
        /// When setting the elements of an array, the right hand side must be a fresh variable with no other uses.
        /// The right-hand side must be an item variable indexed by exactly the ranges of the array, but possibly in a different order.
        /// </remarks>
        public TItem this[Variable<int> range0, Variable<int> range1, Variable<int> range2]
        {
            get { return base[range0, range1, range2]; }
        }

        /// <summary>
        /// Get elements of the array.
        /// </summary>
        /// <param name="range0">The first range used to create the array.</param>
        /// <param name="range1">The second range used to create the array.</param>
        /// <param name="range2">The third range used to create the array.</param>
        /// <returns>A derived variable that indexes <c>this</c> by the given ranges.</returns>
        /// <remarks>
        /// When setting the elements of an array, the right hand side must be a fresh variable with no other uses.
        /// The right-hand side must be an item variable indexed by exactly the ranges of the array, but possibly in a different order.
        /// </remarks>
        public TItem this[Variable<int> range0, Range range1, Variable<int> range2]
        {
            get { return base[range0, range1, range2]; }
        }

        /// <summary>
        /// Get elements of the array.
        /// </summary>
        /// <param name="range0">The first range used to create the array.</param>
        /// <param name="range1">The second range used to create the array.</param>
        /// <param name="range2">The third range used to create the array.</param>
        /// <returns>A derived variable that indexes <c>this</c> by the given ranges.</returns>
        /// <remarks>
        /// When setting the elements of an array, the right hand side must be a fresh variable with no other uses.
        /// The right-hand side must be an item variable indexed by exactly the ranges of the array, but possibly in a different order.
        /// </remarks>
        public TItem this[Range range0, Variable<int> range1, Range range2]
        {
            get { return base[range0, range1, range2]; }
        }

        /// <summary>
        /// Get elements of the array.
        /// </summary>
        /// <param name="range0">The first range used to create the array.</param>
        /// <param name="range1">The second range used to create the array.</param>
        /// <param name="range2">The third range used to create the array.</param>
        /// <returns>A derived variable that indexes <c>this</c> by the given ranges.</returns>
        /// <remarks>
        /// When setting the elements of an array, the right hand side must be a fresh variable with no other uses.
        /// The right-hand side must be an item variable indexed by exactly the ranges of the array, but possibly in a different order.
        /// </remarks>
        public TItem this[Range range0, Range range1, Variable<int> range2]
        {
            get { return base[range0, range1, range2]; }
        }

        /// <summary>
        /// Get elements of the array.
        /// </summary>
        /// <param name="range0">The first range used to create the array.</param>
        /// <param name="range1">The second range used to create the array.</param>
        /// <param name="range2">The third range used to create the array.</param>
        /// <returns>A derived variable that indexes <c>this</c> by the given ranges.</returns>
        /// <remarks>
        /// When setting the elements of an array, the right hand side must be a fresh variable with no other uses.
        /// The right-hand side must be an item variable indexed by exactly the ranges of the array, but possibly in a different order.
        /// </remarks>
        public TItem this[Range range0, Variable<int> range1, Variable<int> range2]
        {
            get { return base[range0, range1, range2]; }
        }

        /// <summary>
        /// Set the name of the array.
        /// </summary>
        /// <param name="name"></param>
        /// <returns><c>this</c></returns>
        public new VariableArray3D<TItem, TArray> Named(string name)
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
        public new VariableArray3D<TItem, TArray> Attrib(ICompilerAttribute attr)
        {
            base.Attrib(attr);
            return this;
        }

        /// <summary>
        /// Set the 3-D jagged array to the specified value
        /// </summary>
        /// <param name="that"></param>
        public void SetTo(VariableArray3D<TItem, TArray> that)
        {
            base.SetTo(that);
        }

        /// <summary>
        /// Clone the 3-D jagged array
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            return new VariableArray3D<TItem, TArray>(this);
        }

        IVariableArray IVariableArray.ReplaceRanges(Dictionary<Range, Range> rangeReplacements, Dictionary<IModelExpression, IModelExpression> expressionReplacements, bool deepCopy)
        {
            // must do this replacement first, since it will influence how we replace the itemPrototype
            Range newRange0 = Range0.Replace(rangeReplacements, expressionReplacements);
            Range newRange1 = Range1.Replace(rangeReplacements, expressionReplacements);
            Range newRange2 = Range2.Replace(rangeReplacements, expressionReplacements);
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
            return new VariableArray3D<TItem, TArray>(itemPrototype, newRange0, newRange1, newRange2);
        }
    }

    /// <summary>
    /// Three-dimensional flat variable array.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class VariableArray3D<T> : VariableArray3D<Variable<T>, T[,]>, IVariableArray3D<T>, SettableTo<VariableArray3D<T>>
    {
        internal VariableArray3D(Range range0, Range range1, Range range2)
            : base(new Variable<T>(), range0, range1, range2)
        {
        }

        internal VariableArray3D(Variable<T> itemPrototype, Range range0, Range range1, Range range2)
            : base(itemPrototype, range0, range1, range2)
        {
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        protected VariableArray3D(VariableArray3D<T> that)
            : base(that)
        {
        }

        /// <summary>
        /// Set the name of the array.
        /// </summary>
        /// <param name="name"></param>
        /// <returns><c>this</c></returns>
        public new VariableArray3D<T> Named(string name)
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
        public new VariableArray3D<T> Attrib(ICompilerAttribute attr)
        {
            base.Attrib(attr);
            return this;
        }

        /// <summary>
        /// Set the 3-D array to a specified value
        /// </summary>
        /// <param name="that"></param>
        public void SetTo(VariableArray3D<T> that)
        {
            base.SetTo(that);
        }

        /// <summary>
        /// Clone the 3-D array
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            return new VariableArray3D<T>(this);
        }

        /// <summary>
        /// Sets/Gets element in array given by the index expressions
        /// </summary>
        /// <param name="index0">First index expression</param>
        /// <param name="index1">Second index expression</param>
        /// <param name="index2">Third index expression</param>
        /// <returns></returns>
        Variable<T> IVariableArray3D<T>.this[IModelExpression index0, IModelExpression index1, IModelExpression index2]
        {
            get { return this[index0, index1, index2]; }
            set { this[(Range) index0, (Range) index1, (Range) index2] = value; }
        }
    }
}