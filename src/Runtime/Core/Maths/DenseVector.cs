// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Probabilistic.Math
{
    using Microsoft.ML.Probabilistic.Serialization;
    using System.Runtime.Serialization;

    /// <summary>
    /// 1-dimensional dense container of double precision data that supports vector operations.
    /// </summary>
    [Serializable]
    [DataContract]
    public class DenseVector : Vector, ICursor, ICloneable, SettableTo<Vector>
    {
#pragma warning disable 1591
        // Storage for the elements, which may be in the middle of the array.
        [DataMember]
        protected double[] data;
        // The starting position of the vector elements in the data array.
        [DataMember]
        protected int start;
        // Number of elements in the vector.
        // This is not necessarily the same as data.Length.
        [DataMember]
        protected readonly int count;
#pragma warning restore 1591

        #region XML Serialization

        /// <summary>
        /// Read a DenseVector from XML of the form <example>&lt;DenseVector&gt;&lt;double&gt;1.2&lt;/double&gt;...&lt;/DenseVector&gt;</example>
        /// </summary>
        /// <param name="reader"></param>
        /// <returns>DenseVector</returns>
        internal static DenseVector FromXml(System.Xml.XmlReader reader)
        {
            reader.Read();
            List<double> list = new List<double>();
            while (reader.NodeType != System.Xml.XmlNodeType.EndElement)
            {
                reader.ReadStartElement("double");
                double x = reader.ReadContentAsDouble();
                list.Add(x);
                reader.ReadEndElement();
            }
            reader.ReadEndElement();
            return DenseVector.FromArray(list.ToArray());
        }

        /// <summary>
        /// Writer XML of the form <example>&lt;DenseVector&gt;&lt;double&gt;1.2&lt;/double&gt;...&lt;/DenseVector&gt;</example>
        /// </summary>
        /// <param name="writer"></param>
        internal void WriteXml(System.Xml.XmlWriter writer)
        {
            writer.WriteStartElement("DenseVector");
            for (int i = 0; i < count; i++)
            {
                writer.WriteStartElement("double");
                writer.WriteValue(this[i]);
                writer.WriteEndElement();
            }
            writer.WriteEndElement();
        }

        #endregion

        #region Factory methods and constructors

        /// <summary>
        /// Create a dense vector of given length with elements all 0.0
        /// </summary>
        /// <param name="count">Number of elements in vector</param>
        /// <returns></returns>
        public new static DenseVector Zero(int count)
        {
            return new DenseVector(count);
        }

        /// <summary>
        /// Create a dense vector of given length with elements all equal
        /// to a specified value
        /// </summary>
        /// <param name="count">Number of elements in vector</param>
        /// <param name="value">value for each element</param>
        /// <returns></returns>
        public new static DenseVector Constant(int count, double value)
        {
            DenseVector v = new DenseVector(count);
            v.SetAllElementsTo(value);
            return v;
        }

        /// <summary>
        /// Creates a dense vector as a copy of another vector
        /// </summary>
        /// <param name="that">The source vector - can be dense or sparse</param>
        public new static DenseVector Copy(Vector that)
        {
            DenseVector v = new DenseVector(that.Count);
            v.SetTo(that);
            return v;
        }

        /// <summary>
        /// Constructs a dense vector from an array.
        /// </summary>
        /// <param name="data">1D array of elements.</param>
        /// <remarks>The array data is copied into new storage.
        /// The size of the vector is taken from the array.
        /// </remarks>
        [Construction("ToArray")]
        public new static DenseVector FromArray(params double[] data)
        {
            return new DenseVector(data);
        }

        /// <summary>
        /// Constructs a vector from part of an array.
        /// </summary>
        /// <param name="data">Storage for the vector elements.</param>
        /// <param name="count">The number of elements in the vector.</param>
        /// <param name="start">The starting index in the array for the vector elements.</param>
        /// <remarks><para>
        /// Throws an exception if Data is null, start &lt; 0, or count &lt; 0.
        /// </para></remarks>
        public new static DenseVector FromArray(int count, double[] data, int start)
        {
            if (data == null)
            {
                throw new ArgumentNullException("Data");
            }
            if (count < 0)
            {
                throw new ArgumentOutOfRangeException("Count",
                                                      count,
                                                      "Count cannot be negative");
            }
            DenseVector v = new DenseVector(count);
            v.SetToSubarray(data, start);
            return v;
        }

        /// <summary>
        /// Constructs a vector by referencing an array.
        /// </summary>
        /// <param name="data">Storage for the vector elements.</param>
        /// <param name="count">The number of elements in the vector.</param>
        /// <param name="start">The starting index in the array for the vector elements.</param>
        /// <remarks><para>
        /// The vector will not copy the array but only reference it, 
        /// so any numerical changes to the array will also apply to the vector.
        /// If the array grows larger, the extra elements are ignored.
        /// The array must not shrink or else the vector will become inconsistent.
        /// </para><para>
        /// Throws an exception if Data is null, start &lt; 0, or count &lt; 0.
        /// </para></remarks>
        public static DenseVector FromArrayReference(int count, double[] data, int start)
        {
            if (data == null)
            {
                throw new ArgumentNullException("Data");
            }
            if (count < 0)
            {
                throw new ArgumentOutOfRangeException("Count",
                                                      count,
                                                      "Count cannot be negative");
            }
            DenseVector v = new DenseVector(count);
            v.data = data;
            v.start = start;
            return v;
        }

        #endregion

        /// <summary>Gets and sets an element.</summary>
        public override double this[int index]
        {
            get { return (data[index + start]); }
            set { data[index + start] = value; }
        }

        /// <summary>
        /// Gets/sets source array for the vector
        /// </summary>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Performance", "CA1819:PropertiesShouldNotReturnArrays")]
        [IgnoreDataMember]
        public double[] SourceArray
        {
            get { return data; }
            set { data = value; }
        }

        #region IList<double> Members

        /// <summary>
        /// Returns the index of the first occurence of the given value in the vector.
        /// Returns -1 if the value is not in the array
        /// </summary>
        /// <param name="item">The item to check for</param>
        /// <returns>Its index in the vector</returns>
        public override int IndexOf(double item)
        {
            double end = start + count;
            for (int i = start, j = 0; i < end; i++, j++)
            {
                if (this.data[i] == item) return j;
            }
            return -1;
        }

        /// <summary>
        /// Is read only
        /// </summary>
        public override bool IsReadOnly
        {
            get { return false; }
        }

        /// <summary>
        /// Returns true if the Vector contains the specified item value
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public override bool Contains(double item)
        {
            double end = start + count;
            for (int i = start; i < end; i++)
            {
                if (this.data[i] == item) return true;
            }
            return false;
        }

        /// <summary>
        /// Copies this vector to the given array starting at the specified index
        /// in the target array
        /// </summary>
        /// <param name="array">The target array</param>
        /// <param name="index">The start index in the target array</param>
        public override void CopyTo(double[] array, int index)
        {
            Array.Copy(data, start, array, index, count);
        }

        /// <summary>
        /// Gets a typed enumerator which yields the vector elements
        /// </summary>
        /// <returns></returns>
        public override IEnumerator<double> GetEnumerator()
        {
            double end = start + count;
            for (int i = start; i < end; i++)
                yield return this.data[i];
        }

        #endregion

        #region LINQ-like operators

        /// <inheritdoc/>
        public override bool All(Func<double, bool> fun)
        {
            int end = start + count;
            for (int i = start; i < end; ++i)
            {
                if (!fun(data[i]))
                    return false;
            }
            return true;
        }

        /// <inheritdoc/>
        public override bool Any(Func<double, bool> fun)
        {
            int end = start + count;
            for (int i = start; i < end; ++i)
            {
                if (fun(data[i]))
                    return true;
            }
            return false;
        }

        /// <inheritdoc/>
        public override bool Any(Vector that, Func<double, double, bool> fun)
        {
            if (that.Sparsity == Sparsity.Dense)
                return Any((DenseVector)that, fun);

            CheckCompatible(that, nameof(that));
            IEnumerator<double> thatEnum = that.GetEnumerator();
            int i = start;
            while (thatEnum.MoveNext())
            {
                if (fun(data[i], thatEnum.Current)) return true;
                i++;
            }
            return false;
        }

        public bool Any(DenseVector that, Func<double, double, bool> fun)
        {
            CheckCompatible(that, nameof(that));
            int end = start + count;
            for (int i = start, j = that.start; i < end; ++i, j++)
            {
                if (fun(data[i], that.data[j])) return true;
            }
            return false;
        }

        /// <inheritdoc/>
        public override IEnumerable<ValueAtIndex<double>> FindAll(Func<double, bool> fun)
        {
            if (fun == null)
            {
                throw new ArgumentNullException(nameof(fun));
            }

            int end = this.start + this.count;
            for (int i = this.start, j = 0; i < end; ++i, j++)
            {
                if (fun(this.data[i]))
                {
                    yield return new ValueAtIndex<double>(j, this.data[i]);
                }
            }
        }

        /// <inheritdoc/>
        public override int CountAll(Func<double, bool> fun)
        {
            if (fun == null)
            {
                throw new ArgumentNullException(nameof(fun));
            }

            int end = this.start + this.count;
            int result = 0;
            for (int i = this.start; i < end; ++i)
            {
                if (fun(this.data[i]))
                {
                    ++result;
                }
            }

            return result;
        }

        /// <inheritdoc/>
        public override int FindFirstIndex(Func<double, bool> fun)
        {
            return Array.FindIndex(this.data, elt => fun(elt));
        }

        /// <inheritdoc/>
        public override int FindLastIndex(Func<double, bool> fun)
        {
            return Array.FindLastIndex(this.data, elt => fun(elt));
        }

        #endregion

        #region ICursor methods

        /// <summary>
        /// Number of elements in vector
        /// </summary>
        public override int Count
        {
            get { return count; }
            protected set { throw new NotSupportedException(); }
        }

        /// <summary>
        /// Gets/sets the start index in the source array
        /// </summary>
        [IgnoreDataMember]
        public int Start
        {
            get { return start; }
            set
            {
                if (value < 0)
                {
                    throw new ArgumentOutOfRangeException("Start",
                                                          Start,
                                                          "Start cannot be negative");
                }
                Assert.IsTrue(value + count <= data.Length);
                start = value;
            }
        }

        /// <summary>
        /// Creates a source array with a given number of records
        /// </summary>
        /// <param name="nRecords"></param>
        public void CreateSourceArray(int nRecords)
        {
            data = new double[count * nRecords];
            start = 0;
        }

        /// <summary>
        /// Creates a clone of this instance which references the source array
        /// </summary>
        /// <returns></returns>
        public ICursor ReferenceClone()
        {
            return new DenseVector(count, data, start);
        }

        #endregion

        #region Creation methods

        /// <summary>
        /// Converts this vector to an array of doubles
        /// </summary>
        /// <returns></returns>
        public override double[] ToArray()
        {
            double[] result = new double[count];
            CopyTo(result, 0);
            return result;
        }

        // this constructor is preferred for serialization.
        /// <summary>
        /// Constructs a zero vector with the given number of elements.
        /// </summary>
        /// <param name="count">Number of elements to allocate (>= 0).</param>
        private DenseVector(int count)
        {
            data = new double[count];
            // Not required: start  = 0;
            this.count = count;
            Sparsity = Sparsity.Dense;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        private DenseVector(Vector that)
            : this(that.Count)
        {
            SetTo(that);
        }

        private DenseVector() { }

        /// <summary>
        /// Constructs a vector from an array.
        /// </summary>
        /// <param name="data">1D array of elements.</param>
        /// <remarks>The array data is copied into new storage.
        /// The size of the vector is taken from the array.
        /// </remarks>
        private DenseVector(params double[] data)
        {
            this.data = (double[])data.Clone();
            // Not required: start  = 0;
            count = data.Length;
            Sparsity = Sparsity.Dense;
        }

        /// <summary>
        /// Constructs a vector by referencing an array.
        /// </summary>
        /// <param name="data">Storage for the vector elements.</param>
        /// <param name="count">The number of elements in the vector.</param>
        /// <param name="start">The starting index in the array for the vector elements.</param>
        /// <remarks><para>
        /// The vector will not copy the array but only reference it, 
        /// so any numerical changes to the array will also apply to the vector.
        /// If the array grows larger, the extra elements are ignored.
        /// The array must not shrink or else the vector will become inconsistent.
        /// </para><para>
        /// Throws an exception if Data is null, start &lt; 0, or count &lt; 0.
        /// </para></remarks>
        private DenseVector(int count, double[] data, int start)
        {
            if (data == null)
            {
                throw new ArgumentNullException("Data");
            }
            if (count < 0)
            {
                throw new ArgumentOutOfRangeException("Count",
                                                      Count,
                                                      "Count cannot be negative");
            }

            this.data = data;
            this.count = count;
            this.start = start;
            Sparsity = Sparsity.Dense;
        }

        /// <summary>
        /// Constructs a vector of a given length and assigns all elements the given value
        /// </summary>
        /// <param name="count"></param>
        /// <param name="value"></param>
        private DenseVector(int count, double value)
            : this(count)
        {
            SetAllElementsTo(value);
        }

        /// <summary>
        /// Appends an item to a vector - returns a new vector
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public override Vector Append(double item)
        {
            DenseVector result = DenseVector.Zero(count + 1);
            Array.Copy(data, start, result.data, result.start, count);
            result[count] = item;
            return result;
        }

        /// <summary>
        /// Returns a new vector which appends a second vector to this vector
        /// </summary>
        /// <param name="second">Second vector</param>
        /// <returns></returns>
        public override Vector Append(Vector second)
        {
            DenseVector result = DenseVector.Zero(count + second.Count);
            Array.Copy(data, start, result.data, 0, count);
            if (second.Sparsity == Sparsity.Dense)
                return Append((DenseVector)second);

            IEnumerator<double> secondEnum = second.GetEnumerator();
            int i = count;
            while (secondEnum.MoveNext())
                result[i++] = secondEnum.Current;

            return result;
        }

        /// <summary>
        /// Returns a new vector which appends a second dense vector to this dense vector
        /// </summary>
        /// <param name="second">Second vector</param>
        /// <returns></returns>
        public DenseVector Append(DenseVector second)
        {
            DenseVector result = DenseVector.Zero(count + second.Count);
            Array.Copy(data, start, result.data, 0, count);
            Array.Copy(second.data, second.start, result.data, count, second.count);
            return result;
        }

        #endregion

        #region Copying

        /// <summary>
        /// Clones this vector - return as a vector
        /// </summary>
        /// <returns></returns>
        public override Vector Clone()
        {
            return new DenseVector(this);
        }

        /// <summary>
        /// Clones this vector - return as an object
        /// </summary>
        /// <returns></returns>
        object ICloneable.Clone()
        {
            return Clone();
        }


        /// <summary>
        /// Sets all elements to a given value.
        /// </summary>
        /// <param name="value">The new value.</param>
        // SetAll might be a better name.
        public override void SetAllElementsTo(double value)
        {
            int end = start + count;
            for (int i = start; i < end; ++i)
                this.data[i] = value;
        }

        /// <summary>
        /// Copies values from an array.
        /// </summary>
        /// <param name="values">An array whose length matches <c>this.Count</c>.</param>
        public override void SetTo(double[] values)
        {
            if (count != values.Length) throw new ArgumentException("array does not match the Vector length", nameof(values));
            SetToSubarray(values, 0);
        }

        /// <summary>
        /// Copies values from a vector to this vector
        /// </summary>
        /// <param name="that"></param>
        public override void SetTo(Vector that)
        {
            if (that.Sparsity == Sparsity.Dense)
            {
                SetTo((DenseVector)that);
                return;
            }

            if (!Object.ReferenceEquals(this, that))
            {
                CheckCompatible(that, nameof(that));
                IEnumerator<double> thatEnum = that.GetEnumerator();
                int i = start;
                while (thatEnum.MoveNext())
                    this.data[i++] = thatEnum.Current;
            }
        }

        /// <summary>
        /// Copies values from a dense vector to this dense vector
        /// </summary>
        /// <param name="that"></param>
        public void SetTo(DenseVector that)
        {
            if (!Object.ReferenceEquals(this, that))
            {
                CheckCompatible(that, nameof(that));
                Array.Copy(that.data, that.start, data, start, count);
            }
        }

        /// <summary>
        /// Copies values from an Enumerable to this vector
        /// </summary>
        /// <param name="that"></param>
        public override void SetTo(IEnumerable<double> that)
        {
            if (!Object.ReferenceEquals(this, that))
            {
                if (that is Vector)
                    this.SetTo(that as Vector);
                else
                    SetTo(that.ToArray());
            }
        }

        /// <summary>
        /// Copies values from an array.
        /// </summary>
        /// <param name="values">An array whose length is at least <c>this.Count + startIndex</c>.</param>
        /// <param name="startIndex">The index of the first value in <paramref name="values"/> to copy.</param>
        public override void SetToSubarray(double[] values, int startIndex)
        {
            Array.Copy(values, startIndex, data, start, count);
        }

        /// <summary>
        /// Copies value from a vector.
        /// </summary>
        /// <param name="that">A vector whose length is at least <c>this.Count + startIndex</c>.</param>
        /// <param name="startIndex">The index of the first value in <paramref name="that"/> to copy.</param>
        public override void SetToSubvector(Vector that, int startIndex)
        {
            if (that.Sparsity == Sparsity.Dense)
            {
                SetToSubvector((DenseVector)that, startIndex);
                return;
            }
            int end = start + count;
            for (int i = start, j = startIndex; i < end; i++)
            {
                this.data[i] = that[j++];
            }
        }

        /// <summary>
        /// Copies value from a vector.
        /// </summary>
        /// <param name="that">A dense vector whose length is at least <c>this.Count + startIndex</c>.</param>
        /// <param name="startIndex">The index of the first value in <paramref name="that"/> to copy.</param>
        public void SetToSubvector(DenseVector that, int startIndex)
        {
            Array.Copy(that.data, startIndex + that.start, data, start, count);
        }

        /// <summary>
        /// Create a subvector of this vector
        /// </summary>
        /// <param name="startIndex"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        public override Vector Subvector(int startIndex, int count)
        {
            DenseVector result = new DenseVector(count);
            result.SetToSubvector(this, startIndex);
            return result;
        }

        /// <summary>
        /// Set a subvector of this to another vector.
        /// </summary>
        /// <param name="startIndex">The index of the first element of this to copy to.</param>
        /// <param name="that">A vector whose length is at most <c>this.Count - startIndex</c>.</param>
        public override void SetSubvector(int startIndex, Vector that)
        {
            if (that.Sparsity == Sparsity.Dense)
            {
                SetSubvector(startIndex, (DenseVector)that);
                return;
            }
            if (startIndex + that.Count > count)
                throw new ArgumentException("startIndex (" + startIndex + ") + that.Count (" + that.Count + ") > this.Count (" + this.Count + ")");
            int end = start + startIndex + that.Count;
            for (int i = start + startIndex, j = 0; i < end; i++)
                this.data[i] = that[j++];
        }

        /// <summary>
        /// Set a subvector of this to another vector.
        /// </summary>
        /// <param name="startIndex">The index of the first element of to copy to.</param>
        /// <param name="that">A dense vector whose length is at most <c>this.Count - startIndex</c>.</param>
        public void SetSubvector(int startIndex, DenseVector that)
        {
            if (startIndex + that.Count > count)
                throw new ArgumentException("startIndex (" + startIndex + ") + that.Count (" + that.Count + ") > this.Count (" + this.Count + ")");
            Array.Copy(that.data, that.start, data, start + startIndex, that.count);
        }

        #endregion

        #region sum, difference etc.

        /// <summary>
        /// Sets the elements of this vector to a function of the elements of a given vector
        /// </summary>
        /// <param name="fun">The function which maps doubles to doubles</param>
        /// <param name="that">The given vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public override Vector SetToFunction(Vector that, Converter<double, double> fun)
        {
            if (that.Sparsity == Sparsity.Dense)
                return SetToFunction((DenseVector)that, fun);

            CheckCompatible(that, nameof(that));
            IEnumerator<double> thatEnum = that.GetEnumerator();
            int i = start;
            while (thatEnum.MoveNext())
                this.data[i++] = fun(thatEnum.Current);

            return (this);
        }

        /// <summary>
        /// Sets the elements of this vector to a function of the elements of a given vector
        /// </summary>
        /// <param name="fun">The function which maps doubles to doubles</param>
        /// <param name="that">The given vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public DenseVector SetToFunction(DenseVector that, Converter<double, double> fun)
        {
            CheckCompatible(that, nameof(that));
            int end = start + count;
            for (int i = start, j = that.start; i < end; i++)
                this.data[i] = fun(that.data[j++]);
            return (this);
        }

        /// <summary>
        /// Sets the elements of this vector to a function of the elements of two vectors
        /// </summary>
        /// <param name="fun">The function which maps two doubles to a double</param>
        /// <param name="a">The first vector</param>
        /// <param name="b">The second vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public override Vector SetToFunction(Vector a, Vector b, Func<double, double, double> fun)
        {
            if (a.Sparsity == Sparsity.Dense && b.Sparsity == Sparsity.Dense)
                return SetToFunction((DenseVector)a, (DenseVector)b, fun);

            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            IEnumerator<double> aEnum = a.GetEnumerator();
            IEnumerator<double> bEnum = b.GetEnumerator();
            int i = start;
            while (aEnum.MoveNext() && bEnum.MoveNext())
                this.data[i++] = fun(aEnum.Current, bEnum.Current);

            return (this);
        }

        /// <summary>
        /// Sets the elements of this vector to a function of the elements of two vectors
        /// </summary>
        /// <param name="fun">The function which maps two doubles to a double</param>
        /// <param name="a">The first vector</param>
        /// <param name="b">The second vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public DenseVector SetToFunction(DenseVector a, DenseVector b, Func<double, double, double> fun)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            int end = start + count;
            for (int i = start, j = a.start, k = b.start; i < end; i++)
                this.data[i] = fun(a.data[j++], b.data[k++]);
            return (this);
        }

        /// <summary>
        /// Sets this vector to the elementwise power of another vector.
        /// </summary>
        /// <param name="that">A vector, which must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <param name="exponent">A scalar.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.  
        /// If <c>this</c> and <paramref name="that"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public override void SetToPower(Vector that, double exponent)
        {
            if (that.Sparsity == Sparsity.Dense)
            {
                SetToPower((DenseVector)that, exponent);
                return;
            }

            CheckCompatible(that, nameof(that));
            IEnumerator<double> aEnum = that.GetEnumerator();
            int i = start;
            while (aEnum.MoveNext())
                this.data[i++] = System.Math.Pow(aEnum.Current, exponent);
        }

        /// <summary>
        /// Sets this vector to the elementwise power of another vector.
        /// </summary>
        /// <param name="that">A vector, which must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <param name="exponent">A scalar.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.  
        /// If <c>this</c> and <paramref name="that"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public void SetToPower(DenseVector that, double exponent)
        {
            CheckCompatible(that, nameof(that));
            int end = start + count;
            DenseVector thatd = (DenseVector)that;
            for (int i = start, j = that.start; i < end; i++)
                this.data[i] = System.Math.Pow(that.data[j++], exponent);
        }

        /// <summary>
        /// Sets this dense vector to the elementwise product of two other vectors.
        /// </summary>
        /// <param name="a">Must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <param name="b">Must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.  
        /// If <c>this</c> and <paramref name="a"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public override void SetToProduct(Vector a, Vector b)
        {
            if (a.Sparsity == Sparsity.Dense && b.Sparsity == Sparsity.Dense)
            {
                SetToProduct((DenseVector)a, (DenseVector)b);
                return;
            }
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            IEnumerator<double> aEnum = a.GetEnumerator();
            IEnumerator<double> bEnum = b.GetEnumerator();
            int i = start;
            while (aEnum.MoveNext() && bEnum.MoveNext())
                this.data[i++] = aEnum.Current * bEnum.Current;
        }

        /// <summary>
        /// Sets this dense vector to the elementwise product of two other dense vectors.
        /// </summary>
        /// <param name="a">Must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <param name="b">Must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.  
        /// If <c>this</c> and <paramref name="a"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public void SetToProduct(DenseVector a, DenseVector b)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            int end = start + count;
            for (int i = start, j = a.start, k = b.start; i < end; i++)
                this.data[i] = a.data[j++] * b.data[k++];
        }

        /// <summary>
        /// Set this vector to a linear combination of two other vectors
        /// </summary>
        /// <param name="aScale">The multiplier for vector a</param>
        /// <param name="a">Vector a</param>
        /// <param name="bScale">The multiplier for vector b</param>
        /// <param name="b">Vector b</param>
        public override void SetToSum(double aScale, Vector a, double bScale, Vector b)
        {
            if (a.Sparsity == Sparsity.Dense && b.Sparsity == Sparsity.Dense)
            {
                SetToSum(aScale, (DenseVector)a, bScale, (DenseVector)b);
                return;
            }
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            IEnumerator<double> aEnum = a.GetEnumerator();
            IEnumerator<double> bEnum = b.GetEnumerator();
            int i = start;
            while (aEnum.MoveNext() && bEnum.MoveNext())
                this.data[i++] = aScale * aEnum.Current + bScale * bEnum.Current;
        }

        /// <summary>
        /// Set this vector to a linear combination of two other vectors
        /// </summary>
        /// <param name="aScale">The multiplier for vector a</param>
        /// <param name="a">Vector a</param>
        /// <param name="bScale">The multiplier for vector b</param>
        /// <param name="b">Vector b</param>
        public void SetToSum(double aScale, DenseVector a, double bScale, DenseVector b)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            int end = start + count;
            for (int i = start, j = a.start, k = b.start; i < end; i++)
                this.data[i] = aScale * a.data[j++] + bScale * b.data[k++];
        }

        /// <summary>
        /// Sets this vector to the elementwise sum of two other vectors.
        /// </summary>
        /// <param name="a">First vector, which must have the same size as <c>this</c>.</param>
        /// <param name="b">Second vector, which must have the same size as <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the sum, and must already be the correct size.
        /// <paramref name="a"/> and/or <paramref name="b"/> may be the same object as <c>this</c>.
        /// If <c>this</c> and <paramref name="a"/>/<paramref name="b"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public override Vector SetToSum(Vector a, Vector b)
        {
            if (a.Sparsity == Sparsity.Dense && b.Sparsity == Sparsity.Dense)
                return SetToSum((DenseVector)a, (DenseVector)b);

            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            IEnumerator<double> aEnum = a.GetEnumerator();
            IEnumerator<double> bEnum = b.GetEnumerator();
            int i = start;
            while (aEnum.MoveNext() && bEnum.MoveNext())
                this.data[i++] = aEnum.Current + bEnum.Current;

            return this;
        }

        /// <summary>
        /// Sets this dense vector to the elementwise sum of two other dense vectors.
        /// </summary>
        /// <param name="a">First vector, which must have the same size as <c>this</c>.</param>
        /// <param name="b">Second vector, which must have the same size as <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the sum, and must already be the correct size.
        /// <paramref name="a"/> and/or <paramref name="b"/> may be the same object as <c>this</c>.
        /// If <c>this</c> and <paramref name="a"/>/<paramref name="b"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public DenseVector SetToSum(DenseVector a, DenseVector b)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            int end = start + count;
            for (int i = start, j = a.start, k = b.start; i < end; i++)
                this.data[i] = a.data[j++] + b.data[k++];
            return this;
        }

        /// <summary>
        /// Sets this vector to another vector plus a scalar.
        /// </summary>
        /// <param name="a">A vector, which must have the same size as <c>this</c>.  Can be the same object as <c>this</c>.</param>
        /// <param name="b">A scalar.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the sum, and must already be the correct size.
        /// If <c>this</c> and <paramref name="a"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public override Vector SetToSum(Vector a, double b)
        {
            if (a.Sparsity == Sparsity.Dense)
                return SetToSum((DenseVector)a, b);

            CheckCompatible(a, nameof(a));
            IEnumerator<double> aEnum = a.GetEnumerator();
            int i = start;
            while (aEnum.MoveNext())
                this.data[i++] = aEnum.Current + b;
            return this;
        }

        /// <summary>
        /// Sets this dense vector to another dense vector plus a scalar.
        /// </summary>
        /// <param name="a">A vector, which must have the same size as <c>this</c>.  Can be the same object as <c>this</c>.</param>
        /// <param name="b">A scalar.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the sum, and must already be the correct size.
        /// If <c>this</c> and <paramref name="a"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public DenseVector SetToSum(DenseVector a, double b)
        {
            CheckCompatible(a, nameof(a));
            int end = start + count;
            for (int i = start, j = a.start; i < end; i++)
                this.data[i] = a.data[j++] + b;
            return this;
        }

        /// <summary>
        /// Sets this vector to the difference of two vectors
        /// </summary>
        /// <param name="a">First vector, which must have the same size as <c>this</c>.</param>
        /// <param name="b">Second vector, which must have the same size as <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks>
        /// <c>this</c> receives the difference, and must already be the correct size.
        /// <paramref name="a"/> and/or <paramref name="b"/> may be the same object as <c>this</c>.
        /// If <c>this</c> and <paramref name="a"/>/<paramref name="b"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public override Vector SetToDifference(Vector a, Vector b)
        {
            if (a.Sparsity == Sparsity.Dense && b.Sparsity == Sparsity.Dense)
                return SetToDifference((DenseVector)a, (DenseVector)b);
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            IEnumerator<double> aEnum = a.GetEnumerator();
            IEnumerator<double> bEnum = b.GetEnumerator();
            int i = start;
            while (aEnum.MoveNext() && bEnum.MoveNext())
                this.data[i++] = aEnum.Current - bEnum.Current;

            return this;
        }

        /// <summary>
        /// Sets this dense vector to the difference of two othe dense vectors
        /// </summary>
        /// <param name="a">First vector, which must have the same size as <c>this</c>.</param>
        /// <param name="b">Second vector, which must have the same size as <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks>
        /// <c>this</c> receives the difference, and must already be the correct size.
        /// <paramref name="a"/> and/or <paramref name="b"/> may be the same object as <c>this</c>.
        /// If <c>this</c> and <paramref name="a"/>/<paramref name="b"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public DenseVector SetToDifference(DenseVector a, DenseVector b)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            int end = start + count;
            for (int i = start, j = a.start, k = 0; i < end; i++)
                this.data[i] = a.data[j++] - b[k++];
            return this;
        }

        /// <summary>
        /// Set this vector to another vector minus a constant
        /// </summary>
        /// <param name="a">The other vector</param>
        /// <param name="b">The constant</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public override Vector SetToDifference(Vector a, double b)
        {
            if (a.Sparsity == Sparsity.Dense)
                return SetToDifference((DenseVector)a, b);

            CheckCompatible(a, nameof(a));
            IEnumerator<double> aEnum = a.GetEnumerator();
            int i = start;
            while (aEnum.MoveNext())
                this.data[i++] = aEnum.Current - b;
            return this;
        }

        /// <summary>
        /// Set this dense vector to another dense vector minus a constant
        /// </summary>
        /// <param name="a">The other vector</param>
        /// <param name="b">The constant</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public DenseVector SetToDifference(DenseVector a, double b)
        {
            CheckCompatible(a, nameof(a));
            int end = start + count;
            for (int i = start, j = a.start; i < end; i++)
                this.data[i] = a.data[j++] - b;
            return this;
        }

        /// <summary>
        /// Set this vector to a constant minus another vector
        /// </summary>
        /// <param name="a">The constant</param>
        /// <param name="b">The other vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public override Vector SetToDifference(double a, Vector b)
        {
            if (b.Sparsity == Sparsity.Dense)
                return SetToDifference(a, (DenseVector)b);

            CheckCompatible(b, nameof(b));
            IEnumerator<double> bEnum = b.GetEnumerator();
            int i = start;
            while (bEnum.MoveNext())
                this.data[i++] = a - bEnum.Current;
            return this;
        }

        /// <summary>
        /// Set this dense vector to a constant minus another dense vector
        /// </summary>
        /// <param name="a">The constant</param>
        /// <param name="b">The other vector</param>
        /// <returns></returns>
        /// <remarks>Assumes the vectors are compatible</remarks>
        public DenseVector SetToDifference(double a, DenseVector b)
        {
            CheckCompatible(b, nameof(b));
            int end = start + count;
            for (int i = start, j = b.start; i < end; i++)
                this.data[i] = a - b.data[j++];
            return this;
        }

        /// <summary>
        /// Sets this vector to a vector times a scalar.
        /// </summary>
        /// <param name="a">A vector, which must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <param name="b">A scalar.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.  
        /// If <c>this</c> and <paramref name="a"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public override Vector SetToProduct(Vector a, double b)
        {
            if (a.Sparsity == Sparsity.Dense)
                return SetToProduct((DenseVector)a, b);

            CheckCompatible(a, nameof(a));
            IEnumerator<double> aEnum = a.GetEnumerator();
            int i = start;
            while (aEnum.MoveNext())
                this.data[i++] = aEnum.Current * b;
            return this;
        }

        /// <summary>
        /// Sets this dense vector to a dense vector times a scalar.
        /// </summary>
        /// <param name="a">A vector, which must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <param name="b">A scalar.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.  
        /// If <c>this</c> and <paramref name="a"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public DenseVector SetToProduct(DenseVector a, double b)
        {
            CheckCompatible(a, nameof(a));
            int end = start + count;
            for (int i = start, j = a.start; i < end; i++)
                this.data[i] = a.data[j++] * b;
            return this;
        }

        /// <summary>
        /// Sets this vector to the product of a vector by a matrix (i.e. x*A).
        /// </summary>
        /// <param name="x">A vector.  Cannot be <c>this</c>.</param>
        /// <param name="A">A matrix.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.
        /// If <c>this</c> and <paramref name="A"/>/<paramref name="x"/> occupy overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public override Vector SetToProduct(Vector x, Matrix A)
        {
            if (x.Sparsity == Sparsity.Dense)
                return SetToProduct((DenseVector)x, A);

            Assert.IsTrue(!object.ReferenceEquals(this, x));
            if (x.Count != A.Rows)
            {
                throw new ArgumentException("Incompatible matrix/vector dimensions", nameof(x));
            }
            if (count != A.Cols)
            {
                throw new ArgumentException("Output vector is incompatible with the product", nameof(A));
            }

            for (int i = 0; i < A.Cols; ++i)
            {
                double sum = 0.0;
                for (int j = 0, k = 0; j < A.Rows; ++j)
                {
                    sum += A[j, i] * x[k++];
                }
                this[i] = sum;
            }
            return (this);
        }

        /// <summary>
        /// Sets this dense vector to the product of a dense vector by a matrix (i.e. x*A).
        /// </summary>
        /// <param name="x">A vector.  Cannot be <c>this</c>.</param>
        /// <param name="A">A matrix.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.
        /// If <c>this</c> and <paramref name="A"/>/<paramref name="x"/> occupy overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public DenseVector SetToProduct(DenseVector x, Matrix A)
        {
            Assert.IsTrue(!object.ReferenceEquals(this, x));
            if (x.Count != A.Rows)
            {
                throw new ArgumentException("Incompatible matrix/vector dimensions", nameof(x));
            }
            if (count != A.Cols)
            {
                throw new ArgumentException("Output vector is incompatible with the product", nameof(A));
            }

            for (int i = 0; i < A.Cols; ++i)
            {
                double sum = 0.0;
                for (int j = 0, k = x.start; j < A.Rows; ++j)
                {
                    sum += A[j, i] * x.data[k++];
                }
                this[i] = sum;
            }
            return (this);
        }

        /// <summary>
        /// Set this vector to the product of a matrix by a vector (i.e. A*x).
        /// </summary>
        /// <param name="A">A matrix.</param>
        /// <param name="x">A vector.  Cannot be <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.
        /// If <c>this</c> and <paramref name="A"/>/<paramref name="x"/> occupy overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public override Vector SetToProduct(Matrix A, Vector x)
        {
            if (x.Sparsity == Sparsity.Dense)
                return SetToProduct(A, (DenseVector)x);

            if (object.ReferenceEquals(this, x)) throw new ArgumentException("x is the same object as this", nameof(x));
            if (x.Count != A.Cols)
            {
                throw new ArgumentException("Incompatible matrix/vector dimensions", nameof(x));
            }
            if (count != A.Rows)
            {
                throw new ArgumentException("Output vector is incompatible with the product", nameof(A));
            }

            for (int i = 0; i < A.Rows; ++i)
            {
                double sum = 0.0;
                for (int j = 0, k = 0; j < A.Cols; ++j)
                {
                    sum += A[i, j] * x[k++];
                }
                this[i] = sum;
            }
            return (this);
        }

        /// <summary>
        /// Set this dense vector to the product of a matrix by a dense vector (i.e. A*x).
        /// </summary>
        /// <param name="A">A matrix.</param>
        /// <param name="x">A vector.  Cannot be <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.
        /// If <c>this</c> and <paramref name="A"/>/<paramref name="x"/> occupy overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public DenseVector SetToProduct(Matrix A, DenseVector x)
        {
            if (object.ReferenceEquals(this, x)) throw new ArgumentException("x is the same object as this", nameof(x));
            if (x.Count != A.Cols)
            {
                throw new ArgumentException("Incompatible matrix/vector dimensions", nameof(x));
            }
            if (count != A.Rows)
            {
                throw new ArgumentException("Output vector is incompatible with the product", nameof(A));
            }

            for (int i = 0; i < A.Rows; ++i)
            {
                double sum = 0.0;
                for (int j = 0, k = x.start; j < A.Cols; ++j)
                {
                    sum += A[i, j] * x.data[k++];
                }
                this[i] = sum;
            }
            return (this);
        }

        /// <summary>
        /// Sets this vector to the elementwise ratio of two other vectors.
        /// </summary>
        /// <param name="a">Must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <param name="b">Must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.  
        /// If <c>this</c> and <paramref name="a"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public override Vector SetToRatio(Vector a, Vector b)
        {
            if (a.Sparsity == Sparsity.Dense && b.Sparsity == Sparsity.Dense)
                return SetToRatio((DenseVector)a, (DenseVector)b);

            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            IEnumerator<double> aEnum = a.GetEnumerator();
            IEnumerator<double> bEnum = b.GetEnumerator();
            int i = start;
            while (aEnum.MoveNext() && bEnum.MoveNext())
                this.data[i++] = aEnum.Current / bEnum.Current;

            return this;
        }

        /// <summary>
        /// Sets this dense vector to the elementwise ratio of two other dense vectors.
        /// </summary>
        /// <param name="a">Must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <param name="b">Must have the same size as <c>this</c>.  Can be <c>this</c>.</param>
        /// <returns><c>this</c></returns>
        /// <remarks><c>this</c> receives the product, and must already be the correct size.  
        /// If <c>this</c> and <paramref name="a"/> occupy distinct yet overlapping portions of the same source array, the results are undefined.
        /// </remarks>
        public DenseVector SetToRatio(DenseVector a, DenseVector b)
        {
            CheckCompatible(a, nameof(a));
            CheckCompatible(b, nameof(b));
            int end = start + count;
            if (b.Sparsity == Sparsity.Dense)
            {
                for (int i = start, j = a.start, k = b.start; i < end; i++)
                    this.data[i] = a.data[j++] / b.data[k++];
            }
            return this;
        }

        #endregion

        #region Arithmetic operations

        /// <summary>
        /// Reduce method. Operates on this vector
        /// </summary>
        /// <param name="initial">Initial value</param>
        /// <param name="fun">Reduction function taking partial result and current element</param>
        /// <returns></returns>
        public override double Reduce(double initial, Func<double, double, double> fun)
        {
            int end = start + count;
            for (int i = start; i < end; i++)
                initial = fun(initial, this.data[i]);
            return initial;
        }

        /// <summary>
        /// Reduce method. Operates on this dense vector and that dense vector
        /// </summary>
        /// <param name="fun">Reduction function taking partial result, current element, and current element of <paramref name="that"/></param>
        /// <param name="initial">Initial value</param>
        /// <param name="that">A second vector</param>
        /// <returns></returns>
        public override double Reduce(double initial, Vector that, Func<double, double, double, double> fun)
        {
            if (that.Sparsity == Sparsity.Dense)
                return Reduce(initial, (DenseVector)that, fun);

            IEnumerator<double> thatEnum = that.GetEnumerator();
            int i = start;
            while (thatEnum.MoveNext())
                initial = fun(initial, this.data[i++], thatEnum.Current);

            return initial;
        }

        /// <summary>
        /// Reduce method. Operates on this vector and that vector
        /// </summary>
        /// <param name="fun">Reduction function taking partial result, current element, and current element of <paramref name="that"/></param>
        /// <param name="initial">Initial value</param>
        /// <param name="that">A second vector</param>
        /// <returns></returns>
        public double Reduce(double initial, DenseVector that, Func<double, double, double, double> fun)
        {
            int end = start + count;
            for (int i = start, j = that.start; i < end; i++)
                initial = fun(initial, this.data[i], that.data[j++]);
            return initial;
        }

        /// <summary>
        /// Reduce method. Operates on this vector and two other vectors
        /// </summary>
        /// <param name="fun">Reduction function</param>
        /// <param name="initial">Initial value</param>
        /// <param name="a">A second vector</param>
        /// <param name="b">A third vector</param>
        /// <returns></returns>
        public override double Reduce(double initial, Vector a, Vector b, Func<double, double, double, double, double> fun)
        {
            double result = initial;
            IEnumerator<double> thisEnum = GetEnumerator();
            IEnumerator<double> aEnum = a.GetEnumerator();
            IEnumerator<double> bEnum = b.GetEnumerator();

            while (thisEnum.MoveNext() && aEnum.MoveNext() && bEnum.MoveNext())
                result = fun(result, thisEnum.Current, aEnum.Current, bEnum.Current);
            return result;
        }

        /// <summary>
        /// Returns the sum of all elements.
        /// </summary>
        public override double Sum()
        {
            int end = start + count;
            double sum = 0.0;
            for (int i = start; i < end; i++)
                sum += data[i];
            return sum;
        }

        /// <summary>
        /// Returns the sum of a function of all elements.
        /// </summary>
        /// <param name="fun">Conversion function</param>
        public override double Sum(Converter<double, double> fun)
        {
            int end = start + count;
            double sum = 0.0;
            for (int i = start; i < end; i++)
                sum += fun(data[i]);
            return sum;
        }

        /// <summary>
        /// Returns the sum of a function of this vector filtered by a function of a second vector.
        /// </summary>
        /// <param name="fun">Function to convert the elements of this vector</param>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="cond">Function to convert the elements of that vector to give the filter condition</param>
        /// <returns>The filtered and mapped sum</returns>
        public override double Sum(Converter<double, double> fun, Vector that, Converter<double, bool> cond)
        {
            if (that.Sparsity == Sparsity.Dense)
                return Sum(fun, (DenseVector)that, cond);

            CheckCompatible(that, nameof(that));
            double sum = 0.0;

            IEnumerator<double> thatEnum = that.GetEnumerator();
            int i = start;
            while (thatEnum.MoveNext())
            {
                if (cond(thatEnum.Current))
                    sum += fun(this.data[i]);
                i++;
            }
            return sum;
        }

        /// <summary>
        /// Returns the sum of a function of this dense vector filtered by a function of a second dense vector.
        /// </summary>
        /// <param name="fun">Function to convert the elements of this vector</param>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="cond">Function to convert the elements of that vector to give the filter condition</param>
        /// <returns>The filtered and mapped sum</returns>
        public double Sum(Converter<double, double> fun, DenseVector that, Converter<double, bool> cond)
        {
            CheckCompatible(that, nameof(that));
            double sum = 0.0;
            int end = start + count;
            for (int i = start, j = that.start; i < end; i++)
                if (cond(that.data[j++]))
                    sum += fun(this.data[i]);
            return sum;
        }

        /// <summary>
        /// Returns the sum of over zero-based index times element.
        /// </summary>
        public override double SumI()
        {
            double result = 0.0;
            for (int i = 0, j = start; i < count; i++, j++)
                result += i * data[j];
            return result;
        }

        /// <summary>
        /// Returns the sum of over square of index^2 times element.
        /// </summary>
        public override double SumISq()
        {
            double result = 0.0;
            for (int i = 0, j = start; i < count; i++, j++)
                result += i * i * data[j];
            return result;
        }

        /// <summary>
        /// Returns the maximum of the elements in the vector
        /// </summary>
        /// <returns></returns>
        public override double Max()
        {
            int end = start + count;
            double max = data[start];
            for (int i = start + 1; i < end; i++)
                max = System.Math.Max(max, data[i]);
            return max;
        }

        /// <summary>
        /// Returns the maximum of a function of the elements in the vector
        /// </summary>
        /// <param name="fun">Conversion function</param>
        /// <returns></returns>
        public override double Max(Converter<double, double> fun)
        {
            int end = start + count;
            double max = fun(data[start]);
            for (int i = start + 1; i < end; i++)
                max = System.Math.Max(max, fun(data[i]));
            return max;
        }

        /// <summary>
        /// Returns the median of all elements.
        /// </summary>
        public override double Median()
        {
            return MMath.Median(data, start, count);
        }

        /// <summary>
        /// Returns the minimum of the elements in the vector
        /// </summary>
        /// <returns></returns>
        public override double Min()
        {
            int end = start + count;
            double min = data[start];
            for (int i = start + 1; i < end; i++)
                min = System.Math.Min(min, data[i]);
            return min;
        }

        /// <summary>
        /// Returns the minimum of the elements in the vector
        /// </summary>
        /// <param name="fun">Conversion function</param>
        /// <returns></returns>
        public override double Min(Converter<double, double> fun)
        {
            int end = start + count;
            double min = fun(data[start]);
            for (int i = start + 1; i < end; i++)
                min = System.Math.Min(min, fun(data[i]));
            return min;
        }

        /// <summary>
        /// Returns the log of the sum of exponentials of the elements of the vector
        /// computed to high accuracy
        /// </summary>
        /// <returns></returns>
        public override double LogSumExp()
        {
            double max = Max();
            int end = start + count;
            double Z = 0.0;
            for (int i = start; i < end; i++)
                Z += System.Math.Exp(data[i] - max);
            return System.Math.Log(Z) + max;
        }

        /// <summary>
        /// Returns the index of the minimum element.
        /// </summary>
        /// <returns>The index of the minimum element.</returns>
        public override int IndexOfMinimum()
        {
            if (count == 0) return -1;
            double min = this.data[start];
            int pos = 0;
            int end = start + count;
            for (int i = start + 1, j = 1; i < end; i++, j++)
            {
                if (min > this.data[i])
                {
                    min = this.data[i];
                    pos = j;
                }
            }
            return pos;
        }

        /// <summary>
        /// Returns the index of the maximum element.
        /// </summary>
        /// <returns>The index of the maximum element.</returns>
        public override int IndexOfMaximum()
        {
            if (count == 0) return -1;
            double max = this.data[start];
            int pos = 0;
            int end = start + count;
            for (int i = start + 1, j = 1; i < end; i++, j++)
            {
                if (max < this.data[i])
                {
                    max = this.data[i];
                    pos = j;
                }
            }
            return pos;
        }

        /// <summary>
        /// Returns the index of the first element at which the sum of all elements so far is greater
        /// than a particular value. Useful for finding the median of a Discrete distribution.
        /// </summary>
        /// <param name="targetSum">The sum of interest</param>
        /// <returns>
        /// The index of the element where <paramref name="targetSum"/> is exceeded 
        /// or -1 if <paramref name="targetSum"/> cannot be exceeded.
        /// </returns>
        public override int IndexAtCumulativeSum(double targetSum)
        {
            const int TargetSumNotExceeded = -1;

            int result = TargetSumNotExceeded;
            double sum = 0.0;

            for (int i = this.start; i < this.start + this.count; ++i)
            {
                sum += this.data[i];
                if (sum > targetSum)
                {
                    result = i - this.start;
                    break;
                }
            }

            return result;
        }

        /// <summary>
        /// Returns the inner product of this vector with another vector.
        /// </summary>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <returns>Their inner product.</returns>
        public override double Inner(Vector that)
        {
            if (that.Sparsity == Sparsity.Dense)
                return Inner((DenseVector)that);

            CheckCompatible(that, nameof(that));
            double inner = 0.0;

            IEnumerator<double> thatEnum = that.GetEnumerator();
            int i = start;
            while (thatEnum.MoveNext())
                inner += (this.data[i++] * thatEnum.Current);

            return inner;
        }

        /// <summary>
        /// Returns the inner product of this dense vector with another dense vector.
        /// </summary>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <returns>Their inner product.</returns>
        public double Inner(DenseVector that)
        {
            CheckCompatible(that, nameof(that));
            double inner = 0.0;
            int end = start + count;
            for (int i = start, j = that.start; i < end; i++)
                inner += (this.data[i] * that.data[j++]);
            return inner;
        }

        /// <summary>
        /// Returns the inner product of this vector with a function of a second vector.
        /// </summary>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="fun">Function to convert the elements of the second vector</param>
        /// <returns>Their inner product.</returns>
        public override double Inner(Vector that, Converter<double, double> fun)
        {
            if (that.Sparsity == Sparsity.Dense)
                return Inner((DenseVector)that, fun);

            CheckCompatible(that, nameof(that));
            double inner = 0.0;

            IEnumerator<double> thatEnum = that.GetEnumerator();
            int i = start;
            while (thatEnum.MoveNext())
                inner += (this.data[i++] * fun(thatEnum.Current));

            return inner;
        }

        /// <summary>
        /// Returns the inner product of this dense vector with a function of a second dense vector.
        /// </summary>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="fun">Function to convert the elements of the second vector</param>
        /// <returns>Their inner product.</returns>
        public double Inner(DenseVector that, Converter<double, double> fun)
        {
            CheckCompatible(that, nameof(that));
            double inner = 0.0;
            int end = start + count;
            for (int i = start, j = that.start; i < end; i++)
                inner += (this.data[i] * fun(that.data[j++]));
            return inner;
        }

        /// <summary>
        /// Returns the inner product of a function of this vector with a function of a second vector.
        /// </summary>
        /// <param name="thisFun">Function to convert the elements of this vector</param>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="thatFun">Function to convert the elements of that vector</param>
        /// <returns>Their inner product.</returns>
        public override double Inner(Converter<double, double> thisFun, Vector that, Converter<double, double> thatFun)
        {
            if (that.Sparsity == Sparsity.Dense)
                return Inner(thisFun, (DenseVector)that, thatFun);

            CheckCompatible(that, nameof(that));
            IEnumerator<double> thatEnum = that.GetEnumerator();
            double inner = 0.0;
            int i = start;
            while (thatEnum.MoveNext())
                inner += (thisFun(this.data[i++]) * thatFun(thatEnum.Current));

            return inner;
        }

        /// <summary>
        /// Returns the inner product of a function of this dense vector with a function of a second dense vector.
        /// </summary>
        /// <param name="thisFun">Function to convert the elements of this vector</param>
        /// <param name="that">Second vector, which must have the same size as <c>this</c>.</param>
        /// <param name="thatFun">Function to convert the elements of that vector</param>
        /// <returns>Their inner product.</returns>
        public double Inner(Converter<double, double> thisFun, DenseVector that, Converter<double, double> thatFun)
        {
            CheckCompatible(that, nameof(that));
            double inner = 0.0;
            int end = start + count;
            for (int i = start, j = that.start; i < end; i++)
                inner += (thisFun(this.data[i]) * thatFun(that.data[j++]));
            return inner;
        }

        /// <summary>
        /// Returns the outer product of this vector with another vector.
        /// </summary>
        /// <param name="that">Second vector.</param>
        /// <returns>Their outer product.</returns>
        public override Matrix Outer(Vector that)
        {
            if (that.Sparsity == Sparsity.Dense)
                return Outer((DenseVector)that);

            Matrix outer = new Matrix(Count, that.Count);
            int end = start + count;
            for (int i = start; i < end; i++)
            {
                double v = this.data[i];
                IEnumerator<double> thatEnum = that.GetEnumerator();
                int j = 0;
                while (thatEnum.MoveNext())
                {
                    outer[i, j++] = v * thatEnum.Current;
                }
            }
            return outer;
        }

        /// <summary>
        /// Returns the outer product of this dense vector with another dense vector.
        /// </summary>
        /// <param name="that">Second vector.</param>
        /// <returns>Their outer product.</returns>
        public PositiveDefiniteMatrix Outer(DenseVector that)
        {
            PositiveDefiniteMatrix outer = new PositiveDefiniteMatrix(Count, that.Count);
            int end = start + count;
            int thatend = that.start + that.count;
            for (int i = start; i < end; ++i)
            {
                double v = this.data[i];
                for (int j = that.start; j < thatend; ++j)
                    outer[i, j] = v * that.data[j];
            }

            return outer;
        }

        /// <summary>
        /// Sets this vector to the diagonal of a matrix.
        /// </summary>
        /// <param name="m">A matrix with Rows==Cols==this.Count.</param>
        public override void SetToDiagonal(Matrix m)
        {
            int end = start + count;
            for (int i = start; i < end; i++)
                this.data[i] = m[i, i];
        }

        /// <summary>
        /// Multiplies this vector by a scalar.
        /// </summary>
        /// <param name="scale">The scalar.</param>
        /// <returns></returns>
        /// <remarks>this receives the product.
        /// This method is a synonym for SetToProduct(this, scale)
        /// </remarks>
        public override void Scale(double scale)
        {
            int end = start + count;
            for (int i = start; i < end; ++i)
                this.data[i] *= scale;
        }

        #endregion

        #region Object overrides

        /// <inheritdoc/>
        public override bool Equals(object obj)
        {
            Vector that = obj as Vector;

            if (Object.ReferenceEquals(that, null))
                return false;
            if (Object.ReferenceEquals(this, that))
                return true;

            if (this.count != that.Count)
                return false;

            if (that.Sparsity == Sparsity.Dense)
            {
                int end = start + count;
                DenseVector thatd = that as DenseVector;
                for (int i = start, j = thatd.start; i < end; i++)
                    if (this.data[i] != thatd.data[j++])
                        return false;
            }
            else
            {
                IEnumerator<double> thatEnum = that.GetEnumerator();
                int i = start;
                while (thatEnum.MoveNext())
                {
                    if (this.data[i++] != thatEnum.Current)
                        return false;
                }
            }
            return true;
        }

        /// <inheritdoc/>
        public override bool EqualsAll(double value)
        {
            int end = start + count;
            for (int i = start; i < end; ++i)
            {
                if (this.data[i] != value) return false;
            }
            return true;
        }

        /// <inheritdoc/>
        public override bool GreaterThan(double value)
        {
            int end = start + count;
            for (int i = start; i < end; ++i)
            {
                if (data[i] <= value) return false;
            }
            return true;
        }

        /// <inheritdoc/>
        public override bool LessThan(double value)
        {
            int end = start + count;
            for (int i = start; i < end; ++i)
            {
                if (data[i] >= value) return false;
            }
            return true;
        }

        /// <inheritdoc/>
        public override bool GreaterThan(Vector that)
        {
            if (that.Sparsity == Sparsity.Dense)
                return GreaterThan((DenseVector)that);

            CheckCompatible(that, nameof(that));
            IEnumerator<double> thatEnum = that.GetEnumerator();
            int i = start;
            while (thatEnum.MoveNext())
            {
                if (this.data[i] <= thatEnum.Current)
                    return false;
            }

            return (true);
        }

        /// <inheritdoc cref="GreaterThan(Vector)"/>
        public bool GreaterThan(DenseVector that)
        {
            int end = start + count;
            for (int i = start, j = that.start; i < end; i++)
                if (this.data[i] <= that.data[j++])
                    return false;
            return (true);
        }

        /// <inheritdoc/>
        public override bool LessThan(Vector that)
        {
            if (that.Sparsity == Sparsity.Dense)
                return LessThan((DenseVector)that);

            CheckCompatible(that, nameof(that));
            IEnumerator<double> thatEnum = that.GetEnumerator();
            int i = start;
            while (thatEnum.MoveNext())
            {
                if (this.data[i] >= thatEnum.Current)
                    return false;
            }

            return (true);
        }

        /// <inheritdoc cref="LessThan(Vector)"/>
        public bool LessThan(DenseVector that)
        {
            int end = start + count;
            for (int i = start, j = that.start; i < end; i++)
                if (this.data[i] >= that.data[j++])
                    return false;
            return (true);
        }

        /// <inheritdoc/>
        public override bool GreaterThanOrEqual(double value)
        {
            int end = start + count;
            for (int i = start; i < end; ++i)
            {
                if (data[i] < value) return false;
            }
            return true;
        }

        /// <inheritdoc/>
        public override bool LessThanOrEqual(double value)
        {
            int end = start + count;
            for (int i = start; i < end; ++i)
            {
                if (data[i] > value) return false;
            }
            return true;
        }

        /// <inheritdoc/>
        public override bool GreaterThanOrEqual(Vector that)
        {
            if (that.Sparsity == Sparsity.Dense)
                return GreaterThanOrEqual((DenseVector)that);

            CheckCompatible(that, nameof(that));
            IEnumerator<double> thatEnum = that.GetEnumerator();
            int i = start;
            while (thatEnum.MoveNext())
            {
                if (this.data[i] < thatEnum.Current)
                    return false;
            }

            return true;
        }

        /// <inheritdoc cref="GreaterThanOrEqual(Vector)"/>
        public bool GreaterThanOrEqual(DenseVector that)
        {
            int end = start + count;
            for (int i = start, j = that.start; i < end; i++)
                if (this.data[i] < that.data[j++])
                    return false;
            return (true);
        }

        /// <inheritdoc/>
        public override bool LessThanOrEqual(Vector that)
        {
            if (that.Sparsity == Sparsity.Dense)
                return LessThanOrEqual((DenseVector)that);

            CheckCompatible(that, nameof(that));
            IEnumerator<double> thatEnum = that.GetEnumerator();
            int i = start;
            while (thatEnum.MoveNext())
            {
                if (this.data[i] > thatEnum.Current)
                    return false;
            }

            return (true);
        }

        /// <inheritdoc cref="LessThanOrEqual(Vector)"/>
        public bool LessThanOrEqual(DenseVector that)
        {
            int end = start + count;
            for (int i = start, j = that.start; i < end; i++)
                if (this.data[i] > that.data[j++])
                    return false;
            return (true);
        }

        /// <inheritdoc/>
        public override double MaxDiff(Vector that)
        {
            if (that.Sparsity == Sparsity.Dense)
                return MaxDiff((DenseVector)that);

            if (count != that.Count)
            {
                return Double.PositiveInfinity;
            }

            IEnumerator<double> thatEnum = that.GetEnumerator();
            double max = 0.0;
            int i = start;
            while (thatEnum.MoveNext())
            {
                double x = this.data[i++];
                double y = thatEnum.Current;
                bool xnan = Double.IsNaN(x);
                bool ynan = Double.IsNaN(y);
                if (xnan != ynan)
                    return Double.PositiveInfinity;
                else
                {
                    // matching infinities or NaNs will not change max
                    double diff = System.Math.Abs(x - y);
                    if (diff > max)
                    {
                        max = diff;
                    }
                }
            }

            return max;
        }

        /// <inheritdoc cref="MaxDiff(Vector)"/>
        public double MaxDiff(DenseVector that)
        {
            if (count != that.Count)
            {
                return Double.PositiveInfinity;
            }

            double max = 0.0;
            int end = start + count;
            for (int i = start, j = that.start; i < end;)
            {
                double x = this.data[i++];
                double y = that.data[j++];
                bool xnan = Double.IsNaN(x);
                bool ynan = Double.IsNaN(y);
                if (xnan != ynan)
                    return Double.PositiveInfinity;
                else
                {
                    // matching infinities or NaNs will not change max
                    double diff = System.Math.Abs(x - y);
                    if (diff > max)
                    {
                        max = diff;
                    }
                }
            }
            return max;
        }

        /// <inheritdoc/>
        public override double MaxDiff(Vector that, double rel)
        {
            if (that.Sparsity == Sparsity.Dense)
                return MaxDiff((DenseVector)that, rel);

            if (count != that.Count)
            {
                return Double.PositiveInfinity;
            }

            IEnumerator<double> thatEnum = that.GetEnumerator();
            double max = 0.0;
            int i = start;
            while (thatEnum.MoveNext())
            {
                double x = this.data[i++];
                double y = thatEnum.Current;
                bool xnan = Double.IsNaN(x);
                bool ynan = Double.IsNaN(y);
                if (xnan != ynan)
                    return Double.PositiveInfinity;
                else
                {
                    // matching infinities or NaNs will not change max
                    double diff = MMath.AbsDiff(x, y, rel);
                    if (diff > max)
                    {
                        max = diff;
                    }
                }
            }

            return max;
        }

        /// <inheritdoc cref="MaxDiff(Vector, double)"/>
        public double MaxDiff(DenseVector that, double rel)
        {
            if (count != that.Count)
            {
                return Double.PositiveInfinity;
            }

            double max = 0.0;
            int end = start + count;
            for (int i = start, j = that.start; i < end;)
            {
                double x = this.data[i++];
                double y = that.data[j++];
                bool xnan = Double.IsNaN(x);
                bool ynan = Double.IsNaN(y);
                if (xnan != ynan)
                    return Double.PositiveInfinity;
                else
                {
                    // matching infinities or NaNs will not change max
                    double diff = MMath.AbsDiff(x, y, rel);
                    if (diff > max)
                    {
                        max = diff;
                    }
                }
            }
            return max;
        }

        /// <summary>
        /// Gets a hash code for the instance.
        /// </summary>
        /// <returns>The code.</returns>
        /// <exclude/>
        public override int GetHashCode()
        {
            int end = start + count;
            int hash = Hash.Start;
            for (int i = start; i < end; i++)
                hash = Hash.Combine(hash, this.data[i]);
            return hash;
        }

        /// <summary>
        /// String representation of vector with a specified format and delimiter
        /// </summary>
        /// <param name="format"></param>
        /// <param name="delimiter"></param>
        /// <returns></returns>
        public override string ToString(string format, string delimiter)
        {
            StringBuilder s = new StringBuilder();
            int end = start + count;
            for (int i = start; i < end; ++i)
            {
                if (i > 0) s.Append(delimiter);
                s.Append(this.data[i].ToString(format));
            }
            return s.ToString();
        }

        /// <summary>
        /// String representation of vector with a specified format and delimiter
        /// and a function for converting integers to display strings.
        /// </summary>
        public override string ToString(string format, string delimiter, Func<int, string> intToString)
        {
            StringBuilder s = new StringBuilder();
            int end = start + count;
            for (int i = start; i < end; ++i)
            {
                if (i > 0) s.Append(delimiter);
                s.Append(intToString(i));
                s.Append(":");
                s.Append(this.data[i].ToString(format));
            }
            return s.ToString();
        }

        #endregion

        #region Linear algebra

        /// <summary>
        /// Gets the solution to Ax=b, where A is an upper triangular matrix, and b is this vector.
        /// Equivalent to the left-division x = A\b.
        /// </summary>
        /// <param name="A">An upper triangular matrix.</param>
        /// <returns><c>this</c></returns>
        /// <remarks>
        /// <c>this</c> is used as the right-hand side vector b, and it also
        /// receives the solution.
        /// Throws an exception if <paramref name="A"/> is singular.</remarks>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1709:IdentifiersShouldBeCasedCorrectly", MessageId = "0#")]
        public override Vector PredivideBy(UpperTriangularMatrix A)
        {
#if LAPACK
            Lapack.PredivideBy(this, A);
#else
            DenseVector b = this;
            Assert.IsTrue(A.Rows == A.Cols);

            if (A.Rows != b.count)
            {
                throw new ArgumentException("matrix dimensions incompatible", nameof(A));
            }

            for (int i = A.Rows - 1; i >= 0; i--)
            {
                double sum = b[i];
                for (int k = i + 1; k < A.Rows; k++)
                {
                    sum -= A[i, k] * b[k];
                }
                if (System.Math.Abs(A[i, i]) < double.Epsilon)
                {
                    throw new MatrixSingularException(A);
                }
                b[i] = sum / A[i, i];
            }
#endif
            return this;
        }

        /// <summary>
        /// Gets the solution to A'x=b, where A is a lower triangular matrix, and b is this vector.
        /// Equivalent to the left-division x = A'\b.
        /// </summary>
        /// <param name="A">A lower triangular matrix.</param>
        /// <returns><c>this</c></returns>
        /// <remarks>
        /// <c>this</c> is used as the right-hand side vector b, and it also
        /// receives the solution.
        /// Throws an exception if <paramref name="A"/> is singular.</remarks>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1709:IdentifiersShouldBeCasedCorrectly", MessageId = "0#")]
        public override Vector PredivideByTranspose(LowerTriangularMatrix A)
        {
#if LAPACK
            Lapack.PredivideByTranspose(this, A);
#else
            DenseVector b = this;
            Assert.IsTrue(A.Rows == A.Cols);
            Assert.IsTrue(A.Rows == b.count);

            for (int i = A.Rows - 1; i >= 0; i--)
            {
                double sum = b[i];
                for (int k = i + 1; k < A.Rows; k++)
                {
                    sum -= A[k, i] * b[k];
                }
                if (System.Math.Abs(A[i, i]) < double.Epsilon)
                {
                    throw new MatrixSingularException(A);
                }
                b[i] = sum / A[i, i];
            }
#endif
            return this;
        }

        /// <summary>
        /// Gets the solution to Ax=b, where A is a lower triangular matrix, and b is this vector.
        /// Equivalent to the left-division x = A\b.
        /// </summary>
        /// <param name="A">A lower triangular matrix.</param>
        /// <returns><c>this</c></returns>
        /// <remarks>
        /// <c>this</c> is used as the right-hand side vector b, and it also
        /// receives the solution.
        /// Throws an exception if <paramref name="A"/> is singular.</remarks>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Naming", "CA1709:IdentifiersShouldBeCasedCorrectly", MessageId = "0#")]
        public override Vector PredivideBy(LowerTriangularMatrix A)
        {
#if LAPACK
            Lapack.PredivideBy(this, A);
#else
            DenseVector b = this;
            Assert.IsTrue(A.Rows == A.Cols);

            if (A.Rows != b.count)
            {
                throw new ArgumentException("matrix and vector dimensions incompatible", nameof(A));
            }

            for (int i = 0; i < A.Rows; i++)
            {
                double sum = b[i];
                for (int j = 0; j < i; j++)
                {
                    sum -= A[i, j] * b[j];
                }
                if (A[i, i] == 0) throw new MatrixSingularException(A);
                b[i] = sum / A[i, i];
            }
#endif
            return this;
        }

        /// <summary>
        /// Premultiply this vector by the inverse of the given positive definite matrix
        /// </summary>
        /// <param name="A"></param>
        /// <returns></returns>
        public override Vector PredivideBy(PositiveDefiniteMatrix A)
        {
            LowerTriangularMatrix L = new LowerTriangularMatrix(A.Rows, A.Cols);
            bool isPD = L.SetToCholesky(A);
            if (!isPD) throw new PositiveDefiniteMatrixException();
            PredivideBy(L);
            PredivideByTranspose(L);
            return this;
        }

        /// <summary>
        /// Solve Y = X*A
        /// </summary>
        /// <param name="Y">Vector of target values</param>
        /// <param name="X">Portrait matrix</param>
        /// <returns>The smallest squared singular value of X.  This is useful for detecting an ill-conditioned problem.</returns>
        public double SetToLeastSquares(DenseVector Y, Matrix X)
        {
            DenseVector result = this;
            // for best numerical accuracy, we use the SVD of X: USV' = X
            // A = inv(X'X)*X'Y
            //   = inv(VSU'USV')*VSU'Y
            //   = V*inv(S^2)*SU'Y
            Matrix V = new Matrix(X.Cols, X.Cols);
            Matrix US = (Matrix)X.Clone();
            V.SetToRightSingularVectors(US);
            double minSingularValueSqr = double.PositiveInfinity;
            for (int k = 0; k < US.Cols; k++)
            {
                double singularValueSqr = 0;
                for (int j = 0; j < US.Rows; j++)
                {
                    singularValueSqr += US[j, k] * US[j, k];
                }
                //Debug.WriteLine("singularValueSqr[{0}] = {1}", k, singularValueSqr);
                minSingularValueSqr = System.Math.Min(minSingularValueSqr, singularValueSqr);
                if (singularValueSqr > 0)
                {
                    for (int j = 0; j < US.Rows; j++)
                    {
                        US[j, k] /= singularValueSqr;
                    }
                }
            }
            DenseVector yUS = DenseVector.Zero(US.Cols);
            yUS.SetToProduct(Y, US);
            if (yUS.Any(x => x > double.MaxValue || x < double.MinValue))
            {
                // Overflow occurred.
                // Rescale Y to avoid overflow.
                double limit = double.MaxValue / Y.Count;
                double upperBound = limit;
                for (int j = 0; j < Y.Count; j++)
                {
                    for (int k = 0; k < US.Cols; k++)
                    {
                        // We want scale*abs(sum_j Y[j]*US[j,k]) <= double.MaxValue
                        // A sufficient condition is scale*abs(Y[j]*US[j,k]) <= double.MaxValue/Y.Count
                        // Therefore scale <= double.MaxValue/Y.Count/abs(Y[j]*US[j,k]).
                        double absY = System.Math.Abs(Y[j]);
                        double absU = System.Math.Abs(US[j, k]);
                        double thisUpperBound;
                        if (absY > 1)
                        {
                            thisUpperBound = limit / absY / absU;
                        }
                        else
                        {
                            thisUpperBound = limit / absU / absY;
                        }
                        upperBound = System.Math.Min(upperBound, thisUpperBound);
                    }
                }
                if (upperBound < 1)
                {
                    Y = (DenseVector)(Y * upperBound);
                    yUS.SetToProduct(Y, US);
                }
            }
            result.SetToProduct(V, yUS);
            if (!result.Any(x => double.IsNaN(x) || double.IsInfinity(x)))
            {
                // iterate again on the residual for higher accuracy
                var residual = X * result;
                residual.SetToDifference(Y, residual);
                yUS.SetToProduct(residual, US);
                result.SetToSum(result, V * yUS);
            }
            return minSingularValueSqr;
        }

        #endregion
    }
}