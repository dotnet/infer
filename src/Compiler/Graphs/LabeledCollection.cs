// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Text;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// A collection which is the union of labeled subcollections.
    /// </summary>
    /// <typeparam name="ItemType"></typeparam>
    /// <typeparam name="LabelType"></typeparam>
    /// <remarks><p>
    /// This interface corresponds to an inverted index, where each label maps to a subset of items.
    /// This is unlike an IDictionary, where each key maps to a single item.
    /// When the labels are sparse, an inverted index is more efficient than attaching a label to each item and searching
    /// through the items.
    /// The subcollections may overlap.
    /// </p><p>
    /// Add(ItemType) is equivalent to using the default label, which may be default(LabelType) or some
    /// other value such as an empty string ("").
    /// The other ICollection methods, such as Count and Clear, apply to all items regardless of label.
    /// </p></remarks>
    internal interface ILabeledCollection<ItemType, LabelType> : ICollection<ItemType>
    {
        /// <summary>
        /// The labels of the subcollections.
        /// </summary>
        /// <remarks>
        /// Labels must be unique.  The returned collection must not be modified.
        /// </remarks>
        ICollection<LabelType> Labels { get; }

        /// <summary>
        /// Get a subcollection.
        /// </summary>
        /// <param name="label">The label of an existing subcollection or a subcollection to be created.</param>
        /// <returns>A subcollection of items.</returns>
        /// <remarks>If the subcollection already exists, it is returned.  Otherwise, a new subcollection is created and returned.
        /// The result is mutable.  Some LabeledCollection classes may not allow certain labels.
        /// </remarks>
        /// <exception cref="InvalidLabelException">If the label is not allowed by the collection.</exception>
        ICollection<ItemType> WithLabel(LabelType label);
    }

    internal class InvalidLabelException : Exception
    {
        public object Label;

        public InvalidLabelException(object label)
        {
            Label = label;
        }

        public InvalidLabelException()
        {
        }

        // This constructor is needed for serialization.
        protected InvalidLabelException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }

    // This is thrown if you try to Add to a list which is full.
    // It is more informative than NotSupportedException.
    internal class ListOverflowException : Exception
    {
        public ListOverflowException()
        {
        }

        // This constructor is needed for serialization.
        protected ListOverflowException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }

    /// <summary>
    /// A default implementation of ILabeledCollection.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="LabelType"></typeparam>
    /// <remarks>
    /// This base class implements all of the ICollection methods in terms of the 
    /// two ILabeledCollection methods.
    /// This makes it easy to create new ILabeledCollection classes, just by 
    /// implementing the two ILabeledCollection methods.
    /// It assumes that the subcollections do not overlap.
    /// </remarks>
    internal abstract class LabeledCollection<T, LabelType> : ILabeledCollection<T, LabelType>
    {
        public abstract ICollection<LabelType> Labels { get; }
        public abstract ICollection<T> WithLabel(LabelType label);

        public virtual LabelType DefaultLabel
        {
            get { return default(LabelType); }
        }

#if zero
    //ICollection ILabeledCollection<T>.Labels { 
        public virtual ICollection Labels {
            get { throw new NotSupportedException(); }
        }
        //ICollection<T> ILabeledCollection<T>.WithLabel(object label)
        public virtual ICollection<T> WithLabel(object label)
        {
            throw new NotSupportedException();
        }
#endif

        #region IEnumerable methods

        public virtual IEnumerator<T> GetEnumerator()
        {
            foreach (LabelType label in Labels)
            {
                foreach (T value in WithLabel(label))
                {
                    yield return value;
                }
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        #endregion

        #region ICollection methods

        public virtual int Count
        {
            get
            {
                int count = 0;
                foreach (LabelType label in Labels)
                {
                    count += WithLabel(label).Count;
                }
                return count;
            }
        }

        public virtual bool IsReadOnly
        {
            get { return false; }
        }

        public virtual void Add(T item)
        {
            WithLabel(DefaultLabel).Add(item);
        }

        public virtual void Clear()
        {
            foreach (LabelType label in Labels)
            {
                WithLabel(label).Clear();
            }
        }

        public virtual bool Contains(T item)
        {
            foreach (LabelType label in Labels)
            {
                if (WithLabel(label).Contains(item))
                {
                    return true;
                }
            }
            return false;
        }

        public virtual void CopyTo(T[] array, int index)
        {
            foreach (LabelType label in Labels)
            {
                ICollection<T> list = WithLabel(label);
                list.CopyTo(array, index);
                index += list.Count;
            }
        }

        public virtual bool Remove(T item)
        {
            foreach (LabelType label in Labels)
            {
                ICollection<T> list = WithLabel(label);
                if (list.Contains(item))
                {
                    // remove only one instance
                    return list.Remove(item);
                }
            }
            return false;
        }

        #endregion

        public override string ToString()
        {
            StringBuilder s = new StringBuilder();
            foreach (LabelType label in Labels)
            {
                if (!label.Equals(DefaultLabel))
                {
                    s.Append(String.Format("<{0}>", label));
                }
                int count = 0;
                foreach (T value in WithLabel(label))
                {
                    if (count > 0) s.Append(" ");
                    count++;
                    s.Append(value);
                }
                if (!label.Equals(DefaultLabel))
                {
                    s.Append(String.Format("</{0}>", label));
                }
            }
            return s.ToString();
        }
    }
}