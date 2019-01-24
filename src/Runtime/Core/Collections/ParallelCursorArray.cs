// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Collections
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// A multidimensional array of objects where each field is in a CursorArray.
    /// </summary>
    /// <typeparam name="CursorType"></typeparam>
    /// <typeparam name="ArrayType">A cursor-based array type, such as CursorArray or ParallelCursorArray.</typeparam>
    /// <remarks><para>
    /// A ParallelCursorArray is meant to behave like an ordinary Array, while 
    /// being more memory-efficient.  It uses a cursor similar to a CursorArray.
    /// However, the storage layout is different.  Each field of the cursor
    /// is stored in a separate ICursorArray, with its own inner cursor.
    /// By advancing these inner cursors, the outer cursor is automatically
    /// updated.  Thus <typeparamref name="CursorType"/> does not need to implement <c>ICursor</c>.
    /// For example, <typeparamref name="CursorType"/> might be <c>Array&lt;ICursor&gt;</c>, holding an array of  
    /// inner cursors.
    /// </para><para>
    /// There must be at least one member array, and 
    /// all member arrays must have the same dimensions.
    /// </para></remarks>
    /// <example>See CursorArrayTest.cs.</example>
    public class ParallelCursorArray<CursorType, ArrayType> : ICursor, ICursorArray<CursorType>
        where ArrayType : ICursorArray
    {
        protected CursorType cursor;
        protected IList<ArrayType> members;

        #region ICursorArray methods

        public void MoveTo(params int[] index)
        {
            foreach (ArrayType a in members)
            {
                a.MoveTo(index);
            }
        }

        public CursorType this[params int[] index]
        {
            get
            {
                MoveTo(index);
                return cursor;
            }
        }

#if false
        ICursor ICursorArray.this[int [] index] {    
            get {
                return this[index];
            }
        }
#endif

        public void MoveTo(int index)
        {
            foreach (ArrayType a in members)
            {
                a.MoveTo(index);
            }
        }

        public CursorType this[int index]
        {
            get
            {
                MoveTo(index);
                return cursor;
            }
        }

#if false
        ICursor ICursorArray.this[int index] {    
            get {
                return this[index];
            }
        }
#endif

        public int Count
        {
            get { return ((ICursorArray) members[0]).Count; }
        }

        public IList<int> Lengths
        {
            get { return members[0].Lengths; }
        }

        public int Rank
        {
            get { return Lengths.Count; }
        }

        public int GetLength(int dimension)
        {
            return Lengths[dimension];
        }

        #endregion

        #region IEnumerable methods

        public IEnumerator<CursorType> GetEnumerator()
        {
            for (int index = 0; index < Count; index++)
            {
                yield return this[index];
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        #endregion

        #region ICursor methods

        int ICursor.Count
        {
            get { throw new NotSupportedException(); }
        }

        public int Start
        {
            get { throw new NotSupportedException(); }
            set { throw new NotSupportedException(); }
        }

        public void CreateSourceArray(int nRecords)
        {
            throw new NotSupportedException();
        }

        public ICursor ReferenceClone()
        {
            throw new NotSupportedException();
#if false
            ArrayType [] new_members = members.MemberwiseClone();
            for(int i = 0; i < members.Length; i++) {
                new_members[i] = (ArrayType)members[i].ReferenceClone();
            }
            return new ParallelStructArray<CursorType,ArrayType>((CursorType)cursor.ReferenceClone(), new_members);
#endif
        }

        #endregion

        #region Copying

        public object Clone()
        {
            throw new NotSupportedException();
#if false
            ArrayType [] new_members = members.MemberwiseClone();
            for(int i = 0; i < members.Length; i++) {
                new_members[i] = (ArrayType)members[i].Clone();
            }
            return new ParallelStructArray<CursorType,ArrayType>((CursorType)cursor.RClone(), new_members);
#endif
        }

        #endregion

        public bool IsCompatibleWith(ICursorArray that)
        {
            return Lengths.ValueEquals(that.Lengths);
        }

        protected void CheckMembers()
        {
            if (members.Count == 0)
                throw new InferRuntimeException("must have at least one member");
            // all members must have the same Lengths
            for (int i = 0; i < members.Count; i++)
            {
                if (!IsCompatibleWith(members[i]))
                {
                    throw new InferRuntimeException("all member arrays must have the same dimensions");
                }
            }
        }

        /// <summary>
        /// Create a new ParallelCursorArray.
        /// </summary>
        /// <param name="cursor">An object to use as the cursor.</param>
        /// <param name="members">A list of cursor-based arrays.</param>
        /// <remarks>
        /// <paramref name="cursor"/> must already be initialized to contain
        /// the cursors of the arrays in <paramref name="members"/>.
        /// There must be at least one member array, and 
        /// all member arrays must have the same dimensions.
        /// </remarks>
        public ParallelCursorArray(CursorType cursor, IList<ArrayType> members)
        {
            this.cursor = cursor;
            this.members = members;
            CheckMembers();
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}