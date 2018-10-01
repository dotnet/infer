// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Factors.Attributes;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// Helpful methods used by generated code.
    /// </summary>
    [Quality(QualityBand.Stable)]
    public class ArrayHelper
    {
        /// <summary>
        /// Fill an array with copies of an object.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array">The array to modify.</param>
        /// <param name="margprot">The object to copy into each element</param>
        /// <returns>The modified array</returns>
        public static T[] Fill<T>(T[] array, T margprot) where T : ICloneable
        {
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = MakeCopy(margprot);
            }
            return array;
        }

        /// <summary>
        /// Fill a 2D array with copies of an object.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array">The array to modify</param>
        /// <param name="margprot">The object to copy into each element</param>
        /// <returns>The modified array</returns>
        public static T[,] Fill2D<T>(T[,] array, T margprot) where T : ICloneable
        {
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    array[i, j] = MakeCopy(margprot);
                }
            }
            return array;
        }

        /// <summary>
        /// Create a clone of obj with uniform value.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="obj"></param>
        /// <returns></returns>
        [Skip]
        public static T MakeUniform<T>(T obj) where T : ICloneable, SettableToUniform
        {
            obj = (T) obj.Clone();
            obj.SetToUniform();
            return obj;
        }

        /// <summary>
        /// Call obj.SetToUniform() and return the object.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="obj"></param>
        /// <returns></returns>
        public static T SetToUniform<T>(T obj) where T : SettableToUniform
        {
            obj.SetToUniform();
            return obj;
        }

        // TODO: move this out of ArrayHelper since it will confuse HoistingTransform
        public static T SetToProductWith<T>(T result, [SkipIfUniform] T value) where T : SettableToProduct<T>
        {
            result.SetToProduct(result, value);
            return result;
        }

        /// <summary>
        /// Call result.SetTo(value) and return result.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="result">The object to modify</param>
        /// <param name="value">The desired value</param>
        /// <returns>The modified object</returns>
        [Fresh] // needed for CopyPropagationTransform
        public static T SetTo<T>(T result, [SkipIfUniform] T value) where T : SettableTo<T>
        {
            result.SetTo(value);
            return result;
        }

        /// <summary>
        /// Call result.SetTo(value) and return result.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="result">The array to modify</param>
        /// <param name="values">The desired values</param>
        /// <returns>The modified array</returns>
        [Fresh] // needed for CopyPropagationTransform
        public static T[] SetTo<T>(T[] result, [SkipIfUniform] T[] values) where T : SettableTo<T>
        {
            Distribution.SetTo(result, values);
            return result;
        }

        // since we are using this function in inference code, it must have annotations so that dependencies are computed correctly
        /// <summary>
        /// Set all elements of an array to the same value.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="T2"></typeparam>
        /// <param name="result">The array to modify</param>
        /// <param name="item">The desired value</param>
        /// <returns>The modified array</returns>
        [Fresh] // needed for CopyPropagationTransform
        public static T SetAllElementsTo<T, T2>(T result, [SkipIfUniform] T2 item) where T : CanSetAllElementsTo<T2>
        {
            result.SetAllElementsTo(item);
            return result;
        }

        /// <summary>
        /// Set all elements of an array to the same value.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="result">The array to modify</param>
        /// <param name="item">The desired value</param>
        /// <returns>The modified array</returns>
        [Fresh] // needed for CopyPropagationTransform
        public static T[] SetAllElementsTo<T>(T[] result, [SkipIfUniform] T item)
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = item;
            }
            return result;
        }

        /// <summary>
        /// Make a deep copy of an object.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="obj"></param>
        /// <returns></returns>
        public static T MakeCopy<T>([SkipIfUniform] T obj) where T : ICloneable
        {
            Array arr = obj as Array;
            if (arr == null) return (T) obj.Clone();
            Array arr2 = (Array) arr.Clone();
            if (arr2.Length > 0 && arr.GetValue(0) is ICloneable)
            {
                if (arr2.Rank != 1) throw new InferRuntimeException("Cannot clone arrays of rank>1");
                for (int i = 0; i < arr2.Length; i++)
                {
                    object item = arr.GetValue(i);
                    item = MakeCopy((ICloneable) item);
                    arr2.SetValue(item, i);
                }
            }
            return (T) (object) arr2;
        }

        /// <summary>
        /// Copies the storage of the passed in argument, without caring about its value.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="result"></param>
        /// <returns></returns>
        [Skip]
        public static T CopyStorage<T>(T result) where T : ICloneable
        {
            // The argument must be named 'result' for the framework to understand that only
            // the storage is being used.
            return MakeCopy(result);
        }
    }

#if false
    public class DistArray<T> : SettableToProduct<DistArray<T>>, ICloneable where T : SettableToProduct<T>, ICloneable
    {
        T[] array;

        public T[] Array { get { return array; } }

        public DistArray(T[] array)
        {
            this.array = array;
        }

        public DistArray(int size, T prototype)
        {
            this.array = ArrayHelper.Fill(size, prototype);
        }

        public int GetLength(int dim) { return array.GetLength(dim); }
        public int Length { get { return array.Length; } }


        public void SetToProduct(DistArray<T> a, DistArray<T> b)
        {
            for (int i = 0; i < array.Length; i++) {
                array[i].SetToProduct(a[i], b[i]);
            }
        }

        public T this[int index]
        {
            get { return array[index]; }
            set { array[index] = value; }
        }


        public object Clone()
        {
            return new DistArray<T>((T[])array.Clone());
        }

    }

    public class DistArray2D<T> : SettableToProduct<DistArray2D<T>>, ICloneable where T : SettableToProduct<T>, ICloneable
    {
        T[,] array;

        public T[,] Array { get { return array; } }

        public DistArray2D(T[,] array)
        {
            this.array = array;
        }

        public DistArray2D(int size, int size2, T prototype)
        {
            this.array = ArrayHelper.Fill2D(size, size2, prototype);
        }

        public void ModifyAll(Converter<T,T> converter)
        {
            for (int i = 0; i < Array.GetLength(0); i++) {
                for (int j = 0; j < Array.GetLength(1); j++) {
                    this[i,j] = converter(this[i,j]);
                }
            }
        }

        public int GetLength(int dim) { return array.GetLength(dim); }
        public int Rows { get { return array.GetLength(0); } }
        public int Columns { get { return array.GetLength(1); } }


        public void SetToProduct(DistArray2D<T> a, DistArray2D<T> b)
        {
            for (int i = 0; i < array.GetLength(0); i++) {
                for (int j = 0; j < array.GetLength(1); j++) {
                    array[i, j].SetToProduct(a[i, j], b[i, j]);
                }
            }
        }

        public T this[int index, int index2]
        {
            get { return array[index, index2]; }
            set { array[index, index2] = value; }
        }


        public object Clone()
        {
            return new DistArray2D<T>((T[,])array.Clone());
        }


    }
#endif
}