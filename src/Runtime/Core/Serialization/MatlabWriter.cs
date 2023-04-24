// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Serialization
{
    /// <summary>
    /// Write data objects to a MAT file
    /// </summary>
    /// <remarks>
    /// The MAT file format is defined in <a href="http://www.mathworks.com/access/helpdesk/help/pdf_doc/matlab/matfile_format.pdf">The MAT file format</a>.
    /// </remarks>
    public class MatlabWriter : IDisposable
    {
        private Stream writer;

        /// <summary>
        /// Create a MatlabWriter to write to a given file name.
        /// </summary>
        /// <param name="fileName">The file name to write to (will be created if missing, or truncated if already exists)</param>
        public MatlabWriter(string fileName)
        {
            writer = new FileStream(fileName, FileMode.Create);
            Write(Matlab5Header("MATLAB 5.0 MAT-file, Created by Microsoft.ML.Probabilistic.Utililies.MatlabWriter"));
        }

        /// <summary>
        /// Write an array of bytes to the stream
        /// </summary>
        /// <param name="bytes"></param>
        protected void Write(byte[] bytes)
        {
            writer.Write(bytes, 0, bytes.Length);
        }

        protected void Write(MatType type)
        {
            Write((int) type);
        }

        protected void Write(short value)
        {
            Write(BitConverter.GetBytes(value));
        }

        protected void Write(ushort value)
        {
            Write(BitConverter.GetBytes(value));
        }

        protected void Write(int value)
        {
            Write(BitConverter.GetBytes(value));
        }

        protected void Write(uint value)
        {
            Write(BitConverter.GetBytes(value));
        }

        protected void Write(long value)
        {
            Write(BitConverter.GetBytes(value));
        }

        protected void Write(ulong value)
        {
            Write(BitConverter.GetBytes(value));
        }

        protected void Write(double value)
        {
            Write(BitConverter.GetBytes(value));
        }

        protected void Write(Single value)
        {
            Write(BitConverter.GetBytes(value));
        }

        protected void Write(byte value)
        {
            writer.WriteByte(value);
        }

        /// <summary>
        /// Write a string value with zero padding to write numBytes in total.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="numBytes"></param>
        protected void Write(string value, int numBytes)
        {
            for (int i = 0; i < value.Length; i++)
            {
                writer.WriteByte((byte) value[i]);
            }
            for (int i = value.Length; i < numBytes; i++)
            {
                writer.WriteByte(0);
            }
        }

        /// <summary>
        /// Write a named object to the stream
        /// </summary>
        /// <param name="name"></param>
        /// <param name="value"></param>
        public void Write(string name, object value)
        {
            if (value is Matrix) Write(name, (Matrix)value);
            else if (value is bool) Write(name, (bool)value ? (byte)1 : (byte)0);
            else if (value is byte) Write(name, (byte)value);
            else if (value is short) Write(name, (short)value);
            else if (value is ushort) Write(name, (ushort)value);
            else if (value is int) Write(name, (int)value);
            else if (value is uint) Write(name, (uint)value);
            else if (value is long) Write(name, (long)value);
            else if (value is ulong) Write(name, (ulong)value);
            else if (value is Single) Write(name, (Single)value);
            else if (value is double) Write(name, (double)value);
            else if (value is MatlabReader.ComplexMatrix) Write(name, (MatlabReader.ComplexMatrix)value);
            else if (value is string) Write(name, (string)value);
            else if (value is object[,]) Write(name, (object[,])value);
            else if (value is IReadOnlyList<bool>) Write(name, (IReadOnlyList<bool>)value);
            else if (value is IReadOnlyList<int>) Write(name, (IReadOnlyList<int>)value);
            else if (value is IReadOnlyList<uint>) Write(name, (IReadOnlyList<uint>)value);
            else if (value is IReadOnlyList<long>) Write(name, (IReadOnlyList<long>)value);
            else if (value is IReadOnlyList<double>) Write(name, (IReadOnlyList<double>)value);
            else if (value is IReadOnlyList<object>) Write<object>(name, (IReadOnlyList<object>)value);
            else if (value is IReadOnlyList<string>) Write<string>(name, (IReadOnlyList<string>)value);
            else if (value is Array) WriteArray(name, (Array)value);
            else if (ReferenceEquals(value, null))
            {
                Write(MatType.MATRIX);
                Write(0);
            }
            else if (value is IReadOnlyDictionary<string, object>) Write(name, (IReadOnlyDictionary<string, object>)value);
            else if (TryWriteDictionary(name, value)) return;
            else throw new NotImplementedException("Cannot write type " + StringUtil.TypeToString(value.GetType()));
        }

        protected bool TryWriteDictionary(string name, object value)
        {
            Type type = value.GetType();
            foreach(Type interfaceType in type.GetInterfaces()) {
                if (interfaceType.Name == typeof(IReadOnlyDictionary<,>).Name)
                {
                    Type[] args = interfaceType.GetGenericArguments();
                    if (args[0].Equals(typeof(string)))
                    {
                        var method = new Action<string, IReadOnlyDictionary<string, object>>(Write).Method;
                        method = method.GetGenericMethodDefinition();
                        Util.Invoke(method.MakeGenericMethod(args[1]), this, name, value);
                        return true;
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Write a named bool array to the stream
        /// </summary>
        /// <param name="name"></param>
        /// <param name="array"></param>
        public void Write(string name, IReadOnlyList<bool> array)
        {
            WriteMatrixHeader(name, array.Count, 1, mxClass.UINT8, 1, false, true);
            Write(MatType.UINT8);
            int numBytes = 1*array.Count;
            int numPaddedBytes = ((numBytes + 7)/8)*8;
            Write(numBytes); // bytes to follow
            for (int i = 0; i < array.Count; i++)
                Write(array[i] ? (byte) 1 : (byte) 0);
            int padding = numPaddedBytes - numBytes;
            for (int i = 0; i < padding; i++)
            {
                Write((byte) 0);
            }
        }

        /// <summary>
        /// Write a named int array to the stream
        /// </summary>
        /// <param name="name"></param>
        /// <param name="array"></param>
        public void Write(string name, IReadOnlyList<int> array)
        {
            WriteMatrixHeader(name, array.Count, 1, mxClass.INT32, 4);
            Write(MatType.INT32);
            int numBytes = 4*array.Count;
            int numPaddedBytes = ((numBytes + 7)/8)*8;
            Write(numBytes); // bytes to follow
            for (int i = 0; i < array.Count; i++)
                Write(array[i]);
            int padding = numPaddedBytes - numBytes;
            for (int i = 0; i < padding; i++)
            {
                Write((byte) 0);
            }
        }

        /// <summary>
        /// Write a named int array to the stream
        /// </summary>
        /// <param name="name"></param>
        /// <param name="array"></param>
        public void Write(string name, IReadOnlyList<uint> array)
        {
            WriteMatrixHeader(name, array.Count, 1, mxClass.UINT32, 4);
            Write(MatType.UINT32);
            int numBytes = 4 * array.Count;
            int numPaddedBytes = ((numBytes + 7) / 8) * 8;
            Write(numBytes); // bytes to follow
            for (int i = 0; i < array.Count; i++)
                Write(array[i]);
            int padding = numPaddedBytes - numBytes;
            for (int i = 0; i < padding; i++)
            {
                Write((byte)0);
            }
        }

        /// <summary>
        /// Write a named long array to the stream
        /// </summary>
        /// <param name="name"></param>
        /// <param name="array"></param>
        public void Write(string name, IReadOnlyList<long> array)
        {
            WriteMatrixHeader(name, array.Count, 1, mxClass.INT64, 8);
            Write(MatType.INT64);
            int numBytes = 8 * array.Count;
            int numPaddedBytes = ((numBytes + 7) / 8) * 8;
            Write(numBytes); // bytes to follow
            for (int i = 0; i < array.Count; i++)
                Write(array[i]);
            int padding = numPaddedBytes - numBytes;
            for (int i = 0; i < padding; i++)
            {
                Write((byte)0);
            }
        }

        /// <summary>
        /// Write a named ulong array to the stream
        /// </summary>
        /// <param name="name"></param>
        /// <param name="array"></param>
        public void Write(string name, IReadOnlyList<ulong> array)
        {
            WriteMatrixHeader(name, array.Count, 1, mxClass.UINT64, 8);
            Write(MatType.UINT64);
            int numBytes = 8 * array.Count;
            int numPaddedBytes = ((numBytes + 7) / 8) * 8;
            Write(numBytes); // bytes to follow
            for (int i = 0; i < array.Count; i++)
                Write(array[i]);
            int padding = numPaddedBytes - numBytes;
            for (int i = 0; i < padding; i++)
            {
                Write((byte)0);
            }
        }

        /// <summary>
        /// Write a named scalar to the stream
        /// </summary>
        /// <param name="name"></param>
        /// <param name="value"></param>
        public void Write(string name, double value)
        {
            Write(name, new double[] {value});
        }

        /// <summary>
        /// Write a named vector to the stream
        /// </summary>
        /// <param name="name"></param>
        /// <param name="array"></param>
        public void Write(string name, IReadOnlyList<double> array)
        {
            Matrix m = new Matrix(array.Count, 1);
            for (int i = 0; i < array.Count; i++)
            {
                m[i, 0] = array[i];
            }
            Write(name, m);
        }

        /// <summary>
        /// Write a named cell vector to the stream
        /// </summary>
        /// <param name="name"></param>
        /// <param name="array"></param>
        public void Write<T>(string name, IReadOnlyList<T> array)
        {
            object[,] m = new object[array.Count,1];
            for (int i = 0; i < array.Count; i++)
            {
                m[i, 0] = array[i];
            }
            Write(name, m);
        }

        /// <summary>
        /// Write a named Matrix to the stream
        /// </summary>
        /// <param name="name">Name of the matrix.</param>
        /// <param name="matrix">A matrix.</param>
        /// <param name="imaginary">If writing a complex matrix, the imaginary part of the matrix.</param>
        public void Write(string name, Matrix matrix, Matrix imaginary = null)
        {
            WriteMatrixHeader(name, matrix.Rows, matrix.Cols, mxClass.DOUBLE, 8, imaginary != null);

            // write the matrix data
            Write(matrix);
            if (imaginary != null) Write(imaginary);
        }

        protected void WriteMatrixHeader(string name, int rows, int cols, mxClass dataClass, int elementSize, bool isComplex = false, bool isLogical = false)
        {
            // convert to a valid variable name
            name = MakeValidVariableName(name);

            int numNameBytes = ((name.Length + 7)/8)*8;

            // compute total size of buffer
            int numericBytes = 8 + elementSize*rows*cols;
            numericBytes = ((numericBytes + 7)/8)*8;
            int numBytes = 48 + numNameBytes + numericBytes;
            if (isComplex) numBytes += numericBytes;

            // write the data type field
            Write(MatType.MATRIX);
            Write(numBytes - 8); // number of bytes 

            // write the array flags
            Write(MatType.UINT32);
            Write(8); // 8 bytes to follow
            int flags = (int) dataClass;
            if (isComplex) flags |= 0x0800;
            if (isLogical) flags |= 0x0200;
            Write(flags);
            Write(0);

            // write the dimension field
            Write(MatType.INT32);
            Write(8); // 8 bytes to follow
            Write(rows); // number of rows
            Write(cols); // number of columns

            // write the matrix name
            Write(MatType.INT8);
            Write(name.Length); // length of name in bytes
            Write(name, numNameBytes);
        }

        protected void Write(Matrix matrix)
        {
            Write(MatType.DOUBLE);
            Write(8*matrix.Rows*matrix.Cols); // bytes to follow
            for (int i = 0; i < matrix.Cols; i++)
                for (int j = 0; j < matrix.Rows; j++)
                    Write(matrix[j, i]);
        }

        /// <summary>
        /// Write a named ComplexMatrix to the stream
        /// </summary>
        /// <param name="name"></param>
        /// <param name="matrix"></param>
        public void Write(string name, MatlabReader.ComplexMatrix matrix)
        {
            Write(name, matrix.Real, matrix.Imaginary);
        }

        public static void WriteFromCsvFolder(string outputFilename, string inputFolder)
        {
            using (var writer = new MatlabWriter(outputFilename))
            {
                foreach (var filename in Directory.EnumerateFiles(inputFolder, "*.csv"))
                {
                    // EnumerateFiles may return files that don't exactly match the given searchPattern
                    if (Path.GetExtension(filename) != ".csv") continue;
                    string name = Path.GetFileNameWithoutExtension(filename);
                    writer.WriteFromCsv(name, filename);
                }
            }
        }

        /// <summary>
        /// Write a named struct whose fields are the columns of a csv file
        /// </summary>
        /// <param name="name"></param>
        /// <param name="path"></param>
        public void WriteFromCsv(string name, string path)
        {
            Dictionary<string,List<string>> dict = MatlabReader.ReadCsv(path);
            // try to convert strings into numbers
            Dictionary<string, object> dict2 = new Dictionary<string, object>();
            foreach (var entry in dict)
            {
                bool isDouble = true;
                List<double> doubles = new List<double>();
                foreach (string s in entry.Value)
                {
                    double d;
                    if (s.Length == 0)
                    {
                        d = double.NaN;
                        doubles.Add(d);
                    }
                    else if (double.TryParse(s, out d)) doubles.Add(d);
                    else
                    {
                        isDouble = false;
                        break;
                    }
                }
                if (isDouble) dict2.Add(entry.Key, doubles);
                else dict2.Add(entry.Key, entry.Value);
            }
            Write(name, dict2);
        }

        /// <summary>
        /// Write a named struct to the stream
        /// </summary>
        /// <param name="name"></param>
        /// <param name="dict"></param>
        public void Write<T>(string name, IReadOnlyDictionary<string, T> dict)
        {
            // convert to a valid variable name
            name = MakeValidVariableName(name);

            // pad the name to an 8-byte boundary
            int numNameBytes = ((name.Length + 7) / 8) * 8;

            // precompute the field bytes
            int numDataBytes = 0;
            Dictionary<string, byte[]> bytes = new Dictionary<string, byte[]>();
            Stream oldWriter = writer;
            try
            {
                foreach (var entry in dict)
                {
                    MemoryStream ms = new MemoryStream(128);
                    writer = ms;
                    Write("", entry.Value);
                    byte[] valueBytes = ms.ToArray();
                    bytes[entry.Key] = valueBytes;
                    numDataBytes += valueBytes.Length;
                }
            }
            finally
            {
                writer = oldWriter;
            }

            // compute total size of buffer
            const int fieldNameLimit = 64;
            int maxFieldNameLength = 0;
            foreach (var key in dict.Keys)
            {
                maxFieldNameLength = System.Math.Max(maxFieldNameLength, key.Length);
            }
            // add 1 byte for the null terminator
            maxFieldNameLength++;
            maxFieldNameLength = System.Math.Min(maxFieldNameLength, fieldNameLimit);
            // round up to multiple of 8 bytes
            maxFieldNameLength = ((maxFieldNameLength + 7)/8)*8;
            //maxFieldNameLength = fieldNameLimit;
            int numFieldNameBytes = maxFieldNameLength * dict.Count;
            int numBytes = 48 + numNameBytes + 16 + numFieldNameBytes + numDataBytes;

            // write the data type field
            Write(MatType.MATRIX);
            Write(numBytes - 8); // number of bytes 

            // write the array flags
            Write(MatType.UINT32);
            Write(8); // 8 bytes to follow
            int dataClass = (int) mxClass.STRUCT;
            Write(dataClass);
            Write(0); // reserved

            // write the dimension field
            Write(MatType.INT32);
            Write(8); // 8 bytes to follow
            Write(1); // number of rows
            Write(1); // number of columns

            // write the name
            Write(MatType.INT8);
            Write(name.Length); // length of (padded) name in bytes
            Write(name, numNameBytes);

            // write the field name length
            Write((int) MatType.INT32 | (4 << 16));
            Write(maxFieldNameLength);

            // write the field names
            Write(MatType.INT8);
            Write(numFieldNameBytes);
            foreach (var entry in dict)
            {
                string truncatedKey = entry.Key;
                if (entry.Key.Length >= maxFieldNameLength)
                {
                    truncatedKey = entry.Key.Substring(0, maxFieldNameLength - 1);
                    Console.WriteLine(string.Format("dictionary key '{0}' exceeds the MATLAB maximum name length of {1} characters and will be truncated to '{2}'", 
                        entry.Key, (maxFieldNameLength - 1), truncatedKey));
                }
                Write(truncatedKey, maxFieldNameLength);
            }

            // write the field values
            foreach (var entry in dict)
            {
                Write(bytes[entry.Key]);
            }
        }

        private static mxClass GetDataClass(Type type)
        {
            if (type.Equals(typeof(object))) return mxClass.CELL;
            else if (type.Equals(typeof(double))) return mxClass.DOUBLE;
            else if (type.Equals(typeof(Single))) return mxClass.SINGLE;
            else if (type.Equals(typeof(byte)))
                return mxClass.INT8;
            else if (type.Equals(typeof(short)))
                return mxClass.INT16;
            else if (type.Equals(typeof(ushort)))
                return mxClass.UINT16;
            else if (type.Equals(typeof(int)))
                return mxClass.INT32;
            else if (type.Equals(typeof(uint)))
                return mxClass.UINT32;
            else if (type.Equals(typeof(long)))
                return mxClass.INT64;
            else if (type.Equals(typeof(ulong)))
                return mxClass.UINT64;
            else
                throw new NotSupportedException("type " + StringUtil.TypeToString(type) + " not supported");
        }

        private static MatType GetElementClass(Type type)
        {
            if (type.Equals(typeof(double))) return MatType.DOUBLE;
            else if (type.Equals(typeof(Single))) return MatType.SINGLE;
            else if (type.Equals(typeof(byte)))
                return MatType.INT8;
            else if (type.Equals(typeof(short)))
                return MatType.INT16;
            else if (type.Equals(typeof(ushort)))
                return MatType.UINT16;
            else if (type.Equals(typeof(int)))
                return MatType.INT32;
            else if (type.Equals(typeof(uint)))
                return MatType.UINT32;
            else if (type.Equals(typeof(long)))
                return MatType.INT64;
            else if (type.Equals(typeof(ulong)))
                return MatType.UINT64;
            else
                throw new NotSupportedException("type " + StringUtil.TypeToString(type) + " not supported");
        }

        /// <summary>
        /// Write a named array to the stream
        /// </summary>
        /// <param name="name"></param>
        /// <param name="array"></param>
        public void WriteArray(string name, Array array)
        {
            Type eltType = array.GetType().GetElementType();
            mxClass dataClass = GetDataClass(eltType);

            // convert to a valid variable name
            name = MakeValidVariableName(name);

            // pad the name to an 8-byte boundary
            int numNameBytes = ((name.Length + 7) / 8) * 8;

            // precompute the cell bytes
            int numDataBytes = 0;
            int numDims = array.Rank;
            int numElts = array.Length;
            int[] sizes = new int[numDims];
            for (int i = 0; i < numDims; i++)
            {
                sizes[i] = array.GetLength(i);
            }
            MatlabReader.Reverse(sizes);
            int[] strides = StringUtil.ArrayStrides(sizes);
            MatlabReader.Reverse(sizes);
            int[] index = new int[numDims];
            List<byte[]> bytes = new List<byte[]>();
            Stream oldWriter = writer;
            try
            {
                for (int i = 0; i < numElts; i++)
                {
                    StringUtil.LinearIndexToMultidimensionalIndex(i, strides, index);
                    MatlabReader.Reverse(index);
                    object cell = array.GetValue(index);
                    MemoryStream ms = new MemoryStream(128);
                    writer = ms;
                    if (dataClass == mxClass.CELL) Write("", cell);
                    else if (dataClass == mxClass.DOUBLE) Write((double)cell);
                    else if (dataClass == mxClass.SINGLE) Write((Single)cell);
                    else if (dataClass == mxClass.INT8) Write((byte)cell);
                    else if (dataClass == mxClass.INT16) Write((short)cell);
                    else if (dataClass == mxClass.UINT16) Write((ushort)cell);
                    else if (dataClass == mxClass.INT32) Write((int)cell);
                    else if (dataClass == mxClass.UINT32) Write((uint)cell);
                    else if (dataClass == mxClass.INT64) Write((long)cell);
                    else if (dataClass == mxClass.UINT64) Write((ulong)cell);
                    else throw new NotImplementedException(dataClass.ToString());
                    byte[] valueBytes = ms.ToArray();
                    bytes.Add(valueBytes);
                    numDataBytes += valueBytes.Length;
                }
            }
            finally
            {
                writer = oldWriter;
            }

            // compute total size of buffer
            int sizeBytes = numDims * 4;
            if (numDims % 2 == 1) sizeBytes += 4;
            int numBytes = 32 + sizeBytes + numNameBytes + numDataBytes;
            if (dataClass != mxClass.CELL) numBytes += 8;

            // write the data type field
            Write(MatType.MATRIX);
            Write(numBytes); // number of bytes 

            // write the array flags
            Write(MatType.UINT32);
            Write(8); // 8 bytes to follow
            Write((int)dataClass);
            Write(0); // reserved

            // write the dimension field
            Write(MatType.INT32);
            if (numDims == 1)
            {
                Write(8); // 8 bytes to follow
                Write(sizes[0]);
                Write(1);
            }
            else
            {
                Write(numDims * 4);
                for (int i = 0; i < numDims; i++)
                {
                    Write(sizes[i]);
                }
                if (numDims % 2 == 1) Write(0);
            }

            // write the name
            Write(MatType.INT8);
            Write(name.Length); // length of name in bytes
            Write(name, numNameBytes);

            if (dataClass != mxClass.CELL)
            {
                MatType elementClass = GetElementClass(eltType);
                Write((int)elementClass);
                Write(numDataBytes);
            }
            // write the cell values
            for (int i = 0; i < bytes.Count; i++)
            {
                Write(bytes[i]);
            }
        }

        /// <summary>
        /// Write a named cell array to the stream
        /// </summary>
        /// <param name="name"></param>
        /// <param name="array"></param>
        public void Write(string name, object[,] array)
        {
            WriteArray(name, array);
        }

        /// <summary>
        /// Write a named string to the stream
        /// </summary>
        /// <param name="name"></param>
        /// <param name="value"></param>
        public void Write(string name, string value)
        {
            // convert to a valid variable name
            name = MakeValidVariableName(name);

            // pad the name to an 8-byte boundary
            int numNameBytes = ((name.Length + 7)/8)*8;

            // pad the data to an 8-byte boundary
            int numDataBytes = ((value.Length + 7)/8)*8;

            // compute total size of buffer
            int numBytes = 48 + numNameBytes + 8 + numDataBytes;

            // write the data type field
            Write(MatType.MATRIX);
            Write(numBytes - 8); // number of bytes 

            // write the array flags
            Write(MatType.UINT32);
            Write(8); // 8 bytes to follow
            int dataClass = (int) mxClass.CHAR;
            Write(dataClass);
            Write(0);

            // write the dimension field
            Write(MatType.INT32);
            Write(8); // 8 bytes to follow
            Write(1); // number of rows
            Write(value.Length); // number of columns

            // write the name
            Write(MatType.INT8);
            Write(name.Length); // length of name in bytes
            Write(name, numNameBytes);

            // write the string
            Write(MatType.UTF8);
            Write(value.Length); // bytes to follow
            Write(value, numDataBytes);
        }

        /// <summary>
        /// Converts a variable name to a valid variable name by changing all invalid characters to an underscore.
        /// </summary>
        /// <param name="name">Variable name</param>
        /// <returns>A valid MATLAB variable name.</returns>
        private static string MakeValidVariableName(string name)
        {
            string result = "";
            for (int i = 0; i < name.Length; i++)
            {
                if (!Char.IsDigit(name[i]) && !Char.IsLetter(name[i]))
                    result += '_';
                else
                    result += name[i];
            }

            // Matlab variable names cannot start with an underscore.
            while (result.Length > 0 && result[0] == '_') result = result.Substring(1);
            return result;
        }

        /// <summary>
        /// Generate a Matlab V5 compatible header byte array
        /// </summary>
        /// <param name="comments">The comments that will go into the header of the Matlab file.</param>
        /// <returns>The header as a byte array.</returns>
        protected static byte[] Matlab5Header(string comments)
        {
            byte[] header = new byte[128];
            if (comments.Length > 124)
                comments = comments.Substring(0, 124);
            else
                comments = comments.PadRight(124);
            for (int i = 0; i < comments.Length; i++)
                header[i] = (byte) comments[i];
            header[124] = 0x00;
            header[125] = 0x01;
            header[126] = (byte) 'I';
            header[127] = (byte) 'M';
            return header;
        }

        /// <summary>
        /// Release the stream used by a MatlabWriter
        /// </summary>
        public void Dispose()
        {
            writer.Dispose();
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// A number type in a MAT file
    /// </summary>
    public enum MatType
    {
        INT8 = 1,
        UINT8 = 2,
        INT16 = 3,
        UINT16 = 4,
        INT32 = 5,
        UINT32 = 6,
        SINGLE = 7,
        DOUBLE = 9,
        INT64 = 12,
        UINT64 = 13,
        MATRIX = 14,
        COMPRESSED = 15,
        UTF8 = 16,
        UTF16 = 17,
        UTF32 = 18
    }

    /// <summary>
    /// An object class in a MAT file
    /// </summary>
    public enum mxClass
    {
        CELL = 1,
        STRUCT = 2,
        OBJECT = 3,
        CHAR = 4,
        SPARSE = 5,
        DOUBLE = 6,
        SINGLE = 7,
        INT8 = 8,
        UINT8 = 9,
        INT16 = 10,
        UINT16 = 11,
        INT32 = 12,
        UINT32 = 13,
        INT64 = 14,
        UINT64 = 15
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}