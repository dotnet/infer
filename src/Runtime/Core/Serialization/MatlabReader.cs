// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Reference: "The MAT file format"
// http://www.mathworks.com/access/helpdesk/help/pdf_doc/matlab/matfile_format.pdf

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.IO.Compression;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using System.Globalization;

namespace Microsoft.ML.Probabilistic.Serialization
{
    /// <summary>
    /// Reads the contents of a MAT file.
    /// </summary>
    /// <remarks>
    /// The MAT file format is defined in <a href="http://www.mathworks.com/access/helpdesk/help/pdf_doc/matlab/matfile_format.pdf">The MAT file format</a>.
    /// </remarks>
    public class MatlabReader : IDisposable
    {
        private Stream reader;

        /// <summary>
        /// Create a MatlabReader that reads from the given file name.
        /// </summary>
        /// <param name="fileName">The MAT file name to read from</param>
        protected MatlabReader(string fileName)
        {
            reader = File.OpenRead(fileName);
            string header = Matlab5Header();
        }

        /// <summary>
        /// Create a MatlabReader that reads from the given stream.
        /// </summary>
        /// <param name="reader">The stream to read from</param>
        protected MatlabReader(Stream reader)
        {
            this.reader = reader;
        }

        /// <summary>
        /// Read all variables from a file and store them in a dictionary
        /// </summary>
        /// <param name="fileName">The name of a MAT file</param>
        /// <returns>A dictionary that maps variable names into values</returns>
        public static Dictionary<string, object> Read(string fileName)
        {
            using (MatlabReader r = new MatlabReader(fileName))
            {
                return r.ReadAll();
            }
        }

        /// <summary>
        /// Read all variables from the current stream and store them in a dictionary
        /// </summary>
        /// <returns>A dictionary that maps variable names into values</returns>
        protected Dictionary<string, object> ReadAll()
        {
            Dictionary<string, object> dict = new Dictionary<string, object>();
            while (true)
            {
                try
                {
                    string name;
                    object m = Matlab5DataElement(out name);
                    dict[name] = m;
                }
                catch (EndOfStreamException)
                {
                    break;
                }
                catch (NotImplementedException ex)
                {
                    Console.WriteLine("MatlabReader could not parse data element: "+ex.Message);
                }
            }
            return dict;
        }

        /// <summary>
        /// Read the next data element from the stream
        /// </summary>
        /// <param name="name">On exit, the variable name</param>
        /// <returns>The data value, as a .NET object</returns>
        protected object Matlab5DataElement(out string name)
        {
            MatType dataType = (MatType) ReadInt();
            int numBytes = ReadInt();
            byte[] bytes = new byte[numBytes];
            if (Read(reader, bytes, 0, numBytes) < numBytes) throw new EndOfStreamException();
            if (dataType == MatType.COMPRESSED)
            {
                // MAT file compression uses Zlib format, which is the same as Deflate but with an extra header and footer.
                // To use DeflateStream, we must first remove this header.
                bytes = RemoveRfc1950Header(bytes);
                var stream = new DeflateStream(new MemoryStream(bytes), CompressionMode.Decompress);
                MatlabReader unzipReader = new MatlabReader(stream);
                return unzipReader.Matlab5DataElement(out name);
            }
            else
            {
                MatlabReader eltReader = new MatlabReader(new MemoryStream(bytes));
                var result = eltReader.Parse(dataType, out name);
                return result;
            }
        }

        /// <summary>
        /// Removes the header defined by <a href="http://www.ietf.org/rfc/rfc1950.txt">RFC 1950</a>
        /// </summary>
        /// <param name="bytes"></param>
        /// <returns>A smaller array of bytes</returns>
        /// <remarks>
        /// The header to remove is 2 or 6 bytes at the start, and 4 bytes at the end.
        /// </remarks>
        protected static byte[] RemoveRfc1950Header(byte[] bytes)
        {
            int trimBegin = 2;
            bool fdict = (bytes[1] & (1 << 5)) > 0;
            if (fdict) trimBegin += 4;
            int trimEnd = 4;
            byte[] result = new byte[bytes.Length - trimBegin - trimEnd];
            Array.Copy(bytes, trimBegin, result, 0, result.Length);
            return result;
        }

        /// <summary>
        /// Read a specific data type from the stream
        /// </summary>
        /// <param name="dataType">The type number as documented by the MAT format</param>
        /// <param name="name">On exit, the variable name</param>
        /// <returns>The data value, as a .NET object</returns>
        protected object Parse(MatType dataType, out string name)
        {
            if (dataType == MatType.COMPRESSED)
            {
                throw new NotImplementedException();
            }
            if (dataType != MatType.MATRIX)
                throw new NotImplementedException("dataType = " + dataType + " not implemented");
            // Flags
            MatType flagsType = (MatType)ReadInt();
            if (flagsType != MatType.UINT32)
                throw new NotImplementedException("flagsType = " + flagsType + " not recognized");
            int numFlagsBytes = ReadInt();
            if (numFlagsBytes != 8)
                throw new NotImplementedException("numFlagsBytes = " + numFlagsBytes + " not recognized");
            int flags = ReadInt();
            bool isLogical = (flags & 0x0200) > 0;
            bool isComplex = (flags & 0x0800) > 0;
            mxClass dataClass = (mxClass)(flags & 0xff);
            if (dataClass > mxClass.UINT64)
                throw new NotImplementedException("dataClass = " + dataClass + " not recognized");
            ReadInt(); // ignored

            // Size
            MatType sizeType = (MatType)ReadInt();
            if (sizeType != MatType.INT32)
                throw new NotImplementedException("sizeType = " + sizeType + " not recognized");
            int numSizeBytes = ReadInt();
            int numDims = numSizeBytes / 4;
            int[] sizes = new int[numDims];
            int numDimsGreaterThanOne = 0;
            for (int i = 0; i < numDims; i++)
            {
                sizes[i] = ReadInt();
                if (sizes[i] > 1)
                    numDimsGreaterThanOne++;
            }
            ReadPadding(numSizeBytes, false);
            bool isVector = (numDimsGreaterThanOne <= 1);
            int length = sizes[0]*sizes[1];

            // Name
            name = ReadString();

            if (dataClass == mxClass.CHAR)
            {
                return ReadString();
            }
            else if (dataClass == mxClass.CELL)
            {
                return ReadCellArray(sizes);
            }
            else if (dataClass == mxClass.STRUCT)
            {
                return ReadStruct();
            }
            else if (dataClass == mxClass.OBJECT || dataClass == mxClass.SPARSE)
            {
                throw new NotImplementedException("dataClass = " + dataClass + " not implemented");
            }
            else if (isVector && dataClass == mxClass.INT32)
            {
                return ReadIntArray(length);
            }
            else if (isVector && dataClass == mxClass.UINT32)
            {
                return ReadUIntArray(length);
            }
            else if (isVector && dataClass == mxClass.INT64)
            {
                return ReadLongArray(length);
            }
            else if (isVector && dataClass == mxClass.UINT64)
            {
                return ReadULongArray(length);
            }
            else if (isVector && isLogical)
            {
                return ReadBoolArray(length);
            }
            else if (numDims == 2)
            {
                // Elements
                int rows = sizes[0];
                int cols = sizes[1];
                object real = ReadMatrix(dataClass, rows, cols);
                if (isComplex)
                {
                    object imaginary = ReadMatrix(dataClass, rows, cols);
                    ComplexMatrix cm = new ComplexMatrix();
                    cm.Real = (Matrix)real;
                    cm.Imaginary = (Matrix)imaginary;
                    return cm;
                }
                else
                {
                    return real;
                }
            }
            else
            {
                if (isComplex)
                    throw new NotSupportedException("complex multi-dimensional arrays are not supported");
                // multi-dimensional array
                return ReadArray(dataClass, sizes);
            }
        }

        protected string ReadString()
        {
            MatType type;
            int length;
            bool isSmallFormat = ReadTypeAndSize(out type, out length);
            if (type != MatType.INT8 && type != MatType.UTF8) throw new NotImplementedException("string type = " + type + " not implemented");
            string name = ReadString(length);
            ReadPadding(length, isSmallFormat);
            return name;
        }

        /// <summary>
        /// Represents a matrix of complex numbers, when reading and writing MAT files.
        /// </summary>
        public class ComplexMatrix
        {
            public Matrix Real, Imaginary;
        }

        /// <summary>
        /// Get the size in bytes of a MatType
        /// </summary>
        /// <param name="type"></param>
        /// <returns>Number of bytes</returns>
        public static int SizeOf(MatType type)
        {
            switch (type)
            {
                case MatType.INT8:
                    return 1;
                case MatType.UINT8:
                    return 1;
                case MatType.INT16:
                    return 2;
                case MatType.UINT16:
                    return 2;
                case MatType.INT32:
                    return 4;
                case MatType.UINT32:
                    return 4;
                case MatType.SINGLE:
                    return 4;
                case MatType.DOUBLE:
                    return 8;
                case MatType.INT64:
                    return 8;
                case MatType.UINT64:
                    return 8;
                default:
                    throw new ArgumentException();
            }
        }

        protected bool ReadTypeAndSize(out MatType type, out int numBytes)
        {
            int typeAndSize = ReadInt();
            bool isSmallFormat = (typeAndSize > 0xffff);
            if (isSmallFormat)
            {
                // small element format
                numBytes = typeAndSize >> 16;
                type = (MatType) (typeAndSize & 0xffff);
            }
            else
            {
                numBytes = ReadInt();
                type = (MatType) typeAndSize;
            }
            return isSmallFormat;
        }

        protected void ReadPadding(int numBytesRead, bool isSmallFormat)
        {
            if (isSmallFormat)
            {
                for (int i = numBytesRead; i < 4; i++)
                {
                    reader.ReadByte();
                }
            }
            else
            {
                for (int i = numBytesRead%8; (i > 0) && (i < 8); i++)
                {
                    reader.ReadByte();
                }
            }
        }

        protected Dictionary<string, object> ReadStruct()
        {
            ReadInt(); // ignored
            int fieldNameLength = ReadInt();
            MatType fieldNameType = (MatType) ReadInt();
            if (fieldNameType != MatType.INT8) throw new NotImplementedException("fieldNameType = " + fieldNameType + " not recognized");
            int numFieldNameBytes = ReadInt();
            int numberOfFields = numFieldNameBytes/fieldNameLength;
            string[] fields = new string[numberOfFields];
            for (int i = 0; i < numberOfFields; i++)
            {
                fields[i] = ReadString(fieldNameLength);
            }
            ReadPadding(numFieldNameBytes, false);
            Dictionary<string, object> dict = new Dictionary<string, object>();
            for (int i = 0; i < numberOfFields; i++)
            {
                string dummyName;
                MatType fieldValueType = (MatType) ReadInt();
                int fieldValueSize = ReadInt();
                dict[fields[i]] = Parse(fieldValueType, out dummyName);
            }
            return dict;
        }

        protected object ReadBoolArray(int count)
        {
            MatType elementClass;
            int numDataBytes;
            bool isSmallFormat = ReadTypeAndSize(out elementClass, out numDataBytes);
            bool[] array = new bool[count];
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = ((int)ReadElement(elementClass) != 0);
            }
            ReadPadding(numDataBytes, isSmallFormat);
            return array;
        }

        protected object ReadIntArray(int count)
        {
            MatType elementClass;
            int numDataBytes;
            bool isSmallFormat = ReadTypeAndSize(out elementClass, out numDataBytes);
            int[] array = new int[count];
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = (int) ReadElement(elementClass);
            }
            ReadPadding(numDataBytes, isSmallFormat);
            return array;
        }

        protected object ReadUIntArray(int count)
        {
            MatType elementClass;
            int numDataBytes;
            bool isSmallFormat = ReadTypeAndSize(out elementClass, out numDataBytes);
            uint[] array = new uint[count];
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = (uint)ReadElement(elementClass);
            }
            ReadPadding(numDataBytes, isSmallFormat);
            return array;
        }

        protected object ReadLongArray(int count)
        {
            MatType elementClass;
            int numDataBytes;
            bool isSmallFormat = ReadTypeAndSize(out elementClass, out numDataBytes);
            long[] array = new long[count];
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = (long)ReadElement(elementClass);
            }
            ReadPadding(numDataBytes, isSmallFormat);
            return array;
        }

        protected object ReadULongArray(int count)
        {
            MatType elementClass;
            int numDataBytes;
            bool isSmallFormat = ReadTypeAndSize(out elementClass, out numDataBytes);
            ulong[] array = new ulong[count];
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = (ulong)ReadElement(elementClass);
            }
            ReadPadding(numDataBytes, isSmallFormat);
            return array;
        }

        protected Matrix ReadMatrix(mxClass dataClass, int rows, int cols)
        {
            Matrix result = new Matrix(rows, cols);
            MatType elementClass;
            int numDataBytes;
            bool isSmallFormat = ReadTypeAndSize(out elementClass, out numDataBytes);
            for (int j = 0; j < result.Cols; j++)
            {
                for (int i = 0; i < result.Rows; i++)
                {
                    result[i, j] = (double)Convert.ChangeType(ReadElement(elementClass), typeof(double));
                }
            }
            ReadPadding(numDataBytes, isSmallFormat);
            return result;
        }

        protected Array ReadCellArray(int[] sizes)
        {
            Array cellArray = Array.CreateInstance(typeof (object), sizes);
            int numElts = cellArray.Length;
            Reverse(sizes);
            int[] strides = StringUtil.ArrayStrides(sizes);
            Reverse(sizes);
            int[] index = new int[sizes.Length];
            bool allStrings = true;
            for (int i = 0; i < numElts; i++)
            {
                string dummyName;
                MatType cellValueType = (MatType) ReadInt();
                int cellValueSize = ReadInt();
                object value = (cellValueSize == 0) ? null : Parse(cellValueType, out dummyName);
                StringUtil.LinearIndexToMultidimensionalIndex(i, strides, index);
                Reverse(index);
                cellArray.SetValue(value, index);
                if (!(value is string))
                    allStrings = false;
            }
            if (numElts > 0 && allStrings && sizes.Length == 2 && sizes[1] == 1)
            {
                string[] stringArray = new string[numElts];
                for (int i = 0; i < numElts; i++)
                {
                    StringUtil.LinearIndexToMultidimensionalIndex(i, strides, index);
                    Reverse(index);
                    object value = cellArray.GetValue(index);
                    stringArray[i] = (string)value;
                }
                return stringArray;
            }
            return cellArray;
        }

        protected Array ReadArray(mxClass dataClass, int[] sizes)
        {
            Array result = Array.CreateInstance(GetType(dataClass), sizes);
            int numElts = result.Length;
            Reverse(sizes);
            int[] strides = StringUtil.ArrayStrides(sizes);
            Reverse(sizes);
            int[] index = new int[sizes.Length];
            MatType elementClass;
            int numDataBytes;
            bool isSmallFormat = ReadTypeAndSize(out elementClass, out numDataBytes);
            for (int i = 0; i < numElts; i++)
            {
                StringUtil.LinearIndexToMultidimensionalIndex(i, strides, index);
                Reverse(index);
                object value = ReadElement(elementClass);
                result.SetValue(value, index);
            }
            ReadPadding(numDataBytes, isSmallFormat);
            return result;
        }

        internal static void Reverse<T>(T[] array)
        {
            int count = array.Length;
            int half = count/2;
            for (int i = 0; i < half; i++)
            {
                int j = count - 1 - i;
                T temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }

        protected Type GetType(mxClass dataClass)
        {
            switch (dataClass)
            {
                case mxClass.DOUBLE:
                    return typeof(double);
                case mxClass.SINGLE:
                    return typeof(Single);
                case mxClass.INT8:
                    return typeof(byte);
                case mxClass.INT32:
                    return typeof(int);
                case mxClass.UINT32:
                    return typeof(uint);
                case mxClass.INT64:
                    return typeof(long);
                case mxClass.UINT64:
                    return typeof(ulong);
                default:
                    throw new ArgumentException("Cannot convert mxClass " + dataClass + " to Type");
            }
        }

        protected object ReadElement(MatType elementClass)
        {
            if (elementClass == MatType.DOUBLE)
                return ReadDouble();
            else if (elementClass == MatType.UINT8)
                return reader.ReadByte();
            else if (elementClass == MatType.INT16)
                return ReadInt16();
            else if (elementClass == MatType.UINT16)
                return ReadUInt16();
            else if (elementClass == MatType.INT32)
                return ReadInt();
            else if (elementClass == MatType.UINT32)
                return ReadUInt();
            else if (elementClass == MatType.SINGLE)
                return ReadSingle();
            else if (elementClass == MatType.INT64)
                return ReadInt64();
            else if (elementClass == MatType.UINT64)
                return ReadUInt64();
            else throw new NotImplementedException("elementClass = " + elementClass + " not recognized");
        }

        /// <summary>
        /// Read a MAT file header from the stream
        /// </summary>
        /// <returns>The header as a string</returns>
        protected string Matlab5Header()
        {
            string header = ReadString(124);
            byte[] bytes = new byte[4];
            if (Read(reader, bytes, 0, 4) < 4) throw new EndOfStreamException();
            if ((bytes[0] != 0x00) || (bytes[1] != 0x01) || (bytes[2] != 'I') || (bytes[3] != 'M'))
                throw new NotImplementedException("Unrecognized header");
            return header.TrimEnd(' ');
        }

        /// <summary>
        /// Read a 32-bit integer from the stream
        /// </summary>
        /// <returns></returns>
        protected int ReadInt()
        {
            byte[] bytes = new byte[4];
            if (Read(reader, bytes, 0, 4) < 4) throw new EndOfStreamException();
            return BitConverter.ToInt32(bytes, 0);
        }

        /// <summary>
        /// Read an unsigned 32-bit integer from the stream
        /// </summary>
        /// <returns></returns>
        protected uint ReadUInt()
        {
            byte[] bytes = new byte[4];
            if (Read(reader, bytes, 0, 4) < 4) throw new EndOfStreamException();
            return BitConverter.ToUInt32(bytes, 0);
        }

        /// <summary>
        /// Read a 16-bit integer from the stream
        /// </summary>
        /// <returns></returns>
        protected short ReadInt16()
        {
            byte[] bytes = new byte[2];
            if (Read(reader, bytes, 0, 2) < 2) throw new EndOfStreamException();
            return BitConverter.ToInt16(bytes, 0);
        }

        /// <summary>
        /// Read an unsigned 16-bit integer from the stream
        /// </summary>
        /// <returns></returns>
        protected ushort ReadUInt16()
        {
            byte[] bytes = new byte[2];
            if (Read(reader, bytes, 0, 2) < 2) throw new EndOfStreamException();
            return BitConverter.ToUInt16(bytes, 0);
        }

        /// <summary>
        /// Read a 64-bit integer from the stream
        /// </summary>
        /// <returns></returns>
        protected long ReadInt64()
        {
            byte[] bytes = new byte[8];
            if (Read(reader, bytes, 0, 8) < 8) throw new EndOfStreamException();
            return BitConverter.ToInt64(bytes, 0);
        }

        /// <summary>
        /// Read an unsigned 64-bit integer from the stream
        /// </summary>
        /// <returns></returns>
        protected ulong ReadUInt64()
        {
            byte[] bytes = new byte[8];
            if (Read(reader, bytes, 0, 8) < 8) throw new EndOfStreamException();
            return BitConverter.ToUInt64(bytes, 0);
        }

        /// <summary>
        /// Read a single-precision float from the stream
        /// </summary>
        /// <returns></returns>
        protected float ReadSingle()
        {
            byte[] bytes = new byte[4];
            if (Read(reader, bytes, 0, 4) < 4) throw new EndOfStreamException();
            return BitConverter.ToSingle(bytes, 0);
        }

        /// <summary>
        /// Read a double-precision float from the stream
        /// </summary>
        /// <returns></returns>
        protected double ReadDouble()
        {
            byte[] bytes = new byte[8];
            if (Read(reader, bytes, 0, 8) < 8) throw new EndOfStreamException();
            return BitConverter.ToDouble(bytes, 0);
        }

        /// <summary>
        /// Read a string of known length from the stream
        /// </summary>
        /// <param name="length">The number of bytes in the string</param>
        /// <returns></returns>
        protected string ReadString(int length)
        {
            byte[] bytes = new byte[length];
            if (Read(reader, bytes, 0, length) < length) throw new EndOfStreamException();
            return ParseString(bytes, 0, bytes.Length);
        }

        /// <summary>
        /// Convert an array of bytes into a string
        /// </summary>
        /// <param name="bytes">The bytes</param>
        /// <param name="start">The position of the first character in the string</param>
        /// <param name="length">An upper bound on the length of the string</param>
        /// <returns></returns>
        protected static string ParseString(byte[] bytes, int start, int length)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < length; i++)
            {
                byte b = bytes[start + i];
                if (b == 0x00) break;
                sb.Append((char) b);
            }
            return sb.ToString();
        }

        /// <summary>
        /// Release the stream used by this MatlabReader
        /// </summary>
        public void Dispose()
        {
            reader.Dispose();
        }

        /// <summary>
        /// Read a csv file (with a header row) as a set of columns
        /// </summary>
        /// <param name="path"></param>
        /// <remarks>Also see Microsoft.VisualBasic.FileIO.TextFieldParser</remarks>
        /// <returns>A mapping from column name to a list of column values</returns>
        public static Dictionary<string, List<string>> ReadCsv(string path)
        {
            Dictionary<string, List<string>> result = new Dictionary<string, List<string>>();
            using (StreamReader reader = File.OpenText(path))
            {
                string header = reader.ReadLine();
                if (header != null)
                {
                    if (header.Contains('"')) throw new NotSupportedException("double quotes are not supported");
                    string[] columnNames = header.Split(',');
                    foreach (var columnName in columnNames)
                    {
                        result[columnName] = new List<string>();
                    }
                    while (!reader.EndOfStream)
                    {
                        string line = reader.ReadLine();
                        if (line.Contains('"')) throw new NotSupportedException("double quotes are not supported");
                        string[] values = line.Split(',');
                        for (int i = 0; i < values.Length; i++)
                        {
                            var list = result[columnNames[i]];
                            list.Add(values[i]);
                        }
                    }
                }
            }
            return result;
        }

        public static double[,] ReadMatrix(double[,] matrix, string filename)
        {
            return ReadMatrix(matrix, filename, ' ');
        }

        public static double[,] ReadMatrix(double[,] matrix, string filename, char separator)
        {
            StreamReader sr = File.OpenText(filename);
            int row = 0;
            while (true)
            {
                int col = 0;
                string s = sr.ReadLine();
                if (s == null) break;
                string[] nums = s.Split(new char[] { separator }, StringSplitOptions.RemoveEmptyEntries);
                foreach (string s2 in nums)
                {
                    matrix[row, col] = Double.Parse(s2, CultureInfo.InvariantCulture);
                    col++;
                }
                if (col != matrix.GetLength(1)) throw new InferRuntimeException("Expected " + matrix.GetLength(1) + " columns, but were " + col + " columns.");
                row++;
                if (row == matrix.GetLength(0)) break;
            }
            if (row != matrix.GetLength(0)) throw new InferRuntimeException("Expected " + matrix.GetLength(0) + " rows, but were " + row + " rows.");
            return matrix;
        }

        protected int Read(Stream stream, byte[] bytes, int offset, int count)
        {
            // https://docs.microsoft.com/en-us/dotnet/core/compatibility/core-libraries/6.0/partial-byte-reads-in-streams
            var totalRead = 0;
            while (totalRead < count)
            {
                var byteRead = stream.Read(bytes, offset + totalRead, count - totalRead);
                if (byteRead == 0) break;
                totalRead += byteRead;
            }
            return totalRead;
        }
    }
}