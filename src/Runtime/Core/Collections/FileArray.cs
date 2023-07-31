// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.Serialization;
using System.Xml;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Serialization;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// An IArray where each element is stored as a disk file.
    /// </summary>
    /// <typeparam name="T">The element type</typeparam>
    /// <remarks><para>
    /// The elements (whose values do not equal default(T)) are stored as individual files in a temp folder.
    /// Elements equal to default(T) are not stored.
    /// A FileArray may contain FileArrays as elements: these are called child FileArrays.
    /// Child FileArrays must be created using the special constructor.
    /// When a FileArray (that is not a child) is Disposed or garbage collected, the folder is deleted.
    /// A child FileArray is deleted only when its parent is deleted.
    /// Thus if the program exits normally, all temp folders should be deleted.
    /// However, if the program halts with an exception, temp folders will not be deleted and must be cleaned up manually.
    /// </para><para>
    /// Be careful when modifying elements of the FileArray in place.  If fa is a FileArray, then the syntax
    /// <code>fa[i][j] = 0;</code> or <code>fa[i].property = 0;</code> will not have any effect.  This happens
    /// because <code>fa[i]</code> reads the element from disk but does not write it back again.  You must explicitly
    /// read the element, modify it, then write it back like so: <code>T item = fa[i];  item.property=0;  fa[i] = item;</code>
    /// </para>
    /// </remarks>
    [Serializable]
    public class FileArray<T> : IArray<T>, IReadOnlyList<T>, IDisposable, ICloneable
    {
        protected readonly int count;
        protected readonly string prefix;
        protected bool doNotDelete;
        internal bool containsFileArrays;

        public string Folder
        {
            get { return prefix; }
        }

        private static readonly object folderLock = new object();

        /// <summary>
        /// Create a file array
        /// </summary>
        /// <param name="folder">Temporary folder for storing data.  Will be deleted when the FileArray is disposed.</param>
        /// <param name="count"></param>
        /// <param name="func"></param>
        public FileArray(string folder, int count, [SkipIfUniform] Func<int, T> func)
        {
            this.count = count;
            folder = folder.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            this.prefix = folder + Path.DirectorySeparatorChar;
            lock (folderLock)
            {
                int suffix = 1;
                while (Directory.Exists(prefix))
                {
                    suffix++;
                    prefix = folder + suffix + Path.DirectorySeparatorChar;
                }
                //Console.WriteLine("creating "+prefix);
                Directory.CreateDirectory(prefix);
            }
            Init(func);
        }

        [Skip]
        public FileArray(string folder, int count) : this(folder, count, i => default(T))
        {
        }

        [Skip]
        public FileArray([IgnoreDeclaration] FileArray<FileArray<T>> parent, int index, int count)
            : this(parent.GetItemFolder(index), count)
        {
            parent.containsFileArrays = true;
            this.doNotDelete = true;
            parent.StoreItem(index, this);
        }

        private void Init(Func<int, T> func)
        {
            for (int i = 0; i < count; i++)
            {
                T item = func(i);
                this[i] = item;
            }
        }

        public virtual object Clone()
        {
            if (containsFileArrays) throw new NotImplementedException();
            string folder = prefix.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return new FileArray<T>(folder + "_clone", count, i => this[i]);
        }

        internal string GetItemFolder(int index)
        {
            string path = prefix + index;
            return path;
        }

        public static string GetTempFolder(string name)
        {
            string path = "FileArray" + Path.DirectorySeparatorChar + name;
            return path;
        }

        public int Rank
        {
            get { return 1; }
        }

        public int GetLength(int dimension)
        {
            if (dimension == 0) return Count;
            else throw new IndexOutOfRangeException("requested dimension " + dimension + " of a 1D array.");
        }

        public int IndexOf(T item)
        {
            throw new NotImplementedException();
        }

        public void Insert(int index, T item)
        {
            throw new NotImplementedException();
        }

        public void RemoveAt(int index)
        {
            throw new NotImplementedException();
        }

        public T this[int index]
        {
            get
            {
                FileStats.AddRead();
                string path = GetFilePath(index);
                if (!File.Exists(path)) return default(T);
                var serializer = GetSerializer();
                using (var reader = XmlDictionaryReader.CreateTextReader(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read), new XmlDictionaryReaderQuotas()))
                {
                    return (T)serializer.ReadObject(reader);
                }
            }
            set { if (!containsFileArrays) StoreItem(index, value); }
        }

        private string GetFilePath(int index) =>
            prefix + index.ToString(CultureInfo.InvariantCulture) + ".bin";

        private DataContractSerializer GetSerializer() =>
            new DataContractSerializer(
                typeof(T),
                new DataContractSerializerSettings
                {
                    DataContractResolver = new InferDataContractResolver()
                });

        internal void StoreItem(int index, T value)
        {
            FileStats.AddWrite();
            string path = GetFilePath(index);
            if (object.ReferenceEquals(value, null) || value.Equals(default(T))) File.Delete(path);
            else
            {
                var serializer = GetSerializer();
                using (var writer = XmlDictionaryWriter.CreateTextWriter(new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None)))
                {
                    serializer.WriteObject(writer, value);
                }
            }
        }

        public void Add(T item)
        {
            throw new NotImplementedException();
        }

        public void Clear()
        {
            for (int i = 0; i < count; i++)
            {
                string path = prefix + i.ToString(CultureInfo.InvariantCulture) + ".bin";
                File.Delete(path);
            }
        }

        public bool Contains(T item)
        {
            return (IndexOf(item) != -1);
        }

        public void CopyTo(T[] array, int arrayIndex)
        {
            for (int i = 0; i < count; i++)
            {
                array[i + arrayIndex] = this[i];
            }
        }

        public int Count
        {
            get { return count; }
        }

        public bool IsReadOnly
        {
            get { return false; }
        }

        public bool Remove(T item)
        {
            throw new NotImplementedException();
        }

        public IEnumerator<T> GetEnumerator()
        {
            for (int i = 0; i < count; i++)
            {
                yield return this[i];
            }
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (!doNotDelete)
            {
                //Console.WriteLine("deleting "+prefix);
                Directory.Delete(prefix, true);
                doNotDelete = true;
            }
        }

        ~FileArray()
        {
            Dispose(false);
        }
    }

    /// <summary>
    /// Stores global counters for monitoring FileArrays
    /// </summary>
    internal static class FileStats
    {
        internal static bool debug;
        private static object countLock = new object();
        private static int readCount, writeCount;

        public static void Clear()
        {
            lock (countLock)
            {
                readCount = 0;
                writeCount = 0;
            }
        }

        public static int ReadCount
        {
            get { return readCount; }
        }

        public static int WriteCount
        {
            get { return writeCount; }
        }

        public static void AddRead()
        {
            lock (countLock)
            {
                readCount++;
            }
        }

        public static void AddWrite()
        {
            lock (countLock)
            {
                writeCount++;
            }
        }
    }

#if false
    public class TempFolderGenerator : IDisposable
    {
        string prefix = Path.GetTempPath()+@"\";
        Set<string> folders = new Set<string>();

        public string GetFolder(string name)
        {
            string path = prefix+name+@"\";
            int count = 1;
            while (Directory.Exists(path)) {
                count++;
                path = prefix+name+count+@"\";
            }
            folders.Add(path);
            Directory.CreateDirectory(path);
            return path;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        void Dispose(bool disposing)
        {
            foreach (string path in folders) {
                // will throw if directory is not empty
                Directory.Delete(path);
            }
        }

        ~TempFolderGenerator()
        {
            Dispose(false);
        }
    }
#endif
}