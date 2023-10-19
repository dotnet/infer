// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.IO;
    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// A reader that takes a delegate when parsing objects.
    /// </summary>
    public class DelegatingWrappedBinaryReader : IReader, IDisposable
    {
        /// <summary>
        /// Parse a string into the requested type.
        /// </summary>
        /// <param name="str">A formatted string.</param>
        /// <param name="type">The requested type to deserialize into.</param>
        /// <returns>An instance of the requested type.</returns>
        public delegate object ParseDelegate(string str, Type type);

        readonly BinaryReader binaryReader;
        readonly ParseDelegate objectReader;

        /// <summary>
        /// Initialises a new instance of <see cref="DelegatingWrappedBinaryReader"/>.
        /// </summary>
        /// <param name="binaryReader">A binary reader.</param>
        /// <param name="objectReader">An object reader.</param>
        public DelegatingWrappedBinaryReader(
            BinaryReader binaryReader,
            ParseDelegate objectReader)
        {
            this.binaryReader = binaryReader;
            this.objectReader = objectReader;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            binaryReader.Dispose();
        }

        /// <inheritdoc/>
        public bool ReadBoolean()
        {
            return binaryReader.ReadBoolean();
        }

        /// <inheritdoc/>
        public double ReadDouble()
        {
            return binaryReader.ReadDouble();
        }

        /// <inheritdoc/>
        public Guid ReadGuid()
        {
            return binaryReader.ReadGuid();
        }

        /// <inheritdoc/>
        public int ReadInt32()
        {
            return binaryReader.ReadInt32();
        }

        /// <inheritdoc/>
        public T ReadObject<T>()
        {
            var str = ReadString();
            var obj = this.objectReader(str, typeof(T));

            if (obj is T requestedObj)
            {
                return requestedObj;

            }

            throw new InvalidOperationException($"Object reader returned a '{obj?.GetType()}' instead of requested type '{typeof(T)}'.");
        }

        /// <inheritdoc/>
        public string ReadString()
        {
            return binaryReader.ReadString();
        }
    }
}
