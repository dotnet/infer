// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Tests
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using Microsoft.ML.Probabilistic.Serialization;

    internal class DelegatingWrappedBinaryReader : IReader, IDisposable
    {
        // We need to use an interface because you can't have open generic delegate types in C#.
        public interface IObjectReader
        {
            T ReadObject<T>(string str);
        }

        readonly BinaryReader binaryReader;
        readonly IObjectReader objectReader;

        public DelegatingWrappedBinaryReader(
            BinaryReader binaryReader,
            IObjectReader objectReader)
        {
            this.binaryReader = binaryReader;
            this.objectReader = objectReader;
        }

        public void Dispose()
        {
            binaryReader.Dispose();
        }

        public bool ReadBoolean()
        {
            return binaryReader.ReadBoolean();
        }

        public double ReadDouble()
        {
            return binaryReader.ReadDouble();
        }

        public Guid ReadGuid()
        {
            return binaryReader.ReadGuid();
        }

        public int ReadInt32()
        {
            return binaryReader.ReadInt32();
        }

        public T ReadObject<T>()
        {
            var str = ReadString();
            return this.objectReader.ReadObject<T>(str);
        }

        public string ReadString()
        {
            return binaryReader.ReadString();
        }
    }
}
