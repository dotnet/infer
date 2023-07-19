using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization;
using System.Text;

namespace Microsoft.ML.Probabilistic.Serialization
{
    public class WrappedBinaryReader : IReader, IDisposable
    {
        readonly BinaryReader binaryReader;
        readonly IFormatter formatter;

        public WrappedBinaryReader(BinaryReader binaryReader, IFormatter formatter)
        {
            this.binaryReader = binaryReader;
            this.formatter = formatter;
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
            return binaryReader.ReadObject<T>(formatter);
        }

        public string ReadString()
        {
            return binaryReader.ReadString();
        }
    }
}
