using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Microsoft.ML.Probabilistic.Serialization
{
    public class WrappedBinaryReader : IReader, IDisposable
    {
        BinaryReader binaryReader;

        public WrappedBinaryReader(BinaryReader binaryReader)
        {
            this.binaryReader = binaryReader;
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
            throw new NotImplementedException();
        }

        public string ReadString()
        {
            return binaryReader.ReadString();
        }
    }
}
