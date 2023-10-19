// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Serialization
{
    using System;
    using System.IO;

    public class WrappedBinaryWriter : IWriter, IDisposable
    {
        BinaryWriter binaryWriter;

        public WrappedBinaryWriter(BinaryWriter binaryWriter)
        {
            this.binaryWriter = binaryWriter;
        }

        public void Dispose()
        {
            binaryWriter.Dispose();
        }

        public void Write(bool value)
        {
            binaryWriter.Write(value);
        }

        public void Write(int value)
        {
            binaryWriter.Write(value);
        }

        public void Write(string value)
        {
            binaryWriter.Write(value);
        }

        public void Write(double value)
        {
            binaryWriter.Write(value);
        }

        public void Write(Guid value)
        {
            binaryWriter.Write(value);
        }

        public void WriteObject(object value)
        {
            throw new NotImplementedException();
        }
    }
}
