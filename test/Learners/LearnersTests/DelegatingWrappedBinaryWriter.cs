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

    internal class DelegatingWrappedBinaryWriter : IWriter, IDisposable
    {
        readonly BinaryWriter binaryWriter;
        readonly Func<object, string> writeObject;

        public DelegatingWrappedBinaryWriter(
            BinaryWriter binaryWriter,
            Func<object, string> writeObject)
        {
            this.binaryWriter = binaryWriter;
            this.writeObject = writeObject;
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
            var objectToWrite = writeObject(value);
            Write(objectToWrite);
        }
    }
}
