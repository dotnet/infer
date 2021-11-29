// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Serialization
{
    using System;

    public interface IWriter
    {
        void Write(bool value);
        void Write(int value);
        void Write(string value);
        void Write(double value);
        void Write(Guid value);
        void WriteObject(object value);
    }
}
