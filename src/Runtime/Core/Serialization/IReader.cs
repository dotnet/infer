// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Serialization
{
    using System;

    public interface IReader
    {
        bool ReadBoolean();
        int ReadInt32();
        double ReadDouble();
        string ReadString();
        Guid ReadGuid();
        T ReadObject<T>();
    }
}
