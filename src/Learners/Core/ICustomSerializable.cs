// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using Serialization;

    /// <summary>
    /// Interface to any object that needs to control its own serialization.
    /// </summary>
    public interface ICustomSerializable
    {
        /// <summary>
        /// Saves the state of the object to a writer.
        /// </summary>
        /// <param name="writer">The writer to save the state of the object to.</param>
        void SaveForwardCompatible(IWriter writer);
    }
}