// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    /// <summary>
    /// Interface to the settings of an implementation of <see cref="ILearner"/>.
    /// These should be set once to configure the learner before calling any query methods on it.
    /// </summary>
    /// <remarks>
    /// This design is a subject to change.
    /// </remarks>
    public interface ISettings
    {
    }
}