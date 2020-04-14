// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Math;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace Microsoft.ML.Probabilistic.Tests
{

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

    using Assert = Xunit.Assert;
    using BernoulliArray = DistributionStructArray<Bernoulli, bool>;
    using BernoulliArrayArray = DistributionRefArray<DistributionStructArray<Bernoulli, bool>, bool[]>;
    using GammaArray = DistributionStructArray<Gamma, double>;
    using GaussianArray = DistributionStructArray<Gaussian, double>;
    using GaussianArrayArray = DistributionRefArray<DistributionStructArray<Gaussian, double>, double[]>;
    using DirichletArray = DistributionRefArray<Dirichlet, Vector>;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Compiler.Transforms;
    using System.IO;
    using Microsoft.ML.Probabilistic.Algorithms;
    using Microsoft.ML.Probabilistic.Models.Attributes;
    using Microsoft.ML.Probabilistic.Serialization;

    public class BayesPointMachineTests
    {
        // Currently fails because the initialization schedule isn't sequential.
        // Manually adding the sequential updates makes it work.
        /// <summary>
        /// Test the interaction of sequential scheduling with initialization.
        /// </summary>
        [Fact]
        [Trait("Category", "OpenBug")]
        public void HierarchicalPrecisionTest4()
        {
            double[][] denseFeatureVectors =
            {
            new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.376020021010442, 0.623979978989558, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0.187587626244131, 0.812412373755869, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0863830616572763, 0.913616938342724, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0.979400086720373, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0.0205999132796268, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0.565473235138128, 0.434526764861872, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.187587626244131, 0.812412373755869, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0.948500216800943, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0514997831990573, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0.528419686577811, 0.471580313422189, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0.0980391997148686, 0.901960800285131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0.376020021010442, 0.623979978989558, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.187587626244131, 0.812412373755869, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0.0980391997148686, 0.901960800285131, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.767507096020999, 0.232492903979001, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0.559319556497247, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.440680443502753, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.66531544420414, 0.33468455579586, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0.685210829577452, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0.314789170422548, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0.279021420642831, 0.720978579357169, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0.222192947339188, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.777807052660812, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0.559319556497247, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0.440680443502753, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0.685210829577452, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0.314789170422548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.187587626244131, 0.812412373755869, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0.979400086720378, 0.0205999132796215, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0.228787452803378, 0.771212547196622, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0.979400086720373, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0.0205999132796268, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0.0863830616572763, 0.913616938342724, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0.528419686577811, 0.471580313422189, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0.802112417116058, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0.197887582883942, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0.979400086720373, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0.0205999132796268, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0.565473235138128, 0.434526764861872, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0.376020021010442, 0.623979978989558, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0.202164033831902, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.797835966168098, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0.228787452803378, 0.771212547196622, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0.0863830616572763, 0.913616938342724, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
new double[] {1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0.317982759330054, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0.682017240669946, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.187587626244131, 0.812412373755869, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.187587626244131, 0.812412373755869, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0.0893539297350134, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0.910646070264987, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1},
new double[] {1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.979400086720378, 0.0205999132796215, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0},
new double[] {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0.685210829577452, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0.314789170422548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0.317982759330054, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0.682017240669946, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
new double[] {1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0.565473235138128, 0.434526764861872, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0.559319556497247, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0.440680443502753, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},
new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0.979400086720373, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0205999132796268, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
new double[] {1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0.317982759330054, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0.682017240669946, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0.559319556497247, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0.440680443502753, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0},
new double[] {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},

        };

            bool[] observations = new bool[]{
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        };

            int numParams = 70;
            IList<SparseVector[]> featureVectors = new List<SparseVector[]>();
            IList<bool[]> labels = new List<bool[]>();
            IList<int[]> indexMapping = new List<int[]>();

            featureVectors.Add(denseFeatureVectors.Select(x => SparseVector.FromArray(x)).ToArray());
            labels.Add(observations);
            int[] mapping = new int[numParams];

            for (int i = 0; i < numParams; i++)
            {
                mapping[i] = i;
            }

            indexMapping.Add(mapping);

            Dictionary<int, Gaussian> specialCase = new Dictionary<int, Gaussian>()
      {
          {21, Gaussian.FromMeanAndPrecision(0, 0.120158107244073)},
          {22, Gaussian.FromMeanAndPrecision(0, 0.174441509106506)},
          {37, Gaussian.FromMeanAndPrecision(0, 0.103619073096587)},
          {38, Gaussian.FromMeanAndPrecision(0, 0.301898999873288)},
          {49, Gaussian.FromMeanAndPrecision(0, 0.0155494933847474)}
      };

            Gaussian[] meanPriors = new Gaussian[numParams];

            for (int i = 0; i < numParams; i++)
            {
                if (specialCase.ContainsKey(i))
                {
                    meanPriors[i] = specialCase[i];
                }
                else
                {
                    meanPriors[i] = Gaussian.FromMeanAndPrecision(0.0, 0.5);
                }
            }

            Gamma[] precisionPriors = new Gamma[numParams];

            for (int i = 0; i < numParams; i++)
            {
                precisionPriors[i] = Gamma.FromShapeAndRate(5, 5);
            }

            HierarchicalBPMPrecisionLearning2(numParams, featureVectors, labels, indexMapping, meanPriors, precisionPriors);
        }

        private void HierarchicalBPMPrecisionLearning2(
            int nFeatures,
            IList<SparseVector[]> featureVectors,
            IList<bool[]> labels,
            IList<int[]> indexMapping,
            Gaussian[] meanPriors,
            Gamma[] precisionPriors)
        {
            IList<IList<int>> idx = new List<IList<int>>();
            IList<IList<double>> value = new List<IList<double>>();

            for (int i = 0; i < featureVectors.First().Length; i++)
            {
                idx.Add(new List<int>());
                value.Add(new List<double>());

                for (int k = 0; k < featureVectors.First()[i].Count; k++)
                {
                    if (featureVectors.First()[i][k] != 0)
                    {
                        idx[i].Add(k);
                        value[i].Add(featureVectors.First()[i][k]);
                    }
                }
            }

            // ------------------
            // Model
            // ------------------

            // Range sizes
            var numberOfMessages = Variable.New<int>().Named("NumberOfMessages");
            var numberOfFeatures = Variable.New<int>().Named("NumberOfFeatures");
            var numberOfFeatureGroups = Variable.New<int>().Named("NumberOfFeatureGroups");

            // Ranges
            Range messageRange = new Range(numberOfMessages).Named("MessageRange");
            Range featureRange = new Range(numberOfFeatures).Named("FeatureRange");
            Range featureGroupRange = new Range(numberOfFeatureGroups).Named("FeatureGroupRange");

            // Mapping of personal feature index to corresponding raw feature index
            var groupOf = Variable.Array<int>(featureRange).Named("GroupOf");
            groupOf.SetValueRange(featureGroupRange);

            // Request a sequential schedule
            messageRange.AddAttribute(new Sequential());

            var weightMeanPrior = Variable.New<GaussianArray>().Named("WeightMeanPrior");
            var weightMean = Variable.Array<double>(featureGroupRange).Named("WeightMean");
            weightMean.SetTo(Variable<double[]>.Random(weightMeanPrior));
            var weightPrecisionPrior = Variable.New<GammaArray>().Named("weightPrecisionPrior");
            var weightPrecision = Variable.Array<double>(featureGroupRange).Named("WeightPrecision");
            weightPrecision.SetTo(Variable<double[]>.Random(weightPrecisionPrior));
            var weights = Variable.Array<double>(featureRange).Named("Weight");
            weights[featureRange] = Variable.GaussianFromMeanAndPrecision(weightMean[groupOf[featureRange]], weightPrecision[groupOf[featureRange]]);
            var weightInitialiser = Variable.New<GaussianArray>().Named("WeightInitialiser");
            weights.InitialiseTo(weightInitialiser);
            var collapsedMessages = Variable.Array<Gaussian>(featureRange).Named("CollapsedMessages");
            Variable.ConstrainEqualRandom(weights[featureRange], collapsedMessages[featureRange]);

            // Number of features active per item
            var featureCounts = Variable.Array<int>(messageRange).Named("FeatureCounts");
            Range featureItemRange = new Range(featureCounts[messageRange]).Named("featureItemRange");

            // The observed features
            var featureIndices = Variable.Array(Variable.Array<int>(featureItemRange), messageRange).Named("FeatureIndices"); // observed data
            var featureValues = Variable.Array(Variable.Array<double>(featureItemRange), messageRange).Named("FeatureValues"); // observed data
                                                                                                                               // The label
            var isActionPerformed = Variable.Array<bool>(messageRange).Named("IsActionPerformed");
            var noise = Variable.New<double>().Named("noise");

            // Loop over emails in a batch
            using (Variable.ForEach(messageRange))
            {
                VariableArray<double> sparseWeights = Variable.Subarray(weights, featureIndices[messageRange]);
                VariableArray<double> sparseProduct = Variable.Array<double>(featureItemRange).Named("SparseProduct");
                sparseProduct[featureItemRange] = featureValues[messageRange][featureItemRange] * sparseWeights[featureItemRange];
                Variable<double> dotProduct = Variable.Sum(sparseProduct).Named("DotProduct");
                isActionPerformed[messageRange] = Variable.GaussianFromMeanAndPrecision(dotProduct, noise).Named("DotProductWithNoise") > 0;
            }

            // ----------------------
            // Observations
            // ----------------------
            numberOfMessages.ObservedValue = featureVectors.First().Length;
            numberOfFeatures.ObservedValue = nFeatures;
            featureCounts.ObservedValue = featureVectors.First().Select(x => x.SparseCount).ToArray();
            featureIndices.ObservedValue = idx.Select(x => x.ToArray()).ToArray();
            featureValues.ObservedValue = value.Select(x => x.ToArray()).ToArray();
            numberOfFeatureGroups.ObservedValue = nFeatures;
            groupOf.ObservedValue = indexMapping.First();
            isActionPerformed.ObservedValue = labels.First();
            noise.ObservedValue = 0.1;
            weightMeanPrior.ObservedValue = new GaussianArray(meanPriors);

            weightPrecisionPrior.ObservedValue = new GammaArray(precisionPriors);

            //weightInitialiser.ObservedValue = new GaussianArray(Util.ArrayInit(nFeatures, f => Gaussian.FromMeanAndVariance(0, 2.333)));
            weightInitialiser.ObservedValue = new GaussianArray(nFeatures, valueIdx =>
            {
                double mean = meanPriors[valueIdx].GetMean();
                double variance = meanPriors[valueIdx].GetVariance() + (1.0 / precisionPriors[valueIdx].GetMean());
                return Gaussian.FromMeanAndVariance(mean, variance);
            });

            collapsedMessages.ObservedValue = Util.ArrayInit(nFeatures, f => Gaussian.Uniform());

            // --------------------
            // Inference
            // --------------------
            InferenceEngine engine = new InferenceEngine();
            //engine.Compiler.UseSerialSchedules = true;
            //engine.Compiler.UseExistingSourceFiles = true;
            //engine.Compiler.AllowSerialInitialisers = false;
            //engine.Compiler.FreeMemory = true;
            var weightMeanPost = engine.Infer(weightMean);
            var weightPrecisionPost = engine.Infer(weightPrecision);
            Console.WriteLine(weightMeanPost);
        }

        // This model gives a bad schedule under .NET 4.5
        // Weights_2_rep2_B_toDef is not uniform (due to InitSchedule) but Weights_2_uses_F_marginal is uniform
        // because Weights_2_uses_B_toDef is not updated before marginal
        // The schedule constraints allow this since Weights_2_uses_B_toDef is not considered stale until it is scheduled.
        internal void MulticlassTest()
        {
            int nClasses = 3;
            double noisePrecision = 0.1;
            int nFeatures = 2;

            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var nDenseItems = new Variable<int>[nClasses];
            var denseItems = new Range[nClasses];

            var denseFeatureVectors = new VariableArray<Vector>[nClasses];

            var denseWeights = new Variable<Vector>[nClasses];
            var denseWeightsPriors = new Variable<VectorGaussian>[nClasses];

            var denseScores = new VariableArray<double>[nClasses, nClasses];
            var denseNoisyScores = new VariableArray<double>[nClasses, nClasses];
            var denseNoisyScoreDifferences = new VariableArray<double>[nClasses, nClasses];

            for (int c = 0; c < nClasses; c++)
            {
                denseWeightsPriors[c] = Variable.New<VectorGaussian>().Named("WeightsPrior_" + (c + 1));
                denseWeightsPriors[c].ObservedValue = VectorGaussian.Uniform(nFeatures);
                denseWeights[c] = Variable<Vector>.Random(denseWeightsPriors[c]).Named("Weights_" + (c + 1));

                nDenseItems[c] = Variable.New<int>().Named("N_" + (c + 1));
                nDenseItems[c].ObservedValue = 0;
                denseItems[c] = new Range(nDenseItems[c]).Named("n_" + (c + 1));
                denseItems[c].AddAttribute(new Sequential());

                denseFeatureVectors[c] = Variable.Array<Vector>(denseItems[c]).Named("Features_" + (c + 1));
                denseFeatureVectors[c].ObservedValue = new Vector[0];

                for (int k = 0; k < nClasses; k++)
                {
                    denseScores[c, k] = Variable.Array<double>(denseItems[c]).Named("Scores_" + (c + 1) + "," + (k + 1));
                    denseNoisyScores[c, k] = Variable.Array<double>(denseItems[c]).Named("NoisyScores_" + (c + 1) + "," + (k + 1));
                    denseNoisyScoreDifferences[c, k] = Variable.Array<double>(denseItems[c]).Named("NoisyScoreDifferences_" + (c + 1) + "," + (k + 1));
                }
            }
            for (int c = 0; c < nClasses; c++)
            {
                using (Variable.ForEach(denseItems[c]))
                {
                    // ...compute noisy scores of all classes...
                    for (int k = 0; k < nClasses; k++)
                    {
                        denseScores[c, k][denseItems[c]] = Variable.InnerProduct(denseWeights[k], denseFeatureVectors[c][denseItems[c]]);
                        denseNoisyScores[c, k][denseItems[c]] = Variable.GaussianFromMeanAndPrecision(denseScores[c, k][denseItems[c]], noisePrecision);
                    }

                    // ...and contrain the noisy score of the true class to be greater than the noisy scores of all other classes
                    for (int k = 0; k < nClasses; k++)
                    {
                        if (k != c)
                        {
                            denseNoisyScoreDifferences[c, k][denseItems[c]] = denseNoisyScores[c, c][denseItems[c]] - denseNoisyScores[c, k][denseItems[c]];
                            Variable.ConstrainPositive(denseNoisyScoreDifferences[c, k][denseItems[c]]);
                        }
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(denseWeights[0]));
        }

        [Fact]
        public void HierarchicalBPM()
        {
            int NumberOfMessages = 1;
            int NumberOfPersonalFeatures = 1;
            int NumberOfSharedFeatures = 1;
            // Ranges
            Range N = new Range(NumberOfMessages).Named("N");
            Range PF = new Range(NumberOfPersonalFeatures).Named("PF");
            Range SF = new Range(NumberOfSharedFeatures).Named("SF");

            var WPersonalFeaturesDist = Variable.New<GaussianArray>().Named("WPersonalFeaturePrior");
            VariableArray<double> WPersonalFeatures = Variable.Array<double>(PF).Named("WPersonalFeatures");
            WPersonalFeatures.SetTo(Variable<double[]>.Random(WPersonalFeaturesDist));

            var WFeaturesSharedMeanPrior = Variable.New<GaussianArray>().Named("WFeaturesSharedMeanPrior");
            var WFeaturesSharedMean = Variable.Array<double>(SF).Named("WFeaturesSharedMean");
            WFeaturesSharedMean.SetTo(Variable<double[]>.Random(WFeaturesSharedMeanPrior));
            var WFeaturesSharedPrecision = Variable.Array<double>(SF).Named("WFeaturesSharedPrecision");
            var WSharedFeatures = Variable.Array<double>(SF).Named("WSharedFeatures"); // parameter array
            WSharedFeatures[SF].SetTo(Variable.GaussianFromMeanAndPrecision(WFeaturesSharedMean[SF], WFeaturesSharedPrecision[SF]));

            var WSharedFeaturesConstraintDist = Variable.Array<Gaussian>(SF).Named("WFeaturesContraintDist");
            Variable.ConstrainEqualRandom(WSharedFeatures[SF], WSharedFeaturesConstraintDist[SF]);

            // Number of features active per item
            var SharedFeatureCounts = Variable.Array<int>(N).Named("SharedFeatureCounts");
            Range SFitem = new Range(SharedFeatureCounts[N]).Named("SFItem");

            // The observed features
            var SharedFeatureIndices = Variable.Array(Variable.Array<int>(SFitem), N).Named("SharedFeatureIndices"); // observed data
            var SharedFeatureValues = Variable.Array(Variable.Array<double>(SFitem), N).Named("SharedFeatureValues"); // observed data

            // Number of features active per item
            var PersonalFeatureCounts = Variable.Array<int>(N).Named("PersonalFeatureCounts");
            Range PFitem = new Range(PersonalFeatureCounts[N]).Named("PFItem");

            // The observed features
            var PersonalFeatureIndices = Variable.Array(Variable.Array<int>(PFitem), N).Named("PersonalFeatureIndices"); // observed data
            var PersonalFeatureValues = Variable.Array(Variable.Array<double>(PFitem), N).Named("PersonalFeatureValues"); // observed data

            // Target: is the e-mail replied to?
            var IsRepliedTo = Variable.Array<bool>(N).Named("IsRepliedTo");

            // Loop over emails in a batch
            using (Variable.ForEach(N))
            {
                var SWsparse = Variable.Subarray(WSharedFeatures, SharedFeatureIndices[N]).Named("SWsparse");
                var Sproduct = Variable.Array<double>(SFitem).Named("Sproduct");
                Sproduct[SFitem] = SharedFeatureValues[N][SFitem] * SWsparse[SFitem];

                var PWSparse = Variable.Subarray(WPersonalFeatures, PersonalFeatureIndices[N]).Named("PWSparse");
                var Pproduct = Variable.Array<double>(PFitem).Named("Pproduct");
                Pproduct[PFitem] = PersonalFeatureValues[N][PFitem] * PWSparse[PFitem];

                var W = Variable.Sum(Sproduct) + Variable.Sum(Pproduct);
                W.Name = "W";

                IsRepliedTo[N] = Variable.GaussianFromMeanAndPrecision(W, 0.1) > 0;
            }
            N.AddAttribute(new Sequential());

            WPersonalFeaturesDist.ObservedValue = new GaussianArray(new Gaussian[] { new Gaussian(0, 1) });
            WSharedFeaturesConstraintDist.ObservedValue = new Gaussian[] { new Gaussian(0, 1) };
            WFeaturesSharedMeanPrior.ObservedValue = new GaussianArray(new Gaussian[] { new Gaussian(0, 1) });
            WFeaturesSharedPrecision.ObservedValue = new double[] { 1 };
            PersonalFeatureCounts.ObservedValue = new int[] { 1 };
            PersonalFeatureIndices.ObservedValue = new int[][] { new int[] { 0 } };
            PersonalFeatureValues.ObservedValue = new double[][] { new double[] { 1 } };
            SharedFeatureCounts.ObservedValue = new int[] { 1 };
            SharedFeatureIndices.ObservedValue = new int[][] { new int[] { 0 } };
            SharedFeatureValues.ObservedValue = new double[][] { new double[] { 1 } };
            IsRepliedTo.ObservedValue = new bool[] { true };

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine(engine.Infer(WPersonalFeatures));
        }

        /// <summary>
        /// Fails with Opt=F because schedule is not sequential
        /// because SparseWeights_use = DerivedVariable(...) is defined at depth 0 instead of depth 1,
        /// because SparseWeights was converted to use JaggedSubarray.
        /// VariableTransform needs to work differently to support Sequential here.
        /// </summary>
        [Trait("Category", "OpenBug")]
        [Fact]
        public void HierarchicalBPM4Test()
        {
            HierarchicalBPM2(true);
        }

        [Fact]
        public void HierarchicalBPM2Test()
        {
            HierarchicalBPM2(false);
        }
        private void HierarchicalBPM2(bool useJaggedSubarray)
        {
            int numberOfMessages = 3;
            int numberOfFeatures = 3;
            int numberOfFeatureGroups = 1;
            Range messageRange = new Range(numberOfMessages).Named("MessageRange");
            Range featureRange = new Range(numberOfFeatures).Named("FeatureRange");
            Range featureGroupRange = new Range(numberOfFeatureGroups).Named("FeatureGroupRange");

            // Mapping of personal feature index to corresponding raw feature index
            var groupOf = Variable.Array<int>(featureRange).Named("GroupOf");
            groupOf.SetValueRange(featureGroupRange);

            // Request a sequential schedule
            messageRange.AddAttribute(new Sequential());

            var weightMeanPriors = Variable.New<GaussianArray>().Named("WeightMeanPrior");
            var weightMeans = Variable.Array<double>(featureGroupRange).Named("WeightMean");
            weightMeans.SetTo(Variable<double[]>.Random(weightMeanPriors));
            var weightPrecisions = Variable.Array<double>(featureGroupRange).Named("WeightPrecision");
            var weights = Variable.Array<double>(featureRange).Named("Weight");
            weights[featureRange].SetTo(Variable.GaussianFromMeanAndPrecision(weightMeans[groupOf[featureRange]], weightPrecisions[groupOf[featureRange]]));
            if (false)
            {
                // Collapsed meesages from all other batches for this person
                var collapsedMessages = Variable.Array<Gaussian>(featureRange).Named("CollapsedMessages");
                Variable.ConstrainEqualRandom(weights[featureRange], collapsedMessages[featureRange]);
            }

            // Number of features active per item
            var featureCounts = Variable.Array<int>(messageRange);
            Range sFitem = new Range(featureCounts[messageRange]).Named("SFItem");

            // The observed features
            var featureIndices = Variable.Array(Variable.Array<int>(sFitem), messageRange).Named("SharedFeatureIndices"); // observed data
            var featureValues = Variable.Array(Variable.Array<double>(sFitem), messageRange).Named("SharedFeatureValues"); // observed data
                                                                                                                           // The label
            var isActionPerformed = Variable.Array<bool>(messageRange).Named("IsActionPerformed");

            // Use jagged sub-array feature for efficiency
            VariableArray<VariableArray<double>, double[][]> sparseWeightsArray = null;
            if (useJaggedSubarray)
                sparseWeightsArray = Variable.JaggedSubarray(weights, featureIndices).Named("SparseWeights");

            // Loop over emails in a batch
            using (Variable.ForEach(messageRange))
            {
                VariableArray<double> sparseWeights;
                if (useJaggedSubarray)
                    sparseWeights = sparseWeightsArray[messageRange];
                else
                    sparseWeights = Variable.Subarray(weights, featureIndices[messageRange]).Named("SparseWeights");

                VariableArray<double> sparseProduct = Variable.Array<double>(sFitem).Named("SparseProduct");
                sparseProduct[sFitem] = featureValues[messageRange][sFitem] * sparseWeights[sFitem];
                Variable<double> dotp = Variable.Sum(sparseProduct).Named("DotProduct");
                double NoisePrecision = 1;
                isActionPerformed[messageRange] = Variable.GaussianFromMeanAndPrecision(dotp, NoisePrecision) > 0;
            }

            weightPrecisions.ObservedValue = new double[] { 1 };
            weightMeanPriors.ObservedValue = new GaussianArray(1, i => new Gaussian(0, 100));
            groupOf.ObservedValue = Util.ArrayInit(numberOfFeatures, f => 0);
            featureCounts.ObservedValue = Util.ArrayInit(numberOfMessages, i => numberOfFeatures);
            featureIndices.ObservedValue = Util.ArrayInit(numberOfMessages, i => Util.ArrayInit(numberOfFeatures, f => f));
            featureValues.ObservedValue = Util.ArrayInit(numberOfMessages, i => Util.ArrayInit(numberOfFeatures, f => 1.0));
            isActionPerformed.ObservedValue = Util.ArrayInit(numberOfMessages, i => (i > 0));

            InferenceEngine engine = new InferenceEngine();
            Console.WriteLine("weights = {0}", engine.Infer(weights));
            Console.WriteLine("weightMeans = {0}", engine.Infer(weightMeans));
            Gaussian weightMeansActual = engine.Infer<IList<Gaussian>>(weightMeans)[0];
            Gaussian weightMeansExpected = new Gaussian(0.1868, 0.7815);
            Assert.True(weightMeansExpected.MaxDiff(weightMeansActual) < 1e-3);
            weightMeanPriors.ObservedValue = new GaussianArray(1, i => new Gaussian(0, 1));
            Console.WriteLine("weights = {0}", engine.Infer(weights));
            Console.WriteLine("weightMeans = {0}", engine.Infer(weightMeans));
            weightMeanPriors.ObservedValue = new GaussianArray(1, i => new Gaussian(0.2, 1));
            Console.WriteLine("weights = {0}", engine.Infer(weights));
            Console.WriteLine("weightMeans = {0}", engine.Infer(weightMeans));
        }

        internal void HierarchicalBPM3()
        {
            var numberOfPersons = Variable.New<int>().Named("NumberOfPersons");
            var numberOfFeatureGroups = Variable.New<int>().Named("NumberOfFeatureGroups");
            // Ranges
            Range personRange = new Range(numberOfPersons).Named("PersonRange");
            var numberOfMessages = Variable.Array<int>(personRange).Named("NumberOfMessages");
            Range messageRange = new Range(numberOfMessages[personRange]).Named("MessageRange");
            var numberOfFeatures = Variable.Array<int>(personRange).Named("NumberOfFeatures");

            Range featureRange = new Range(numberOfFeatures[personRange]).Named("FeatureRange");
            Range featureGroupRange = new Range(numberOfFeatureGroups).Named("FeatureGroupRange");
            // Mapping of personal feature index to corresponding raw feature index
            var groupOf = Variable.Array(Variable.Array<int>(featureRange), personRange).Named("GroupOf");
            groupOf.SetValueRange(featureGroupRange);

            // Request a sequential schedule
            //personRange.AddAttribute(new Sequential());
            messageRange.AddAttribute(new Sequential());

            var weightMeanPriors = Variable.New<GaussianArray>().Named("WeightMeanPrior");
            var weightMeans = Variable.Array<double>(featureGroupRange).Named("WeightMean");
            weightMeans.SetTo(Variable<double[]>.Random(weightMeanPriors));
            var weightPrecisions = Variable.Array<double>(featureGroupRange).Named("WeightPrecision");
#if removegrouping

                        var weights = Variable.Array(Variable.Array<double>(featureGroupRange), personRange).Named("Weight");
                        var weights[personRange][featureGroupRange] = Variable.GaussianFromMeanAndPrecision(weightMeans[featureGroupRange], var weightPrecisions[featureGroupRange]).ForEach(personRange);
#else
            var weights = Variable.Array(Variable.Array<double>(featureRange), personRange).Named("Weight");
            weights[personRange][featureRange] = Variable.GaussianFromMeanAndPrecision(weightMeans[groupOf[personRange][featureRange]],
                                                                                       weightPrecisions[groupOf[personRange][featureRange]]);
#endif
            bool useSparse = true;
            bool useRegression = true;

            // Number of features active per item
            var featureCounts = Variable.Array(Variable.Array<int>(messageRange), personRange).Named("FeatureCounts");
            Range sFitem = new Range(featureCounts[personRange][messageRange]).Named("SFItem");
            var featureCountsZero = Variable.Array(Variable.Array<int>(messageRange), personRange).Named("FeatureCountsZero");
            Range sFitemZero = new Range(featureCountsZero[personRange][messageRange]).Named("SFItemZero");

            // The observed features
            var featureIndices = Variable.Array(Variable.Array(Variable.Array<int>(sFitem), messageRange), personRange).Named("FeatureIndices"); // observed data
            var featureIndicesZero = Variable.Array(Variable.Array(Variable.Array<int>(sFitemZero), messageRange), personRange).Named("FeatureIndicesZero"); // observed data
            var featureValues = Variable.Array(Variable.Array(Variable.Array<double>(useSparse ? sFitem : featureRange), messageRange), personRange).Named("FeatureValues"); // observed data
                                                                                                                                                                             // The label
            var isActionPerformed = Variable.Array(Variable.Array<bool>(messageRange), personRange).Named("IsActionPerformed");

            var bias = Variable.Array<double>(personRange).Named("bias");
            bias[personRange] = Variable.GaussianFromMeanAndVariance(0, 100).ForEach(personRange);
            var shift = Variable.Array(Variable.Array<double>(featureRange), personRange).Named("shift");
            shift[personRange][featureRange] = Variable.GaussianFromMeanAndVariance(0, 10).ForEach(personRange, featureRange);

            using (Variable.ForEach(personRange))
            {
                if (useSparse)
                {
                    // Use jagged sub-array feature for efficiency
                    VariableArray<VariableArray<double>, double[][]> sparseWeights =
                              Variable.JaggedSubarray(weights[personRange], featureIndices[personRange]).Named("SparseWeights");
                    VariableArray<VariableArray<double>, double[][]> sparseWeightsZero =
                          Variable.JaggedSubarray(weights[personRange], featureIndicesZero[personRange]).Named("SparseWeightsZero");
                    VariableArray<VariableArray<double>, double[][]> sparseFeatureMeans =
                          Variable.JaggedSubarray(shift[personRange], featureIndices[personRange]).Named("sparseFeatureMeans");
                    VariableArray<VariableArray<double>, double[][]> sparseFeatureMeansZero =
                          Variable.JaggedSubarray(shift[personRange], featureIndicesZero[personRange]).Named("sparseFeatureMeansZero");
                    // Loop over emails in a batch
                    using (Variable.ForEach(messageRange))
                    {
                        VariableArray<double> sparseProduct = Variable.Array<double>(sFitem).Named("SparseProduct");
                        sparseProduct[sFitem] = (featureValues[personRange][messageRange][sFitem] - sparseFeatureMeans[messageRange][sFitem]) *
                                                sparseWeights[messageRange][sFitem];
                        Variable<double> dotp = Variable.Sum(sparseProduct).Named("DotProduct");
                        VariableArray<double> sparseProductZero = Variable.Array<double>(sFitemZero).Named("SparseProductZero");
                        sparseProductZero[sFitemZero] = sparseFeatureMeansZero[messageRange][sFitemZero] * sparseWeightsZero[messageRange][sFitemZero];
                        Variable<double> dotpZero = Variable.Sum(sparseProductZero).Named("DotProductZero");
                        var noisysum = Variable.GaussianFromMeanAndPrecision(dotp - dotpZero + bias[personRange], 1);
                        if (useRegression)
                        {
                            using (Variable.If(isActionPerformed[personRange][messageRange]))
                            {
                                Variable.ConstrainEqual(noisysum, 1.0);
                            }
                            using (Variable.IfNot(isActionPerformed[personRange][messageRange]))
                            {
                                Variable.ConstrainEqual(noisysum, -1.0);
                            }
                        }
                        else
                        {
                            isActionPerformed[personRange][messageRange] = noisysum > 0;
                        }
                    }
                }
                else
                {
                    // Loop over emails in a batch
                    using (Variable.ForEach(messageRange))
                    {
                        VariableArray<double> product = Variable.Array<double>(featureRange).Named("product");
                        product[featureRange] = (featureValues[personRange][messageRange][featureRange] - shift[personRange][featureRange]).Named("shiftedFeature") *
                                                weights[personRange][featureRange];
                        Variable<double> sum = Variable.Sum(product).Named("sum");
                        var noisysum = Variable.GaussianFromMeanAndPrecision((sum + bias[personRange]).Named("sumPlusBias"), 1).Named("noisySum");
                        if (useRegression)
                        {
                            using (Variable.If(isActionPerformed[personRange][messageRange]))
                            {
                                Variable.ConstrainEqual(noisysum, 1.0);
                            }
                            using (Variable.IfNot(isActionPerformed[personRange][messageRange]))
                            {
                                Variable.ConstrainEqual(noisysum, -1.0);
                            }
                        }
                        else
                        {
                            isActionPerformed[personRange][messageRange] = noisysum > 0;
                        }
                    }
                }
            }

            Rand.Restart(12347);
            int n = 1000;
            int nf = 100;
            Vector[] x = Util.ArrayInit(n, i => Vector.FromArray(Util.ArrayInit(nf, f => Rand.Double())));
            if (true)
            {
                // empirical moments of features
                Gaussian[] featureDist2 = new Gaussian[nf];
                for (int f = 0; f < nf; f++)
                {
                    GaussianEstimator est = new GaussianEstimator();
                    for (int i = 0; i < n; i++)
                    {
                        est.Add(x[i][f]);
                    }
                    featureDist2[f] = est.GetDistribution(new Gaussian());
                    if (f < 3)
                        Console.Write(featureDist2[f] + " ");
                }
                Console.WriteLine();
                // construct a whitening matrix
                Matrix whiten = new Matrix(nf, nf);
                for (int i = 0; i < nf; i++)
                {
                    whiten[i, 0] = featureDist2[i].GetMean();
                }
                for (int j = 1; j < nf; j++)
                {
                    double sumsq = 0;
                    for (int i = 0; i < j; i++)
                    {
                        whiten[i, j] = whiten[i, 0];
                        sumsq += whiten[i, j] * whiten[i, j];
                    }
                    whiten[j, j] = -sumsq / whiten[j, 0];
                }
                // whiten the data
                Matrix whitenTrans = whiten.Transpose();
                if (true)
                {
                    // check that all inner products are zero
                    Matrix temp = whitenTrans * whiten;
                    double max = double.NegativeInfinity;
                    for (int i = 0; i < nf; i++)
                    {
                        for (int j = 0; j < nf; j++)
                        {
                            if (i == j)
                                continue;
                            max = System.Math.Max(max, System.Math.Abs(temp[i, j]));
                        }
                    }
                    Console.WriteLine("max inner product = {0}", max);
                }
                var basis = new Basis(Vector.FromArray(Util.ArrayInit(nf, f => featureDist2[f].GetMean())));
                double maxDiff = double.NegativeInfinity;
                for (int i = 0; i < n; i++)
                {
                    Vector y = basis.Project(x[i]);
                    x[i] = whitenTrans * x[i];
                    maxDiff = System.Math.Max(maxDiff, y.MaxDiff(x[i]));
                }
                Console.WriteLine("maxDiff = {0}", maxDiff);
            }

            numberOfFeatureGroups.ObservedValue = nf;
            weightMeanPriors.ObservedValue = new GaussianArray(numberOfFeatureGroups.ObservedValue, i => Gaussian.FromMeanAndPrecision(0.0, 0.01));
            weightMeans.ObservedValue = new double[numberOfFeatureGroups.ObservedValue];
            weightPrecisions.ObservedValue = Util.ArrayInit(numberOfFeatureGroups.ObservedValue, i => 0.1);
            numberOfPersons.ObservedValue = 1;
            numberOfMessages.ObservedValue = new int[] { n };
            numberOfFeatures.ObservedValue = new int[] { nf };
            featureCounts.ObservedValue = Util.ArrayInit(numberOfPersons.ObservedValue,
                                                         person => Util.ArrayInit(numberOfMessages.ObservedValue[person], m => numberOfFeatures.ObservedValue[person]));
            featureCountsZero.ObservedValue = Util.ArrayInit(numberOfPersons.ObservedValue,
                                                             person => Util.ArrayInit(numberOfMessages.ObservedValue[person], m => nf - numberOfFeatures.ObservedValue[person]));
            featureIndices.ObservedValue = Util.ArrayInit(numberOfPersons.ObservedValue,
                                                          person =>
                                                          Util.ArrayInit(numberOfMessages.ObservedValue[person],
                                                                         m => Util.ArrayInit(featureCounts.ObservedValue[person][m], f => f)));
            featureIndicesZero.ObservedValue = Util.ArrayInit(numberOfPersons.ObservedValue,
                                                              person =>
                                                              Util.ArrayInit(numberOfMessages.ObservedValue[person],
                                                                             m => Util.ArrayInit(featureCountsZero.ObservedValue[person][m], f => f)));
            featureValues.ObservedValue = Util.ArrayInit(numberOfPersons.ObservedValue,
                                                         person =>
                                                         Util.ArrayInit(numberOfMessages.ObservedValue[person],
                                                                        m => Util.ArrayInit(featureCounts.ObservedValue[person][m], f => x[m][f])));
            double trueBias = 0;
            double[] trueWeights = Util.ArrayInit(numberOfFeatures.ObservedValue[0], f => 1.0 * f / (numberOfFeatures.ObservedValue[0] - 1) - 0.5);
            Vector trueVector = Vector.FromArray(trueWeights);
            GaussianEstimator innerEst = new GaussianEstimator();
            isActionPerformed.ObservedValue = Util.ArrayInit(numberOfPersons.ObservedValue, person => Util.ArrayInit(numberOfMessages.ObservedValue[person], m =>
            {
                double inner = Vector.FromArray(featureValues.ObservedValue[person][m]).Inner(trueVector) + trueBias;
                innerEst.Add(inner);
                return inner + Rand.Normal() > 0.0;
            }));
            Gaussian innerDist = innerEst.GetDistribution(new Gaussian());
            groupOf.ObservedValue = Util.ArrayInit(numberOfPersons.ObservedValue, person => Util.ArrayInit(numberOfFeatures.ObservedValue[person], f => f));

            // empirical moments of features
            Gaussian[] featureDist = new Gaussian[nf];
            for (int f = 0; f < nf; f++)
            {
                GaussianEstimator est = new GaussianEstimator();
                for (int i = 0; i < n; i++)
                {
                    est.Add(featureValues.ObservedValue[0][i][f]);
                }
                featureDist[f] = est.GetDistribution(new Gaussian());
                if (f < 3)
                    Console.Write(featureDist[f] + " ");
            }
            Console.WriteLine();
            if (false)
            {
                // wTrue*x + bTrue = wTrue*(x - shift) + bTrue + wTrue*shift
                shift.ObservedValue = Util.ArrayInit(numberOfPersons.ObservedValue, person => Util.ArrayInit(nf, f => featureDist[f].GetMean()));
                double shiftedBias = trueBias;
                for (int f = 0; f < nf; f++)
                {
                    shiftedBias += trueWeights[f] * shift.ObservedValue[0][f];
                }
                bias.ObservedValue = Util.ArrayInit(numberOfPersons.ObservedValue, person => shiftedBias);
            }
            else
            {
                shift.ObservedValue = Util.ArrayInit(numberOfPersons.ObservedValue, person => Util.ArrayInit(nf, f => 0.0));
                bias.ObservedValue = Util.ArrayInit(numberOfPersons.ObservedValue, person => 0.0);
            }
            if (false)
            {
                // change the first feature to be sum_i x[i]*m[i]
                for (int i = 0; i < n; i++)
                {
                    double sum = 0;
                    for (int f = 0; f < nf; f++)
                    {
                        sum += featureValues.ObservedValue[0][i][f] * featureDist[f].GetMean();
                    }
                    featureValues.ObservedValue[0][i][0] = sum;
                }
            }

            Console.WriteLine("empirical inner = {0}", innerDist);
            Gaussian innerExpected;
            double ms = 0, vs = 0;
            for (int f = 0; f < nf; f++)
            {
                Gaussian productDist = GaussianProductOp.ProductAverageConditional(trueVector[f], featureDist[f]);
                ms += productDist.GetMean();
                vs += productDist.GetVariance();
            }
            innerExpected = Gaussian.FromMeanAndVariance(ms, vs);
            Console.WriteLine("expected inner = {0}", innerExpected);

            InferenceEngine engine = new InferenceEngine();
            //engine.Algorithm = new VariationalMessagePassing();
            engine.Compiler.GivePriorityTo(typeof(GaussianProductOp_SHG09));
            GaussianProductOp.ForceProper = true;
            for (int iter = 1; iter < 50; iter++)
            {
                engine.NumberOfIterations = iter;
                var weightsPost = engine.Infer<IList<GaussianArray>>(weights);
                Console.WriteLine("weights[0][0] = {0} {1}", weightsPost[0][0], weightsPost[0][1]);
                var biasPost = engine.Infer<IList<Gaussian>>(bias);
                Console.WriteLine("bias[0] = {0}", biasPost[0]);
                //Console.WriteLine("weightMeans = {0}", engine.Infer(weightMeans));
                var shiftPost = engine.Infer<IList<GaussianArray>>(shift);
                Console.WriteLine("shift[0][0] = {0}", shiftPost[0][0]);
            }
            var weightMeansPost = engine.Infer<IList<Gaussian>>(weightMeans);
            Console.WriteLine("weightMeans[0] = {0} {1}", weightMeansPost[0], weightMeansPost[1]);
        }

        [Fact]
        public void HierarchicalBPMPrecisionLearningTest()
        {
            // ----------------
            // Data
            // ----------------
            int nPersons = 2;
            int nData = 20;
            int[] nFeatures = new int[] { 3, 6 };
            SparseVector[][] featureVectors = new SparseVector[nPersons][];
            bool[][] actionPerformed = new bool[nPersons][];
            for (int p = 0; p < nPersons; p++)
            {
                featureVectors[p] = new SparseVector[nData];
                actionPerformed[p] = new bool[nData];
                for (int n = 0; n < nData; n++)
                {
                    featureVectors[p][n] = SparseVector.Zero(nFeatures[p]);
                    featureVectors[p][n][0] = 1;
                    featureVectors[p][n][1] = 1;
                    actionPerformed[p][n] = false;
                }
            }

            int[][] valueIndexToParamIndex = new int[][]
                  {
                new int[] { 0, 1, 2},
                new int[] { 0, 1, 2, 2, 2, 2}
                  };

            featureVectors[0][3][2] = 1;
            featureVectors[0][9][2] = 1;
            featureVectors[0][10][2] = 1;
            featureVectors[0][12][2] = 1;
            featureVectors[0][13][2] = 1;
            featureVectors[0][16][2] = 1;

            featureVectors[1][0][2] = 1.0;
            featureVectors[1][5][2] = 1.0;
            featureVectors[1][7][2] = 1.0;
            featureVectors[1][10][2] = 1.0;
            featureVectors[1][14][2] = 1.0;
            featureVectors[1][17][2] = 1.0;

            double oneThird = 1.0 / 3.0;
            featureVectors[1][1][3] = oneThird;
            featureVectors[1][1][4] = oneThird;
            featureVectors[1][1][5] = oneThird;
            featureVectors[1][11][3] = oneThird;
            featureVectors[1][11][4] = oneThird;
            featureVectors[1][11][5] = oneThird;
            featureVectors[1][12][3] = oneThird;
            featureVectors[1][12][4] = oneThird;
            featureVectors[1][12][5] = oneThird;
            featureVectors[1][15][3] = oneThird;
            featureVectors[1][15][4] = oneThird;
            featureVectors[1][15][5] = oneThird;
            featureVectors[1][18][3] = oneThird;
            featureVectors[1][18][4] = oneThird;
            featureVectors[1][18][5] = oneThird;

            actionPerformed[0][11] = true;
            actionPerformed[1][2] = true;
            actionPerformed[1][11] = true;
            actionPerformed[1][12] = true;
            actionPerformed[1][14] = true;
            actionPerformed[1][15] = true;
            actionPerformed[1][18] = true;

            HierarchicalBPMPrecisionLearning(3, featureVectors, actionPerformed, valueIndexToParamIndex);
        }

        private void HierarchicalBPMPrecisionLearning(
          int numParams,
          IList<SparseVector[]> featureVectors,
          IList<bool[]> labels,
          IList<int[]> indexMapping)
        {
            // ------------------
            // Model
            // ------------------

            var evidence = Variable.Bernoulli(0.5).Named("evidence");
            var block = Variable.If(evidence);

            // Range sizes
            var numberOfPersons = Variable.New<int>().Named("NumberOfPersons");
            var numberOfFeatureGroups = Variable.New<int>().Named("NumberOfFeatureGroups");

            // Ranges
            Range personRange = new Range(numberOfPersons).Named("PersonRange");
            var numberOfMessages = Variable.Array<int>(personRange).Named("NumberOfMessages");
            Range messageRange = new Range(numberOfMessages[personRange]).Named("MessageRange");
            var numberOfFeatures = Variable.Array<int>(personRange).Named("NumberOfFeatures");

            Range featureRange = new Range(numberOfFeatures[personRange]).Named("FeatureRange");
            Range featureGroupRange = new Range(numberOfFeatureGroups).Named("FeatureGroupRange");

            // Mapping of personal feature index to corresponding raw feature index
            var groupOf = Variable.Array(Variable.Array<int>(featureRange), personRange).Named("GroupOf");
            groupOf.SetValueRange(featureGroupRange);

            // Request a sequential schedule
            messageRange.AddAttribute(new Sequential());
            //featureRange.AddAttribute(new Sequential());
            //featureGroupRange.AddAttribute(new Sequential());

            var weightMeanPrior = Variable.New<GaussianArray>().Named("WeightMeanPrior");
            var weightMean = Variable.Array<double>(featureGroupRange).Named("WeightMean");
            weightMean.SetTo(Variable<double[]>.Random(weightMeanPrior));
            var weightPrecisionPrior = Variable.New<GammaArray>().Named("weightPrecisionPrior");
            var weightPrecision = Variable.Array<double>(featureGroupRange).Named("WeightPrecision");
            weightPrecision.SetTo(Variable<double[]>.Random(weightPrecisionPrior));
            var weights = Variable.Array(Variable.Array<double>(featureRange), personRange).Named("Weight");
            weights[personRange][featureRange] = Variable.GaussianFromMeanAndPrecision(weightMean[groupOf[personRange][featureRange]],
                                                                                       weightPrecision[groupOf[personRange][featureRange]]);

            // Number of features active per item
            var featureCounts = Variable.Array(Variable.Array<int>(messageRange), personRange).Named("FeatureCounts");
            Range featureItemRange = new Range(featureCounts[personRange][messageRange]).Named("featureItemRange");

            var isTrain = Variable.Array(Variable.Array<bool>(messageRange), personRange).Named("isTrain");

            // The observed features
            var featureIndices = Variable.Array(Variable.Array(Variable.Array<int>(featureItemRange), messageRange), personRange).Named("FeatureIndices"); // observed data
            var featureValues = Variable.Array(Variable.Array(Variable.Array<double>(featureItemRange), messageRange), personRange).Named("FeatureValues"); // observed data
                                                                                                                                                            // The label
            var isActionPerformed = Variable.Array(Variable.Array<bool>(messageRange), personRange).Named("IsActionPerformed");
            double noisePrecision = 0.1;
            using (Variable.ForEach(personRange))
            {
                // Loop over emails in a batch
                using (Variable.ForEach(messageRange))
                {
                    using (Variable.If(isTrain[personRange][messageRange]))
                    {
                        VariableArray<double> sparseWeights = Variable.Subarray(weights[personRange], featureIndices[personRange][messageRange]).Named("sparseWeights");
                        VariableArray<double> sparseProduct = Variable.Array<double>(featureItemRange).Named("SparseProduct");
                        sparseProduct[featureItemRange] = featureValues[personRange][messageRange][featureItemRange] * sparseWeights[featureItemRange];
                        Variable<double> dotProduct = Variable.Sum(sparseProduct).Named("DotProduct");
                        isActionPerformed[personRange][messageRange] = Variable.GaussianFromMeanAndPrecision(dotProduct, noisePrecision).Named("DotProductWithNoise") > 0;
                    }
                    //using (Variable.IfNot(isTrain[personRange][messageRange])) {
                    //    isActionPerformed[personRange][messageRange] = Variable.Bernoulli(0.5);
                    //}
                }
            }
            block.CloseBlock();

            // --------------------
            // Observations
            // --------------------
            int numUsers = featureVectors.Count;
            int[] nMessages = Util.ArrayInit(numUsers, p => featureVectors[p].Length);
            int[][] fCount = Util.ArrayInit(numUsers, p => Util.ArrayInit(nMessages[p], m => featureVectors[p][m].SparseCount));
            int[][][] fIndices = Util.ArrayInit(numUsers, p => Util.ArrayInit(nMessages[p], m => Util.ArrayInit(fCount[p][m], f => featureVectors[p][m].SparseValues[f].Index)));
            double[][][] fValues = Util.ArrayInit(numUsers, p => Util.ArrayInit(nMessages[p], m => Util.ArrayInit(fCount[p][m], f => featureVectors[p][m].SparseValues[f].Value)));
            int[] nFeatures = Util.ArrayInit(numUsers, p => indexMapping[p].Length);
            int[][] valueIndexToParamIndex = Util.ArrayInit(numUsers, p => indexMapping[p]);
            bool[][] lbl = Util.ArrayInit(numUsers, p => labels[p]);

            numberOfPersons.ObservedValue = numUsers;
            numberOfFeatureGroups.ObservedValue = numParams;
            numberOfMessages.ObservedValue = nMessages;

            featureCounts.ObservedValue = fCount;
            featureIndices.ObservedValue = fIndices;
            featureValues.ObservedValue = fValues;

            numberOfFeatures.ObservedValue = nFeatures;
            groupOf.ObservedValue = valueIndexToParamIndex;
            isActionPerformed.ObservedValue = lbl;
            weightMeanPrior.ObservedValue = new GaussianArray(numberOfFeatureGroups.ObservedValue, i => Gaussian.FromMeanAndPrecision(0.0, 0.5));
            weightPrecisionPrior.ObservedValue = new GammaArray(numberOfFeatureGroups.ObservedValue, i => Gamma.FromShapeAndRate(3, 3));

            isTrain.ObservedValue = Util.ArrayInit(numUsers, p => Util.ArrayInit(nMessages[p], m => Bernoulli.Sample(0.5)));
            int nTrain = 0, nTest = 0;
            for (int i = 0; i < isTrain.ObservedValue.Length; i++)
            {
                for (int j = 0; j < isTrain.ObservedValue[i].Length; j++)
                {
                    if (isTrain.ObservedValue[i][j])
                        nTrain++;
                    else
                        nTest++;
                }
            }
            weightMean.ObservedValue = new double[numParams];

            // --------------------
            // Inference
            // --------------------
            InferenceEngine engine = new InferenceEngine();
            if (false)
            {
                for (int iter = 1; iter < 20; iter++)
                {
                    engine.NumberOfIterations = iter;
                    var weightsPost = engine.Infer<IList<GaussianArray>>(weights);
                    Console.WriteLine("weights[0][0] = {0} {1}", weightsPost[0][0], weightsPost[0][1]);
                    Console.WriteLine("weightPrec = {0}", engine.Infer<Gamma[]>(weightPrecision)[1]);
                }
            }
            if (false)
            {
                double pmin = 0.002;
                double pmax = 0.003;
                int n = 1;
                double inc = (pmax - pmin) / n;
                //weightMean.ObservedValue = new double[] { -2, -2 };
                //weightMean.ObservedValue = new double[] { -1.4, 0.7 };
                for (int i = 0; i < n; i++)
                {
                    weightPrecision.ObservedValue = new double[] { 0.5, i * inc + pmin };
                    Console.WriteLine("{1} evidence = {0}", engine.Infer<Bernoulli>(evidence).LogOdds, weightPrecision.ObservedValue[1]);
                }
            }
            if (false)
            {
                // use importance sampling from the posterior to verify the evidence value
                Console.WriteLine("evidence = {0}", engine.Infer<Bernoulli>(evidence).LogOdds);
                var weightPost = engine.Infer<GaussianArrayArray>(weights);
                var weightPrecPost2 = engine.Infer<IList<Gamma>>(weightPrecision);
                MeanVarianceAccumulator trainAcc = new MeanVarianceAccumulator();
                MeanVarianceAccumulator testAcc = new MeanVarianceAccumulator();
                MeanVarianceAccumulator evAcc = new MeanVarianceAccumulator();
                double evShift = 0;
                MeanVarianceAccumulator trainLogProbAcc = new MeanVarianceAccumulator();
                MeanVarianceAccumulator testLogProbAcc = new MeanVarianceAccumulator();
                for (int iter = 0; iter < 100; iter++)
                {
                    // draw sample of the weights
                    var weightsSample = weightPost.Sample();
                    // compute predictions
                    double ev = -weightPost.GetLogProb(weightsSample);
                    int trainErrors = 0, testErrors = 0;
                    double trainLogProb = 0, testLogProb = 0;
                    for (int i = 0; i < numUsers; i++)
                    {
                        for (int j = 0; j < nMessages[i]; j++)
                        {
                            int count = fCount[i][j];
                            Gaussian[] products = new Gaussian[count];
                            for (int k = 0; k < count; k++)
                            {
                                int fi = fIndices[i][j][k];
                                double fv = fValues[i][j][k];
                                //Gaussian w = weightPost[i][fi];
                                Gaussian w = Gaussian.PointMass(weightsSample[i][fi]);
                                products[k] = GaussianProductOp.ProductAverageConditional(w, fv);
                            }
                            Gaussian score = FastSumOp.SumAverageConditional(products);
                            Gaussian noisyScore = GaussianOp.SampleAverageConditional(score, noisePrecision);
                            //Console.WriteLine("{1} score = {0}", score.GetMean(), lbl[i][j]);
                            if ((score.GetMean() > 0) != lbl[i][j])
                                if (isTrain.ObservedValue[i][j])
                                    trainErrors++;
                                else
                                    testErrors++;
                            double logProb = IsPositiveOp.LogEvidenceRatio(lbl[i][j], noisyScore);
                            if (isTrain.ObservedValue[i][j])
                            {
                                ev += logProb;
                                trainLogProb += logProb;
                            }
                            else
                                testLogProb += logProb;
                        }
                    }
                    trainAcc.Add(trainErrors);
                    testAcc.Add(testErrors);
                    trainLogProbAcc.Add(trainLogProb);
                    testLogProbAcc.Add(testLogProb);
                    for (int i = 0; i < weightPost.Count; i++)
                    {
                        for (int j = 0; j < weightPost[i].Count; j++)
                        {
                            int group = valueIndexToParamIndex[i][j];
                            Gaussian prior = Gaussian.FromMeanAndPrecision(weightMean.ObservedValue[group], weightPrecPost2[group].GetMean());
                            //sumLogProb += prior.GetLogProb((weightPost[i][j]/prior).GetMean());
                            //sumLogProb += GaussianOp.LogAverageFactor(weightPost[i][j]/prior, 0, weightPrecPost2[group]);
                            //sumLogProb += prior.GetLogAverageOf(weightPost[i][j]/prior);
                            ev += prior.GetLogProb(weightsSample[i][j]);
                        }
                    }
                    if (iter == 1)
                        evShift = ev;
                    evAcc.Add(System.Math.Exp(ev - evShift));
                    //Console.WriteLine("sampled evidence = {0}", ev);
                }
                Console.WriteLine("train error = {0}% +/- {1}", (trainAcc.Mean / nTrain * 100).ToString("g2"), (System.Math.Sqrt(trainAcc.Variance) / nTrain * 100).ToString("g2"));
                Console.WriteLine("test error = {0}% +/- {1}", (testAcc.Mean / nTest * 100).ToString("g2"), (System.Math.Sqrt(testAcc.Variance) / nTest * 100).ToString("g2"));
                Console.WriteLine("avg train logProb = {0}", trainLogProbAcc.Mean / nTrain);
                Console.WriteLine("avg test logProb = {0}", testLogProbAcc.Mean / nTest);
                Console.WriteLine("evidence = {0}", System.Math.Log(evAcc.Mean) + evShift);
                using (var writer = new MatlabWriter("weightsEP.mat"))
                {
                    for (int i = 0; i < weightPost.Count; i++)
                    {
                        Matrix m = new Matrix(weightPost[i].Count, 2);
                        for (int j = 0; j < weightPost[i].Count; j++)
                        {
                            m[j, 0] = weightPost[i][j].GetMean();
                            m[j, 1] = weightPost[i][j].GetVariance();
                        }
                        writer.Write("w" + i, m);
                    }
                    for (int i = 0; i < nMessages.Length; i++)
                    {
                        Matrix x = new Matrix(nMessages[i], nFeatures[i]);
                        for (int j = 0; j < nMessages[i]; j++)
                        {
                            for (int k = 0; k < fCount[i][j]; k++)
                            {
                                int fi = fIndices[i][j][k];
                                double fv = fValues[i][j][k];
                                x[j, fi] = fv;
                            }
                        }
                        writer.Write("x" + i, x);
                    }
                }
            }

            engine.NumberOfIterations = 1;
            var weightsPost1 = engine.Infer<Diffable>(weights);
            engine.NumberOfIterations = 50;

            var weightMeanPost = engine.Infer<IList<Gaussian>>(weightMean);
            var weightPrecPost = engine.Infer<IList<Gamma>>(weightPrecision);

            Console.WriteLine("Mean posteriors");
            Console.WriteLine(weightMeanPost);

            Console.WriteLine("Precision posteriors");
            Console.WriteLine(weightPrecPost);
            foreach (var wpp in weightPrecPost)
            {
                Console.WriteLine("Mean = {0}, Variance = {1}", wpp.GetMean(), wpp.GetVariance());
            }

            // check that the generated algorithm is reset correctly
            engine.NumberOfIterations = 1;
            var weightsPost2 = engine.Infer<Diffable>(weights);
            Assert.True(weightsPost2.MaxDiff(weightsPost1) < 1e-10);
        }
        

        [Fact]
        public void HierarchicalBPMPrecisionLearning2Test()
        {
            // ------------------
            // Model
            // ------------------

            // Range sizes
            var numberOfMessages = Variable.New<int>().Named("NumberOfMessages");
            var numberOfFeatures = Variable.New<int>().Named("NumberOfFeatures");
            var numberOfFeatureGroups = Variable.New<int>().Named("NumberOfFeatureGroups");

            // Ranges
            Range messageRange = new Range(numberOfMessages).Named("MessageRange");
            Range featureRange = new Range(numberOfFeatures).Named("FeatureRange");
            Range featureGroupRange = new Range(numberOfFeatureGroups).Named("FeatureGroupRange");

            // Mapping of personal feature index to corresponding raw feature index
            var groupOf = Variable.Array<int>(featureRange).Named("GroupOf");
            groupOf.SetValueRange(featureGroupRange);

            // Request a sequential schedule
            messageRange.AddAttribute(new Sequential());

            var weightMeanPrior = Variable.New<GaussianArray>().Named("WeightMeanPrior");
            var weightMean = Variable.Array<double>(featureGroupRange).Named("WeightMean");
            weightMean.SetTo(Variable<double[]>.Random(weightMeanPrior));
            var weightPrecisionPrior = Variable.New<GammaArray>().Named("weightPrecisionPrior");
            var weightPrecision = Variable.Array<double>(featureGroupRange).Named("WeightPrecision");
            weightPrecision.SetTo(Variable<double[]>.Random(weightPrecisionPrior));
            var weights = Variable.Array<double>(featureRange).Named("Weight");
            weights[featureRange] = Variable.GaussianFromMeanAndPrecision(weightMean[groupOf[featureRange]], weightPrecision[groupOf[featureRange]]);
            var weightInitialiser = Variable.New<GaussianArray>().Named("WeightInitialiser");
            weights.InitialiseTo(weightInitialiser);
            var collapsedMessages = Variable.Array<Gaussian>(featureRange).Named("CollapsedMessages");
            Variable.ConstrainEqualRandom(weights[featureRange], collapsedMessages[featureRange]);

            // Number of features active per item
            var featureCounts = Variable.Array<int>(messageRange).Named("FeatureCounts");
            Range featureItemRange = new Range(featureCounts[messageRange]).Named("featureItemRange");

            // The observed features
            var featureIndices = Variable.Array(Variable.Array<int>(featureItemRange), messageRange).Named("FeatureIndices"); // observed data
            var featureValues = Variable.Array(Variable.Array<double>(featureItemRange), messageRange).Named("FeatureValues"); // observed data
                                                                                                                               // The label
            var isActionPerformed = Variable.Array<bool>(messageRange).Named("IsActionPerformed");
            var noise = Variable.New<double>().Named("noise");

            // Loop over emails in a batch
            using (Variable.ForEach(messageRange))
            {
                VariableArray<double> sparseWeights = Variable.Subarray(weights, featureIndices[messageRange]);
                VariableArray<double> sparseProduct = Variable.Array<double>(featureItemRange).Named("SparseProduct");
                sparseProduct[featureItemRange] = featureValues[messageRange][featureItemRange] * sparseWeights[featureItemRange];
                Variable<double> dotProduct = Variable.Sum(sparseProduct).Named("DotProduct");
                isActionPerformed[messageRange] = Variable.GaussianFromMeanAndPrecision(dotProduct, noise).Named("DotProductWithNoise") > 0;
            }

            // ----------------
            // Data
            // ----------------
            int nData = 20;
            int nFeatures = 6;
            SparseVector[] featureVectors = new SparseVector[nData];
            bool[] actionPerformed = new bool[nData];

            for (int n = 0; n < nData; n++)
            {
                featureVectors[n] = SparseVector.Zero(nFeatures);
                featureVectors[n][0] = 1;
                featureVectors[n][1] = 1;
                actionPerformed[n] = false;
            }

            int[] valueIndexToParamIndex = new int[] { 0, 1, 2, 2, 2, 2 };

            featureVectors[0][2] = 1.0;
            featureVectors[5][2] = 1.0;
            featureVectors[7][2] = 1.0;
            featureVectors[10][2] = 1.0;
            featureVectors[14][2] = 1.0;
            featureVectors[17][2] = 1.0;

            double oneThird = 1.0 / 3.0;
            featureVectors[1][3] = oneThird;
            featureVectors[1][4] = oneThird;
            featureVectors[1][5] = oneThird;
            featureVectors[11][3] = oneThird;
            featureVectors[11][4] = oneThird;
            featureVectors[11][5] = oneThird;
            featureVectors[12][3] = oneThird;
            featureVectors[12][4] = oneThird;
            featureVectors[12][5] = oneThird;
            featureVectors[15][3] = oneThird;
            featureVectors[15][4] = oneThird;
            featureVectors[15][5] = oneThird;
            featureVectors[18][3] = oneThird;
            featureVectors[18][4] = oneThird;
            featureVectors[18][5] = oneThird;

            actionPerformed[2] = true;
            actionPerformed[11] = true;
            actionPerformed[12] = true;
            actionPerformed[14] = true;
            actionPerformed[15] = true;
            actionPerformed[18] = true;

            int[] fCount = Util.ArrayInit(nData, m => featureVectors[m].SparseCount);
            int[][] fIndices = Util.ArrayInit(nData, m => Util.ArrayInit(fCount[m], f => featureVectors[m].SparseValues[f].Index));
            double[][] fValues = Util.ArrayInit(nData, m => Util.ArrayInit(fCount[m], f => featureVectors[m].SparseValues[f].Value));

            // ----------------------
            // Observations
            // ----------------------
            numberOfMessages.ObservedValue = nData;
            numberOfFeatures.ObservedValue = nFeatures;
            featureCounts.ObservedValue = fCount;
            featureIndices.ObservedValue = fIndices;
            featureValues.ObservedValue = fValues;
            numberOfFeatureGroups.ObservedValue = 3;
            groupOf.ObservedValue = valueIndexToParamIndex;
            isActionPerformed.ObservedValue = actionPerformed;
            noise.ObservedValue = 0.1;
            weightMeanPrior.ObservedValue = new GaussianArray(
                      new Gaussian[]
                          {
                    Gaussian.FromMeanAndVariance(-1.046, 1.42),
                    Gaussian.FromMeanAndVariance(-1.046, 1.42),
                    Gaussian.FromMeanAndVariance(-0.5519, 1.71),
                      });

            weightPrecisionPrior.ObservedValue = new GammaArray(
                      new Gamma[]
                          {
                    Gamma.FromShapeAndRate(2, 0.3193),
                    Gamma.FromShapeAndRate(2, 0.3193),
                    Gamma.FromShapeAndRate(2.026, 0.3302)
                      });

            weightInitialiser.ObservedValue = new GaussianArray(Util.ArrayInit(nFeatures, f => Gaussian.FromMeanAndVariance(0, 2.333)));
            collapsedMessages.ObservedValue = Util.ArrayInit(nFeatures, f => Gaussian.Uniform());

            // --------------------
            // Inference
            // --------------------
            InferenceEngine engine = new InferenceEngine();
            //engine.Compiler.FreeMemory = true;
            var weightMeanPost = engine.Infer<Gaussian[]>(weightMean);
            var weightPrecisionPost = engine.Infer<Gamma[]>(weightPrecision);
        }

        private class Basis
        {
            private Vector seed;
            private double[] diagonal;

            /// <summary>
            /// Construct an orthogonal basis from a seed vector
            /// </summary>
            /// <param name="seed"></param>
            public Basis(Vector seed)
            {
                this.seed = seed;
                int d = seed.Count;
                diagonal = new double[d];
                double sumsq = 0;
                for (int i = 0; i < d; i++)
                {
                    if (seed[i] == 0.0)
                        throw new ArgumentException("seed[" + i + "] == 0");
                    diagonal[i] = -sumsq / seed[i];
                    sumsq += seed[i] * seed[i];
                }
            }
            /// <summary>
            /// Project a vector onto the basis
            /// </summary>
            /// <param name="x"></param>
            /// <returns></returns>
            public Vector Project(Vector x)
            {
                int d = seed.Count;
                Vector result = Vector.Zero(d);
                double partialInner = 0;
                for (int i = 0; i < d; i++)
                {
                    result[i] = partialInner + x[i] * diagonal[i];
                    partialInner += x[i] * seed[i];
                }
                result[0] = partialInner;
                return result;
            }
        }

        // Test sequential attribute for hierarchical three state importance model
        [Fact]
        [Trait("Category", "OpenBug")]
        public void HierarchicalThreeStateImportanceModelTest()
        {
            var wB = HierarchicalThreeStateImportanceModel(false);
            var wA = HierarchicalThreeStateImportanceModel(true);

            Assert.True(wA.MaxDiff(wB) < 1e-10);
        }

        // Test sequential attribute for three state importance model
        private Gaussian HierarchicalThreeStateImportanceModel(bool unroll)
        {
            double noisePrecision = 0.1;
            // Ranges
            int numMessages = 2;
            int numPersonalFeatures = 1;
            int numSharedFeatures = 1;
            Range N = new Range(numMessages).Named("N");
            Range PF = new Range(numPersonalFeatures).Named("PF");
            Range SF = new Range(numSharedFeatures).Named("SF");

            // This is what we are testing
            N.AddAttribute(new Sequential());

            var WPersonalFeaturesDist = Variable.New<GaussianArray>().Named("WPersonalFeaturePrior");
            var WPersonalFeatures = Variable.Array<double>(PF).Named("WPersonalFeatures");
            WPersonalFeatures.SetTo(Variable<double[]>.Random(WPersonalFeaturesDist));

            var WFeaturesSharedMeanPrior = Variable.New<GaussianArray>().Named("WFeaturesSharedMeanPrior");
            var WFeaturesSharedMean = Variable.Array<double>(SF).Named("WFeaturesSharedMean");
            WFeaturesSharedMean.SetTo(Variable<double[]>.Random(WFeaturesSharedMeanPrior));
            var WFeaturesSharedPrecision = Variable.Array<double>(SF).Named("WFeaturesSharedPrecision");
            var WSharedFeatures = Variable.Array<double>(SF).Named("WFeatures"); // parameter array
            WSharedFeatures[SF].SetTo(Variable.GaussianFromMeanAndPrecision(WFeaturesSharedMean[SF], WFeaturesSharedPrecision[SF]));

            var WSharedFeaturesConstraintDist = Variable.Array<Gaussian>(SF).Named("WFeaturesContraintDist");
            Variable.ConstrainEqualRandom(WSharedFeatures[SF], WSharedFeaturesConstraintDist[SF]);

            var UnimportantThresholdMeanPrior = Variable.New<Gaussian>().Named("UnimportantThresholdMeanPrior");
            var UnimportantThresholdSharedMean = Variable.New<double>().Named("UnimportantThresholdSharedMean");
            UnimportantThresholdSharedMean.SetTo(Variable<double>.Random(UnimportantThresholdMeanPrior));
            var UnimportantThresholdSharedPrecision = Variable.New<double>().Named("UnimportantThresholdSharedPrecision");
            var UnimportantThreshold = Variable.New<double>().Named("UnimportantThreshold");
            UnimportantThreshold.SetTo(Variable.GaussianFromMeanAndPrecision(UnimportantThresholdSharedMean, UnimportantThresholdSharedPrecision));

            var UnimportantThresholdConstraintDist = Variable.New<Gaussian>().Named("UnimportantThresholdConstraintDist");
            Variable.ConstrainEqualRandom(UnimportantThreshold, UnimportantThresholdConstraintDist);

            // Number of features active per item
            var SharedFeatureCounts = Variable.Array<int>(N);
            Range SFitem = new Range(SharedFeatureCounts[N]).Named("SFItem");

            // The observed features
            var SharedFeatureIndices = Variable.Array(Variable.Array<int>(SFitem), N).Named("SharedFeatureIndices"); // observed data
            var SharedFeatureValues = Variable.Array(Variable.Array<double>(SFitem), N).Named("SharedFeatureValues"); // observed data

            // Number of features active per item
            var PersonalFeatureCounts = Variable.Array<int>(N);
            Range PFitem = new Range(PersonalFeatureCounts[N]).Named("PFItem");

            // The observed features
            var PersonalFeatureIndices = Variable.Array(Variable.Array<int>(PFitem), N).Named("PersonalFeatureIndices"); // observed data
            var PersonalFeatureValues = Variable.Array(Variable.Array<double>(PFitem), N).Named("PersonalFeatureValues"); // observed data

            // is a message important?
            var IsImportant = Variable.Array<bool>(N);

            // is a message unimportant?
            var IsUnimportant = Variable.Array<bool>(N);

            // Loop over emails
            using (Variable.ForEach(N))
            {
                var SWsparse = Variable.Subarray(WSharedFeatures, SharedFeatureIndices[N]);
                var Sproduct = Variable.Array<double>(SFitem);
                Sproduct[SFitem] = SharedFeatureValues[N][SFitem] * SWsparse[SFitem];

                var PWSparse = Variable.Subarray(WPersonalFeatures, PersonalFeatureIndices[N]);
                var Pproduct = Variable.Array<double>(PFitem);
                Pproduct[PFitem] = PersonalFeatureValues[N][PFitem] * PWSparse[PFitem];

                var W = Variable.Sum(Sproduct) + Variable.Sum(Pproduct);

                var score = Variable.GaussianFromMeanAndPrecision(W, noisePrecision);

                IsImportant[N] = score > 0;
                IsUnimportant[N] = score < UnimportantThreshold;
            }

            // Don't allow division for variables indexed by the range we want to unroll
            SharedFeatureIndices.AddAttribute(new DivideMessages(false));
            SharedFeatureValues.AddAttribute(new DivideMessages(false));
            PersonalFeatureIndices.AddAttribute(new DivideMessages(false));
            PersonalFeatureValues.AddAttribute(new DivideMessages(false));
            IsImportant.AddAttribute(new DivideMessages(false));
            IsUnimportant.AddAttribute(new DivideMessages(false));

            WPersonalFeaturesDist.ObservedValue = new GaussianArray(new Gaussian[] { new Gaussian(0, 1) });
            WSharedFeaturesConstraintDist.ObservedValue = new Gaussian[] { new Gaussian(0, 1) };
            WFeaturesSharedMeanPrior.ObservedValue = new GaussianArray(new Gaussian[] { new Gaussian(0, 1) });
            WFeaturesSharedPrecision.ObservedValue = new double[] { 1 };
            PersonalFeatureCounts.ObservedValue = new int[] { 1, 1 };
            PersonalFeatureIndices.ObservedValue = new int[][] { new int[] { 0 }, new int[] { 0 } };
            PersonalFeatureValues.ObservedValue = new double[][] { new double[] { 1 }, new double[] { 1 } };
            SharedFeatureCounts.ObservedValue = new int[] { 1, 1 };
            SharedFeatureIndices.ObservedValue = new int[][] { new int[] { 0 }, new int[] { 0 } };
            SharedFeatureValues.ObservedValue = new double[][] { new double[] { 1 }, new double[] { 1 } };
            IsImportant.ObservedValue = new bool[] { true, true };
            IsUnimportant.ObservedValue = new bool[] { false, false };
            UnimportantThresholdMeanPrior.ObservedValue = Gaussian.FromMeanAndPrecision(-1, 0.1);
            UnimportantThresholdSharedPrecision.ObservedValue = 1.0;
            UnimportantThresholdConstraintDist.ObservedValue = Gaussian.Uniform();
            InferenceEngine engine = new InferenceEngine();

            engine.NumberOfIterations = 1;
            engine.Compiler.UnrollLoops = unroll;
            return engine.Infer<Gaussian[]>(WFeaturesSharedMean)[0];
        }

        // Test sequential attribute for featureWeights
        // TODO: This test is flawed since it passes even when the update for featureWeights is not sequential
        // Must use JaggedSubarray to have a chance of being sequential
        [Fact]
        public void ThreeStateImportanceModelTest()
        {
            var wB = ThreeStateImportanceModel(false);
            var wA = ThreeStateImportanceModel(true);
            Console.WriteLine("w = {0} should be {1}", wB, wA);
            Assert.True(wA.MaxDiff(wB) < 1e-10);
        }

        // Test sequential attribute for three state importance model
        public Gaussian ThreeStateImportanceModel(bool unroll)
        {
            double noisePrecision = 0.1;
            // Ranges
            int numberOfMessages = 2;
            int numberOfFeatures = 1;
            Range featureRange = new Range(numberOfFeatures).Named("featureRange");
            Range messageRange = new Range(numberOfMessages).Named("messageRange");

            // Make sure that the range across messages is handled sequentially
            // - this is necessary to ensure the model converges when training with large batch sizes
            messageRange.AddAttribute(new Sequential());

            // Number of features active per item
            var featureCounts = Variable.Array<int>(messageRange).Named("featureCounts");
            Range featureItemRange = new Range(featureCounts[messageRange]).Named("featureItemRange");

            // The sparse observed feature vector
            var featureIndices = Variable.Array(Variable.Array<int>(featureItemRange), messageRange).Named("featureIndices"); // observed data
            var featureValues = Variable.Array(Variable.Array<double>(featureItemRange), messageRange).Named("featureValues"); // observed data

            // The priors on the weights
            var weightPriors = Variable.New<GaussianArray>().Named("weightPriors");

            // The weights
            var featureWeights = Variable.Array<double>(featureRange).Named("featureWeights");
            featureWeights.SetTo(Variable<double[]>.Random(weightPriors));

            // Initialisation - all the batched version messages to
            // be initialised with marginals
            //var initialWeights = Variable.New<GaussianArray>().Named("initialWeights");
            //featureWeights.InitialiseTo(initialWeights);

            // Threshold on the score for unimportant messages
            var unimportantThresholdPrior = Variable.New<Gaussian>().Named("unimportantThresholdPrior");
            var unimportantThreshold = Variable.Random<double, Gaussian>(unimportantThresholdPrior).Named("unimportantThreshold");

            // Target: is the message important?
            var isImportant = Variable.Array<bool>(messageRange).Named("isImportant");

            // Target: is the message unimportant?
            var isUnimportant = Variable.Array<bool>(messageRange).Named("isUnimportant");

            // softLabels = Variable.Array<Bernoulli>(messageRange).Named("softLabels");

            // This will cause the engine to loop over all the e-mails when
            // training and classifying.
            using (Variable.ForEach(messageRange))
            {
                VariableArray<double> sparseWeights = Variable.Subarray(featureWeights, featureIndices[messageRange]).Named("sparseWeights");
                VariableArray<double> product = Variable.Array<double>(featureItemRange).Named("product");
                product[featureItemRange] = featureValues[messageRange][featureItemRange] * sparseWeights[featureItemRange];
                Variable<double> weight = Variable.Sum(product).Named("featureWeight");
                var score = Variable.GaussianFromMeanAndPrecision(weight, noisePrecision).Named("Score");
                isImportant[messageRange] = score > 0;
                isUnimportant[messageRange] = score < unimportantThreshold;
                // Variable.ConstrainEqualRandom(isImportant[messageRange], softLabels[messageRange]);
            }
            // Don't allow division for variables indexed by the range we want to unroll
            //isImportant.AddAttribute(new DivideMessages(false));
            //isUnimportant.AddAttribute(new DivideMessages(false));
            //featureCounts.AddAttribute(new DivideMessages(false));
            //featureIndices.AddAttribute(new DivideMessages(false));
            //featureValues.AddAttribute(new DivideMessages(false));
            //softLabels.AddAttribute(new DivideMessages(false));

            weightPriors.ObservedValue = new GaussianArray(new Gaussian[] { new Gaussian(0, 1) });
            featureCounts.ObservedValue = new int[] { 1, 1 };
            featureIndices.ObservedValue = new int[][] { new int[] { 0 }, new int[] { 0 } };
            featureValues.ObservedValue = new double[][] { new double[] { 1 }, new double[] { 1 } };
            isImportant.ObservedValue = new bool[] { true, true };
            isUnimportant.ObservedValue = new bool[] { false, false };
            unimportantThresholdPrior.ObservedValue = Gaussian.FromMeanAndPrecision(-1, 0.1);
            //initialWeights.ObservedValue = (GaussianArray)Distribution<double>.Array(new Gaussian[] { Gaussian.Uniform() });
            InferenceEngine engine = new InferenceEngine();
            engine.ModelName = "ThreeStateImportanceModel";
            engine.Compiler.UnrollLoops = unroll;
            return engine.Infer<Gaussian[]>(featureWeights)[0];
        }

        internal void SpeedTest2()
        {
            Variable<Vector> trainingWeightVector;
            VariableArray<bool> willBuyTraining;
            Variable<double> noise = Variable.New<double>();

            //age and income Training data
            double[] incomeTrainingData;
            double[] ageTrainingData;
            bool[] willBuyTrainingData;
            double[] noiseValues = { 0.01, 0.04, 0.16, 0.64, 2.56, 5.12, 10.24, 20.48, 40.96 };
            int numData = 500;
            double stdDev = 5.0; //standard deviation of the noise

            GenerateData(numData, stdDev, out incomeTrainingData, out ageTrainingData, out willBuyTrainingData);
            //      for (int i = 0; i < numData; i++)
            //      {
            //        Console.WriteLine("income: {0:f2}  age: {1:f2}  willBuy: {2}", incomeTrainingData[i], ageTrainingData[i], willBuyTrainingData[i]);
            //      }
            //Create x* array
            Range trainingDataRange = new Range(incomeTrainingData.Length);
            Vector[] trainingDataArray = new Vector[incomeTrainingData.Length];
            for (int i = 0; i < incomeTrainingData.Length; i++)
            {
                trainingDataArray[i] = Vector.FromArray(incomeTrainingData[i], ageTrainingData[i], 1);
            }
            VariableArray<Vector> trainingData = Variable.Array<Vector>(trainingDataRange);
            trainingData.ObservedValue = trainingDataArray;

            Variable<bool> evidence = Variable.Bernoulli(0.5);
            using (Variable.If(evidence))
            {
                //Create w* array
                trainingWeightVector = Variable.Random(new VectorGaussian(Vector.Zero(3),
                                                                PositiveDefiniteMatrix.Identity(3))).Named("w");
                //Create result array
                willBuyTraining = Variable.Array<bool>(trainingDataRange);
                willBuyTraining.ObservedValue = willBuyTrainingData;

                //Create model bayes point machine model for training data
                willBuyTraining[trainingDataRange] = Variable.IsPositive(
                  Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(trainingWeightVector, trainingData[trainingDataRange]), noise));
            }
            //trainingWeightVector.AddAttribute(new DivideMessages(false));

            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;

            Console.WriteLine("Number of samples: {0}", numData);
            for (int i = 0; i < noiseValues.Length; i++)
            {
                //Infer trained weights
                noise.ObservedValue = noiseValues[i];
                VectorGaussian trainedWeightDist = engine.Infer<VectorGaussian>(trainingWeightVector);
                double logEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;

                //print the weight vector's mean values and covariance matrix
                Vector meanTrained = Vector.Zero(3);
                PositiveDefiniteMatrix varianceTrained = new PositiveDefiniteMatrix(3, 3);
                trainedWeightDist.GetMeanAndVariance(meanTrained, varianceTrained);

                Console.WriteLine("Noise: {0:f2}   Log evidence: {1:f2} Weights: {2}", noiseValues[i], logEvidence, meanTrained.ToString());
            }
        }
        private static void GenerateData(int numData, double stdDev, out double[] income, out double[] age, out bool[] willBuy)
        {
            income = new double[numData];
            age = new double[numData];
            willBuy = new bool[numData];
            Rand.Restart(42);
            int i;

            for (i = 0; i < numData; i++)
            {
                income[i] = (Rand.Double() * 50) + 20.0;
                age[i] = (Rand.Double() * 40) + 20;
                willBuy[i] = (income[i] * 2 - age[i] * 2.5) > 0;
            }
            if (stdDev > 0.01) //add noise
            {
                for (i = 0; i < numData; i++)
                {
                    while ((income[i] += Rand.Normal(0, stdDev)) < 10) //discard ages<10 years old
                        continue;
                }
            }
        }

        /// <summary>
        /// Same as FactorizedBayesPointEvidence but xdata is represented sparsely.
        /// </summary>
        [Fact]
        public void SparseFactorizedBayesPointEvidence()
        {
            // Bayes Point Machine on the 3-point problem
            // Start model definition
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            Variable<int> nUser = Variable.New<int>().Named("nUser");
            Range user = new Range(nUser).Named("user");
            Variable<int> nFeatures = Variable.New<int>().Named("nFeatures");
            Range feature = new Range(nFeatures).Named("feature");
            VariableArray<double> w = Variable.Array<double>(feature).Named("w");
            w[feature] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(feature);
            VariableArray<int> xValueCount = Variable.Array<int>(user).Named("xCount");
            Range userFeature = new Range(xValueCount[user]).Named("userFeature");
            var xValues = Variable.Array<double>(Variable.Array<double>(userFeature), user).Named("xValues");
            var xIndices = Variable.Array<int>(Variable.Array<int>(userFeature), user).Named("xIndices");
            VariableArray<bool> y = Variable.Array<bool>(user).Named("y");
            using (Variable.ForEach(user))
            {
                VariableArray<double> product = Variable.Array<double>(userFeature).Named("product");
                VariableArray<double> wSparse = Variable.Subarray(w, xIndices[user]);
                product[userFeature] = xValues[user][userFeature] * wSparse[userFeature];
                // The following also works, but is slower:
                //product[userFeature] = xValues[user][userFeature] * w[xIndices[user][userFeature]];
                Variable<double> score = Variable.Sum(product).Named("score");
                y[user] = (score > 0);
            }
            block.CloseBlock();
            // End of model definition
            user.AddAttribute(new Sequential());

            // this is a sparse encoding of the three vectors:
            // {0,0,1} {0,1,1} {1,0,1}
            int[][] xIndicesData = { new int[] { 2 }, new int[] { 1, 2 }, new int[] { 0, 2 } };
            double[][] xValuesData = { new double[] { 1 }, new double[] { 1, 1 }, new double[] { 1, 1 } };
            bool[] yData = { true, true, false };

            nFeatures.ObservedValue = 3;
            nUser.ObservedValue = yData.Length;
            y.ObservedValue = yData;
            xValues.ObservedValue = xValuesData;
            xIndices.ObservedValue = xIndicesData;

            int[] xValueCountData = new int[xIndicesData.Length];
            for (int i = 0; i < xIndicesData.Length; i++)
            {
                xValueCountData[i] = xIndicesData[i].Length;
            }
            xValueCount.ObservedValue = xValueCountData;


            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            object wActual = engine.Infer<object>(w);
            Console.WriteLine(wActual);
            Vector mwTrue = Vector.FromArray(-1.178410192370038, 0.514204026416866, 0.533030909628306);
            PositiveDefiniteMatrix vwTrue = new PositiveDefiniteMatrix(new double[,]
                {
                {0.3578619594789516, 0.05893838160003953, -0.1183834842062737,},
                {0.05893838160003954, 0.5424263963482457, -0.08091881706709862,},
                {-0.1183834842062736, -0.08091881706709868, 0.1625333312892799,}
            });
            Gaussian[] wExpected = new Gaussian[3];
            for (int i = 0; i < 3; i++)
            {
                wExpected[i] = new Gaussian(mwTrue[i], vwTrue[i, i]);
            }
            Assert.True(Distribution<double>.Array(wExpected).MaxDiff(wActual) < 1e-6);

            double evExpected = -2.486152518337251;
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-10) < 1e-10);
        }

        // TODO: why is w_rep_F_marginal reversed here?
        [Fact]
        public void FactorizedBayesPointEvidence()
        {
            // Bayes Point Machine on the 3-point problem
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double[][] xdata = { new double[] { 0, 0, 1 }, new double[] { 0, 1, 1 }, new double[] { 1, 0, 1 } };
            bool[] ydata = { true, true, false };
            int n = ydata.Length;
            Range user = new Range(n);
            VariableArray<bool> y = Variable.Constant(ydata, user).Named("y");
            Range feature = new Range(3);
            var x = Variable.Constant(xdata, user, feature).Named("x");
            VariableArray<double> w = Variable.Array<double>(feature).Named("w");
            w[feature] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(feature);
            using (Variable.ForEach(user))
            {
                VariableArray<double> product = Variable.Array<double>(feature).Named("product");
                product[feature] = x[user][feature] * w[feature];
                Variable<double> score = Variable.Sum(product).Named("score");
                y[user] = (score > 0);
            }
            block.CloseBlock();
            user.AddAttribute(new Sequential());

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            //engine.BrowserMode = BrowserMode.Always;
            DistributionArray<Gaussian> wActual = engine.Infer<DistributionArray<Gaussian>>(w);
            Vector mwTrue = Vector.FromArray(-1.178410192370038, 0.514204026416866, 0.533030909628306);
            PositiveDefiniteMatrix vwTrue = new PositiveDefiniteMatrix(new double[,] {
                {0.3578619594789516, 0.05893838160003953, -0.1183834842062737,},
                {0.05893838160003954, 0.5424263963482457, -0.08091881706709862,},
                {-0.1183834842062736, -0.08091881706709868, 0.1625333312892799,}
            });
            var wExpected = new GaussianArray(3);
            for (int i = 0; i < 3; i++)
            {
                wExpected[i] = new Gaussian(mwTrue[i], vwTrue[i, i]);
            }
            Console.WriteLine(StringUtil.JoinColumns("w = ", wActual, " should be ", wExpected));
            Assert.True(wExpected.MaxDiff(wActual) < 1e-6);

            double evExpected = -2.486152518337251;
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-10) < 1e-10);
        }

        [Fact]
        public void FactorizedBayesPointEvidence2()
        {
            // Bayes Point Machine on the 3-point problem
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double[][] xdata = { new double[] { 0, 0, 1 }, new double[] { 0, 1, 1 }, new double[] { 1, 0, 1 } };
            bool[] ydata = { true, true, false };
            int n = ydata.Length;
            Range user = new Range(n).Named("user");
            VariableArray<bool> y = Variable.Constant(ydata, user).Named("y");
            Range feature = new Range(3).Named("feature");
            var x = Variable.Constant(xdata, user, feature).Named("x");
            VariableArray<double> w = Variable.Array<double>(feature).Named("w");
            w[feature] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(feature);
            using (Variable.ForEach(user))
            {
                VariableArray<double> product = Variable.Array<double>(feature).Named("product");
                using (var fb = Variable.ForEach(feature))
                {
                    var c = (fb.Index == 0);
                    //var c = (fb.Index != 0);
                    using (Variable.If(c))
                    {
                        product[feature] = x[user][feature] * w[feature];
                    }
                    using (Variable.IfNot(c))
                    {
                        product[feature] = x[user][feature] * w[feature];
                    }
                }
                Variable<double> score = Variable.Sum(product).Named("score");
                y[user] = (score > 0);
            }
            block.CloseBlock();
            user.AddAttribute(new Sequential());

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            DistributionArray<Gaussian> wActual = engine.Infer<DistributionArray<Gaussian>>(w);
            Console.WriteLine(wActual);
            Vector mwTrue = Vector.FromArray(-1.178410192370038, 0.514204026416866, 0.533030909628306);
            PositiveDefiniteMatrix vwTrue = new PositiveDefiniteMatrix(new double[,]
                {
                {0.3578619594789516, 0.05893838160003953, -0.1183834842062737,},
                {0.05893838160003954, 0.5424263963482457, -0.08091881706709862,},
                {-0.1183834842062736, -0.08091881706709868, 0.1625333312892799,}
            });
            Gaussian[] wExpected = new Gaussian[3];
            for (int i = 0; i < 3; i++)
            {
                wExpected[i] = new Gaussian(mwTrue[i], vwTrue[i, i]);
            }
            Assert.True(Distribution<double>.Array(wExpected).MaxDiff(wActual) < 1e-6);

            double evExpected = -2.486152518337251;
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-10) < 1e-10);
        }

        // Duplicate MarginalIncrement statements, missing OffsetIndex dependency
        [Fact]
        public void FactorizedBayesPointEvidence3()
        {
            // Bayes Point Machine on the 3-point problem
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double[][] xdata = { new double[] { 0, 0, 1 }, new double[] { 0, 1, 1 }, new double[] { 1, 0, 1 } };
            bool[] ydata = { true, true, false };
            int n = ydata.Length;
            Range user = new Range(n).Named("user");
            VariableArray<bool> y = Variable.Constant(ydata, user).Named("y");
            Range feature = new Range(3).Named("feature");
            var x = Variable.Constant(xdata, user, feature).Named("x");
            Range dummy = new Range(1);
            var w = Variable.Array(Variable.Array<double>(feature), dummy).Named("w");
            w[dummy][feature] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(dummy, feature);
            var dummyIndex = Variable.Array<int>(user).Named("dummyIndex");
            dummyIndex.ObservedValue = Util.ArrayInit(n, i => 0);
            using (Variable.ForEach(user))
            {
                VariableArray<double> product = Variable.Array<double>(feature).Named("product");
                using (var fb = Variable.ForEach(feature))
                {
                    var c = (fb.Index != 0);
                    using (Variable.If(c))
                    {
                        product[feature] = x[user][feature] * w[dummyIndex[user]][feature];
                    }
                    using (Variable.IfNot(c))
                    {
                        product[feature] = x[user][feature] * w[dummyIndex[user]][feature];
                    }
                }
                Variable<double> score = Variable.Sum(product).Named("score");
                y[user] = (score > 0);
            }
            block.CloseBlock();
            user.AddAttribute(new Sequential());

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            var wActual = engine.Infer<GaussianArrayArray>(w)[0];
            Console.WriteLine(wActual);
            Vector mwTrue = Vector.FromArray(-1.178410192370038, 0.514204026416866, 0.533030909628306);
            PositiveDefiniteMatrix vwTrue = new PositiveDefiniteMatrix(new double[,] {
                {0.3578619594789516, 0.05893838160003953, -0.1183834842062737,},
                {0.05893838160003954, 0.5424263963482457, -0.08091881706709862,},
                {-0.1183834842062736, -0.08091881706709868, 0.1625333312892799,}
            });
            Gaussian[] wExpected = new Gaussian[3];
            for (int i = 0; i < 3; i++)
            {
                wExpected[i] = new Gaussian(mwTrue[i], vwTrue[i, i]);
            }
            Assert.True(Distribution<double>.Array(wExpected).MaxDiff(wActual) < 1e-6);

            double evExpected = -2.486152518337251;
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-10) < 1e-10);
        }

        internal void MulticlassBiasTest()
        {
            Gaussian[] priors = new Gaussian[] {
        new Gaussian(1,1),
        new Gaussian(1,2),
        new Gaussian(1,3)
      };
            int nc = priors.Length;
            double[] scores = new double[nc];
            double x = 3;

            int nsamples = 1000000;
            DiscreteEstimator est = new DiscreteEstimator(nc);
            for (int i = 0; i < nsamples; i++)
            {
                for (int c = 0; c < nc; c++)
                {
                    double w = priors[c].Sample();
                    scores[c] = x * w - 0.5 * w * w;
                }
                int argmax = MMath.IndexOfMaximumDouble(scores);
                est.Add(argmax);
            }
            Discrete p = est.GetDistribution(Discrete.Uniform(nc));
            Console.WriteLine(p);
        }

        // new convergence iters: mean 20.276, stddev 10.0003911923484
        //     convergence stmts: mean 486.624, stddev 240.009388616362
        // old convergence iters: mean 23.444, stddev 10.6726221707695
        //     convergence stmts: mean 421.992, stddev 192.107199073851
        // sparse,n=3:
        // new convergence iters: mean 8.98989999999996, stddev 12.3339935945337
        //     convergence stmts: mean 215.757599999999, stddev 296.01584626881
        // old convergence iters: mean 9.83079999999998, stddev 13.5381228890862
        //     convergence stmts: mean 176.9544, stddev 243.686212003551
        // chain,n=4:
        // convergence iters: mean 3.082, stddev 0.274364720764168
        //convergence stmts: mean 73.968, stddev 6.58475329834003
        //convergence iters: mean 2, stddev 0
        //convergence stmts: mean 96, stddev 0
        // chain,n=5:
        // convergence iters: mean 3.468, stddev 0.541272574586963
        // convergence stmts: mean 104.04, stddev 16.2381772376089
        // convergence iters: mean 2, stddev 0
        // convergence stmts: mean 192, stddev 0
        // chain,n=3:
        // convergence iters: mean 2.324, stddev 0.468
        // convergence stmts: mean 41.832, stddev 8.424
        // convergence iters: mean 2, stddev 0
        // convergence stmts: mean 48, stddev 0
        // old schedule requires approx n/2+1 iters
        // new schedule requires 2 iters, but is floor(n/2) times as long
        // so new schedule needs approx twice as many stmts to converge
        // loop,n=3:
        // convergence iters: mean 40.532, stddev 1.37875886216554
        // convergence stmts: mean 729.576000000001, stddev 24.8176595189796
        // convergence iters: mean 21.072, stddev 1.23483440185314
        // convergence stmts: mean 505.728, stddev 29.6360256444753
        // permutations of 3, with #back edges:
        // 123,231,312 = 1
        // 132,321,213 = 2
        // so you'd expect to use 1.5 times more iterations on average
        // why is it taking twice as many iters? because you need both a clockwise and anticlockwise cycle around the loop, and one of these 
        // will always have 2 back edges
        // loop,n=4:
        // convergence iters: mean 35.9379999999999, stddev 7.91897442854819
        // convergence stmts: mean 862.511999999997, stddev 190.055386285156
        // convergence iters: mean 15.664, stddev 0.873558240760168
        // convergence stmts: mean 751.871999999999, stddev 41.9307955564881
        // loop,n=5:
        // convergence iters: mean 38.332, stddev 3.40584438869424
        // convergence stmts: mean 1149.96, stddev 102.175331660827
        // convergence iters: mean 13.284, stddev 0.710875516528739
        // convergence stmts: mean 1275.264, stddev 68.2440495867589
        internal void FactorizedRegression()
        {
            double[][] xdata;
            double[] ydata;
            int n = 3;
            int d = n;
            double noiseVariance = 0.1 * 3;
            var model = new FactorizedRegressionUnrolled(n, d);
            MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
            for (int trial = 0; trial < 1000; trial++)
            {
                model.GenerateData(n, d, System.Math.Sqrt(noiseVariance), out xdata, out ydata);
                model.SetData(xdata, ydata, noiseVariance);
                int niters = model.GetNumberOfIterations(50);
                mva.Add((double)niters);
            }
            Console.WriteLine("convergence iters: mean {0}, stddev {1}", mva.Mean, System.Math.Sqrt(mva.Variance));
            int count = Microsoft.ML.Probabilistic.Compiler.Transforms.SchedulingTransform.LastScheduleLength;
            Console.WriteLine("convergence stmts: mean {0}, stddev {1}", mva.Mean * count, System.Math.Sqrt(mva.Variance) * count);
        }
        public class FactorizedRegressionUnrolled
        {
            public VariableArray<double> w;
            public Variable<double>[] y;
            public VariableArray<double>[] x;
            public Variable<double> noise;
            public InferenceEngine engine = new InferenceEngine();

            public FactorizedRegressionUnrolled(int n, int d)
            {
                y = new Variable<double>[n];
                x = new VariableArray<double>[n];
                Range feature = new Range(d);
                w = Variable.Array<double>(feature).Named("w");
                w[feature] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(feature);
                noise = Variable.New<double>().Named("noise");
                for (int user = 0; user < n; user++)
                {
                    x[user] = Variable.Array<double>(feature).Named("x" + user);
                    var wDamped = Variable.Array<double>(feature);
                    wDamped[feature] = Variable<double>.Factor(Damp.Backward<double>, w[feature], 0.5);
                    VariableArray<double> product = Variable.Array<double>(feature).Named("product" + user);
                    product[feature] = x[user][feature] * wDamped[feature];
                    Variable<double> score = Variable.Sum(product).Named("score" + user);
                    y[user] = Variable.GaussianFromMeanAndVariance(score, noise).Named("y" + user);
                }
                w.AddAttribute(new DivideMessages(false));
                engine.OptimiseForVariables = new IVariable[] { w };
            }
            public void GenerateData(int n, int d, double noiseStd, out double[][] xdata, out double[] ydata)
            {
                xdata = new double[n][];
                ydata = new double[n];
                double[] w = new double[d];
                for (int j = 0; j < d; j++)
                {
                    w[j] = Rand.Normal();
                }
                for (int i = 0; i < n; i++)
                {
                    xdata[i] = new double[d];
                    for (int j = 0; j < d; j++)
                    {
                        //xdata[i][j] = Rand.Normal()*Rand.Normal();
                        //xdata[i][j] = (Rand.Double() < 1.0/n) ? 0.0 : 1.0;
                    }
                }
                if (true)
                {
                    int[] perm = Rand.Perm(n);
                    //perm = new int[] { 0, 2, 1 };
                    for (int i = 0; i < n; i++)
                    {
                        for (int j = 0; j < d; j++)
                        {
                            // chain
                            //xdata[perm[i]][j] = (j==i || j==i+1) ? 1.0 : 0.0;
                            // loop
                            xdata[perm[i]][j] = (j == i || j == (i + 1) % n) ? 1.0 : 0.0;
                        }
                    }
                    //xdata[0][0] = 1;
                    //xdata[0][1] = 1;
                    //xdata[1][1] = 1;
                    //xdata[1][2] = 1;
                    //xdata[2][2] = 1;
                    //xdata[2][3] = 1;
                    //xdata[3][3] = 1;
                    //xdata[3][4] = 1;
                }
                for (int i = 0; i < n; i++)
                {
                    double score = 0;
                    for (int j = 0; j < d; j++)
                    {
                        score += xdata[i][j] * w[j];
                    }
                    ydata[i] = score + Rand.Normal() * noiseStd;
                }
            }
            public void SetData(double[][] xdata, double[] ydata, double noiseVariance)
            {
                this.noise.ObservedValue = noiseVariance;
                for (int i = 0; i < this.y.Length; i++)
                {
                    this.x[i].ObservedValue = xdata[i];
                    this.y[i].ObservedValue = ydata[i];
                }
            }
            public int GetNumberOfIterations(int maxIter)
            {
                engine.ShowProgress = false;
                List<Diffable> wResults = new List<Diffable>();
                for (int iter = 1; iter <= maxIter; iter++)
                {
                    engine.NumberOfIterations = iter;
                    var wActual = engine.Infer<IReadOnlyList<Gaussian>>(w);
                    if (wResults.Count > 10)
                    {
                        var wPrev = wResults[wResults.Count - 1];
                        double diff = wPrev.MaxDiff(wActual);
                        if (diff < 1e-10 || diff > 1e10)
                            break;
                    }
                    wResults.Add((Diffable)wActual);
                    //Trace.WriteLine($"{iter} {wActual[0]}");
                }
                var wLast = wResults[wResults.Count - 1];
                //Console.WriteLine(wLast);
                for (int iter = 0; iter < wResults.Count - 1; iter++)
                {
                    var wActual = wResults[iter];
                    double diff = wLast.MaxDiff(wActual);
                    if (diff < 1e-8)
                        return iter + 1;
                    //Console.WriteLine(diff);
                }
                return maxIter;
            }
        }

        public class FactorizedRegressionModel
        {
            public VariableArray<double> w;
            public VariableArray<double> y;
            public VariableArray<double> offset;
            public VariableArray<VariableArray<double>, double[][]> x;
            public Variable<double> noise;
            public InferenceEngine engine = new InferenceEngine();

            public FactorizedRegressionModel(int n, int d, bool useOffset = false)
            {
                // item does not have a Sequential attribute, therefore it will use a parallel schedule.
                Range item = new Range(n);
                Range feature = new Range(d);
                y = Variable.Array<double>(item).Named("y");
                x = Variable.Array(Variable.Array<double>(feature), item).Named("x");
                w = Variable.Array<double>(feature).Named("w");
                w[feature] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(feature);
                noise = Variable.New<double>().Named("noise");
                offset = Variable.Observed(default(double[]), item).Named(nameof(offset));
                var wSum = Variable.Sum(w).Named("wSum");
                var wAverage = wSum / d;
                wAverage.Name = nameof(wAverage);
                using (Variable.ForEach(item))
                {
                    var wDamped = Variable.Array<double>(feature);
                    wDamped[feature] = Variable<double>.Factor(Damp.Backward<double>, w[feature], 1.0);
                    VariableArray<double> product = Variable.Array<double>(feature).Named("product");
                    product[feature] = x[item][feature] * wDamped[feature];
                    Variable<double> score = Variable.Sum(product).Named("score");
                    if(useOffset)
                    {
                        score = (score + offset[item] * wAverage).Named("score2");
                    }
                    y[item] = Variable.GaussianFromMeanAndVariance(score, noise);
                }
                //w.AddAttribute(new DivideMessages(false));
                engine.OptimiseForVariables = new IVariable[] { w };
            }
            public void SetData(double[][] xdata, double[] ydata, double noiseVariance, double[] offsetData = null)
            {
                this.noise.ObservedValue = noiseVariance;
                this.x.ObservedValue = xdata;
                this.y.ObservedValue = ydata;
                if (offsetData != null) this.offset.ObservedValue = offsetData;
            }
            public int GetNumberOfIterations(int maxIter)
            {
                engine.ShowProgress = false;
                List<Diffable> wResults = new List<Diffable>();
                for (int iter = 1; iter <= maxIter; iter++)
                {
                    engine.NumberOfIterations = iter;
                    var wActual = engine.Infer<IReadOnlyList<Gaussian>>(w);
                    if (wResults.Count > 10)
                    {
                        var wPrev = wResults[wResults.Count - 1];
                        double diff = wPrev.MaxDiff(wActual);
                        if (diff < 1e-10 || diff > 1e10)
                            break;
                    }
                    wResults.Add((Diffable)wActual);
                    //Trace.WriteLine($"{iter} {wActual[0]}");
                }
                var wLast = wResults[wResults.Count - 1];
                //Console.WriteLine(wLast);
                for (int iter = 0; iter < wResults.Count-1; iter++)
                {
                    var wActual = wResults[iter];
                    double diff = wLast.MaxDiff(wActual);
                    if (diff < 1e-8)
                        return iter + 1;
                    //Console.WriteLine(diff);
                }
                return maxIter;
            }
        }

        /// <summary>
        /// Test a case where EP does not converge.
        /// </summary>
        internal void FactorizedRegression2()
        {
            int n = 6;
            int d = 6;
            double pm = 1;
            //model.SetData(xdata, ydata, noiseVariance);
            //int niters = model.GetNumberOfIterations();
            //Console.WriteLine(niters);

            bool useOffset = true;
            //var model = new FactorizedRegressionUnrolled(n, d);
            var model = new FactorizedRegressionModel(n, d, useOffset);
            //var ems = EpTests.linspace(0, 1.0/(n-1), 40);
            //var ems = EpTests.linspace(0, 2.0, 40);
            var ems = EpTests.linspace(-1.0, 0, 40);
            foreach (var em in ems)
            {
                int h = n / 2;
                double[][] xdata = Util.ArrayInit(n, i => Util.ArrayInit(d, j =>
                    (i == j) ? pm : ((i < h && j >= h) || (i >= h && j < h)) ? em : 0));
                // FFA case
                xdata = Util.ArrayInit(n, i => Util.ArrayInit(d, j =>
                    (i == j) ? pm : em));
                //xdata = Util.ArrayInit(n, i => Util.ArrayInit(d, j =>
                //    (i == j) ? pm : (i == j-1) ? em : 0));
                double[] offsetData = null;
                if(useOffset)
                {
                    xdata = Util.ArrayInit(n, i => Util.ArrayInit(d, j =>
                        (i == j) ? (pm - em) : 0));
                    offsetData = Util.ArrayInit(n, i => em*d);
                    Console.WriteLine(Vector.FromArray(offsetData));
                }
                Matrix X = new Matrix(GetArray(xdata));
                bool orthogonalize = false;
                if (orthogonalize)
                {
                    Matrix Xt = X.Transpose();
                    Matrix right = new Matrix(n, n);
                    right.SetToRightSingularVectors(Xt);
                    X = Xt.Transpose();
                    xdata = GetJaggedArray(X);
                }
                Console.WriteLine(X);

                double[] ydata = Util.ArrayInit(n, i => 1.0);
                var highestStablePrecision = 0.0;
                var precs = EpTests.linspace(1, 200, 20);
                foreach (var prec in precs)
                {
                    double noiseVariance = 1.0 / prec;
                    model.SetData(xdata, ydata, noiseVariance, offsetData);
                    int maxIter = 20000;
                    int niters = model.GetNumberOfIterations(maxIter);
                    bool isUnstable = (niters == maxIter);
                    //Trace.WriteLine($"em={em} prec={prec} {isUnstable}");
                    if (!isUnstable)
                        highestStablePrecision = prec;
                }
                Trace.WriteLine($"em={em} highest stable prec={highestStablePrecision}");
            }
        }

        public static double[,] GetArray(double[][] array)
        {
            double[,] array2d = new double[array.Length, array[0].Length];
            for (int i = 0; i < array.Length; i++)
            {
                for (int j = 0; j < array[i].Length; j++)
                {
                    array2d[i, j] = array[i][j];
                }
            }
            return array2d;
        }

        public static double[][] GetJaggedArray(Matrix A)
        {
            return Util.ArrayInit(A.Rows, i => Util.ArrayInit(A.Cols, j => A[i, j]));
        }

        public static PositiveDefiniteMatrix GetCorrelationCoefficients(PositiveDefiniteMatrix A)
        {
            Vector d = Vector.FromArray(Util.ArrayInit(A.Rows, i => 1 / System.Math.Sqrt(A[i, i])));
            A.ScaleRows(d).ScaleCols(d);
            for (int i = 0; i < A.Rows; i++)
            {
                A[i, i] -= 1;
                for (int j = 0; j < A.Rows; j++)
                {
                    A[i, j] = System.Math.Abs(A[i, j]);
                }
            }
            return A;
        }

        [Fact]
        public void BayesPointEvidence2()
        {
            // this issue was found by Vincent Tan
            double[] incomes = { 63, 16, 28, 55, 22, 20 };
            double[] ages = { 38, 23, 40, 27, 18, 40 };
            bool[] willBuy = { true, false, true, true, false, false };

            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock model = Variable.If(evidence);

            // Create target y
            VariableArray<bool> y = Variable.Constant(willBuy).Named("y");
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(3),
                PositiveDefiniteMatrix.Identity(3))).Named("w");
            // Create x vector, augmented by 1
            Range j = y.Range.Named("j");
            Vector[] xdata = new Vector[incomes.Length];
            for (int i = 0; i < xdata.Length; i++)
                xdata[i] = Vector.FromArray(incomes[i], ages[i], 1);
            VariableArray<Vector> x = Variable.Constant(xdata, j).Named("x");

            // Bayes Point Machine
            Variable<double> wInnerX = Variable.InnerProduct(w, x[j]).Named("wInnerX");
            y[j] = Variable.IsPositive(Variable.GaussianFromMeanAndPrecision(wInnerX, 1.0));

            model.CloseBlock();

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            // test passes with 20 iters but not 40 iters
            engine.NumberOfIterations = 40;
            VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
            Console.WriteLine("Dist over w=\n" + wPosterior);

            VectorGaussian wExpected = new VectorGaussian(Vector.FromArray(new double[]
                {
                    0.109666345041522, -0.062557928550261, -0.827699303548466
                }),
                                                          new PositiveDefiniteMatrix(new double[,]
                                                              {
                    { 0.004203498829365, -0.002901565701299, -0.012119192593608},
                    {-0.002901565701299,  0.003226764582933, -0.016479796269154},
                    {-0.012119192593608, -0.016479796269154,  0.843335367330159}
                }));
            Assert.True(wExpected.MaxDiff(wPosterior) < 1e-4);

            double logEvidence = engine.Infer<Bernoulli>(evidence).LogOdds;
            double evExpected = -9.477096220081785;
            Console.WriteLine("evidence = {0} (should be {1})", logEvidence, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, logEvidence) < 1e-4);
        }


        [Fact]
        public void BayesPointEvidence()
        {
            // Bayes Point Machine on the 3-point problem
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double[,] xdata = { { 0, 0 }, { 0, 1 }, { 1, 0 } };
            bool[] ydata = { true, true, false };
            Variable<Vector> w = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3)).Named("w");
            int n = ydata.Length;
            Variable<Vector>[] x = new Variable<Vector>[n];
            Variable<bool>[] y = new Variable<bool>[n];
            for (int i = 0; i < n; i++)
            {
                x[i] = Variable.Constant(Vector.FromArray(xdata[i, 0], xdata[i, 1], 1)).Named("x" + i);
                Variable<double> h = Variable.InnerProduct(x[i], w).Named("h" + i);
                y[i] = Variable.Constant(ydata[i]).Named("y" + i);
                if (y[i].ObservedValue)
                {
                    Variable.ConstrainPositive(h);
                }
                else
                {
                    Variable.ConstrainFalse(Variable.IsPositive(h));
                }
            }
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            VectorGaussian wActual = engine.Infer<VectorGaussian>(w);
            //Console.WriteLine(wActual);
            Vector mwTrue = Vector.FromArray(-1.178410192370038, 0.514204026416866, 0.533030909628306);
            PositiveDefiniteMatrix vwTrue = new PositiveDefiniteMatrix(new double[,]
                {
                {0.3578619594789516, 0.05893838160003953, -0.1183834842062737,},
                {0.05893838160003954, 0.5424263963482457, -0.08091881706709862,},
                {-0.1183834842062736, -0.08091881706709868, 0.1625333312892799,}
            });
            VectorGaussian wExpected = new VectorGaussian(mwTrue, vwTrue);
            Assert.True(wExpected.MaxDiff(wActual) < 1e-6);

            double evExpected = -2.486152518337251;
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-10) < 1e-10);
        }

        [Fact]
        public void BayesPointProbitEvidence()
        {
            // probit Bayes Point Machine on the 6-point problem
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double[,] xdata = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 0, 0.5 }, { 1.5, 0 }, { 0.1, 1.5 } };
            bool[] ydata = { true, true, false, true, false, false };
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3))).Named("w");
            int n = ydata.Length;
            Variable<Vector>[] x = new Variable<Vector>[n];
            Variable<bool>[] y = new Variable<bool>[n];
            double noise = 1.0;
            for (int i = 0; i < n; i++)
            {
                x[i] = Variable.Constant(Vector.FromArray(xdata[i, 0], xdata[i, 1], 1)).Named("x" + i);
                Variable<double> h = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(x[i], w), noise).Named("h" + i);
                y[i] = Variable.Constant(ydata[i]).Named("y" + i);
                if (y[i].ObservedValue)
                {
                    Variable.ConstrainPositive(h);
                }
                else
                {
                    Variable.ConstrainFalse(Variable.IsPositive(h));
                }
            }
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            VectorGaussian wActual = engine.Infer<VectorGaussian>(w);
            Console.WriteLine(wActual);
            Vector mwTrue = Vector.FromArray(-1.077521554289827, -0.2008897723072238, 0.3660247485925804);
            PositiveDefiniteMatrix vwTrue = new PositiveDefiniteMatrix(new double[,]
                {
                {0.5410449066272626,0.07674046058414256,-0.161630781607686},
                {0.07674046058414252,0.4505294996327321,-0.213693905957742},
                    {-0.161630781607686, -0.213693905957742, 0.3821742571834561}
                });
            VectorGaussian wExpected = new VectorGaussian(mwTrue, vwTrue);
            Assert.True(wExpected.MaxDiff(wActual) < 1e-6);

            double evExpected = -4.652691936530638;
            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-10) < 1e-10);
        }

        [Fact]
        public void BayesPointNoisyStepEvidence()
        {
            // noisy-step Bayes Point Machine on the 6-point problem
            Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");
            IfBlock block = Variable.If(evidence);
            double[,] xdata = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 0, 0.5 }, { 1.5, 0 }, { 0.1, 1.5 } };
            bool[] ydata = { true, true, false, true, false, false };
            double errorRate = 0.1;
            Bernoulli matchDist = new Bernoulli(1 - errorRate);
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3))).Named("w");
            int n = ydata.Length;
            Variable<Vector>[] x = new Variable<Vector>[n];
            for (int i = 0; i < n; i++)
            {
#if false
                x[i] = Variable.Constant(Vector.FromArray(xdata[i, 0], xdata[i, 1], 1)).Named("x" + i);
                Variable<bool> yPredicted = (Variable.InnerProduct(x[i], w) > 0.0);
                Variable.ConstrainEqualRandom(yPredicted == ydata[i], matchDist);
#else
                x[i] = Variable.Constant(Vector.FromArray(xdata[i, 0], xdata[i, 1], 1) * (ydata[i] ? 1 : -1)).Named("x" + i);
                Variable<bool> yPredicted = (Variable.InnerProduct(x[i], w) > 0.0);
                Variable.ConstrainEqualRandom(yPredicted, matchDist);
#endif
            }
            block.CloseBlock();

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            VectorGaussian wActual = engine.Infer<VectorGaussian>(w);
            Console.WriteLine(wActual);
            // Vector mwTrue = new DenseVector(-1.3633, -0.4944, 0.6543); // error rate = 0
            // error rate = 0.1
            Vector mwTrue = Vector.FromArray(-1.16220429443624, -0.0338583325319568, 0.5428335422151358);
            PositiveDefiniteMatrix vwTrue = new PositiveDefiniteMatrix(new double[,]
                {
                {0.3730596036503481,0.09642738507267575,-0.1256797047507377},
                {0.09642738507267563,0.7684442285629477,-0.1530917637786459},
                    {-0.1256797047507378, -0.1530917637786461, 0.2119627016754658}
                });
            double tolerance = 1e-6;
            double evExpected = -4.553117742706541;
            if (FactorManager.IsDefaultOperator(typeof(IsPositiveOp_Proper)))
            {
                mwTrue = Vector.FromArray(-1.164, -0.07971, 0.5411);
                vwTrue = new PositiveDefiniteMatrix(new double[,]
                    {
                    {0.3656,  0.09801, -0.1145},
                    {0.09801, 0.5187,  -0.1651},
                    {-0.1145, -0.1651, 0.1929}
                });
                tolerance = 2e-3;
                evExpected = -4.57316358225882;
            }
            VectorGaussian wExpected = new VectorGaussian(mwTrue, vwTrue);
            Assert.True(wExpected.MaxDiff(wActual) < tolerance);

            double evActual = engine.Infer<Bernoulli>(evidence).LogOdds;
            Console.WriteLine("evidence = {0} (should be {1})", evActual, evExpected);
            Assert.True(MMath.AbsDiff(evExpected, evActual, 1e-10) < 1e-6);
        }

        [Fact]
        public void BayesPoint()
        {
            // Bayes Point Machine on the 3-point problem
            double[,] xdata = { { 0, 0 }, { 0, 1 }, { 1, 0 } };
            bool[] ydata = { true, true, false };
            Variable<Vector> w = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3)).Named("w");
            //Variable<Vector> w = Variable.Random(new VectorGaussian(new DenseVector(3), PositiveDefiniteMatrix.Identity(3))).Named("w");
            int n = ydata.Length;
            Variable<Vector>[] x = new Variable<Vector>[n];
            Variable<bool>[] y = new Variable<bool>[n];
            //Variable<double>[] h = new Variable<double>[n];
            for (int i = 0; i < n; i++)
            {
                x[i] = Variable.Constant(Vector.FromArray(xdata[i, 0], xdata[i, 1], 1)).Named("x" + i);
                Variable<double> h = Variable.InnerProduct(x[i], w).Named("h" + i);
                y[i] = Variable.Constant(ydata[i]).Named("y" + i);
                if (y[i].ObservedValue)
                {
                    Variable.ConstrainPositive(h);
                }
                else
                {
                    Variable.ConstrainFalse(Variable.IsPositive(h));
                }
                //Variable.ConstrainEqual(Variable.IsPositive(h[i]),y[i]);
                //y[i].SetTo(Variable.IsPositive(h[i]));
            }

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            VectorGaussian wActual = engine.Infer<VectorGaussian>(w);
            //Console.WriteLine(wActual);
            Vector mwTrue = Vector.FromArray(-1.178410192370038, 0.514204026416866, 0.533030909628306);
            PositiveDefiniteMatrix vwTrue = new PositiveDefiniteMatrix(new double[,]
                {
                {0.3578619594789516, 0.05893838160003953, -0.1183834842062737,},
                {0.05893838160003954, 0.5424263963482457, -0.08091881706709862,},
                {-0.1183834842062736, -0.08091881706709868, 0.1625333312892799,}
            });
            VectorGaussian wExpected = new VectorGaussian(mwTrue, vwTrue);
            Assert.True(wExpected.MaxDiff(wActual) < 1e-6);
        }

        [Fact]
        public void BayesPointProbit()
        {
            // probit Bayes Point Machine on the 6-point problem
            double[,] xdata = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 0, 0.5 }, { 1.5, 0 }, { 0.1, 1.5 } };
            bool[] ydata = { true, true, false, true, false, false };
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3))).Named("w");
            int n = ydata.Length;
            Variable<Vector>[] x = new Variable<Vector>[n];
            Variable<bool>[] y = new Variable<bool>[n];
            double noise = 1.0;
            for (int i = 0; i < n; i++)
            {
                x[i] = Variable.Constant(Vector.FromArray(xdata[i, 0], xdata[i, 1], 1)).Named("x" + i);
                Variable<double> h = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(x[i], w), noise).Named("h" + i);
                y[i] = Variable.Constant(ydata[i]).Named("y" + i);
                if (y[i].ObservedValue)
                {
                    Variable.ConstrainPositive(h);
                }
                else
                {
                    Variable.ConstrainFalse(Variable.IsPositive(h));
                }
            }

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            VectorGaussian wActual = engine.Infer<VectorGaussian>(w);
            Console.WriteLine(wActual);
            Vector mwTrue = Vector.FromArray(-1.077521554289827, -0.2008897723072238, 0.3660247485925804);
            PositiveDefiniteMatrix vwTrue = new PositiveDefiniteMatrix(new double[,]
                {
                    {0.5410449066272626, 0.07674046058414256, -0.161630781607686},
                    {0.07674046058414252, 0.4505294996327321, -0.213693905957742},
                    {-0.161630781607686, -0.213693905957742, 0.3821742571834561}
                });
            VectorGaussian wExpected = new VectorGaussian(mwTrue, vwTrue);
            Assert.True(wExpected.MaxDiff(wActual) < 1e-6);
        }

        [Fact]
        public void BayesPointNoisyStep()
        {
            // noisy-step Bayes Point Machine on the 6-point problem
            double[,] xdata = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 0, 0.5 }, { 1.5, 0 }, { 0.1, 1.5 } };
            bool[] ydata = { true, true, false, true, false, false };
            double errorRate = 0.1;
            Bernoulli matchDist = new Bernoulli(1 - errorRate);
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3))).Named("w");
            int n = ydata.Length;
            Variable<Vector>[] x = new Variable<Vector>[n];
            for (int i = 0; i < n; i++)
            {
#if false
                x[i] = Variable.Constant(Vector.FromArray(xdata[i, 0], xdata[i, 1], 1)).Named("x" + i);
                Variable<bool> yPredicted = (Variable.InnerProduct(x[i], w) > 0.0);
                Variable.ConstrainEqualRandom(yPredicted == ydata[i], matchDist);
#else
                x[i] = Variable.Constant(Vector.FromArray(xdata[i, 0], xdata[i, 1], 1) * (ydata[i] ? 1 : -1)).Named("x" + i);
                Variable<bool> yPredicted = (Variable.InnerProduct(x[i], w) > 0.0);
                Variable.ConstrainEqualRandom(yPredicted, matchDist);
#endif
            }

            InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
            engine.NumberOfIterations = 100;
            VectorGaussian wActual = engine.Infer<VectorGaussian>(w);
            Console.WriteLine("w = ");
            Console.WriteLine(wActual);
            // Vector mwTrue = new DenseVector(-1.3633, -0.4944, 0.6543); // error rate = 0
            // error rate = 0.1
            Vector mwTrue = Vector.FromArray(-1.16220429443624, -0.0338583325319568, 0.5428335422151358);
            PositiveDefiniteMatrix vwTrue = new PositiveDefiniteMatrix(new double[,]
                {
                    {0.3730596036503481, 0.09642738507267575, -0.1256797047507377},
                    {0.09642738507267563, 0.7684442285629477, -0.1530917637786459},
                    {-0.1256797047507378, -0.1530917637786461, 0.2119627016754658}
                });
            double tolerance = 1e-6;
            if (FactorManager.IsDefaultOperator(typeof(IsPositiveOp_Proper)))
            {
                mwTrue = Vector.FromArray(-1.164, -0.07971, 0.5411);
                vwTrue = new PositiveDefiniteMatrix(new double[,]
                    {
                    {0.3656,  0.09801, -0.1145},
                    {0.09801, 0.5187,  -0.1651},
                    {-0.1145, -0.1651, 0.1929}
                });
                tolerance = 2e-3;
            }
            VectorGaussian wExpected = new VectorGaussian(mwTrue, vwTrue);
            Console.WriteLine(" should be");
            Console.WriteLine(wExpected);
            Console.WriteLine("MaxDiff = {0}", wExpected.MaxDiff(wActual));
            Assert.True(wExpected.MaxDiff(wActual) < tolerance);
        }


        //[Fact]
        internal void SpeedTest()
        {
            BayesPoint();
            Stopwatch watch = new Stopwatch();
            watch.Start();
            BayesPoint();
            watch.Stop();
            // Infer1: 20ms
            // Infer2 21 Nov 2007: 366ms
            // Infer2 26 Feb 2008: 900ms
            // Infer2 13 Dec 2010: 150ms
            Console.WriteLine("stopwatch: " + watch.ElapsedMilliseconds + "ms");
            //Assert.True(watch.ElapsedMilliseconds < 200);
        }

        [Fact]
        public void BPMSparseTest()
        {
            int nClass = 2;
            int nItems = 10;
            int totalFeatures = 4;

            GenerateData gd = new GenerateData(nClass, totalFeatures);

            int[][][] indices;
            double[][][] x = gd.Sample(nItems, out indices);

            Console.WriteLine("------------- bpm sparse--------------");
            BPMSparseTest bpm = new BPMSparseTest(nClass, totalFeatures);
            bpm.Train(indices, x);

        }
    }

    //-----------------------------------------------------------------------------------------

    internal class BPMSparseVarsForTrain
    {
        public VariableArray<double>[] w;
        public BPMDataVars[] dataVars;  // curUserTrain[k]: variables for kth component
        public InferenceEngine ie;
    }


    public class BPMDataVars
    {
        public VariableArray<VariableArray<double>, double[][]> xValues;
        public VariableArray<VariableArray<int>, int[][]> xIndices;
        public Variable<int> nItem;
        public Range item;
        public VariableArray<int> xValueCount;
        public BPMDataVars()
        {
        }

        public BPMDataVars(Variable<int> nUser, Range user, VariableArray<VariableArray<int>, int[][]> xIndices, VariableArray<int> xValueCount,
                           VariableArray<VariableArray<double>, double[][]> xValues)
        {
            this.nItem = nUser;
            this.item = user;
            this.xValueCount = xValueCount;
            this.xValues = xValues;
            this.xIndices = xIndices;
        }
    }


    internal class BPMSparseTest
    {
        private BPMSparseVarsForTrain trainModel;
        private int nComponents, nFeatures;
        private Range feature;

        public BPMSparseTest(int nClass, int featureCount)
        {
            nComponents = nClass;
            nFeatures = featureCount;
            feature = new Range(nFeatures).Named("feature");
            trainModel = SpecifyTrainModel("_train");
        }

        private BPMSparseVarsForTrain SpecifyTrainModel(string s)
        {
            VariableArray<VariableArray<double>, double[][]>[] xValues = new VariableArray<VariableArray<double>, double[][]>[nComponents];
            VariableArray<VariableArray<int>, int[][]>[] xIndices = new VariableArray<VariableArray<int>, int[][]>[nComponents];
            VariableArray<int>[] xValueCount = new VariableArray<int>[nComponents];

            Range[] itemFeature = new Range[nComponents];
            Variable<int>[] nItem = new Variable<int>[nComponents];
            Range[] item = new Range[nComponents];

            VariableArray<double>[] w = new VariableArray<double>[nComponents];
            for (int c = 0; c < nComponents; c++)
            {
                w[c] = Variable.Array<double>(feature).Named("w" + c + s);
                w[c][feature] = Variable.Random(new Gaussian(0, 1)).ForEach(feature);
            }
            for (int c = 0; c < nComponents; c++)
            {
                nItem[c] = Variable.New<int>().Named("nItem" + "_" + c + s);
                item[c] = new Range(nItem[c]).Named("item" + "_" + c + s);
                xValueCount[c] = Variable.Array<int>(item[c]).Named("xCount" + "_" + c + s);
                itemFeature[c] = new Range(xValueCount[c][item[c]]).Named("itemFeature" + "_" + c + s);
                xValues[c] = Variable.Array(Variable.Array<double>(itemFeature[c]), item[c]).Named("xValues_" + c + s);
                xIndices[c] = Variable.Array(Variable.Array<int>(itemFeature[c]), item[c]).Named("xIndices_" + c + s);

                using (Variable.ForEach(item[c]))
                {
                    Variable<double>[] score = ComputeClassScores(w, xValues[c][item[c]], xIndices[c][item[c]], itemFeature[c], "_" + c + s);
                    ConstrainArgMax(c, score);
                }
            }

            BPMSparseVarsForTrain bpmVar = new BPMSparseVarsForTrain();
            bpmVar.ie = new InferenceEngine();
            bpmVar.dataVars = new BPMDataVars[nComponents];
            bpmVar.w = w;
            for (int c = 0; c < nComponents; c++)
            {
                bpmVar.dataVars[c] = new BPMDataVars(nItem[c], item[c], xIndices[c], xValueCount[c], xValues[c]);
            }
            return bpmVar;
        }


        private Variable<double>[] ComputeClassScores(VariableArray<double>[] w, VariableArray<double>[] xValues, VariableArray<int>[] xIndices, Range[] itemFeature, int curK,
                                                      string s)
        {
            return ComputeClassScores(w, xValues[curK], xIndices[curK], itemFeature[curK], s);
        }


        private Variable<double>[] ComputeClassScores(VariableArray<double>[] w, VariableArray<double> xValues, VariableArray<int> xIndices, Range itemFeature, string s)
        {
            Variable<double>[] score = new Variable<double>[nComponents];
            for (int c = 0; c < nComponents; c++)
            {
                //VariableArray<double> wSparse = Variable.GetItems(w[c], xIndices);
                VariableArray<double> wSparse = Variable.Subarray<double>(w[c], xIndices);
                VariableArray<double> product = Variable.Array<double>(itemFeature).Named("product_" + c + s);
                product[itemFeature] = xValues[itemFeature] * wSparse[itemFeature];
                score[c] = Variable.Sum(product).Named("score_" + c + s);
            }
            return score;
        }


        public void Train(int[][][] xIndicesData, double[][][] xValuesData)
        {
            AssignGivensForTrain(trainModel, xIndicesData, xValuesData);

            for (int c = 0; c < nComponents; c++)
            {
                Console.WriteLine((trainModel.ie).Infer(trainModel.w[c]));
            }
        }

        private static void SetDataVar(ref BPMDataVars dataVar, int[][] xIndicesData, double[][] xValuesData)
        {
            dataVar.nItem.ObservedValue = xIndicesData.Length;
            dataVar.xValues.ObservedValue = xValuesData;
            dataVar.xIndices.ObservedValue = xIndicesData;

            int[] xValueCountData = new int[xIndicesData.Length];
            for (int i = 0; i < xIndicesData.Length; i++)
            {
                xValueCountData[i] = xIndicesData[i].Length;
            }
            dataVar.xValueCount.ObservedValue = xValueCountData;
        }


        public void AssignGivensForTrain(BPMSparseVarsForTrain bpm, int[][][] xIndicesData, double[][][] xValuesData)
        {
            for (int c = 0; c < nComponents; c++)
            {
                SetDataVar(ref bpm.dataVars[c], xIndicesData[c], xValuesData[c]);
            }
        }


        private void ConstrainMaximum(Variable<int> ytrain, Variable<double>[] score)
        {
            for (int c = 0; c < nComponents; c++)
            {
                using (Variable.Case(ytrain, c))
                {
                    ConstrainArgMax(c, score);
                }
            }
        }

        private void ConstrainArgMax(int argmax, Variable<double>[] score)
        {
            for (int c = 0; c < score.Length; c++)
            {
                if (c != argmax)
                    Variable.ConstrainPositive(score[argmax] - score[c]);
            }
        }
    }

    internal class GenerateData
    {
        public int nClass;
        public int nFeatures;
        public Vector[] w;

        public GenerateData(int nClass, int nFeatures)
        {
            this.nClass = nClass;
            this.nFeatures = nFeatures;
            SetW();
        }

        public Vector[][] Sample(int totalItems)
        {
            List<Vector>[] samples = new List<Vector>[nClass];
            for (int c = 0; c < nClass; c++)
            {
                samples[c] = new List<Vector>();
            }
            for (int t = 0; t < totalItems; t++)
            {
                Vector y = VectorGaussian.Sample(Vector.Zero(nFeatures), PositiveDefiniteMatrix.Identity(nFeatures));
                double[] score = new double[w.Length];
                for (int i = 0; i < w.Length; i++)
                {
                    score[i] = y.Inner(w[i]);
                }
                int id = FindMaxIndex(score);
                samples[id].Add(y);
            }
            Vector[][] x = new Vector[nClass][];
            for (int c = 0; c < nClass; c++)
            {
                x[c] = new Vector[samples[c].Count];
                for (int j = 0; j < samples[c].Count; j++)
                {
                    x[c][j] = samples[c][j];
                }
            }
            return x;
        }

        public double[][][] Sample(int totalItems, out int[][][] indices)
        {
            Vector[][] sampleVector = Sample(totalItems);

            double[][][] x = new double[nClass][][];
            indices = new int[nClass][][];
            for (int i = 0; i < nClass; i++)
            {
                x[i] = new double[sampleVector[i].Length][];
                indices[i] = new int[sampleVector[i].Length][];

                for (int j = 0; j < sampleVector[i].Length; j++)
                {
                    Vector q = sampleVector[i][j];
                    x[i][j] = new double[q.Count];
                    indices[i][j] = new int[q.Count];
                    for (int k = 0; k < q.Count; k++)
                    {
                        indices[i][j][k] = k;
                        x[i][j][k] = q[k];
                    }
                }
            }
            return x;
        }


        private int FindMaxIndex(double[] score)
        {
            int id;
            if (score.Length == 1)
                id = (score[0] > 0) ? 0 : 1;
            else
            {
                id = 0;
                for (int j = 1; j < score.Length; j++)
                {
                    id = (score[j] > score[id]) ? j : id;
                }
            }
            return id;
        }

        private void SetW()
        {
            w = new Vector[nClass == 2 ? 1 : nClass];
            for (int c = 0; c < w.Length; c++)
            {
                w[c] = VectorGaussian.Sample(Vector.Zero(nFeatures), PositiveDefiniteMatrix.Identity(nFeatures));
            }
        }
    }

    //-----------------------------------------------------------------------------------------

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif
}
