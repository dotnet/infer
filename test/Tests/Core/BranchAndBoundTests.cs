// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class BranchAndBoundTests
    {
        // Developer test
        public void BranchAndBound_IgnoresIrrelevantDimensions()
        {
            for (int relevantCount = 1; relevantCount <= 1; relevantCount++)
            {
                for (int irrelevantCount = 0; irrelevantCount <= 4; irrelevantCount++)
                {
                    Trace.WriteLine($"relevantCount = {relevantCount}, irrelevantCount = {irrelevantCount}");
                    BranchAndBound_IgnoresIrrelevantDimensions(relevantCount, irrelevantCount);
                }
            }
        }

        public void BranchAndBound_IgnoresIrrelevantDimensions(int relevantCount, int irrelevantCount)
        {
            Region bounds = new Region(relevantCount + irrelevantCount);
            for (int i = 0; i < bounds.Lower.Count; i++)
            {
                bounds.Lower[i] = -1000;
                bounds.Upper[i] = 1000;
            }
            Vector slope = Vector.Zero(relevantCount + irrelevantCount);
            for (int i = 0; i < slope.Count; i++)
            {
                if (i < relevantCount) slope[i] = i + 1;
                else slope[i] = 0;
            }
            Func<Vector, double> Evaluate = vector => vector.Inner(slope);
            Func<Region, double> GetUpperBound = region => region.Upper.Inner(slope);
            //BranchAndBound.Debug = true;
            Vector best = BranchAndBound.Search(bounds, Evaluate, GetUpperBound);
            Trace.WriteLine($"best = {best}");
        }
    }
}
