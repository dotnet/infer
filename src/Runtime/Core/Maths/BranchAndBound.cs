// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Math
{
    using System;
    using System.Diagnostics;
    using Microsoft.ML.Probabilistic.Collections;

    public class BranchAndBound
    {
        protected class QueueNode : IComparable<QueueNode>
        {
            public Region Region;
            public double UpperBound;

            public int CompareTo(QueueNode other)
            {
                return other.UpperBound.CompareTo(UpperBound);
            }
        }

        public static bool Debug;
        public enum SplitMethod
        {
            Random,
            Cycle,
            LowestChild
        }
        static SplitMethod splitMethod = SplitMethod.Cycle;
        static MeanVarianceAccumulator timeAccumulator = new MeanVarianceAccumulator();

        /// <summary>
        /// Finds the input that maximizes a function.
        /// </summary>
        /// <param name="bounds">Bounds for the search.</param>
        /// <param name="Evaluate">The function to maximize.</param>
        /// <param name="GetUpperBound">Returns an upper bound to the function in a region.  Need not be tight, but must become tight as the region shrinks.</param>
        /// <param name="xTolerance">Allowable relative error in the solution on any dimension.  Must be greater than zero.</param>
        /// <returns>A Vector close to the global maximum of the function.</returns>
        public static Vector Search(Region bounds, Func<Vector, double> Evaluate, Func<Region, double> GetUpperBound, double xTolerance = 1e-4)
        {
            if (xTolerance <= 0) throw new ArgumentException($"xTolerance <= 0");
            int dim = bounds.Lower.Count;
            if (dim == 0) return Vector.Zero(dim);
            PriorityQueue<QueueNode> queue = new PriorityQueue<QueueNode>();
            double lowerBound = double.NegativeInfinity;
            Vector argmax = bounds.GetMidpoint();
            Action<Region> addRegion = delegate (Region region)
            {
                Stopwatch watch = Stopwatch.StartNew();
                double upperBoundF = GetUpperBound(region);
                watch.Stop();
                if (Debug)
                {
                    if (Debug && timeAccumulator.Count > 10 && watch.ElapsedMilliseconds > timeAccumulator.Mean + 4 * Math.Sqrt(timeAccumulator.Variance))
                        Trace.WriteLine($"GetUpperBound took {watch.ElapsedMilliseconds}ms");
                    timeAccumulator.Add(watch.ElapsedMilliseconds);
                }
                double upperBound = upperBoundF;
                if (upperBound > lowerBound)
                {
                    if (Debug)
                        Trace.WriteLine($"added region {region} with upperBoundF = {upperBoundF}");
                    QueueNode node = new QueueNode()
                    {
                        Region = region,
                        UpperBound = upperBound
                    };
                    queue.Add(node);
                }
                else if (Debug)
                    Trace.WriteLine($"rejected region {region} with upperBound {upperBound} <= lowerBound {lowerBound}");
            };
            addRegion(bounds);
            int splitDim = 0;
            while (queue.Count > 0)
            {
                var node = queue.ExtractMinimum();  // gets the node with highest upper bound
                if (node.UpperBound <= lowerBound)
                    continue;
                Region region = node.Region;
                // compute the lower bound
                Vector midpoint = region.GetMidpoint();
                Stopwatch watch = Stopwatch.StartNew();
                double nodeLowerBound = Evaluate(midpoint);
                watch.Stop();
                if (Debug)
                {
                    if (Debug && timeAccumulator.Count > 10 && watch.ElapsedMilliseconds > timeAccumulator.Mean + 4 * Math.Sqrt(timeAccumulator.Variance))
                        Trace.WriteLine($"Evaluate took {watch.ElapsedMilliseconds}ms");
                    timeAccumulator.Add(watch.ElapsedMilliseconds);
                }
                if (Debug)
                    Trace.WriteLine($"expanding region {region} with upper bound = {node.UpperBound} lower bound = {nodeLowerBound}");
                if (nodeLowerBound > node.UpperBound) throw new Exception("nodeLowerBound > node.UpperBound");
                if (nodeLowerBound > lowerBound)
                {
                    argmax = midpoint;
                    lowerBound = nodeLowerBound;
                }

                if (splitMethod == SplitMethod.Cycle || dim == 1)
                {
                    // cycle the split dimension
                    splitDim++;
                    if (splitDim == dim)
                        splitDim = 0;
                }
                else if (splitMethod == SplitMethod.Random)
                {
                    splitDim = Rand.Int(dim);
                }
                else if (splitMethod == SplitMethod.LowestChild)
                {
                    // find the best dimension to split on
                    int bestSplitDim = 0;
                    double bestSplitScore = double.PositiveInfinity;

                    for (int i = 0; i < dim; i++)
                    {
                        if (region.Upper[i] - region.Lower[i] < xTolerance)
                            continue;
                        int splitDim2 = i;
                        double splitValue2 = midpoint[i];
                        Region leftRegion2 = new Region(region);
                        leftRegion2.Upper[splitDim2] = splitValue2;
                        double leftUpperBound = GetUpperBound(leftRegion2);
                        Region rightRegion2 = new Region(region);
                        rightRegion2.Lower[splitDim2] = splitValue2;
                        double rightUpperBound = GetUpperBound(rightRegion2);
                        double score = Math.Min(leftUpperBound, rightUpperBound);
                        if (score < bestSplitScore)
                        {
                            bestSplitScore = score;
                            bestSplitDim = i;
                        }
                    }
                    if (bestSplitScore < double.MaxValue)
                    {
                        //Console.WriteLine("bestSplitValue = {0} (bestSplitScore = {1})", bestSplitValue, bestSplitScore);
                        splitDim = bestSplitDim;
                        //Console.WriteLine("argmaxGumbel = {0}  splitDim = {1}", argmaxGumbel, splitDim);
                    }
                    else
                    {
                        // use the previous splitDim
                    }
                }

                // find a dimension to split on
                bool foundSplit = false;
                for (int i = 0; i < dim; i++)
                {
                    if (MMath.AbsDiff(region.Upper[splitDim], region.Lower[splitDim], 1e-10) < xTolerance)
                    {
                        if (++splitDim == dim)
                            splitDim = 0;
                    }
                    else
                    {
                        foundSplit = true;
                        break;
                    }
                }
                if (!foundSplit)
                {
                    break;
                }

                // split the node
                double splitValue = midpoint[splitDim];
                if (region.Upper[splitDim] != splitValue)
                {
                    Region leftRegion = new Region(region);
                    leftRegion.Upper[splitDim] = splitValue;
                    addRegion(leftRegion);
                }
                if (region.Lower[splitDim] != splitValue)
                {
                    Region rightRegion = new Region(region);
                    rightRegion.Lower[splitDim] = splitValue;
                    addRegion(rightRegion);
                }
            }
            return argmax;
        }
    }
}
