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
            public readonly Region Region;
            public readonly double UpperBound;
            public readonly int SplitDim;

            public QueueNode(Region region, double upperBound, int splitDim)
            {
                this.Region = region;
                this.UpperBound = upperBound;
                this.SplitDim = splitDim;
            }

            public int CompareTo(QueueNode other)
            {
                return other.UpperBound.CompareTo(UpperBound);
            }

            public override string ToString()
            {
                return $"QueueNode({Region}, {UpperBound}, {SplitDim})";
            }
        }

        public static bool Debug;
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
            if (xTolerance <= 0) throw new ArgumentOutOfRangeException($"xTolerance <= 0");
            int dim = bounds.Lower.Count;
            if (dim == 0) return Vector.Zero(dim);
            PriorityQueue<QueueNode> queue = new PriorityQueue<QueueNode>();
            double lowerBound = double.NegativeInfinity;
            Vector argmax = bounds.GetMidpoint();
            long upperBoundCount = 0;
            Action<Region,int> addRegion = delegate (Region region, int splitDim)
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
                upperBoundCount++;
                //if (upperBoundCount % 100 == 0) Trace.WriteLine($"lowerBound = {lowerBound}");
                double upperBound = upperBoundF;
                if (upperBound > lowerBound)
                {
                    if (Debug)
                        Trace.WriteLine($"added region {region} with upperBoundF = {upperBoundF}");
                    QueueNode node = new QueueNode(region, upperBound, splitDim);
                    queue.Add(node);
                }
                else if (Debug)
                    Trace.WriteLine($"rejected region {region} with upperBound {upperBound} <= lowerBound {lowerBound}");
            };
            addRegion(bounds, 0);
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
                    Trace.WriteLine($"expanding {node} lower bound = {nodeLowerBound}");
                if (nodeLowerBound > node.UpperBound) throw new Exception("nodeLowerBound > node.UpperBound");
                if (nodeLowerBound > lowerBound)
                {
                    argmax = midpoint;
                    lowerBound = nodeLowerBound;
                }

                // Find a dimension to split on.
                // As a region gets split, this will cycle through the dimensions.
                int splitDim = (node.SplitDim + 1) % dim;
                // To avoid storing SplitDims, we could make a random choice.
                // However, this takes much longer to converge.
                //int splitDim = Rand.Int(dim);
                bool foundSplit = false;
                for (int i = 0; i < dim; i++)
                {
                    if (MMath.AbsDiff(region.Upper[splitDim], region.Lower[splitDim], 1e-10) < xTolerance)
                    {
                        splitDim++;
                        if (splitDim == dim)
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
                    addRegion(leftRegion, splitDim);
                }
                if (region.Lower[splitDim] != splitValue)
                {
                    Region rightRegion = new Region(region);
                    rightRegion.Lower[splitDim] = splitValue;
                    addRegion(rightRegion, splitDim);
                }
            }
            //Trace.WriteLine($"BranchAndBound.Search upperBoundCount = {upperBoundCount}");
            return argmax;
        }
    }
}
