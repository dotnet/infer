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

            public QueueNode(Region region, double upperBound)
            {
                this.Region = region;
                this.UpperBound = upperBound;
            }

            public int CompareTo(QueueNode other)
            {
                return other.UpperBound.CompareTo(UpperBound);
            }

            public override string ToString()
            {
                return $"QueueNode({Region}, {UpperBound})";
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
            if (Debug)
            {
                Func<Region, double> GetUpperBound1 = GetUpperBound;
                GetUpperBound = region =>
                {
                    Stopwatch watch = Stopwatch.StartNew();
                    double upperBound = GetUpperBound1(region);
                    watch.Stop();
                    if (timeAccumulator.Count > 10 && watch.ElapsedMilliseconds > timeAccumulator.Mean + 4 * Math.Sqrt(timeAccumulator.Variance))
                        Trace.WriteLine($"GetUpperBound took {watch.ElapsedMilliseconds}ms");
                    timeAccumulator.Add(watch.ElapsedMilliseconds);
                    //if (upperBoundCount % 100 == 0) Trace.WriteLine($"lowerBound = {lowerBound}");
                    return upperBound;
                };
            }
            Action<Region, int, double> addRegion = delegate (Region region, int splitDim, double upperBound)
            {
                if (upperBound > lowerBound)
                {
                    if (Debug)
                        Trace.WriteLine($"added region {region} with upperBound = {upperBound}");
                    QueueNode node = new QueueNode(region, upperBound);
                    queue.Add(node);
                }
                else if (Debug)
                    Trace.WriteLine($"rejected region {region} with upperBound {upperBound} <= lowerBound {lowerBound}");
            };
            upperBoundCount++;
            addRegion(bounds, 0, GetUpperBound(bounds));
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
                    if (timeAccumulator.Count > 10 && watch.ElapsedMilliseconds > timeAccumulator.Mean + 4 * Math.Sqrt(timeAccumulator.Variance))
                        Trace.WriteLine($"Evaluate took {watch.ElapsedMilliseconds}ms");
                    timeAccumulator.Add(watch.ElapsedMilliseconds);
                    Trace.WriteLine($"expanding {node} lower bound = {nodeLowerBound}");
                }
                if (nodeLowerBound > node.UpperBound) throw new Exception("nodeLowerBound > node.UpperBound");
                if (nodeLowerBound > lowerBound)
                {
                    argmax = midpoint;
                    lowerBound = nodeLowerBound;
                }

                Func<int, bool> DimensionCanSplit = i => MMath.AbsDiff(region.Upper[i], region.Lower[i], 1e-10) >= xTolerance;

                int splitDim;
                bool lowestChild = false;
                Region leftRegion = null;
                double upperBoundLeft = default(double);
                Region rightRegion = null;
                double upperBoundRight = default(double);
                if (lowestChild)
                {
                    splitDim = -1;
                    double lowestUpperBound = double.PositiveInfinity;
                    for (int i = 0; i < dim; i++)
                    {
                        if (DimensionCanSplit(i))
                        {
                            double splitValue2 = midpoint[i];
                            Region leftRegion2 = new Region(region);
                            leftRegion2.Upper[i] = splitValue2;
                            upperBoundCount++;
                            double upperBoundLeft2 = GetUpperBound(leftRegion2);
                            Region rightRegion2 = new Region(region);
                            rightRegion2.Lower[i] = splitValue2;
                            upperBoundCount++;
                            double upperBoundRight2 = GetUpperBound(rightRegion2);
                            double lowerUpperBound = Math.Min(upperBoundLeft2, upperBoundRight2);
                            if (lowerUpperBound < lowestUpperBound)
                            {
                                lowestUpperBound = lowerUpperBound;
                                upperBoundLeft = upperBoundLeft2;
                                upperBoundRight = upperBoundRight2;
                                leftRegion = leftRegion2;
                                rightRegion = rightRegion2;
                                splitDim = i;
                            }
                        }
                    }
                    if (splitDim < 0)
                        break;
                }
                else
                {
                    // Find a dimension to split on.
                    splitDim = Rand.Int(dim);
                    bool foundSplit = false;
                    for (int i = 0; i < dim; i++)
                    {
                        if (!DimensionCanSplit(splitDim))
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
                }

                // split the node
                double splitValue = midpoint[splitDim];
                if(Debug)
                    Trace.WriteLine($"splitting dimension {splitDim}");
                if (region.Upper[splitDim] != splitValue)
                {
                    if(leftRegion == null)
                    {
                        leftRegion = new Region(region);
                        leftRegion.Upper[splitDim] = splitValue;
                        upperBoundCount++;
                        upperBoundLeft = GetUpperBound(leftRegion);
                    }
                    addRegion(leftRegion, splitDim, upperBoundLeft);
                }
                if (region.Lower[splitDim] != splitValue)
                {
                    if(rightRegion == null)
                    {
                        rightRegion = new Region(region);
                        rightRegion.Lower[splitDim] = splitValue;
                        upperBoundCount++;
                        upperBoundRight = GetUpperBound(rightRegion);
                    }
                    addRegion(rightRegion, splitDim, upperBoundRight);
                }
            }
            if(Debug)
                Trace.WriteLine($"BranchAndBound.Search upperBoundCount = {upperBoundCount}");
            return argmax;
        }
    }
}
