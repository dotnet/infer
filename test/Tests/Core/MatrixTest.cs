// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Math;
using Assert = Microsoft.ML.Probabilistic.Tests.AssertHelper;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using System.Diagnostics;

namespace Microsoft.ML.Probabilistic.Tests
{

    public class MatrixTests
    {
        public const double TOLERANCE = 1e-7;

        [Fact]
        public void VectorFactoryTests()
        {
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            // Zero dense
            Vector v1dense = Vector.Zero(2);
            Assert.Equal(2, v1dense.Count);
            Assert.Equal(0.0, v1dense[0]);
            Assert.Equal(0.0, v1dense[1]);
            Assert.Equal(v1dense.Sparsity, Sparsity.Dense);
            // Zero sparse
            Vector v1sparse = Vector.Zero(2, Sparsity.Sparse);
            Assert.Equal(2, v1sparse.Count);
            Assert.Equal(0.0, v1sparse[0]);
            Assert.Equal(0.0, v1sparse[1]);
            Assert.Equal(v1sparse.Sparsity, Sparsity.Sparse);
            // Zero approximate Sparse
            Vector v1approx = Vector.Zero(2, approxSparsity);
            Assert.Equal(2, v1approx.Count);
            Assert.Equal(0.0, v1approx[0]);
            Assert.Equal(0.0, v1approx[1]);
            Assert.Equal(v1approx.Sparsity, approxSparsity);
            // Zero piecewise
            Vector v1piece = Vector.Zero(2, Sparsity.Piecewise);
            Assert.Equal(2, v1piece.Count);
            Assert.Equal(0.0, v1piece[0]);
            Assert.Equal(0.0, v1piece[1]);
            Assert.Equal(v1piece.Sparsity, Sparsity.Piecewise);

            // Constant dense
            Vector v2dense = Vector.Constant(2, 3.0);
            Assert.Equal(2, v2dense.Count);
            Assert.Equal(3.0, v2dense[0]);
            Assert.Equal(3.0, v2dense[1]);
            Assert.Equal(v2dense.Sparsity, Sparsity.Dense);
            // Constant sparse
            Vector v2sparse = Vector.Constant(2, 3.0, Sparsity.Sparse);
            Assert.Equal(2, v2sparse.Count);
            Assert.Equal(3.0, v2sparse[0]);
            Assert.Equal(3.0, v2sparse[1]);
            Assert.True(v2sparse.Sparsity.IsSparse);
            // Constant approximate
            Vector v2approx = Vector.Constant(2, 3.0, approxSparsity);
            Assert.Equal(2, v2approx.Count);
            Assert.Equal(3.0, v2approx[0]);
            Assert.Equal(3.0, v2approx[1]);
            Assert.Equal(v2approx.Sparsity, approxSparsity);
            // Constant piecewise
            Vector v2piece = Vector.Constant(2, 3.0, Sparsity.Piecewise);
            Assert.Equal(2, v2piece.Count);
            Assert.Equal(3.0, v2piece[0]);
            Assert.Equal(3.0, v2piece[1]);
            Assert.Equal(v2piece.Sparsity, Sparsity.Piecewise);

            double[] fromArray = new double[] { 1.2, 2.3, 2.3, 3.4, 1.2, 1.2 };
            // FromArray dense
            Vector v3dense = Vector.FromArray(fromArray);
            Assert.Equal(6, v3dense.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v3dense[i], fromArray[i]);
            Assert.Equal(v3dense.Sparsity, Sparsity.Dense);

            // FromArray sparse
            Vector v3sparse = Vector.FromArray(fromArray, Sparsity.Sparse);
            Assert.Equal(6, v3sparse.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v3sparse[i], fromArray[i]);
            Assert.Equal(v3sparse.Sparsity, Sparsity.Sparse);
            SparseVector v3sparses = ((SparseVector)v3sparse);
            Assert.Equal(1.2, v3sparses.CommonValue);
            Assert.Equal(3, v3sparses.SparseValues.Count);
            Assert.True(v3sparses.HasCommonElements);

            // FromArray approximate
            Vector v3approx = Vector.FromArray(fromArray, approxSparsity);
            Assert.Equal(6, v3approx.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v3approx[i], fromArray[i]);
            Assert.Equal(v3approx.Sparsity, approxSparsity);
            ApproximateSparseVector v3approxa = ((ApproximateSparseVector)v3approx);
            Assert.Equal(1.2, v3approxa.CommonValue);
            Assert.Equal(3, v3approxa.SparseValues.Count);
            Assert.True(v3approxa.HasCommonElements);


            // FromArray piecewise
            Vector v3piece = Vector.FromArray(fromArray, Sparsity.Piecewise);
            Assert.Equal(6, v3piece.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v3piece[i], fromArray[i]);
            Assert.Equal(v3piece.Sparsity, Sparsity.Piecewise);
            PiecewiseVector v3pieces = ((PiecewiseVector)v3piece);
            Assert.Equal(1.2, v3pieces.CommonValue);
            Assert.Equal(2, v3pieces.Pieces.Count);
            Assert.True(v3pieces.HasCommonElements());

            // Copy to dense from dense vector
            Vector v4denseFromDense = Vector.Copy(v3dense);
            Assert.Equal(6, v4denseFromDense.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v4denseFromDense[i], fromArray[i]);
            Assert.Equal(v4denseFromDense.Sparsity, Sparsity.Dense);

            // Copy to dense from sparse vector
            Vector v4denseFromSparse = Vector.Copy(v3sparse, Sparsity.Dense);
            Assert.Equal(6, v4denseFromSparse.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v4denseFromSparse[i], fromArray[i]);
            Assert.Equal(v4denseFromSparse.Sparsity, Sparsity.Dense);

            // Copy to dense from approximate sparse vector
            Vector v4denseFromApprox = Vector.Copy(v3approx, Sparsity.Dense);
            Assert.Equal(6, v4denseFromApprox.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v4denseFromApprox[i], fromArray[i]);
            Assert.Equal(v4denseFromApprox.Sparsity, Sparsity.Dense);

            // Copy to dense from piecewise vector
            Vector v4denseFromPiece = Vector.Copy(v3piece, Sparsity.Dense);
            Assert.Equal(6, v4denseFromPiece.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v4denseFromPiece[i], fromArray[i]);
            Assert.Equal(v4denseFromPiece.Sparsity, Sparsity.Dense);

            // Copy to sparse from dense vector
            Vector v4sparseFromDense = Vector.Copy(v3dense, Sparsity.Sparse);
            Assert.Equal(6, v4sparseFromDense.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v4sparseFromDense[i], fromArray[i]);
            Assert.Equal(v4sparseFromDense.Sparsity, Sparsity.Sparse);
            SparseVector v4sparseFromDenses = ((SparseVector)v4sparseFromDense);
            Assert.Equal(1.2, v4sparseFromDenses.CommonValue);
            Assert.Equal(3, v4sparseFromDenses.SparseValues.Count);
            Assert.True(v4sparseFromDenses.HasCommonElements);

            // Copy to sparse from sparse vector
            Vector v4sparseFromSparse = Vector.Copy(v3sparse, Sparsity.Sparse);
            Assert.Equal(6, v4sparseFromSparse.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v4sparseFromSparse[i], fromArray[i]);
            Assert.Equal(v4sparseFromSparse.Sparsity, Sparsity.Sparse);
            SparseVector v4sparseFromSparses = ((SparseVector)v4sparseFromSparse);
            Assert.Equal(1.2, v4sparseFromSparses.CommonValue);
            Assert.Equal(3, v4sparseFromSparses.SparseValues.Count);
            Assert.True(v4sparseFromSparses.HasCommonElements);

            // Copy to sparse from approx sparse vector
            Vector v4sparseFromApprox = Vector.Copy(v3approx, Sparsity.Sparse);
            Assert.Equal(6, v4sparseFromApprox.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v4sparseFromApprox[i], fromArray[i]);
            Assert.Equal(v4sparseFromApprox.Sparsity, Sparsity.Sparse);
            SparseVector v4sparseFromApproxs = ((SparseVector)v4sparseFromApprox);
            Assert.Equal(1.2, v4sparseFromApproxs.CommonValue);
            Assert.Equal(3, v4sparseFromApproxs.SparseValues.Count);
            Assert.True(v4sparseFromApproxs.HasCommonElements);

            // Copy to sparse from piecewise vector
            Vector v4sparseFromPiece = Vector.Copy(v3piece, Sparsity.Sparse);
            Assert.Equal(6, v4sparseFromPiece.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v4sparseFromPiece[i], fromArray[i]);
            Assert.Equal(v4sparseFromPiece.Sparsity, Sparsity.Sparse);
            SparseVector v4sparseFromPieces = ((SparseVector)v4sparseFromPiece);
            Assert.Equal(1.2, v4sparseFromSparses.CommonValue);
            Assert.Equal(3, v4sparseFromSparses.SparseValues.Count);
            Assert.True(v4sparseFromSparses.HasCommonElements);

            // Copy to approx from dense vector
            Vector v4approxFromDense = Vector.Copy(v3dense, approxSparsity);
            Assert.Equal(6, v4approxFromDense.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v4approxFromDense[i], fromArray[i]);
            Assert.Equal(v4approxFromDense.Sparsity, approxSparsity);
            ApproximateSparseVector v4approxFromDensea = ((ApproximateSparseVector)v4approxFromDense);
            Assert.Equal(1.2, v4approxFromDensea.CommonValue);
            Assert.Equal(3, v4approxFromDensea.SparseValues.Count);
            Assert.True(v4approxFromDensea.HasCommonElements);

            // Copy to approx from sparse vector
            Vector v4approxFromSparse = Vector.Copy(v3sparse, approxSparsity);
            Assert.Equal(6, v4approxFromSparse.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v4approxFromSparse[i], fromArray[i]);
            Assert.Equal(v4approxFromSparse.Sparsity, approxSparsity);
            ApproximateSparseVector v4approxFromSparsea = ((ApproximateSparseVector)v4approxFromSparse);
            Assert.Equal(1.2, v4approxFromSparsea.CommonValue);
            Assert.Equal(3, v4approxFromSparsea.SparseValues.Count);
            Assert.True(v4approxFromSparsea.HasCommonElements);

            // Copy to approx from approx sparse vector
            Vector v4approxFromApprox = Vector.Copy(v3approx, approxSparsity);
            Assert.Equal(6, v4approxFromApprox.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v4approxFromApprox[i], fromArray[i]);
            Assert.Equal(v4approxFromApprox.Sparsity, approxSparsity);
            ApproximateSparseVector v4approxFromApproxa = ((ApproximateSparseVector)v4approxFromApprox);
            Assert.Equal(1.2, v4approxFromApproxa.CommonValue);
            Assert.Equal(3, v4approxFromApproxa.SparseValues.Count);
            Assert.True(v4approxFromApproxa.HasCommonElements);

            // Copy to approx from sparse vector
            Vector v4approxFromPiece = Vector.Copy(v3piece, approxSparsity);
            Assert.Equal(6, v4approxFromPiece.Count);
            for (int i = 0; i < fromArray.Length; i++) Assert.Equal(v4approxFromPiece[i], fromArray[i]);
            Assert.Equal(v4approxFromPiece.Sparsity, approxSparsity);
            ApproximateSparseVector v4approxFromSparseb = ((ApproximateSparseVector)v4approxFromPiece);
            Assert.Equal(1.2, v4approxFromSparseb.CommonValue);
            Assert.Equal(3, v4approxFromSparseb.SparseValues.Count);
            Assert.True(v4approxFromSparseb.HasCommonElements);

            // Dense vector from reference
            Vector v5Dense = DenseVector.FromArrayReference(3, fromArray, 1);
            Assert.Equal(3, v5Dense.Count);
            for (int i = 0; i < v5Dense.Count; i++) Assert.Equal(v5Dense[i], fromArray[i + 1]);
            Assert.Equal(v5Dense.Sparsity, Sparsity.Dense);
        }

        private void VectorSet(Vector v)
        {
            for (int i = 0; i < v.Count; i++)
            {
                double d = Rand.Double();
                v[i] = d;
                Assert.Equal(v[i], d);
            }
            if (v.IsApproximate)
            {
                ApproximateSparseVector asv = (ApproximateSparseVector)v;
                int idx = asv.Count / 2;
                int numCV = asv.SparseValues.Count;
                v[idx] = asv.CommonValue + 0.5 * asv.Tolerance;
                Assert.Equal(v[idx], asv.CommonValue);
                Assert.Equal(numCV - 1, asv.SparseValues.Count);
            }
        }

        [Fact]
        public void VectorSetTests()
        {
            Rand.Restart(12347);
            VectorSet(Vector.Zero(6));
            VectorSet(Vector.Zero(6, Sparsity.Sparse));
            VectorSet(Vector.Zero(6, Sparsity.ApproximateWithTolerance(0.001)));
            VectorSet(DenseVector.FromArrayReference(6, new double[10], 3));
            VectorSet(Vector.Zero(6, Sparsity.Piecewise));
        }

        private void VectorIndexOf(Vector v, double commonValue)
        {
            int commonCnt = 0;
            for (int i = 0; i < v.Count; i++)
            {
                if ((i % 2) == 0)
                    v[i] = commonValue + i + 0.5;
                else
                {
                    v[i] = commonValue;
                    commonCnt++;
                }
            }
            for (int i = 0; i < v.Count; i++)
            {
                if ((i % 2) == 0)
                    Assert.Equal(v.IndexOf(v[i]), i);
            }
            Assert.Equal(1, v.IndexOf(commonValue));

            // Test IndexOfAll/FindAll
            Func<double, bool> filter = x => x >= commonValue && x < v[2];
            int[] indexOfAll = v.IndexOfAll(filter).ToArray();
            ValueAtIndex<double>[] all = v.FindAll(filter).ToArray();
            int allCount = v.CountAll(filter);
            Assert.Equal(indexOfAll.Length, all.Length);
            Assert.Equal(indexOfAll.Length, allCount);
            int expectedIdx = 0;
            int observedCnt = 0;
            int expectedCnt = commonCnt + 1;
            for (int i = 0; i < all.Length; ++i)
            {
                Assert.Equal(indexOfAll[i], all[i].Index);
                Assert.Equal(expectedIdx, indexOfAll[i]);
                double expectedValue = expectedIdx == 0 ? commonValue + 0.5 : commonValue;
                Assert.Equal(expectedValue, all[i].Value);

                expectedIdx += (expectedIdx == 0) ? 1 : 2;
                observedCnt++;
            }
            Assert.Equal(expectedCnt, observedCnt);

            filter = x => x > v[0] && x <= v[2];
            indexOfAll = v.IndexOfAll(filter).ToArray();
            all = v.FindAll(filter).ToArray();
            allCount = v.CountAll(filter);
            Assert.Single(indexOfAll);
            Assert.Single(all);
            Assert.Equal(1, allCount);
            Assert.Equal(2, indexOfAll[0]);
            Assert.Equal(2, all[0].Index);
            Assert.Equal(v[2], all[0].Value);

            // Index of minimum/maximum
            Assert.Equal(1, v.IndexOfMinimum());
            Assert.Equal(v.IndexOfMaximum(), v.Count - 2);

            // Contains
            for (int i = 0; i < v.Count; i++)
            {
                Assert.Contains(v[i], v);
                Assert.DoesNotContain(v[i] + 0.1, v);
            }
        }

        [Fact]
        public void VectorIndexOfTests()
        {
            VectorIndexOf(Vector.Zero(6), 0.0);
            VectorIndexOf(Vector.Zero(6, Sparsity.Sparse), 0.0);
            VectorIndexOf(DenseVector.FromArrayReference(6, new double[10], 3), 0.0);
            VectorIndexOf(Vector.Zero(6, Sparsity.ApproximateWithTolerance(0.001)), 0.0);
            VectorIndexOf(Vector.Zero(6, Sparsity.Piecewise), 0.0);
        }

        private void VectorFindFirstAndLastIndex(Sparsity sparsity)
        {
            var v = Vector.Zero(6, sparsity);
            v[0] = 1;
            v[2] = 2;
            v[3] = 2;
            v[5] = 1;

            int firstIndex = v.FindFirstIndex(elt => elt > 0.0);
            int lastIndex = v.FindLastIndex(elt => elt > 0.0);

            Assert.Equal(0, firstIndex);
            Assert.Equal(5, lastIndex);

            firstIndex = v.FindFirstIndex(elt => elt == 0.0);
            lastIndex = v.FindLastIndex(elt => elt == 0.0);
            Assert.Equal(1, firstIndex);
            Assert.Equal(4, lastIndex);

            firstIndex = v.FindFirstIndex(elt => elt == 2.0);
            lastIndex = v.FindLastIndex(elt => elt == 2.0);
            Assert.Equal(2, firstIndex);
            Assert.Equal(3, lastIndex);

            v[1] = 3.0;
            v[4] = 3.0;

            firstIndex = v.FindFirstIndex(elt => elt == 0.0);
            lastIndex = v.FindLastIndex(elt => elt == 0.0);
            Assert.Equal(-1, firstIndex);
            Assert.Equal(-1, lastIndex);
        }

        [Fact]
        public void VectorFindFirstAndLastIndexTests()
        {
            VectorFindFirstAndLastIndex(Sparsity.Dense);
            VectorFindFirstAndLastIndex(Sparsity.Sparse);
            VectorFindFirstAndLastIndex(Sparsity.ApproximateWithTolerance(0.001));
            VectorFindFirstAndLastIndex(Sparsity.Piecewise);
        }

        [Fact]
        public void VectorIndexAtCumulativeSumTests()
        {
            const int count = 13;
            const double commonValue = 0.02;
            Vector dense = DenseVector.Constant(count, commonValue);
            Vector sparse = SparseVector.Constant(count, commonValue);
            Vector piecewise = PiecewiseVector.Constant(count, commonValue);

            // Test vectors have the form { (0.02, 0.02, 0.02), [0.3, 0.3], (0.02, 0.02), 0.2, (0.02, 0.02, 0.02, 0.02, 0.02) }
            dense[3] = sparse[3] = piecewise[3] = 0.3;
            dense[4] = sparse[4] = piecewise[4] = 0.3;
            dense[7] = sparse[7] = piecewise[7] = 0.2;

            foreach (Vector v in new[] { dense, sparse, piecewise })
            {
                Assert.Equal(-1, v.IndexAtCumulativeSum(1.0));
                Assert.Equal(0, v.IndexAtCumulativeSum(0.01));
                Assert.Equal(1, v.IndexAtCumulativeSum(0.02));
                Assert.Equal(2, v.IndexAtCumulativeSum(0.05));
                Assert.Equal(3, v.IndexAtCumulativeSum(0.35));
                Assert.Equal(4, v.IndexAtCumulativeSum(0.4));
                Assert.Equal(4, v.IndexAtCumulativeSum(0.65));
                Assert.Equal(5, v.IndexAtCumulativeSum(0.67));
                Assert.Equal(6, v.IndexAtCumulativeSum(0.69));
                Assert.Equal(7, v.IndexAtCumulativeSum(0.79));
                Assert.Equal(8, v.IndexAtCumulativeSum(0.901));
                Assert.Equal(9, v.IndexAtCumulativeSum(0.93));
                Assert.Equal(12, v.IndexAtCumulativeSum(0.99));
            }

            // Negative values
            // Test vectors have the form { (0.02, 0.02, 0.02), [-0.01, -0.01], (0.02, 0.02), 0.2, (0.02, 0.02, 0.02, 0.02, 0.02) }
            dense[3] = sparse[3] = piecewise[3] = -0.01;
            dense[4] = sparse[4] = piecewise[4] = -0.01;
            dense[7] = sparse[7] = piecewise[7] = 0.2;

            foreach (Vector v in new[] { dense, sparse, piecewise })
            {
                Assert.Equal(6, v.IndexAtCumulativeSum(0.07));
                Assert.Equal(7, v.IndexAtCumulativeSum(0.15));
                Assert.Equal(9, v.IndexAtCumulativeSum(0.31));
            }

            // No common values
            double[] array = new[] { 0.2, 0.2, 0.0, 0.6 };
            dense = DenseVector.FromArray(array);
            sparse = SparseVector.FromArray(array);
            piecewise = PiecewiseVector.FromArray(array);

            foreach (Vector v in new[] { dense, sparse, piecewise })
            {
                Assert.Equal(-1, v.IndexAtCumulativeSum(1.0));
                Assert.Equal(0, v.IndexAtCumulativeSum(0.05));
                Assert.Equal(1, v.IndexAtCumulativeSum(0.25));
                Assert.Equal(3, v.IndexAtCumulativeSum(0.4));
                Assert.Equal(3, v.IndexAtCumulativeSum(0.99));
            }

            // Empty vectors
            dense = DenseVector.Zero(0);
            sparse = SparseVector.Zero(0);
            piecewise = PiecewiseVector.Zero(0);

            foreach (Vector v in new[] { dense, sparse, piecewise })
            {
                Assert.Equal(-1, v.IndexAtCumulativeSum(-1));
            }
        }

        private void VectorAppendVector(Vector thisVector, Vector thatVector)
        {
            int jointCount = thisVector.Count + thatVector.Count;
            double[] expected = new double[jointCount];
            int i = 0;
            for (int j = 0; j < thisVector.Count; j++)
                expected[i++] = thisVector[j];
            for (int j = 0; j < thatVector.Count; j++)
                expected[i++] = thatVector[j];

            SparseVector thisSparseVector = thisVector as SparseVector;
            SparseVector thatSparseVector = thatVector as SparseVector;

            Vector vAppend = thisVector.Append(thatVector);
            Assert.Equal(vAppend.Count, jointCount);
            for (int j = 0; j < jointCount; j++)
                Assert.Equal(expected[j], vAppend[j]);

            if (thisVector.Sparsity == Sparsity.Piecewise || thatVector.Sparsity == Sparsity.Piecewise)
            {
                // todo
            }
            else if (thisVector.Sparsity == Sparsity.Dense || thatVector.Sparsity == Sparsity.Dense)
                Assert.True(vAppend.Sparsity.IsDense);
            else if (thisSparseVector.CommonValue != thatSparseVector.CommonValue)
                Assert.True(vAppend.Sparsity.IsDense);
            else
                Assert.Equal(vAppend.Sparsity, thisVector.Sparsity);

            // Static concatenator
            Vector vConcat = Vector.Concat(thisVector, thatVector);
            Assert.Equal(vConcat.Count, jointCount);
            for (int j = 0; j < jointCount; j++)
                Assert.Equal(expected[j], vConcat[j]);

            if (thisVector.Sparsity == Sparsity.Piecewise || thatVector.Sparsity == Sparsity.Piecewise)
            {
                // todo
            }
            else if (thisVector.Sparsity == Sparsity.Dense || thatVector.Sparsity == Sparsity.Dense)
                Assert.True(vConcat.IsDense);
            else if (thisSparseVector.CommonValue != thatSparseVector.CommonValue)
                Assert.True(vConcat.IsDense);
            else
                Assert.Equal(vConcat.Sparsity, thisVector.Sparsity);
        }

        private void VectorAppendScalar(Vector thisVector, double thatScalar)
        {
            int jointCount = thisVector.Count + 1;
            double[] expected = new double[jointCount];
            int i = 0;
            for (int j = 0; j < thisVector.Count; j++)
                expected[i++] = thisVector[j];
            expected[i] = thatScalar;

            Vector vAppend = thisVector.Append(thatScalar);
            Assert.Equal(vAppend.Count, jointCount);
            for (int j = 0; j < jointCount; j++)
                Assert.Equal(expected[j], vAppend[j]);

            Assert.Equal(vAppend.Sparsity, thisVector.Sparsity);

            if (thisVector.IsApproximate)
            {
                var asv = (ApproximateSparseVector)thisVector;
                int numCV = asv.SparseValues.Count;
                int len = thisVector.Count;
                var asvAppend = (ApproximateSparseVector)thisVector.Append(asv.CommonValue + 0.5 * asv.Tolerance);
                Assert.Equal(asv.CommonValue, asvAppend[len]);
                Assert.Equal(numCV, asvAppend.SparseValues.Count);
            }
        }

        [Fact]
        public void VectorAppendTests()
        {
            double[] thisArray = new double[] { 1.2, 2.3, 3.4, 1.2, 1.2, 2.3 };
            double[] thatArray = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8 };
            Vector[] thisVector = new Vector[5];
            Vector[] thatVector = new Vector[5];
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            thisVector[0] = Vector.FromArray(thisArray);
            thisVector[1] = Vector.FromArray(thisArray, Sparsity.Sparse);
            thisVector[2] = Vector.FromArray(thisArray, approxSparsity);
            thisVector[3] = DenseVector.FromArrayReference(3, thisArray, 2);
            thisVector[4] = Vector.FromArray(thisArray, Sparsity.Piecewise);
            thatVector[0] = Vector.FromArray(thatArray);
            thatVector[1] = Vector.FromArray(thatArray, Sparsity.Sparse);
            thatVector[2] = Vector.FromArray(thatArray, approxSparsity);
            thatVector[3] = DenseVector.FromArrayReference(2, thatArray, 1);
            thatVector[4] = Vector.FromArray(thatArray, Sparsity.Piecewise);

            for (int i = 0; i < thisVector.Length; i++)
                for (int j = 0; j < thisVector.Length; j++)
                    VectorAppendVector(thisVector[i], thatVector[j]);

            for (int i = 0; i < thisVector.Length; i++)
            {
                VectorAppendScalar(thisVector[i], 1.2); // Common value
                VectorAppendScalar(thisVector[i], 3.4);
            }
        }

        private void VectorSetToValue(Vector thisVector, double val)
        {
            // Create a copy of the correct sparsity
            Vector a = Vector.Constant(thisVector.Count, val, thisVector.Sparsity);
            Vector b = Vector.Zero(thisVector.Count, thisVector.Sparsity);
            b.SetAllElementsTo(val);
            Assert.True(a.Equals(b));
        }

        private void VectorSetToArray(Vector thisVector, double[] arr)
        {
            // Create copies of the correct sparsity
            Vector a = Vector.FromArray(arr, thisVector.Sparsity);
            Vector b = Vector.Zero(thisVector.Count, thisVector.Sparsity);
            b.SetTo(arr);
            Assert.True(a.Equals(b));

            if (thisVector.IsApproximate)
            {
                ApproximateSparseVector asv = (ApproximateSparseVector)a;
                double[] arr2 = (double[])arr.Clone();
                double scale = 0.5 * asv.Tolerance;

                // Minimum value decides the common value, so keep the first of these
                int idx = asv.IndexOf(asv.CommonValue);
                for (int i = 0; i < arr2.Length; i++)
                    arr2[i] = (arr[i] == asv.CommonValue && i != idx) ? (asv.CommonValue + scale * (Rand.Double())) : arr[i];

                b.SetTo(arr2);
                Assert.True(a.Equals(b));
            }
        }

        private void VectorSetToSubarray(Vector thisVector, double[] arr, double[] ext_arr, int startIndex)
        {
            // Create copies of the correct sparsity
            Vector a = Vector.FromArray(arr, thisVector.Sparsity);
            Vector b = Vector.Zero(thisVector.Count, thisVector.Sparsity);
            b.SetToSubarray(ext_arr, startIndex);
            Assert.True(a.Equals(b));
        }

        private void VectorSetToVector(Vector thisVector, Vector thatVector)
        {
            thisVector.SetTo(thatVector);
            Assert.True(thisVector.Equals(thatVector));
        }

        private void VectorCopyTo(Vector thisVector)
        {
            double[] target = new double[thisVector.Count + 2];
            target[0] = 10.0;
            target[target.Length - 1] = 10.0;
            thisVector.CopyTo(target, 1);
            for (int i = 1; i < target.Length - 1; i++)
                Assert.Equal(thisVector[i - 1], target[i]);
            Assert.Equal(10.0, target[0]);
            Assert.Equal(10.0, target[target.Length - 1]);
        }


        [Fact]
        public void VectorSetToEnumerableTests()
        {
            double[] thisArray = new double[] { 1.2, 2.3, 3.4, 1.2, 1.2, 2.3 };
            double[] thatArray = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8, 1.2 };
            double[] thisExtendedArray = new double[] { 5.6, 1.2, 2.3, 3.4, 1.2, 1.2, 2.3, 6.7 };
            double[] thatExtendedArray = new double[] { 5.6, 5.6, 1.2, 1.2, 6.7, 7.8, 1.2, 1.2 };
            Vector[] thisVector = new Vector[5];
            var that = new IEnumerable<double>[8];
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            thisVector[0] = Vector.FromArray(thisArray);
            thisVector[1] = Vector.FromArray(thisArray, Sparsity.Sparse);
            thisVector[2] = Vector.FromArray(thisArray, approxSparsity);
            thisVector[3] = DenseVector.FromArrayReference(6, thisExtendedArray, 1);
            thisVector[4] = Vector.FromArray(thisArray, Sparsity.Piecewise);
            that[0] = Vector.FromArray(thatArray);
            that[1] = Vector.FromArray(thatArray, Sparsity.Sparse);
            that[2] = Vector.FromArray(thatArray, approxSparsity);
            that[3] = DenseVector.FromArrayReference(6, thatExtendedArray, 1);
            that[4] = thatArray;
            that[5] = thatArray.ToList();
            var sparseValues = thatArray.Select((x, i) => new ValueAtIndex<double>(i, x)).ToList();
            that[6] = SparseList<double>.FromSparseValues(thatArray.Length, 0.0, sparseValues);
            that[7] = Vector.FromArray(thatArray, Sparsity.Piecewise);

            for (int i = 0; i < thisVector.Length; i++)
                for (int j = 0; j < that.Length; j++)
                    VectorSetToEnumerable(thisVector[i], that[j]);
        }

        private void VectorSetToEnumerable(Vector thisVector, IEnumerable<double> that)
        {
            thisVector.SetTo(that);
            var thatList = that.ToList();
            Assert.True(thisVector.Count == thatList.Count);
            for (int i = 0; i < thisVector.Count; i++)
            {
                Assert.True(thisVector[i] == thatList[i]);
            }
        }

        [Fact]
        public void VectorSetToTests()
        {
            double[] thisArray = new double[] { 1.2, 2.3, 3.4, 1.2, 1.2, 2.3 };
            double[] thatArray = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8, 1.2 };
            double[] thisExtendedArray = new double[] { 5.6, 1.2, 2.3, 3.4, 1.2, 1.2, 2.3, 6.7 };
            double[] thatExtendedArray = new double[] { 5.6, 5.6, 1.2, 1.2, 6.7, 7.8, 1.2, 1.2 };
            Vector[] thisVector = new Vector[5];
            Vector[] thatVector = new Vector[5];
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            thisVector[0] = Vector.FromArray(thisArray);
            thisVector[1] = Vector.FromArray(thisArray, Sparsity.Sparse);
            thisVector[2] = Vector.FromArray(thisArray, approxSparsity);
            thisVector[3] = DenseVector.FromArrayReference(6, thisExtendedArray, 1);
            thisVector[4] = Vector.FromArray(thisArray, Sparsity.Piecewise);
            thatVector[0] = Vector.FromArray(thatArray);
            thatVector[1] = Vector.FromArray(thatArray, Sparsity.Sparse);
            thatVector[2] = Vector.FromArray(thatArray, approxSparsity);
            thatVector[3] = DenseVector.FromArrayReference(6, thatExtendedArray, 1);
            thatVector[4] = Vector.FromArray(thatArray, Sparsity.Piecewise);

            for (int i = 0; i < thisVector.Length - 1; i++) // no case 3
                VectorSetToValue(thisVector[i], 0.1);

            for (int i = 0; i < thisVector.Length - 1; i++) // no case 3
                VectorSetToArray(thisVector[i], thatArray);

            for (int i = 0; i < thisVector.Length - 1; i++) // no case 3
                VectorSetToSubarray(thisVector[i], thatArray, thatExtendedArray, 1);

            for (int i = 0; i < thisVector.Length; i++)
                VectorCopyTo(thisVector[i]);

            for (int i = 0; i < thisVector.Length; i++)
                for (int j = 0; j < thisVector.Length; j++)
                    VectorSetToVector(thisVector[i], thatVector[j]);
        }

        private void VectorSubvector(Vector thisVector, int startIndex, int count)
        {
            Vector subvec = thisVector.Subvector(startIndex, count);
            Assert.Equal(subvec.Count, count);
            Assert.Equal(subvec.Sparsity, thisVector.Sparsity);
            for (int i = 0, j = startIndex; i < subvec.Count; i++, j++)
                Assert.Equal(subvec[i], thisVector[j]);
        }

        private void VectorSetSubvector(Vector thisVector, Vector thatVector, int startIndex)
        {
            double[] expected = thisVector.ToArray();
            for (int i = startIndex, j = 0; i < startIndex + thatVector.Count; i++)
                expected[i] = thatVector[j++];

            thisVector.SetSubvector(1, thatVector);

            for (int i = 0; i < thisVector.Count; i++)
                Assert.Equal(expected[i], thisVector[i]);
        }

        [Fact]
        public void VectorSubvectorTests()
        {
            double[] thisArray = new double[] { 1.2, 2.3, 3.4, 1.2, 1.2, 2.3 };
            double[] thisExtendedArray = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8, 1.2, 1.2, 6.7 };
            double[] thatArray = new double[] { 5.6, 1.2, 1.2, 6.7 };
            double[] thatExtendedArray = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8, 1.2 };
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            Vector[] thisVector = new Vector[5];
            Vector[] thatVector = new Vector[5];
            thisVector[0] = Vector.FromArray(thisArray);
            thisVector[1] = Vector.FromArray(thisArray, Sparsity.Sparse);
            thisVector[2] = Vector.FromArray(thisArray, approxSparsity);
            thisVector[3] = DenseVector.FromArrayReference(6, thisExtendedArray, 1);
            thisVector[4] = Vector.FromArray(thisArray, Sparsity.Piecewise);
            thatVector[0] = Vector.FromArray(thatArray);
            thatVector[1] = Vector.FromArray(thatArray, Sparsity.Sparse);
            thatVector[2] = Vector.FromArray(thatArray, approxSparsity);
            thatVector[3] = DenseVector.FromArrayReference(4, thatExtendedArray, 1);
            thatVector[4] = Vector.FromArray(thatArray, Sparsity.Piecewise);

            for (int i = 0; i < thisVector.Length; i++)
                VectorSubvector(thisVector[i], 1, thisArray.Length - 2);

            for (int i = 0; i < thisVector.Length; i++)
                for (int j = 0; j < thisVector.Length; j++)
                    VectorSetSubvector(thisVector[i], thatVector[j], 1);
        }

        private void VectorSetToFunction(Vector a, Vector b, Vector c)
        {
            Vector aCopy = Vector.Copy(a);
            Vector bCopy = Vector.Copy(b);
            Vector cCopy = Vector.Copy(c);
            double cConst = 10.0;

            double[] expected = new double[a.Count];
            // Single Vector argument
            for (int i = 0; i < a.Count; i++)
                expected[i] = System.Math.Pow(b[i], 2.0);
            a.SetToFunction(b, x => System.Math.Pow(x, 2.0));
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], a[i]);
            if (a.IsApproximate && b.IsSparse)
            {
                ApproximateSparseVector asv = (ApproximateSparseVector)a;
                SparseVector bsv = (SparseVector)b;
                Vector b1 = Vector.Copy(b);
                double cv = bsv.CommonValue;
                int cc = 0;
                for (int i = 0; i < b1.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        cc++;
                        b1[i] = cv + 0.1 * asv.Tolerance;
                    }
                    else
                    {
                        b1[i] = cv + 2 * asv.Tolerance;
                    }
                }
                a.SetToFunction(b1, x => System.Math.Pow(x, 2.0));
                Assert.Equal(cv * cv, asv.CommonValue);
                Assert.Equal(cc, asv.SparseValues.Count);
            }

            a = Vector.Copy(aCopy);
            b = Vector.Copy(bCopy);
            c = Vector.Copy(cCopy);
            // Two vector arguments
            for (int i = 0; i < a.Count; i++)
                expected[i] = b[i] - c[i];
            a.SetToFunction(b, c, (x, y) => x - y);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], a[i]);
            if (a.IsApproximate && b.IsSparse && c.IsSparse)
            {
                ApproximateSparseVector asv = (ApproximateSparseVector)a;
                SparseVector bsv = (SparseVector)b;
                SparseVector csv = (SparseVector)c;
                Vector b1 = Vector.Copy(b);
                double cv = bsv.CommonValue - csv.CommonValue;
                int cc = 0;
                for (int i = 0; i < b1.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        cc++;
                        b1[i] = c[i] + cv + 0.1 * asv.Tolerance;
                    }
                    else
                    {
                        b1[i] = c[i] + cv + 2 * asv.Tolerance;
                    }
                }
                a.SetToFunction(b1, c, (x, y) => x - y);
                Assert.Equal(cv, asv.CommonValue);
                Assert.Equal(cc, asv.SparseValues.Count);
            }

            a = Vector.Copy(aCopy);
            b = Vector.Copy(bCopy);
            c = Vector.Copy(cCopy);
            // Set to power
            for (int i = 0; i < a.Count; i++)
                expected[i] = System.Math.Pow(b[i], 2.0);
            a.SetToPower(b, 2.0);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], a[i]);

            a = Vector.Copy(aCopy);
            b = Vector.Copy(bCopy);
            c = Vector.Copy(cCopy);
            // Set to sum with vector
            for (int i = 0; i < a.Count; i++)
                expected[i] = b[i] + c[i];
            a.SetToSum(b, c);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], a[i]);
            if (a.IsApproximate && b.IsSparse && c.IsSparse)
            {
                ApproximateSparseVector asv = (ApproximateSparseVector)a;
                SparseVector bsv = (SparseVector)b;
                SparseVector csv = (SparseVector)c;
                Vector b1 = Vector.Copy(b);
                double cv = bsv.CommonValue + csv.CommonValue;
                int cc = 0;
                for (int i = 0; i < b1.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        cc++;
                        b1[i] = -c[i] + cv + 0.1 * asv.Tolerance;
                    }
                    else
                    {
                        b1[i] = -c[i] + cv + 2 * asv.Tolerance;
                    }
                }
                a.SetToSum(b1, c);
                Assert.Equal(cv, asv.CommonValue);
                Assert.Equal(cc, asv.SparseValues.Count);
            }

            a = Vector.Copy(aCopy);
            b = Vector.Copy(bCopy);
            c = Vector.Copy(cCopy);
            // Set to sum with scalar
            for (int i = 0; i < a.Count; i++)
                expected[i] = b[i] + cConst;
            a.SetToSum(b, cConst);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], a[i]);
            if (a.IsApproximate && b.IsSparse)
            {
                ApproximateSparseVector asv = (ApproximateSparseVector)a;
                SparseVector bsv = (SparseVector)b;
                Vector b1 = Vector.Copy(b);
                double cv = bsv.CommonValue + cConst;
                int cc = 0;
                for (int i = 0; i < b1.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        cc++;
                        b1[i] = -cConst + cv + 0.1 * asv.Tolerance;
                    }
                    else
                    {
                        b1[i] = -cConst + cv + 2 * asv.Tolerance;
                    }
                }
                a.SetToSum(b1, cConst);
                Assert.Equal(cv, asv.CommonValue);
                Assert.Equal(cc, asv.SparseValues.Count);
            }

            a = Vector.Copy(aCopy);
            b = Vector.Copy(bCopy);
            c = Vector.Copy(cCopy);
            // Set to mixture sum
            double bScale = 3.0;
            double cScale = 4.0;
            for (int i = 0; i < a.Count; i++)
                expected[i] = bScale * b[i] + cScale * c[i];
            a.SetToSum(bScale, b, cScale, c);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], a[i]);
            if (a.IsApproximate && b.IsSparse && c.IsSparse)
            {
                ApproximateSparseVector asv = (ApproximateSparseVector)a;
                SparseVector bsv = (SparseVector)b;
                SparseVector csv = (SparseVector)c;
                Vector b1 = Vector.Copy(b);
                double cv = bScale * bsv.CommonValue + cScale * csv.CommonValue;
                int cc = 0;
                for (int i = 0; i < b1.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        cc++;
                        b1[i] = (-(cScale * c[i]) + cv + 0.1 * asv.Tolerance) / bScale;
                    }
                    else
                    {
                        b1[i] = (-(cScale * c[i]) + cv + 2 * bScale * asv.Tolerance) / bScale;
                    }
                }
                a.SetToSum(bScale, b1, cScale, c);
                Assert.Equal(cv, asv.CommonValue);
                Assert.Equal(cc, asv.SparseValues.Count);
            }

            a = Vector.Copy(aCopy);
            b = Vector.Copy(bCopy);
            c = Vector.Copy(cCopy);
            // Set to difference with vector
            for (int i = 0; i < a.Count; i++)
                expected[i] = b[i] - c[i];
            a.SetToDifference(b, c);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], a[i]);
            if (a.IsApproximate && b.IsSparse && c.IsSparse)
            {
                ApproximateSparseVector asv = (ApproximateSparseVector)a;
                SparseVector bsv = (SparseVector)b;
                SparseVector csv = (SparseVector)c;
                Vector b1 = Vector.Copy(b);
                double cv = bsv.CommonValue - csv.CommonValue;
                int cc = 0;
                for (int i = 0; i < b1.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        cc++;
                        b1[i] = c[i] + cv + 0.1 * asv.Tolerance;
                    }
                    else
                    {
                        b1[i] = c[i] + cv + 2 * asv.Tolerance;
                    }
                }
                a.SetToDifference(b1, c);
                Assert.Equal(cv, asv.CommonValue);
                Assert.Equal(cc, asv.SparseValues.Count);
            }

            a = Vector.Copy(aCopy);
            b = Vector.Copy(bCopy);
            c = Vector.Copy(cCopy);
            // Set to difference with scalar, scalar first
            for (int i = 0; i < a.Count; i++)
                expected[i] = cConst - b[i];
            a.SetToDifference(cConst, b);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], a[i]);
            if (a.IsApproximate && b.IsSparse)
            {
                ApproximateSparseVector asv = (ApproximateSparseVector)a;
                SparseVector bsv = (SparseVector)b;
                Vector b1 = Vector.Copy(b);
                double cv = -bsv.CommonValue + cConst;
                int cc = 0;
                for (int i = 0; i < b1.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        cc++;
                        b1[i] = cConst - cv - 0.1 * asv.Tolerance;
                    }
                    else
                    {
                        b1[i] = cConst - cv - 2 * asv.Tolerance;
                    }
                }
                a.SetToDifference(cConst, b1);
                Assert.Equal(cv, asv.CommonValue);
                Assert.Equal(cc, asv.SparseValues.Count);
            }

            a = Vector.Copy(aCopy);
            b = Vector.Copy(bCopy);
            c = Vector.Copy(cCopy);
            // Set to difference with scalar, scalar second
            for (int i = 0; i < a.Count; i++)
                expected[i] = b[i] - cConst;
            a.SetToDifference(b, cConst);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], a[i]);
            if (a.IsApproximate && b.IsSparse)
            {
                ApproximateSparseVector asv = (ApproximateSparseVector)a;
                SparseVector bsv = (SparseVector)b;
                Vector b1 = Vector.Copy(b);
                double cv = bsv.CommonValue - cConst;
                int cc = 0;
                for (int i = 0; i < b1.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        cc++;
                        b1[i] = cConst + cv + 0.1 * asv.Tolerance;
                    }
                    else
                    {
                        b1[i] = cConst + cv + 2 * asv.Tolerance;
                    }
                }
                a.SetToDifference(b1, cConst);
                Assert.Equal(cv, asv.CommonValue);
                Assert.Equal(cc, asv.SparseValues.Count);
            }

            a = Vector.Copy(aCopy);
            b = Vector.Copy(bCopy);
            c = Vector.Copy(cCopy);
            // Set to product with vector
            for (int i = 0; i < a.Count; i++)
                expected[i] = b[i] * c[i];
            a.SetToProduct(b, c);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], a[i]);
            if (a.IsApproximate && b.IsSparse && c.IsSparse)
            {
                ApproximateSparseVector asv = (ApproximateSparseVector)a;
                SparseVector bsv = (SparseVector)b;
                SparseVector csv = (SparseVector)c;
                Vector b1 = Vector.Copy(b);
                double cv = bsv.CommonValue * csv.CommonValue;
                int cc = 0;
                for (int i = 0; i < b1.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        cc++;
                        b1[i] = (cv + 0.1 * asv.Tolerance) / c[i];
                    }
                    else
                    {
                        b1[i] = (cv + 2 * c[i] * asv.Tolerance) / c[i];
                    }
                }
                a.SetToProduct(b1, c);
                Assert.Equal(cv, asv.CommonValue);
                Assert.Equal(cc, asv.SparseValues.Count);
            }

            a = Vector.Copy(aCopy);
            b = Vector.Copy(bCopy);
            c = Vector.Copy(cCopy);
            // Set to product with scalar
            for (int i = 0; i < a.Count; i++)
                expected[i] = b[i] * cConst;
            a.SetToProduct(b, cConst);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], a[i]);
            if (a.IsApproximate && b.IsSparse)
            {
                ApproximateSparseVector asv = (ApproximateSparseVector)a;
                SparseVector bsv = (SparseVector)b;
                Vector b1 = Vector.Copy(b);
                double cv = bsv.CommonValue * cConst;
                int cc = 0;
                for (int i = 0; i < b1.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        cc++;
                        b1[i] = (cv + 0.1 * asv.Tolerance) / cConst;
                    }
                    else
                    {
                        b1[i] = (cv + 2 * cConst * asv.Tolerance) / cConst;
                    }
                }
                a.SetToProduct(b1, cConst);
                Assert.Equal(cv, asv.CommonValue);
                Assert.Equal(cc, asv.SparseValues.Count);
            }

            a = Vector.Copy(aCopy);
            b = Vector.Copy(bCopy);
            c = Vector.Copy(cCopy);
            // Scale method (synonym for SetToProduct(this, scale))
            for (int i = 0; i < a.Count; i++)
                expected[i] = a[i] * cConst;
            a.Scale(cConst);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], a[i]);

            a = Vector.Copy(aCopy);
            b = Vector.Copy(bCopy);
            c = Vector.Copy(cCopy);
            // Set to ratio with vector
            for (int i = 0; i < a.Count; i++)
                expected[i] = b[i] / c[i];
            a.SetToRatio(b, c);
            for (int i = 0; i < a.Count; i++)
                Assert.Equal(expected[i], a[i]);

            if (a.IsApproximate && b.IsSparse && c.IsSparse)
            {
                ApproximateSparseVector asv = (ApproximateSparseVector)a;
                SparseVector bsv = (SparseVector)b;
                SparseVector csv = (SparseVector)c;
                Vector b1 = Vector.Copy(b);
                double cv = bsv.CommonValue / csv.CommonValue;
                int cc = 0;
                for (int i = 0; i < b1.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        cc++;
                        b1[i] = (cv + 0.1 * asv.Tolerance / c[i]) * c[i];
                    }
                    else
                    {
                        b1[i] = (cv + 2 * asv.Tolerance) * c[i];
                    }
                }
                a.SetToRatio(b1, c);
                Assert.Equal(cv, asv.CommonValue);
                Assert.Equal(cc, asv.SparseValues.Count);
            }
        }

        [Fact]
        public void VectorSetToFunctionTests()
        {
            double[] a = new double[] { 1.2, 2.3, 3.4, 1.2, 1.2, 2.3 };
            double[] aExt = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8, 1.2, 1.2, 6.7 };
            double[] b = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8, 1.2 };
            double[] bExt = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8, 1.2, 5.6, 1.2 };
            double[] c = new double[] { 2.3, 4.5, 1.2, 1.2, 5.6, 1.2 };
            double[] cExt = new double[] { 1.2, 2.3, 4.5, 1.2, 1.2, 5.6, 1.2, 1.2 };
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            Vector[] aVector = new Vector[5];
            Vector[] bVector = new Vector[5];
            Vector[] cVector = new Vector[5];
            aVector[0] = Vector.FromArray(a);
            aVector[1] = Vector.FromArray(a, Sparsity.Sparse);
            aVector[2] = Vector.FromArray(a, approxSparsity);
            aVector[3] = DenseVector.FromArrayReference(6, aExt, 1);
            aVector[4] = Vector.FromArray(a, Sparsity.Piecewise);
            bVector[0] = Vector.FromArray(b);
            bVector[1] = Vector.FromArray(b, Sparsity.Sparse);
            bVector[2] = Vector.FromArray(b, approxSparsity);
            bVector[3] = DenseVector.FromArrayReference(6, bExt, 1);
            bVector[4] = Vector.FromArray(b, Sparsity.Piecewise);
            cVector[0] = Vector.FromArray(c);
            cVector[1] = Vector.FromArray(c, Sparsity.Sparse);
            cVector[2] = Vector.FromArray(c, approxSparsity);
            cVector[3] = DenseVector.FromArrayReference(6, cExt, 1);
            cVector[4] = Vector.FromArray(c, Sparsity.Piecewise);

            for (int i = 0; i < aVector.Length; i++)
                for (int j = 0; j < bVector.Length; j++)
                    for (int k = 0; k < cVector.Length; k++)
                        VectorSetToFunction(aVector[i], bVector[j], cVector[k]);

            // Now repeat, with one of the target instance the same as one of the source instances
            for (int j = 0; j < bVector.Length; j++)
            {
                for (int k = 0; k < cVector.Length; k++)
                {
                    var bCopy = Vector.Copy(bVector[j]);
                    VectorSetToFunction(bCopy, bCopy, cVector[k]);
                }
            }
        }

        private void VectorOperator(Vector b, Vector c)
        {
            double[] expected = new double[b.Count];
            Vector a;

            // Set to power
            for (int i = 0; i < b.Count; i++)
                expected[i] = System.Math.Pow(b[i], 2.0);
            a = b ^ 2.0;
            for (int i = 0; i < b.Count; i++)
                Assert.Equal(expected[i], a[i]);

            // Set to sum with vector
            for (int i = 0; i < b.Count; i++)
                expected[i] = b[i] + c[i];
            a = b + c;
            for (int i = 0; i < b.Count; i++)
                Assert.Equal(expected[i], a[i]);

            // Set to sum with scalar
            for (int i = 0; i < b.Count; i++)
                expected[i] = b[i] + 10.0;
            a = b + 10.0;
            for (int i = 0; i < b.Count; i++)
                Assert.Equal(expected[i], a[i]);

            // Set to difference with vector
            for (int i = 0; i < b.Count; i++)
                expected[i] = b[i] - c[i];
            a = b - c;
            for (int i = 0; i < b.Count; i++)
                Assert.Equal(expected[i], a[i]);

            // Set to difference with scalar, scalar first
            for (int i = 0; i < b.Count; i++)
                expected[i] = 10.0 - c[i];
            a = 10.0 - c;
            for (int i = 0; i < b.Count; i++)
                Assert.Equal(expected[i], a[i]);

            // Set to difference with scalar, scalar second
            for (int i = 0; i < b.Count; i++)
                expected[i] = b[i] - 10.0;
            a = b - 10.0;
            for (int i = 0; i < b.Count; i++)
                Assert.Equal(expected[i], a[i]);

            // Set to product with vector
            for (int i = 0; i < b.Count; i++)
                expected[i] = b[i] * c[i];
            a = b * c;
            for (int i = 0; i < b.Count; i++)
                Assert.Equal(expected[i], a[i]);

            // Set to product with scalar
            for (int i = 0; i < b.Count; i++)
                expected[i] = b[i] * 10.0;
            a = b * 10.0;
            for (int i = 0; i < b.Count; i++)
                Assert.Equal(expected[i], a[i]);

            // Set to product with scalar
            for (int i = 0; i < b.Count; i++)
                expected[i] = 10.0 * c[i];
            a = 10.0 * c;
            for (int i = 0; i < b.Count; i++)
                Assert.Equal(expected[i], a[i]);

            // Set to ratio with vector
            for (int i = 0; i < b.Count; i++)
                expected[i] = b[i] / c[i];
            a = b / c;
            for (int i = 0; i < b.Count; i++)
                Assert.Equal(expected[i], a[i]);

            // Unary negation
            for (int i = 0; i < b.Count; i++)
                expected[i] = -b[i];
            a = -b;
            for (int i = 0; i < b.Count; i++)
                Assert.Equal(expected[i], a[i]);
        }

        [Fact]
        public void VectorOperatorTests()
        {
            double[] b = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8, 1.2 };
            double[] bExt = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8, 1.2, 5.6, 1.2 };
            double[] c = new double[] { 2.3, 4.5, 1.2, 1.2, 5.6, 1.2 };
            double[] cExt = new double[] { 1.2, 2.3, 4.5, 1.2, 1.2, 5.6, 1.2, 1.2 };
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            Vector[] bVector = new Vector[5];
            Vector[] cVector = new Vector[5];
            bVector[0] = Vector.FromArray(b);
            bVector[1] = Vector.FromArray(b, Sparsity.Sparse);
            bVector[2] = Vector.FromArray(b, approxSparsity);
            bVector[3] = DenseVector.FromArrayReference(6, bExt, 1);
            bVector[4] = Vector.FromArray(b, Sparsity.Piecewise);
            cVector[0] = Vector.FromArray(c);
            cVector[1] = Vector.FromArray(c, Sparsity.Sparse);
            cVector[2] = Vector.FromArray(c, approxSparsity);
            cVector[3] = DenseVector.FromArrayReference(6, cExt, 1);
            cVector[4] = Vector.FromArray(c, Sparsity.Piecewise);

            for (int j = 0; j < bVector.Length; j++)
                for (int k = 0; k < cVector.Length; k++)
                    VectorOperator(bVector[j], cVector[k]);
        }

        private void VectorReduce(Vector thisVector, Vector thatVector, double initial)
        {
            double expected = 0.0;
            double actual = 0.0;

            // Reduce
            expected = initial;
            for (int i = 0; i < thisVector.Count; i++)
                expected += thisVector[i];
            actual = thisVector.Reduce(initial, (x, y) => x + y);
            Assert.Equal(expected, actual);

            // Sum
            expected = 0.0;
            for (int i = 0; i < thisVector.Count; i++)
                expected += thisVector[i];
            actual = thisVector.Sum();
            Assert.Equal(expected, actual);

            // Sum of func
            expected = 0.0;
            for (int i = 0; i < thisVector.Count; i++)
                expected += System.Math.Exp(thisVector[i]);
            actual = thisVector.Sum(System.Math.Exp);
            Assert.Equal(expected, actual);

            // Filtered sum of func
            double mean = thatVector.Sum() / thatVector.Count;
            expected = 0.0;
            for (int i = 0; i < thisVector.Count; i++)
                if (thatVector[i] < mean)
                    expected += System.Math.Exp(thisVector[i]);
            actual = thisVector.Sum(System.Math.Exp, thatVector, x => x < mean);
            Assert.Equal(expected, actual);

            // SumI
            expected = 0.0;
            for (int i = 0; i < thisVector.Count; i++)
                expected += i * thisVector[i];
            actual = thisVector.SumI();
            Assert.Equal(expected, actual);

            // SumISq
            expected = 0.0;
            for (int i = 0; i < thisVector.Count; i++)
                expected += i * i * thisVector[i];
            actual = thisVector.SumISq();
            Assert.Equal(expected, actual, 1e-11);

            // Max
            expected = double.MinValue;
            for (int i = 0; i < thisVector.Count; i++)
                if (thisVector[i] > expected)
                    expected = thisVector[i];
            actual = thisVector.Max();
            Assert.Equal(expected, actual);

            // Max func
            mean = thisVector.Sum() / thisVector.Count;
            expected = double.MinValue;
            for (int i = 0; i < thisVector.Count; i++)
                if (System.Math.Abs(thisVector[i] - mean) > expected)
                    expected = System.Math.Abs(thisVector[i] - mean);
            actual = thisVector.Max(x => System.Math.Abs(x - mean));
            Assert.Equal(expected, actual);

            // Min
            expected = double.MaxValue;
            for (int i = 0; i < thisVector.Count; i++)
                if (thisVector[i] < expected)
                    expected = thisVector[i];
            actual = thisVector.Min();
            Assert.Equal(expected, actual);

            // Min func
            expected = double.MaxValue;
            for (int i = 0; i < thisVector.Count; i++)
                if (System.Math.Abs(thisVector[i] - mean) < expected)
                    expected = System.Math.Abs(thisVector[i] - mean);
            actual = thisVector.Min(x => System.Math.Abs(x - mean));
            Assert.Equal(expected, actual);

            // Log sum exp
            double Z = 0.0;
            for (int i = 0; i < thisVector.Count; i++)
                Z += System.Math.Exp(thisVector[i]);
            expected = System.Math.Log(Z);
            actual = thisVector.LogSumExp();
            Assert.Equal(expected, actual);

            // Reduce with second vector
            expected = initial;
            for (int i = 0; i < thisVector.Count; i++)
                expected += thisVector[i] * thatVector[i];
            actual = thisVector.Reduce(initial, thatVector, (x, y, z) => x + y * z);
            Assert.Equal(expected, actual);

            // Inner product
            expected = 0.0;
            for (int i = 0; i < thisVector.Count; i++)
                expected += thisVector[i] * thatVector[i];
            actual = thisVector.Inner(thatVector);
            Assert.Equal(expected, actual);

            // Inner product fun
            expected = 0.0;
            for (int i = 0; i < thisVector.Count; i++)
                expected += thisVector[i] * System.Math.Exp(thatVector[i]);
            actual = thisVector.Inner(thatVector, System.Math.Exp);
            Assert.Equal(expected, actual);

            // Inner product fun fun
            expected = 0.0;
            for (int i = 0; i < thisVector.Count; i++)
                expected += System.Math.Pow(thisVector[i], 2.0) * System.Math.Exp(thatVector[i]);
            actual = thisVector.Inner(x => System.Math.Pow(x, 2.0), thatVector, System.Math.Exp);
            Assert.Equal(expected, actual);

            // Static inner product
            expected = 0.0;
            for (int i = 0; i < thisVector.Count; i++)
                expected += thisVector[i] * thatVector[i];
            actual = Vector.InnerProduct(thisVector, thatVector);
            Assert.Equal(expected, actual);

            // Outer product
            expected = 0.0;
            Vector a = thisVector.Subvector(1, 3);
            Vector b = thatVector.Subvector(1, 2);

            Matrix expectedMat = new PositiveDefiniteMatrix(a.Count, b.Count);
            for (int i = 0; i < a.Count; i++)
                for (int j = 0; j < b.Count; j++)
                    expectedMat[i, j] = a[i] * b[j];
            Matrix actualMat = a.Outer(b);
            for (int i = 0; i < a.Count; i++)
                for (int j = 0; j < b.Count; j++)
                    Assert.Equal(expectedMat[i, j], actualMat[i, j], 1e-7);

            // Any
            try
            {
                bool expectedAnyResult = false;
                for (int i = 0; i < thisVector.Count; i++)
                {
                    expectedAnyResult |= thisVector[i] > thatVector[i];
                }

                bool actualAnyResult = thisVector.Any(thatVector, (x, y) => x > y);
                Assert.Equal(expectedAnyResult, actualAnyResult);
            }
            catch (NotImplementedException)
            {
                // Not every case is implemented, but those that are should work
            }
        }

        [Fact]
        public void VectorReduceTests()
        {
            double[] a = new double[] { 1.2, 2.3, 3.4, 1.2, 1.2, 2.3 };
            double[] aExt = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8, 1.2, 1.2, 6.7 };
            double[] b = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8, 1.2 };
            double[] bExt = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8, 1.2, 5.6, 1.2 };
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            Vector[] aVector = new Vector[5];
            Vector[] bVector = new Vector[5];
            aVector[0] = Vector.FromArray(a);
            aVector[1] = Vector.FromArray(a, approxSparsity);
            aVector[2] = Vector.FromArray(a, Sparsity.Sparse);
            aVector[3] = DenseVector.FromArrayReference(6, aExt, 1);
            aVector[4] = Vector.FromArray(a, Sparsity.Piecewise);
            bVector[0] = Vector.FromArray(b);
            bVector[1] = Vector.FromArray(b, Sparsity.Sparse);
            bVector[2] = Vector.FromArray(b, approxSparsity);
            bVector[3] = DenseVector.FromArrayReference(6, bExt, 1);
            bVector[4] = Vector.FromArray(b, Sparsity.Piecewise);
            for (int i = 0; i < aVector.Length; i++)
                for (int j = 0; j < bVector.Length; j++)
                    VectorReduce(aVector[i], bVector[j], 0.1);
        }

        private void VectorSparseReduce(SparseVector thisVector, ISparseList<double> b, ISparseList<int> c)
        {
            double delta = 1e-9;
            double expected = 0.0;
            double actual = 0.0;
            double initial;
            // Single argument sum
            initial = 1.0;
            expected = initial;
            for (int i = 0; i < thisVector.Count; i++)
                expected += thisVector[i];

            actual = thisVector.Reduce(initial, (x, y) => x + y, (x, y, count) => x + count * y);
            Assert.Equal(expected, actual, delta);

            // Single argument max
            initial = double.MinValue;
            expected = initial;
            for (int i = 0; i < thisVector.Count; i++)
                expected = System.Math.Max(expected, thisVector[i]);

            actual = thisVector.Reduce(initial, (x, y) => System.Math.Max(x, y), (x, y, count) => System.Math.Max(x, y));
            Assert.Equal(expected, actual, delta);

            // Double argument sum of product
            initial = 1.0;
            expected = initial;
            for (int i = 0; i < thisVector.Count; i++)
                expected += thisVector[i] * b[i];
            actual = thisVector.Reduce(initial, b, (x, y, z) => x + y * z, (x, y, z, count) => x + count * y * z);
            Assert.Equal(expected, actual, delta);

            // Double argument max of product
            initial = double.MinValue;
            expected = initial;
            for (int i = 0; i < thisVector.Count; i++)
                expected = System.Math.Max(expected, thisVector[i] * b[i]);
            actual = thisVector.Reduce(initial, b, (x, y, z) => System.Math.Max(x, y * z), (x, y, z, count) => System.Math.Max(x, y * z));
            Assert.Equal(expected, actual, delta);

            // Triple argument - sum of product to a power
            initial = 1.0;
            expected = initial;
            for (int i = 0; i < thisVector.Count; i++)
                expected += System.Math.Pow(thisVector[i] * b[i], c[i]);
            actual = thisVector.Reduce(initial, b, c, (w, x, y, z) => w + System.Math.Pow(x * y, z), (w, x, y, z, count) => w + count * System.Math.Pow(x * y, z));
            Assert.Equal(expected, actual, delta);

            // Triple argument - max of product to a power
            initial = 1.0;
            expected = initial;
            for (int i = 0; i < thisVector.Count; i++)
                expected = System.Math.Max(expected, System.Math.Pow(thisVector[i] * b[i], c[i]));
            actual = thisVector.Reduce(initial, b, c, (w, x, y, z) => System.Math.Max(w, System.Math.Pow(x * y, z)), (w, x, y, z, count) => System.Math.Max(w, System.Math.Pow(x * y, z)));
            Assert.Equal(expected, actual, delta);
        }

        [Fact]
        public void VectorSparseReduceTests()
        {
            Rand.Restart(12347);
            int len = 200;
            for (int i = 0; i < 100; i++)
            {
                // Create random sparse vectors for the reduce operations
                // Random numbers of non-common values are in random positions. 
                double aCom = 2.0 * Rand.Double() - 1.0;
                double bCom = 2.0 * Rand.Double() - 1.0;
                int cCom = Rand.Int(0, 3);
                var a = SparseVector.Constant(len, aCom);
                var b = SparseVector.Constant(len, bCom);
                var c = SparseList<int>.Constant(len, cCom);

                int numA = Rand.Poisson(20);
                int numB = Rand.Poisson(20);
                int numC = Rand.Poisson(20);

                int[] permA = Rand.Perm(len);
                int[] permB = Rand.Perm(len);
                int[] permC = Rand.Perm(len);

                for (int j = 0; j < numA; j++)
                {
                    double aval = 2.0 * Rand.Double() - 1.0;
                    a[permA[j]] = aval;
                }
                for (int j = 0; j < numB; j++)
                {
                    double bval = 2.0 * Rand.Double() - 1.0;
                    b[permB[j]] = bval;
                }
                for (int j = 0; j < numC; j++)
                {
                    int cval = Rand.Int(0, 3);
                    c[permC[j]] = cval;
                }
                VectorSparseReduce(a, b, c);
            }
        }

        [Fact]
        public void VectorEqualityTests()
        {
            double[] a = new double[] { 1.2, 2.3, 3.4, 1.2, 1.2, 2.3, double.PositiveInfinity, double.NegativeInfinity };
            double[] aExt = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8, 1.2, 1.2, 6.7 };
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            VectorEquality(Vector.FromArray(a));
            VectorEquality(Vector.FromArray(a, Sparsity.Sparse));
            VectorEquality(Vector.FromArray(a, approxSparsity));
            VectorEquality(Vector.FromArray(a, Sparsity.Piecewise));
            VectorEquality(DenseVector.FromArrayReference(6, aExt, 1));

            void VectorEquality(Vector v)
            {
                Vector copy = Vector.Copy(v);
                Assert.True(v == copy);
                Assert.False(v != copy);
                Assert.Equal(v, copy);
                Assert.Equal(0.0, v.MaxDiff(copy));

                double delta = 0.25;
                Vector noisy = v.Clone();
                noisy[0] += delta;
                Assert.False(v == noisy);
                Assert.True(v != noisy);
                Assert.NotEqual(Vector.Zero(noisy.Count), noisy);
                double diffExpected = delta / (v[0] + TOLERANCE);
                double diff = v.MaxDiff(noisy, TOLERANCE);
                Assert.Equal(diffExpected, diff);

                v[0] = double.NaN;
                Vector withNaN = v.Clone();
                Assert.Equal(0.0, v.MaxDiff(withNaN));
            }
        }

        [Fact]
        public void VectorInequalityTests()
        {
            double[] a = new double[] { 1.2, 2.3, 3.4, 1.2, 1.2, 2.3 };
            double[] aExt = new double[] { 5.6, 1.2, 1.2, 6.7, 7.8, 1.2, 1.2, 6.7 };
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            VectorInequality(Vector.FromArray(a));
            VectorInequality(Vector.FromArray(a, Sparsity.Sparse));
            VectorInequality(Vector.FromArray(a, approxSparsity));
            VectorInequality(Vector.FromArray(a, Sparsity.Piecewise));
            VectorInequality(DenseVector.FromArrayReference(6, aExt, 1));

            void VectorInequality(Vector v)
            {
                double min = v.Min();
                double max = v.Max();
                Assert.True(v.All(x => x < max + 0.1));
                Assert.True(v.Any(x => x > max - 0.1));

                int i2 = v.Count / 2;

                Vector b = Vector.Copy(v);
                b[i2] = v[i2] + 0.1;

                Assert.True(b >= v);
                Assert.False(b > v);
                Assert.False(b < v);
                Assert.False(b <= v);
                Assert.True(v <= b);
                Assert.False(v < b);
                Assert.False(v > b);
                Assert.False(v >= b);

                for (int i = 0; i < v.Count; i++)
                    b[i] = v[i] + 0.1;
                Assert.True(b >= v);
                Assert.True(b > v);
                Assert.False(b < v);
                Assert.False(b <= v);
                Assert.True(v <= b);
                Assert.True(v < b);
                Assert.False(v > b);
                Assert.False(v >= b);

                for (int i = 0; i < v.Count; i++)
                    b[i] = (i % 2) == 0 ? v[i] + 0.1 : v[i] - 0.1;
                Assert.False(b >= v);
                Assert.False(b > v);
                Assert.False(b < v);
                Assert.False(b <= v);
                Assert.False(v <= b);
                Assert.False(v < b);
                Assert.False(v > b);
                Assert.False(v >= b);

                double d = 0.5 * (min + max);
                Assert.False(v <= d);
                Assert.False(v < d);
                Assert.False(v > d);
                Assert.False(v >= d);

                Assert.True(v <= max);
                Assert.True(v < max + 0.1);
                Assert.False(v < max);
                Assert.False(v > max);
                Assert.False(v >= max);

                Assert.True(v >= min);
                Assert.True(v > min - 0.1);
                Assert.False(v > min);
                Assert.False(v < min);
                Assert.False(v <= min);

                v[0] = double.NegativeInfinity;
                v[1] = double.PositiveInfinity;
                b.SetTo(v);
                Assert.True(b >= v);
                Assert.True(b <= v);
                Assert.False(b > v);
                Assert.False(b < v);
            }
        }

        [Fact]
        public void DeterminantTest()
        {
            Matrix m = Matrix.Parse("3.0 2.0" + Environment.NewLine + "1.5 3.0");
            double determinant = m.Determinant();
            const double expectedDeterminant = 6.0;
            Assert.Equal(expectedDeterminant, determinant);
        }

        // b must be [4,4,4], c must be [1,1,1]
        private void Predivide(Vector b, Vector c)
        {
            // rectangular matrix tests
            Matrix R = new Matrix(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            Matrix Rt = R.Transpose();
            Matrix Rtt = new Matrix(R.Rows, R.Cols);
            Rtt.SetToTranspose(Rt);
            Assert.True(Rtt.Equals(R));
            Assert.True(Rtt.GetHashCode() == R.GetHashCode());

            Matrix RRTrue = new Matrix(new double[,] { { 14, 32 }, { 32, 77 } });
            Matrix RR = new Matrix(R.Rows, Rt.Cols);
            RR.SetToProduct(R, Rt);
            //Rt.SetToProduct(R,Rt); RR.SetTo(Rt);
            Assert.True(RR.MaxDiff(RRTrue) < TOLERANCE);

            RR.SetTo(RRTrue);
            Assert.Equal(RR, RRTrue);
            Assert.Equal(RR.GetHashCode(), RRTrue.GetHashCode());

            RRTrue[0, 1] = RRTrue[0, 1] - 1;
            RR[1, 0] = RR[1, 0] - 1;
            RR.SetToTranspose(RR); // in-place transpose
            Assert.True(RR.Equals(RRTrue));

            // square matrix tests
            // A = [2 1 1; 1 2 1; 1 1 2]
            PositiveDefiniteMatrix A = new PositiveDefiniteMatrix(new double[,] { { 2, 1, 1 }, { 1, 2, 1 }, { 1, 1, 2 } });
            Assert.True(A.SymmetryError() == 0.0);

            double det = A.Determinant();
            Assert.True(System.Math.Abs(det - 4) < TOLERANCE);

            PositiveDefiniteMatrix AinvTrue = new PositiveDefiniteMatrix(new double[,]
                {
                    {0.75, -0.25, -0.25},
                    {-0.25, 0.75, -0.25},
                    {-0.25, -0.25, 0.75}
                });
            PositiveDefiniteMatrix Ainv = new PositiveDefiniteMatrix(3, 3);
            Ainv.SetToInverse(A);
            Assert.True(Ainv.MaxDiff(AinvTrue) < TOLERANCE);
            Assert.True(Ainv.SymmetryError() == 0.0);

            Vector v = Ainv * b;
            Assert.True(v.MaxDiff(c) < TOLERANCE);

            v.SetTo(b);
            Assert.Equal(v, b);
            //Assert.Equal(v.GetHashCode(), b.GetHashCode());

            (new LuDecomposition(A)).Solve((DenseVector)v);
            // should be same as above
            Assert.True(v.MaxDiff(c) < TOLERANCE);
            Vector b2 = Vector.Zero(b.Count);
            b2.SetToProduct(A, v);
            Assert.True(b.MaxDiff(b2) < TOLERANCE);

            LowerTriangularMatrix LTrue = new LowerTriangularMatrix(new double[,]
                {
                    {1.414213562373095, 0, 0},
                    {0.707106781186547, 1.224744871391589, 0},
                    {0.707106781186547, 0.408248290463863, 1.154700538379252}
                });
            LowerTriangularMatrix L = new LowerTriangularMatrix(3, 3);
            bool isPosDef = L.SetToCholesky(A);
            Assert.True(isPosDef);
            Assert.True(L.MaxDiff(LTrue) < TOLERANCE);

            LowerTriangularMatrix LinvTrue = new LowerTriangularMatrix(new double[,]
                {
                    {0.707106781186547, 0, 0},
                    {-0.408248290463863, 0.816496580927726, 0},
                    {-0.288675134594813, -0.288675134594813, 0.866025403784439}
                });
            LowerTriangularMatrix Linv = new LowerTriangularMatrix(3, 3);
            Linv.SetToInverse(L);
            Assert.True(Linv.MaxDiff(LinvTrue) < TOLERANCE);
            Linv.SetTo(L);
            Linv.SetToInverse(Linv);
            Assert.True(Linv.MaxDiff(LinvTrue) < TOLERANCE);

            // L*L' = A, so L'\L\A = I
            PositiveDefiniteMatrix eye = new PositiveDefiniteMatrix(3, 3);
            eye.SetTo(A);
            eye.PredivideBy(L);
            eye.PredivideBy(L.Transpose());
            Assert.True(eye.MaxDiff(Matrix.Identity(3)) < TOLERANCE);
            eye.SetTo(A);
            eye.PredivideBy(L);
            eye.PredivideByTranspose(L);
            Assert.True(eye.MaxDiff(Matrix.Identity(3)) < TOLERANCE);

            // L*L' = A, so inv(A)*b = inv(L')*inv(L)*b
            v.SetTo(b);
            v.PredivideBy(L);
            v.PredivideBy(L.Transpose());
            Assert.True(v.MaxDiff(c) < TOLERANCE);

            v.SetTo(b);
            v.PredivideBy(L);
            v.PredivideByTranspose(L);
            Assert.True(v.MaxDiff(c) < TOLERANCE);
        }

        [Fact]
        public void PredivideTests()
        {
            double[] b = new double[] { 4.0, 4.0, 4.0 };
            double[] bExt = new double[] { 5.6, 4.0, 4.0, 4.0, 5.6 };
            double[] c = new double[] { 1.0, 1.0, 1.0 };
            double[] cExt = new double[] { 1.2, 1.0, 1.0, 1.0, 1.2 };
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            Vector[] bVector = new Vector[5];
            Vector[] cVector = new Vector[5];
            bVector[0] = Vector.FromArray(b);
            bVector[1] = Vector.FromArray(b, Sparsity.Sparse);
            bVector[2] = Vector.FromArray(b, approxSparsity);
            bVector[3] = DenseVector.FromArrayReference(3, bExt, 1);
            bVector[4] = Vector.FromArray(b, Sparsity.Piecewise);
            cVector[0] = Vector.FromArray(c);
            cVector[1] = Vector.FromArray(c, Sparsity.Sparse);
            cVector[2] = Vector.FromArray(c, approxSparsity);
            cVector[3] = DenseVector.FromArrayReference(3, cExt, 1);
            cVector[4] = Vector.FromArray(c, Sparsity.Piecewise);

            for (int j = 0; j < bVector.Length; j++)
                for (int k = 0; k < cVector.Length; k++)
                    Predivide(bVector[j], cVector[k]);
        }

        [Fact]
        public void SetToLeastSquaresTest()
        {
            Matrix X = Matrix.FromArray(new double[,] { { 1, -2 }, { 1, -1 } });
            DenseVector Y = DenseVector.FromArray(double.MinValue, -31);
            DenseVector Yorig = (DenseVector)Y.Clone();
            DenseVector A = DenseVector.Zero(2);
            A.SetToLeastSquares(Y, X);
            // X[1] - X[0] = {0, 1} therefore A[1] = Y[1]-Y[0]
            //A[1] = Y[1] - Y[0];
            //A[0] = (Y[1] - X[1,1] * A[1])/X[1,0];
            // The above does not yield Y=X*A, therefore there is no exact solution.
            // The solution that minimizes squared error is:
            //A[1] = double.MaxValue / 2;
            //A[0] = double.MaxValue / 2;
            DenseVector Y2 = DenseVector.Zero(2);
            Y2.SetToProduct(X, A);
            double maxdiff = Y.MaxDiff(Y2);
            Assert.True(maxdiff < double.MaxValue * 0.66);
            Assert.True(Yorig.Equals(Y));
        }

        private void MatrixVector(Matrix M, Vector v, Sparsity resultSparsity)
        {
            int rows = M.Rows;
            int cols = M.Cols;
            Vector MvExpected = Vector.Zero(rows); // Make this dense
            Vector MvActual = Vector.Zero(rows, resultSparsity);
            for (int r = 0; r < rows; r++)
            {
                double sum = 0.0;
                for (int c = 0; c < cols; c++)
                    sum += M[r, c] * v[c];
                MvExpected[r] = sum;
            }
            MvActual.SetToProduct(M, v);
            Assert.Equal(MvExpected, MvActual);
        }

        private void VectorMatrix(Vector v, Matrix M, Sparsity resultSparsity)
        {
            int rows = M.Rows;
            int cols = M.Cols;
            Vector vMExpected = Vector.Zero(cols); // Make this dense
            Vector vMActual = Vector.Zero(cols, resultSparsity);
            for (int c = 0; c < cols; c++)
            {
                double sum = 0.0;
                for (int r = 0; r < rows; r++)
                {
                    sum += v[r] * M[r, c];
                }
                vMExpected[c] = sum;
            }
            vMActual.SetToProduct(v, M);
            Assert.Equal(vMExpected, vMActual);
        }

        [Fact]
        public void MatrixVectorTests()
        {
            // The vector
            double[] a = new double[] { 1.2, 2.3, 1.2 };
            double[] aExt = new double[] { 5.6, 1.2, 2.3, 1.2, 6.7 };
            Sparsity approxSparsity = Sparsity.ApproximateWithTolerance(0.001);
            Vector[] aVector = new Vector[5];
            aVector[0] = Vector.FromArray(a);
            aVector[1] = Vector.FromArray(a, Sparsity.Sparse);
            aVector[2] = Vector.FromArray(a, approxSparsity);
            aVector[3] = DenseVector.FromArrayReference(3, aExt, 1);
            aVector[4] = Vector.FromArray(a, Sparsity.Piecewise);

            // The matrices
            Matrix ML = new Matrix(4, 3);
            Matrix MR = new Matrix(3, 4);
            Rand.Restart(12347);
            for (int i = 0; i < ML.Rows; i++)
                for (int j = 0; j < ML.Cols; j++)
                    ML[i, j] = Rand.Double();
            for (int i = 0; i < MR.Rows; i++)
                for (int j = 0; j < MR.Cols; j++)
                    MR[i, j] = Rand.Double();

            for (int i = 0; i < aVector.Length; i++)
            {
                VectorMatrix(aVector[i], MR, Sparsity.Dense);
                MatrixVector(ML, aVector[i], Sparsity.Dense);
                VectorMatrix(aVector[i], MR, Sparsity.Sparse);
                MatrixVector(ML, aVector[i], Sparsity.Sparse);
                VectorMatrix(aVector[i], MR, Sparsity.Piecewise);
                MatrixVector(ML, aVector[i], Sparsity.Piecewise);
            }
        }

        [Fact]
        public void VectorSetTo_WithWrongArraySize_ThrowsException()
        {
            Assert.Throws<ArgumentException>(() =>
            {

                Vector x = Vector.Zero(3);
                double[] array = new double[] { 1, 2, 3 };
                x.SetTo(array);
                for (int i = 0; i < array.Length; i++)
                {
                    Assert.Equal(array[i], x[i]);
                }

                x.SetTo(new double[] { 1, 2 });

            });
        }

        [Fact]
        public void OuterTransposeTest()
        {
            Matrix big = new Matrix(100, 100);
            for (int i = 0; i < big.Rows; i++)
            {
                for (int j = 0; j < big.Cols; j++)
                {
                    big[i, j] = Rand.Double();
                }
            }
            Matrix bigt = big.Transpose();
            Matrix bigr = new Matrix(100, 100);
            bigr.SetToOuterTranspose(big);
            Assert.True(bigr.MaxDiff(bigt * big) < TOLERANCE);
        }

        [Fact]
        public void MatrixTest()
        {
            // rectangular matrix tests
            Matrix R = new Matrix(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            Matrix Rt = R.Transpose();
            Matrix Rtt = new Matrix(R.Rows, R.Cols);
            Rtt.SetToTranspose(Rt);
            Assert.True(Rtt.Equals(R));
            Assert.True(Rtt.GetHashCode() == R.GetHashCode());

            Matrix RRTrue = new Matrix(new double[,] { { 14, 32 }, { 32, 77 } });
            Matrix RR = new Matrix(R.Rows, Rt.Cols);
            RR.SetToProduct(R, Rt);
            //Rt.SetToProduct(R,Rt); RR.SetTo(Rt);
            Assert.True(RR.MaxDiff(RRTrue) < TOLERANCE);
            RR.SetToOuter(R);
            Assert.True(RR.MaxDiff(RRTrue) < TOLERANCE);
            RR.SetToOuterTranspose(Rt);
            Assert.True(RR.MaxDiff(RRTrue) < TOLERANCE);

            RR.SetTo(RRTrue);
            Assert.Equal(RR, RRTrue);
            Assert.Equal(RR.GetHashCode(), RRTrue.GetHashCode());

            RRTrue[0, 1] = RRTrue[0, 1] - 1;
            RR[1, 0] = RR[1, 0] - 1;
            RR.SetToTranspose(RR); // in-place transpose
            Assert.True(RR.Equals(RRTrue));

            // square matrix tests
            // A = [2 1 1; 1 2 1; 1 1 2]
            PositiveDefiniteMatrix A = new PositiveDefiniteMatrix(new double[,] { { 2, 1, 1 }, { 1, 2, 1 }, { 1, 1, 2 } });
            Assert.True(A.SymmetryError() == 0.0);

            double det = A.Determinant();
            Assert.True(System.Math.Abs(det - 4) < TOLERANCE);

            PositiveDefiniteMatrix AinvTrue = new PositiveDefiniteMatrix(new double[,]
                {
                    {0.75, -0.25, -0.25},
                    {-0.25, 0.75, -0.25},
                    {-0.25, -0.25, 0.75}
                });
            PositiveDefiniteMatrix Ainv = new PositiveDefiniteMatrix(3, 3);
            Ainv.SetToInverse(A);
#if LAPACK
            var t = new Matrix(A);
            Lapack.SymmetricInverseInPlace(t); 
            Assert.True(Ainv.MaxDiff(t) < TOLERANCE);
#endif
            Assert.True(Ainv.MaxDiff(AinvTrue) < TOLERANCE);
            Assert.True(Ainv.SymmetryError() == 0.0);

            Vector b = Vector.FromArray(new double[] { 4, 4, 4 });
            Vector c = Vector.FromArray(new double[] { 1, 1, 1 });
            Vector v = Ainv * b;
            Assert.True(v.MaxDiff(c) < TOLERANCE);

            v.SetTo(b);
            Assert.Equal(v, b);
            Assert.Equal(v.GetHashCode(), b.GetHashCode());

            (new LuDecomposition(A)).Solve(v);
            // should be same as above
            Assert.True(v.MaxDiff(c) < TOLERANCE);
            Vector b2 = Vector.Zero(b.Count);
            b2.SetToProduct(A, v);
            Assert.True(b.MaxDiff(b2) < TOLERANCE);

            LowerTriangularMatrix LTrue = new LowerTriangularMatrix(new double[,]
                {
                    {1.414213562373095, 0, 0},
                    {0.707106781186547, 1.224744871391589, 0},
                    {0.707106781186547, 0.408248290463863, 1.154700538379252}
                });
            LowerTriangularMatrix L = new LowerTriangularMatrix(3, 3);
            bool isPosDef = L.SetToCholesky(A);
            Assert.True(isPosDef);
            Assert.True(L.MaxDiff(LTrue) < TOLERANCE);

            PositiveDefiniteMatrix Acopy = (PositiveDefiniteMatrix)A.Clone();
            L = Acopy.CholeskyInPlace(out isPosDef);
            Assert.True(isPosDef);
            Assert.True(L.MaxDiff(LTrue) < TOLERANCE);

            LowerTriangularMatrix LinvTrue = new LowerTriangularMatrix(new double[,]
                {
                    {0.707106781186547, 0, 0},
                    {-0.408248290463863, 0.816496580927726, 0},
                    {-0.288675134594813, -0.288675134594813, 0.866025403784439}
                });
            LowerTriangularMatrix Linv = new LowerTriangularMatrix(3, 3);
            Linv.SetToInverse(L);
            Assert.True(Linv.MaxDiff(LinvTrue) < TOLERANCE);
            Linv.SetTo(L);
            Linv.SetToInverse(Linv);
            Assert.True(Linv.MaxDiff(LinvTrue) < TOLERANCE);

            // L*L' = A, so L'\L\A = I
            PositiveDefiniteMatrix eye = new PositiveDefiniteMatrix(3, 3);
            eye.SetTo(A);
            eye.PredivideBy(L);
            eye.PredivideBy(L.Transpose());
            Assert.True(eye.MaxDiff(Matrix.Identity(3)) < TOLERANCE);
            eye.SetTo(A);
            eye.PredivideBy(L);
            eye.PredivideByTranspose(L);
            Assert.True(eye.MaxDiff(Matrix.Identity(3)) < TOLERANCE);

            // L*L' = A, so inv(A)*b = inv(L')*inv(L)*b
            v.SetTo(b);
            v.PredivideBy(L);
            v.PredivideBy(L.Transpose());
            Assert.True(v.MaxDiff(c) < TOLERANCE);

            v.SetTo(b);
            v.PredivideBy(L);
            v.PredivideByTranspose(L);
            Assert.True(v.MaxDiff(c) < TOLERANCE);

            UpperTriangularMatrix U = L.Transpose();
            LowerTriangularMatrix L2 = U.Transpose();
            Assert.Equal(L, L2);

            Assert.True(R.Equals(Matrix.Parse(R.ToString())));
            Assert.True(R.Equals(Matrix.Parse(R.ToString("g"))));
        }

        [Fact]
        public void MatrixLeastSquaresTest()
        {
            // in matlab:
            // X = [1 0.7; 1 0.8; 1 0.9]
            // Y = [2 3; 4 5; 6 7]
            // Y - X*(X\Y)
            Matrix X = new Matrix(new double[,] {
                { 1,  0.7 },
                { 1,  0.8 },
                { 1,  0.9 }
            });
            Matrix Y = new Matrix(new double[,] {
                { 2, 3 },
                { 4, 5 },
                { 6, 7 }
            });
            Matrix A = new Matrix(2, 2);
            A.SetToLeastSquares(Y, X);

            var residual = new Matrix(Y.Rows, Y.Cols);
            residual.SetToProduct(X, A);
            residual.SetToDifference(Y, residual);
            Assert.True(residual.All(x => System.Math.Abs(x) < 1e-14));
        }

        [Fact]
        public void MatrixLeastSquaresTest2()
        {
            // in matlab:
            // X = [1,  0.767680972352692; 1,  0.767680973268405; 1,  0.767680974304166]
            // Y = [2 3; 4 5; 6 7]
            // Y - X*(X\Y)
            Matrix X = new Matrix(new double[,] {
                { 1,  0.767680972352692 },
                { 1,  0.767680973268405 },
                { 1,  0.767680974304166 }
            });
            Matrix Y = new Matrix(new double[,] {
                { 2, 3 },
                { 4, 5 },
                { 6, 7 }
            });
            Matrix A = new Matrix(2, 2);
            A.SetToLeastSquares(Y, X);

            var residual = new Matrix(Y.Rows, Y.Cols);
            residual.SetToProduct(X, A);
            residual.SetToDifference(Y, residual);
            Assert.True(residual.All(x => System.Math.Abs(x) < 1e-1));
        }

        [Fact]
        public void MatrixSvdTest()
        {
            Matrix A = new Matrix(new double[,] {
                { 1.000000000000000,  0.767680972352692 },
                { 1.000000000000000,  0.767680973268405 },
                { 1.000000000000000,  0.767680974304166 }
            });
            Matrix V = new Matrix(A.Cols, A.Cols);
            V.SetToRightSingularVectors(A);
            // A now contains the left singular vectors scaled by the singular values.
            Matrix US = A;
            DenseVector S = DenseVector.Zero(A.Cols);
            for (int i = 0; i < A.Cols; i++)
            {
                double sum = 0;
                double compensation = 0;
                for (int j = 0; j < A.Rows; j++)
                {
                    double y = A[j, i] * A[j, i] - compensation;
                    double nextSum = sum + y;
                    compensation = (nextSum - sum) - y;
                    sum = nextSum;
                }
                S[i] = System.Math.Sqrt(sum);
            }
            var Sinv = DenseVector.Zero(S.Count);
            Sinv.SetToFunction(S, x => 1.0 / x);
            Matrix U = (Matrix)US.Clone();
            U.ScaleCols(Sinv);

            // these results are slightly different from Matlab, but seem to be more accurate.
            // They were computed in extended precision.
            Matrix UExpected = new Matrix(new double[,] {
                { 0.57735026892309971,  -0.69217076042542958 },
                { 0.57735026917846644,  -0.028980919646426295 },
                { 0.57735026946731116,   0.72115167940491542 }
            });
            DenseVector SExpected = DenseVector.FromArray(2.1835755609411125, 1.0952516532160563e-09);
            Matrix VExpected = new Matrix(new double[,] {
                { 0.79321771069024527,   -0.60893814418816494 },
                { 0.60893814418816494,  0.79321771069024527 }
            });

            double UError = UExpected.MaxDiff(U);
            double SError = SExpected.MaxDiff(S);
            double VError = VExpected.MaxDiff(V);
            Matrix USExpected = UExpected * Matrix.FromDiagonal(SExpected);
            double USError = USExpected.MaxDiff(US);
            //Console.WriteLine(StringUtil.JoinColumns("US = ", US.ToString("g17"), " expected ", USExpected.ToString("g17"), " error ", USError));
            //Console.WriteLine(StringUtil.JoinColumns("U = ", U.ToString("g17"), " expected ", UExpected.ToString("g17"), " error ", UError));
            //Console.WriteLine(StringUtil.JoinColumns("S = ", S, " expected ", SExpected, " error ", SError));
            //Console.WriteLine(StringUtil.JoinColumns("V = ", V, " expected ", VExpected, " error ", VError));
            Matrix AExpected = UExpected * Matrix.FromDiagonal(SExpected) * VExpected.Transpose();
            Matrix USVActual = U * Matrix.FromDiagonal(S) * V.Transpose();
            Matrix AActual = US * V.Transpose();
            //Console.WriteLine(StringUtil.JoinColumns("A = ", AActual.ToString("g17"), " expected ", AExpected.ToString("g17")));
            Assert.True(UError < 1e-7);
            Assert.True(SError < 1e-10);
            Assert.True(VError < 1e-10);
        }

        // TODO: change this test to use SetToLeftSingularVectors
        //[Fact]
        internal void MatrixSvdTest2()
        {
            Matrix A = new Matrix(new double[,] {
                { 1.000000000000000,  0.767680972352692 },
                { 1.000000000000000,  0.767680973268405 },
                { 1.000000000000000,  0.767680974304166 }
            }).Transpose();
            Matrix V = new Matrix(A.Cols, A.Cols);
            V.SetToRightSingularVectors(A);
            //Matrix A2 = new Matrix(A.Rows, A.Cols);
            //A2.SetToProduct(A, V.Transpose());
            DenseVector S = DenseVector.Zero(A.Cols);
            for (int i = 0; i < A.Cols; i++)
            {
                double sum = 0;
                for (int j = 0; j < A.Rows; j++)
                {
                    sum += A[j, i] * A[j, i];
                }
                S[i] = System.Math.Sqrt(sum);
            }
            var Sinv = DenseVector.Zero(S.Count);
            Sinv.SetToFunction(S, x => (x == 0) ? 1.0 : 1.0 / x);
            A.ScaleCols(Sinv);
            Matrix U = A;

            // these results are slightly different from Matlab, but seem to be more accurate.
            Matrix VExpected = new Matrix(new double[,] {
                { 0.43308922367497854, 0.577350268923100,  -0.69217074896108965 },
                { -0.81598209252455245, 0.577350269178466,  -0.028980941246352369 },
                { 0.38289286884957391, 0.577350269467311,   0.72115168954050146 }
            });
            DenseVector SExpected = DenseVector.FromArray(0, 2.183575560941113, 0.000000001095252);
            Matrix UExpected = new Matrix(new double[,] {
                { 0 /* -0.78350323396415822 */, 0.793217710690245,   -0.608938144188165 },
                { 0 /* -0.60148066629825048 */, 0.608938144188165,  0.793217710690245 }
            });

            //Console.WriteLine(StringUtil.JoinColumns("U = ", U, " expected ", UExpected));
            //Console.WriteLine(StringUtil.JoinColumns("S = ", S, " expected ", SExpected));
            //Console.WriteLine(StringUtil.JoinColumns("V = ", V, " expected ", VExpected));
            Assert.True(UExpected.MaxDiff(U) < 1e-10);
            Assert.True(SExpected.MaxDiff(S) < 1e-10);
            Assert.True(VExpected.MaxDiff(V) < 1e-10);
        }

        [Fact]
        public void MatrixEigenvectorsSymmetricTest()
        {
            PositiveDefiniteMatrix A = new PositiveDefiniteMatrix(new double[,] { { 2, 1, 1 }, { 1, 2, 1 }, { 1, 1, 2 } });
            PositiveDefiniteMatrix Aev;
            double[] eigenvaluesReal = new double[A.Rows];
            double[] eigenvaluesImag = new double[A.Rows];
#if LAPACK
            double[] eigenvalues = new double[A.Rows];
            Aev = (PositiveDefiniteMatrix)A.Clone();
            Lapack.EigenvectorsSymmetricInPlace(Aev, eigenvalues);
            Console.WriteLine("eigenvectors as rows:\n{0}", Aev);
            Console.WriteLine("eigenvalues: {0}", StringUtil.CollectionToString(eigenvalues, " "));
            Aev = (PositiveDefiniteMatrix)A.Clone();
            Lapack.EigenvaluesInPlace(Aev, eigenvaluesReal, eigenvaluesImag);
            Console.WriteLine("eigenvalues Real: {0}", StringUtil.CollectionToString(eigenvaluesReal, " "));
            Console.WriteLine("eigenvalues Imag: {0}", StringUtil.CollectionToString(eigenvaluesImag, " "));
#endif
            Matrix eigenvectors = new Matrix(A.Rows, A.Cols);
            Aev = (PositiveDefiniteMatrix)A.Clone();
            eigenvectors.SetToEigenvectorsOfSymmetric(Aev);
            Matrix product = eigenvectors * Aev * eigenvectors.Transpose();
            Assert.True(product.MaxDiff(A) < TOLERANCE);

            //A = new PositiveDefiniteMatrix(new double[,] { { 1, 5, 7 }, { 3, 0, 6 }, { 4, 3, 1 } });
            Aev = (PositiveDefiniteMatrix)A.Clone();
            Aev.EigenvaluesInPlace(eigenvaluesReal, eigenvaluesImag);
            //Console.WriteLine("eigenvalues Real: {0}", StringUtil.CollectionToString(eigenvaluesReal, " "));
            //Console.WriteLine("eigenvalues Imag: {0}", StringUtil.CollectionToString(eigenvaluesImag, " "));
            // TODO: assertion

            A.SetTo(new double[,] { { double.PositiveInfinity, 0, 0 }, { 0, 2, 1 }, { 0, 1, 2 } });
            Aev = (PositiveDefiniteMatrix)A.Clone();
            eigenvectors.SetToEigenvectorsOfSymmetric(Aev);
            //Console.WriteLine("eigenvectors as cols:\n{0}", eigenvectors);
            //Console.WriteLine("eigenvalues:\n{0}", Aev);
        }

        // Test a case where balancing is required
        [Fact]
        public void MatrixEigenvaluesTest()
        {
            var A = new Matrix(new double[,] { { -2e+015, -3.85e+031, 1.05e+031 }, { 1, 0, 0 }, { 0, 1, 0 } });
            double[] eigenvaluesRealExpected = { 0.27272727, -1e+015, -1e+015 };
            double[] eigenvaluesImagExpected = { 0, 6.1237e+015, 6.1237e+015 };
            MatrixEigenvalues(A, eigenvaluesRealExpected, eigenvaluesImagExpected);
        }
        // Test a case where Francis shifts do not work (eigenvalues have the same magnitude)
        [Fact]
        public void MatrixEigenvaluesTest2()
        {
            var A = new Matrix(new double[,] { { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 } });
            double[] eigenvaluesRealExpected = { 1, -0.5, -0.5 };
            double[] eigenvaluesImagExpected = { 0, 0.86603, -0.86603 };
            MatrixEigenvalues(A, eigenvaluesRealExpected, eigenvaluesImagExpected);
        }
        // Test a case where Francis shifts do not work (matrix has triple eigenvalues)
        // and Day shifts do not work
        [Trait("Category", "OpenBug")]
        [Fact]
        public void MatrixEigenvaluesTest3()
        {
            var A = new Matrix(new double[,] { { 0, -3, 0, -3, 0, -1 }, { 1, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0, 0 },
                                               { 0, 0, 1, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0 }, { 0, 0, 0, 0, 1, 0 } });
            double[] eigenvaluesRealExpected = { 0, 0, 0, 0, 0, 0 };
            double[] eigenvaluesImagExpected = { 1, -1, 1, -1, 1, -1 };
            MatrixEigenvalues(A, eigenvaluesRealExpected, eigenvaluesImagExpected);
        }
        // Test a case where Francis shifts do not work (Demmel's example)
        [Trait("Category", "OpenBug")]
        [Fact]
        public void MatrixEigenvaluesTest4()
        {
            double a = 1e-8;
            var A = new Matrix(new double[,] { { 0, 1, 0, 0 }, { 1, 0, a, 0 }, { 0, -a, 0, 1 }, { 0, 0, 1, 0 } });
            double[] eigenvaluesRealExpected = { 1, 1, -1, -1 };
            double[] eigenvaluesImagExpected = { 0, 0, 0, 0 };
            MatrixEigenvalues(A, eigenvaluesRealExpected, eigenvaluesImagExpected);
        }
        private void MatrixEigenvalues(Matrix A, double[] eigenvaluesRealExpected, double[] eigenvaluesImagExpected)
        {
            double[] eigenvaluesReal = new double[A.Rows];
            double[] eigenvaluesImag = new double[A.Rows];
            var Aev = (Matrix)A.Clone();
            Aev.EigenvaluesInPlace(eigenvaluesReal, eigenvaluesImag);
            for (int i = 0; i < A.Rows; i++)
            {
                bool found = false;
                for (int j = 0; j < A.Rows; j++)
                {
                    double errorReal = MMath.AbsDiff(eigenvaluesReal[j], eigenvaluesRealExpected[i], 1e-10);
                    double errorImag = MMath.AbsDiff(eigenvaluesImag[j], eigenvaluesImagExpected[i], 1e-10);
                    if (errorReal < 1e-4 && errorImag < 1e-4)
                    {
                        found = true;
                        break;
                    }
                }
                Assert.True(found);
            }
        }

        internal void MatrixLogDeterminantTest()
        {
            PositiveDefiniteMatrix A =
                new PositiveDefiniteMatrix(
                    Matrix.Parse(
                        "24579592121.578629  12931091059.564314  -7998777904.3176851 90470385.802461445  5234576251.4642344  24678272370.717808  -70271649501.98642  -36306145609.258133\r\n12931091059.564314  7154278685.4261942  -4679099403.9732046 51222178.446217276  4118983860.199861   12530207422.652084  -43796270978.827423 -18838836338.201038\r\n-7998777904.3176851 -4679099403.9732046 88488028774.557251  -89025414.088132828 -21106264746.96204  10187828975.625618  498768222425.4101   -8011613681.8100595\r\n90470385.802461445  51222178.446217276  -89025414.088132828 409474.09008759452  40001448.611181676  79202860.046776846  -637572590.76677167 -123261117.92935695\r\n5234576251.4642344  4118983860.199861   -21106264746.96204  40001448.611181676  15520700972.575539  -5273256737.9989862 -127223473190.46838 3006895768.3404646\r\n24678272370.717808  12530207422.652084  10187828975.625618  79202860.046776846  -5273256737.9989862 33819227784.856228  24841323447.594578  -46166014096.263481\r\n-70271649501.98642  -43796270978.827423 498768222425.4101   -637572590.76677167 -127223473190.46838 24841323447.594578  2908938880988.3267  3000757203.60425\r\n-36306145609.258133 -18838836338.201038 -8011613681.8100595 -123261117.92935695 3006895768.3404646  -46166014096.263481 3000757203.60425    64217624470.000877"));
            double s = 5.5428051142289373e-15;
            s = 2e-5;
            //s = 5.5428051142289e-15;  // works
            PositiveDefiniteMatrix B = (PositiveDefiniteMatrix)A.Clone();
            B.Scale(s);
            double logdet1 = A.LogDeterminant() + A.Rows * System.Math.Log(s);
            double logdet2 = B.LogDeterminant();
            Assert.Equal(logdet1, logdet2, 1);
        }

        internal void KroneckerTest()
        {
            Matrix A = new Matrix(new double[,] { { 1, 2 }, { 3, 4 } });
            Matrix C = A.Kronecker(A);
            //Console.WriteLine(C);
            // TODO: assertion

            Matrix K = Matrix.Commutation(4, 3);
            //Console.WriteLine(K);
        }
    }
}