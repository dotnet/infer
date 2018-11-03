// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace CrowdsourcingWithWords
{
    /// <summary>
    ///   Receiver Operating Characteristic (ROC) Curve
    /// </summary>
    /// <remarks>
    ///   In signal detection theory, a receiver operating characteristic (ROC), or simply
    ///   ROC curve, is a graphical plot of the sensitivity vs. (1 − specificity) for a 
    ///   binary classifier system as its discrimination threshold is varied. 
    ///  
    /// References: 
    ///   http://en.wikipedia.org/wiki/Receiver_operating_characteristic
    ///   http://www.anaesthetist.com/mnm/stats/roc/Findex.htm
    ///   http://radiology.rsna.org/content/148/3/839.full.pdf
    /// </remarks>
    public class ReceiverOperatingCharacteristic
    {

        private double area;

        // The actual, measured data
        private double[] measurement;

        // The data, as predicted by a test
        private double[] prediction;


        // The real number of positives and negatives in the measured (actual) data
        private int positiveCount;
        private int negativeCount;

        // The values which represent positive and negative values in our
        //  measurement data (such as presence or absence of some disease)
        double dtrue;
        double dfalse;

        // The collection to hold our curve point information
        public PointCollection collection;

        /// <summary>
        ///   Constructs a new Receiver Operating Characteristic model
        /// </summary>
        /// <param name="measurement">An array of binary values. Typically 0 and 1, or -1 and 1, indicating negative and positive cases, respectively.</param>
        /// <param name="prediction">An array of continuous values trying to approximate the measurement array.</param>
        public ReceiverOperatingCharacteristic(double[] measurement, double[] prediction)
        {
            this.measurement = measurement;
            this.prediction = prediction;

            // Determine which numbers correspond to each binary category
            dtrue = measurement.Min();
            dfalse = measurement.Max();

            // Count the real number of positive and negative cases
            this.positiveCount = measurement.Count(m => m == dtrue);

            // Negative cases is just the number of cases minus the number of positives
            this.negativeCount = this.measurement.Length - this.positiveCount;
        }

        #region Public Methods

        /// <summary>
        ///   Computes a ROC curve with 1/increment points
        /// </summary>
        /// <param name="increment">The increment over the previous point for each point in the curve.</param>
        public void Compute(double increment)
        {
            List<Point> points = new List<Point>();
            double cutoff;

            // Create the curve, computing a point for each cutoff value
            for (cutoff = dfalse; cutoff <= dtrue; cutoff += increment)
            {
                points.Add(ComputePoint(cutoff));
            }
            if (cutoff < dtrue) points.Add(ComputePoint(dtrue));

            // Sort the curve by descending specificity
            points.Sort((a, b) => a.Specificity.CompareTo(b.Specificity));

            // Create the point collection
            this.collection = new PointCollection(points.ToArray());

            // Calculate area and error associated with this curve
            this.area = calculateAreaUnderCurve();
            calculateStandardError();
        }


        Point ComputePoint(double threshold)
        {
            int truePositives = 0;
            int trueNegatives = 0;

            for (int i = 0; i < this.measurement.Length; i++)
            {
                bool measured = (this.measurement[i] == dtrue);
                bool predicted = (this.prediction[i] >= threshold);


                // If the prediction equals the true measured value
                if (predicted == measured)
                {
                    // We have a hit. Now we have to see
                    //  if it was a positive or negative hit
                    if (predicted)
                        truePositives++; // Positive hit
                    else trueNegatives++;// Negative hit
                }
            }



            // The other values can be computed from available variables
            int falsePositives = negativeCount - trueNegatives;
            int falseNegatives = positiveCount - truePositives;

            return new Point(this, threshold,
                truePositives, trueNegatives,
                falsePositives, falseNegatives);
        }
        #endregion


        #region Private Methods
        /// <summary>
        ///   Calculates the area under the ROC curve using the trapezium method
        /// </summary>
        private double calculateAreaUnderCurve()
        {
            double sum = 0.0;

            for (int i = 0; i < collection.Count - 1; i++)
            {
                // Obs: False Positive Rate = (1-specificity)
                var tpz = collection[i].Sensitivity + collection[i + 1].Sensitivity;
                tpz = tpz * (collection[i].FalsePositiveRate - collection[i + 1].FalsePositiveRate) / 2.0;
                sum += tpz;
            }
            return sum;
        }

        /// <summary>
        ///   Calculates the standard error associated with this curve
        /// </summary>
        private double calculateStandardError()
        {
            double A = area;

            // real positive cases
            int Na = positiveCount;

            // real negative cases
            int Nn = negativeCount;

            double Q1 = A / (2.0 - A);
            double Q2 = 2 * A * A / (1.0 + A);

            return Math.Sqrt((A * (1.0 - A) +
                (Na - 1.0) * (Q1 - A * A) +
                (Nn - 1.0) * (Q2 - A * A)) / (Na * Nn));
        }
        #endregion

        #region Nested Classes

        /// <summary>
        /// The confusion matrix for the classified instances
        /// </summary>
        public class ConfusionMatrix
        {

            //  2x2 confusion matrix
            private int truePositives;
            private int trueNegatives;
            private int falsePositives;
            private int falseNegatives;

            /// <summary>
            ///   Constructs a new Confusion Matrix.
            /// </summary>
            public ConfusionMatrix(int truePositives, int trueNegatives,
                int falsePositives, int falseNegatives)
            {
                this.truePositives = truePositives;
                this.trueNegatives = trueNegatives;
                this.falsePositives = falsePositives;
                this.falseNegatives = falseNegatives;
            }

            /// <summary>
            ///   Sensitivity, also known as True Positive Rate
            /// </summary>
            /// <remarks>
            ///   Sensitivity = TPR = TP / (TP + FN)
            /// </remarks>
            public double Sensitivity => (double)truePositives / (truePositives + falseNegatives);

            /// <summary>
            ///   Specificity, also known as True Negative Rate
            /// </summary>
            /// <remarks>
            ///   Specificity = TNR = TN / (FP + TN)
            ///    or also as:  TNR = (1-False Positive Rate)
            /// </remarks>
            public double Specificity => (double)trueNegatives / (trueNegatives + falsePositives);

            /// <summary>
            ///   False Positive Rate, also known as false alarm rate.
            /// </summary>
            /// <remarks>
            ///   It can be calculated as: FPR = FP / (FP + TN)
            ///                or also as: FPR = (1-specifity)
            /// </remarks>
            public double FalsePositiveRate => (double)falsePositives / (falsePositives + trueNegatives);
        }

        /// <summary>
        ///   Object to hold information about a Receiver Operating Characteristic Curve Point
        /// </summary>
        public class Point : ConfusionMatrix
        {
            /// <summary>
            ///   Constructs a new Receiver Operating Characteristic point.
            /// </summary>
            internal Point(ReceiverOperatingCharacteristic curve, double cutoff,
                int truePositives, int trueNegatives, int falsePositives, int falseNegatives)
                : base(truePositives, trueNegatives, falsePositives, falseNegatives)
            {
                this.Cutoff = cutoff;
            }

            /// <summary>
            ///   Gets the cutoff value (discrimination threshold) for this point.
            /// </summary>
            public double Cutoff { get; private set; }
        }

        /// <summary>
        ///   Represents a Collection of Receiver Operating Characteristic (ROC) Curve points.
        ///   This class cannot be instantiated.
        /// </summary>
        public class PointCollection : ReadOnlyCollection<Point>
        {
            internal PointCollection(Point[] points)
                : base(points)
            {
            }

        }
        #endregion
    }
}
