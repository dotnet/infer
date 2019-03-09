// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Vector = Microsoft.ML.Probabilistic.Math.Vector;

namespace ImageClassifier
{
    /// <summary>
    /// The model for the application.
    /// </summary>
    public class ItemsModel
    {
        ObservableCollection<Item> items = new ObservableCollection<Item>();

        public ObservableCollection<Item> Items { get { return items; } }

        public string Name { get { return "Testing"; } }

        public Form1 form1 = new Form1();

        public ItemsModel()
        {
        }

        bool classifying = false;

        internal bool Reclassify()
        {
            if (classifying) return false;
            classifying = true;
            List<Item> positive = new List<Item>();
            List<Item> negative = new List<Item>();
            foreach (Item item in Items)
            {
                if (item.State == 1) positive.Add(item);
                if (item.State == -1) negative.Add(item);
            }

            int count = positive.Count + negative.Count;
            if (count == 0)
            {
                Reset();
                classifying = false;
                return false;
            }

            Vector[] trainData = new Vector[count];
            bool[] trainLabels = new bool[count];
            Vector[] testData = new Vector[Items.Count - count];
            int trainCount = 0, testCount = 0;
            foreach (Item item in Items)
            {
                if (item.State != 0)
                {
                    trainData[trainCount] = item.Data;
                    trainLabels[trainCount] = item.State == 1;
                    trainCount++;
                }
                else
                {
                    testData[testCount++] = item.Data;
                }
            }

            form1.bpm.Train(trainData, trainLabels);
            double[] probs = form1.bpm.Test(testData);
            testCount = 0;
            foreach (Item item in Items)
            {
                if (item.State == 0)
                {
                    item.probTrue = probs[testCount++];
                }
            }

            classifying = false;
            return true;
        }

        public void Reset()
        {
            foreach (Item item in Items) item.Reset();
        }

        public void PopulateFromStringsAndVectors(IReadOnlyList<string> filenames, IReadOnlyList<Vector> data)
        {
            for (int i = 0; i < filenames.Count; i++)
            {
                items.Add(new Item(form1.folder + filenames[i], data[i]));
            }
        }
    }

    public class Item
    {
        static Random rnd = new Random();

        internal double probTrue = -1;

        public int State = 0; // +1 is positive, -1 is negative, 0 is unlabelled

        public string Filename { get; set; }

        public Vector Data { get; set; }

        public Item(string filename, Vector data)
        {
            this.Filename = filename;
            this.Data = data;
        }

        public override string ToString()
        {
            return "Item("+probTrue+")";
        }

        internal void Classify()
        {
            probTrue = rnd.NextDouble();
        }

        internal void Reset()
        {
            State = 0;
            probTrue = -1;
        }
    }
}
