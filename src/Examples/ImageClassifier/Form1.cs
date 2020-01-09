// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Collections;
using MRC = Microsoft.ML.Probabilistic.Collections;

namespace ImageClassifier
{
    public partial class Form1 : Form
    {
        internal BayesPointMachine bpm;
        internal List<Vector> data;
        Vector[] trainData, testData;
        bool[] trainLabels;
        Label[] probLabels;
        MRC.SortedSet<int> testSet = new MRC.SortedSet<int>();
        Dictionary<int, bool> labelMap = new Dictionary<int, bool>();
        public string folder = ImageFeatures.folder;

        public Form1()
        {
            InitializeComponent();
            ReadData();
            ReadImages();
        }

        public void ReadData()
        {
            data = ReadVectors(folder + "Features.txt");
            Normalize(data);
            int nFeatures = data[0].Count;
            bpm = new BayesPointMachine(nFeatures, 0.01);
        }

        public void ReadImages()
        {
            string[] filenames = File.ReadAllLines(folder + "Images.txt");
            int n = filenames.Length;

            probLabels = new Label[n];
            int xpos = 5;
            int ypos = 5;
            for (int i = 0; i < n; i++)
            {
                Panel panel = new Panel();
                panel.AutoSize = false;
                panel.Size = new Size(200, 200 + 150);
                panel.Name = "panel" + i;
                panel.Location = new Point(xpos, ypos);

                Label image = new Label();
                image.AutoSize = false;
                image.Name = "image" + i;
                image.Image = new Bitmap(folder + filenames[i]);
                image.Size = new Size(image.Image.Width, image.Image.Height);
                image.ForeColor = Color.DodgerBlue;
                image.Location = new Point(0, 0);
                panel.Controls.Add(image);

                Panel radioPanel = new Panel();
                panel.Controls.Add(radioPanel);
                radioPanel.Location = new Point(0, 205);
                RadioButton PosButton = new RadioButton();
                PosButton.AutoSize = true;
                PosButton.Name = "Pos" + i;
                PosButton.Text = "Positive";
                PosButton.Location = new Point(0, 0);
                PosButton.TabIndex = 0;
                PosButton.TabStop = true;
                PosButton.CheckedChanged += radioButton_CheckedChanged;
                radioPanel.Controls.Add(PosButton);
                RadioButton NegButton = new RadioButton();
                NegButton.AutoSize = true;
                NegButton.Name = "Neg" + i;
                NegButton.Text = "Negative";
                NegButton.Location = new Point(0, 30);
                NegButton.TabIndex = 1;
                NegButton.TabStop = true;
                NegButton.CheckedChanged += radioButton_CheckedChanged;
                radioPanel.Controls.Add(NegButton);
                RadioButton NoneButton = new RadioButton();
                NoneButton.Name = "Non" + i;
                NoneButton.Text = "No label";
                NoneButton.Location = new Point(0, 60);
                NoneButton.CheckedChanged += radioButton_CheckedChanged;
                NoneButton.TabIndex = 2;
                NoneButton.TabStop = true;
                radioPanel.Controls.Add(NoneButton);

                Label prob = new Label();
                prob.AutoSize = true;
                prob.Name = "prob" + i;
                prob.Dock = DockStyle.Bottom;
                prob.Text = prob.Name;
                prob.TextAlign = ContentAlignment.BottomCenter;
                probLabels[i] = prob;
                radioPanel.Controls.Add(prob);

                this.Controls.Add(panel);
                xpos += panel.Width + 5;
                if (xpos > 1500)
                {
                    ypos += panel.Height + 5;
                    xpos = 5;
                }
            }
        }

        public void radioButton_CheckedChanged(object sender, EventArgs e)
        {
            RadioButton button = (RadioButton)sender;
            if (button.Checked)
            {
                int i = int.Parse(button.Name.Substring(3));
                if (button.Text == "Positive") labelMap[i] = true;
                else if (button.Text == "Negative") labelMap[i] = false;
                else labelMap.Remove(i);
                LabelsChanged();
            }
        }

        public static List<Vector> ReadVectors(string path)
        {
            List<Vector> result = new List<Vector>();
            foreach (string line in File.ReadLines(path))
            {
                string[] entries = line.Split(',');
                Vector v = Vector.Zero(entries.Length - 1);
                for (int i = 0; i < v.Count; i++)
                {
                    v[i] = double.Parse(entries[i + 1], CultureInfo.InvariantCulture);
                }

                result.Add(v);
            }

            return result;
        }

        public void Normalize(IList<Vector> data)
        {
            if (data.Count == 0) return;
            VectorMeanVarianceAccumulator acc = new VectorMeanVarianceAccumulator(data[0].Count);
            foreach (Vector v in data)
            {
                acc.Add(v);
            }

            Vector mean = acc.Mean;
            Vector variance = acc.Variance.Diagonal();
            Vector stddev = Vector.Zero(variance.Count);
            stddev.SetToFunction(variance, Math.Sqrt);
            for (int i = 0; i < stddev.Count; i++)
            {
                if (stddev[i] == 0.0)
                {
                    mean[i] = 0.0;
                    stddev[i] = 1.0;
                }
            }

            foreach (Vector v in data)
            {
                v.SetToDifference(v, mean);
                v.SetToRatio(v, stddev);
            }
        }

        private void backgroundWorker1_DoWork(object sender, DoWorkEventArgs e)
        {
            Stopwatch watch = new Stopwatch();
            watch.Start();
            BackgroundWorker worker = (BackgroundWorker)sender;

            bpm.Train(trainData, trainLabels);
            e.Result = bpm.Test(testData);
            watch.Stop();
            Console.WriteLine("DoWork: {0} ms", watch.ElapsedMilliseconds);
        }

        public void LabelsChanged()
        {
            if (backgroundWorker1.IsBusy) return;

            trainData = new Vector[labelMap.Count];
            trainLabels = new bool[labelMap.Count];
            testData = new Vector[data.Count - labelMap.Count];
            testSet.Clear();
            int trainCount = 0, testCount = 0;
            for (int i = 0; i < data.Count; i++)
            {
                if (labelMap.ContainsKey(i))
                {
                    trainData[trainCount] = data[i];
                    trainLabels[trainCount] = labelMap[i];
                    trainCount++;
                }
                else
                {
                    testData[testCount++] = data[i];
                    testSet.Add(i);
                }
            }

            backgroundWorker1.RunWorkerAsync();
        }

        void ShowProbs(double[] probs)
        {
            int count = 0;
            for (int i = 0; i < probLabels.Length; i++)
            {
                probLabels[i].Text = "";
            }

            foreach (int item in testSet)
            {
                probLabels[item].Text = probs[count++].ToString("g4");
            }
        }

        private void backgroundWorker1_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (e.Error != null)
            {
                MessageBox.Show(e.Error.Message);
            }
            else if (e.Cancelled) { } //testLabels.Text = "Cancelled";
            else
            {
                double[] probs = (double[])e.Result;
                ShowProbs(probs);
            }
        }
    }
}