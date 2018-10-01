// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;
using Microsoft.ML.Probabilistic.Distributions;
using System.Windows.Media.Animation;

namespace ClinicalTrial
{
    /// <summary>
    /// Interaction logic for ClinicalTrial.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public ClinicalTrialModel model = null;
        public int[] counts = new int[] { 0, 0, 0, 0 };

        public MainWindow()
        {
            InitializeComponent();
            model = new ClinicalTrialModel();
            infer(new bool[0], new bool[0]);
        }

        private static void drawBetaDistribution(ListBox lb, Beta dist)
        {
            lb.Items.Clear();
            int numItems = (int)(lb.Width / 2.0);
            double max = 6.0;
            double mult = ((double)lb.Height) / max;
            double inc = 1.0 / ((double)(numItems - 1));
            double curr = 0.0;
            lb.Margin = new Thickness(0);
            for (int i = 0; i < numItems; i++)
            {
                if (curr > 1.0)
                    curr = 1.0;
                double d = Math.Exp(dist.GetLogProb(curr));
                double height = mult * d;
                lb.Items.Add(new Rectangle() { Margin = new Thickness(0), Height = height, Width = 2, Fill = Brushes.Yellow, ClipToBounds = true });
                curr += inc;
            }
        }

        private void infer(bool[] treated, bool[] placebo)
        {
            model.Infer(treated, placebo);
            ProbIsEffectiveSlider.Value = model.posteriorTreatmentIsEffective.GetProbTrue();
            drawBetaDistribution(TreatedPDF, model.posteriorProbIfTreated);
            drawBetaDistribution(PlaceboPDF, model.posteriorProbIfPlacebo);
        }

        private void UpdateList(double val, ListBox lb, int index)
        {
            // Following assumes 20 items max.
            int new_count = (int)val;
            int cur_count = counts[index];

            // If count has changed...
            if (new_count != cur_count)
            {
                // ... update the list box ...
                lb.Items.Clear();
                for (int i = 0; i < new_count; i++)
                {
                    // the actual object added here does not matter (it will be ignored)
                    // the size of the collection is all that matters
                    lb.Items.Add("patient"); 
                }

                counts[index] = new_count;

                // ... construct the true/false arrays...
                int numTreated = counts[0] + counts[2];
                int numPlacebo = counts[1] + counts[3];
                bool[] treated = new bool[numTreated];
                bool[] placebo = new bool[numPlacebo];
                int j = 0;
                for (; j < counts[0]; j++)
                    treated[j] = true;
                for (; j < numTreated; j++)
                    treated[j] = false;
                j = 0;
                for (; j < counts[1]; j++)
                    placebo[j] = true;
                for (; j < numPlacebo; j++)
                    placebo[j] = false;

                // ... and do the inference
                infer(treated, placebo);
            }
        }

        FrameworkElement startEl;

        private void Rectangle_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            FrameworkElement el = (FrameworkElement)sender;
            startEl = el;
            Point pos = e.GetPosition(el);

            // Size of a child element
            pos = OnClickOrDrag(el, pos);
            e.Handled = true;
        }

        Size childSize = new Size(35 + 12 * 2, 60);

        private Point OnClickOrDrag(FrameworkElement el, Point pos)
        {
            // Size of a child element
            int numPerRow = (int)(el.ActualWidth / childSize.Width);
            int col = (int)((pos.X + 35 + 4) / childSize.Width); col = Math.Min(col, 7);
            int row = (int)(pos.Y / childSize.Height); row = Math.Min(row, 6);
            if ((string)el.Tag == "TreatedGood") UpdateList(row * numPerRow + col, ListBoxTreatedGood, 0);
            if ((string)el.Tag == "PlaceboGood") UpdateList(row * numPerRow + col, ListBoxPlaceboGood, 1);
            if ((string)el.Tag == "TreatedBad") UpdateList(row * numPerRow + col, ListBoxTreatedBad, 2);
            if ((string)el.Tag == "PlaceboBad") UpdateList(row * numPerRow + col, ListBoxPlaceboBad, 3);
            return pos;
        }

        private void Rectangle_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton != MouseButtonState.Pressed) return;

            FrameworkElement el = (FrameworkElement)sender;
            if (el != startEl) return;
            Point pos = e.GetPosition(el);
            pos = OnClickOrDrag(el, pos);
            e.Handled = true;
        }

        private void Reset_Clicked(object sender, RoutedEventArgs e)
        {
            UpdateList(0, ListBoxTreatedGood, 0);
            UpdateList(0, ListBoxPlaceboGood, 1);
            UpdateList(0, ListBoxTreatedBad, 2);
            UpdateList(0, ListBoxPlaceboBad, 3);
        }

        static Random rnd = new Random();

        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            // An Easter egg
            if (e.Key == Key.D) // D for 'die'
            {
                FrameworkElement el = GetRandomElement(ListBoxPlaceboBad);
                if (el == null) el = GetRandomElement(ListBoxTreatedBad);
                if (el == null) return;
                el.Tag = "Dead";
                RotateTransform rt = new RotateTransform { CenterX = childSize.Width / 2 + 3, CenterY = childSize.Height - 14 };
                el.RenderTransform = rt;
                rt.BeginAnimation(RotateTransform.AngleProperty, new DoubleAnimation(90, new Duration(TimeSpan.FromMilliseconds(300))));
            }
        }

        private FrameworkElement GetRandomElement(ListBox lb)
        {
            FrameworkElement el = null;
            if (lb.Items.Count > 0)
            {
                for (int i = 0; i < 20; i++)
                {
                    int ind = rnd.Next(lb.Items.Count);
                    el = (FrameworkElement)lb.ItemContainerGenerator.ContainerFromIndex(ind);
                    if (el.Tag != null) el = null;
                    if (el != null) break;
                }
            }

            return el;
        }

        private void EndDemoClicked(object sender, RoutedEventArgs e)
        {
            WindowState = WindowState.Minimized;
        }
    }
}
