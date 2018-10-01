// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Media.Animation;

namespace ImageClassifier
{
    /// <summary>
    /// Interaction logic for UserControl1.xaml
    /// </summary>
    public partial class ClassifierView : UserControl
    {
        static Random rnd = new Random();

        public ClassifierView()
        {
            InitializeComponent();
        }

        int marg = 5;

        internal void ShowInForm(string title)
        {
            Window w = new Window();
            w.Title = title;
            w.Content = this;
            w.SizeToContent = SizeToContent.WidthAndHeight;
            w.ResizeMode = ResizeMode.CanMinimize;
            Application app = new Application();
            app.Run(w);
        }

        Brush normalStroke = Brushes.Black;

        Point dragPoint = new Point(-1, -1);

        private void Rectangle_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            Rectangle r = (Rectangle)sender;
            dragPoint = e.GetPosition(this);
            r.CaptureMouse();
        }

        private void Rectangle_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            Rectangle r = (Rectangle)sender;
            r.ReleaseMouseCapture();
            ContentPresenter dobj = (ContentPresenter)itemsControl.ItemContainerGenerator.ContainerFromItem(r.DataContext);
            Point ct = new Point((double)dobj.GetValue(Canvas.LeftProperty) + r.Width / 2, (double)dobj.GetValue(Canvas.TopProperty) + r.Height / 2);
            r.IsHitTestVisible = false;
            List<Rectangle> hitTests = new List<Rectangle>();
            hitTests.Add(r);
            Rectangle hit = null;
            while (hit == null)
            {
                IInputElement el = InputHitTest(ct);
                if (!(el is Rectangle)) break;
                Rectangle r2 = (Rectangle)el;
                if (r2.DataContext is Item)
                {
                    r2.IsHitTestVisible = false;
                    hitTests.Add(r2);
                    continue;
                }

                hit = r2;
            }

            foreach (Rectangle r3 in hitTests) r3.IsHitTestVisible = true;

            Item item = (Item)r.DataContext;
            int oldState = item.State;
            if (hit == positiveBlock)
            {
                item.State = 1;
                r.Stroke = Brushes.LimeGreen;
            }

            if (hit == negativeBlock)
            {
                item.State = -1;
                r.Stroke = Brushes.Red;
            }

            if (hit == null)
            {
                item.State = 0;
                r.Stroke = normalStroke;
            }

            if (oldState != item.State)
            {
                // Update classifier
                ItemsModel model = (ItemsModel)DataContext;
                r.Cursor = Cursors.Wait;
                bool isClassifying = model.Reclassify();
                BottomPanel.Visibility = isClassifying ? Visibility.Visible : Visibility.Hidden;
                r.Cursor = null;
                foreach (Item it2 in model.Items)
                {
                    if (it2 != item) StartAnimation(it2);
                }
            }

            StartAnimation(item);
            dragPoint.X = -1;
        }

        protected void StartAnimation(Item item)
        {
            ContentPresenter dobj = (ContentPresenter)itemsControl.ItemContainerGenerator.ContainerFromItem(item);
            double x = (double)dobj.GetValue(Canvas.LeftProperty);
            double y = (double)dobj.GetValue(Canvas.TopProperty);
            Point targetPoint = GetTargetPosition(dobj, item);
            double dist = Math.Sqrt((x - targetPoint.X) * (x - targetPoint.X) + (y - targetPoint.Y) * (y - targetPoint.Y));
            double dur = Math.Pow(dist / 400, 0.5);
            if (dur > 1) dur = 1;
            if (dur < 0.1) dur = 0.1;
            Duration duration = new Duration(TimeSpan.FromSeconds(dur));
            DoubleAnimation xanim = new DoubleAnimation(targetPoint.X, duration);
            DoubleAnimation yanim = new DoubleAnimation(targetPoint.Y, duration);
            dobj.BeginAnimation(Canvas.LeftProperty, xanim);
            dobj.BeginAnimation(Canvas.TopProperty, yanim);
        }

        int bottomMargin = 45;

        internal Point GetTargetPosition(ContentPresenter dobj, Item item)
        {
            Rectangle r = (Rectangle)dobj.ContentTemplate.FindName("itemRect", dobj);
            double x = (double)dobj.GetValue(Canvas.LeftProperty);
            double y = (double)dobj.GetValue(Canvas.TopProperty);
            double newx = x;
            double newy = y;
            int width = (int)r.Width;
            int height = (int)r.Height;
            if (item.State == -2)
            {
                Point pt = GetRandomPosition(dobj);
                newx = pt.X;
                newy = pt.Y;
            }

            int binHeight = (int)binRow.Height.Value;
            
            if (x < marg) newx = marg;
            if (x + width > Width - marg) newx = Width - marg - width;
            if (y < marg) newy = marg;
            if (y + height > Height - marg - bottomMargin) newy = Height - marg - height - bottomMargin;
            if (item.State == 0)
            {
                if (y < binHeight + marg) newy = binHeight + marg;
            }
            else
            {
                if (item.State != -2)
                {
                    if (y + height > binHeight - marg) newy = binHeight - marg - height;
                    if (item.State == 1)
                    {
                        if (x + width > Width / 2 - marg) newx = Width / 2 - marg - width;
                    }
                    else
                    {
                        if (x < marg + Width / 2) newx = Width / 2 + marg;
                    }
                }
            }

            if ((item.State == 0) && (item.probTrue >= 0))
            {
                newx = marg + Squash(Squash(1 - item.probTrue)) * (Width - width - marg * 2);
            }

            return new Point(newx, newy);
        }

        /// <summary>
        /// Maps [0,1] to [0,1] with a sigmoidal nonlinearity.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private double Squash(double x)
        {
            return (Math.Sin(Math.PI * (x - 0.5)) + 1) * 0.5;
        }

        private void Rectangle_MouseMove(object sender, MouseEventArgs e)
        {
            if (dragPoint.X < 0) return;
            Rectangle r = (Rectangle)sender;
            Point pt = e.GetPosition(this);
            Item item = (Item)r.DataContext;
            FrameworkElement dobj = (FrameworkElement)itemsControl.ItemContainerGenerator.ContainerFromItem(item);

            double x = (double)dobj.GetValue(Canvas.LeftProperty);
            double y = (double)dobj.GetValue(Canvas.TopProperty);
            dobj.BeginAnimation(Canvas.LeftProperty, null);
            dobj.BeginAnimation(Canvas.TopProperty, null);

            double newx = pt.X - dragPoint.X + x;
            double newy = pt.Y - dragPoint.Y + y;
            dobj.SetValue(Canvas.LeftProperty, newx);
            dobj.SetValue(Canvas.TopProperty, newy);
            dragPoint = pt;
        }

        private void itemRect_Loaded(object sender, RoutedEventArgs e)
        {
            Rectangle r = (Rectangle)sender;
            Item item = (Item)r.DataContext;
            r.Stroke = normalStroke;
            BitmapImage im = new BitmapImage(new Uri(item.Filename, UriKind.Relative));
            SetRectSizeFromImage(r, im);
            r.Fill = new ImageBrush(im);
            DependencyObject dobj = itemsControl.ItemContainerGenerator.ContainerFromItem(item);
            Point pt = GetRandomPosition((ContentPresenter)dobj);
            dobj.SetValue(Canvas.LeftProperty, (double)pt.X);
            dobj.SetValue(Canvas.TopProperty, (double)pt.Y);
        }

        protected Point GetRandomPosition(ContentPresenter dobj)
        {
            Rectangle r = (Rectangle)dobj.ContentTemplate.FindName("itemRect", dobj);
            int binHeight = (int)binRow.Height.Value;
            int x = marg + rnd.Next((int)Width - marg * 2 - (int)r.Width);
            int y = marg + rnd.Next((int)Height - binHeight - marg * 2 - (int)r.Height - bottomMargin) + binHeight;
            return new Point(x, y);
        }

        int maxWidthOrHeight = 128;

        private void SetRectSizeFromImage(Rectangle r, BitmapImage im)
        {
            int width = im.PixelWidth;
            int height = im.PixelHeight;
            if (width > height)
            {
                r.Width = maxWidthOrHeight;
                r.Height = Math.Round(((double) height * maxWidthOrHeight) / width);
            }
            else
            {
                r.Width = Math.Round(((double) width * maxWidthOrHeight) / height);
                r.Height = maxWidthOrHeight;
            }
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Reset();
        }

        private void Reset()
        {
            ItemsModel model = (ItemsModel)DataContext;
            foreach (Item item in model.Items)
            {
                ContentPresenter dobj = (ContentPresenter)itemsControl.ItemContainerGenerator.ContainerFromItem(item);
                Rectangle r = (Rectangle)dobj.ContentTemplate.FindName("itemRect", dobj);
                r.Stroke = normalStroke;
                item.State = -2;
                StartAnimation(item);
                item.State = 0;
            }

            model.Reset();
            BottomPanel.Visibility = Visibility.Hidden;
        }
    }
}
