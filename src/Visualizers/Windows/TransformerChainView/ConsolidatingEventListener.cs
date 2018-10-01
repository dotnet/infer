// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Threading;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    /// <summary>
    /// An event listener which consolidates events which are close by in time, effectively suppressing
    /// the earlier events.
    /// </summary>
    public class ConsolidatingEventListener<TEventArgs> where TEventArgs : EventArgs
    {
        protected DispatcherTimer timer;
        protected EventHandler<TEventArgs> childHandler;
        public bool IsEnabled { get; set; }

        public ConsolidatingEventListener(long windowInMillis, EventHandler<TEventArgs> childHandler)
        {
            IsEnabled = true;
            timer = new DispatcherTimer();
            timer.IsEnabled = false;
            timer.Interval = TimeSpan.FromMilliseconds(windowInMillis);
            timer.Tick += new EventHandler(timer_Tick);
            this.childHandler = childHandler;
        }

        void timer_Tick(object sender, EventArgs e)
        {
            if (!IsEnabled) return;
            timer.Stop();
            childHandler.Method.Invoke(childHandler.Target, new object[] { timer.Tag, args });
        }

        object args;
        public void HandlerMethod(object sender, TEventArgs args)
        {
            if (!IsEnabled) return;
            timer.Stop();
            timer.Tag = sender;
            this.args = args;
            timer.Start();
        }
    }
}
