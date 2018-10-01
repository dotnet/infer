// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Reflection;

#if DEBUG
[assembly: AssemblyConfiguration("Debug")]
#else
[assembly: AssemblyConfiguration("Release")]
#endif
[assembly: AssemblyCompany("Microsoft Research Limited")]
[assembly: AssemblyProduct("Microsoft.ML.Probabilistic")]
[assembly: AssemblyCopyright("Copyright © Microsoft Research Limited 2008-2014")]
[assembly: AssemblyTrademark("")]
[assembly: AssemblyCulture("")]