// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

// General Information about an assembly is controlled through the following 
// set of attributes. Change these attribute values to modify the information
// associated with an assembly.

[assembly: AssemblyDescription("Run-time library for Infer.NET")]

// Setting ComVisible to false makes the types in this assembly not visible 
// to COM components.  If you need to access a type in this assembly from 
// COM, set the ComVisible attribute to true on that type.

[assembly: ComVisible(false)]

// The following GUID is for the ID of the typelib if this project is exposed to COM

[assembly: Guid("b35ca8e5-42b7-4dc6-9c75-8b94393d4c1d")]

// Tests need to be friend assemblies

[assembly:
    InternalsVisibleTo(
        "Microsoft.ML.Probabilistic.Tests,PublicKey=0024000004800000940000000602000000240000525341310004000001000100551f07a755a3e3f2901fa321ab631d13d6192b4e6ac9c87279500f49d6635cde6902587752eff20402f46f6ea9c3d80e827580a799840aaab9a49b1d2597e4c1798ee93c5cb66851e9d22f4d6e8110571f4a2e59f1d760f7be04fb10e7dc43ee7ed2831907731427b9815c5fe7f4888f9933ee7a1ad5d1f293fd8ab834fac1be"
        )]