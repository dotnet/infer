// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Reflection;

namespace Microsoft.ML.Probabilistic.Compiler.Reflection
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

  public interface CanGetDelegate
  {
    Delegate GetDelegate(Type type);
  }

    public class DelegateGroup : CanGetDelegate
    {
        public MethodBase[] methods;
        public object target;

        public DelegateGroup()
        {
        }

        public DelegateGroup(Type type, string methodName, BindingFlags flags, object target)
        {
            //methods = type.GetMethods(flags);
            //methods = Array.FindAll<MethodInfo>(methods, delegate(MethodInfo method) { return method.Name == methodName; });
            MemberInfo[] members = type.FindMembers(MemberTypes.Method, flags, Type.FilterName, methodName);
            methods = Array.ConvertAll<MemberInfo, MethodBase>(members, delegate(MemberInfo info) { return (MethodBase) info; });
            this.target = target;
        }

        public object DynamicInvoke(params object[] args)
        {
            return Invoker.Invoke(methods, target, args);
        }

        public Delegate GetDelegate(Type type)
        {
            // find a method which is compatible with the delegate type
            // compatibility is defined at: 
            // ms-help://MS.VSCC.v80/MS.MSDN.v80/MS.NETDEVFX.v20.en/cpref2/html/M_System_Delegate_CreateDelegate_2_1ee8f399.htm
            foreach (MethodBase method in methods)
            {
                Delegate result = Delegate.CreateDelegate(type, target, (MethodInfo) method, false);
                if (result != null) return result;
            }
            return null;
        }

        public DelegateGroup MakeGenericMethod(params Type[] types)
        {
            List<MethodBase> newmethods = new List<MethodBase>();
            foreach (MethodBase method in methods)
            {
                MethodInfo info = method as MethodInfo;
                if (info != null && info.IsGenericMethodDefinition)
                {
                    try
                    {
                        MethodInfo rmethod = info.MakeGenericMethod(types);
                        newmethods.Add(rmethod);
                    }
                    catch (ArgumentException)
                    {
                    }
                }
            }
            DelegateGroup result = new DelegateGroup();
            result.methods = newmethods.ToArray();
            result.target = target;
            return result;
        }

        public override string ToString()
        {
            StringBuilder s = new StringBuilder();
            for (int i = 0; i < methods.Length; i++)
            {
                if (i > 0) s.AppendLine();
                s.Append(methods[i].ToString());
            }
            return s.ToString();
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}