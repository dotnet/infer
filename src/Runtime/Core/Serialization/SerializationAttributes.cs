// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Utilities;
using System;
using System.Collections.Generic;
using System.Reflection;

namespace Microsoft.ML.Probabilistic.Serialization
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Attribute that can be attached to a constructor or factory method, to provide information about
    /// how to set the parameters of the constructor/method to copy an instance of the object.
    /// For example, it can indicate that the parameters correspond to properties of the object.
    /// </summary>
    [AttributeUsage(AttributeTargets.Constructor | AttributeTargets.Method)]
    public class ConstructionAttribute : Attribute, IComparable
    {
        /// <summary>
        /// The names of the properties, fields or methods to call on an object instance
        /// to get parameter values to construct a duplicate of that instance.
        /// </summary>
        public string[] Params { get; }

        public ConstructionAttribute()
        {
        }

        public ConstructionAttribute(params string[] parameters)
        {
            this.Params = parameters;
            this.paramMembers = new MemberInfo[this.Params.Length];
        }

        /// <summary>
        /// The construction method
        /// </summary>
        public MemberInfo TargetMember;

        /// <summary>
        /// The methods/properties to call to get constructor parameters.
        /// </summary>
        internal MemberInfo[] paramMembers { get; }

        /// <summary>
        /// The name of a boolean property or method on the object instance which
        /// indicates when this construction method should be used.
        /// </summary>
        /// <remarks>
        /// There are often special case constructors or factory methods which apply
        /// when an object is in a particular state e.g. when a distribution is uniform.  
        /// This parameter allows a constructor to be used only when the object is in
        /// a state, as indicated by a bool property or method e.g. IsUniform().
        /// </remarks>
        public string UseWhen { get; set; }

        /// <summary>
        /// Gets the value of the constructor parameter at the given index needed to
        /// reconstruct the supplied instance.
        /// </summary>
        /// <param name="instance">The instance</param>
        /// <param name="paramIndex">The parameter index</param>
        /// <param name="type">Type of the parameter (output)</param>
        /// <returns></returns>
        public object GetParamValue(int paramIndex, object instance, out Type type)
        {
            MemberInfo paramMember = GetParamMember(paramIndex);
            if (paramMember == null)
            {
                throw new ArgumentException("Invalid property or method name '" + Params[paramIndex] + "' for parameter " + paramIndex + " in type " + this.TargetMember?.Name + ".");
            }
            else if (paramMember is PropertyInfo)
            {
                PropertyInfo prop = (PropertyInfo) paramMember;
                type = prop.PropertyType;
                return prop.GetValue(instance, null);
            }
            else if (paramMember is FieldInfo)
            {
                FieldInfo field = (FieldInfo) paramMember;
                type = field.FieldType;
                return field.GetValue(instance);
            }
            else if (paramMember is MethodInfo)
            {
                MethodInfo method = (MethodInfo) paramMember;
                type = method.ReturnType;
                return method.Invoke(instance, null);
            }
            else throw new NotSupportedException("param must be property, field, or method; was " + paramMember);
        }

        private MemberInfo GetParamMember(int paramIndex)
        {
            if (paramMembers[paramIndex] == null)
            {
                paramMembers[paramIndex] = TargetMember.DeclaringType.GetProperty(
                    Params[paramIndex],
                    BindingFlags.FlattenHierarchy | BindingFlags.Instance | BindingFlags.Public |
                    BindingFlags.GetProperty);

                if (paramMembers[paramIndex] == null)
                {
                    // get method must have no parameters
                    paramMembers[paramIndex] = TargetMember.DeclaringType.GetMethod(Params[paramIndex], Type.EmptyTypes);
                }
                if (paramMembers[paramIndex] == null)
                {
                    paramMembers[paramIndex] = TargetMember.DeclaringType.GetField(Params[paramIndex]);
                }
            }

            return paramMembers[paramIndex];
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>
        /// Whether this construction attribute should be used for the supplied instance.
        /// </summary>
        /// <param name="instance"></param>
        /// <returns></returns>
        internal bool IsApplicable(object instance)
        {
            if (UseWhen == null) return true;
            Type type = instance.GetType();
            var pinfo = type.GetProperty(UseWhen);
            if (pinfo != null)
            {
                return (bool) pinfo.GetValue(instance, null);
            }
            var minfo = type.GetMethod(UseWhen, new Type[0]);
            if ((minfo != null) && (minfo.ReturnType == typeof (bool)))
            {
                return (bool) minfo.Invoke(instance, null);
            }
            throw new Exception("In type " + StringUtil.TypeToString(type) + ", ContructionAttribute UseWhen=" + UseWhen + " is not recognized");
            return false;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <summary>
        /// Cache of construction attributes
        /// </summary>
        private static Dictionary<Type, List<ConstructionAttribute>> cache = new Dictionary<Type, List<ConstructionAttribute>>();

        public static List<ConstructionAttribute> GetConstructionAttribute(Type type)
        {
            lock (cache)
            {
                if (!cache.ContainsKey(type))
                {
                    var cas = new List<ConstructionAttribute>();
                    // Look for constructors
                    foreach (ConstructorInfo ci in type.GetConstructors())
                    {
                        if (!ci.IsDefined(typeof (ConstructionAttribute), true)) continue;
                        ConstructionAttribute ca = ci.GetCustomAttributes(typeof (ConstructionAttribute), true)[0] as ConstructionAttribute;
                        ca.TargetMember = ci;
                        cas.Add(ca);
                    }
                    // Look for factory methods
                    foreach (MethodInfo mi in type.GetMethods(BindingFlags.Static | BindingFlags.Public | BindingFlags.FlattenHierarchy))
                    {
                        if (!mi.IsDefined(typeof (ConstructionAttribute), true)) continue;
                        if (mi.ReturnType != type) continue;
                        ConstructionAttribute ca = mi.GetCustomAttributes(typeof (ConstructionAttribute), true)[0] as ConstructionAttribute;
                        ca.TargetMember = mi;
                        cas.Add(ca);
                    }
                    // Order so that the UseWhen==null case comes last.
                    // Any other priorities could be included in here at some point if needed.
                    cas.Sort();
                    cache[type] = cas;
                }
                return cache[type];
            }
        }

        // Order so that the UseWhen==null case comes last.
        public int CompareTo(object obj)
        {
            ConstructionAttribute ca = obj as ConstructionAttribute;
            if (ca == null) return -1;
            if ((UseWhen == null) && (ca.UseWhen != null)) return 1;
            if ((UseWhen != null) && (ca.UseWhen == null)) return -1;
            return 0;
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}