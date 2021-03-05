// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler
{
    /// <summary>
    /// An attribute registry allows attributes to be associated with objects.
    /// </summary>
    public class AttributeRegistry<TObject, TAttribute> : IEnumerable<KeyValuePair<TObject, Set<TAttribute>>>, ICloneable
        where TObject : class
        where TAttribute : class
    {
        /// <summary>
        /// The attribute registry - dictionary from objects to attributes
        /// </summary>
        protected Dictionary<TObject, Set<TAttribute>> registry;

        /// <summary>
        /// Attribute registry constructor
        /// </summary>
        /// <param name="useIdentityEquality">Attributes are consider equal if they reference the same instance</param>
        public AttributeRegistry(bool useIdentityEquality)
        {
            if (!useIdentityEquality)
            {
                registry = new Dictionary<TObject, Set<TAttribute>>();
            }
            else
            {
                registry = new Dictionary<TObject, Set<TAttribute>>(ReferenceEqualityComparer<TObject>.Instance);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        protected bool IsIdentityEquality
        {
            get { return (registry.Comparer is ReferenceEqualityComparer<TObject>); }
        }

        /*public AttributeRegistry(IEqualityComparer<TObject> comparer)
        {
                registry = new Dictionary<TObject, List<TAttribute>>(comparer);
        }

        public AttributeRegistry(bool compareByIdentity)
        {
                if (compareByIdentity)
                {
                        registry = new Dictionary<TObject, List<TAttribute>>(new IdentityComparer<TObject>());
                }
                else
                {
                        registry = new Dictionary<TObject, List<TAttribute>>();
                }
        }*/


        /// <summary>
        /// Whether a given object has attributes
        /// </summary>
        /// <param name="obj">The object</param>
        /// <returns></returns>
        public bool HasAttributes(TObject obj)
        {
            return registry.ContainsKey(obj);
        }

        /// <summary>
        /// Copies object attributes associated with source object
        /// to target object in a target registry
        /// </summary>
        /// <param name="sourceObject">Source object</param>
        /// <param name="targetRegistry">Target registry</param>
        /// <param name="targetObject">Target object</param>
        public void CopyObjectAttributesTo(
            TObject sourceObject,
            AttributeRegistry<TObject, TAttribute> targetRegistry,
            TObject targetObject)
        {
            if (ReferenceEquals(sourceObject, targetObject))
                return;
            if (!registry.ContainsKey(sourceObject))
                return;
            foreach (TAttribute attr in registry[sourceObject])
                targetRegistry.Add(targetObject, attr);
        }

        /// <summary>
        /// Copies object attributes of type T associated with source object
        /// to target object in a target registry
        /// </summary>
        /// <typeparam name="T">Attribute type</typeparam>
        /// <param name="sourceObject">Source object</param>
        /// <param name="targetRegistry">Target registry</param>
        /// <param name="targetObject">Target object</param>
        public void CopyObjectAttributesTo<T>(
            TObject sourceObject,
            AttributeRegistry<TObject, TAttribute> targetRegistry,
            TObject targetObject)
        {
            if (ReferenceEquals(sourceObject, targetObject))
                return;
            if (!registry.ContainsKey(sourceObject))
                return;
            foreach (TAttribute attr in registry[sourceObject])
                if (attr is T)
                    targetRegistry.Add(targetObject, attr);
        }

        /// <summary>
        /// Removes all attributes of the specified object assignable to class T.
        /// </summary>
        /// <typeparam name="T">Specified type</typeparam>
        /// <param name="obj">Specified object</param>
        public void Remove<T>(TObject obj) where T : TAttribute
        {
            if (!registry.ContainsKey(obj))
                return;
            Set<TAttribute> oldList = registry[obj];
            Set<TAttribute> newList = new Set<TAttribute>();
            foreach (TAttribute attr in oldList)
                if (!(attr is T))
                    newList.Add(attr);
            registry[obj] = newList;
        }

        /// <summary>
        /// Removes all attributes of the specified object assignable to class T.
        /// </summary>
        /// <param name="obj">Specified object</param>
        /// <param name="t">Specified type</param>
        public void RemoveOfType(TObject obj, Type t)
        {
            if (!registry.ContainsKey(obj))
                return;
            Set<TAttribute> oldList = registry[obj];
            Set<TAttribute> newList = new Set<TAttribute>();
            foreach (TAttribute attr in oldList)
                if (!t.IsAssignableFrom(attr.GetType()))
                    newList.Add(attr);
            registry[obj] = newList;
        }

        /// <summary>
        /// Whether an object has an attribute of a particular type
        /// </summary>
        /// <typeparam name="T">The type</typeparam>
        /// <param name="obj">The object</param>
        /// <returns></returns>
        public bool Has<T>(TObject obj) where T : class, TAttribute
        {
            if (!registry.ContainsKey(obj))
                return false;
            foreach (TAttribute attr in registry[obj])
                if (attr is T)
                    return true;
            return false;
        }

        /// <summary>
        /// Returns the attributes of type T associated with the specified object. 
        /// </summary>
        /// <returns></returns>
        public List<T> GetAll<T>(TObject mi) where T : TAttribute
        {
            List<T> result = new List<T>();
            if (!registry.ContainsKey(mi))
                return result;
            foreach (TAttribute attr in registry[mi])
            {
                if (attr is T)
                    result.Add((T)attr);
            }
            return result;
        }

        /// <summary>
        /// Gets the unique attribute of a given type.  Returns null if none.  Throws an exception if not unique.
        /// </summary>
        /// <typeparam name="T">The type</typeparam>
        /// <param name="obj">The object</param>
        /// <returns>The attribute or null</returns>
        public T Get<T>(TObject obj) where T : class, TAttribute
        {
            if (!registry.ContainsKey(obj))
                return null;
            bool found = false;
            T result = null;
            foreach (TAttribute attr in registry[obj])
            {
                if (attr is T)
                {
                    if (found)
                    {
                        throw new InferCompilerException($"{obj} has multiple {StringUtil.TypeToString(typeof(T))} attributes");
                    }
                    else
                    {
                        result = (T)attr;
                        found = true;
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// Gets an attribute of a given type, or creates one if it doesn't already exist
        /// </summary>
        /// <typeparam name="T">The type</typeparam>
        /// <param name="obj">The object</param>
        /// <param name="generator"></param>
        /// <returns>The attribute</returns>
        public T GetOrCreate<T>(TObject obj, Func<T> generator) where T : class, TAttribute
        {
            T val = Get<T>(obj);
            if (val == null)
            {
                val = generator();
                Set(obj, val);
            }
            return val;
        }

        /// <summary>
        /// Sets the unique attribute of a given type against an object.  Throws an exception if not unique.
        /// </summary>
        /// <param name="obj">The object to associate the attribute with</param>
        /// <param name="attr">The attribute to associate</param>
        public void Set<T>(TObject obj, T attr)
            where T : class, TAttribute
        {
            if (Has<T>(obj))
                throw new InferCompilerException($"{obj} already has a {StringUtil.TypeToString(typeof(T))} attribute");
            Add(obj, attr);
        }

        /// <summary>
        /// Registers an attribute against the specified object.  Unlike Set, the attribute need not be unique.
        /// </summary>
        /// <param name="obj">The object to associate the attribute with</param>
        /// <param name="attr">The attribute to associate</param>
        public void Add(TObject obj, TAttribute attr)
        {
            if (!registry.ContainsKey(obj))
                registry[obj] = new Set<TAttribute>();
            registry[obj].Add(attr);
        }

        /// <summary>
        /// Gets an enumerator that iterates through the attribute registry
        /// </summary>
        /// <returns></returns>
        public IEnumerator<KeyValuePair<TObject, Set<TAttribute>>> GetEnumerator()
        {
            return registry.GetEnumerator();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        /// <summary>
        /// Write out the resgistry and the keyed registry to a text writer
        /// </summary>
        /// <param name="writer"></param>
        public void WriteTo(System.IO.TextWriter writer)
        {
            foreach (KeyValuePair<TObject, Set<TAttribute>> kvp in registry)
            {
                writer.WriteLine(kvp.Key.GetType().Name + " " + kvp.Key + ": ");
                foreach (TAttribute attr in kvp.Value)
                    writer.WriteLine("  " + attr);
                writer.WriteLine();
            }
        }

        /// <summary>
        /// Write out the attributes of a specified object to a text writer
        /// </summary>
        /// <param name="obj"></param>
        /// <param name="writer"></param>
        public void WriteObjectAttributesTo(TObject obj, System.IO.TextWriter writer)
        {
            writer.WriteLine(obj.GetType().Name + " " + obj + "(hash=" + System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj) + "): ");
            if (registry.ContainsKey(obj))
            {
                int i = 0;
                foreach (TAttribute attr in registry[obj])
                    writer.WriteLine((i++) + "  " + attr);
                writer.WriteLine();
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            AttributeRegistry<TObject, TAttribute> that = new AttributeRegistry<TObject, TAttribute>(IsIdentityEquality);
            foreach (KeyValuePair<TObject, Set<TAttribute>> entry in this.registry)
            {
                foreach (TAttribute attr in entry.Value)
                {
                    that.Add(entry.Key, attr);
                }
            }
            return that;
        }
    }

    internal class ReferenceEqualityComparer<T> : IEqualityComparer<T>
        where T : class
    {
        private ReferenceEqualityComparer()
        {
        }

        public static ReferenceEqualityComparer<T> Instance { get; } = new ReferenceEqualityComparer<T>();

        #region IEqualityComparer<T> Members

        public bool Equals(T x, T y) => ReferenceEquals(x, y);

        public int GetHashCode(T obj) => RuntimeHelpers.GetHashCode(obj);

        #endregion
    }
}