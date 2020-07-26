// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using System.Reflection;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.Reflection
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    // another name might be GenericActivator
    /// <summary>
    /// Static methods to dynamically invoke methods and access fields of an object.
    /// </summary>
    public static class Invoker
    {
        /// <summary>
        /// Get a type by name.
        /// </summary>
        /// <param name="typeName">The name of a type in the System library or any loaded assembly.</param>
        /// <returns>The Type object.</returns>
        public static Type GetLoadedType(string typeName)
        {
            Type type = Type.GetType(typeName);
            if (type == null)
            {
                // search through loaded assemblies
                AppDomain app = AppDomain.CurrentDomain;
                Assembly[] assemblies = app.GetAssemblies();
                foreach (Assembly assembly in assemblies)
                {
                    type = assembly.GetType(typeName);
                    if (type != null) break;
                }
                if (type == null) throw new TypeLoadException("The type '" + typeName + "' does not exist (perhaps you need a qualifier?)");
            }
            return type;
        }

        /// <summary>
        /// Invoke the static member which best matches the argument types.
        /// </summary>
        /// <param name="type"></param>
        /// <param name="methodName"></param>
        /// <param name="args"></param>
        /// <returns></returns>
        public static object InvokeStatic(Type type, string methodName, params object[] args)
        {
            if (methodName == "new")
            {
                if (args.Length == 0) return Activator.CreateInstance(type);
                else
                {
                    // when there are arguments, we may need to perform conversions.
                    ConstructorInfo[] ctors = type.GetConstructors();
                    if (ctors.Length == 0) throw new MissingMethodException(type + " has no constructors");
                    return Invoke(ctors, null, args);
                }
            }
            else
            {
                return InvokeMember(type, methodName,
                                    BindingFlags.Public | BindingFlags.Static | BindingFlags.GetField | BindingFlags.GetProperty | BindingFlags.InvokeMethod |
                                    BindingFlags.FlattenHierarchy, null, args);
            }
        }

        /// <summary>
        /// Invoke the instance member which best matches the argument types.
        /// </summary>
        /// <param name="methodName"></param>
        /// <param name="target"></param>
        /// <param name="args"></param>
        /// <returns></returns>
        public static object InvokeInstance(string methodName, object target, params object[] args)
        {
            return InvokeMember(target.GetType(), methodName,
                                BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetField | BindingFlags.GetProperty | BindingFlags.InvokeMethod |
                                BindingFlags.FlattenHierarchy, target, args);
        }

        /// <summary>
        /// Get an element from an array or collection, or invoke a delegate.
        /// </summary>
        /// <param name="target">An array, collection, delegate, or method group.</param>
        /// <param name="args">Indices for the collection or arguments for the delegate.  May be null.</param>
        /// <returns>The collection element or return value of the delegate.  If args == null, the target itself.</returns>
        public static object GetValue(object target, params object[] args)
        {
            if (args == null) return target;
            Delegate d = target as Delegate;
            if (d != null)
            {
                return InvokeMember(d.GetType(), "Invoke", BindingFlags.Public | BindingFlags.Instance | BindingFlags.InvokeMethod, d, args);
            }
            DelegateGroup dg = target as DelegateGroup;
            if (dg != null)
            {
                return dg.DynamicInvoke(args);
            }
            if (args.Length == 0) return target;
            Array array = target as Array;
            if (array != null)
            {
                int[] index = Array.ConvertAll<object, int>(args, delegate(object o) { return Convert.ToInt32(o); });
                return array.GetValue(index);
            }
            if (target == null) return target;
            //throw new ArgumentNullException("The field/property value is null");
            // collection
            return InvokeMember(target.GetType(), "Item", BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetProperty, target, args);
        }

        /// <summary>
        /// Set an element in an array or collection.
        /// </summary>
        /// <param name="target">An array or collection.</param>
        /// <param name="args">Indices followed by the value to set.  Length > 0.</param>
        public static void SetValue(object target, params object[] args)
        {
            Array array = target as Array;
            if (array != null)
            {
                object value = args[args.Length - 1];
                Conversion conv;
                Type elementType = array.GetType().GetElementType();
                if (!Conversion.TryGetConversion(value.GetType(), elementType, out conv))
                    throw new ArgumentException("The value (" + args[0] + ") does not match the element type (" + elementType.Name + ")");
                if (conv.Converter != null) value = conv.Converter(value);
                int[] index = new int[args.Length - 1];
                for (int i = 0; i < index.Length; i++) index[i] = Convert.ToInt32(args[i]);
                array.SetValue(value, index);
                return;
            }
            // collection
            InvokeMember(target.GetType(), "Item", BindingFlags.Public | BindingFlags.Instance | BindingFlags.SetProperty, target, args);
        }

        /// <summary>
        /// Invoke the member which best matches the argument types.
        /// </summary>
        /// <param name="type">The type defining the member.</param>
        /// <param name="memberName">The name of a field, property, event, instance method, or static method.</param>
        /// <param name="flags"></param>
        /// <param name="target">The object whose member to invoke.  Ignored for a static field or property.
        /// If target is non-null, then it is provided as the first argument to a static method.</param>
        /// <param name="args">Can be empty or null.  Empty means a function call with no arguments.  
        /// null means get the member itself.</param>
        /// <returns>The result of the invocation. For SetField/SetProperty the result is null.</returns>
        /// <exception cref="MissingMemberException"></exception>
        /// <exception cref="ArgumentException"></exception>
        /// <remarks><para>
        /// This routine is patterned after Type.InvokeMember.
        /// flags must specify Instance, Static, or both.
        /// </para><para>
        /// If flags contains CreateInstance, then name is ignored and a constructor is invoked.
        /// </para><para>
        /// If memberName names a field/property and flags contains SetField/SetProperty, 
        /// then the field/property's value is changed to args[args.Length-1].
        /// If args.Length > 1, the field/property is indexed by args[0:(args.Length-2)].
        /// </para><para>
        /// If memberName names a field/property and flags contains GetField/GetProperty, 
        /// then the field/property's value is returned.  
        /// If args != null and the field/property is a delegate, then it is invoked with args.
        /// Otherwise if args != null, the field/property is indexed by args.
        /// </para><para>
        /// If memberName names an event and flags contains GetField, 
        /// then the event's EventInfo is returned.  
        /// If args != null, then the event is raised with args.
        /// </para><para>
        /// If memberName names a method and flags contains InvokeMethod, 
        /// then it is invoked with args.  A static method is invoked with target and args.
        /// If args == null, then the result is a DelegateGroup containing all overloads of the method.
        /// </para><para>
        /// Other flag values are implemented as in Type.InvokeMember.
        /// In each case, overloading is resolved by matching the argument types, possibly with conversions.
        /// </para><para>
        /// If a matching member is not found, the interfaces of the type are also searched.
        /// As a last resort, if the memberName is op_Equality or op_Inequality, then a default implementation
        /// is provided (as in C#).
        /// </para></remarks>
        public static object InvokeMember(Type type, string memberName, BindingFlags flags, object target, params object[] args)
        {
            if ((flags & BindingFlags.GetField) == BindingFlags.GetField)
            {
                FieldInfo field = type.GetField(memberName, flags);
                if (field != null)
                {
                    object value = field.GetValue(target);
                    value = GetValue(value, args);
                    return value;
                }
                EventInfo evnt = type.GetEvent(memberName, flags);
                if (evnt != null)
                {
                    if (args == null) return evnt;
                    //MethodInfo raiser = evnt.GetRaiseMethod();
                    //return raiser.Invoke(target, args);
                    //return type.InvokeMember(memberName, BindingFlags.Public | BindingFlags.Instance | BindingFlags.GetField | BindingFlags.GetProperty | BindingFlags.InvokeMethod, null, target, args);
                    // http://forums.microsoft.com/MSDN/ShowPost.aspx?PostID=130529&SiteID=1
                    return new NotSupportedException("Raising events is not supported by Reflection");
                }
            }
            if ((flags & BindingFlags.SetField) == BindingFlags.SetField)
            {
                FieldInfo field = type.GetField(memberName, flags);
                if (field != null)
                {
                    if (args == null || args.Length == 0) throw new ArgumentException("No value was provided to set");
                    if (args.Length == 1)
                    {
                        object value = args[0];
                        Conversion conv;
                        if (!Conversion.TryGetConversion(value == null ? null : value.GetType(), field.FieldType, out conv))
                            throw new ArgumentException("The value (" + args[0] + ") does not match the field type (" + field.FieldType.Name + ")");
                        if (conv.Converter != null) value = conv.Converter(value);
                        field.SetValue(target, value);
                    }
                    else
                    {
                        SetValue(field.GetValue(target), args);
                    }
                    return null;
                }
            }
            if ((flags & BindingFlags.GetProperty) == BindingFlags.GetProperty)
            {
                PropertyInfo[] props = type.GetProperties(flags);
                props = Array.FindAll<PropertyInfo>(props, delegate(PropertyInfo p) { return p.Name == memberName; });
                if (props.Length > 0)
                {
                    PropertyInfo prop = props[0];
                    int rank = prop.GetIndexParameters().Length;
                    if (rank > 0)
                    {
                        //if(rank != args.Length) throw new ArgumentException("Not enough arguments for indexed property");
                        // indexed property
                        memberName = "get_" + memberName;
                        //MethodInfo method = type.GetMethod(memberName);
                        return InvokeMember(type, memberName, flags | BindingFlags.InvokeMethod, target, args);
                    }
                    // not indexed property
                    object value = prop.GetValue(target, null);
                    value = GetValue(value, args);
                    return value;
                }
            }
            if ((flags & BindingFlags.SetProperty) == BindingFlags.SetProperty)
            {
                PropertyInfo[] props = type.GetProperties(flags);
                props = Array.FindAll<PropertyInfo>(props, delegate(PropertyInfo p) { return p.Name == memberName; });
                if (props.Length > 0)
                {
                    if (args == null || args.Length == 0) throw new ArgumentException("No value was provided to set");
                    //PropertyInfo prop = type.GetProperty(memberName, flags);
                    PropertyInfo prop = props[0];
                    int rank = prop.GetIndexParameters().Length;
                    if (rank > 0 || args.Length == 1)
                    {
                        // indexed property
                        memberName = "set_" + memberName;
                        //MethodInfo method = type.GetMethod(memberName);
                        return InvokeMember(type, memberName, flags | BindingFlags.InvokeMethod, target, args);
                    }
                    // not indexed property
                    SetValue(prop.GetValue(target, null), args);
                    return null;
                }
            }
            if ((flags & BindingFlags.InvokeMethod) == BindingFlags.InvokeMethod)
            {
                // must explicitly search through BaseTypes because static methods are not inherited
                Type baseType = type;
                while (baseType != null)
                {
                    MethodInfo[] methods = baseType.GetMethods(flags);
                    methods = Array.FindAll(methods, delegate(MethodInfo method) { return method.Name == memberName; });
                    if (methods.Length > 0)
                    {
                        if (args == null)
                        {
                            // even if there is only one method, we can't create a delegate because we don't know the 
                            // desired delegate type.
                            DelegateGroup dg = new DelegateGroup();
                            dg.target = target;
                            dg.methods = methods;
                            return dg;
                        }
                        return Invoke(methods, target, args);
                    }
                    baseType = baseType.BaseType;
                }
            }
            // search through interfaces
            Type[] faces = type.GetInterfaces();
            foreach (Type face in faces)
            {
                try
                {
                    return InvokeMember(face, memberName, flags, target, args);
                }
                catch (MissingMemberException)
                {
                }
            }
            // default operator implementations
            if ((flags & BindingFlags.InvokeMethod) == BindingFlags.InvokeMethod &&
                (flags & BindingFlags.Static) == BindingFlags.Static)
            {
                if (memberName == "op_Equality" || memberName == "op_Inequality" ||
                    memberName == "op_GreaterThan" || memberName == "op_LessThan" ||
                    memberName == "op_GreaterThanOrEqual" || memberName == "op_LessThanOrEqual" ||
                    memberName == "op_Subtraction" || memberName == "op_Addition" ||
                    memberName == "op_BooleanOr" || memberName == "op_BooleanAnd" || memberName == "op_BooleanNot" ||
                    memberName == "op_UnaryNegation")
                {
                    return InvokeMember(typeof (Invoker), memberName, BindingFlags.Public | BindingFlags.Static | BindingFlags.InvokeMethod, target, args);
                }
            }
            throw new MissingMemberException(type.ToString() + " has no member named " + memberName + " under the binding flags " + flags);
        }

        // used by InvokeMember
        public static bool op_Equality(object a, object b)
        {
          if (a == null)
            return (a == b);
          else
            return a.Equals(b);
        }

        public static bool op_Inequality(object a, object b)
        {
          return !op_Equality(a, b);
        }

        public static int op_UnaryNegation(int a)
        {
            return -a;
        }

        public static int op_Addition(int a, int b)
        {
          return (a + b);
        }

        public static int op_Subtraction(int a, int b)
        {
          return (a - b);
        }

        public static bool op_GreaterThan(int a, int b)
        {
          return (a > b);
        }

        public static bool op_GreaterThanOrEqual(int a, int b)
        {
          return (a >= b);
        }

        public static bool op_LessThan(int a, int b)
        {
          return (a < b);
        }

        public static bool op_LessThanOrEqual(int a, int b)
        {
          return (a <= b);
        }

        public static bool op_BooleanOr(bool a, bool b)
        {
          return (a || b);
        }

        public static bool op_BooleanAnd(bool a, bool b)
        {
          return (a && b);
        }

        public static bool op_BooleanNot(bool b)
        {
          return !b;
        }

        /// <summary>
        /// Gets the types of the objects in the specified array. 
        /// </summary>
        /// <param name="args">An array of objects whose types to determine.  args[i] can be null, whose type is assumed to be typeof(Nullable).</param>
        /// <returns>An array of Type objects representing the types of the corresponding elements in args. </returns>
        /// <remarks>This method is the same as Type.GetTypeArray except it allows null values.</remarks>
        public static Type[] GetTypeArray(object[] args)
        {
            Type[] actuals = new Type[args.Length];
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == null) actuals[i] = typeof (Nullable);
                else actuals[i] = args[i].GetType();
            }
            return actuals;
        }

        public static string PrintTypes(Type[] types)
        {
            StringBuilder s = new StringBuilder();
            for (int i = 0; i < types.Length; i++)
            {
                if (i > 0) s.Append(",");
                if (types[i] != null) s.Append(types[i].Name);
            }
            return s.ToString();
        }

        public static Type[] GetParameterTypes(MethodBase method)
        {
            ParameterInfo[] parameters = method.GetParameters();
            Type[] formals = new Type[parameters.Length];
            int i = 0;
            foreach (ParameterInfo param in parameters) formals[i++] = param.ParameterType;
            return formals;
        }

        public static int GenericParameterCount(MethodBase method)
        {
            MethodInfo info = method as MethodInfo;
            if (info == null) return 0;
            Type[] args = info.GetGenericArguments();
            int count = 0;
            foreach (Type arg in args)
            {
                if (arg.IsGenericParameter) count++;
            }
            return count;
        }

        public static int GenericParameterCount(Type type)
        {
            Type[] args = type.GetGenericArguments();
            int count = 0;
            foreach (Type arg in args)
            {
                if (arg.IsGenericParameter) count++;
            }
            return count;
        }

        private static T[] AddFirst<T>(T[] array, T item)
        {
            T[] result = new T[array.Length + 1];
            result[0] = item;
            array.CopyTo(result, 1);
            return result;
        }

        private static T[] ButFirst<T>(T[] array)
        {
            T[] result = new T[array.Length - 1];
            Array.Copy(array, 1, result, 0, result.Length);
            return result;
        }

        /// <summary>
        /// Invoke the method which best matches the argument types.
        /// </summary>
        /// <param name="methods">A non-empty list of methods, exactly one of which will be invoked.  Can include both static and instance methods.</param>
        /// <param name="target">The instance for an instance method, or if non-null, the first argument of a static method.</param>
        /// <param name="args">The remaining arguments of the method.</param>
        /// <returns>The return value of the method.</returns>
        public static object Invoke(MethodBase[] methods, object target, params object[] args)
        {
            if (methods.Length == 0) throw new ArgumentException("The method list is empty");
            Binding binding;
            Exception exception;
            Type[] actuals = GetTypeArray(args);
            MethodBase method = GetBestMethod(methods, (target == null) ? null : target.GetType(), actuals, ConversionOptions.AllConversions, out binding, out exception);
            if (method == null) throw exception;
            if (method.IsStatic && target != null)
            {
                args = AddFirst(args, target);
                target = null;
            }
            else if (!method.IsStatic && !method.IsConstructor && target == null)
            {
                if (args.Length == 0) throw new ArgumentException("The target is null");
                target = args[0];
                args = ButFirst(args);
            }
            // apply argument conversions
            binding.ConvertAll(args);
            //for (int i = 0; i < args.Length; i++) {
            //  Converter conv = binding.Conversions[i].Converter;
            //  if (conv != null) args[i] = conv(args[i]);
            //}
            object result = Util.Invoke(method, target, args);
            if (!method.IsConstructor && ((MethodInfo)method).ReturnType == typeof(void))
                return Missing.Value;
            else
                return result;
        }

        /// <summary>
        /// Invoke a generic method by inferring type parameters from the method arguments.
        /// </summary>
        /// <param name="method"></param>
        /// <param name="target"></param>
        /// <param name="args"></param>
        /// <returns></returns>
        public static object Invoke(MethodBase method, object target, params object[] args)
        {
            MethodBase[] methods = new MethodBase[] {method};
            return Invoke(methods, target, args);
        }

        /// <summary>
        /// Find the method which best matches the given arguments.
        /// </summary>
        /// <param name="type"></param>
        /// <param name="memberName"></param>
        /// <param name="flags"></param>
        /// <param name="targetType">The type of <c>this</c>, for instance methods.  If looking for a static method, use null.</param>
        /// <param name="argTypes">Types.  argTypes.Length == number of method parameters.  argTypes[i] may be null to allow any type, or typeof(Nullable) to mean "any nullable type".</param>
        /// <param name="exception">Exception created on failure</param>
        /// <returns>null on failure.</returns>
        /// <exception cref="ArgumentException">The best matching type parameters did not satisfy the constraints of the generic method.</exception>
        /// <exception cref="MissingMethodException">No match was found.</exception>
        public static MethodBase GetBestMethod(Type type, string memberName, BindingFlags flags, Type targetType, Type[] argTypes, out Exception exception)
        {
            MethodInfo[] methods = type.GetMethods(flags);
            methods = Array.FindAll<MethodInfo>(methods, delegate(MethodInfo method) { return method.Name == memberName; });
            if (methods.Length == 0)
            {
                exception = new MissingMethodException(type.Name + " does not have any methods named " + memberName + " under the binding flags " + flags);
                return null;
            }
            else return GetBestMethod(methods, targetType, argTypes, out exception);
        }

        /// <summary>
        /// Find the method which best matches the given arguments.
        /// </summary>
        /// <param name="methods"></param>
        /// <param name="targetType">The type of <c>this</c>, for instance methods.  If looking for a static method, use null.</param>
        /// <param name="argTypes">Types.  argTypes.Length == number of method parameters.  argTypes[i] may be null to allow any type, or typeof(Nullable) to mean "any nullable type".</param>
        /// <param name="exception">Exception created on failure</param>
        /// <returns>A non-null MethodBase.</returns>
        /// <exception cref="ArgumentException">The best matching type parameters did not satisfy the constraints of the generic method.</exception>
        /// <exception cref="MissingMethodException">No match was found.</exception>
        public static MethodBase GetBestMethod(MethodBase[] methods, Type targetType, Type[] argTypes, out Exception exception)
        {
            Binding binding;
            return GetBestMethod(methods, targetType, argTypes, ConversionOptions.NoConversions, out binding, out exception);
        }

        /// <summary>
        /// Find the method which best matches the given arguments.
        /// </summary>
        /// <param name="methods">Methods to search through</param>
        /// <param name="targetType">The type of <c>this</c>, for instance methods.  If looking for a static method, use null.</param>
        /// <param name="argTypes">Types.  argTypes.Length == number of method parameters.  argTypes[i] may be null to allow any type, or typeof(Nullable) to mean "any nullable type".</param>
        /// <param name="conversionOptions">Specifies which conversions are allowed</param>
        /// <param name="binding">Modified to contain the generic type arguments and argument conversions needed for calling the method</param>
        /// <param name="exception">Exception created on failure</param>
        /// <returns>A non-null MethodBase.</returns>
        /// <exception cref="ArgumentException">The best matching type parameters did not satisfy the constraints of the generic method.</exception>
        /// <exception cref="MissingMethodException">No match was found.</exception>
        public static MethodBase GetBestMethod(MethodBase[] methods, Type targetType, Type[] argTypes, ConversionOptions conversionOptions, out Binding binding,
                                               out Exception exception)
        {
            Type[] instance_actuals = argTypes;
            Type[] static_actuals = argTypes;
            if (targetType != null)
            {
                static_actuals = AddFirst(argTypes, targetType);
            }
            else if (argTypes.Length > 0)
            {
                // targetType == null
                instance_actuals = ButFirst(argTypes);
            }
            exception = new MissingMethodException("The arguments (" + PrintTypes(argTypes) + ") do not match any overload of " + methods[0].Name);
            binding = null;
            MethodBase bestMethod = null;
            foreach (MethodBase method in methods)
            {
                Binding b = Binding.GetBestBinding(method, (method.IsStatic || method.IsConstructor) ? static_actuals : instance_actuals, conversionOptions, out exception);
                if (b != null && b < binding)
                {
                    binding = b;
                    bestMethod = method;
                }
            }
            if (bestMethod == null) return null;
            // If the method is generic, specialize on the inferred type parameters.
            return binding.Bind(bestMethod);
        }

        #region Cloning

        public class DoNotCloneAttribute : Attribute
        {
        }

        public class DoNotCloneItemsAttribute : Attribute
        {
        }

        public static bool HasAttribute(MemberInfo member, Type attributeType)
        {
            object[] attrs = member.GetCustomAttributes(true);
            foreach (object attr in attrs)
            {
                if (attr.GetType().Equals(attributeType)) return true;
            }
            return false;
        }

        /// <summary>
        /// Clone an object by reflection on its fields.
        /// </summary>
        /// <param name="o"></param>
        /// <returns></returns>
        public static object Clone(object o)
        {
            return Clone(o, true);
        }

        public static object Clone(object o, bool cloneFields)
        {
            bool debug = true;
            Type type = o.GetType();
            if (type.IsPrimitive)
            {
                return o; // no need to clone
            }
            else if (type.IsArray && !type.GetElementType().IsPrimitive)
            {
                if (type.GetArrayRank() == 1)
                {
                    Array array = (Array) o;
                    int length = array.Length;
                    Array result = (Array) Activator.CreateInstance(type, length); // or array.Clone();
                    for (int i = 0; i < length; i++)
                    {
                        object value = array.GetValue(i);
                        if (cloneFields) value = Clone(value);
                        result.SetValue(value, i);
                    }
                    return result;
                }
                else throw new NotImplementedException("Cannot clone an array of rank > 1");
            }
            else if (o is ICloneable)
            {
                // if it has a Clone() method, use it
                return ((ICloneable) o).Clone();
            }
            else
            {
                // if it has a copy constructor, use it
                try
                {
                    return Activator.CreateInstance(type, o);
                }
                catch (MissingMethodException)
                {
                }
                if (debug) Console.WriteLine("(Cloning a " + type);
                // make an empty instance and clone each field.
                object clone = Activator.CreateInstance(type);
                FieldInfo[] fields = type.GetFields(BindingFlags.Public | BindingFlags.Instance);
                foreach (FieldInfo field in fields)
                {
                    object value = field.GetValue(o);
                    // must be operator != here, not the Equals method
                    if (field.GetValue(clone) != value)
                    {
                        if (debug) Console.WriteLine(field.Name);
                        if (cloneFields && !HasAttribute(field, typeof (DoNotCloneAttribute)))
                        {
                            value = Clone(value, !HasAttribute(field, typeof (DoNotCloneItemsAttribute)));
                        }
                        field.SetValue(clone, value);
                    }
                }
                // what about events?
                if (debug) Console.WriteLine(")");
                return clone;
            }
        }

        #endregion
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}