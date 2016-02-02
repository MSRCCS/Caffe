using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CmdParser
{
    /// <summary>
    /// This class is designed to extend CmdParser, so that it supports multiple task parsing
    /// For example:
    ///     Application <taskname> [options]
    /// The [options] will be parsed by CmdParser.
    /// </summary>
    public class ParserX
    {
        static Dictionary<string, Tuple<string, object, Action<object>>> _tasks = new Dictionary<string,Tuple<string, object, Action<object>>>(StringComparer.OrdinalIgnoreCase);
        static string _currentTask = string.Empty;
        
        /// <summary>
        /// Add task by specifying taskFunc and description for help text
        /// </summary>
        /// <param name="taskFunc">Task function to be invoked</param>
        /// <param name="taskDescription">Task description</param>
        public static void AddTask<T>(Action<T> taskFunc, string taskDescription) where T: new ()
        {
            string taskName = taskFunc.Method.Name;
            _tasks.Add(taskName, new Tuple<string, object, Action<object>>(taskDescription, new T(), new Action<object>(obj => taskFunc((T)obj))));
        }

        /// <summary>
        /// Parses Command Line Arguments. Displays usage message to Console.Out
        /// if /?, /help or invalid arguments are encounterd.
        /// Errors are output on Console.Error.
        /// Use ArgumentAttributes to control parsing behaviour.
        /// </summary>
        /// <returns></returns>
        public static bool ParseArgumentsWithUsage(string[] arguments)
        {
            if (Parser.ParseHelp(arguments) 
                || arguments.Length == 0
                || (arguments.Length > 0 && !_tasks.ContainsKey(arguments[0]))
               )
            {
                // Help needed or error encountered in arguments. Display usage message
                string exeName = System.IO.Path.GetFileNameWithoutExtension(System.Reflection.Assembly.GetEntryAssembly().Location);
                Console.WriteLine("Usage: {0} [tasktype]\n", exeName);
                Console.WriteLine("Task type:");
                foreach (var kv in _tasks)
                    Console.WriteLine("{0, 16}  {1}", kv.Key, kv.Value.Item1);
                return false;
            }

            _currentTask = arguments[0];

            if (!Parser.ParseArgumentsWithUsage(arguments.Skip(1).ToArray(), _tasks[arguments[0]].Item2))
                return false;

            return true;
        }

        /// <summary>
        /// Run task according to the cmd parsing result
        /// </summary>
        public static void RunTask()
        {
            if (string.IsNullOrEmpty(_currentTask))
                throw new ArgumentException("Please call ParseArgumentsWithUsage first");

            var task = _tasks[_currentTask];
            object cmd = task.Item2;
            Action<object> action = task.Item3;
            action(cmd);
        }

    }
}
