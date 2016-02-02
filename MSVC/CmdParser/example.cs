using System;

namespace CmdParser
{
    class AppArguments
    {
        [Argument(ArgumentType.Required, ShortName = "", HelpText = "Starting number of connections.")]
        public int start;
        [Argument(ArgumentType.Required, HelpText = "Maximum number of connections.")]
        public int max;
        [Argument(ArgumentType.Required, ShortName = "inc", HelpText = "Number of connections to increment, if needed.")]
        public int increment;
        [DefaultArgument(ArgumentType.AtMostOnce, DefaultValue = @"C:\foo\config.xml", HelpText = "Path to TServer config file.")]
        public string configpath;
    }

    class WCArguments
    {
        [Argument(ArgumentType.AtMostOnce, DefaultValue = true, HelpText = "Count number of lines in the input text.")]
        public bool lines;
        [Argument(ArgumentType.AtMostOnce, DefaultValue = false, HelpText = "Count number of words in the input text.")]
        public bool words;
        [Argument(ArgumentType.AtMostOnce, HelpText = "Count number of chars in the input text.")]
        public bool chars;
        [DefaultArgument(ArgumentType.MultipleUnique, DefaultValue = new string[] { "a", "b" }, HelpText = "Input files to count.")]
        public string[] files;
    }

    class WC
    {
        static void Main(string[] args)
        {
            WCArguments parsedArgs = new WCArguments();
            if (Parser.ParseArgumentsWithUsage(args, parsedArgs))
            {
                // insert application code here
                Console.WriteLine(parsedArgs.lines);
                Console.WriteLine(parsedArgs.words);
                Console.WriteLine(parsedArgs.chars);
                Console.WriteLine(parsedArgs.files.Length);
            }
            Parser.ParseArgumentsWithUsage(args, new AppArguments());
        }
    }
}