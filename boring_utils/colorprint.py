import pprint
import inspect
from datetime import datetime

# color patterns
RED_PATTERN          = '\033[31m%s\033[0m'
GREEN_PATTERN        = '\033[32m%s\033[0m'
BROWN_PATTERN        = '\033[33m%s\033[0m'
BLUE_PATTERN         = '\033[34m%s\033[0m'
PURPLE_PATTERN       = '\033[35m%s\033[0m'
CYAN_PATTERN         = '\033[36m%s\033[0m'

LIGHT_GRAY_PATTERN   = '\033[90m%s\033[0m'
LIGHT_RED_PATTERN    = '\033[91m%s\033[0m'
LIGHT_GREEN_PATTERN  = '\033[92m%s\033[0m'
LIGHT_YELLOW_PATTERN = '\033[93m%s\033[0m'
LIGHT_BLUE_PATTERN   = '\033[94m%s\033[0m'
LIGHT_PURPLE_PATTERN = '\033[95m%s\033[0m'
LIGHT_CYAN_PATTERN   = '\033[96m%s\033[0m'
LIGHT_WHITE_PATTERN  = '\033[97m%s\033[0m'

BG_RED_PATTERN       = '\033[41m%s\033[0m'
BG_GREEN_PATTERN     = '\033[42m%s\033[0m'
BG_BLUE_PATTERN      = '\033[44m%s\033[0m'

BOLD_PATTERN         = '\033[1m%s\033[0m'
UNDERLINE_PATTERN    = '\033[4m%s\033[0m'
ITALIC_PATTERN       = '\033[3m%s\033[0m'


def _get_caller_path(frame, include_method=True):
    class_or_func_name = frame.f_code.co_name
    
    if 'self' not in frame.f_locals:
        return class_or_func_name
        
    class_name = frame.f_locals['self'].__class__.__name__

    if include_method:
        method_path = []
        current_frame = frame
        
        ignore_methods = {'<module>', '_bootstrap', '_bootstrap_inner', 
                         'wrapper', 'decorator', '__init__', 'execute_function'}
        
        while current_frame:
            func_name = current_frame.f_code.co_name
            if (func_name not in ignore_methods and 
            not func_name.startswith('_') and 
            func_name != class_or_func_name):
                method_path.insert(0, func_name)
            current_frame = current_frame.f_back
        
        method_path.insert(0, class_name)
        return '.'.join(method_path)
    else:
        return '.'.join([class_name, class_or_func_name])


def mprint(obj, magic_methods=False, private_methods=True, public_methods=True):
    # Split items based on their types
    
    if private_methods:
        magic_methods = [x for x in dir(obj) if x.startswith("__") and x.endswith("__")]
        private_methods = [x for x in dir(obj) if x.startswith("_") and not x in magic_methods]
        print("\n\033[93mPrivate Methods:\033[0m")
        for item in sorted(private_methods):
            print(f"    {item}")

        if magic_methods:
            print("\n\033[93mMagic Methods:\033[0m")
            for item in sorted(magic_methods):
                print(f"    {item}")
    
    if public_methods:
        public_methods = [x for x in dir(obj) if not x.startswith("_")]
        print("\n\033[93mPublic Methods:\033[0m")
        for item in sorted(public_methods):
            print(f"    {item}")


def cprint(*exprs, c=None, class_name=True, use_pprint=True, new_line=False, include_method=False, timestamp=False):
    """
    Custom print function that prints the name of the variable/expression
    alongside its value.
    
    Parameters:
    - *exprs: Variable-length argument list of expressions to evaluate.
    - class_name (bool, optional): If True, prints the class name or function name along with the variable name.
    - timestamp (bool, optional): If True, adds timestamp info at the beginning of the output.
    """
    try:
        p = pprint.pprint if use_pprint else print
        # Fetch the line of code that called cprint
        frame = inspect.currentframe().f_back

        # Get multiple lines of context to handle multiline calls
        frameinfo = inspect.getframeinfo(frame)
        lineno = frameinfo.lineno
        filename = frameinfo.filename

        # Try to get the complete function call across multiple lines
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                start_line = lineno - 1

                # Find the opening parenthesis
                call_line = ''
                paren_count = 0
                for i in range(start_line, min(start_line + 20, len(lines))):
                    line = lines[i]
                    call_line += line
                    paren_count += line.count('(') - line.count(')')
                    if paren_count == 0 and '(' in call_line:
                        break

                call_line = call_line.strip()
        except:
            # Fallback to original method
            call_line = frameinfo.code_context[0].strip()

        # Extract the arguments from the line
        arg_str = call_line[call_line.index('(') + 1:call_line.rindex(')')].strip()

        # Split the arguments by comma, keeping expressions intact
        arg_list = []
        bracket_count = 0
        current_arg = []
        for char in arg_str:
            if char == ',' and bracket_count == 0:
                arg_list.append(''.join(current_arg).strip())
                current_arg = []
            else:
                if char in '([{':
                    bracket_count += 1
                elif char in ')]}':
                    bracket_count -= 1
                current_arg.append(char)
        if current_arg:
            arg_list.append(''.join(current_arg).strip())

        # Check if there are any arguments
        if not arg_list or (len(arg_list) == 1 and not arg_list[0]):
            print()  # Print an empty line
            return

        for arg, expr in zip(arg_list, exprs):
            try:
                if class_name:
                    frame = inspect.currentframe().f_back
                    caller_path = _get_caller_path(frame, include_method)
                    arg = f"{caller_path} -> {arg}"
                
                # Add timestamp if requested
                if timestamp:
                    timestamp_str = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    arg = timestamp_str + arg

                end = '\n' if new_line else ' '
                if not c:               print(LIGHT_YELLOW_PATTERN % f"{arg}:", end=end)
                elif c == 'red':        print(RED_PATTERN % f"{arg}:", end=end)
                elif c == 'green':      print(GREEN_PATTERN % f"{arg}:", end=end)
                elif c == 'blue':       print(BLUE_PATTERN % f"{arg}:", end=end)
                elif c == 'brown':      print(BROWN_PATTERN % f"{arg}:", end=end)
                elif c == 'purple':     print(PURPLE_PATTERN % f"{arg}:", end=end)
                elif c == 'cyan':       print(CYAN_PATTERN % f"{arg}:", end=end)
                elif c == 'gray':       print(LIGHT_GRAY_PATTERN % f"{arg}:", end=end)
                elif c == 'light_red':  print(LIGHT_RED_PATTERN % f"{arg}:", end=end)
                elif c == 'light_green':print(LIGHT_GREEN_PATTERN % f"{arg}:", end=end)
                elif c == 'yellow':     print(LIGHT_YELLOW_PATTERN % f"{arg}:", end=end)
                elif c == 'light_blue': print(LIGHT_BLUE_PATTERN % f"{arg}:", end=end)
                elif c == 'light_purple':print(LIGHT_PURPLE_PATTERN % f"{arg}:", end=end)
                elif c == 'light_cyan': print(LIGHT_CYAN_PATTERN % f"{arg}:", end=end)
                elif c == 'white':      print(LIGHT_WHITE_PATTERN % f"{arg}:", end=end)
                elif c == 'bg_red':     print(BG_RED_PATTERN % f"{arg}:", end=end)
                elif c == 'bg_green':   print(BG_GREEN_PATTERN % f"{arg}:", end=end)
                elif c == 'bg_blue':    print(BG_BLUE_PATTERN % f"{arg}:", end=end)
                elif c == 'bold':       print(BOLD_PATTERN % f"{arg}:", end=end)
                elif c == 'underline':  print(UNDERLINE_PATTERN % f"{arg}:", end=end)
                elif c == 'italic':     print(ITALIC_PATTERN % f"{arg}:", end=end)
                else:                   print(arg, end=end)
                p(expr)

            except Exception as e:
                print(f"Error evaluating {arg}: {e}")

    except (AttributeError, ValueError, IndexError):
        for expr in exprs:
            if class_name:
                frame = inspect.currentframe().f_back
                caller_path = _get_caller_path(frame, include_method)
                prefix = f"{caller_path} ->"
                if timestamp:
                    prefix = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {prefix}"
                print(prefix, end=' ')
            elif timestamp:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", end=' ')
            p(expr)


def sprint(*exprs, globals=None, locals=None):
    """
    Custom print function that prints the name of the variable/expression
    alongside its value.
    
    Parameters:
    - *exprs (str): Variable-length argument list of expressions to evaluate as strings.
    - globals, locals (dict, optional): Optional dictionaries to specify the namespace
      for the evaluation. This allows the function to access variables outside of its
      local scope.
    """
    # Check if there are any arguments
    if not exprs:
        print()  # Print an empty line
        return
    
    for expr in exprs:
        try:
            # Evaluate the expression
            value = eval(expr, globals, locals)
            print(f"\033[93m{expr}\033[0m: \n{value}\n")
        except Exception as e:
            print(f"Error evaluating {expr}: {e}")


def tprint(title='', sep='=', c=None, class_name=True, include_method=False, new_line=False, timestamp=True):
    """
    Print a title with separators.
    
    Parameters:
    - title (str): The title to print.
    - sep (str, optional): The separator character. Default is '='.
    - c (str, optional): The color of the output. Options are 'red', 'green', 'blue', 'yellow', 'pep', 'brown', or None for default color.
    - class_name (bool, optional): If True, prints the class name or function name along with the title.
    - new_line (bool, optional): If True, starts the output with a newline. Default is True.
    - timestamp (bool, optional): If True, adds timestamp info at the beginning of the output.
    """
    try:
        separator = sep * (5 // len(sep))
        
        # Add timestamp prefix if requested
        timestamp_prefix = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " if timestamp else ""

        if class_name:
            frame = inspect.currentframe().f_back
            caller_path = _get_caller_path(frame, include_method)

            if title == '': 
                content = f'{separator} {caller_path} {separator}'
            else: 
                content = f'{separator} {caller_path} -> {title} {separator}'
        else:
            if title == '': 
                content = f'{separator}{separator}'
            else: 
                content = f'{separator} {title} {separator}'
        
        # Combine timestamp and content, add newline if requested
        newline_prefix = '\n' if new_line else ''
        output = f'{newline_prefix}{timestamp_prefix}{content}'

        if not c:
            if sep == '=': print(PURPLE_PATTERN % output)
            else:          print(BROWN_PATTERN % output)
        elif c == 'red':          print(RED_PATTERN % output)
        elif c == 'green':        print(GREEN_PATTERN % output)
        elif c == 'blue':         print(BLUE_PATTERN % output)
        elif c == 'brown':        print(BROWN_PATTERN % output)
        elif c == 'purple':       print(PURPLE_PATTERN % output)
        elif c == 'cyan':         print(CYAN_PATTERN % output)
        elif c == 'gray':         print(LIGHT_GRAY_PATTERN % output)
        elif c == 'light_red':    print(LIGHT_RED_PATTERN % output)
        elif c == 'light_green':  print(LIGHT_GREEN_PATTERN % output)
        elif c == 'yellow':       print(LIGHT_YELLOW_PATTERN % output)
        elif c == 'light_blue':   print(LIGHT_BLUE_PATTERN % output)
        elif c == 'light_purple': print(LIGHT_PURPLE_PATTERN % output)
        elif c == 'light_cyan':   print(LIGHT_CYAN_PATTERN % output)
        elif c == 'white':        print(LIGHT_WHITE_PATTERN % output)
        elif c == 'bg_red':       print(BG_RED_PATTERN % output)
        elif c == 'bg_green':     print(BG_GREEN_PATTERN % output)
        elif c == 'bg_blue':      print(BG_BLUE_PATTERN % output)
        elif c == 'bold':         print(BOLD_PATTERN % output)
        elif c == 'underline':    print(UNDERLINE_PATTERN % output)
        elif c == 'italic':       print(ITALIC_PATTERN % output)
        else:                     print(output)

    except (AttributeError, ValueError, IndexError):
        separator = sep * (20 // len(sep))
        timestamp_prefix = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " if timestamp else ""
        newline_prefix = '\n' if new_line else ''
        
        if title:
            content = f'{separator} {title} {separator}'
        else:
            content = f'{separator}{separator}'
            
        output = f'{newline_prefix}{timestamp_prefix}{content}'
        print(PURPLE_PATTERN % output if sep == '=' else BROWN_PATTERN % output) 
