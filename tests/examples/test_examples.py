import sys
import os
import doctest
import timecave.validation_strategy_metrics

#sys.path.append("C:/Users/User/Documents/Beatriz/timecave/timecave");

#module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'C:/Users/User/Documents/Beatriz/timecave/timecave'))
#sys.path.insert(0, module_path)

#print(sys.path);

doctest.testmod(timecave.validation_strategy_metrics, verbose=True);

#doctest.testfile("timecave.validation_strategy_metrics.py", verbose=True);
#doctest.run_docstring_examples(metrics.PAE, globals(), verbose=True, name="PAE");

#finder = doctest.DocTestFinder();
#tests = finder.find(metrics);

#print(tests);