import multiprocessing as mp
from tobacco.utilities.type_checker import check_param_type
from typing import Union, Iterable, Callable

FLAG_ALL_DONE = "WORK FINISHED"
FLAG_WORKER_FINISHED_PROCESSING = "WORKER FINISHED PROCESSING"

class MultiProcessor:
    """
    Class to run parallelizable tasks

    >>> m = MultiProcessor(square, [i for i in range(10)], {})
    >>> results = m.run_parallel(5)
    >>> results
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    """

    def __init__(self, worker_function: Callable, values_to_process: Iterable,
                 function_statics: Union[dict, None]=None):
#                 global_args: dict):

        check_param_type(worker_function, Callable, 'worker_function', 'MultiProcessor')
#        check_param_type(global_args, dict, 'global_args', 'MultiProcessor')

        self.worker_function = worker_function
        self.values_to_process = values_to_process
        self.function_statics = function_statics
#        self.global_args = global_args


    def run_parallel(self, no_processes: Union[int, None]):

        check_param_type(no_processes, Union[None, int], 'no_processes', 'Multiprocessor')
        if not no_processes:
            no_processes = mp.cpu_count() - 1

        entry_queue = mp.Queue()
        results_queue = mp.Queue()
        for idx, value_to_process in enumerate(self.values_to_process):
            entry_queue.put({'entry_id': idx, 'value_to_process': value_to_process})
        for i in range(no_processes): entry_queue.put(FLAG_ALL_DONE)

        worker_args = (entry_queue, results_queue)
        for process_n in range(no_processes):
            p = mp.Process(target = self._worker_process, args=(worker_args))
            p.start()

        threads_finished = 0
        results = []
        temp_results = {}
        cur_results_id = 0
        while True:
            new_result = results_queue.get()
            if new_result == FLAG_WORKER_FINISHED_PROCESSING:
                threads_finished += 1
                if threads_finished == no_processes:
                    break
            else:
                temp_results[new_result['entry_id']] = new_result['result']

                while True:
                    if not cur_results_id in temp_results:
                        break
                    else:
                        results.append(temp_results[cur_results_id])
                        del temp_results[cur_results_id]
                        cur_results_id += 1

        return results

    def run_single_threaded(self):
        """
        >>> m = MultiProcessor(square, [i for i in range(10)], {})
        >>> results = m.run_single_threaded()
        >>> results
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

        """

        results = []
        for value_to_process in self.values_to_process:
            if self.function_statics:
                result = self.worker_function(value_to_process, self.function_statics)
            else:
                result = self.worker_function(value_to_process)
            results.append(result)
        return results

    def _worker_process(self, entry_queue, results_queue):

        while True:
            entry = entry_queue.get()
            if entry == FLAG_ALL_DONE:
                results_queue.put(FLAG_WORKER_FINISHED_PROCESSING)
                break
            else:


                if self.function_statics:
                    result = self.worker_function(entry_queue['value_to_process'],
                                                  self.function_statics)
                else:
                    result = self.worker_function(entry['value_to_process'])
                results_queue.put({'entry_id': entry['entry_id'], 'result': result})


def square(i):
    return i * i

def squarew(i, statics):

    x = statics['x']
    return i*i+x



if __name__ == '__main__':

    m = MultiProcessor(square, [i for i in range(10)])
    results = m.run_parallel(5)
    results = m.run_single_threaded()
    print(results)

    n = MultiProcessor(squarew, [i for i in range(10)], function_statics={'x': 10})
    results = n.run_single_threaded()
    print(results)