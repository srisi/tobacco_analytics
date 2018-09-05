import json
import time

import redis
from tobacco.configuration import REDIS_HOST, CURRENT_MACHINE
from tobacco.secrets import get_secret


class Redis_Con:
    """ Redis_Con handles all of the interaction with the redis task manager

    This means primarily that it adds and retrieves frequency and text passage tasks.

    """

    def __init__(self):
        self.con = redis.StrictRedis(host=REDIS_HOST, port=6379, db=0,
                      password=get_secret('redis_pw_aws'))

    def push_task_passages(self, task):
        self.con.rpush('passages_tasks', json.dumps(task))

    def push_task_sections(self, task):
        self.con.rpush('sections_tasks', json.dumps(task))

    def push_task_frequencies(self, task):
        self.con.rpush('frequencies_tasks', json.dumps(task))

    def get_task_frequencies(self):
        count = 0
        while True:
            # preference for local machine.
            if CURRENT_MACHINE == 'local':
                task = self.con.lpop('frequencies_tasks')
                if task:
                    task = task.decode('utf-8')
                    return json.loads(task)
            # if task queue longer than 1, execute a task on aws
            else:
                if self.con.llen('frequencies_tasks') > 0:
                    task = self.con.lpop('frequencies_tasks')
                    if task:
                        task = task.decode('utf-8')
                        return json.loads(task)
                else:
                    time.sleep(0.2)

            count += 1
            if count == 3000:
                print("Frequencies task queue still alive...")
                count = 0
            time.sleep(0.2)


    def get_task_passages(self):
        count = 0
        while True:
            if CURRENT_MACHINE == 'aws' or CURRENT_MACHINE == 'local':
                task = self.con.blpop(['sections_tasks', 'passages_tasks'], 0)
                if task:
                    task = json.loads(task[1].decode('utf-8'))
                    task_type = task[0]
                    task_params = task[1:]
                    return task_type, task_params
            else:
                pass


    def get_task_sections(self):
        task = self.con.lpop('sections_tasks')
        if task:
            task = json.loads(task.decode('utf-8'))
            task_type = task[0]
            task_params = task[1:]
            return task_type, task_params
        else:
            return None, None


    def push_sample_task_passages(self):
        self.con.rpush('passages_tasks', json.dumps(
            (('complete', ['american spirit'], [], [], [], 1901, 2016, 400, 2000, 0.85))
        ))


if __name__ == "__main__":
    redis_con = Redis_Con()
    f = redis_con.push_sample_task_passages()
    print(f)