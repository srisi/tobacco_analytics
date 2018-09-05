import socket
from tobacco.configuration import PATH_OCR_FILES, PATH_TOKENIZED, DOC_COUNT
from tobacco.utilities.databases import Database

import io
import pickle
import gzip

import numpy as np




'''
6/9/17

Major observations on string encoding and loading speed:

The main problem is actually not loading data from disk but rather encoding it (at least on my desktop system)
For the text passage search, it therefore makes sense to work by and large with byte arrays instead of strings
and to only decode the small passages where it's absolutely necessary.

The new load_ocr_by_tid first loads the doc into a bytearray and only converts it if necessary.

The logic for the function is a bit convoluted: I haven't figured out how to best find the end of the bytearray and
initializing long bytearrays is expensive. Hence, I created a dict that stores the exact number of bytes needed for
every single document.
Which is not very efficient but seems to work.

'''

# LOCAL = False
# if socket.gethostname() == 'stephan-desktop':
#     LOCAL = True


def get_ocr_by_tid(tid, return_bytearray=True):

    filepath = '{}{}/{}/{}/{}/{}'.format(PATH_OCR_FILES, tid[0], tid[1], tid[2], tid[3], tid + ".txt")

    try:
        b = bytearray(TID_TO_FILELENGTH_DICT[tid])
        try:
            f = io.open(filepath, 'rb')
            f.readinto(b)
        except (IOError, FileNotFoundError):
            print("Could not load {}. Could not find file in {}".format(tid, filepath))
            b = bytearray()

    except KeyError:
        print("Could not load {}. TID not in tid_to_filelength_dict".format(tid))
        b = bytearray()

    if return_bytearray:
        return b
    else:
        return b.decode('cp1252', errors='ignore')


def get_ocr_sections(tid, sections, return_bytearray=True):
    '''

    :param tid:
    :param sections: list of tuples with start and end points
    :param return_bytearray:
    :return:
    '''


    filepath = '{}{}/{}/{}/{}/{}'.format(PATH_OCR_FILES, tid[0], tid[1], tid[2], tid[3], tid + ".txt")
    text_sections = []

    try:
        with io.open(filepath, 'rb') as f:
            for section in sections:
                start_id, end_id = section

                if end_id+1 - start_id < 0:
                    print(tid, start_id, end_id, end_id - start_id)

                b = bytearray(end_id + 1 - start_id)
                f.seek(start_id)
                f.readinto(b)
                if return_bytearray:
                    text_sections.append(b)
                else:
                    text_sections.append(b.decode('cp1252', errors='ignore'))

#    except (IOError, FileNotFoundError):
    except KeyError:
        print("Could not load {}. Could not find file in {}".format(tid, filepath))
        return []

    return text_sections


def get_tid_to_filelength_dict():

    try:
        tid_to_filelength_dict = pickle.load(gzip.open(PATH_TOKENIZED + 'tid_to_filelength_dict.pickle', 'rb'))

    except IOError:
        print("Preprocessed tid_to_filelength_dict not available. Creating a new one.")

        # tid_to_filelength_dict = {}
        tid_to_filelength_arr = np.zeros(DOC_COUNT, dtype=np.int64)

        db = Database('TOB_FULL')
        con, cur = db.connect()

        cur.execute('SELECT tid from docs')
        count = 0
        while True:
            count += 1
            if count % 10000 == 0:
                print(count)
            row = cur.fetchone()
            if not row: break
            tid = row['tid']

            filepath = '{}{}/{}/{}/{}/{}'.format(PATH_OCR_FILES, tid[0], tid[1], tid[2], tid[3], tid + ".txt")

            array_len = 10000
            end = None
            while True:
                b = bytearray(array_len)
                f = io.open(filepath, 'rb')
                f.readinto(b)
                str = b.decode('cp1252', errors='ignore')
                end = str.find('\x00')
                if end > -1:
                    break
                else:
                    array_len *= 10

#            tid_to_filelength_dict[tid] = end
            tid_to_filelength_arr[tid] = end
#        pickle.dump(tid_to_filelength_dict, gzip.open(PATH_TOKENIZED + 'tid_to_filelength_dict.pickle', 'wb'))
        pickle.save(PATH_TOKENIZED + 'tid_to_filelength_arr.npy', tid_to_filelength_arr)
        print("Longest file is {} bytes long.".format(max(tid_to_filelength_dict.values())))


    # if the number of tids in the dict != DOC_COUNT, something is wrong
    assert len(tid_to_filelength_dict) == DOC_COUNT, "Length of tid_to_filelength_dict ({}) does not equal DOC_COUNT ({})".format(len(tid_to_filelength_dict), DOC_COUNT)
    return tid_to_filelength_dict

TID_TO_FILELENGTH_DICT = get_tid_to_filelength_dict()

if __name__ == "__main__":

    pass
#    get_tid_to_filelength_dict()
#    t = get_ocr_section('rrnn0216', 23646, 26342)
    sections = [(5473947, 5475041)]
    # sections = [(0, 1000)]
    t = get_ocr_sections('sydw0179', sections)
    print(t)
    t = get_ocr_sections('sydw0179', [(5453947, 5495041)])
    print(t)



    '''
        '''