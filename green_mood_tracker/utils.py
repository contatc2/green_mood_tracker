import time


def map_example_to_dict(input_ids, attention_masks, label):
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    }, label


################
#  DECORATORS  #
################

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed
