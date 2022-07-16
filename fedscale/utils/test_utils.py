from fedscale.core.logger.aggragation import *

def aggregate_test_result(test_result_accumulator, task, round_num, global_virtual_clock, testing_history):    
    
    accumulator = test_result_accumulator[0]
    for i in range(1, len(test_result_accumulator)):
        if task == "detection":
            for key in accumulator:
                if key == "boxes":
                    for j in range(596):
                        accumulator[key][j] = accumulator[key][j] + \
                            test_result_accumulator[i][key][j]
                else:
                    accumulator[key] += test_result_accumulator[i][key]
        else:
            for key in accumulator:
                accumulator[key] += test_result_accumulator[i][key]
    if task == "detection":
        testing_history['perf'][round_num] = {'round': round_num, 'clock': global_virtual_clock,
                                                    'top_1': round(accumulator['top_1']*100.0/len(test_result_accumulator), 4),
                                                    'top_5': round(accumulator['top_5']*100.0/len(test_result_accumulator), 4),
                                                    'loss': accumulator['test_loss'],
                                                    'test_len': accumulator['test_len']
                                                    }
    else:
        testing_history['perf'][round_num] = {'round': round_num, 'clock': global_virtual_clock,
                                                    'top_1': round(accumulator['top_1']/accumulator['test_len']*100.0, 4),
                                                    'top_5': round(accumulator['top_5']/accumulator['test_len']*100.0, 4),
                                                    'loss': accumulator['test_loss']/accumulator['test_len'],
                                                    'test_len': accumulator['test_len']
                                                    }

    logging.info("FL Testing in round: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, test loss: {:.4f}, test len: {}"
                    .format(round_num, global_virtual_clock, testing_history['perf'][round_num]['top_1'],
                            testing_history['perf'][round_num]['top_5'], testing_history['perf'][round_num]['loss'],
                            testing_history['perf'][round_num]['test_len']))