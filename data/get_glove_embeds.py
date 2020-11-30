from multiprocessing import Process, Queue
import pdb


def read_file_to_dict(file_location, queue, is_embeds=False):
    '''Reads in a file at a location and stores the data in the return item'''
    if is_embeds:
        return_item = {}
    else:
        return_item = []

    with open(file_location, 'r') as f:
        for line in f:
            if is_embeds:
                data_arr = line.split(' ')
                token = data_arr[0].strip()
                embed = data_arr[1:]
                return_item[token] = embed
            else:
                return_item.append(line.strip())

    if is_embeds:
        queue.put(('EMBEDS', return_item))
    else:
        queue.put(('WORDS', return_item))


def main():
    '''Gets the requested embeds for a list of words'''

    embeds_file_location = 'data/glove_embeds.txt'
    desired_words_file_location = 'data/all_words.txt'

    queue = Queue()

    get_words_thread = Process(target=read_file_to_dict,
                               args=(desired_words_file_location, queue))
    get_embeds_thread = Process(target=read_file_to_dict,
                                args=(embeds_file_location,
                                      queue, True))

    threads = [get_words_thread, get_embeds_thread]

    for t in threads:
        t.start()

    embeds_dict = {}
    words_list = []

    for _ in range(2):
        out = queue.get()
        if out[0] == 'EMBEDS':
            embeds_dict = out[1]
        elif out[0] == 'WORDS':
            words_list = out[1]

    output_file_location = 'data/vocab_glove_embeds.txt'

    # Write the embeds out
    with open(output_file_location, 'w') as output_file:
        unk_embed = ' '.join(embeds_dict['unk'])
        for word in words_list:
            output_file.write(word + ' ')
            if word in embeds_dict:
                output_file.write(' '.join(embeds_dict[word]))
            else:
                output_file.write(unk_embed)


main()
