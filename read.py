from collections import Counter
import numpy as np
import pickle
import time
word_counter = Counter()
counter_per_word = dict()
pmi_matrix = {}
word_to_index = dict()
probability_per_word = dict()
attribute_to_set = dict()
functional_words = set(['IN','DT','TO','CC',','])

def get_list_of_sentences(file_name):
    global word_counter
    all_sentences  = []
    s = []
    with open(file_name) as f:
        for i,line in enumerate(f):
            line = line.strip()
            if line == "":
                if len(s) > 0:
                    all_sentences.append(s)
                    s = []
                continue
            line = line.replace("\t", " ")
            parts = line.split()
            word = parts[2]
            if not word_to_index.has_key(parts[2]):
                word_to_index[word] = len(word_to_index)
            word_counter[word_to_index[word]] += 1
            s.append((word_to_index[word],parts[4]))

    save_conuter_to_file([all_sentences,word_counter],"all_sentences")
    return all_sentences


def get_tuples(sen):
    all_tup =[]
    for i in range(len(sen)):
        for j in range(i+1,len(sen)):
            w_first = sen[i]
            w_two   = sen[j]
            tup = (w_first,w_two)
            all_tup.append(tup)

    return all_tup


def insert_tupples_into_matrix(all_tupples):
    for tup in all_tupples:
        w_0 = tup[0]
        w_1 = tup[1]
        if not counter_per_word.has_key(w_0):
            counter_per_word[w_0] = Counter()

        if not counter_per_word.has_key(w_1):
            counter_per_word[w_1] = Counter()

        counter_per_word[w_0][w_1] += 1
        counter_per_word[w_1][w_0] += 1


def convert_to_pmi():
    sum_of_all_word = get_sum_all_words_from_counter()
    create_probability_per_word(sum_of_all_word)
    for word_index in counter_per_word.keys():
        shared_probability = counter_per_word[word_index]
        word_probability = np.power(probability_per_word[word_index],0.75)
        pmi_matrix[word_index] = {}
        this_pmi = pmi_matrix[word_index]
        for contex_index in shared_probability:
            this_pmi[contex_index] = np.maximum(0,np.log( float(shared_probability[contex_index])/ sum_of_all_word / probability_per_word[contex_index] / word_probability ))


def get_sum_all_words_from_counter():
    this_sum = 0
    for c in counter_per_word:
        this_sum += sum(counter_per_word[c].values())

    return this_sum


def create_probability_per_word(sum_of_all_word):
    for index in counter_per_word.keys():
        sum_of_word = sum(counter_per_word[index].values())
        probability_per_word[index] = float(sum_of_word) / sum_of_all_word


def reduce_matrix(counter, thrershold = 5 ):
    for i in range(len(word_to_index)):
        if not counter.has_key(i): continue
        vector = counter[i]
        to_del = []
        for contex in vector:
            if vector[contex] <= thrershold:
                print "word %0d removed %0d"%(i,contex)
                to_del.append( contex )

        for d in to_del:
            del vector[d]


def save_conuter_to_file(counter,file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(counter, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_counter_from_file(file_name):
    with open(file_name, 'rb') as handle:
        c = pickle.load(handle)
    return c


def remove_function_word_from_sentence(sentence):
    line = []
    for w in sentence:
        if not w[1] in functional_words:
            line.append(w[0])

    return line


def remove_words_which_appear_less(sentence):
    new_sen = []
    for w,pos in sentence:
        if word_counter[w] > 99:
            new_sen.append((w,pos))
    return new_sen


def create_vectors(all_sentences,k = 2 ):
    for full_sen in all_sentences:
        reduced_sentence = remove_words_which_appear_less(full_sen)
        if k ==-1:
            reduced_sentence = [s[0] for s in reduced_sentence]
            all_tups = get_tuples(reduced_sentence)
        else:
            all_tups = []
            reduced_sentence = remove_function_word_from_sentence(reduced_sentence)
            for index_word, word in enumerate(reduced_sentence):
                # word_vector = matrix_word_vectors[word]
                all_tups.extend(get_tuples(reduced_sentence[max(0,index_word-k):index_word+k]))

        insert_tupples_into_matrix(all_tups)


def fill_attribute_sets():
    for c in pmi_matrix:
        for att in pmi_matrix[c].keys():
            if not attribute_to_set.has_key(att):
                attribute_to_set[att] = set()
            attribute_to_set[att].add(c)




def find_similarities(check_for_these_words):
    final_result= []

    for w in check_for_these_words:
        counter_result_for_word = Counter()
        index_word = word_to_index[w] # change to number
        pmi_vector = pmi_matrix[index_word]  #pmi_values of word
        for p_value in pmi_vector:
            all_other_word_to_check = attribute_to_set[p_value]
            for different_word in all_other_word_to_check:
                #get_dist(index_word,different_word)
                multiplication = pmi_vector[p_value] * pmi_matrix[different_word][p_value]
                counter_result_for_word[different_word] += multiplication

        final_result.append((w,counter_result_for_word.most_common(20)))

    return final_result


def normelize_pmi():
    for word in pmi_matrix:
        word_pmi_values = pmi_matrix[word].values()
        l2_norm = np.sqrt(sum(map(lambda x: x**2,word_pmi_values)))
        this_pmi_dict = pmi_matrix[word]
        for p in this_pmi_dict:
            this_pmi_dict[p] /= l2_norm


def create_from_wikipedia():
    global pmi_matrix
    global word_to_index
    global counter_per_word
    all_sentences = get_list_of_sentences("wikipedia.sample.trees.lemmatized")
    #all_sentences,word_counter = load_counter_from_file("all_sentences")
    print "done loading"
    create_vectors(all_sentences, k=5)  # k determines size if context window. -1 means full sentence as context.
    reduce_matrix(counter_per_word, 5)

    # with open('word_to_index_reduced.pickle', 'rb') as handle:
    #     word_to_index = pickle.load(handle)


    # with open('word_vector_reduced.pickle', 'rb') as handle:
    #     counter_per_word = pickle.load(handle)

    # save_conuter_to_file(counter_per_word,"small_window_vector_per_word")

    # counter_per_word = load_counter_from_file("small_window_vector_per_word")
    convert_to_pmi()
    reduce_matrix(pmi_matrix,0.1)
    normelize_pmi()
    # save_conuter_to_file(pmi_matrix,"pmi_matrix_normalized.pickle")

    #save_conuter_to_file(pmi_matrix,"pmi_matrix.pickle")
    #pmi_matrix = load_counter_from_file("pmi_matrix_normalized.pickle")
    return pmi_matrix

def read_w2v(file = "bow5.words"):
    list_of_vec = []
    with open(file) as f:
        for i,line in enumerate(f):
            parts = line.split()
            word  = parts[0]
            vec   = parts[1:]
            word_to_index[word] = len(word_to_index)
            word_index = word_to_index[word]
            list_of_vec.append(np.array(vec, dtype=np.float64))

    return np.matrix(list_of_vec) , word_to_index


def main():
    global word_to_index
    read_from_wiki = True

    _start_time = time.time()

    check_for_these_words = "car bus hospital hotel gun bomb horse fox table bowl guitar piano"
    check_for_these_words = check_for_these_words.split()

    if (read_from_wiki):
        create_from_wikipedia()
        inv_map = {v: k for k, v in word_to_index.iteritems()}
        fill_attribute_sets()
        similarities = find_similarities(check_for_these_words)
        for w_similarity in similarities:
            print "word checked is %s" % w_similarity[0]
            for diffrernt_word in w_similarity[1]:
                other_word = inv_map[diffrernt_word[0]]
                angle = diffrernt_word[1]
                print "word %s angle = %f" % (other_word, angle)
            print
    else:
        matrix, word_to_index = read_w2v("deps.words")
        inv_map = {v: k for k, v in word_to_index.iteritems()}
        intersting_word = [ word_to_index[c] for c in check_for_these_words ]
        rows = matrix[intersting_word]
        result = np.dot(rows,matrix.T)
        indecies = np.argsort(result,axis=1)
        for i in range(len(result)):
            word  = inv_map[intersting_word[i]]
            print "word checked is %s" % word
            this_row = indecies[i,:]
            for differnt_word_index in range(20):
                other_word = this_row[0,-1-differnt_word_index]
                angle = result[i,other_word]
                print "word %s angle = %f" % (inv_map[other_word], angle)
            print ""



    print "time took %0f" % (time.time() - _start_time)


if __name__ == '__main__':
    main()

