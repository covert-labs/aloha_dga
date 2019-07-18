"""
Code from: https://github.com/baderj/domain_generation_algorithms/tree/master/suppobox

    generate domains according to: 
    - https://www.endgame.com/blog/malware-with-a-personal-touch.html
    - http://www.rsaconference.com/writable/presentations/file_upload/br-r01-end-to-end-analysis-of-a-domain-generating-algorithm-malware-family.pdf 

    requires words1.txt, words2.txt and words3.txt

    Thanks to Sandor Nemes who provided the third wordlist. It is taken
    from this sample:
    https://www.virustotal.com/en/file/4ee8484b95d924fe032feb8f26a44796f37fb45eca3593ab533a06785c6da8f8/analysis/
"""
import time
from datetime import datetime
import argparse
import os

def generate_domains(num_domains, include_tld=True):
    time_ = time.mktime(datetime.strptime("2015-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple())
    words1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'suppobox_words1.txt')
    words2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'suppobox_words2.txt')
    words3 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'suppobox_words3.txt')
    ret = []
    ret += _generate_domains(time_, words1, num_domains//3 + 1, include_tld=include_tld)
    ret += _generate_domains(time_, words2, num_domains//3, include_tld=include_tld)
    ret += _generate_domains(time_, words3, num_domains//3, include_tld=include_tld)
    return ret

def _generate_domains(time_, word_list, num_domains, include_tld=True):
    results = []
    with open(word_list, "r") as r:
        words = [w.strip() for w in r.readlines()]

    if not time_:
        time_ = time.time()
    seed = int(time_) >> 9
    for c in range(num_domains):
        nr = seed
        res = 16*[0]
        shuffle = [3, 9, 13, 6, 2, 4, 11, 7, 14, 1, 10, 5, 8, 12, 0]
        for i in range(15):
            res[shuffle[i]] = nr % 2
            nr = nr >> 1

        first_word_index = 0
        for i in range(7):
            first_word_index <<= 1
            first_word_index ^= res[i]

        second_word_index = 0
        for i in range(7,15):
            second_word_index <<= 1
            second_word_index ^= res[i]
        second_word_index += 0x80

        first_word = words[first_word_index]
        second_word = words[second_word_index]
        tld = ''
        if include_tld:
            tld = ".net"
        results.append("{}{}{}".format(first_word, second_word, tld))
        seed += 1

    return results
