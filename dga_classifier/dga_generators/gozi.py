from datetime import datetime, timedelta
from ctypes import c_uint
import argparse
import os


wordlists = {'luther': (4, '.com'), 'rfc4343': (3, '.com'), 'nasa': (5, '.com')}

seeds = {
        'luther': {'div': 4, 'tld': '.com', 'nr': 12},
        'rfc4343': {'div': 3, 'tld': '.com', 'nr': 10},
        'nasa': {'div': 5, 'tld': '.com', 'nr': 12},
        'gpl': {'div': 5, 'tld': '.ru', 'nr': 10}
        }
        
        
class Rand:

    def __init__(self, seed):
        self.r = c_uint(seed) 

    def rand(self):
        self.r.value = 1664525*self.r.value + 1013904223
        return self.r.value

def get_words(wordlist):
    fn = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gozi_{}.txt'.format(wordlist))
    with open(fn, 'r') as r:
        return [w.strip() for w in r if w.strip()]

def dga(date, wordlist, include_tld=True):
    words = get_words(wordlist)
    diff = date - datetime.strptime("2015-01-01", "%Y-%m-%d")
    days_passed = (diff.days // seeds[wordlist]['div'])
    flag = 1
    seed = (flag << 16) + days_passed - 306607824
    r = Rand(seed) 

    for i in range(12):
        r.rand()
        v = r.rand()
        length = v % 12 + 12
        domain = ""
        while len(domain) < length:
            v = r.rand() % len(words)
            word = words[v] 
            l = len(word)
            if not r.rand() % 3:
                l >>= 1
            if len(domain) + l <= 24:
                domain += word[:l]
        if include_tld:
            domain += seeds[wordlist]['tld']
        yield domain

def generate_domains(num_domains, include_tld=True):
    base_date = datetime.strptime("2015-01-01", "%Y-%m-%d")
    
    ret = []

    # 12 domains per day
    for delta in range(num_domains//12):
        d = base_date + timedelta(days=delta)
        ret += [x for x in dga(d, 'luther', include_tld=include_tld)]
        ret += [x for x in dga(d, 'nasa', include_tld=include_tld)]
        ret += [x for x in dga(d, 'rfc4343', include_tld=include_tld)]
        ret += [x for x in dga(d, 'gpl', include_tld=include_tld)]
        if len(ret) > num_domains:
            break
    return ret[:num_domains]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gozi dga")
    parser.add_argument("-d", "--date", 
            help="date for which to generate domains")
    parser.add_argument("-w", "--wordlist", help="wordlist", 
            choices=seeds.keys(), default='luther')
    args = parser.parse_args()
    if args.date:
        d = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        d = datetime.now()

    for domain in dga(d, args.wordlist):
        print(domain)
