#!/usr/bin/env python 

'''
Code adapted from https://blog.checkpoint.com/wp-content/uploads/2015/07/matsnu-malwareid-technical-brief.pdf
'''

import os
import sys
import datetime
import string

def is_hex(s):
    if not s.startswith('0x'):
        return False
    s = s[2:]
    hex_digits = set(string.hexdigits)
    # if s is long, then it is faster to check against a set
    return all(c in hex_digits for c in s)

def is_valid_int(arg):
    if not is_hex(arg):
        if not arg.isdigit():
            return None
        else:
            value = int(arg)
    else:
        value = int(arg, 16)
    return value


def parse_dict_file(fname):
    dict0 = []
    dict1 = []
    try:
        with open(fname, 'rb') as f:
            dict0 = f.read().split('\n')
            for i in range(len(dict0)):
                dict0[i] = dict0[i].rstrip()
                if dict0[i]:
                    dict1.append(dict0[i])
    except Exception as e:
        print 'read error: ' + str(e)
        sys.exit(1)
    return dict1


def write_file(fname, cont, separator = ''):
    try:
        with open(fname, 'wb') as f:
            for d in cont:
                f.write(d + separator)
    except Exception as e:
        print 'Write error: ' + str(e)
        sys.exit(1)


def append_file(fname, cont, separator = ''):
    try:
        with open(fname, 'a') as f:
            for d in cont:
                f.write(d + separator)
    except Exception as e:
        print 'Write error: ' + str(e)
        sys.exit(1)


class domain_generator:
    def __init__(self, dict1, dict2):
        self.const1 = 0xef5eb
        self.const2 = 0x39339
        self.dict1 = dict1
        self.dict2 = dict2

    def get_days_since_epoch(self):
        epoch = datetime.datetime.utcfromtimestamp(0)
        today = datetime.datetime.today()
        d = today - epoch
        return d.days


    def choose_next_word(self, dictionary):
        self.seed &= 0xffff
        self.seed = (self.seed * self.const1) & 0xffff
        self.seed = (self.seed * self.time) & 0xffff
        self.seed = (self.seed * self.const2) & 0xffff
        self.seed = (self.seed * self.next_domain_no) & 0xffff
        self.seed = (self.seed ^ self.const1) & 0xffff
        rem = self.seed % len(dictionary)
        return dictionary[self.seed % len(dictionary)]

    def generate_domain(self, include_tld=True):
        domain = ''
        self.parity_flag = 0
        
        while len(domain) < 0x18:
            if len(domain) > 0xc:
                break
            if len(domain) == 0:
                domain += self.choose_next_word(self.dict1)
            elif self.parity_flag == 0:
                domain += self.choose_next_word(self.dict1)
            else:
                domain += self.choose_next_word(self.dict2)
            
            self.parity_flag = (self.parity_flag + 1) % 2
            if self.seed & 0x1 == 0x1:
                domain += '-'

        if domain[-1] == '-':
            domain = domain[:-1]

        if include_tld:
            domain += '.com'
        self.next_domain_no += 1
        return domain

    def generate_domains(self, loops, domains, time, include_tld=True):
        domains_list = []
        # DGA works as follows: generate domains for the current and loops - 1 previous days
        time -= (loops - 1)
        for l in range(loops):
            self.seed = 1
            self.next_domain_no = 1
            self.time = time + l

            for d in range(domains):
                domains_list.append(self.generate_domain(include_tld=include_tld))
        return domains_list

def unique_list(l):
    rl = []
    for e in l:
        if e not in rl:
            rl.append(e)
    return rl

def days_since_epoch(d):
    epoch = datetime.datetime.utcfromtimestamp(0)
    dse = d - epoch
    return dse.days

def domains_gen(date_from, date_to, dict1, dict2, include_tld=True):
    dga = domain_generator(dict1, dict2)
    domains = []
    for d in range(date_from, date_to + 1):
        dd = dga.generate_domains(3, 10, d, include_tld=include_tld)
        domains += dd
    return domains


def generate_domains(num_domains, include_tld=True):
    dict1 = parse_dict_file(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'matsnu_dict1.txt'))
    dict2 = parse_dict_file(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'matsnu_dict2.txt'))
    
    date_from = datetime.datetime.strptime('01.01.2015', '%d.%m.%Y')
    days_from = days_since_epoch(date_from)

    date_to = datetime.datetime.strptime('01.01.2019', '%d.%m.%Y')
    days_to = days_since_epoch(date_to)
    return domains_gen(days_from, days_to, dict1, dict2, include_tld=include_tld)[:num_domains]


def main():
    if len(sys.argv) < 8:
        print 'usage: ' + sys.argv[0] + '--from from-date --to to-date dict1 dict2 outfile [--unique-domains]'
        sys.exit(1)
    
    dict1 = parse_dict_file(sys.argv[5])
    dict2 = parse_dict_file(sys.argv[6])

    if sys.argv[1] != '--from':
        print 'Invalid arg: ' + sys.argv[1] + ', should be --from'
        sys.exit(1)
    
    date_from = datetime.datetime.strptime(sys.argv[2], '%d.%m.%Y')
    days_from = days_since_epoch(date_from)
    if sys.argv[3] != '--to':
        print 'Invalid arg: ' + sys.argv[3] + ', should be --to'
        sys.exit(1)
    
    date_to = datetime.datetime.strptime(sys.argv[4], '%d.%m.%Y')
    days_to = days_since_epoch(date_to)
    if days_from > days_to:
        print '--from date should be less equal than --to date'
        return sys.exit(1)

    print '[+] Generating domains...'
    domains = domains_gen(days_from, days_to, dict1, dict2)
    print '[+] Domains were generated'
    if len(sys.argv) > 8:
        if sys.argv[8] == '--unique-domains':
            print '[+] Cleaning domains...'
            domains = unique_list(domains)
            print '[+] Domains were cleaned'
    
    dom_metadata = [ 'From: ' + sys.argv[2], 'To:' + sys.argv[4], 'DGA:' ]
    for d in domains:
        dom_metadata.append(d)
        write_file(sys.argv[7], dom_metadata, '\r\n')

if __name__ == '__main__':
    main()
    sys.exit(0)