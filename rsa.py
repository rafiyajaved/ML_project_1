#!/usr/bin/python
import json, sys, hashlib


#TODO: calculate the message m given ciphertext c and e, d, N
def rsa(c, e, d, N):

    """
    :param c: The cipher text
    :param N,e: The public key pair
    :param d: The private key
    :return: The plain text
    """

    m = None

    # your code starts here

    return (c^d)mod(n);
    # your code ends here

    return m

def usage():
    print """Usage:
        python rsa.py student_id (i.e., qchenxiong3)"""
    sys.exit(1)

def to_text(n):
    s = ""
    while n != 0:
        curr = n % 256
        n /= 256
        s = chr(curr) + s
    return s

def main():
    if len(sys.argv) != 2:
        usage()

    n = 0
    e = 0

    all_keys = None
    with open("keys.json", 'r') as f:
        all_keys = json.load(f)

    name = hashlib.sha224(sys.argv[1].strip()).hexdigest()
    if name not in all_keys:
        print sys.argv[1], "not in keylist"
        usage()

    k = all_keys[name]
    m = rsa(int(k['c'],16), int(k['e'],16), int(k['d'],16), int(k['N'],16))
    print 'Message after decryption:'
    print to_text(m)

if __name__ == "__main__":
    main()
