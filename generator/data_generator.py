from random import randint
import random

def calc_check_digit(upc):
    check_sum = sum(int(x) * (3 if i % 2 == 0 else 1) for i, x in enumerate(upc))
    check_digit = (10 - (check_sum % 10)) % 10
    return str(check_digit)

def generate_upca():
    upc_prefix = "0"
    upc_middle = str(randint(10000, 99999))
    upc_check_digit = calc_check_digit(upc_prefix + upc_middle)
    upc = upc_prefix + upc_middle + upc_check_digit
    return upc

def generate_code39():
    # Define the characters allowed in a Code 39 barcode
    code39_characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%'
    
    code39_data = ''.join(random.choices(code39_characters, k=10))  # Generate random data of length 10
    
    # Calculate the checksum character for Code 39
    checksum = sum([code39_characters.index(char) for char in code39_data]) % 43
    code39_encoded = code39_data + code39_characters[checksum]
    
    return code39_encoded

def generate_code93(length=None):
    if length is None:
        length = randint(1, 50)
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%'
    input_data = [random.choice(characters) for _ in range(length)]
    return ''.join(input_data)

def random_ascii(sz=256):
    return ''.join([chr(randint(0, 127)) for i in range(randint(1, sz))])

gens = {
    'ean13': lambda: str(randint(10**11, 10**12-1)),
    'upca': generate_upca,
    'plessey': lambda: hex(randint(0, 16**20)).upper()[2:],
    'code39': generate_code39,
    'code93': generate_code93,
    'datamatrix': random_ascii,
    'qrcode': random_ascii,
    'azteccode': random_ascii,
    'aztecrune': lambda: str(randint(0, 255)),
    'microqrcode': lambda: random_ascii(35),
}

dims = {
    'ean13': '1d',
    'upca': '1d',
    'plessey': '1d',
    'code39': '1d',
    'code93': '1d',
    'datamatrix': '2d',
    'qrcode': '2d',
    'azteccode': '2d',
    'aztecrune': '2d',
    'microqrcode': '2d',
}