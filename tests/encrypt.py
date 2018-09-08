
import os
from Crypto.Cipher import AES

#KEY = '1234567890123456' 
#IV  = '1234567890123456' 

KEY = '140B41B22A29DEB4061BDA6Fb6747E14'
IV  = '0D79A874BE09C72F'

def encrypt(data):
    cipher = AES.new(KEY, AES.MODE_CFB , IV )
    return cipher.encrypt(data)

data = "!!!BlarBlarBlar!!!"
print( encrypt( data ) )

file = open("encrypted.txt", "wb")
file.write( encrypt( data ) )
file.close()


