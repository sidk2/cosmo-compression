import numpy as np
import os
import fpzip
import h5py
import bitshuffle.h5
import lzma

class Compressor():
    def __init__(self, input):
        self.input = input
    def get_bpp(self):
        return self.bpp
    def lossless_verification(self):
        b = self.decompress()
        return np.allclose(b, self.input)
    
class GZIP_SHUFFLE(Compressor):
    def __init__(self, input):
        super().__init__(input)
        self.output_file = "bytes.h5"
    def compress(self):
        shape = self.input.shape
        h5f = h5py.File(self.output_file, 'w')
        h5f.create_dataset('dataset_1', data=self.input, compression="gzip", shuffle=True, compression_opts=9)
        h5f.close()
        self.bpp = os.path.getsize(self.output_file) * 8 / np.prod(shape)
      
    def decompress(self):
        h5f = h5py.File(self.output_file,'r')
        b = h5f['dataset_1'][:]
        h5f.close()
        return b

class FPZIP(Compressor):
    def compress(self):
        shape = self.input.shape
        compressed_bytes = fpzip.compress(self.input, precision=0, order='C')
        # with open(self.output_file, "wb") as binary_file:
        #     binary_file.write(compressed_bytes)
        self.compressed_bytes =  compressed_bytes
        self.bpp = len(self.compressed_bytes) * 8 / np.prod(shape)
        # self.bpp = os.path.getsize(self.output_file) * 8 / np.prod(shape)   
        
    def decompress(self):
        b = fpzip.decompress(self.compressed_bytes, order='C')
        # b = fpzip.decompress(open(self.output_file, "rb").read(), order='C')
        return b

class LZMA(Compressor):
    def compress(self):
        shape = self.input.shape
        compressed_bytes = lzma.compress(self.input)
        self.compressed_bytes =  compressed_bytes
        self.bpp = len(self.compressed_bytes) * 8 / np.prod(shape)  
        
    def decompress(self):
        b = lzma.decompress(self.compressed_bytes)
        b = np.frombuffer(b, dtype=np.float32)
        b = b.reshape(self.input.shape[1:])
        return b