import sys
import json

MEM_SIZE = 1 << 20

STAT_AOK = 1
STAT_HLT = 2
STAT_ADR = 3
STAT_INS = 4

CC_NAMES = ["ZF", "SF", "OF"]

REG_NAMES = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
             "r8", "r9", "r10", "r11", "r12", "r13", "r14"]

I_HALT = 0
I_NOP = 1
I_RRMOVQ = 2
I_IRMOVQ = 3
I_RMMOVQ = 4
I_MRMOVQ = 5
I_OPQ = 6
I_JXX = 7
I_CALL = 8
I_RET = 9
I_PUSHQ = 10
I_POPQ = 11

class Simulator:
    def __init__(self):
        self.PC = 0

    def fetch(self):
        pass

    def decode(self):
        pass

    def execute(self):
        pass

    def run(self):
        pass

if __name__ == "__main__":
    pass
