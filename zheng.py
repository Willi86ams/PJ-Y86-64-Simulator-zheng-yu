import sys
import json

MEM_SIZE = 1 << 20
STAT_AOK = 1
STAT_HLT = 2
STAT_ADR = 3
STAT_INS = 4

I_HALT   = 0x0
I_NOP    = 0x1
I_RRMOVQ = 0x2
I_IRMOVQ = 0x3
I_RMMOVQ = 0x4
I_MRMOVQ = 0x5
I_OPQ    = 0x6
I_JXX    = 0x7
I_CALL   = 0x8
I_RET    = 0x9
I_PUSHQ  = 0xA
I_POPQ   = 0xB

REG_RAX = 0x0
REG_RCX = 0x1
REG_RDX = 0x2
REG_RBX = 0x3
REG_RSP = 0x4
REG_RBP = 0x5
REG_RSI = 0x6
REG_RDI = 0x7
REG_R8  = 0x8
REG_R9  = 0x9
REG_R10 = 0xA
REG_R11 = 0xB
REG_R12 = 0xC
REG_R13 = 0xD
REG_R14 = 0xE
REG_NONE = 0xF

REG_NAMES = [
    "rax", "rcx", "rdx", "rbx",
    "rsp", "rbp", "rsi", "rdi",
    "r8",  "r9",  "r10", "r11",
    "r12", "r13", "r14"
]

MASK64 = (1 << 64) - 1

def mask64(x):
    return x & MASK64

def to_signed64(x):
    x &= MASK64
    if x & (1 << 63):
        return x - (1 << 64)
    return x

class Simulator:
    def __init__(self):
        self.PC = 0
        self.mem = bytearray(MEM_SIZE)
        self.regs = [0] * len(REG_NAMES)
        self.cc = {"ZF": 1, "SF": 0, "OF": 0}
        self.STAT = STAT_AOK
        self.logs = []

    def load_from_stdin(self):
        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            addr_part, rest = line.split(":", 1)
            addr_part = addr_part.strip()
            try:
                addr = int(addr_part, 16)
            except:
                continue
            code_part = rest.split("|")[0].strip()
            if not code_part:
                continue
            hex_str = code_part.replace(" ", "")
            for i in range(0, len(hex_str), 2):
                byte_str = hex_str[i:i+2]
                if not byte_str:
                    continue
                b = int(byte_str, 16)
                if addr < 0 or addr >= MEM_SIZE:
                    self.STAT = STAT_ADR
                    return
                self.mem[addr] = b
                addr += 1

    def get_reg(self, r):
        if r == REG_NONE:
            return 0
        if 0 <= r < len(self.regs):
            return self.regs[r]
        self.STAT = STAT_INS
        return 0

    def set_reg(self, r, val):
        if r == REG_NONE:
            return
        if 0 <= r < len(self.regs):
            self.regs[r] = mask64(val)
        else:
            self.STAT = STAT_INS

    def check_mem_range(self, addr, size):
        if addr < 0 or addr + size > MEM_SIZE:
            self.STAT = STAT_ADR
            return False
        return True

    def mem_read8(self, addr):
        if not self.check_mem_range(addr, 8):
            return 0
        chunk = self.mem[addr:addr+8]
        return int.from_bytes(chunk, "little", signed=True)

    def mem_write8(self, addr, val):
        if not self.check_mem_range(addr, 8):
            return
        u = mask64(val)
        self.mem[addr:addr+8] = u.to_bytes(8, "little")

    def cond_holds(self, ifun):
        ZF = self.cc["ZF"]
        SF = self.cc["SF"]
        OF = self.cc["OF"]
        if ifun == 0:
            return True
        elif ifun == 1:
            return (SF ^ OF) or ZF
        elif ifun == 2:
            return SF ^ OF
        elif ifun == 3:
            return ZF == 1
        elif ifun == 4:
            return ZF == 0
        elif ifun == 5:
            return (SF ^ OF) == 0
        elif ifun == 6:
            return (SF ^ OF) == 0 and ZF == 0
        else:
            self.STAT = STAT_INS
            return False

    def set_cc_for_op(self, op, a, b, r):
        a_s = to_signed64(a)
        b_s = to_signed64(b)
        r_s = to_signed64(r)
        self.cc["ZF"] = 1 if r_s == 0 else 0
        self.cc["SF"] = 1 if r_s < 0 else 0
        if op == 0:
            if (a_s > 0 and b_s > 0 and r_s < 0) or (a_s < 0 and b_s < 0 and r_s >= 0):
                self.cc["OF"] = 1
            else:
                self.cc["OF"] = 0
        elif op == 1:
            neg_a = -a_s
            if (b_s > 0 and neg_a > 0 and r_s < 0) or (b_s < 0 and neg_a < 0 and r_s >= 0):
                self.cc["OF"] = 1
            else:
                self.cc["OF"] = 0
        else:
            self.cc["OF"] = 0

    def step(self):
        pc = self.PC
        if pc < 0 or pc >= MEM_SIZE:
            self.STAT = STAT_ADR
            return
        byte0 = self.mem[pc]
        icode = (byte0 >> 4) & 0xF
        ifun = byte0 & 0xF
        rA = REG_NONE
        rB = REG_NONE
        valC = None
        valP = pc + 1
        need_reg = False
        need_valC = False
        valC_after_reg = True

        if icode in (I_RRMOVQ, I_OPQ, I_PUSHQ, I_POPQ):
            need_reg = True
            valP = pc + 2
        elif icode in (I_IRMOVQ, I_RMMOVQ, I_MRMOVQ):
            need_reg = True
            need_valC = True
            valC_after_reg = True
            valP = pc + 10
        elif icode in (I_JXX, I_CALL):
            need_valC = True
            valC_after_reg = False
            valP = pc + 9
        elif icode in (I_HALT, I_NOP, I_RET):
            valP = pc + 1
        else:
            self.STAT = STAT_INS
            return

        if icode in (I_HALT, I_NOP, I_IRMOVQ, I_RMMOVQ, I_MRMOVQ, I_CALL, I_RET, I_PUSHQ, I_POPQ):
            if ifun != 0:
                self.STAT = STAT_INS
                return
        elif icode == I_OPQ:
            if ifun not in (0, 1, 2, 3):
                self.STAT = STAT_INS
                return
        elif icode in (I_RRMOVQ, I_JXX):
            if ifun not in range(0, 7):
                self.STAT = STAT_INS
                return

        if need_reg:
            if pc + 1 >= MEM_SIZE:
                self.STAT = STAT_ADR
                return
            regbyte = self.mem[pc + 1]
            rA = (regbyte >> 4) & 0xF
            rB = regbyte & 0xF

        if need_valC:
            if valC_after_reg:
                addr_c = pc + 2
            else:
                addr_c = pc + 1
            if not self.check_mem_range(addr_c, 8):
                return
            valC = int.from_bytes(self.mem[addr_c:addr_c+8], "little", signed=True)

        if icode == I_HALT:
            self.STAT = STAT_HLT
            return

        elif icode == I_NOP:
            self.PC = valP

        elif icode == I_RRMOVQ:
            cond = self.cond_holds(ifun)
            if cond:
                valA = self.get_reg(rA)
                self.set_reg(rB, valA)
            self.PC = valP

        elif icode == I_IRMOVQ:
            self.set_reg(rB, valC)
            self.PC = valP

        elif icode == I_RMMOVQ:
            valA = self.get_reg(rA)
            valB = self.get_reg(rB)
            addr = to_signed64(valB) + valC
            if not self.check_mem_range(addr, 8):
                return
            self.mem_write8(addr, valA)
            self.PC = valP

        elif icode == I_MRMOVQ:
            valB = self.get_reg(rB)
            addr = to_signed64(valB) + valC
            if not self.check_mem_range(addr, 8):
                return
            valM = self.mem_read8(addr)
            self.set_reg(rA, valM)
            self.PC = valP

        elif icode == I_OPQ:
            valA = self.get_reg(rA)
            valB = self.get_reg(rB)
            if ifun == 0:
                res = to_signed64(valB) + to_signed64(valA)
            elif ifun == 1:
                res = to_signed64(valB) - to_signed64(valA)
            elif ifun == 2:
                res = to_signed64(valB) & to_signed64(valA)
            elif ifun == 3:
                res = to_signed64(valB) ^ to_signed64(valA)
            else:
                self.STAT = STAT_INS
                return
            ures = mask64(res)
            self.set_reg(rB, ures)
            self.set_cc_for_op(ifun, valA, valB, ures)
            self.PC = valP

        elif icode == I_JXX:
            cond = self.cond_holds(ifun)
            if cond:
                self.PC = valC
            else:
                self.PC = valP

        elif icode == I_CALL:
            rsp = self.get_reg(REG_RSP)
            valE = to_signed64(rsp) - 8
            if not self.check_mem_range(valE, 8):
                return
            self.mem_write8(valE, valP)
            self.set_reg(REG_RSP, valE)
            self.PC = valC

        elif icode == I_RET:
            rsp = self.get_reg(REG_RSP)
            valA = to_signed64(rsp)
            if not self.check_mem_range(valA, 8):
                return
            valM = self.mem_read8(valA)
            valE = valA + 8
            self.set_reg(REG_RSP, valE)
            self.PC = valM

        elif icode == I_PUSHQ:
            valA = self.get_reg(rA)
            rsp = self.get_reg(REG_RSP)
            valE = to_signed64(rsp) - 8
            if not self.check_mem_range(valE, 8):
                return
            self.mem_write8(valE, valA)
            self.set_reg(REG_RSP, valE)
            self.PC = valP

        elif icode == I_POPQ:
            rsp = self.get_reg(REG_RSP)
            valA = to_signed64(rsp)
            if not self.check_mem_range(valA, 8):
                return
            valM = self.mem_read8(valA)
            valE = valA + 8
            self.set_reg(REG_RSP, valE)
            self.set_reg(rA, valM)
            self.PC = valP

        else:
            self.STAT = STAT_INS

    def snapshot(self):
        reg_dict = {name: to_signed64(self.regs[i]) for i, name in enumerate(REG_NAMES)}
        cc_dict = {k: int(v) for k, v in self.cc.items()}
        mem_dict = {}
        for addr in range(0, MEM_SIZE, 8):
            chunk = self.mem[addr:addr+8]
            if len(chunk) < 8:
                break
            val = int.from_bytes(chunk, "little", signed=True)
            if val != 0:
                mem_dict[str(addr)] = val
        return {
            "CC": cc_dict,
            "MEM": mem_dict,
            "PC": int(self.PC),
            "REG": reg_dict,
            "STAT": int(self.STAT)
        }

    def run(self):
        while self.STAT == STAT_AOK:
            self.step()
            self.logs.append(self.snapshot())
            if self.STAT != STAT_AOK:
                break

def main():
    sim = Simulator()
    sim.load_from_stdin()
    if sim.STAT == STAT_ADR:
        sim.logs.append(sim.snapshot())
    else:
        sim.run()
    print(json.dumps(sim.logs, indent=4))

if __name__ == "__main__":
    main()
