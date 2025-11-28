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

I_HALT = 2
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

import sys
import json

class Simulator:
    def __init__(self):
        # 程序计数器
        self.PC = 0
        
        # 寄存器初始化 - 15个寄存器
        self.registers = [0] * 15
        
        # 条件码寄存器
        self.CC = {'ZF': 0, 'SF': 0, 'OF': 0}
        
        # 内存（稀疏表示）
        self.memory = {}
        
        # 处理器状态
        self.stat = 'AOK'  # AOK, HLT, ADR, INS
    
    def read_memory(self, addr):
        """读取一个字节的内存"""
        return self.memory.get(addr, 0)
    
    def write_memory(self, addr, value):
        """向内存写入一个字节"""
        if 0 <= value <= 0xFF:
            self.memory[addr] = value
        else:
            raise ValueError(f"Invalid byte value: {value}")
    
    def write_memory_qword(self, addr, value):
        """向内存写入8字节(小端序)"""
        # 处理有符号数
        if value < 0:
            value = (1 << 64) + value
    
        for i in range(8):
            byte_val = (value >> (i * 8)) & 0xFF
            self.write_memory(addr + i, byte_val)
    
    def read_memory_qword(self, addr):
        """读取8字节(小端序)"""
        value = 0
        for i in range(8):
            byte_val = self.read_memory(addr + i)
            value |= (byte_val << (i * 8))
        # 处理有符号扩展
        if value & (1 << 63):
            value = value - (1 << 64)
        return value

    def fetch(self):
        """取指阶段 - 读取指令字节"""
        if self.stat != 'AOK':
            return None, None, None, None, None, None
        
        # 读取第一个字节
        byte1 = self.read_memory(self.PC)
        icode = (byte1 >> 4) & 0xF  # 高4位
        ifun = byte1 & 0xF          # 低4位
        
        # 默认值
        rA = 0xF  # 无寄存器
        rB = 0xF  # 无寄存器
        valC = 0
        valP = self.PC + 1  # 默认下一条指令地址
        # 根据指令类型解析更多字节
        if icode in [2, 3, 4, 5, 6, 10, 11]:  # 需要寄存器字节的指令
            if self.PC + 1 in self.memory:
                reg_byte = self.read_memory(self.PC + 1)
                rA = (reg_byte >> 4) & 0xF
                rB = reg_byte & 0xF
                valP += 1
        
        # 需要常数字的指令
        if icode in [3, 4, 5]:  # irmovq, rmmovq, mrmovq
            if self.PC + 2 in self.memory:
                valC = self.read_memory_qword(self.PC + 2)
                valP += 8
        
        return icode, ifun, rA, rB, valC, valP
        pass

    def decode(self,rA,rB):
        """译码阶段 - 读取寄存器值"""
        valA = self.registers[rA] if rA != 0xF else 0
        valB = self.registers[rB] if rB != 0xF else 0
        return valA, valB
        pass

    def execute(self,icode,ifun,valA,valB,valC):    
        valE = 0
        set_cc = False

        if icode == 2:  # rrmovq / cmovXX
            if ifun == 0:  # rrmovq - 无条件移动
                valE = valA
            else:  # cmovXX - 条件移动
                if self.check_condition(ifun):
                    valE = valA
        
        elif icode == 3:  # irmovq - 立即数移动
            valE = valC
        
        elif icode == 6:  # 算术指令
            set_cc = True
            if ifun == 0:  # addq
                valE = valA + valB
            elif ifun == 1:  # subq
                valE = valB - valA
            elif ifun == 2:  # andq
                valE = valA & valB
            elif ifun == 3:  # xorq
                valE = valA ^ valB
            # 更新条件码
            if set_cc:
                self.update_condition_codes(valE)
        
        return valE, set_cc
        pass

    def check_condition(self, ifun):
        """检查条件码是否满足条件移动"""
        ZF, SF, OF = self.CC['ZF'], self.CC['SF'], self.CC['OF']
        
        conditions = {
            1: (ZF == 1) or (SF != OF),  # le (less or equal)
            2: (SF != OF),               # l (less)
            3: (ZF == 1),                # e (equal)
            4: (ZF == 0),                # ne (not equal)
            5: (SF == OF),               # ge (greater or equal)
            6: (ZF == 0) and (SF == OF)  # g (greater)
        }
        return conditions.get(ifun, False)
    
    def update_condition_codes(self, result):
        """更新条件码 - 算术指令使用"""
        # 零标志
        self.CC['ZF'] = 1 if result == 0 else 0
        # 符号标志
        self.CC['SF'] = 1 if result < 0 else 0
        # 溢出标志 - 简化实现，实际需要更精确的溢出检测
        # 注意：这是一个简化版本，实际需要根据具体操作检测溢出
        self.CC['OF'] = 0
    
    def write_back(self, icode, rA, rB, valE, valM):
        """写回阶段 - 更新寄存器"""
        if icode == 2:  # rrmovq / cmovXX
            if rB != 0xF:
                self.registers[rB] = valE
        elif icode == 3:  # irmovq
            if rB != 0xF:
                self.registers[rB] = valE
        elif icode == 6:  # 算术指令
            if rB != 0xF:
                self.registers[rB] = valE
    
    def dump_state(self):
        """输出当前状态为JSON格式 - 这是关键方法！"""
        # 寄存器名称映射
        reg_names = ['rax', 'rcx', 'rdx', 'rbx', 'rsp', 'rbp', 
                    'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14']
        reg_state = {}
        for i, name in enumerate(reg_names):
            reg_state[name] = self.registers[i]
        
        # 内存状态 - 只输出非零的8字节对齐值
        mem_state = {}
        processed_addrs = set()
        
        # 收集所有内存地址
        all_addrs = sorted(self.memory.keys())
        for addr in all_addrs:
            base_addr = addr - (addr % 8)  # 8字节对齐
            # 避免重复处理同一个对齐地址
            if base_addr in processed_addrs:
                continue
            processed_addrs.add(base_addr)
            
            # 读取8字节值
            qword_val = self.read_memory_qword(base_addr)
            
            # 只有当值不为0时才输出
            if qword_val != 0:
                mem_state[str(base_addr)] = qword_val
        
        # 状态码映射
        stat_map = {'AOK': 1, 'HLT': 2, 'ADR': 3, 'INS': 4}
        
        # 构建输出
        output = {
            "PC": self.PC,
            "REG": reg_state,
            "MEM": mem_state,
            "CC": self.CC.copy(),
            "STAT": stat_map.get(self.stat, 0)
        }
        print(json.dumps(output, indent=4))

    def run(self):
        """主循环 - 从标准输入加载.yo文件并执"""
        # 从标准输入加载程序（老师提供的.yo文件）
        self.load_program_from_stdin()
    
        instruction_count = 0
        max_instructions = 1000  # 防止无限循环
    
        while self.stat == 'AOK' and instruction_count < max_instructions:
            instruction_count += 1
        
            # 取指
            icode, ifun, rA, rB, valC, valP = self.fetch()
        
            if icode is None:
                self.stat = 'INS'  # 无效指令
                break
        
            # 特殊指令处理
            if icode == 0:  # halt
                self.stat = 'HLT'
                self.dump_state()
                break
            elif icode == 1:  # nop
                self.PC = valP
                self.dump_state()
                continue
            
            # 译码
            valA, valB = self.decode(rA, rB)
        
            # 执行
            valE, set_cc = self.execute(icode, ifun, valA, valB, valC)
        
            # 访存阶段（主要由成员B负责，但需要基础实现）
            valM = self.memory_access(icode, valE, valA, valP)
        
            # 写回
            self.write_back(icode, rA, rB, valE, valM)
        
            # 更新PC
            self.update_pc(icode, valC, valM, valP)
        
            # 输出状态（符合项目要求的JSON格式）
            self.dump_state()
        if instruction_count >= max_instructions:
            print("警告：达到最大指令数限制", file=sys.stderr)
        pass

    def load_program_from_stdin(self):
        """从标准输入加载.yo格式的程序"""
        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if ':' in line:
                parts = line.split(':', 1)
                addr_str = parts[0].strip()
                data_part = parts[1].strip()
            
                data_str = data_part.split('#')[0].strip('|')[0].strip()  # 去掉注释
                
                try:
                    # 转换地址
                    addr = int(addr_str, 16)

                    # 将十六进制字符串转换为字节列表
                    bytes_list = []
                    clean_data_str = data_str.replace(' ', '')  # 移除空格
                    for i in range(0, len(data_str), 2):
                        byte_str = data_str[i:i+2]
                        if len(byte_str) == 2:  # 确保是完整的字节
                            try:
                                byte_val = int(byte_str, 16)
                                bytes_list.append(byte_val)
                            except ValueError:
                                # 跳过无效的字节
                                continue
                    for offset, byte_val in enumerate(bytes_list):
                        self.write_memory(addr + offset, byte_val)
                except ValueError as e:
                    # 跳过无法解析的行
                    print(f"跳过无法解析的行: {line}", file=sys.stderr)
                    continue
    
    def memory_access(self, icode, valE, valA, valP):
        """访存阶段 - 基础实现"""
        valM = 0
    
        # 基础的内存访问实现
        if icode == 4:  # rmmovq
             self.write_memory_qword(valE, valA)
        elif icode == 5:  # mrmovq
            valM = self.read_memory_qword(valE)
        elif icode == 8:  # call
            self.registers[4] -= 8  # rsp减8
            self.write_memory_qword(self.registers[4], valP)
        elif icode == 9:  # ret
            valM = self.read_memory_qword(self.registers[4])
            self.registers[4] += 8
        elif icode == 10:  # pushq
            self.registers[4] -= 8
            self.write_memory_qword(self.registers[4], valA)
        elif icode == 11:  # popq
            valM = self.read_memory_qword(self.registers[4])
            self.registers[4] += 8
    
        return valM
    
    def update_pc(self, icode, valC, valM, valP):
        """更新程序计数器"""
        if icode == 7:  # jmp / jXX
            # 这里需要检查条件，简化实现
            self.PC = valC
        elif icode == 8:  # call
            self.PC = valC
        elif icode == 9:  # ret
            self.PC = valM
        else:
            self.PC = valP

if __name__ == "__main__":
    # 创建模拟器实例并运行
    simulator = Simulator()
    simulator.run()
    pass
