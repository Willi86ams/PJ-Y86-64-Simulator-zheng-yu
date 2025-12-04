import sys
from copy import deepcopy

from zheng import (
    Simulator,
    MEM_SIZE,
    REG_NAMES,
    STAT_AOK,
    STAT_HLT,
    STAT_ADR,
    STAT_INS,
    I_HALT, I_NOP, I_RRMOVQ, I_IRMOVQ, I_RMMOVQ, I_MRMOVQ,
    I_OPQ, I_JXX, I_CALL, I_RET, I_PUSHQ, I_POPQ,
)

from PIL import Image, ImageDraw, ImageFont

STAT_NAME = {
    STAT_AOK: "AOK",
    STAT_HLT: "HLT",
    STAT_ADR: "ADR",
    STAT_INS: "INS",
}

ERROR_DESC = {
    STAT_AOK: "OK (no error)",
    STAT_HLT: "Program halted normally",
    STAT_ADR: "Address error (invalid PC or memory access)",
    STAT_INS: "Invalid instruction encoding",
}

# ============== 读 .yo 到模拟器 ==============

def load_yo_into_sim(sim: Simulator, yo_path: str):
    with open(yo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            addr_part, rest = line.split(":", 1)
            addr_part = addr_part.strip()
            try:
                addr = int(addr_part, 16)
            except ValueError:
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
                    sim.STAT = STAT_ADR
                    return
                sim.mem[addr] = b
                addr += 1


def estimate_total_steps(yo_path: str, limit: int = 1000):
    """
    预跑一次，返回:
      logical_steps: 在 AOK 状态下真正执行了多少 step()
      total_display_steps: 动画里要显示多少步（如果最后是错误/HLT，会多一帧）
      ended_stat: 预跑结束时的 STAT
    """
    sim = Simulator()
    load_yo_into_sim(sim, yo_path)
    steps = 0
    while sim.STAT == STAT_AOK and steps < limit:
        sim.step()
        steps += 1
    ended_stat = sim.STAT
    total_display_steps = steps if ended_stat == STAT_AOK else steps + 1
    return steps, total_display_steps, ended_stat

# ============== 译码 ==============

def decode_icode_name(icode, ifun):
    if icode == I_HALT:
        return "halt"
    if icode == I_NOP:
        return "nop"
    if icode == I_RRMOVQ:
        return f"rrmovq/cmov({ifun})"
    if icode == I_IRMOVQ:
        return "irmovq"
    if icode == I_RMMOVQ:
        return "rmmovq"
    if icode == I_MRMOVQ:
        return "mrmovq"
    if icode == I_OPQ:
        return {0: "addq", 1: "subq", 2: "andq", 3: "xorq"}.get(ifun, f"opq({ifun})")
    if icode == I_JXX:
        return {0: "jmp", 1: "jle", 2: "jl", 3: "je", 4: "jne", 5: "jge", 6: "jg"}.get(ifun, f"jXX({ifun})")
    if icode == I_CALL:
        return "call"
    if icode == I_RET:
        return "ret"
    if icode == I_PUSHQ:
        return "pushq"
    if icode == I_POPQ:
        return "popq"
    return f"unknown({icode})"


def fetch_decode_once(sim: Simulator):
    pc = sim.PC
    if pc < 0 or pc >= MEM_SIZE:
        return {"error": "PC out of range", "pc": pc}

    byte0 = sim.mem[pc]
    icode = (byte0 >> 4) & 0xF
    ifun = byte0 & 0xF

    rA = 0xF
    rB = 0xF
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
        return {"pc": pc, "icode": icode, "ifun": ifun, "error": "invalid icode"}

    if need_reg:
        if pc + 1 >= MEM_SIZE:
            return {"pc": pc, "icode": icode, "ifun": ifun, "error": "reg byte out of range"}
        regbyte = sim.mem[pc + 1]
        rA = (regbyte >> 4) & 0xF
        rB = regbyte & 0xF

    if need_valC:
        addr_c = pc + 2 if valC_after_reg else pc + 1
        if addr_c < 0 or addr_c + 8 > MEM_SIZE:
            return {"pc": pc, "icode": icode, "ifun": ifun, "rA": rA, "rB": rB, "error": "valC out of range"}
        valC = int.from_bytes(sim.mem[addr_c:addr_c+8], "little", signed=True)

    raw_len = max(1, valP - pc)
    raw_bytes = bytes(sim.mem[pc:pc+raw_len])

    return {
        "pc": pc,
        "raw_bytes": raw_bytes,
        "icode": icode,
        "ifun": ifun,
        "rA": rA,
        "rB": rB,
        "valC": valC,
        "valP": valP,
        "instr_name": decode_icode_name(icode, ifun),
    }

# ============== 帧描述结构 ==============

def make_line(text, spans=None):
    """
    spans:
      - 局部高亮: (start, end, (r,g,b))
      - 整行着色: ("FULL", (r,g,b))
    """
    return {
        "text": text,
        "spans": spans or [],
    }


def build_frame(sim: Simulator, fd_info, step_idx, total_steps, prev_pc, prev_regs):
    frame = {
        "header": [],
        "regs": [],
        "mem": [],
        "progress": [],
    }

    # ---------- 如果是 fetch/decode 级别的错误：单独走一条路径 ----------
    if "error" in fd_info:
        pc = fd_info.get("pc", sim.PC)
        err = fd_info.get("error", "")

        title = f"*** ERROR FRAME (step {step_idx}/{total_steps}) ***"
        frame["header"].append(make_line(title, spans=[("FULL", (200, 0, 0))]))

        frame["header"].append(make_line(f"PC   : {pc} (0x{pc:03x})"))

        if err:
            frame["header"].append(make_line(f"Detail: {err}"))

        stat = sim.STAT
        cc = sim.cc
        stat_str = STAT_NAME.get(stat, str(stat))
        desc = ERROR_DESC.get(stat, "")
        if desc:
            stat_line = f"STAT : {stat_str:<3} ({desc})   CC : ZF={cc['ZF']} SF={cc['SF']} OF={cc['OF']}"
        else:
            stat_line = f"STAT : {stat_str:<3}   CC : ZF={cc['ZF']} SF={cc['SF']} OF={cc['OF']}"
        frame["header"].append(make_line(stat_line, spans=[("FULL", (200, 0, 0))]))

        frame["header"].append(make_line("Execution stopped; cannot fetch/decode further."))

        # Registers at error
        frame["regs"].append(
            make_line("Registers at error:",
                      spans=[("FULL", (0, 128, 0))])
        )

        regs = sim.regs
        if prev_regs is None:
            prev_regs = [0] * len(REG_NAMES)

        left_names = REG_NAMES[:8]
        right_names = REG_NAMES[8:]
        max_rows = max(len(left_names), len(right_names))

        for i in range(max_rows):
            text = ""
            spans = []

            if i < len(left_names):
                name = left_names[i]
                idx = i
                val = regs[idx]
                prev_val = prev_regs[idx]
                seg = f"{name:>3} = {val:<4d}"
                text += f"{seg:<34}"
                if val != prev_val:
                    spans.append((0, len(seg), (200, 160, 0)))
            else:
                text += " " * 34

            if i < len(right_names):
                name = right_names[i]
                idx = 8 + i
                val = regs[idx]
                prev_val = prev_regs[idx]
                seg = f"{name:>3} = {val:<4d}"
                start = len(text)
                text += f"{seg:<34}"
                if val != prev_val:
                    spans.append((start, start + len(seg), (200, 160, 0)))
            else:
                text += " " * 34

            frame["regs"].append(make_line(text, spans=spans))

        # Memory snapshot
        frame["mem"].append(
            make_line("Non-zero memory at error (first few 8-byte blocks)",
                      spans=[("FULL", (128, 0, 128))])
        )
        state = sim.snapshot()
        mem = state["MEM"]
        if not mem:
            frame["mem"].append(make_line("  (none)"))
        else:
            cnt = 0
            for addr_str in sorted(mem.keys(), key=lambda x: int(x)):
                val = mem[addr_str]
                frame["mem"].append(make_line(f"  M[{addr_str:>5}] = {val}"))
                cnt += 1
                if cnt >= 6:
                    break

        # Progress bar
        ratio = step_idx / total_steps
        bar_len = 20
        filled = max(1, int(round(ratio * bar_len)))
        bar = "#" * filled + "-" * (bar_len - filled)
        prog_text = f"Progress: [{bar}] {step_idx}/{total_steps}"
        bar_start = prog_text.index("[")
        bar_end = prog_text.index("]") + 1
        bar_color = (200, 0, 0)
        frame["progress"].append(
            make_line(prog_text, spans=[(bar_start, bar_end, bar_color)])
        )

        return frame

    # ---------- 正常 Fetch/Decode 帧 ----------

    title = f"Fetch & Decode (before exec) - Step {step_idx}/{total_steps}"
    frame["header"].append(
        make_line(title, spans=[("FULL", (0, 128, 255))])
    )

    pc = fd_info["pc"]
    raw = fd_info["raw_bytes"]
    icode = fd_info["icode"]
    ifun = fd_info["ifun"]
    rA = fd_info["rA"]
    rB = fd_info["rB"]
    valC = fd_info["valC"]
    valP = fd_info["valP"]
    instr_name = fd_info["instr_name"]

    # PC 行
    if prev_pc is not None and pc != prev_pc and step_idx > 1:
        pc_line = f"PC   : {pc:<4d} (0x{pc:03x})   instr : {instr_name}   (from {prev_pc})"
        frame["header"].append(
            make_line(pc_line, spans=[("FULL", (80, 40, 200))])
        )
    else:
        pc_line = f"PC   : {pc:<4d} (0x{pc:03x})   instr : {instr_name}"
        frame["header"].append(make_line(pc_line))

    # STAT + CC 行（这里也加上错误描述，但只有非 AOK 才显示）
    stat = sim.STAT
    cc = sim.cc
    stat_str = STAT_NAME.get(stat, str(stat))
    desc = ERROR_DESC.get(stat, "")
    if stat == STAT_AOK:
        color = (0, 150, 0)
        stat_line = f"STAT : {stat_str:<3}        CC : ZF={cc['ZF']} SF={cc['SF']} OF={cc['OF']}"
    else:
        color = (200, 0, 0)
        if desc:
            stat_line = f"STAT : {stat_str:<3} ({desc})   CC : ZF={cc['ZF']} SF={cc['SF']} OF={cc['OF']}"
        else:
            stat_line = f"STAT : {stat_str:<3}   CC : ZF={cc['ZF']} SF={cc['SF']} OF={cc['OF']}"
    frame["header"].append(make_line(stat_line, spans=[("FULL", color)]))

    # icode / ifun
    frame["header"].append(make_line(f"icode: {icode}    ifun: {ifun}"))

    # bytes
    hex_bytes = " ".join(f"{b:02x}" for b in raw)
    frame["header"].append(make_line(f"bytes[{pc:03x}..]: {hex_bytes}"))

    # rA / rB
    if rA != 0xF or rB != 0xF:
        nameA = REG_NAMES[rA] if rA != 0xF and rA < len(REG_NAMES) else "NONE"
        nameB = REG_NAMES[rB] if rB != 0xF and rB < len(REG_NAMES) else "NONE"
        ra_rb_line = f"rA   : {nameA} (#{rA}),   rB : {nameB} (#{rB})"
    else:
        ra_rb_line = "No register specifier in this instruction."
    frame["header"].append(make_line(ra_rb_line))

    # valC / valP
    frame["header"].append(make_line(f"valC : {valC}    next PC (valP) = {valP}"))

    # --- Registers ---
    frame["regs"].append(
        make_line("Registers (before executing this instruction)",
                  spans=[("FULL", (0, 128, 0))])
    )

    regs = sim.regs
    if prev_regs is None:
        prev_regs = [0] * len(REG_NAMES)

    left_names = REG_NAMES[:8]
    right_names = REG_NAMES[8:]
    max_rows = max(len(left_names), len(right_names))

    for i in range(max_rows):
        text = ""
        spans = []

        if i < len(left_names):
            name = left_names[i]
            idx = i
            val = regs[idx]
            prev_val = prev_regs[idx]
            seg = f"{name:>3} = {val:<4d}"
            text += f"{seg:<34}"
            if val != prev_val:
                spans.append((0, len(seg), (200, 160, 0)))
        else:
            text += " " * 34

        if i < len(right_names):
            name = right_names[i]
            idx = 8 + i
            val = regs[idx]
            prev_val = prev_regs[idx]
            seg = f"{name:>3} = {val:<4d}"
            start = len(text)
            text += f"{seg:<34}"
            if val != prev_val:
                spans.append((start, start + len(seg), (200, 160, 0)))
        else:
            text += " " * 34

        frame["regs"].append(make_line(text, spans=spans))

    # --- Memory ---
    frame["mem"].append(
        make_line("Non-zero memory (first few 8-byte blocks)",
                  spans=[("FULL", (128, 0, 128))])
    )
    state = sim.snapshot()
    mem = state["MEM"]
    if not mem:
        frame["mem"].append(make_line("  (none)"))
    else:
        cnt = 0
        for addr_str in sorted(mem.keys(), key=lambda x: int(x)):
            val = mem[addr_str]
            frame["mem"].append(make_line(f"  M[{addr_str:>5}] = {val}"))
            cnt += 1
            if cnt >= 6:
                break

    # --- Progress ---
    ratio = step_idx / total_steps
    bar_len = 20
    filled = max(1, int(round(ratio * bar_len)))
    bar = "#" * filled + "-" * (bar_len - filled)
    prog_text = f"Progress: [{bar}] {step_idx}/{total_steps}"
    bar_start = prog_text.index("[")
    bar_end = prog_text.index("]") + 1
    bar_color = (0, 180, 0) if step_idx % 2 == 0 else (0, 130, 0)
    frame["progress"].append(
        make_line(prog_text, spans=[(bar_start, bar_end, bar_color)])
    )

    return frame


def build_all_frames(yo_path: str):
    logical_steps, total_display_steps, ended_stat = estimate_total_steps(yo_path)

    sim = Simulator()
    load_yo_into_sim(sim, yo_path)
    if sim.STAT != STAT_AOK:
        print("加载 .yo 出错，STAT =", sim.STAT)
        return [], total_display_steps, sim.STAT

    frames = []
    prev_pc = None
    prev_regs = None
    step_idx = 0

    # 先画所有 AOK 步骤
    while sim.STAT == STAT_AOK and step_idx < logical_steps:
        step_idx += 1
        fd_info = fetch_decode_once(sim)
        frame = build_frame(sim, fd_info, step_idx, total_display_steps, prev_pc, prev_regs)
        frames.append(frame)

        prev_pc = sim.PC
        prev_regs = deepcopy(sim.regs)
        sim.step()

    # 最后一帧（HLT / ADR / INS），再画一帧
    if sim.STAT != STAT_AOK:
        step_idx += 1
        fd_info = fetch_decode_once(sim)
        frame = build_frame(sim, fd_info, step_idx, total_display_steps, prev_pc, prev_regs)
        frames.append(frame)

    return frames, total_display_steps, ended_stat

# ============== 字体处理 ==============

def is_monospace(font: ImageFont.FreeTypeFont) -> bool:
    try:
        box_i = font.getbbox("iiii")
        box_M = font.getbbox("MMMM")
        w_i = box_i[2] - box_i[0]
        w_M = box_M[2] - box_M[0]
        return abs(w_i - w_M) < 2
    except Exception:
        return False


def pick_monospace_font(size=16):
    candidates = [
        "Consolas.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/Consola.ttf",
        "DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "Menlo.ttc",
        "Courier New.ttf",
        "Courier_New.ttf",
    ]
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size=size)
            if is_monospace(font):
                print(f"[info] 使用字体: {path}")
                return font
        except Exception:
            continue
    print("[warn] 未找到理想等宽字体，使用默认字体")
    return ImageFont.load_default()

# ============== 渲染到图片 ==============

def render_frame_to_image(frame, inner_width, font, bgcolor=(255, 255, 255)):
    visual_lines = []

    top_text = "┌" + "─" * inner_width + "┐"
    visual_lines.append({"text": top_text, "spans": []})

    sections = ["header", "regs", "mem", "progress"]
    for sec_i, sec in enumerate(sections):
        for line_spec in frame[sec]:
            content = line_spec["text"]
            content_padded = content.ljust(inner_width)
            full_text = "│" + content_padded + "│"
            spans_full = []
            for span in line_spec["spans"]:
                if isinstance(span[0], str) and span[0] == "FULL":
                    spans_full.append(span)
                else:
                    start, end, color = span
                    spans_full.append((start + 1, end + 1, color))
            visual_lines.append({"text": full_text, "spans": spans_full})
        if sec_i != len(sections) - 1:
            sep_text = "├" + "─" * inner_width + "┤"
            visual_lines.append({"text": sep_text, "spans": []})

    bottom_text = "└" + "─" * inner_width + "┘"
    visual_lines.append({"text": bottom_text, "spans": []})

    max_len = max(len(v["text"]) for v in visual_lines)
    for v in visual_lines:
        v["text"] = v["text"].ljust(max_len)

    try:
        box = font.getbbox("M")
        char_w = box[2] - box[0]
        char_h = box[3] - box[1]
    except Exception:
        char_w = 10
        char_h = 18

    # 行距调大一点
    line_height = char_h + 6
    margin_x = 10    # 左右边距
    margin_y = 10    # 上下边距

    img_width = max_len * char_w + 2 * margin_x
    img_height = len(visual_lines) * line_height + 2 * margin_y

    img = Image.new("RGB", (img_width, img_height), bgcolor)
    draw = ImageDraw.Draw(img)

    for line_idx, v in enumerate(visual_lines):
        text = v["text"]
        spans = v["spans"]
        y = margin_y + line_idx * line_height

        # 整行着色
        if spans and isinstance(spans[0][0], str) and spans[0][0] == "FULL":
            full_color = spans[0][1]
            draw.text((margin_x, y), text, font=font, fill=full_color)
            continue

        # 没有 span → 整行黑色
        if not spans:
            draw.text((margin_x, y), text, font=font, fill=(0, 0, 0))
            continue

        # 有局部高亮 span → 逐字符画
        for x_idx, ch in enumerate(text):
            x = margin_x + x_idx * char_w
            color = (0, 0, 0)
            for span in spans:
                if isinstance(span[0], str):
                    continue
                s, e, c = span
                if s <= x_idx < e:
                    color = c
                    break
            draw.text((x, y), ch, font=font, fill=color)

    return img

# ============== 生成 GIF（最后一帧更慢） ==============

def make_gif_advanced(
    yo_path: str,
    out_path: str = "y86_advanced.gif",
    delay_sec: float = 1.0,
    repeat_per_step: int = 3,
    error_last_factor: float = 3.0,  # 错误 / HLT 最后一帧放慢倍数
):
    frames, total_steps, ended_stat = build_all_frames(yo_path)
    if not frames:
        print("没有帧可渲染（可能加载失败）")
        return

    max_inner_width = 0
    for frame in frames:
        for sec in ["header", "regs", "mem", "progress"]:
            for line_spec in frame[sec]:
                max_inner_width = max(max_inner_width, len(line_spec["text"]))
    max_inner_width = max(max_inner_width, 50)

    font = pick_monospace_font(size=16)

    pil_frames = []
    for frame in frames:
        img = render_frame_to_image(frame, inner_width=max_inner_width, font=font)
        for _ in range(repeat_per_step):
            pil_frames.append(img)

    if not pil_frames:
        print("没有可保存的帧")
        return

    base_ms = int(delay_sec * 1000)
    durations = [base_ms] * len(pil_frames)

    # 如果程序以 ADR/INS 结束，就把最后一步的所有帧放慢很多
    if ended_stat in (STAT_ADR, STAT_INS, STAT_HLT):
        start = max(0, len(pil_frames) - repeat_per_step)
        slow_ms = int(base_ms * error_last_factor)
        for i in range(start, len(pil_frames)):
            durations[i] = slow_ms

    first, rest = pil_frames[0], pil_frames[1:]
    first.save(
        out_path,
        save_all=True,
        append_images=rest,
        format="GIF",
        loop=0,
        duration=durations,
    )
    print(
        f"GIF 已生成: {out_path} "
        f"(普通帧 {base_ms} ms, 最后帧 {durations[-1]} ms, 每步重复 {repeat_per_step} 次)"
    )

# ============== main ==============

def main():
    if len(sys.argv) < 2:
        print("用法: python animate_gif.py test/prog1.yo")
        sys.exit(1)
    yo_path = sys.argv[1]
    make_gif_advanced(
        yo_path,
        out_path="y86_advanced.gif",
        delay_sec=1.0,        # 普通帧每帧 1 秒
        repeat_per_step=2,    # 每条指令 2 帧
        error_last_factor=3.0 # 错误/HLT 最后一帧 = 3 倍时间
    )

if __name__ == "__main__":
    main()
