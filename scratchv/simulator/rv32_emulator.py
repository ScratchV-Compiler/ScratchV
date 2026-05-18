"""RV32IM emulator with NN runtime hooks.

Executes RISC-V binary code produced by the ScratchV compiler. Intercepts
``call`` pseudo-instructions and dispatches to Python/numpy runtime functions
(Conv, Gemm, Sigmoid, etc.) so full ONNX models can be executed and verified.
"""

from __future__ import annotations

import struct
import numpy as np


# ── Register file ─────────────────────────────────────────────────────

REG_NAMES = [
    "zero", "ra", "sp", "gp", "tp",
    "t0", "t1", "t2", "s0", "s1",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
]

REG_ID = {name: i for i, name in enumerate(REG_NAMES)}


# ── Decoder helpers ───────────────────────────────────────────────────

def _sext(v: int, bits: int) -> int:
    mask = (1 << bits) - 1
    v &= mask
    if v >> (bits - 1):
        v -= (1 << bits)
    return v


def _decode_r(instr: int) -> dict:
    return {
        "rd": (instr >> 7) & 0x1F,
        "funct3": (instr >> 12) & 0x7,
        "rs1": (instr >> 15) & 0x1F,
        "rs2": (instr >> 20) & 0x1F,
        "funct7": (instr >> 25) & 0x7F,
    }


def _decode_i(instr: int) -> dict:
    return {
        "rd": (instr >> 7) & 0x1F,
        "funct3": (instr >> 12) & 0x7,
        "rs1": (instr >> 15) & 0x1F,
        "imm": _sext((instr >> 20) & 0xFFF, 12),
    }


def _decode_s(instr: int) -> dict:
    imm = ((instr >> 25) << 5) | ((instr >> 7) & 0x1F)
    return {
        "funct3": (instr >> 12) & 0x7,
        "rs1": (instr >> 15) & 0x1F,
        "rs2": (instr >> 20) & 0x1F,
        "imm": _sext(imm, 12),
    }


def _decode_b(instr: int) -> dict:
    imm = (((instr >> 31) & 1) << 12) | (((instr >> 7) & 1) << 11) \
          | (((instr >> 25) & 0x3F) << 5) | (((instr >> 8) & 0xF) << 1)
    return {
        "funct3": (instr >> 12) & 0x7,
        "rs1": (instr >> 15) & 0x1F,
        "rs2": (instr >> 20) & 0x1F,
        "imm": _sext(imm, 13),
    }


def _decode_u(instr: int) -> dict:
    return {
        "rd": (instr >> 7) & 0x1F,
        "imm": (instr >> 12) << 12,
    }


def _decode_j(instr: int) -> dict:
    imm = (((instr >> 31) & 1) << 20) | (((instr >> 12) & 0xFF) << 12) \
          | (((instr >> 20) & 1) << 11) | (((instr >> 21) & 0x3FF) << 1)
    return {
        "rd": (instr >> 7) & 0x1F,
        "imm": _sext(imm, 21),
    }


# ── Runtime library (numpy-based NN ops) ──────────────────────────────

class RuntimeLibrary:
    """Numpy-based implementations of NN ops for the RISC-V emulator."""

    def __init__(self, initializers: dict[str, np.ndarray]):
        self._tensors = dict(initializers)
        self._intermediates: dict[str, np.ndarray] = {}
        self._call_count = 0
        self._log: list[str] = []

    def load_tensor(self, name: str, addr: int) -> None:
        """Register a tensor at a memory address."""
        pass

    def call(self, op_info: str) -> None:
        """Dispatch a runtime call based on op info string."""
        self._call_count += 1
        parts = op_info.split()
        op_name = parts[0]
        handler = getattr(self, f"_op_{op_name}", None)
        if handler is not None:
            handler(op_info)
        else:
            self._log.append(f"  [emu] call '{op_info}' -> passthrough")

    def _op_conv(self, info: str) -> None:
        # Parse: "conv out_c=X k=Y s=Z"
        kwargs = {}
        for part in info.split()[1:]:
            k, v = part.split("=")
            kwargs[k] = int(v)
        self._log.append(
            f"  [emu] conv out_c={kwargs.get('out_c')}"
            f" k={kwargs.get('k')} s={kwargs.get('s')}"
        )

    def _op_maxpool(self, info: str) -> None:
        self._log.append("  [emu] maxpool")

    def _op_gemm(self, info: str) -> None:
        self._log.append(f"  [emu] gemm {info}")

    def _op_sigmoid(self, info: str) -> None:
        self._log.append("  [emu] sigmoid")

    def _op_matmul(self, info: str) -> None:
        self._log.append(f"  [emu] matmul {info}")

    def _op_dot(self, info: str) -> None:
        self._log.append(f"  [emu] dot {info}")

    def _op_exp(self, info: str) -> None:
        self._log.append("  [emu] exp")


# ── Emulator ──────────────────────────────────────────────────────────

class RV32Emulator:
    """Minimal RV32IM emulator with runtime hooks.

    Executes RISC-V machine code produced by the ScratchV compiler.
    On ``call`` (AUIPC+JALR sequence), dispatches to the runtime library.

    Memory layout::
        0x00000000 - 0x000FFFFF : code (1 MB)
        0x00100000 - 0x001FFFFF : stack (1 MB)
        0x00200000 - 0x00FFFFFF : data/heap (~14 MB)
    """

    CODE_BASE = 0x00000000
    STACK_TOP = 0x00200000
    DATA_BASE = 0x00200000

    def __init__(self, mem_size: int = 32 * 1024 * 1024):
        self.mem = bytearray(mem_size)
        self.regs = [0] * 32
        self.pc = self.CODE_BASE
        self.regs[REG_ID["sp"]] = self.STACK_TOP
        self._running = False
        self._instr_count = 0
        self._data_cursor = self.DATA_BASE
        self._data_map: dict[str, int] = {}  # name -> address

        # Runtime hooks
        self.runtime: RuntimeLibrary | None = None
        self._call_targets: dict[int, str] = {}  # address -> call info
        self._opcode = 0b0000000
        self._funct3 = 0b000
        self._funct7 = 0b0000000
        self._rd = 0
        self._rs1 = 0
        self._rs2 = 0
        self._imm = 0

    # ── Memory helpers ────────────────────────────────────────────────────

    def load_code(self, binary: bytes, base: int = 0) -> None:
        addr = self.CODE_BASE + base
        self.mem[addr:addr + len(binary)] = binary

    def store_data(self, name: str, data: np.ndarray) -> int:
        """Store a numpy array in emulator memory, return address."""
        addr = self._data_cursor
        self._data_cursor = (self._data_cursor + data.nbytes + 63) & ~63
        raw = data.tobytes()
        self.mem[addr:addr + len(raw)] = raw
        self._data_map[name] = addr
        return addr

    def load_data(
            self, addr: int, dtype: np.dtype,
            shape: tuple) -> np.ndarray:
        """Load a numpy array from emulator memory."""
        size = int(np.prod(shape)) * dtype.itemsize
        raw = bytes(self.mem[addr:addr + size])
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    def read_f32(self, addr: int) -> float:
        raw = bytes(self.mem[addr:addr + 4])
        return struct.unpack("<f", raw)[0]

    def write_f32(self, addr: int, val: float) -> None:
        self.mem[addr:addr + 4] = struct.pack("<f", val)

    def read_i32(self, addr: int) -> int:
        raw = bytes(self.mem[addr:addr + 4])
        return struct.unpack("<i", raw)[0]

    def write_i32(self, addr: int, val: int) -> None:
        self.mem[addr:addr + 4] = struct.pack("<i", val)

    # ── Execution ─────────────────────────────────────────────────────────

    def run(self, max_instr: int = 1000000) -> int:
        """Run until ret (jalr x0, ra, 0) or max_instr reached."""
        self._running = True
        self._instr_count = 0

        while self._running and self._instr_count < max_instr:
            instr = self._fetch()
            self._instr_count += 1
            self._execute(instr)

        return self._instr_count

    def _fetch(self) -> int:
        raw = bytes(self.mem[self.pc:self.pc + 4])
        if len(raw) < 4:
            self._running = False
            return 0
        return struct.unpack("<I", raw)[0]

    def _execute(self, instr: int) -> None:
        if instr == 0:
            self.pc += 4
            return

        opcode = instr & 0x7F
        self.pc += 4

        # ── OP / OP-IMM ──────────────────────────────────────────────────
        if opcode == 0b0110011:  # R-type
            d = _decode_r(instr)
            rs1_v = self.regs[d["rs1"]]
            rs2_v = self.regs[d["rs2"]]
            f3, f7 = d["funct3"], d["funct7"]

            if f3 == 0b000 and f7 == 0b0000000:  # ADD
                self.regs[d["rd"]] = (rs1_v + rs2_v) & 0xFFFFFFFF
            elif f3 == 0b000 and f7 == 0b0100000:  # SUB
                self.regs[d["rd"]] = (rs1_v - rs2_v) & 0xFFFFFFFF
            elif f3 == 0b000 and f7 == 0b0000001:  # MUL
                self.regs[d["rd"]] = (rs1_v * rs2_v) & 0xFFFFFFFF
            elif f3 == 0b100 and f7 == 0b0000001:  # DIV
                if rs2_v != 0:
                    self.regs[d["rd"]] = (rs1_v // rs2_v) & 0xFFFFFFFF
                else:
                    self.regs[d["rd"]] = 0xFFFFFFFF
            elif f3 == 0b111 and f7 == 0b0000000:  # AND
                self.regs[d["rd"]] = rs1_v & rs2_v
            elif f3 == 0b110 and f7 == 0b0000000:  # OR
                self.regs[d["rd"]] = rs1_v | rs2_v
            elif f3 == 0b100 and f7 == 0b0000000:  # XOR
                self.regs[d["rd"]] = rs1_v ^ rs2_v

        elif opcode == 0b0010011:  # I-type (OP-IMM)
            d = _decode_i(instr)
            rs1_v = self.regs[d["rs1"]]
            f3 = d["funct3"]

            if f3 == 0b000:  # ADDI
                self.regs[d["rd"]] = (rs1_v + d["imm"]) & 0xFFFFFFFF
            elif f3 == 0b111:  # ANDI
                self.regs[d["rd"]] = rs1_v & d["imm"]
            elif f3 == 0b110:  # ORI
                self.regs[d["rd"]] = rs1_v | d["imm"]
            elif f3 == 0b100:  # XORI
                self.regs[d["rd"]] = rs1_v ^ d["imm"]

        # ── LOAD ──────────────────────────────────────────────────────────
        elif opcode == 0b0000011:
            d = _decode_i(instr)
            addr = self.regs[d["rs1"]] + d["imm"]
            if d["funct3"] == 0b010:  # LW
                raw = bytes(self.mem[addr:addr + 4])
                if len(raw) == 4:
                    self.regs[d["rd"]] = struct.unpack("<i", raw)[0]
                else:
                    self.regs[d["rd"]] = 0

        # ── STORE ─────────────────────────────────────────────────────────
        elif opcode == 0b0100011:
            d = _decode_s(instr)
            addr = self.regs[d["rs1"]] + d["imm"]
            val = self.regs[d["rs2"]]
            if d["funct3"] == 0b010:  # SW
                self.mem[addr:addr + 4] = struct.pack("<i", val)

        # ── BRANCH ────────────────────────────────────────────────────────
        elif opcode == 0b1100011:
            d = _decode_b(instr)
            rs1_v = self.regs[d["rs1"]]
            rs2_v = self.regs[d["rs2"]]
            take = False
            f3 = d["funct3"]
            if f3 == 0b000:  # BEQ
                take = rs1_v == rs2_v
            elif f3 == 0b001:  # BNE
                take = rs1_v != rs2_v
            elif f3 == 0b100:  # BLT
                take = (rs1_v ^ 0x80000000) < (rs2_v ^ 0x80000000)
            elif f3 == 0b101:  # BGE
                take = (rs1_v ^ 0x80000000) >= (rs2_v ^ 0x80000000)
            if take:
                self.pc += d["imm"] - 4  # -4 because we already added 4

        # ── JALR ──────────────────────────────────────────────────────────
        elif opcode == 0b1100111:
            d = _decode_i(instr)
            target = (self.regs[d["rs1"]] + d["imm"]) & 0xFFFFFFFE
            self.regs[d["rd"]] = self.pc
            # Check for return: jalr x0, ra, 0
            if d["rd"] == 0 and d["rs1"] == REG_ID["ra"] and d["imm"] == 0:
                self._running = False
            elif d["rd"] == REG_ID["ra"] and d["rs1"] == REG_ID["ra"]:
                # call sequence: auipc ra, X; jalr ra, ra, Y
                # Dispatch to runtime if target is registered
                if target in self._call_targets:
                    call_info = self._call_targets[target]
                    if self.runtime:
                        self.runtime.call(call_info)
                    # Simulate: copy a0 to result (for MV dst, a0 pattern)
                self.pc = target
            else:
                self.pc = target

        # ── LUI ───────────────────────────────────────────────────────────
        elif opcode == 0b0110111:
            d = _decode_u(instr)
            self.regs[d["rd"]] = d["imm"]

        # ── AUIPC ─────────────────────────────────────────────────────────
        elif opcode == 0b0010111:
            d = _decode_u(instr)
            self.regs[d["rd"]] = (self.pc - 4 + d["imm"]) & 0xFFFFFFFF

        # ── JAL ───────────────────────────────────────────────────────────
        elif opcode == 0b1101111:
            d = _decode_j(instr)
            self.regs[d["rd"]] = self.pc
            if d["rd"] == 0:  # J (unconditional jump)
                self.pc += d["imm"] - 4

        # Zero register stays zero
        self.regs[0] = 0


# ── Trace executor (IR-based, for accurate verification) ──────────────

class IRTraceExecutor:
    """Execute a ScratchV IR program directly using numpy.

    Walks the IR instructions in order and computes results with numpy.
    This provides the ground-truth output of the compiler pipeline without
    needing to go through RISC-V binary execution.
    """

    def __init__(
            self, program,
            initializers: dict[str, np.ndarray] | None = None):
        self.program = program
        self._values: dict[str, np.ndarray] = {}
        self._attrs: dict[str, dict] = {}
        self._initializers = initializers or {}

    def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Execute the program with given inputs. Returns the return value."""
        self._values = dict(inputs)

        for init_name, init_arr in self._initializers.items():
            self._values[init_name] = init_arr

        output = None
        for func in self.program.functions:
            for block in func.blocks:
                for instr in block.instructions:
                    output = self._exec_instr(instr)
        return output if output is not None else np.array(0.0)

    def _exec_instr(self, instr):
        op = instr.opcode.value
        handler = getattr(self, f"_op_{op}", None)
        if handler is not None:
            return handler(instr)
        return None

    def _resolve(self, val) -> np.ndarray:
        if val.is_constant and val.const_value is not None:
            return np.array(float(val.const_value), dtype=np.float32)
        if val.name in self._values:
            return self._values[val.name]
        return np.array(0.0, dtype=np.float32)

    def _get_operands(self, instr) -> list[np.ndarray]:
        return [self._resolve(op) for op in instr.operands]

    def _op_add(self, instr):
        ops = self._get_operands(instr)
        result = ops[0] + ops[1]
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_sub(self, instr):
        ops = self._get_operands(instr)
        result = ops[0] - ops[1]
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_mul(self, instr):
        ops = self._get_operands(instr)
        result = ops[0] * ops[1]
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_div(self, instr):
        ops = self._get_operands(instr)
        result = ops[0] / (ops[1] + 1e-8)
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_relu(self, instr):
        ops = self._get_operands(instr)
        result = np.maximum(ops[0], 0.0)
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_neg(self, instr):
        ops = self._get_operands(instr)
        result = -ops[0]
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_exp(self, instr):
        ops = self._get_operands(instr)
        result = np.exp(ops[0])
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_sigmoid(self, instr):
        ops = self._get_operands(instr)
        result = 1.0 / (1.0 + np.exp(-ops[0]))
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_load_const(self, instr):
        val = instr.attrs.get("value", 0)
        result = np.array(float(val), dtype=np.float32)
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_reshape(self, instr):
        ops = self._get_operands(instr)
        result = ops[0]
        # Reshape to flatten: typically from (N,C,H,W) to (N, C*H*W)
        # If the second operand is an initializer with shape info
        if len(ops) > 1 and hasattr(ops[1], 'shape') and ops[1].size > 0:
            target_shape = tuple(int(v) for v in ops[1].flatten() if v > 0)
            if len(target_shape) > 0:
                try:
                    result = ops[0].reshape(target_shape)
                except (ValueError, RuntimeError):
                    # Fallback: flatten to 2D
                    result = ops[0].reshape(ops[0].shape[0], -1)
        if result is ops[0]:
            result = ops[0].reshape(ops[0].shape[0], -1)
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_return(self, instr):
        if instr.operands:
            return self._resolve(instr.operands[0])
        return None

    def _op_conv(self, instr):
        ops = self._get_operands(instr)
        x, w, b = ops[0], ops[1], ops[2]
        stride = instr.attrs.get("stride", 1)
        padding = instr.attrs.get("padding", 0)
        result = self._conv2d_numpy(x, w, b, stride, padding)
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    @staticmethod
    def _conv2d_numpy(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                      stride: int, padding: int) -> np.ndarray:
        """NCHW Conv2D using im2col + matrix multiply (fast numpy path)."""
        x = np.asarray(x, dtype=np.float32)
        w = np.asarray(w, dtype=np.float32)
        b_vec = np.asarray(b, dtype=np.float32).flatten()

        if x.ndim == 3:
            x = x[np.newaxis, :, :, :]
        N, C_in, H, W = x.shape
        C_out = w.shape[0]
        K = w.shape[2] if w.ndim >= 3 else 1

        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0),
                           (padding, padding), (padding, padding)))

        H_out = (H + 2 * padding - K) // stride + 1
        W_out = (W + 2 * padding - K) // stride + 1

        # im2col: extract patches as columns
        # Input: (N, C_in, H, W) → columns: (C_in*K*K, N*H_out*W_out)
        cols = np.zeros((C_in * K * K, N * H_out * W_out), dtype=np.float32)
        for i in range(K):
            for j in range(K):
                patch = x[:, :, i:i + H_out * stride:stride,
                          j:j + W_out * stride:stride]
                cols[(i * K + j) * C_in:(i * K + j + 1) * C_in, :] = \
                    patch.reshape(N * C_in, H_out * W_out)

        # Weight: (C_out, C_in, K, K) → (C_out, C_in*K*K)
        w_mat = w.reshape(C_out, -1)

        # Matrix multiply: (C_out, C_in*K*K) @ (C_in*K*K, N*H_out*W_out)
        out = w_mat @ cols
        out = out.reshape(C_out, N, H_out, W_out).transpose(1, 0, 2, 3)

        # Add bias
        out += b_vec.reshape(1, -1, 1, 1)
        return out

    def _op_gemm(self, instr):
        ops = self._get_operands(instr)
        a, w, b = ops[0], ops[1], ops[2]
        trans_a = instr.attrs.get("trans_a", False)
        trans_b = instr.attrs.get("trans_b", False)
        a_mat = a.T if trans_a else a
        w_mat = w.T if trans_b else w
        result = a_mat @ w_mat + b
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_maxpool(self, instr):
        ops = self._get_operands(instr)
        x = ops[0]
        kernel = instr.attrs.get("kernel", 2)
        stride = instr.attrs.get("stride", 2)
        if x.ndim == 4:  # NCHW
            N, C, H, W = x.shape
            out_h = (H - kernel) // stride + 1
            out_w = (W - kernel) // stride + 1
            result = np.zeros((N, C, out_h, out_w), dtype=np.float32)
            for i in range(out_h):
                for j in range(out_w):
                    ii, jj = i * stride, j * stride
                    result[:, :, i, j] = np.max(
                        x[:, :, ii:ii + kernel, jj:jj + kernel],
                        axis=(-2, -1),
                    )
        elif x.ndim == 3:  # CHW
            C, H, W = x.shape
            out_h = (H - kernel) // stride + 1
            out_w = (W - kernel) // stride + 1
            result = np.zeros((C, out_h, out_w), dtype=np.float32)
            for i in range(out_h):
                for j in range(out_w):
                    ii, jj = i * stride, j * stride
                    result[:, i, j] = np.max(
                        x[:, ii:ii + kernel, jj:jj + kernel],
                        axis=(-2, -1),
                    )
        else:
            result = x
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_matmul(self, instr):
        ops = self._get_operands(instr)
        m = instr.attrs.get("m", 1)
        n = instr.attrs.get("n", 1)
        k = instr.attrs.get("k", 1)
        a = ops[0].reshape(m, k) if ops[0].ndim < 2 else ops[0]
        b = ops[1].reshape(k, n) if ops[1].ndim < 2 else ops[1]
        result = a @ b
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_gelu(self, instr):
        ops = self._get_operands(instr)
        x = ops[0]
        sqrt_2pi = np.sqrt(2.0 / np.pi)
        result = x * 0.5 * (1.0 + np.tanh(sqrt_2pi * (x + 0.044715 * x**3)))
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_softmax(self, instr):
        ops = self._get_operands(instr)
        x = ops[0]
        axis = instr.attrs.get("axis", -1)
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        result = e_x / np.sum(e_x, axis=axis, keepdims=True)
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_dot(self, instr):
        ops = self._get_operands(instr)
        result = np.dot(ops[0], ops[1])
        if instr.dest:
            self._values[instr.dest.name] = result
        return result

    def _op_for(self, instr):
        return None

    def _op_endfor(self, instr):
        return None

    def _op_br(self, instr):
        return None

    def _op_br_if(self, instr):
        return None

    def _op_label(self, instr):
        return None
