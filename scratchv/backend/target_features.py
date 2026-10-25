"""ISA/ABI requirements for scalar RISC-V instruction selection."""

from dataclasses import dataclass

from scratchv.ir.types import DataType, OpCode


@dataclass(frozen=True)
class TargetFeatures:
    isa: str = "rv32im"
    abi: str = "ilp32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "isa", self.isa.lower())
        object.__setattr__(self, "abi", self.abi.lower())
        if (self.isa, self.abi) not in {
            ("rv32im", "ilp32"),
            ("rv32imf", "ilp32f"),
            ("rv32imfd", "ilp32d"),
        }:
            raise ValueError(
                f"unsupported RISC-V ISA/ABI combination: {self.isa} + {self.abi}"
            )

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset(char.upper() for char in self.isa.removeprefix("rv32"))

    def require_float(self, opcode: OpCode, dtype: DataType) -> None:
        if dtype not in {DataType.FLOAT32, DataType.FLOAT64}:
            raise ValueError(f"{opcode.value} requires an f32 or f64 dtype")
        required = "D" if dtype is DataType.FLOAT64 else "F"
        if required not in self.extensions:
            raise ValueError(
                f"{opcode.value} {dtype.value} requires the RISC-V {required} "
                f"extension (target {self.isa}/{self.abi})"
            )
