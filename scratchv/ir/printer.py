"""IR printer: pretty-prints IR programs for debugging."""

from scratchv.ir.types import Program


class IRPrinter:
    """Prints an IR Program in a human-readable text format."""

    def __init__(self, program: Program):
        self.program = program

    def dump(self) -> str:
        return self.program.dump()

    def print(self) -> None:
        print(self.dump())
