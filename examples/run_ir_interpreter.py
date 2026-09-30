"""Run a direct Program example: python -m examples.run_ir_interpreter."""

import argparse

from benchmarks.ir_interpreter_cases import CASE_FACTORIES, compare, make_case
from scratchv.verification.ir_interpreter import IRInterpreter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=list(CASE_FACTORIES), default="matmul_add_softmax"
    )
    args = parser.parse_args(argv)
    case = make_case(args.case)
    result = IRInterpreter(case.program).run(
        case.inputs, initializers=case.initializers
    )
    error = compare(result.return_value, case)
    print(f"{case.name}: PASS, steps={result.executed_steps}, max_abs_error={error}")
    print(result.return_value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
