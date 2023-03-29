# Rust 🦀 SSA Compiler WIP

A compiler that uses SSA (single static assignment form) as its definitive IR written entirely in Rust!

The compiler hosts an optimization pipeline and native assembly generation.

## Usage

Supply the binary with command-line arguments:

Display Options:
```sh
$ cargo run -- --help
```

Compile `correct3.lang` into RISCV assembly:
```sh
$ mkdir build
$ cargo run -- examples/correct3.lang -o build
```

Disable optimization:
```sh
$ mkdir build
$ cargo run -- examples/correct3.lang -o build --no-optimize
```

## Roadmap

- [x] Parser (DONE)
- [x] Intermediate Representation (DONE)
    - [x] Definition (DONE)
    - [x] Translation from AST (DONE)
    - [x] CFG construction (DONE)
    - [x] SSA transformation (DONE)
- [x] Optimization passes (DONE)
    - [x] GVN-PRE (DONE)
    - [x] Copy propagation (DONE)
- [x] Backend (DONE)
    - [x] Register allocation (DONE)
    - [x] Instruction selection (DONE)
- [ ] Global data (TODO)
- [ ] Assembler integration (TODO)

## Cargo features

| Feature | Description |
| --- | --- |
| `print-cfgs` | Displays every constructed CFG as a [dot](https://graphviz.org/doc/info/lang.html) graph. |
| `print-gvn` | Displays debug information of the GVN-PRE pass. |
| `print-linear` | Displays generated linear code. |