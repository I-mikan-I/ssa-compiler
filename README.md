# Rust 🦀 SSA Compiler WIP

A compiler that uses SSA (single static assignment) form as its definitive IR written in pure Rust!

The compiler will host an optimization pipeline and native machine code generation.

## Roadmap

- [x] Parser (DONE)
- [ ] Intermediate Representation
    - [x] Definition (DONE)
    - [x] Translation from AST (DONE)
    - [x] CFG construction (DONE)
    - [x] SSA transformation (DONE)
- [ ] Optimization passes (WIP)
    - [ ] GVN-PRE (WIP)
- [ ] Backend

## Cargo features

| Feature | Description |
| --- | --- |
| `print-cfgs` | Displays every constructed CFG as a [dot](https://graphviz.org/doc/info/lang.html) graph. |
