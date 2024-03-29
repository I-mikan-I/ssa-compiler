// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
//! Intermediate Representation
//!
//! The compiler's IR is a three-address, register based code with an infinite number of registers.
//!
//! | Instruction | Ad1 | Ad2 | Ad3 |
//! | ----------- | --- | --- | --- |
//! | add ad1, ad2, ad3 | result | op1 | op2 |
//! | sub ad1, ad2, ad3 | result | op1 | op2 |
//! | mult ad1, ad2, ad3 | result | op1 | op2 |
//! | div ad1, ad2, ad3 | result | op1 | op2 |
//! | and ad1, ad2, ad3 | result | op1 | op2 |
//! | or ad1, ad2, ad3 | result | op1 | op2 |
//! | mv ad1, ad2 | result | op1 | NA |
//! | xor ad1, ad2, ad3 | result | op1 | op2 |
//! | load ad1, ad2, ad3 | result | offset | memory base |
//! | store ad1, ad2, ad3 | value | offset | memory base |
//! | la ad1, @label | result | NA | NA |
//! | bgt ad1, ad2, @taken, @else | op1 | op2 | NA |
//! | bl ad1, ad2, @taken, @else | op1 | op2 | NA |
//! | j @label | NA | NA | NA |
//! | beq ad1, ad2, @taken, @else | op1 | op2 | NA |
//! | li ad1, imm  | result | NA | NA |
//! | slt ad1, ad2, ad3 | result | op1 | op2 |
//! | call ad1, @label (ad2 ... adn) | result | arg1 | ... |
//! | return ad1 | result | NA | NA |
//!
//! - Labels are represented as immediate string arguments to the op.
//! - Memory locations are symbolic, an execution could for example
//!   assign a random value to each label.
//! - ad1, ad2, ad3 are registers.
//! - Registers hold 8 bytes.
//! - Negative numbers use 2's complement.
//! - Memory locations store a single register.
//!
//! ## Code Shape
//!
//! ```bnf
//! lin: transforms AST into IR before inserting line, returns register name of result.  
//! name: allocates register name.  
//! label: allocates memory location.  
//! body: inserts IR of AST.  
//!
//! (L) + (R) =>    add name() lin(L) lin(R)
//! (L) - (R) =>    sub name() lin(L) lin(R)
//! (L) * (R) =>    mul name() lin(L) lin(R)
//! (L) / (R) =>    div name() lin(L) lin(R)
//! (L) or (R) =>   ...
//! (L) and (R) =>  ...
//! (L) xor (R) =>  ...
//! not (L) =>      li name() -1 // r_new
//!                 xor lin(L) r_new
//! (L) >= (R) =>   slt name() lin(L) lin(R) // r_new
//!                 li name() 1 // r_new2
//!                 xor name() r_new r_new2
//! (L) < (R) =>    slt name() lin(L) lin(R)
//! (L) == (R) =>   xor name() lin(L) lin(R) // r_new
//!                 li name() 1 // r_new2
//!                 slt name() r_new r_new2
//! true =>         li name() 1
//! false =>        li name() 0
//! number =>       li name() number
//! ident = (R)     mv name() lin(R)
//! ident           name(ident) // reuse register
//! global          la name() global
//! global = (R)    la name() global // r_new
//!                 store lin(R) 0 r_new
//! if (C) then (I)
//! else (E) =>     li name() 1 // r_new
//!                 beq lin(C) r_new @l1 @l2
//!                 label(): body(I) // @l1
//!                 j @l3
//!                 label(): body(E) // @l2
//!                 j @l3
//!                 label(): ... // @l3
//! while (C)
//! do (B) =>       j @l1
//!                 label():    // @l1
//!                 li name() 0 // r_new
//!                 beq lin(C) r_ne2 @l2 @l3
//!                 label():    // @l3
//!                 body(B)
//!                 j @l1
//!                 label(): ... // @l2
//! identifier((arg0), (arg1), ...)
//! =>              j @l1
//!                 label():    // @l1
//!                 call @label(identifier) (lin(arg0), lin(arg1), ...)
//!                 j @l2
//!                 label():    // @l2
//! return (E) =>   return lin(E)
//!                 label():    // dead
//!
//! ```

use parser_defs::{Any, Type};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
    rc::Rc,
};

use crate::util::SheafTable;

pub type VReg = u32;
pub struct VRegGenerator(u32, u32, String);
impl VRegGenerator {
    pub fn starting_at_reg(at: u32) -> Self {
        Self(at, 0, "_LABEL_".into())
    }
    pub fn with_prefix(prefix: String) -> Self {
        Self(0, 0, prefix)
    }
    pub fn new() -> Self {
        Self(0, 0, "_LABEL_".into())
    }
    pub fn next_reg(&mut self) -> u32 {
        let res = self.0;
        self.0 += 1;
        res
    }
    pub fn next_label(&mut self) -> String {
        let res = format!("{}{}", self.2, self.1);
        self.1 += 1;
        res
    }
}
impl Default for VRegGenerator {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Operator {
    Add(VReg, VReg, VReg),
    Sub(VReg, VReg, VReg),
    Mult(VReg, VReg, VReg),
    Div(VReg, VReg, VReg),
    And(VReg, VReg, VReg),
    Or(VReg, VReg, VReg),
    Mv(VReg, VReg),
    Xor(VReg, VReg, VReg),
    Load(VReg, VReg, VReg),
    Store(VReg, VReg, VReg),
    La(VReg, Rc<str>),
    Bgt(VReg, VReg, Rc<str>, Rc<str>),
    Bl(VReg, VReg, Rc<str>, Rc<str>),
    J(Rc<str>),
    Beq(VReg, VReg, Rc<str>, Rc<str>),
    Li(VReg, i64),
    Slt(VReg, VReg, VReg),
    Call(VReg, Rc<str>, Vec<VReg>),
    Return(VReg),
    Label(Rc<str>),
    /// Loads param 2nd into fst
    GetParameter(VReg, u64),
    StoreLocal(VReg, u64),
    LoadLocal(VReg, u64),
    Nop,
}
impl Operator {
    pub fn dependencies(&self) -> Vec<VReg> {
        match self {
            Operator::Add(_, y, z)
            | Operator::Sub(_, y, z)
            | Operator::Mult(_, y, z)
            | Operator::Div(_, y, z)
            | Operator::Bl(z, y, _, _)
            | Operator::Slt(_, y, z)
            | Operator::Bgt(z, y, _, _)
            | Operator::Load(_, y, z)
            | Operator::Beq(z, y, _, _)
            | Operator::And(_, y, z)
            | Operator::Xor(_, y, z)
            | Operator::Or(_, y, z) => {
                vec![*y, *z]
            }
            Operator::StoreLocal(y, _) | Operator::Mv(_, y) => {
                vec![*y]
            }
            Operator::Store(x, y, z) => {
                vec![*x, *y, *z]
            }
            Operator::Call(_, _, z) => z.clone(),
            Operator::Return(x) => vec![*x],
            _ => vec![],
        }
    }
    pub fn dependencies_mut(&mut self) -> Vec<&mut VReg> {
        match self {
            Operator::Add(_, y, z)
            | Operator::Sub(_, y, z)
            | Operator::Mult(_, y, z)
            | Operator::Div(_, y, z)
            | Operator::Bl(z, y, _, _)
            | Operator::Slt(_, y, z)
            | Operator::Bgt(z, y, _, _)
            | Operator::Load(_, y, z)
            | Operator::Beq(z, y, _, _)
            | Operator::And(_, y, z)
            | Operator::Xor(_, y, z)
            | Operator::Or(_, y, z) => {
                vec![y, z]
            }
            Operator::StoreLocal(y, _) | Operator::Mv(_, y) => {
                vec![y]
            }
            Operator::Store(x, y, z) => {
                vec![x, y, z]
            }
            Operator::Call(_, _, z) => z.iter_mut().collect(),
            Operator::Return(x) => vec![x],
            _ => vec![],
        }
    }
    pub fn receiver(&self) -> Option<VReg> {
        match self {
            Operator::Add(x, _, _)
            | Operator::Sub(x, _, _)
            | Operator::Mult(x, _, _)
            | Operator::Div(x, _, _)
            | Operator::And(x, _, _)
            | Operator::Or(x, _, _)
            | Operator::Mv(x, _)
            | Operator::Xor(x, _, _)
            | Operator::Load(x, _, _)
            | Operator::La(x, _)
            | Operator::Li(x, _)
            | Operator::Slt(x, _, _)
            | Operator::Call(x, _, _)
            | Operator::LoadLocal(x, _)
            | Operator::GetParameter(x, _) => Some(*x),
            _ => None,
        }
    }
    pub fn receiver_mut(&mut self) -> Option<&mut VReg> {
        match self {
            Operator::Add(x, _, _)
            | Operator::Sub(x, _, _)
            | Operator::Mult(x, _, _)
            | Operator::Div(x, _, _)
            | Operator::And(x, _, _)
            | Operator::Or(x, _, _)
            | Operator::Mv(x, _)
            | Operator::Xor(x, _, _)
            | Operator::Load(x, _, _)
            | Operator::La(x, _)
            | Operator::Li(x, _)
            | Operator::Slt(x, _, _)
            | Operator::Call(x, _, _)
            | Operator::LoadLocal(x, _)
            | Operator::GetParameter(x, _) => Some(x),
            _ => None,
        }
    }
    pub fn replace_reg(&mut self, reg: VReg, with: VReg) {
        macro_rules! replace {
            ($($x:expr),+) => {
            {
                $(
                if *$x == reg {
                    *$x = with;
                }
                )+
            }
            };
        }
        match self {
            Operator::Add(_, y, z)
            | Operator::Sub(_, y, z)
            | Operator::Mult(_, y, z)
            | Operator::Div(_, y, z)
            | Operator::And(_, y, z)
            | Operator::Slt(_, y, z)
            | Operator::Or(_, y, z)
            | Operator::Xor(_, y, z)
            | Operator::Bgt(z, y, _, _)
            | Operator::Bl(z, y, _, _)
            | Operator::Beq(z, y, _, _)
            | Operator::Load(_, y, z) => replace!(y, z),
            Operator::StoreLocal(y, _) | Operator::Return(y) | Operator::Mv(_, y) => replace!(y),
            Operator::Store(x, y, z) => replace!(x, y, z),
            Operator::Call(.., z) => {
                for r in z {
                    replace!(r);
                }
            }
            _ => {}
        }
    }
}

pub struct Displayable<'a, O>(pub &'a [O]);

impl<'a, O> Display for Displayable<'a, O>
where
    O: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for op in self.0 {
            writeln!(f, "{op}")?;
        }
        Ok(())
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Add(rd1, rd2, rd3) => write!(f, "\tadd rd{rd1}, rd{rd2}, rd{rd3}"),
            Operator::Sub(rd1, rd2, rd3) => write!(f, "\tsub rd{rd1}, rd{rd2}, rd{rd3}"),
            Operator::Mult(rd1, rd2, rd3) => write!(f, "\tmult rd{rd1}, rd{rd2}, rd{rd3}"),
            Operator::Div(rd1, rd2, rd3) => write!(f, "\tdiv rd{rd1}, rd{rd2}, rd{rd3}"),
            Operator::And(rd1, rd2, rd3) => write!(f, "\tand rd{rd1}, rd{rd2}, rd{rd3}"),
            Operator::Or(rd1, rd2, rd3) => write!(f, "\tor rd{rd1}, rd{rd2}, rd{rd3}"),
            Operator::Mv(rd1, rd2) => write!(f, "\tmv rd{rd1}, rd{rd2}"),
            Operator::Xor(rd1, rd2, rd3) => write!(f, "\txor rd{rd1}, rd{rd2}, rd{rd3}"),
            Operator::Load(rd1, rd2, rd3) => write!(f, "\tload rd{rd1}, rd{rd2}, rd{rd3}"),
            Operator::Store(rd1, rd2, rd3) => write!(f, "\tstore rd{rd1}, rd{rd2}, rd{rd3}"),
            Operator::La(rd1, rd2) => write!(f, "\tla rd{rd1}, @{rd2}"),
            Operator::Bgt(rd1, rd2, rd3, rd4) => {
                write!(f, "\tbgt rd{rd1}, rd{rd2}, @{rd3}, @{rd4}")
            }
            Operator::Bl(rd1, rd2, rd3, rd4) => write!(f, "\tbl rd{rd1}, rd{rd2}, @{rd3}, @{rd4}"),
            Operator::J(rd1) => write!(f, "\tj @{rd1}"),
            Operator::Beq(rd1, rd2, rd3, rd4) => {
                write!(f, "\tbeq rd{rd1}, rd{rd2}, @{rd3}, @{rd4}")
            }
            Operator::Li(rd1, rd2) => write!(f, "\tli rd{rd1}, #{rd2}"),
            Operator::Slt(rd1, rd2, rd3) => write!(f, "\tslt rd{rd1}, rd{rd2}, rd{rd3}"),
            Operator::Call(rd1, rd2, rd3) => {
                write!(f, "\tcall rd{rd1}, @{rd2}, ( ")?;
                for reg in rd3 {
                    write!(f, "{reg} ")?;
                }
                write!(f, ")")
            }
            Operator::Return(rd1) => write!(f, "\treturn rd{rd1}"),
            Operator::LoadLocal(rd1, imm) => write!(f, "\tloadAR rd{rd1}, AR({imm})"),
            Operator::StoreLocal(rd1, imm) => write!(f, "\tstoreAR rd{rd1}, AR({imm})"),
            Operator::Label(rd1) => write!(f, "@{rd1}:"),
            Operator::GetParameter(rd1, rd2) => write!(f, "\tgetParam rd{rd1}, {rd2}"),
            Operator::Nop => write!(f, "nop"),
        }
    }
}

impl Display for SSAOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SSAOperator::IROp(op) => write!(f, "{op}"),
            SSAOperator::Phi(rd1, args) => {
                write!(f, "\trd{rd1} <- \u{03D5} ( ")?;
                for arg in args {
                    if *arg == u32::MAX {
                        write!(f, "\u{22a5} ")?
                    } else {
                        write!(f, "{arg} ")?;
                    }
                }
                write!(f, ")")
            }
        }
    }
}
pub struct Function<B> {
    body: B,
    params: Vec<VReg>,
    max_reg: VReg,
}
impl<B> Function<B> {
    pub fn get_body(&self) -> &B {
        &self.body
    }
    pub fn into_body(self) -> B {
        self.body
    }
    pub fn get_body_mut(&mut self) -> &mut B {
        &mut self.body
    }
    pub fn get_params(&self) -> &Vec<VReg> {
        &self.params
    }
    pub fn get_max_reg(&self) -> VReg {
        self.max_reg
    }
}
impl Function<Vec<Operator>> {
    pub fn into_cfg(self) -> Function<CFG<Operator>> {
        let mut res = Function {
            body: CFG::from_linear(&self.body, self.get_params(), self.get_max_reg()),
            params: self.params,
            max_reg: 0,
        };
        res.max_reg = res.body.get_max_reg();
        res
    }
}
impl Function<CFG<Operator>> {
    pub fn into_ssa(self) -> Function<CFG<SSAOperator>> {
        let mut res = Function {
            body: self.body.into_ssa(),
            params: self.params,
            max_reg: 0,
        };
        res.max_reg = res.body.get_max_reg();
        res
    }
}
pub struct Context<B> {
    pub functions: HashMap<String, Function<B>>,
    pub globals: HashMap<String, Type>,
}
impl<B> Context<B> {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            globals: HashMap::new(),
        }
    }
    pub fn get_functions(&self) -> &HashMap<String, Function<B>> {
        &self.functions
    }
}
impl<B> Default for Context<B> {
    fn default() -> Self {
        Self::new()
    }
}

enum Scope {
    Local(VReg),
    Global,
}

pub fn translate_program(context: &mut Context<Vec<Operator>>, program: &parser_defs::Program) {
    let defs = &program.0;
    let mut scope = SheafTable::new();
    for g in defs {
        if let parser_defs::GlobalDecl(i, ..)
        | parser_defs::FunctionDecl(i, ..)
        | parser_defs::FunctionDef(i, ..) = g
        {
            scope.insert(i.as_str().into(), Scope::Global);
        }
    }
    for f in defs {
        match f {
            parser_defs::FunctionDef(..) => {
                scope.push();
                translate_function(context, &mut scope, f, VRegGenerator::new());
                scope.pop();
            }
            parser_defs::FunctionDecl(..) => {}
            parser_defs::GlobalDecl(ident, t) => {
                debug_assert!(!context.globals.contains_key(ident));
                context.globals.insert(ident.clone(), *t);
            }
            parser_defs::LocalVarDef(..) => panic!("local var def at global scope"),
        }
    }
}

fn translate_function(
    context: &mut Context<Vec<Operator>>,
    scope: &mut SheafTable<Rc<str>, Scope>,
    function: &parser_defs::Defs,
    mut gen: VRegGenerator,
) {
    if let parser_defs::FunctionDef(i, p, _, b) = function {
        let code = Vec::new();
        let mut params = Vec::new();
        for p in p.0.iter() {
            let reg = gen.next_reg();
            scope.insert(p.0.as_str().into(), Scope::Local(reg));
            params.push(reg);
        }
        context.functions.insert(
            i.clone(),
            Function {
                body: code,
                params,
                max_reg: 0,
            },
        );
        let code = context.functions.get_mut(i).unwrap().get_body_mut();
        translate_block(code, scope, Any::B(b), &mut gen);
        #[cfg(feature = "print-linear")]
        {
            println!("translated function {i}:\n{}", Displayable(&code));
        }
        context.functions.get_mut(i).unwrap().max_reg = gen.next_reg();
    } else {
        panic!("Expected function def")
    }
}

fn translate_block(
    vec: &mut Vec<Operator>,
    scope: &mut SheafTable<Rc<str>, Scope>,
    instr: parser_defs::Any<'_>,
    gen: &mut VRegGenerator,
) -> Option<VReg> {
    match instr {
        Any::E(e) => match e {
            parser_defs::Expr::Or(l, r) => {
                let left = translate_block(vec, scope, Any::E(l), gen).unwrap();
                let right = translate_block(vec, scope, Any::T(r), gen).unwrap();
                let next = gen.next_reg();
                vec.push(Operator::Or(next, left, right));
                Some(next)
            }
            parser_defs::Expr::Xor(l, r) => {
                let left = translate_block(vec, scope, Any::E(l), gen).unwrap();
                let right = translate_block(vec, scope, Any::T(r), gen).unwrap();
                let next = gen.next_reg();
                vec.push(Operator::Xor(next, left, right));
                Some(next)
            }
            parser_defs::Expr::ETerm(i) => translate_block(vec, scope, Any::T(i), gen),
        },
        Any::S(s) => match s {
            parser_defs::Statement::Def(d) => translate_block(vec, scope, Any::D(d), gen),
            parser_defs::Statement::Assign(i, v) => {
                let value = translate_block(vec, scope, Any::E(v), gen).unwrap();
                if let Some(Scope::Local(reg)) = scope.get(i.as_str()) {
                    vec.push(Operator::Mv(*reg, value));
                } else if let Some((i, Scope::Global)) = scope.get_key_value(i.as_str()) {
                    let intermediate = gen.next_reg();
                    vec.push(Operator::Li(intermediate, 0));
                    let intermediate2 = gen.next_reg();
                    vec.push(Operator::La(intermediate2, Rc::clone(i)));
                    vec.push(Operator::Store(value, intermediate, intermediate2));
                } else {
                    panic!("Did not find variable")
                }
                None
            }
            parser_defs::Statement::IfElse(c, t, e) => {
                let c = translate_block(vec, scope, Any::E(c), gen).unwrap();
                let l1: Rc<_> = gen.next_label().into();
                let l2: Rc<_> = gen.next_label().into();
                let l3: Rc<_> = gen.next_label().into();
                let next = gen.next_reg();
                vec.push(Operator::Li(next, 1));
                vec.push(Operator::Beq(c, next, Rc::clone(&l1), Rc::clone(&l2)));
                vec.push(Operator::Label(Rc::clone(&l1)));
                translate_block(vec, scope, Any::B(t), gen);
                vec.push(Operator::J(Rc::clone(&l3)));
                vec.push(Operator::Label(Rc::clone(&l2)));
                translate_block(vec, scope, Any::B(e), gen);
                vec.push(Operator::J(Rc::clone(&l3)));
                vec.push(Operator::Label(Rc::clone(&l3)));
                None
            }
            parser_defs::Statement::While(c, b) => {
                let l1: Rc<_> = gen.next_label().into();
                let l2: Rc<_> = gen.next_label().into();
                let l3: Rc<_> = gen.next_label().into();
                let next = gen.next_reg();
                vec.push(Operator::J(Rc::clone(&l1)));
                vec.push(Operator::Label(Rc::clone(&l1)));
                let c = translate_block(vec, scope, Any::E(c), gen).unwrap();
                vec.push(Operator::Li(next, 0));
                vec.push(Operator::Beq(c, next, Rc::clone(&l2), Rc::clone(&l3)));
                vec.push(Operator::Label(Rc::clone(&l3)));
                translate_block(vec, scope, Any::B(b), gen);
                vec.push(Operator::J(Rc::clone(&l1)));
                vec.push(Operator::Label(Rc::clone(&l2)));
                None
            }
            parser_defs::Statement::Return(e) => {
                let l1: Rc<_> = gen.next_label().into();
                let reg = translate_block(vec, scope, Any::E(e), gen).unwrap();
                vec.push(Operator::Return(reg));
                vec.push(Operator::Label(Rc::clone(&l1)));
                None
            }
        },
        Any::D(def) => match def {
            parser_defs::Defs::LocalVarDef(i, _, e) => {
                let res = translate_block(vec, scope, Any::E(e), gen).unwrap();
                scope.insert(i.as_str().into(), Scope::Local(res));
                Some(res)
            }
            parser_defs::Defs::GlobalDecl(..) => panic!("Nested global decl"),
            parser_defs::Defs::FunctionDef(..) => panic!("Nested function def"),
            parser_defs::Defs::FunctionDecl(..) => panic!("Nested function decl"),
        },
        Any::B(parser_defs::Body(v)) => {
            for v in v {
                translate_block(vec, scope, Any::S(v), gen);
            }
            None
        }
        Any::As(_) => panic!("Args handled in call"),
        Any::Ps(_) | Any::P(_) => panic!("Params handled in function def"),
        Any::A(atom) => match atom {
            parser_defs::Atom::Mult(l, r) => {
                let left = translate_block(vec, scope, Any::A(l), gen).unwrap();
                let right = translate_block(vec, scope, Any::U(r), gen).unwrap();
                let next = gen.next_reg();
                vec.push(Operator::Mult(next, left, right));
                Some(next)
            }
            parser_defs::Atom::Div(l, r) => {
                let left = translate_block(vec, scope, Any::A(l), gen).unwrap();
                let right = translate_block(vec, scope, Any::U(r), gen).unwrap();
                let next = gen.next_reg();
                vec.push(Operator::Div(next, left, right));
                Some(next)
            }
            parser_defs::Atom::AUnit(i) => translate_block(vec, scope, Any::U(i), gen),
        },
        Any::U(u) => match u {
            parser_defs::Unit::Identifier(i) => {
                if let Some(Scope::Local(reg)) = scope.get(i.as_str()) {
                    Some(*reg)
                } else if let Some((i, Scope::Global)) = scope.get_key_value(i.as_str()) {
                    let intermediate = gen.next_reg();
                    let intermediate2 = gen.next_reg();
                    vec.push(Operator::Li(intermediate, 0));
                    vec.push(Operator::La(intermediate2, i.clone()));
                    let result = gen.next_reg();
                    vec.push(Operator::Load(result, intermediate, intermediate2));
                    Some(result)
                } else {
                    panic!("Did not find variable")
                }
            }
            parser_defs::Unit::True => {
                let res = gen.next_reg();
                vec.push(Operator::Li(res, 1));
                Some(res)
            }
            parser_defs::Unit::False => {
                let res = gen.next_reg();
                vec.push(Operator::Li(res, 0));
                Some(res)
            }
            parser_defs::Unit::Call(f, parser_defs::Args(args)) => {
                let mut regs = Vec::with_capacity(args.capacity());
                for a in args {
                    regs.push(translate_block(vec, scope, Any::E(a), gen).unwrap());
                }
                let result = gen.next_reg();
                let (name, _) = scope.get_key_value(f.as_str()).unwrap();
                let l1 = gen.next_label().into();
                let l2 = gen.next_label().into();
                vec.push(Operator::J(Rc::clone(&l1)));
                vec.push(Operator::Label(Rc::clone(&l1)));
                vec.push(Operator::Call(result, Rc::clone(name), regs));
                vec.push(Operator::J(Rc::clone(&l2)));
                vec.push(Operator::Label(Rc::clone(&l2)));
                Some(result)
            }
            parser_defs::Unit::Grouping(e) => translate_block(vec, scope, Any::E(e), gen),
            parser_defs::Unit::Number(c) => {
                let res = gen.next_reg();
                vec.push(Operator::Li(res, *c));
                Some(res)
            }
        },
        Any::F(f) => match f {
            parser_defs::Factor::Plus(l, r) => {
                let left = translate_block(vec, scope, Any::F(l), gen).unwrap();
                let right = translate_block(vec, scope, Any::A(r), gen).unwrap();
                let next = gen.next_reg();
                vec.push(Operator::Add(next, left, right));
                Some(next)
            }
            parser_defs::Factor::Minus(l, r) => {
                let left = translate_block(vec, scope, Any::F(l), gen).unwrap();
                let right = translate_block(vec, scope, Any::A(r), gen).unwrap();
                let next = gen.next_reg();
                vec.push(Operator::Sub(next, left, right));
                Some(next)
            }
            parser_defs::Factor::FAtom(i) => translate_block(vec, scope, Any::A(i), gen),
        },
        Any::BT(bt) => match bt {
            parser_defs::BTerm::Not(i) => {
                let expr = translate_block(vec, scope, Any::CT(i), gen).unwrap();
                let intermediate = gen.next_reg();
                vec.push(Operator::Li(intermediate, -1));
                let result = gen.next_reg();
                vec.push(Operator::Xor(result, intermediate, expr));
                Some(result)
            }
            parser_defs::BTerm::BCTerm(i) => translate_block(vec, scope, Any::CT(i), gen),
        },
        Any::CT(ct) => match ct {
            parser_defs::CTerm::GEq(l, r) => {
                let left = translate_block(vec, scope, Any::CT(l), gen).unwrap();
                let right = translate_block(vec, scope, Any::F(r), gen).unwrap();
                let next = gen.next_reg();
                let next2 = gen.next_reg();
                let next3 = gen.next_reg();
                vec.push(Operator::Slt(next, left, right));
                vec.push(Operator::Li(next2, 1));
                vec.push(Operator::Xor(next3, next2, next));
                Some(next3)
            }
            parser_defs::CTerm::LT(l, r) => {
                let left = translate_block(vec, scope, Any::CT(l), gen).unwrap();
                let right = translate_block(vec, scope, Any::F(r), gen).unwrap();
                let next = gen.next_reg();
                vec.push(Operator::Slt(next, left, right));
                Some(next)
            }
            parser_defs::CTerm::EQ(l, r) => {
                let left = translate_block(vec, scope, Any::CT(l), gen).unwrap();
                let right = translate_block(vec, scope, Any::F(r), gen).unwrap();
                let next = gen.next_reg();
                let next2 = gen.next_reg();
                let next3 = gen.next_reg();
                vec.push(Operator::Xor(next, left, right));
                vec.push(Operator::Li(next2, 1));
                vec.push(Operator::Slt(next3, next, next2));
                Some(next3)
            }
            parser_defs::CTerm::CTFactor(i) => translate_block(vec, scope, Any::F(i), gen),
        },
        Any::T(t) => match t {
            parser_defs::Term::And(l, r) => {
                let left = translate_block(vec, scope, Any::T(l), gen).unwrap();
                let right = translate_block(vec, scope, Any::BT(r), gen).unwrap();
                let next = gen.next_reg();
                vec.push(Operator::And(next, left, right));
                Some(next)
            }
            parser_defs::Term::TCTerm(i) => translate_block(vec, scope, Any::BT(i), gen),
        },
        _ => panic!("Unexpected instr"),
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SSAOperator {
    IROp(Operator),
    Phi(VReg, Vec<VReg>),
}

impl SSAOperator {
    pub fn receiver(&self) -> Option<VReg> {
        match self {
            SSAOperator::IROp(op) => op.receiver(),
            SSAOperator::Phi(x, _) => Some(*x),
        }
    }
    pub fn replace_reg(&mut self, reg: VReg, with: VReg) {
        match self {
            SSAOperator::IROp(o) => o.replace_reg(reg, with),
            SSAOperator::Phi(_, vec) => {
                for r in vec {
                    if *r == reg {
                        *r = with;
                    }
                }
            }
        }
    }
}
#[derive(Debug)]
pub struct Block<O> {
    pub label: Rc<str>,
    pub body: Vec<O>,
    pub preds: Vec<usize>,
    pub children: Vec<usize>,
    /// Idom of dominator tree.
    /// None if entry node, else Some.
    pub idom: Option<usize>,
    pub idom_of: Vec<usize>,
    pub r_idom: Option<usize>,
    pub r_idom_of: Vec<usize>,
}

impl<O> Block<O> {
    fn empty(label: &Rc<str>) -> Self {
        Self {
            label: Rc::clone(label),
            body: Vec::new(),
            preds: Vec::new(),
            children: Vec::new(),
            idom: None,
            idom_of: Vec::new(),
            r_idom: None,
            r_idom_of: Vec::new(),
        }
    }
    pub fn into_other<T>(self, body: Vec<T>) -> Block<T> {
        Block {
            label: self.label,
            body,
            preds: self.preds,
            children: self.children,
            idom: self.idom,
            idom_of: self.idom_of,
            r_idom: self.r_idom,
            r_idom_of: self.r_idom_of,
        }
    }
}
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct CFG<O> {
    pub blocks: Vec<Block<O>>,
    entry: usize,
    exit: usize,
    max_reg: VReg,
    allocated_ars: u64,
}
impl<O> CFG<O> {
    pub fn get_allocated_ars_mut(&mut self) -> &mut u64 {
        &mut self.allocated_ars
    }
    pub fn into_other<A>(self, blocks: Vec<Block<A>>) -> CFG<A> {
        CFG {
            blocks,
            entry: self.entry,
            exit: self.exit,
            max_reg: self.max_reg,
            allocated_ars: 0,
        }
    }
    pub fn get_blocks(&self) -> &[Block<O>] {
        &self.blocks
    }
    pub fn get_blocks_mut(&mut self) -> &mut [Block<O>] {
        &mut self.blocks
    }
    pub fn get_entry(&self) -> usize {
        self.entry
    }
    pub fn get_exit(&self) -> usize {
        self.exit
    }
    pub fn get_max_reg(&self) -> VReg {
        self.max_reg
    }
    pub fn set_max_reg(&mut self, max_reg: VReg) {
        self.max_reg = max_reg;
    }
    pub fn get_block(&self, i: usize) -> &Block<O> {
        self.blocks.get(i).unwrap()
    }
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
    pub fn len(&self) -> usize {
        self.blocks.len()
    }
    pub fn get_block_mut(&mut self, i: usize) -> &mut Block<O> {
        self.blocks.get_mut(i).unwrap()
    }
    /// Returns the indices of a reverse post-order walk on the dominator tree.
    /// Last() constitutes first element;
    pub fn get_dom_rpo(&self) -> Vec<usize> {
        fn postorder<O>(s: &CFG<O>, current: usize, acc: &mut Vec<usize>) {
            let block = &s.blocks[current];
            for &child in &block.idom_of {
                postorder(s, child, acc);
            }
            acc.push(current);
        }
        let mut res = Vec::with_capacity(self.len());
        postorder(self, 0, &mut res);
        res
    }
    fn bfs_reverse(
        &self,
        total_blocks: usize,
        block: usize,
        selector: fn(&Block<O>) -> &Vec<usize>,
    ) -> Vec<usize> {
        let mut outputs = vec![usize::MAX; total_blocks];
        outputs[block] = 0;
        let block = &self.blocks[block];
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
        queue.extend(selector(block).iter().cloned().zip(std::iter::repeat(1)));

        while let Some((pred, dist)) = queue.pop_front() {
            let old = outputs[pred];
            if dist < old {
                outputs[pred] = dist;
                let block = &self.blocks[pred];
                queue.extend(
                    selector(block)
                        .iter()
                        .cloned()
                        .zip(std::iter::repeat(dist + 1)),
                );
            }
        }
        outputs
    }
    fn apsp_reverse(&self, postdominance: bool) -> Vec<Vec<usize>> {
        let selector: fn(&Block<O>) -> &Vec<usize> = if !postdominance {
            |block| &block.preds
        } else {
            |block| &block.children
        };
        let block_len = self.blocks.len();
        let inner = vec![usize::MAX; block_len];
        let mut output: Vec<Vec<usize>> = Vec::from_iter(std::iter::repeat(inner).take(block_len));
        for (i, val) in output.iter_mut().enumerate() {
            *val = self.bfs_reverse(block_len, i, selector);
        }
        output
    }
    /// calculates dominance frontiers for each block.
    /// Must be a valid CFG as constructed by 'from_linear'.
    pub fn get_dominance_frontiers(&self) -> Vec<HashSet<usize>> {
        let blocks = self.blocks.len();
        let mut frontier = vec![HashSet::new(); blocks];
        for (i, block) in self.blocks.iter().enumerate() {
            for &pred in &block.preds {
                if let Some(idom) = block.idom {
                    let mut current = pred;
                    while idom != current {
                        frontier[current].insert(i);
                        if let Some(next) = self.blocks[current].idom {
                            current = next;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        frontier
    }
    pub fn get_postdominance_frontiers(&self) -> Vec<HashSet<usize>> {
        let blocks = self.blocks.len();
        let mut frontier = vec![HashSet::new(); blocks];
        for (i, block) in self.blocks.iter().enumerate() {
            for &child in &block.children {
                if let Some(r_idom) = block.r_idom {
                    let mut current = child;
                    while r_idom != current {
                        frontier[current].insert(i);
                        if let Some(next) = self.blocks[current].r_idom {
                            current = next;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        frontier
    }
    pub fn calculate_idoms(&self, postdominance: bool) -> Vec<Option<usize>> {
        let mut doms = vec![HashSet::from_iter(0..self.blocks.len()); self.blocks.len()];
        let mut idoms = vec![None; self.blocks.len()];
        let root = if !postdominance {
            self.get_entry()
        } else {
            self.get_exit()
        };
        doms[root] = HashSet::from([root]);
        let mut changed = true;

        let selector: fn(&Block<O>) -> &Vec<usize> = if !postdominance {
            |block: &Block<O>| &block.preds
        } else {
            |block: &Block<O>| &block.children
        };

        while changed {
            changed = false;

            for (i, block) in self.blocks.iter().enumerate() {
                let mut new: HashSet<_> = if let Some(fst) = selector(block).first() {
                    selector(block)
                        .iter()
                        .map(|n| &doms[*n])
                        .fold(doms[*fst].clone(), |acc, next| {
                            acc.intersection(next).cloned().collect()
                        })
                } else {
                    HashSet::new()
                };
                new.insert(i);
                if new != doms[i] {
                    changed = true;
                    doms[i] = new;
                }
            }
        }

        let apsp_reverse = self.apsp_reverse(postdominance);

        for (i, mut set) in doms.into_iter().enumerate() {
            set.remove(&i);
            let idom = set.into_iter().min_by_key(|&v| apsp_reverse[i][v]);
            idoms[i] = idom;
        }
        idoms
    }
}
impl<O> CFG<O>
where
    O: Display,
{
    pub fn to_dot(&self) -> String {
        let mut adjacencies = String::new();
        let mut attributes = String::new();
        let mut dominance = String::new();
        let mut postdominance = String::new();
        for (i, block) in self.blocks.iter().enumerate() {
            adjacencies.extend(block.children.iter().map(|j| format!("{i}->{j}\n")));
            attributes.push_str(&format!(
                "{i}[label=\"{{{0}|{1}}}\"]\ndom{i}[label=\"{0}\"]\npdom{i}[label=\"{0}\"]\n",
                block.label,
                Displayable(&block.body)
                    .to_string()
                    .chars()
                    .filter(|&c| c != '\t')
                    .flat_map(|c| match c {
                        '\n' => "\\n".chars().collect(),
                        '<' => "\\<".chars().collect(),
                        c => vec![c],
                    })
                    .collect::<String>()
            ));
            if let Some(idom) = block.idom {
                dominance.push_str(&format!("dom{}->dom{i}\n", idom));
            }
            if let Some(idom) = block.r_idom {
                postdominance.push_str(&format!("pdom{}->pdom{i}\n", idom))
            }
        }
        format!(
            "
digraph G {{
node [shape=record]

{adjacencies}

subgraph cluster_dominance {{
label=\"dom tree\"
{dominance}
}}

subgraph cluster_postdominance {{
label=\"postdom tree\"
{postdominance}
}}

{attributes}}}"
        )
    }
}
impl CFG<Operator> {
    pub fn from_linear(
        code: impl AsRef<[Operator]>,
        params: impl AsRef<[VReg]>,
        max_reg: VReg,
    ) -> Self {
        let code = code.as_ref();
        let mut labels: HashMap<Rc<str>, usize> = HashMap::new();
        let mut blocks = Vec::from([Block::empty(&Rc::from("ENTRY"))]);
        let mut i = blocks.len();

        for op in code.iter() {
            if let Operator::Label(s) = op {
                labels.insert(Rc::clone(s), i);
                i += 1;
                blocks.push(Block::empty(s));
            }
        }
        blocks.push(Block::empty(&Rc::from("EXIT")));
        let exit_idx = blocks.len() - 1;
        let mut start = 0;
        let mut block_idx = 0;
        for (i, op) in code.iter().enumerate() {
            match op {
                Operator::Label(_) => {
                    let block = &mut blocks[block_idx];
                    block.body = Vec::from(&code[start..i]);
                    if block.children.is_empty() {
                        block.children.push(exit_idx); //exit
                        blocks[exit_idx].preds.push(block_idx);
                    }
                    block_idx += 1;
                    start = i + 1;
                }
                Operator::J(s) => {
                    let block = &mut blocks[block_idx];
                    let target = *labels.get(s.as_ref()).unwrap();
                    if !block.children.contains(&target) {
                        block.children.push(target);
                        blocks[target].preds.push(block_idx);
                    }
                }
                Operator::Beq(_, _, t, f)
                | Operator::Bl(_, _, t, f)
                | Operator::Bgt(_, _, t, f) => {
                    for s in [t, f] {
                        let block = &mut blocks[block_idx];
                        let target = *labels.get(s.as_ref()).unwrap();
                        if !block.children.contains(&target) {
                            block.children.push(target);
                            blocks[target].preds.push(block_idx);
                        }
                    }
                }
                _ => {}
            }
            blocks[block_idx].body = Vec::from(&code[start..]);
        }
        blocks[0].body = params
            .as_ref()
            .iter()
            .map(|&vr| Operator::GetParameter(vr, vr as u64))
            .chain(std::mem::take(&mut blocks[0].body).into_iter())
            .collect();

        let mut gc_map = vec![usize::MAX; blocks.len()];
        let exit = blocks.len() - 1;
        let mut blocks: Vec<_> = blocks.into_iter().enumerate().collect();
        let mut pruned = HashSet::new();
        let mut changed = true;
        while changed {
            changed = false;
            blocks.retain(|(old, block)| {
                if (block.preds.is_empty() || block.preds.iter().all(|idx| pruned.contains(idx)))
                    && *old != 0
                    && *old != exit
                {
                    changed = true;
                    pruned.insert(*old);
                    false
                } else {
                    true
                }
            });
        }
        let mut blocks: Vec<Block<_>> = blocks
            .into_iter()
            .enumerate()
            .map(|(new, (old, mut b))| {
                gc_map[old] = new;
                debug_assert_ne!(new, usize::MAX);
                b.preds.retain(|pred| !pruned.contains(pred));
                b
            })
            .collect();
        for block in &mut blocks {
            block.preds = std::mem::take(&mut block.preds)
                .into_iter()
                .map(|i| gc_map[i])
                .collect();
            block.children = std::mem::take(&mut block.children)
                .into_iter()
                .map(|i| gc_map[i])
                .collect();
        }
        let exit = blocks.len() - 1;
        debug_assert_eq!(blocks[exit].label, Rc::from("EXIT"));
        let mut result = Self {
            blocks,
            entry: 0,
            exit,
            max_reg,
            allocated_ars: 0,
        };
        result.split_critical();
        let idoms = result.calculate_idoms(false);
        let r_idoms = result.calculate_idoms(true);
        for (i, k) in idoms.into_iter().enumerate() {
            if let Some(_k) = k {
                result.blocks[i].idom = k;
                result.blocks[_k].idom_of.push(i);
            }
        }
        for (i, k) in r_idoms.into_iter().enumerate() {
            if let Some(_k) = k {
                result.blocks[i].r_idom = k;
                result.blocks[_k].r_idom_of.push(i);
            }
        }

        #[cfg(feature = "print-cfgs")]
        {
            println!("CFG <from-linear>:");
            println!("{}", result.to_dot());
        }
        result
    }
    fn split_critical(&mut self) {
        let mut gen = VRegGenerator::with_prefix("_CRITICAL_".into());
        let mut to_append = vec![];
        let mut current_idx = self.blocks.len();
        for i in 0..self.blocks.len() {
            if self.blocks[i].children.len() <= 1 {
                continue;
            }
            let block = &self.blocks[i];
            for (k, child_) in block.children.clone().into_iter().enumerate() {
                if self.blocks[child_].preds.len() > 1 {
                    let label: Rc<str> = gen.next_label().into();
                    let child_label = self.blocks[child_].label.clone();
                    let block = &mut self.blocks[i];
                    if let Some(
                        Operator::Bgt(_, _, s1, s2)
                        | Operator::Bl(_, _, s1, s2)
                        | Operator::Beq(_, _, s1, s2),
                    ) = block.body.last_mut()
                    {
                        if *s1 == child_label {
                            *s1 = Rc::clone(&label);
                        } else if *s2 == child_label {
                            *s2 = Rc::clone(&label);
                        }
                    }

                    let mut new = Block::empty(&label);
                    let child = &mut self.blocks[child_];
                    new.children = vec![child_];
                    new.preds = vec![i];
                    new.body = vec![Operator::J(Rc::clone(&child.label))];
                    to_append.push(new);
                    child.preds.iter_mut().for_each(|pred| {
                        if *pred == i {
                            *pred = current_idx;
                        }
                    });
                    self.blocks[i].children[k] = current_idx;
                    current_idx += 1;
                }
            }
        }
        self.blocks.append(&mut to_append);
    }
    fn rename_blocks(
        &mut self,
        block: usize,
        globals: &Vec<VReg>,
        names: &mut HashMap<VReg, Vec<VReg>>,
        generator: &mut VRegGenerator,
        phis: &mut Vec<Vec<SSAOperator>>,
    ) {
        let current = block;
        for &global in globals.iter() {
            let last = names.entry(global).or_default();
            let next = last.last().cloned().unwrap_or(u32::MAX); // undefined
            last.push(next);
        }
        for op in &mut phis[current] {
            if let SSAOperator::Phi(rec, _) = op {
                let old = *rec;
                *(names.get_mut(&old).unwrap().last_mut().unwrap()) = generator.next_reg();
                *rec = *names[&old].last().unwrap();
            }
        }
        let mut locals = HashMap::new();
        macro_rules! update_name {
            ($name:expr) => {
                #[allow(clippy::unnecessary_mut_passed)]
                if globals.contains($name) {
                    let old = *$name;
                    *$name = *names[&old].last().unwrap();
                } else if let Some(&v) = locals.get($name) {
                    *$name = v;
                }
            };
        }
        macro_rules! set_name {
            ($name:expr) => {
                if globals.contains($name) {
                    let old = *$name;
                    *(names.get_mut(&old).unwrap().last_mut().unwrap()) = generator.next_reg();
                    *$name = *names[&old].last().unwrap();
                } else {
                    let old = *$name;
                    *$name = generator.next_reg();
                    locals.insert(old, *$name);
                }
            };
        }
        for op in &mut self.blocks[current].body {
            match op {
                Operator::Add(x, y, z)
                | Operator::Sub(x, y, z)
                | Operator::Mult(x, y, z)
                | Operator::Div(x, y, z)
                | Operator::Xor(x, y, z)
                | Operator::Slt(x, y, z)
                | Operator::Load(x, y, z)
                | Operator::And(x, y, z)
                | Operator::Or(x, y, z) => {
                    update_name!(y);
                    update_name!(z);
                    set_name!(x);
                }
                Operator::Store(x, y, z) => {
                    update_name!(x);
                    update_name!(y);
                    update_name!(z);
                }
                Operator::LoadLocal(x, _)
                | Operator::Li(x, _)
                | Operator::La(x, _)
                | Operator::GetParameter(x, _) => {
                    set_name!(x);
                }
                Operator::Mv(x, y) => {
                    update_name!(y);
                    set_name!(x);
                }
                Operator::Bgt(x, y, _, _)
                | Operator::Bl(x, y, _, _)
                | Operator::Beq(x, y, _, _) => {
                    update_name!(y);
                    update_name!(x);
                }
                Operator::Call(x, _, z) => {
                    for name in z {
                        update_name!(name);
                    }
                    set_name!(x);
                }
                Operator::StoreLocal(x, _) | Operator::Return(x) => {
                    update_name!(x);
                }
                _ => {}
            }
        }
        std::mem::take(&mut locals);
        for &child in self.blocks[current].children.iter() {
            for phi in &mut phis[child] {
                if let SSAOperator::Phi(_, args) = phi {
                    let selfpos = self.blocks[child]
                        .preds
                        .iter()
                        .position(|&v| v == current)
                        .unwrap();

                    update_name!(&mut args[selfpos]);
                }
            }
        }
        for idomsucc in self.blocks[current].idom_of.clone() {
            self.rename_blocks(idomsucc, globals, names, generator, phis);
        }
        for global in globals.iter() {
            names.get_mut(global).unwrap().pop();
        }
    }
    /// to_ssa transforms the CFG into SSA form.
    pub fn into_ssa(mut self) -> CFG<SSAOperator> {
        let frontiers = self.get_dominance_frontiers();
        let (globals, defined_in) = self.get_global_regs();
        let mut phis = vec![vec![]; self.blocks.len()];
        //phase 1: generate phi functions
        for &global in globals.iter() {
            let mut queue: Vec<_> = defined_in[&global].clone();
            while let Some(next) = queue.pop() {
                for &succ in &frontiers[next] {
                    let entry =
                        SSAOperator::Phi(global, vec![global; self.blocks[succ].preds.len()]);
                    if !phis[succ].contains(&entry) {
                        phis[succ].push(entry);
                        queue.push(succ);
                    }
                }
            }
        }
        let mut generator = VRegGenerator::starting_at_reg(self.max_reg + 1);
        let mut names: HashMap<VReg, Vec<VReg>> = HashMap::new();
        self.rename_blocks(0, &globals, &mut names, &mut generator, &mut phis);
        let new_blocks: Vec<Block<SSAOperator>> = phis
            .into_iter()
            .zip(self.blocks.into_iter())
            .map(|(mut vec, mut block)| {
                let old = std::mem::take(&mut block.body);
                let mut modified = old
                    .into_iter()
                    .map(SSAOperator::IROp)
                    .collect::<Vec<SSAOperator>>();
                vec.append(&mut modified);
                block.into_other(vec)
            })
            .collect();
        let result = CFG {
            blocks: new_blocks,
            entry: self.entry,
            exit: self.exit,
            max_reg: generator.next_reg(),
            allocated_ars: 0,
        };
        let _ = result;
        #[cfg(feature = "print-cfgs")]
        {
            println!("CFG <to-ssa>:");
            println!("{}", result.to_dot());
        }
        result
    }

    /// gets global regs and the blocks that they are used in.
    fn get_global_regs(&self) -> (Vec<VReg>, HashMap<VReg, Vec<usize>>) {
        let mut exposed = HashSet::new();
        let mut defined_in: HashMap<VReg, HashSet<usize>> = HashMap::new();
        for (i, block) in self.blocks.iter().enumerate() {
            let mut killed = HashSet::new();
            for op in &block.body {
                macro_rules! test_and_insert {
                    ($reg:expr) => {
                        if !killed.contains($reg) {
                            exposed.insert(*$reg);
                        }
                    };
                }
                match op {
                    Operator::Add(x, y, z)
                    | Operator::Slt(x, y, z)
                    | Operator::Sub(x, y, z)
                    | Operator::Mult(x, y, z)
                    | Operator::Div(x, y, z)
                    | Operator::And(x, y, z)
                    | Operator::Or(x, y, z)
                    | Operator::Xor(x, y, z)
                    | Operator::Load(x, y, z)
                    | Operator::Store(x, y, z) => {
                        test_and_insert!(y);
                        test_and_insert!(z);
                        killed.insert(x);
                        defined_in.entry(*x).or_default().insert(i);
                    }
                    Operator::LoadLocal(x, _)
                    | Operator::GetParameter(x, ..)
                    | Operator::La(x, ..)
                    | Operator::Li(x, ..) => {
                        killed.insert(x);
                        defined_in.entry(*x).or_default().insert(i);
                    }
                    Operator::Mv(x, y) => {
                        test_and_insert!(y);
                        killed.insert(x);
                        defined_in.entry(*x).or_default().insert(i);
                    }
                    Operator::Call(x, _, args) => {
                        for vr in args {
                            test_and_insert!(vr);
                        }
                        killed.insert(x);
                        defined_in.entry(*x).or_default().insert(i);
                    }
                    Operator::Bgt(y, z, _, _)
                    | Operator::Bl(y, z, _, _)
                    | Operator::Beq(y, z, _, _) => {
                        test_and_insert!(y);
                        test_and_insert!(z);
                    }
                    Operator::StoreLocal(y, _) | Operator::Return(y) => {
                        test_and_insert!(y);
                    }
                    _ => {}
                }
            }
        }
        (
            exposed.into_iter().collect(),
            defined_in
                .into_iter()
                .map(|(k, v)| (k, v.into_iter().collect()))
                .collect(),
        )
    }
    pub fn live_out(&self) -> Vec<HashSet<VReg>> {
        let mut live_ins = vec![HashSet::new(); self.len()];
        let mut live_outs = vec![HashSet::new(); self.len()];
        let mut defs = vec![HashSet::new(); self.len()];

        for (i, block) in self.get_blocks().iter().enumerate() {
            for op in block.body.iter() {
                macro_rules! insert_deps {
                    ($($x:expr),+) => {
                        {
                            $(if !defs[i].contains($x) {
                                live_ins[i].insert(*$x);
                            })+
                        }
                    };
                }
                match op {
                    crate::ir::Operator::Add(_, y, z)
                    | crate::ir::Operator::Sub(_, y, z)
                    | crate::ir::Operator::Mult(_, y, z)
                    | crate::ir::Operator::Slt(_, y, z)
                    | crate::ir::Operator::Div(_, y, z)
                    | crate::ir::Operator::Or(_, y, z)
                    | crate::ir::Operator::Xor(_, y, z)
                    | crate::ir::Operator::Load(_, y, z)
                    | crate::ir::Operator::And(_, y, z) => insert_deps!(y, z),
                    crate::ir::Operator::Store(x, y, z) => insert_deps!(x, y, z),
                    crate::ir::Operator::StoreLocal(y, _) | crate::ir::Operator::Mv(_, y) => {
                        insert_deps!(y)
                    }
                    crate::ir::Operator::Bl(x, y, ..)
                    | crate::ir::Operator::Beq(x, y, ..)
                    | crate::ir::Operator::Bgt(x, y, ..) => insert_deps!(x, y),
                    crate::ir::Operator::Call(.., z) => {
                        for rc in z {
                            insert_deps!(rc);
                        }
                    }
                    crate::ir::Operator::Return(x) => insert_deps!(x),
                    _ => {}
                }
                if let Some(rec) = op.receiver() {
                    defs[i].insert(rec);
                }
            }
        }

        let mut changed = true;

        while changed {
            changed = false;
            for i in 0..self.len() {
                let succs = self.get_block(i).children.iter();
                let mut new = HashSet::new();
                for succ in succs {
                    new.extend(live_ins[*succ].iter());
                }
                if new != live_outs[i] {
                    changed = true;
                    live_outs[i] = new;
                }

                let old = std::mem::take(&mut live_ins[i]);
                let mut new = old.clone();
                new.extend(live_outs[i].clone().difference(&defs[i]));
                if new != old {
                    changed = true;
                }
                live_ins[i] = new;
            }
        }

        live_outs
    }
}

impl CFG<SSAOperator> {
    pub fn live_out(&self) -> Vec<HashSet<VReg>> {
        let ssa = self;
        let mut live_ins = vec![HashSet::new(); ssa.len()];
        let mut live_outs = vec![HashSet::new(); ssa.len()];
        let mut phi_uses = vec![HashSet::new(); ssa.len()];
        let mut defs = vec![HashSet::new(); ssa.len()];

        for (i, block) in ssa.get_blocks().iter().enumerate() {
            for op in block.body.iter() {
                macro_rules! insert_deps {
                    ($($x:expr),+) => {
                        {
                            $(if !defs[i].contains($x) {
                                live_ins[i].insert(*$x);
                            })+
                        }
                    };
                }
                match op {
                    SSAOperator::IROp(op) => match op {
                        crate::ir::Operator::Add(_, y, z)
                        | crate::ir::Operator::Sub(_, y, z)
                        | crate::ir::Operator::Mult(_, y, z)
                        | crate::ir::Operator::Slt(_, y, z)
                        | crate::ir::Operator::Div(_, y, z)
                        | crate::ir::Operator::Or(_, y, z)
                        | crate::ir::Operator::Xor(_, y, z)
                        | crate::ir::Operator::Load(_, y, z)
                        | crate::ir::Operator::And(_, y, z) => insert_deps!(y, z),
                        crate::ir::Operator::Store(x, y, z) => insert_deps!(x, y, z),
                        crate::ir::Operator::StoreLocal(y, _) | crate::ir::Operator::Mv(_, y) => {
                            insert_deps!(y)
                        }
                        crate::ir::Operator::Bl(x, y, ..)
                        | crate::ir::Operator::Beq(x, y, ..)
                        | crate::ir::Operator::Bgt(x, y, ..) => insert_deps!(x, y),
                        crate::ir::Operator::Call(.., z) => {
                            for rc in z {
                                insert_deps!(rc);
                            }
                        }
                        crate::ir::Operator::Return(x) => insert_deps!(x),
                        _ => {}
                    },
                    SSAOperator::Phi(_, vec) => {
                        for (i, reg) in vec.iter().enumerate() {
                            phi_uses[block.preds[i]].insert(*reg);
                        }
                    }
                }
                if let Some(rec) = op.receiver() {
                    defs[i].insert(rec);
                }
            }
        }

        let mut changed = true;

        while changed {
            changed = false;
            for i in 0..ssa.len() {
                let succs = ssa.get_block(i).children.iter();
                let mut new = phi_uses[i].clone();
                for succ in succs {
                    new.extend(live_ins[*succ].iter());
                }
                if new != live_outs[i] {
                    changed = true;
                    live_outs[i] = new;
                }

                let old = std::mem::take(&mut live_ins[i]);
                let mut new = old.clone();
                new.extend(live_outs[i].clone().difference(&defs[i]));
                if new != old {
                    changed = true;
                }
                live_ins[i] = new;
            }
        }

        live_outs
    }
    pub fn ssa_graph(&self) -> Vec<Vec<(&SSAOperator, usize)>> {
        let mut result = vec![Vec::default(); self.max_reg as usize];
        for (i, block) in self.blocks.iter().enumerate() {
            for op in &block.body {
                macro_rules! push_operands {
                    ($($x:expr),+) => {
                        {$(result[*$x as usize].push((op, i));)+}
                    };
                }
                match op {
                    SSAOperator::IROp(op_) => match op_ {
                        Operator::Add(_, x, y) => push_operands!(x, y),
                        Operator::Sub(_, x, y) => push_operands!(x, y),
                        Operator::Mult(_, x, y) => push_operands!(x, y),
                        Operator::Div(_, x, y) => push_operands!(x, y),
                        Operator::And(_, x, y) => push_operands!(x, y),
                        Operator::Or(_, x, y) => push_operands!(x, y),
                        Operator::StoreLocal(x, _) => push_operands!(x),
                        Operator::Mv(_, x) => push_operands!(x),
                        Operator::Xor(_, x, y) => push_operands!(x, y),
                        Operator::Load(_, x, y) => push_operands!(x, y),
                        Operator::Store(z, x, y) => push_operands!(x, y, z),
                        Operator::Bgt(x, y, _, _) => push_operands!(x, y),
                        Operator::Bl(x, y, _, _) => push_operands!(x, y),
                        Operator::Beq(x, y, _, _) => push_operands!(x, y),
                        Operator::Slt(_, x, y) => push_operands!(x, y),
                        Operator::Call(_, _, vec) => {
                            for &o in vec {
                                result[o as usize].push((op, i))
                            }
                        }
                        Operator::Return(x) => push_operands!(x),
                        _ => {}
                    },
                    SSAOperator::Phi(_, v) => {
                        for &o in v.iter().filter(|&&r| r != u32::MAX) {
                            result[o as usize].push((op, i));
                        }
                    }
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod test {
    use proptest::prelude::*;
    use proptest::prop_oneof;
    use proptest::strategy::Strategy;

    use crate::util::SheafTable;

    use super::Operator;
    use super::CFG;
    use std::collections::HashSet;
    use std::collections::VecDeque;
    use std::rc::Rc;

    #[test]
    fn build_cfg_jmp() {
        let l1 = Rc::from("L1");
        let input = vec![
            Operator::Nop,
            Operator::Nop,
            Operator::J(Rc::clone(&l1)),
            Operator::Label(Rc::clone(&l1)),
            Operator::Nop,
            Operator::Nop,
            Operator::Return(2),
        ];
        let output = CFG::from_linear(input, [], 2);
        println!("{output:?}")
    }

    #[test]
    fn build_cfg_branch() {
        let l1 = Rc::from("L1");
        let l2 = Rc::from("L2");
        let l3 = Rc::from("L3");
        let header = Rc::from("Header");
        let input = vec![
            Operator::Nop,
            Operator::J(Rc::clone(&header)),
            Operator::Label(Rc::clone(&header)),
            Operator::Nop,
            Operator::Beq(0, 0, Rc::clone(&l1), Rc::clone(&l2)),
            Operator::Label(Rc::clone(&l1)),
            Operator::Nop,
            Operator::J(Rc::clone(&l3)),
            Operator::Label(Rc::clone(&l2)),
            Operator::Nop,
            Operator::J(Rc::clone(&l3)),
            Operator::Label(Rc::clone(&l3)),
            Operator::Return(2),
        ];
        let output = CFG::from_linear(input, [], 2);
        assert_eq!(output.blocks[0].children.len(), 1);
        assert_eq!(output.blocks[1].children.len(), 2);
        assert_eq!(output.blocks[2].children.len(), 1);
        assert_eq!(output.blocks[3].children.len(), 1);
        assert_eq!(output.blocks[4].children.len(), 0);
        assert_eq!(output.blocks[4].preds.len(), 2);

        assert_eq!(output.blocks[0].idom, None);
        assert_eq!(output.blocks[1].idom.unwrap(), 0);
        assert_eq!(output.blocks[2].idom.unwrap(), 1);
        assert_eq!(output.blocks[3].idom.unwrap(), 1);
        assert_eq!(output.blocks[4].idom.unwrap(), 1);
        println!("{output:?}")
    }

    #[test]
    fn build_cfg_while() {
        let l1 = Rc::from("L1");
        let l2 = Rc::from("L2");
        let l3 = Rc::from("L3");
        let input = vec![
            Operator::Nop,
            Operator::J(Rc::clone(&l1)),
            Operator::Label(Rc::clone(&l1)),
            Operator::Li(33, 0),
            Operator::Beq(34, 33, Rc::clone(&l2), Rc::clone(&l3)),
            Operator::Label(Rc::clone(&l3)),
            Operator::Nop,
            Operator::J(Rc::clone(&l1)),
            Operator::Label(Rc::clone(&l2)),
            Operator::Nop,
        ];
        let output = CFG::from_linear(input, [], 34);
        assert_eq!(output.blocks[0].children.len(), 1);
        assert_eq!(output.blocks[1].children.len(), 2);
        assert_eq!(output.blocks[2].children.len(), 1);
        assert_eq!(output.blocks[3].children.len(), 0);
        assert_eq!(output.blocks[1].preds.len(), 2);

        println!("{output:?}");

        assert_eq!(output.blocks[0].idom, None);
        assert_eq!(output.blocks[1].idom.unwrap(), 0);
        assert_eq!(output.blocks[2].idom.unwrap(), 1);
    }

    proptest! {
        #[test]
        fn test_reachable(cfg in any_with::<CFG<Operator>>((20, 20, 20))) {
            let dot = cfg.to_dot();
            let mut found = HashSet::new();
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            found.insert(cfg.get_exit());
            queue.push_back(0);
            while let Some(nxt) = queue.pop_front() {
                visited.insert(nxt);
                found.insert(nxt);
                queue.extend(cfg.blocks[nxt].children.iter().filter(|child| !visited.contains(child)));
            }
            assert!(found.len() == cfg.blocks.len(), "unreachable block in {dot}\n");
        }

        #[test]
        fn test_correct_labels(cfg in any_with::<CFG<Operator>>((20, 20, 20))) {
            for block in &cfg.blocks {
                if let Some(Operator::Bgt(_, _, s1, s2) | Operator::Bl(_, _, s1, s2) | Operator::Beq(_, _, s1, s2)) = block.body.last() {
                    assert!(block.children.len() <= 2 && !block.children.is_empty(), "{}", cfg.to_dot());
                    for &child in &block.children {
                        let child = &cfg.blocks[child];
                        assert!(child.label == *s1 || child.label == *s2);
                    }
                } else if let Some(Operator::J(s1)) = block.body.last() {
                    assert_eq!(block.children.len(), 1);
                    assert_eq!(&cfg.blocks[block.children[0]].label, s1);
                }
            }
        }

        #[test]
        fn test_ssa_one_definition(cfg in any_with::<CFG<Operator>>((20, 20, 20))) {
            let dot = cfg.to_dot();
            let ssa = cfg.into_ssa();
            let ssa_dot = ssa.to_dot();
            let mut defined = HashSet::new();
            for block in ssa.blocks {
                for op in block.body {
                    match op {
                        crate::ir::SSAOperator::IROp(op) => match op {
                            Operator::Add(x, _, _)  |
                            Operator::Sub(x, _, _)  |
                            Operator::Mult(x, _, _)  |
                            Operator::Div(x, _, _)  |
                            Operator::And(x, _, _)  |
                            Operator::Or(x, _, _)  |
                            Operator::Mv(x, _)  |
                            Operator::Xor(x, _, _)  |
                            Operator::Load(x, _, _)  |
                            Operator::La(x, _)  |
                            Operator::LoadLocal(x, _) |
                            Operator::Li(x, _)  |
                            Operator::Slt(x, _, _)  |
                            Operator::Call(x, _, _)  |
                            Operator::GetParameter(x, _) =>  {
                                assert!(!defined.contains(&x), "original: {dot}\n ssa: {ssa_dot}\n");
                                defined.insert(x);
                            },
                            _ => {}
                        },
                        crate::ir::SSAOperator::Phi(x, _) => {
                            assert!(!defined.contains(&x), "original: {dot}\n ssa: {ssa_dot}\n");
                            defined.insert(x);
                        }
                    }
                }
            }
        }

        #[test]
        fn test_ssa_reaching_definition(cfg in any_with::<CFG<Operator>>((20, 20, 20))) {
            fn check_reaching_recursive(ssa: & CFG<super::SSAOperator>, names: &mut SheafTable<super::VReg, bool>, current: usize) {
                let block = &ssa.blocks[current];
                let ssa_dot = ssa.to_dot();
                names.push();
                macro_rules! check_def {
                    ($name:expr) => {
                        assert!(names.get($name).is_some(), "\nssa: {ssa_dot}\nin block: {}\nchecking name: {}\n", block.label, *$name);
                    }
                }

                macro_rules! create_def {
                    ($name:expr) => {
                        names.insert(*$name, true);
                    }
                }
                for op in &block.body {
                    match op {
                        crate::ir::SSAOperator::IROp(op) => match op {
                            Operator::Add(x, y, z)
                            | Operator::Sub(x, y, z)
                            | Operator::Mult(x, y, z)
                            | Operator::Div(x, y, z)
                            | Operator::Xor(x, y, z)
                            | Operator::Slt(x, y, z)
                            | Operator::Load(x, y, z)
                            | Operator::Store(x, y, z)
                            | Operator::And(x, y, z)
                            | Operator::Or(x, y, z) => {
                                check_def!(y);
                                check_def!(z);
                                create_def!(x);
                            }
                            Operator::LoadLocal(x, _) | Operator::Li(x, _) | Operator::La(x, _) | Operator::GetParameter(x, _) => {
                                create_def!(x);
                            }
                            Operator::Mv(x, y) => {
                                check_def!(y);
                                create_def!(x);
                            }
                            Operator::Bgt(x, y, _, _)
                            | Operator::Bl(x, y, _, _)
                            | Operator::Beq(x, y, _, _) => {
                                check_def!(y);
                                check_def!(x);
                            }
                            Operator::Call(x, _, z) => {
                                for name in z {
                                    check_def!(name);
                                }
                                create_def!(x);
                            }
                            Operator::StoreLocal(x, _) | Operator::Return(x) => {
                                check_def!(x);
                            }
                            _ => {}
                                    },
                        crate::ir::SSAOperator::Phi(x, _) => {
                            create_def!(x);
                        },
                    }
                }
                for &child in &block.idom_of {
                    check_reaching_recursive(ssa, names, child);
                }
                names.pop();
            }
            let ssa = cfg.into_ssa();
            let mut names = SheafTable::new();
            check_reaching_recursive(&ssa, &mut names, 0);
        }
    }

    proptest! {

        #![proptest_config(ProptestConfig {
            .. ProptestConfig::default()
        })]

        #[test]
        fn test_ssa_non_critical_edge(cfg in any_with::<CFG<Operator>>((10, 10, 10))) {
            let ssa = cfg.into_ssa();

            for block in &ssa.blocks {
                if block.children.len() > 1 {
                    for &child in &block.children {
                        let child = &ssa.blocks[child];
                        assert!(child.preds.len() <= 1);
                    }
                }
            }
        }

        // #[test]
        // fn test_ssa_print(cfg in any_with::<CFG<Operator>>((20, 20, 30))) {
        //     println!("\nLinear:\n{}\n", cfg.to_dot());
        //     let ssa = cfg.to_ssa();
        //     println!("\nSSA:\n{}\n", ssa.to_dot());
        // }

    }

    fn arbitrary_stmt(
        reg1: super::VReg,
        reg2: super::VReg,
        reg3: super::VReg,
        label: Rc<str>,
    ) -> impl Strategy<Value = Operator> {
        prop_oneof![
            Just(Operator::Nop),
            Just(Operator::Add(reg1, reg2, reg3)),
            Just(Operator::Sub(reg1, reg2, reg3)),
            Just(Operator::Mult(reg1, reg2, reg3)),
            Just(Operator::Div(reg1, reg2, reg3)),
            Just(Operator::And(reg1, reg2, reg3)),
            Just(Operator::Or(reg1, reg2, reg3)),
            Just(Operator::Xor(reg1, reg2, reg3)),
            Just(Operator::La(reg1, label)),
            Just(Operator::Load(reg1, reg2, reg3)),
            Just(Operator::Slt(reg1, reg2, reg3)),
            Just(Operator::Store(reg1, reg2, reg3)),
            Just(Operator::Load(reg1, reg2, reg3)),
            Just(Operator::Mv(reg1, reg2)),
            any::<i64>().prop_map(move |i| Operator::Li(reg1, i)),
            any::<u64>().prop_map(move |i| Operator::LoadLocal(reg1, i)),
            any::<u64>().prop_map(move |i| Operator::StoreLocal(reg1, i)),
        ]
    }

    fn arbitrary_branch(
        reg1: super::VReg,
        reg2: super::VReg,
        label1: Rc<str>,
        label2: Rc<str>,
    ) -> impl Strategy<Value = Operator> {
        prop_oneof![
            Just(Operator::Beq(
                reg1,
                reg2,
                Rc::clone(&label1),
                Rc::clone(&label2)
            )),
            Just(Operator::Bl(
                reg1,
                reg2,
                Rc::clone(&label1),
                Rc::clone(&label2)
            )),
            Just(Operator::Bgt(
                reg1,
                reg2,
                Rc::clone(&label1),
                Rc::clone(&label2)
            )),
        ]
    }

    fn arbitrary_block(
        len: usize,
        children: Vec<Rc<str>>,
        reg_range: impl Strategy<Value = super::VReg> + Clone + 'static,
    ) -> impl Strategy<Value = Vec<Operator>> {
        let body = proptest::collection::vec(
            (
                reg_range.clone(),
                reg_range.clone(),
                reg_range.clone(),
                "[a-z]*",
            )
                .prop_flat_map(|(reg1, reg2, reg3, lb)| {
                    arbitrary_stmt(reg1, reg2, reg3, Rc::from(&lb[..]))
                }),
            len,
        );
        let body: BoxedStrategy<Vec<Operator>> = if children.len() == 2 {
            body.prop_flat_map(move |v| {
                let b: BoxedStrategy<Operator> = (
                    reg_range.clone(),
                    reg_range.clone(),
                    Just(Rc::clone(&children[0])),
                    Just(Rc::clone(&children[1])),
                )
                    .prop_flat_map(|(reg1, reg2, lb1, lb2)| arbitrary_branch(reg1, reg2, lb1, lb2))
                    .boxed();
                b.prop_map(move |b| v.iter().cloned().chain(std::iter::once(b)).collect())
            })
            .boxed()
        } else if children.len() == 1 {
            body.prop_map(move |v| {
                v.into_iter()
                    .chain(std::iter::once(Operator::J(Rc::clone(&children[0]))))
                    .collect()
            })
            .boxed()
        } else {
            body.boxed()
        };

        body
    }

    fn generate_linear(
        reg_range: u32,
        block_len: usize,
        mut worklist: VecDeque<Rc<str>>,
        potential_blocks: Vec<Rc<str>>,
        done: HashSet<Rc<str>>,
        result: Vec<Vec<Operator>>,
    ) -> BoxedStrategy<Vec<Vec<Operator>>> {
        if let Some(nxt) = worklist.pop_front() {
            let copy_nxt = Rc::clone(&nxt);
            (
                proptest::collection::vec(proptest::sample::select(potential_blocks.clone()), 0..3)
                    .prop_filter("try to branch to self", move |vals| {
                        !vals.contains(&copy_nxt)
                    }),
                Just(worklist),
                Just(potential_blocks),
                Just(done),
                Just(result),
                Just(nxt),
            )
                .prop_flat_map(
                    move |(vals, mut worklist, potential_blocks, mut done, result, nxt)| {
                        for lb in &vals {
                            if !done.contains(lb) {
                                worklist.push_back(Rc::clone(lb));
                                done.insert(Rc::clone(lb));
                            }
                        }
                        let res = (
                            Just(result),
                            arbitrary_block(block_len, vals, 0_u32..reg_range),
                        )
                            .prop_map(move |(result, new)| {
                                result
                                    .into_iter()
                                    .chain(std::iter::once(
                                        std::iter::once(Operator::Label(Rc::clone(&nxt)))
                                            .chain(new.into_iter())
                                            .collect::<Vec<_>>(),
                                    ))
                                    .collect::<Vec<_>>()
                            });
                        (Just(worklist), Just(potential_blocks), Just(done), res).prop_flat_map(
                            move |(worklist, potential_blocks, done, res)| {
                                generate_linear(
                                    reg_range,
                                    block_len,
                                    worklist,
                                    potential_blocks,
                                    done,
                                    res,
                                )
                            },
                        )
                    },
                )
                .boxed()
        } else {
            Just(result).boxed()
        }
    }
    impl Arbitrary for CFG<Operator> {
        type Parameters = (usize, u32, usize);

        fn arbitrary_with((len, reg_range, block_len): Self::Parameters) -> Self::Strategy {
            let potential_blocks: Vec<Rc<str>> = (0..len)
                .map(|n| format!("LABEL_{}", n))
                .map(|s| Rc::from(&s[..]))
                .collect();
            let mut worklist: VecDeque<Rc<str>> = VecDeque::new();
            worklist.push_back(Rc::from("ENTRY"));
            let done: HashSet<Rc<str>> = HashSet::new();

            let linear = generate_linear(
                reg_range,
                block_len,
                worklist,
                potential_blocks,
                done,
                Vec::new(),
            )
            .prop_map(move |result| result.into_iter().flatten().skip(1).collect::<Vec<_>>());

            let cfg = linear.prop_map(move |linear| {
                CFG::from_linear(linear, (0..=reg_range).collect::<Vec<_>>(), reg_range + 1)
            });

            cfg.boxed()
        }

        type Strategy = BoxedStrategy<CFG<Operator>>;
    }
}
