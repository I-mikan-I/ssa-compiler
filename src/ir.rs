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
//! | bgt ad1, ad2, @label | op1 | op2 | NA |
//! | bl ad1, ad2, @label | op1 | op2 | NA |
//! | j @label | NA | NA | NA |
//! | beq ad1, ad2, @label | op1 | op2 | NA |
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
//! (L) >= (R) =>   slt name() lin(R) lin(L) // r_new
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
//! if (C) then (I)
//! else (E) =>     li name() 1 // r_new
//!                 beq lin(C) r_new @l1
//!                 j @l2
//!                 label(): body(I) // @l1
//!                 j @l3
//!                 label(): body(E) // @l2
//!                 j @l3
//!                 label(): ... // @l3
//! while (C)
//! do (B) =>       label():    // @l1
//!                 li name() 0 // r_new
//!                 beq lin(C) r_ne2 @l2
//!                 body(B)
//!                 j @l1
//!                 label(): ... // @l2
//! identifier((arg0), (arg1), ...)
//! =>              call @label(identifier) (lin(arg0), lin(arg1), ...)
//! return (E) =>   return lin(E)```

use parser_defs::Any;
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    rc::Rc,
};

use crate::util::SheafTable;

type VReg = u32;
struct VRegGenerator(u32, u32);
impl VRegGenerator {
    pub fn new() -> Self {
        Self(0, 0)
    }
    pub fn next_reg(&mut self) -> u32 {
        let res = self.0;
        self.0 += 1;
        res
    }
    pub fn next_label(&mut self) -> String {
        let res = format!("_LABEL_{}", self.1);
        self.1 += 1;
        res
    }
}
#[derive(Debug, PartialEq, Eq)]
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
    Bgt(VReg, VReg, Rc<str>),
    Bl(VReg, VReg, Rc<str>),
    J(Rc<str>),
    Beq(VReg, VReg, Rc<str>),
    Li(VReg, i64),
    Slt(VReg, VReg, VReg),
    Call(VReg, Rc<str>, Vec<VReg>),
    Return(VReg),
    Label(Rc<str>),
}

pub struct Displayable<'a>(pub &'a [Operator]);

impl<'a> Display for Displayable<'a> {
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
            Operator::Bgt(rd1, rd2, rd3) => write!(f, "\tbgt rd{rd1}, rd{rd2}, @{rd3}"),
            Operator::Bl(rd1, rd2, rd3) => write!(f, "\tbl rd{rd1}, rd{rd2}, @{rd3}"),
            Operator::J(rd1) => write!(f, "\tj @{rd1}"),
            Operator::Beq(rd1, rd2, rd3) => write!(f, "\tbeq rd{rd1}, rd{rd2}, @{rd3}"),
            Operator::Li(rd1, rd2) => write!(f, "\tli rd{rd1}, #{rd2}"),
            Operator::Slt(rd1, rd2, rd3) => write!(f, "\tslt rd{rd1}, rd{rd2}, rd{rd3}"),
            Operator::Call(rd1, rd2, rd3) => {
                write!(f, "\tcall rd{rd1}, @{rd2}, ( ")?;
                for reg in rd3 {
                    write!(f, "{reg} ")?;
                }
                write!(f, " )")
            }
            Operator::Return(rd1) => write!(f, "\treturn rd{rd1}"),
            Operator::Label(rd1) => write!(f, "@{rd1}:"),
        }
    }
}
pub struct Function {
    body: Vec<Operator>,
    params: Vec<VReg>,
}
impl Function {
    pub fn get_body(&self) -> &Vec<Operator> {
        &self.body
    }
    pub fn get_body_mut(&mut self) -> &mut Vec<Operator> {
        &mut self.body
    }
    pub fn get_params(&self) -> &Vec<VReg> {
        &self.params
    }
}
pub struct Context {
    functions: HashMap<String, Function>,
}
impl Context {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }
    pub fn get_functions(&self) -> &HashMap<String, Function> {
        &self.functions
    }
}

enum Scope {
    Local(VReg),
    Global,
}

pub fn translate_program(context: &mut Context, program: &parser_defs::Program) {
    let defs = &program.0;
    let mut scope = SheafTable::new();
    for g in defs {
        if let parser_defs::VarDef(i, ..) = g {
            scope.insert(i.as_str().into(), Scope::Global);
        }
        if let parser_defs::FunctionDef(i, ..) = g {
            scope.insert(i.as_str().into(), Scope::Global);
        }
    }
    for f in defs {
        if let parser_defs::FunctionDef(..) = f {
            scope.push();
            translate_function(context, &mut scope, f, VRegGenerator::new());
            scope.pop();
        }
    }
}

fn translate_function<'a>(
    context: &mut Context,
    scope: &mut SheafTable<Rc<str>, Scope>,
    function: &'a parser_defs::Defs,
    mut gen: VRegGenerator,
) {
    if let parser_defs::FunctionDef(i, p, t, b) = function {
        let code = Vec::new();
        let mut params = Vec::new();
        for p in p.0.iter() {
            let reg = gen.next_reg();
            scope.insert(p.0.as_str().into(), Scope::Local(reg));
            params.push(reg);
        }
        context
            .functions
            .insert(i.clone(), Function { body: code, params });
        let code = context.functions.get_mut(i).unwrap().get_body_mut();
        translate_block(code, scope, Any::B(b), &mut gen);
    } else {
        panic!("Expected function def")
    }
}

fn translate_block<'a>(
    vec: &mut Vec<Operator>,
    scope: &mut SheafTable<Rc<str>, Scope>,
    instr: parser_defs::Any<'a>,
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
                vec.push(Operator::Beq(c, next, Rc::clone(&l1)));
                vec.push(Operator::J(Rc::clone(&l2)));
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
                let c = translate_block(vec, scope, Any::E(c), gen).unwrap();
                let l1: Rc<_> = gen.next_label().into();
                let l2: Rc<_> = gen.next_label().into();
                let next = gen.next_reg();
                vec.push(Operator::Label(Rc::clone(&l1)));
                vec.push(Operator::Li(next, 0));
                vec.push(Operator::Beq(c, next, Rc::clone(&l2)));
                translate_block(vec, scope, Any::B(b), gen);
                vec.push(Operator::J(Rc::clone(&l1)));
                vec.push(Operator::Label(Rc::clone(&l2)));
                None
            }
            parser_defs::Statement::Return(e) => {
                let reg = translate_block(vec, scope, Any::E(e), gen).unwrap();
                vec.push(Operator::Return(reg));
                None
            }
        },
        Any::D(def) => match def {
            parser_defs::Defs::VarDef(i, t, e) => {
                let res = translate_block(vec, scope, Any::E(e), gen).unwrap();
                scope.insert(i.as_str().into(), Scope::Local(res));
                Some(res)
            }
            parser_defs::Defs::FunctionDef(..) => panic!("Nested function def"),
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
                vec.push(Operator::Call(result, Rc::clone(name), regs));
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
                vec.push(Operator::Slt(next, right, left));
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
