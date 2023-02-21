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
//! =>              call @label(identifier) (lin(arg0), lin(arg1), ...)
//! return (E) =>   return lin(E)```

use parser_defs::Any;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
    rc::Rc,
    result,
};

use crate::ssa::{Linear, SSA};
use crate::util::SheafTable;

type VReg = u32;
struct VRegGenerator(u32, u32);
impl VRegGenerator {
    pub fn starting_at(at: u32) -> Self {
        Self(at, 0)
    }
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
    Nop,
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
                    write!(f, "{arg} ")?;
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
pub struct Context<B> {
    functions: HashMap<String, Function<B>>,
}
impl<B> Context<B> {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
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
                let reg = translate_block(vec, scope, Any::E(e), gen).unwrap();
                vec.push(Operator::Return(reg));
                None
            }
        },
        Any::D(def) => match def {
            parser_defs::Defs::VarDef(i, _, e) => {
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

pub struct IRLinear(Function<Vec<Operator>>);
impl IRLinear {
    pub fn get_mut(&mut self) -> &mut Function<Vec<Operator>> {
        &mut self.0
    }
}
pub struct IRSSA(Function<CFG<SSAOperator>>);
impl SSA for IRSSA {
    type Operation = Operator;

    type LinearReg = VReg;

    type SSAReg = VReg;
}

impl Linear for IRLinear {
    type Operation = Operator;

    type LinearReg = VReg;

    type SSaReg = VReg;

    type AsSSA = IRSSA;

    fn transpile_ssa(&self) -> Self::AsSSA {
        todo!()
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SSAOperator {
    IROp(Operator),
    Phi(VReg, Vec<VReg>),
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
    mark: bool,
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
            mark: false,
        }
    }
    fn into_other<T>(self, body: Vec<T>) -> Block<T> {
        Block {
            label: self.label,
            body: body,
            preds: self.preds,
            children: self.children,
            idom: self.idom,
            idom_of: self.idom_of,
            mark: self.mark,
        }
    }
}
#[derive(Debug)]
pub struct CFG<O> {
    blocks: Vec<Block<O>>,
    entry: usize,
    max_reg: VReg,
}
impl<O> CFG<O>
where
    O: Display,
{
    pub fn to_dot(&self) -> String {
        let mut adjacencies = String::new();
        let mut attributes = String::new();
        let mut dominance = String::new();
        for (i, block) in self.blocks.iter().enumerate() {
            adjacencies.extend(block.children.iter().map(|j| format!("{i}->{j}\n")));
            attributes.push_str(&format!(
                "{i}[label=\"{{{0}|{1}}}\"]\ndom{i}[label=\"{0}\"]\n",
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

{attributes}}}"
        )
    }
}
impl CFG<Operator> {
    pub fn get_entry(&self) -> &Block<Operator> {
        self.blocks.get(self.entry).unwrap()
    }
    pub fn get_block(&self, i: usize) -> &Block<Operator> {
        self.blocks.get(i).unwrap()
    }
    fn bfs_reverse(&self, total_blocks: usize, block: usize) -> Vec<usize> {
        let mut outputs = vec![usize::MAX; total_blocks];
        outputs[block] = 0;
        let block = &self.blocks[block];
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
        queue.extend(block.preds.iter().cloned().zip(std::iter::repeat(1)));

        while let Some((pred, dist)) = queue.pop_front() {
            let old = outputs[pred];
            if dist < old {
                outputs[pred] = dist;
                let block = &self.blocks[pred];
                queue.extend(block.preds.iter().cloned().zip(std::iter::repeat(dist + 1)));
            }
        }
        outputs
    }
    fn apsp_reverse(&self) -> Vec<Vec<usize>> {
        let block_len = self.blocks.len();
        let inner = vec![usize::MAX; block_len];
        let mut output: Vec<Vec<usize>> = Vec::from_iter(std::iter::repeat(inner).take(block_len));
        for (i, val) in output.iter_mut().enumerate() {
            *val = self.bfs_reverse(block_len, i);
        }
        output
    }
    pub fn from_linear(
        code: impl AsRef<[Operator]>,
        params: impl AsRef<[VReg]>,
        max_reg: VReg,
    ) -> Self {
        let code = code.as_ref();
        let mut i = 1;
        let mut labels: HashMap<Rc<str>, usize> = HashMap::new();
        let mut blocks = Vec::from([Block::empty(&Rc::from("ENTRY"))]);

        for op in code.iter() {
            if let Operator::Label(s) = op {
                labels.insert(Rc::clone(s), i);
                i += 1;
                blocks.push(Block::empty(s));
            }
        }
        let mut start = 0;
        let mut block_idx = 0;
        for (i, op) in code.iter().enumerate() {
            match op {
                Operator::Label(_) => {
                    let block = &mut blocks[block_idx];
                    block_idx += 1;
                    block.body = Vec::from(&code[start..i]);
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

        let mut doms = vec![HashSet::from_iter(0..blocks.len()); blocks.len()];
        doms[0] = HashSet::from([0]);

        let mut changed = true;
        while changed {
            changed = false;

            for (i, block) in blocks.iter().enumerate() {
                let mut new: HashSet<_> = if let Some(fst) = block.preds.first() {
                    block
                        .preds
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
        let mut result = Self {
            blocks,
            entry: 0,
            max_reg,
        };
        let apsp_reverse = result.apsp_reverse();

        for (i, mut set) in doms.into_iter().enumerate() {
            set.remove(&i);
            let idom = set.into_iter().min_by_key(|&v| apsp_reverse[i][v]);
            result.blocks[i].idom = idom;
            if let Some(idom) = idom {
                result.blocks[idom].idom_of.push(i);
            }
        }
        #[cfg(feature = "print-cfgs")]
        {
            println!("CFG <from-linear>:");
            println!("{}", result.to_dot());
        }
        result
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
            let next = last.last().cloned().unwrap_or(u32::MAX); // should never be used
            last.push(next);
        }
        for op in &mut phis[current] {
            if let SSAOperator::Phi(rec, _) = op {
                let old = *rec;
                *(names.get_mut(&old).unwrap().last_mut().unwrap()) = generator.next_reg();
                *rec = *names[&old].last().unwrap();
            }
        }
        macro_rules! update_name {
            ($name:expr) => {
                if globals.contains($name) {
                    let old = *$name;
                    *$name = *names[&old].last().unwrap();
                }
            };
        }
        macro_rules! set_name {
            ($name:expr) => {
                if globals.contains($name) {
                    let old = *$name;
                    *(names.get_mut(&old).unwrap().last_mut().unwrap()) = generator.next_reg();
                    *$name = *names[&old].last().unwrap();
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
                | Operator::Store(x, y, z)
                | Operator::And(x, y, z)
                | Operator::Or(x, y, z) => {
                    update_name!(y);
                    update_name!(z);
                    set_name!(x);
                }
                Operator::Li(x, _) | Operator::La(x, _) => {
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
                Operator::Call(x, y, z) => {
                    for name in z {
                        update_name!(name);
                    }
                    set_name!(x);
                }
                Operator::Return(x) | Operator::GetParameter(x, _) => {
                    update_name!(x);
                }
                _ => {}
            }
        }
        for (i, &child) in self.blocks[current].children.iter().enumerate() {
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
    pub fn to_ssa(mut self) -> CFG<SSAOperator> {
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
        let mut generator = VRegGenerator::starting_at(self.max_reg + 1);
        let mut names: HashMap<VReg, Vec<VReg>> = HashMap::new();
        self.rename_blocks(0, &globals, &mut names, &mut generator, &mut phis);
        let new_blocks: Vec<Block<SSAOperator>> = phis
            .into_iter()
            .zip(self.blocks.into_iter())
            .map(|(mut vec, mut block)| {
                let old = std::mem::take(&mut block.body);
                let mut modified = old
                    .into_iter()
                    .map(|op| SSAOperator::IROp(op))
                    .collect::<Vec<SSAOperator>>();
                vec.append(&mut modified);
                block.into_other(vec)
            })
            .collect();
        let result = CFG {
            blocks: new_blocks,
            entry: self.entry,
            max_reg: self.max_reg,
        };
        #[cfg(feature = "print-cfgs")]
        {
            println!("CFG <to-ssa>:");
            println!("{}", result.to_dot());
        }
        result
    }
    /// calculates dominance frontiers for each block.
    /// Must be a valid CFG as constructed by 'from_linear'.
    fn get_dominance_frontiers(&self) -> Vec<HashSet<usize>> {
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
                    Operator::GetParameter(x, ..) | Operator::La(x, ..) | Operator::Li(x, ..) => {
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
                    Operator::Return(y) => {
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
}

#[cfg(test)]
mod test {
    use super::Operator;
    use super::CFG;
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
        let output = CFG::from_linear(input, &[], 2);
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
        let output = CFG::from_linear(input, &[], 2);
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
        let output = CFG::from_linear(input, &[], 34);
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
}

//todo: label could be first instr!
