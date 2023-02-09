use std::{error::Error, fmt::{Debug, Display}};

pub use self::{
    Atom::*, BTerm::*, CTerm::*, Defs::*, Expr::*, Factor::*, Statement::*, Term::*, Type::*,
    Unit::*,
};
pub fn do_nothing() {}
pub struct ParseErr {
    message: Box<dyn Error>,
    span: Option<lrpar::Span>,
}
impl<E> From<E> for ParseErr
where
    E: Into<Box<dyn Error>>,
{
    fn from(v: E) -> Self {
        Self {
            message: v.into(),
            span: None,
        }
    }
}
impl Debug for ParseErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}
impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Debug>::fmt(self, f)
    }
}
pub type Ident = String;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Int,
    Bool,
    Never,
}
#[derive(Debug)]
pub enum Expr {
    Or(Box<Expr>, Term),
    Xor(Box<Expr>, Term),
    ETerm(Term),
}
#[derive(Debug)]
pub enum Term {
    And(Box<Term>, BTerm),
    TCTerm(BTerm),
}
#[derive(Debug)]
pub enum BTerm {
    Not(CTerm),
    BCTerm(CTerm),
}
#[derive(Debug)]
pub enum CTerm {
    GEq(Box<CTerm>, Factor),
    LT(Box<CTerm>, Factor),
    EQ(Box<CTerm>, Factor),
    CTFactor(Factor),
}
#[derive(Debug)]
pub enum Factor {
    Plus(Box<Factor>, Atom),
    Minus(Box<Factor>, Atom),
    FAtom(Atom),
}
#[derive(Debug)]
pub enum Atom {
    Mult(Box<Atom>, Unit),
    Div(Box<Atom>, Unit),
    AUnit(Unit),
}
#[derive(Debug)]
pub enum Unit {
    Identifier(Ident),
    True,
    False,
    Call(Ident, Args),
    Grouping(Box<Expr>),
    Number(i64),
}

#[derive(Debug)]
pub struct Parameter(pub Ident, pub Type);
#[derive(Debug)]
pub struct Params(pub Vec<Parameter>);
#[derive(Debug)]
pub struct Args(pub Vec<Expr>);
#[derive(Debug)]
pub struct Body(pub Vec<Statement>);
#[derive(Debug)]
pub enum Statement {
    Def(Defs),
    Assign(Ident, Expr),
    IfElse(Expr, Body, Body),
    While(Expr, Body),
    Return(Expr),
}
#[derive(Debug)]
pub enum Defs {
    VarDef(Ident, Type, Expr),
    FunctionDef(Ident, Params, Type, Body),
}
#[derive(Debug)]
pub struct Program(pub Vec<Defs>);

#[derive(Debug, Clone, Copy)]
pub enum Any<'a> {
    Ty(&'a Type),
    E(&'a Expr),
    S(&'a Statement),
    D(&'a Defs),
    B(&'a Body),
    As(&'a Args),
    Ps(&'a Params),
    P(&'a Parameter),
    A(&'a Atom),
    U(&'a Unit),
    F(&'a Factor),
    BT(&'a BTerm),
    CT(&'a CTerm),
    T(&'a Term),
    PR(&'a Program),
}

pub fn append<U, E>(lhs: Result<Vec<U>, E>, rhs: Result<U, E>) -> Result<Vec<U>, ParseErr>
where
    E: Into<ParseErr> + Debug + 'static,
{
    let mut lhs_: Vec<U> = lhs.map_err(|err| err.into())?;
    lhs_.push(rhs.map_err(|err| err.into())?);
    Ok(lhs_)
}
