use std::{error::Error, fmt::Debug};

pub use self::{
    Expr::*, Term::*, CTerm::*, Factor::*, Atom::*, Unit::*, Defs::*, Statement::*,
    Type::*,
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
pub type Ident = String;
#[derive(Debug)]
pub enum Type {
    Int,
    Bool
}
#[derive(Debug)]
pub enum Expr {
    Or(Box<Expr>, Term),
    Xor(Box<Expr>, Term),
    ETerm(Term)
}
#[derive(Debug)]
pub enum Term {
    And(Box<Term>, CTerm),
    TCTerm(CTerm)
}
#[derive(Debug)]
pub enum CTerm {
    GEq(Box<CTerm>, Factor),
    LT(Box<CTerm>, Factor),
    EQ(Box<CTerm>, Factor),
    CTFactor(Factor)
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
    AUnit(Unit)
}
#[derive(Debug)]
pub enum Unit {
    Identifier(Ident),
    True,
    False,
    Call(Ident, Args),
    Grouping(Box<Expr>),
    Number(i64)
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
    FunctionDef(Ident, Params, Body),
}
#[derive(Debug)]
pub struct Program(pub Vec<Defs>);

pub fn append<U, E>(lhs: Result<Vec<U>, E>, rhs: Result<U, E>) -> Result<Vec<U>, ParseErr>
where
    E: Into<ParseErr> + Debug + 'static,
{
    let mut lhs_: Vec<U> = lhs.map_err(|err| err.into())?;
    lhs_.push(rhs.map_err(|err| err.into())?);
    Ok(lhs_)
}
