use std::error::Error;

use lrlex::{lrlex_mod, DefaultLexerTypes};
use lrpar::{lrpar_mod, LexParseError};
use parser_defs::{
    Defs, Parameter, Params, ParseErr, Program,
    Type::{self, Bool, Never},
};

use crate::util::SheafTable;
lrlex_mod!("language.l");
lrpar_mod!("language.y");

pub fn validate(program: &parser_defs::Program) -> Option<Box<dyn Error>> {
    let mut sheaf = SheafTable::new();
    resolution(parser_defs::Any::PR(program), &mut sheaf).err()
}

enum ResolutionState<'a> {
    Variable(Type),
    Function(&'a [Parameter], Type),
}

fn resolution<'a, 'b>(
    program: parser_defs::Any<'a>,
    state: &mut SheafTable<&'b str, ResolutionState<'a>>,
) -> Result<Type, Box<dyn Error>>
where
    'a: 'b,
{
    use parser_defs::Any;
    match program {
        parser_defs::Any::E(expr) => match expr {
            parser_defs::Expr::Or(l, r) | parser_defs::Expr::Xor(l, r) => {
                let l = resolution(Any::E(l), state)?;
                let r = resolution(Any::T(r), state)?;
                if l == Bool && r == Bool {
                    Ok(Bool)
                } else {
                    Err(format!("'OR': Type mismatch. Expected: Bool. Got: {l}, {r}.").into())
                }
            }
            parser_defs::Expr::ETerm(t) => resolution(Any::T(t), state),
        },
        parser_defs::Any::S(s) => match s {
            parser_defs::Statement::Def(d) => resolution(Any::D(d), state),
            parser_defs::Statement::Assign(l, e) => {
                if let Some(ResolutionState::Variable(t)) = state.get(l.as_str()) {
                    let t = *t;
                    let te = resolution(Any::E(e), state)?;
                    if t == te {
                        Ok(t)
                    } else {
                        Err(format!("'=': Type mismatch. Expected: {t}. Got: {te}.").into())
                    }
                } else {
                    Err(format!("Undeclared variable {l}.").into())
                }
            }
            parser_defs::Statement::IfElse(c, t, e) => {
                if Bool != resolution(Any::E(c), state)? {
                    return Err("'IF': Type mismatch. Expected: Bool.".to_string().into());
                }
                state.push();
                resolution(Any::B(t), state)?;
                state.pop();
                state.push();
                resolution(Any::B(e), state)?;
                state.pop();
                Ok(Type::Never)
            }
            parser_defs::Statement::While(c, b) => {
                if Bool != resolution(Any::E(c), state)? {
                    return Err("'WHILE': Type mismatch. Expected: Bool.".to_string().into());
                }
                state.push();
                resolution(Any::B(b), state)?;
                state.pop();
                Ok(Type::Never)
            }
            parser_defs::Statement::Return(e) => {
                let tr = resolution(Any::E(e), state)?;
                if let Some(ResolutionState::Function(_, t)) = state.get("0CUR_FUN") {
                    if *t == tr {
                        return Ok(Type::Never);
                    }
                }
                Err("'RETURN': Type mismatch.".to_string().into())
            }
        },
        parser_defs::Any::D(d) => match d {
            parser_defs::Defs::VarDef(i, t, e) => {
                if state.get(i.as_str()).is_some() {
                    Err(format!("Dual declaration for {i}").into())
                } else {
                    let te = resolution(Any::E(e), state)?;
                    if te != *t {
                        return Err(format!("'=': Type mismatch. Expected: {t}. Got: {te}.").into());
                    }
                    state.insert(i, ResolutionState::Variable(*t));
                    Ok(Never)
                }
            }
            parser_defs::Defs::FunctionDef(i, Params(vec), t, b) => {
                if state.get(i.as_str()).is_some() {
                    return Err(format!("Dual declaration for {i}").into());
                }
                state.insert(i, ResolutionState::Function(vec.as_slice(), *t));
                state.push();
                state.insert("0CUR_FUN", ResolutionState::Function(vec.as_slice(), *t));
                for p in vec {
                    state.insert(&p.0, ResolutionState::Variable(p.1));
                }
                resolution(Any::B(b), state)?;
                state.pop();
                Ok(Never)
            }
        },
        parser_defs::Any::B(parser_defs::Body(vec)) => vec
            .iter()
            .map(|s| resolution(Any::S(s), state))
            .find(|res| res.is_err())
            .unwrap_or(Ok(Never)),
        parser_defs::Any::A(a) => match a {
            parser_defs::Atom::Mult(l, r) | parser_defs::Atom::Div(l, r) => {
                let l = resolution(Any::A(l), state)?;
                let r = resolution(Any::U(r), state)?;
                if l == Type::Int && r == Type::Int {
                    Ok(Type::Int)
                } else {
                    Err(format!("'*//': Type mismatch. Expected: Int. Got: {l}, {r}.").into())
                }
            }
            parser_defs::Atom::AUnit(u) => resolution(Any::U(u), state),
        },
        parser_defs::Any::U(u) => match u {
            parser_defs::Unit::Identifier(i) => {
                if let Some(ResolutionState::Variable(t)) = state.get(i.as_str()) {
                    Ok(*t)
                } else {
                    Err(format!("Undeclared variable {i}.").into())
                }
            }
            parser_defs::Unit::Call(i, a) => {
                if let Some(ResolutionState::Function(p, t)) = state.get(i.as_str()) {
                    let t = *t;
                    if p.len() == a.0.len() {
                        a.0.iter()
                            .zip(p.iter())
                            .map(|(e, p)| -> Result<Type, Box<dyn Error>> {
                                let ta = resolution(Any::E(e), state)?;
                                if ta != p.1 {
                                    Err(format!(
                                        "'CALL': Typ mismatch. Expected: {}. Got: {ta}.",
                                        p.1
                                    )
                                    .into())
                                } else {
                                    Ok(t)
                                }
                            })
                            .find(|res| res.is_err())
                            .unwrap_or(Ok(t))
                    } else {
                        Err(format!("Wrong number of arguments in call to {i}.").into())
                    }
                } else {
                    Err(format!("Undeclared function {i}.").into())
                }
            }
            parser_defs::Unit::Grouping(e) => resolution(Any::E(e), state),
            parser_defs::Unit::True | parser_defs::Unit::False => Ok(Bool),
            parser_defs::Unit::Number(_) => Ok(Type::Int),
        },
        parser_defs::Any::F(f) => match f {
            parser_defs::Factor::Plus(l, r) | parser_defs::Factor::Minus(l, r) => {
                let l = resolution(Any::F(l), state)?;
                let r = resolution(Any::A(r), state)?;
                if l == Type::Int && r == Type::Int {
                    Ok(Type::Int)
                } else {
                    Err(format!("'+/-': Type mismatch. Expected: Int. Got: {l}, {r}.").into())
                }
            }
            parser_defs::Factor::FAtom(a) => resolution(Any::A(a), state),
        },
        parser_defs::Any::CT(ct) => match ct {
            parser_defs::CTerm::GEq(l, r)
            | parser_defs::CTerm::LT(l, r)
            | parser_defs::CTerm::EQ(l, r) => {
                let l = resolution(Any::CT(l), state)?;
                let r = resolution(Any::F(r), state)?;
                if l == Type::Int && r == Type::Int {
                    Ok(Type::Bool)
                } else {
                    Err(format!("'>=/</==': Type mismatch. Expected: Int. Got: {l}, {r}.").into())
                }
            }
            parser_defs::CTerm::CTFactor(f) => resolution(Any::F(f), state),
        },
        parser_defs::Any::T(t) => match t {
            parser_defs::Term::And(l, r) => {
                let l = resolution(Any::T(l), state)?;
                let r = resolution(Any::BT(r), state)?;
                if l == Type::Bool && r == Type::Bool {
                    Ok(Type::Bool)
                } else {
                    Err(format!("'AND': Type mismatch. Expected: Bool. Got: {l}, {r}.").into())
                }
            }
            parser_defs::Term::TCTerm(ct) => resolution(Any::BT(ct), state),
        },
        parser_defs::Any::BT(bt) => match bt {
            parser_defs::BTerm::Not(ct) => {
                let t = resolution(Any::CT(ct), state)?;
                if t != Bool {
                    Err(format!("'NOT': Type mismatch. Expected: Bool. Got: {t}.").into())
                } else {
                    Ok(Bool)
                }
            }
            parser_defs::BTerm::BCTerm(ct) => resolution(Any::CT(ct), state),
        },
        parser_defs::Any::PR(Program(defs)) => {
            for v in defs.iter().filter(|&d| matches!(d, Defs::VarDef(_, _, _))) {
                resolution(Any::D(v), state)?;
            }
            for f in defs.iter() {
                if let Defs::FunctionDef(ident, params, t, ..) = f {
                    state.insert(ident.as_str(), ResolutionState::Function(&params.0[..], *t));
                }
            }
            for f in defs.iter() {
                if let Defs::FunctionDef(_, params, t, b) = f {
                    state.push();
                    state.insert(
                        "0CUR_FUN",
                        ResolutionState::Function(params.0.as_slice(), *t),
                    );
                    for p in &params.0 {
                        state.insert(&p.0, ResolutionState::Variable(p.1));
                    }
                    resolution(Any::B(b), state)?;
                    state.pop();
                }
            }
            Ok(Never)
        }
        _ => Ok(Never),
    }
}

type GRMResult<V, E> = (
    Option<Result<V, E>>,
    Vec<LexParseError<u32, DefaultLexerTypes>>,
);
pub fn parse(s: &str) -> GRMResult<Program, ParseErr> {
    let lexerdef = language_l::lexerdef();
    let lexer = lexerdef.lexer(s);
    language_y::parse(&lexer)
}

#[cfg(test)]
mod tests {
    use crate::parser::validate;

    macro_rules! expect_correct {
        ($path:expr, $($name:ident),+) => {
          $(
            #[test]
            fn $name() {
                let input = std::fs::read_to_string(
                    concat!($path, "/", stringify!($name), ".lang")).unwrap();
                let result = super::parse(&input);
                assert!(result.1.is_empty());
                assert!(result.0.is_some());
                assert!(result.0.as_ref().unwrap().is_ok());
                let p = result.0.unwrap().unwrap();
                let res = validate(&p);
                assert!(res.is_none(), "{}", res.unwrap());
            }
          )+
        };
    }
    macro_rules! expect_wrong_semantics {
        ($path:expr, $($name:ident),+) => {
          $(
            #[test]
            fn $name() {
                let input = std::fs::read_to_string(
                    concat!($path, "/", stringify!($name), ".lang")).unwrap();
                let result = super::parse(&input);
                assert!(result.1.is_empty());
                assert!(result.0.is_some());
                assert!(result.0.as_ref().unwrap().is_ok());
                let p = result.0.unwrap().unwrap();
                let res = validate(&p);
                assert!(res.is_some(), "Expected validate error, but found None");
            }
          )+
        };
    }

    expect_correct!("examples", correct2, example, fibb, recurse, correct3);
    expect_wrong_semantics!(
        "examples",
        wrong1,
        wrong_type1,
        wrong2,
        wrong_type2,
        wrong_type3,
        wrong_type4
    );
}
