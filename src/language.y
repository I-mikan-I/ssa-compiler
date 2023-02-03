%start Program
%%
Digit -> Result<i64, ParseErr> :
    'DIGIT' {
                Ok($lexer.span_str($1?.span()).parse()?)
            } ;
Number -> Result<i64, ParseErr> :
    Digit {$1}
    | '-' Digit {
                        $2.map(|v| -1 * v)
                } ;
Identifier -> Result<Ident, ParseErr> :
    'IDENT' {
        Ok($lexer.span_str($1?.span()).into())
    } ;
Arguments -> Result<Args, ParseErr> :
    '(' ')' {
        Ok(Args(Vec::new()))
    }
    | '(' ArgList ')' {
        Ok(Args($2?))
    } ;
ArgList -> Result<Vec<Expr>, ParseErr> :
    Expr {
        Ok(vec![$1?])
    }
    | ArgList ',' Expr {
        append($1, $3)
    } ;
Parameters -> Result<Params, ParseErr> :
    '(' ')' {
        Ok(Params(Vec::new()))
    }
    | '(' ParamList ')' {
        Ok(Params($2?))
    } ;
ParamList -> Result<Vec<Parameter>, ParseErr> :
    Identifier '::' Type {
        Ok(vec![Parameter($1?, $3?)])
    }
    | ParamList ',' Identifier '::' Type {
        append($1, Ok(Parameter($3?, $5?)))
    } ;
Factor -> Result<Factor, ParseErr> :
    Number {
        Ok(Number($1?))
    }
    | Identifier {
        Ok(FIdent($1?))
    }
    | Identifier Arguments  {
        Ok(Call($1?, $2?))
    } ;

Term -> Result<Term, ParseErr> :
    Term '*' Factor {
        Ok(Mult($1?.into(), $3?.into()))
    }
    | Term '/' Factor {
        Ok(Div($1?.into(), $3?.into()))
    }
    | Factor {
        Ok(TFactor($1?))
    } ;

Expr -> Result<Expr, ParseErr> :
    Expr '+' Term {
        Ok(Plus($1?.into(), $3?))
    }
    | Expr '-' Term {
        Ok(Minus($1?.into(), $3?))
    }
    | Term {
        Ok(ETerm($1?))
    } ;
Type -> Result<Type, ParseErr> :
    'INT' {
        Ok(Int)
    } ;
VarDef -> Result<Defs, ParseErr> :
    Identifier '::' Type '=' Expr ';' {
        Ok(VarDef($1?, $3?, $5?))
    } ;
Body -> Result<Body, ParseErr> :
    '{' '}' {
        Ok(Body(Vec::new()))
    }
    | '{' StatementList '}' {
        Ok(Body($2?))
    } ;
Statement -> Result<Statement, ParseErr> :
    VarDef {Ok(Def($1?))}
    | FunctionDef {Ok(Def($1?))}
    | Identifier '=' Expr ';' {
        Ok(Assign($1?, $3?))
    }
    | 'IF' Expr "THEN" Body "ELSE" Body {
        Ok(IfElse($2?, $4?, $6?))
    }
    | 'WHILE' Expr 'DO' Body {
        Ok(While($2?, $4?))
    }
    | 'RETURN' Expr ';' {
        Ok(Return($2?))
    } ;
StatementList -> Result<Vec<Statement>, ParseErr> :
    Statement {
        Ok(vec![$1?])
    }
    | StatementList Statement {
        append($1, $2)
    } ;
FunctionDef -> Result<Defs, ParseErr> :
    'LAMBDA' Identifier Parameters Body {
        Ok(FunctionDef($2?, $3?, $4?))
    } ;
DefList -> Result<Vec<Defs>, ParseErr> :
    VarDef {
        Ok(vec![$1?])
    }
    | FunctionDef {
        Ok(vec![$1?])
    }
    | DefList VarDef {
        append($1, $2)
    }
    | DefList FunctionDef {
        append($1, $2)
    } ;
Program -> Result<Program, ParseErr> :
    {Ok(Program(Vec::new()))}
    | DefList {
        Ok(Program($1?))
    } ;
%%

use parser_defs::*;