use lrlex::{lrlex_mod, DefaultLexerTypes};
use lrpar::{lrpar_mod, LexParseError};
use parser_defs::{ParseErr, Program};
lrlex_mod!("language.l");
lrpar_mod!("language.y");

pub fn parse(
    s: &str,
) -> (
    Option<Result<Program, ParseErr>>,
    Vec<LexParseError<u32, DefaultLexerTypes>>,
) {
    let lexerdef = language_l::lexerdef();
    let lexer = lexerdef.lexer(s);
    return language_y::parse(&lexer);
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse_simple() {
        let input = std::fs::read_to_string("examples/example.lang").unwrap();
        let result = super::parse(&input);
        assert!(result.1.is_empty());
        assert!(result.0.is_some());
        assert!(result.0.unwrap().is_ok());
    }
}
