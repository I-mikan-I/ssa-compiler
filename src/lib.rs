pub mod ir;
pub mod parser;
pub mod util;

#[cfg(test)]
mod tests {
    use super::ir::*;
    use super::parser::*;

    #[test]
    fn test1() {
        let input = std::fs::read_to_string(concat!("examples", "/", "correct2", ".lang")).unwrap();
        let result = parse(&input);
        assert!(result.1.is_empty());
        assert!(result.0.is_some());
        assert!(result.0.as_ref().unwrap().is_ok());
        let p = result.0.unwrap().unwrap();
        let res = validate(&p);
        assert!(res.is_none(), "{}", res.unwrap());

        let mut context = Context::new();
        translate_program(&mut context, &p);
        let funs = context.get_functions();
        for (name, fun) in funs {
            println!("\nFN {name} ::");
            println!("{}", Displayable(&fun[..]));
        }
    }
}
