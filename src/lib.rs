#![allow(dead_code)]
mod backend;
mod ir;
mod parser;
mod ssa;
mod util;

use std::{collections::HashMap, error::Error, path::PathBuf};

use clap::Parser;

const TEMP_DIR: &str = "build";
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    input_files: Vec<PathBuf>,
    #[arg(short = 'o', long = "output")]
    output_file: PathBuf,
    #[arg(short = 'g', long, default_value_t = false)]
    no_optimize: bool,
}

pub fn lib_main() {
    let args = Args::parse();
    if let Err(e) = { validate_input(&args.input_files) } {
        eprintln!("Error during argument validation: {}", e);
        return;
    }
    let strings = args
        .input_files
        .iter()
        .map(std::fs::read_to_string)
        .map(|res| {
            res.map_err(|e| -> Box<dyn Error> {
                format!("cannot read from input file: {e}").into()
            })
        })
        .collect::<Result<Vec<_>, _>>();
    if let Err(e) = strings {
        eprintln!("Error during parsing: {e}");
        return;
    }
    let strings = strings.unwrap();

    let parsed: Result<Vec<_>, _> = strings
        .iter()
        .map(|s| parser::parse_and_validate(s))
        .collect();
    if let Err(e) = parsed {
        eprintln!("Error during parsing: {e}");
        return;
    }
    let linearized: Vec<_> = parsed
        .unwrap()
        .into_iter()
        .map(|p| {
            let mut ctx = ir::Context::default();
            ir::translate_program(&mut ctx, &p);
            ctx
        })
        .collect();
    let funs_iter = linearized
        .into_iter()
        .flat_map(|ctx| ctx.functions.into_iter());
    let mut funs = HashMap::new();
    for (name, fun) in funs_iter {
        if funs.contains_key(&name) {
            eprintln!("Two functions with conflicting names {name}");
            return;
        }
        funs.insert(name, fun.into_cfg().into_ssa());
    }

    for v in funs.values_mut() {
        if !args.no_optimize {
            ssa::optimization_sequence(v.get_body_mut())
                .unwrap_or_else(|_| panic!("Bug found, please open an issue"));
        }
    }
    let funs = funs.into_iter().map(|(k, v)| {
        let v = backend::to_assembly(v.into_body(), &k);
        (k, v)
    });
    for (name, body) in funs {
        let out_path = format!("{TEMP_DIR}/{name}.s");
        let res = std::fs::write(&out_path, body);
        if let Err(e) = res {
            eprintln!("Error writing output file: {out_path}\n{e}");
            return;
        }
    }
}

fn validate_input(input: &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
    for input in input {
        if !input.is_file() {
            return Err(format!(
                "input {} is not a valid file",
                input.to_str().unwrap_or("INVALID_UTF8")
            )
            .into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ir::*;

    #[test]
    fn test_ir() {
        let input = "
        myvar3 :: Bool = false;
        lambda myfun(myvar3 :: Int) :: Int {
            myvar4 :: Int = 0;
            if myvar2 then {
                myvar4 = myvar3;
            } else {
                myvar4 = 33;
            }
            return myvar4;
        }
        myvar2 :: Bool = true;
        ";

        let p = super::parser::parse_and_validate(&input).unwrap();

        let mut context = Context::new();
        translate_program(&mut context, &p);
        let funs = context.get_functions();
        let fun = funs.get("myfun").unwrap();
        let body = fun.get_body();
        let params = fun.get_params();
        let expected = "
        li rd1, #0
        li rd2, #0
        la rd3, @myvar2
        load rd4, rd2, rd3
        li rd5, #1
        beq rd4, rd5, @_LABEL_0, @_LABEL_1
@_LABEL_0:
        mv rd1, rd0
        j @_LABEL_2
@_LABEL_1:
        li rd6, #33
        mv rd1, rd6
        j @_LABEL_2
@_LABEL_2:
        return rd1
@_LABEL_3:";
        assert_eq!(
            expected
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>(),
            Displayable(&body[..])
                .to_string()
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
        );
        assert_eq!(&params[..], &[0][..])
    }

    #[test]
    fn test_cfg() {
        let input = "
        myvar3 :: Bool = false;
        lambda myfun(myvar3 :: Int) :: Int {
            myvar4 :: Int = 0;
            i :: Int = 100;
            while (i >= 0) do {
                if i / 2 / 2 / 2 < 2 then {
                    myvar4 = myvar4 * 3;
                } else {
                    myvar4 = myvar4 / 2;
                }
            }
           return myvar4;
        }
        myvar2 :: Bool = true;
        ";

        let p = super::parser::parse_and_validate(&input).unwrap();
        let mut context = Context::new();
        translate_program(&mut context, &p);
        let funs = context.get_functions();
        let fun = funs.get("myfun").unwrap();
        let body = fun.get_body();
        println!("{}", Displayable(&body[..]));
        let params = fun.get_params();
        /*
        rd1 = myvar4
        rd2 = i
         */
        let expected = "
        li rd1, #0
        li rd2, #100
        j @_LABEL_0
@_LABEL_0:
        li rd4, #0
        slt rd5, rd2, rd4
        li rd6, #1
        xor rd7, rd6, rd5
        li rd3, #0
        beq rd7, rd3, @_LABEL_1, @_LABEL_2
@_LABEL_2:
        li rd8, #2
        div rd9, rd2, rd8
        li rd10, #2
        div rd11, rd9, rd10
        li rd12, #2
        div rd13, rd11, rd12
        li rd14, #2
        slt rd15, rd13, rd14
        li rd16, #1
        beq rd15, rd16, @_LABEL_3, @_LABEL_4
@_LABEL_3:
        li rd17, #3
        mult rd18, rd1, rd17
        mv rd1, rd18
        j @_LABEL_5
@_LABEL_4:
        li rd19, #2
        div rd20, rd1, rd19
        mv rd1, rd20
        j @_LABEL_5
@_LABEL_5:
        j @_LABEL_0
@_LABEL_1:
        return rd1
@_LABEL_6:";
        assert_eq!(
            expected
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>(),
            Displayable(&body[..])
                .to_string()
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
        );
        assert_eq!(&params[..], &[0][..]);

        let cfg = CFG::from_linear(body, fun.get_params(), fun.get_max_reg());
        println!("{:?}", cfg);
        assert_eq!(
            cfg.get_block(0).body,
            vec![
                Operator::GetParameter(0, 0),
                Operator::Li(1, 0),
                Operator::Li(2, 100),
                Operator::J(std::rc::Rc::from("_LABEL_0"))
            ]
        );
        assert_eq!(cfg.get_block(0).children, vec![1]);
        assert_eq!(cfg.get_block(0).idom, None);
        assert_eq!(cfg.get_block(1).children, vec![6, 2]);
        assert_eq!(cfg.get_block(1).idom.unwrap(), 0);
        assert_eq!(cfg.get_block(2).children, vec![3, 4]);
        assert_eq!(cfg.get_block(2).idom.unwrap(), 1);
        assert_eq!(cfg.get_block(3).children, vec![5]);
        assert_eq!(cfg.get_block(3).idom.unwrap(), 2);
        assert_eq!(cfg.get_block(4).children, vec![5]);
        assert_eq!(cfg.get_block(4).idom.unwrap(), 2);
        assert_eq!(cfg.get_block(5).children, vec![1]);
        assert_eq!(cfg.get_block(5).idom.unwrap(), 2);
        assert_eq!(cfg.get_block(6).children, vec![7]);
        assert_eq!(cfg.get_block(6).idom.unwrap(), 1);
        assert_eq!(cfg.get_block(7).children, vec![]);
        assert_eq!(cfg.get_block(7).idom.unwrap(), 6);
        cfg.into_ssa();
    }

    #[test]
    fn test_cfg_empty() {
        let input = "
        lambda myfun() :: Int {

        }
        ";

        let p = super::parser::parse_and_validate(&input).unwrap();
        let mut context = Context::new();
        translate_program(&mut context, &p);
        let funs = context.get_functions();
        let fun = funs.get("myfun").unwrap();
        let body = fun.get_body();
        println!("{}", Displayable(&body[..]));
        let params = fun.get_params();
        /*
        rd1 = myvar4
        rd2 = i
         */
        let expected = "";
        assert_eq!(
            expected
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>(),
            Displayable(&body[..])
                .to_string()
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
        );
        assert_eq!(&params[..], &[][..]);

        let cfg = CFG::from_linear(body, fun.get_params(), fun.get_max_reg());
        println!("{:?}", cfg);
        assert_eq!(cfg.get_block(0).children, vec![]);
        assert_eq!(cfg.get_block(0).idom, None);
    }
}
