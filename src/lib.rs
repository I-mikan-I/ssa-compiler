#![allow(dead_code)]
pub mod ir;
pub mod parser;
pub mod ssa;
pub mod util;

#[cfg(test)]
mod tests {
    use super::ir::*;
    use super::parser::*;

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
        return rd1";
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
        return rd1";
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
        assert_eq!(cfg.get_block(6).children, vec![]);
        assert_eq!(cfg.get_block(6).idom.unwrap(), 1);
        //DEBUG
        cfg.to_ssa();
    }

    #[test]
    fn test_cfg_empty() {
        let input = "
        lambda myfun() :: Int {

        }
        ";
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
