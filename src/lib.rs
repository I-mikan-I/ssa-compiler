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
        beq rd4, rd5, @_LABEL_0
        j @_LABEL_1
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
}
