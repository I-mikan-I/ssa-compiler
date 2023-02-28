pub mod gvn_pre {
    use std::collections::{HashMap, HashSet};

    use crate::ir::{Block, Operator, SSAOperator, VReg, CFG};

    pub fn optimize(cfg: &mut CFG<SSAOperator>) {
        todo!()
    }

    type Value = usize;
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    enum Expression {
        Plus(Value, Value),
        Sub(Value, Value),
        Mult(Value, Value),
        Div(Value, Value),
        And(Value, Value),
        Or(Value, Value),
        Xor(Value, Value),
        Immediate(i64),
        Reg(VReg),
        Phi(Vec<VReg>),
    }
    impl Expression {
        fn canon(self) -> Self {
            match self {
                Expression::Plus(v1, v2) => {
                    Expression::Plus(std::cmp::min(v1, v2), std::cmp::max(v1, v2))
                }
                Expression::Mult(v1, v2) => {
                    Expression::Mult(std::cmp::min(v1, v2), std::cmp::max(v1, v2))
                }
                Expression::And(v1, v2) => {
                    Expression::And(std::cmp::min(v1, v2), std::cmp::max(v1, v2))
                }
                Expression::Or(v1, v2) => {
                    Expression::Or(std::cmp::min(v1, v2), std::cmp::max(v1, v2))
                }
                Expression::Xor(v1, v2) => {
                    Expression::Or(std::cmp::min(v1, v2), std::cmp::max(v1, v2))
                }
                e => e,
            }
        }
    }
    struct ValueTable {
        expressions: HashMap<Expression, Value>,
        number: Value,
    }
    impl ValueTable {
        fn new() -> Self {
            Self {
                expressions: HashMap::new(),
                number: 5,
            }
        }
        fn maybe_insert(&mut self, exp: Expression) -> Value {
            let canon = exp.canon();
            let entry = self.expressions.entry(canon);
            let value = *entry.or_insert(self.number);
            self.number += 1;
            value
        }
        fn insert_with(&mut self, exp: Expression, val: Value) {
            let canon = exp.canon();
            let res = self.expressions.insert(canon, val);
            debug_assert!(res.is_none())
        }
        fn maybe_insert_op(&mut self, op: &SSAOperator) -> Option<Value> {
            macro_rules! value_regs {
                ($x:expr, $y:expr) => {
                    (
                        self.maybe_insert(Expression::Reg(*$x)),
                        self.maybe_insert(Expression::Reg(*$y)),
                    )
                };
                ($x: expr) => {
                    self.maybe_insert(Expression::Reg($x))
                };
            }
            match op {
                SSAOperator::IROp(op) => match op {
                    Operator::Add(rec, x, y) => {
                        let (x, y) = value_regs!(x, y);
                        let new = self.maybe_insert(Expression::Plus(x, y));
                        self.insert_with(Expression::Reg(*rec), new);
                        Some(new)
                    }
                    Operator::Sub(rec, x, y) => {
                        let (x, y) = value_regs!(x, y);
                        let new = self.maybe_insert(Expression::Sub(x, y));
                        self.insert_with(Expression::Reg(*rec), new);
                        Some(new)
                    }
                    Operator::Mult(rec, x, y) => {
                        let (x, y) = value_regs!(x, y);
                        let new = self.maybe_insert(Expression::Mult(x, y));
                        self.insert_with(Expression::Reg(*rec), new);
                        Some(new)
                    }
                    Operator::Div(rec, x, y) => {
                        let (x, y) = value_regs!(x, y);
                        let new = self.maybe_insert(Expression::Div(x, y));
                        self.insert_with(Expression::Reg(*rec), new);
                        Some(new)
                    }
                    Operator::And(rec, x, y) => {
                        let (x, y) = value_regs!(x, y);
                        let new = self.maybe_insert(Expression::And(x, y));
                        self.insert_with(Expression::Reg(*rec), new);
                        Some(new)
                    }
                    Operator::Or(rec, x, y) => {
                        let (x, y) = value_regs!(x, y);
                        let new = self.maybe_insert(Expression::Or(x, y));
                        self.insert_with(Expression::Reg(*rec), new);
                        Some(new)
                    }
                    Operator::Xor(rec, x, y) => {
                        let (x, y) = value_regs!(x, y);
                        let new = self.maybe_insert(Expression::Xor(x, y));
                        self.insert_with(Expression::Reg(*rec), new);
                        Some(new)
                    }
                    Operator::Li(rec, x) => {
                        let new = self.maybe_insert(Expression::Immediate(*x));
                        self.insert_with(Expression::Reg(*rec), new);
                        Some(new)
                    }
                    _ => None,
                },
                SSAOperator::Phi(rec, operands) => {
                    let new = self.maybe_insert(Expression::Phi(operands.clone()));
                    self.insert_with(Expression::Reg(*rec), new);
                    Some(new)
                }
            }
        }
    }

    type Leader = VReg; //placeholder
    type AntiLeader = bool;

    fn build_sets_block(block: &Block<SSAOperator>) {
        todo!()
    }

    fn generate_value_table(cfg: &mut CFG<SSAOperator>) -> ValueTable {
        let mut value_table = ValueTable::new();

        let mut rpo = cfg.get_dom_rpo();

        while let Some(next) = rpo.pop() {
            let block = cfg.get_block(next);
            for op in &block.body {
                value_table.maybe_insert_op(op);
            }
        }

        #[cfg(feature = "print-gvn")]
        {
            let partitioned = value_table.expressions.iter().fold(
                HashMap::<Value, HashSet<Expression>>::new(),
                |mut map, (k, v)| {
                    map.entry(*v).or_default().insert(k.clone());
                    map
                },
            );
            println!("GVN:\n{:?}", partitioned);
        }
        value_table
    }

    fn build_sets(cfg: &mut CFG<SSAOperator>) -> () {
        let mut avail_out: Vec<HashSet<Leader>> = vec![HashSet::new(); cfg.len()];
        let mut antic_int: Vec<HashSet<AntiLeader>> = vec![HashSet::new(); cfg.len()];
        let value_table = generate_value_table(cfg);

        todo!()
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::ir::CFG;

        #[test]
        fn generate_table() {
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
            let result = crate::parser::parse(&input);
            assert!(result.1.is_empty());
            assert!(result.0.is_some());
            assert!(result.0.as_ref().unwrap().is_ok());
            let p = result.0.unwrap().unwrap();
            let res = crate::parser::validate(&p);
            assert!(res.is_none(), "{}", res.unwrap());

            let mut context = crate::ir::Context::new();
            crate::ir::translate_program(&mut context, &p);
            let funs = context.get_functions();
            let fun = funs.get("myfun").unwrap();
            let body = fun.get_body();
            println!("{}", crate::ir::Displayable(&body[..]));

            let cfg = CFG::from_linear(body, fun.get_params(), fun.get_max_reg());
            let mut ssa = cfg.to_ssa();
            let table = generate_value_table(&mut ssa);

            for val in [
                (Expression::Immediate(100), 1),
                (Expression::Immediate(0), 3),
                (Expression::Immediate(2), 5),
                (Expression::Immediate(3), 1),
                (Expression::Immediate(1), 2),
            ] {
                let number = *table.expressions.get(&val.0).unwrap();
                let count = table
                    .expressions
                    .iter()
                    .filter(|(_, &v)| v == number)
                    .count();
                assert_eq!(count, val.1 + 1);
            }
        }
    }
}
