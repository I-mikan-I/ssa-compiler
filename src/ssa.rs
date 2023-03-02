pub mod gvn_pre {
    #![allow(clippy::too_many_arguments)]
    use std::collections::{HashMap, HashSet, LinkedList};

    use crate::ir::{Operator, SSAOperator, VReg, CFG};

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
        Mv(Value),
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
                    Expression::Xor(std::cmp::min(v1, v2), std::cmp::max(v1, v2))
                }
                e => e,
            }
        }
        fn depends_on(&self) -> Vec<Value> {
            match self {
                Expression::Sub(v1, v2)
                | Expression::Mult(v1, v2)
                | Expression::Div(v1, v2)
                | Expression::And(v1, v2)
                | Expression::Or(v1, v2)
                | Expression::Xor(v1, v2)
                | Expression::Plus(v1, v2) => vec![*v1, *v2],
                Expression::Mv(v1) => vec![*v1],
                _ => vec![],
            }
        }
    }
    struct ValueTable {
        expressions: HashMap<Expression, Value>,
        number: Value,
        blackbox_regs: HashSet<VReg>,
    }
    impl ValueTable {
        fn new() -> Self {
            Self {
                expressions: HashMap::new(),
                number: 5,
                blackbox_regs: HashSet::new(),
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
            debug_assert!(res.is_none() || res.unwrap() == val);
        }
        fn maybe_insert_op(
            &mut self,
            op: &SSAOperator,
            exp_gen: &mut LinkedList<(Value, Expression)>,
            added_exps: &mut HashSet<Value>,
        ) -> Result<(Value, Option<VReg>, Expression), Option<VReg>> {
            macro_rules! value_regs {
                ($($x: expr),+) => {
                {
                    ($({
                    let new = Expression::Reg(*$x);
                    if !self.expressions.contains_key(&new) {
                        self.blackbox_regs.insert(*$x);
                    }
                    let val = self.maybe_insert(new);
                    if !added_exps.contains(&val) {
                        exp_gen.push_back((val, Expression::Reg(*$x)));
                        added_exps.insert(val);
                    }
                    val}
                    ),+)
                }
                };
            }
            let res = match op {
                SSAOperator::IROp(op) => match op {
                    Operator::Add(rec, x, y) => {
                        let (x, y) = value_regs!(x, y);
                        let res = Expression::Plus(x, y).canon();
                        let new = self.maybe_insert(res.clone());
                        self.insert_with(Expression::Reg(*rec), new);
                        Ok((new, Some(*rec), res))
                    }
                    Operator::Sub(rec, x, y) => {
                        let (x, y) = value_regs!(x, y);
                        let res = Expression::Sub(x, y).canon();
                        let new = self.maybe_insert(Expression::Sub(x, y));
                        self.insert_with(Expression::Reg(*rec), new);
                        Ok((new, Some(*rec), res))
                    }
                    Operator::Mult(rec, x, y) => {
                        let (x, y) = value_regs!(x, y);
                        let res = Expression::Mult(x, y).canon();
                        let new = self.maybe_insert(Expression::Mult(x, y));
                        self.insert_with(Expression::Reg(*rec), new);
                        Ok((new, Some(*rec), res))
                    }
                    Operator::Div(rec, x, y) => {
                        let (x, y) = value_regs!(x, y);
                        let res = Expression::Div(x, y).canon();
                        let new = self.maybe_insert(Expression::Div(x, y));
                        self.insert_with(Expression::Reg(*rec), new);
                        Ok((new, Some(*rec), res))
                    }
                    Operator::And(rec, x, y) => {
                        let (x, y) = value_regs!(x, y);
                        let res = Expression::And(x, y).canon();
                        let new = self.maybe_insert(Expression::And(x, y));
                        self.insert_with(Expression::Reg(*rec), new);
                        Ok((new, Some(*rec), res))
                    }
                    Operator::Or(rec, x, y) => {
                        let (x, y) = value_regs!(x, y);
                        let res = Expression::Or(x, y).canon();
                        let new = self.maybe_insert(Expression::Or(x, y));
                        self.insert_with(Expression::Reg(*rec), new);
                        Ok((new, Some(*rec), res))
                    }
                    Operator::Xor(rec, x, y) => {
                        let (x, y) = value_regs!(x, y);
                        let res = Expression::Xor(x, y).canon();
                        let new = self.maybe_insert(Expression::Xor(x, y));
                        self.insert_with(Expression::Reg(*rec), new);
                        Ok((new, Some(*rec), res))
                    }
                    Operator::Li(rec, x) => {
                        let res = Expression::Immediate(*x).canon();
                        let new = self.maybe_insert(Expression::Immediate(*x));
                        self.insert_with(Expression::Reg(*rec), new);
                        Ok((new, Some(*rec), res))
                    }
                    Operator::Mv(rec, x) => {
                        let x = value_regs!(x);
                        let res = Expression::Mv(x).canon();
                        let new = self.maybe_insert(Expression::Mv(x));
                        self.insert_with(Expression::Reg(*rec), new);
                        Ok((new, Some(*rec), res))
                    }
                    Operator::Return(x) => {
                        let new = value_regs!(x);
                        Ok((new, None, Expression::Reg(*x)))
                    }
                    //unsupported; kill sets
                    Operator::Load(x, _, _)
                    | Operator::Store(x, _, _)
                    | Operator::La(x, _)
                    | Operator::Slt(x, _, _)
                    | Operator::Call(x, _, _)
                    | Operator::GetParameter(x, _) => Err(Some(*x)),
                    _ => Err(None),
                },
                SSAOperator::Phi(rec, operands) => {
                    let res = Expression::Phi(operands.clone()).canon();
                    let new = self.maybe_insert(Expression::Phi(operands.clone()));
                    self.insert_with(Expression::Reg(*rec), new);
                    return Ok((new, Some(*rec), res)); //skip adding to exp_gen
                }
            };
            if let Ok((val, _, exp)) = &res {
                if !added_exps.contains(val) {
                    added_exps.insert(*val);
                    exp_gen.push_back((*val, exp.clone()));
                }
            }
            res
        }
    }

    type Leader = VReg; //placeholder
    type AntiLeader = bool;

    fn build_sets_phase1(
        cfg: &CFG<SSAOperator>,
        current: usize,
        exp_gen: &mut Vec<LinkedList<(Value, Expression)>>,
        phi_gen: &mut Vec<HashMap<Value, VReg>>,
        tmp_gen: &mut Vec<HashSet<VReg>>,
        leaders: &mut Vec<HashMap<Value, VReg>>,
        table: &mut ValueTable,
    ) {
        let block = cfg.get_block(current);
        let mut added_exps = HashSet::new();
        for op in &block.body {
            match table.maybe_insert_op(op, &mut exp_gen[current], &mut added_exps) {
                Ok((val, Some(reg), exp)) => {
                    leaders[current].entry(val).or_insert(reg);
                    if let Expression::Phi(_) = exp {
                        phi_gen[current].entry(val).or_insert(reg);
                    }
                }
                Err(Some(killed)) => {
                    tmp_gen[current].insert(killed);
                }
                _ => {}
            }
        }
        for dom_child in block.idom_of.clone() {
            leaders[dom_child] = leaders[current].clone();
            build_sets_phase1(cfg, dom_child, exp_gen, phi_gen, tmp_gen, leaders, table);
        }
    }

    fn build_sets_phase2(
        cfg: &CFG<SSAOperator>,
        current: usize,
        changed: &mut bool,
        antic_out: &mut Vec<LinkedList<(Value, Expression)>>,
        antic_in: &mut Vec<LinkedList<(Value, Expression)>>,
        exp_gen: &Vec<LinkedList<(Value, Expression)>>,
        tmp_gen: &Vec<HashSet<VReg>>,
        phi_gen: &Vec<HashMap<Value, VReg>>,
        value_table: &mut ValueTable,
    ) {
        let block = cfg.get_block(current);
        for &child in &block.idom_of {
            build_sets_phase2(
                cfg,
                child,
                changed,
                antic_out,
                antic_in,
                exp_gen,
                tmp_gen,
                phi_gen,
                value_table,
            );
        }
        #[allow(clippy::comparison_chain)]
        if block.children.len() > 1 {
            let potential_out = &antic_in[block.children[0]];
            let mut result = LinkedList::new();
            let rest = block.children[1..].iter().map(|child| &antic_in[*child]);
            for (val, exp) in potential_out {
                if rest
                    .clone()
                    .map(|child_exp| child_exp.iter().map(|t| t.0).any(|v| v == *val))
                    .all(|b| b)
                {
                    result.push_back((*val, exp.clone()))
                };
            }
            if antic_out[current] != result {
                *changed = true;
            }
            antic_out[current] = result;
        } else if block.children.len() == 1 {
            let mut result = LinkedList::new();
            let mut translated = HashMap::new();
            let child = block.children[0];
            let antic_in_succ = &antic_in[child];
            for (val, exp) in antic_in_succ {
                if let Some(&reg) = phi_gen[child].get(val) {
                    let child_block = cfg.get_block(child);
                    let self_index = child_block
                        .preds
                        .iter()
                        .position(|&blk| blk == current)
                        .unwrap();
                    let mut iter = child_block.body.iter();
                    while let Some(SSAOperator::Phi(rec, vec)) = iter.next() {
                        if *rec == reg {
                            let translated_reg = vec[self_index];
                            let translated_val =
                                value_table.maybe_insert(Expression::Reg(translated_reg));
                            result.push_back((translated_val, Expression::Reg(translated_reg)));
                            translated.insert(*val, translated_val);
                        }
                    }
                } else {
                    macro_rules! maybe_translate {
                        ($t:path, $($x:expr),+) => {
                            $t($(if let Some(new) = translated.get($x) {*new} else {*$x}),+)
                        }
                    }
                    let updated = match exp {
                        Expression::Plus(x, y) => maybe_translate!(Expression::Plus, x, y),
                        Expression::Sub(x, y) => maybe_translate!(Expression::Sub, x, y),
                        Expression::Mult(x, y) => maybe_translate!(Expression::Mult, x, y),
                        Expression::Div(x, y) => maybe_translate!(Expression::Div, x, y),
                        Expression::And(x, y) => maybe_translate!(Expression::And, x, y),
                        Expression::Or(x, y) => maybe_translate!(Expression::Or, x, y),
                        Expression::Xor(x, y) => maybe_translate!(Expression::Xor, x, y),
                        Expression::Phi(..) => continue,
                        e => e.clone(),
                    };
                    let updated_val = value_table.maybe_insert(updated.clone());
                    result.push_back((updated_val, updated));
                    translated.insert(*val, updated_val);
                }
            }
            if antic_out[current] != result {
                *changed = true;
            }
            antic_out[current] = result;
        }

        let mut killed = HashSet::new();
        let cleaned = exp_gen[current]
            .iter()
            .chain(antic_out[current].iter())
            .filter_map(|(val, exp)| {
                for dependency in exp.depends_on() {
                    if killed.contains(&dependency) {
                        killed.insert(*val);
                        return None;
                    }
                }
                if let Expression::Reg(r) = exp {
                    if tmp_gen[current].contains(r) {
                        killed.insert(*val);
                        return None;
                    }
                }
                Some((*val, exp.clone()))
                // not needed because exp already contains and overwrites any regs in antic_out!
                // if let Some(updated) = tmp_gen[current].get(val) {
                //     Some((*val, updated.clone()))
                // } else {
                //     Some((*val, exp.clone()))
                // }
            });
        let mut added = HashSet::new();
        let mut result = LinkedList::new();
        for v in cleaned {
            if !added.contains(&v.0) {
                added.insert(v.0);
                result.push_back(v);
            }
        }
        if antic_in[current] != result {
            *changed = true;
        }
        antic_in[current] = result;
    }

    fn generate_value_table(cfg: &mut CFG<SSAOperator>) -> ValueTable {
        let mut value_table = ValueTable::new();

        let mut rpo = cfg.get_dom_rpo();

        while let Some(next) = rpo.pop() {
            let block = cfg.get_block(next);
            for op in &block.body {
                let _ = value_table.maybe_insert_op(op, &mut LinkedList::new(), &mut HashSet::new());
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

    fn build_sets(
        cfg: &mut CFG<SSAOperator>,
    ) -> (
        Vec<HashMap<Value, VReg>>,
        Vec<LinkedList<(Value, Expression)>>,
        ValueTable,
    ) {
        let mut value_table = generate_value_table(cfg);

        let mut exp_gen = vec![LinkedList::default(); cfg.len()];
        let mut tmp_gen = vec![HashSet::default(); cfg.len()];
        let mut phi_gen = vec![HashMap::default(); cfg.len()];
        let mut leaders = vec![HashMap::default(); cfg.len()];
        build_sets_phase1(
            cfg,
            cfg.get_entry(),
            &mut exp_gen,
            &mut phi_gen,
            &mut tmp_gen,
            &mut leaders,
            &mut value_table,
        );
        #[cfg(feature = "print-gvn")]
        {
            println! {"exp_gens: \n{:?}\n", exp_gen};
            println! {"tmp_gen: \n{:?}\n", tmp_gen};
            println! {"phi_gens: \n{:?}\n", phi_gen};
            println! {"leaders: \n{:?}\n", leaders};
        }
        let mut changed = true;
        let mut antic_out = vec![LinkedList::default(); cfg.len()];
        let mut antic_in = vec![LinkedList::default(); cfg.len()];
        while changed {
            changed = false;
            build_sets_phase2(
                cfg,
                cfg.get_entry(),
                &mut changed,
                &mut antic_out,
                &mut antic_in,
                &exp_gen,
                &tmp_gen,
                &phi_gen,
                &mut value_table,
            );
        }
        #[cfg(feature = "print-gvn")]
        {
            println! {"antic_in: \n{:?}\n", antic_in};
        }
        (leaders, antic_in, value_table)
    }

    #[cfg(test)]
    mod tests {
        use std::rc::Rc;

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

        #[test]
        fn build_sets_simple() {
            let body = vec![
                Operator::Li(1, 3),
                Operator::Li(2, 5),
                Operator::Add(3, 1, 2),
            ];
            println!("{}", crate::ir::Displayable(&body[..]));

            let cfg = CFG::from_linear(body, vec![], 3);
            let mut ssa = cfg.to_ssa();
            let (leaders, antic_in, mut table) = build_sets(&mut ssa);
            assert_eq!(leaders.len(), 1);
            let value_3 = table.maybe_insert(Expression::Immediate(3));
            let value_5 = table.maybe_insert(Expression::Immediate(5));
            let value_sum = table.maybe_insert(Expression::Plus(value_5, value_3));
            assert_eq!(*leaders[0].get(&value_3).unwrap(), 4);
            assert_eq!(*leaders[0].get(&value_5).unwrap(), 5);
            assert_eq!(*leaders[0].get(&value_sum).unwrap(), 6);

            assert_eq!(
                Vec::from_iter(antic_in[0].iter().cloned()),
                &[
                    (value_3, Expression::Immediate(3)),
                    (value_5, Expression::Immediate(5)),
                    (value_sum, Expression::Plus(value_3, value_5))
                ]
            );
        }

        #[test]
        fn build_sets_trans() {
            let l1: Rc<str> = "Label1".into();
            let body = vec![
                Operator::Li(1, 3),
                Operator::Li(2, 5),
                Operator::Add(3, 1, 2),
                Operator::J(Rc::clone(&l1)),
                Operator::Label(Rc::clone(&l1)),
                Operator::Sub(4, 3, 2),
                Operator::Xor(5, 4, 3),
                Operator::Return(5),
            ];
            println!("{}", crate::ir::Displayable(&body[..]));

            let cfg = CFG::from_linear(body, vec![], 3);
            let mut ssa = cfg.to_ssa();
            let (leaders, antic_in, mut table) = build_sets(&mut ssa);
            assert_eq!(leaders.len(), 2);
            assert_eq!(antic_in.len(), 2);
            let value_3 = table.maybe_insert(Expression::Immediate(3));
            let value_5 = table.maybe_insert(Expression::Immediate(5));
            let value_sum = table.maybe_insert(Expression::Plus(value_3, value_5));
            let value_sub = table.maybe_insert(Expression::Sub(value_sum, value_5));
            let value_xor = table.maybe_insert(Expression::Xor(value_sub, value_sum));
            assert_eq!(leaders[0].len(), 3);
            assert_eq!(*leaders[0].get(&value_3).unwrap(), 4);
            assert_eq!(*leaders[0].get(&value_5).unwrap(), 5);
            assert_eq!(*leaders[0].get(&value_sum).unwrap(), 6);
            assert_eq!(leaders[1].len(), 5);
            assert_eq!(*leaders[1].get(&value_3).unwrap(), 4);
            assert_eq!(*leaders[1].get(&value_5).unwrap(), 5);
            assert_eq!(*leaders[1].get(&value_sum).unwrap(), 6);
            assert_eq!(*leaders[1].get(&value_sub).unwrap(), 7);
            assert_eq!(*leaders[1].get(&value_xor).unwrap(), 8);

            assert_eq!(
                Vec::from_iter(antic_in[0].iter().cloned()),
                &[
                    (value_3, Expression::Immediate(3)),
                    (value_5, Expression::Immediate(5)),
                    (value_sum, Expression::Plus(value_3, value_5)),
                    (value_sub, Expression::Sub(value_sum, value_5)),
                    (value_xor, Expression::Xor(value_sum, value_sub)),
                ]
            );
            assert_eq!(
                Vec::from_iter(antic_in[1].iter().cloned()),
                &[
                    (value_sum, Expression::Reg(6)),
                    (value_5, Expression::Reg(5)),
                    (value_sub, Expression::Sub(value_sum, value_5)),
                    (value_xor, Expression::Xor(value_sum, value_sub)),
                ]
            );
        }

        #[test]
        fn build_sets_trans_kill() {
            let l1: Rc<str> = "Label1".into();
            let body = vec![
                Operator::Li(1, 3),
                Operator::Li(2, 5),
                Operator::Slt(3, 1, 2),
                Operator::J(Rc::clone(&l1)),
                Operator::Label(Rc::clone(&l1)),
                Operator::Sub(4, 3, 2),
                Operator::Xor(5, 4, 3),
                Operator::Return(5),
            ];
            println!("{}", crate::ir::Displayable(&body[..]));

            let cfg = CFG::from_linear(body, vec![], 3);
            let mut ssa = cfg.to_ssa();
            let (leaders, antic_in, mut table) = build_sets(&mut ssa);
            assert_eq!(leaders.len(), 2);
            assert_eq!(antic_in.len(), 2);
            let value_3 = table.maybe_insert(Expression::Immediate(3));
            let value_5 = table.maybe_insert(Expression::Immediate(5));
            let value_slt = table.maybe_insert(Expression::Reg(6));
            let value_sub = table.maybe_insert(Expression::Sub(value_slt, value_5));
            let value_xor = table.maybe_insert(Expression::Xor(value_sub, value_slt));
            assert_eq!(leaders[0].len(), 2);
            assert_eq!(*leaders[0].get(&value_3).unwrap(), 4);
            assert_eq!(*leaders[0].get(&value_5).unwrap(), 5);
            assert_eq!(leaders[1].len(), 4);
            assert_eq!(*leaders[1].get(&value_3).unwrap(), 4);
            assert_eq!(*leaders[1].get(&value_5).unwrap(), 5);
            assert_eq!(*leaders[1].get(&value_sub).unwrap(), 7);
            assert_eq!(*leaders[1].get(&value_xor).unwrap(), 8);

            assert_eq!(
                Vec::from_iter(antic_in[0].iter().cloned()),
                &[
                    (value_3, Expression::Immediate(3)),
                    (value_5, Expression::Immediate(5)),
                ]
            );
            assert_eq!(
                Vec::from_iter(antic_in[1].iter().cloned()),
                &[
                    (value_slt, Expression::Reg(6)),
                    (value_5, Expression::Reg(5)),
                    (value_sub, Expression::Sub(value_slt, value_5)),
                    (value_xor, Expression::Xor(value_slt, value_sub)),
                ]
            );
        }

        #[test]
        fn build_sets_if() {
            let l1: Rc<str> = "Label1".into();
            let l2: Rc<str> = "Label2".into();
            let body = vec![
                Operator::Li(1, 3),
                Operator::Li(2, 5),
                Operator::Add(3, 1, 2),
                Operator::Beq(1, 2, Rc::clone(&l1), Rc::clone(&l2)),
                Operator::Label(Rc::clone(&l1)),
                Operator::Sub(4, 3, 2),
                Operator::Xor(5, 4, 3),
                Operator::Return(5),
                Operator::Label(Rc::clone(&l2)),
                Operator::Sub(4, 3, 2),
                Operator::And(5, 4, 3),
                Operator::Return(5),
            ];
            println!("{}", crate::ir::Displayable(&body[..]));

            let cfg = CFG::from_linear(body, vec![], 5);
            let mut ssa = cfg.to_ssa();
            let (leaders, antic_in, mut table) = build_sets(&mut ssa);
            assert_eq!(leaders.len(), 3);
            assert_eq!(antic_in.len(), 3);
            let value_3 = table.maybe_insert(Expression::Immediate(3));
            let value_5 = table.maybe_insert(Expression::Immediate(5));
            let value_sum = table.maybe_insert(Expression::Plus(value_3, value_5));
            let value_sub = table.maybe_insert(Expression::Sub(value_sum, value_5));
            let value_xor = table.maybe_insert(Expression::Xor(value_sub, value_sum));
            let value_and = table.maybe_insert(Expression::And(value_sub, value_sum));
            assert_eq!(leaders[0].len(), 3);
            assert_eq!(*leaders[0].get(&value_3).unwrap(), 6);
            assert_eq!(*leaders[0].get(&value_5).unwrap(), 7);
            assert_eq!(*leaders[0].get(&value_sum).unwrap(), 8);
            assert_eq!(leaders[1].len(), 5);
            assert_eq!(*leaders[1].get(&value_3).unwrap(), 6);
            assert_eq!(*leaders[1].get(&value_5).unwrap(), 7);
            assert_eq!(*leaders[1].get(&value_sum).unwrap(), 8);
            assert_eq!(*leaders[1].get(&value_sub).unwrap(), 9);
            assert_eq!(*leaders[1].get(&value_xor).unwrap(), 10);
            assert_eq!(leaders[2].len(), 5);
            assert_eq!(*leaders[2].get(&value_3).unwrap(), 6);
            assert_eq!(*leaders[2].get(&value_5).unwrap(), 7);
            assert_eq!(*leaders[2].get(&value_sum).unwrap(), 8);
            assert_eq!(*leaders[2].get(&value_sub).unwrap(), 11);
            assert_eq!(*leaders[2].get(&value_and).unwrap(), 12);

            assert_eq!(
                Vec::from_iter(antic_in[0].iter().cloned()),
                &[
                    (value_3, Expression::Immediate(3)),
                    (value_5, Expression::Immediate(5)),
                    (value_sum, Expression::Plus(value_3, value_5)),
                    (value_sub, Expression::Sub(value_sum, value_5)),
                ]
            );
            assert_eq!(
                Vec::from_iter(antic_in[1].iter().cloned()),
                &[
                    (value_sum, Expression::Reg(8)),
                    (value_5, Expression::Reg(7)),
                    (value_sub, Expression::Sub(value_sum, value_5)),
                    (value_xor, Expression::Xor(value_sum, value_sub)),
                ]
            );
            assert_eq!(
                Vec::from_iter(antic_in[2].iter().cloned()),
                &[
                    (value_sum, Expression::Reg(8)),
                    (value_5, Expression::Reg(7)),
                    (value_sub, Expression::Sub(value_sum, value_5)),
                    (value_and, Expression::And(value_sum, value_sub)),
                ]
            );
        }
        #[test]
        fn build_sets_loop() {
            let l1: Rc<str> = "Label1".into();
            let l2: Rc<str> = "Label2".into();
            let l3: Rc<str> = "Label3".into();
            let l4: Rc<str> = "Label4".into();
            let body = vec![
                Operator::Li(1, 3),
                Operator::Li(2, 5),
                Operator::Add(3, 1, 2),
                Operator::J(Rc::clone(&l4)),
                Operator::Label(Rc::clone(&l4)),
                Operator::Beq(1, 2, Rc::clone(&l1), Rc::clone(&l2)),
                Operator::Label(Rc::clone(&l1)),
                Operator::Sub(4, 3, 2),
                Operator::Xor(5, 4, 3),
                Operator::J(Rc::clone(&l3)),
                Operator::Label(Rc::clone(&l2)),
                Operator::Sub(4, 3, 2),
                Operator::And(5, 4, 3),
                Operator::J(Rc::clone(&l3)),
                Operator::Label(Rc::clone(&l3)),
                Operator::J(Rc::clone(&l4)),
            ];
            println!("{}", crate::ir::Displayable(&body[..]));

            let cfg = CFG::from_linear(body, vec![], 5);
            let mut ssa = cfg.to_ssa();
            let (leaders, antic_in, mut table) = build_sets(&mut ssa);
            assert_eq!(leaders.len(), 5);
            assert_eq!(antic_in.len(), 5);
            let value_3 = table.maybe_insert(Expression::Immediate(3));
            let value_5 = table.maybe_insert(Expression::Immediate(5));
            let value_sum = table.maybe_insert(Expression::Plus(value_3, value_5));
            let value_sub = table.maybe_insert(Expression::Sub(value_sum, value_5));
            let value_xor = table.maybe_insert(Expression::Xor(value_sub, value_sum));
            let value_and = table.maybe_insert(Expression::And(value_sub, value_sum));
            assert_eq!(leaders[0].len(), 3);
            assert_eq!(*leaders[0].get(&value_3).unwrap(), 6);
            assert_eq!(*leaders[0].get(&value_5).unwrap(), 7);
            assert_eq!(*leaders[0].get(&value_sum).unwrap(), 8);
            assert_eq!(leaders[1].len(), 3);
            assert_eq!(leaders[2].len(), 5);
            assert_eq!(*leaders[2].get(&value_3).unwrap(), 6);
            assert_eq!(*leaders[2].get(&value_5).unwrap(), 7);
            assert_eq!(*leaders[2].get(&value_sum).unwrap(), 8);
            assert_eq!(*leaders[2].get(&value_sub).unwrap(), 9);
            assert_eq!(*leaders[2].get(&value_xor).unwrap(), 10);
            assert_eq!(leaders[3].len(), 5);
            assert_eq!(*leaders[3].get(&value_3).unwrap(), 6);
            assert_eq!(*leaders[3].get(&value_5).unwrap(), 7);
            assert_eq!(*leaders[3].get(&value_sum).unwrap(), 8);
            assert_eq!(*leaders[3].get(&value_sub).unwrap(), 11);
            assert_eq!(*leaders[3].get(&value_and).unwrap(), 12);

            assert_eq!(
                Vec::from_iter(antic_in[0].iter().cloned()),
                &[
                    (value_3, Expression::Immediate(3)),
                    (value_5, Expression::Immediate(5)),
                    (value_sum, Expression::Plus(value_3, value_5)),
                    (value_sub, Expression::Sub(value_sum, value_5)),
                ]
            );
            assert_eq!(
                Vec::from_iter(antic_in[1].iter().cloned()),
                &[
                    (value_sum, Expression::Reg(8)),
                    (value_5, Expression::Reg(7)),
                    (value_sub, Expression::Sub(value_sum, value_5)),
                ]
            );
            assert_eq!(
                Vec::from_iter(antic_in[2].iter().cloned()),
                &[
                    (value_sum, Expression::Reg(8)),
                    (value_5, Expression::Reg(7)),
                    (value_sub, Expression::Sub(value_sum, value_5)),
                    (value_xor, Expression::Xor(value_sum, value_sub)),
                ]
            );
            assert_eq!(
                Vec::from_iter(antic_in[3].iter().cloned()),
                &[
                    (value_sum, Expression::Reg(8)),
                    (value_5, Expression::Reg(7)),
                    (value_sub, Expression::Sub(value_sum, value_5)),
                    (value_and, Expression::And(value_sum, value_sub)),
                ]
            );
            assert_eq!(
                Vec::from_iter(antic_in[4].iter().cloned()),
                &[
                    (value_sum, Expression::Reg(8)),
                    (value_5, Expression::Reg(7)),
                    (value_sub, Expression::Sub(value_sum, value_5)),
                ]
            );
        }
        #[test]
        fn build_sets_phi() {
            let l1: Rc<str> = "Label1".into();
            let l2: Rc<str> = "Label2".into();
            let l3: Rc<str> = "Label3".into();
            let l4: Rc<str> = "Label4".into();
            let body = vec![
                Operator::Li(1, 3),
                Operator::Li(2, 5),
                Operator::Add(3, 1, 2),
                Operator::J(Rc::clone(&l4)),
                Operator::Label(Rc::clone(&l4)),
                Operator::Beq(1, 2, Rc::clone(&l1), Rc::clone(&l2)),
                Operator::Label(Rc::clone(&l1)),
                Operator::Sub(4, 3, 2),
                Operator::Xor(3, 4, 3),
                Operator::J(Rc::clone(&l3)),
                Operator::Label(Rc::clone(&l2)),
                Operator::Sub(4, 3, 2),
                Operator::And(3, 4, 3),
                Operator::J(Rc::clone(&l3)),
                Operator::Label(Rc::clone(&l3)),
                Operator::J(Rc::clone(&l4)),
            ];
            println!("{}", crate::ir::Displayable(&body[..]));

            let cfg = CFG::from_linear(body, vec![], 5);
            let mut ssa = cfg.to_ssa();
            let (leaders, antic_in, mut table) = build_sets(&mut ssa);
            assert_eq!(leaders.len(), 5);
            assert_eq!(antic_in.len(), 5);
            let value_3 = table.maybe_insert(Expression::Immediate(3));
            let value_5 = table.maybe_insert(Expression::Immediate(5));
            let value_phi1 = table.maybe_insert(Expression::Phi(vec![8, 14]));
            let value_phi2 = table.maybe_insert(Expression::Phi(vec![11, 13]));
            let value_sum = table.maybe_insert(Expression::Plus(value_3, value_5));
            let value_sub = table.maybe_insert(Expression::Sub(value_sum, value_5));
            let value_sub_phi = table.maybe_insert(Expression::Sub(value_phi1, value_5));
            let value_xor = table.maybe_insert(Expression::Xor(value_sub_phi, value_phi1));
            let value_and = table.maybe_insert(Expression::And(value_sub_phi, value_phi1));
            assert_eq!(leaders[0].len(), 3);
            assert_eq!(*leaders[0].get(&value_3).unwrap(), 6);
            assert_eq!(*leaders[0].get(&value_5).unwrap(), 7);
            assert_eq!(*leaders[0].get(&value_sum).unwrap(), 8);
            assert_eq!(leaders[1].len(), 4);
            assert_eq!(leaders[2].len(), 6);
            assert_eq!(*leaders[2].get(&value_3).unwrap(), 6);
            assert_eq!(*leaders[2].get(&value_5).unwrap(), 7);
            assert_eq!(*leaders[2].get(&value_sum).unwrap(), 8);
            assert_eq!(*leaders[2].get(&value_sub_phi).unwrap(), 10);
            assert_eq!(*leaders[2].get(&value_xor).unwrap(), 11);
            assert_eq!(leaders[3].len(), 6);
            assert_eq!(*leaders[3].get(&value_3).unwrap(), 6);
            assert_eq!(*leaders[3].get(&value_5).unwrap(), 7);
            assert_eq!(*leaders[3].get(&value_sum).unwrap(), 8);
            assert_eq!(*leaders[3].get(&value_sub_phi).unwrap(), 12);
            assert_eq!(*leaders[3].get(&value_and).unwrap(), 13);
            assert_eq!(leaders[4].len(), 5);
            assert_eq!(*leaders[3].get(&value_3).unwrap(), 6);
            assert_eq!(*leaders[3].get(&value_5).unwrap(), 7);
            assert_eq!(*leaders[3].get(&value_sum).unwrap(), 8);
            assert_eq!(*leaders[3].get(&value_sub_phi).unwrap(), 12);

            assert_eq!(
                Vec::from_iter(antic_in[0].iter().cloned()),
                &[
                    (value_3, Expression::Immediate(3)),
                    (value_5, Expression::Immediate(5)),
                    (value_sum, Expression::Plus(value_3, value_5)),
                    (value_sub, Expression::Sub(value_sum, value_5)),
                ]
            );
            assert_eq!(
                Vec::from_iter(antic_in[1].iter().cloned()),
                &[
                    (value_phi1, Expression::Reg(9)),
                    (value_5, Expression::Reg(7)),
                    (value_sub_phi, Expression::Sub(value_phi1, value_5)),
                ]
            );
            assert_eq!(
                Vec::from_iter(antic_in[2].iter().cloned()),
                &[
                    (value_phi1, Expression::Reg(9)),
                    (value_5, Expression::Reg(7)),
                    (value_sub_phi, Expression::Sub(value_phi1, value_5)),
                    (value_xor, Expression::Xor(value_phi1, value_sub_phi)),
                    (
                        table.maybe_insert(Expression::Sub(value_xor, value_5)),
                        Expression::Sub(value_xor, value_5)
                    ),
                ]
            );
            assert_eq!(
                Vec::from_iter(antic_in[3].iter().cloned()),
                &[
                    (value_phi1, Expression::Reg(9)),
                    (value_5, Expression::Reg(7)),
                    (value_sub_phi, Expression::Sub(value_phi1, value_5)),
                    (value_and, Expression::And(value_phi1, value_sub_phi)),
                    (
                        table.maybe_insert(Expression::Sub(value_and, value_5)),
                        Expression::Sub(value_and, value_5)
                    ),
                ]
            );
            assert_eq!(
                Vec::from_iter(antic_in[4].iter().cloned()),
                &[
                    (value_phi2, Expression::Reg(14)),
                    (value_5, Expression::Reg(7)),
                    (
                        table.maybe_insert(Expression::Sub(value_phi2, value_5)),
                        Expression::Sub(value_phi2, value_5)
                    ),
                ]
            );
        }
    }
}
