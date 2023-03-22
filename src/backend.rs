pub mod register_allocation {
    use std::collections::{HashMap, HashSet};

    use crate::ir::{Operator, SSAOperator, VReg, VRegGenerator, CFG};
    use crate::util;

    pub trait Allocator<NR> {
        fn allocate(ssa: CFG<SSAOperator>) -> HashMap<VReg, NR>;
    }

    pub struct RISCV64 {}
    pub enum RV64Reg {
        PLACEHOLDER,
    }

    impl RV64Reg {
        fn get_param_reg(n: u64) -> Option<RV64Reg> {
            Some(Self::PLACEHOLDER)
        }
    }

    struct InterferenceNode {
        live_range: VReg,
        edge_with: Vec<usize>,
    }
    struct InterferenceGraph {
        nodes: Vec<InterferenceNode>,
        index: HashMap<VReg, usize>,
    }

    impl InterferenceGraph {
        fn new() -> Self {
            Self {
                nodes: Vec::new(),
                index: HashMap::new(),
            }
        }
        fn insert(&mut self, vreg: VReg) -> Option<&InterferenceNode> {
            if let Some(&index) = self.index.get(&vreg) {
                return Some(&self.nodes[index]);
            }
            self.index.insert(vreg, self.nodes.len());
            self.nodes.push(InterferenceNode {
                live_range: vreg,
                edge_with: Vec::new(),
            });
            None
        }
        fn get_mut(&mut self, vreg: &VReg) -> Option<&mut InterferenceNode> {
            self.index
                .get(vreg)
                .and_then(|index| self.nodes.get_mut(*index))
        }
        fn get(&self, vreg: &VReg) -> Option<&InterferenceNode> {
            self.index
                .get(vreg)
                .and_then(|index| self.nodes.get(*index))
        }
        fn add_edge(&mut self, vreg1: VReg, vreg2: VReg) {
            if !self.index.contains_key(&vreg1) {
                self.insert(vreg1);
            }
            if !self.index.contains_key(&vreg2) {
                self.insert(vreg2);
            }
            let (&index1, &index2) = (
                self.index.get(&vreg1).unwrap(),
                self.index.get(&vreg2).unwrap(),
            );
            if !self.nodes[index1].edge_with.contains(&index2) {
                self.nodes[index1].edge_with.push(index2);
                self.nodes[index2].edge_with.push(index1);
            }
        }
    }

    impl InterferenceGraph {
        fn to_dot(&self) -> String {
            self.to_dot_colored(None)
        }
        fn to_dot_colored(&self, colors: Option<Vec<u64>>) -> String {
            let max_color = colors
                .as_deref()
                .and_then(|vec| vec.iter().max().cloned())
                .map(|n| n + 1);
            let mut interferences = String::new();
            let mut attributes = String::new();
            for (i, node) in self.nodes.iter().enumerate() {
                if let Some(ref vec) = colors {
                    attributes.push_str(&format!(
                        "{}[style=filled; fillcolor=\"{},1.0,1.0\"]\n",
                        node.live_range,
                        vec[i] as f64 / max_color.unwrap() as f64
                    ));
                }
                for node2 in &node.edge_with {
                    interferences.push_str(&format!(
                        "{} -- {}\n",
                        node.live_range, self.nodes[*node2].live_range
                    ));
                }
            }
            format!(
                "
strict graph G {{
{attributes}
{interferences}
}}"
            )
        }
        fn find_coloring(
            &self,
            max_colors: usize,
            pins: HashMap<VReg, u64>,
        ) -> Result<Vec<u64>, ()> {
            use z3::ast::*;
            let mut config = z3::Config::new();
            config.set_model_generation(true);
            let context = z3::Context::new(&config);
            let solver = z3::Solver::new(&context);

            let nodes_z3 = self
                .nodes
                .iter()
                .map(|node| Int::new_const(&context, node.live_range.to_string()))
                .collect::<Vec<_>>();

            let nodes_native_regs: Vec<_> = (0..max_colors)
                .into_iter()
                .map(|num| Int::new_const(&context, format!("REG_{num}")))
                .collect();

            for i in 0..max_colors {
                for k in (i + 1)..max_colors {
                    // all regs distinct, complete subgraph
                    solver.assert(
                        &nodes_native_regs[i]
                            ._safe_eq(&nodes_native_regs[k])
                            .unwrap()
                            .not(),
                    );
                }
            }
            for (i, node) in self.nodes.iter().enumerate() {
                if let Some(reg) = pins.get(&node.live_range) {
                    solver.assert(
                        &nodes_native_regs[*reg as usize]
                            ._safe_eq(&nodes_z3[i])
                            .unwrap(),
                    );
                }
            }

            let max = Int::from_u64(&context, max_colors as u64);
            let min = Int::from_u64(&context, 0 as u64);
            for node in &nodes_z3 {
                solver.assert(&node.lt(&max));
                solver.assert(&node.ge(&min));
            }

            let mut added_edges = HashSet::new();
            for edge in self
                .nodes
                .iter()
                .enumerate()
                .flat_map(|(i, node)| std::iter::repeat(i).zip(node.edge_with.iter().cloned()))
            {
                if !added_edges.contains(&edge) {
                    let (n1, n2) = edge;
                    added_edges.insert((n2, n1));
                    solver.assert(&nodes_z3[n1]._safe_eq(&nodes_z3[n2]).unwrap().not());
                }
            }

            if !matches!(solver.check(), z3::SatResult::Sat) {
                return Err(());
            }
            let model = solver.get_model().unwrap();

            Ok(nodes_z3
                .into_iter()
                .map(|node| model.eval(&node, true).unwrap().as_u64().unwrap())
                .collect())
        }
    }

    //todo could be made more efficient
    fn spill_liverange(
        cfg: &mut CFG<Operator>,
        live_out: &mut [HashSet<VReg>],
        live_range: VReg,
        ar_offset: u64,
    ) {
        let mut gen = VRegGenerator::starting_at_reg(cfg.get_max_reg());
        for (i, block) in cfg.get_blocks_mut().into_iter().enumerate() {
            live_out[i].remove(&live_range);
            block.body = std::mem::take(&mut block.body)
                .into_iter()
                .flat_map(|mut op| {
                    let new_lr = gen.next_reg();
                    let dependencies = op.dependencies_mut();
                    let mut res = if dependencies.contains(&&mut live_range.clone()) {
                        vec![Operator::LoadLocal(new_lr, ar_offset)]
                    } else {
                        vec![]
                    };
                    dependencies.into_iter().for_each(|reg| {
                        if reg == &live_range {
                            *reg = new_lr
                        }
                    });
                    let new_lr = gen.next_reg();
                    if let Some(lr) = op.receiver_mut() {
                        if lr == &live_range {
                            *lr = new_lr;
                            res.append(&mut vec![op, Operator::StoreLocal(new_lr, ar_offset)]);
                            return res;
                        }
                    }
                    res.push(op);
                    res
                })
                .collect();
        }
    }

    impl RISCV64 {
        //todo refactor some general fns out of RV64
        fn pin_liveranges(
            cfg: &mut CFG<Operator>,
            live_out: &mut [HashSet<VReg>],
        ) -> HashMap<VReg, RV64Reg> {
            let mut res = HashMap::new();
            let mut done = false;
            while !done {
                done = true;
                let mut spill = None;
                for block in cfg.get_blocks() {
                    for op in block.body.iter() {
                        match op {
                            Operator::GetParameter(x, n) => {
                                if res.contains_key(x) {
                                    spill = Some(*x);
                                    break;
                                }
                                res.insert(*x, RV64Reg::get_param_reg(*n).unwrap());
                                // todo add support for params > n
                            }
                            Operator::Call(_, _, ops) => {
                                for (i, op) in ops.iter().enumerate() {
                                    if res.contains_key(op) {
                                        spill = Some(*op);
                                        break;
                                    }
                                    res.insert(*op, RV64Reg::get_param_reg(i as u64).unwrap());
                                }
                            }
                            _ => {}
                        }
                    }
                }
                if let Some(lr) = spill {
                    let ar_offset = *cfg.get_allocated_ars_mut();
                    spill_liverange(cfg, live_out, lr, ar_offset);
                    *cfg.get_allocated_ars_mut() += 1;
                    done = false;
                }
            }
            res
        }
        fn rewrite_liveranges(
            mut ssa: CFG<SSAOperator>,
            live_out: &mut [HashSet<VReg>],
        ) -> CFG<Operator> {
            let mut union_find = util::UnionFind::new();
            let mut new_blocks = Vec::with_capacity(ssa.len());
            for block in ssa.get_blocks() {
                let mut ops = block.body.iter();
                while let Some(SSAOperator::Phi(rec, operands)) = ops.next() {
                    union_find.new_set(*rec);
                    for &op in operands {
                        union_find.new_set(op);
                        union_find.union(rec, &op);
                    }
                }
            }

            for block in live_out.iter_mut() {
                *block = std::mem::take(block)
                    .into_iter()
                    .map(|reg| union_find.find(&reg).cloned().unwrap_or(reg))
                    .collect::<HashSet<u32>>();
            }

            for mut block in std::mem::take(&mut ssa.blocks) {
                let mut old = std::mem::take(&mut block.body);
                for op in old.iter_mut() {
                    macro_rules! rewrite {
                        ($($x:expr),+) => {
                            {
                                $(if let Some(leader) = union_find.find($x) {
                                    *$x = *leader;
                                })+
                            }
                        };
                    }
                    match op {
                        SSAOperator::IROp(op_) => match op_ {
                            crate::ir::Operator::Add(x, y, z) => rewrite!(x, y, z),
                            crate::ir::Operator::Sub(x, y, z) => rewrite!(x, y, z),
                            crate::ir::Operator::Mult(x, y, z) => rewrite!(x, y, z),
                            crate::ir::Operator::Div(x, y, z) => rewrite!(x, y, z),
                            crate::ir::Operator::And(x, y, z) => rewrite!(x, y, z),
                            crate::ir::Operator::Or(x, y, z) => rewrite!(x, y, z),
                            crate::ir::Operator::Mv(x, y) => rewrite!(x, y),
                            crate::ir::Operator::Xor(x, y, z) => rewrite!(x, y, z),
                            crate::ir::Operator::Load(x, y, z) => rewrite!(x, y, z),
                            crate::ir::Operator::Store(x, y, z) => rewrite!(x, y, z),
                            crate::ir::Operator::La(x, _) => rewrite!(x),
                            crate::ir::Operator::Bgt(x, y, _, _) => rewrite!(x, y),
                            crate::ir::Operator::Bl(x, y, _, _) => rewrite!(x, y),
                            crate::ir::Operator::Beq(x, y, _, _) => rewrite!(x, y),
                            crate::ir::Operator::Li(x, _) => rewrite!(x),
                            crate::ir::Operator::Slt(x, y, z) => rewrite!(x, y, z),
                            crate::ir::Operator::Call(x, _, z) => {
                                rewrite!(x);
                                for op in z {
                                    rewrite!(op);
                                }
                            }
                            crate::ir::Operator::Return(x) => rewrite!(x),
                            crate::ir::Operator::StoreLocal(x, _) => rewrite!(x),
                            crate::ir::Operator::LoadLocal(x, _) => rewrite!(x),
                            crate::ir::Operator::GetParameter(x, _) => rewrite!(x),
                            _ => {}
                        },
                        SSAOperator::Phi(..) => {}
                    }
                }
                let new: Vec<_> = old
                    .into_iter()
                    .filter_map(|op| match op {
                        SSAOperator::IROp(op) => Some(op),
                        SSAOperator::Phi(_, _) => None,
                    })
                    .collect();
                new_blocks.push(block.into_other(new));
            }
            ssa.into_other(new_blocks)
        }
        fn build_interference_graph(
            cfg: &CFG<Operator>,
            live_out: &[HashSet<VReg>],
        ) -> InterferenceGraph {
            let mut graph = InterferenceGraph::new();
            for (i, block) in cfg.get_blocks().iter().enumerate() {
                let mut live_now = live_out[i].clone();
                for op in block.body.iter() {
                    if let Some(rec) = op.receiver() {
                        live_now.remove(&rec);
                        live_now.iter().for_each(|&lr| graph.add_edge(lr, rec));
                    }
                    live_now.extend(op.dependencies());
                }
            }
            graph
        }
    }

    impl Allocator<RV64Reg> for RISCV64 {
        fn allocate(ssa: CFG<SSAOperator>) -> HashMap<VReg, RV64Reg> {
            let mut live_out = crate::ssa::liveness::live_out(&ssa);
            let mut lr_cfg = RISCV64::rewrite_liveranges(ssa, &mut live_out);
            let pins: Vec<_> = RISCV64::pin_liveranges(&mut lr_cfg, &mut live_out)
                .into_iter()
                .map(|(lr, reg)| (lr, reg as u64))
                .collect();

            todo!()
        }
    }
    #[cfg(test)]
    mod tests {
        use std::collections::HashMap;

        use crate::ir::CFG;

        #[test]
        fn construct_graph() {
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
                i = i - 1;
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
            crate::ssa::optimization_sequence(&mut ssa).unwrap();

            let mut live_out = crate::ssa::liveness::live_out(&ssa);
            let cfg = super::RISCV64::rewrite_liveranges(ssa, &mut live_out);
            println!("live ranges: {}", cfg.to_dot());
            let graph = super::RISCV64::build_interference_graph(&cfg, &live_out);

            println!("{}", graph.to_dot());
            let expected = [
                (26, 24),
                (26, 25),
                (26, 53),
                (26, 30),
                (26, 32),
                (26, 37),
                (26, 39),
                (26, 41),
                (26, 43),
                (24, 26),
                (24, 53),
                (24, 25),
                (53, 24),
                (53, 25),
                (53, 26),
                (53, 30),
                (53, 32),
                (53, 43),
                (25, 24),
                (25, 26),
                (25, 53),
                (25, 30),
                (25, 32),
                (25, 34),
                (25, 35),
                (25, 37),
                (25, 39),
                (25, 41),
                (25, 43),
                (30, 26),
                (30, 25),
                (30, 34),
                (30, 53),
                (34, 30),
                (34, 32),
                (34, 25),
                (34, 37),
                (34, 39),
                (34, 41),
                (32, 26),
                (32, 25),
                (32, 34),
                (32, 53),
                (35, 25),
                (35, 39),
                (35, 41),
                (37, 25),
                (37, 26),
                (37, 34),
                (37, 41),
                (39, 25),
                (39, 26),
                (39, 34),
                (39, 35),
                (41, 25),
                (41, 26),
                (41, 34),
                (41, 35),
                (41, 37),
                (43, 26),
                (43, 25),
                (43, 53),
            ];
            for (n1_, n2) in expected {
                let (n1, n2) = (graph.get(&n1_).unwrap(), graph.index.get(&n2).unwrap());
                assert!(
                    n1.edge_with.contains(n2),
                    "node {n1_} does not connect to node {n2}\n"
                );
            }

            let coloring = graph.find_coloring(4, HashMap::default());
            assert!(coloring.is_ok());
            println!(
                "found coloring:\n{}",
                graph.to_dot_colored(Some(coloring.unwrap()))
            );
        }
    }
}
