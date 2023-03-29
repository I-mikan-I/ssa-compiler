pub mod register_allocation {
    use std::cmp::max_by_key;
    use std::collections::{HashMap, HashSet};
    use std::fmt::Display;

    use crate::ir::{Operator, SSAOperator, VReg, VRegGenerator, CFG};
    use crate::util;

    pub trait Allocator<NR> {
        fn allocate(ssa: CFG<SSAOperator>) -> (CFG<Operator>, HashMap<VReg, NR>);
        fn add_procedure_prologues(cfg: &mut CFG<Operator>, allocs: &HashMap<VReg, NR>);
    }

    pub trait RegisterSet: Sized + Into<usize> + TryFrom<usize> + Clone {
        type AllocationError;
        fn max_regs() -> usize;
        fn from_colors(
            colors: &[u64],
            pins: &HashMap<u64, Self>,
        ) -> Result<Vec<Self>, Self::AllocationError>;
    }

    pub struct RISCV64 {}

    #[repr(usize)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum RV64Reg {
        X0 = 0,   // zero
        X1 = 1,   // ra
        X2 = 2,   // sp
        X3 = 3,   // gp
        X4 = 4,   // tp
        X5 = 5,   // t0
        X6 = 6,   // t1
        X7 = 7,   // t2
        X8 = 8,   // s0
        X9 = 9,   // s1
        X10 = 10, // a0
        X11 = 11,
        X12 = 12,
        X13 = 13,
        X14 = 14,
        X15 = 15,
        X16 = 16,
        X17 = 17, // a7
        X18 = 18, // s2
        X19 = 19,
        X20 = 20,
        X21 = 21,
        X22 = 22,
        X23 = 23,
        X24 = 24,
        X25 = 25,
        X26 = 26,
        X27 = 27, // s11
        X28 = 28, // t3
        X29 = 29,
        X30 = 30,
        X31 = 31, // t6
    }
    impl Display for RV64Reg {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                RV64Reg::X0 => write!(f, "x0"),
                RV64Reg::X1 => write!(f, "x1"),
                RV64Reg::X2 => write!(f, "x2"),
                RV64Reg::X3 => write!(f, "x3"),
                RV64Reg::X4 => write!(f, "x4"),
                RV64Reg::X5 => write!(f, "x5"),
                RV64Reg::X6 => write!(f, "x6"),
                RV64Reg::X7 => write!(f, "x7"),
                RV64Reg::X8 => write!(f, "x8"),
                RV64Reg::X9 => write!(f, "x9"),
                RV64Reg::X10 => write!(f, "x10"),
                RV64Reg::X11 => write!(f, "x11"),
                RV64Reg::X12 => write!(f, "x12"),
                RV64Reg::X13 => write!(f, "x13"),
                RV64Reg::X14 => write!(f, "x14"),
                RV64Reg::X15 => write!(f, "x15"),
                RV64Reg::X16 => write!(f, "x16"),
                RV64Reg::X17 => write!(f, "x17"),
                RV64Reg::X18 => write!(f, "x18"),
                RV64Reg::X19 => write!(f, "x19"),
                RV64Reg::X20 => write!(f, "x20"),
                RV64Reg::X21 => write!(f, "x21"),
                RV64Reg::X22 => write!(f, "x22"),
                RV64Reg::X23 => write!(f, "x23"),
                RV64Reg::X24 => write!(f, "x24"),
                RV64Reg::X25 => write!(f, "x25"),
                RV64Reg::X26 => write!(f, "x26"),
                RV64Reg::X27 => write!(f, "x27"),
                RV64Reg::X28 => write!(f, "x28"),
                RV64Reg::X29 => write!(f, "x29"),
                RV64Reg::X30 => write!(f, "x30"),
                RV64Reg::X31 => write!(f, "x31"),
            }
        }
    }
    impl RV64Reg {
        const ALLOCATION_ORDER: [Self; 24] = [
            Self::X5,
            Self::X6,
            Self::X7,
            Self::X28,
            Self::X29,
            Self::X30,
            Self::X31,
            Self::X10,
            Self::X11,
            Self::X12,
            Self::X13,
            Self::X17,
            Self::X8,
            Self::X9,
            Self::X18,
            Self::X19,
            Self::X20,
            Self::X21,
            Self::X22,
            Self::X23,
            Self::X24,
            Self::X25,
            Self::X26,
            Self::X27,
        ];
        fn callee_saved(&self) -> bool {
            #[allow(clippy::match_like_matches_macro)]
            match self {
                RV64Reg::X8
                | RV64Reg::X9
                | RV64Reg::X18
                | RV64Reg::X19
                | RV64Reg::X20
                | RV64Reg::X21
                | RV64Reg::X22
                | RV64Reg::X23
                | RV64Reg::X24
                | RV64Reg::X25
                | RV64Reg::X26
                | RV64Reg::X27 => true,
                _ => false,
            }
        }
        fn get_param_reg(n: u64) -> Option<RV64Reg> {
            use RV64Reg::*;
            match n {
                0 => Some(X10),
                1 => Some(X11),
                2 => Some(X12),
                3 => Some(X13),
                4 => Some(X14),
                5 => Some(X15),
                6 => Some(X16),
                7 => Some(X17),
                _ => None,
            }
        }
    }
    impl From<RV64Reg> for usize {
        fn from(reg: RV64Reg) -> Self {
            reg as usize
        }
    }
    impl RegisterSet for RV64Reg {
        type AllocationError = ();

        fn max_regs() -> usize {
            24
        }

        fn from_colors(
            colors: &[u64],
            pins: &HashMap<u64, Self>,
        ) -> Result<Vec<Self>, Self::AllocationError> {
            let mut avail = Self::ALLOCATION_ORDER
                .into_iter()
                .filter(|k| !pins.values().any(|k2| k2 == k));
            colors
                .iter()
                .map(|color| pins.get(color).cloned().or_else(|| avail.next()))
                .collect::<Option<Vec<Self>>>()
                .ok_or(())
        }
    }

    impl std::convert::TryFrom<usize> for RV64Reg {
        type Error = &'static str;

        fn try_from(value: usize) -> Result<Self, Self::Error> {
            use RV64Reg::*;
            match value {
                0 => Ok(X0),
                1 => Ok(X1),
                2 => Ok(X2),
                3 => Ok(X3),
                4 => Ok(X4),
                5 => Ok(X5),
                6 => Ok(X6),
                7 => Ok(X7),
                8 => Ok(X8),
                9 => Ok(X9),
                10 => Ok(X10),
                11 => Ok(X11),
                12 => Ok(X12),
                13 => Ok(X13),
                14 => Ok(X14),
                15 => Ok(X15),
                16 => Ok(X16),
                17 => Ok(X17),
                18 => Ok(X18),
                19 => Ok(X19),
                20 => Ok(X20),
                21 => Ok(X21),
                22 => Ok(X22),
                23 => Ok(X23),
                24 => Ok(X24),
                25 => Ok(X25),
                26 => Ok(X26),
                27 => Ok(X27),
                28 => Ok(X28),
                29 => Ok(X29),
                30 => Ok(X30),
                31 => Ok(X31),
                _ => Err("register number too high"),
            }
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
        fn merge(&mut self, into: VReg, from: VReg) {
            if !self.index.contains_key(&into) {
                self.maybe_insert(into);
            }
            if !self.index.contains_key(&from) {
                self.maybe_insert(from);
            }
            let into_node = self.index[&into];
            let from_node = self.index[&from];
            for edge in std::mem::take(&mut self.nodes[from_node].edge_with) {
                self.nodes[edge].edge_with = std::mem::take(&mut self.nodes[edge].edge_with)
                    .into_iter()
                    .filter(|&n| n != from_node && n != into_node)
                    .collect();
                self.nodes[edge].edge_with.push(into_node);
                self.nodes[into_node].edge_with.push(edge);
            }
            self.nodes[into_node].edge_with.sort();
            self.nodes[into_node].edge_with.dedup();
            self.index.remove(&from); // garbage left
        }
        fn maybe_insert(&mut self, vreg: VReg) -> Option<&InterferenceNode> {
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
        fn check_edge(&self, vreg1: &VReg, vreg2: &VReg) -> bool {
            if let (Some(index), Some(index2)) = (self.index.get(vreg1), self.index.get(vreg2)) {
                self.nodes[*index].edge_with.contains(index2)
            } else {
                false
            }
        }
        fn add_edge(&mut self, vreg1: VReg, vreg2: VReg) {
            if !self.index.contains_key(&vreg1) {
                self.maybe_insert(vreg1);
            }
            if !self.index.contains_key(&vreg2) {
                self.maybe_insert(vreg2);
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
            self.to_dot_colored::<RV64Reg>(None)
        }
        fn to_dot_colored<R: RegisterSet>(&self, colors: Option<Vec<R>>) -> String {
            let max_color = colors
                .as_deref()
                .and_then(|vec| vec.iter().map(|c| c.clone().into()).max())
                .map(|n| n + 1);
            let mut interferences = String::new();
            let mut attributes = String::new();
            for (i, node) in self.nodes.iter().enumerate() {
                if let Some(ref vec) = colors {
                    attributes.push_str(&format!(
                        "{}[style=filled; fillcolor=\"{},1.0,1.0\"]\n",
                        node.live_range,
                        vec[i].clone().into() as f64 / max_color.unwrap() as f64
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
        fn find_coloring<R: RegisterSet>(&self, pins: &HashMap<VReg, R>) -> Result<Vec<R>, ()> {
            use z3::ast::*;
            let max_colors = R::max_regs();
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
                        &nodes_native_regs[reg.clone().into()]
                            ._safe_eq(&nodes_z3[i])
                            .unwrap(),
                    );
                }
            }

            let max = Int::from_u64(&context, max_colors as u64);
            let min = Int::from_u64(&context, 0);
            for node in nodes_z3.iter().chain(nodes_native_regs.iter()) {
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

            let colors = nodes_z3
                .iter()
                .map(|node| model.eval(node, true).unwrap().as_u64().unwrap())
                .collect::<Vec<_>>();
            let color_pins = pins
                .iter()
                .map(|(k, v)| (colors[self.index[k]], v.clone()))
                .collect();

            R::from_colors(&colors, &color_pins).map_err(|_| ())
        }
    }

    //todo could be made more efficient
    fn spill_liverange(cfg: &mut CFG<Operator>, live_range: VReg, ar_offset: u64) {
        let mut gen = VRegGenerator::starting_at_reg(cfg.get_max_reg());
        for block in cfg.get_blocks_mut().iter_mut() {
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
        cfg.set_max_reg(gen.next_reg());
    }
    fn build_interference_graph(
        cfg: &CFG<Operator>,
    ) -> (
        InterferenceGraph,
        HashSet<(VReg, VReg)>,
        HashMap<VReg, usize>,
    ) {
        let mut graph = InterferenceGraph::new();
        let mut coalescable = HashSet::new();
        let mut spill_weights = HashMap::new();
        let live_out = cfg.live_out();
        for (i, block) in cfg.get_blocks().iter().enumerate() {
            let mut live_now = live_out[i].clone();
            for op in block.body.iter().rev() {
                for &live in &live_now {
                    *spill_weights.entry(live).or_insert(0_usize) += 1;
                }
                if let Some(rec) = op.receiver() {
                    live_now.remove(&rec);
                    if let Operator::Mv(lr1, lr2) = op {
                        coalescable.insert((*lr1, *lr2));
                        live_now
                            .iter()
                            .filter(|&lr| lr != lr2)
                            .for_each(|&lr| graph.add_edge(lr, rec));
                    } else {
                        live_now.iter().for_each(|&lr| graph.add_edge(lr, rec));
                    }
                }
                live_now.extend(op.dependencies());
            }
        }
        (graph, coalescable, spill_weights)
    }
    /// no critical edges allowed
    pub fn conventionalize_ssa(ssa: &mut CFG<SSAOperator>) {
        let mut parallel_copies = vec![Vec::new(); ssa.len()];
        let mut gen = VRegGenerator::starting_at_reg(ssa.get_max_reg());
        for block in ssa.get_blocks_mut().iter_mut() {
            let mut ops = block.body.iter_mut();
            while let Some(SSAOperator::Phi(_, vec)) = ops.next() {
                let new_args = std::iter::repeat_with(|| gen.next_reg())
                    .take(vec.len())
                    .collect::<Vec<_>>();
                for (i, pred) in block.preds.iter().enumerate() {
                    if vec[i] == u32::MAX {
                        continue; // dead phi function
                    }
                    parallel_copies[*pred].push(Operator::Mv(new_args[i], vec[i]));
                }
                *vec = new_args;
            }
        }

        for (i, mut copies) in parallel_copies.into_iter().enumerate() {
            let len = ssa.get_block(i).body.len();
            // found copy to non-live name
            while !copies.is_empty() {
                if let Some(op) = {
                    let mut choices = HashMap::new();
                    let mut iter = copies.iter().enumerate();
                    while let Some((i, Operator::Mv(rec, _))) = iter.next() {
                        choices.insert(rec, i);
                    }
                    let mut iter = copies.iter();
                    while let Some(Operator::Mv(_, op)) = iter.next() {
                        choices.remove(&op);
                    }
                    choices
                        .values()
                        .next()
                        .cloned()
                        .map(|index| copies.remove(index))
                } {
                    ssa.get_block_mut(i)
                        .body
                        .insert(len - 1, SSAOperator::IROp(op.clone()));
                } else {
                    // break cycle
                    if let Some(Operator::Mv(_, op)) = copies.last_mut() {
                        let new_name = gen.next_reg();
                        ssa.get_block_mut(i)
                            .body
                            .insert(len - 1, SSAOperator::IROp(Operator::Mv(new_name, *op)));
                        *op = new_name;
                    }
                }
            }
        }
        #[cfg(feature = "print-cfgs")]
        {
            println!("after conventionalizing SSA:{}\n", ssa.to_dot());
        }
        ssa.set_max_reg(gen.next_reg());
    }
    fn rewrite_liveranges(mut ssa: CFG<SSAOperator>) -> CFG<Operator> {
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
    impl RISCV64 {
        //todo refactor some general fns out of RV64
        fn pin_liveranges(cfg: &mut CFG<Operator>) -> (HashMap<VReg, RV64Reg>, bool) {
            let mut spilled = false;
            let mut gen = VRegGenerator::starting_at_reg(cfg.get_max_reg());
            let mut res = HashMap::new();
            for block in cfg.get_blocks_mut() {
                let mut i = 0;
                while i < block.body.len() {
                    match &mut block.body[i] {
                        Operator::GetParameter(x, n) => {
                            let pin = RV64Reg::get_param_reg(*n).unwrap();
                            if let Some(pinned) = res.get(x) {
                                if pinned != &pin {
                                    spilled = true;
                                    let next = gen.next_reg();
                                    let old = *x;
                                    *x = next;
                                    block.body.insert(i + 1, Operator::Mv(old, next));
                                    res.insert(next, pin);
                                }
                            } else {
                                res.insert(*x, pin);
                            }
                        }
                        Operator::Call(rec, _, ops) => {
                            let pin = RV64Reg::get_param_reg(0).unwrap();
                            let mut post = None;
                            let mut before = None;
                            if let Some(pinned) = res.get(rec) {
                                if pinned != &pin {
                                    spilled = true;
                                    let next = gen.next_reg();
                                    let old = *rec;
                                    *rec = next;
                                    post = Some(Operator::Mv(old, next));
                                    res.insert(next, pin);
                                }
                            } else {
                                res.insert(*rec, pin);
                            }
                            for (i, op) in ops.iter_mut().enumerate() {
                                let pin = RV64Reg::get_param_reg(i as u64).unwrap();
                                if let Some(pinned) = res.get(op) {
                                    if pinned != &pin {
                                        spilled = true;
                                        let next = gen.next_reg();
                                        let old = *op;
                                        *op = next;
                                        before = Some(Operator::Mv(next, old));
                                        res.insert(next, pin);
                                    }
                                } else {
                                    res.insert(*op, pin);
                                }
                            }
                            if let Some(operation) = post {
                                block.body.insert(i + 1, operation)
                            }
                            if let Some(operation) = before {
                                block.body.insert(i, operation)
                            }
                        }
                        Operator::Return(op) => {
                            let pin = RV64Reg::get_param_reg(0).unwrap();
                            if let Some(pinned) = res.get(op) {
                                if pinned != &pin {
                                    spilled = true;
                                    let next = gen.next_reg();
                                    let old = *op;
                                    *op = next;
                                    block.body.insert(i + 1, Operator::Mv(old, next));
                                    res.insert(next, pin);
                                }
                            } else {
                                res.insert(*op, pin);
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
            }
            cfg.set_max_reg(gen.next_reg());
            (res, spilled)
        }
    }

    impl Allocator<RV64Reg> for RISCV64 {
        fn allocate(ssa: CFG<SSAOperator>) -> (CFG<Operator>, HashMap<VReg, RV64Reg>) {
            let mut lr_cfg = rewrite_liveranges(ssa);

            let (graph, coloring) = 'build_allocate: loop {
                let (mut graph, coalescable, mut spill_weights) = build_interference_graph(&lr_cfg);
                // coalesce build loop
                for (lr1, lr2) in coalescable.into_iter() {
                    if graph.check_edge(&lr1, &lr2) {
                        continue;
                    } else {
                        graph.merge(lr2, lr1);
                        // todo make more efficient (i.e. all coalescing in one pass)
                        for block in lr_cfg.get_blocks_mut() {
                            let mut i = 0;
                            while i < block.body.len() {
                                if let Operator::Mv(left, right) = block.body[i] {
                                    if left == lr1 && right == lr2 {
                                        block.body.remove(i);
                                        continue;
                                    }
                                }
                                let op = &mut block.body[i];
                                if let Some(rec) = op.receiver_mut() {
                                    if *rec == lr1 {
                                        *rec = lr2;
                                    }
                                }
                                for op in op.dependencies_mut() {
                                    if *op == lr1 {
                                        *op = lr2;
                                    }
                                }
                                i += 1;
                            }
                        }
                        continue 'build_allocate;
                    }
                }
                let (pins, rebuild_graph) = RISCV64::pin_liveranges(&mut lr_cfg);
                if rebuild_graph {
                    (graph, _, spill_weights) = build_interference_graph(&lr_cfg);
                }
                for lr in pins.keys() {
                    graph.maybe_insert(*lr);
                }
                for (reg1, value1) in &pins {
                    for (reg2, value2) in &pins {
                        if value1 == value2
                            && graph
                                .get(reg1)
                                .and_then(|n1| {
                                    graph.index.get(reg2).map(|i2| n1.edge_with.contains(i2))
                                })
                                .unwrap_or(false)
                        {
                            let ar = *lr_cfg.get_allocated_ars_mut();
                            let to_spill = max_by_key(reg1, reg2, |&t| spill_weights[t]);
                            spill_liverange(&mut lr_cfg, *to_spill, ar);
                            *lr_cfg.get_allocated_ars_mut() += 1;
                            continue 'build_allocate;
                        }
                    }
                }
                match graph.find_coloring(&pins) {
                    Err(_) => {
                        println!("conflict! {}", lr_cfg.to_dot());
                        let lr_to_spill = graph
                            .nodes
                            .iter()
                            .max_by_key(|node| spill_weights[&node.live_range])
                            .unwrap()
                            .live_range;
                        let ar = *lr_cfg.get_allocated_ars_mut();
                        spill_liverange(&mut lr_cfg, lr_to_spill, ar);
                        *lr_cfg.get_allocated_ars_mut() += 1;
                    }
                    Ok(coloring) => break (graph, coloring),
                }
            };

            for block in lr_cfg.get_blocks_mut() {
                for op in block.body.iter_mut() {
                    if let Some(rec) = op.receiver_mut() {
                        *rec = graph
                            .index
                            .get(rec)
                            .map(|&idx| <RV64Reg as Into<usize>>::into(coloring[idx]))
                            .unwrap_or(coloring[0].into()) as u32;
                        // unwrap = no interferences at all
                    }
                    for dep in op.dependencies_mut() {
                        *dep = graph
                            .index
                            .get(dep)
                            .map(|&idx| <RV64Reg as Into<usize>>::into(coloring[idx]))
                            .unwrap_or(coloring[0].into()) as u32;
                    }
                }
            }
            #[cfg(feature = "print-cfgs")]
            {
                println!("After register allocation:\n{}", lr_cfg.to_dot());
            }
            (
                lr_cfg,
                coloring
                    .into_iter()
                    .map(|color| (<RV64Reg as Into<usize>>::into(color) as VReg, color))
                    .collect(),
            )
        }

        /// callsites must be isolated (excluding spills);
        fn add_procedure_prologues(cfg: &mut CFG<Operator>, allocs: &HashMap<VReg, RV64Reg>) {
            let ar = *cfg.get_allocated_ars_mut();
            let mut ar_max = ar;
            let live_out = cfg.live_out();
            let callee_saved = allocs.values().filter(|reg| reg.callee_saved());

            for (i, block) in cfg.get_blocks_mut().iter_mut().enumerate() {
                if block.body.iter().any(|op| matches!(op, Operator::Call(..))) {
                    let mut ar_ = ar;
                    debug_assert_eq!(block.preds.len(), 1);
                    let to_save: Vec<_> = live_out[i]
                        .intersection(&live_out[block.preds[0]])
                        .filter(|reg| !allocs[reg].callee_saved())
                        .collect();
                    let mut prologue: Vec<_> = to_save
                        .iter()
                        .map(|&&reg| {
                            let res = Operator::StoreLocal(reg, ar_);
                            ar_ += 1;
                            res
                        })
                        .collect();
                    let mut ar_ = ar;
                    let mut epilogue: Vec<_> = to_save
                        .iter()
                        .map(|&&reg| {
                            let res = Operator::LoadLocal(reg, ar_);
                            ar_ += 1;
                            res
                        })
                        .collect();
                    prologue.append(&mut block.body);
                    let jump = prologue.pop();
                    prologue.append(&mut epilogue);
                    if let Some(op) = jump {
                        prologue.push(op)
                    }
                    block.body = prologue;
                    ar_max = std::cmp::max(ar_max, ar_);
                }
            }

            let mut ar = ar_max;
            *cfg.get_allocated_ars_mut() = ar_max;

            for &saved in callee_saved {
                cfg.get_block_mut(cfg.get_entry())
                    .body
                    .insert(0, Operator::StoreLocal(usize::from(saved) as VReg, ar));
                cfg.get_block_mut(cfg.get_exit())
                    .body
                    .push(Operator::LoadLocal(usize::from(saved) as VReg, ar));
                ar += 1;
            }
            *cfg.get_allocated_ars_mut() = ar;
        }
    }
    #[cfg(test)]
    mod tests {
        use std::collections::HashMap;

        use crate::{
            backend::register_allocation::{Allocator, RV64Reg},
            ir::{Operator, CFG},
        };

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
            super::conventionalize_ssa(&mut ssa);

            let cfg = super::rewrite_liveranges(ssa);
            println!("live ranges: {}", cfg.to_dot());
            let (graph, _, _) = super::build_interference_graph(&cfg);

            println!("{}", graph.to_dot());

            let coloring = graph.find_coloring::<RV64Reg>(&HashMap::default());
            assert!(coloring.is_ok());
            println!(
                "found coloring:\n{}",
                graph.to_dot_colored(Some(coloring.unwrap()))
            );
        }
        #[test]
        fn allocate_program() {
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

            let cfg = CFG::from_linear(body, fun.get_params(), fun.get_max_reg());
            let mut ssa = cfg.to_ssa();
            crate::ssa::optimization_sequence(&mut ssa).unwrap();
            super::conventionalize_ssa(&mut ssa);

            let allocation = super::RISCV64::allocate(ssa);
            println!("allocation: {:?}", allocation.1);
            println!("allocated_graph: {}", allocation.0.to_dot());
        }
        #[test]
        fn allocate_program_calls_no_spill() {
            let input = "
        myvar3 :: Bool = false;
        lambda myfun(myvar3 :: Int) :: Int {
            myvar4 :: Int = 0;
            i :: Int = 100;
            while (i >= 0) do {
                myvar4 = myfun(myvar4);
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

            let cfg = CFG::from_linear(body, fun.get_params(), fun.get_max_reg());
            let mut ssa = cfg.to_ssa();
            crate::ssa::optimization_sequence(&mut ssa).unwrap();
            super::conventionalize_ssa(&mut ssa);

            let allocation = super::RISCV64::allocate(ssa);
            assert_eq!(
                allocation.0.blocks[3].body[0],
                Operator::Call(10, "myfun".into(), vec![10])
            );
            assert_eq!(
                allocation.0.blocks[0].body[0],
                Operator::GetParameter(10, 0)
            );
            assert_eq!(allocation.0.blocks[5].body[0], Operator::Return(10));
            println!("allocation: {:?}", allocation.1);
            println!("allocated_graph: {}", allocation.0.to_dot());
        }
        #[test]
        fn allocate_program_calls_spill() {
            let input = "
        myvar3 :: Bool = false;
        lambda myfun(myvar3 :: Int, myvar5 :: Int) :: Int {
            myvar4 :: Int = 0;
            i :: Int = 100;
            while (i >= 0) do {
                myvar4 = myfun(3, myvar4);
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

            let cfg = CFG::from_linear(body, fun.get_params(), fun.get_max_reg());
            let mut ssa = cfg.to_ssa();
            crate::ssa::optimization_sequence(&mut ssa).unwrap();
            super::conventionalize_ssa(&mut ssa);

            let allocation = super::RISCV64::allocate(ssa);
            assert_eq!(
                &allocation.0.blocks[3].body[0..=2],
                &[
                    Operator::LoadLocal(11, 0),
                    Operator::Call(10, "myfun".into(), vec![10, 11]),
                    Operator::StoreLocal(10, 0)
                ]
            );
            assert_eq!(
                allocation.0.blocks[0].body[0],
                Operator::GetParameter(10, 0)
            );
            println!("allocation: {:?}", allocation.1);
            println!("allocated_graph: {}", allocation.0.to_dot());
        }
        #[test]
        fn program_calls_spill_prologues() {
            let input = "
        myvar3 :: Bool = false;
        lambda myfun(myvar3 :: Int, myvar5 :: Int) :: Int {
            myvar4 :: Int = 0;
            i :: Int = 100;
            while (i >= 0) do {
                myvar4 = myfun(3, myvar4);
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

            let cfg = CFG::from_linear(body, fun.get_params(), fun.get_max_reg());
            let mut ssa = cfg.to_ssa();
            crate::ssa::optimization_sequence(&mut ssa).unwrap();
            super::conventionalize_ssa(&mut ssa);

            let mut allocation = super::RISCV64::allocate(ssa);
            super::RISCV64::add_procedure_prologues(&mut allocation.0, &allocation.1);
            assert!(allocation.0.blocks[3]
                .body
                .iter()
                .take(3)
                .all(|op| matches!(op, Operator::StoreLocal(..))));
        }
    }
}

pub mod instruction_selection {
    use std::{
        collections::{HashMap, HashSet, VecDeque},
        fmt::Display,
        rc::Rc,
    };

    use crate::ir::{Operator, VReg, CFG};

    use super::register_allocation::RV64Reg;
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    #[allow(non_camel_case_types)]
    pub enum RV64Operation {
        ADD(RV64Reg, RV64Reg, RV64Reg),
        ADDI(RV64Reg, RV64Reg, i16),
        SUB(RV64Reg, RV64Reg, RV64Reg),
        SQ(RV64Reg, RV64Reg, i16),
        LQ(RV64Reg, RV64Reg, i16),
        AND(RV64Reg, RV64Reg, RV64Reg),
        OR(RV64Reg, RV64Reg, RV64Reg),
        XOR(RV64Reg, RV64Reg, RV64Reg),
        MUL(RV64Reg, RV64Reg, RV64Reg),
        DIV(RV64Reg, RV64Reg, RV64Reg),
        SLT(RV64Reg, RV64Reg, RV64Reg),
        PSEUD_LA(RV64Reg, Rc<str>),
        PSEUD_CALL(Rc<str>),
        PSEUD_RET,
        PSEUD_J(Rc<str>),
        PSEUD_LI(RV64Reg, i64),
        PSEUD_LABEL(Rc<str>),
        BEQ(RV64Reg, RV64Reg, Rc<str>),
        BGE(RV64Reg, RV64Reg, Rc<str>),
        BL(RV64Reg, RV64Reg, Rc<str>),
    }

    impl CFG<RV64Operation> {
        fn linearize(self) -> Vec<RV64Operation> {
            let mut placed = HashSet::new();
            let mut worklist = VecDeque::new();
            worklist.push_back(self.get_block(self.get_entry()));
            let mut result = Vec::new();
            while let Some(to_place) = worklist.pop_front() {
                if placed.contains(&to_place.label) {
                    continue;
                }
                result.push(RV64Operation::PSEUD_LABEL(Rc::clone(&to_place.label)));
                for op in to_place.body.iter() {
                    match op {
                        RV64Operation::PSEUD_J(t) => {
                            if placed.contains(t) {
                                result.push(op.clone());
                            } else {
                                worklist.push_front(
                                    self.blocks.iter().find(|b| &b.label == t).unwrap(),
                                );
                            }
                        }

                        RV64Operation::BEQ(_, _, l)
                        | RV64Operation::BGE(_, _, l)
                        | RV64Operation::BL(_, _, l) => {
                            debug_assert_eq!(to_place.children.len(), 2);
                            let r = self.get_block(to_place.children[1]);
                            result.push(op.clone());
                            if placed.contains(&r.label) {
                                result.push(RV64Operation::PSEUD_J(Rc::clone(&r.label)));
                            } else {
                                worklist.push_front(r);
                            }
                            worklist.push_back(self.blocks.iter().find(|b| &b.label == l).unwrap());
                        }
                        op => {
                            result.push(op.clone());
                        }
                    }
                }
                placed.insert(Rc::clone(&to_place.label));
            }
            result
        }
    }

    impl Display for RV64Operation {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                RV64Operation::ADD(r1, r2, r3) => write!(f, "add {r1},{r2},{r3}"),
                RV64Operation::ADDI(r1, r2, r3) => write!(f, "addi {r1},{r2},{r3}"),
                RV64Operation::SUB(r1, r2, r3) => write!(f, "sub {r1},{r2},{r3}"),
                RV64Operation::SQ(r1, r2, r3) => write!(f, "sd {r1},{r3}({r2})"),
                RV64Operation::LQ(r1, r2, r3) => write!(f, "ld {r1},{r3}({r2})"),
                RV64Operation::AND(r1, r2, r3) => write!(f, "and {r1},{r2},{r3}"),
                RV64Operation::OR(r1, r2, r3) => write!(f, "or {r1},{r2},{r3}"),
                RV64Operation::XOR(r1, r2, r3) => write!(f, "xor {r1},{r2},{r3}"),
                RV64Operation::MUL(r1, r2, r3) => write!(f, "mul {r1},{r2},{r3}"),
                RV64Operation::DIV(r1, r2, r3) => write!(f, "div {r1},{r2},{r3}"),
                RV64Operation::SLT(r1, r2, r3) => write!(f, "slt {r1},{r2},{r3}"),
                RV64Operation::PSEUD_LA(r1, r2) => write!(f, "la {r1},{r2}"),
                RV64Operation::PSEUD_CALL(r1) => write!(f, "call {r1}"),
                RV64Operation::PSEUD_RET => write!(f, "ret"),
                RV64Operation::PSEUD_J(r1) => write!(f, "j .{r1}"),
                RV64Operation::PSEUD_LI(r1, r2) => write!(f, "li {r1},{r2}"),
                RV64Operation::PSEUD_LABEL(l) => write!(f, ".{l}:"),
                RV64Operation::BEQ(r1, r2, r3) => write!(f, "beq {r1},{r2},.{r3}"),
                RV64Operation::BGE(r1, r2, r3) => write!(f, "bge {r1},{r2},.{r3}"),
                RV64Operation::BL(r1, r2, r3) => write!(f, "bl {r1},{r2},.{r3}"),
            }
        }
    }

    pub fn select_instructions(
        mut cfg: CFG<Operator>,
        allocation: &HashMap<VReg, RV64Reg>,
    ) -> CFG<RV64Operation> {
        use RV64Operation::*;
        let mut new_bodies: Vec<Vec<RV64Operation>> = vec![Vec::new(); cfg.len()];
        let exit_label = Rc::clone(&cfg.get_block(cfg.get_exit()).label);

        for (i, block) in cfg.get_blocks_mut().iter_mut().enumerate() {
            let body = std::mem::take(&mut block.body);

            for op in body {
                macro_rules! binary_op {
                    ($x:expr, $y:expr, $z:expr, $p:path) => {{
                        new_bodies[i].push($p(allocation[&$x], allocation[&$y], allocation[&$z]));
                    }};
                }
                macro_rules! get_regs {
                    ($($x:expr),+) => {
                        ($(allocation[&$x]),+)
                    };
                }
                match op {
                    Operator::Add(x, y, z) => binary_op!(x, y, z, ADD),
                    Operator::Sub(x, y, z) => binary_op!(x, y, z, SUB),
                    Operator::Mult(x, y, z) => binary_op!(x, y, z, MUL),
                    Operator::Div(x, y, z) => binary_op!(x, y, z, DIV),
                    Operator::And(x, y, z) => binary_op!(x, y, z, AND),
                    Operator::Or(x, y, z) => binary_op!(x, y, z, OR),
                    Operator::Mv(x, y) => {
                        let (x, y) = get_regs!(x, y);
                        new_bodies[i].push(ADDI(x, y, 0));
                    }
                    Operator::Xor(x, y, z) => binary_op!(x, y, z, XOR),
                    Operator::Load(x, y, z) => {
                        let (x, y, z) = get_regs!(x, y, z);
                        new_bodies[i].push(ADD(x, y, z));
                        new_bodies[i].push(LQ(x, x, 0));
                    }
                    Operator::Store(x, y, z) => {
                        let (x, y, z) = get_regs!(x, y, z);
                        new_bodies[i].push(ADD(y, y, z));
                        new_bodies[i].push(SQ(x, y, 0));
                        new_bodies[i].push(SUB(y, y, z));
                    }
                    Operator::La(x, y) => {
                        let x = get_regs!(x);
                        new_bodies[i].push(PSEUD_LA(x, y));
                    }
                    Operator::Bgt(x, y, z, _) => {
                        let (x, y) = get_regs!(x, y);
                        new_bodies[i].push(BGE(x, y, z));
                    }
                    Operator::Bl(x, y, z, _) => {
                        let (x, y) = get_regs!(x, y);
                        new_bodies[i].push(BL(x, y, z));
                    }
                    Operator::J(x) => {
                        new_bodies[i].push(PSEUD_J(x));
                    }
                    Operator::Beq(x, y, z, _) => {
                        let (x, y) = get_regs!(x, y);
                        new_bodies[i].push(BEQ(x, y, z));
                    }
                    Operator::Li(x, y) => {
                        let x = get_regs!(x);
                        new_bodies[i].push(PSEUD_LI(x, y));
                    }
                    Operator::Slt(x, y, z) => binary_op!(x, y, z, SLT),
                    Operator::Call(_, y, z) => {
                        debug_assert!(z.len() <= 7);
                        // z already in correct regs due to allocation
                        new_bodies[i].push(PSEUD_CALL(y));
                    }
                    Operator::Return(_) => {
                        // already in x10
                        new_bodies[i].push(PSEUD_J(Rc::clone(&exit_label)));
                    }
                    Operator::Label(_) => {
                        #[cfg(debug_assertions)]
                        unreachable!("label should be excluded in CFG");
                        #[cfg(not(debug_assertions))]
                        unsafe {
                            unreachable_unchecked()
                        }
                    }
                    Operator::GetParameter(_, _) => {
                        //already in correct reg
                    }
                    Operator::StoreLocal(x, y) => {
                        let x = get_regs!(x);
                        new_bodies[i].push(SQ(x, RV64Reg::X2, (y * 8) as i16));
                    }
                    Operator::LoadLocal(x, y) => {
                        let x = get_regs!(x);
                        new_bodies[i].push(LQ(x, RV64Reg::X2, (y * 8) as i16));
                    }
                    Operator::Nop => {}
                }
            }
        }
        let mut new_blocks: Vec<_> = std::mem::take(&mut cfg.blocks)
            .into_iter()
            .enumerate()
            .map(|(i, block)| block.into_other(std::mem::take(&mut new_bodies[i])))
            .collect();
        new_blocks[cfg.get_entry()].body.insert(
            0,
            SQ(
                RV64Reg::X1,
                RV64Reg::X2,
                (*cfg.get_allocated_ars_mut() as i16) * 8,
            ),
        );
        *cfg.get_allocated_ars_mut() += 1;
        new_blocks[cfg.get_entry()].body.insert(
            0,
            ADDI(
                RV64Reg::X2,
                RV64Reg::X2,
                (*cfg.get_allocated_ars_mut() as i16) * -8,
            ),
        );
        new_blocks[cfg.get_exit()].body.extend([
            LQ(
                RV64Reg::X1,
                RV64Reg::X2,
                (*cfg.get_allocated_ars_mut() as i16 - 1) * 8,
            ),
            ADDI(
                RV64Reg::X2,
                RV64Reg::X2,
                (*cfg.get_allocated_ars_mut() as i16) * 8,
            ),
            PSEUD_RET,
        ]);
        cfg.into_other(new_blocks)
    }
    #[cfg(test)]
    mod tests {
        use super::super::register_allocation::*;
        use crate::ir::{Displayable, CFG};

        #[test]
        fn select_instructions() {
            let input = "
        myvar3 :: Bool = false;
        lambda myfun(myvar3 :: Int, myvar5 :: Int) :: Int {
            myvar4 :: Int = 0;
            i :: Int = 100;
            while (i >= 0) do {
                myvar4 = myfun(3, myvar4);
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

            let cfg = CFG::from_linear(body, fun.get_params(), fun.get_max_reg());
            let mut ssa = cfg.to_ssa();
            crate::ssa::optimization_sequence(&mut ssa).unwrap();
            conventionalize_ssa(&mut ssa);

            let mut allocation = RISCV64::allocate(ssa);
            RISCV64::add_procedure_prologues(&mut allocation.0, &allocation.1);
            let native = super::select_instructions(allocation.0, &allocation.1);
            println!("native allocated: {}", native.to_dot());
            let linear = native.linearize();
            println!("linear:\n{}", Displayable(&linear));
        }
        #[test]
        fn select_instructions_fib() {
            let input = "
        lambda fib(n :: Int) :: Int {
            if n < 2 then {
                return 1;
            } else {
                return fib(n - 1) + fib(n - 2);
            }
        }
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
            let fun = funs.get("fib").unwrap();
            let body = fun.get_body();

            let cfg = CFG::from_linear(body, fun.get_params(), fun.get_max_reg());
            let mut ssa = cfg.to_ssa();
            crate::ssa::optimization_sequence(&mut ssa).unwrap();
            conventionalize_ssa(&mut ssa);

            let mut allocation = RISCV64::allocate(ssa);
            RISCV64::add_procedure_prologues(&mut allocation.0, &allocation.1);
            let native = super::select_instructions(allocation.0, &allocation.1);
            println!("native allocated: {}", native.to_dot());
            let linear = native.linearize();
            println!("linear:\n{}", Displayable(&linear));
        }
    }
}
