pub trait SSA {
    type Operation;
    type LinearReg;
    type SSAReg;
}

pub trait Linear {
    type Operation;
    type LinearReg;
    type SSaReg;
    type AsSSA: SSA<Operation = Self::Operation, LinearReg = Self::LinearReg, SSAReg = Self::SSaReg>;

    fn transpile_ssa(&self) -> Self::AsSSA;
}
