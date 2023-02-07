//! Intermediate Representation
//! 
//! The compiler's IR is a three-address, register based code with an infinite number of registers.
//! 
//! | Instruction | Ad1 | Ad2 | Ad3 |
//! | ----------- | --- | --- | --- |
//! | add ad1, ad2, ad3 | result | op1 | op2 |
//! | sub ad1, ad2, ad3 | result | op1 | op2 |
//! | mult ad1, ad2, ad3 | result | op1 | op2 |
//! | div ad1, ad2, ad3 | result | op1 | op2 |
//! | and ad1, ad2, ad3 | result | op1 | op2 |
//! | or ad1, ad2, ad3 | result | op1 | op2 |
//! | xor ad1, ad2, ad3 | result | op1 | op2 |
//! | load ad1, ad2, ad3 | result | offset | memory base |
//! | store ad1, ad2, ad3 | value | offset | memory base |
//! | la ad1, @label | result | NA | NA |
//! | bgt ad1, ad2, @label | op1 | op2 | NA |
//! | bl ad1, ad2, @label | op1 | op2 | NA |
//! | j @label | NA | NA | NA |
//! | beq ad1, ad2, @label | op1 | op2 | NA |
//! | li ad1, imm  | result | NA | NA |
//! | slt ad1, ad2, ad3 | result | op1 | op2 |
//! | call ad1, @label (ad2 ... adn) | result | arg1 | ... |
//! | return ad1 | result | NA | NA |
//! 
//! - Labels are represented as immediate string arguments to the op.
//! - Memory locations are symbolic, an execution could for example
//!   assign a random value to each label.
//! - ad1, ad2, ad3 are registers.
//! - Registers hold 8 bytes.
//! - Negative numbers use 2's complement.
//! - Memory locations store a single register.
//! 
//! ## Code Shape
//! 
//! lin: transforms AST into IR before inserting line, returns register name of result.  
//! name: allocates register name.  
//! label: allocates memory location.  
//! body: inserts IR of AST.  
//! 
//! (L) + (R) =>    add name() lin(L) lin(R)
//! (L) - (R) =>    sub name() lin(L) lin(R)
//! (L) * (R) =>    mul name() lin(L) lin(R)
//! (L) / (R) =>    div name() lin(L) lin(R)
//! (L) or (R) =>   ...
//! (L) and (R) =>  ...
//! (L) xor (R) =>  ...
//! not (L) =>      li name() -1 // r_new
//!                 xor lin(L) r_new
//! (L) >= (R) =>   slt name() lin(R) lin(L) // r_new
//!                 li name() -1 // r_new2
//!                 xor name() r_new r_new2
//! (L) < (R) =>    slt name() lin(L) lin(R)
//! (L) == (R) =>   xor name() lin(L) lin(R) // r_new
//!                 li name() 1 // r_new2
//!                 slt name() r_new r_new2
//! true =>         li name() 1
//! false =>        li name() 0
//! number =>       li name() number
//! ident = (R)     store lin(R) 0 @label() 
//! ident           load name() 0 @label(ident) // reuse label
//! if (C) then (I) 
//! else (E) =>     li name() 1 // r_new
//!                 beq lin(C) r_new @l1
//!                 j @l2
//!                 label(): body(I) // @l1
//!                 j @l3
//!                 label(): body(E) // @l2
//!                 j @l3
//!                 label(): ... // @l3
//! while (C)
//! do (B) =>       label():    // @l1
//!                 li name() 0 // r_new
//!                 beq lin(C) r_ne2 @l2
//!                 body(B)
//!                 j @l1
//!                 label(): ... // @l2
//! identifier((arg0), (arg1), ...)
//! =>              call @label(identifier) (lin(arg0), lin(arg1), ...)
//! return (E) =>   return lin(E)