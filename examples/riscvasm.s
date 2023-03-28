.globl fib
.attribute arch, "rv64im"

fib:
.ENTRY:
addi x2,x2,-32
sd x1,24(x2)
sd x10,0(x2)
li x5,2
sd x5,8(x2)
ld x6,0(x2)
ld x10,8(x2)
slt x5,x6,x10
li x10,1
beq x5,x10,._LABEL_0
._LABEL_1:
ld x7,0(x2)
sub x10,x7,x10
._LABEL_5:
call fib
sd x10,16(x2)
._LABEL_6:
ld x28,0(x2)
ld x10,8(x2)
sub x10,x28,x10
._LABEL_7:
call fib
._LABEL_8:
ld x29,16(x2)
add x10,x29,x10
.EXIT:
ld x1,24(x2)
addi x2,x2,32
ret
._LABEL_0:
j .EXIT