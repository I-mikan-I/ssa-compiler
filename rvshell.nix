{pkgs ? import <nixos> {}, lib ? import <nixos/lib>, rv64 ? import <nixos> {
    crossSystem = {
        config = "riscv64-unknown-linux-gnu";
    };
}} :

let
    z3 = pkgs.z3.overrideAttrs (oldAttrs: {
        meta = oldAttrs.meta // {outputsToInstall = [ "dev" ]; };
    });
in
rv64.callPackage (
    {mkShell, stdenv, gcc}:
    mkShell {
        buildInputs = [ stdenv ];
        nativeBuildInputs = [ gcc ];
    }
) {}